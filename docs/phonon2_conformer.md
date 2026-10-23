# Phonon-2 bounded Conformer and frontend qualification

This adds one complete Conformer eval block on the existing ISA, plus audited
offline frontend vectors. **ASR remains blocked.** No real audio has traversed
the FPGA, no FPGA was accessed, and no image, service, lock or paid resource was
changed. The Inferlab contract still returns `ASR_FPGA_NOT_READY` and advertises
`ready=false`, `hardware_validated=false`, and no CPU/GPU fallback.

Model: `FermionResearch/Phonon-2` at
`fb09063c9dc8e130d97b3a2191ed5aed6b71bb31`. The architecture/license investigation
and checkpoint inventory remain in [phonon2_asr.md](phonon2_asr.md). Weights stay
on the existing EC2 instance. None are included here.

## Pinned semantics and golden vectors

[semantic_policy.json](phonon2/semantic_policy.json) is the explicit qualification
policy, following the inspected [Transformers frontend][frontend] and
[Conformer source][graph]. This is not a claim of identical MLX or NeMo output.

* Input: mono 16 kHz PCM16, scaled by 1/32768, 320..480000 samples. Reject short
  clips with fewer than two valid feature frames (unbiased variance otherwise
  divides by zero), partial PCM samples and clips over 30 seconds. No automatic
  resampling, channel averaging, chunking or truncation is introduced.
* Dither is explicitly **zero** for the selected Transformers path. The archive's
  NeMo config says `1e-5`; that unresolved runtime difference is preserved.
* Preemphasis 0.97, centered zero-padded FFT512, hop160, nonperiodic Hann400,
  128 Slaney mel filters, `log(mel + 2^-24)`, valid-frame mean and unbiased
  variance, divide by `std + 1e-5`, then mask. The float32 Slaney table builder
  was audited against [librosa 0.11.0's filter implementation][mel] and
  [frequency conversion][mel-convert]; no librosa code was installed/executed.
* For N samples, stored frames = floor(N/160)+1, valid frames = floor(N/160).
  Subsampling propagates stored lengths separately from valid-prefix lengths.
  At 30 seconds the counts are 3001/3000, then 376/375 after three stride-2 stages.
* Attention masks valid queries **and** keys using finite float32 minimum in the
  relative-score term. Fully masked padded queries retain the reference's
  uniform attention behavior. GLU output is zeroed before depthwise convolution;
  residual outputs and final padded rows are not forcibly zeroed.

[frontend_vectors.npz](phonon2/frontend_vectors.npz) contains six deterministic
PCM fixtures, preemphasis, log-mel, normalized features, masks and the mel bank.
The [manifest](phonon2/frontend_vectors.json) records input/output hashes, library
versions, host architecture and errors against a separately written NumPy FFT
reference. The vectors use our own audited PyTorch CPU equations, not an imported
upstream model class. Regenerate only intentionally:

```sh
python -B -m tools.phonon2_frontend_vectors --output-dir /tmp/phonon2-vectors
```

There is a real numerical qualification gap: 100 ms silence is zero after
normalization in this PyTorch CPU fixture, but the same equations with NumPy's
FP32 reduction produce about -0.158806 on valid rows. Both have identical
log-mel values. Tiny mean-rounding errors are amplified by near-zero variance.
The fixture retains this case, records the discrepancy, and does not waive it
with a broad all-input tolerance. On nondegenerate fixtures, measured maximum
log-mel error is 0.000208 and feature error 0.000110. Device frontend acceptance
must explicitly resolve low-variance reduction behavior before serving is enabled.

TDT was not implemented or altered. The semantic policy retains separate token
and duration argmax, durations `[0,1,2,3,4]`, blank=8192 and the blank/zero-duration
advance-by-one guard. It also records that the inspected Fermion CUDA path forces
advance after ten zero-duration symbols, while the pinned Transformers TDT update
does not apply that guard. No decoder policy is silently substituted. A future
watchdog must report failure/incomplete output rather than changing transcript
semantics to terminate a loop. See the pinned [generation source][generation].

## Complete block lowering

`opentpu/asr/phonon2_conformer.py` compiles batch 1, H=1024, FF=4096, eight
128-dimensional heads, depthwise kernel9, and 1..8 stored encoder frames with a
nonempty contiguous valid prefix. Smaller widths are supported only for unit
tests. The selected length/mask is static, derived from length metadata; dynamic
device mask handoff has not been integrated.

The complete order is half-scaled FF1 residual, relative-attention residual,
GLU/depthwise/BatchNorm/SiLU/pointwise residual, half-scaled FF2 residual, then
final LayerNorm. All projection weights retain their exact stored five levels in
two disjoint signed-INT8 planes with row scales. INT6 depthwise weights and FP16
normalization parameters are unpacked offline without additional quantization.
Only activations entering projections use QACT. Attention scores/probabilities,
depthwise convolution, normalization and residuals remain FP32.

| Operation | Existing device instructions |
|---|---|
| Five-value affine | QACT once, MM low plane, MM high plane with accumulation |
| LayerNorm | RSUM, subtraction, RSSQ, RSQRT, gain/bias VOPs; epsilon 1e-5 |
| SiLU and GLU | EXP2, RECIP, MUL; GLU halves in upstream order |
| Relative attention | FP32 RDOT, bias ADD, relative-key row addressing, finite-min FILL, RMAX/EXP2/RSUM/RECIP softmax |
| Value aggregation | On-device column-copy transpose and FP32 RDOT |
| Depthwise kernel9 | Boundary-aware LD plus per-channel MUL/ADD; zero padding |
| Eval BatchNorm | Subtract running mean, RSQRT(var+1e-5), gain/bias, SiLU |
| Phase control | LI, ADDI, LOOP, LD, ST, HALT |

There are 17 phases for eight heads. Device DRAM persists across all phases;
the host only sequences programs. The qualification harness reads activations
after all phases complete. No host attention, convolution, normalization or
intermediate transpose is hidden between launches. Sinusoidal position tables
are immutable compile-time constants; their learned projection executes on device.

No missing ISA primitive was encountered for **this bounded block**. Its programs
do not require STREAM, DSTEP, PAIR, a new opcode, or a new image. This is an ISA
functional result, not RTL simulation, timing closure or evidence about the loaded
card's capabilities.

## Resources and numerical evidence

For eight stored / seven valid frames, measured compiler budgets are:

| Resource | Used | Saved F2 limit |
|---|---:|---:|
| DRAM image | 51,097,600 bytes | 4 GiB logical DRAM; qualification simulator uses 64 MiB |
| Weights, aligned | 50,581,504 bytes | Included above |
| Peak live TMEM | 57,352 words | 65,536 words |
| Largest phase program | 2,760 words | 32,768 words |
| QACT projection depth | 32 blocks of 128 | 128 blocks |
| Projection rows | 4 | 16 ACT rows, four MXU columns |

The arithmetic count is approximately 417.3 million signed-INT8 MACs across the
two weight planes for this sequence, plus about 196.6 thousand FP32 attention
dot-product MACs and at most 73.7 thousand depthwise MACs. These are operation
counts, **not** latency or throughput measurements. Full-model weight planning
remains about 1.239 GB before alignment, activations and programs; complete graph
fit and bandwidth have not been established.

The real checkpoint's first layer was run on the existing EC2 CPU ISA simulator.
Input was eight seeded synthetic 1024-wide activation rows, not subsampled audio.
[conformer_validation.json](phonon2/conformer_validation.json) records exact record
hashes, source hashes, every phase's budget, per-stage errors and acceptance limits.
The selected runtime source is the exact published PR #39 commit
`8815ac80226aa3a0d0b07d03344f61839580415e`, with new block/audit files isolated.

Final relative L2 error versus the dense NumPy reference is about **1.70%**.
Against the reference using ISA activation quantization it is about **1.8e-7**.
The independent dense NumPy and PyTorch equations agree to about **6.4e-7**
relative L2. Quantization accuracy on real encoder activations and transcript WER
remain unmeasured. One passing synthetic sequence is not a model-quality claim.

Reproduce with the existing approved checkpoint and installed dependencies:

```sh
python -B -m tools.phonon2_qualify_block \
  --container /path/to/existing/model.fermion --report /tmp/phonon2-block.json
python -B -m pytest -q tests/test_phonon2_conformer.py tests/test_phonon2_asr.py \
  tests/test_kernels.py tests/test_autodecode.py
```

## Remaining blockers and next bounded work

The production endpoint remains unavailable. The frontend exists only as audited
CPU test equations/vectors; these are never imported by the serving adapter.
Missing device work includes PCM16 conversion, FFT/mel/normalization, the learned
Conv2D subsampler, length-mask handoff, tiled attention for up to 376 stored frames,
all 24 blocks in one resident graph, LSTM predictor, joint head, TDT control and
text output. The compiler rejects more than eight frames; do not split audio into
eight-frame chunks and call it equivalent noncausal attention.

Next, qualify a short device frontend against these exact vectors, first proving
PCM unpacking/normalization and one FFT512 frame's permutations/reductions on
the existing ISA. There is no native PCM16 conversion or FFT instruction; safe
lowering and numerical behavior remain to be demonstrated, not assumed. Then
implement the actual subsampler with its three valid-prefix masks, and qualify
this block on its output. Tiled full-context attention is a separate bounded
step. If a required operation cannot be expressed safely or the measured budget
fails, stop dependent work and propose the exact primitive/image change before
synthesis. No new image is currently justified by this block's evidence.

Hardware acceptance still requires a parent-coordinated window, normal ownership
locks and loaded-image checks. The final gate remains real audio-to-transcript
FPGA evidence with all model computation device-side.

[frontend]: https://github.com/huggingface/transformers/blob/ea52711b4850c332be3f778396ce882ab84a9011/src/transformers/models/parakeet/feature_extraction_parakeet.py
[graph]: https://github.com/huggingface/transformers/blob/ea52711b4850c332be3f778396ce882ab84a9011/src/transformers/models/parakeet/modeling_parakeet.py
[generation]: https://github.com/huggingface/transformers/blob/ea52711b4850c332be3f778396ce882ab84a9011/src/transformers/models/parakeet/generation_parakeet.py
[mel]: https://github.com/librosa/librosa/blob/0.11.0/librosa/filters.py#L121
[mel-convert]: https://github.com/librosa/librosa/blob/0.11.0/librosa/core/convert.py#L1113
