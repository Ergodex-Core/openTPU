# Qwen3-TTS eight-layer decoder transformer

This increment extends the resident decoder front at `85a42df9` through input
projection, all eight transformer layers, final RMS normalization and output
projection. It produces one 1024-feature decoder row per codec frame. It does
not produce waveform samples. The component passes software ISA acceptance;
the preserved overall main-model gate and full-TTS readiness remain false.

## Pinned source and execution

The model is exactly `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, revision
`85e237c12c027371202489a0ec509ded67b5e4b5`. The reviewed source is Qwen3-TTS revision
`022e286b98fbec7e1e916cb940cdf532cd9f488e`; model/source license is Apache-2.0.
Primary references are the [decoder source](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py),
[decoder config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/blob/85e237c12c027371202489a0ec509ded67b5e4b5/speech_tokenizer/config.json),
[configuration source](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py)
and [Transformers 4.57.3 mask source](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/masking_utils.py).
These files were inspected as source; upstream executable model code was not
imported or run. The NumPy references are independent authored algebra, not an
execution of upstream Torch/CUDA kernels.

The existing builder data was reused without copying model weights to the Mac.
The tokenizer safetensors file is 682,293,092 bytes, SHA-256
`836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`. Its full digest,
496 F32 tensors, header extents and requested tensor shapes were checked again.
This transformer uses 93 official parameter tensors in addition to the 36
unique front-end tensors. Main-checkpoint provenance and both public LFS hashes
remain in the [front-end evidence](qwen3_tts_audio_front.md).

All tests and checkpoint computation ran in the isolated
`/home/ubuntu/qwen3-tts-audio-transformer-20261023-01` directory on the existing
`opentpu-m8a-build`. Python 3.12.3 and NumPy 2.5.4 came from the existing
`/home/ubuntu/otpu-venv`; no Torch installation or new dependency environment was
needed. Jobs used one BLAS/OMP thread, nice 19, a 64 GiB address-space cap and
bounded timeouts. The Mac was the source/Git and receipt-transport bridge.

## Device arithmetic and integration

The [arithmetic contract](qwen3_tts_arithmetic_contract.md) is unchanged: pinned
source weights evaluated with FP32 algebra, approximated by the documented
corrected MX affine representation. Each affine uses two INT8 weight terms and
two device-generated activation terms, with four MM products. This is not native
FP32 MM or bit-exact BF16 execution. Norms, attention, residuals, layer scales
and KV remain FP32 in the device program.

The source decoder differs from the talker: hidden width 512, sixteen Q and KV
heads of width 64, FFN width 1024, RMS epsilon 1e-5, theta 10000, Identity Q/K
norms, and learned scales on attention and MLP branches before residual addition.
Input/output projections map 1024-to-512 and 512-to-1024, both with bias. All
eight layers use SwiGLU and a sliding causal window of 72 total positions,
including the current position. The width-64 heads use FP32 VOP attention;
affine contraction widths remain aligned to the shared D=128 MXU. No talker
head-normalization or other-model compatibility is assumed.

Each layer keeps exactly 72 FP32 K/V slots per head in a resident ring. Keys and
values are traversed in chronological order across wrap. Only current/past
visible slots are read. Rotary tables for positions 0 through 7999 are fixed
offline constants in device memory; no host trig or learned request-time
arithmetic occurs. Position 8000 is rejected rather than wrapped or truncated.
The 8000-position bound is this component's qualified bound; it does not claim
an upstream default-RoPE mathematical limit. The integrated main path still has
its separate 512-position talker limit, including text prefill.

`AudioTransformerImage` and its diagnostic session extend the standalone front.
`AudioTransformerFrameImage` consumes the same resident `audio.preconv` tensor
after generated codec IDs and exposes `desc.audio_transformed` with shape
`(1,1024)`. `AudioTransformerFrameSession` supplies codec-frame position separately
from text/talker position. A small overridable `feedback_parts()` hook in the
base diagnostic session preserves the old dispatch by default; existing front
regressions exercise that unchanged behavior.

Reset clears front convolution history, decoder ring caches and inherited
caches/counters. Failed execution requires reset. Diagnostic reads feed only
independent comparisons after execution. No host hidden-state, context, code or
feature upload participates in the integrated generated-frame path.

## Numeric and state acceptance

`transformer-tests-02` passed **9 tests in 22.84 seconds**, including the four
existing front tests. New tests cover layerwise FP64 source algebra, repeated
ring wrap, independent full-sequence masking, poisoned future slots, poisoned
reset, position bounds, execution-failure latching, chunk recomputation and
resident frame position independent of three-row text prefill. Native compilation
checks cover positions 0, 71, 72, 143 and 7999 on the exact shared target.

`transformer-acceptance-01` completed with **exit 0 in 296.07 seconds**:

* Nine retained three-prompt prefix frames were tested against real checkpoint
  weights. This corpus contains five unique codec-ID frames; it is not a broad
  speech-quality corpus.
* An additional 80-frame sequence repeats those previously generated IDs solely
  to test cache evolution and the real 72-frame boundary. It is labeled as a
  diagnostic in every per-frame row, not as a generated utterance.
* Independent FP32 and FP64 references start from unquantized source front-end
  output. A separate conditional FP32 reference consumes the measured device
  pre-convolution output to distinguish transformer error from accumulated
  front-end error. All layer/boundary checks retain the 3.5% relative L2 bound.
* A full-sequence masked FP32 reference independently checks the incremental
  source-algebra reference. Maximum difference across all stages is
  0.000347744%, below the separate 0.001% reference-consistency bound.
* Poisoned-state reset replay of the original prefix was bitwise equal.
* Actual main-checkpoint generation of three `Hello.` frames was executed twice,
  with full reset between runs. All 48/48 codec IDs per run match the preserved
  FP32 path; transformed features replay bitwise. Maximum integrated output
  error is 0.013374521%, maximum absolute error 0.000008163974.

Maximum per-row relative L2 errors over the 89 component frames, in percent:

| Boundary | FP32 source front | FP64 source front | Conditional FP32 |
| --- | ---: | ---: | ---: |
| Input projection | 0.014967739% | 0.014970307% | 0.012242623% |
| Layer 0 | 0.015068873% | 0.015072079% | 0.012126903% |
| Layer 1 | 0.014856215% | 0.014855940% | 0.011693842% |
| Layer 2 | 0.013910090% | 0.013856731% | 0.016728705% |
| Layer 3 | 0.026383586% | 0.026303435% | 0.025944985% |
| Layer 4 | 0.036047241% | 0.035941718% | 0.039916842% |
| Layer 5 | 0.044218813% | 0.044106105% | 0.053490606% |
| Layer 6 | 0.044173838% | 0.043945219% | 0.053601512% |
| Layer 7 | 0.051804335% | 0.051535179% | 0.061892667% |
| Final RMS normalization | 0.026177023% | 0.026249572% | 0.034114657% |
| Output projection | 0.032112370% | 0.031969098% | 0.041705243% |

These are feature errors, not waveform or perceptual quality measures. The
largest conditional error is larger than the accumulated error in this corpus;
neither column is hidden or substituted for the other.

## Continuous and source chunk semantics

The integrated per-frame path is equivalent within the measured error bound to
one continuous, causal full-sequence decoder-transformer call. The public
tokenizer `decode()` calls `decoder.chunked_decode()` with its defaults. That
method recomputes each chunk with up to
25 preceding codec frames, resetting pre-convolution, rotary position and all
transformer state, then discards context outputs. Its default chunk size is 300.

`chunked_features()` implements that control for this feature boundary, including
the source defaults. The real test used an explicit 64-frame chunk size so the
80-frame diagnostic crossed a boundary. It recomputed 41 frames (25 context plus
16 emitted rows) after reset and matched the independent source chunk reference
within **0.026490665%** relative L2. The small synthetic tests also execute the
public helper and verify replay.

Continuous output and recomputed output differed by **7.1936486%** feature L2 on
the second chunk. That is a measured difference between two source modes, not
an implementation error against either mode's reference. They must not be
silently interchanged in future serving. The generated-frame integration tests
continuous mode; it does not claim the full waveform `chunked_decode` API or a
300-frame real-checkpoint sequence has been validated. A production controller
for generated frames must retain codec IDs in device memory and replay the
preceding 25 at the 300-frame boundary, restarting decoder state/positions and
discarding context features. That resident replay controller is not implemented
here; the standalone diagnostic helper accepts supplied codec IDs. It must be
integrated before claiming long-utterance parity with the public decode API.

## Shared image and resources

The target remains AGFI `agfi-0001de1dd3df8e21e`, profile
`f2-hbm:p4rl8a64c150`, build `417243f1`, configuration hash `dd6b92a6`, with runtime
`30100afb075374381601a6e01d9f914a58dbec91`. The runtime is a required dependency,
not an assumption of equivalence with a different image/profile. Evidence uses
the pinned target and prior register receipt; there was no live card query.

The transformer adds 146 sequential programs per frame: input projection,
eight groups of QKV, sixteen head programs and finish, then final output.
Together with front/main schedules this is 544 programs per generated frame.
These counts describe the current split; no hardware throughput is inferred.

| Representative program at position 72 | Instructions | IMEM words | Peak TMEM words |
| --- | ---: | ---: | ---: |
| Input projection | 13 | 104 | 3,073 |
| QKV | 87 | 696 | 7,811 |
| One attention head | 241 | 1,928 | 6,312 |
| Attention residual and MLP | 121 | 968 | 5,636 |
| Final norm/output projection | 21 | 168 | 3,585 |

All tested positions fit IMEM 32,768 and TMEM 65,536 words. Standalone front plus
transformer residency is 107,370,624 bytes. Combined main/front/transformer
residency is **2,781,933,440 bytes**, 68,556,800 bytes above the prior combined
front, leaving **1,513,033,856 bytes** of 4 GiB. The standalone path additionally
needs a 1 MiB identity already present in the main path. Future waveform weights,
buffers, programs and throughput have not been qualified.

## Remaining gates, host audit and reproduction

Implemented learned operations are device programs: input projection, all norms,
Q/K/V and output projections, rotary application, attention/softmax, learned
branch scales, SwiGLU and residuals, final norm/projection and KV writes. Host
work is offline verified weight packing/constants, deterministic program/position
and chunk scheduling, state reset, transport, and explicit CPU test references.
No CPU/GPU/API fallback or production backend was added.

All historical receipts remain intact. The main-model receipt still fails its
unchanged gate: BF16 agrees on 127/144 generated IDs and 6/9 complete frames;
maximum conditional BF16 logit error is 7.51369%, above 3.5%. FP32 source-algebra
feature acceptance does not relabel that gate as passed or establish quality.

Remaining stages include resident generated-code chunk replay, the two factor-2 causal upsampling/ConvNeXt blocks, waveform
convolution/SnakeBeta blocks with rates 8/5/4/3, raw output clamp/scaling, complete
utterances, quality and coordinated physical/serving acceptance. No missing
primitive blocked this transformer. Later exact-GELU and sine-based SnakeBeta
still need source-specific bounded ISA lowerings; existing tanh-GELU is not
silently substituted. No new RTL or synthesis is proposed by this result.
`waveform_produced`, raw-scaling/utterance/quality acceptance, physical acceptance
and `tts_enabled` all remain false. Service interruption was zero; its permitted
maximum remains 30 seconds. Hardware/image/production were untouched.

From a builder checkout with the pinned runtime and this increment:

```sh
/home/ubuntu/otpu-venv/bin/python -m pytest -q tests/test_qwen3_tts_audio_front.py tests/test_qwen3_tts_audio_transformer.py
/home/ubuntu/otpu-venv/bin/python -u tools/qwen3_tts_audio_transformer_acceptance.py --data /home/ubuntu/qwen3-tts-audio-transformer-20261023-01/data --output /path/to/new-transformer-receipt.json --integrated
```

Saved jobs are terminal; inspect their receipts before any rerun. The evidence
directory preserves requests, results, full stage/frame metrics, source hashes,
environment and runner provenance. The first launcher failed with a quoting
SyntaxError before any tests executed; its log/result and exact failed wrapper
bytes are preserved (the wrapper has `.txt` appended locally, recorded in the
verification alias map). The corrected launcher produced the nine passing tests;
no numerical threshold changed. A separate metadata-packaging attempt used
system Python without NumPy and failed before writing artifacts; packaging then
used the same existing venv as the tests, with no tests restarted.

All 31 acceptance-start source hashes and the per-row receipt hash match the
review checkout. The post-run tree verifies 54 component/fixture files plus 95
runtime files against the pinned Git object. The one unused copied main receipt
has an explicit local path alias. These source/resource receipts establish
reproducibility of software simulation, not physical FPGA execution.
