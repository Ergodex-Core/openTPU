# Phonon-2 raw PCM to text on the shared-profile ISA

This increment preserves `09f3ee76` and supplies the remaining device waveform
frontend. Raw mono PCM16 bytes pass through compiled signal processing, learned
subsampling, all 24 Conformer layers, LSTM/TDT decoding and UTF-8 selection.
The qualification backend is CPU ISA simulation. No FPGA, image change,
synthesis, production worker or readiness enablement is involved.

The required target remains `f2-hbm:p4rl8a64c150`, CFG `dd6b92a6`, with
parent-designated AGFI `agfi-0001de1dd3df8e21e`. Its live identity and capabilities
have not been queried. The [runtime dependency manifest](phonon2/waveform_runtime_dependency.json)
pins the same 17 common-runtime source files at `df69c0a9d970ba4f78b8ac6e979b41dc69d4f181`,
the preserved learned-stage sources, relevant RTL and normalization source.
Changed compiler/runtime files require requalification.

## Device input and preprocessing

The input contract is unchanged: 16 kHz, one channel, signed little-endian
PCM16, 320 through 480,000 samples. Other sample rates, channel counts, dither
or durations are rejected. Host work in the proposed path is byte transport,
immutable program/constant staging, orchestration and UTF-8 byte output I/O.
There is no host signed conversion, scaling, resampling, transform, masking,
normalization, learned computation or token-driven decode loop.

The frontend has six device programs:

1. Reset its scratch while preserving the transported PCM input.
2. Extract signed bytes using INT8 MM with an immutable one-hot ACT basis.
   ARGMAX identifies negative low bytes; unsigned-low correction plus signed
   high-byte arithmetic reconstructs PCM16, then multiplies by `1/32768`.
   All 65,536 PCM16 values were tested and are bit-exact.
3. Compute preemphasis `x[n] - 0.97*x[n-1]`, preserving the first sample, and
   place samples inside 256 zeros on each side for centered STFT.
4. Apply the nonperiodic Hann400 window centered in 512 samples, with hop160.
   A direct 512-point DFT uses existing FP32 RDOT in 32-bin tiles, producing
   257 bins. It preserves the source's magnitude-square graph, including its
   square-root operation represented by reciprocal/rsqrt composition.
5. Apply 128 Slaney area-normalized mel filters and `ln(mel + 2^-24)` using
   existing log2 and multiplication by ln2. Retain all stored log-mel frames;
   mask the last frame in statistics input.
6. Compute masked mean and unbiased variance, divide by `sqrt(var)+1e-5`, and
   leave invalid output rows zero. Stored frames are `floor(N/160)+1`; valid
   frames are `floor(N/160)`. These lengths propagate independently through
   the learned subsampler.

No FFT opcode is required for correctness. The direct DFT is a conservative
implementation with unmeasured physical throughput; no timing promise follows
from its memory fit. The 7,134-byte coefficient asset holds input-independent
roots of unity, Hann and mel coefficients. Host expansion of static coefficient
tables is compilation/constant preparation, independent of audio.

## Pinned numerical policy

Dither is explicitly zero, matching the inspected Transformers path. The NeMo
archive's `1e-5` dither setting remains a documented runtime disagreement; it
is not silently introduced. The original [semantic policy](phonon2/semantic_policy.json)
and frontend goldens are preserved.

The initial generic ISA reduction differed from Torch on long silence. The
normalization divides a one-ulp mean error by approximately `1e-5`, so that
apparently small discrepancy can become an output difference near 0.16.
The fix lowers the retained ARM64 Torch FP32 summation order: Vec4, ILP4 and
the 16-group cascade from PyTorch commit `5c4886908584029761b579af026dcfb627c84070`.
It includes the masked final frame and corrects reciprocal-multiply count
division with a split-product residual, using existing VPU operations.

This preserves source equations and their nonzero silence results; it does not
detect silence and force its features to zero. Recorded normalization boundaries
through 30 seconds match the retained ARM64 reference within `5e-7` absolute
error. Native x86 Torch can use a different vector reduction order. The full
qualification records that implementation difference separately and compares
complete transcripts/traces against both the pinned statistics policy and the
native CPU Torch path. There is no claim of universal bitwise equality across
FFT libraries or CPU backends.

The adapted reduction is attributed to PyTorch under its preserved
[license](phonon2/TORCH-LICENSE.txt). Model weights/tokenizer data remain
CC-BY-4.0 with NVIDIA/Fermion attribution; the earlier [architecture/license
audit](phonon2_asr.md) remains applicable. No checkpoint weights are in Git or
on the Mac, and no upstream model code was executed.

## Complete device composition and resources

Frontend output is copied directly to subsampler input before immutable weights
are staged. Those regions overlap for some durations; a tested device memmove
chooses forward or reverse row order from their static addresses. Activations
never pass through host preprocessing or re-upload between model stages.
The subsequent encoder and TDT programs retain their earlier source/receipt
bindings. All model and DSP operations remain inside the device program stream.

At the 30-second compile envelope:

| Frontend resource | Actual compiled budget | Shared-profile limit |
|---|---:|---:|
| Graph DRAM | 15,286,144 bytes | 4 GiB |
| Immutable frontend constants | 1,251,840 bytes | included above |
| Peak live TMEM | 33,409 words | 65,536 words |
| Largest phase | 25,328 words | 32,768 words |
| Largest program + wait image | 126,704 bytes | 163,840-byte slot |
| Stored / valid mel frames | 3001 / 3000 | pinned duration guard |

The learned subsampler remains the largest graph allocation at 180,715,008
bytes for the maximum context. Graph memory is reused across phases rather
than summing all component allocations. The top-of-DRAM program/wait slot
remains separate. The raw-PCM complete path has 693 dispatches, including reset,
frontend programs, copies, learned stages and autonomous TDT. These are software
budgets; bandwidth, clocked execution and the user's maximum 30-second serving
interruption still require coordinated physical measurement.

## Validation and reproduction

The shared-runtime regression passes 193 tests with one optional Transformers
test skipped. Frontend coverage includes every PCM16 value, retained silence,
impulse/chirp/noise vectors, exact and partial hops, masks, duration/dither guards,
30-second normalization boundaries, reset and overlapping copies in both
directions. No existing projection, subsampler, encoder or TDT receipt is
rewritten as evidence for a changed compiler.

The original existing-EC2 qualification completed successfully before the
execution environment was restored. Its saved receipt was recovered and
hash-verified without launching a duplicate model run. All five cases match
the complete token/duration/frame traces and UTF-8 text of both references:

| PCM case | Samples | Decode steps | Maximum feature absolute error |
|---|---:|---:|---:|
| Short speech crop | 8960 | 7 | 7.80821e-5 |
| Full speech recording | 112960 | 53 | 7.12872e-5 |
| Minimum-duration silence | 320 | 1 | 0 |
| Silence reduction boundary | 5280 | 4 | 0 |
| Short speech after reset | 8960 | 7 | 7.80821e-5 |

The full transcript is:

> I really was very much afraid of showing him how much shocked I was at some parts of what he said.

PCM conversion and preemphasis are bit-exact in every case, all masks pass, and
reset reproduces frontend features, encoder output, decoder state/cache, trace
and text bit-for-bit. The full recording's final encoder relative L2 error
against the dense pinned reference is `9.32875e-5` (0.009329%). Native CPU versus
pinned statistics feature differences were at most `9.53674e-7` in these cases.
The full recording exercises a reverse device copy for an overlapping handoff.

[waveform_validation.json](phonon2/waveform_validation.json) binds the model,
runtime, sources, constants, input PCM hashes, every encoder layer, decoder
traces, masks, reset and budgets. The full case uses 56,851,456 graph bytes,
1,272,022,400 staged immutable bytes and 3,191,720 uploaded program/wait bytes
across 693 dispatches. These are CPU/ISA execution measurements, not FPGA timing.
The frontend reference is separate audit work and never feeds features into the
device path. The full case does not separately run the wait checker; short-case
and unit tests exercise the conservative dependency schedule.

After receipt recovery, comments were corrected to describe the ISA's actual
zero specials (`rsqrt(0)=0`, `recip(0)=0`) and add PyTorch attribution. Executable
AST, encoded programs, images and budgets remain identical at four contexts,
including 30 seconds. [The equivalence record](phonon2/waveform_source_equivalence.json)
preserves both original qualification and delivered source hashes explicitly.

Reproduce using the existing EC2 environment and checkpoint:

```sh
python -B -m tools.phonon2_common_runtime \
  --prepare-from-git /existing/common-runtime-checkout \
  --destination /tmp/phonon2-common-runtime
python -B -m tools.phonon2_run_shared_waveform \
  --runtime-root /tmp/phonon2-common-runtime \
  --container /existing/ec2/model.fermion \
  --pcm docs/phonon2/speech_1688_crop.pcm \
  --wave /existing/ec2/speech_1688_full.wav \
  --output-dir /tmp/phonon2-waveform-qualification
```

The tutorial float-WAV conversion is offline fixture preparation. The inference
input is the resulting PCM16 bytes, whose hash is recorded. It is not a serving
resampling/conversion path. Final physical audio-to-transcript acceptance and
Inferlab availability remain gated, with the same image, parent-coordinated
ownership locks and no CPU/GPU fallback.
