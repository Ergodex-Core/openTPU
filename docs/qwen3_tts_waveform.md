# Qwen3-TTS resident waveform decoder

This bounded increment adds the remaining causal waveform network after the
resident upsampling output. It targets the exact deployed shared image and
produces 1,920 mono FP32 waveform samples per codec frame in ISA simulation.
This is component software acceptance, not physical or complete-utterance TTS
readiness. All serving paths remain disabled.

## Pinned architecture and arithmetic

The checkpoint is the Apache-2.0
[Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tree/85e237c12c027371202489a0ec509ded67b5e4b5),
revision `85e237c12c027371202489a0ec509ded67b5e4b5`; the
[official decoder source](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py)
is read-only revision `022e286b98fbec7e1e916cb940cdf532cd9f488e`.
No upstream executable model code is imported. The cached tokenizer checkpoint
has 682,293,092 bytes and SHA-256
`836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`.
Its reader verifies the full digest, header, all tensor extents, exact F32 shapes
and finite parameters before packing. The main BF16 checkpoint is independently
verified by the existing reader for generated-code integration.

Four resident rows of 1,024 features feed a causal kernel-7 convolution to
1,536 channels. Four waveform blocks use rates [8,5,4,3] and widths
1,536 -> 768 -> 384 -> 192 -> 96. Each has SnakeBeta, a stride-r causal
transposed convolution with kernel 2r/right trim r, and three residual units
with kernel-7 dilations 1,3,9 and kernel-1 output convolutions. Each residual
unit has two SnakeBeta operations. The final SnakeBeta and 96-to-1 causal
kernel-7 convolution produce raw samples. Source scaling is exactly
`clamp(raw, -1, 1)`: no output tanh, peak normalization or gain is introduced.
The total rate is 1,920 samples per 12.5-Hz codec frame at 24 kHz.

The stage reads **118 unique F32 tensors**, totaling 52,370,113 elements /
209,480,452 raw bytes. These become **238 physical parameter views**: 150
convolution taps, 30 biases and 58 folded SnakeBeta constants. Transposed tap
weights explicitly convert [input,output,tap] to MX [output,input] offline.
Contraction widths 192 and 96 pad with device zeros to 256 and 128; matching
weight columns are zero-padded offline. Padding does not change channel counts
or source algebra. Biases, residuals, history, output and static Snake constants
are resident FP32. Every affine uses the existing two-term INT8 weight and
two-term activation representation with four MM products and FP32 accumulation.

The arithmetic contract remains source weights evaluated with FP32 algebra,
not bit-exact BF16 execution. SnakeBeta is
`x + sin(x * exp(alpha))**2 / (exp(beta) + 1e-9)`.
Static exp(alpha) and reciprocal-exp(beta) constants are folded offline in F32.
Input-dependent periodic computation runs entirely in device VOP programs.

There is no native sine/floor opcode on this target. `sin2_bounded` uses FP32
round-to-nearest range reduction, split pi constants (201/64 plus the residual),
a degree-13 sine polynomial, square and clamp. The qualified argument domain
is [-262144,262144]. The native ISA probe checked 16,384 dense and 16,384 random
values over that domain, 4,096 near zero, and 108 pi/half-pi/ULP-neighbor cases.
Maximum absolute sin-squared error was **4.172325134277344e-6**, below the fixed
2e-5 bound. These are sampled-domain results, not an exhaustive FP32 proof.
All 29 checkpoint SnakeBeta parameter groups yield finite folded constants;
maximum exp(alpha) is 8.5410814 and inverse-beta is 360.7342.

Each device frame accumulates maximum absolute sine argument into a resident
guard scalar. After a frame's waveform finish, the session reads that scalar
and rejects nonfinite/out-of-domain values, latches failure and requires reset
before exposing output or applying further main-model feedback. This scalar
comparison is fail-closed orchestration, not host model computation. There is
no CPU/GPU/API sine or model fallback. No RTL or synthesis is introduced.

## Causal state, chunks and output interface

Causal convolutions retain (kernel-1)*dilation input rows; transposed convolutions
retain one activated input row. History is chronological, including when the
54-row dilation-9 history exceeds the current 32-row early-stage frame. At a
chunk restart every convolution uses fresh zero history, then consumes all
retained context frames. Normal frames preserve history; full reset clears
all waveform state. No residual output is substituted for a convolution's
input history. Transposed phase p consumes current input tap p and previous
input tap p+r; explicit strided Tensor descriptors store phase-interleaved output
using the existing compiler/ISA.

`WaveformChunkImage` / `WaveformChunkISASession` provide a recorded-code diagnostic
path. `WaveformChunkFrameImage` / `WaveformChunkFrameSession` attach the waveform
stage to neural-generated resident IDs, decoder and upsampling. The standalone
output is `desc.waveform`; the generated-code output is `desc.audio_waveform`,
shape `(1920,)`, float32, 24,000 samples/second. The existing device controller
saves/restores current IDs and recomputes retained IDs without host ID upload.
This increment adds no serving adapter, HTTP endpoint or UI enablement.

## Builder validation

All tests and model data remain on the existing `opentpu-m8a-build`, under
`/home/ubuntu/qwen3-tts-waveform-20261023-02`, using the existing Python 3.12.3 /
NumPy 2.5.4 venv, one BLAS/OMP thread, nice 19, 64-GiB virtual-memory cap and
bounded timeouts. The Mac is the source/Git/control/receipt bridge. No packages,
credentials, services, instances, new downloads or production files changed.

`wave-tests-04`: **4 passed in 37.58 seconds**. Independent NumPy references cover
all waveform layers, asymmetric transpose phases, causal dilation/history,
padded contractions, raw clamp, 5-frame 3/2 chunks, bitwise reset, domain rejection
and failure/reset behavior. A small neural producer exercises three frames with
1/1 chunk replay and restored current IDs. Native positions 0/25/324 compile
within the exact target limits. Earlier compile failures and one fixture count
assertion failure are retained in full with their fixes: stepped slicing became
an explicit strided Tensor; an unsupported flat Tile reshape became a contiguous
2-D Tensor destination; the tiny rate-2 fixture's 214 tap/parameter views were
distinguished from the native model's 238. Numerical bounds were unchanged.

`wave-native-01`: **exit 0 in 75.12 seconds**. One retained prefix codec frame
passes through the full actual-checkpoint decoder and waveform stage, followed
by a bitwise-equal reset replay. Independent full-sequence FP32 and FP64 source
references check every one of the 59 operation outputs plus all 12 pre-residual
branches, raw samples and clamped samples. Maximum relative L2 across operation
boundaries is **0.130505223% FP32 / 0.130501431% FP64**; raw and clamped waveform
error is **0.083438226% FP32 / 0.083474462% FP64**, below the unchanged 3.5% bound.
Maximum absolute waveform error is 1.422e-6 against FP32. The first frame's RMS
is 0.00050206, with no clipped samples; this short prefix is not audible-utterance
quality evidence. Maximum device sine argument is 7.14038, within the domain.

`wave-integrated-01`: **exit 0 in 264.96 seconds** (264.80 harness seconds).
The exact main checkpoint consumed the 13-row source-format `Hello.` prefill,
generated two codec frames, and fed their resident 16-ID rows into the full
waveform path with diagnostic 1/1 chunks. All **32/32 IDs** matched the preserved
FP32 golden path, and current IDs were restored after context replay. Maximum
operation/raw/waveform relative L2 was **0.155928060% FP32 / 0.155934337% FP64**.
Full reset regenerated the first frame's 16 IDs and waveform bitwise. The second
frame explicitly exercised context recomputation through the complete decoder,
upsampling and waveform stack. The references recompute source-format chunks
independently, including all waveform context, before trimming. These 160 ms of
prefix audio are not a complete utterance or speech-quality test.

The separate public-boundary acceptance commit `b1987014` validates 301 actual-
checkpoint recorded-code decoder/upsampling frames, including the 300/25 boundary,
state poisoning and reset. Maximum output error is 0.0181051% against FP64.
**A native 300-frame boundary including waveform is still unvalidated.** Short
waveform chunk tests establish composition semantics; they do not substitute for
that long native run. Full native main generation through frame 300 and complete
utterances remain separate work, subject to the existing 512-position limit
including prefill. No speech quality or physical throughput is inferred from
reference agreement or ISA-simulator runtime.

## Shared image, resources and remaining gates

Only AGFI `agfi-0001de1dd3df8e21e`, profile `f2-hbm:p4rl8a64c150`, build
`417243f1`, configuration hash `dd6b92a6`, runtime
`30100afb075374381601a6e01d9f914a58dbec91` is qualified in software. This does
not assert opcode equivalence with burst200/other images. No live query or
physical card access occurred. Apply these component files over that pinned
runtime: the component branch's older compiler is not a substitute.

The waveform stage adds **79 programs per decoded codec frame**: 29 SnakeBeta,
30 convolution/transpose, 18 history preparations, begin and finish. Programs
loop over four output rows, avoiding one host dispatch per audio sample. Native
program maxima are **1,104 IMEM words / 61,449 TMEM words**, below 32,768 / 65,536.
Only 4,087 TMEM words remain at the worst point, so future fusion/batching must
requalify memory. Standalone full decoder residency is **294,402,304 bytes**;
waveform adds 144,324,992 bytes to the preceding upsampling image.
Combined main/decoder/waveform residency is **2,968,816,128 bytes**, leaving
**1,326,151,168 bytes** of 4 GiB. Shared identity reuse makes the combined stage
increment 144,176,000 bytes. The preceding stages' maximum IMEM remains 3,088
words; waveform raises the overall TMEM maximum to 61,449 words. Ordinary frame
dispatch grows from 554 to 633 programs including the main path. A full native
public-boundary dispatch is 6,586 programs including all 25 waveform replays;
these are software schedule counts, not measured card latency.
Public boundary replay executes waveform on all 25 retained frames as well as
the current frame; instruction counts are not hardware speed measurements.

The upstream BF16 acceptance remains failed at **127/144 IDs, 6/9 complete
frames and 7.5136935% maximum conditional predictor error**. The historical failed
receipts and unchanged 3.5% criteria are preserved. Passing FP32-source-algebra
component comparisons does not relabel that overall gate. Full TTS remains
disabled. Remaining gates are longer generated waveforms/EOS/complete utterances,
actual native public-boundary waveform validation, numerical/quality assessment,
and coordinated same-image physical acceptance. No missing ISA primitive is
currently demonstrated within the qualified sine domain; out-of-domain sine
requires a new device-only lowering decision, never host substitution.

The [evidence directory](qwen3_tts_waveform_evidence/) preserves every failed and
passed test, actual-checkpoint rows, requests/results, runner, source hashes,
probe, environment and post-run verification. All 166 source snapshot files
match: 95 runtime files against Git `30100afb`, and 71 component/fixture files
against this tree. Both 58-entry acceptance source manifests, the two-entry sine
probe manifest, comparison-row hashes and archive artifacts were verified.
Original failed-source overlays and all three failure receipts remain retained.
Every builder job is terminal; inspect saved results before any rerun. Service
interruption was zero;
the user's 30-second maximum remains. No FPGA, image or serving changes occurred.

```sh
/home/ubuntu/otpu-venv/bin/python -m pytest -q tests/test_qwen3_tts_waveform.py
/home/ubuntu/otpu-venv/bin/python tools/qwen3_tts_snake_probe.py --data /home/ubuntu/qwen3-tts-waveform-20261023-02/data --output /path/to/new-snake-receipt.json
/home/ubuntu/otpu-venv/bin/python -u tools/qwen3_tts_waveform_acceptance.py --data /home/ubuntu/qwen3-tts-waveform-20261023-02/data --output /path/to/new-waveform-receipt.json --frames 2 --integrated --reset
```
