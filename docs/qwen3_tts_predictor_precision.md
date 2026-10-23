# Qwen3-TTS predictor replay and device precision experiment

This isolated increment preserves `0a99508d` and all earlier failed receipts.
It first checks the original predictor lowering with exact-MX replay, then tests
an optional device-only precision remedy against the same real-checkpoint
generated-frame acceptance. It does not enable TTS, change a serving route,
access the FPGA or change its image. Here “Image” in Python class names means a
resident memory layout, not a new FPGA bitstream.

## Exact replay establishes the exercised arithmetic

`PredictorExactReplay` runs locally reviewed operator algebra without importing
the compiler, kernels or ISA interpreter for its arithmetic. It shares the
documented FP32 primitives and offline MX packer, so it is an arithmetic
isolation tool, not an independent quality oracle. `ExactMX.hidden_step` simply
exposes the existing independent transformer algebra for predictor inputs; the
original talker entry point keeps its arithmetic/order.

With the pinned checkpoint and valid `Hello.` CustomVoice prompt, all **160
layer outputs and 30 codec heads** across two generated frames match the original
ISA path **bitwise**. This includes predictor cache restart for the second
frame. There are no mismatching rows among those 190 comparisons. The replay
exercises all five predictor layers and all 15 heads twice. This rules out a
lowering discrepancy in those exercised paths, not every possible input.

The full receipt is `qwen3_tts_predictor_exact.json`, 41,844 bytes, SHA-256
`72f7feb74e4ac1e9b1fdab1715ff13a1321ff570c6d98327f85a20cd75e88994`.
The run took 69.78 seconds of CPU/ISA diagnostics. The preceding regression
suite passed 53 tests in 38.38 seconds. The generated IDs match the preserved
baseline, including its numerical disagreement with source-format FP32/BF16
algebra; replay is not a replacement for that failed acceptance.

## Existing-ISA precision remedy

`PredictorPrecisionImage` is opt-in through the diagnostic tool's
`--predictor-precision` flag. The original `CodecFrameImage` path remains the
default, and `synthesize` remains fail-closed. The experiment changes only the
five-layer predictor and its 15 codec heads. The talker, text projection and
primary codec head retain their existing INT8 path.

Each predictor affine weight is represented by two resident INT8 terms. Offline
packing computes the second term from the source FP32-expanded checkpoint minus
the first term's represented weight. The host never computes activation
residuals during inference. Each normalized activation is quantized on device;
an unscaled integer identity MM reconstructs that operand's unit-weight MM
effect, and device VOP subtraction forms the second activation term. Four
existing MM operations then sum the high/high, high/low, low/high and low/low
products in FP32. The identity pass is another existing MM operation.

This is not a native INT16/FP16/BF16 matrix format. The MX shift window still
exists, including its interaction between weight and activation group offsets;
the experiment's accuracy must be measured rather than inferred from a nominal
bit count. Native MM formats remain INT8, INT4, FP4 and NVFP4.

The predictor's short per-frame K/V cache is also retained in FP32. Q.K, softmax
and P.V use existing LD/ST and FP32 VOP reductions/multiply/add/exp2/reciprocal.
The prefix is at most 16 positions, so bounded row loops fit without a new
attention opcode. Source Q/K normalization, rotate-half RoPE, final norm and
head order are unchanged. Generated IDs, all embedding lookups, sum and talker
feedback remain device operations. Host reads remain diagnostic/transport or
EOS orchestration; there is no host activation correction or backend fallback.

New tests check nonzero-head exact replay, a residual affine's error reduction,
the corrected predictor against independent identical-input FP32 algebra,
poisoned caches across frames, and native maximum-context residency and program
bounds. Four focused tests passed in 24.04 seconds before real-weight execution.
These synthetic tests do not establish speech quality.

## Measured memory and instruction costs

The target remains exactly `agfi-0001de1dd3df8e21e`, profile
`f2-hbm:p4rl8a64c150`, build `417243f1`, hash `dd6b92a6`, using runtime
`30100afb075374381601a6e01d9f914a58dbec91`. No generic F2 or alternate image is
substituted. `qwen3_tts_predictor_precision_profile.json` records actual compiled
opcodes, VOP functions, MM/QACT flags, memory and instruction counts.

The additional 121,962,496 bytes comprise 111,329,280 bytes of residual matrices
and scales, 5,390,336 bytes of integer identity constants, and 5,242,880 bytes of
FP32 predictor K/V. Original predictor INT8 KV remains allocated to preserve the
baseline layout; the precision path does not read it. Both cache sets are
included in reset and poison tests. All learned source tensors remain resident.

| Talker capacity | Total resident bytes | Free within 4 GiB | Aggregate IMEM words/frame | Largest program | Peak TMEM words |
|---|---:|---:|---:|---:|---:|
| 512 | 2,103,400,320 | 2,191,566,976 | 625,448 | 9,560 | 20,181 |
| 32,768 | 4,027,406,208 | 267,561,088 | 693,992 | 9,560 | 20,181 |

There are still 143 sequential programs per frame. At talker position 511,
MM instructions increase from 5,218 to 11,598 and LD instructions from 1,932 to
12,812; aggregate IMEM increases from 291,928 to 625,448 words. These are material
costs. There is no measured FPGA latency, throughput or real-time audio claim.
The maximum-context free space does not include a waveform decoder or any
further talker precision remedy. Layouts begin at logical device address zero
and require coordinated model residency; no current Gemma data was read,
evicted or changed. Serving interruption is zero in this software work; the
future maximum remains 30 seconds.

## Provenance and acceptance boundary

The model remains exactly Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice at revision
[`85e237c12c027371202489a0ec509ded67b5e4b5`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tree/85e237c12c027371202489a0ec509ded67b5e4b5).
The 1,811,626,576-byte checkpoint's LFS SHA-256
`bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`
is rechecked before execution. All 402 original BF16 tensor names/shapes remain
the same; extra residual terms are derived offline, not new source parameters.
Source algebra and Apache-2.0 provenance remain pinned to
[`022e286b98fbec7e1e916cb940cdf532cd9f488e`](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py).
No upstream executable model code is imported or executed.

Real-weight acceptance retains the three valid non-streaming CustomVoice
prompts, three generated frames each, explicit greedy configuration, 3.5%
relative-logit-L2 bound and complete generated-ID agreement. Identical-prefix
predictor diagnostics remain separate from independently generated sequences.
Neither a corrected conditional predictor nor exact-MX replay may replace
end-to-end generated-frame acceptance. FP32/BF16 oracle differences and the
unchanged INT8 talker remain explicit potential blockers. Full waveform TTS
stays disabled until its decoder and full end-to-end acceptance are implemented.

## Completed real-weight result

The original validation command `6185de96-7657-45c4-b678-333ef184a043` finished
during the local execution-environment outage. It was recovered by reading that
same command and its existing files, without resubmitting model computation.
**56 tests passed in 53.39 seconds.** The completed numerical run took 419.71
seconds of CPU/ISA diagnostics and returned **exit 2, accepted=false** under the
unchanged thresholds. The full receipt is
`qwen3_tts_predictor_precision_acceptance.json` (231,400 bytes, SHA-256
`969680e0a9a558107a159fb9d1ac376531548447e9f7a6fd4344647d7de81b2c`).
`qwen3_tts_predictor_precision_validation.json` records the recovered command,
tests and verification. All recorded tested-source hashes match the local
worktree. The 42 prefill rows are byte-identical to the preserved
`qwen3_tts_frame_prefill.json`, and both independent references' generated
sequences are unchanged from `0a99508d`; this is the same corpus and oracle.

| Measurement | Prior INT8 FP32 / BF16 | Corrected predictor FP32 / BF16 |
|---|---:|---:|
| Independently generated IDs matching ISA | 121/144 / 123/144 | 134/144 / 130/144 |
| Complete matching frames | 3/9 / 3/9 | 6/9 / 6/9 |
| Identical-prefix predictor greedy decisions | 125/135 / 127/135 | 135/135 / 129/135 |
| Maximum identical-prefix predictor logit relative L2 | 8.8992% / 9.2039% | 0.0214245% / 4.82404% |
| Maximum independent-sequence predictor logit relative L2 | 36.9503% / 31.0127% | 27.71696% / 36.48475% |
| Maximum subsequent talker codec-logit relative L2 | 10.0528% / 11.0454% | 7.47727% / 10.66005% |

The FP32 predictor numerical bound and greedy decisions are fixed for all 135
identical-input real-weight decisions in this corpus. This is measured source
parity at the predictor boundary, not full-model acceptance. Those diagnostics
take the actual device talker hidden state and actual generated prefix as inputs
to the independent reference; they never feed reference hidden states back into
the device program. The predictor's FP32 embedding sum remains exact. All nine
EOS decisions agree, and poisoned-cache/reset replay retains identical IDs and
bitwise identical final logits.

Meaningful free-running output remains blocked: the unchanged INT8 talker and
text projection still supply perturbed hidden states/KV to the corrected
predictor. Its complete generated sequences still miss 10 FP32 IDs and 14 BF16
IDs. The larger independent-sequence predictor error includes the effects of
those differing inputs and prefixes; it is not hidden by the corrected
conditional result. The BF16 identical-prefix reference differs from the FP32
reference on six greedy decisions and reaches 4.82404% logit error. Agreement
with both arithmetic oracles is therefore not simultaneously achievable on
those decisions. Neither issue is used to loosen the gate: accepted remains
false against the original full-sequence criterion.

The next independent increment should extend a measured precision remedy to
the talker/text projection and test complete generated-frame parity, with an
explicit bounded context/resource contract. The FP32/BF16 oracle precision
decision also remains explicit. Waveform decoder work must retain a separate
reviewable increment; this result is not authorization to enable TTS or claim
audio quality, full-waveform inference or hardware readiness. No image,
hardware, production, credentials or network service changed during recovery.
