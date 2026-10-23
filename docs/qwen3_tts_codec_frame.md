# Qwen3-TTS complete codec-frame compiler slice

This increment adds the five-layer code predictor, all 15 predictor heads and
embedding tables, device greedy selection, the 16-codebook embedding sum, and
feedback into the 28-layer talker. It preserves `6f029c63` and every historical
failed receipt. Full waveform TTS remains disabled. There is no production
transport, FPGA execution receipt, audio decoder or speech-quality claim here.

## Pinned source and data

The model is exactly
[`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tree/85e237c12c027371202489a0ec509ded67b5e4b5),
revision `85e237c12c027371202489a0ec509ded67b5e4b5`. The checkpoint is
1,811,626,576 bytes, SHA-256
`bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`.
The data-only reader rechecks that digest before execution. The complete 402
checkpoint tensors, 905,788,672 BF16 elements, match the resident descriptor's
canonical name/shape/dtype hash in `tests/fixtures/qwen3_tts_frame_contract.json`.
The model's “0.6B” label is not its complete packaged parameter count.

The source is Qwen-TTS 0.1.1 at
[`022e286b98fbec7e1e916cb940cdf532cd9f488e`](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py),
under [Apache-2.0](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/LICENSE).
Upstream custom executable model code was read only. No upstream executable
model code, remote model loader or pickle was imported or run.

The independent references implement the reviewed source algebra with ordinary
locally installed Torch operators. They are diagnostic CPU programs, never
runtime fallback paths. BF16 reference results describe this explicit eager
algebra, not a claim of bitwise identity to every upstream SDPA/CUDA kernel.

## Device sequence and interface

`CodecFrameImage` extends the existing talker image with a separate resident
predictor hidden vector, five distinct layer weight/KV sets, 15 INT8 2048-row
heads, 15 FP32 2048-row embedding tables, a 16-ID output buffer and an FP32 sum.
The native widths are both 1024, making the source's `small_to_mtp_projection`
an identity. The predictor's 1D rotate-half RoPE has theta 1,000,000 and matches
the existing equal-axis talker lowering numerically; its configuration is
checked explicitly rather than inferred from the talker name.

After the source-format non-streaming CustomVoice prefill:

1. Device argmax selects the primary code under fixed source suppression.
2. Predictor position 0 consumes the final normalized talker hidden state.
3. Position 1 consumes the primary code's talker embedding and head 0 predicts
   the second codebook ID. Each following predictor position consumes the
   preceding codebook's own embedding and uses the next head.
4. Every ID is written in device memory. The corresponding embedding is loaded
   with ISA RLD addressing and added to the device-resident sum. The last
   predicted code's embedding contributes to the sum without another predictor
   transformer step. There are exactly 16 predictor positions, not 17.
5. The device adds projected TTS text padding to the sum, runs all 28 talker
   layers and the codec head, leaving the next frame's logits/hidden resident.

`predictor_parts(position, head)` and `feedback_parts(talker_position)` return
sequential ISA programs with per-part bounds. `CodecFrameISASession` is an
explicit CPU ISA diagnostic session, with no card opener or production route.
Its `frame()` returns the 16 IDs or `None` on an already device-selected EOS.
It performs no host learned arithmetic or token selection. Output reads are
diagnostic/transport only. After an execution failure it requires explicit
reset before reuse; no retry or backend substitution is hidden.

The supported diagnostic generation configuration is explicit:
`do_sample=False`, `subtalker_dosample=False`, `repetition_penalty=1.0`,
`min_new_tokens=2`. The first two primary choices suppress EOS on device. This
is a supported upstream configuration, not the sampled default (temperature
0.9, top-k 50, repetition penalty 1.05). Stochastic sampling, that repetition
penalty, batched generation and streaming text assembly are not implemented by
this slice. The text prompt, speaker/language controls and style restrictions
remain those documented in the prior prefill slice.

## Cache and host dependency audit

Predictor KV positions restart at zero for every frame, with D=128 allocated
positions per layer and only positions 0..15 visible. Every visible entry is
overwritten before attention uses it; exact causal lengths mask the stale tail
and the preceding frame. Talker KV continues across feedback frames. Diagnostic
reset clears all KV and both position counters; embeddings, constants and
weights stay resident. Poison/reset tests check this behavior independently.

The host performs offline tensor verification/packing, constant RoPE preparation,
BPE tokenization, deterministic prompt/control scheduling, program launch and
EOS control branching. Its EOS branch reads the ID selected by a device program;
it does not compute logits or choose the token. FP32 embedding lookup, projection,
all 28+5 transformer layers, normalization, heads, greedy reductions, codebook
sum and feedback are ISA operations. FP32 embeddings/norms/biases/residuals/RoPE
and INT8 matrices/activations/KV are explicit; no mixed precision extension was
silently substituted. CPU reference and ISA simulation are acceptance tools.

Full `synthesize` remains fail-closed. There is no host neural audio decoding,
CPU/GPU generation, external API, remote-code execution or production fallback.

## Shared image and resource limits

The required target remains `agfi-0001de1dd3df8e21e`,
`f2-hbm:p4rl8a64c150`, build `417243f1`, configuration hash `dd6b92a6`.
Qualification uses runtime `30100afb075374381601a6e01d9f914a58dbec91` in an
isolated EC2 clone. The older runtime in this component's base cannot represent
the wide-port profile and intentionally fails closed. Publisher integration
must use the qualified runtime or an independently verified compatible one.

`qwen3_tts_frame_profile_audit.json` records descriptor-only compilation:

| Talker cache capacity | Resident bytes | KV bytes | Free within 4 GiB | Max part IMEM words | Peak TMEM words |
|---|---:|---:|---:|---:|---:|
| 512 | 1,981,437,824 | 31,629,312 | 2,313,529,472 | 3,096 | 15,160 |
| 32,768 | 3,905,443,712 | 1,939,120,128 | 389,523,584 | 5,544 | 15,160 |

Each frame uses 143 sequential program launches. Aggregate IMEM is 291,928 words
at talker position 511 and 360,472 at position 32,767; the program is deliberately
split because those totals exceed IMEM. Every individual part fits the 32,768
word IMEM and 65,536 word TMEM. The opcode and actual MM/QACT flag sets are
recorded in the audit. There is no newly required ISA primitive for this path.
The 389,523,584-byte maximum-context margin does not include the waveform decoder
or any future higher-precision remedy. No spill/offload is allowed.

The 143 launches and 16 predictor passes through five layers are material
throughput costs. Software simulator duration is not FPGA performance evidence.
This layout begins at logical device address zero and assumes its own model
residency; it does not reserve space alongside the current Gemma session. Future
integration must coordinate the allowed weights/programs switch on the same
image and prove the serving interruption bound. No resident Gemma data was
read, evicted or overwritten by this software work.
No real-time audio claim is made. No card, live registers, ownership lock, image,
service or deployment was accessed or changed. Serving interruption was zero;
the user's 30-second maximum remains a requirement for future coordination.

## Acceptance and remaining gates

The bounded real-checkpoint tool uses the three pinned valid source prompts
`Hello.`, `Good morning.`, and `The FPGA speaks.`, with up to three generated
frames each. It retains source-format prefill diagnostics separately, records
every generated code and per-head logit error, and compares independent FP32
and BF16 free-running sequences. It also teacher-forces the device's actual
hidden state and selected prefix into independent predictor algebra, isolating
conditional predictor numerical error from accumulated talker/ID divergence.
Those conditional rows are diagnostic, not substitute speech-quality evidence.

Acceptance retains relative logit L2 <= 0.035 and complete generated greedy ID
agreement, with matching EOS decisions and exact poisoned/reset replay. Historical
arbitrary aligned-row failures are not reclassified as meaningful output rows.
The unchanged legacy receipt remains in `qwen3_tts_numerics_legacy.json`, including
its 10/16 FP64 and 12/16 BF16 masked-greedy failures. No threshold is relaxed.

The completed EC2 validation passed **52 tests in 31.31 seconds**, including all
46 prior tests. The suite retains 35-step block-boundary/cache tests and native
maximum-context compilation. New checks cover the five-layer/15-head sequence,
all 16 embeddings including the final code, greedy ties/masking/minimum EOS,
no predictor/KV update on EOS, poisoned frame-local caches, reset replay,
execution-failure latching, the complete checkpoint tensor contract and native
IMEM/TMEM/residency. Real-weight generated sequences here are short; this is not
a real-weight 32,768-position or naturally encountered EOS acceptance claim.

The final run's durable full receipt is `qwen3_tts_frame_acceptance.json`
(229,919 bytes, SHA-256
`db07ef058bca6dfdcfb739ce55fb2690ae0f1c05947b477f85af1364e192bcfa`).
All 42 source-prefill rows are retained separately in
`qwen3_tts_frame_prefill.json` (23,228 bytes, SHA-256
`aa5ed0f2e9dc8f4180b800cf8b5865a6151c65f32ee35bb79b3a32812accf47c`).
`qwen3_tts_frame_validation.json` records the SSM command, runtime, tests,
checksums, earlier reporting defects and unchanged historical-receipt checksum.
Both receipt bytes and every recorded tested-source hash were verified locally.

**Numerical acceptance failed, exit 2.** There are nine generated frames/144
IDs, not merely prefill diagnostic rows:

| Generated-output measurement | FP32 reference | BF16 reference |
|---|---:|---:|
| Independently generated IDs matching ISA | 121/144 | 123/144 |
| Complete matching frames | 3/9 | 3/9 |
| Predictor greedy decisions, independent full sequences | 112/135 | 114/135 |
| Predictor greedy decisions under identical ISA hidden/ID prefixes | 125/135 | 127/135 |
| Maximum predictor logit relative L2, independent sequences | 36.9503% | 31.0127% |
| Maximum predictor logit relative L2, identical-prefix diagnostic | 8.8992% | 9.2039% |
| Maximum subsequent talker codec-logit relative L2 | 10.0528% | 11.0454% |
| Identical-ID embedding-sum relative L2 | 0% | 0.18953% maximum |

Every meaningful final-prefill decision still matches both references (3/3).
The non-output source-prefill masked-greedy counts remain 33/42 FP32 and 34/42
BF16. Neither these rows nor the legacy arbitrary aligned rows are substituted
for generated-output acceptance. The poisoned-cache vs reset replay produces
identical generated IDs and bitwise identical final logits. All nine EOS
decisions agree, with no natural EOS among these short generated sequences.

The identical-ID FP32 embedding sum is exact, so the new accumulation path is
not responsible for that reference disagreement. Predictor error persists
under identical device hidden states/IDs; accumulated free-running differences
are larger. This separates the local predictor numerical issue from the later
effects of diverging token histories, but does not yet prove whether every
predictor error is quantization rather than a lowering defect. The reused
transformer arithmetic was independently checked in the previous talker slice;
a predictor-specific exact-MX replay is the smallest next isolation step.

FP32 and BF16 references themselves agree on only 127/144 independently
generated IDs and 129/135 identical-prefix predictor decisions. Exact agreement
with both is therefore impossible on all these rows. This is reported as an
oracle-precision issue to resolve explicitly, not used to relax the unchanged
gate; the ISA path currently fails the numeric bound against either reference.

After exact replay, a residual-INT8 or selective FP32-vector lowering on this
same ISA is a candidate precision remedy, requiring its own numeric and resource
proof. Native MM exposes INT8/INT4/FP4/NVFP4, not FP16/BF16 matrix arithmetic.
The 512-position layout has 2.31 GB free; maximum-context free space is only
389.52 MB. Additional weights/KV and passes must be counted before selecting a
remedy. No such remedy, speed result or new RTL is claimed by this increment.

Actual generated-code parity is the next gate before any physical validation.
After numerical acceptance, remaining stages include RVQ/audio decoder execution
(transformer, causal/depthwise/transposed convolutions, ConvNeXt/LayerNorm,
exact-GELU and SnakeBeta), finite error bounds for nonlinear lowerings, complete
text-to-waveform acceptance, strict serving integration and coordinated physical
acceptance on the same image. Current ISA has no native sine or erf; a software
lowering or other explicit implementation decision must be validated before
claiming those decoder stages. No synthesis or image change is proposed here.
