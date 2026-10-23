# Qwen3-TTS corrected talker, projection and codec-frame experiment

This separate increment preserves predictor commit
`a9c02e7aa8fe07d40a9fe1b140a3a299b9785ad5` and its failed full-sequence receipt.
It extends the measured existing-ISA precision remedy to the text projection,
all 28 talker layers and primary codec head. The predictor's corrected path,
all 16 codebook lookups/accumulation and feedback are retained. No waveform
decoder, production route or hardware readiness is implied. TTS remains disabled.

## Arithmetic and execution contract

`TalkerPrecisionFrameImage` is an optional resident-memory layout, selected only
by `tools/qwen3_tts_frame_acceptance.py --talker-precision`. It uses the same two
INT8 weight terms and two on-device activation terms described in
`qwen3_tts_predictor_precision.md`. Offline weight residual packing is allowed;
activation reconstruction, subtraction, four MM terms, all attention/norms,
nonlinearities, heads, argmax, lookups and feedback are device programs. There is
no host activation correction or reference hidden state fed into inference.
All 402 original learned tensors remain resident; derived terms do not alter the
source tensor contract. This is not a native FP16/BF16 or exact FP32 MM format.

Talker K/V is stored in FP32 in addition to the unused baseline INT8 cache. To
bound program/scratch size, each layer is split into Q/K/V projection and cache
write, one program per KV attention head, then output projection/MLP/residual.
Head scores are assembled from 64-row key tiles in original token order, followed
by FP32 softmax and a deterministic sequential weighted-value sum. No attention
window or future token is substituted; every visible causal token is used.

Text-only assistant-role rows, aligned text/codec rows and generated feedback
each have explicit corrected projection programs. The primary codec head also
uses corrected matrix products. Fixed suppression, minimum two generated tokens
before EOS, device argmax and the explicit greedy configuration are unchanged.
Host BPE, prompt scheduling, launch/control, offline packing/constants and output
transport remain orchestration. No CPU/GPU/API inference fallback is connected.

## Bounded context and shared-image resource evidence

The corrected talker mode explicitly supports at most **512 resident positions**,
including prefill and generated feedback. Oversized context is rejected; it is
never truncated, windowed or offloaded. The source model's 32,768-position maximum
is not claimed for this mode. The original INT8 and predictor-only modes retain
their unchanged maximum-context tests and descriptors. FP32 talker K/V at the
full source context would itself exceed the current 4 GiB address space, so a
larger corrected-context contract needs a separately measured representation.

The image remains `agfi-0001de1dd3df8e21e`, profile
`f2-hbm:p4rl8a64c150`, build `417243f1`, configuration hash `dd6b92a6`, runtime
`30100afb075374381601a6e01d9f914a58dbec91`. No bitstream, synthesis, card query,
ownership lock, service or deployment was touched. The word “Image” in the Python
class denotes memory layout only. Weight/program switching would require future
parent coordination on the same already-loaded FPGA image, with the user's
maximum 30-second serving interruption bound; actual interruption here is zero.

`qwen3_tts_talker_precision_profile.json` records descriptor-only native limits:

| Resource at capacity 512 | Measured value |
|---|---:|
| Resident device bytes | 2,675,611,520 |
| Additional bytes over corrected-predictor mode | 572,211,200 |
| Free bytes within 4 GiB | 1,619,355,776 |
| FP32 talker K/V bytes | 117,440,512 |
| Programs per prefill row | 282 |
| Programs per generated codec frame | 395 |
| Largest representative program, context position 511 | 25,224 IMEM words |
| Largest representative TMEM peak | 28,737 words |

Representative maximum-context programs are measured for input projection,
feedback, Q/K/V, one KV head, output/MLP and codec head. All 28 talker layers have
the same program geometry, with different resident addresses. The 32,768-word
IMEM and 65,536-word TMEM limits are unchanged. The existing corrected predictor
programs also fit their previously measured limits. There is no new ISA opcode
or RTL requirement for this slice. Launch count, repeated matrix passes and FP32
attention are substantial costs; no FPGA throughput or real-time audio claim is
made from simulator durations or these counts.

The first maximum-context compile exposed TMEM fragmentation with 128-row key
tiles while the synthetic source/cache test passed. Reducing key tiles to 64
fixed the demonstrated allocation failure without changing token order,
arithmetic within a key row, context or numerical thresholds. Both focused tests
then passed in 48.13 seconds. They cover source-reference error through 35
positions, crossing the tiny target's block boundary; generated multi-frame
feedback; poisoned caches; reset equivalence; maximum native program limits;
and explicit oversized-context rejection. This is not a real-weight or physical
512-position execution receipt.

## Provenance and acceptance boundary

The model remains exactly
[`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tree/85e237c12c027371202489a0ec509ded67b5e4b5),
revision `85e237c12c027371202489a0ec509ded67b5e4b5`, with the verified
1,811,626,576-byte BF16 checkpoint SHA-256
`bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`.
The data-only reader rechecks that digest. Source algebra and Apache-2.0
provenance remain pinned to
[`022e286b98fbec7e1e916cb940cdf532cd9f488e`](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py).
No upstream executable model code or remote model loader is imported or run.

The independent FP32/BF16 source references and the valid source-format corpus
are unchanged: `Hello.`, `Good morning.`, `The FPGA speaks.`, three greedy codec
frames per prompt. These are only three prefix frames (0.24 seconds at 12.5
frames/s) per prompt, not complete spoken utterances. There is no decoded
waveform or intelligibility/speaker-quality evidence here. Meaningful generated
output comparisons stay separate from non-output prefill diagnostics and all
historical failures remain intact. The relative-logit-L2 bound remains 3.5%,
with complete greedy agreement, matching EOS decisions and bitwise reset/poison
replay. Both FP32 and BF16 comparisons remain visible; no arithmetic-oracle
decision or threshold relaxation is hidden.

## Completed real-weight result

Command `adbf1122-b4f4-4c4d-a7a2-105801582b8e` ran once on the existing
`opentpu-f2` instance. **58 tests passed in 102.32 seconds.** The numerical
diagnostic took 671.02 seconds; that duration is not FPGA performance evidence.
Full receipts are `qwen3_tts_talker_precision_acceptance.json` (232,713 bytes,
SHA-256 `d553b53081fcb0ff2f4bbfa325ebd53944cc590ea87c84f8dbc32913ba6abc1d`)
and `qwen3_tts_talker_precision_prefill.json` (23,580 bytes, SHA-256
`053ce75b9fc63ab0cf07dc472710020ac670abbacb62b26aab4cda0e08dca246`).
Receipt bytes, all recorded tested-source hashes, original thresholds and both
independent references' complete sequences were verified after retrieval.

| Measurement | FP32 reference | BF16 reference |
|---|---:|---:|
| Independently generated IDs matching ISA | **144/144** | 127/144 |
| Complete matching frames | **9/9** | 6/9 |
| Predictor decisions matching across independent sequences | 135/135 | 118/135 |
| Identical-prefix predictor decisions matching | 135/135 | 127/135 |
| Maximum generated predictor logit relative L2 | **0.0271066%** | 36.64298% |
| Maximum identical-prefix predictor logit relative L2 | **0.0198885%** | 7.51369% |
| Maximum subsequent talker codec-logit relative L2 | **0.0267498%** | 7.02542% |
| Exact FP32 embedding sum | Yes | BF16 sum differs up to 0.18412% |
| Source-prefill masked-greedy decisions matching | 42/42 | 37/42 |

All nine EOS decisions match both references. Cache poison/reset replay produces
identical generated IDs and bitwise identical final logits. The FP32 generated
sequence and numerical bounds now pass for the entire bounded corpus, improving
from 121/144 IDs before correction and 134/144 with predictor-only correction.
This fixes the demonstrated FP32 source-format codec-frame numerical defect.

**Overall acceptance still fails, exit 2, accepted=false.** The unchanged gate
requires both reference comparisons. FP32 and BF16 source references disagree
on 17/144 generated IDs; matching all FP32 IDs therefore cannot also match every
BF16 ID. Both references, every failed comparison and the unchanged 3.5% bound
remain in the receipt. An explicit arithmetic-oracle/representation contract is
needed before treating this as a deployable precision policy. No criterion was
weakened, and no waveform, full-sentence quality or physical readiness follows.

## Builder preference and remaining work

The user requested `opentpu-m8a-build` after this job had started. Read-only EC2
inventory verified `i-0ec8d33362b3100be`, m8a.16xlarge, running in us-east-1.
It was absent from SSM inventory; no agent, IAM, security-group or registration
change was attempted. Its existing SSH route was verified using the existing
key and trusted host-key record with strict checking. Read-only preflight found
64 CPUs, ample memory/disk, an existing NumPy 2.5.4/pytest 9.1.1 environment, and
no ancestor AGENTS.md. Torch and runtime commit 30100afb were absent from the
checked environments/repositories; this is not a claim that every directory was
searched. Subsequent compute/data should use an isolated builder workspace with
verified source/dependencies. The Mac remains the available CLI/Git control
bridge; no registered remote or credentials changed. The existing F2 job was
left intact and was not migrated or duplicated.

The next waveform increment can begin with decoder RVQ lookup/normalization,
separate semantic/acoustic projections and their sum, then the audio transformer,
causal/depthwise/transposed convolution blocks and waveform head. Reviewed decoder
source also requires LayerNorm, exact-GELU and SnakeBeta; the current ISA lacks
native erf/sine, so validated bounded software lowerings are needed. No RTL or
new image is proposed. Decoder weights/source parity, causal boundary/reset
behavior, full utterances and end-to-end physical/serving acceptance remain
outstanding. Full TTS continues to fail closed; no UI or production route changed.
