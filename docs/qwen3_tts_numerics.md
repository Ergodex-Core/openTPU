# Qwen3-TTS: source prefill fix and shared-image numerical diagnosis

This isolated follow-up preserves failed-result commit `5ee48c27` and its
receipts. It fixes a demonstrated input-assembly defect, separates numerical
sources on actual checkpoint weights, and qualifies the talker programs against
the required shared Gemma image in software. It does not enable TTS or claim
that the unchanged all-row acceptance gate passed.

## Fixed source contract

The pinned [generation source](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py)
constructs the first three assistant-role rows from **projected text only**.
The earlier aligned-row interface always added a codec embedding. It also
accepted arbitrary text/codec row pairs, which were never a valid CustomVoice
prompt. Such rows must not be interpreted as speech-quality evidence.

`custom_voice_prefill` now reproduces the source's fixed-speaker,
`non_streaming_mode=True`, no-instruction schedule. It checks the assistant
wrapper, emits three `role` rows, then language/speaker/control rows, the text
body plus TTS EOS over codec padding, and TTS padding over codec BOS. Auto
language and the source's Chinese speaker-dialect override are covered by tests.
The 0.6B variant still rejects style instructions through its existing request
contract. No cloning or streaming-generation capability is implied.

The `role` device program performs text lookup/projection with no codec lookup,
then passes its resident output through layer 0. Ordinary `aligned` rows retain
the original compiled path. `compile_parts(position, input_kind)` and the
explicit diagnostic session accept these two row kinds. All learned lookup,
projection, mixing, layer and head computation remains in device programs.
Tokenization and scheduling are deterministic host control, not host NN math.

The three texts are `Hello.`, `Good morning.`, and `The FPGA speaks.`. Their
source wrapper and body IDs are retained in `tests/fixtures/qwen3_tts_prompts.json`.
The installed built-in Qwen2 tokenizer ran locally on EC2 with pinned data and
`local_files_only=True`; no Qwen executable model code was imported. The
vocabulary, merges and tokenizer config were checked against official Git blob
IDs at model revision `85e237c12c027371202489a0ec509ded67b5e4b5`. SHA-256 values are
in the fixture. The source revision remains
`022e286b98fbec7e1e916cb940cdf532cd9f488e`, package 0.1.1, Apache-2.0. The full
checkpoint's previously verified LFS checksum is checked again before execution.

## Exact shared image qualification

Required image: **agfi-0001de1dd3df8e21e**, profile
**f2-hbm:p4rl8a64c150**, build `417243f1`, configuration hash `dd6b92a6`.
This is not the prior generic F2 DDR profile or a burst200 target.

The recorded image's source/runtime is
`30100afb075374381601a6e01d9f914a58dbec91`, already available on EC2 in
`/home/ubuntu/opt/bench-wide-port-alexshell`. A separate shared clone was checked
out at that revision; no active checkout, service or card was changed. Software
runs now use its exact target: D=128, MCOLS=8, LANES=8, PORTB=4, ROWLANES=true,
ALINE=64, ACT_BLOCKS=128, ACT_ROWS=16, 65,536 TMEM words, 32,768 IMEM words and
4 GiB logical device DRAM. The `a64` suffix names the 64-byte port-A line, not
64 ACT blocks.

`shared_tts_config` requires the wide-port Config fields and the exact hash.
It fails closed on the older runtime instead of substituting a generic target.
The component must therefore be integrated into revision 30100afb or a
compatible shared-image runtime; the earlier e55992e4 host is not qualified for
this image. No whole-runtime migration or deployment is hidden in this commit.

Historical exact-image register evidence comes from the publisher's
`41a9f4c48d449885a71e600350ff832aad09bfa2` evidence snapshot,
`tools/evolve-harness/evidence/sources/afbfd5b4-L33211.json`: CAPS `f70a0ef3`,
CAPS2 `39`, VERSION `00800808`, and the above build/hash. This evidence is
historical, not current ownership or a live card probe. The current required
identity is supplied by the parent. Fresh identity/capability checks remain
mandatory during coordinated physical acceptance.

The opcode/budget audit is in `qwen3_tts_shared_profile_audit.json`. No missing
native primitive was found for the current talker path. No physical TTS
acceptance, hardware speed, or full-model compatibility follows from this
software qualification. The same image must remain loaded; only weights and
programs may change. No serving interruption occurred, and the user's maximum
30-second interruption constraint remains explicit in the profile metadata.

The native audit emits LD/ST, RLD, MM, QACT/QST, VOP, LI/ADDI, LOOP and HALT;
there is no WAITW, DSTEP, STREAM or cross-slice GATHER. It records the actual
wide-port MM/QACT flag sets, rather than assuming old instructions are identical.
At capacity 512, the role-row program has 30 parts, 70,920 aggregate IMEM words,
and a largest part of 2,520 words. At position 32,767, aligned-row parts total
155,592 words but each part uses at most 5,744; peak TMEM is 15,162 words.
The source prefill change leaves the 4 GiB address-window/residency limits intact.

## Quantization and reference separation

There are 42 source-format prefill rows over the three prompts. Each prompt
starts with empty per-layer KV. The same real 316-tensor checkpoint subset is
used throughout. Comparisons retain layerwise traces rather than relying only
on layer-0 thresholds. Standalone ablations use FP32 algebra; their effects are
nonlinear, occur at different positions, and must not be added together.

| Path compared with clean FP32 | Maximum logit relative L2, all prefill rows |
|---|---:|
| Ideal FP64 algebra | 0.000514% |
| Source-cast eager BF16 | 6.2645% |
| INT8 weights only | 10.5211% |
| INT8 affine activations only | 13.1404% |
| INT8 KV only | 5.5415% |
| Functional weights/activation/KV/query/probability quantization | 20.1356% |
| Compiled ISA INT8/FP32 path | 16.1722% |

The functional quantization ablation intentionally uses dequantized FP32
matmuls. It is not an exact implementation of MX integer accumulation,
shift-window dropping, reduction order or the ISA's FP32 elementary functions.
Those differences can cross subsequent quantization boundaries. Its mismatch
with compiled ISA must not be mislabeled as a lowering defect.

A separate operator replay (`ExactMX`) uses the documented numeric primitives,
INT8 shift window 16, explicit integer accumulation, exact weight scales,
activation/KV quantizers, ISA reductions/elementary functions, and static MLP
chunk order. It imports no compiler, kernels or Machine. On all 13 rows of the
valid `Hello.` prefill, **every one of 28 layer outputs and every codec logit
matches the compiled ISA bit-for-bit** (364/364 layer snapshots; zero logit
error). This isolates lowering for that replayed case. It shares numeric
primitives/packing and is not an independent speech-quality reference.

## Acceptance stayed unchanged

The original 16 aligned-row diagnostic was rerun with the same input IDs,
references, greedy policies and criteria, now on the shared image's software
profile. It still exits **2**: 10/16 raw/fixed-mask greedy matches against FP64,
12/16 against BF16, and the same 51.19% / 56.63% maximum logit error as before.
The old trace checksum is unchanged. No threshold or dataset was silently
replaced, and the failed receipt remains intact. Because the two references
agree on only 11/16 rows, 100% agreement with both is impossible on that set.

The new valid-prefill evidence is separate:

| Text | Final prefill greedy, ISA/FP64/FP32/BF16 | ISA logit L2 vs FP32 | vs BF16 |
|---|---|---:|---:|
| Hello. | 1995 / 1995 / 1995 / 1995 | 2.8208% | 2.8226% |
| Good morning. | 1995 / 1995 / 1995 / 1995 | 3.0288% | 3.2127% |
| The FPGA speaks. | 1995 / 1995 / 1995 / 1995 | 3.2273% | 3.3998% |

These three final points satisfy the unchanged 0.035 diagnostic error level and
100% raw/fixed-mask greedy agreement. They are initial codec-head decisions,
not generated audio, free-running speech, or proof that the quantized whole
model is accepted. Across all 42 rows, greedy agreement is only 39/42 raw and
33/42 fixed-mask against FP32; against BF16 it is 42/42 raw and 34/42 fixed-mask.
The all-row gate therefore remains failed. Earlier prefix logits are not used
as generated codec outputs by the source, but their failures are still reported.
The full source's minimum-new-token EOS constraint, repetition penalty and
stochastic sampler are not replaced by this diagnostic greedy policy.

No mixed-precision recipe was introduced or tuned to these three texts. The
standalone ablations identify weight/activation/KV sensitivities; any precision
change still needs valid generated-code and eventual waveform acceptance on
this same image. There is no basis here for an image change or synthesis.

## Residency, throughput implications and next integration

The source-format fix adds no parameter storage. Device residency at capacity
512 remains 1,742,852,480 bytes with INT8 matrices/KV and FP32 tables/vectors.
Each role row adds one projection program, so there are 30 parts for a role row
and 29 for an aligned row: three additional launches per prefill. It also omits
the incorrect codec lookup/addition. Program parts and scratch remain bounded
by the recorded shared-profile audit. These are static resource facts, not an
FPGA timing or tokens-per-second claim. Regular-row weights are used; no Gemma
row-lane performance result is attributed to TTS.

The integration interface now connects pinned BPE IDs to the complete device
prefill and first codec-head result. `desc.normalized` and `desc.greedy_masked`
remain resident for the next stage. The next substantive integration is the
five-layer/15-head code predictor and device codebook-embedding accumulation,
followed by sampling/EOS state and the RVQ/waveform decoder. The shared image
has no native sine or erf operation; SnakeBeta and exact-GELU still need tested
ISA software lowerings or an explicit capability decision. Another model's
image is not an allowed workaround.

All 46 regression tests passed on the pinned shared-profile runtime, including
causal poisoning, reset, native maximum-context compilation and long-context
hardware-loop tests. Stable `5ee48c27` files/receipts are recoverable unchanged
from their commit; new evidence lives in separately named numerical receipts.
No hardware, synthesis, service, UI or readiness change occurred.

## Receipts and reproducibility

The software-only source checkout is
`/home/ubuntu/qwen3-tts-numerics-20261023-01/source`, based on the exact shared
runtime revision above. Existing active checkouts were read only. Matrix data
comes from the already cached, fully checksummed official safetensors file.
Runs use one numerical thread and low process priority on existing EC2.

| Evidence | SSM command / result |
|---|---|
| Pinned tokenizer/BPE data | `cebcad7e-6886-42e0-909d-c17b6c802e2c`, success |
| Shared-profile tests and seven-path ablation | `1fea6e5a-8e10-4282-99ca-edabf034ca4e`, 45 tests passed; ablation completed |
| Final regression, exact replay, unchanged legacy gate | `420fa0ab-98f9-4cc1-819d-e085e74c2ceb`, 46 passed in 12.79s; exact replay exit 0; legacy exit 2 retained |
| Receipt/opcode audit | `8df9dc3d-8f6e-4eb7-8989-6d692e552752`, success |

`qwen3_tts_numerics_ablation.json` retains the full layerwise summary and final
prefill points. The full unmodified 72,342-byte EC2 result, `ablation-01.json`,
has SHA-256 `504b69c731e4b09c011a994bcee3be1f1df3ff0162faad8083f8ea007d66ce50`;
its complete per-position trace path/hash is in the summary. The two additional
committed receipts are exact unmodified EC2 files:

- `qwen3_tts_numerics_exact.json`: 16,513 bytes, SHA-256
  `25176fa2cd3672782d4a7ca7ce87fe86cc0b4374f405136a2779afcdb7f720f4`.
- `qwen3_tts_numerics_legacy.json`: 7,071 bytes, SHA-256
  `597e51b22fc05cedbadc597572ac39ccccc7217b9249d63b173ab673203c470f`.

The seven-path ablation took 172.92 seconds and the exact replay 63.27 seconds;
these include CPU simulation/reference work and are not hardware throughput
measurements. The acceptance runner's gate, IDs, reference definitions and
thresholds were preserved. Only its target selection was changed to the required
shared profile. All reported tested-code hashes match the delivered files.
