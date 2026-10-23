# Qwen3-TTS: composed talker and codec logits

This isolated follow-up to `3a57ab00` composes all **28 talker layers**, final
RMSNorm and the 3,072-way codec head for the exact
**Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice** variant. It adds raw and fixed-mask greedy
selection in the device program. It leaves full text-to-waveform synthesis gated.

## Source contract and implemented interface

The pinned model revision remains
`85e237c12c027371202489a0ec509ded67b5e4b5`; source remains
`022e286b98fbec7e1e916cb940cdf532cd9f488e`, `qwen-tts 0.1.1`, Apache-2.0.
The [official model implementation](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py)
runs every decoder layer, then `talker.model.norm`, then `talker.codec_head`.
The generation wrapper suppresses codec IDs 2048..3071 except EOS 2150.

`TalkerStackImage` owns all 316 talker/projection/head tensors: 764,218,368
stored checkpoint elements. `tests/fixtures/qwen3_tts_stack_contract.json`
records a canonical shape/dtype/name hash from the actual safetensors header;
the native descriptor must match it. The exact official 1,811,626,576-byte file
is already cached on EC2 and is verified again against SHA-256
`bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`.
No additional model download or remote model-code execution is required.

`compile_parts(position)` returns sequential one-slice programs. Part 0 reuses
the previously validated resident embedding/projection/layer-0 program verbatim.
Parts 1..27 consume the preceding hidden vector from device DRAM, use their own
weights/KV, and overwrite that vector on device. The last part executes final
RMSNorm, codec projection, and two VOP ARGMAX reductions. It stores complete
logits plus `[score, id]` pairs for raw and fixed-mask selection. Equal maxima
select the lowest ID. No host normalization, hidden-vector transfer between
layers, codec matmul or argmax is part of the target compute path.

The fixed mask allows acoustic IDs 0..2047 and EOS 2150. It is not the complete
generation policy: the default repetition penalty and stochastic sampling are
not implemented or silently replaced. The raw/fixed-mask greedy policies are
explicit diagnostics. The text/codec input remains one aligned row per position;
complete CustomVoice prompt assembly and 16-code predictor embedding summation
are still absent.

The unchanged precision recipe is MX INT8 matrices, activations and KV, with
FP32 embedding tables, vectors, residuals, fixed RoPE tables and ISA arithmetic.
All layers share immutable RoPE constants, while every layer owns distinct KV.
`StackISASession` is an explicit CPU ISA simulator, never a production backend.
The host writes two validated token IDs, launches the parts in order, and may
read diagnostic snapshots. Reset zeroes all layer KV regions and restarts the
position; tests compare dirty-state replay with a fresh machine.
Diagnostic snapshot copies are never written back or used as the next layer's
input; all inter-layer data comes from the resident device hidden buffer.

## Instruction and memory boundary

The f2 target has 32,768 IMEM words, 65,536 TMEM words and a 4 GiB device address
window. A concatenation of all layer instructions exceeds IMEM. Sequential
programs solve this without a new opcode or synthesis: each part fits and all
learned state stays resident. Calling the inherited single-program `compile`
interface is explicitly rejected, preventing accidental execution of layer 0
alone. A future hardware runner must schedule these parts or provide reviewed
program chaining; the diagnostic does not install such a runner on the card.

The new runtime uses only existing load/store, integer address, MXU, quantizer,
VOP and loop operations. It adds no hardware-specific transport, generic-model
adapter shortcut, CPU/GPU fallback, API call, or image dependency.

Measured descriptor budgets (no allocation at maximum context):

| Capacity / position | Resident bytes | KV bytes | Total IMEM words across 29 parts | Largest part |
|---|---:|---:|---:|---:|
| 512 / 0 | 1,742,852,480 | 30,277,632 | 70,928 | 2,728 |
| 512 / 256 | 1,742,852,480 | 30,277,632 | 90,640 | 3,432 |
| 32,768 / 32,767 | 3,666,858,368 | 1,937,768,448 | 155,592 | 5,744 |

Peak live TMEM is 15,162 words in every checked case. The maximum-context
talker leaves 628,108,928 bytes of the compile-time address window; this does
not establish room for the remaining predictor, decoder and stream state.
There is no current talker memory blocker, but concatenating the instructions
is an actual IMEM blocker. The explicit sequential-parts interface resolves it.
Board memory availability/transport is unqueried and remains unvalidated.

## Validation scope

The EC2 regression includes **42 passed in 12.19s**. All prior 36 tests remain,
including poisoned-cache causal checks, reset, native max-context descriptor
compilation and the long-cache hardware-loop reference test. The prior one-layer
260-position real-weight receipt is preserved separately. New tests cover three-layer composition across
a 32-token block boundary, per-layer KV isolation, bitwise future-cache
poisoning, reset, reference equivalence with the prior independent layer math,
device argmax ties/EOS suppression, full native parameter contracts and 29-part
compilation at position 32,767.

The new checkpoint diagnostic uses all 28 layers and compares accumulated
hidden vectors, final normalization and codec logits with independent locally
reviewed FP64 and eager-BF16 Torch algebra. The references import no Qwen model
classes. It reports per-layer accumulated error, final-logit error, raw/masked
greedy agreement, reference top-two margins and winner ranks. It also checks
that device ARGMAX agrees with the emitted device logits, and compares two
machines with/without poisoned future KV at every layer.

The real-checkpoint run is intentionally bounded to 16 common-input diagnostic
rows, not a valid CustomVoice prompt, free-running audio, or a whole-TTS
acceptance. The older long-context coverage remains separate; this does not
claim 260-position real-weight acceptance of all 28 layers. A mismatch after
composition is reported directly rather than judged only by a layer-local
3.5% threshold. The runner exits 2 for any observed greedy disagreement while
preserving its numeric receipt, and exits 1 for masking/reset/device-selection
invariant failures.

## Actual checkpoint outcome: greedy compatibility failed

The composed program executes and all structural/state invariants pass, but
the previous INT8/FP32 precision recipe does **not** establish greedy parity
after 28 layers. No quantizer, threshold or input was tuned after observing the
failure. Maximum accumulated hidden-vector relative L2 errors over 16 rows:

| Layer, one-based | ISA vs FP64 | ISA vs eager BF16 |
|---:|---:|---:|
| 1 | 2.16% | 2.49% |
| 2 | 5.10% | 5.76% |
| 8 | 9.24% | 9.81% |
| 16 | 25.25% | 27.08% |
| 28 | 54.15% | 59.71% |

Final-logit relative L2 is at most **51.19% vs FP64 / 56.63% vs BF16**, with
mean 25.27% / 27.88%, minimum cosine 0.8744 / 0.8455 and maximum absolute logit
error 8.2455 / 10.0993. Raw and fixed-mask greedy agree on **10/16** rows against
FP64 and **12/16** against eager BF16. The two independent-precision reference
policies themselves agree on 11/16 rows. These diagnostic rows are outside a
complete valid CustomVoice prompt, so the result does not isolate all error
to INT8 or establish behavior on valid speech prompts.

There are **zero** future-cache bitwise mismatches, device-argmax/logit invariant
mismatches, or reset bitwise mismatches. Reset checks two rows against a pristine
28-layer machine. Existing long-context tests remain passing. The checkpoint
run completed in 106.91 seconds with **exit 2**, the deliberately failed greedy
compatibility gate, rather than an execution exception or permission denial.
Peak measured host RSS is 14,641,668 KiB; this is a CPU diagnostic measurement,
not FPGA inference latency, memory use or performance.

The concrete acceptance blocker is accumulated numerical/greedy disagreement,
not a missing talker ISA primitive. The minimum next decision is a valid-prompt
input contract plus a bounded precision/quantization ablation, including a
source-rounded FP32 baseline, to locate sensitivity before choosing calibrated
INT8, residual/multi-pass INT8 representations or higher-precision ISA work.
Neither the layer-local 3.5% threshold nor observed argmax correctness certifies
composed-model compatibility. No RTL/synthesis change is justified by this
receipt alone, and no automatic host fallback is permitted.

Some mismatches exceed a near-tie explanation. At position 9, device greedy is
1930 while both references select 628. The reference top-two margins are
0.8538 (FP64) and 1.5 (BF16), and the reference winner ranks third in device
logits. At position 6, device selects 1247, FP64 selects 703 (device rank nine),
and BF16 selects 14 (device rank five). Per-position measurements retain margins,
maximum logit error and the sufficient `margin > 2*max_error` stability check.

## Remaining end-to-end stages

The full talker backbone/head closes the previously missing 27 layers and final
logits. Still required: actual text/control/speaker/language prompt assembly;
the five-layer code predictor, 15 acoustic heads and device embedding sum;
repetition penalty, stochastic RNG/sampling, EOS/generation state; RVQ; and the
codec transformer/convolution waveform decoder with streaming state. Dynamic
sine for SnakeBeta and exact-GELU/erf remain unresolved ISA lowering decisions.

Greedy agreement on common rows cannot establish agreement after feedback,
long prompts or downstream acoustic reconstruction. The default sampled TTS
policy and speech quality remain unvalidated. Physical FPGA execution and
performance remain pending parent coordination. No FPGA, synthesis, serving,
production UI, new resources, or full-TTS enablement is part of this change.

## Reproducible evidence

- Source/test SSM command: `10fb2e2a-b86a-429f-9298-ffb2c305a01a`, pytest exit 0.
- Real-checkpoint SSM command: `1b1606af-8d68-4ae8-b588-22bfc7cef7a1`, completed
  with diagnostic exit 2 (greedy mismatch), not an execution or approval error.
- Receipt read command: `c5c607dd-0661-4e2a-b17f-3f1cf4900d36`.
- Exact unmodified EC2 result: `qwen3_tts_stack_acceptance.json`, 6,341 bytes,
  SHA-256 `1136a09c5d17e83f0e0dd7aeee4964d0da5321b9824073d826400faeb9897a6e`.
- Full per-position result remains on EC2 at
  `/home/ubuntu/qwen3-tts-stack-20261023-01/stack-acceptance-01.positions.json`,
  185,138 bytes, SHA-256
  `8663d1d6ede90b40158119eeceaea226905fd9729d31a4fb99d54521e79faed1`.

The isolated source directory is
`/home/ubuntu/qwen3-tts-stack-20261023-01/source`. The command uses
`tools/qwen3_tts_stack_acceptance.py --steps 16` with the already verified
checkpoint/manifest paths from the prior acceptance, and an output path of
`/home/ubuntu/qwen3-tts-stack-20261023-01/stack-acceptance-01.json`.
Execution is low priority, one numerical thread, bounded by a 480-second timeout
and 64 GiB address-space ceiling. Python 3.12.3, NumPy 2.5.4 and CPU PyTorch
2.14.1 are unchanged. Tested code hashes match local deliverables. The prior
`3a57ab00` files and receipt are unchanged; this commit contains only new files.
