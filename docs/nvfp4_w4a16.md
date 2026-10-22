# NVFP4 W4A16 numerical slice

This implements the proposal's contract/reference, a single F24 decode-align-dot
block, and a folded FP32 epilogue. It is independently testable and is not
connected to the sequencer, DMA, existing ACT RAM, model runtime or F2 shell.
The [third gate](nvfp4_local_job.md) adds the strict BF16/E2M1 tree in RTL and a
bounded descriptor-to-private-result controller. The earlier F24 mode remains
separately labeled; it does not substitute for strict tree validation.

## Second gate: FP32 epilogue and ownership

`rtl/mxu/otpu_nvfp4_fp.sv` implements portable RNE FP32 conversion, multiplication,
addition and E4M3 scale decoding. Gradual underflow and DAZ/FTZ are distinct
policies. Exact wide integer addition handles cancellation across the entire
FP32 exponent range. This prioritizes conformance, with a 280-bit arithmetic
workspace; it is not an area-optimized floating-point pipeline or qualified F2 IP.
There is no native FP32 DSP assumption. Overflow/nonfinite inputs fault with
canonical zero bits; a later zero or compensating scale cannot undo a fault.

`rtl/mxu/otpu_nvfp4_scale_acc.sv` is a single-context folded backend with separate
CONVERT, SCALE, ACCUMULATE and GLOBAL register boundaries. Every block first
normalizes the signed 41-bit F24 subtotal to RNE32 (or consumes a strict FP32
tree result), then multiplies the E4M3 scale, then adds into a +0 accumulator in
increasing K16 order. The last block invokes a separate final global multiply.
There is no fusion or reassociation. A non-last block occupies four clocks from
acceptance to the next input-ready state; final output becomes valid only after
the additional global stage. These are state-machine cycles, not measured F2
throughput. The FSM serializes the FP32 recurrence and holds terminal output
until consumed.

| RTL mode | Subtotal input | Underflow policy |
|---|---|---|
| 0 | External strict FP32 tree result | Gradual |
| 1 | External strict FP32 tree result | DAZ before each FP32 op; FTZ after RNE |
| 2 | F24 integer plus e_star | Gradual; remains approximate relative to strict tree |

Mode 3 is rejected. Mode 2 rejects discarded alignment, because no bounded-error
profile is approved. Even mode 2 without discarded bits is not strict tree
arithmetic. No descriptor capability or binary ISA allocation was added.

Each input atom includes group, K16 index and epoch, with an independently checked
scale identity. A scale can arrive late using `in_scale_valid`; the complete atom
is accepted only when the scale is available. This qualifies one composite atom,
not an independently handshaken scale channel. The source must hold the operand,
scale and identities stable until acceptance. First/last, monotonic K16 index,
profile, job tag, group, epoch and global-factor consistency are checked. An
unexpected first or changed identity poisons the current row while preserving its
original output identity. Fault bits are sticky through last: bit 0 protocol/input,
bit 1 conversion, bit 2 scale multiplication, bit 3 accumulation, bit 4 final
global multiplication. Failed output is +0 and must not be consumed as a result.

`buffer_free` is the input-atom acceptance event, with `free_group`,
`free_k16_index`, and `free_epoch` valid on that event. The full atom has been
snapshotted, so its staging slot may be reused. This releases neither an entire
DMA allocation nor the unfinished accumulator context. `output_committed` occurs
only on successful terminal `out_valid && out_ready`, after every required
rounding stage. It describes **local numerical result acceptance**, not host/DMA
visibility, model-state commit, or speculative token publication. Terminal faults
still drain via out_valid/out_ready but never assert output_committed.

No next row is accepted until the previous terminal result drains. Group/epoch
reuse, including wrap to zero, is tested only after that drain. Reset gates both
events and abandons in-flight contexts; the enclosing job controller must advance
its generation and prevent stale external traffic from reentering. It is not
implemented by a CPU speculative issuer or by these local epoch tags.
Reuse also requires every external operand/scale/subtotal holder to drain or be
generation-fenced. A local output handshake alone cannot distinguish stale traffic
after a complete group/K/epoch key is reused; that lifetime gate is upstream.

The dot module now has `TAG_W` (default 32, preserving its original interface).
The composed test uses 128 bits for group/epoch/K16/block identity. It directly
connects dot valid/ready to epilogue valid/ready and verifies atomic scale/tag
association through the subtotal pipeline under accumulator backpressure.

An independent `gpt-6.1-sol` reviewer at high reasoning reviewed arithmetic,
protocol and golden tests. No blocking finding was reported. Its suggested
overflow-threshold, partial-row reset, and composed-pipeline checks were added.
No independent physical implementation or hardware review is claimed.

## Provenance and repository audit

Implementation input: `OpenTPU_NVFP4_MXU_Implementation_Proposal_AWS_F2.docx`,
Library `libfile_a4e69a055c008191884ebcc48061ab61`, version 1, 309,595 bytes,
SHA256 `fcc3c03ce0d232bba1e812b3d430ef60f315ff2189dfe431fb896adf4f9718c5`.
The current Library helper materialized readable ZIP/XML bytes locally, retaining
Library identity. Its initial download failure was resolved using the installed
CA certificate bundle (the default Python CA file was absent).

Baseline: `/Users/rachittibrewal/ergodex-03/openTPU`, clean `main` at
`f237dab2f196855a59a083b1454786df8f01291f`. Its branches/worktrees and the separate
ergodex-04 checkout were inspected. Neither has an NVFP4 W4A16 branch.
The repository already implements NVFP4 **weights** in MM with int8 activations,
an MX shift window and per-row weight scale. It cannot be relabeled W4A16.
No applicable AGENTS.md or FPGA/Vivado skill under `~/.agents/skills` was found;
the personal FPGA, simulation and revision-control skills were reviewed.
The revision-control export workflow does not apply to this standalone RTL slice.

Work is in an independent local clone at
`/Users/rachittibrewal/Documents/Codex/2026-10-22/task-11/opentpu-nvfp4`,
branch `codex/nvfp4-w4a16-block`. This avoids writes to the source repository's Git
metadata and preserves existing user worktrees. The active wide-port checkout
`/Users/rachittibrewal/Documents/Codex/2026-10-22/task/wide-port-200mhz-opt`
was inspected read-only and was not edited by this work.

## Numerical contracts

Canonical weights: `q * s * g`, with E2M1 q, unsigned finite E4M3FN s per K16,
and an unsigned finite FP32 multiplicative global factor g. E2M1 magnitudes are
`0, .5, 1, 1.5, 2, 3, 4, 6`; sign is bit 3 and both zeros are legal.
Even K is in the low nibble, odd K in the high nibble. Unused high tail nibble
is ignored; the packer writes it as zero. Scale codes `00..7e` are legal;
`7f` and all signed scales (including negative zero) fault. Scale `00` denotes
a zero-weight block regardless of its payload, and canonical exports should use
zero codes there. The strict operation order still evaluates its dot before
scaling, so a zero scale cannot hide nonfinite input or intermediate overflow.
BF16 storage is little endian for a future memory front end.
No checkpoint tensors were imported. Actual checkpoint scale direction and
packing still require validation against captured golden tensor samples.

`opentpu/nvfp4.py` consumes raw integer bit patterns. Its exact rational arithmetic
does not reuse host BLAS or MM's existing FTZ arithmetic. Each FP32 operation is
rounded to nearest, ties to even; signed zeros and rounding into the smallest
normal are explicit. Nonfinite operands and overflow fault, including overflow
before a later compensating scale. Underflow follows one of two distinct IDs:

| Profile | Contract | Implemented backend |
|---|---|---|
| `nvfp4-w4a16-tree16-rne-gradual-v1` | Gradual underflow throughout | Software tree; RTL epilogue from strict subtotal |
| `nvfp4-w4a16-tree16-rne-daz-ftz-v1` | DAZ before each operation; FTZ after RNE | Software tree; RTL epilogue from strict subtotal |
| `nvfp4-w4a16-align-f24-rne-gradual-v1` | BF16 common exponent F24 alignment | RTL/software subtotal and epilogue, rejecting discarded alignment |

Strict row evaluation widens BF16 exactly, forms separately rounded FP32 products,
reduces adjacent pairs through a fixed four-level K16 tree, multiplies the block
scale, and adds blocks into a +0 accumulator in increasing K order. It multiplies
g once after all K blocks. Tail terms are +0, with no bias, residual or BF16 cast.
No GPU bit identity or model-quality equivalence is asserted.

The F24 block decodes `a = sign * M * 2^e`, where normals use `M=128+fraction`,
`e=exponent_field-134` and subnormals use `M=fraction`, `e=-133`. Zero lanes do
not affect `e_star`; an all-zero block uses `e_star=-133`.

```
t[i] = RNE_integer(sign[i] * M[i] * 2^(24 + e[i] - e_star))
d2[i] = 2 * E2M1(code[i])
z = sum_i(t[i] * d2[i])
approximate dot = z * 2^(e_star - 25)
```

The signed dot is 41 bits. `discarded` detects any nonzero alignment bits lost,
even if rounded to zero or multiplied by a zero weight. A consumer must select
strict fallback or an independently approved error policy when it is set. There
is no implicit approximate GEMM launch capability. Even with no lost bits, one
rounding of the integer dot differs from a tree rounding each FP32 addition:
the tests include `2^25 + 1 - 2^25 + 1`, whose strict tree is 0 and exact dot is 2.
The pre-scale alignment error bound is `48 * 2^(e_star-24)`; the F16 software
experiment uses the corresponding bound with 16. This is not an approved model
error envelope.

## Standalone RTL interface

`rtl/mxu/otpu_nvfp4_dot16.sv` implements one token/output K16 subtotal. It uses
shift/mux constant products with shared `t` and `3t`, then explicitly extended
37/38/39/40/41-bit product and reduction registers. It has no generic multiplier
per weight. Seven pipeline registers carry decode, alignment, product and four
reduction levels. It can accept one block per clock while unstalled.

The producer holds `in_valid` and its complete payload until `in_ready`.
`in_a[16*i +:16]` and `in_w[4*i +:4]` identify lane i. A zero mask bit suppresses
that lane and its activation validity check. The E4M3 scale and opaque 32-bit tag
travel atomically with the subtotal. The scale is **not applied** here. An active
nonfinite BF16 lane or invalid scale sets `out_invalid`; dot, exponent and
discarded are then canonical zero, with the original tag and scale retained.

If the output is valid and not ready, all pipeline stages freeze, the producer
is backpressured and the complete output payload remains stable. Output valid
does not depend on ready. Synchronous reset clears all valid state, discards
in-flight work and gates both handshakes. Data registers need no reset. The
upstream job controller must own reset generations, source lifetime and tags;
this primitive does not implement cancellation, completion or memory visibility.
The first accepted input reaches the final register after seven rising edges
including its acceptance edge, provided the pipeline advances on each edge.

## Proposed logical ISA boundary

`opentpu/mxu_job.py` defines immutable `W4A16Job` and `Buffer` records. A complete
mapping round-trip is supported; unknown or missing fields reject. This is a
logical contract without assigned opcodes, CSRs, binary offsets or wire length.
`MX_LAUNCH` is a symbolic operation. `SOFTWARE_CAPABILITIES` advertises only the
two complete strict software profile tuples. Each profile's SHA256 digest is
derived from its pinned ASCII semantic contract in `PROFILE_SPECS`.

The checker validates ABI major/minor, reserved fields, format tuple, profile
digest, positive 32-bit shapes, unsigned 64-bit addresses/strides/spans, alignment,
minimum row extents, address overflow and conservative output/input allocation
aliasing. Addresses above 4 GiB are preserved. It requires one FP32 global factor,
K16 scales, FP32 private output, post-operation NONE and transform bypass.
Read-only input allocations may overlap. A driver must still authorize every
mapping; structural validation grants no DMA access.

Context, generation, request, position, epoch and dependency identities are
retained as immutable metadata. There is no job queue or runtime completion
implementation yet. Epoch tags alone do not implement speculative decoding.
Future target verification requires explicit commit/rollback of KV, GDN,
convolution and position state after ordinary target decoding is correct.

Existing `Instr`, MM opcode `0x20`, WF=3, compiler, emulator and RTL behavior stay
as defined by ISA v1. Neither a new device capability bit nor a CX adapter is
advertised. CX remains an optional draft transport, not a DMA standard.

RHT is rejected by this first interface. It may only be enabled with matching
transformed weights, sign/normalization manifest and a separate quality gate.
The Holo4 checkpoint is mixed precision: FP8 attention and the BF16 head still
need their own implementations. This block does not establish full-model coverage.

## Verification and reproduction

Use the installed Python environment with NumPy and pytest. On this Mac it is
`/Users/rachittibrewal/anaconda3/bin/python` (NumPy 1.23.5, pytest 7.1.2).
The default `python3` lacks NumPy. No software was installed for these checks.

```
python -m pytest -q tests/test_nvfp4.py tests/test_nvfp4_epilogue.py tests/test_mx.py tests/test_isa.py tests/test_quant.py tests/test_fp.py
python -m pytest -q -s --basetemp=build/nvfp4-evidence/rtl tests/test_nvfp4_rtl.py
python -m pytest -q tests/test_nvfp4_epilogue.py
python -m pytest -q -s --basetemp=build/nvfp4-gate2/final-rtl tests/test_nvfp4_rtl.py tests/test_nvfp4_scale_acc_rtl.py
verilator --lint-only --top-module otpu_nvfp4_dot16 rtl/mxu/otpu_nvfp4_dot16.sv
verilator --lint-only --top-module otpu_nvfp4_scale_acc rtl/mxu/otpu_nvfp4_fp.sv rtl/mxu/otpu_nvfp4_scale_acc.sv
```

The software checks cover all format codes, canonical packing, RNE ties,
subnormals, overflow before scaling, signed zero, strict ordering and global
factor placement. Exact rational FP32 primitives are checked against 4,000 random
word pairs with independent NumPy IEEE operations. Strict rows are checked
against a separate NumPy pairwise implementation under both subnormal policies.
F16/F24 width and error bounds use 2,400 seeded rational block comparisons.
Descriptor tests reject unsupported profiles, unknown layouts, RHT, inverse
scales, bad spans/strides, 64-bit wrap and output aliasing.
At the first gate (`00087551`), the selected software/legacy arithmetic and ISA regression completed with
**144 passed**; the standalone RTL regression completed with **2 passed**.
Verilator standalone lint completed without warnings.

The RTL test runs the same 5,835 vectors through Icarus Verilog 13.0 and Verilator
5.048. Every finite BF16 exponent crosses all 16 FP4 codes; directed and seeded
cases add subnormals, maximum magnitudes, cancellation, ties, invalid scales,
nonfinite/masked lanes, random bubbles, long stalls and reset with work in flight.
It requires a terminal PASS, no fatal/error/timeout, exact ordered bit matches,
stable stalled payloads, depth seven and sustained simultaneous acceptance/retire.
Each run checks 5,825 outputs and explicitly flushes ten accepted transactions
on reset, with 3,335 output stall cycles and 5,100 simultaneous accept/retire
cycles. The retained `build/nvfp4-evidence` directory contains input vectors,
generated testbench, compile transcripts and simulation logs.

The testbench's terminal control was revised after Verilator continued to its
watchdog following a PASS/$finish path. The final harness exits its bounded loop
before calling $finish and rejects fatal text even when a simulator returns zero.
This was a verification harness issue; no arithmetic acceptance check was relaxed.

The second gate adds `tests/test_nvfp4_epilogue.py` and
`tests/test_nvfp4_scale_acc_rtl.py`. Primitive vectors cover every E4M3 byte,
all 512 signed e_star encodings crossed with subtotal extrema/ties, exact converter
overflow-threshold neighbors, FP32 cancellation, signed zeros, normal/subnormal
promotion, invalid operands, and seeded IEEE word pairs under both policies.
Row vectors cover separate rounding witnesses, increasing K16 order, immutable
group/epoch/profile/global identity, coherent operand/scale identity mutations,
independent scale identity errors, sticky faults, delayed scales, seven reset
phases including a partial row awaiting its next block, stalls, and drained
owner reuse across epoch wrap. A composed BF16/E2M1-dot-to-epilogue test adds
masked tails, invalid/discarded poisoning, 128-bit identities, thirteen in-flight
reset phases, and accumulator-driven backpressure. It does not model independently
reordered memory responses or prove global epoch lifetime fencing.

Both Icarus 13.0 and Verilator 5.048 execute all RTL families. Two portability
issues in verification were corrected before the passing aggregate run: Icarus
needed direct enum-state if/else assignment and declarations before composed
testbench instance references. The composed testbench also snapshots stimulus
identity before the sampling edge, avoiding a source/scoreboard race. No numeric
comparison, fault condition or watchdog was removed.

Final second-gate results: **149 software tests passed** and **8 RTL tests passed**;
standalone dot and scale/accumulator Verilator lint both pass without warnings.
Each simulator completes the following identical scoreboard totals:

| Family | Completed evidence |
|---|---|
| F24 dot pipeline | 5,835 vectors; 5,825 checked and ten reset-flushed |
| FP32 primitives | 14,088 exact golden comparisons |
| Ordered epilogue | 2,665 blocks / 1,425 rows; seven reset phases; 7,205 stalled-output and 9,098 delayed-scale cycles |
| Composed dot/epilogue | 800 blocks / 220 rows; thirteen reset phases; 1,463 stalled-output and 2,219 delayed-scale cycles |

The final logs, generated benches/vectors and hashes are retained under
`build/nvfp4-gate2/final-rtl` with aggregate logs in `build/nvfp4-gate2`.
Every run requires explicit TEST_PASS and rejects fatal/error/watchdog text,
including simulator failures that return exit status zero. These are behavioral
simulations; clock-cycle counts are not F2 frequency or TPS measurements.

## Next implementation gate and exclusions

The strict folded tree and bounded local-stream job controller are now implemented
and described in [the third gate](nvfp4_local_job.md). Next connect authorized
memory/tile buffers and real output visibility to the checked logical job.
Retain explicit generation/drain ownership and qualify a selected implementation
under the actual F2 Vivado/HDK flow before pipelining or replicating arithmetic.
The wide-port worker owns the separate two-credit prefetch A/B experiment.
No Megakernels/GPU TPS result is imported into this numerical backend.

F2/VU47P at 200 MHz remains a target. There is no native FP32 DSP on VU47P.
No Vivado executable or MCP was available through the configured environment.
Vivado/XSim, synthesis (including resource mapping), place/route, timing, CDC,
shell integration, DCP/AFI generation, FPGA loading and hardware measurement are
unrun. No timing, area, throughput, inference or model-quality claim follows from
these tests. No paid resource, model-weight download, Mac inference, license or
security change, publication or push was performed. The full repository suite
includes model execution and was not run during the local inference pause;
the selected arithmetic/ISA regressions are recorded separately.
