# Round-three validation: closed-loop RTL and Ramulator

This is a bounded validation of an OpenTPU instruction simulator. It is not a physical
hardware measurement, a pretrained-model benchmark, or proof of cycle accuracy for every
accepted parameter. Existing RTL, compiler, LPN model/results and prior golden observations
are preserved. The earlier 0.8–4.2% LPN results still apply only to their five configurations.

## Observed results

The fixed-memory corpus has **112/112 exact total-cycle and dispatch/start/end matches**,
with numerical agreement against the sequential ISA reference. It covers all 20 VPU
functions, four broadcast modes, partial rows/columns, reductions/nonlinear recurrence,
OUTER modes, scalar/loop/BAR/RLD, QACT scaling/duplication, QST aligned/strided/half/row modes,
packed weights, MXU overlap/RMAX, long DMA, contention, compiler GEMM and three tiny-model
stages. Coverage inventory lists actual encoded shapes and flags; it is not exhaustive.

The native-memory corpus has **21/21 exact matches** for total cycles, every D/S/E event,
every per-port acceptance/completion cycle and request payload, and physical bytes and
transactions. Both sides independently generate requests against the same controller
implementation and pinned Ramulator configuration. No replay or forced arrival times occur.

| Workload | HBM2, core 200 MHz | DDR4, core 200 MHz | HBM2, hypothetical core 1 GHz / eight outstanding |
|---|---:|---:|---:|
| Unaligned 2048-word LD/ST | 585 | 1310 | 1606 |
| DMA/VPU contention | 456 | 456 | 479 |
| Two-row QST | 229 | 348 | 726 |
| Compiled partial GEMM | 228 | 255 | 466 |
| Tiny Qwen prompt position 0 | 5748 | 5783 | 18319 |
| Tiny Qwen prompt position 1 | 5750 | 5785 | 18314 |
| Tiny Qwen decode position 2 | 5752 | 5787 | 18314 |

Each cell is both the **observed RTL cycle count and matched simulator count**, with a cold
memory controller at that stage's start. HBM2 controller/memory tCK is 1000 ps; DDR4 is
833 ps. The 200 MHz core period is 5000 ps, the stress period 1000 ps. RTL's simulation
clock is an edge driver; the explicit profile periods drive Ramulator physical time.

The tiny workload executes the complete random two-layer Qwen3 topology: hidden 64,
two query heads, one KV head, head dimension 32, FFN 96, vocabulary 64. Deterministic weights
and teacher-forced tokens are generated locally. Input bundles carry prior ISA-generated KV
for standalone stages. RTL and the performance simulator independently match ISA DRAM/TMEM
bit for bit. Independent NumPy float-forward logits pass the separately declared 5% relative
L2 / 0.995 cosine gate: observed relative L2 is approximately 2.04–2.11%, cosine >0.99978.
The float reference does not reuse the ISA quantizer. Sharing ISA execution alone would not
constitute independent numerical validation.

**170 pytest cases pass with the real pinned Ramulator library enabled.** Frozen schedules
are explicitly replay tests, not new RTL measurements. They include 21 opt-in native-memory
cases; an absent library skips those tests and never silently emulates Ramulator.

## Model changes justified by first divergence

The previous three 32-cycle tiny-model gaps were QST's actual unbuffered two-pass schedule
and queue/overlap details. New traces also established scale-before-data writes, bounded
backpressure freeze, registered VPU/DMA bank writes, MXU head/drain ordering, scale/weight
FIFO visibility, store-run occupancy snapshots, quantizer readiness before sequencer done,
and BAR/RLD setup. Initial failed comparisons are retained alongside final results.
No evaluator tolerance, RTL golden output, measurement boundary or counter was changed.
The stricter port-event gate caught differences even when aggregate cycles already matched.

The new harness generates a derived top/testbench in an isolated build directory. The
original compute RTL and counter/reset/dump logic are unchanged. The sole functional
substitution is the external-memory endpoint, imported through DPI from the same native-port
controller used by the simulator. This validates that controller contract, not an F2 AXI shell.
Build receipts hash original and derived sources, executable, pinned bridge and configuration.

## Persistent sequence prediction and limits

The separately reported HBM2/200 MHz persistent-controller sequence predicts 5748, 5758,
and 5731 cycles. Device-only serial prefill/TTFT is **57.53 microseconds**; the single decode
step is **28.655 microseconds**, or **34,897.9 conditional tokens/second**. These numbers are
**unvalidated timing predictions for this tiny random workload**, not measured hardware TPS
and not pretrained-model throughput. They exclude uploads, sampling, host scheduling,
serving queues and network time. The sequence retains KV/controller/row/refresh state with
no launch gaps. A persistent RTL process has not yet validated that multi-launch boundary.
Layer spans, stalls, physical/logical traffic and utilization are in each stage's metrics.
Layer spans may overlap and cannot be summed to recover wall time.

Persistent-launch testing found and fixed a local/global completion-clock mixup; its failed
diagnostic is retained and a deterministic repeated-launch regression guards the correction.
Do not reuse historical sequence predictions as current results.

Remaining coverage gaps: full-size/pretrained model execution; other D/LANES/MCOLS geometries;
exhaustive PAIR, ACC/ASCALE and packed scaling variants; alternate TMEM/WBUF/pumped topologies;
multi-slice collectives, streaming/wait-count/host-poll/program-chain paths; and persistent
multi-launch RTL timing. Unsupported opcodes/configurations fail explicitly. Physical Fmax,
resource fit, area, power and FPGA shell behaviour require separate implementation work.

## Retained artifacts and reproduction

`tools/perfsim/receipts/2026-10-19-round3/` retains summaries, raw RTL logs (compressed),
per-case metrics, memory event logs, source/build/input hashes, coverage and test output.
Frozen fixtures retain compressed exact program images and observed traces. The dependency
lock pins Ramulator and transitive sources; the existing notice/patch/license are preserved.
The architecture is S=1, D=32, LANES=8, MCOLS=4, WIN=16, RPB=32, WPB=1, FIFO_DEPTH=1024.
No synthesis, paid resources, push or PR publication is part of this work.

Follow [README](README.md) for building and running the fixed and native comparisons.
`coverage.py --fixed FIXED_CORPUS --native NATIVE_CORPUS --out coverage.json` derives the
machine-readable inventory from retained run summaries. `sequence.py` reports the distinct
persistent-state predictions. Exact comparison failures return exit 2; numerical/execution
failures return exit 1. Prior round-two and initial failed evidence remain unchanged.
