# Validation round two: RTL trace diagnosis and matched memory

The implementation remains experimental. **76 of 79 cases match both total cycles and every
instruction dispatch/start/end event. All 79 pass numeric DRAM/TMEM comparisons. The three
tiny-model stages fail the exact timing gate.** The complete pretrained-model goal is not
validated. The preserved [initial results](validation.md) and receipts remain unchanged.
No LPN result or claim is extended to this model.

## What changed and why

The original FILL(67), MUL(67), RSUM(67) program predicted 77 cycles against 107 measured.
The RTL trace showed missing frontend/start stages, vector pipeline tails and reduction-tree
scheduling. The simulator now predicts **107**, with the exact measured instruction events.
This was addressed with RTL-derived stages, not by adding a fitted whole-program offset.

The follow-up corpus exposed additional issues: the default C8 nonlinear path uses all eight
lanes with recirculation, output arbitration freezes the VPU pipeline, DMA gathers stores into
runs and prefetches loads, QACT overlaps row scan/scale/pack phases, and MXU result draining
must run concurrently with array advances. MXU scale lines are requested on each applicable
advance, with coupled A/B acceptance and an issuer-busy condition. This increased reported
scale traffic compared with the historical model; the old HBM projections are superseded.
Sequencer selection now distinguishes oldest-ready DMA/Q/VPU from oldest-only MXU/collectives.

The compiled 3×64×19 GEMM moved from **313 predicted / 228 measured** in the first follow-up
corpus to **228 / 228**, including every instruction trace event. Seven new partial-tile MXU
cases (M=1–4, N=1–33, KB=1–4) and five new DMA length/boundary cases were then executed and
matched without further fitting. Those cases and earlier diagnostics are now regression
fixtures, not a perpetual held-out validation set.

## Evidence and scope

Receipts: [`tools/perfsim/receipts/2026-10-19-round2`](../../tools/perfsim/receipts/2026-10-19-round2).

* **108 pytest tests pass**, including 75 frozen RTL trace regressions and two tests against
  the freshly built real Ramulator HBM2/DDR4 backend. The remaining tests cover numerical
  references, packed formats, queue limits, clocks, retry/completion IDs, dependencies and
  coupled A/B backpressure. Frozen fixtures replay actual observations; they do not execute RTL.
* **79 actual Verilator executions** used the unchanged `tb_top` and identical program/image
  inputs. Cases cover partial vectors, reductions, nonlinear pipeline overlap, a full issue
  window, loops, RAW/WAR/WAW dependencies, aligned/unaligned DMA, long buffered DMA, VPU/DMA
  contention, QACT row buffering, partial MXU tiles, compiled GEMM and three tiny-model stages.
* Matched memory uses the explicitly selected `rtl_fixed` backend: D=32, DRAM_LAT=8, AXI=0,
  registered native A/B/SW ports and masked writes. It is a behavioral-memory comparison,
  **not Ramulator/DDR/HBM timing validation**. Core and memory edges both use 5 ns; frequency
  only supplies units for this contract. RAM width and clock mismatches are rejected.
* The cached RTL binary was parameter-checked for S=1, D=32, MCOLS=4, LANES=8, ACT_BLOCKS=64,
  ACT_ROWS=4, TMEM_WORDS=8192, IMEM_WORDS=65536, DRAM_WORDS=262144, WIN=16, RPB=32, WPB=1,
  MXU_IMPL=2 and FIFO_DEPTH=1024. The receipts contain binary/build-receipt hashes; this round
  did not freshly rebuild the RTL. Other parameter combinations have no matched timing claim.
* The evaluator records raw RTL stdout, exact programs/configs, input/output hashes, instruction
  traces, metrics and a simulator/evaluator source manifest. It leaves the RTL counter and
  testbench unchanged. Exact trace/cycle failure returns exit 2; numeric/execution failure is
  an error. A separate ±5% aggregate check cannot make a trace mismatch pass.

| Bounded compiled program | RTL cycles | Predicted cycles, matched memory | Error | Exact trace gate |
|---|---:|---:|---:|---|
| Partial GEMM | 228 | 228 | 0% | Pass |
| Tiny Qwen prompt position 0 | 5,716 | 5,748 | +0.560% | Fail |
| Tiny Qwen prompt position 1 | 5,718 | 5,750 | +0.560% | Fail |
| Tiny Qwen decode position 2 | 5,720 | 5,752 | +0.559% | Fail |

The model stages use fixed input/KV snapshots for each RTL launch; they are not continuous
RTL serving measurements. Small aggregate errors do not hide sizeable per-instruction start
and completion differences. Remaining sources include QST serialization, overlapping VPU
commands and multi-command MXU release/drain/response scheduling. A wider corpus, alternate
bank topologies and a shared RTL Ramulator adapter are still needed.

## Independent numerical checks and fresh Ramulator predictions

The complete tiny random two-layer Qwen3 topology includes attention, normalization, MLP and
head paths. All stages match sequential ISA arithmetic bit-for-bit. That comparison shares
functional code, so it is not an independent arithmetic oracle. Unchanged RTL independently
matches DRAM/TMEM, and the NumPy float forward gives relative L2 errors **2.0373%, 2.0884%,
2.1099%**, with cosines above 0.99978. These pass the previously declared 5% L2 / 0.995 cosine
limits. No pretrained checkpoint was downloaded.

A clean offline build of pinned Ramulator v2.0a and dependencies succeeded in non-synced `/tmp`
source/build directories, followed by a successful incremental repeat build and actual HBM2
and DDR4 integration tests. `dependency-build.json` records the patch and library hashes;
both build logs are retained. The earlier synced-filesystem stalls were not successful builds.
The repeat used the same build script; no system installation or paid resource was required.

The following are **unvalidated predictions using actual Ramulator**, with persistent controller
and KV state across two serial prompt steps and one teacher-forced decode step. HBM2 controller
and memory periods are both 1 ns. They are not measured hardware performance or full-size-model
TPS. The fixed-memory RTL comparison above does not validate these HBM timings.

| Core clock assumption | Prompt cycles | Decode cycles | Device prefill / TTFT | Decode time | Conditional tiny-workload TPS |
|---|---:|---:|---:|---:|---:|
| User-supplied 200 MHz | 5,794 + 5,773 | 5,748 | 57.835 µs | 28.740 µs | 34,795 |
| Hypothetical 1 GHz | 11,447 + 11,760 | 11,747 | 23.207 µs | 11.747 µs | 85,128 |

The start boundary is resident-program execution; the end is drained HALT and final write
completion. Host transfer, queueing, sampling and networking are excluded. Prefill here means
two serial token programs. At 200 MHz the stage read bytes are 137,600 / 137,728 / 137,856,
with 8,600 / 8,608 / 8,616 read transactions. Each stage uses 2,368 physical write bytes in
148 transactions, including the declared byte-store RMW adapter overhead. Per-layer/region
spans, stalls and utilization are in the retained metrics; overlapping spans are not additive.

## Remaining blockers to the full requested goal

There is no access/approval blocker pending. The blocking issue is timing fidelity and coverage:
QST and several overlapping-unit paths still differ from RTL; all three compiled-model exact
trace gates fail. HBM/DDR timing has not been checked against RTL driven by the same closed-loop
Ramulator adapter. Full pretrained-model execution and timing validation remain absent.
Multi-slice, WAITC/streaming and other explicitly rejected modes remain unsupported. Physical
resource fit, routing, power and Fmax require implementation work that was not performed.
