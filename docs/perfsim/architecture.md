# Architecture and fidelity contract

The model is instruction-driven and cycle-stepped. Functional execution uses the unchanged
`opentpu.isasim.Slice.execute` at command completion, with dispatch-time register snapshots.
The timing scheduler does not use its result to synthesize latency or grant sparse speedups.
A separate sequential ISA run checks scheduler correctness, and independent NumPy/RTL checks
cover arithmetic. Sharing ISA arithmetic is explicitly **not** an independent numerical oracle.

## Dataflow

1. Four elastic footprint stages (R→P→S→Q→C) resolve scalar register-relative addresses and expand LOOP dynamically. The first R edge is cycle 1, earliest dispatch is cycle 5, and the unit start is observed one cycle later.
   A finite issue window retains outstanding commands. RLD blocks younger dispatch until its
   scalar result commits. BAR waits for older commands; HALT waits for all commands and writes.
2. `opentpu.waits.footprint/conflict` supplies the same conservative RAW/WAR/WAW address-range
   semantics as `otpu_pkg`. MXU and collectives consider their oldest unstarted command; DMA, quantizer and VPU choose the oldest ready candidate. Started same-unit dependencies are cleared for internally ordered units, but not VPU. Registered starts cannot occur on consecutive cycles. MXU prefetch is released
   on DRAM dependencies separately from activation/TMEM dependencies.
3. Unit steps arbitrate in DMA→collective→VPU→MXU→quantizer priority order on banked TMEM at `word_address % LANES`, with separate per-bank read
   and write capacities. Atomic lane operations stall as a group. MXU scattered drains split
   into conflict-free groups. Buffered outputs exert backpressure at finite queue limits.
4. Every memory command reaches a finite adapter. Port A (32-byte scale lines), B (D-byte DMA/
   weight requests), SW (quantizer writes) each accept at most one logical request per core
   cycle. The adapter splits unaligned transfers into physical transactions, retries rejection,
   and releases tickets only on completion. Partial writes explicitly perform RMW; read/write
   ordering is enforced per transaction line. No bandwidth average replaces requests.
5. Ramulator advances in integer-picosecond time. Core, controller and memory periods are
   recorded separately. This bridge supports a controller tick equal to DRAM tCK and rejects
   unequal controller/memory periods. Core ticks may be slower/faster/non-integral relative
   to memory ticks. Requests generated at a core edge first reach a later memory edge.
   Ingress bandwidth is independently bounded per controller tick. Persistent sequences retain
   the memory clock, open rows, refresh phase and controller queues between launches.

## Unit-specific model

| Unit | Implemented behaviour | RTL reference and remaining approximation |
|---|---|---|
| Issue | finite window; control/register/loop handling; range conflicts; separate MM stream dependencies; 2-entry MXU queue | `rtl/seq/otpu_seq.sv`, `opentpu/waits.py`. Four-stage footprint pipeline, registered unit starts and drained HALT boundary; optional two-stage WAITC frontend with unit-count waits and a 31-entry VPU mask |
| MXU | actual D-byte block stream, row-gated consumption, finite prefetch FIFO, scale-line requests, ACT row replay, int8/int4/fp4/nvfp4 packing, PAIR, partial M/N and KB; result FIFO/drain, ACC/ASCALE/RMAX | `rtl/mxu/otpu_mxu.sv`: CLS=min(D/8,CL), LROW=CLS+MCOLS+5, epilogue lanes default MCOLS/2. Independent prefetch, array advance and result drain pipelines; LROW followed by S0/row/FIFO stages, nine-cycle conversion/scaling epilogue. Coupled A/B requests are atomic, A is fetched on each scale advance, and an active issuer blocks the next command. B response FIFO visibility takes one additional core edge; native A line/tag plus scale-count visibility takes two. Next-head pop and serialized drain, including epilogue freeze, are explicit. PAIR/alternate scaling variants and other geometries remain incompletely validated |
| VPU | all current arithmetic/copy/fill/nonlinear/reduction/OUTER functions, broadcast addressing, masked partial lanes, 64-partial reduction tail, buffered pipeline writes | `rtl/vpu/otpu_vpu.sv`: LM=2, LA=4, SL=7; ADD/MUL/OUTER 1 slot, EXP2/EXP2SUB 9, RECIP 6, RSQRT/LOG2 10. Default C8 composite path uses eight lanes and 25-cycle recirculation passes (HAS_DSTEP defaults on even without DSTEP opcodes). Two-entry command queue, arithmetic latency ordering, padded reduction streams, RTL folding-tree latency and pipeline freeze on denied output writes. Cross-command read arbitration and non-C8 configurations lack matched timing coverage |
| Quantizer | row-buffer capacity, max/prescale read pass, reciprocal/Newton latency, second ACT pack pass, QACT DUP/ROW/CSCALE/RSCALE; QST ROW/HALF/strided bytes and scale writes | `rtl/vpu/otpu_quant.sv`: LM=2, LA=4, QL=5, reciprocal tail=30. QACT overlaps four row banks with finite chunk credits, scale readiness and ACT packing. Overflow reread path is rejected. QST is an unbuffered two-pass path: full-row max reads, scale records first, a second TMEM read pass, then four-byte writes only for aligned ES=1 (otherwise byte writes). A denied SW write freezes its finite 12-stage reader/writer pipeline. QACT and QST readiness precede sequencer completion by one edge |
| DMA | D-chunk boundaries, lane gathering/scattering, finite gathered chunk, port-B sharing and read/write completion | `rtl/dma/otpu_dma.sv`. Independent request/gather and completion/drain paths, 128-chunk buffer, store run threshold 32 and pipeline stages; tested aligned/unaligned transfers, partial chunks and long runs |
| TMEM | modulo banking, independent configurable read/write capacity, address-specific contention, pipeline drain stalls | `rtl/mem/otpu_tmem.sv`. Abstract bank model; replicated-port guests, subbank/hash/pump and WBUF variants are not implemented; only the tested default topology is validated |
| Memory adapter | real addresses/transactions, finite occupancy, per-line RMW ordering, callback completion, explicit clocks | Deliberately a documented abstract adapter, not cycle-exact `otpu_native_dram`/F2 shell. Partial QST writes incur explicit per-request RMW rather than undocumented shell combining; this adapter contract is identical on both sides of native RTL comparisons |

MXU counters count actual useful `M*N*KB*D` work and packed weight traffic; zero values do not
change those counts. K is block-granular in the ISA: compiler zero padding remains real work.
There is no assumed ternary compression or sparse bypass. Capacity does not imply hardware fit.

## Supported and rejected configurations

Supported in this initial implementation: one slice, D=32/64/128, LANES=4/8/16, MCOLS=2/4/8/16,
ACT row replay, PAIR where the ISA permits it, NOP/HALT/LI/ADDI/LOOP/BAR/RLD/LD/ST/MM/QACT/QST/
VOP/DSTEP/STREAM and optional SEQ_WC. The stream engine requires eight lanes. Zero/illegal shapes, buffer overflows and unknown functions/flags fail rather than invent
cycles. Functional semantic errors still raise the original ISA errors.

Explicitly rejected: S>1/GATHER interconnect timing, WAITW host polling,
HALT CHAIN/program-loader timing, unknown wide-export configuration
fields, unsupported memory backends and divided controller clocks. These are coverage gaps,
not operations assigned zero cost. Architecture sweeps beyond the accepted domain fail.

### Ramulator presets

The supplied HBM2 profile selects upstream `HBM2_8Gb`, 8 channels, `channel_width=64`,
`HBM2_2Gbps`, `tCK=1000 ps`, FRFCFS, OpenRowPolicy, all-bank refresh and RoBaRaCoCh mapping.
**This pinned upstream HBM2 implementation reports 16 bytes/transaction** (prefetch=2,
channel width=64); the bridge reads and checks this value. Do not reinterpret its bandwidth
as a measured JEDEC device/F2 subsystem. DDR4 selects two channels of `DDR4_8Gb_x8`, one rank,
`DDR4_2400R`, tCK=833 ps, 64 bytes/transaction. Only these supplied presets have matched closed-loop RTL tests.
Other Ramulator standards are not claimed supported merely because upstream contains them.

The upstream Generic controller lacks write callbacks. `prepare.py` adds a callback on write
command issue; the bridge delays completion by nCWL+nBL memory ticks. This is write data-bus
completion, not power-safe persistence. Read completions use upstream's callback. The patch
also selects the first level under channel for refresh (rank for DDR4, pseudochannel for HBM2)
and adds the C++ dependent-template keyword required by strict Clang. No model code or testbench
in the OpenTPU repository is patched for this integration.

## Metrics definitions

`cycles`: half-open interval [0,cycles), core start through drained HALT. `seconds`: cycles times
core period. Physical `read_bytes/write_bytes`: accepted backend transactions times backend
transaction size, including RMW. Logical bytes count port payloads before splitting. Transactions
and completions carry unique IDs. Retried requests do not increment accepted byte/transaction
counts. `peak_outstanding` covers queued plus inflight transactions, including RMW write tails.

`unit_busy_fraction`: union of each unit's command start/end intervals divided by total cycles;
it includes unit waiting on memory. `mxu_advance_fraction`: actual model advances / cycles.
`useful_mac_fraction`: useful MACs / (cycles*D*MCOLS). Stalls are **micro-operation blocked events**;
producer and consumer can block in the same cycle, so counters are not mutually exclusive or
additive wall-time buckets. Region `span_cycles` includes waits and can overlap other regions.

## Matched behavioral-memory validation backend

`rtl_fixed` models the existing `otpu_dram` AXI=0 contract for identical RTL-program comparisons. It must be explicitly selected; it is never substituted for Ramulator. Core/controller/memory periods must agree, `transaction_bytes` must equal D, and `rmw` must be false. B uses D-byte physical transfers, A reads 32-byte scale lines, and SW writes four physical bytes with native masks. Local requests are registered before the DRAM edge; read tickets become unit-visible at local-request cycle + DRAM_LAT + 2, writes at +2. Each port accepts at most once per core cycle. This behavioral memory has no controller queue or channel-contention model. Its pending response pipeline follows DRAM_LAT, not the generic adapter's outstanding setting. It does not model DDR/HBM or the FPGA shell.

The matching receipts use D=32, LANES=8, MCOLS=4, RPB=32, WPB=1, WIN=16 and DRAM_LAT=8. Exact results for this subset do not establish all supported parameter combinations. The legacy `vpu_slots` field accepts only its default (10); arithmetic paths are fixed and the actual command queue depth is two. Non-default C8 lane widths and TMEM read latencies are rejected. `dma_buffer_chunks` and `dma_store_run` are explicit capacities; overrides are design experiments until compared against corresponding RTL parameters.

## Registered native-port contract (`memory.native_ports=true`)

`ports.py` adds one elastic request register per A/B/SW port and one input register for
B read responses. `PortController` accepts B, A, then SW subject to finite adapter capacity,
returns reads in per-port acceptance order (at most one response per port per core edge),
and drains write completions into global write-idle. The same class runs inside the RTL DPI
driver. Scale addresses align to 32-byte lines. Native accepted byte masks must be contiguous.
No outstanding capacity or backend rejection is bypassed; the deterministic simple backend
uses the same controller when selected explicitly. `native_ports=false` retains the original
abstract core-facing adapter for historical receipts and experiments.

VPU and DMA bank writes include their registered TMEM outputs. VPU arithmetic-tail readiness
is distinct from physical write and sequencer completion; a denied output freezes all active
VPU arithmetic and pending done state. DMA store-run lengths snapshot the registered buffer
occupancy before the next producer update. BAR takes the singleton request/ack sequence;
RLD includes pending-read, conversion, multiply and scalar writeback stages. These details are
necessary even when total cycles accidentally agree without them.

Memory timestamps are global across launches; unit-local comparisons add the launch offset.
A dedicated persistent-launch regression guards against comparing local core cycles with
absolute memory completion cycles. Round four compares five-launch sequences against a persistent RTL process, including explicit
reset/host gaps and retained TMEM/ACT probes. See [the exact scope](validation-round4.md).

MXU weight FIFO credits are reserved per accepted chunk across active commands and released
on the actual final replay consumption; packed int4 halves do not release early. DMA store
chunks remain occupied until port acceptance. A full output FIFO freezes its read pipeline,
including pending stages. Long-K and 128-by-128 DeltaNet regressions cover these boundaries.

The production board target remains outside the validated topology: pumped/replay TMEM,
MXU_E=4 and the DDR3/LiteDRAM shell are not validated. Round five adds baseline WAITC and
DSTEP/STREAM, with evidence and remaining limits in [validation-round5.md](validation-round5.md).


## WAITC and DMA stream engine

With `Config.SEQ_WC=true`, the unchanged compiler's `wait_counts` emits one 64-bit entry per
static instruction. The default uses conservative unknown-runtime-argument ranges. A caller
can supply `wait_entries` explicitly, with one valid entry per instruction. Seven four-bit
unit distances plus a 31-bit older-VPU mask determine readiness; same-unit counts are ignored,
while the VPU mask still applies. A two-stage frontend dispatches first at cycle 3, registered
readiness takes two edges, non-VPU units preserve unit order, and the VPU selects oldest ready.
The 31-entry VPU tag lifetime stalls dispatch before aliasing. Dynamic LOOP iterations consume
the static compiler table and maintain dynamic unit history. No average issue-rate shortcut.

`stream.py` follows the original DMA/SE control path: descriptor TMEM reads; VPU drain/ownership;
q/k/optional-A/column-gate/x/constant fill; 128 input chunks; 64-chunk read/write runs; three
registered enable stages; 128-chunk output gather; read availability and write-idle completion.
The SE arithmetic progresses only on enabled edges, with Y after X + T_U(97) + U(7) = 105
steps and q output after another 35 steps. Backpressure freezes enable propagation. Columns64
use sixteen eight-lane segments, with eight pad segments discarded from output. Independent
arithmetic still executes through the ISA at completion and is checked separately.

DSTEP supports rows1..256 and columns64/128/192/256, zero initialization and in-place state.
STREAM accepts the unchanged `stream_hw_cfg` hardware subset: DELTA, DELTA1, SCALE and DOT,
constant/column/unit gate, optional q output and A slot1/2, aligned identical or disjoint DRAM
ranges. TMEM-source/destination, nodst, partial-overlap ranges, arbitrary programs, persistent
scalar registers and other descriptors raise `Unsupported`. The measured modes and geometries
are finite coverage, not an exhaustive descriptor proof.

## Exact host-side batching

The optional `otpu_advance` ABI executes every Ramulator tick in C++ until the first callback or
the next core boundary. Python then processes completion/RMW/ingress before advancing again.
No core or memory cycles are skipped, and refresh still runs. With ingress queued, only one
tick is fused with completion polling. `OTPU_PERFSIM_MEMORY_BATCH=0` selects scalar tick/poll;
older libraries use that same scalar path. There is no backend substitution. Metrics include
`host_execution.ramulator_tick_batching` and `core_cycles_skipped` (zero).
