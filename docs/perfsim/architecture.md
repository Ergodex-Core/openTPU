# Architecture and fidelity contract

The model is instruction-driven and cycle-stepped. Functional execution uses the unchanged
`opentpu.isasim.Slice.execute` at command completion, with dispatch-time register snapshots.
The timing scheduler does not use its result to synthesize latency or grant sparse speedups.
A separate sequential ISA run checks scheduler correctness, and independent NumPy/RTL checks
cover arithmetic. Sharing ISA arithmetic is explicitly **not** an independent numerical oracle.

## Dataflow

1. In-order dispatch resolves scalar register-relative addresses and expands LOOP dynamically.
   A finite issue window retains outstanding commands. RLD blocks younger dispatch until its
   scalar result commits. BAR waits for older commands; HALT waits for all commands and writes.
2. `opentpu.waits.footprint/conflict` supplies the same conservative RAW/WAR/WAW address-range
   semantics as `otpu_pkg`. Per-unit queues prevent in-order units from overtaking an older
   unstarted command; VPU can choose an independent younger command. MXU prefetch is released
   on DRAM dependencies separately from activation/TMEM dependencies.
3. All unit steps arbitrate banked TMEM at `word_address % LANES`, with separate per-bank read
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
| Issue | finite window; control/register/loop handling; range conflicts; separate MM stream dependencies; 2-entry MXU queue | `rtl/seq/otpu_seq.sv`, `opentpu/waits.py`. Fetch/resolve overlap represented by configurable dispatch delay, not every sequencer register |
| MXU | actual D-byte block stream, row-gated consumption, finite prefetch FIFO, scale-line requests, ACT row replay, int8/int4/fp4/nvfp4 packing, PAIR, partial M/N and KB; result FIFO/drain, ACC/ASCALE/RMAX | `rtl/mxu/otpu_mxu.sv`: CLS=min(D/8,CL), LROW=CLS+MCOLS+5, epilogue lanes default MCOLS/2. Micro-op epilogue/command overlap and scale holding-register scheduling are structural approximations, not an RTL port |
| VPU | all current arithmetic/copy/fill/nonlinear/reduction/OUTER functions, broadcast addressing, masked partial lanes, 64-partial reduction tail, buffered pipeline writes | `rtl/vpu/otpu_vpu.sv`: LM=2, LA=4, SL=7; ADD/MUL/OUTER 1 slot, EXP2/EXP2SUB 9, RECIP 6, RSQRT/LOG2 10. Lane tapping, per-partial recurrence and inter-command issue arbitration are simplified |
| Quantizer | row-buffer capacity, max/prescale read pass, reciprocal/Newton latency, second ACT pack pass, QACT DUP/ROW/CSCALE/RSCALE; QST ROW/HALF/strided bytes and scale writes | `rtl/vpu/otpu_quant.sv`: LM=2, LA=4, QL=5, reciprocal tail=30. Rows processed serially; four-bank row overlap and overflow reread path not implemented (overflow rejected) |
| DMA | D-chunk boundaries, lane gathering/scattering, finite gathered chunk, port-B sharing and read/write completion | `rtl/dma/otpu_dma.sv`. Conservative single-chunk scheduling, not the full RTL lookahead pipeline |
| TMEM | modulo banking, independent configurable read/write capacity, address-specific contention, pipeline drain stalls | `rtl/mem/otpu_tmem.sv`. Abstract bank model; exact replicated-port guests, subbank/hash/pump modes and priority encoding not implemented |
| Memory adapter | real addresses/transactions, finite occupancy, per-line RMW ordering, callback completion, explicit clocks | Deliberately a documented abstract adapter, not cycle-exact `otpu_native_dram`/F2 shell. QST byte writes currently incur individual RMW traffic rather than the hardware's possible write combining; results can be pessimistic |

MXU counters count actual useful `M*N*KB*D` work and packed weight traffic; zero values do not
change those counts. K is block-granular in the ISA: compiler zero padding remains real work.
There is no assumed ternary compression or sparse bypass. Capacity does not imply hardware fit.

## Supported and rejected configurations

Supported in this initial implementation: one slice, D=32/64/128, LANES=4/8/16, MCOLS=2/4/8/16,
ACT row replay, PAIR where the ISA permits it, NOP/HALT/LI/ADDI/LOOP/BAR/RLD/LD/ST/MM/QACT/QST/
VOP. Zero/illegal shapes, buffer overflows and unknown functions/flags fail rather than invent
cycles. Functional semantic errors still raise the original ISA errors.

Explicitly rejected: S>1/GATHER interconnect timing, WAITW host polling, DSTEP/STREAM,
HALT CHAIN/program-loader timing, SEQ_WC wait-count issue, unknown wide-export configuration
fields, unsupported memory backends and divided controller clocks. These are coverage gaps,
not operations assigned zero cost. Architecture sweeps beyond the accepted domain fail.

### Ramulator presets

The supplied HBM2 profile selects upstream `HBM2_8Gb`, 8 channels, `channel_width=64`,
`HBM2_2Gbps`, `tCK=1000 ps`, FRFCFS, OpenRowPolicy, all-bank refresh and RoBaRaCoCh mapping.
**This pinned upstream HBM2 implementation reports 16 bytes/transaction** (prefetch=2,
channel width=64); the bridge reads and checks this value. Do not reinterpret its bandwidth
as a measured JEDEC device/F2 subsystem. DDR4 selects two channels of `DDR4_8Gb_x8`, one rank,
`DDR4_2400R`, tCK=833 ps, 64 bytes/transaction. Only these supplied presets have smoke tests.
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
