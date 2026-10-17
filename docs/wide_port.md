# A wide port B: 2x and 4x the weight bandwidth of a slice

Status: ISA, simulator and compiler done (Config.PORTB, tests/test_wide_port.py); RTL in progress.
Validated in Verilator first; FPGA synthesis only in the nightly build.

## Why

A slice streams one D = 128-byte weight chunk a cycle on port B: 25.6 GB/s at 200 MHz. The F2's
HBM gives ~460 GB/s over 32 pseudo-channels, and Alex's burst200 image already streams ~20 GB/s,
~78% of the port. Decode at batch 1 is weight-bound, so a slice's tokens/s is the port's rate
over the bytes a token streams. More HBM ports only help once a slice takes more than one
chunk a cycle.

At M = 1 (one decode row) an int8 MM uses one of the MXU's MCOLS = 4 columns: the other three
are idle. Column reuse (docs/isa.md "Column reuse": PAIR for 4-bit, QUAD for t2) feeds them more
blocks of the same row. Generalized to every format, and with the port taking as many chunks a
cycle, a slice streams 2x or 4x the bytes with no more multipliers:

| format | reuse | blocks a cycle | chunks a cycle (PORTB needed) |
|---|---|---|---|
| int8 | PAIR | 2 | 2 |
| int8 | QUAD | 4 | 4 |
| int4 / fp4 / nvfp4 | PAIR | 2 | 1 (today) |
| int4 / fp4 / nvfp4 | QUAD | 4 | 2 |
| t2 | QUAD | 4 | 1 (docs/ternary.md) |

## ISA

* `Config.PORTB` in {1, 2, 4}: the D-byte chunks a cycle port B and the MXU's stream take (CAPS
  and CFG register: RTL work). In the arch hash only when not 1.
* MM PAIR (flags bit6) for any format; QUAD is rs bit 1 with PAIR, for any format (t2's rs bit 0
  is separate). Reuse r = 4 (QUAD), 2 (PAIR), 1; block k of output row j comes from ACT row
  j + (k mod r) M; r M <= MCOLS; k0 a multiple of r. An MM's rate, r / blocks a chunk, must be
  at most PORTB (isa.chunks_per_cycle; the simulator refuses more).
* QACT DUP4 with T2 or PORTB > 1.
* The result is the same integer per output: reuse changes no bit.

The compiler takes the widest reuse the port allows: a stationary operand of at most MCOLS/4
rows is quantized with DUP4 (MCOLS/2: DUP), and each MM pairs or quads where its format's rate
fits PORTB.

## RTL (to do)

Port B carries PORTB chunks a cycle (PORTB x 1024 bits) from the slice through otpu_native_dram
(PORTB x 2 beats a cycle over its NCH channels: f2-hbm's 8 channels feed PORTB 4), the MXU's
chunk FIFO takes PORTB chunks a cycle, and the S1 decode hands each column of a reuse group its
block (the QUAD datapath of docs/ternary.md 6 generalized: int8 blocks are one chunk each).
Port A delivers the reuse group's scale words a cycle (int8: r dw bytes after Sw). The DMA's
LD / ST stay one chunk a cycle.

### Placement (f2-hbm:p4rl8, the first placed build, 2026-10-17)

The first synthesis and placement of `f2-hbm:p4rl8` (PORTB 4 / MCOLS 8 / RL 8 / NCH 8 on the
VU47P with the F2 HDK shell, Vivado 2025.2; post-synthesis 251,520 LUT, 196,289 FF, 426 BRAM,
1,275 DSP) placed at WNS -4.888 ns, TNS -123,690 ns, 102,854 failing endpoints of 1,019,569.
The post-synthesis report had only the MXU's `mkr` family failing (the other work on this
branch); everything else appeared at placement, 90-97% of each path routing: fan-out, bus
width and SLR distance, not logic depth. By domain, with what changed for it (every change
bit-exact with the ISA simulator; nothing in tb_top's sources, so the regression suites'
cycles are the same by construction):

* **hclk (450 MHz, 2.222 ns): WNS -4.888, 34,489 failing endpoints of 110,006**, the worst 600
  into `u_port/m_wdata_reg` (327) and `m_wstrb_reg` (115), `u_host/btarget_reg`, `m_awaddr_reg`
  and the bridges' `u_rq` LUT RAMs, from `hbm_reset_n_reg` (120), `u_host/hw_s_reg` (115),
  `hw_d_reg` (110), `ws_reg`'s replicas (213) and `rq_d_reg` (18): 2 levels and 6.9 ns of
  route. The arithmetic: the host path's head beat is one 512-bit register (`hw_d`, two AXI
  beats) whose half, selected by `ws`, went on a shared 256-bit bus to all 8 ports' arbiters,
  whose W registers (8 x 256 = 2,048 bits, 8 x 32 of strobe) took it on their own `wsel`: each
  `hw_d` bit 8 loads across the ports, `ws` 8 x 288 = 2,304 mux selects (hence the replicas),
  and the 8 ports' R data (8 x 256 bits, from the HBM pins through the arbiters) AND-ORed into
  `rq_d` in one cycle. None of it is PORTB's: the hclk domain has no PORTB, the host's bus is
  256 bits at PORTB 1 and 4 alike (the "4096-bit" register is the 8 ports' 256-bit W and R
  registers seen together), so this is inherited from f2-hbm and will show in its base build
  too. The change (the smallest of the three options: not a narrower host path, which is
  already a port's width; not the host into clk through the adapter, which would move the
  ordering contract): **a register per port for the host's master** (`otpu_f2_hbm_hreg` in
  otpu_f2_hbm_mem: AW, W, AR taken from the host when empty, the ready a flip-flop; R and B
  registered back), so every net between the host block and a port starts and ends at a
  register with the cycle for its route, and no hclk net carries more than one port's 256
  bits. The host's write rate goes from a beat per ~4 to ~7 hclk cycles (4 GB/s against the
  PCIe's 2). The reset: `hbm_reset_n[2]` (an ASYNC_REG flip-flop, not replicated) drove ~1,250
  reset pins along the HBM's 8 ports (a bridge's hclk side ~130 with its two crossings, the
  arbiters, the host path's ~150); now a register in otpu_f2_hbm_mem (max_fanout 4) and one in
  every module (nch, port, hreg, host), three flip-flops from the synchronizer to any pin.
* **clk (200 MHz, 5.0 ns): WNS -4.052, 66,008 failing endpoints of 894,614**: `core_reset_n_reg`
  (406 of the worst 600: 4 levels, 92% route), `u_nch/cf_h_reg` into the bridges' `u_xq`
  crossing (160: 9 levels), `u_mxu/mk_reg -> mkr_reg` (32: the other work). The reset:
  `core_reset_n[2]` drove, with no register of its own, ~4,700 pins: the adapter's ~2,200
  (per channel the queue, tag and FIFO counters, the 64 hazard-bucket flags, the 8 B-tail
  copies), the 8 bridges' core sides (~230 each with the crossings' pointers and the 64 read
  tags), the control block, the trace, the loader and the slice's port registers; the units'
  ~15k are already behind core_rst's tree (otpu_board, otpu_slice). Now otpu_board registers
  it (`rst_q`, max_fanout 128: synthesis replicates it as it does core_rst, each replica by its
  loads) for everything but go_ok / core_rst, and otpu_f2_hbm_mem and each bridge register it
  again. `cf_h` is not the channel hash (that is the adapter's `pair_of`, registered at BREG
  since f2-hbm): it is the bridge's command FIFO head pointer, and the 9 levels were the head's
  LUT RAM read into the order check's 80 compares of 25 bits (16 writes and 64 reads in
  flight), the issue, and the crossing's 600 write enables. Now the head is a register
  (`h_*`: the compares and the issue start from flip-flops) and the crossing writes its tail
  slot every cycle there is room (`otpu_afifo` `WT`, with `RWR`): the issue moves only the
  pointer. A command issues a core cycle later than before (5 ns on the HBM's ~400); the order
  check, the request order and the adapter are untouched, so the two-part reads
  (`NATIVEB split`: 0 with row lanes, 7.3% at PORTB 2 QUAD on Gemma's 9-chunk rows) are the
  same counters on the same adapter.
* **clk -> clk_f (the pumped TMEM, 2.5 ns): WNS -2.356, 2,067 endpoints**: `u_vpu/ta_addr_reg`
  (60) and `rl_pq_reg` (40) into the pumped banks' ADDRARDADDR, 7-10 levels, 90% route. These
  paths are the ones otpu_clk2x.tcl gives one core period (multicycle 2 -end from clk through
  `u_pb`'s read pins; the HBM synthesis script errors out if it is not applied, and the
  post-synthesis report met it): the VPU's `ta_addr` *is* the register at the pump boundary
  (release r2's input stage), and what follows it, TMEM's rotation, sub-bank hash and rank
  multiplexer into 128 RAMB36 spread over the slice, is the read's own cycle. A further
  register would be a cycle of TMEM read latency on every port, which every unit's data
  pipeline and the replay protocol assume (not a local change; not made). What the report
  says is that 5 ns did not cover 7-10 levels plus ~6 ns of route: the placement spread the
  slice (a flow matter, below), and the path requirement to check in the report is 5.000, not
  2.500.
* **clk_main_a0 (250 MHz): WNS -0.744, 265 endpoints; clk_main_a0 -> hclk -2.861 (45)**: the host
  path's aclk side and its crossings into hclk. The PCIS W crossing now writes its tail every
  cycle (`WT`), so the PCIS W take reaches 604 fewer write enables; the crossing bounds (below)
  are the rest.

**What the build flow must add** (not RTL): (1) the placement strategy: build.py runs the HDK's
build_all with `SSI_SpreadLogic_high`, which spreads one slice of 250K LUTs over the three
SLRs; the slice with its adapter wants one SLR and the HBM fabric (u_sys/u_mem: bridges, host
registers, arbiters, host path) the bottom of SLR0 by the HBM's AXI ports: pblocks for the
two, and `Default` or `SSI_SpreadLogic_low`; (2) the crossings' bounds: cl_timing_user_hbm.tcl
now bounds each direction by the *destination* clock's period (it bounded both by the smaller
of the pair: the bridges' return FIFOs' data into the core at 2.2 ns across an SLR instead of
5); (3) with the pblocks, the bridges' core sides (the adapter's 512-bit native buses, 5 ns)
are the SLR crossing, as docs/f2_hbm.md planned. The base f2-hbm build at the same commit
shows what of this the shell and the host path already fail at PORTB 1.

## Expected effect (projections, Gemma-3-1B int8, ~1.0 GB a token)

| PORTB | weight ceiling | at the measured ~78% |
|---|---|---|
| 1 | 25 tok/s | ~20 |
| 2 | 50 | ~38 |
| 4 | 100 | ~70 |

The per-token fixed work (VPU, attention, norms, launches) does not shrink, so the gain is
below 2x / 4x by its share; the RTL timing runs (tools/perf_qwen.py on the f2-hbm memory model)
measure it.

## Baseline on the RTL (PORTB 1)

tools/perf_qwen.py, Gemma-3-1B int8 (int8 head), OTPU_TARGET=f2-hbm OTPU_NCH=8, --lat 60, 4
layers and the whole 262K-row head: 3,228,204 cycles against a port-B roofline of 3,207,980
(99.4%); MXU busy 100%, VPU 2%, MXU starved 18 cycles. By phase: the head 2.37 M cycles, the
4 layers' MLP 0.75 M, attention 0.11 M. The whole model (26 layers): about 7.96 M cycles a
token, 39.8 ms at 200 MHz, ~25 tok/s, the port's rate: decode is port-bound on the RTL, so
PORTB 2 and 4 should come near 2x and 4x (the board's ~78% port efficiency on Alex's image is
the memory path's, which more HBM channels a slice address).

## Future improvements

### Port B as independent lanes (from the SGMV streaming GEMV engine)

Ergodex-Core/coral-gemma-f2's SGMV engine (rtl/engine) streams int8 weights from HBM in L = 4
lanes, each its own AXI master on its own rows: 64 B a cycle per lane at 250 MHz (64 GB/s
peak), bursts of at most 8 beats that never cross 512 B, a 128-256-beat read FIFO as credits
with 16-32 bursts outstanding (rready always 1), ~0.97 beat/cycle per lane at 150-cycle HBM
latency; ~36 GB/s measured on F2 (Gemma 3 1B: ~1 GB a token in 27.7 ms), ~20K LUT and 284 DSP
for 4 lanes. Its 15.8 tok/s is bound by its 25 MHz Coral NPU, not by the engine.

What is worth taking, and what is not:

* **Take: the lane structure for PORTB > 1.** Build the wide port as PORTB independent
  one-chunk lanes, each on its own HBM channel pair with SGMV-style credits and bursts, each
  feeding one MXU column group of a reuse (column c of QUAD / PAIR), instead of one 2048-4096-
  bit port B through the slice and otpu_native_dram. Same ISA and numerics (an MM's blocks k
  with k mod r = c are lane c's stream, contiguous per row), timing stays local to a lane at
  200-250 MHz, and it scales with MCOLS (8 lanes at MCOLS 8) and with HBM ports.
* **Take: 250 MHz** if the lanes close it there (SGMV routed at 250 on the same part): +25%.
* **Not now: a separate GEMV engine beside the MXU.** At M = 1 the MXU already is a streaming
  GEMV at 99.4% of its port on the RTL (above), with three idle columns that reuse feeds; the
  VPU is ~2% of a decode token, so SGMV's fused MLP / ARGMAX post-passes save little; and a
  second datapath would have to reimplement openTPU's MX numerics (scale rows, the shift window,
  int4, t2) to stay bit-exact with the ISA. Revisit if decode needs more than MCOLS x D weights
  a cycle, or GEMVs must run beside prefill / batched MMs on the MXU.
