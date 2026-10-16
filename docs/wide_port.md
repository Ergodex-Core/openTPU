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
