# A wide port B: 2x and 4x the weight bandwidth of a slice

Status: ISA, simulator and compiler done (Config.PORTB, tests/test_wide_port.py); RTL built and
validated in Verilator (section RTL: its state on 2026-10-16, the decision to finish it as one
wide port, the tests and the measurements); not synthesized: FPGA synthesis only in the nightly
build (section "What the nightly needs").

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

* `Config.PORTB` in {1, 2, 4}: the D-byte chunks a cycle port B and the MXU's stream take
  (CAPS2 [3:1], section RTL). In the arch hash only when not 1.
* MM PAIR (flags bit6) for any format; QUAD is rs bit 1 with PAIR, for any format (t2's rs bit 0
  is separate). Reuse r = 4 (QUAD), 2 (PAIR), 1; block k of output row j comes from ACT row
  j + (k mod r) M; r M <= MCOLS; k0 a multiple of r. An MM's rate, r / blocks a chunk, must be
  at most PORTB (isa.chunks_per_cycle; the simulator refuses more).
* QACT DUP4 with T2 or PORTB > 1.
* The result is the same integer per output: reuse changes no bit.

The compiler takes the widest reuse the port allows: a stationary operand of at most MCOLS/4
rows is quantized with DUP4 (MCOLS/2: DUP), and each MM pairs or quads where its format's rate
fits PORTB.

## RTL

### State on 2026-10-16

What the stopped agent left, as found (commits 585ebe85 "WIP wide port" and c68be81c, its
uncommitted edits committed as they were):

* **Complete in the RTL, one wide port.** Port B carries a request's chunk count (`b_n`) and
  a response's chunk mask and last flag (`b_rcm`, `b_rlast`) from otpu_mxu through otpu_slice's
  port registers to otpu_dram (one response a request) and otpu_native_dram (a chunk a channel
  pair, a request in one or two parts); the MXU's chunk FIFO entry is PORTB chunks with
  per-chunk write enables, its issue asks for an advance's chunks in one request, the scale
  path holds a reuse group's data (a third entry RAM for blocks 2 and 3, the two-line window
  for 4-bit / NVFP4 QUAD), and S1 feeds the QUAD datapath's four streams from the entry's
  blocks for every format. `PORTB` goes through otpu_top / tb_top, otpu_board, otpu_f2_sys,
  otpu_f2_hbm_sys, otpu_asic_top (one chunk), otpu_ctrl (CAPS2 [3:1]), `target.rtl_params`,
  `build_config`, `rtlsim.top_params` (and the adapter's NATIVEB counters into
  `stats["native_b"]`), tools/perf_qwen.py `--portb`. 585ebe85 was the RTL and the slice; the
  uncommitted edits were the host side (regs.caps2 `portb`, device_config / check_hash /
  FakeTransport / otpu-diag) and CAPS2's encoding (0 = one chunk, so a PORTB 1 bitstream's
  CAPS2 is as before).
* **Verified here:** tests/test_wide_port_rtl.py, 53 tests, pass (Verilator 5.048, 65 s on the
  warm cache): every reuse format on the behavioural DRAM and on the native path (NCH 4 / 8 /
  16, random stalls), bit for bit the ISA simulator's, and the throughput test (PORTB chunks a
  cycle).
* **Left undone.** (1) tests/test_host.py still had two asserts of `caps2` without the `portb`
  key (the suite was red); FakeTransport's CFG_HASH left PORTB out, so a PORTB 2 fake card
  would not pass check_hash. (2) The board model the host tests drive (sim/verilator/tb_board.sv,
  SimTransport) had no PORTB and no NCH parameter: it was two channels a slice, and
  otpu_native_dram needs NCH >= 2 PORTB, so the board path (CAPS2, CFG_HASH, the host's
  register protocol) could not be built at PORTB 2 at all. (3) No measurement of a whole
  model: the Gemma numbers below were two layers and the head; nothing on Bonsai-2. (4) The
  doc said "RTL done".

### Design

A top-level parameter `PORTB` (1, 2, 4; default 1) through otpu_top / tb_top, otpu_board, the F2
system tops (otpu_f2_sys, otpu_f2_hbm_sys) and otpu_slice; `opentpu.target.rtl_params` passes the
target's, `rtlsim.top_params` a `Config(PORTB=...)`'s (OTPU_PORTB through `board_config`). The
control block reports it in CAPS2 [3:1] (2 or 4; 0 on a one-chunk build, so a PORTB 1
bitstream's CAPS2 is as before); `regs.caps2` reads it, `device_config` and `check_hash` take
Config.PORTB from the bitstream (OTPU_PORTB must agree), FakeTransport(portb=...) models it,
`otpu-diag` accepts the field. PORTB 1 builds none of what follows (every term below is a
constant or a generate branch not taken): the regression suites pass unchanged, and the
two-channel adapter is still every-cycle equal to its reference (tests/test_native_nch.py).

**Port B.** A request carries `b_n` (chunks - 1, consecutive chunks; the MXU's only, the DMA's
and the loader's stay one chunk, their writes too), a response up to PORTB chunks (`b_rdata`
PORTB x 1024 bits, chunk c at bit 1024 c), `b_rcm` the chunks it carries and `b_rlast` (the
request's last part). The slice's port-B output register takes `b_n`, its input register the
wide data. The behavioural DRAM (otpu_dram) answers a request in one response.

**otpu_native_dram** (NCH >= 2 PORTB). At BREG (the register after the take, NCH > 2) each
chunk of a request gets its address (+32 words a chunk), pair, halves and channel beat, from the
slice's b_addr flip-flops. Chunk m is on pair `(m mod NP) ^ fold(m / NP)`, a bijection over each
aligned group of NP = NCH / 2 chunks, so a request inside one group needs PORTB distinct pairs
and goes out in one cycle, PORTB x 2 beats over 2 PORTB channels. A request that crosses a group
boundary can need a pair twice (the next group's fold): it then goes in two parts, the first
occurrence of every pair, then the rest, a cycle later (b_rdy low in the first part's cycle,
the queues keep QD - 2, the tag FIFO OD - 2). Each part is a tag entry (per chunk: valid, pair,
halves; the last flag) and a response once every one of its pairs' two channels holds a beat,
so no queue takes two beats in a cycle and the merge pops a channel at most once a cycle: the
MXU writes a response's chunks into its FIFO entry (per-chunk write enables) and counts the entry
on `b_rlast`. On the Gemma decode below at NCH 8: 7% of the wide reads at PORTB 2, 57% at PORTB 4 (Gemma's
9-chunk rows put most 4-chunk reads across a group); each costs a port-B cycle.

**MXU.** `QD = T2 || PORTB > 1` builds the QUAD datapath of docs/ternary.md 6 (four weight
streams through the column hops, the PE input select by `hq`, two more scale decoders, the
four-column row sum); `cmd_pair` / `cmd_qd` hold for every format on a wide port, and an MM's
chunks an advance are 1, 2 (int8 PAIR, 4-bit QUAD) or 4 (int8 QUAD; checked against PORTB).
- *Issue.* A FIFO entry is PORTB chunks; a wide MM asks for an advance's chunks in one request
  (a row's last advance for the chunks left: `i_nl` from KB mod 4) and steps its chunk address
  by them. A narrow MM's entries hold one chunk, as before (a replayed row, M > MCOLS, still
  needs KB entries).
- *Scales.* An advance's line holds the reuse group's data: int8 PAIR two bytes (k0, k even:
  never across a word), QUAD four (an aligned word, k0 % 4 == 0 and srs, ssa % 4 == 0: the
  compiler's `al` 4), step 2 / 4 bytes; 4-bit QUAD four words (16 bytes), NVFP4 QUAD four 8-byte
  groups (32). The ISA aligns those to 8 / 16 bytes only, so they cross a 32-byte line at offset
  24 / 16 (int4 rows of `8 + 8 ceil(KB/2)` bytes alternate); the issuer then asks for the first
  line alone (a cycle without a chunk, as a row's lone Sw request), stages it (`r_stl`, 256
  bits; its Sw loads r_sw when the row's Sw is in it) and completes the entry from both lines
  with the chunks' request (an 8-byte-granular 4:1 window over 512 bits). The entry's third RAM
  (128 x DEPTH) holds blocks 2 and 3's data; slots 0 / 1 are the old {Sw, low} / high halves.
- *S1.* The four streams are the entry's blocks: int8 chunks 0..3 (hi = chunk 1), 4-bit chunk
  0's halves and chunk 1's; t2 keeps its 2-bit decode. Column cM + j takes block rc' + c, the
  column end its block's B (the 4:1 of decoded scales QUAD had), the row sum adds columns j, j+M
  (PAIR) and j+2M, j+3M (QUAD).
- *ACT RAM / quantizer.* Built with `T2 = QD`: QACT DUP4 and the four-block reads (r_blk3 / 4).

**Resources and timing at 200 MHz (estimated, not synthesized: the user's rule is Verilator
only).** D = 128, MCOLS = 4, FIFO_DEPTH 1024:
| | PORTB 2 | PORTB 4 |
|---|---|---|
| MXU chunk FIFO (block RAM) | 2048 x 1024 b: 58 RAMB36 (+29) | 4096 x 1024 b: 116 RAMB36 (+87) |
| scale entry third RAM | 4 RAMB36 | 4 RAMB36 |
| port-B input register (slice) + MXU write data | +1.0K FF | +3.1K FF |
| QUAD datapath (if not T2 already) | ~7K LUT, ~10K FF (docs/ternary.md 6) | same |
| S1's wide stream select (int8 chunk / 4-bit half / t2) | ~2K LUT | ~3K LUT |
| scale window (4:1 x 256 b), r_stl | ~0.3K LUT, 0.3K FF | same |
| adapter: per-chunk address, fold, pair compare | ~0.3K LUT | ~0.6K LUT |
| adapter: merge (NP:1 + halves swap a chunk, at NCH 8) | +1.5K LUT (2 x 1024 b) | +4.5K LUT (4 x 1024 b) |
The wide datapaths are 2048 / 4096 bits from the adapter's B FIFO heads through the merge into
the slice's input register, then into the chunk FIFO's RAMs: one LUT level more than today's
1024-bit path (the merge's NP:1 per chunk instead of one), with its selects (a chunk's pair and
halves, from the tag FIFO's registered head) fanning out to 1024 LUTs each: they want
replication, as the B FIFO's tail copies have. The chunk FIFO's groups are per chunk (each its
own address copies, four column groups a chunk). The new logic levels in the MXU are one LUT
each: the stream select after the FIFO's read register (int8 / 4-bit / t2 for hi, q2, q3), the
scale window and the `i_spx` test (two address bits) in the issue, which ends at registers
except a_addr's extra mux input (i_sa2, a register). In the adapter the per-chunk +32 adder and
fold start at the slice's b_addr flip-flops and end at BREG; b_rdy gains b_hold, a register
(the first part of a two-part request). At PORTB 4 on 8 channels every channel carries one beat
a cycle on average: the native port's limit (12.8 GB/s a channel) and above the HBM bridge's
~10 GB/s reads in flight (docs/f2_hbm.md): f2-hbm (NCH 8) suits PORTB 2; PORTB 4 wants
f2-hbm16 (NCH 16) or deeper bridges.

**Tests** (Verilator 5.048, D = 128, MCOLS 4 / 8, against the ISA simulator, DRAM and TMEM bit
for bit): tests/test_wide_port_rtl.py, 53 tests, each case on the behavioural DRAM and on the
board's memory path (otpu_native_dram + otpu_native_mem, OTPU_NCH 4 at PORTB 2, 8 at PORTB 4,
20% random stalls): int8 PAIR at PORTB 2 (KB 1, 5, 9; k0 0 / 2; M 1 / 2), int8 QUAD at 4 (KB 1,
6, 9, 7; k0 0 / 4; M 1 / 2 at MCOLS 8), int4 / fp4 / NVFP4 QUAD at 2 (KB 3 to 9, k0 0 / 4 / 8,
scale rows offset 8 / 16 so advances cross lines), int4 PAIR, int8 PAIR and fp4 QUAD on a port
of four, a plain int8 MM of M 4, t2 QUAD with T2 and PORTB 2; each with scales, UNIT + RMAX, ACC
+ RMAX and UNIT + ACC + ASCALE. A replayed row (M 8 > MCOLS 4) then int8 QUAD on a PORTB 4 build;
QUAD's row choice on distinct ACT rows (M 1 / 2); one program of int8 QUAD / PAIR, int4 QUAD,
fp4 PAIR, NVFP4 QUAD, t2 QUAD, plain int8 / int4 / NVFP4, QUAD ACC, M 4 and int8 / fp4 QUAD with
RSC back to back (DRAM, NCH 8, NCH 16); the compiler's one-row int8 dot at PORTB 2 / 4. The
native runs check that wide requests happened (the adapter's NATIVEB counters: wide reads,
their chunks, the two-part ones; rtlsim stats["native_b"]).

**Throughput** (test_wide_port_rtl.py::test_wide_port_throughput_rtl, one-row int8 MM, KB 16,
the cycles 256 more weight rows add):
| PORTB (reuse) | behavioural DRAM | native path, NCH 8, 20% stalls |
|---|---|---|
| 1 (plain) | 4096 (N KB) | 4096 |
| 2 (PAIR) | 2048 (N KB / 2) | 2048 |
| 4 (QUAD) | 1024 (N KB / 4) | 1311 (3.1 chunks a cycle: each channel needs every cycle, the model withholds 20%) |

**Gemma-3-1B int8, two layers** (tools/perf_qwen.py --layers 2 --wformat int8 --head-format
int8 --lat 60 --mhz 200 --portb N, OTPU_TARGET=f2-hbm OTPU_NCH=8; the 262K-row head dominates,
84% of the PORTB 1 cycles):
| PORTB | cycles | speedup | port-B chunks | core-port roofline (port B / PORTB, port A, MXU) | efficiency | MXU blocked | wide reads (two-part) | tok/s at 200 MHz (2 layers + head) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2,798,478 | 1.00x | 2,787,778 | 2,787,778 (port B) | 99.6% | 0.3% | - | 71.5 |
| 2 | 1,643,595 | 1.70x | 2,787,778 | 1,536,296 (MXU: 5 advances a 9-chunk row) | 93.5% | 5.7% | 1,242,880 (87,619: 7.0%) | 121.7 |
| 4 | 1,308,691 | 2.14x | 2,787,778 | 925,224 (MXU: 3 advances a row) | 70.7% | 27.6% | 631,808 (358,331: 56.7%) | 152.8 |
PORTB 2 is 1.70x: Gemma's rows are 9 chunks (K = 1152), so PAIR takes 5 advances a row
(2, 2, 2, 2, 1: 1.8 chunks a cycle) and the head's MMs run at 1.8 / 2 of the port; PORTB 4 is
2.14x: QUAD takes 3 advances a row (4, 4, 1: 3 chunks a cycle, the MXU roofline 925K), and the
rest is the two-part requests: 57% of the wide reads cross a group of 4 pairs and need a pair
twice, each a cycle on port B (925K + 358K = 1.28M of the 1.31M; MXU blocked 27.6%). At PORTB 2
7% split (87K of the 107K cycles over the roofline). Rows aligned to PORTB chunks would never
split; the next step is to let a request's second part go with the next request's first when
their pairs differ (or to pad weight rows to whole groups), not measured here.

## Expected effect (projections, Gemma-3-1B int8, ~1.0 GB a token)

| PORTB | weight ceiling | at the measured ~78% |
|---|---|---|
| 1 | 25 tok/s | ~20 |
| 2 | 50 | ~38 |
| 4 | 100 | ~70 |

The per-token fixed work (VPU, attention, norms, launches) does not shrink, so the gain is
below 2x / 4x by its share; the RTL timing runs (tools/perf_qwen.py on the f2-hbm memory model)
measure it.
