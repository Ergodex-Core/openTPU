# A wide port B: 2x and 4x the weight bandwidth of a slice

Status: ISA, simulator and compiler done (Config.PORTB, tests/test_wide_port.py); RTL built and
validated in Verilator at PORTB 2 and 4, the board model and the host at PORTB 2 (section RTL:
its state on 2026-10-16, the decision to finish it as one wide port, the tests, the
measurements: Gemma-3-1B int8 4,652,571 cycles a token at PORTB 2, 1.71x, 43 tok/s at 200
MHz); not synthesized: FPGA synthesis only in the nightly build (section "What the nightly
needs", targets f2-hbm:p2 and f2-hbm16:p4). PORTB 8 and int4 at PORTB 4 need a reuse the ISA
does not have (section "The whole-model tally"): the row lanes of docs/row_lanes.md, built and
measured in section "Row lanes": int4 with four lanes at PORTB 4 on MCOLS 8 reaches 135.7
tok/s on the RTL (the whole model measured), int8 83.5.

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

### Decision: finish the one wide port (PORTB 2); the lanes stay the PORTB 4 / MCOLS 8 path

The alternative ("Future improvements" below: PORTB independent one-chunk lanes, each on its own
HBM channel pair with credits and bursts, each feeding one column group of a reuse) was weighed
against what the code holds:

* **What a lane rebuild would and would not change.** The MXU side is the same under both:
  a chunk FIFO entry of PORTB chunks (one a lane), an issue that asks for an advance's chunks
  at once, the reuse group's scale data in the entry, S1's four streams from the entry's
  blocks. Only otpu_slice's port B (PORTB request / response channels instead of one wide one)
  and otpu_native_dram's front (a queue set a lane) would differ. The part the WIP had to get
  right for bit-exactness, and did (53 tests), is the part a rebuild would keep.
* **The adapter already spreads a wide request over PORTB channel pairs.** Chunk m is on pair
  `(m mod NP) ^ fold(m / NP)` (docs/f2_hbm.md), a bijection within an aligned group of NP
  chunks, so a request's chunks go to PORTB distinct pairs in one cycle and come back as a
  PORTB-pair join: at the HBM each chunk of a cycle is on its own pseudo-channel pair, as a
  lane would be, and each channel's bridge (otpu_f2_hbm_nch) already does the lane engine's
  job -- its own bursts and credits at 450 MHz, timing local to the channel. What the slice
  does not have is a lane's fixed pair: lane c's chunk of group u sits on pair `c ^ fold(u)`,
  which changes with the group. A lane on a fixed pair means dropping the fold, which exists to
  spread power-of-two strides (port A's columns, a transposed V) over the pairs, and which the
  host's address map (opentpu/host/board.py `split`, `beat_at`), every image on the F2's HBM
  and docs/f2_hbm.md encode. Not a change to make for PORTB 2.
* **What the shared structure costs, measured.** The two-part requests (a request across a
  group boundary that needs a pair twice): 7% of the wide reads at PORTB 2 on Gemma's 9-chunk
  rows (87K of 1.64M cycles), 57% at PORTB 4 (358K of 1.31M: the main loss there). Lanes with
  rows aligned to PORTB chunks would never split -- but the same is had inside the adapter by
  letting a split request's second part go with the next request's first part (their pairs
  differ), or in the compiler by padding weight rows to whole groups; both are PORTB 4 work.
  The merge is NP:1 a chunk instead of one (+1.5K LUT estimated at NCH 8, PORTB 2) and the
  2048-bit path from the B FIFO heads to the slice's input register is one LUT level more than
  today's 1024: synthesizable only in the nightly, and if it misses 200 MHz the fix is a
  register per chunk at the merge (latency only), not a restructure.
* **Risk and time.** Finishing: the board model, the host's hash, the regression suites at
  PORTB 2 and the measurements (this page). Rebuilding: a new slice port, a new adapter front,
  a new address map and the 53 tests over again, for a structure whose gain shows at PORTB 4
  (the F2's NCH 8 feeds PORTB 2 at 22% of a port's limit each; PORTB 4 is the one that wants
  f2-hbm16 or lanes, see "Resources and timing").

So: the WIP's one wide port is finished for PORTB 2 (below). The lane structure stays the
recorded direction for PORTB 4 and for MCOLS 8 (8 lanes), where the fixed-pair address map
and the split-free rows are worth their change.

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

**The regression suites at PORTB 1 and 2** (2026-10-16, Verilator 5.048; PORTB 2 under
`OTPU_PORTB=2 OTPU_PAIR=1`, the native path on `OTPU_NCH=4`; the machine shared with other
sessions' builds, load 100-185):
| suite | PORTB 1 | PORTB 2 |
|---|---|---|
| tests/test_wide_port_rtl.py | 53 passed (its own matrix: PORTB 2 and 4 builds) | " |
| tests/test_rtl.py (the board path: otpu_native_dram + the memory model) | 176 passed, 1 skipped (test_board_memory_path_stress[0-0] on a rerun: in the full run clang crashed compiling its Verilator build under the load, and the later tests' identical build then succeeded) | 176 passed, 1 skipped (the same) |
| tests/test_t2_rtl.py | 24 passed (test_t2_rules updated: a9d539a6 made 4-bit QUAD legal) | 24 passed |
| tests/test_observability.py (tb_board, SimTransport) | 18 passed | 18 passed (NCH 4, CAPS2 [3:1] = 2, CFG_HASH of Config(PORTB=2); the starve test's bounds for the two-part requests) |
| tests/test_host.py | 84 passed, 1 skipped; 4 wall-clock timing tests and one needing `transformers` fail on the loaded machine (5 of 9 pass on a rerun) | 77 passed, 1 skipped; the same environmental ones, 5 load-flaky ones that pass on a rerun, and 4 selftest cases on register-map 2 / 3 fakes, which cannot report PORTB and so mismatch OTPU_PORTB=2 by design (the host follows the bitstream) |
| tests/test_target.py, test_wide_port.py, test_pair.py, test_f2_build.py (snapshots) | 100 + 7 passed | - |
Cycles at PORTB 1 are unchanged: every PORTB term is a constant or a generate branch not taken,
and the PORTB 1 builds' cache keys do not change (top_params adds PORTB only when wide).

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

### Measured on the RTL (2026-10-16)

Verilator 5.048, the board uarch and the HBM memory model (OTPU_TARGET=f2-hbm, OTPU_NCH=8:
otpu_native_dram on 8 channels, otpu_native_mem, --lat 60, no stalls), as "Baseline on the
RTL" below. tools/perf_qwen.py `--model gemma3 --layers 4 --wformat int8 --head-format int8
--lat 60 --mhz 200 --portb N` (4 layers and the whole 262K-row head; the whole model is
26 layers: the layers' phases scale by 26/4, the head is whole -- checked by the 26-layer run at
PORTB 2, `--layers 0`, 16 minutes of simulation, which landed on the extrapolation):

| Gemma-3-1B int8, 4 layers + head | PORTB 1 | PORTB 2 | PORTB 4 |
|---|---|---|---|
| cycles | 3,228,204 | 1,894,313 | 1,514,346 |
| speedup | 1.00x | 1.70x | 2.13x |
| roofline (port B / PORTB, port A, MXU) | 3,207,980 (port B) | 1,761,872 (MXU) | 1,064,016 (MXU) |
| efficiency to it | 99.4% | 93.0% | 70.3% |
| port B's utilization (chunks / (PORTB x cycles)) | 99.4% | 84.7% | 53.0% |
| MXU blocked (chunks but no MAC) | 0.2% | 5.7% | 27.5% |
| wide reads, two-part | - | 1,437,184, 105,337 (7.3%) | 739,328, 419,565 (56.8%) |
| DRAM read beats a cycle a channel | 1.007 | 1.716 | 2.146 |
| head / MLP / attention cycles | 2,368,742 / 753,084 / 106,376 | 1,392,823 / 436,534 / 64,954 | 1,102,284 / 360,907 / 51,153 |
| **whole model (26 layers)** | **7.96 M, 25.1 tok/s** (extrapolated; the 26-layer run hit rtlsim's 3600 s simulator timeout on the loaded machine) | **4,652,571 measured, 43.0 tok/s** (the extrapolation said 4.65 M; 91.2% of the MXU roofline 4,243,208, port B at 84.1%, 300,202 of 3,574,528 wide reads two-part) | **3.78 M, 52.9 tok/s** (extrapolated) |

PORTB 2 lands on the projection (38 tok/s at the board's 78% port efficiency, 50 at the
ceiling: the RTL's 43 is at 85% of its port). PORTB 4 does not reach 70: Gemma's rows are 9
chunks (K = 1152), so a QUAD row is 3 advances of 4, 4, 1 chunks (the MXU's roofline is 75% of
the port's), and 57% of the wide reads cross a group of 4 pairs and go in two parts (each a
port-B cycle): 1.06 M roofline + 0.42 M two-part cycles = 1.48 M of the 1.51 M. The one-row
int8 MM with KB 16 (test_wide_port_rtl.py's throughput test) streams 4 chunks a cycle on the
behavioural DRAM, 3.1 on the native path with the model's 20% stalls.

**Bonsai-2 2B (docs/ternary.md)**, tools/perf_bonsai2.py `MODEL f2-hbm --layers 1 --portb N`
(1 layer and the head; the same memory model and latency), in t2 QUAD with the rotation on the
device (`--wformat t2 --rotate`) and in the unfolded int4 of the board baseline (`--wformat
int4 --head-format int8`):

| Bonsai-2 2B, 1 layer + head | PORTB 1 | PORTB 2 | PORTB 4 |
|---|---|---|---|
| t2 QUAD: cycles | 1,171,020 | 1,170,808 | 1,170,715 |
| t2 QUAD: roofline (port B chunks 1,144,993 / PORTB, port A, MXU) and efficiency | 1,144,993 (port B), 97.8% | 1,122,560 (MXU), 95.9% | 1,122,432 (MXU), 95.9% |
| t2 QUAD: port B's utilization | 97.8% | 48.9% | 24.4% |
| t2 QUAD: wide reads (two-part) | - | 256 (0) | 128 (51) |
| int4, int8 head: cycles (speedup) | 4,234,618 | 2,203,107 (1.92x) | - |
| int4, int8 head: roofline and efficiency | 4,229,089 (port B), 99.9% | 2,114,544 (port B / 2), 96.0% | |
| int4, int8 head: port B's utilization; wide reads (two-part) | 99.9% | 96.0%; 2,091,136 (1,092: 0.05%) | |

**Bonsai-2 at MCOLS 8 (2026-10-17, the merged tree with the row lanes; docs/ternary.md 8.1)**,
the same tool with `--mcols 8 [--rowlanes]`, 1 layer + head, then the whole model:

| Bonsai-2, t2 QUAD with the rotation | MCOLS 4 / PORTB 1 | MCOLS 8 / PORTB 2 (QUAD) | MCOLS 8 / PORTB 2 / RL 2 |
|---|---|---|---|
| 2B, 1 layer + head: cycles (speedup) | 1,171,020 | 1,170,384 (1.00x) | 786,339 (1.49x) |
| 2B: roofline and efficiency | 1,144,993 (port B), 97.8% | 1,119,488 (MXU), 95.7% | 705,024 (port A lines), 89.7% |
| 2B: port B's utilization; wide reads (two-part) | 97.8% | 48.9%; 256 (0) | 72.8%; 554,240 (0) |
| 2B, whole model (24 layers): cycles, tok/s at 200 MHz | 4,766,291, 42.0 (measured) | (still running at hand-back: /tmp/wp-merge/rtl_chain3.log, rtl_chain4.log, rtl_whole.jsonl) | 3,698,534 (1.29x), 54.1 (measured; the head 1.54x, a layer 1.23x: its VPU work no longer hides behind the MMs) |
| 4B, 1 layer + head: cycles (speedup) | 1,554,988 | 1,549,005 (1.00x) | 1,005,571 (1.55x) |
| 4B: roofline and efficiency | 1,517,376 (port B), 97.6% | 1,477,248 (MXU), 95.4% | 892,800 (port A lines), 88.8% |
| 4B: port B's utilization; wide reads (two-part) | 97.6% | 49.0%; 640 (0) | 75.4%; 731,520 (0) |
| 4B, whole model (32 layers): cycles, tok/s | (still running at hand-back: /tmp/wp-merge/rtl_chain3.log, rtl_chain4.log, rtl_whole.jsonl) | (the baseline's) | (still running at hand-back: /tmp/wp-merge/rtl_chain3.log, rtl_chain4.log, rtl_whole.jsonl) |
| ISA unit cycles a token, 2B / 4B / 9B | 5.44 / 12.16 / 20.47 M | 5.37 / 11.83 / 20.04 M | 3.53 / 7.72 / 12.29 M |

MCOLS 8 alone leaves t2 where it was (a one-row QUAD fills four of the eight columns from its
one chunk a cycle); the row lanes (t2 L 2: two rows' chunks a cycle into eight columns) give
1.5-1.6x, bound by port A's scale lines (an RL 2 t2 advance reads 1.25 lines for two chunks),
not by port B: the target for it is `f2-hbm:p2rl8` (CFG_HASH 4317000c), `f2-hbm:p2m8`
(8a2b063d) the M = 2 build for MTP on t2. Section 8.1 of docs/ternary.md has the analysis.

t2 QUAD is the same at every PORTB, as its compile says: the step is 98 t2 QUAD MMs
(1,105,408 chunks; a t2 chunk is four blocks, QUAD takes them in one cycle, so a wider port has
nothing more to stream for it) and one int8 MM of 512 chunks that becomes PAIR (the 256 wide
reads). The 2B at 5.44 M unit cycles a token in t2 stays 5.44 M: for t2 the port is not the
bound, the MXU's four columns are (MCOLS 8 would be t2's doubling). The unfolded int4 decode
(the board baseline's format, 9.69 M chunks a token) is where PORTB 2 pays for Bonsai: 72 of
its 73 MMs go PAIR -> QUAD (2,193,664 of 2,195,712 chunks at two a cycle), and the RTL gives
1.92x at 96% of the doubled port: Bonsai's int4 rows (K = 2048: 8 chunks) align to the pair
groups, so only 0.05% of the wide reads split (Gemma's 9-chunk rows: 7.3%). The whole 2B in
int4 + int8 head (12.36 M cycles a token measured on the board at 78%) would be ~6.4 M on the
RTL at PORTB 2, near its t2 QUAD figure (5.44 M) with twice the bytes.

### The whole-model tally on the ISA, and what > 100 tok/s needs (PORTB 8, int4)

tools/tally_decode.py (the step run on the ISA simulator over a zeroed image, every executed
instruction at opentpu.profile's nominal cost; `bound` = max(port B at PORTB a cycle, the
MXU's advances) + the other units' cycles, which the RTL mostly overlaps: 8.25 M here against
7.96 M measured at PORTB 1). Gemma-3-1B, 26 layers, 200 MHz:

| weights | PORTB | bytes a token | chunks | port-B cycles | MXU advances | bound | tok/s |
|---|---|---|---|---|---|---|---|
| int8 (890 MMs) | 1 | 1,033,510,176 | 7,830,202 | 7,830,202 | 7,817,736 | 8,247,141 | 24.3 |
| int8 (838 PAIR) | 2 | " | " | 3,915,101 | 4,243,208 | 4,660,147 | 42.9 |
| int8 (838 QUAD) | 4 | " | " | 1,957,551 | 2,590,728 | 3,007,667 | 66.5 |
| int8 | 8 | " | " | 978,776 | 2,590,728 (QUAD: the ISA's widest) | 3,007,667 | 66.5 |
| int4 (838 PAIR) | 1 | 575,970,592 | 4,255,674 | 4,255,674 | 4,243,208 | 4,672,613 | 42.8 |
| int4 (838 QUAD) | 2 | " | " | 2,127,837 | 2,590,728 | 3,007,667 | 66.5 |
| int4 (QUAD) | 4 | " | " | 1,063,919 | 2,590,728 | 3,007,667 | 66.5 |

What the table says:

* **PORTB 8 and int4 at PORTB 4 buy nothing with the ISA's reuse.** Column reuse is x1, x2
  (PAIR), x4 (QUAD): an MM takes at most 4 blocks a cycle -- 4 int8 chunks, 2 int4 chunks. A
  port of 8 chunks has no instruction that asks for 8, and int4 QUAD at PORTB 4 is int4 QUAD
  at PORTB 2 with the port half idle. Both need an 8-way reuse (and MCOLS 8 at M = 1), a new
  ISA mode: QACT DUP8, the 8-block ACT RAM read, 8 scale decoders, a 4:1 PE input select, the
  8-column row sum, a FIFO entry of 8 chunks. Not built here; the RTL is parameterized to
  PORTB 4 (the ISA's widest need), on NCH 8 (f2-hbm) or 16 (f2-hbm16:p4).
* **Even 8-way block reuse (OCT) stops short of 100.** Gemma's rows are 9 blocks: OCT takes
  2 advances a row (8 + 1), 1,629,444 advances a token, 97.7 tok/s with the other units
  serial, ~115 with them overlapped as the RTL does -- at 56% of the port's rate (PORTB 8:
  all 16 channels of f2-hbm16 at a beat a cycle each, the native port's limit).
* **Reuse across rows (lanes) removes the rounding.** With PORTB lanes each streaming its own
  weight row (column group c takes row n + c: the SGMV engine's structure, "Future
  improvements"), the MXU's cycles are N KB / PORTB with no rounding of KB: int8 at PORTB 4
  1,954,460 (84 tok/s with the other units serial, ~96 overlapped: the int8 ceiling of a
  25.6 GB/s x 4 port), int4 at PORTB 4 1,060,828 (135 / ~170 tok/s at 54 GB/s of HBM), int8
  at PORTB 8 977,256 (143 / ~180, but at the native port's limit on 16 channels, where the
  bridge's ~10 GB/s of reads in flight a channel caps it near 78%: ~140). **So > 100 tok/s
  is int4 weights with 4 row lanes (MCOLS 8: each lane's chunk is two blocks on two columns),
  or int8 with 8 lanes on f2-hbm16**; both are the lane structure, with a new MM mode (the
  output row from the lane, `n mod L`) rather than a wider block reuse. The int4 route also
  halves the bytes a token (576 MB), which is what the board's 78% port efficiency needs.
* The other units' 417K nominal cycles (VPU, quantizer, DMA; ~130K net on the RTL) are 7% of
  a PORTB 4 token and 13% of a lanes-int4 one: they start to matter there (docs/vpu_pump.md,
  docs/seq_vpu.md).

### Row lanes (MM RL, docs/row_lanes.md), measured on the RTL (2026-10-16)

The same method and model as the table above (Verilator 5.048, OTPU_TARGET=f2-hbm OTPU_NCH=8,
--lat 60, 200 MHz, tools/perf_qwen.py `--model gemma3 --layers 4 --portb N --rowlanes
[--mcols 8]`: 4 layers and the whole 262K-row head; the whole model is 26 layers), with
`Config.ROWLANES`: the image in row-lane groups and every projection and head MM an MM RL of
the widest L (int8 L 4 at PORTB 4, L 2 at PORTB 2; int4 L 4 at PORTB 4 on MCOLS 8, L 2 at
PORTB 2); the attention's K / V MMs stay PAIR / QUAD. The RL builds are `RL` 4 at MCOLS 4 and
8 at MCOLS 8 (otpu_mxu's streams).

| Gemma-3-1B, 4 layers + head | int8 PORTB 4 QUAD (above) | int8 PORTB 4, RL 4 (MCOLS 4) | int8 PORTB 4, RL 4 (MCOLS 8) | int8 PORTB 2, RL 2 | int4 PORTB 4 QUAD (MCOLS 8) | int4 PORTB 4, RL 4 (MCOLS 8) | int4 PORTB 2, RL 2 |
|---|---|---|---|---|---|---|---|
| cycles | 1,514,346 | 949,485 | 949,474 | 1,637,409 | 1,610,911 | 579,306 | 922,723 |
| roofline (port B / PORTB, port A lines, MXU) | 1,064,016 (MXU) | 801,995 (port B; port A 801,632, MXU 800,592) | 801,995 | 1,603,990 (port B) | 1,063,976 (MXU) | 534,712 (port A; MXU 441,256, port B 442,699) | 885,398 (port B) |
| efficiency to it | 70.3% | 84.5% | 84.5% | 98.0% | 66.0% | 92.3% | 96.0% |
| port B's utilization (chunks / (PORTB x cycles)) | 53.0% | 84.5% | 84.5% | 98.0% | 27.5% | 76.4% | 95.9% |
| MXU: MAC / blocked (chunks, no MAC) / starved | 70% / 27.5% / 0.2% | 84.3% / 12.8% / 0.3% | 84.3% / 12.9% / 0.2% | 97.7% / 0.9% / 0.0% | 66.0% / 31.3% / 1.1% | 76.2% / 16.9% / 2.4% | 95.5% / 1.6% / 0.2% |
| wide reads, two-part | 739,328, 419,565 (56.8%) | 799,488, **0** | 799,528, 0 | 1,598,976, 0 | 697,896, 47,136 (6.8%) | 440,232, **0** | 880,384, 0 |
| DRAM read beats a cycle a channel | 2.146 | 3.428 | 3.428 | 1.988 | 1.184 | 3.293 | 2.068 |
| head / MLP / attention cycles | 1,102,284 / 360,907 / 51,153 | 686,818 / 222,479 / 40,186 | 686,825 / 222,470 / 40,177 | 1,196,334 / 380,182 / 60,891 | 1,188,686 / 374,504 / 47,719 | 416,615 / 132,244 / 30,445 | 670,921 / 210,382 / 41,418 |
| tok/s at 200 MHz (4 layers + head) | 132.1 | 210.6 | 210.6 | 122.1 | 124.2 | 345.2 | 216.7 |
| **whole model (26 layers): cycles, tok/s** | 3.78 M, 52.9 (extrapolated) | **2,393,990 measured, 83.5** (the extrapolation 2,394,141; 699 s of simulation; 82.1% of the roofline 1,966,576 (port A lines), MAC 81.9%, blocked 11.6%, 0 of 1,952,640 wide reads two-part) | " | 4,063,309, 49.2 | 3,933,136, 50.9 | **1,474,025 measured, 135.7** (the extrapolation 1,474,094; 288 s; 89.1% of the roofline 1,312,940 (port A lines; the MXU 1,065,924), MAC 72.3%, blocked 14.6%, starved 2.5%, 0 of 1,059,268 wide reads two-part) | 2,307,621, 86.7 |

Against the tally (docs/row_lanes.md): int8 RL 4 at PORTB 4 was 1,959,816 MXU advances, 84.1
tok/s with the other units serial; the RTL's whole model is 2,393,990 cycles measured, 83.5 tok/s: the
head runs at 85% of the port (the MXU and port A at one advance and one line a cycle, each
hiccup of port A's line stream -- 12.8% blocked -- a cycle of the array), the layers at 90%+,
and the other units overlap. int4 RL 4 on MCOLS 8 was 1,065,924 advances but 1,299,524 port-A
lines (the group's Sw line is a request of its own: H = A = 32), 116.5 serial / ~150
overlapped; the RTL gives 1,474,025 cycles measured, 135.7 tok/s, port-A bound as predicted (the roofline
534,712 of the 4-layer run is port A's, the MXU 441,256). The two-part reads are gone in every
RL run (0 of 799,488 wide reads at PORTB 4 where QUAD had 56.8%): an advance's L chunks are one
aligned run of L pairs. int8 RL 4 is the same on MCOLS 4 and 8 (11 cycles apart): int8 L 4
needs four columns; int4 L 4 needs eight.

**So > 100 tok/s is reached on the RTL in int4 with four lanes at PORTB 4 on MCOLS 8:
135.7 tok/s (the int4 image, 0.58 GB a token). int8 with four lanes stops at the port's 102
GB/s: 83.5 tok/s (1.03 GB a token at 85% of the port).** On the card (78% of the port on
Alex's image, docs/wide_port.md "Baseline on the board") the int4 point would be ~105-120: the
int4 route halves the bytes, which is what the card's port efficiency needs.

**The regression suites with the mode** (2026-10-17, Verilator 5.048; the mode on under
`OTPU_ROWLANES=1 OTPU_PORTB=2 OTPU_PAIR=1`, the native path on `OTPU_NCH=4`; the machine
shared, load ~24):
| suite | mode off | mode on |
|---|---|---|
| tests/test_wide_port_rtl.py | 53 + 43 row-lane tests passed (its own matrix) | " |
| tests/test_rtl.py (the board path) | 176 passed, 1 skipped (41 min) | 176 passed, 1 skipped |
| tests/test_t2_rtl.py | 24 passed | 24 passed |
| tests/test_observability.py (tb_board, SimTransport) | 18 passed | 18 passed (CAPS2 bit4, CFG_HASH of Config(PORTB=2, ROWLANES=True)) |
| tests/test_row_lanes.py (ISA), test_wide_port, test_pair, test_target, test_compiler, test_t2, test_mx, test_f2_build (the p4rl / p4rl8 / p2rl snapshots) | 38 + 186 + 105 passed | - |
Cycles at PORTB 1 and at PORTB 2 without the mode are unchanged: every RL term is a constant
or a generate branch not taken (the PORTB 1 build's RL is 0), and the exact-cycle tests
(test_wide_port_rtl's throughput, test_rtl's) pass as before.

**Bonsai-2 2B**, the unfolded int4 path of docs/ternary.md 5 (9.69 M chunks a token: int4 5.70
M, the int8 head 3.99 M; K = 2048, KB 16): at PORTB 2 row lanes buy nothing (int4 L 2 = N / 2 x
8 advances = QUAD's N x 4; the rows align to the pair groups already); at PORTB 4 on MCOLS 8
int4 L 4 takes the int4 chunks in 1.43 M advances (1.7 M port-A lines) and int8 L 4 the head in
1.0 M, ~2.7 M MXU cycles a token against 12.36 M measured on the board today (the ISA tally;
not run on the RTL).

### Port A at 64 bytes a line, and prefill over the row-lane image (2026-10-17)

docs/row_lanes.md "Port A at 64 bytes" (the decision and its arithmetic) and "Prefill over the
group layout" (the group walk). The same method as the row-lane table above (Verilator 5.048,
OTPU_TARGET=f2-hbm OTPU_NCH=8, --lat 60, 200 MHz; tools/perf_qwen.py `--model gemma3 --portb 4
--mcols 8 --rowlanes --aline 64 [--wformat int4 --head-format int4] --layers 4 | 0`), the
builds RL 8 / ALINE 64 (`f2-hbm:p4rl8a64`, CFG_HASH dd6b92a6):

| Gemma-3-1B, RL 4 at PORTB 4 / MCOLS 8 | int4, ALINE 32 (above) | **int4, ALINE 64** | int8, ALINE 32 (above) | **int8, ALINE 64** |
|---|---|---|---|---|
| 4 layers + head: cycles | 579,306 | **565,739** | 949,474 | 949,474 |
| roofline (port B / 4, port A lines, MXU) | 534,712 (port A; MXU 441,256, port B 442,699) | 442,699 (port B; port A 442,296, MXU 441,256) | 801,995 (port B; port A 801,592) | 801,995 (port B; port A 801,592) |
| efficiency to it | 92.3% | 78.3% | 84.5% | 84.5% |
| port A: lines, Sw alone | 528,278, 88,046 (16.7%: 1.2 lines an advance) | **440,232, 0 (one line an advance)** | 799,528, 0 | 799,528, 0 |
| port B's utilization | 76.4% | 77.8% | 84.5% | 84.5% |
| MXU: MAC / blocked / starved | 76.2% / 16.9% / 2.4% | 78.0% / 15.2% / 2.2% | 84.3% / 12.9% / 0.2% | 84.3% / 12.9% / 0.2% |
| DRAM read beats a cycle a channel | 3.293 | 3.373 | 3.428 | 3.428 |
| 4 layers + head: tok/s | 345.2 | 353.5 | 210.6 | 210.6 |
| **whole model (26 layers): cycles, tok/s** | 1,474,025, 135.7 | **1,440,727, 138.8** (352 s; the roofline now 1,072,684 = port B's and port A's alike, the MXU 1,065,924: 74.5%; MAC 74.0%, blocked 12.6%, starved 2.6%; 1,059,268 lines, 0 alone; 3.230 beats) | 2,393,990, 83.5 | 2,394,111, 83.5 (681 s; 82.1%, MAC 81.8%, blocked 11.6%) |

What it says:

* **int4 RL 4 is off the port-A bound: one line an advance, 0 Sw-alone requests** (the group's
  Sw rides its first advance's 64-byte line; 440,232 lines where 528,278 were needed at 32),
  and the whole model gains 33 K cycles, 135.7 -> **138.8 tok/s**. Not the ~170 of the MXU's
  roofline: with port A gone the next bound is port B itself, which int4 L 4 at four chunks an
  advance runs at every cycle (1,063,919 port-B cycles against the MXU's 1,065,924: the two
  rooflines coincide), and the native path delivers 3.23-3.43 beats a cycle a channel of 4 on
  this model (int8 RL 4, port-B bound since the row-lane commit, reaches 84.5% the same way),
  plus the other units' ~130 K net cycles (9% of an int4 token, 5% of an int8 one). So int4 at
  ALINE 64 is where int8 was: the port's rate on 8 channels, and the VPU / quantizer / DMA
  work the MXU does not overlap. The MXU's own roofline needs port B past 85% (f2-hbm16's 16
  channels at half a beat each, docs/f2_hbm.md) and the other units overlapped (docs/vpu_pump.md).
* **int8 is unchanged to the cycle** (949,474 at 4 layers; the whole model 2,394,111 against
  2,393,990, the two runs' noise): int8 L 4's Sw shared its first advance's line at 32 already
  (H 16 + A 4), as the arithmetic said.
* The 32-byte builds and images are untouched (`ALINE` defaults to 32; `f2-hbm:p4rl8` keeps
  172e5c73; the PORTB 1 build's netlist has no ALINE term).

**Prefill over the row-lane image** (the group walk, docs/row_lanes.md "Prefill over the
group layout"; the same image as decode; `--rows 4` at 4 layers + the head, logits for the
last row; the row-lane image at ALINE 64, the adapter keeping 16 beats for port-A reuse on
RL builds):

| Gemma-3-1B prefill, 4 rows, PORTB 4 / MCOLS 8 | int4, standard image | **int4, row-lane image** | int8, standard image | **int8, row-lane image** |
|---|---|---|---|---|
| cycles (a row) | 1,761,842 (440,460) | **1,334,346 (333,586)** | 1,675,893 (418,973) | **1,614,575 (403,644)** |
| roofline (MXU; port B / 4; port A) | 1,240,760; 443,581; 1,017,592 | 1,174,776 (port A = MXU 1,170,616); 443,581 | 1,240,760; 802,877; 1,017,592 | 1,436,920 (port A; MXU 1,432,760); 802,877 |
| efficiency | 70.4% | 88.0% | 74.0% | 89.0% |
| MXU: MAC / blocked / starved | 70.4% / 25.1% / 0.8% | 87.7% / 7.0% / 0.7% | 74.0% / 21.6% / 0.5% | 88.7% / 7.4% / 0.1% |
| wide reads, two-part | 524,472, 34,952 | 327,864, **0** | 913,080, 332,527 | 590,008, **0** |
| port A lines, Sw alone | 1,629,880, 0 | 1,166,520, 0 | 1,236,664, 0 | 1,428,664, 0 |

No regression: the row-lane image's prefill is 24% faster in int4 and 3.7% in int8. The
standard image's 4-row MMs take PAIR (M 4 = MCOLS / 2) and QUAD, whose wide reads go in two
parts (34,952 int4, 332,527 int8: a pair twice) and whose scale windows wait on port A (the
MXU blocked 21-25%); the walk reads one aligned chunk a request from the groups (0 two-part)
and, with the adapter keeping a group's beats, one line a block with every row after a
group's first a reuse (blocked 7%). (The walk has no PAIR / QUAD, so its MXU roofline is the
plain MM's -- int8: 1,432,760 against the standard image's 1,240,760 -- and it still finishes
first.) Before the adapter's beat reuse the int4 walk measured 4,753,621 cycles (2.7x the
standard image, blocked 74%): a 4-bit row's scale data spans three beats of its group and
every row restarted the adapter's run at the group's first beat; int8's rows sit in one beat
and were 3.7% faster already.

### LPN estimate of the decode token (tools/wideport_lpn.py; estimated, calibrated per phase on the Verilator receipts, 2026-10-18)

The decode token as a Latency Petri Net (MJChku/LPNDialect's compiled nets: the hardware
loop's LPN tier on the hwloop branch, tools/hwloop/lpn), so that a configuration's tok/s comes
in seconds instead of the minutes to hours of Verilator, with the error stated per phase. The
model is the tier's **wide model** (hwloop's `tools/hwloop/lpn/wide.py`: upstream's
CompactOpenTPU with port B as a credit of PORTB chunks a cycle plus the two-part reads counted
from the MM's addresses, a row-lane group of L rows as one row of the model, the NCH channels
each a credit held for its beats at RD / round trip = 64 / 74 = 0.865 beats a cycle; graded +1
to +4% on the 96a90c2f export) imported from that worktree and extended in
`tools/wideport_lpn.py` (the hwloop branch untouched): **port A at ALINE bytes** (a group's Sw
line a request of its own only where `quant.rl_sw_alone` says so; a QUAD advance's two-line
crossings at the line's boundaries), **t2 weights** (a chunk of four blocks, QUAD one chunk an
advance with its scale words in one 16-byte group, RL 2 two chunks an advance: docs/ternary.md
8.2), **DSTEP / STREAM** on the DMA unit (profile's lanes and chunks, the chunks on port B, a
read's round trip first; upstream's frontend refuses them, so the command trace is recorded by
the tool with the same rule), the units as the RTL has them (the board's MXU_E 4 epilogue
lanes, WIN 16, FIFO_DEPTH 1024 from targets/f2-hbm.toml, otpu_board's NRD 64 reads in flight a
channel, the round trip --lat + the adapter's 14 cycles of path) and **the phases**: one
transition walks the runs of same-phase commands (perf_qwen's `_phase` by kernel source) and
fires when every command of a run is complete, so the native run's `--trace` gives each run's
end and the phase cycles add up as `opentpu.profile.Profile.phases` charges them (the running
max of the ends in dispatch order). **Nothing is fitted**: every constant is the RTL's or
upstream's validated model's. The one sensitivity is the adapter's path (`--adapter-latency`,
derived from its structure, not measured): on the int8 point 10 / 12 / 14 / 16 cycles give
-1.1% / +1.5% / +4.1% / +6.7% of the token (mean |phase error| 2.3 / 2.2 / 2.9 / 4.4%); 14
stays.

The receipts (Verilator 5.048, this tree, `OTPU_TARGET=f2-hbm OTPU_NCH=8`, `--lat 60`, 200
MHz; `--json` for the phases): int8 `tools/perf_qwen.py --model gemma3 --layers 4 --portb 4
--mcols 8 --rowlanes --aline 64` (949,474 cycles, 265 s of simulation; identical to the table
above). The LPN runs: `python -I tools/wideport_lpn.py --model gemma3 --layers 4 --wformat int8
--head-format int8 --portb 4 --mcols 8 --rowlanes --aline 64 --lat 60 --measured RECEIPT.jsonl`,
a few seconds each (frontend 1-4 s, model 0.3-1.5 s, compile 0.4-0.6 s, native 0.2-0.8 s).
The rule: no tok/s is reported from a point whose mean |error| over the phases carrying >= 2%
of the token is above 10%.

**Calibration, Gemma-3-1B int8, 4 layers + head, PORTB 4 / MCOLS 8 / RL 4 / ALINE 64**
(cycles; the RTL column measured, the LPN column estimated):

| phase | RTL (measured) | LPN (estimate) | error | what binds in the model (port B / MXU / port A / channel) |
|---|---|---|---|---|
| LM head (`other`) | 686,825 | 723,866 | +5.4% | channel (589,824 / 589,824 / 589,824 / 691,456) |
| MLP | 222,470 | 224,931 | +1.1% | channel (186,624 / 186,624 / 186,624 / 219,281) |
| attention | 40,177 | 39,344 | -2.1% | channel (24,104 / 24,104 / 23,080 / 27,363) |
| **token** | **949,474** | **988,141** | **+4.1%** | channel 938,100 (port B = MXU 800,552, port A 799,528) |

Mean |error| 2.9% over the three phases. The head is where the error is: RL 4's four chunks an
advance put a beat on every channel every cycle, the model's channels sustain 0.865 of that
(RD 64 beats in flight over a 74-cycle round trip) and the RTL's head runs at 0.859 of its
589,824 requests a cycle with 12.9% of the MXU's cycles blocked (the receipt), i.e. the two
agree on what binds and differ by the ~37 K cycles of the model's per-command tails and
the credits' granularity. int4 (`--wformat int4 --head-format int4`, ALINE 64) has no
per-phase receipt on this tree (the user's priority was int8; the 4-layer total 565,739 is the
table above): the LPN gives 599,271, **+5.9%**, channel-bound likewise.

**The whole model (26 layers + head, `--layers 0`; every LPN number estimated, 200 MHz):**

| Gemma-3-1B, PORTB 4 / MCOLS 8 / RL 4 / ALINE 64 | RTL (measured) | LPN (estimate) | error | the model's bound | tok/s RTL / LPN |
|---|---|---|---|---|---|
| int8 / int8 head | 2,394,111 | 2,441,652 | +2.0% | channel 2,294,625 (port B = MXU 1,959,556, port A 1,952,900): the port's channels at 0.865 beats a cycle, as the RTL's 82% of its port-B roofline says | **83.5 / 81.9** |
| int4 / int4 head | 1,440,727 | 1,476,251 | +2.5% | channel 1,321,067 (port B = MXU 1,065,924, port A 1,059,268) | **138.8 / 135.5** |

Per phase of the int8 token (LPN): MLP 1,462,050, head 723,866, attention 255,736; int4: MLP
847,642, head 439,818, attention 188,790. Both tokens are **port-bound on the channels** in the
model as on the RTL (docs/row_lanes.md "Port A at 64 bytes": port B at four chunks every cycle
is the same 1.07 M cycles as the MXU; what the model adds is that the native path's 64 reads in
flight a channel cap it at 0.865 of that), so the next lever the model agrees with is the
port's depth or f2-hbm16's channels, not the scale path. Bonsai-2's t2 RL 2 points are in
docs/ternary.md 8.2 (4B 27.1 estimated against 26.7 measured, 2B 56.5 against 54.1). Not
modelled (as the tier's README lists): the DMA's and QST's beats on the channels, QACT DUP8's
copies, the adapter's A runs, SEQ_WC's wait entries (perf_bonsai2's runs use that sequencer,
perf_qwen's the range scoreboard; the programs are the same), HALT CHAIN. The runs' JSON
(`--json`, and `result.json` under `--work`) carry every number with the label `lpn-estimate`.

### What the nightly needs (the f2-hbm target at PORTB 2)

Verilator cannot give LUTs or a clock: the lanes' cost in silicon and the 2048-bit path's
timing come from the nightly's synthesis. What it needs:

* **The target:** `f2-hbm:p2` (targets/f2-hbm.toml `[variants.p2]`, PORTB = 2 over the f2-hbm
  arch; `f2-hbm16:p4` is PORTB 4 on 16 channels). `python boards/f2/build.py --hdk ... --target
  f2-hbm:p2 --out RUN --flow SynthCL` (then `--flow BuildAll --package` as docs/f2_hbm.md
  "Build"): `target.rtl_params` carries `PORTB` into otpu_f2_hbm_sys -> otpu_board -> the slice
  and its adapter (build.py checks every parameter against the top's), the geometry stays
  S = 1, NCH = 8: a two-chunk request is two pairs' four beats, half a beat a channel a cycle
  (6.4 GB/s a channel at full rate, under the native port's 12.8 and the bridge's ~10 GB/s of
  reads in flight). The DDR target (f2, NCH 2) cannot take PORTB 2: otpu_native_dram needs
  NCH >= 2 PORTB. tests/test_f2_build.py checks the p2 / p4 snapshots (`.PORTB(32'h2)` in
  opentpu_params.vh, the hash).
* **The hash:** CFG_HASH a9d1d2b0 for f2-hbm:p2 (f2-hbm's 1f260779 stays: PORTB is in the hash
  only when wide). Programs are compiled for `Config(PORTB=2)`; the host takes PORTB from
  CAPS2 [3:1] (device_config), so no OTPU_PORTB is needed on the card, and a PORTB 1 image's
  CAPS2 reads as before.
* **To read off the synthesis report** against the estimates table above: the slice's MXU
  (RAMB36: the chunk FIFO at 2048 x 1024 b, +29; the scale entry's third RAM, +4; the QUAD
  datapath's ~7K LUT / ~10K FF where T2 did not already build it -- f2-hbm has T2, so that
  part is already paid), the adapter (the per-chunk address and fold, the NP:1 merge a chunk:
  ~+1.8K LUT), the slice's 2048-bit port-B input register (+1K FF); and the timing of the
  merge path (B FIFO heads -> NP:1 a chunk -> the slice's input register) and of the per-chunk
  selects' fanout (1024 LUTs each: replication expected), the S1 stream select and the scale
  window in the MXU (one LUT level each). If the merge path misses 200 MHz, a register per chunk
  at the merge (b_rvalid a cycle later: latency only, the MXU's FIFO absorbs it) is the fix.
* **The row-lane builds:** `f2-hbm:p4rl` (PORTB 4, ROWLANES, MCOLS 4: int8 L 4; CFG_HASH
  d6ed551d), `f2-hbm:p4rl8` (MCOLS 8: int4 L 4, the > 100 tok/s point; 172e5c73), `f2-hbm:p2rl`
  (e476f0b7), `f2-hbm:p2rl8` (PORTB 2, MCOLS 8: Bonsai-2's t2 L 2, docs/ternary.md 8.1 and 10;
  4317000c) and, without lanes, `f2-hbm:p2m8` (PORTB 2, MCOLS 8: t2 QUAD at M = 2 for MTP on
  t2; 8a2b063d). `rtl_params` carries `RL` (4 or 8, `target.rl_streams`) into otpu_f2_hbm_sys ->
  otpu_board -> the slice (the MXU's streams, the quantizer's and ACT RAM's DUP8) and otpu_ctrl
  (CAPS2 bit4); the host takes ROWLANES from CAPS2 bit4 (device_config) and OTPU_ROWLANES must
  agree. Programs and images are compiled for `Config(PORTB=4, ROWLANES=True[, MCOLS=8])`. To
  read off the synthesis report against docs/row_lanes.md's cost table: at RL 8 the hops (64
  bits a position), the PE select (4:1), the four more scale decoders and the two more entry
  RAMs (11 RAMB36), the result FIFO's Sw per column; at RL 4 the entry RAMs and the selects
  only. PORTB 4 on 8 channels runs every channel at a beat a cycle (3.43 read beats a cycle over
  8 channels in the int8 run): f2-hbm16 (NCH 16) halves that; the bridge's reads in flight
  (docs/f2_hbm.md) are the card's limit to check first.
* **The 64-byte port-A line:** `f2-hbm:p4rl8a64` (PORTB 4, ROWLANES, MCOLS 8, ALINE 64; CFG_HASH
  dd6b92a6; CAPS2 bit5): `rtl_params` carries `ALINE` (32 on every other target) into the F2
  tops -> otpu_board -> the slice's MXU (the 512-bit line register and tag RAM, the window
  selects), otpu_dram / otpu_native_dram (the beat returned whole) and otpu_ctrl; the adapter's
  `APR` (16 kept port-A beats on RL builds) comes with RL. Programs and images for
  `Config(PORTB=4, ROWLANES=True, MCOLS=8, ALINE=64)` (OTPU_ALINE=64 must agree with the
  bitstream). To read off the synthesis: the MXU's line path (ln1, r_stl, the tags) at twice
  the width, the adapter's 16 x 512-bit reuse RAM and comparators; the measured gain is the
  table above (int4 RL 4 decode 135.7 -> 138.8 tok/s, prefill on the one image).
* **On the board:** tools/perf_qwen.py's run with `--portb 2` is the expected cycles a token
  (table above); otpu-serve on the image with Gemma-3-1B int8 gives the tok/s to set against
  the RTL's and against the PORTB 1 board baseline (22.3 tok/s, ~89% of its RTL figure).

### Synthesis (f2-hbm:p4rl8, the first run, 2026-10-17)

The first synthesis of `f2-hbm:p4rl8` (PORTB 4, MCOLS 8, RL 8, NCH 8, clk_raw 200 MHz = 5.0 ns)
on the VU47P, post-synthesis:

* **Utilization**, the whole CL: 251,520 LUT (19%), 196,289 FF (7.5%), 426 BRAM (21%), 1,275
  DSP (14%).
* **Timing:** every path family positive (the HBM AXI, the shell clocks, hold) but one, 32
  endpoints: `u_sys/u_board/u_slice/u_mxu/mx_have_reg[0]/C` -> `.../u_mxu/mkr_reg[31:0]/D`,
  slack **-2.427 ns** (data path 7.250 ns = logic 2.249, 31%, + route 5.001, 69%; 33 logic
  levels: CARRY8 x14, LUT3 x4, LUT4 x2, LUT5 x9, LUT6 x4).

What the path was: `mkr` is the RL RMAX word (an MM RL writes one RMAX over the group's L rows,
where the plain MM writes one per column), and it was a reduce over the MCOLS columns'
running maxima in one cycle -- `for j: if (mx_have[j] && (!have || mk[j] > m)) m = mk[j]` --
MCOLS - 1 chained 32-bit key compares, each compare's result selecting the next compare's
operand, with the `have` flag starting the chain at `mx_have[0]`. At MCOLS 4 that is three
chained compares; MCOLS 8 (the first RL 8 synthesis: p2rl8 and p4rl8 are the MCOLS 8 builds)
doubles it to seven, 14 CARRY8 deep with the selects between them. Nothing in it is PORTB's
or RL 8's: it is the MCOLS 8 term of an MM RL's one-word RMAX.

What changed (otpu_mxu `mkr`): the word is a running max over the candidates the drain's lanes
offer the columns (max is associative: the max over the columns' maxima that saw a value is
the max over everything they were offered), in two register stages -- the cycle's candidates'
max over the NL = 4 lanes (a lane without a hit as key 0, the unsigned minimum: two compare
levels) and `mkr <= max(mkr, that)` (one level), cleared with `mx_have`. Its value at the
RMAX write (`mx_go`) is the same word on the same cycle (an assertion in the RTL checks it
against the columns' reduce at every RL RMAX write), so cycles are unchanged on every build:
the longest path is now two 32-bit compares (8 CARRY8) and a select. Its sibling in the RL
request path, the group walk's row-end step (`so = i_soff + w_ds` selected and added to
`srow_next`, then the `same_line` compares that decide the next row's Sw request: two adds and
a compare in the `go_iss` cycle), keeps the sum as a register (`w_sod`, advanced with
`i_soff`; asserted equal), one add less on that path. The other RL-widened arithmetic in the
issuer is one add deep (the command's first cycle's `so` -> `srow_addr + so` -> `same_line`,
the group walk's `w_rd` / `w_sd` three-operand sums into registers) and was left. Measured
after the change (the method of the tables above): Gemma-3-1B 4 layers + head at PORTB 4 /
MCOLS 8 / RL 4, int4 579,306 and int8 949,474 cycles, the baselines to the cycle;
tests/test_wide_port_rtl.py 168 passed, test_rtl's replay / act_reads / attention cases 25 (the
PORTB 1 exact cycles), test_t2_rtl 24, test_observability at OTPU_ROWLANES=1 OTPU_PORTB=4
OTPU_MCOLS=8 16 of 18 (the two that fail, test_mxu_starve_counter's cyc[20] < 0.3 cyc[400] at
13,137 / 41,089 and test_trace_long_wait_is_exact's 52,189-cycle wait under its 65,535, fail
identically on the commit before: their thresholds are PORTB 1 / 2's). The re-synthesis of
this fix is the nightly's to report.

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

**What the build flow must add** (the proposal as written after round 1; done in the section
after this one: the pblocks are cl_pnr_user_hbm.tcl with the board in SLR2, not
SLR1, and the crossings are register chains, not the bridges' FIFOs): (1) the placement directive: boards/f2/build.py (the `cmd = ['vivado', ...,
'-tclargs', 'SSI_SpreadLogic_high', 'AggressiveExplore', 'AggressiveExplore', 'A1', 'B2',
'C0', 'H2']` line) hands the HDK's build_all.tcl `SSI_SpreadLogic_high` as place_design's
directive, which spreads one slice of 250K LUTs (19% of the device) over the three SLRs on
purpose; change it to `Default` (or `SSI_BalanceSLLs` once the pblocks below exist), keeping
the phys_opt / route directives and the clock recipes. (2) Pblocks, in
boards/f2/build/constraints/cl_pnr_user.tcl (read unmanaged at placement, see round 3;
the hierarchy is WRAPPER/CL/u_sys/...): the HBM fabric in the SLR with the HBM's AXI ports
(the VU47P's stacks are below SLR0) and the slice with its adapter and control block in one
SLR next to it:

```
create_pblock pb_hbm_fabric
resize_pblock pb_hbm_fabric -add SLR0
add_cells_to_pblock pb_hbm_fabric [get_cells WRAPPER/CL/u_sys/u_mem]
create_pblock pb_slice
resize_pblock pb_slice -add SLR1
add_cells_to_pblock pb_slice [get_cells WRAPPER/CL/u_sys/u_board]
```

(u_sys/u_mem: the 8 bridges, the 8 host registers, the 8 arbiters, the host path; u_sys/u_board:
the slice, otpu_native_dram, otpu_ctrl, the trace. Check against the base build's
`report_utilization -slr` which SLR the small shell's PCIS and OCL logic occupies and how
much of SLR0 the shell and the HBM wrapper take; if the shell is in SLR0, the fabric's pblock
can be SLR0's lower clock regions rather than the whole SLR, and the slice SLR1; S = 2 puts
the second slice in SLR2.) With them the SLR crossing is the bridges' core sides (the adapter's
512-bit native buses, registered in the adapter's queue LUT RAMs and the bridges' FIFOs, 5 ns)
as docs/f2_hbm.md planned, and the host path's aclk side crosses from the shell to the fabric
through its crossing FIFOs (4 ns into aclk, 2.2 into hclk: an hclk register stage in the host
path, as the ports now have, if that direction still fails). (3) The crossings' bounds:
cl_timing_user_hbm.tcl now bounds each direction by the *destination* clock's period (it
bounded both by the smaller of the pair: the bridges' return FIFOs' data into the core at 2.2
ns across an SLR instead of 5). The base f2-hbm build at the same commit shows what of this
the shell and the host path already fail at PORTB 1.

**Status (2026-10-17, Verilator 5.048, bit-exact with the ISA simulator):**
tests/test_wide_port_rtl.py 168 passed (1:17 h, cold cache); tests/test_rtl.py -k "replay or
act_reads or attention or lane_count" 27 passed, 1 skipped; tests/test_t2_rtl.py 24 passed;
tests/test_observability.py 18 passed and tests/test_host.py 90 passed, 1 skipped (tb_board:
the suites that build otpu_board's rst_q; 3 host-side failures that build no RTL and are the
base's: a FakeTransport caps dict without the t2 key, FakeTransport without close, no
transformers in the venv); tests/test_f2_rtl.py 13 passed (tb_f2hbm on the fabric as
committed); tests/test_f2_build.py 82 passed. tools/perf_qwen.py Gemma-3-1B at PORTB 4 / MCOLS
8 / RL 4 / ALINE 64, 4 layers + head, --lat 60: int4 565,739 cycles (baseline 565,739, delta 0),
int8 949,474 (949,474, delta 0): tb_top's sources are untouched by these commits, so the
regression cycles and the NATIVEB counters are the base's by construction; the cycle costs are
in the F2 fabric only (a command one core cycle later in the bridge; the host's write beat per
~7 hclk cycles instead of ~4; the logic out of reset 1-3 cycles later).

### Placement, round 2, and the floorplan (2026-10-17)

The second placed build (fb42c14e: the round above plus the mkr fix, directive
SSI_SpreadLogic_low) placed at WNS -4.608 ns. SLR usage: SLR0 161,809 LUTs, SLR1 91,118, SLR2
21; SLR0 <-> SLR1 7,693 of 23,040 SLLs (5,508 up, 2,185 down), 2 of them registered (TX / RX_REG
2 of 32). By domain:

* **hclk (450 MHz)**: the worst 300 are still the host write path, now from `u_host/ws_reg`'s
  replicas (216), `hw_s_reg` (35), `rq_d_reg` (25) and `hw_d_reg` into
  `g_port[*].u_hreg/m_wdata_reg` (224) and `m_wstrb_reg` (39): one level, 6.338 ns of which
  6.127 route. The host block and the eight port registers were placed far apart; the half
  select `ws` fed every port's 288 register loads through the mux.
* **clk (200 MHz)**: `rl_pq_reg` (216 of the worst 300: otpu_slice's pending RLD read, into the
  TMEM request arrays and the bank arbiter, whose grant is the MXU's `t_gnt`) into
  `u_mxu/alpha_reg` (160), `t_raddr_reg` (31), `xo_reg` and the lanes' `m_reg`; and
  `u_board/u_mem/wh_r_reg` (70, the adapter's SW-queue bucket hash, whose readies reach the
  units' grants) into `u_quant` and `u_mxu/g_lane.u_l/u_m1/m_reg`: 4 levels, 8.455 ns of which
  8.147 route: the adapter, the arbiter and the units across the SLR split.
* **clk -> clk_f** (the pumped TMEM, multicycle 2 = 5.0 ns): -2.995 ns, 2,063 of 2,272
  endpoints, `u_vpu/ta_addr_reg` and `rl_pq_reg` into the pumped banks' ADDRARDADDR, 7-10
  levels, 90% route.
* **clk_main_a0 -> hclk** -1.279 (77: the host path's PCIS crossings).
* The base f2-hbm (PORTB 1, MCOLS 4, 96a90c2f, SpreadLogic_high) placed at -4.102 with the
  same host path and the bridge's `cf_h`: the shell fails on its own.

Module sizes (synthesis, fb42c14e; LUT / FF / RAMB36 / DSP): cl_opentpu 256,609 / 194,413 / 420
/ 1,275; `u_sys/u_board` 225,960 / 153,362 / 416 / 1,275 (`u_slice` 174,618 / 131,573 / 388 /
1,275: `u_mxu` 97,272 / 70,237 / 128 / 1,122, `u_vpu` 36,165 / 29,560 / 4 / 86, `u_tmem` 11,913 /
5,129 / 64, `u_quant` 9,383, `u_seq` 9,044, `u_dma` 9,251 / 3,955 / 14; the adapter
`u_board/u_mem` 46,717 / 14,508); `u_sys/u_mem` 29,514 / 39,848 (8 bridges at ~3,300 / 3,745
each, the host path, the port registers and arbiters). A VU47P clock region holds ~13.6K LUTs
and ~27K flip-flops; the SLRs are rows Y0-3, Y4-7, Y8-11; the HBM's ports are below SLR0.

**The floorplan** (boards/f2/build/constraints/cl_pnr_user_hbm.tcl, appended by build.py to
cl_pnr_user.tcl for an HBM target, which build_level_1_cl.tcl reads unmanaged before the HDK's
opt_design (round 3 below: the HDK's own read of the user XDC is managed and drops Tcl); every
pblock a child of the HDK's `pblock_CL`, whole clock regions, `-add` ranges as the HDK's, then
cut down to the sites the parent holds (`otpu_contain`), `CONTAIN_ROUTING` 0, `SNAPPING_MODE`
off, the cell queries failing closed; tests/test_f2_build.py checks it under a Tcl mock):

| pblock | hierarchy (WRAPPER/CL/u_sys/...) | clock regions | LUT budget | used |
|---|---|---|---|---|
| `pb_otpu_hbm` (SLR0) | `u_mem/g_port[*].u_nch`, `.u_hreg`, `.u_port`, `u_mem/u_host`, `u_ocl`, `u_cx/g_xr.u_m`, the SLR0 stage of every chain (`u_mem/g_xr.g_x[0]`, `u_cx/g_xr.g_x[0]`, `g_rs[0]`) | X0Y0:X3Y3 + X4Y0:X7Y0 (20, the CL's whole SLR0 area) | ~272K | ~31K (11%) |
| `pb_otpu_mid` (SLR1) | the middle stage of every chain (`g_x[1]`, `g_rs[1]`) | X0Y4:X3Y7 (16) | ~217K | 0 LUT, ~18K FF |
| `pb_otpu_board` (SLR2) | `u_board`, the SLR2 stages (`g_x[2]`, `u_cx/g_xr.u_s`, `g_rs[2]`) | X0Y8:X7Y11 (32, all of SLR2) | ~435K | 226K (52%); 416 of 672 RAMB36, 1,275 of 3,008 DSP |
| `pb_otpu_tmem` (child of `pb_otpu_board`) | `u_board/u_slice/u_tmem`, `u_vpu` | X2Y9:X5Y10 (8) | ~109K | 48K (44%), 68 RAMB36 |

Why this split: SLR1's CL area is 23 clock regions (~313K LUTs), 72% full with the board, too
full for 200 MHz routing; SLR2 holds it at 52% with the MXU, TMEM and VPU in one SLR, so the
adapter's readies into the units, the grants into the MXU and the pumped banks' addresses
never cross an SLR. The alternative, the adapter in SLR1 as the bridge to SLR0 and the slice in
SLR2, would put port B's 4,096-bit data and the ports' same-cycle readies (`b_rdy`, `sw_rdy`,
`a_rdy`: a request is taken when req && rdy in the cycle) across an SLR, which needs a
credit-based port, not a register; the native ports already have the credit form (a FIFO behind
a registered ready) and cross instead. The TMEM beside the VPU: the pump paths (clk -> clk_f,
one core period) start at the VPU's `ta_addr` and the MXU's `t_raddr`; the MXU (97K) wraps
around the box. Round 2's directive (SSI_SpreadLogic_low) is moot with the pblocks; build.py's
default stays until the parent decides (`--place-directive Default` is the natural choice).

**The SLR crossing registers**: every net between SLR0 and SLR2 is a chain of three `otpu_xr`
stages (rtl/shell/otpu_xr.sv, otpu_board's S > 1 crossing stage moved to its own file: its
`r_reg` with max_fanout 256, the cells a floorplan marks USER_SLL_REG), one per SLR, placed by
the pblocks above, so each SLR crossing runs flip-flop to flip-flop with nothing between
(TX_REG / RX_REG eligible). No stage resets (a word in flight at a reset flushes in three
cycles while both sides hold it).

| crossing | module | down (SLR2 -> SLR0) | up (SLR0 -> SLR2) | cycles added |
|---|---|---|---|---|
| the native ports (port B's requests and data, port A's, the SW queue's: everything the adapter sends) | otpu_f2_hbm_mem `NXR` = 3, `g_xr.g_x[k].u_dn` / `.u_up` | per channel the adapter's takes (`n_cvalid && n_cready`, `n_wvalid && n_wready`) with the command (we, 25-bit beat) and the beat (512 + 64): 604 bits x 8 | the bridge's readies, `n_rvalid` + 512, `n_wdone` 16, `n_err`: 532 x 8 | +3 each way: a read's data 6 cycles later, a write counted 6 later. The bridge takes what arrives (`XL` = 6: the adapter sends what it saw a ready for, 3 cycles earlier, and the beat lands 3 later) and its native FIFOs go from 4 to 16 entries with the ready's threshold at 16 - 2 - 6 = 8 |
| the control path (OCL -> the board's registers) | otpu_axil_xr `u_cx` (rtl/boards/f2/otpu_axil_xr.sv): `g_xr.u_m` (SLR0), `g_xr.u_s` (SLR2), `g_xr.g_x[k]` | AW, W, AR as pulses with their words (63 bits) | the takes (the master's readies for a cycle), B and R as pulses (41 bits) | +3 each way per access (an access every microseconds) |
| the board's reset and the HBM's ready | otpu_f2_hbm_sys `g_rs[k].u_x` | | `rst`, `hbm_ready` (2 bits) | the board enters and leaves reset 3 cycles after the fabric (idle both times) |

Measured on the RTL (tools/perf_qwen.py `--model gemma3 --layers 4 --portb 4 --mcols 8
--rowlanes --aline 64 --mhz 200`, OTPU_TARGET=f2-hbm OTPU_NCH=8, int4 and int8 with the head in
the same format, 4 layers + head): at --lat 60 int4 565,739 cycles and int8 949,474, the
baselines to the cycle (the stages are in the F2 fabric only, which tb_top does not build, so
the regression cycles are the base's by construction). The equivalent on the native model,
6 more cycles of read latency (--lat 66): int4 598,971 (+5.9%: MXU starved 3.0% instead of
2.2%, blocked 19.0% instead of 15.2%), int8 1,015,899 (+7.0%). So the port is latency-bound
at this depth (the adapter's 64 reads a channel in flight, the MXU's chunk FIFO): 30 ns on the
HBM's ~400 would cost of that order on the card, and two stages a direction (a net spanning
SLR1 flip-flop to flip-flop, if the placed build shows the 5 ns hold it) or more reads in
flight would take it back. The host's write path and the control path are not on a run's
cycle count.

**Fan-out** (`(* max_fanout *)` attributes; "before" is the net's loads in the RTL, "after" the
loads per copy synthesis leaves):

| net | where | before | after |
|---|---|---|---|
| `ws` (the host path's W half select) -> the ports' `m_wdata` / `m_wstrb` | otpu_f2_hbm_host | 288 LUT selects (the 256 + 32 mux), each LUT into 8 ports: 2,304 register loads behind `ws` (216 replicas placed) | 0: the W beat shown to the ports is a register (`hw_wd`, `hw_ws`, `hw_wl`: the low half loads with the head, the high half as the low is taken), 8 register loads per bit, no logic; `ws` keeps its 17 valids and the one load select |
| `gnt` (the slice's grants: `t_gnt` is the MXU's enable) | otpu_slice | ~5,000 register enables in the MXU (the drain, the lanes, the epilogue), the VPU's, QUANT's | max_fanout 256: the arbiter's last level replicated per 256 loads |
| `rl_pq` (the pending RLD read) | otpu_slice | ~40 (the request array's en and 32 address bits, the rl_* registers) | max_fanout 8 |
| `ta_addr`, `tb_addr` (the VPU's TMEM addresses) | otpu_vpu | ~16-40 per bit over the TMEM (lane rotation, bank compares, the 16 pumped banks' address AND-OR) | max_fanout 16 on the register (`ta_q`, `tb_q`) |
| `t_raddr` (the MXU's TMEM read address) | otpu_mxu | the same | max_fanout 16 (`t_rq`) |
| `wh_r` (the adapter's SW-queue bucket hash, 3 bits x 8 channels) | otpu_native_dram | ~30 per bit, all in the adapter (the bucket compares, the block flag's mux) | unchanged: the 8 ns were the SLR split, not the fan-out (the adapter and the units share SLR2 now) |

**The pumped TMEM address** (clk -> clk_f): otpu_clk2x.tcl's multicycle is `-from` the core
clock `-through` every otpu_pump_bank's `ra`, `rb`, `rena`, `renb` pins `-to` clk_f, so it covers
every start point that reaches a bank's address, `rl_pq` through the collective port's request
included (the report's 5.000 ns requirement on those paths confirms it); nothing to extend, and
no register added (that would be a cycle of TMEM read latency on every port). With the slice
in one SLR and the TMEM beside the VPU the 90%-route paths shorten; the attributes above place
a copy of each address register per 16 loads.

**Status (2026-10-17, Verilator 5.048, bit-exact with the ISA simulator):**
tests/test_f2_rtl.py 16 passed (the fabric at NXR 3 and 0, the OCL crossing with and without
otpu_axil_xr); tests/test_wide_port_rtl.py 168 passed (1:09 h, cold cache); tests/test_rtl.py
-k "replay or act_reads or attention or lane_count" 27 passed, 1 skipped;
tests/test_observability.py 18 passed; tests/test_host.py 90 passed, 1 skipped, the base's
3 host-side failures (no RTL: the FakeTransport caps / close, no transformers in the venv);
tests/test_f2_build.py 84 passed (the HBM snapshot's constraints and the floorplan under the
Tcl mock); tests/test_target.py and tests/test_tourney.py 95 passed (the source lists with
rtl/shell/otpu_xr.sv). The perf runs above.

### Placement, round 3: the floorplanned build (2026-10-18)

The first floorplanned build (d9cdda12, `--place-directive Default`, host dir
/home/ubuntu/opt/hwloop-nightly/p4rl8-d9cdda12) synthesized and linked, then failed opt_design's
DRC: `[DRC HDPR-23] Nested Pblock ranges must be a subset of parent Pblock ranges: The child
Pblock 'pb_otpu_board' is not contained by the parent Pblock 'pblock_CL'`, the child's range
`CLOCKREGION_X0Y8:CLOCKREGION_X7Y11`, the parent's a list of site ranges with holes in SLR2
(`SLICE_X8Y660:X8Y719`, `SLICE_X182Y480:X182Y539`, `URAM288_X4Y32:X4Y127`, ...). Two things
the log showed on the way:

1. **The HDK reads the user placement file as a managed XDC.** Its build_level_1_cl.tcl does
   `read_xdc ${constraints_dir}/${SHELL_MODE}_cl_pnr_user.xdc` (no `-unmanaged`), and a managed
   XDC accepts constraints only: every `proc`, `if`, `foreach`, `puts`, `get_msg_config` line
   is dropped with `CRITICAL WARNING: [Designutils 20-1307] Command '...' is not supported in
   the xdc constraint file`, and the read goes on. So since round 1 the linked reset exception
   ran as `set_false_path -through <pin> -to {}` (`[Vivado 12-4739] No valid object(s) found
   for '-to '`: the synchronizers' CLR recovery / removal paths were timed, never exempted),
   and in round 3 the four pblocks were created with their ranges and **no cells**
   (`otpu_cells` unknown, every `add_cells_to_pblock` dropped): the DRC fired on empty pblocks,
   and had it passed the placer would have been free as in round 2. The rounds' timing numbers
   stand (nothing the file said was applied), the floorplan had not been tried.
2. **The parent's ranges are not whole clock regions.** small_shell_level_1_fp_cl.xdc adds
   clock regions and takes the shell's own slice columns out, and `SNAPPING_MODE ON` on a
   reconfigurable partition's pblock derives site ranges (DERIVED_RANGES) that leave out the
   static shell's sites and the SLR boundaries' LAGUNA rows; a child added by clock regions
   holds those sites and is not a subset.

Fix (this commit): the Tcl moved out of the managed file into
boards/f2/build/constraints/cl_pnr_user.tcl (the reset exception; build.py appends
cl_pnr_user_hbm.tcl, the floorplan, for an HBM target as before), small_shell_cl_pnr_user.xdc
holds comments only, and boards/f2/build/scripts/build_level_1_cl.tcl, our wrapper that sources
the HDK's sequence, wraps its `opt_design` once: `read_xdc -unmanaged
${constraints_dir}/cl_pnr_user.tcl`, then the HDK's opt_design (a Tcl error in the file stops
the build; a sequence that never reaches opt_design stops it too). The floorplan's
`otpu_contain`: after a pblock's clock regions, every site of it that its parent does not hold
(`get_sites -of_objects`, the parent's DERIVED_RANGES) is taken out with `resize_pblock
-remove`, children in parent-first order, `SNAPPING_MODE` off on them (a child is not a
reconfigurable partition; snapping would put the sites back). tests/test_f2_build.py: the
floorplan mock's `pblock_CL` lacks one site of SLR2's clock regions and the test expects exactly
that site removed from `pb_otpu_board` (87 passed); a new test runs the wrapper against a mock
HDK sequence (the unmanaged read lands after the managed one and before opt_design, once;
no opt_design or a bad file fails).

**Round 3 placed (76a8f878, `--place-directive Default`, the floorplan applied, host dir
/home/ubuntu/opt/hwloop-nightly/p4rl8-76a8f878): WNS -5.567 ns** after place_design, the
pblocks as intended (board: 168K LUTs, 50% of SLR2's, 422 of 672 RAMB36; hbm: 30K LUTs, 11% of
SLR0's). After phys_opt_design, by clock (report_timing on the post_phys_opt checkpoint,
place_rpt/ in the run dir):

| clock | round 2 (free placement) | round 3 | worst path (route share) |
|---|---|---|---|
| clk_raw 200 MHz | -4.05 | **-1.42** | inside the MXU: `mx_gq_reg` -> `p1_sw_reg` 9 levels, 5.2 of 6.1 ns route; the weight FIFO's BRAM -> `wd_reg[63]` 5.0 of 6.3 ns |
| hclk_raw 450 MHz | -4.89 | **-1.98** | the bridge's `wf_n_reg` -> `k_d_reg[*]/CE` (604 enables from one LUT, 3.6 of 3.8 ns route) |
| clk_raw -> clk_f_raw 400 MHz | -2.36 | **-1.34** | the pump bank's `pk_a_reg` -> its BRAMs' ADDR, 1.3 of 1.4 ns route (the register far from the bank's 16 RAMB36) |
| clk_main_a0 250 MHz | -0.74 | **-5.01** | shell cells both ends (names hidden), 8.4 of 8.6 ns route: the shell's AXI-Lite interface to the OCL side of the control path, which the floorplan put in SLR0 (`u_ocl`, `u_cx/g_xr.u_m` in pb_otpu_hbm) |

So the SLR split is gone from the core paths (the three core-side clocks each gained 1-3 ns and
are now 80-95% route inside one SLR, the fan-out / register-placement kind), and the one
regression is the control path's placement against the shell. Round 4: `u_ocl` and the
control chain's master next to the shell's OCL interface (its SLR from the path's sites),
`wf_n` replicated per 64 `k_d` enables, `pk_a` registers duplicated per 4 RAMB36 (or the bank's
address pipelined one stage, a cycle of TMEM latency), `mx_gq` replicated per MXU row group and
the weight FIFO's read data registered in the BRAM's output register.

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

## Baseline on the board (PORTB 1)

Gemma-3-1B int8 (int8 head) served by otpu-serve on Alex's burst200 image
(agfi-0d06cb48544a02106: 200 MHz, HBM over 2 channels), tools/f2_bench.py, 2026-10-16: decode
**22.3 tok/s** (short prompt) / 21.7 (long), ~89% of the RTL's port-bound ~25 tok/s; prefill
3.7-4.5 tok/s (TTFT 4.3 s for 16 tokens, 59.6 s for 268). For reference, coral-gemma-f2's SGMV
engine build runs the same model at 15.8 tok/s (its 25 MHz Coral NPU bound).

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
