# Milestone 0: the "MXU not consuming" loss split by cause

Branch `v4-m0-gating-split` (off `tpubench`). The v4 plan ([README.md](README.md)) hangs every
candidate's estimate on one number of the Gemma 3 1B decode trace: "MXU not consuming (drain /
row credits)" 18.2% of the token's 10.28 M cycles, the `mb` bucket counter (chunks in the MXU's
FIFO, no pop). P1, P2 and P4 each propose a fix in `otpu_mxu.sv` for it (result-FIFO credits,
a row lead, Sw prefetch) and each says "milestone 0 decides which part carries it". This
milestone makes the RTL say why each blocked cycle did not pop, shows it in Lens, and measures
Gemma 3 1B with it, at the memory model's default stalls and without them.

## 1. What was added

**RTL** (`rtl/mxu/otpu_mxu.sv`, `rtl/top/otpu_slice.sv`). A row's first advance pops only once
(1) all of the row's chunks are in the chunk FIFO, (2) all of its scale entries are in the scale
FIFO (port A), and (3) a result-FIFO credit is free (`rows_live + rows_p < RF`, RF = 32 at MCOLS
4); a later advance never stalls (the existing `$fatal`). So a blocked cycle (`pf_block = more &&
f_count != 0 && !pop`) is at a row's first advance with one of the three unmet, and the MXU now
names the first unmet one in a fixed priority, `pf_bcause[3:0]`:

| flag | Q key | cause | what it says |
|:--|:--|:--|:--|
| `[0]` | `mbc` | credits | no result-FIFO credit: RF rows popped but not yet written to TMEM, the drain side (the epilogue lanes E, or a lost TMEM grant `fm`) |
| `[1]` | `mbr` | row incomplete | the row's chunks are not all in the chunk FIFO (`f_count < c_rch`): the stream is irregular or behind the MXU |
| `[2]` | `mbs` | scales missing | the chunks are in, not all of the row's scale entries (`s_count < c_rch`, port A) |
| `[3]` | `mbo` | other | the remainder: none by construction (`f_av` holds in a blocked cycle; replay groups past the first find the row whole) |

Credits come first because with no credit nothing could pop whatever the FIFOs hold, and a full
result FIFO fills the chunk FIFO within DEPTH cycles (the issuer stops at `occ < DEPTH`), so the
port's idle cycles in that state are the drain's. Then the chunk side, then the scale side.
Exactly one flag is set in a blocked cycle, so the four sum to `mb`.

`otpu_slice` counts the four in simulation-only window sums beside the P/Q counters and prints
them on the Q trace line after `ld` (`mbc= mbr= mbs= mbo=`); the `pf` struct, `NQ` and the
board's trace-buffer records are unchanged, so the card's profiles and old profiles parse as
before. The run's totals (`xb_*`) are printed at the dump as `MXBLOCK` (`rtlsim` stats
`mxu_block`). Counters only: every run's results and cycle counts are the same.

**Lens** (`opentpu/profile.py`, `opentpu/lens.py`, `opentpu/lens_app.html`, [lens.md](../lens.md)
"Bucketed counters"). `parse` takes the keys as it takes any Q key. `lens.blocked_split(buckets,
k0, k1)` totals `mb` and its causes over a bucket range; `summary_data` carries it as
`blocked`; `lens summary` prints "MXU not consuming with chunks in its FIFO ... by cause" and
`notes()` names the causes when `mb` is over 2%. In the port-B attribution ("Where the cycles
went": `CAUSES`, `_bucket_causes`, the app's `bucketCauses`) the window's blocked take is split
in its counters' proportions into `blocked_credits` / `blocked_row` / `blocked_scales` /
`blocked_other`, each with its own colour in the bar and the legend, in the roofline gap's "held
back by" and its advice, and as rows of the MXU card's stall table under "has chunks, not
consuming". A profile without the keys shows the one "blocked" cause as before.

**Tests** (`tests/test_lens.py`): `test_blocked_split_in_causes` (a synthetic bucket profile
with and without the keys: the four sum to what "blocked" took, the summary prints them, the
old profile still renders) and `test_rtl_blocked_cycles_sum_their_causes` (the MLP decode on
the board model's memory path: every bucket's `mb` is the sum of its causes, the run totals
match the buckets, the port-B causes carry the split). The existing RTL tests are unchanged
(counters only).

## 2. Measured: Gemma 3 1B, one decode token

The same token as [gemma3.md](../gemma3.md) "Lens" (position 14, int8 weights and head, the
resident decode program, 5,350 instructions, 922 MMs, 721,408 streamed rows: 459,264 in the 26
layers, 262,144 in the LM head; the board model, `--axi`), recorded three times with the memory
model at different settings: `+axi_stall=20` (the default, fitted to the card's DDR3: the
trace in gemma3.md), 5% and 0%, the latter two with `OTPU_LAT=30` (the coordinator's HBM-like
run; `build/lens/v4/today-rtl-200mhz-measured.otpuprof`, reproduced to the cycle: 8,221,777).
The cycle counts of the stall-20 run are the earlier trace's to the cycle (10,279,178): the
counters change no behaviour.

| memory model | cycles | of the roofline | port B idle attributed to `mb` | `mb` cycles (of the run) | credits `mbc` | row `mbr` | scales `mbs` | other | `mb` a row |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| stall 20, LAT 20 (the card's DDR3) | 10,279,178 | 76.4% | 18.2% | 2,213,424 (21.5%) | 3,012 (0.1%) | 2,151,907 (97.2%) | 58,505 (2.6%) | 0 | 3.07 |
| stall 5, LAT 30 | 8,651,201 | 90.7% | 5.0% | 623,803 (7.2%) | 5,507 (0.9%) | 522,418 (83.7%) | 95,878 (15.4%) | 0 | 0.86 |
| stall 0, LAT 30 (HBM-like) | 8,221,777 | 95.5% | 1.4% | 171,243 (2.1%) | 5,795 (3.4%) | 96,814 (56.5%) | 68,634 (40.1%) | 0 | 0.24 |

(The percentages in the cause columns are of `mb`. "Port B idle attributed to `mb`" is Lens's
"Where the cycles went" figure, the idle cycles of the windows that had blocked cycles; `mb`
itself also counts blocked cycles in windows where port B was streaming.)

By phase (`tools/mxu_block_split.py`, the LM head from the first `mm 1x1152 . 8192x1152^T`):

| | layers: `mb` (of the phase) | credits | row | scales | a row | LM head: `mb` (of the phase) | credits | row | scales | a row |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| stall 20 | 1,536,342 (21.2%) | 0.2% | 96.0% | 3.8% | 3.35 | 677,082 (22.2%) | 0 | 100% | 0 | 2.58 |
| stall 5 | 442,342 (7.2%) | 1.2% | 77.1% | 21.7% | 0.96 | 181,461 (7.1%) | 0 | 100% | 0 | 0.69 |
| stall 0 | 132,516 (2.3%) | 4.4% | 43.8% | 51.8% | 0.29 | 38,727 (1.6%) | 0 | 100% | 0 | 0.15 |

The other counters of the same runs: the MXU drain lost 221-226 k cycles to TMEM arbitration
(`fm`, all in the layers; it does not reach port B), the chunk FIFO was empty with work (`ms`)
61 k / 16 k / 39 k cycles, port B requests waited for the memory (`bs`) 5 k / 2 k / 0.6 k, port
A's (`as`) 9 k / 15 k / 16 k; "MXU computing, not streaming" (the per-MM fill and drain, 922
MMs) 3.9% / 2.8% / 1.5%; waits on the VPU and quantizer 0.7% / 1.1% / 1.3%.

**What the split says.**

1. **Credits never gate.** `mbc` is 3-6 k cycles a token in every regime, 0.03-0.07% of the
   run, none in the LM head. With M = 1 a row's result drains in a cycle; at KB 9 (54) a row
   pops every 9 (54) cycles and the pop-to-drain pipeline is ~40 cycles, so 4-5 rows are live
   of the RF = 32 credits. The 2.1% TMEM arbitration loss of the drain (`fm`) is absorbed by the
   same margin: it never backs up into the credits.
2. **Scales gate a little, only in the layers.** `mbs` 59-96 k cycles (0.6-1.1% of the run),
   none in the LM head (rows of KB 9 whose scale lines arrive with the chunks): the layers'
   rows whose scale entries complete after their chunks (port A's requests go out with the
   chunks' but return through their own queue; `as` 9-16 k). At stall 0 it is 40% of what is
   left of `mb` and the largest MXU-side item.
3. **Row-incomplete carries the rest, and it is the memory's rate, not the gating.** `mbr`
   is 97% / 84% / 57% of `mb`, and in every regime and both phases the chunk FIFO held **6-8
   chunks** (`lens.fifo_level_at`: the mean level in the windows with late rows), under one KB-9
   row, out of 1,024: the stream never gets ahead of the MXU during an MM. The FIFO fills
   (>= 160 chunks) only in the 4-23% of cycles between MMs, when the MXU waits on the VPU or
   the quantizer. The MXU consumes a chunk a cycle; the memory model returns 0.76 chunks a cycle
   at stall 20 (7.84 M chunks in 10.28 M cycles), 0.9 at stall 5, and at stall 0 the port's own
   rate with the DDR3 model's refresh and row-miss hiccups (tRFC, tREFI: `ddr3_plusargs`). In
   that regime every chunk is consumed the cycle after it arrives, and a row whose last chunk
   has not arrived shows as `mbr` for exactly the memory's shortfall: the row-granular pop adds
   nothing in steady state (a row's first pop waits for its last chunk, and while its 9 pops
   run the next row's 9 chunks arrive), only the fill at each MM's start (~8 cycles x 922 MMs =
   7 k, inside the LM head's 39 k at stall 0 with the hiccups). The `+axi_stall` knob withholds
   the port's handshakes, so it caps the port's throughput at 1 - stall; it is a memory that
   cannot keep the port's rate (the card's DDR3-1066 peak, 17.07 GB/s, is exactly 128 B x
   133.33 MHz), not jitter with headroom.

So of the 18.2%: **~17.7 points are the memory delivering below the port's rate** (gone with a
memory that keeps it: the stall-0 run's 95.5%), **0.5 are scales**, **0.0 credits**. On an HBM
that keeps the port's rate the whole "MXU not consuming" item is 2.1% of the token (0.24 cycles
a row): 0.8% scales, 1.2% rows late at the DDR3 model's hiccups, 0.07% credits.

**A second data point: qwen-tiny** (`lens record qwen-tiny --board --axi --pos 64`, the small
random Qwen3 at position 64, stall 20: 22,107 cycles, 70.5% of its roofline, 255 instructions).
`mb` 1,734 cycles (7.8%): **credits 470 (27.1%)**, row 1,264 (72.9%), scales 0, other 0; the
LM head's 136 all row. Its hidden size is 256, so its rows are KB 2 (two-cycle rows: ~20 live in
the pipeline's ~40 cycles) and the MXU drain lost 2,148 cycles (9.7%) to TMEM arbitration,
which here does back up into the 32 credits. The row-incomplete part has the FIFO at 1.8 chunks
while rows were late. So the credit case exists, for short rows with a contended drain; Gemma
3's rows of 9 and 54 are not that case, and a candidate that shortens rows (P4's note on
splitting Wo by K to KB 2) would be.

## 3. What this means for the proposals' row-gating fixes

Each proposal's section 2 and 5 assumed the 18.2% was per row and in the MXU (P1: "2.6 cycles a
row, unchanged unless fixed", 0.36-0.72 M cycles after its fix; P2: "per row + memory rate",
0.10-0.57 M; P4: "2.8 cycles a row", the fix "worth ~14%"). The split says:

- **The 18.2% is not a row-gating loss.** 97% of it is the memory model delivering 0.76 chunks
  a cycle to a port that requests one, with the MXU consuming each chunk as it lands. It is
  the card's DDR3 (whose peak equals the port's rate) and the model fitted to it, and no change
  in `otpu_mxu.sv` recovers any of it; a memory that keeps the port's rate does (stall 0: 95.5%
  of the roofline, 8.22 M cycles, 0.24 blocked cycles a row). The proposals' "RTL as it is"
  baselines (P1 31 tok/s at 2.6 a row, P4 72 tok/s at 2.8 a row) are too pessimistic by that
  much; their "fixed" cases (P1's 0.5 a row, 41 tok/s; P4's 1 cycle a row, 89 tok/s) are about
  what the unfixed RTL already does on a memory with headroom. The README's "the RTL will lose
  more than the model: 0.81x" ratio is the memory model's, not the RTL's: against the ISA
  model's 8.28 M the stall-0 RTL takes 8.22 M, 1% under it.
- **P1 (a) RF 32 -> 128 credits per row group, P2's per-row-group credits, P4's RF 64:** not
  what carries the loss. Credits gate 0.03-0.07% of Gemma 3's token at every stall setting.
  Keep them only as insurance where rows get short (KB 2 at P4's K split, qwen-tiny's 27%) or
  where C chunks a cycle shorten a KB-9 row to 9/C cycles: at C = 4 a row pops every 2.25
  cycles, ~18 rows live of 32 (P2's row-group entries hold 4 rows each, so its 32 row-groups
  are ample); the drain's TMEM-arbitration loss (2.1-2.7% of the cycles at `fm`, absorbed by the
  margin today) is what eats into the margin first. A per-row-group credit is cheap and sound;
  it is not a performance item for Gemma 3.
- **P1 (c) and P4's Sw / scale-line prefetch:** the one MXU-side item that is measured. Scales
  are 0.6-1.1% of the token (none in the LM head; 52% of what is left of `mb` at stall 0),
  from rows whose scale entries complete after their chunks. At C = 2-4 the same per-row
  lateness is 2-4x the share unless port A keeps pace (each proposal's section 9: port A at C
  lines a cycle, or the scale entries fetched a row ahead). Worth doing; worth ~1% at C = 1,
  2-4% at the candidates' widths if left alone.
- **P1 (b) two-row lead, P2's zero-bubble `row_in` lookahead:** the row-incomplete part they
  address is 0.6% of port B's idle at stall 0 (`mbr` 1.2% of the cycles, the DDR3 model's hiccups), with the FIFO
  at 6-8 chunks while rows were late: a lead cannot pop a row whose chunks have not arrived.
  The row-granular pop adds no steady-state bubble (a row's pops overlap the next row's
  arrivals) and ~8 cycles of fill per MM start (922 MMs: 7 k cycles, 0.1%). What a lead buys is
  the hiccup absorption that a stocked FIFO gives for free: on the F2's HBM the port's rate is
  11-45% of the memory's, so the FIFO stays stocked during a stream and the lead is moot; on a
  memory at the port's rate (the card) it recovers nothing. Not a priority.
- **The per-row costs that do scale with C** are the rest of the stall-0 gap: "MXU computing,
  not streaming" 1.5% (the fill and drain of each MM, 123 k cycles = 0.17 a row, 130 a command)
  and the VPU / quantizer waits 1.3% (per instruction, software-optimizations.md). At C = 4 a
  KB-9 row is 2.25 cycles, so today's 0.24 + 0.17 = 0.41 cycles a row of MXU-side fixed cost
  would be 18% of the stream unless the per-row items shrink with the width (P2's lanes pop a
  row group at once, so its per-row cost is per group of 4; P1's groups likewise per pair).
  That, the scale prefetch and port A's rate are the row-level work the candidates still have;
  the result-FIFO credit changes can be dropped from their critical path.

**For the F2 estimates:** take the stall-0 run (8.22 M cycles at C = 1, 95.5% of the roofline)
as "today's RTL on an HBM that keeps the port's rate", not the 10.28 M trace; the ISA model
(8.28 M) and it agree to 1%. The Lens candidate re-timings in the README (P1 4.36 M, P2 2.19 M,
P4 2.40 M) are then the right starting point with a ~5% RTL loss, not 19%, provided the
per-row items above are kept per row group at the candidates' widths.

## 4. Reproducing

```sh
PYTHONPATH=. python -m opentpu.lens record model --model gemma3 --axi -o build/lens/m0-gemma3.otpuprof
OTPU_LAT=30 PYTHONPATH=. python -m opentpu.lens record model --model gemma3 --axi --stall 0 -o build/lens/m0-gemma3-stall0.otpuprof
OTPU_LAT=30 PYTHONPATH=. python -m opentpu.lens record model --model gemma3 --axi --stall 5 -o build/lens/m0-gemma3-stall5.otpuprof
PYTHONPATH=. python -m opentpu.lens record qwen-tiny --board --axi --pos 64 -o build/lens/m0-qwen-tiny.otpuprof
python -m opentpu.lens summary build/lens/m0-gemma3.otpuprof       # the split under "by cause"
python tools/mxu_block_split.py build/lens/m0-gemma3*.otpuprof     # by phase, with the FIFO level
python -m pytest -q tests/test_lens.py                              # the split's tests
```

Each Gemma 3 run is 10-15 minutes on the Verilator board model (`models/gemma-3-1b-it`, the
Hugging Face checkpoint). The profiles open in the app (`lens open`), where the four causes
have their own colours in "Where the cycles went", the MXU card lists them under "has chunks,
not consuming", and the roofline gap's "held back by" names the one that leads.
