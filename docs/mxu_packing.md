# MXU DSP packing: two int8 products a DSP48E2 on the CL 16 cascade (phase 1 of the VLA plan)

Status: feasibility study (2026-10-16), the plan's phase 1 (docs/vla_frequency_plan.md L1, section
7; docs/vla_frequency_plan_review.md finding 1). Verilator results are measured; the
out-of-context synthesis is prepared for the nightly host and **not run**. Nothing here is in a
file list or in otpu_mxu's array: the integration is phase 3 (section 6).

**Short answer.** Yes: one DSP48E2 computes two exact int8 products a cycle on this MXU's 16-deep
cascade, with the lanes split at the chain end and a one-bit fix-up that the column end's
pre-adder absorbs. A packed CL 16 chain and a packed physical column (128 packed PEs, two column
ends) are bit-exact against numpy in Verilator over the int8 corners, K up to 16384 and the MX
shifts (section 4: 9 tests, ~11,500 blocks, 0 mismatches). The right thing to pack on this MXU
is the **weights** (two weight rows a DSP, section 2), not the activations the plan's L1 names:
it keeps the ACT RAM at 16 RAMB36 a physical column (512 at MCOLS 64 logical, not 1,024), port
B's bytes per MAC unchanged, and the drain's row order. MCOLS 64 logical costs **32 x 146 =
4,672 DSPs** for the array (+192 for 32 epilogue lanes: ~4.9K), against 8,768 (+192) unpacked:
the whole part's 9,024. The only operand the packing cannot take is -128, which quant.py's
int8 never produces (section 3.4). The fmax and the routed DSP / LUT counts of one packed column
on the xcvu47p are the nightly's (section 5).

## 1. The question

docs/mxu_systolic.md ("One product per DSP"): "Two int8 products per DSP48E1 is not exact on a
cascade (the study: the low field overflows after 3 accumulations), so each column has 128
product DSPs". The review's argument (finding 1): the DSP48E2's 27-bit pre-adder / A input
leaves a 19-bit low lane under an 8-bit high one, and a CL = 16 chain's sum of products in
[-127, 127] is at most 16 x 127^2 = 258,064 < 2^18, so a chain's packed sum is exact if the
lanes are split at the chain end, before the column end's accumulation over the row's blocks.

The earlier finding is the DSP48E1's: its pre-adder and multiplier input are 25 bits, so an 8-bit
high lane leaves a 17-bit low lane (|L| < 2^16 = 65,536: four products of 127^2, not sixteen).
The DSP48E2's extra two bits are what make a whole chain fit. The repo already has a packed
DSP48E2 column (rtl/exp/mx_mxu/otpu_mx_us.sv, docs/mx_mxu.md 5: `otpu_pack_pe`, A = x1 << 19,
D = x0, B = w, "S0 = sext(P[18:0]), S1 = P[47:19] + P[18]"), bit-exact in xsim on the unisim
models (260 rows, 0 mismatches) and routed out of context on an xcku5p (288 DSPs for two packed
columns of four rows, +2.997 ns at 133.33 MHz). It packs activations (two stationary rows a
column); neither the plan nor the review cites it. This study settles the arithmetic for the
production array (otpu_mxu's chains, column ends and skews), decides what to pack, and prepares
the measurement on the F2's part.

## 2. Two ways to pack: weights (a) or activations (b)

A physical column is one ACT row's 128 DSPs in eight chains of 16; a logical column is one
output stream, one (ACT row, weight row) pair per advance. Today they coincide: an advance
streams one weight chunk (row n, block k) past MCOLS physical columns = MCOLS logical columns.
Packing gives each physical column two logical columns; the two options differ in which operand
the DSP's pre-adder packs and which one the two lanes share.

| | (a) pack weights: A = sext(w_a), D = w_b << 19, B = act | (b) pack activations: A = a_1, D = a_2 << 19, B = w |
|---|---|---|
| the two logical columns of a physical column | ACT row j x weight rows (2p, 2p+1): two weight chunks an advance, one ACT row | ACT rows (j, j + MCOLS_phys) x weight row n: one chunk, two ACT rows |
| DSPs a physical column | 128 + 2 x 9 (two otpu_mxce) = 146 | 146 |
| DSPs a logical column (array) | **73** (today 137: 128 + 9) | 73 |
| the epilogue lanes | E = MCOLS_log / 2 lanes x 6 DSPs, shared: 3 a logical column either way (docs/mx_mxu.md: a lane is 6 DSPs, 1.2K LUTs) | the same |
| MCOLS 64 logical | 32 physical x 146 = **4,672** + 192 lanes = ~4.9K of 9,024 | 4,672 + 192 |
| ACT RAM read width (the review's finding 2: width-bound, 16 RAMB36 a physical column) | **unchanged: 16 a physical column, 512 RAMB36 at MCOLS 64 logical** | two rows a physical column a cycle: 1,024 RAMB36 (double width), or 512 using both ports of each RAMB36 for reads, which takes the quantizer's write port while an MM runs (today a QACT into other rows can overlap an MM) |
| port B | two chunks a pass over a row group: 2 / ceil(min(M, ACT_ROWS) / 32) chunks a cycle: at M = 255, 2 / 8; at the expert's M = 50, 2 / 2 = 1; at M <= 32, 2 a cycle (PORTB 2) for the full rate, half the rate at PORTB 1. **Bytes per MAC as unpacked MCOLS 64**: a pass streams the same N x KB x 128 bytes for the same M x N x K MACs, and the passes over the prefix (ceil(rows / ACT_ROWS)) do not depend on MCOLS | 1 / ceil(min(M, ACT_ROWS) / 64) a cycle: unpacked MCOLS 64's; at M <= 32, half the columns idle |
| MXU cycles a pass (N weight rows, KB blocks, M rows) | N / 2 x KB x ceil(min(M, 255) / 32): at M >= 64 and at M = 50 **the same as unpacked MCOLS 64** (N x KB x 4 at M = 255: the tally's); at M <= 32, half of unpacked's with PORTB 2 | N x KB x ceil(min(M, 255) / 64) |
| the chunk FIFO's replay (a row must fit: KB <= DEPTH) | a row pair: 2 KB <= DEPTH (f2-hbm's FIFO_DEPTH 1024; pi0.5's largest KB is 128), or one FIFO a lane at PORTB 2 | unchanged |
| weight formats packed | **int8 (W8) only**: the pre-adder is the packing's, so PAIR's A / D stream select cannot run on a packed advance; 4-bit PAIR / QUAD and t2 stay unpacked, one product a DSP; W8A16 too (a 16-bit activation's lane needs 28 bits: |L| <= 16 x 127 x 32767 > 2^26) | int8 only, the same |
| the lanes' B = m << sh | each lane its own (two weight rows have their own dw[n][k]): the bm shifter doubles (2 x 8 x 17 bits a column) | each lane its own (two ACT rows have their own da[j][k][g]): the same |
| the result FIFO and drain | a pass yields 32 integers for row 2p and 32 for 2p+1: **two FIFO rows** in today's format (MCOLS_phys integers, their Sa', one Sw), pushed one after the other; the drain is unchanged, rows n, n + 1 in order (TMEM column n, then n + 1) | a pass yields 64 integers for row n: one FIFO row twice as wide (unpacked MCOLS 64's); the drain unchanged |
| column reuse at small M (int8 PAIR at PORTB 2: columns j < M block 2c, j + M block 2c + 1) | subsumed: a packed advance at PORTB 2 is two rows x M columns, the same 2 x M x 128 MAC a cycle as int8 PAIR's M x 2 blocks, and it needs no 2M <= MCOLS | not possible with the lanes sharing the weight: PAIR's two blocks are two weight bytes; it stays unpacked |

Both are exact by the same arithmetic (section 3). (a) is the choice: at MCOLS 64 logical it
halves the BRAM the review's finding 2 made the binding cost (512 RAMB36 against 1,024, or
against losing the QACT / MM overlap), and it changes nothing in the ACT RAM, the drain or the
result FIFO's row format. What it asks for is already on the plan's path: PORTB 2 (L7, a named
dependency) so that M <= 32 commands run at the full rate, and a chunk FIFO that holds a row
pair. The plan's L1 line, "one weight x two activation rows a DSP, the pre-adder forming
a1 + (a2 << 19)", is (b); section 7 says what that changes in the plan.

## 3. The lane arithmetic

### 3.1 The packed product and the chain

For position i of a chain, with weights w_a, w_b (two rows, the same K index) and the activation
a, all int8, the DSP48E2's pre-adder forms

    AD = w_a + w_b * 2^19          (27 bits: |AD| <= 127 * 2^19 + 127 = 66,584,703 < 2^26)

and the 27 x 18 multiplier M = AD * a = w_a * a + (w_b * a) * 2^19. The cascade adds sixteen of
these: P = sum M = L + H * 2^19 with L = sum w_a * a and H = sum w_b * a, each a 16-product int8
dot product. For operands in [-127, 127]:

    |L|, |H| <= 16 * 127 * 127 = 258,064 < 2^18 = 262,144

so L lies in the 19-bit signed range [-2^18, 2^18 - 1] and the low 19 bits of P determine it:
P mod 2^19 = L mod 2^19, and sext19 of that is L because |L| < 2^18. Then P - L = H * 2^19
exactly and H = (P - L) / 2^19. P itself is at most 258,064 x (2^19 + 1) < 2^37: the 48-bit
cascade never overflows.

### 3.2 The sign fix-up

Shifting P right by 19 does not give H when L is negative: P = (H - 1) * 2^19 + (2^19 + L) with
2^19 + L in [2^18, 2^19), so floor(P / 2^19) = H - 1 and the 19-bit field's top bit, P[18], is
set exactly then. Hence

    L = sext19(P[18:0])
    H = P[47:19] + P[18]             (= (P - L) >>> 19)

one bit added to the high field. In the RTL (rtl/mxu/otpu_mxu_pk_col.sv) neither costs fabric
logic or a cycle: the low lane's column end takes {5{P[18]}, P[18:0]} on its A port (wiring), and
the high lane's column end is an `otpu_mxce` with CPG 2, whose pre-adder forms (a + d) * bm: a
= P[42:19], d = P[18]. The chain's last P feeds the column ends' input registers directly, as
in otpu_mxu (the LAST position leaves PREG off, "the column end's A / D input registers take
its place"). H fits 19 bits too, so taking P[42:19] (24 bits) of the 29-bit field is exact.

### 3.3 Where the split happens, and why there

At the chain end, before the column end, for two reasons that each suffice:

1. the column end multiplies each chain end by the group's B = m << sh (< 2^17) and sums eight
   chains and up to 128 blocks of a row into one 48-bit integer; a packed sum would have to keep
   the low lane under 2^18 through all of that, and it fails at the second chain already (2 x
   258,064 = 516,128 > 2^18), let alone x 2^16 over 128 blocks (the earlier study's "overflows
   after 3 accumulations" is this effect on a 17-bit lane);
2. the two lanes are two weight rows with their own block offsets dw[n][k], so their B differ: the
   multiply by B has to happen on the separated lanes. (Under (b) the same holds for the two ACT
   rows' da.)

So a physical column has two column ends (18 DSPs) and the split sits between the array and
them, in the wires.

### 3.4 -128

* quant.py: int8 is [-127, 127] (`q8`: "saturate to [-127, 127]"), for weights (`_int8x` is
  `fp32.quantize_mx`) and for activations (the ISA simulator's QACT and QST are `_quant_rows` ->
  `quantize_mx` -> `q8`, opentpu/isasim.py). The ISA simulator's ACT RAM (`self.act`, int8) is
  written only by QACT, so **it never holds -128**. A weight byte of -128 can only come from a
  host that writes int8 rows outside quant.py: the simulator's MM reads DRAM as int8
  (`wv.view(np.int8)`) and accepts it. The simulator's own group-sum bound comment says |c| <
  2^19.
* With one operand at -128 the packing stays exact: |L| <= 16 x 128 x 127 = 260,096 < 2^18
  (measured: -128 activations against weights in [-127, 127], a -128 weight in the low lane,
  section 4). Fifteen (-128)^2 products and one -128 x -127 (262,016) are still exact; sixteen
  (-128)^2 (+262,144 = 2^18) are the one pattern that breaks the lane bound: the low lane reads
  -262,144 and the high lane one more (the borrow), measured.
* A second limit, the pre-adder's: w_b = -128 puts -2^26 on D, the 27-bit range's end, and any
  negative w_a beside it wraps AD below -2^26, which makes the high lane's product wrong by
  256 x a (measured: -2,080,768 for +2,080,768 at w_a = -1, w_b = -128, a = -127). w_b = -128
  beside w_a >= 0 is exact. The generic model wraps exactly as the DSP (27-bit signed add).

Neither needs a hardware change (CL 8 and twice the column ends, the review's contingency):
both need an operand the quantizer never produces. The integration should keep -128 out of the
weight path by contract (the host's int8 rows are quant.py's), or reject it where the rows are
built, as it does for the other formats' codes.

## 4. The RTL and the Verilator results

* `otpu_pe_pk` (rtl/tech/generic/otpu_tech_dsp.sv, rtl/tech/xilinxus/otpu_tech_dsp.sv): otpu_pe's
  DSP48E2 (AMULTSEL "AD", PREADDINSEL "A", the activation in B (BREG 1), MREG, P = PCIN + M
  down the cascade, PREG off at a chain's LAST, WREG the A / D input registers loaded with en)
  with INMODE fixed at 5'b00100 (A2 to the pre-adder, D enabled, add) instead of the
  per-command stream select, A = sext(w_a), D = {w_b, 19'b0}; no INMODEREG. The generic model is
  the same pipeline cycle for cycle (its 27-bit pre-adder wraps as the DSP's). rtl/tech/xilinx7
  has no packed cell (section 1).
* `otpu_mxu_pk_col` (rtl/mxu/otpu_mxu_pk_col.sv): NCH chains of 16 packed PEs with otpu_mxu's
  skews (the activation through S1 and the stage skew into BREG, the weights through S1 and S2
  or the DSP's A / D registers), the unpack of 3.2, two `otpu_mxce` column ends (CPG 1 and 2),
  bm per lane and group (0 on a cycle without a block, as otpu_mxu's bm2), first / last
  travelling with the block. acc lands CLS + 7 cycles after the ports (otpu_mxu: CLS + 6 + j
  edges after S0). PACK 0 builds the same column on `otpu_pe` with one column end: the
  out-of-context comparison.
* `sim/verilator/tb_mxu_pk.sv` reads a stimulus file (one line a cycle: v first last act wa wb
  bma bmb) and prints the rows' two accumulators; `tests/test_mxu_pk.py` builds it with
  `opentpu.rtlsim.build` (NCH / PACK / WREG through -G) and compares against a python-int
  reference of the ISA's integer, acc = sum_k sum_g bm[g] * sum_i act[i] w[i].

Measured (Verilator 5.048, `pytest tests/test_mxu_pk.py`: 9 passed, 45 s):

| test | what | result |
|---|---|---|
| packed_corners, one chain and the column, WREG 0 and 1 | every sign combination of act, w_a, w_b at +-127 (both lanes at +-258,064), under B = 2^16 and over 128 blocks; lanes at -1 / 0 / +1 in all nine combinations (the fix-up's borrow); L = -1 beside H = +-242,062; alternating patterns in and out of phase, 7 against 9, chains alternating | 54 rows x 4 builds exact; lanes reached +-258,064; |acc| to 1.73e13 (8 chains x 128 blocks x 2^16 x 258,064, inside 2^47) |
| packed_random, 3 seeds | 160 rows each of 1..128 random blocks in [-127, 127], B per lane and group 0 / 2^sh (sh <= 16) / m << sh (6-bit m), random bubbles between rows | 2,656 + 3,135 + 2,534 blocks exact |
| minus_128 | -128 activations vs [-127, 127] weights (incl. 128 blocks at B = 2^16); a -128 weight in the low lane; w_b = -128 beside w_a >= 0; 15 x (-128)^2 + (-128)(-127) | 46 rows exact, lanes to 262,016 / 260,096 |
| minus_128, the failing patterns | 16 x (-128)^2 in the low lane: RTL -2,097,152 / 2,080,776 for 2,097,152 / 2,080,768 (8 chains: -262,144 and 260,096 + 1 each); w_a = -1, w_b = -128, a = -127: RTL 16,256 / -2,080,768 for 16,256 / 2,080,768 | as section 3.4 predicts, pinned by the test |
| unpacked_column | PACK 0, acc_a against the reference, acc_b 0 | 13 rows exact |

The existing MXU RTL tests that build the touched tech file (`pytest tests/test_rtl.py -k
"replay or act_reads"`: the replay fuzz and the ACT-read gating at MCOLS 2 / 4 / 8): 20 passed,
1 skipped (its own "PAIR needs 2M <= MCOLS" condition), 221 s. otpu_mxu.sv is untouched.

## 5. Out of context on the F2's part (prepared, not run)

`tools/mxu_pk_ooc.tcl` synthesizes, places and routes `otpu_mxu_pk_col` with the xilinxus cells
on xcvu47p-fsvh2892-2-e (targets/f2-hbm.toml), PACK 1 (128 otpu_pe_pk, the unpack, two column
ends) or PACK 0 (the existing otpu_pe and one column end), WREG 1 (a column past the first), at
one clock period, in tools/tourney/remote.py's `ooc_tcl()` shape: `synth_design -mode
out_of_context -flatten_hierarchy rebuilt`, opt, place, phys_opt, route, `util.txt` and
`timing.txt` as `tools/tourney/synth.parse_ooc()` reads them, OTPU_WNS / OTPU_WHS / OTPU_PERIOD,
plus OTPU_DSP / OTPU_LUT / OTPU_FF from the routed netlist and the worst path's endpoints.
`tools/mxu_pk_ooc.sh` runs the four steps one at a time, rerunnable.

On the nightly host (remote.py's build hosts: `opentpu`, else `omarchy.tail5bd214.ts.net`,
native Vivado 2026.1 in ~/.local/bin; never on the Mac), from a checkout of this branch:

    tools/mxu_pk_ooc.sh build/mxu_pk_ooc            # pk5 pk4 un5 un4: packed / unpacked at 5.0 / 4.0 ns

Results land in `build/mxu_pk_ooc/<step>/` (`util.txt`, `timing.txt`, `timing_summary.txt`,
`util_hier.txt`, `routed.dcp`, `vivado.log`) and the summary lines in
`build/mxu_pk_ooc/progress.txt` (`DONE <step> rc=0 <s>s`, then that step's OTPU_* lines). Each
run is one column: minutes, not the hours of a board build (rtl/exp/mx_mxu's columns took 3 to 8
minutes on omarchy). To read them: `synth.parse_ooc(vivado.log, util.txt, period)` gives wns,
fmax, dsp, lut, ff.

What to expect and what decides: OTPU_DSP 146 for pk, 137 for un (the DSP48E2 count is fixed
by the instances; synthesis cannot add or remove any); the LUT difference is the unpack (none)
and the second column end's bm skew; the question is fmax. The packed PE's multiplier path is
otpu_pe's (A / D registers -> pre-adder -> 27 x 18 multiplier -> MREG, then the cascade's
post-adder), with the 27-bit pre-adder carrying a wider operand but the same depth; the only
new fabric is the sign-extension wiring and the CPG 2 column end, which the production design
already uses above D = 128. If pk closes 5.0 ns with the margin un has, MCOLS 64 packed is the
f2-vla point (docs/vla_frequency_plan.md 6); if pk closes 5.0 and not 4.0 while un closes both,
250 MHz is the cost of packing; if pk fails 5.0, the fallback is MCOLS 32 unpacked (section 8 of
the plan).

## 6. The integration (phase 3, not implemented here)

In otpu_mxu's array, one physical column j (ACT row j and its replay groups) with 128 DSPs on
`otpu_pe` whose D port is {w_hi, 19'b0} instead of sext(w_hi):

* **The decode (S1)** hands each position {w_hi, w_lo} as today (16 bits a position a hop; the
  hop width does not change). Under a packed W8 command the two streams are weight rows 2p and
  2p+1 of the same block (two chunks an advance, from the chunk FIFO holding the pair, or one
  FIFO a lane at PORTB 2); under every other command they are what they are today (a 4-bit
  chunk's low and high blocks, t2's four).
* **INMODE per command, three values** (the per-column `hi0[j]` becomes two bits, loaded in the
  DSP's INMODE register as today): 5'b00000 the low stream alone (AD = A: lane a), 5'b00110 the
  high stream alone (A gated, AD = D = w_hi << 19: lane b), 5'b00100 both (packed). So an
  unpacked column's product lands in one of the two lanes, and PAIR's columns j >= M read their
  integer from lane b's accumulator: `rowi[j] = acc_a[j] + acc_b[j + M]`. The D port's format is
  the same in every mode; nothing is formatted per command. The INMODE fan-out (hi0 into a
  column's 128 INMODE registers, the MCOLS 4 OOC's worst path) is the same net as today; a
  register per chain removes it, as docs/mxu_systolic.md planned.
* **Two column ends a physical column** (`otpu_mxce` CPG 1 for lane a, CPG 2 with d = P[18] for
  lane b), two bm shifters (bm2 per lane: the low stream's s / m and the high stream's, the same
  `hi1[j]` selection applied twice), the first / last skews shared.
* **The deskew and the result FIFO**: a packed pass yields two rows of MCOLS_phys integers (row 2p
  from the a lanes, row 2p+1 from the b lanes) in today's row format, pushed on consecutive
  cycles (acc_b delayed one cycle; the row's last-block flag doubled), with Sw[2p] and
  Sw[2p+1]; the drain, the epilogue lanes (E = MCOLS_phys: 64 outputs per pass at 2 a cycle,
  as E = MCOLS / 2 at MCOLS 64 unpacked) and the TMEM addressing are unchanged (the drain steps
  TMEM by one column per row as it does; the rows come in order n, n+1).
* **The issuer**: a W8 command with PACK streams weight rows in pairs (N even, or a zero row at the
  end), two chunks an advance over one row group, 2 KB <= DEPTH; its MM count and the replay are
  per pair. The ISA does not change: the compiler sees MCOLS_phys stationary rows (the ACT RAM's
  row groups) and an MXU that takes two weight rows an advance; `isa.chunks_per_cycle` gets a
  packed int8 case (2, as int8 PAIR's) so that PORTB 1 configurations issue it at half rate.
* **Not packed**: 4-bit formats (PAIR / QUAD, the stream select), t2, W8A16 (one product a DSP,
  lane a). The packed array's unpacked rate is MCOLS_phys x 128 MAC a cycle: the plan's MCOLS 32
  unpacked point for those formats.
* **Phase 3's bit-exact gate**: the MM tests at MCOLS_phys 4 / 8 with PACK on W8 (the ISA
  simulator does not change: an MM's result is the same integer), `test_fuzz_mm_replay` and
  `test_act_reads_follow_the_mm_rows` with packed W8 commands, mxu_bench on the pi0.5 shapes.

## 7. What this changes in the plan and the review

* **docs/vla_frequency_plan.md L1** says "(one weight x two activation rows a DSP, the pre-adder
  forming a1 + (a2 << 19))": that is option (b). Pack the weights instead (section 2): the ACT
  RAM stays 16 RAMB36 a physical column, so **L2's and section 6's "BRAM 512 / 1,024 RAMB36 of
  2,016 for the ACT RAM" is 512 at MCOLS 64 packed**, not 1,024 (1,024 is MCOLS 64 unpacked,
  which the DSPs rule out anyway), and the review's option (b) "pump the ACT RAM at 2x" is not
  needed for the ACT RAM.
* L1: "The pre-adder is taken by PAIR's stream select (INMODE via `hi0`), so the packed column
  forms the pair in fabric or redesigns that select": neither. The pre-adder forms the pair, and
  the select stays as a third INMODE value (section 6); no fabric mux per position.
* L1's "est. ~4.6K DSP" at MCOLS 64 packed is 4,672 for the array (32 x 146: the second column
  end a physical column is 9 DSPs, not 0) plus the epilogue lanes (~192): ~4.9K, 54% of the
  part, leaving ~4.1K for the VPU (L4 est. 300-500), QUANT and the fused unit. MCOLS 32 unpacked
  is 32 x 137 + 96 = 4,480 ("4.4K": right).
* The review's finding 1, "the two-products-per-DSP packing that MCOLS 64 depends on was already
  found inexact on this MXU's 16-deep cascade": the finding quoted is the DSP48E1's (17-bit
  lane), and the repo's own DSP48E2 packed column (rtl/exp/mx_mxu/otpu_mx_us.sv) was already
  exact in xsim. The review's bound (258,064 < 2^18) and fix-up are right and are now measured
  on the production cells. Its contingency "if weights can be -128 anywhere ... it needs CL 8" is
  not needed: a -128 weight is exact on its own, and the quantizer never produces one (3.4).
* L9 / the review's 8: W8A16 is half the packed rate, as the plan says once packing holds.
* The tally's MCOLS 64 numbers (0.95 s of MXU, port B 0.51 s at ACT_ROWS 255) stand for the
  packed array at M >= 64 and at the expert's M = 50 (section 2's cycle count: N / 2 x KB x
  ceil(M / 32) = N x KB x ceil(M / 64) there), with port B's bytes unchanged. Row groups are 32
  wide, not 64, so TMEM's row-group arithmetic in tools/pi05_cost.py should be run with
  MCOLS_phys 32 and a 2x MAC rate to confirm (the passes over the prefix, 3 at ACT_ROWS 255, do
  not depend on MCOLS).
* The packed advance subsumes int8 PAIR (docs/wide_port.md's column reuse at PORTB 2) for W8:
  the same MAC rate without 2M <= MCOLS.
