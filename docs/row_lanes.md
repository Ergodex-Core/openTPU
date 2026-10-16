# Row lanes: an MM mode for decode past the ISA's column reuse

Status: design (this page), with the cycle tally of tools/tally_decode.py. The ISA, the
simulator, the compiler, the RTL and the measurements follow in the sections marked as they
land (docs/wide_port.md "Row lanes" holds the measured table).

## Why

docs/wide_port.md "The whole-model tally": with the wide port the ISA's column reuse (x1, x2
PAIR, x4 QUAD) caps any MM at four K-blocks an advance, and a row of KB blocks takes ceil(KB /
4) advances. Gemma-3-1B's rows are 9 blocks (K = 1152): QUAD takes 3 advances (4, 4, 1) where
the port could deliver them in 2.25, PORTB 8 has no instruction that asks for 8 chunks, and
int4 at PORTB 4 is int4 at PORTB 2 with the port half idle. The measured PORTB 4 run (4 layers
and the head, int8) is 1,514,346 cycles against a port of 783,692: the MXU's rounding
(1,064,016 advances) and the two-part wide reads (419,565 of 739,328 cross a group of four
channel pairs) take the rest.

A row lane removes both. At M = 1 (decode) the MXU has MCOLS columns and one activation row:
instead of feeding the idle columns more blocks of the *same* weight row, feed each column group
its *own* weight row. L lanes stream L rows at once, a chunk of each a cycle; column group l
computes output row n + l against the one stationary row; the MXU's cycles are ceil(N / L) x
ceil(KB / blocks a chunk), no rounding of KB to the reuse. This is the SGMV engine's structure
(Ergodex-Core/coral-gemma-f2 rtl/engine: four lanes, each its own rows) inside the MXU, with
openTPU's numerics unchanged: every output is the same integer and the same three fp32
multiplies as a plain MM.

## The ISA shape

**MM RL** (`isa.mm(..., rl=L)`, L in {2, 4, 8}: `rs` bits 3:2 = log2 L; `rs` is a multiple of
D >= 16, so its bits 3:0 are free: bits 1:0 are t2 and QUAD, `RS_MASK` is now 0xF). The MM
streams its N rows L at a time, the L rows of a group side by side:

* `M = 1`; not with PAIR or QUAD (a lane's reuse is its whole chunk: int8 one block, 4-bit two,
  t2 four, so the blocks a chunk, `bpc`, play PAIR / QUAD's part); `L <= Config.PORTB` (the
  chunks a cycle); `L bpc <= MCOLS` (the columns); `L bpc bs <= 32` (the advance's scale data
  in one port-A line, bs = a block's scale bytes: int8 1, int4 / fp4 / t2 4, nvfp4 8). So:
  int8 L 2, 4, 8; int4 / fp4 L 2 (4 columns), 4 (8 columns); nvfp4 and t2 L 2 (4 and 8
  columns). `k0` a multiple of bpc. `Config.ROWLANES` (the build has the mode; CAPS2 bit 4).
* **The weights in groups of L rows** (`quant.rl_rows`): group g = rows gL .. gL + L - 1 at
  `sa + g rs`, its chunks interleaved by chunk: chunk q of row gL + l at `sa + g rs + (q L + l)
  D`. An advance (chunk q of the group) is one aligned run of L chunks. `rs` is the group's
  stride, L x the row's bytes; `sa` is L D aligned. Rows past N in the last group are streamed
  and dropped (the image zeroes them; the ISA says their bytes must be readable).
* **The scale rows in groups** (`quant.rl_scales`): group g at `ssa + g srs`: the L rows' `Sw`
  (4 L bytes), padded to `H = max(4 L, A)` where `A = L bpc bs` is what an advance reads, then
  block k's data of the L rows: row gL + l's at `H + k L bs + l bs` (the same byte / word /
  eight bytes as docs/isa.md's per-row scale rows: `dw` and `m` are read as before). An
  advance reads blocks `k .. k + bpc - 1` of the L rows: A contiguous bytes at `H + k L bs`,
  A-aligned, never across a line. `ssa` and `srs` are multiples of 32 (the Sw and the first
  advances share a line where `H + A <= 32`: int8 L 2 / 4, int4 L 2; else the group's Sw is a
  request of its own, as a row's lone Sw is today). RSC: the group is its L `Sw` words, which
  is what the standard RSC rows (the K cache, one word a token) already are. UNIT: nothing.
* **The result** is a plain MM's: `T[out + n]` for n < N, RMAX the max over them (into `out +
  ors`), ACC and ASCALE as before (ASCALE's one factor, M = 1).
* **QACT DUP8** (flag bit 5, `F_DUP8`; 8 rows <= MCOLS): the row in ACT rows r .. r + 7 rows,
  as DUP (2) and DUP4 (4). An RL MM of L bpc columns reads ACT rows 0 .. L bpc - 1, which must
  hold the same row (DUP for 2, DUP4 for 4, DUP8 for 8), as PAIR / QUAD do.

The ISA simulator computes the mode as the plain MM over the same values (the group layout
read back into rows and scale rows), so the results are bit-identical with and without it:
the mode changes cycles, not values (tests/test_row_lanes.py).

### Why the groups, and not the standard rows with L row pointers

Three reasons, each measured or counted:

1. **Port A.** The MXU's scale reader fetches one 32-byte line a cycle, one request per
   advance. With the standard per-row scale rows, lane l's byte for block k sits in row n + l's
   row, srs apart from the others: an int8 advance would need L lines (L requests a cycle), and
   int4 at L 4 (a word a block, 4 rows x 44 bytes a group of 5 advances) 35 bytes a cycle, over
   the port's 32 even with perfect line caching, which the FIFO-per-advance structure does not
   have (a line register per lane, entries written without a response to trigger them, and
   the issuer stalling on responses in flight: not a small change). Interleaved by block, an
   advance's data is one aligned read, exactly the shape of QUAD's four bytes / four words
   today (the extraction windows `xw`, `x16`, `x32` are reused as they are).
2. **The two-part reads vanish.** otpu_native_dram puts chunk m on pair `(m mod NP) ^ fold(m /
   NP)`, a bijection within an aligned group of NP = NCH / 2 chunks. A wide request of PORTB
   consecutive chunks that crosses a group boundary needs a pair twice and goes in two parts:
   57% of the wide reads at PORTB 4 on Gemma's 9-chunk rows (docs/wide_port.md: 419,565 of
   739,328, 0.42 M port-B cycles of the 1.51 M). An RL advance is L chunks at `sa + g rs + q L
   D`, L D aligned (rs = L x a row's bytes, sa L D aligned): always inside one aligned group of
   NP >= L chunks, so never split. No change to the adapter, the address map, the host's split
   or the fold: the "pad rows to pair groups" option of the two, chosen because it costs no
   silicon and no new address map, and the rows are not padded, only regrouped.
3. **Page locality stays sequential.** The alternative that keeps a row's address affine in its
   index (so a table row can still be gathered by `tok * stride`: the tied head's embedding
   row) is a chunk-column layout, chunk q of every row at `q N D + n D`. Its advances are N D
   apart (32 MB on the 262K-row head): a new DRAM row every beat on every pseudo-channel,
   which the fixed-latency HBM model of the RTL runs would not show and the card would. The
   group layout streams a group's L x KBc chunks as one contiguous run, as today's rows.

The cost of 3 is the embedding gather (kernels/gather.py: row `tok` of the tied head by `tok x
row bytes`, an RLD MUL on the sampled token): row tok of a group layout is at `(tok / L) rs +
(tok mod L) D`, which the ISA cannot form (no divide; `floor(tok / L)` as an fp32 magic-number
round on the VPU plus RLD is possible and exact, four VOPs a token, left for later). The image
keeps the head's element rows in the standard layout for the gather beside the row-lane copy
for the MM: +302 MB on the 1B's int8 head (+29% of its 1.03 GB image), +151 MB in int4; the
layers' projections have no gather and are stored once. A standard-layout copy of the *scales*
is not needed (the gather has its own gather scales).

### What stays out (and how it would go in)

* **M > 1** (`M <= MCOLS / (L bpc)`: a few prefill rows with lanes): the pipeline's structure is
  the same (column (p, l, j) = (p L + l) M + j); the drain's addresses become `out + j ors + n
  + l` and its bank-conflict run depends on ors and L together. M = 1 first.
* **A plain MM over the group layout** (prefill at M = MCOLS over decode's weights): the
  issuer would walk a group's rows (Sw 4 apart, block data bs apart, the block step L bs, a
  wrap at the group's end). Until then an image with ROWLANES is a decode image; its prefill
  rows (rows > 1) are refused at compile time (an RL matrix streamed at M > 1 is a
  CompileError), and a second scale copy (+3% int8 / +6% int4) or that issuer path lifts it.
* **PORTB 8** (int8 L 8 on f2-hbm16, 8 streams at MCOLS 8): the ISA and the simulator take
  it; the RTL's port B, slice and adapter are built to PORTB 4 (`b_n` two bits), so the RTL
  measures L 2 and 4.

## The MXU

The mode is the QUAD datapath with the four (eight) streams reinterpreted. Streams are numbered
block-major, `s = p L + l` (block p of lane l's chunk), and the columns likewise: column J = p L
+ l takes stream J (M = 1). So the lane structure is PAIR / QUAD's with M' = L: columns J >= L
take block 1 (`hi`), J >= 2 L blocks 2 and 3 (`hq`), the row sum adds columns j + L, j + 2 L, j
+ 3 L into column j < L, and what differs from QUAD is (a) which stream a column multiplies
(stream J, not block J / M) and (b) that the L results of a pipeline row are L output rows.

* **Issue.** Every advance asks for L chunks (`b_n = L - 1`, no shorter last advance: KBc =
  ceil(KB / bpc) advances a group, the chunk address stepping L D), one entry of L chunks (the
  wide port's FIFO entry: PORTB >= L chunks, per-chunk write enables), and one port-A line of
  A bytes (the scale offset `so = H + k0 L bs` for the first advance, then a step of A); the group's
  Sw words (4 L bytes at the group's start, word-aligned, within one line) by the existing
  Sw-line logic (`i_swh` / `i_sws`: with the first advance's line where `H + A <= 32`, else a
  request of its own). The command's total is ceil(N / L) x KBc advances.
* **The scale entry** holds the advance's A bytes as slots s = 0 .. L bpc - 1 (slot s = bs
  bytes at s bs: lane s mod L, block s / L) in the wide port's three entry RAMs (slots 0 .. 3:
  `{Sw0, slot 0}`, `{slot 1}`, `{slot 3, slot 2}`) and, for eight streams, a fourth (slots 4 ..
  7, 32 bits each: int8 and int4 only), plus the L - 1 other lanes' Sw (a fifth RAM, 224
  bits). The response extraction is QUAD's: int8 L 4 is the four bytes of a word (`xw`), int4
  L 2 the four words of 16 bytes (`x16`), nvfp4 L 2 the 32 bytes (`x32`); int8 L 2 the byte
  pair (PAIR's), int8 L 8 eight bytes (`x8`), int4 L 4 and t2 L 2 eight words (`x32`).
* **S1** decodes the entry into L bpc streams in block-major order: int8 lane l = chunk l (the
  wide port's order already); 4-bit stream p L + l = chunk l's half p (where the wide port's
  QUAD order is chunk 0's halves then chunk 1's: a per-command select, `rl0`); t2 block p of
  chunk l. The hops carry NS = 8 x 8 bits where the build has eight streams (`RL = 8`: MCOLS 8
  with PORTB 4), else QUAD's 32.
* **The PE select.** Column J's DSP pre-adder takes `{s(2q+1), s(2q)}` by a fabric select
  `s_hq[J]` (one bit for four streams, two for eight: a 4:1, one LUT6 level as today's 2:1)
  and picks within the pair by its INMODE register `s_hi[J]`; in RL `{s_hq, s_hi}[J] = J`
  (stream J), else they are the command's `hq`, `hi` as before. The ACT RAM's selects stay the
  block's (`act_hi[J] = J >= L` and `act_hq` for bpc 4: block p = J / L), so column J reads
  ACT block `ab + k bpc + p` from its ACT row J (the DUP8 copy); the scale decoders (eight:
  four more of ~160 LUT each) feed the column's B by `{s_hq, s_hi}` and the block valid (past an
  odd KB) by `p`, as QUAD's.
* **Sw per lane.** The entry's L Sw words go with the advance to S1 as a per-column Sw (column J:
  lane J mod L) and from the row's last advance into the result FIFO row (MCOLS x 32 bits
  instead of 32: +7 x 32 x 64 x 3 LUT-RAM bits at MCOLS 8), and the drain's lane k multiplies by
  its column's Sw. Sa' per column is unchanged (every copy's Sa).
* **The drain** writes a pipeline row's L results (columns 0 .. L - 1 after the row sum) at
  `out + n0 + l`: consecutive words, distinct banks, min(L, E) lanes a cycle (E = MCOLS / 2
  epilogue lanes: two cycles a row at MCOLS 4 and L 4, which a row's KBc >= 2 advances cover;
  the result FIFO absorbs KB 1 rows); then `+ L` for the next row (today + 1). The last row
  writes N - (G - 1) L results (`Mz`, a per-command field, with a row countdown in the drain).
  RMAX: the per-column running maxima (today's `mk[j]`) are reduced over the L columns at the
  end (an L-input key max, registered) into the one word; ASCALE's factor is `alpha[0]` for
  every lane; ACC reads and writes per column as before.
* **Two heads.** The pop head moves to the next command only if the pipeline treats both
  alike: the condition adds `rl` and L (`n.rl == h.rl`, M' = L compared under rl as M is under
  PAIR).

**The MXU utilization bound at M = 1.** L bpc of the MCOLS columns are busy: int8 L 4 is 4
columns (every column at MCOLS 4; 4 of 8 at MCOLS 8), int4 L 4 is 8 (MCOLS 8), int8 L 8 is 8.
At MCOLS 8 int8 L 4 leaves four columns idle: filling them with PAIR-style block reuse (each
lane two blocks a cycle) would be 8 int8 chunks an advance, PORTB 8; with a second activation
row (M = 2) a batch of two tokens at the same weight bandwidth. The mode runs the port at L
chunks a cycle and the array at L bpc / MCOLS: int8 L 4 is 100% of MCOLS 4 and 50% of 8.

## otpu_native_dram

No change: an RL request is an aligned run of L <= NP chunks (the group layout above), one
chunk on each of L distinct pairs, in one part. The adapter's two-part counter (`NATIVEB
split`, rtlsim stats["native_b"]) on an RL decode is the evidence (the measured table).

## The cycle tally (tools/tally_decode.py, Gemma-3-1B, 26 layers, 200 MHz)

The decode step on the ISA simulator over a zeroed image, every executed instruction at
opentpu.profile's nominal cost: port-B cycles = chunks / PORTB, MXU advances, port-A lines (one
a cycle: an advance's line, and a group's Sw line where it is not the first advance's), `bound`
= max of the three + the other units' 417 K nominal cycles (VPU, quantizer, DMA; the RTL
overlaps most of them: ~130 K net). Standard = the ISA's column reuse as the compiler emits it
(PAIR at PORTB 2, QUAD at 4); row lanes = every scaled, non-RSC one-row MM (838 of 890: the
projections and the head; the attention's K-cache (RSC) and V^T (UNIT) MMs stay PAIR / QUAD)
as an RL MM of the widest L the configuration allows (`quant.rl_lanes`). `--rowlanes`
compiles the step for Config.ROWLANES and tallies the compiler's own RL MMs (the same numbers:
section "ISA, simulator, compiler").

| weights | PORTB | MCOLS | standard: MXU advances, bound, tok/s | row lanes: L, MXU advances, port-A lines, port-B cycles, bound, tok/s |
|---|---|---|---|---|
| int8 | 2 | 4 | 4,243,208 (PAIR), 4,660,147, 42.9 | 2, 3,912,456, 3,905,800, 3,915,101, 4,332,040, 46.2 |
| int8 | 4 | 4 | 2,590,728 (QUAD), 3,007,667, 66.5 | 4, 1,959,816, 1,953,160, 1,957,551, 2,376,755, 84.1 |
| int8 | 4 | 8 | 2,590,468, 3,007,407, 66.5 | 4, 1,959,556, 1,952,900, 1,957,551, 2,376,495, 84.2 |
| int8 | 8 | 8 | 2,590,468 (QUAD: the ISA's widest), 3,007,407, 66.5 | 8, 983,236, 1,096,708, 978,776, 1,513,647, 132.1 |
| int4 | 2 | 4 | 2,590,728 (QUAD), 3,007,667, 66.5 | 2, 2,125,192, 2,118,536, 2,127,837, 2,544,776, 78.6 |
| int4 | 4 | 4 | 2,590,728, 3,007,667, 66.5 | 2 (int4 L 4 needs 8 columns), 2,125,192, 2,118,536, 1,063,919, 2,542,131, 78.7 |
| int4 | 4 | 8 | 2,590,468, 3,007,407, 66.5 | 4, 1,065,924, 1,299,524, 1,063,919, 1,716,463, 116.5 |
| int4 | 8 | 8 | 2,590,468, 3,007,407, 66.5 | 4 (int4 L 8: 64 scale bytes an advance), 1,065,924, 1,299,524, 531,960, 1,716,463, 116.5 |

(int8 weights with the int8 head: 1.03 GB a token; int4 with the int4 head: 0.58 GB. PORTB 1 is
24.3 tok/s in both tallies; the tally's MM counts: int8 890 MMs, 838 of them RL.)

What it says:

* **int8 at PORTB 4 with four lanes: 1.96 M advances, 84 tok/s with the other units serial, ~96
  with them overlapped** (the 1.96 M port-B cycles are the 102 GB/s ceiling of a 4-chunk port at
  200 MHz: 1.03 GB a token). MCOLS 4 suffices (L bpc = 4 columns); MCOLS 8 changes nothing for
  int8 L 4.
* **int4 at PORTB 4 needs MCOLS 8 (L 4 = 8 columns) and is port-A bound: 1.30 M lines against
  1.07 M advances** (the group's Sw line, 16 bytes of 32, is a request of its own every 5
  advances: H = A = 32), 116 tok/s serial, ~150 overlapped. At MCOLS 4 int4 can only take L 2 (4
  columns), 2.13 M advances, 79 tok/s: the same as at PORTB 2. The next step for int4 past 150
  is a 64-byte port-A line (the adapter's beat is 64 bytes already) or two groups' Sw in one
  line.
* **int8 L 8 at PORTB 8 (f2-hbm16, MCOLS 8): 0.98 M advances but 1.10 M port-A lines** (eight Sw
  words fill the header line: a request of its own a group), 132 tok/s serial; the ISA and
  simulator take it, the RTL's port is built to PORTB 4.
* **> 100 tok/s on the tally: int4 with four lanes at PORTB 4 on MCOLS 8 (116 / ~150), or int8
  with eight lanes on f2-hbm16 (132).** int8 with four lanes stops at the port's 102 GB/s: 84 /
  ~96.
* The other units' 417 K nominal cycles are 18% of an int8 L 4 token's bound and 24% of an int4
  L 4 one's: where the RTL's overlap of them (docs/vpu_pump.md, docs/seq_vpu.md) now matters.

## Hardware cost (estimated; Verilator only, no synthesis here)

Per lane beyond QUAD's four streams (the eight-stream build, `RL = 8`, at D = 128, MCOLS 8):

| | cost |
|---|---|
| the hops' streams (32 -> 64 bits a position, columns 1 .. MCOLS - 2) | +6 x 128 x 32 = 24.6 K FF |
| S1's four more streams, and the block-major remap (per stream and position: the lane's chunk by L, the block's half or t2 code, the 4-bit decode: ~20 LUT) and their skew lines | ~15-20 K LUT (as built: `wrl`, a per-command select over three L values), ~1 K FF; the four-stream build's remap is a 2:1 on two streams |
| the PE input select 2:1 -> 4:1 (one LUT6 level, 8 x 128 x 16 bits) | ~+8 K LUT |
| four more scale decoders and the per-column 8:1 of (s, m, v) | ~0.7 K + 2.2 K LUT |
| the scale entry's fourth and fifth RAMs (128 + 224 bits x DEPTH 1024) | 11 RAMB36 |
| the result FIFO's Sw per column (+224 bits x 64 x 3 copies, LUT RAM) | ~1.4 K LUT |
| the drain's row step (+ L), the last row's count, the RMAX key max (an 8-input tree, registered), the per-lane Sw pick | ~0.5 K LUT |
| the issuer: the RL scale offsets (so = H + k0 L bs, step A) and the total ceil(N / L) KBa | ~0.1 K LUT |
| the quantizer's DUP8 and the ACT RAM's column select | ~0.1 K LUT |

The four-stream build (`RL = 4`: MCOLS 4 at PORTB 2 / 4, int8 L 4 and int4 / t2 L 2) adds only
the last six rows (~2.5 K LUT, 11 RAMB36) over the wide port's QUAD datapath: the hops, the PE
select and the stream count are QUAD's. Timing: the new levels are the 4:1 PE select (one LUT6
in place of the 2:1), the S1 4-bit remap (a level after the FIFO's read register, before the
wd register), the scale slot select (one more level before the dw subtract: 8:1 instead of
4:1) and the drain's +L adder (4 bits of carry on a 32-bit incrementer): nothing on the
sequencer's or the grant's paths. The nightly's synthesis reads them off against this table.

## ISA, simulator, compiler (landed)

* **ISA** (docs/isa.md "Row lanes", opentpu/isa.py): `isa.mm(rl=L)`, rs bits 3:2 (`RS_RL`;
  `RS_MASK` 0xF, rs a multiple of 16), `mm_lanes(ins)`, `chunks_per_cycle(rl=)`; QACT
  `F_DUP8` (one of DUP, DUP4, DUP8). The layout: opentpu/quant.py `rl_lanes` (the widest L of
  a format on a PORTB / MCOLS), `rl_rows`, `rl_scales`, `rl_row_stride`, `rl_scale_bytes`,
  `rl_header`, `rl_adv_bytes`.
* **Simulator** (opentpu/isasim.py): `Config.ROWLANES` (OTPU_ROWLANES; in the arch hash only
  when on: every existing hash stays), the MM's RL branch reads the groups back as rows (an
  as_strided view of [G, KBc, L, D]) and the scale groups (`_mx_scales_rl`), then computes the
  plain MM; the legality (ROWLANES, L <= PORTB, L bpc <= MCOLS, the 32-byte line, k0, M = 1, no
  PAIR, L D alignment, 32-byte / 4 L (RSC) scale alignment); QACT DUP8.
* **Profiler** (opentpu/profile.py): an RL MM is ceil(N / L) ceil(KB / bpc) cycles, its chunks
  the padded groups', its port A the lines (an advance's, and the group's Sw line where it is
  not the first advance's); `roofline()` takes them as cycles.
* **Compiler** (opentpu/compiler.py): `QTensor.rl` (`weight(rl=)`; row slices at multiples of L,
  column slices by L D chunks), `Stationary.copies` (DUP8 where eight copies fit), `dot()`
  emits MM RL for an RL matrix at M = 1 and refuses anything else (a CompileError: an RL matrix
  has no other reader). opentpu/runtime.py `Weight(rl=True)` for kernels.
* **Images** (opentpu/llm/qwen3.py `rl_lanes_of` / `rl_alloc`, gemma4.py for Gemma 3): with
  `Config.ROWLANES` the projections, the down parts, the PLE projection and the LM head are
  stored in groups of the format's widest L (L D / 32-byte aligned) and their descriptors
  carry `rl`; the head keeps its standard rows beside them for the embedding gather. The
  decode kernels do not change: the descriptors are the switch.
* **Targets**: `f2-hbm:p4rl` (PORTB 4, ROWLANES; CFG_HASH d6ed551d), `f2-hbm:p4rl8` (and MCOLS
  8; 172e5c73), `f2-hbm:p2rl` (e476f0b7); `target.rl_streams` (4 at MCOLS 4, 8 at MCOLS 8) is
  the RTL's `RL`; `rtl_params["RL"]`, `build_config(ROWLANES=)`; the host reads CAPS2 bit4
  (`regs.caps2["rowlanes"]`, `device_config`, OTPU_ROWLANES must agree, FakeTransport).
* **Tests** (tests/test_row_lanes.py, 38): the RL MM bit for bit the plain MM's over every
  format, L 2 / 4 / 8, UNIT + RMAX, ACC + RMAX, ASCALE, RSC, N not a multiple of L; the
  refusals; the compiler's dot at PORTB 2 / 4, MCOLS 4 / 8; a tiny Gemma 3 and Qwen3 decode
  step with and without the mode, the same logits (int8 and int4); Gemma-3-1B's real decode
  token (26 layers and the head) bit-identical in int8 (PORTB 4) and int4 (PORTB 4, MCOLS 8);
  `tools/tally_decode.py --rowlanes` tallies the compiled mode: the table above, exactly.

## RTL (landed)

`RL` (0, 4, 8: the MXU's streams, `target.rl_streams`: the columns, at least QUAD's four) through
tb_top / otpu_top, otpu_board, otpu_f2_sys / otpu_f2_hbm_sys, tb_board and otpu_slice
(`rtlsim.top_params` only when on: every other build's cache key stays); otpu_ctrl CAPS2 bit4;
otpu_quant / otpu_actram `DUP8` (flag bit 5: rows r + c off, c = 1 .. 7). In otpu_mxu
(rtl/mxu/otpu_mxu.sv), as the design above: `cmd_rl` / `cmd_L` from rs bits 3:2 with the
pipeline's M' = L (`cmd_M`: the hi / hq columns, the row sum's partner distance, Ml), the
structure flags pair' / qd' from the format's blocks a chunk, the chunk count L an advance
(`cmd_nb`, no shorter last advance), the total ceil(N / L) x KBa, the drain's lanes (jo[l] =
l, run = drain_run(1), dinc = L, the last row's Mz with a row countdown d_rem / d_lz), the
issuer's offsets (so = H + k0 L bs, step A, the stride rs & ~0xF), the tag's rl / lg and the
A-byte window into slots (the entry's third RAM plus a fourth for slots 4 .. 7 at RL 8 and a
fifth for the other lanes' Sw), S1's block-major streams (`wrl`: stream s = block s / L of
lane s mod L's chunk), the per-column stream select (s_hi / s_hq = J's bits in RL) beside the
ACT RAM's block select, the eight decoders and the `dec1[{s_hq, s_hi}]` pick with the block's
validity by `{hq, hi}`, Sw per column into the result FIFO (MCOLS words a row) and the lanes,
RMAX reduced over the lanes' column maxima (`mkr`), ASCALE's alpha[0], the pop-head overlap
condition on rl and L. Without RL every term is a constant or a generate branch not taken
(the per-column Sw is one value), and the PORTB 1 / 2 regression suites are unchanged.

**What MCOLS the mode needs for L = 4:** int8 L 4 takes four columns: MCOLS 4 (the f2-hbm
build as it is); int4 L 4 is eight columns (each lane's chunk two blocks): MCOLS 8 (`RL` 8, the
eight-stream datapath). Measured identical for int8 at MCOLS 4 and 8 (949,485 and 949,474
cycles).

**Tests** (Verilator 5.048, D = 128, bit for bit the ISA simulator's DRAM and TMEM;
tests/test_wide_port_rtl.py "row lanes"): 16 cases x the behavioural DRAM and the native path
(the adapter on 4 / 8 channels with 20% random stalls): int8 L 2 (PORTB 2 / 4), L 4 (PORTB 4,
MCOLS 4 and 8), int4 / fp4 L 2 and L 4 (MCOLS 8), nvfp4 L 2, t2 L 2 (MCOLS 8), KB 1 to 9, k0
offsets, N not a multiple of L, each with scales, UNIT + RMAX, ACC + RMAX, UNIT + ACC + ASCALE
and (int8 / int4 / fp4) RSC; the native runs check that no wide read went in two parts; one
program at PORTB 4 / MCOLS 8 / T2 of int8 RL 4, int4 RL 4, int8 QUAD, int4 RL 2, t2 RL 2, plain
int8, int8 PAIR, nvfp4 RL 2, int8 RL 4 with ACC, fp4 QUAD, int8 RL 2 and RL 4 with RSC back to
back (DRAM, NCH 8, NCH 16); the compiler's one-row dot over `Weight(rl=True)` at PORTB 2 / 4
and MCOLS 4 / 8; the throughput: int8 over 9-block rows, QUAD's 3 advances a row against RL
4's 2.25, and 0 two-part reads where QUAD's split.

## Measurements

docs/wide_port.md "Row lanes": the table (4 layers + head, every point; the whole model), the
comparison with the tally, the Bonsai-2 line, what the nightly needs.
