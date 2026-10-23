# openTPU v4: the kernel / software side of Gemma 3 1B decode

A study beside the four hardware candidates of [README.md](README.md) (2026-10-18): what the
decode program already fuses, what a "megakernel" still leaves on the table, how much of the
vector / quantizer / store work the RTL hides under the weight stream today and in each candidate,
and what the compiler and the image layout can do about the row-gating loss without new RTL.
Every number is **measured** (the 133.33 MHz Lens trace, docs/gemma3.md "Lens"), **modelled**
(README's ISA re-timings, or the cycle model of section 7, which reproduces the measured token to
0.7%) or **assumed** (marked). Nothing was simulated for this note.

## 1. Summary

| | today (card RTL at 200 MHz) | P1 C=2 | P2 C=4 | P3 C=8 | P4 4 slices |
|:--|--:|--:|--:|--:|--:|
| hardware alone, host loop, RTL expected (section 7: gating fixed / unfixed) | 22.7 / 19.3 tok/s | 41 / 31 | 86 / 74 | 165 / 143 | 83 / 72 |
| x the card's own decode loop (no host, no logits ST, no fill) | 1.02 | 1.02 | 1.04 | 1.07 | 1.05 |
| x row folding (LM head x4, MLP chunk 2304) | 1.03 / 1.09 | 1.06 / 1.16 | 1.03 / 1.09 | 1.03 / 1.09 | 1.03 / 1.09 |
| x speculative decoding, n-gram, chat text (assumed) | 1.08 | 1.08 | 1.07 | 1.06 | 1.06 |
| **all software, int8** | **26 / 23** | **48 / 40** | **100 / 90** | **197 / 178** | **98 / 88** |
| x fp4 layers, int8 head (bytes 0.66x; accuracy caveat, section 5) | 1.47 | 1.42 | 1.44 | 1.44 | 1.43 |
| all software, fp4 layers | 38 / 32 | 68 / 52 | 144 / 125 | 283 / 245 | 140 / 122 |

The software multipliers are modest (1.1-1.2x in int8 for every candidate) because the resident
decode program plus the card's generate loop already is the megakernel: one program per token,
the loop on the card, no host work per token, and the RTL already hides three quarters of the
non-MM work under the stream. The three items worth doing, by payoff / effort:

1. **Measure and ship the F2 on the card's decode loop, not the host loop** (`Engine.generate_card`,
   docs/autodecode.md). It exists; at P3 the host loop's 0.3 ms + logits store + fill is 7% of a
   5.5 ms token. Effort: none (it is what otpu-chat runs).
2. **Row folding in the image and two kernels**: 4 vocabulary rows per streamed LM-head row (KB 36,
   M = 4) and the MLP's F chunk 768 -> 2304 at decode: 961 K -> 585 K streamed rows a token
   (section 4). Worth 3% if the RTL's gating fix lands at 0.5-1 cycle a row group, 9-16% if it does
   not; at C = 8 it turns a 2-cycle head row into a 4.5-cycle one. 1-2 weeks, no RTL; not bit-exact
   to today's image (a requantized head, section 4.3).
3. **n-gram speculative decoding for Gemma 3** on the card's loop (docs/mtp.md 6.2, 8): the
   verify / accept machinery of opentpu/llm/mtp.py is Qwen3.5's; Gemma needs the drafter and a
   2-row verify at a run-time position. 1.03-1.10x on chat, 1.2-1.5x on summaries and code. 3-4
   weeks; Gemma 3 is the easiest MTP target in the tree (no recurrent state to roll back).

4-bit layers are the one large software lever (1.4x in every candidate) and the one with an
accuracy price the user has declined for the baseline; section 5 gives both numbers.

## 2. What the resident decode and autodecode already fuse

A "megakernel" here means (docs/gemma4.md "How it maps", docs/autodecode.md):

- **One program per token**, resident in IMEM: `gemma4_step` (opentpu/llm/gemma4.py:1879) traces
  the 26 layers as hardware loops over the plan's repeated units (`for it in ol.range(reps)`,
  gemma4.py:1931; Gemma 3: (5 sliding + 1 global) x 4, then 2 sliding), the final norm and the LM
  head (`_lm_head`, qwen3.py:1675): 5,350 instructions. The token's rows are gathered on the device
  (`_gathered`, gemma4.py:1774: 32 one-hot MMs, kernels/gather.py:102-144); the host passes the id
  and position as 6 of the 8 argument words.
- **The generate loop on the card** (`generate.compile_generate`, `Engine.generate_card`,
  qwen3.py:3253): the step inside a `LOOP` over up to 256 positions, each LM-head chunk handed to
  `Greedy` (generate.py:227: one ARGMAX a chunk while the next chunk streams) or the `Sampler`, the
  token fed back through the gather, `HALT CHAIN` to the next bucket. The host reads 64 bytes a
  token. Measured against the resident step: greedy -0.2%, sampled +0.7%, with the repetition
  penalty +2.0% (docs/autodecode.md "Cost").
- **Inside a layer** the kernels are software-pipelined already: gate / up of F chunk c + 1 stream
  while the VPU computes `gelu(g) * u` of chunk c (kernels/mlp.py:56-76; the loop form :93); flash
  attention keeps `ATTN_DEPTH` = 3 score blocks in flight across head boundaries (kernels/attention
  .py:206-290, qwen3.py:863); the drain does `acc * alpha` (docs/compiler.md). The 16-slot window
  with WAITC entries (opentpu/waits.py:1-40, docs/isa.md:531-534) dispatches across it all: an MM's
  stream starts once its DRAM operands are free (counts j = 5, 6), its release waits for its QACT;
  the MXU holds two commands (rtl/mxu/otpu_mxu.sv:41-43) and a 1024-chunk FIFO, so the next MM's
  weights stream through the norm -> QACT chain in front of it (docs/litedram.md "The core's own
  gaps": those chains "cost nothing"; a 2048-chunk FIFO changed 0.00% on the card).
- **No barriers at S = 1**: no BAR is emitted; `all_gather` (mlp.py:67) is the identity. The only
  serializations are the units' in-order queues (DMA, MXU, QUANT) and the VPU's 31-deep mask.

What is still not fused, or still costs per token (cycles at 133 MHz unless said):

| item | what it is | today | at P3 (1.13 M-cycle token) |
|:--|:--|--:|--:|
| the 32 embedding gather MMs | 32 MMs of 9 streamed rows + 9 COPY VOPs before layer 0; they hold the MXU's two command slots, so no weight prefetch runs under them | ~32 x 435 = 14 K (435 = the trace's 0.40 M "computing, not streaming" / 922 MMs) | 1.3% |
| the host in the loop | `Engine.step` (qwen3.py:2603): HALTED -> RUN 0.303 ms steady (docs/autodecode.md) | 40 K (0.4%) | 60 K: 5.3% |
| the logits store | 32 ST of 32 KiB (host path), each under the next chunk's MM; the last one exposed | ISA model 33 K serialized, RTL ~1-2 K exposed | 0.2% |
| `fill_logits` | 32 ST of the -inf sentinel + an RLD gate at the program's start, streamed-logits path only (qwen3.py:1401-1424); layer 0's first MM is 60 instructions away, outside the window | 32 K (0.3%) | 8 K: 0.7% |
| the LM head's tail | the chunk maxima's ARGMAX, the id pick, the state update (generate.py:248) | ~2-5 K | 0.4% |
| the KV appends | 52 QST, a byte a cycle on port SW: 13.7 K in every candidate (README) | 0.13% | 1.2% |
| per-layer chains | ~5 VPU chains a layer the MXU waits on (section 3) | 72 K exposed (VPU 0.2% + quantizer 0.5%, measured) | ~3% |

The card loop removes the second, third and fourth rows: the "+ card loop" column of section 7.

## 3. Compute / memory overlap: what is exposed, what ordering could hide

**The ISA model serializes every unit** (README "Lens profiles": VOP 151 K cycles at 8 lanes, QACT
84 K, ST 33 K, LD 21 K, QST's write 13.7 K: 0.30 M of 8.28 M, 3.6%). **The RTL hides most of it**:
port-B idle attributed to the VPU is 0.2% and to the quantizer 0.5%, 72 K of 10.28 M. So the RTL
already overlaps three quarters of the vector work and all of the LD / ST work. What is exposed is
dependent chains, not throughput, about five a layer:

| chain (a layer) | between the MXU's last and next MM | at 8 lanes (est.) | covered by |
|:--|:--|--:|:--|
| layer start: last down MM -> `_add_norm` (gemma4.py:1680-1690) -> pre-norm + QACT (`_dense`, :1693) -> q MM | ~10 VOPs + QACT on 1152 words | ~1.5 K | the q MM's stream, issued ahead: FIFO 1024 chunks |
| k norm, RoPE, K QST (:1636-1648; 263 cycles on port SW) -> score MM (its K read waits for the QST, count j = 5) | | ~0.6 K + 263 | the q norm / RoPE / QACT chain runs beside the QST |
| softmax: EXP2SUB, RSUM, QACT(P) -> PV MM | 4 x 256 scores at position 14; 4 x 2048 at a full context | ~0.5 K (2 K at 2048) | `ATTN_DEPTH` 3: the next block's score MM |
| PV -> RECIP, emit, QACT(o) -> o MM; o MM -> `_add_norm` -> pre-norm -> QACT -> gate MM | | ~0.6 K; ~1.5 K | the next MM's stream |
| gelu(g) * u -> QACT -> down MM of chunk j | 5 VOPs on 768 + QACT | ~1.0 K | gate / up of chunk j + 1 (13.8 K chunks): hidden except the last chunk |

The exposed part is `max(0, chain - FIFO / C)`: the FIFO buys 1024 / C cycles of prefetch and the
chain shrinks with the lanes. Today: 1.5 K - 1024 = ~0.5 K x ~5 chains x 26 = 65 K (measured 72 K).
The candidates widen the lanes with C, so the chain shrinks as fast as the cover and **the exposed
part stays ~3% of the token in every candidate**:

| | chain at the lanes | FIFO cover 1024 / C | exposed a chain | a token (~130 chains) | share |
|:--|--:|--:|--:|--:|--:|
| today | 1.5 K | 1024 | ~0.5 K | 72 K (measured) | 0.7% |
| P1 (C=2, 8 lanes) | 1.5 K | 512 | ~1.0 K | 130 K | 2.7% |
| P2 (C=4, 16 lanes) | 0.75 K | 256 | ~0.5 K | 65 K | 2.9% |
| P3 (C=8, 32 lanes) | 0.38 K | 128 | ~0.25 K | 32 K (+40 K attention at 2048) | 2.9% |
| P4 (4 x C=1, 8 lanes) | 1.5 K | 1024 | ~0.5 K | 72 K + collective 35 K | 4.7% |

**The FIFO depth that hides the chains is the same in every candidate: chain x C = ~3,000 chunks
(384 KiB)**, 3x today's; 6,000 with a full-context softmax. That is an RTL knob (P3 proposes a 4
MiB FIFO), not a program change: the issue side already streams ahead.

What program ordering can and cannot do:

- **Prefetch the next layer's weights across the norm: already happens.** The q MM is 10
  instructions after the last down MM, inside the window; its stream starts when the MXU's second
  command slot frees. What stops it streaming further is the FIFO, not a dependency; a deeper
  prefetch needs the FIFO or a third command slot. The program-side help is **fewer, longer MMs**:
  q, k and v as one 1536-row MM (contiguous weights; three result views) saves two command
  turnarounds a layer, 2 x 435 x 26 = 23 K cycles (0.2% today, ~2% at P3), bit-exact.
- **Shorter chains**: `_add_norm` + the pre-norm are two RMSNorms back to back (RSSQ, RECIP, mul,
  add, RSSQ, ..., QACT). A fused add-norm-norm-quantize composite would halve the chain (~1% a
  candidate); the fusion peepholes (docs/compiler.md) fire on unnamed temporaries only, so the
  kernel must be written for it. Bit-exact if the fp32 order is kept.
- **The MLP's last chunk** has no next gate / up to hide under and the post-norm follows it;
  nothing can move before it (q needs y). Folding the F chunks (section 4) lengthens this tail
  (3x at 2304: 2 K a layer at 8 lanes, 0.5 K at 32), which the row saving outweighs under any
  gating cost above 0.3 cycles a row.
- **The K / V append under the next MM**: K is appended before the q chain, so the QST's 263
  cycles already run beside it; at 32 lanes the q chain is ~150 cycles, so ~110 x 26 = 3 K (0.3%)
  become exposed at P3. Scoring the token's own key by an RDOT in TMEM and reading the cache only
  to the previous position would hide it, but changes the summation order: port SW's widening
  (each proposal's section 9) is the fix.
- **The logits ST under the head's MM**: the RTL does it in the host path; absent in the card loop,
  as is `fill_logits`.
- **The embedding gathers under layer 0**: the gathers are MXU commands, so layer 0's q stream
  cannot start under them. A DMA gather (LD of the 1,584-byte row + a VPU dequantization) needs an
  int8-to-fp32 VOP that does not exist (RTL); issuing the q MM before the last gathers saves ~1 K
  cycles. The 14 K (1.3% at P3) is the per-MM start cost; fewer, longer gathers (the one-hot
  operand at M = 4 gives 32 MMs of 9 rows; a [36, 128] view of the row's blocks at 4 MMs would need
  the gather's transpose rewritten) are a second-tier item.

Net: the program-side overlap items are worth 1-3% a candidate; the card loop is the one that
matters (section 2), and the FIFO depth is the RTL's.

## 4. The row-gating loss as a software / layout problem

### 4.1 Per row, not per byte

The layers lose 24.3% of their chunk-cycles and the LM head 23.9% (proposal-1-wide-port.md 5),
and the head's rows are all KB 9: the loss is per streamed row. Rows the program streams (the
proposals count down as 1,152 rows a layer; it is 9 MMs of 1,152 rows of KB 6 at `MLP_CHUNK` 768,
gemma4.py:669, kernels/mlp.py:33-76):

| a layer | rows | KB | chunks |
|:--|--:|--:|--:|
| q / k / v | 1024 / 256 / 256 | 9 | 13,824 |
| o | 1152 | 8 | 9,216 |
| gate, up | 2 x 6912 | 9 | 124,416 |
| down (9 F chunks of 768) | 9 x 1152 = 10,368 | 6 | 62,208 |
| a layer / 26 layers | 26,880 / 698,880 | | 209,664 / 5.45 M |
| LM head (32 MMs of 8,192) | 262,144 | 9 | 2.36 M |
| a token | **961,024** | | 7.81 M |

"MXU not consuming" 18.2% + "computing, not streaming" 3.9% = 2.27 M cycles over 961 K rows =
**2.36 cycles a row** (the head alone 0.74 M / 262 K = 2.8; the layers (7.2 M - 5.45 M - 72 K) /
699 K = 2.4). Which mechanism carries it is unmeasured (P1's milestone 0): the result-FIFO credit
(`rows_live + rows_p < RF`, RF 32, otpu_mxu.sv:115-118, :338), the row-complete gate (`row_in`,
:331-336), the scale reader's own port-A request for a row's Sw ("a cycle without a chunk",
:30-32), the command turnaround. Each is paid once per streamed row, or per row group where C lanes
pop in lock step, so **software's lever is the same whatever the cause: fewer, longer rows**.

### 4.2 The cycles-a-row budget per candidate

| | a KB 9 row streams in | a 1-cycle bubble costs | budget for <= 5% | the RTL fix's target |
|:--|--:|--:|--:|:--|
| today, P4 (C=1 a slice) | 9 cycles | 11% | 0.45 a row | P4: 1 a row (RF 64, Sw prefetched) |
| P1 (C=2) | 5 | 20% | 0.25 a row | 0.5-1.0 a row (RF 128, two-row lead, Sw a row ahead) |
| P2 (C=4, 4 rows a group) | 9 a group | 11% a group | 0.45 a group | 0.1-0.57 M a token |
| P3 (C=8, 8 rows a group) | 9 a group | 11% a group | 0.45 a group | 0.5-1 a group |

Left at 2.36 a row the loss is 2.27 M cycles in P1 (37% of a 6.4 M token) and, if it stayed per
row rather than per group, 2.27 M at P3 (a 3.4 M token: 59 tok/s instead of 165). The lock-step
grouping of P2 / P3 is the first fix, the RTL's credit / lead / prefetch changes the second, the
layouts below the third and the only one without RTL.

### 4.3 Layout tricks without new hardware

**(a) Fold the LM head: 4 vocabulary rows per streamed row.** Store the head as [65,536, 36 blocks],
row n holding vocabulary rows n, n + 65,536, n + 131,072, n + 196,608 in its four 9-block parts;
the stationary operand is M = 4 rows, row j = xs in blocks 9j .. 9j + 8 and zero elsewhere (36 of
the ACT RAM's 128 blocks). One MM of N = 65,536 gives out[j, n] = logit(j x 65,536 + n), and the
[4, chunk] result read row-major has the vocabulary id as its ARGMAX index (Greedy's `base=col0`,
generate.py:246, becomes the chunk's first n plus j x 65,536). M = 4 is free on the stream (a chunk
is reused across MCOLS = 4 columns), so the head costs the same 2.36 M chunks in 65,536 rows of
36 instead of 262,144 of 9: at C = 8 a row streams in 4.5 cycles and a 1-cycle bubble is 22%, not
50%; E = 4 drain lanes against 4 results every 4.5 cycles keep up (otpu_mxu.sv:107). Needs: the
folded layout and scale rows in the image builder (qwen3.py `_Bump`, gemma4's Image), the 4-row
stationary in `_lm_head`, the embedding gather reading part t mod 4 of row t div 4 (gather_row's
row slice becomes a part slice; the gather scales are per block already), the id mapping in
`Greedy` / `Sampler`. **Numerics**: an int8x row has one fp32 Sw and a power-of-two shift a block
(docs/mx_mxu.md 3.3), so four rows under one Sw lose at most one bit on the rows below the group's
amax, a quarter bit on average, inside the int8 head's KL 0.007 (docs/quant.md) but **not
bit-exact to today's image**; exactness would need a per-column Sw in the epilogue (RTL).

**(b) The MLP's F chunk at decode: 768 -> 2304.** `_mlp_chunk` (gemma4.py:675) sizes it for 4
prefill rows' gate / up tiles; at R = 1 a 2304 chunk is 2 x 2 x [1, 2304] = 9.2 K of 64 K words.
Down becomes 3 MMs of 1,152 rows of KB 18: 3,456 rows a layer instead of 10,368, 180 K fewer a
token. One QACT range a chunk (docs/quant.md "Activations") means coarser int8 for a chunk's small
columns: `down_chunk` (mlp.py:27) feeds the emulation, so the emulation follows; prefill keeps 768
and stays bit-exact. Whole-F (1 chunk, 1,152 rows) saves 60 K rows more but loses the gelu overlap
(~7 K cycles a layer exposed at 8 lanes; 1.7 K at 32 = 44 K a token against 60 K x 0.75 / 8): no.

**(c) Fold the layer matrices x2** (q / k / v / o / gate / up at KB 16-18, M = 2): 215 K fewer rows,
but the result lands as [2, N / 2] at an odd row stride (docs/compiler.md "Bank-aware strides"),
so RoPE's pair structure on k, the heads' split of q, the norms' gamma and the gelu product must
be written for the folded view, and the Sw sharing touches the layers. Second tier.

**(d) K-major tiles** (docs/dataflow.md 4.11: NT rows a tile, the stationary block read once a
tile, the column end accumulating NT rows) amortize every per-row cost over NT x KB chunks and are
bit-exact, but are an MXU change: the proper fix at C = 8; (a)-(c) are its software approximation.

**(e) The result FIFO**: at M = 1 a row yields a result every ceil(9 / C) cycles against E = 4 a
cycle, so the FIFO cannot fill unless the TMEM grant is withheld (the drain lost 2.1% to
arbitration; VPU-first puts the VPU ahead of it, docs/litedram.md). If milestone 0 names the
credit, interleave Greedy's ARGMAX of chunk c with the MM of chunk c + 2 (two chunk tiles live, 32
KiB of TMEM). **Interleaving MMs of different matrices** changes no per-row cost (one in-order MXU).

### 4.4 Worth (section 7's model)

Rows 961 K -> 585 K with (a) + (b) (370 K with (c)). At the trace's 2.36 a row: P1 6.35 -> 5.46 M
cycles (+16%), P2 / P3 / P4 +9%. At the proposals' fixed 0.75 a row or group: +3% to +6%. The fold
is insurance on the RTL milestone with a small certain return.

## 5. Speculative decoding, and 4-bit weights

**The MXU side is free.** An int8 MM reuses each chunk across MCOLS = 4 stationary rows, so a
verify of 2-4 rows streams the weights once (docs/mtp.md 1's "three rows do not" is the fp4 PAIR
rule 2M <= MCOLS; int8 pairs up to 4). What a row costs is what does not share weights: the VOPs
(151 K cycles a row at 8 lanes, 38 K at 32), QACT (84 K / 21 K), the 32 gather MMs (~14 K), the
row's KV reads and K / V QSTs, the head's per-row ARGMAX (hidden). Estimated c_2 (a 2-row verify
over a step; docs/mtp.md 1 measured 1.09-1.21 on the card's models at position 544): today ~1.04,
P1 1.05, P2 1.07, P3 and P4 1.10 (**assumed**; measure with `tools/perf_qwen.py --model gemma3
--rows 2 --logits all`). c_4 ~1.1 today, ~1.25 at P3. Speedup = (1 + a) / (c_2 + c_draft) at k = 1
(docs/mtp.md 7):

| drafter for Gemma 3 1B | c_draft | acceptance a | speedup today / P3 |
|:--|--:|:--|:--|
| n-gram prompt lookup, g = 3 > 2 (docs/mtp.md 6.2): VOPs over the history, no model | 0 | chat checkpoints 0.47-0.60 on the 22-36% of tokens with a draft: 1.03-1.09x; summaries to 1.18x; Gemma 4 E2B pre-trained 1.23x (copying); **gemma-3-1b-it unmeasured** | 1.03-1.10x / 1.03-1.09x; section 7 assumes 1.08x (a = 0.5 on a third of the tokens) |
| a draft model | no smaller Gemma 3 exists (no 270M); a 4-bit copy of the 1B costs 0.5 of a step a draft | 0.5 | (1 + a) / 1.55 < 1 for a < 0.55: never pays at batch 1 |
| an MTP / EAGLE head trained for Gemma 3 (one layer + a 32K-id fp4 head, as Qwen3.5's) | ~0.05 | 0.7 (Qwen3.5's 32K head, docs/mtp.md 7.1) | ~1.5x / 1.45x; needs training: out of scope |

What mtp.py provides: the k = 1 loop (a 2-row verify at a run-time position, accept, draft, commit,
`MTPDecoder.loop_card`, speculative sampling; docs/mtp.md 9-11) for Qwen3.5's head. What Gemma 3
lacks: the n-gram drafter on the card ("n-gram waits", mtp.md 9: a VOP match + ARGMAX + RLD + an
LD), the verify over `gemma4_step`'s rows form (RunRows, gemma4.py:1891-1895; row 1's ring slot
is pos + 1 and a rejected row's K / V is overwritten by the next token, as the positional caches
in mtp.py's docstring), and nothing to roll back. k = 2-3 pays only with a trained drafter (+2-3%
with n-gram: the second draft's acceptance ~0.25 against c_3 - c_2 ~0.05-0.1).

**4-bit weights.** fp4x6 is 4.25 bits a weight against int8's 8.06-8.25 (docs/quant.md "The MX
formats"): layers 698 MB -> 360 MB with the int8 head (302 MB): 662 MB, 0.66x; the head in fp4 too:
516 MB, 0.52x. PAIR streams a 4-bit chunk in one advance at M <= MCOLS / 2 (docs/quant.md
"Hardware"), so decode runs at the stream rate in every candidate. Caveats: (1) a KB 9 row is 5
chunks in fp4, so every per-row cost's share doubles (section 4.2's budgets halve); (2) accuracy on
small models: Qwen3-0.6B KL 0.19 against int8's 0.007 (docs/quant.md "Accuracy"), Gemma 4 E2B fp4
layers ppl 5.63 against int8 4.35 with top-1 agreement 0.77 (docs/gemma4.md "Accuracy"), and the
-it checkpoints' mixes all fail their bars; Gemma 3 1B is smaller still. Attention in int8 with the
MLP in fp4 (0.78x bytes, ~1.3x) recovered a third of E2B's loss. Section 7's fp4 column is a bound,
not a recommendation; `tools/formats_scan.py` on gemma-3-1b-it decides.

## 6. Batching as the throughput lever

An int8 MM takes up to MCOLS = 4 rows a weight pass for free; past that, ACT_ROWS replay consumes
each streamed row G = ceil(M / 4) times from the FIFO (otpu_mxu.sv:34-40), so 8 rows cost 2x the
MXU time and the aggregate does not grow past 4 (docs/serve.md "Measured on the card": 8 rows =
two runs of 4). otpu-serve's batched worker (KV slots, span programs, a sink per row) is Qwen3's;
Gemma runs in turns (one sequence an image). Per extra row: section 5's per-row cost (0.28 M
cycles today, P1 and P4 at 8 lanes; 0.15 M at P2; 0.08 M at P3), plus KV reads at long contexts.

| rows a pass | today | P1 | P2 | P3 | P4 |
|:--|--:|--:|--:|--:|--:|
| 1 (card loop, gating fixed) | 23 tok/s | 42 | 89 | 176 | 88 |
| 2 | 45 | 79 | 168 | 330 | 156 |
| 4 | 84 | 142 | 298 | 583 | 256 |

R x 200 MHz / (T_1 + (R - 1) x per-row), T_1 from section 7. Measured reference: Qwen3 int8 on the
card, 1 row 24.1 -> 4 rows 79.1 tok/s (3.3x; docs/serve.md) against this model's 3.6x. Past 4 rows
the lever is MCOLS 8 (docs/mx_mxu.md), RTL. Software: `batch.py`'s slots and sinks on the gemma4
model code (its one-sequence and `cfg.S == 1` assumptions, `Image(batch=b)` KV slots and rings).

## 7. The table: candidate x software, decode tok/s at 200 MHz

Cycle model, one token: `T = 7.85 M / C + rows / g x b + chains + fixed + collective + host`.
7.85 M the measured stream (chunk-cycles with KV and tables); rows 961 K (585 K folded); g rows a
lock-step group (1 today, P1, P4; C for P2, P3); b 2.36 cycles (measured, unfixed) or 0.75 (the
proposals' fix, assumed mid of 0.5-1); chains section 3; fixed 30 K (QST 13.7 K + gathers 14 K +
head tail; modelled); collective P4 35 K (proposal 4); host 0.3 ms = 60 K + logits ST (33 K /
lanes) + fill (32 K / lanes), host loop only. Check: today's host loop at 133.33 MHz gives
10.35 M against the measured 10.28 M. The serialized ISA models (README) are 8.28 / 4.36 / 2.19 /
1.10 / 2.40 M.

| candidate, gating | hardware, host loop | + card loop (overlap) | + head x4, chunk 2304 | + n-gram spec (x1.08, assumed) | + fp4 layers (bound) |
|:--|--:|--:|--:|--:|--:|
| today, unfixed (measured 2.36 / row) | 10.35 M, 19.3 | 10.22 M, 19.6 | 9.33 M, 21.4 | 23 | 32 |
| today, fixed 0.75 | 8.80 M, 22.7 | 8.67 M, 23.1 | 8.39 M, 23.8 | 26 | 38 |
| P1, unfixed | 6.48 M, 30.9 | 6.35 M, 31.5 | 5.46 M, 36.6 | 40 | 52 |
| P1, fixed | 4.93 M, 40.6 | 4.81 M, 41.6 | 4.52 M, 44.2 | 48 | 68 |
| P2, unfixed (a group) | 2.72 M, 73.6 | 2.62 M, 76.2 | 2.40 M, 83.2 | 90 | 125 |
| P2, fixed | 2.33 M, 85.8 | 2.24 M, 89.4 | 2.17 M, 92.3 | 100 | 144 |
| P3, unfixed (a group) | 1.40 M, 142.6 | 1.33 M, 150.7 | 1.22 M, 164.5 | 178 | 245 |
| P3, fixed | 1.21 M, 165.4 | 1.13 M, 176.5 | 1.10 M, 182.1 | 197 | 283 |
| P4, unfixed | 2.79 M, 71.6 | 2.67 M, 75.0 | 2.44 M, 81.8 | 88 | 122 |
| P4, fixed | 2.40 M, 83.2 | 2.28 M, 87.7 | 2.21 M, 90.5 | 98 | 140 |

Status: hardware = the proposals' RTL expectations re-derived on one model (proposal 1: 41 / 38;
2: 69-79; 3: 148-175; 4: 72-89: within their ranges); card loop = a measured mechanism, modelled
cycles; folding = modelled on the per-row loss, numerics unmeasured; spec = assumed acceptance
(`tools/mtp_accept.py` on the -it checkpoint is the first measurement); fp4 = modelled bytes,
accuracy unmeasured on Gemma 3. If P3's unfixed loss were per row rather than per group, its first
column is 3.4 M = 59 tok/s and the folds are worth 1.37x there: where software matters most.

## 8. Implementation plan, by payoff / effort

| # | item | files | test (bit-exactness to the per-position programs is the standard) | Lens check |
|---|:--|:--|:--|:--|
| 1 | Record the F2 numbers on the card loop: `lens record model --generate` (the bucket's loop program instead of the resident step) | opentpu/lens (record); tools/decode_profile.py `--card-loop` exists | tests/test_autodecode.py (tiny Gemma 4, greedy / sampled, exists) | ST / fill gone from the per-class table; host path 0 |
| 2 | `tools/mtp_accept.py --model gemma3`: n-gram acceptance of gemma-3-1b-it on the nine prompts | tools/mtp_accept.py (Gemma 3's chat template) | offline, HF | decides item 6 |
| 3 | LM head x4 fold: image layout, `_lm_head`'s 4-row stationary, gather of a part, Greedy / Sampler ids, the folded quantizer in the emulation | qwen3.py (`_lm_head`, `_Bump`), gemma4.py (Image, `_gathered`), kernels/gather.py, llm/generate.py, opentpu/quant.py | tests/test_gemma3.py: device = `emulated_logits` bit for bit; HF greedy as docs/gemma3.md; `tools/formats_scan.py` dKL of the folded head vs today's (bar: inside the int8 head's noise) | head rows 65,536; head cycles 2.36 M / C x (1 + b / 4.5); "MXU not consuming" on the head <= 1/4 of the layers' per chunk |
| 4 | MLP chunk 2304 at R = 1 (`_mlp_chunk` takes the rows; `down_chunk` for the emulation) | gemma4.py:675, kernels/mlp.py:8-31, opentpu/quant.py (`fake_act` ranges) | device = emulation; prefill's chunk unchanged (bit-exact prefill) | down rows 3 x 1,152 a layer; the last chunk's tail <= 2 K a layer |
| 5 | q / k / v as one 1,536-row MM (contiguous weights; three result views) | gemma4.py `_attention`, the Image's layer block | bit-exact (the same per-row arithmetic) | 2 fewer MM commands a layer |
| 6 | n-gram speculative decoding for Gemma 3: the drafter on the card, the 2-row verify at a run-time position, accept / commit in the generate loop | opentpu/llm/mtp.py (a generic verify over `gemma4_step` rows), llm/generate.py, a history-match kernel; docs/mtp.md 8 items 2 and 5 | tests/test_autodecode.py style: tokens equal the plain card loop's, greedy and sampled (`reference_pick`), across a bucket boundary and at a stop id, tiny Gemma 3 | verify = step x c_2; iterations and acceptance in the run's stats |
| 7 | Fused add-norm-norm-quantize chain (a peephole on `_add_norm` + the pre-norm, or a composite) | opentpu/compiler.py, kernels/lib.py `rmsnorm`, gemma4.py:1680-1693 | bit-exact if the fp32 order is kept | the layer-start gap before the q MM halved |
| 8 | Batched slots for Gemma (otpu-serve's `BatchWorker` on the gemma4 model code) | opentpu/llm/batch.py, gemma4.py, serve/batching.py | replies alone = replies together, bit for bit (tests/test_serve.py pattern) | 4 rows a step at the stream + 3 x per-row |
| 9 | Layer matrices x2 fold (after 3 and 4 measure the gating on the F2) | gemma4.py, kernels/mlp.py, lib.py (rope, rmsnorm on folded views), the image | emulation = device; HF greedy; dKL bar | rows 370 K a token |

Items 1-2 are days; 3-5 are 1-2 weeks each and need no RTL; 6 is 3-4 weeks; 7-9 follow the F2
measurements. The decision they inform is section 4.2's: if P1's milestone 0 shows the per-row loss
is the credit or the Sw request (fixable in RTL to 0.5 a row), the folds are a 3% item; if it is
the row turnaround (per row by construction), they are the 10-16% item and K-major tiles
(dataflow.md 4.11) are the v5 MXU.
