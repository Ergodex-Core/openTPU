# Profiling findings: memory stalls and bottlenecks in Gemma 3 1B decode

What the Lens traces of today's design say about where a decode token loses cycles, and the
improvement opportunities they point at, ranked. All numbers are from the RTL board model
(one slice, D 128, MCOLS 4, 8 lanes, one 128-byte chunk a cycle) tracing one resident-decode
token of gemma-3-1b-it in int8 at position 14, with the memory model at three settings
(`lens record model --model gemma3 --axi [--stall N]`, `OTPU_LAT=30` for the HBM-like runs;
`build/lens/v4/m0-gemma3*.otpuprof`, with Milestone 0's blocked-cycle split,
[m0-gating-split.md](m0-gating-split.md)); the long-context token is in section 6. The
profiles are in the user's `lens-v4/all-profiles.html` and `build/lens/v4/v4-candidates.otpuprof`.

## 1. The three memory settings

| Memory model | Cycles | Of the bound (7.85 M) | Port B streaming | Row incomplete | Scales late | Credits | MXU turnarounds | VPU / quant waits | FIFO starved |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DDR3 default, 20% random stalls (the card) | 10.28 M | 76.4% | 76.4% | 17.7% | 0.5% | 0.0% | 3.9% | 0.7% | 0.6% |
| 5% stalls (HBM jitter) | 8.65 M | 90.7% | 90.8% | 4.0% | 1.0% | 0.1% | 2.8% | 1.1% | 0.1% |
| no stalls (HBM-like) | 8.22 M | 95.5% | 95.5% | 0.6% | 0.8% | 0.1% | 1.5% | 1.3% | 0.2% |

Percentages are of the token's cycles, as port B sees them: streaming, or idle for the
named reason.

**Finding 1: on the card, the only large loss is the memory's rate.** With the DDR3 model
the chunk FIFO holds 7 chunks on average while rows are late (under one 9-chunk row) during
94% of the cycles: the memory delivers 0.76 chunks a cycle, the MXU would take 1.0, so every
row completes late. 97% of the 2.21 M blocked cycles are "row's chunks not all in"; credits
are 0.1%. No change in the MXU recovers this; a faster or smoother memory does. On the card
the lever is the controller's efficiency (`docs/litedram.md`: 99% busy, 94% column commands,
2.7% refresh), on F2 the HBM makes it vanish. The 18.2% "MXU not consuming" in the original
trace was this, not row gating.

**Finding 2: the loss follows the stall rate linearly** (20%: 17.7%, 5%: 4.0%, 0%: 0.6%
row-incomplete), because the port's rate equals the MXU's consumption: there is no slack.
The FIFO cannot fill ahead when the stream runs at exactly one chunk a cycle, so every
withheld beat is a lost MXU cycle. An HBM adapter must therefore deliver a steady 1 (or C)
chunks a cycle at the port, which means enough requests in flight across the pseudo-channels
to absorb per-channel jitter (refresh, bank conflicts): the 5%-stall run says a 5% jitter
costs 5% of the token.

## 2. What is left at the roofline (the HBM-like run), by cause

8.22 M cycles against 7.85 M of chunks: 371 k cycles, 4.5%.

| Loss | Cycles | Share | What it is |
|:--|--:|--:|:--|
| MXU computing, not streaming | 161.7 k | 2.0% | 131 gaps between matmuls, median 1,242 cycles, every one ended by a QACT (section 3) |
| waiting on quantizer / VPU work | 107 k | 1.3% | the same chains, the part attributed to the units in flight |
| scales not all in | 68.6 k | 0.8% | port A's scale lines behind their chunks at matmul starts, all in the layers |
| row incomplete | 96.8 k (part hidden) | 0.6% | the DDR3 model's refresh / row-miss hiccups at a 1:1 rate |
| FIFO starved (latency) | 38.6 k | 0.2% | matmul starts: 30-cycle latency not covered |
| credits (result FIFO) | 5.8 k | 0.07% | negligible |
| prologue + epilogue | 5.8 k | 0.07% | the embedding gather before the first stream, the logits after the last |

The MXU drain lost 226 k cycles (2.7%) to TMEM bank arbitration, but port B did not see it
(0.0%): the result FIFO absorbed it. The LM head (the last 2.42 M cycles, 29% of the token)
streams at 98%; the layers at 94.5%.

## 3. The dependent chains between matmuls: the bottleneck that scales with C

Every gap between two matmuls is ended by a QACT, the quantization of the next matmul's
activations: 131 gaps = 5 a layer plus one at the head, 1,242 cycles each, 6,150 cycles a
layer, 160 k a token. By source line (`gemma4.py`, `kernels/attention.py`):

| Chain | Cycles a token | What runs in the gap |
|:--|--:|:--|
| input norm, quantize (`_attention` 1634) | 45.6 k | W_o drain of the previous layer, the residual add, RMSNorm, QACT |
| MLP pre-norm, quantize (`_dense` 1695) | 40.9 k | W_o drain, residual add + post-attention norm, pre-MLP norm, QACT |
| query quantize (`attention.py` 117) | 32.5 k | q projection drain, q_norm, RoPE, QACT |
| softmax probabilities, quantize (`attention.py` 195) | 29.1 k | the QK^T drain, max, exp2, sum, QACT |
| attention output, quantize (`_attention` 1675) | 11.9 k | the PV drain, the division by the row sum, QACT |
| final norm (`_lm_head`) | 1.7 k | the last layer's drain, final norm |

These are not memory stalls: the port is idle because the next matmul's weight stream is
not issued until its activations are quantized. The stream of a matmul depends on data it
does not read (the weights are known at program time) only through the instruction's release.
At one chunk a cycle this is 2.0% + 1.3% = 3.3% of the token; it is a fixed cost in cycles,
so at C chunks a cycle its share is C times larger: **13% at C=4 (P2, P4), 26% at C=8 (P3)**
unless the chains shrink or the stream decouples. This is the largest single item for v4
after the port width itself, and none of the hardware proposals' MXU changes touch it.

Opportunities, in order of payoff:

1. **Decouple the weight stream from the activation dependency (hardware + ISA).** Let the
   MXU prefetch the next matmul's chunks into the FIFO as soon as the previous stream ends,
   releasing the compute side when the QACT lands: an MM whose stream side has no data
   dependency (a flag, or a PREFETCH instruction naming the weight tensor). The FIFO (1,024
   chunks = 1,024 cycles at C=1) covers the 1,242-cycle chains almost entirely; at C=4 the
   same chain is 4,968 chunks: the FIFO must grow to about 4 K chunks (512 KiB, URAM) for
   full cover, as the software study's "3,000 chunks" estimated. Gain: up to 3.3% today,
   13% at C=4, 26% at C=8. The sequencer already prefetches the next layer's weights into
   its two command slots; this extends it across the activation dependency.
2. **Shorten the chains (software).** Fuse the residual add, the post-attention norm and
   the pre-MLP norm into one VOP pass (today three passes over the row of 1,152 fp32 in 8
   lanes: 144 cycles each), quantize in the same pass where the quantizer allows; fold the
   attention output's division into the PV epilogue; issue the q projection first and the
   K / V projections after it so the q chain runs under the K / V stream. Each chain loses
   200-400 of its 1,242 cycles: about 1% a token at C=1, 4% at C=4.
3. **Widen the VPU and quantizer with C** (P2, P3 do; P4 does not): the chains' VOP and
   QACT work is 8-lane work (VOP busy 418 k, QACT 101 k cycles a token); at 16 lanes the
   chains halve. P4 keeps 8 lanes per slice and repeats the norms on every slice: its chains
   stay 1,242 cycles against a 2.06 M-cycle token (6.3% x 2).

## 4. Memory-side opportunities

4. **Port A scale lines (0.8% today, 68.6 k cycles, all in the layers).** The weight
   scale lines (32 B each, one per chunk) arrive on port A behind their chunks at matmul
   starts. Prefetch the scale lines of a row ahead of its chunks (P1's item c): the loss
   goes to the FIFO-starved floor. At C lanes port A must deliver C scale entries a cycle
   or this term grows C times (P2's section 9).
5. **HBM request depth.** From Finding 2: the adapter needs enough outstanding requests
   per pseudo-channel that a channel's refresh or bank conflict never leaves the port short
   of a beat. With 32 channels at 450 MHz feeding a 200 MHz port of C x 128 B, each channel
   carries C/32 of the bytes; a reorder / slot-reserved FIFO of a few hundred chunks per
   lane (P1's reorder buffer, P2's slot-reserved FIFO) absorbs jitter of tens of cycles. The
   5%-stall run is the test: a 5% random-withhold model must cost well under 5%.
6. **The LM head is 29% of the token** (302 MB of int8 for 262,144 rows). It streams at
   98% today, so its only levers are bytes: a 4-bit head halves it (-14% of the token, at
   the accuracy cost docs/quant.md reports: Gemma 4's fp4 head takes a third choice at a near
   tie), and the software study's row folding (4 vocab rows per streamed row) matters only
   when per-row costs grow with C. Vocabulary pruning for a chat deployment (the 262 k
   vocabulary's rare tokens) is the software lever that costs no accuracy on the kept tokens.
7. **TMEM arbitration on the MXU drain (2.7%, hidden today).** At C lanes the drain writes
   C times as many rows a cycle into TMEM while the VPU and quantizer run the chains: the
   result FIFO will fill (credits appear, as qwen-tiny's KB-2 rows already show: 27% credits).
   Give the drain its own TMEM write port or priority over the VPU (P2 / P3's TMEM section).

## 5. What is not a problem

- Result-FIFO credits: 0.1-3% of the blocked cycles at every setting. The proposals' RF
  widenings are insurance, not performance.
- DRAM backpressure ("DRAM not ready"): 0.0% in every run; the port never pushed back.
- DMA: 0.9% in flight; the logits store (1 MiB, 35 k cycles) and the K / V appends (QST
  33.6 k) are queued behind the matmuls, not exposed.
- The embedding gather (32 one-hot matmuls) and the final norm: 5.8 k cycles, 0.07%.
- Dispatch: "nothing in flight" is 0.0%; the 16-slot window never ran dry.

## 6. Long context

(Position 1,208 after a 1,208-token prompt; the measurement is running: the sliding layers'
K / V rings and the global layers' KV reads are the only per-position memory traffic, and the
flash-attention entries over 5 blocks of 256 at the global layers are the only per-position
compute. This section is filled in when the two runs, at 0% and 20% stalls, complete.)

## 7. Ranked summary

| # | Opportunity | Where | Gain at C=1 | Gain at C=4 | Effort |
|--:|:--|:--|--:|--:|:--|
| 1 | Decoupled weight prefetch across the activation dependency, FIFO 4 K chunks | ISA + MXU issue side + FIFO | 3% | 13% | medium (RTL + ISA sim + compiler flag) |
| 2 | HBM adapter with deep per-channel request queues (jitter to < 1%) | memory adapter | 0% (card: 24% but DDR3-bound) | the difference between 90% and 95% | the F2 adapter's design |
| 3 | Fused norm / add / quantize passes, q before K / V | kernels (software) | 1% | 4% | low |
| 4 | Scale-line prefetch on port A | MXU issue side | 0.8% | 2-3% | low |
| 5 | VPU / quantizer at C x 8 lanes | P2 / P3 (P4: no) | - | 6% (P4's shortfall) | in the proposals |
| 6 | 4-bit LM head (accuracy caveat) | image format | 14% | 14% | none (a flag) |
| 7 | MXU drain TMEM port / priority | TMEM | 0% (hidden) | credits appear | in P2 / P3 |

The card's own 24% (the DDR3 rate) is outside this list: it is the memory, and the F2
target replaces it.
