# Ternary weights: the t2 format, MM QUAD and Ternary-Bonsai 2

Ternary-Bonsai 2 (prism-ml; Qwen3.5-2B, 4B, 9B and Qwen3.8-27B with ternary weights) stores
every projection as {-1, 0, +1} with an fp16 scale per group of 128, in a rotated input basis
(a signed 512-wide Walsh-Hadamard transform folded into the weights). Decode on openTPU is
DRAM-bound: its time per token is the weight bytes it streams. In int4 a ternary weight costs
4 bits; t2 stores it in 2, and MM QUAD lets one-row decode MMs take a whole 2-bit chunk per
cycle. This note is the format's specification (ISA, simulator, quantizer, RTL) and how the
models use it.

## 1. The format (MM WF 1 with rs bit 0)

* **Elements.** 2-bit codes `c`, the integer `w = c - 1` in {-1, 0, 1, 2}; four per byte,
  element `i` of a block in bits `2 (i mod 4) +: 2` of byte `i / 4`. Block `k` of row `n` is
  the `D/4` bytes at `sa + n*rs' + k*D/4` (rs' = rs with bits 1:0 cleared): a D-byte chunk holds
  four blocks; a row's last chunk is partly used when KB is not a multiple of 4. This is the
  byte layout of llama.cpp's Q2_0 / PQ2_0 `qs` (PQ2_0's group is 128 = D): a PQ2_0 tensor's
  code bytes are its t2 rows as they are.
* **Scale rows.** `Sw` (fp32), three reserved words, then a word per block, int4's word: `dw` =
  bits 7:0, the multiplier `m` of sub-block `b` (D/4 elements) bits `8 + 6b +: 6`. A row is
  `16 + 16 ceil(KB/4)` bytes; `ssa` and `srs` are multiples of 16, so a chunk's four words are
  one aligned 16-byte read (never across a 32-byte port-A line). A weight is
  `w * m * 2^-dw * Sw[n]`.
* **Encoding.** The MM flags byte and WF are full, so t2 is WF = 1 (int4) with bit 0 of the row
  stride `rs` (w5) set; `rs` is a multiple of D >= 4, so its bits 1:0 are free. The MXU uses
  `rs & ~3` as the stride. In the assembler `wf=isa.WT2` encodes it; `isa.mm_format(ins)`
  decodes (format, PAIR, QUAD).
* **The shift window** W is 11, as int4's (a t2 group's `|c| <= 32 * 2 * 127 < 2^13`).
* **The capability.** `Config.T2`, CAPS bit28 (opentpu/host/regs.py `CAP_T2`); the arch hash
  includes `T2` only when it is on. The ISA simulator refuses t2, QUAD and DUP4 without it.

## 2. Column reuse x4: MM QUAD and QACT DUP4

A 4-bit MM takes one block a cycle, PAIR two (columns M..2M-1 take the odd blocks). A t2 chunk
holds four: **QUAD** (rs bit 1, with t2 and PAIR; `4 M <= MCOLS`; `k0` a multiple of 4) lets
columns `c M + j` (c = 0..3) take the blocks `k = c mod 4`, so one row (M = 1, MCOLS = 4) takes
the whole chunk, 512 weights, each cycle. Output row `j` gets block `k` from ACT row
`j + (k mod 4) M`:

```
r = QUAD ? j + (k mod 4) M : PAIR and k odd ? j + M : j
```

**QACT DUP4** (flag bit4, `F_DUP4`; not with DUP; `4 rows <= MCOLS`) writes the rows to ACT rows
`r`, `r + rows`, `r + 2 rows` and `r + 3 rows` in the same cycles. As with PAIR, the result is
the same integer per output: QUAD changes no bit (tests/test_t2.py).

Without QUAD, t2 at PAIR takes half a chunk a cycle: the bytes halve against int4 but the MXU
consumes 64 B a cycle (12.8 GB/s at 200 MHz). QUAD consumes 128 B a cycle, the port's width.

## 3. Quantizer (opentpu/quant.py "t2")

`quantize_mxu(W, "t2")` uses int4's search (`_x4`) on the element grid {-1, 0, 1}: each
sub-block's ideal scale is its amax, quantized to `Sw 2^-dw m`. On a ternary matrix (each
block `q * d`) the codes come back exactly and each block's scale within a 6-bit multiplier:
the relative error of a block scale is at most 1/64 (measured on Bonsai-2-2B's ffn_up: 0.50%
Frobenius, the same as int4 on these weights at half the bytes).

## 4. Ternary-Bonsai 2 on openTPU (opentpu/llm/bonsai2.py)

A model directory holds the GGUF (PQ2_0; or the `-MTP-Q8_0` one, whose `nextn` layer maps to
`mtp.*`) beside Qwen/Qwen3.5-<size>'s config.json and tokenizer. `bonsai2.Weights` reads it
(opentpu/llm/gguf_file.py: the prism types PQ2_0 = 142 and PTQ1_0 = 143 included), undoes
llama.cpp's conversion (norm gains + 1, A_log, tiled value heads) and either
* unfolds the rotation (`W = W_f H diag(s)`), so any image quantizes the model's own weights
  (int4 / int8: the baseline), or
* keeps the latent weights (`ternary`), for a t2 image that applies the rotation on the device:
  `R x = H (s * x)`, H the normalized Sylvester Hadamard matrix of 512, block-diagonal.

### The rotation on the device

The sign vectors fold exactly: into the RMSNorm gains (the inputs of qkv, z, q, k, v, gate,
up), into `up_proj`'s rows (down's input `silu(g) * u`: negating rows of a ternary matrix keeps
it ternary), into `v_proj`'s rows (o_proj's input is linear in V) and into the DeltaNet output
norm's gain expanded per head (out_proj's input). H is one MM of the activation, viewed as
`K/512` rows of 512, by H as a t2 matrix (codes +-1, `Sw = 2^-4.5`): 64 KB a transform.

## 5. Baseline (2026-10-16, Alex's burst200 image agfi-0d06cb48544a02106)

200 MHz core, HBM through 2 channels, 4 GiB. int4 weights (rotation unfolded), int8 tied head,
64-step greedy chat decode, `tools/fpga_inference_check.py --golden numpy`:

| Model | Image | Decode tok/s | argmax vs fp32 reference |
|---|---|---|---|
| Ternary-Bonsai-2-2B | 1.30 GB | 9.06 | 16 / 16 |
| Ternary-Bonsai-2-4B | 2.66 GB | 6.28 | 20 / 20 |
| Ternary-Bonsai-2-9B | 5.45 GB | does not fit 4 GiB | |
| Ternary-Bonsai-2-27B | ~7 GB (t2) | does not fit 4 GiB | |

### Serving (otpu-serve, the card's greedy sampler; tools/f2_bench.py, 128 tokens at most)

On the same image, 2026-10-16. The probe above reads every token's 248K logits back over the
BAR (about 45 ms a 2B token); the server samples on the card. "rotated int4": the latent
ternary weights (exact codes) and the latent ternary head in int4, the rotation on the card
(section 4; runs on any image); its prefill runs rotate too.

| Model | Weights | Decode tok/s (short / long) | Prefill tok/s (274-token prompt) |
|---|---|---|---|
| 2B | int4, int8 head | 16.8 / 16.7 | 4.6 |
| 2B | rotated int4 | 19.3 / 19.3 | 14.3 |
| 4B | int4, int8 head | 8.0 / 8.0 | 3.6 |
| 4B | rotated int4 | 8.6 / 8.6 | 5.0 |
| 4B + MTP | int4, int8 head, --mtp | 10.9 / 6.2 | 3.6 |

One decode step's cost on the ISA simulator (opentpu.profile's nominal unit cycles, the MMs
at their chunk counts; b2-scripts/step_costs.py): the 2B in int4 + int8 head streams 9.69 M
D-byte chunks (the int8 head 3.99 M of them); measured 12.36 M cycles a token, 78% of the
port's rate. In t2 with QUAD and the rotation: 4.26 M chunks (the rotations' H MMs 0.03 M),
5.44 M unit cycles: about 2.3x fewer cycles a token at the same efficiency.
## 6. The RTL (T2 builds)

A top-level parameter `T2` (otpu_top, otpu_board, otpu_asic_top, otpu_slice; default 0) builds
the format into otpu_mxu, otpu_actram and otpu_quant and sets CAPS bit28 (otpu_ctrl).
`rtl_params` and `sim_uarch` pass the target's `T2`, rtlsim builds tb_top with `T2 = 1` for a
`Config(T2=True)`. Without T2 every t2 term is the constant 0 (the selects fold away: int8, int4,
fp4, NVFP4 and PAIR keep their paths); the one change for every build is the stride, `rs & ~3`.

* **Issue.** A t2 row is ceil(KB/4) chunks; a chunk request every fourth advance (PAIR every
  second, QUAD each), the scale offset `16 + 4 k0` and step 4 / 8 / 16 bytes. A scale entry holds
  the chunk's four words, `{Sw, w1, w0}` and `{w3, w2}` in the two entry RAMs: QUAD writes both
  halves from one 16-byte group of the line, PAIR one half an advance (its 8-byte pair, as int4
  without PAIR writes a word), plain t2 stages block 4c (4c+2) in a register and writes it with
  4c+1 (4c+3), a row's odd last block alone. The FIFOs, the tags and the port A requests are
  as before (one line per advance).
* **Decode (S1).** Element `i` of block `q` is bits `2D q + 2i`: per lane a 4:1 select of 2-bit
  fields and `{6{c == 0}, c - 1}`, on the 8-bit element path. The low stream is block `q` (plain:
  `k mod 4`; PAIR: `2 (k mod 2)`; QUAD: 0), the high `q | 1`; the block scales the same words.
* **QUAD (four columns, two DSP inputs).** The array's DSPs take two weight streams on their
  pre-adder (A low, D high, INMODE by the column's `hi`). T2 makes it four: S1 decodes all four
  blocks of the chunk, every column hop carries 32 bits instead of 16, and in front of each PE's
  A / D a 2:1 select by the column's `hq` (a per-command constant: `j >= 2M`) picks `{4c+3,
  4c+2}` instead of `{high, low}`; `hi` is then `c` odd. So column `c M + j` multiplies block
  `4c' + c` with its ACT row's block, the column end applies that block's B (a 4:1 of the four
  decoded scale streams per column, where PAIR had a 2:1) and the four columns' integers of row
  `j` are added before the result FIFO, two 48-bit adders deep. The ACT RAM reads four blocks a
  cycle: r_blk3 / r_blk4 for the rows in `r_hq` (a 4:1 on each column's block address); past a KB
  that is not a multiple of 4 a column reads block 4c again (its multipliers are zero), as PAIR.
  Consecutive QUAD MMs of one M share the pipeline (the two heads), QUAD and non-QUAD do not.
* **DUP4.** The quantizer marks its writes (flag bit4); the ACT RAM's column select takes rows
  `r + c rows` (c = 1..3; all below MCOLS, so group 0) in the same cycles.
* **Cost at D = 128, MCOLS = 4 (estimated, not synthesized).** The hop registers of columns
  1..3 (+6.1K flip-flops), the S1 decode's two more streams and their skew lines (~2K SRL LUTs,
  ~2K flip-flops), the PE input selects (384 PEs x 16 bits, ~3K LUTs), two more scale decoders
  and the 4:1 per column end, the QUAD row adders (one 48-bit adder per QUAD row) and the ACT
  address 4:1: ~7K LUTs and ~10K flip-flops, no DSP or block RAM. Timing at 200 MHz (5 ns): the
  new levels are one LUT each (the PE select between a hop register and the DSP's A/D register,
  the 4:1 of 2-bit fields after the chunk FIFO's read register, the 4:1 of scale words before
  the dw subtract) and the row sum's second adder; the per-column `hq` drives D x 8 select LUTs
  (a high fanout, like `hi` into the DSPs' INMODE registers today) and may want replication.
* **Tests** (tests/test_t2_rtl.py, Verilator, D = 32): plain / PAIR / QUAD with KB 1 to 9, k0
  0 to 8, M 1, 2, 4, MCOLS 4 and 8, UNIT, ACC, RMAX, ASCALE, a replayed t2 row (M 8 > MCOLS 4),
  QUAD's row choice on distinct ACT rows, a program of every format back to back, and the other
  formats' MM cases on a T2 build: bit-exact against the ISA simulator.
* **Throughput** (Verilator, behavioural DRAM, one row, M = 1, the cycles 256 more weight rows
  add): QUAD takes a chunk a cycle with one port A line request a chunk (its four words in one
  16-byte group, no Sw-alone requests at srs = 16 + 16 ceil(KB/4)). D = 32, KB 16: QUAD 1,024
  cycles for 1,024 chunks, int4 PAIR 2,048 for 2,048; D = 128, KB 48: QUAD 3,072 for 3,072,
  int4 PAIR 6,144 for 6,144 (1.000 cycles a chunk both: QUAD half PAIR's cycles for the same
  weights). tests/test_t2_rtl.py test_quad_takes_a_chunk_a_cycle_rtl checks it.

### Multi-slice (f2-hbm:s2, :s4)

The rotated t2 decode compiles for two slices (2B: 0.28 GiB a slice, 4B: 0.62) and four (4B:
0.33 GiB; the 2B's two KV heads stop at two slices). On the ISA simulator the 2B on two
slices (f2-hbm:s2, t2 weights and head) gives 4 / 4 greedy tokens the fp32 reference's.

## 7. The 9B and the 27B in t2 within a slice's 4 GiB (ISA)

`tools/b2_image_size.py` reads the image's own layout (every allocation's size is the gap to
the next) and classes it; the classes add up to `Image.nbytes`. The images below are the
serving path's: t2 weights and head, the rotation on the device, the int8 embedding rows from
the host (`embed_host`: a slot of rows, not a 248320-row table), `cap` 2048 tokens of KV cache
(otpu-serve's default), one slice of f2-hbm (4 GiB) unless noted. MiB.

| Model (rotation block) | ternary matrices: t2 data + scale rows | LM head (t2) | DeltaNet states + windows | KV cache (2048) | norms, tables, I/O | total | of 4 GiB |
|---|---|---|---|---|---|---|---|
| 2B (512) | 328 + 60 | 121 + 19 | 18 + 3 | 12 | 11 | **572 MiB** (0.56 GiB) | fits, 3524 MiB free |
| 4B (512) | 854 + 140 | 152 + 23 | 48 + 5 | 33 | 12 | **1266 MiB** (1.24 GiB) | fits, 2830 MiB free |
| 9B (4096) | 1654 + 242 | 243 + 34 | 48 + 5 | 33 | 12 | **2271 MiB** (2.22 GiB) | fits, 1825 MiB free |
| 27B (1024), one slice | 6302 + 972 | 303 + 42 | 144 + 24 | 66 | 14 | **7866 MiB** (7.68 GiB) | does not fit (-3770 MiB) |
| 27B, f2-hbm:s2, a slice | 3271 + 504 | 152 + 21 | 72 + 12 | 33 | 13 | **4078 MiB** (3.98 GiB) | fits, 18 MiB free |

* **The 9B fits with 1.8 GiB to spare.** Its weights are 2.2 GB in the GGUF and 1.9 GiB
  here (the t2 rows are the PQ2_0 code bytes; a t2 scale row is 16 + 16 ceil(KB/4) bytes, 14%
  of a 4096-wide row against PQ2_0's fp16 per 128, 1.6%). The KV cache is 4.1 MiB an
  attention layer at 2048 tokens (8 layers, 4 KV heads of 256, int8 K with its scales, V
  transposed); the DeltaNet state 2 MiB a layer (32 heads x 128 x 128 fp32). Headroom for a
  longer context: 16.4 MiB per 1024 tokens, so cap 65536 (1 GiB of KV) still fits.
* **The 9B's head** is its own tensor (`output.weight`, untied, Hadamard-folded like the
  projections: prism.hadamard.version 1); in t2 277 MiB, in int4 517 MiB (the image 2511 MiB):
  7.1 checks both.
* **The 9B's rotation block is 4096** (the 2B's and 4B's 512, the 27B's 1024): the hadamard
  kernel's H_512 MM followed by three butterfly stages of VOP adds and subtracts over each
  block's 8 rows of 512 (kernels/hadamard.py), the same 64 KB H matrix.
* **The 27B does not fit one slice**: 7.68 GiB in t2 (the header's tensor table; the GGUF
  itself is not on disk here, only its first 48 MiB). On **f2-hbm:s2** (two slices, 8 GiB: each
  holds half the heads, half the MLP rows, half the vocabulary, docs/f2_hbm.md) it is 3.98 GiB
  a slice at cap 2048, 18 MiB under the slice's 4 GiB: it fits, with the context limited to
  about 4000 tokens (a slice's KV cache grows 8.2 MiB per 1024 tokens). Its MLP width 17408
  (17 blocks of 1024) stays unpadded on two slices (8704 = 8.5 blocks: padded to 18432, as
  `bonsai2.spec(slices=2)` does: the table's figure).
* **f2-hbm:s4 does not map the 27B, nor s2 the 9B**: a slice's DeltaNet heads must make
  whole rotation blocks before out_proj (Spec.check): 12 heads x 128 = 1536 is not a multiple
  of 1024 (27B on four slices), 16 x 128 = 2048 not of 4096 (9B on two). Splitting the
  out_proj input's rotation across slices would need the heads' outputs gathered before the
  Hadamard (an all-gather of 2048 floats a token), not done.
* **ACT RAM and the 27B.** Spec.check refused the 27B on any slice count: "an inner
  dimension exceeds ACT RAM" (its MLP width 17408 > 128 blocks x 128). W_down's input goes
  through ACT a chunk at a time (swiglu_down's chunks of whole rotation blocks: 1024 here), so
  the check now takes the widest chunk the layouts can make (`Spec._down_chunk`), not the
  width; the check of the other dimensions (hidden, the attention heads' width) is as before.

### 7.1 The 9B's greedy check on the ISA simulator

`tools/fpga_inference_check.py --backend isa --target f2-hbm --wformat t2 --golden numpy
--chat --steps 16 --cap 2048` (the 2B / 4B baselines' probe, section 5: the chat template,
thinking off, "Write one sentence about the Moon.", 19 prompt tokens; the device's argmax
against the fp32 reference's logits over the same history):

| 9B image | head | image bytes | tokens | argmax vs fp32 reference | cosine of the logits |
|---|---|---|---|---|---|
| t2 weights, rotation on the device | t2 (latent ternary) | 2,381,125,248 (2271 MiB) | "The Moon is Earth's only natural satellite, a rocky celestial body that orbits our" | **16 / 16** | 0.9998-1.0000 |
| t2 weights, rotation on the device | int4 (latent ternary) | 2,633,418,368 (2511 MiB) | the same 16 tokens | **16 / 16** | 0.9998-1.0000 (the same logits to the printed digits: both heads hold the ternary codes exactly) |

The t2 head is exact on the ternary codes (section 3), as the int4 head is, and 240 MiB
smaller: the same tokens and logits. A 9B decode step on the ISA simulator takes 130-210 s on
this host, the image 1040 s to build under load. `bonsai2.Weights.unfold` had multiplied a [N, K / B, B] stack by H, which numpy
runs as one GEMV a row: with the 9B's B = 4096 (K / B = 1) the 248320-row embedding table
took 248320 GEMVs each re-reading the 64 MB H, longer than the image's quantization; it is
one [N K / B, B] GEMM now (bit-identical: tests/test_bonsai2.py).

## 8. Decode cost a token: t2 QUAD with the rotation, PORTB 1 / 2 / 4 (ISA), and the RTL

`tools/b2_step_costs.py` (b2-scripts/step_costs.py's method): one resident decode step on
the ISA simulator after four warm-up steps, every executed instruction at opentpu.profile's
nominal unit cycles and port-B D-byte chunks (the MMs at their advance counts: t2 QUAD one
advance a chunk), cap 2048. "tok/s" is the port-bound figure at 200 MHz: unit cycles a token
at the core clock. The RTL column is the whole token (every layer and the LM head) on the
T2 build in Verilator with the board's memory path (b2-scripts/perf_b2.py: otpu_native_dram
on 8 channels, --lat 60), as section 6 measured the format.

| Model | weights | port-B chunks a token (MMs) | ISA unit cycles: PORTB 1 / 2 / 4 | port B busy at 1 / 2 / 4 | tok/s at 200 MHz (ISA, port-bound) | RTL cycles a token (PORTB 1) | RTL: of the port's rate | RTL tok/s at 200 MHz |
|---|---|---|---|---|---|---|---|---|
| 2B | t2, t2 head | 4.26 M (3.91 M) | 5.44 / 5.43 / 5.43 M | 78 / 39 / 20% | 36.8 | 4,766,291 | 89.1% (4,247,164 chunks) | 42.0 |
| 4B | t2, t2 head | 9.43 M (8.55 M) | 12.16 / 12.14 / 12.13 M | 78 / 39 / 19% | 16.5 | 10,443,310 | 90.1% (9,407,794 chunks) | 19.2 |
| 9B | t2, t2 head | 16.57 M (15.67 M) | 20.47 M (PORTB 1) | 81% | 9.8 | (not run: ~2x the 4B's hour) | | |

Where a token's unit cycles go (2B / 4B / 9B, PORTB 1): the t2 QUAD MMs 3.86 / 8.30 / 15.49 M
(one cycle a chunk), the rotations' H MMs 0.11 / 0.30 / 0.79 M (plain t2 MMs of M = K/512
rows: all four columns busy; the 9B's 4096-blocks add their butterflies' VOPs, 0.45 M of
copies and subtracts: its rotation is 6% of the token, the 2B's and 4B's 2%), DSTEP (the
DeltaNet state updates) 0.59 / 1.57 / 1.57 M, the other VOPs 0.46 / 0.88 / 1.19 M, QACT 0.17 /
0.33 / 0.44 M, LD / ST 0.22 / 0.38 / 0.44 M. The MMs are 71-80% of the unit cycles; the rest
is the VPU's and the DMA's, which the RTL overlaps with the MXU's streaming: the 2B's RTL
token (4.77 M cycles) is below the ISA's serial sum (5.44 M) and 11% above its port-B
roofline; the 4B's (10.44 M) 14% below its serial sum and 10% above its roofline (90.1% of
the port; 1579 s of Verilator for the token). The 9B's port-bound figure is 9.8 tok/s at 200
MHz (16.57 M chunks: 2.1 GB a token); at the 2B's and 4B's RTL efficiency (89-90% of the port)
about 8.8 tok/s. Against the board's rotated int4 baseline (section 5: 2B 19.3, 4B 8.6 tok/s
served at ~78% of the port) the RTL's t2 tokens are 2.2x the 2B's and 2.2x the 4B's.

**A wider port B does not help t2 at MCOLS 4.** t2 QUAD takes a whole chunk a cycle, 512
weights into the MXU's four columns: the MXU's rate and the port's are the same 128 B a
cycle. At PORTB 2 and 4 the compiler finds no wider reuse for t2 (r = 4 is MCOLS), so the
token's cycles do not move (5.44 -> 5.43 M: only the DeltaNet's int8 a / b MMs pair) and port B
idles 61% and 80% of the time. What a wider port needs with t2 is more columns to reuse into:
MCOLS 8 (an 8-way reuse of two chunks a cycle; the MXU's cost, not the port's) or, for the
same effect at MCOLS 4, batching (two sequences' rows share each chunk: docs/offload.md's
batched decode). The 4-bit formats are different: int4 QUAD streams two chunks a cycle, so
a wider port does pay for rotated int4 (the 2B's latent codes in int4, int4 head, the same
tally): 7.92 M chunks a token, 9.10 M unit cycles at PORTB 1 (port B busy 87%), 5.43 M at
PORTB 2 and at 4 (73% and 36% busy): at PORTB 2 int4 reaches t2's cycles at twice the bytes,
and t2 at PORTB 1 already has them.

## 9. MTP speculative decode on the t2 images (ISA)

How the serving path does it (opentpu/llm/mtp.py, docs/mtp.md 9-10; otpu-serve --mtp): the
image holds the -MTP GGUF's nextn layer (mtp.*: its fc, one attention layer with its own KV
cache, its norms) and an fp4 draft head over the 32768 lowest ids, and two slots of every
DeltaNet state and convolution window. An iteration at position p with the token t and the
draft d runs one **verify** (the rows kernel over the two rows (t, d) at p, p + 1, forked: row
0's state goes to the committed slot, row 1's to the other; the final norm's rows kept) and one
**draft** (the MTP layer over (hid, a0) and (hid, a1), two rows; the next draft is the accepted
row's); the draft is accepted when it equals row 0's argmax; a rejected row leaves nothing
behind (its KV rows are positional and overwritten, its state sits in the uncommitted slot).
On the card the loop chains the buckets' programs (V, E per parity, D, D1: six a bucket,
compiled on the host at a bucket's first use). Greedy, the tokens are plain decode's bit for
bit (every emitted token is the argmax of a row the decode step would compute).

`tools/b2_mtp.py` runs it on the ISA simulator against plain greedy decode of the same image
layout (t2 weights and head, the rotation on the device; the MTP layer and its fc int8: the
nextn layer is Q8_0, not ternary, `Spec.formats` attn@L=int8,mlp@L=int8), the prompt through
the chunked prefill (8 rows a run), and tallies each run kind's unit cycles as section 8 does.
Ternary-Bonsai-2-4B-MTP, chat template, thinking off, greedy:

| prompt | tokens | equal to plain greedy | drafts accepted | a plain step | a verify run (2 rows) | a draft run | speedup in unit cycles | tok/s at 200 MHz, port-bound: plain / MTP |
|---|---|---|---|---|---|---|---|---|
| short (19 tokens: "Write one sentence about the Moon.") | 30 (to EOS) | 30 / 30 | 11 / 18 = 0.61 | 12.16 M | 23.86 M = 1.963 steps | 1.35 M = 0.111 steps | **0.78x** (29 tokens: 352 M plain, 452 M MTP) | 16.5 / 12.8 |
| long (274 tokens: tools/f2_bench.py's FPGA passage, "summarize in two sentences") | 16 (--steps 16) | (the host loop's; plain decode was not run on this prompt, its step cost is the short prompt's) | 7 / 8 = 0.88 | 12.16 M (the short prompt's) | 23.95 M = 1.971 steps | 1.36 M = 0.112 steps | **0.91x** (15 tokens: 182 M plain, 201 M MTP) | 16.5 / 14.9 |

The loop on the device (generate_card, one chained run: the serving path's) gives the same
30 tokens as the host loop on the short prompt, 18 iterations, 11 accepted; its loop costs
429 M unit cycles against the host loop's 452 M (its E / D1 kinds at a bucket's end), still
0.82x of plain's 353 M; its host compile was 3.2 s (bucket 1's six programs and the prefill's
per-position programs). A verify run's cycles are the same at every position of the reply
(23.85-23.86 M: the KV reads grow by a few thousand a token); its MMs are "MM t2 pair", 8.30 M
chunks in 16.61 M cycles, two a chunk.

**Why t2 QUAD and MTP do not add up at MCOLS 4.** A verify is two rows. t2 QUAD needs 4 M <=
MCOLS, so it is M = 1 only: a two-row MM in t2 can PAIR at most (2 M = 4 columns, two blocks
of the chunk a cycle per row), half a chunk a cycle, and the verify streams a step's bytes in
twice the MXU cycles: c_2 = 1.96 against the 1.05-1.06 of the int8 and fp4 images in
docs/mtp.md 9.1, where the MXU had three idle columns at M = 1 for the second row to use. With
an acceptance of 0.61 an iteration emits 1.61 tokens for 2.07 steps' cycles: 0.78x. The same
four columns give either t2's 2x over int4 at M = 1 or MTP's second row, not both; MTP pays on
the int4 / int8 images (the board's today, section 5: 4B int4 + MTP 10.9 against 8.0 tok/s),
and would pay on t2 with eight MXU columns (M = 2 QUAD: 4 M = 8 <= MCOLS 8, a chunk a cycle
for both rows, c_2 back near 1.05), the same change a wider port B needs (section 8).
The draft run is 0.11 of a t2 step (0.05 of an int8 one in docs/mtp.md: the step shrank, the
int8 MTP layer and the fp4 head did not).

**The long case on the board** (section 5: 4B int4 + MTP 10.9 tok/s after the short prompt,
6.2 after the 274-token one, where plain decode gives 8.0 / 8.0). What differs between the
two on the model side is the attention bucket: the short prompt's reply runs in bucket 1,
whose six loop programs the warm-up request had compiled; the long prompt's reply starts in
bucket 2 (positions 256-511), whose programs the reply's first loop compiled on the host
before running (MTPDecoder._gen_programs: the `what` keys of the program cache), while plain
decode's bucket programs were compiled ahead at server start (opentpu/host/gen_warm.py), which
otpu-serve left off for --mtp. On this host the six programs of a bucket (V 3024-3200
instructions, E 2500-2588, D 627-803, D1 444-532 for the 4B t2 image) compile in 1.1-1.5 s
(tools/b2_mtp.py --compile-only); the card host is slower and adds each program's wait
entries (gen_warm's note: 3.6 s a bucket for the 35B MoE there) and the chain area's writes.
`gen_warm.warm_mtp` now compiles the MTP loop's programs of every bucket (greedy and the
default sampler's V / E, the split form where a V or E is over IMEM) into the program cache
in the background process at server start, and otpu-serve / otpu-chat --mtp start it
(tests/test_mtp.py test_mtp_loop_programs_warmed_ahead: a decoder then compiles none of them,
its tokens plain decode's). Whether that is the whole 10.9 -> 6.2 gap is for the board to
say (the rest of the loop is position-independent: the KV and state handling of rejected rows
costs nothing per position; a verify at p = 275-290 costs 23.95 M unit cycles against 23.86 M
at p = 20-50, the KV reads of 8 attention layers and the MTP layer's). The long prompt's
acceptance on the ISA simulator, 0.88, is higher than the short one's 0.61 (a summary's
tokens against a one-liner's), so acceptance is not what made the board's long case slower.

## 10. What the board needs

* **A T2 build** (otpu_top T2 = 1, CAPS bit 28; section 6) at **PORTB 1** in the nightly: the
  t2 QUAD decode is port-bound at one chunk a cycle, 89-90% of the port on the RTL (2B 42, 4B
  19, 9B ~9 tok/s at 200 MHz against the rotated int4 baseline's 19.3 / 8.6 / none). PORTB 2
  adds nothing to t2 at MCOLS 4 (section 8); it is the int4 images' gain (their QUAD streams
  two chunks a cycle) and, with MCOLS 8, t2's at M = 2 (a verify's two rows, or two
  sequences).
* **The 9B** fits the slice's 4 GiB with 1.8 GiB to spare (section 7); its image builds in
  the image cache as the others (the GGUF's 4096-block rotation: the butterflies are 6% of
  the token). The 27B needs f2-hbm:s2 (3.98 GiB a slice at cap 2048).
* **MTP** stays off for t2 images at MCOLS 4 (0.78x); keep it for int4 / int8 images, with
  gen_warm's bucket programs ahead.
