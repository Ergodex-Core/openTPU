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
