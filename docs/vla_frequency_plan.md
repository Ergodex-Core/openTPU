# Plan: VLA inference at >= 1 Hz on openTPU (F2 FPGA)

Status: plan (2026-10-16). Every number here is an *estimate* from model shapes and the
openTPU configuration unless marked measured; Phase 0 (the π0.5 port, docs/pi05.md) has
measured section 3's π0.5 numbers with the ISA tally and RTL timing of cut phases. Work order: π0.5 port first, then
RhinoVLA (HuixiAI/RhinoVLA-V3), then the variant below. Reviewed 2026-10-16
(docs/vla_frequency_plan_review.md): its findings are folded in below, chiefly that the DSP
packing is the one gate on the 1 Hz claim (L1), the ACT RAM is width-bound (L2), port B at
PORTB 1 takes 25.6-32 GB/s whatever the HBM offers (L7, 6a), QST bounds the vision tower at
scale (L11), and the numbers are committed at 200 MHz. Everything is validated in the ISA
simulator and Verilator first; FPGA synthesis only in the nightly build.

## 1. VLA and LLM need different openTPU variants

| | LLM decode (Gemma-3-1B, Ternary-Bonsai; branch ternary-bonsai) | VLA (π0.5, RhinoVLA; this branch) |
|---|---|---|
| shape | one token a step: matrix-vector (M = 1) | a ~300-970-token prefix, 3 views of vision, 10 denoising steps of 50 rows: matrix-matrix |
| bound by | weight bytes a token (port B) | multiply-accumulates, and the VPU's element-wise work |
| levers | wide port B / HBM lanes (docs/wide_port.md), t2 weights, MTP | a wider MXU in one slice, ACT RAM rows, TMEM, VPU lanes, a fused activation unit, clock, W8A16 |
| openTPU today on it | 99.4% of the port on the RTL (Gemma, measured) | MXU-bound, then VPU-bound (below) |

One RTL, two targets: `f2-hbm` (+ PORTB) for the LLMs and a new `f2-vla` (section 5). The units
that differ are parameters or optional blocks, as T2, PORTB and NCH already are.

## 2. The workload

Per action chunk (one inference). MACs from the layer shapes; VPU element ops from the
elementwise work (RMSNorm, GELU-tanh at ~9 VOPs an element in kernels/lib.py, softmax ~5).

| | π0.5 (lerobot/pi05_libero_finetuned_v044) | RhinoVLA (paper: arXiv 2606.07383) |
|---|---|---|
| vision | SigLIP so400m, 3 views x 256 tokens: ~0.3 T MAC | Qwen3-VL ViT, 3 views, merged to 64 tokens each: ~0.25 T MAC |
| backbone prefix | Gemma 2B (18 layers, 2048 wide, GeGLU 16384), ~800-970 tokens: ~1.6-1.9 T MAC | Qwen3-VL LM 2.1B, ~300 tokens: ~0.6 T MAC |
| action expert | Gemma 300M, 50 rows x 10 steps: ~0.15 T MAC | 0.40B, 31 rows x steps: ~0.1 T MAC |
| total | **~2.1 T MAC** | **~1.0 T MAC** |
| GELU elements in the prefix MLPs | 800 x 16384 x 18 = 236 M (~2.1 G VPU ops) | ~300 x 6144 x 28 = 52 M (SwiGLU) |
| softmax elements (prefix attention) | 18 x 8 x 800^2 = 92 M | ~28 x 16 x 300^2 = 40 M |

References: π0.5 on a Jetson AGX Orin (FP16) 858 ms; RhinoVLA on Huixi R1 (7 nm, 500 TOPS
int8) 85.5 ms, 11.7 Hz (the paper's measurements).

## 3. Where openTPU on the F2 stands (S = 1, MCOLS 4, ACT_ROWS 16, LANES 8, 200 MHz)

π0.5's numbers are **measured** by Phase 0 (docs/pi05.md 5: tools/pi05_cost.py's ISA tally of
a whole chunk at Np = 558 valid prefix rows, calibrated by Verilator runs of cut phases);
RhinoVLA's are still estimates.

* MACs (measured): **1.47 T a π0.5 chunk**, not ~2.1 T: two real cameras (the third is masked
  and dropped), and 558 valid prefix rows (lerobot pads to 968; the pads are dropped exactly).
  Vision 0.22 T, prefix 1.07 T, expert 0.18 T.
* MXU (measured tally): 512 MAC/cycle, 102 GMAC/s: **14.5 s** a π0.5 chunk (vision 2.2,
  prefix 10.5, 10 expert steps 1.8). RhinoVLA (estimate): ~10 s.
* VPU (measured tally): 2.0 s a chunk. **The prefix's GELU is 0.78 s, not ~1.3 s**: one cycle
  an element at 8 lanes as estimated, but over 558 x 16384 x 17 = 155 M elements (the last
  layer's MLP is dead), not 800 x 16384 x 18; the chunk's GELU in all (tower, expert too)
  1.26 s; softmax 0.28 s, norms 0.13 s. QACT 0.77 s, QST 0.25 s (the K stores a word a
  cycle on port A, the transposed V stores a byte: confirmed on the RTL, docs/pi05.md 5).
* Weight traffic (measured tally): port B 1.43 G chunks, **183 GB a chunk** (7.1 s at a chunk a
  cycle), of it the prefix 138 GB: TMEM (64K words) caps the prefix's row group at 8 rows, so
  the 2B's ~1.9 GB stream 70 times (558 / 8), not 35 (ACT_ROWS 16). ACT_ROWS 128 alone does
  nothing; with a 256K-word TMEM (row groups 64 / 32 / 32) port B falls to 2.3 s.
* RTL overlap (measured, Verilator, cut phases): cycles 1.23x (tower layer) to 1.54x (expert
  step) the slowest unit's.
* Total today: **~15-22 s a π0.5 chunk** (measured tally bound 14.5 s, serial 25.1 s, the RTL
  ratios between); RhinoVLA ~12-15 s (estimate).
* int8 (measured, docs/pi05.md 4): W8A8 as the card runs it moves π0.5's actions by 6.8% of
  their range, the 2B's activations nearly all of it; the 2B at W8A16 with the expert W8A8:
  1.7%. L9 (W8A16) is needed for π0.5's prefix unless a rollout shows 7% harmless.
* Scaled (measured ISA tally, no RTL): MCOLS 64, ACT_ROWS 128, LANES 64, TMEM 1M words: MXU
  0.95 s, port B 0.65 s, VPU 0.26 s, QST 0.34 s (port A's byte a cycle: does not scale with
  LANES) at 200 MHz; ACT_ROWS 256 needs an MM row-count field past 8 bits.

## 4. Scale one openTPU up, not out

More compute comes from one wider slice, not from several openTPUs (slices): a VLA's prefix is
~800 rows of the same weights, so one weight stream can feed many more MXU columns at once, and
one slice keeps every activation, KV cache and DeltaNet-free attention local: no collectives per
layer, no sharding of π0.5's single KV head, no per-slice copies of the vision tower. The
multi-slice route stays the fallback (section 8).

How a wider slice turns into MACs: an MM streams each weight chunk (D = 128 int8 weights) once
per group of ACT_ROWS activation rows, and every MXU column applies it to one row a cycle. So

    MAC/cycle = MCOLS x 128,  weight bytes/cycle = 128 / ceil(min(M, ACT_ROWS) / MCOLS)

A prefix of ~800 rows keeps 64 columns busy with one 128-byte chunk every
ceil(ACT_ROWS / MCOLS) cycles: compute grows 16x while the weight bandwidth a cycle does not.

## 5. The levers inside one slice (gain, cost, validation)

| # | lever | gain (estimate) | cost on the VU47P (estimate) | validation |
|---|---|---|---|---|
| L1 | **MXU columns MCOLS 4 -> 32 physical, 64 logical with DSP48E2 int8 packing**: one activation row x two weight rows a DSP (A = w_a, D = w_b << 19, the pre-adder's AD = w_a + w_b 2^19, B = the activation), a physical column yielding two logical output columns a cycle. A CL = 16 chain's lane sum is at most 16 x 127^2 = 258,064 < 2^18 (quant.py's int8 never holds -128), the lanes split at the chain end (L = sext19(P[18:0]); H = P[47:19] + P[18], the borrow absorbed by the high lane's column end on its pre-adder: no fabric, no cycle), before the column end's accumulation over the row's blocks (where a packed lane overflows at the second chain already). PAIR's A / D select survives as a third INMODE value. The repo's "inexact after 3 accumulations" was the DSP48E1's 25-bit pre-adder (a 17-bit lane); it already had a bit-exact packed DSP48E2 column, rtl/exp/mx_mxu (docs/mx_mxu.md 5) | MCOLS 32 unpacked: 8x, 4096 MAC/cycle; 32 packed (64 logical): 16x, 8192 | 146 DSP a physical column (128 PEs + two column ends), **73 a logical column** against 137 today; MCOLS 64 logical = 4,672 + ~192 epilogue lanes ≈ **4.9K of 9,024**; MCOLS 32 unpacked 4,480; 64 unpacked would be 8.8K, the whole part. LUT est. ~250-320K. W8 only: 4-bit PAIR / QUAD, t2 and W8A16 stay one product a DSP | **Phase 1 done (docs/mxu_packing.md)**: `otpu_pe_pk` (generic and DSP48E2), `otpu_mxu_pk_col`, tests/test_mxu_pk.py in Verilator: 9 passed, 0 mismatches against the ISA's integer over the corners (every sign of +-127 at the lane bound, the borrow, K = 16384, random rows), and the two failing patterns pinned (sixteen (-128)^2 in a lane; w_b = -128 beside a negative w_a wraps the 27-bit pre-adder): both need an operand the quantizer never produces. Left: one packed and one unpacked column out of context on the nightly host (`tools/mxu_pk_ooc.sh`: DSP count, **fmax**, the open question); then MCOLS 8 -> 16 -> 32 physical bit-exact, mxu_bench |
| L2 | **ACT_ROWS 16 -> 255**: an MM's row count is an 8-bit field (isa.mm: 0 < M < 256), and 255 rows reach 256's pass count over the prefix, ceil(558 / 255) = ceil(558 / 256) = 3 and ceil(968 / 255) = 4, so **no ISA change**; a 16-bit M (QST's rows already are) stays an option for M >= 256 | weight passes over the prefix: 3 at Np 558 instead of 70 today (TMEM caps the row group at 8; section 3) | **width-bound, not capacity-bound**: otpu_actram reads one D-byte block a column every cycle (`r_data` is MCOLS x D x 8 bits), 16 RAMB36 a physical column today (4 pieces x 4) beside its DSP column; 32 physical columns (64 logical, packed): **512 RAMB36** of the part's 2,016; 64 unpacked would be 1,024 (or 960 URAM at 72 bits a port: every URAM on the part), which is why L1 packs weights, not activations; rows are free up to the RAM's depth (512). Pumping the ACT RAM at 2x as TMEM_PMP does halves it, another clock-domain item | **ISA tally (done)**: the prefix's 255-row group needs TMEM 1.5M words (at 1M: "TMEM exhausted", the group stays 128 and nothing changes); with it port B falls 0.65 -> 0.51 s a chunk at MCOLS 64 (the prefix 0.40 -> 0.26 s), 0.69 -> 0.55 s at MCOLS 32, the MXU bound unchanged. RTL timing with ROWS 255 in phase 3 |
| L3 | **a fused activation unit in the MXU drain / quantizer**: GELU-tanh / SiLU (table + polynomial, as SGMV's sgmv_act), gate x up, then quantize for the next MM (SGMV's MLP_FUSE); gate and up rows interleaved in the image so a drain yields pairs; one unit per drain lane | removes the prefix MLPs' VPU work: GELU 0.78 s of π0.5 today, 0.14 s at LANES 64 (tally): ~6% of the serial sum at the f2-vla point. Still cheap; built after L4-L6 and sized on the measured remainder | est. ~4K LUT, ~10 DSP, 3 BRAM a lane, 8-16 lanes | bit-exact against an ISA definition of the fused op |
| L4 | **VPU lanes 8 -> 64** (LANES; TMEM banks with them) for softmax over long rows, norms, RoPE, residual adds | 8x element throughput: π0.5's ~92 M softmax elements and norms in ~0.05 s | LUT est. ~150-200K (fp32 lanes), DSP ~300-500 | the VPU tests at LANES 16, 32, 64 |
| L5 | **TMEM 64K -> 1M words** (4 MB in URAM), banked by LANES; the drain's MCOLS outputs and QACT's row reads at 64 words a cycle | prefix tiles of 256 rows x 2048 fp32 (2 MB) in one pass; fewer DRAM round trips | ~114 URAM; a 64-way bank crossbar est. ~80-100K LUT | tiled prefill tests; RTL bank-conflict counters |
| L6 | **QACT / QST at LANES elements a cycle** (they feed 256-row stationary operands) | activations into ACT RAM as fast as the MXU drains them | with L4's lanes | the quantizer tests at LANES 64 |
| L7 | **memory: 8 HBM ports on the one slice** (NCH 8): they buy the 4 GiB and the lanes' headroom, **not bandwidth**: a slice at PORTB 1 takes one D-byte chunk a cycle, 25.6 GB/s at 200 MHz, 32 at 250, whatever the ports offer (docs/f2_hbm.md; docs/wide_port.md: "more HBM ports only help once a slice takes more than one chunk a cycle"), and the board delivers 78-89% of that (burst200). **PORTB 2** (the wide-port lanes built on ternary-bonsai: a named dependency of f2-vla with a merge point, built once there) is on π0.5's critical path: the expert step is port-bound in the tally (0.016 s port B against 0.014 s MXU at M 50 under 64 columns). A DMA lane for activation tiles / KV goes through otpu_native_dram's grant chain unless it gets its own channel path (the LD before an MM is not overlapped with that MM) | keeps the MXU fed at small M; activation traffic off the weight port | the HBM shell exists (f2-hbm: S=1 x 8); lanes est. +~10K LUT | RTL timing on the 8-port memory model; the lanes' tests from ternary-bonsai |
| L8 | **clock 200 -> 250 MHz** (SGMV closed 250 on the same part at 20K LUT; today's full f2 build is 145K LUT at 200) | 1.25x, treated as upside: every number below is at 200 MHz | timing work, nightly | synthesis / route (nightly only) |
| L9 | **W8A16 where accuracy needs it** (docs/pi05.md 4: W8A8 moves π0.5's actions by 6.8% of their range, the 2B at W8A16 1.7%; RhinoVLA's paper: W8A8 hurt success) | accuracy; int8 x int16 is one MAC a DSP: half L1's rate if L1 packs, the same rate unpacked (decided by phase 1) | DSP or a second weight path | action error vs fp32 (docs/pi05.md harness) |
| L10 | **model-side**: the 3 views as MXU rows of one pass, the prefix KV once a chunk, the expert's time conditioning precomputed (docs/pi05.md), fewer visual tokens (RhinoVLA: 64 a view) | π0.5 -> RhinoVLA halves the MACs | none | ISA |
| L11 | **QST stores as D-byte chunks** through the DMA / port B write path, or SigLIP's per-view outputs kept in TMEM and quantized straight into ACT RAM (with L10). Today QST writes on port A a word a cycle for contiguous, word-aligned rows (otpu_quant's `wide`: the K stores) and a byte a cycle for the transposed V stores (es the column stride): ceil(n / LANES) + n / 4 or + n cycles (opentpu/profile.py, confirmed on the RTL: the tower's QKV phase 1.10x its MXU cycles with QST tallied so; the byte-only model exceeded the measured total). Neither shrinks with LANES | the vision tower's bound at scale: QST 0.181 s against 0.137 s of MXU at the f2-vla point, 0.21 s of the chunk's 2.08 s serial sum (nearly the whole VPU after L4) | a chunk FIFO on the quantizer's store side, or V kept untransposed (the MXU's K-block layout) | the quantizer tests; RTL timing of the tower |

## 6. The VLA variant `f2-vla` (one slice; target, estimate)

S = 1, D = 128, MCOLS = 32 physical columns, 64 logical with the packing (exact, phase 1; its
fmax from the nightly's OOC run decides packed 64 against unpacked 32), ACT_ROWS = 255,
ACT_BLOCKS = 128, TMEM_WORDS = 1.5M (the prefix's 255-row group needs it; L2), LANES = 64,
QST chunked (L11), the fused activation unit,
NCH = 8 HBM ports, PORTB 2, core 200 MHz (250 as upside). Peak: MCOLS 32, 4096 MAC/cycle x
200 MHz = 0.82 TMAC/s (8x today); MCOLS 64 packed, 1.64 TMAC/s at 200 MHz, 2.05 at 250 (20x).
Budget (estimates for the nightly synthesis to check): DSP 4,480 unpacked / ~4.9K packed (4,672
+ the epilogue lanes) of 9,024; **BRAM 512 RAMB36 of 2,016 for the ACT RAM** either way (32
physical columns; L2), the rest for the weight
FIFOs, TMEM pieces and the trace; URAM ~345 of 960 (TMEM 1.5M words, 6 MB); LUT ~0.6-0.8M of
1.3M. The
placement scheme of one DSP column and one BRAM column per array column does not extend to 64
columns across three SLRs: phase 2 sketches the floorplan (columns per SLR, where the result
FIFO and the drain sit).

| | today (MCOLS 4; measured tally) | f2-vla, MCOLS 32 unpacked, 200 MHz | f2-vla, MCOLS 64 packed, 200 MHz |
|---|---|---|---|
| π0.5 a chunk | ~15-22 s (section 3) | tally (ACT_ROWS 255, LANES 64, TMEM 1.5M): MXU 1.90 s, port B 0.55, VPU 0.26, QST 0.21; bound 1.9 s, serial 3.1 s: **0.3-0.5 Hz** | tally (ACT_ROWS 255, LANES 64, TMEM 1.5M): MXU 0.95 s, port B 0.51, VPU 0.26, QST 0.21; bound 0.95 s, serial 2.08 s: **0.5-1.05 Hz** (the RTL's overlap, 1.10-1.54x the slowest unit on cut phases, puts it at ~0.7-0.9 Hz) |
| RhinoVLA a chunk | MXU 6.6 s (shape tally, tools/rhinovla_shape.py: 0.66 T MAC, a 233-row prefix; the real chunk more, the row groups TMEM-capped as π0.5's) | shape tally: MXU 0.90 s, port B 0.25: **~1.1 Hz MXU-bound, no margin** (the VPU / QACT / QST not estimated: ~0.3-0.5 s more serial, by π0.5's) | shape tally: MXU 0.53 s, port B 0.25: **~1.3-1.9 Hz** |

So the ≥ 1 Hz deliverable (RhinoVLA) rests on phase 1's packing result: with MCOLS 64 packed
it has margin (MXU 0.53 s), at MCOLS 32 unpacked the MXU alone is 0.90 s, at the line, and the
other units' overlap has to be measured on the real prefix shape. RhinoVLA's shape (the
paper's Table 8; its V3 branch may differ): 3 views x 64 tokens + ~40 prompt tokens + 1 state
token = ~233 prefix rows through Qwen3-VL-2B (28 layers, 2048 wide, MLP 6144) after a 24-layer
ViT at 256 patches a view; the expert 18 x 1024 on 31 rows: 0.66 T MAC a chunk (vision 0.24,
prefix 0.32, 10 expert steps 0.09), against π0.5's 1.47 T. 250 MHz adds 1.25x on top. π0.5 at 1 Hz needs ~2.8 TMAC/s effective:
the MCOLS 96 stretch (packed; 12,288 MAC/cycle, DSP ~6.2K + VPU) is the point within one slice;
MCOLS 128 would need ~8.2K DSPs, past the part with the VPU's. The plan is sized for Np 558
(the LIBERO prompt shape; at 968 rows, lerobot's padded length, every prefix number is 1.7x).
An action chunk is 50 actions: 1 Hz inference still drives 10-50 Hz control, open loop within
a chunk.

### 6a. Memory bandwidth with 8 ports (π0.5, a chunk)

A slice at PORTB 1 takes one chunk a cycle, 25.6 GB/s at 200 MHz (L7); the HBM's ~115 GB/s
behind 8 ports is not the rate. The tally at MCOLS 64 / ACT_ROWS 128 / LANES 64 / TMEM 1M:

| traffic | port B (tally, PORTB 1, 200 MHz) | at PORTB 2 |
|---|---|---|
| weights, prefix: 5 passes over 558 rows at ACT_ROWS 128; 3 at ACT_ROWS 255 with TMEM 1.5M words (tally: 0.26 s) | 0.40 s against 0.68 s of MXU | 0.20 s |
| weights, expert: 10 steps, 50 rows under 64 columns (a chunk a cycle: port-bound, 0.016 s a step against 0.014 s of MXU) | 0.16 s | 0.08 s |
| weights, vision (both cameras) and the activations' LD / ST | ~0.09 s | ~0.05 s |
| chunk | **0.65 s against 0.95 s of MXU** (0.51 s at ACT_ROWS 255 with TMEM 1.5M: the tally) | ~0.33 s |

So 8 ports are enough for one f2-vla slice, but the margin comes from ACT_ROWS 255 (L2) and
PORTB 2 (L7), not from the HBM: the expert is port-bound already, so PORTB 2 is in the variant,
not a contingency; 16 ports only with the MCOLS 96 stretch.

## 7. Phases (each ends in ISA / Verilator results; the nightly synthesizes the best point)

| phase | work | exit criterion |
|---|---|---|
| 0 | π0.5 port: **done** (docs/pi05.md: ISA-exact numerics, the tally, int8 error). RhinoVLA's shape-only tally: **done** (tools/rhinovla_shape.py, the paper's Table 8: ViT 24 x 1024 at 256 patches a view merged to 64 tokens, 3 views; the 2B 28 x 2048, MLP 6144, 16 / 8 heads of 128; the expert 18 x 1024, MLP 3072, 31 rows; 0.66 T MAC a chunk, 45% of π0.5's; MXU 0.90 s at MCOLS 32, 0.53 s at 64, port B 0.25 s). Left: QST's DRAM rate confirmed on the RTL (L11) | the budget per phase; RhinoVLA's seconds a chunk at each point |
| 1 | **the packing's feasibility** (L1): **done** in Verilator (bit-exact, docs/mxu_packing.md); left: the packed and unpacked columns out of context on the nightly host (DSP, fmax). ACT_ROWS 255 in the tally (done: needs TMEM 1.5M words, L2) and the RTL. QST: the K stores confirmed a word a cycle on the RTL; the V stores (L11) still a byte | **MCOLS 64 packed or MCOLS 32 unpacked**: the fmax a column; the tally at ACT_ROWS 255; QST off the vision tower's critical path |
| 2 | the ACT RAM at the chosen MCOLS: BRAM / pumped / URAM from the width math; a floorplan sketch per SLR (columns, the result FIFO, the drain) | the BRAM budget; the SLR plan |
| 3 | L1 MXU columns: MCOLS 8 -> 16 -> 32 (-> 64) (the RTL parameter exists: MCOLS 4 / 8 builds today) with ACT_ROWS 255 | bit-exact at each MCOLS; RTL timing on the **real prefix shape** (558-968 rows, the 16384-wide MLPs, the 2048-wide projections; a cut model hides the drain's and result FIFO's losses): MAC/cycle ~MCOLS x 128 at >= 75% |
| 4 | L4 / L5 / L6: LANES 64, TMEM 1M words, QACT / QST at LANES a cycle | VPU / QACT tests; bank-conflict counters in the RTL timing; VPU + QACT + QST < 15% of the serial sum |
| 5 | L3 fused activation unit, sized on phase 4's remainder: ISA op + RTL + tests | bit-exact; the GELU's share gone |
| 6 | L7 memory: f2-hbm's S=1 x 8 ports, PORTB 2 merged from ternary-bonsai, the DMA path for activations | RTL timing on the 8-port model: the expert step no longer port-bound; port-B stalls < 5% outside it |
| 7 | RhinoVLA port (Qwen3-VL ViT new; Qwen3 LM kernels exist; expert / flow loop shared with π0.5) | action error vs its reference |
| 8 | f2-vla target; nightly synthesis at 200 MHz, then 250; AFI; board run | ≥ 1 Hz RhinoVLA on the F2 (measured) |
| 9 | L9 W8A16 where phases 0 / 7 show int8 activations hurt; the MCOLS 96 stretch for π0.5 at 1 Hz | success-rate parity on LIBERO; RTL timing, then the nightly |

## 8. Risks and the fallback

* The DSP packing (L1) is exact (phase 1); what is left of the gate is its fmax on the real
  part (the OOC run on the nightly host). Unpacked, MCOLS 64 is the whole part's DSPs, and the
  fallback is **MCOLS 32 x 1 slice unpacked** (4,480 DSP), which keeps every argument of
  section 4 and lands RhinoVLA near 1 Hz with no margin. MCOLS 32 x 2 slices is not the fallback: it
  brings back the collectives and the sharding of π0.5's single KV head that section 4 avoids.
* A 32-64-column MXU, 64 VPU lanes and a 64-way TMEM crossbar in one slice: wiring across SLRs
  is the main risk (pblocks for the MXU's column chain, pipelined weight and ACT-RAM paths), and
  the ACT RAM's 512-1,024 RAMB36 have to sit beside their DSP columns. Hence 200 MHz committed
  and 250 as upside (a 0.6-0.8M LUT design across three SLRs closing 250 MHz on its first
  nightly is unlikely).
* The Hz numbers carry the units' overlap: the tally's bound assumes every unit overlapped, the
  serial sum none; the RTL on cut phases sits at 1.23-1.54x the bound. Phase 3 measures it on
  the real prefix shape.
* The VPU estimate rests on VOP counts; the fused unit's numerics must match an ISA definition.
* W8A8 accuracy for actions is unproven for a rollout (6.8% of the range on π0.5; the paper's
  warning); W8A16 halves the packed rate, and costs nothing unpacked.
