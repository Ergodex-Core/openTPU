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
  1.26 s; softmax 0.28 s, norms 0.13 s. QACT 0.77 s, QST 0.15 s with QST T (0.25 before it:
  the K stores a word a cycle on port A, the transposed V stores a byte, confirmed on the
  RTL; now both a word a cycle, docs/pi05.md 5).
* Weight traffic (measured tally): port B 1.43 G chunks, **183 GB a chunk** (7.1 s at a chunk a
  cycle), of it the prefix 138 GB: TMEM (64K words) caps the prefix's row group at 8 rows, so
  the 2B's ~1.9 GB stream 70 times (558 / 8), not 35 (ACT_ROWS 16). ACT_ROWS 128 alone does
  nothing; with a 256K-word TMEM (row groups 64 / 32 / 32) port B falls to 2.3 s.
* RTL overlap (measured, Verilator, cut phases, docs/pi05.md 5): cycles 1.10x (the tower's
  QKV) to 1.54x (the expert step) the slowest unit's at MCOLS 4; the MXU pops 65-91% of the
  cycles. **At the f2-vla sizes (MCOLS 16 / 32, ACT_ROWS 255, TMEM 1.5M, 16 lanes: the RTL's
  VPU takes no more) the prefix's MLP keeps the MXU at 81% at MCOLS 16 (1.23x) and 60% at
  32; the other phases overlap 1.5-2.9x, the MXU 52-66% at MCOLS 16 and 32-50% at 32**
  (docs/pi05.md 5.3): the MXU drain loses the TMEM grant while the VPU runs and the result
  FIFO fills (v_out at MCOLS 16: blocked 738 K of 2.37 M cycles, the drain denied 756 K; the
  loss follows the VPU's share of the time), not the weight stream (port B 9-20% busy, the
  MXU never starved).
* **After phase 4's first round (measured, docs/pi05.md 5.4)**: the drain's writes before
  the VPU's (TMEM_WARB 2), the drain at MCOLS lanes a step (MXU_E 32), the VPU at 64 lanes
  (its sums on a 16-lane sub-datapath), PACK: v_out at MCOLS 32 overlaps **1.77x** the
  slowest unit (the MXU, 627,586 cycles against its 354,816 packed), the packed MXU at
  **56.5%** (36% unpacked at 16 lanes under the old order); at 16 lanes the arbitration alone
  took p_out at MCOLS 16 from 5.86 to 4.85 M cycles (the MXU 68%) and the VPU-bound phases
  stayed put (every phase but p_qkv / p_out is VPU-bound at 16 lanes with PACK). The overlap
  range the plan carries is now 1.5-1.8x on the measured phases at the target; the rest of
  the phases at 64 lanes were running when this was written (the 64-lane model simulates
  at ~42 cycles a second).
* QST T (measured, docs/isa.md QST): the V^T appends through the quantizer's four-row
  transposer, a word a cycle instead of a byte: the tower's QKV phase 1.306 -> 0.751 M cycles
  at MCOLS 16 (-42%), 1.265 -> 0.710 M at MCOLS 32 (-44%); the chunk's QST 0.21 -> 0.09 s at
  every configuration (tally).
* Total today: **~15-22 s a π0.5 chunk** (measured tally bound 14.5 s, serial 25.1 s, the RTL
  ratios between); RhinoVLA ~12-15 s (estimate).
* int8 (measured, docs/pi05.md 4): W8A8 as the card runs it moves π0.5's actions by 6.8% of
  their range, the 2B's activations nearly all of it; the 2B at W8A16 with the expert W8A8:
  1.7%. L9 (W8A16) is needed for π0.5's prefix unless a rollout shows 7% harmless.
* Scaled (measured ISA tally, no RTL): MCOLS 64, ACT_ROWS 128, LANES 64, TMEM 1M words: MXU
  0.95 s, port B 0.65 s, VPU 0.26 s, QST 0.09 s with QST T (0.21 before it; port A's word a
  cycle does not scale with LANES) at 200 MHz; ACT_ROWS 256 needs an MM row-count field past
  8 bits.

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
| L1 | **MXU columns MCOLS 4 -> 32 physical, 64 logical with DSP48E2 int8 packing**: one activation row x two weight rows a DSP (A = w_a, D = w_b << 19, the pre-adder's AD = w_a + w_b 2^19, B = the activation), a physical column yielding two logical output columns a cycle. A CL = 16 chain's lane sum is at most 16 x 127^2 = 258,064 < 2^18 (quant.py's int8 never holds -128), the lanes split at the chain end (L = sext19(P[18:0]); H = P[47:19] + P[18], the borrow absorbed by the high lane's column end on its pre-adder: no fabric, no cycle), before the column end's accumulation over the row's blocks (where a packed lane overflows at the second chain already). PAIR's A / D select survives as a third INMODE value. The repo's "inexact after 3 accumulations" was the DSP48E1's 25-bit pre-adder (a 17-bit lane); it already had a bit-exact packed DSP48E2 column, rtl/exp/mx_mxu (docs/mx_mxu.md 5) | MCOLS 32 unpacked: 8x, 4096 MAC/cycle; 32 packed (64 logical): 16x, 8192 | 146 DSP a physical column (128 PEs + two column ends), **73 a logical column** against 137 today; MCOLS 64 logical = 4,672 + ~192 epilogue lanes ≈ **4.9K of 9,024**; MCOLS 32 unpacked 4,480; 64 unpacked would be 8.8K, the whole part. LUT est. ~250-320K. W8 only: 4-bit PAIR / QUAD, t2 and W8A16 stay one product a DSP. The integration's extra fabric a physical column (docs/mxu_packing.md 7): the second bm shifter and its skew (8 x 17 bits x (CL + MCOLS) SRL stages), the second 48-bit deskew, two more 24-bit chain-end buses into the column ends; and once a slice: the chunk FIFO's second bank (32 more RAMB36 at FIFO_DEPTH 1024), a 40-bit scale bank, a 2 x DEPTH write-address queue (~400 LUT of LUT RAM at DEPTH 1024). **LUT / FF deltas not measured here**: the Mac has no slang-capable yosys for this SystemVerilog and Verilator reports no cell counts; the DSP counts and the LUT / FF of a packed against an unpacked column are the nightly's OOC run (`tools/mxu_pk_ooc.sh`) | **Phase 1 done (docs/mxu_packing.md)**: `otpu_pe_pk` (generic and DSP48E2), `otpu_mxu_pk_col`, tests/test_mxu_pk.py in Verilator: 9 passed, 0 mismatches against the ISA's integer over the corners (every sign of +-127 at the lane bound, the borrow, K = 16384, random rows), and the two failing patterns pinned (sixteen (-128)^2 in a lane; w_b = -128 beside a negative w_a wraps the 27-bit pre-adder): both need an operand the quantizer never produces. **Phase 3 done in Verilator (branch vla-pack, docs/mxu_packing.md 7)**: `otpu_mxu PACK` (Config.PACK, CAPS2 bit1), bit-exact with the ISA simulator at MCOLS 4 / 8 physical and ACT_ROWS to 64 over int8 in every mode, replay, the 4-bit formats and t2 unpacked beside it, odd N, KB 1 (tests/test_mxu_pack_rtl.py: 36 passed); the cut pi0.5 phases at MCOLS 8 packed against MCOLS 16 unpacked (section 7's table). Left: the OOC fmax on the nightly host, then MCOLS 16 -> 32 physical |
| L2 | **ACT_ROWS 16 -> 255**: an MM's row count is an 8-bit field (isa.mm: 0 < M < 256), and 255 rows reach 256's pass count over the prefix, ceil(558 / 255) = ceil(558 / 256) = 3 and ceil(968 / 255) = 4, so **no ISA change**; a 16-bit M (QST's rows already are) stays an option for M >= 256 | weight passes over the prefix: 3 at Np 558 instead of 70 today (TMEM caps the row group at 8; section 3) | **width-bound, not capacity-bound**: otpu_actram reads one D-byte block a column every cycle (`r_data` is MCOLS x D x 8 bits), 16 RAMB36 a physical column today (4 pieces x 4) beside its DSP column; 32 physical columns (64 logical, packed): **512 RAMB36** of the part's 2,016; 64 unpacked would be 1,024 (or 960 URAM at 72 bits a port: every URAM on the part), which is why L1 packs weights, not activations; rows are free up to the RAM's depth (512). Pumping the ACT RAM at 2x as TMEM_PMP does halves it, another clock-domain item | **ISA tally (done)**: the prefix's 255-row group needs TMEM 1.5M words (at 1M: "TMEM exhausted", the group stays 128 and nothing changes); with it port B falls 0.65 -> 0.51 s a chunk at MCOLS 64 (the prefix 0.40 -> 0.26 s), 0.69 -> 0.55 s at MCOLS 32, the MXU bound unchanged. **RTL (done, measured)**: otpu_actram holds ceil(255 / MCOLS) groups (its "ROWS a multiple of MCOLS" check was the only change), the ACT RAM at 255 rows x 16 / 32 columns bit-exact in tests/test_rtl.py (M up to 255 over 8 / 16 groups), the prefix's 255-row group runs on the RTL (p_qkv at MCOLS 16: 2.16 M cycles, 1.51x the MXU's) |
| L3 | **a fused activation unit in the MXU drain / quantizer**: GELU-tanh / SiLU (table + polynomial, as SGMV's sgmv_act), gate x up, then quantize for the next MM (SGMV's MLP_FUSE); gate and up rows interleaved in the image so a drain yields pairs; one unit per drain lane | removes the prefix MLPs' VPU work: GELU 0.78 s of π0.5 today, 0.14 s at LANES 64 (tally): ~6% of the serial sum at the f2-vla point. Still cheap; built after L4-L6 and sized on the measured remainder | est. ~4K LUT, ~10 DSP, 3 BRAM a lane, 8-16 lanes | bit-exact against an ISA definition of the fused op |
| L4 | **VPU lanes 8 -> 64** (LANES; TMEM banks with them) for softmax over long rows, norms, RoPE, residual adds. **Done in the RTL (measured, docs/pi05.md 5.4)**: the isum_64 partial loop (docs/isa.md "Sums": 64 partials, each a chain of fp32 adds) is RL = 64 / LANES chunks long through the 4-cycle adder, so LANES was capped at 16; now the reductions RSUM / RSSQ / RDOT run on a 16-lane sub-datapath whatever LANES is (otpu_vpu RLN: a LANES-wide chunk held and shifted 16 lanes a sub-step into the loop, KS = LANES / 16 sub-steps with KS - 1 issue bubbles, the loop, the tree and the row sums seeing exactly a 16-lane VPU's stream: the same bits), the elementwise VOPs and RMAX / ARGMAX at LANES a cycle, the composites at VPU_CL (16 in the target). Cycle cost: sums rows x ceil(cols / 16) at 16 / 32 / 64 lanes, the rest rows x ceil(cols / LANES). The alternative, four rows interleaved through a 64-lane loop (the sums at LANES a cycle), is the next step if their share demands it (23-30% of p_qkv's, e_step's and p_attend's VOP cycles by the tally, 1-5% of v_out's and p_out's). Bit-exact: test_vops at LANES 32 / 64 (every VOP), the lane-count kernels at 32 / 64, the board-uarch (replaying TMEM, WBUF) cases at 64. The 64-lane f2-vla build needs Verilator a 64 MB stack (rtlsim.build) and simulates ~70x slower than the 16-lane one (~42 cycles a second: open). Measured on v_out at the target: the VPU's tally 718,912 -> 179,728 and the MXU the slowest unit, the VPU still busy 72-79% (the composites at VPU_CL 16, the sums at 16) | 8x element throughput: π0.5's ~92 M softmax elements and norms in ~0.05 s | LUT est. ~150-200K (fp32 lanes), DSP ~300-500 | the VPU tests at LANES 16, 32, 64: **passed** (tests/test_vops.py, test_rtl's lane_count) |
| L5 | **TMEM 64K -> 1.5M words** (6 MB in URAM), banked by LANES; the drain's MCOLS outputs and QACT's row reads at 64 words a cycle. **Measured (RTL, 1.5M words, docs/pi05.md 5.3)**: the one-copy TMEM takes 1.5M words as it is; the bank-conflict counters said the MXU drain was the unit that loses (v_out at MCOLS 16 denied 756 K of 2.37 M cycles). **The drain's arbitration (done, measured, docs/pi05.md 5.4)**: otpu_slice `TMEM_WARB` orders the VPU / MXU-drain write pair (0 the VPU first as before, 1 round-robin on a conflict, 2 the drain first, 3 the drain first while its result FIFO is half full; `+tm_warb` in simulation, TMEMSTAT's pair counters wv / wm / wc / wcm / wcv); the target takes 2: p_out at MCOLS 16 (the MXU-bound phase) 5.86 -> 4.85 M cycles (-17%, the MXU 56 -> 68%) where round-robin gave -1.3%; the VPU-bound phases at 16 lanes lost 1-3% to it, at 64 lanes v_out does not (627,586 against round-robin's 632,495). **And the drain's rate**: with PACK a pass yields two result rows and E = MCOLS / 2 drained them in four steps, the drain's ideal cycles 1.13x the MXU's pops on v_out at MCOLS 32, 2.15x on p_attend; the target's `MXU_E 32` (a pass in two steps) halves it (0.57x, 1.11x). What is left on v_out at the target (MXU 56.5%): the two steps a pass, the VPU's writes (449 K of 628 K cycles) and the QACT gaps | prefix tiles of 256 rows x 2048 fp32 (2 MB) in one pass; fewer DRAM round trips | ~170 URAM; a 64-way bank crossbar est. ~80-100K LUT | tiled prefill tests; RTL bank-conflict counters (TMEMSTAT); tests/test_rtl.py test_write_arbitration_policies_bit_exact: **3 passed** |
| L6 | **QACT / QST at LANES elements a cycle** (they feed 256-row stationary operands). **Done, measured**: the quantizer took LANES 64 by its parameters (D a multiple of LANES) and runs bit-exact at 64 lanes on the RTL (tests/test_quant_rtl.py test_quantizer_modes_rtl and test_quantizer_qst_t_rtl at D 128 / 64 lanes: QACT in every mode, QST, QST T; **passed**) now that the VPU builds past 16 (L4). The tally at 64 lanes: QACT 53,248 of v_out's 821,256 serial cycles at MCOLS 32 packed, 124,992 of p_attend's 923,184; QST's 286,720 on v_qkv is the port A word rate (L11), not the lanes | activations into ACT RAM as fast as the MXU drains them | with L4's lanes | the quantizer tests at LANES 64: **passed** |
| L7 | **memory: 8 HBM ports on the one slice** (NCH 8): they buy the 4 GiB and the lanes' headroom, **not bandwidth**: a slice at PORTB 1 takes one D-byte chunk a cycle, 25.6 GB/s at 200 MHz, 32 at 250, whatever the ports offer (docs/f2_hbm.md; docs/wide_port.md: "more HBM ports only help once a slice takes more than one chunk a cycle"), and the board delivers 78-89% of that (burst200). **PORTB 2** (the wide-port lanes built on ternary-bonsai: a named dependency of f2-vla with a merge point, built once there) is on π0.5's critical path: the expert step is port-bound in the tally (0.016 s port B against 0.014 s MXU at M 50 under 64 columns). A DMA lane for activation tiles / KV goes through otpu_native_dram's grant chain unless it gets its own channel path (the LD before an MM is not overlapped with that MM) | keeps the MXU fed at small M; activation traffic off the weight port | the HBM shell exists (f2-hbm: S=1 x 8); lanes est. +~10K LUT | RTL timing on the 8-port memory model; the lanes' tests from ternary-bonsai |
| L8 | **clock 200 -> 250 MHz** (SGMV closed 250 on the same part at 20K LUT; today's full f2 build is 145K LUT at 200) | 1.25x, treated as upside: every number below is at 200 MHz | timing work, nightly | synthesis / route (nightly only) |
| L9 | **W8A16 where accuracy needs it** (docs/pi05.md 4: W8A8 moves π0.5's actions by 6.8% of their range, the 2B at W8A16 1.7%; RhinoVLA's paper: W8A8 hurt success) | accuracy; int8 x int16 is one MAC a DSP: half L1's rate if L1 packs, the same rate unpacked (decided by phase 1) | DSP or a second weight path | action error vs fp32 (docs/pi05.md harness) |
| L10 | **model-side**: the 3 views as MXU rows of one pass, the prefix KV once a chunk, the expert's time conditioning precomputed (docs/pi05.md), fewer visual tokens (RhinoVLA: 64 a view) | π0.5 -> RhinoVLA halves the MACs | none | ISA |
| L11 | **QST T (done, measured)**: the transposed V stores through a four-row transposer in the quantizer (docs/isa.md QST flag `T`: pass 1 at LANES elements a cycle into two halves of LANES x 4 words, each group of four rows flushed a word a column on port A while the next group's passes run; the compiler sets it wherever drs = 1 and es is a multiple of 4; bit-exact against the ISA simulator in tests/test_quant_rtl.py at every dst alignment, row count and HALF, and in the attention tests). Before it QST wrote a byte a cycle for the V^T stores (es the column stride) and a word for the K stores; now both a word: the tower's QKV phase on the RTL 1.306 -> 0.751 M cycles at MCOLS 16, 1.265 -> 0.710 M at MCOLS 32; the chunk's QST 0.21 -> 0.09 s (tally), the tower's 0.181 -> 0.077 s, under the MXU at MCOLS 64 (0.137 s). The other two options were not free: V untransposed needs the P.V MM to stream V's K dimension (tokens) from DRAM, which the MXU's row layout cannot, or Wv stationary with X streamed, and the ACT RAM has no int8 load path for it. **What is left**: at MCOLS 32 the QKV phase is still QST-bound (0.710 M cycles against 0.249 M of MXU; K and V are 64 port A words a token and head), so the next step is the stores as D-byte chunks on the write path | the vision tower's bound at scale: was QST 0.181 s against 0.137 s of MXU at the f2-vla point; now 0.077 s | the transposer: 2 x LANES x 4 words of flops (16 Kbit at LANES 64), a LANES:1 word mux | the quantizer tests; RTL timing of the tower |

## 6. The VLA variant `f2-vla` (one slice; target, estimate)

S = 1, D = 128, MCOLS = 32 physical columns, 64 logical with the packing (exact, phase 1; its
fmax from the nightly's OOC run decides packed 64 against unpacked 32), ACT_ROWS = 255,
ACT_BLOCKS = 128, TMEM_WORDS = 1.5M (the prefix's 255-row group needs it; L2), LANES = 64,
QST T (L11, done), the fused activation unit,
NCH = 8 HBM ports, PORTB 2, core 200 MHz (250 as upside). **The target exists**
(targets/f2-vla.toml, CFG_HASH 074e2423: PORTB 1 for now, no streams, no t2, the MXU's
epilogue lanes MCOLS / 2) and builds in Verilator at MCOLS 16 and 32 with ACT_ROWS 255 and
TMEM 1.5M words, at 16 lanes (L4). Peak: MCOLS 32, 4096 MAC/cycle x
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
| π0.5 a chunk | ~15-22 s (section 3) | tally (ACT_ROWS 255, LANES 64, TMEM 1.5M, QST T): MXU 1.90 s, port B 0.55, VPU 0.26, QST 0.09; bound 1.9 s, serial 2.96 s: **0.3-0.5 Hz** | tally (ACT_ROWS 255, LANES 64, TMEM 1.5M, QST T): MXU 0.95 s, port B 0.51, VPU 0.26, QST 0.09; bound 0.95 s, serial 1.96 s: **0.5-1.05 Hz** (the RTL's overlap on cut phases, 1.10-1.54x the slowest unit at MCOLS 4, puts it at ~0.7-0.9 Hz) |
| π0.5, measured RTL overlap at the f2-vla sizes (docs/pi05.md 5.3; 16 lanes) | 1.10-1.54x, MXU 65-91% (p_out 68%) | MCOLS 16: **p_out 1.23x, MXU 81%**; the others 1.51-2.13x, MXU 42-66% (v_out 60%, p_qkv 66%, v_qkv 66% with T, e_step 52%) | MCOLS 32 unpacked: p_out 1.67x, MXU 60%; the others 1.97-2.88x, MXU 23-50% (p_qkv 50%, v_out 36%, v_qkv 35%, e_step 32%); the VPU at 16 lanes the longest unit of three phases, the MXU drain's TMEM arbitration the loss on every one |
| π0.5, measured at the target after phase 4's first round (docs/pi05.md 5.4; LANES 64, PACK, MXU_E 32, TMEM_WARB 2) | - | (MCOLS 16 / 32 packed: the other phases running at the time of writing) | MCOLS 32 packed, 64 lanes: **v_out 1.77x, the packed MXU 56.5%** (the VPU busy 79%: its composites at VPU_CL 16 and sums at 16 a cycle); the exit criterion's 15% holds for p_out by the tally (13%), not for v_out (28%), p_qkv (23%), e_step (27%), p_attend (28%), v_qkv (59%: QST) |
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
| 1 | **the packing's feasibility** (L1): **done** in Verilator (bit-exact, docs/mxu_packing.md); left: the packed and unpacked columns out of context on the nightly host (DSP, fmax). ACT_ROWS 255 in the tally (done: needs TMEM 1.5M words, L2) and the RTL (**done**: otpu_actram's groups, the replay tests at 255 rows). QST: the K stores confirmed a word a cycle on the RTL; the V stores **QST T, done** (L11: -42% on the tower's QKV phase at MCOLS 16) | **MCOLS 64 packed or MCOLS 32 unpacked**: the fmax a column; the tally at ACT_ROWS 255; QST off the vision tower's critical path (at MCOLS 32 it is still on it: the port A word rate) |
| 2 | the ACT RAM at the chosen MCOLS: BRAM / pumped / URAM from the width math; a floorplan sketch per SLR (columns, the result FIFO, the drain) | the BRAM budget; the SLR plan |
| 3 | L1 MXU columns: MCOLS 8 -> 16 -> 32 (-> 64) (the RTL parameter exists: MCOLS 4 / 8 builds today) with ACT_ROWS 255. **The packing is in the array** (`otpu_mxu PACK`, merged from vla-pack, docs/mxu_packing.md 7: bit-exact at MCOLS 4 / 8 physical = 8 / 16 logical, ACT_ROWS to 64, tests/test_mxu_pack_rtl.py 36 passed; left: MCOLS 16 / 32 physical PACK builds and the OOC fmax). **Measured (unpacked, docs/pi05.md 5.3)**: MCOLS 16 and 32 build and run bit-exact at ACT_ROWS 255 (tests/test_rtl.py); the RTL timing of the cut phases at Np 558 (p_out at 128) gives the MXU 81% of the cycles on the prefix's MLP at MCOLS 16 (60% at 32) and 32-66% on the other phases, and names the loss: the MXU drain denied the TMEM grant while the VPU runs (the result FIFO then fills; the loss follows the VPU's share of the time), the dependency gaps on QACT, and the 16-lane VPU (L4) | bit-exact at each MCOLS (**done** for 16 and 32); RTL timing on the **real prefix shape** (558-968 rows, the 16384-wide MLPs, the 2048-wide projections; a cut model hides the drain's and result FIFO's losses): MAC/cycle ~MCOLS x 128 at >= 75% (**met by p_out at MCOLS 16, 81%; not by the rest**: 52-66% at MCOLS 16, 32-60% at 32; phase 4's two items are what stands in the way) |
| 4 | L4 / L5 / L6: LANES 64 (the VPU's partial loop past 16 lanes: **done**, the sums on a 16-lane sub-datapath, bit-exact at 32 / 64), TMEM 1.5M words (**builds and runs**), QACT / QST at LANES a cycle (**done**: the quantizer bit-exact at 64 lanes); **and the MXU drain's TMEM arbitration** (**done**: TMEM_WARB 2 in the target, p_out at MCOLS 16 -17%; and its rate, MXU_E 32: a packed pass in two steps) | VPU / QACT tests (**passed** at 32 / 64); bank-conflict counters in the RTL timing (TMEMSTAT, the pair's wv / wm / wc / wcm / wcv); VPU + QACT + QST < 15% of the serial sum: **met by p_out alone** (13% by the tally at the target; v_out 28%, p_qkv 23%, e_step 27%); the MXU >= 75% of the cycles on v_out / p_qkv / p_out at MCOLS 32: **not met** (v_out 56.5% measured at 64 lanes; p_qkv and p_out at 64 lanes running at the time of writing). Next: the drain at one step a packed pass, VPU_CL 64 or the fused unit (L3) for the composites, the sums' four-row interleave |
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
