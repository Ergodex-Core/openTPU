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
  **56.5%** (36% unpacked at 16 lanes under the old order); e_step 466,936 cycles (-47%
  against the same build at 16 lanes), 2.34x port B's 199,264 (the step is port-bound: L7);
  **at MCOLS 16 packed (32 logical columns) v_out keeps the MXU at 80.9%** (876,803 cycles,
  1.24x; the MXU never blocked, the VPU busy 74%), e_step 51.7%; at 16 lanes the arbitration alone
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
| L1 | **MXU columns MCOLS 4 -> 32 physical, 64 logical with DSP48E2 int8 packing**: one activation row x two weight rows a DSP (A = w_a, D = w_b << 19, the pre-adder's AD = w_a + w_b 2^19, B = the activation), a physical column yielding two logical output columns a cycle. A CL = 16 chain's lane sum is at most 16 x 127^2 = 258,064 < 2^18 (quant.py's int8 never holds -128), the lanes split at the chain end (L = sext19(P[18:0]); H = P[47:19] + P[18], the borrow absorbed by the high lane's column end on its pre-adder: no fabric, no cycle), before the column end's accumulation over the row's blocks (where a packed lane overflows at the second chain already). PAIR's A / D select survives as a third INMODE value. The repo's "inexact after 3 accumulations" was the DSP48E1's 25-bit pre-adder (a 17-bit lane); it already had a bit-exact packed DSP48E2 column, rtl/exp/mx_mxu (docs/mx_mxu.md 5) | MCOLS 32 unpacked: 8x, 4096 MAC/cycle; 32 packed (64 logical): 16x, 8192 | 146 DSP a physical column (128 PEs + two column ends), **73 a logical column** against 137 today; MCOLS 64 logical = 4,672 + ~192 epilogue lanes ≈ **4.9K of 9,024**; MCOLS 32 unpacked 4,480; 64 unpacked would be 8.8K, the whole part. LUT est. ~250-320K. W8 only: 4-bit PAIR / QUAD, t2 and W8A16 stay one product a DSP. The integration's extra fabric a physical column (docs/mxu_packing.md 7): the second bm shifter and its skew (8 x 17 bits x (CL + MCOLS) SRL stages), the second 48-bit deskew, two more 24-bit chain-end buses into the column ends; and once a slice: the chunk FIFO's second bank (32 more RAMB36 at FIFO_DEPTH 1024), a 40-bit scale bank, a 2 x DEPTH write-address queue (~400 LUT of LUT RAM at DEPTH 1024). **LUT / FF deltas not measured here**: the Mac has no slang-capable yosys for this SystemVerilog and Verilator reports no cell counts; the DSP counts and the LUT / FF of a packed against an unpacked column are the nightly's OOC run (`tools/mxu_pk_ooc.sh`) | **Phase 1 done (docs/mxu_packing.md)**: `otpu_pe_pk` (generic and DSP48E2), `otpu_mxu_pk_col`, tests/test_mxu_pk.py in Verilator: 9 passed, 0 mismatches against the ISA's integer over the corners (every sign of +-127 at the lane bound, the borrow, K = 16384, random rows), and the two failing patterns pinned (sixteen (-128)^2 in a lane; w_b = -128 beside a negative w_a wraps the 27-bit pre-adder): both need an operand the quantizer never produces. **Phase 3 done in Verilator (branch vla-pack, docs/mxu_packing.md 7)**: `otpu_mxu PACK` (Config.PACK, CAPS2 bit6), bit-exact with the ISA simulator at MCOLS 4 / 8 physical and ACT_ROWS to 64 over int8 in every mode, replay, the 4-bit formats and t2 unpacked beside it, odd N, KB 1 (tests/test_mxu_pack_rtl.py: 36 passed); the cut pi0.5 phases at MCOLS 8 packed against MCOLS 16 unpacked (section 7's table). The OOC fmax (done 2026-10-17): packed WNS +1.986 / +1.010 ns at 5.0 / 4.0 ns, 146 DSP48E2, 2,642 LUT, 6,692 FF a column; unpacked +2.021 / +0.986, 137 DSP, 1,738 LUT, 4,372 FF. Left, then MCOLS 16 -> 32 physical |
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
the OOC run passed on 2026-10-17: +1.99 ns at 200 MHz, +1.01 at 250; packed 64 logical it is), ACT_ROWS = 255,
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
| π0.5, measured at the target after phase 4's first round (docs/pi05.md 5.4; LANES 64, PACK, MXU_E 32, TMEM_WARB 2) | - | (MCOLS 16 / 32 packed: the other phases running at the time of writing) | MCOLS 16 packed (32 logical), 64 lanes: **v_out 1.24x, the packed MXU 80.9%**, e_step 1.93x / 51.7%. MCOLS 32 packed (64 logical), 64 lanes: **v_out 1.77x, 56.5%**, p_qkv 2.28x / 44.0%, e_step 2.34x / 33.2% (port-bound), p_attend 1.72x / 24.0% (port-bound), v_qkv 2.05x / 21.2% (QST-bound) (the VPU busy 66-79%: its composites at VPU_CL 16 and sums at 16 a cycle); the exit criterion's 15% holds for p_out by the tally (13%), not for v_out (28%), p_qkv (23%), e_step (27%), p_attend (28%), v_qkv (59%: QST) |
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

### 6b. First synthesis of f2-vla (measured, 2026-10-17: the design does not fit)

SynthCL on the F2 host (Vivado 2025.2, vla 22ade69b + `TMEM_SBP = 6`, run dir
/home/ubuntu/opt/hwloop-nightly/vla-22ade69b/synth, 2 h 34 min, `OTPU_STAGE_SYNTHESIS_PASSED`,
0 errors). Post-synthesis utilization against the budget above:

| resource | used | available | of the part | budget above |
|---|---|---|---|---|
| CLB LUTs | 2,253,192 | 1,303,680 | **173%** | 0.6-0.8M |
| of them logic | 1,035,402 | 1,303,680 | 79% | |
| LUT as memory (distributed RAM 1,140,236 + SRL 77,554) | 1,217,790 | 600,960 | **203%** | |
| registers | 555,937 | 2,607,360 | 21% | |
| RAMB36 | 2,297 (2,819 before the spill) | 2,016 | **114%** | 512 + FIFOs |
| URAM | **0** | 960 | 0% | ~345 |
| DSP48E2 | 5,876 | 9,024 | 65% | ~4.9K |

By hierarchy (LUTs / LUTRAM / RAMB36 / DSP): `u_act` (ACT RAM, 255 rows x 128 blocks, 4 MB,
otpu_ram_1r1w_be STYLE "auto") 1,149,727 / **1,086,464** / 0 / 0: Vivado announced URAM for it,
then `[Synth 8-5835] BRAM overutilized (5,638 of 4,032)` and implemented it as 131,088
RAM256X1D; `u_tmem` (1.5M words, 64 lanes) 424,760 / 0 / **2,048** / 0: the pumped banks
(otpu_pump_bank, STYLE "block", 64 banks x 2 sub-banks x 16 RAMB36) alone exceed the part's
2,016 RAMB36, and the 425K logic LUTs are the 64-lane x 8-port crossbar (f2-hbm16's TMEM:
10.3K LUTs, 64 RAMB36); `u_mxu` (32 packed columns) 299,960 / 9,508 / 62 / 4,897; `u_vpu` (64
lanes) 205,815 / 5,046 / 0 / 576; `u_quant` 60,104 / - / 117 / 400. Timing estimate (no
placement): clk_raw 200 MHz **-0.298 ns** (31 endpoints: the quantizer's 64-lane max / compare
tree, `u_quant/ca_reg -> cmax_reg`, 20 levels), clk_f_raw 400 MHz +1.10, hclk_raw 450 MHz
+0.15, clk_main_a0 +2.49.

What it means: the memory mapping in the budget (ACT RAM and TMEM in URAM) is not what the RTL
asks for (`STYLE "auto"` on the ACT RAM, `"block"` on the pump banks), and even with both RAMs
in URAM / BRAM the **logic alone is 79% of the part's LUTs** (the TMEM crossbar 425K, the MXU
300K, the VPU 206K): MCOLS 32 packed x LANES 64 x 8 TMEM ports does not fit one VU47P with
the HDK shell, before timing. The OOC packed column (docs/mxu_packing.md 9) still holds for the
column itself. The options and the cut taken are 6c; the decision was the user's (D14 A: cut
at MCOLS 32 before the fallback B, MCOLS 16 / LANES 32). The
f2-hbm16 (MCOLS 4) run for comparison: 241K LUTs, 182K FF, 261 RAMB36, 727 DSP, clk_raw +1.10.

### 6c. The cuts for MCOLS 32 (2026-10-18; estimates for the second synthesis to confirm)

**The memory mapping** (otpu_slice `ACT_RAM` / `TMEM_RAM`, target uarch keys; the other
targets state nothing and keep today's "auto" / "block"):

| memory | geometry | 6b (measured) | 6c (estimated from the geometry) |
|---|---|---|---|
| ACT RAM (u_act), `ACT_RAM 1` | 32 columns x 4 pieces of 1,024 words x 256 bits with byte enables, + 32 scale records of 1,024 x 64 | 131,088 RAM256X1D (1,086,464 LUT RAM) + ~63K LUTs of decode, 0 RAMB36 | **1,024 RAMB36** (8 of 1K x 36 a piece) **+ 64** (2 a column), ~10K LUTs (the 64-lane byte placement, the per-piece address copies; f2-hbm16's u_act is 718 at MCOLS 4) |
| TMEM pumped banks (u_tmem), `TMEM_RAM 2` | 64 banks x 2 sub-banks x 12,288 words x 32 bits, true dual-port on clk_f 400 MHz | 2,048 RAMB36 (16 a sub-bank: Vivado rounds 12,288 to 16K) | **384 URAM288** (3 of 4K x 72 a sub-bank, cascaded, 32 of the 72 bits used), 0 RAMB36 |
| the rest (u_mxu 62, u_quant 117, IMEM / trace / FIFOs / shell ~70) | | 249 RAMB36 | 249 RAMB36 |
| **total** | | RAMB36 2,297 (114%), URAM 0, LUT RAM 1,140K (190% of the 601K LUT-RAM sites) | **RAMB36 1,337 (66%), URAM 384 (40%), LUT RAM ~54K (9%)** |

The URAM's read timing: otpu_pump_bank's reads are issued at the even fast edge and their
words must hold through the next core cycle, a two-fast-period multicycle both ways
(boards/ypcb-00338/constraints/otpu_clk2x.tcl: `ra / rena / rb / renb` into clk_f, `qa / qb`
out of it, 5.0 ns each); the URAM without its output register (OREG off, what `if (en) q <=
mem[a]` infers) reads in one clock like the block RAM, with a longer clock-to-out (~1.5-2 ns
against the RAMB36's ~1 ns) and a 3-deep cascade's mux in the same 5 ns window: no extra cycle,
the TMEM latency contract (a read's word the cycle after) holds. The write lands at the odd
fast edge from the bank's own `pk_*` registers, one fast period as before. A URAM cannot be
initialised (the simulation's `initial` is under `ifndef SYNTHESIS`), and Vivado's URAM
inference wants each port read-or-write a cycle with the output held on a write, which is the
bank's NO_CHANGE form. What synthesis has to confirm: that Vivado infers the 128 TDP RAMs as
URAM (the log's URAM count 384; a fall-back to block RAM would read 2,048 RAMB36 again) and the
ACT RAM's byte-enable pieces as 8 RAMB36 each (1,088 in u_act).

**The TMEM crossbar.** The slice's 8 read ports and 4 write ports (otpu_slice P_* / W_*; all
64 lanes x 32 bits): the MXU (read: ACC's old values, ASCALE's factors, write: the drain's
results, E = 32 lanes a step, and RMAX's maxima), the DMA (LD writes / ST reads, 32 words a
cycle), the quantizer's three (QACT / QST's rows at 64 a cycle, the second operand, the
one-word row factor), the VPU's two operands and its write, the collective's pair (RLD's one
word at S = 1). Only the MXU's two were general crossbars (`GEN_R` / `GEN_W`: its drain writes
the M results of a streamed row `ors` words apart, scattered over the banks); the other ten are
barrel rotators (a contiguous run of words, LANES x log2 LANES multiplexers). The structure's
LUTs at 64 lanes, calibrated on f2-hbm16's whole TMEM (10.3K measured against 12.3K by the same
count at 8 lanes): a 64-lane rotator port ~6K (6 stages x 64 x 32 bits, two 2:1s a LUT6) + 2K
for the fresh / held select; a general read port ~43K of 64:1 lane selects + ~25K of 64 x 64
bank-row AND-OR; a general write port ~70K (64 banks x 64 lanes x 47 bits of AND-OR); the IMPL
3 arbitration, the sub-bank data selects and the write merge ~35K. That puts the ten rotator
ports and the arbitration at ~120K and the MXU's pair at ~140K by the count, which the measured
424,760 says Vivado builds nearer 300K (the 64 x 64 compare-and-OR trees and the 2,048 64:1
selects are what grows 64x from the 8-lane TMEM). The options, by that count:

| option | the MXU's pair (LUTs, est.) | the rate |
|---|---|---|
| as synthesized (two 64 x 64 crossbars) | ~300K (measured, by difference) | the drain's 32 lanes a step at any odd stride |
| A: lane-group ports (a port touches one 16-lane group a cycle: four 16 x 16 crossbars) | ~75K | the drain's 32 lanes span two groups: two steps a row, the drain (already 0.57x the pops on v_out) at 1.14x, the MXU-bound phases -20..-30% |
| B: fewer full-width ports (the MXU's reads as the DMA's guest; the write crossbar stays) | ~150K | ACC / ASCALE / RMAX reads behind the DMA's LDs; the drain's rate kept |
| C: a two-level crossbar (4:1 group select x 16:1 within the group, both ports) | ~100K | any permutation, the rate kept |
| **D (taken): rotators whose lanes carry their rows** (otpu_tmem `ROW_R` / `ROW_W`) | **~20K** (the rotator moves {enable, row} to the banks and the data back: 6 stages x 64 x (15 + 33) bits) | the drain's 32 lanes a step at output row strides = 1 mod LANES, which every π0.5 MM already had at the widths that are multiples of 64 (257, 513, 1153, 2049: the compiler's odd-stride rule gave N + 1); the compiler's rule under `MXROT` is the first stride >= N that is 1 mod LANES (an attention's 558 keys: 577, was 559; the expert's 608: 641); any other stride drains one lane a step (bit-exact, slower) |

D works because the drain's lane k writes word `a0 + k x ors`, whose bank is `(a0 + k) mod
LANES` exactly when `ors = 1 mod LANES`: the rotation pattern of a contiguous run, with the
rows differing per lane. The TMEM takes the rows along the rotator instead of computing them
from a0; the MXU's `drain_run` is its NL lanes at such strides and 1 otherwise (otpu_mxu
MXROT); RMAX, ASCALE and ACC's lanes were contiguous already. It is an arch flag (Config.MXROT,
CAPS2 bit7, in the CFG_HASH when on: the programs' layout differs, every result is the same;
targets/f2-vla.toml alone sets it) so that f2 / f2-hbm's programs and netlists do not change.
Bit-exact against the ISA simulator: tests/test_rtl.py's random MMs at strides N and N + 1
with MXROT forced on every Config (the rotation pattern and the one-lane fallback), the
f2-vla-size suites on the m7a (the step-2 commit's numbers). The LPN (tools/pi05_lpn.py, the
same programs under the new rule: v_out 602,200 cycles against 602,164 before, p_attend
637,861 against 637,878 at the 577 stride, e_step 452,718 against 452,618) sees no rate
change from the strides. **Measured** (tools/pi05_rtl.py on the m7a, the 5.4 configuration,
docs/pi05.md 5.4 "after the 6c cuts"): v_out 627,705 cycles against 627,586 (+0.02%), v_qkv
592,802 against 587,037 (+1.0%, the quantizer's cycle a row), p_attend 722,232 against 718,667
(+0.5%), e_step 467,218 against 466,936 (+0.06%). The first RTL (step 1) had p_attend +10.4%
and e_step +5.2%: the attention's last key block (46 keys at Np 558) came from the language's
empty / zeros / full at stride 47, not 1 mod 64, and its 16 MMs drained one result a step;
compiler.mm_stride (step 2) is now the one rule for every tile the MXU drains into.

**The quantizer's tree** (clk_raw -0.298 ns in 6b: `u_quant/ca_reg -> cmax_reg`, 20 logic
levels, the 64-lane max in two registered stages of 1 + 5 compare-and-select levels): three
stages now (pairs; eights; the rest, 3 levels at 64 lanes), the group maxes taken at the level
that completes them and carried, 12 logic levels at most on that path. A row's scale, and so
its pass 1, comes a cycle later: a QST's rows run one after the other (+1 cycle a row), a
QACT's pipeline through the row buffer (+1 once). In the LPN (`q_row 1`, the new knob) the
chunk is +0.08% (v_qkv, the QST-bound phase, +0.6%: 617,491 against 613,633 cycles; the other
phases within 0.1%).

**The LPN before / after** (tools/pi05_lpn.py --full, the real model, 200 MHz; before =
`OTPU_MXROT=0 --q-row 0`, which reproduces 5.5's numbers): lpnlang 1.476 -> 1.477 s a chunk
(0.68 Hz, 295.2 -> 295.5 M cycles), the dialect 1.413 -> 1.414 s (0.71 Hz, 282.5 -> 282.8 M);
the tally's bound 0.951 s and serial 2.003 s unchanged (no unit's rate moved: the drain's lanes,
the VPU's, QST's word rate are what they were). So the cuts cost nothing the model sees; the
f2-vla Hz stays ~0.65-0.7 at 200 MHz, ~0.8 at 250 (docs/pi05.md 5.5).

**The pre-synthesis estimate** (CLB LUTs per hierarchy as Vivado reports them in 6b, logic +
LUT RAM + SRL; the part 1,303,680 LUTs, 600,960 of them LUT-RAM capable, 2,016 RAMB36, 960
URAM, 9,024 DSP):

| hierarchy | 6b (measured) | 6c (estimated) | what changed |
|---|---:|---:|---|
| u_act | 1,149,727 (1,086,464 LUT RAM) | **~10K** (6-15K) | the RAM in 1,088 RAMB36, the RAM256X1D decode gone |
| u_tmem | 424,760 | **~140K** (120-165K) | the MXU's pair ~300K -> ~20K (D); the ten rotators, the arbitration, the sub-bank selects stay (~120K by the count, x0.85-1.2 for Vivado) |
| u_mxu | 299,960 (9,508 LUT RAM) | 300K | unchanged (the packed columns 32 x 2,642 = 85K, the drain's lane picks, RMAX, the FIFOs) |
| u_vpu | 205,815 (5,046 LUT RAM) | 206K | unchanged |
| u_quant | 60,104 | 60K | a register stage, no LUTs to speak of |
| the rest (DMA, sequencer, control, the HBM adapter, the shell's CL side) | 112,826 (by difference) | 113K | unchanged |
| **CLB LUTs** | 2,253,192 (173%) | **~830K (64%; 810-855K, 62-66%)** | |
| of them logic | 1,035,402 (79%) | **~735K (56%; 54-58%)** | the LUT RAM ~54K + SRL 77.5K are the rest |
| registers | 555,937 (21%) | 556K (21%) | +~1.3K in the quantizer |
| RAMB36 | 2,297 (114%) | **1,337 (66%)** | the ACT RAM in, the TMEM out |
| URAM | 0 | **384 (40%)** | the TMEM's banks |
| DSP48E2 | 5,876 (65%) | 5,876 (65%) | |
| clk_raw 200 MHz (post-synthesis) | -0.298 ns (the quantizer's tree) | the tree at 12 levels: expected positive; the next worst paths of 6b were not reported | clk_f_raw +1.10, hclk +0.15 as before; the URAM's clock-to-out sits in a 5 ns multicycle |

So MCOLS 32 packed reaches the goal on the estimate (logic ~56% of the part, memories in URAM /
BRAM at 40% / 66%): no fallback to B needed before the synthesis. What the synthesis has to
confirm, in order of doubt: u_tmem's LUTs after the cut (the 120-165K spread is Vivado's
handling of the rotators and the IMPL 3 arbitration at 64 lanes; above ~200K the next cut is
the ten rotators' fresh / held selects and the guest ports' lanes, Q3 and COLL use one word),
the URAM inference of the pumped banks, the ACT RAM's RAMB36 count, and clk_raw's next worst
path. The place-and-route questions of section 8 (the MXU's 32 columns and their 1,088 RAMB36
across three SLRs) are untouched by this section.

Repeating the estimate's inputs: the hierarchy numbers above are
/home/ubuntu/opt/hwloop-nightly/vla-22ade69b/synth's utilization report (6b); the structure
counts are otpu_tmem's at LANES 64, IW 15 (24,576 words a bank), NSB 2, NRP 8, NWP 4.

### 6d. Second synthesis of f2-vla (measured, 2026-10-18: it fits)

SynthCL on the F2 host (Vivado 2025.2, vla 92197967 = 6c's RTL, run dir
/home/ubuntu/opt/hwloop-nightly/vla-92197967/synth, synth_design 1 h 40 min, 0 errors, no
"BRAM overutilized"). Against 6b and 6c's estimate:

| resource | 6b (first synthesis) | 6c estimate | **6d measured** | of the part |
|---|---|---|---|---|
| CLB LUTs | 2,253,192 (173%) | ~830K | **931,960** | **71.5%** |
| registers | 555,937 | | 540,250 | 20.7% |
| RAMB36 | 2,297 (114%) | 1,337 | **1,353** | 67% |
| URAM | 0 | 384 | **384** (the 128 pumped banks, 3 cascaded each) | 40% |
| DSP48E2 | 5,876 | 5,876 | 5,876 | 65% |

By hierarchy (LUTs / logic / LUTRAM / SRL / FF / RAMB36 / URAM / DSP): `u_slice` 846,205 /
756,424 / 15,446 / 74,335 / 480,640 / 1,316 / 384 / 5,876; `u_tmem` **297,991** (all logic; 6c
estimated ~140K: the ten rotators, the IMPL 3 arbitration and the two MXROT rotator ports cost
more than the structure count said; still 127K less than 6b's 425K), `u_mxu` 260,691 (was
299,960), `u_vpu` 206,140, `u_quant` 60,467, `u_act` 2,533 + 1,088 RAMB36 (was 1,149,727 LUTs),
`u_seq` 10,262, `u_dma` 7,671; the adapter `u_mem` 38,101, the HBM memory path 28,606, the host
path 1,963, control 813.

Timing estimate (post-synthesis, no placement): clk_raw 200 MHz **-0.739 ns** (2,777 endpoints),
the worst path `u_mxu/rx_reg[nv][0][30]` -> `u_mxu/mk_reg[24][0]/CE`, 23 levels (2 CARRY8, 10
LUT6): the MXU's mk / mkr family that round 1 of the wide-port branch's F2 build hit (docs/wide_port.md
"Placement": `mx_have -> mkr`, fixed there in fb42c14e, not on this branch); clk_raw ->
clk_f_raw **-0.390** (1,808): `g_coll.u_coll/asel_reg[0]` -> the pumped bank's
`mem_reg_uram_2/CAS_IN_ADDR_B[13]`, 18 levels through the URAM cascade under the 2-fast-period
multicycle (otpu_clk2x.tcl matched all 128 pumped RAMs here: 3,840 request pins); clk_f_raw
intra +0.79, hclk_raw +0.15, clk_main_a0 +2.49. Hold violations are the pre-placement kind
(-0.1 ns, 0-1 levels). Next: port the wide-port branch's mkr fix (a deliberate cross-merge,
the one exception to the branch rule), pipeline the collective port's address into the
pumped banks (or start it a cycle earlier), then one placement round (the floorplan from the
wide-port branch's cl_pnr_user_hbm.tcl, with pb_otpu_board at ~70% of SLR2 not 50%: the slice
is 846K LUTs against SLR2's ~435K, so the f2-vla floorplan must span two SLRs; phase 2's
sketch).

### 6e. After 6d: the two timing paths, the TMEM's 298K, the floorplan (2026-10-19; the RTL changes measured in Verilator, the LUT and level counts estimated from the structure)

**The RMAX column update** (6d's clk_raw -0.739 ns: `u_mxu/rx_reg[nv][0][30] -> mk_reg[24][0]/CE`,
23 levels, 2 CARRY8 + 21 LUTs). What the path was: per column j every drain lane k compared its
key against `mk[j]` and the 32 results (and 32 more for the ACC path's `ry`) were OR-ed into
`mk[j]`'s enable: a 32-bit magnitude compare (4 CARRY8) behind the lane's hit decode, then a
64-input OR. The wide-port branch met the same family on its RL word `mkr` (33 levels, 14
CARRY8) and made it a running max in two register stages (c5af9505); vla has no RL word, so
that commit does not apply and the form was re-implemented for `mk`: the cycle's candidate per
column is selected by the hits alone into a register (`mc_k`, `mc_v`: the hit decode `rx.col ==
j` 1 level, the AND with the key bit 1, the OR over 64 hit terms 2-3: ~5 LUT levels), and the
next cycle one compare per column makes the enable (`mk <= max(mk, mc_k)`: one 32-bit compare,
4 CARRY8, then `!mx_have` and `mc_v`: ~7 levels). Counted from the RTL: 23 -> ~7 on the deepest
of the two stages. Two hits on a column in consecutive granted cycles are the drain's normal
case and compare the second against the mk the first wrote. The cost: `mk` is final a cycle
later, so `c_drained` waits for `mc_v` and the RMAX writes of an MM start one cycle later (MMs
with RMAX: the attention's QK^T, 96 a p_attend run). **Measured** (tools/pi05_rtl.py on the
m7a, the 5.4 configuration, this section's RTL against 6c's): p_attend 722,238 cycles (722,232:
+6), e_step 467,218 (unchanged), v_out 627,705 (unchanged), v_qkv 592,802 (unchanged); the
suites at the f2-vla sizes 322 passed + 281 with MXROT forced on every Config (the 13
pre-existing failures of 6c's run, unchanged). No LPN constant moves.

The next deepest clk_raw cones counted from the RTL (no synthesis number yet): the slice's TMEM
grant, from the units' registered requests through otpu_tmem's conflict compares (`cp`: a 15-bit
row compare per asker pair and bank, 2 levels, OR-ed over 64 banks, 2-3) to `blk_x`, the
group's grant and the units' enables (~12-14 levels; r2.1's m8 had it at 14 on 8 lanes, +0.003
ns); the quantizer's offset `dval` (a 9-bit exponent subtract, a 23-bit mantissa compare and the
saturation, ~8) and its group-max fold `gnew` / `nmax` (one 31-bit compare and a select, ~6);
the drain's lane pick `dx_l[k] = rfh_i[dj + k]` (a 32:1 of 48 bits a lane, 3 LUT6 levels after
the head select, ~5). None is at 23; the grant cone is the one to watch at 64 lanes.

**The collective port's address into the pumped banks** (6d's clk_raw -> clk_f_raw -0.390 ns:
`u_coll/asel_reg[0] -> ... mem_reg_uram_2/CAS_IN_ADDR_B[13]`, 18 levels under the two-fast-period
multicycle). The port is a guest of the DMA's (GST): its request is `av && asel && am` (1
level) from otpu_coll's registers, then the 64-lane OR for the guests' take chain (2), the
enables' rotation (3-6), the host's `qe` select (1), the replay mux `ae` (1), the rank and the
RAM's address AND-OR (3-4) and the URAM cascade (2). At S = 1 the collective reads one word
(RLD's, lane 0) and never writes (no GATHER on one slice: the host refuses one, the board builds
the BAR-only unit), so under MXROT with `COLL_GATHER 0` the port is a one-word port (otpu_tmem
`ONE_R`: lane 0's bank decoded, 1 level, no 64-wide OR, no rotation; its data one lane) and its
write port is tied off (`NUL_W`); the quantizer's row-factor port (Q3) is one word on every
MXROT build. Counted from the RTL: 18 -> ~11 (the decode, the take chain's two ANDs on single
bits, qe, ae, the rank and AND-OR, the cascade); no latency change (the collective port still
reads the cycle after). The next port on that path is a full rotator's (the DMA's, the VPU's
two, the quantizer's two): the rotation (3 levels as 4:1 pairs, 6 as 2:1) and, on the non-ROW
ports, the bank row's `+ (b < rot)` incrementer (2 CARRY8) before the same tail: ~13-15 levels
by the count, so the second synthesis's -0.39 should become a small positive or a small
negative on a VPU or DMA address; if negative, the next step is the bank row's incrementer
(carry the rows per lane as the ROW ports do: +15 bits through the rotator, no adder).
`COLL_GATHER` is now a slice parameter (from otpu_top / otpu_board; the F2 sys tops take it as
an optional uarch key, f2-vla 0, so the bitstream builds the BAR-only unit the simulations
always built for one-slice FPGA targets).

**u_tmem's 297,991 LUTs against 6c's ~140K.** The count assumed each rotator port's read data
is a barrel (6 stages of 2:1 on 64 lanes x 32 bits, 3 LUT6 per lane and bit as 4:1 pairs: ~6K
a port). It was written as an indexed select per lane, `lq[l] = qb[(l + rq[l]) % LANES]`, with
the rotation copied per lane (`rot_q[l]`, the card's fan-out fix), which Vivado builds as a
64:1 multiplexer per lane and bit: 16 LUT6 (the F7/F8/F9 muxes free) x 2,048 = ~33K a port,
~262K for the eight read ports; with the write rotators (~22K: three ports of 33 bits as 4:1
pairs, the MXU's 48), the IMPL 3 arbitration and the sub-bank selects (~20K), the enables' and
rows' rotation (~10K) that is the 298K. At 8 lanes an 8:1 select is 2 LUTs a bit, which is why
f2-hbm16's 10.3K TMEM did not show it. The fix is the barrel (lane l's stage k takes lane l + 2^k
on its own copy's bit k; the same words), ~6K a port: u_tmem **~90-100K** by the count (49K of
read data, 22K writes, 20K arbitration, 10K enables / rows, -8K from the one-word collective
port's rotator, data and write port). The ten rotators' widths are the lanes' (64 x 32 bits:
every port reads its unit's full width each cycle) and the arbitration's per-port cost is the
4-asker AND-OR per RAM side: neither has a cheap cut left of the size the data path had; the
next LUT block is the VPU's 206K.

**The floorplan** (phase 2's sketch; the HDK's CL area is SLR2 X0Y8:X7Y11 all 32 clock
regions ~435K LUTs, SLR1 X0Y4:X3Y7 + X5Y1:X5Y7 ~313K, SLR0 X0Y0:X3Y3 + X4Y0:X7Y0 ~272K with the
HBM stacks below it; the wide-port branch's pblocks, boards/f2/build/constraints/cl_pnr_user_hbm.tcl
read unmanaged by its build_level_1_cl.tcl inside the HDK's opt_design, put the whole
226K-LUT slice in SLR2 (pb_otpu_board) with the TMEM + VPU in a child box, the HBM fabric in
SLR0 and three-stage otpu_xr chains on every net between them). f2-vla's slice is 846K LUTs in
6d, ~640K after the TMEM's barrel: it has to span two SLRs, and the only cut through the slice
that does not break a same-cycle loop is **the MXU's array against the rest**:

| side | what | LUTs (6d; after the barrel) | RAM / DSP |
|---|---|---|---|
| SLR1 (+ SLR0 below it for the HBM fabric, 30K, and the adapter, 38K) | the 32 packed columns (otpu_mxu_pk_col), the ACT RAM beside them, the chunk and scale FIFOs, the issuer (port A / B streams) | ~100K (the columns 32 x 2,642 = 85K, u_act 2.5K, the FIFOs and the issuer) | 1,088 + 62 RAMB36, 4,672 DSP |
| SLR2 | the TMEM and its arbiter, the VPU, the quantizer, the DMA, the sequencer and control, and the MXU's drain side (the result FIFO, the 32 epilogue lanes, the ACC read-modify-write, RMAX, the TMEM requests) | 298 + 206 + 60 + 18 + ~160 = **~740K (170%); after the barrel ~530K (122%)** | 384 URAM, 117 + ~70 RAMB36, ~1,200 DSP |

The buses across that cut and whether the RTL registers them today: the quantizer's ACT RAM
writes, registered twice in otpu_actram for exactly this hop (**yes**); port B's chunks and port
A's scale lines from the adapter into the MXU's FIFOs, through the slice's port registers
(**yes**); the sequencer's command and start (registers in otpu_seq) and the MXU's done (a
register) (**yes**), its `rdy` (`!i_act && !i_ini && q_n < 2`, one level into the sequencer's
can_start: **no**, a few bits); the array's result rows into the result FIFO (the chain ends'
accumulators to the three LUT-RAM copies' write, `rowp`, 2.6K bits a row at a time; the write
enable is a register, the data a select of two registered rows: **marginal**, one level and the
crossing); and inside otpu_mxu the issuer and the drain share the head command's entries and
the counters `rf_n`, `rows_live`, `c_left` with the drain's pop `d_pop = t_gnt && drain_row_done`
coming from the slice's grant the same cycle (**no**: an unregistered few-bit loop between the
two sides and the arbiter). So the split is not available as the RTL stands: the MXU would need
its issuer / drain handshake and the result FIFO's push registered (one or two cycles of drain
latency a row, hidden by the 64-row FIFO), and with that SLR2 still holds ~530K against 435K.
What makes SLR2 fit is a smaller VPU / TMEM, not a better cut: at **LANES 32** (MCOLS 32 packed
kept) the TMEM, VPU and quantizer halve (~45 + 103 + 30K) and SLR2 is ~340K (78%) with the MXU
split above, and the LPN (tools/pi05_lpn.py --full, lpnlang, OTPU_LANES=32) puts the chunk at
**1.549 s, 0.65 Hz** against 1.477 s / 0.68 Hz at LANES 64 (the VPU's elementwise VOPs and QACT
at 32 a cycle: +4.9%); the fallback B, MCOLS 16 / LANES 32, is 2.429 s, 0.41 Hz (the MXU
halved: +64%). The other direction, registering the VPU's, quantizer's and DMA's TMEM
interfaces so they can sit in SLR1 (the data return lands in registers in all three, `xa <=
ta_data`, `x1 <= t_rdata`, the DMA's two-stage read; the grant, `lost` and `frz` do not: a unit
would have to issue a cycle ahead and undo), is the bigger change. Recommendation: no placement
round at LANES 64 (it would place Vivado's own split with thousands of unregistered crossings,
as the wide-port rounds 1-2 did at -4.6 ns, and tell us nothing the count above does not); the
next synthesis with this section's RTL (the barrel, the two paths), then the decision between
LANES 32 (+5% on the LPN, the slice ~500K, the SLR2 box ~340K) and the registered-interface
work. The cl_pnr_user_hbm.tcl variant and the build mechanism's cross-merge wait for that
decision: at LANES 32 the pblocks are the wide-port file's with pb_otpu_board as the SLR2 side
above and a new pb_otpu_mxu (SLR1, the columns, u_act and the FIFOs by cell name) once the
MXU's internal crossings are registered.

### 6f. Third synthesis of f2-vla (measured, 2026-10-18: fits, and meets timing post-synthesis)

SynthCL on the F2 host (Vivado 2025.2, vla c2598e54 = 6e's RTL, run dir
/home/ubuntu/opt/hwloop-nightly/vla-c2598e54/synth, 0 errors, no overutilization):

| resource | 6d | **6f** | of the part |
|---|---|---|---|
| CLB LUTs | 931,960 (71.5%) | **718,265** | **55.1%** |
| RAMB36 | 1,353 | 1,353 | 67% |
| URAM | 384 | 384 | 40% |
| DSP48E2 | 5,876 | 5,876 | 65% |

By hierarchy (LUTs): `u_slice` 642,912 (was 846,205); `u_tmem` **124,593** (was 297,991: the
read data path as a barrel rotator, 6e); `u_mxu` 233,490 (was 260,691: the mk fix); `u_vpu`
205,531; `u_quant` 56,268; `u_act` 2,361 + 1,088 RAMB36.

Timing estimate (post-synthesis, no placement), every clock positive: clk_raw 200 MHz
**+0.110 ns** (was -0.739: the mk path), clk_f_raw 400 MHz +0.788, hclk_raw 450 MHz +0.154,
clk_main_a0 +2.489; the clk_raw -> clk_f_raw table has no violation (was -0.390: the
collective port's address); one -0.003 ns clk_f_raw -> clk_raw path (a pumped bank's URAM
clock to an MXU DSP input) and the usual pre-placement hold nits. So the VLA variant fits the
VU47P with margin and starts placement with positive synthesis slack on every domain.

What placement needs (6e's verdict stands): the slice is 643K LUTs against SLR2's ~435K of CL
area, so it cannot sit in one SLR as the wide-port branch's floorplan puts its 226K slice;
the register-clean cut found at 64 lanes is the MXU's array (~100K) against the rest (~540K,
still above one SLR). The options: LANES 32 at MCOLS 32 packed (TMEM / VPU / quant halve, the
rest ~340K = 78% of SLR2; LPN 1.549 s = 0.65 Hz against 0.68), or registering the TMEM's
grant / lost / frz interfaces to the VPU, quant and DMA (a protocol change) so the slice can
span two SLRs at 64 lanes. A free placement across SLRs (the wide-port branch's rounds 1-2,
-4.6 to -4.9 ns) is the measured outcome of neither.

### 6g. The LANES 32 point, f2-vla32 (2026-10-19; the user's D18 A: the placement round at LANES 32, MCOLS 32 packed, in one SLR)

**The target** (targets/f2-vla32.toml): f2-vla with LANES 32, ULANES 32, TMEM_SBP 5 (log2 LANES,
stated: opentpu.target now fails closed on a TMEM_IMPL 3 target whose stated or default SBP is
below log2 LANES, and on ULANES / MXU_E / VPU_CL above LANES), MXU_E 32 (= LANES: a packed pass's
two rows in two steps as at 64 lanes), VPU_CL 16, COLL_GATHER 0; the rest f2-vla's (MCOLS 32
packed, MXROT, ACT_ROWS 255, TMEM 1.5M in URAM, the ACT RAM in block RAM, TMEM_WARB 2).
f2-vla stays the 64-lane point: 6b-6f and docs/pi05.md 5.4 / 5.5 are measured there, and the
>= 1 Hz slice is the 64-lane one once the TMEM's unit interfaces are registered (6e). The two
have their own CFG_HASH (LANES is in it) and programs (the compiler's row groups the tally picks,
128 / 255 / 64 as at 64 lanes: the TMEM is the same).

**Regression and the phases** (the m7a, Verilator 5.048, OTPU_TARGET=f2-vla32 OTPU_MCOLS=32
OTPU_LANES=32 OTPU_PACK=1): tests/test_rtl.py, test_vops.py, test_mxu_pack_rtl.py, test_quant_rtl.py, test_pi05.py
**320 passed**, 6 skipped, and **279 passed** with MXROT forced on every Config, the 13
pre-existing failures of 6c's run (the test_axi_* and additive-mask tests at the f2-vla
environment) and two more that the fix after them removes (test_vops at a 64-lane Config on the
32-lane target: rtlsim now raises TMEM_SBP to the simulated lanes', cb78ea0c; both pass). The
seven measured phases, the same tool and configuration as 5.4 (tools/pi05_rtl.py, the m7a):

| phase (Np 558, rows 128 / 255 / 64) | LANES 64 (5.4, 6e's RTL) | **LANES 32** | the LPN at 32 (lpnlang) |
|---|---|---|---|
| v_out, MCOLS 32 E 32 | 627,705 (MXU 56.5%) | **913,551 (38.8%; +45.5%)**: the VPU busy 85%, the MXU 93% of its cycles waiting on it, the arbitration losses as at 64 (`fv` 179K against 174K, `fm` 4) | 726,600 (+20.7%) |
| p_qkv | 838,610 (44.0%) | **929,476 (39.7%; +10.8%)** | 878,166 (+11.7%) |
| e_step | 467,218 (33.2%) | **556,512 (27.9%; +19.1%)** | 488,929 (+8.0%) |
| p_attend | 722,238 (24.0%) | **957,878 (18.0%; +32.6%)**: the quantizer's losses `fq` 66.9K against 10.2K, the VPU's 194K against 160K | 634,095 (-0.6%) |
| v_qkv | 592,802 (21.0%) | **627,270 (19.8%; +5.8%)** (QST-bound: port A's word rate, not the lanes) | 650,095 (+5.3%) |
| v_out, MCOLS 16 E 16 | 876,803 (80.9%) | **1,143,378 (62.1%; +30.4%)** | - |
| e_step, MCOLS 16 E 16 | 600,209 (51.7%) | **673,748 (46.0%; +12.3%)** | - |

**The LPN** (tools/pi05_lpn.py --full, the real model, 200 MHz): lpnlang **1.549 s = 0.65 Hz** (309.7 M cycles; vision 0.439, prefix
0.816, expert 0.293 s), the dialect **1.532 s = 0.65 Hz** (306.3 M), against 1.477 / 1.414 s
at 64 lanes: +4.9% / +8.3%; the tally's bound 0.951 s (the MXU's: unchanged) and serial 2.356
s (2.003 at 64). The measured phases say more: the LPN has the 32-lane v_out at +21% where the
RTL has +46%, e_step +8% against +19%, p_attend -1% against +33% (its VPU model carries the
elementwise VOPs at the lanes and the composites at VPU_CL, but not the VPU's own pipeline at
half the width under the same instruction stream, nor the quantizer's and VPU's extra TMEM
losses on the attention phase). Weighting 5.5's chunk by the measured ratios (v_out's for the
tower's MLP phases, p_qkv's for the prefix MLP p_out, which the 32-lane model did not run,
p_attend's and e_step's for theirs): **~1.75 s, ~0.57 Hz** at 200 MHz, the number to carry
until p_out is measured at 32 lanes. tools/pi05_cost.py
--target f2-vla32 (its "today" row groups 16 / 8 / 8, the card's shape, not the plan's) reads
port B 6.647 s bound / 10.8 s serial: the 255-row group is what makes the plan's tally (the LPN
prints it at 128 / 255 / 64).

**The pre-synthesis estimate at LANES 32**, from 6f's hierarchy (f2-vla, 64 lanes: u_tmem
124,593, u_vpu 205,531, u_quant 56,268, u_mxu 233,490, u_act 2,361, the sequencer and DMA
~20K; the slice 642,912) and what scales with the lanes:

| hierarchy | 6f (64 lanes) | LANES 32 (est.) | why |
|---|---:|---:|---|
| u_tmem | 124,593 | ~63K | the eight read rotators, three write rotators and the enables / rows scale with the lanes (~86K -> ~43K), the IMPL 3 arbitration and sub-bank selects with the RAMs (128 -> 64: ~20K -> ~10K), the rest ~10K |
| u_vpu | 205,531 | **~160K** (VPU_CL 16) / ~112K (VPU_CL 8) | the 64 plain lanes ~1.7K each halve (~109K -> ~54K); the 16 composite lanes (VPU_CL: EXP2 / RECIP / ... pipelines, ~6K each, ~96K) and the 16-lane sum sub-datapath do not; f2-hbm16's 36K at LANES 8 / VPU_CL 2 fits the same split with ~10K fixed |
| u_quant | 56,268 | ~36K | the lanes' prescale, max and q8 paths halve; the scale unit, the group RAMs and QST T's transposer stay |
| u_mxu | 233,490 | 233K | the 32 packed columns and the drain's 32 epilogue lanes (E = LANES = 32) are what they were |
| u_act, seq, DMA, ctrl | ~22K | ~22K | |
| **the slice** | 642,912 | **~515K (VPU_CL 16), ~465K (VPU_CL 8)** | |
| + the adapter u_mem and the control / trace (SLR2 with it in pb_otpu_board) | ~43K | ~43K | |
| **pb_otpu_board** | | **~560K / ~510K against SLR2's ~435K: 128% / 117%** | |

So the slice does **not** land at ~340K: that number (6e) was the SLR2 side of the array-against-
the-rest split, with the MXU's 85K of columns and the ACT RAM in SLR1; the whole slice in one
SLR at LANES 32 is ~465-515K, over SLR2's CL area before the adapter, because the MXU (233K)
and the VPU's composite lanes (~96K) do not shrink with the lanes. VPU_CL 8 takes ~48K off at
1.667 s = 0.60 Hz (lpnlang --vpu-cl 8, +7.6% over VPU_CL 16; the
measured VPU-bound phases would lose more) on the LPN. RAM is 6f's (the ACT RAM 1,088 RAMB36, the TMEM 384 URAM: the
banks are 32 x 2 x 24,576 words, 6 URAM a sub-bank, the same 384), DSP 6f's.

**The build mechanism** (the second deliberate cross-merge, from the wide-port branch at
223a3bf7, read-only there; the eleven commits cherry-picked with -x before e7b1683e):
boards/f2/build.py's --place-directive and its pnr handling (cl_pnr_user_hbm.tcl appended to
cl_pnr_user.tcl for an HBM target), build_level_1_cl.tcl's opt_design wrapper (cl_pnr_user.tcl
and otpu_clk2x.tcl read unmanaged on the linked design, fail-closed: the reset exception on
exactly 12 endpoints, the floorplan's cells found by name, the pumped RAM cells cross-checked
by module and name and the worst read-request path asserted at two fast periods), the
floorplan (pb_otpu_hbm SLR0, pb_otpu_mid SLR1, pb_otpu_pcis SLR1, pb_otpu_board SLR2,
pb_otpu_tmem inside it: the TMEM alone here, see the file), the three-stage otpu_xr chains on
every net between the HBM fabric and the board (otpu_f2_hbm_mem NXR 3, otpu_axil_xr, the
reset and HBM ready), the fabric's host-path and bridge fixes of that branch's rounds 4-5
(the host registered per port, the PCIS skids, pcis_wbusy from registers, the bridge's beat
register and W count), and tests/test_f2_build.py's tclsh mocks. Not ported: that branch's
round-6 list (item 9, pb_otpu_pcis's hop to u_host, unresolved there and here; item 10, the
adapter's write-address cone; item 11, the pumped banks' write side: on this branch the write
pins come from the per-RAM pk_* flip-flops within one fast period by design and cannot take
the read side's multicycle, nothing to port), and the wide port itself (PORTB > 1, RL,
ROWLANES, ALINE: not on this branch). One conflict by hand (the sys top's u_board instantiation).
otpu_clk2x.tcl's new cross-check counts block RAM primitives under the pump cells with
`PRIMITIVE_TYPE =~ BLOCKRAM.*`, which URAM288 (BLOCKRAM.URAM) satisfies; the synthesis log's
"[otpu_clk2x.tcl] ... 384 block RAM primitives" line is the confirmation.

**Can the synthesis + P&R be launched?** The synthesis of f2-vla32, yes: it is the measurement
this section's estimate needs (u_vpu's and u_tmem's real counts at 32 lanes). The one-SLR P&R
on the estimate, no: pb_otpu_board at ~560K (VPU_CL 16) or ~510K (VPU_CL 8) is over SLR2 and
the HDK's opt_design would stop on the over-full pblock (fail closed, by design); if the
measured slice comes in under ~330K the pblocks apply as they are. What brings it there: the
array-against-the-rest split (6e: the MXU's internal issuer / drain handshake and the result
FIFO's push registered, then pb_otpu_mxu in SLR1 for the columns and the ACT RAM: ~100K out of
SLR2, the rest ~365-415K = 84-95%, still above the 72% line without VPU_CL 8), or the
registered TMEM interfaces at 64 lanes. The plan's numbers to carry: f2-vla32 at ~0.57 Hz by the measured phases (0.65 on the LPN) against f2-vla's ~0.65-0.68,
with its slice ~465-515K LUTs by the estimate..

### 6h. Synthesis of f2-vla32 (measured, 2026-10-18: LANES 32 does not fit one SLR either)

SynthCL on the F2 host (vla b746b90a, target f2-vla32, run dir
/home/ubuntu/opt/hwloop-nightly/vla32-b746b90a/synth, 0 errors, no overutilization):

| resource | f2-vla (6f, 64 lanes) | **f2-vla32** | of the part |
|---|---|---|---|
| CLB LUTs | 718,265 (55.1%) | **574,183** | **44.0%** |
| RAMB36 | 1,353 | 1,324.5 | 66% |
| URAM | 384 | 384 | 40% |
| DSP48E2 | 5,876 | 5,556 | 62% |

By hierarchy (LUTs): `u_slice` **500,230** (was 642,912); `u_vpu` 164,083 (was 205,531: the 16
composite lanes do not scale with LANES, as 6g estimated), `u_tmem` 57,662 (was 124,593),
`u_quant` 32,637 (was 56,268), `u_mxu` 225,678 (was 233,490). Timing estimate
(post-synthesis): clk_raw +0.652, clk_f_raw +0.288, hclk_raw +0.359, clk_main_a0 +2.481; one
clk_f_raw -> clk_raw path at -0.402 (a pumped bank's URAM clock to an MXU DSP input,
`fp_mul_s1` B port: the read data's half-period return, -0.003 at 64 lanes) and the
pre-placement hold nits.

So the 32-lane slice is 500K LUTs against SLR2's ~435K of CL area (115%): 6g's estimate (515K)
was right and the one-SLR floorplan of the wide-port branch does not apply at either lane
count. What does fit, by the numbers: the MXU's array side (~100K: columns, ACT RAM, weight
FIFOs, issuer) in SLR1 with the rest in SLR2 (400K = 92% of SLR2 at 32 lanes; 540K = 124% at
64 lanes, where the VPU or the quantizer would have to move too), which needs the issuer /
drain handshake and the result FIFO's push registered across the boundary (6e's analysis), or
at 64 lanes additionally the TMEM's grant / lost / frz interfaces to the VPU, quant and DMA
registered so those units can leave SLR2. LANES 32 alone buys nothing: it costs the measured
phases 6-46% (6g) and still needs the split.

### 6i. The array split for the f2-vla32 placement (2026-10-19; the user's D20 A: the MXU's array side in SLR1, the rest in SLR2)

**The cut.** Not through the MXU's control (6e: the issuer and the drain share the command
entries, `rows_live`, `rf_n`, `c_left` and the pop head with the slice's grant in the same
cycle), but through its compute pipeline: everything between otpu_mxu's S0 registers (the
popped chunk from the FIFO's read register, the scale entries, the per-command constants) and
the result FIFO is now `otpu_mxu_arr` (rtl/mxu/otpu_mxu_arr.sv: the chunk's decode, the
shifts, the weight skews and hops, the 4,096 PEs, the column ends and deskews, the result
row's assembly and composition), instantiated as `u_mxu/u_arr`. With `ARR` (otpu_slice
`MXU_ARR`, a uarch key, f2-vla32 1; every other target 0, the pipeline it was) every net that
crosses the hierarchy is a flip-flop or a block RAM pin at both ends:

| crossing | direction | what, how wide | the ends | cycles added |
|---|---|---|---|---|
| the S0 bundle | SLR2 -> SLR1 | the popped chunk's flags (m0, 9 bits), the chunk and PACK's pair chunk (2 x 1,024: from the chunk FIFO's read registers, which sit in SLR1 with the array), the scale entries (96 + 64 + 40, the scale FIFO's read registers, SLR1 too), the per-command constants (cu0 .. sel0, ~100) | otpu_mxu's S0 registers -> u_arr's two input stages (NX 2) | 2 |
| the ACT RAM's read request | SLR2 -> SLR1 | act_blk / blk2 / hi / grp / ren (~70) | otpu_mxu's act_* output registers (ARR) -> otpu_actram's RREG input stage, then its block address mux and the 1,088 RAMB36 | 2 (the data meets the delayed bundle at the act skews, as before) |
| the ACT RAM's data and scale records | within SLR1 | 32 x 1,024 + 32 x 64 bits | u_act's registered read -> u_arr's act skews | 0 |
| the chunk FIFO's and scale FIFO's writes | SLR2 -> SLR1 | port B's chunk (1,024) and port A's entries with their slot addresses | the slice's port registers / otpu_mxu's queue-slot and read-address registers -> the RAMs' pins | 0 |
| the result row and its push | SLR1 -> SLR2 | rowp (32 x 80 + 33 = 2,593 bits), rv, rvb, push_n | u_arr's output registers -> the result FIFO's three LUT-RAM copies (their write-enable replicas from push_n) | 1 |
| the quantizer's ACT RAM writes | SLR2 -> SLR1 | 32 bytes, a scale record, the row and index | the quantizer's registers -> otpu_actram's we_i / we_h (the two-register hop it always had) | 0 |
| the adapter's port interfaces (u_mem in SLR1 with the array) | both | port A / B / SW requests and addresses (slice -> adapter), the readies, valids and data (adapter -> slice) | the slice's port registers both ways; the adapter's a_* and b_rvalid are registers, its b_rdy and b_rdata (the 1,024-bit mux of two queue heads) a LUT level or two before the slice's registers | 0 |
| the HBM fabric <-> the adapter | SLR0 <-> SLR1 | the native ports | the three-stage otpu_xr chains (stage 2 in SLR2 now one hop past the adapter) | 0 |

So three cycles more from a pop to the result FIFO (LROW + 3) and nothing else: the drain, the
TMEM, the sequencer's handshake and the DRAM ports do not cross. The adapter's two readies are
the one exception to flip-flop-to-flip-flop (a hop with at most two LUT levels, the slice's
side registered); without the adapter in SLR1, pb_otpu_board is at ~100% of SLR2 (below).
Bit-exact: tests/test_rtl.py mlp, fuzz_mm_replay, attention_decode (RMAX, ACC) with ARR and
MXROT forced, tests/test_mxu_pack_rtl.py at MCOLS 4 (PACK's pair rows through the registered
push); the f2-vla32 suites on the m7a: tests/test_rtl.py, test_vops.py, test_mxu_pack_rtl.py, test_quant_rtl.py,
test_pi05.py **322 passed**, 6 skipped, and **281 passed** with MXROT forced on every Config
(the 13 pre-existing failures of 6c's run, unchanged). The LPN's chunk at f2-vla32 with the MM
latency constant +3 (--lmm 75): 1.547 s against 1.549 (no rate moves).

**The phases** (tools/pi05_rtl.py on the m7a, f2-vla32 with MXU_ARR against 6g's f2-vla32):

| phase | 6g (f2-vla32) | **with the split** | delta |
|---|---|---|---|
| v_out, MCOLS 32 E 32 | 913,551 | **913,707** | +156 (+0.02%) |
| p_qkv | 929,476 | **929,516** | +40 |
| e_step | 556,512 | **556,630** | +118 (+0.02%) |
| p_attend | 957,878 | **958,228** | +350 (+0.04%: 96 MMs, ~3.6 cycles each) |
| v_qkv | 627,270 | **627,542** | +272 (+0.04%) |
| v_out, MCOLS 16 E 16 | 1,143,378 | **1,143,378** | 0 |
| e_step, MCOLS 16 E 16 | 673,748 | **673,748** | 0 |

The LPN carries it as the MM latency constant (l_mm 72 -> 75, no rate moves): the chunk
moves 1.549 -> 1.547 s (0.65 Hz either way).

**The floorplan** (boards/f2/build/constraints/cl_pnr_user_mxu.tcl, appended by build.py
after cl_pnr_user_hbm.tcl for a target whose uarch states MXU_ARR; the six pblocks of the
wide-port file at 788a91c0, pb_otpu_xs cherry-picked with round 6's credit link, as they are):

| pblock | region | cells | LUTs (6h's count) | fill |
|---|---|---|---|---|
| pb_otpu_mxu (new) | SLR1 X0Y4:X3Y7, 16 clock regions, ~217K | u_slice/u_mxu/u_arr (~105K: the OOC column's 2,642 x 32 = 85K, the shifts ~13K, the decode and the row assembly), the chunk and scale FIFO RAMs (62 RAMB36), u_slice/u_act (2.4K, 1,088 RAMB36), u_board/u_mem (38K) | ~146K | **~67%** (and pb_otpu_mid's ~18K flip-flops in the same box) |
| pb_otpu_board | SLR2 X0Y8:X7Y11, 32 regions, ~435K | u_board less the above: the MXU's issuer and drain (~120K), the TMEM (58K), the VPU (164K), the quantizer (33K), the DMA, sequencer, control and trace (~20K) | ~395K | **~91%** |
| pb_otpu_tmem (child) | X2Y9:X5Y10, 8 regions, ~109K | u_tmem (58K, 384 URAM) | 58K | 53% |
| pb_otpu_hbm / mid / pcis / xs | SLR0 / SLR1 / SLR1 / SLR0 | the fabric, the chain stages, the PCIS registers and link ends | ~30K + flip-flops | as on wide-port |

91% of SLR2 is above the wide-port file's own "72%: too full for 200 MHz routing" line: the
VPU's 164K (its 16 composite lanes do not shrink with the lanes) is what fills it; VPU_CL 8
would take ~48K out (6g: 0.60 Hz on the LPN against 0.65). The 6h clk_f -> clk_raw path
(-0.402 ns, the pumped bank's URAM clock to the fmadd's DSP B input): the ACC path's `xo` /
`alr` registers are kept as fabric registers now (`keep`), so the TMEM's half-period read
return ends at a flip-flop, not at the DSP's input register Vivado had folded it into.

**Can the synthesis + P&R be launched?** Yes, as the first placed VLA build: the mechanism
(the unmanaged Tcl, the chains, the pblocks) is the wide-port branch's at its round-6 state, the
split's crossings are registered, and the three fills are 67 / 91 / 53%. What to read first in
its reports: pb_otpu_board's utilization after place_design (the 91% is a count, the placer's
answer is the measurement), the clk_raw paths across the SLR1 / SLR2 boundary (expected none
with logic but the adapter's readies), and otpu_clk2x.tcl's cell cross-check on the 128 pumped
RAMs. If SLR2 is the problem, the next moves are VPU_CL 8 or the quantizer's row buffer and
transposer RAMs to SLR1 (block RAM, not logic: no LUT relief).

## 7. Phases (each ends in ISA / Verilator results; the nightly synthesizes the best point)

| phase | work | exit criterion |
|---|---|---|
| 0 | π0.5 port: **done** (docs/pi05.md: ISA-exact numerics, the tally, int8 error). RhinoVLA's shape-only tally: **done** (tools/rhinovla_shape.py, the paper's Table 8: ViT 24 x 1024 at 256 patches a view merged to 64 tokens, 3 views; the 2B 28 x 2048, MLP 6144, 16 / 8 heads of 128; the expert 18 x 1024, MLP 3072, 31 rows; 0.66 T MAC a chunk, 45% of π0.5's; MXU 0.90 s at MCOLS 32, 0.53 s at 64, port B 0.25 s). Left: QST's DRAM rate confirmed on the RTL (L11) | the budget per phase; RhinoVLA's seconds a chunk at each point |
| 1 | **the packing's feasibility** (L1): **done** in Verilator (bit-exact, docs/mxu_packing.md); the columns out of context on the F2 host (Vivado 2025.2, xcvu47p, 2026-10-17): **done, the gate passed**: the packed column WNS +1.986 ns at 5.0 ns and +1.010 at 4.0 (146 DSP48E2, 2,642 LUT, 6,692 FF a column), the unpacked +2.021 / +0.986 (137 DSP, 1,738 LUT, 4,372 FF); the worst path is the column end's DSP cascade in both (~330 MHz), so packing costs +904 LUT and +2,320 FF a column and no clock. ACT_ROWS 255 in the tally (done: needs TMEM 1.5M words, L2) and the RTL (**done**: otpu_actram's groups, the replay tests at 255 rows). QST: the K stores confirmed a word a cycle on the RTL; the V stores **QST T, done** (L11: -42% on the tower's QKV phase at MCOLS 16) | **MCOLS 64 packed or MCOLS 32 unpacked**: the fmax a column; the tally at ACT_ROWS 255; QST off the vision tower's critical path (at MCOLS 32 it is still on it: the port A word rate) |
| 2 | the ACT RAM at the chosen MCOLS: BRAM / pumped / URAM from the width math (**done, 6c**: the ACT RAM's 1,024 + 64 RAMB36 by its geometry, the TMEM's 384 URAM; the first synthesis, 6b, had the ACT RAM spilled into LUT RAM and the TMEM in 2,048 RAMB36); a floorplan sketch per SLR (columns, the result FIFO, the drain): open | the BRAM budget (6c: RAMB36 66%, URAM 40%, estimated); the SLR plan |
| 3 | L1 MXU columns: MCOLS 8 -> 16 -> 32 (-> 64) (the RTL parameter exists: MCOLS 4 / 8 builds today) with ACT_ROWS 255. **The packing is in the array** (`otpu_mxu PACK`, merged from vla-pack, docs/mxu_packing.md 7: bit-exact at MCOLS 4 / 8 physical = 8 / 16 logical, ACT_ROWS to 64, tests/test_mxu_pack_rtl.py 36 passed; left: MCOLS 16 / 32 physical PACK builds and the OOC fmax). **Measured (unpacked, docs/pi05.md 5.3)**: MCOLS 16 and 32 build and run bit-exact at ACT_ROWS 255 (tests/test_rtl.py); the RTL timing of the cut phases at Np 558 (p_out at 128) gives the MXU 81% of the cycles on the prefix's MLP at MCOLS 16 (60% at 32) and 32-66% on the other phases, and names the loss: the MXU drain denied the TMEM grant while the VPU runs (the result FIFO then fills; the loss follows the VPU's share of the time), the dependency gaps on QACT, and the 16-lane VPU (L4) | bit-exact at each MCOLS (**done** for 16 and 32); RTL timing on the **real prefix shape** (558-968 rows, the 16384-wide MLPs, the 2048-wide projections; a cut model hides the drain's and result FIFO's losses): MAC/cycle ~MCOLS x 128 at >= 75% (**met by p_out at MCOLS 16, 81%; not by the rest**: 52-66% at MCOLS 16, 32-60% at 32; phase 4's two items are what stands in the way) |
| 4 | L4 / L5 / L6: LANES 64 (the VPU's partial loop past 16 lanes: **done**, the sums on a 16-lane sub-datapath, bit-exact at 32 / 64), TMEM 1.5M words (**builds and runs**), QACT / QST at LANES a cycle (**done**: the quantizer bit-exact at 64 lanes); **and the MXU drain's TMEM arbitration** (**done**: TMEM_WARB 2 in the target, p_out at MCOLS 16 -17%; and its rate, MXU_E 32: a packed pass in two steps) | VPU / QACT tests (**passed** at 32 / 64; the regression after the changes: tests/test_rtl.py replay / act_reads / attention / lane_count / arbitration 52 passed, 5 skipped; tests/test_mxu_pack_rtl.py (MCOLS 4 / 8 / 16 / 32) + test_vops.py (LANES 4 to 64) + test_quant_rtl.py 130 passed, 1 skipped); bank-conflict counters in the RTL timing (TMEMSTAT, the pair's wv / wm / wc / wcm / wcv); VPU + QACT + QST < 15% of the serial sum: **met by p_out alone** (13% by the tally at the target; v_out 28%, p_qkv 23%, e_step 27%); the MXU >= 75% of the cycles on v_out / p_qkv / p_out at MCOLS 32: **not met at MCOLS 32 packed** (v_out 56.5%, p_qkv 44.0% measured at 64 lanes; p_out at the target not measured: at Np 32, the run the 64-lane model's speed allows, it is port-bound by the tally), **met at MCOLS 16 packed** (v_out 80.9%: the same logical width as MCOLS 32 unpacked, where 5.3 had 36%). Next: the drain at one step a packed pass, VPU_CL 64 or the fused unit (L3) for the composites, the sums' four-row interleave |
| 5 | L3 fused activation unit, sized on phase 4's remainder: ISA op + RTL + tests | bit-exact; the GELU's share gone |
| 6 | L7 memory: f2-hbm's S=1 x 8 ports, PORTB 2 merged from ternary-bonsai, the DMA path for activations | RTL timing on the 8-port model: the expert step no longer port-bound; port-B stalls < 5% outside it |
| 7 | RhinoVLA port (Qwen3-VL ViT new; Qwen3 LM kernels exist; expert / flow loop shared with π0.5) | action error vs its reference |
| 8 | f2-vla target; nightly synthesis at 200 MHz, then 250; AFI; board run | ≥ 1 Hz RhinoVLA on the F2 (measured) |
| 9 | L9 W8A16 where phases 0 / 7 show int8 activations hurt; the MCOLS 96 stretch for π0.5 at 1 Hz | success-rate parity on LIBERO; RTL timing, then the nightly |

## 8. Risks and the fallback

* The DSP packing (L1) is exact (phase 1); its OOC fmax on the real part passed on 2026-10-17
  (+1.99 ns at 200 MHz, +1.01 at 250 MHz): the gate is closed, PACK stays. Unpacked, MCOLS 64 is the whole part's DSPs, and the
  fallback is **MCOLS 32 x 1 slice unpacked** (4,480 DSP), which keeps every argument of
  section 4 and lands RhinoVLA near 1 Hz with no margin. MCOLS 32 x 2 slices is not the fallback: it
  brings back the collectives and the sharding of π0.5's single KV head that section 4 avoids.
* A 32-64-column MXU, 64 VPU lanes and a 64-way TMEM crossbar in one slice (6b measured the
  crossbar at 425K LUTs; 6c cuts the MXU's two general ports to rotators, ~140K estimated for
  the TMEM): wiring across SLRs
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
