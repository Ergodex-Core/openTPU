# Plan: VLA inference at >= 1 Hz on openTPU (F2 FPGA)

Status: plan (2026-10-16). Every number here is an *estimate* from model shapes and the
openTPU configuration unless marked measured; Phase 0 replaces them with the ISA tally and RTL
timing of the π0.5 port (docs/pi05.md, in progress). Work order: π0.5 port first, then
RhinoVLA (HuixiAI/RhinoVLA-V3), then the variant below. Everything is validated in the ISA
simulator and Verilator first; FPGA synthesis only in the nightly build.

## 1. VLA and LLM need different openTPU variants

| | LLM decode (Gemma-3-1B, Ternary-Bonsai; branch ternary-bonsai) | VLA (π0.5, RhinoVLA; this branch) |
|---|---|---|
| shape | one token a step: matrix-vector (M = 1) | a ~300-970-token prefix, 3 views of vision, 10 denoising steps of 50 rows: matrix-matrix |
| bound by | weight bytes a token (port B) | multiply-accumulates, and the VPU's element-wise work |
| levers | wide port B / HBM lanes (docs/wide_port.md), t2 weights, MTP | MXU columns and slices, ACT RAM rows, TMEM, VPU lanes, a fused activation unit, clock, W8A16 |
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

## 3. Where openTPU on the F2 stands (estimate, S = 1, MCOLS 4, 200 MHz, LANES 8)

* MXU: 4 columns x 128 int8 MACs = 512 MAC/cycle, 102 GMAC/s. π0.5: ~21 s; RhinoVLA: ~10 s.
* VPU: 8 lanes. π0.5's prefix GELU alone ~2.1 G ops / 8 = ~0.27 G cycles = **~1.3 s**, softmax
  ~0.3 s: the VPU is the second bottleneck, and the first once the MXU is scaled.
* Weight traffic in prefill: a prefix of R rows streams every weight ceil(R / ACT_ROWS) times
  (the MXU replays a chunk for ACT_ROWS rows): at ACT_ROWS 16, ~50 passes x 2.9 GB = ~150 GB
  for π0.5, ~6 s at 25.6 GB/s. Port B becomes a limit unless ACT_ROWS grows.
* TMEM 64K words (256 KB): a prefix tile of 128 rows x 2048 fp32 is 1 MB; tiled prefill
  (llm/tiled.py) works in TMEM-sized tiles, at the cost of more passes.
* Total today: **~25-30 s per π0.5 chunk, ~12-15 s per RhinoVLA chunk** (Phase 0 measures it).

## 4. The levers (each with its gain, cost and how it is validated)

| # | lever | gain (estimate) | cost on the VU47P (estimate) | validation |
|---|---|---|---|---|
| L1 | **ACT_ROWS 16 -> 128** (ACT RAM in URAM: 128 x 128 blocks x 128 B = 2 MB a slice) | weight passes /8: prefill stops being port-B-bound | ~57 URAM a slice | ISA tally, RTL timing on a cut model |
| L2 | **MXU columns MCOLS 4 -> 16** (DSP48E2 int8 packing: one weight x two activations a DSP, 2 MACs a DSP) | 4x MAC/cycle | 16 x 128 MACs at 2 a DSP = 1024 DSP a slice (+ ~200 VPU); LUT est. +~100K a slice (S1 decode, accumulators, drain lanes) | Verilator vs ISA bit-exact; mxu_bench |
| L3 | **slices S = 1 -> 4** (f2-hbm:s4: 32 HBM ports exist in RTL) with VLA sharding: MLP and LM-head rows across slices, attention heads (π0.5's Gemma has 1 KV head: replicate K/V, split q heads), vision views across slices | ~3.5x (collectives per layer) | 4 x a slice: DSP ~5K, LUT est. ~750K of 1.3M (58%) | ISA S=4 runs vs reference, RTL S=2 timing |
| L4 | **a fused activation unit** in the MXU drain / quantizer: GELU-tanh / SiLU from a table plus a polynomial (as SGMV's sgmv_act), gate x up, then quantize for the next MM (SGMV's MLP_FUSE), with gate and up rows interleaved in the image so a drain yields pairs | removes ~1.3 s of π0.5's VPU work (the prefix MLPs), ~0.3 s of the expert's and vision's | est. ~4K LUT, ~10 DSP, 3 BRAM a slice | bit-exact against an ISA definition of the fused op |
| L5 | **VPU lanes 8 -> 32** (LANES, TMEM banks) | softmax / norms 4x | LUT est. +~40K a slice, TMEM banking | the existing VPU tests at LANES 32 |
| L6 | **TMEM 64K -> 512K words** (2 MB in URAM) | prefix tiles of 128+ rows in one pass; fewer DRAM round trips | ~57 URAM a slice | tiled prefill tests |
| L7 | **clock 200 -> 250 MHz** (SGMV closed 250 on the same part) | 1.25x | timing work at the nightly | synthesis / route (nightly only) |
| L8 | **W8A16 where accuracy needs it** (RhinoVLA's paper: W8A8 hurt success; W8A16 kept it) | accuracy, not speed: int8 x int16 is one MAC a DSP (no packing), so W8A16 layers run at half L2's rate | DSP or a second weight path | action error vs fp32 (docs/pi05.md harness) |
| L9 | **model-side**: batch the 3 views as MXU rows (one pass of the tower's weights), the prefix KV once per chunk, the expert's time conditioning precomputed (docs/pi05.md), fewer visual tokens (RhinoVLA: 64 a view) | π0.5 -> RhinoVLA halves the MACs | none | ISA |

## 5. The VLA variant `f2-vla` (target, estimate)

S = 4, MCOLS = 16 (packed int8 DSPs), ACT_ROWS = 128, ACT_BLOCKS = 128, TMEM_WORDS = 512K,
LANES = 32, the fused activation unit, NCH = 8 a slice (32 HBM ports), PORTB = 1-2, core 250 MHz.
Peak 4 x 16 x 128 = 8192 MAC/cycle x 250 MHz = 2.05 TMAC/s (20x today); DSP ~5K of 9024,
LUT ~0.75-0.9M of 1.3M, URAM ~460 of 960 (estimates, to be checked by the nightly synthesis).

| | today | f2-vla at 75% MXU efficiency |
|---|---|---|
| π0.5 a chunk | ~25-30 s | ~1.4 s MXU + ~0.1-0.2 s VPU/other: **~0.6-0.7 Hz** |
| RhinoVLA a chunk | ~12-15 s | ~0.65 s + ~0.1 s: **~1.3 Hz** |

≥ 1 Hz is reachable for RhinoVLA on this variant. For π0.5 it needs ~2.8 TMAC/s effective:
MCOLS 24 x 4 slices or 6 slices x 16 (12K MAC/cycle; DSP ~7K, LUT near the part's limit), or
fewer prefix tokens. Note that an action chunk is 50 actions: 1 Hz inference with chunked
execution still drives the robot at 10-50 Hz control, open loop within a chunk.

## 6. Phases (each ends in Verilator / ISA results; the nightly synthesizes the best point)

| phase | work | exit criterion |
|---|---|---|
| 0 | π0.5 port (agent, in progress): ISA-exact numerics, the cost tally per phase (vision / prefix / expert, MXU / VPU / port B) at today's config | measured seconds a chunk, per-unit breakdown, int8 action error |
| 1 | software only: ACT_ROWS / tile sizing, views batched, prefix KV once, expert conditioning precomputed (L9, L1 as a Config change in the ISA) | ISA tally at the new sizes |
| 2 | L4 fused activation unit: ISA op + RTL + tests | bit-exact; VPU share < 10% |
| 3 | L2 MCOLS 16 with DSP packing; L5 LANES 32; L6 TMEM 512K | RTL timing of a cut model at MCOLS 16 |
| 4 | L3 four slices with VLA sharding | ISA S=4 run exact; RTL S=2 timing |
| 5 | RhinoVLA port (Qwen3-VL ViT new; Qwen3 LM kernels exist; expert / flow loop shared with π0.5) | action error vs its reference |
| 6 | f2-vla target; nightly synthesis at 250 MHz; AFI; board run | ≥ 1 Hz RhinoVLA on the F2 (measured) |
| 7 | L8 W8A16 where Phase 0 / 5 show int8 activations hurt | success-rate parity on LIBERO |

## 7. Risks

* Routing a ~1M-LUT design at 250 MHz with 4 slices across SLRs: pblocks per slice, register
  slices on HBM ports (docs/f2_hbm.md risks); fall back to 200 MHz (-20%).
* DSP int8 packing needs the shared operand (a weight times two activation rows): true for the
  MXU's column structure, but its exactness has to be shown bit for bit.
* The VPU estimate rests on VOP counts; the fused unit's exact numerics must match an ISA
  definition (the repo's bit-exact discipline).
* W8A8 accuracy for actions is unproven (the paper's warning); W8A16 halves the packed rate.
