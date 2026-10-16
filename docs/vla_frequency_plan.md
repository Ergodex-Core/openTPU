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
| L1 | **MXU columns MCOLS 4 -> 64**, DSP48E2 int8 packing (one weight x two activation rows a DSP: the column structure shares the weight), the weight chunk systolically across the columns | 16x MAC/cycle: 8192 MAC/cycle | 64 x 128 MACs at 2 a DSP = 4096 DSP; LUT est. ~250-320K (S1 decode, accumulators, per-column epilogue i2f / fmul) | Verilator vs ISA bit-exact at MCOLS 8, 16, 64; mxu_bench |
| L2 | **ACT_ROWS 16 -> 256** (a bank a column, URAM: 256 rows x 128 blocks x 128 B = 4 MB) | weight passes over a prefix /16 (π0.5: ~4 passes instead of ~56) | ~114 URAM | ISA tally, RTL timing of a cut model |
| L3 | **a fused activation unit in the MXU drain / quantizer**: GELU-tanh / SiLU (table + polynomial, as SGMV's sgmv_act), gate x up, then quantize for the next MM (SGMV's MLP_FUSE); gate and up rows interleaved in the image so a drain yields pairs; one unit per drain lane | removes the prefix MLPs' VPU work (~1.3 s of π0.5 today; ~0.1 s even at 64 lanes) | est. ~4K LUT, ~10 DSP, 3 BRAM a lane, 8-16 lanes | bit-exact against an ISA definition of the fused op |
| L4 | **VPU lanes 8 -> 64** (LANES; TMEM banks with them) for softmax over long rows, norms, RoPE, residual adds | 8x element throughput: π0.5's ~92 M softmax elements and norms in ~0.05 s | LUT est. ~150-200K (fp32 lanes), DSP ~300-500 | the VPU tests at LANES 16, 32, 64 |
| L5 | **TMEM 64K -> 1M words** (4 MB in URAM), banked by LANES; the drain's MCOLS outputs and QACT's row reads at 64 words a cycle | prefix tiles of 256 rows x 2048 fp32 (2 MB) in one pass; fewer DRAM round trips | ~114 URAM; a 64-way bank crossbar est. ~80-100K LUT | tiled prefill tests; RTL bank-conflict counters |
| L6 | **QACT / QST at LANES elements a cycle** (they feed 256-row stationary operands) | activations into ACT RAM as fast as the MXU drains them | with L4's lanes | the quantizer tests at LANES 64 |
| L7 | **memory: 8 HBM ports on the one slice** (NCH 8, ~115 GB/s raw, ~100 GB/s usable): port B and a DMA lane for activation tiles / KV; then double port B (PORTB 2, the wide-port lanes from ternary-bonsai) where the small-M phases (expert: 50 rows, vision views) show port-B stalls (section 6) | keeps the MXU fed at small M; activation traffic off the weight port | the HBM shell exists (f2-hbm: S=1 x 8); lanes est. +~10K LUT | RTL timing on the 8-port memory model |
| L8 | **clock 200 -> 250 MHz** (SGMV closed 250 on the same part) | 1.25x | timing work, nightly | synthesis / route (nightly only) |
| L9 | **W8A16 where accuracy needs it** (RhinoVLA's paper: W8A8 hurt success) | accuracy; int8 x int16 is one MAC a DSP (no packing): W8A16 layers at half L1's rate | DSP or a second weight path | action error vs fp32 (docs/pi05.md harness) |
| L10 | **model-side**: the 3 views as MXU rows of one pass, the prefix KV once a chunk, the expert's time conditioning precomputed (docs/pi05.md), fewer visual tokens (RhinoVLA: 64 a view) | π0.5 -> RhinoVLA halves the MACs | none | ISA |

## 6. The VLA variant `f2-vla` (one slice; target, estimate)

S = 1, D = 128, MCOLS = 64 (packed int8 DSPs), ACT_ROWS = 256, ACT_BLOCKS = 128,
TMEM_WORDS = 1M, LANES = 64, the fused activation unit, NCH = 8 HBM ports (~100 GB/s), PORTB 1
(2 if section 6a's stalls ask), a DMA lane for activation tiles, core 250 MHz. Peak 64 x 128 =
8192 MAC/cycle x 250 MHz = 2.05 TMAC/s (20x today). Budget: DSP ~4.6K of 9024, URAM ~230 of
960, LUT ~0.6-0.8M of 1.3M (estimates for the nightly synthesis to check).

| | today (MCOLS 4, 200 MHz) | f2-vla at 75% MXU efficiency |
|---|---|---|
| π0.5 a chunk | ~25-30 s | ~1.4 s MXU + ~0.1-0.2 s VPU / other: **~0.65 Hz** |
| RhinoVLA a chunk | ~12-15 s | ~0.65 s + ~0.1 s: **~1.3 Hz** |

RhinoVLA reaches ≥ 1 Hz on this variant. π0.5 at 1 Hz needs ~2.8 TMAC/s effective: MCOLS 96
(12,288 MAC/cycle, DSP ~6.2K + VPU: ~0.95-1 Hz) is the stretch point within one slice; MCOLS
128 would need ~8.2K DSPs, past the part with the VPU's. An action chunk is 50 actions: 1 Hz
inference still drives 10-50 Hz control, open loop within a chunk.

### 6a. Memory bandwidth with 8 ports (π0.5, a chunk)

| traffic | bytes | at ~100 GB/s |
|---|---|---|
| weights, prefix: ceil(~900 / 256) = 4 passes x ~2.5 GB | ~10 GB | 0.10 s (overlapped: the MXU needs a chunk every 4 cycles) |
| weights, vision: 768 rows, 3 passes x 0.8 GB | ~2.4 GB | 0.02 s |
| weights, expert: 10 steps x ~0.3 GB, 50 rows (< MCOLS: a chunk a cycle, port-B-bound) | ~3 GB | 0.03 s at 32 GB/s port B: 0.09 s |
| activation tiles, KV, the drain's outputs (~40 MB a layer) | ~0.8 GB | 0.01 s |
| total | ~16 GB | ~0.16 s of port time against ~1.4 s of MACs |

So 8 ports are enough for one f2-vla slice. The one phase that is port-B-bound is the expert
(50 rows under 64 columns: a chunk a cycle, 0.09 s at PORTB 1); PORTB 2 halves it, and is the
first doubling if the RTL timing shows it matters; 16 ports only with the MCOLS 96 stretch.

## 7. Phases (each ends in ISA / Verilator results; the nightly synthesizes the best point)

| phase | work | exit criterion |
|---|---|---|
| 0 | π0.5 port (agent, in progress): ISA-exact numerics, the cost tally (vision / prefix / expert; MXU / VPU by op / QACT / port B) at today's config and at ACT_ROWS 128 / LANES 32 | seconds a chunk, per-unit breakdown, int8 action error |
| 1 | software: tile sizing for ACT_ROWS 256, views batched, prefix KV once, conditioning precomputed (L10) | ISA tally at the new sizes |
| 2 | L3 fused activation unit: ISA op + RTL + tests | bit-exact; VPU share < 10% of a chunk |
| 3 | L1 MXU columns: MCOLS 8 -> 16 -> 64 with DSP packing (the RTL parameter exists: MCOLS 4 / 8 builds today), L2 ACT_ROWS 256 | bit-exact at each MCOLS; RTL timing of a cut prefix: MAC/cycle ~MCOLS x 128 at >= 75% |
| 4 | L4 / L5 / L6: LANES 64, TMEM 1M words, QACT at LANES a cycle | VPU / QACT tests; bank-conflict counters in the RTL timing |
| 5 | L7 memory: f2-hbm's S=1 x 8 ports + a DMA lane; PORTB 2 if the expert's port-B stalls matter | RTL timing on the 8-port model: port-B stalls < 5% outside the expert |
| 6 | RhinoVLA port (Qwen3-VL ViT new; Qwen3 LM kernels exist; expert / flow loop shared with π0.5) | action error vs its reference |
| 7 | f2-vla target; nightly synthesis at 250 MHz; AFI; board run | ≥ 1 Hz RhinoVLA on the F2 (measured) |
| 8 | L9 W8A16 where phases 0 / 6 show int8 activations hurt | success-rate parity on LIBERO |
| 9 | the MCOLS 96 stretch for π0.5 at 1 Hz | RTL timing, then the nightly |

## 8. Risks and the fallback

* A 64-column MXU, 64 VPU lanes and a 64-way TMEM crossbar in one slice: wiring across SLRs at
  250 MHz is the main risk (pblocks for the MXU's column chain, pipelined weight and
  ACT-RAM paths); fall back to 200 MHz (-20%) or MCOLS 32 x 2 slices (the multi-slice route,
  f2-hbm:s2: collectives and VLA sharding come back with it).
* DSP int8 packing needs the shared weight across two activation rows: true for the column
  structure, but its exactness has to be shown bit for bit.
* The VPU estimate rests on VOP counts; the fused unit's numerics must match an ISA definition.
* W8A8 accuracy for actions is unproven (the paper's warning); W8A16 halves the packed rate.
