# RhinoVLA V3 (HuixiAI/RhinoVLA, Qwen3-VL-2B + action expert) on openTPU

Status (2026-10-17): Phase 7 of docs/vla_frequency_plan.md, the model side. The fp32 numpy
reference matches the repo's own PyTorch model (on a tiny random model: the real checkpoint,
6.07 GB, was not fetched for lack of disk); the card's image and phase programs run a whole
action chunk on the ISA simulator; the cost of a chunk is tallied per phase and unit, today
and at the f2-vla points (ISA tally, no RTL run). Nothing here ran on an FPGA or on the RTL.
The steps that need the real weights are left as commands (sections 2, 4).

Code: opentpu/llm/rhinovla.py (Spec, Weights, the host side, the fp32 reference, the
quantization emulation, Image / Host / programs / run_chunk), opentpu/kernels/qwen3vl_vit.py
(what Qwen3-VL's ViT adds to kernels/vit.py's SigLIP phases), opentpu/kernels/rhinovla.py (the
text phases and the expert step on kernels/pi05.py's), tools/rhinovla_reference.py (the repo's
model's tensors), tools/rhinovla_check.py (reference and ISA checks),
tools/rhinovla_sensitivity.py (int8 points and the precision map), tools/rhinovla_cost.py
(the tally), tests/test_rhinovla.py (a tiny random model; 22 s).

## 1. The model: what V3 is (the code, not the paper)

The repo (branch main = the V3 release of 2026-10-15; V2 is frozen and incompatible) runs
`rhinovla/model/framework/RhinoVLA.py predict_action`: a Qwen3-VL-2B prefix encoder (HF
transformers' modeling_qwen3_vl.py, built from `rhinovla/assets/qwen3_vl_processor/config.json`,
no official Qwen3-VL weights: the checkpoint carries them) and `modules/action_expert.py`.
Where it differs from arXiv 2606.07383's Table 8 (the paper's V1), the code wins:

* **vision**: Qwen3-VL's ViT, 24 blocks, 1024 wide, 16 heads of 64, MLP 4096 gelu_pytorch_tanh,
  LayerNorms with biases (eps 1e-6), a fused qkv [3072, 1024] with bias, proj with bias. Patch
  16 with a *temporal patch of 2* (the processor feeds the image twice: a [1024, 3 x 2 x 16 x
  16] Conv3d, folded here to a [1024, 768] matrix on the host), a learned 48 x 48 position
  table bilinearly interpolated to the 16 x 16 grid, **2D RoPE** (theta 10000: 16 frequencies
  on the token's grid row, 16 on its column, rotate-half over the head's 64), the tokens in the
  processor's **2 x 2 window order**. The merger: LayerNorm per token (1024), the window's four
  rows as [4096] -> fc1 4096 -> GELU (erf) -> fc2 2048. **Deepstack**: the outputs of blocks 5,
  11 and 17 go through their own mergers (LayerNorm over the window's 4096, fc1, erf GELU, fc2)
  and are added to the text model's image rows after its decoder layers 0, 1 and 2. Images are
  256 x 256 (the configs' image_size; the processor's pixel budget 65536 .. 131072 keeps them:
  no resize), 16 x 16 = 256 patches a view merged to **64 tokens**, **3 views** (top_head,
  hand_left, hand_right). Pixels (x / 255 - 0.5) / 0.5.
* **prompt** (modules/qwen.py, prompt_template v2 with the View Registry): the user turn
  `[top_head|rgb]<img>\n[hand_left|rgb]<img>\n[hand_right|rgb]<img>\n<task>` through the Qwen3-VL
  chat template with the generation prompt: 9 text tokens, a view, 9, a view, 9, a view, 14
  (the task's words among them): **41 text tokens + 3 x 64 = 233 prefix rows** for a 7-word
  task, exactly the plan's estimate.
* **text model**: 28 layers, 2048 wide, 16 query heads and 8 KV heads of 128 with q_norm /
  k_norm (RMSNorm over the head), SwiGLU 6144, RMSNorm w x (not Gemma's 1 + w), **interleaved
  mRoPE** (theta 5e6, sections [24, 20, 20]: of the 64 frequencies, index f < 60 takes the T /
  H / W position for f mod 3 = 0 / 1 / 2, f >= 60 the T; a text token has one position on all
  three axes, an image's tokens the view's start on T and start + row / column of the 8 x 8
  merged grid on H / W, the next text token starts 8 after the view's start), and **causal**
  attention over the prefix: the standard decoder, nothing bidirectional (π0.5's prefix-LM
  mask does not apply). Only every layer's K (after k_norm and RoPE) and V are kept; the last
  layer's attention output and MLP are dead, as π0.5's.
* **action expert** (RhinoActionExpert): **18 layers** (the configs' depth; the dataclass default
  10 is not what is trained), 1024 wide, SwiGLU 3072, 16 / 8 heads of 128 with q_norm /
  k_norm and the same RoPE tables; expert layer i attends to the prefix KV of backbone layer
  **10 + i** (prefix_layer_offset 10: the last 18 of 28) and to the suffix itself
  (bidirectional among the suffix), at positions max prefix position + 1 + i (the rope delta).
  The suffix is 1 **state token** (state_proj(state x state_mask) + state_mask_proj(state_mask))
  and 30 action tokens (action_in_proj(x_t) + action_mask_proj(action_mask)): actions and
  state are **96-D slots** (V3; the paper's 72 was V1) and the masks say which slots are live
  (use_mask_condition true in the configs). adaRMS: Qwen's RMSNorm (with its weight) x (1 +
  scale) + shift and residual += **tanh(gate)** x update, (scale, shift, gate) = Linear(cond)
  per norm; cond is the sinusoidal time embedding itself (width 1024, periods 4e-3 .. 4; **no
  time MLP**). The final adaRMS (no gate), action_out_proj on the last 30 rows in fp32.
* **flow**: 10 Euler steps from t = 1: x_t = (x_t - v_t / 10) x action_mask, t = max(t - 0.1,
  0) accumulated in fp32; the noise N(0, 1) x action_mask. The output is x_t after the last
  step: normalized 96-D slots (the dataset's norm.json unnormalizes the live ones).

Spec (opentpu/llm/rhinovla.Spec): the defaults above; `Spec.from_config(assets_dir, yaml)`
reads them from the assets' config.json and a training yaml's action_expert section
(configs/training/demo_full_finetune.yaml and demo_ae_finetune.yaml agree on every value).

## 2. The reference

tools/rhinovla_reference.py runs the repo's own `RhinoVLA.predict_action` in fp32 on the CPU
(venv `/Users/rachittibrewal/ergodex-03/pi05-venv`: torch 2.11, transformers 5.5.4; the repo
imported with its two GPU-side imports stubbed: qwen_vl_utils.process_vision_info returns the
messages' PIL images, accelerate.logging a logger) on a synthetic observation (three smooth
random images, a 96-D state, a task, seed 0, the noise passed in by replacing sample_noise)
and saves 54 intermediate tensors. opentpu/llm/rhinovla.reference_* is the same model in numpy
fp32. tools/rhinovla_check.py --reference, max |a - b| / max |b| (measured):

| stage | error |
|---|---|
| tower: patch + position embedding / block 0 / block 1 | 2.7e-6 / 2.0e-6 / 2.4e-6 |
| merged rows (the prefix's image rows) / the deepstack rows | 2.1e-6 / 2.8e-6 |
| every text layer's K, V | <= 2.5e-6 |
| the mRoPE positions [3, L], the rope delta, the time schedule | equal |
| every step's time embedding / x_t / velocity | 0 (bit-exact) / <= 1.5e-7 / <= 3.7e-7 |
| actions (normalized slots) | 1.4e-7 relative, 7.2e-7 absolute |

**On a tiny model** (Spec TINY: 2 tower blocks of 128 wide with deepstack from block 0, 4 text
layers of 256 wide with 2 / 1 heads of 128, a 2-layer expert of 128 wide at offset 2, 64 x 64
images = 4 tokens a view, 53 prefix rows, a chunk of 10 x 96, 3 steps), its weights drawn by
rhinovla.random_weights (one source for the tests and the tool) and loaded into the repo's
model **strictly**: every tensor name of the port is the model's (the tool refuses otherwise).
The tiny model is built by the repo's own code path (RhinoVLA(config) on a temporary copy of
the assets with the tiny sizes in config.json), so the processor, the chat template, the
deepstack injection, get_rope_index and the expert's masks are the repo's, not re-implemented.

**The real checkpoint** (HuixiAI/RhinoVLA-V3, `rhinovla_pretrain.ckpt`, 6.07 GB) was not fetched:
the Mac had 14 GiB free against the rule of 35 GiB before a 6 GB download. The commands, once
it is there (a bf16 state dict; the tool loads it through the repo's TrainerUtils and casts to
fp32: ~9 GB of RAM):

    python tools/rhinovla_reference.py --repo RhinoVLA --ckpt rhinovla_pretrain.ckpt \
        --config RhinoVLA/configs/training/demo_full_finetune.yaml --out ref.npz
    python tools/rhinovla_check.py --ref ref.npz --ckpt rhinovla_pretrain.ckpt \
        --assets RhinoVLA/rhinovla/assets/qwen3_vl_processor \
        --config RhinoVLA/configs/training/demo_full_finetune.yaml --reference

(The prompt's token ids come from the processor in the .npz; the numpy side never tokenizes.)

## 3. The mapping onto the card

One device program per phase; the host launches them in order (Image.runs, run_chunk). Every
weight is int8 (MX: per-row scales, power-of-two block offsets) in DRAM; each part's layers lie
in equal blocks, so a layer index (a run argument or a hardware loop's counter) addresses them.
No new op anywhere.

| program | instructions (TINY) | runs a chunk | work |
|---|---|---|---|
| v_embed | 8 | 3 | patches [256, 768] (the host's im2col in window order) @ the folded patch matrix, + bias + the interpolated position rows |
| v_qkv / v_attend / v_out (v_last) | 98 / 51 / 62 (74) | 3 x 24 each | LayerNorm, q / k / v with biases, **2D RoPE on q and k** (rope_rows with the host's [256, 32] table), the int8 KV; attention; proj + MLP (gelu_tanh); v_last LayerNorms the stored rows with the merger's norm |
| v_deep | 87 | 3 x 3 | the deepstack merger k of the residual rows after block 5 / 11 / 17: LayerNorm over [4096], fc1, **erf GELU**, fc2, into the DS rows |
| v_merge | 69 | 3 | the main merger (no norm: v_last did it per token) into the view's 64 prefix rows |
| p_qkv / p_attend / p_out | 154 / 110 / 84 | 28 / 27 / 27 | RMSNorm w (QACT RSCALE / CSCALE), q / k / v, **q_norm / k_norm** (lib.rmsnorm on each head's [R, 128]), mRoPE from the host's per-row tables, K / V of 8 heads into the layer's cache; **causal attention** (16 query heads on 8 KV heads: _attend_heads' per-entry heads); x += O Wo^T, x += SwiGLU (pi05.geglu with act=silu) |
| p_deep | 13 | 3 | the three views' DS rows of deepstack layer k added to the prefix's image rows, after p_out of layers 0, 1, 2 |
| e_step | 2655 | 10 | one Euler step, the 18 layers in one hardware loop, the step a run argument |

* **The causal mask** is an attention.Additive tile: the host writes a table whose row r is
  query row r's additive mask over the keys (-0 where key <= r, -inf past it, a pad word), and
  a group of MCOLS query rows at loop-affine r0 takes the tile at base + r0 x stride, so one
  hardware loop serves every group of the prefix and the masked keys weigh exactly 0 (the
  scores start as the tile, q.K^T is accumulated into it with the row maxima: two
  instructions a block). The attention's MACs are the whole square, not the triangle (13 G of
  the prefix's 325 G: not worth a triangle).
* **The host writes** the view's patch rows before its tower; after the tower the text rows
  (41 rows at their positions among the image rows), the mRoPE cos / sin tables of the prefix
  and the suffix, the mask table; after the prefix the suffix's state row (state_proj +
  state_mask_proj: fp32), the action rows' bias (action_in_proj's + action_mask_proj(mask)),
  action_out_proj's rows and bias times the action mask, and x_0^T = (noise x mask)^T
  [96, 30]. It reads x_t^T once at the end. The time embedding and every norm's adaRMS (gain
  = w (1 + scale), shift, tanh(gate)) for each step are computed at build time in fp32 (torch's
  arithmetic for the embedding: bit-exact with the repo's), the adaRMS table indexed by
  (step, layer).
* **action_in_proj / action_out_proj in fp32 on the VPU** (π0.5's io32): x_t lives transposed in
  DRAM; the in-projection is 96 OUTERs over the 30 action rows, the state row a store of the
  host's; the out-projection 96 RDOTs of the final adaRMS rows of the 30 action rows, then
  v_t + b and x_t += dt v_t per slot. With W_out's rows pre-multiplied by the mask and x_0 =
  noise x mask, the dead slots stay exactly 0 (fp32 x 0 is exact).
* **The prefix KV** of each layer is written once (every KV head: K token-major, V^T rows cap
  wide); the expert appends its 31 rows after the prefix's Np in the cache of backbone layer
  10 + i each step, so its attention is one pass over Np + 31 keys of that layer.
* Rows run in groups of R (tower), RP (prefix), RE (expert), RM (mergers' windows) inside
  hardware loops. The e_step program (18 layers looped) is 2655 instructions at TINY and
  fits IMEM; at the real sizes its loop body does not grow (the chunks of the MLP are the
  same 256), the vision programs grow with the heads (16 heads of q, k, v a row group).

ISA simulator runs (OTPU_TARGET f2-hbm; the card's int8 everywhere except the fp32 pieces
above), against the fp32 reference of the same model and inputs (tools/rhinovla_check.py
--isa --emulated on TINY; 12 s in all):

| | error |
|---|---|
| the merged image rows (after v_merge, the deepstack path included) | cosine >= 0.99973, 2.3% relative |
| every step's velocity | 0.3% relative |
| actions: max abs 0.0063 over the live slots' range 10.06 | **0.063% of the range** against fp32 numpy, 0.063% against the repo's PyTorch |
| the card against Emulated (the card's int8 points in float64) | 0.030% of the range (Emulated itself is 0.084% from fp32) |

So the card sits inside the emulation's own error band; it is not bit-exact with it (float64
sums, a one-pass softmax, π0.5's caveat) and bit-exactness against a float model is not a
property the ISA simulator's fp32 VPU can be held to. tests/test_rhinovla.py runs the same
chunk with 40 of the 96 slots live and checks the dead ones are exactly 0 (5 passed, 22 s).

## 4. int8 accuracy and the precision map

The design's Phase 7 exit metric is max |a - a_ref| over the live slots of the 30 x 96 chunk
(96 in V3; the design wrote 72, V1's) divided by the chunk's range, on the ISA simulator, with
a 2% bound at the Phase 8 precision. tools/rhinovla_sensitivity.py emulates the card's
quantization points in float64 (Emulated, restrictable to a part and to a matrix group) and:

* measures the variants w8a8 (the card today), w8a16, backbone_w8a16 (the paper's precision),
  kv_only, <part>_only, and the sweep `a_<group>`: one group at W8A8 with everything else at
  W8A16 (weights int8 everywhere), the group's attributed error; the groups are the tower's
  patch / qkv / proj / fc1 / fc2 / mergers / attention operands, the backbone's qkv / o /
  gate-up / down / attention operands, the expert's likewise (17 groups);
* `--map` builds the precision map by the design's procedure: start at W8A8 everywhere; while
  the chunk is above the bound, promote to W8A16 the group with the largest attributed error
  (the backbone's groups first among errors within 10% of each other), re-measure; print the
  map, the share of the chunk's MACs at W8A16 and the MXU seconds it implies at the f2-vla
  point (a W8A16 MM runs at one product a DSP, half the packed rate: its seconds double;
  --mxu takes the tally's vision / prefix / expert seconds).

Measured on TINY only (a smoke run: every variant is under 0.1% of the range on a random
model, w8a8 0.08%, w8a16 0.03%, the expert's down / gate-up the largest attributed errors at
0.05 / 0.04%; with the bound forced to 0.05% the map promotes e_down then e_gateup, 4% of the
MACs, MXU 0.440 -> 0.465 s). **The real map needs the real weights**:

    python tools/rhinovla_sensitivity.py --ref ref.npz --ckpt rhinovla_pretrain.ckpt \
        --assets RhinoVLA/rhinovla/assets/qwen3_vl_processor \
        --config RhinoVLA/configs/training/demo_full_finetune.yaml --map --json map.json
    python tools/rhinovla_check.py --ref ref.npz --ckpt ... --assets ... --config ... --isa

(each variant is one fp32 forward of the 2B on 233 rows plus 10 expert steps, ~1-2 min on the
CPU; the sweep and the map ~40 of them; the ISA run of the full chunk is hours, as π0.5's.)
What to expect from π0.5's measurement (docs/pi05.md 4: W8A8 6.8% of the range, the 2B's
activations nearly all of it, W8A16 on the 2B 1.7%) and the paper's (W8A8 hurt success): the
backbone's gate-up and qkv inputs promoted first; the design's Approach B already carries the
backbone-W8A16 case (section 5's table gives the MXU seconds for it).

## 5. The cost of a chunk (ISA tally)

tools/rhinovla_cost.py tallies every instruction a chunk executes (the programs compiled for a
3-view prompt of 41 text tokens = 233 prefix rows, loop counts expanded, nothing run) at
profile._describe's nominal rates, pi05_cost's method: MXU N x KB x ceil(M / MCOLS) cycles an
MM (PACK: ceil(N / 2) x KB), port B the D-byte chunks at PORTB a cycle, VPU rows x ceil(cols /
LANES) a VOP by kind, QACT 2 ceil(n / LANES), QST with QST T, DMA. 200 MHz, seconds. A chunk's
time lies between the slowest unit's (every unit overlapped: "bound") and the sum ("serial").
The row groups are the smallest bound's among those that compile (TMEM holds the work areas).

MACs: **0.686 T a chunk** (vision 0.266 T for the 3 views with the mergers and the deepstack
mergers, prefix 0.325 T at 233 rows, the 10 expert steps 0.095 T), against the shape tally's
0.66 T (tools/rhinovla_shape.py: no mergers) and π0.5's measured 1.47 T.

### Today: f2-hbm, S = 1, D = 128, MCOLS 4, ACT_ROWS 16, LANES 8, TMEM 64K words

Row groups R / RP / RE / RM = 16 / 8 / 8 / 4.

| phase | MXU | port B | VPU | GELU | SiLU | softmax | norms | RoPE | rest | QACT | QST | DMA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vision (3 views) | 2.601 | 0.931 | 0.841 | 0.425 | - | 0.127 | 0.074 | 0.071 | 0.145 | 0.315 | 0.165 | 0.130 |
| prefix (233 rows) | 3.214 | 1.685 | 0.318 | - | 0.121 | 0.081 | 0.054 | 0.038 | 0.025 | 0.146 | 0.029 | 0.079 |
| one expert step | 0.096 | 0.053 | 0.026 | - | 0.005 | 0.005 | 0.006 | 0.003 | 0.006 | 0.011 | 0.003 | 0.005 |
| **chunk (10 steps)** | **6.77** | **3.15** | **1.42** | 0.43 | 0.17 | 0.26 | 0.19 | 0.14 | 0.23 | 0.57 | 0.22 | 0.26 |

Bound 6.8 s, serial 12.4 s: **~7-12 s a chunk today** (0.08-0.15 Hz), the MXU first, then port
B (the prefix's 2B streams 30 times: TMEM caps the row group at 8 rows, as π0.5's). The shape
tally said 6.6 s of MXU; the tower is 0.4 s more than its share by MACs (the heads of 64 are
padded to 128 in q.K^T and P.V, and the mergers were not in the shape tally).

### The f2-vla points (ISA Config only, no RTL; 200 MHz)

Row groups 128 / 255 / 31 / 64 at every point (the prefix in one 233-row pass, the expert's
31 rows in one group).

| config | MXU | port B | VPU (GELU / SiLU) | QACT | QST | DMA | bound | serial | Hz |
|---|---|---|---|---|---|---|---|---|---|
| today | 6.774 | 3.145 | 1.424 (0.43 / 0.17) | 0.568 | 0.222 | 0.259 | 6.77 | 12.39 | 0.08-0.15 |
| MCOLS 32 unpacked, ACT_ROWS 255, LANES 64, TMEM 1.5M, PORTB 1 (Approach A) | 0.881 | 0.341 | 0.188 (0.053 / 0.015) | 0.071 | 0.137 | 0.059 | 0.88 | 1.68 | 0.60-1.14 |
| **MCOLS 32 PACK (64 logical), ACT_ROWS 255, LANES 64, TMEM 1.5M, PORTB 1 (f2-vla as defined)** | **0.440** | 0.341 | 0.188 | 0.071 | 0.137 | 0.059 | **0.44** | **1.24** | **0.81-2.27** |
| the same at PORTB 2 | 0.441 | 0.171 | 0.188 | 0.071 | 0.137 | 0.059 | 0.44 | 1.07 | 0.94-2.27 |

Per phase at the f2-vla point (PACK, PORTB 1 / 2):

| phase | MXU | port B | VPU | QACT | QST | DMA | bound | serial |
|---|---|---|---|---|---|---|---|---|
| vision (3 views) | 0.163 | 0.130 / 0.065 | 0.114 | 0.039 | 0.103 | 0.029 | 0.163 | 0.578 / 0.513 |
| prefix | 0.218 | 0.081 / 0.040 | 0.040 | 0.018 | 0.018 | 0.019 | 0.218 | 0.394 / 0.354 |
| one expert step | 0.006 | 0.013 / 0.007 | 0.003 | 0.001 | 0.002 | 0.001 | 0.013 / 0.007 | 0.026 / 0.020 |

Reading it:
* **The MXU bound at the f2-vla point is 0.44 s (2.3 Hz), under the design's shape estimate of
  0.53 s**: the prefix is 233 rows in one 255-row pass (ceil(233 / 32) = 8 column passes a
  weight chunk; with PACK two chunks an advance), and the tower's 256 rows a view two passes
  of 128. Unpacked (Approach A) 0.88 s, 1.1 Hz at the line, as the design said.
* **The serial sum is 1.24 s, 2.8x the bound**: the units are more balanced than π0.5's (where
  the MXU was 0.95 of a 1.96 s sum). After the MXU: port B 0.34 s (0.17 at PORTB 2; the expert
  step is port-bound at PORTB 1, 0.013 against 0.006 s of MXU, as the design's L7 said; the
  tower streams its weights twice a view, 3 views: 0.13 s, which batching the 3 views as 768
  rows of one pass would cut to ~0.05 s, plan L10), the VPU 0.19 s (the tower's gelu_tanh over
  256 x 4096 x 24 x 3 elements 0.053 s, the mergers' erf GELU, RoPE 0.018, softmax 0.015),
  **QST 0.14 s** (0.10 s of it the tower: 24 blocks x 16 heads x 256 tokens x 3 views of K and
  V^T stores at a word a cycle on port A, L11's remainder; the stores as D-byte chunks on the
  write path would take it to ~0.03 s), QACT 0.07, DMA 0.06.
* So **RhinoVLA at the f2-vla point is 0.8 Hz with no overlap and 2.3 Hz with every unit
  overlapped**; the RTL's measured overlap on π0.5's phases (docs/pi05.md 5.3: 1.2-1.5x the
  slowest unit at MCOLS 16 with 16 lanes, up to 2-3x at MCOLS 32 where the VPU at 16 lanes was
  the longest unit) puts it at **1.5-1.9 Hz at PORTB 1 and 1.6-2.0 at PORTB 2 if the f2-vla
  RTL at 64 lanes overlaps as the MCOLS 16 phases did**, and at the line (~1 Hz) if it
  overlaps as the MCOLS 32 / 16-lane phases did. With the backbone at W8A16 (the expected
  Phase 8 case, half the packed rate on the prefix's MMs) the MXU is 0.66 s and the serial sum
  1.46 s: 0.7-1.5 Hz, ~1.1-1.3 at the MCOLS 16 overlap. The RTL run of a cut phase at the
  f2-vla sizes (tools/pi05_rtl.py's method on p_out / v_out / e_step) is the next measurement;
  it was not run here.
* The real chunk's time also carries 3 x 24 x 3 + 28 x 3 + 10 ≈ 310 program launches (the
  host's; π0.5's ~190): a chain of programs (the sequencer) removes them, as docs/pi05.md 6 says.

## 6. Remaining work

* **The real weights**: the reference against the repo's model on the checkpoint, the ISA run,
  the sensitivity sweep and the precision map (section 2 and 4's commands; a 6 GB download and
  ~10 GB of RAM); then the design's Phase 7 exit number (max |a - a_ref| / range at the map's
  precision on the ISA simulator) instead of TINY's 0.06%.
* The 3 views as one 768-row tower pass (port B 0.13 -> ~0.05 s), the tower's K / V stores on
  the D-byte write path (QST 0.10 -> ~0.03 s), the whole chunk as one program chain.
* W8A16 on the card (plan L9) for the groups the map promotes; the design's Approach C (the
  expert re-stepped on fresh state at ~8 Hz: 0.026 s a step serial here) once a rollout harness
  exists.
* The RTL timing of the cut phases at the f2-vla sizes (p_out at 233 rows, v_out, e_step with
  its 31 rows), and the full chunk once the overlap is measured; FPGA only in the nightly.
