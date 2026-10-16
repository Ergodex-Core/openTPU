"""RhinoVLA V3 (HuixiAI/RhinoVLA, branch main = the V3 release of 2026-10-15; the checkpoint
`rhinovla_pretrain.ckpt`, 6.07 GB, HuixiAI/RhinoVLA-V3) on openTPU: docs/rhinovla.md.

The model, as the repo's code runs it (rhinovla/model/framework/RhinoVLA.py predict_action,
modules/action_expert.py, HF transformers' modeling_qwen3_vl.py), not as the paper's V1
(arXiv 2606.07383, Table 8) describes it; where they differ the code wins:

  * Qwen3-VL-2B as built from rhinovla/assets/qwen3_vl_processor/config.json. The ViT: 24
    blocks, 1024 wide, 16 heads of 64, MLP 4096 gelu_pytorch_tanh, LayerNorms with biases
    (eps 1e-6), a fused qkv [3072, 1024] with bias, proj with bias; patch 16 with a temporal
    patch of 2 (the image repeated: a [1024, 3 x 2 x 16 x 16] Conv3d, folded here to [1024,
    768] on the host), a learned 48 x 48 position table bilinearly interpolated to the grid,
    2D RoPE (theta 10000, 16 frequencies on the row, 16 on the column, rotate-half over the
    64 of a head), the tokens in 2 x 2 window order (the processor's), the merger LayerNorm
    (1024, per token) -> [4096] -> fc1 4096 -> GELU (erf) -> fc2 2048; **deepstack**: the
    outputs of blocks 5, 11 and 17 through their own mergers (LayerNorm over the 4096 of a
    window, fc1, GELU, fc2) are added to the language model's image rows after its decoder
    layers 0, 1 and 2.
  * Images are 256 x 256 (the configs' image_size; the processor's pixel budget 65536 ..
    131072 keeps them), 16 x 16 patches merged to 64 tokens a view, 3 views (the configs'
    top_head, hand_left, hand_right). Pixels (x / 255 - 0.5) / 0.5.
  * The prompt (prompt_template v2 of modules/qwen.py): the user turn "[top_head|rgb]<img>
    \\n[hand_left|rgb]<img>\\n[hand_right|rgb]<img>\\n<task>" with the generation prompt:
    41 text tokens around the 3 x 64 image tokens, **233 prefix rows** for a 7-word task.
  * The text model: 28 layers, 2048 wide, 16 query heads and 8 KV heads of 128 with q_norm /
    k_norm (RMSNorm over the head, eps 1e-6), interleaved mRoPE (theta 5e6, sections [24,
    20, 20]: of the 64 frequencies, index f < 60 takes the T / H / W position for f mod 3 =
    0 / 1 / 2, f >= 60 the T; text tokens have one position on all three axes, an image's
    tokens the view's start on T and start + row / column of the 8 x 8 merged grid on H / W,
    the next text token starts 8 past the view's start), **causal** attention over the
    prefix (the standard decoder; nothing bidirectional), SwiGLU MLP 6144 (silu(gate) up).
    Only every layer's K (after k_norm and RoPE) and V are kept.
  * The action expert (RhinoActionExpert): 18 layers (the configs' depth 18; the dataclass
    default 10 is not what the configs train), 1024 wide, MLP 3072 SwiGLU, 16 / 8 heads of
    128 with q_norm / k_norm and the same mRoPE tables, attending to the prefix KV of
    backbone layer `prefix_layer_offset + i` = 10 + i (the configs' prefix_layer_offset 10:
    the last 18 of 28) and to the suffix itself (bidirectional among the suffix), at
    positions max prefix position + 1 + i. The suffix is 1 state token (state_proj(state x
    state_mask) + state_mask_proj(state_mask)) and 30 action tokens (action_in_proj(x_t) +
    action_mask_proj(action_mask)); actions and state are 96-D slots (V3; the paper's 72 was
    V1), the masks say which slots are live. adaRMS: Qwen's RMSNorm (with its weight) x (1 +
    scale) + shift, residual += tanh(gate) x update, (scale, shift, gate) = Linear(cond)
    per norm; cond = the sinusoidal time embedding (width 1024, periods 4e-3 .. 4) itself:
    no time MLP. The final adaRMS (no gate), action_out_proj on the last 30 rows in fp32.
  * 10 Euler steps: t = 1, x_t = (x_t - v_t / 10) x action_mask, t = max(t - 0.1, 0) (fp32
    accumulated); the noise is N(0, 1) x action_mask.

This module holds Spec, Weights, the host side (patch rows, the position and RoPE tables,
the prompt's layout, the time embedding and every step's adaRMS modulation), reference_*
(numpy fp32, the device's numerics target; tools/rhinovla_reference.py checks it against
the repo's PyTorch model), Exact / Emulated (the card's quantization points, for
tools/rhinovla_sensitivity.py), and Image / programs / run_chunk (opentpu/kernels/rhinovla.py
and kernels/qwen3vl_vit.py on the ISA simulator).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Q = "qwen.model.model."
VIS = Q + "visual."
LM = Q + "language_model."
AE = "action_expert."
AET = AE + "transformer."
AEIO = AE + "io."
IMAGE_PAD = 151655


@dataclass(frozen=True)
class Spec:
    # Qwen3-VL ViT (vision_config)
    v_hidden: int = 1024
    v_heads: int = 16
    v_ffn: int = 4096
    v_layers: int = 24
    patch: int = 16
    image: int = 256
    v_eps: float = 1e-6
    v_theta: float = 10000.0
    merge: int = 2
    v_pos_grid: int = 48            # num_position_embeddings 2304 = 48 x 48
    deepstack: tuple = (5, 11, 17)
    # Qwen3-VL-2B text model (text_config)
    hidden: int = 2048
    layers: int = 28
    ffn: int = 6144
    n_q: int = 16
    n_kv: int = 8
    head_dim: int = 128
    eps: float = 1e-6
    theta: float = 5_000_000.0
    mrope: tuple = (24, 20, 20)
    # the action expert (configs/training/*.yaml action_expert)
    e_hidden: int = 1024
    e_layers: int = 18
    e_ffn: int = 3072
    offset: int = 10                # prefix_layer_offset
    action_dim: int = 96
    state_dim: int = 96
    chunk: int = 30                 # action_horizon
    steps: int = 10                 # num_inference_timesteps
    min_period: float = 4e-3
    max_period: float = 4.0
    mask_condition: bool = True
    views: int = 3

    @property
    def v_head_dim(self) -> int:
        return self.v_hidden // self.v_heads

    @property
    def side(self) -> int:
        """Patches a side (16)."""
        return self.image // self.patch

    @property
    def n_patch(self) -> int:
        """Patches a view (256)."""
        return self.side ** 2

    @property
    def n_tok(self) -> int:
        """Prefix rows a view after the 2 x 2 merge (64)."""
        return self.n_patch // self.merge ** 2

    @property
    def rows(self) -> int:
        """Suffix rows: the state token and the action chunk (31)."""
        return 1 + self.chunk

    @property
    def e_q(self) -> int:
        return self.n_q * self.head_dim

    @property
    def e_kv(self) -> int:
        return self.n_kv * self.head_dim

    @staticmethod
    def from_config(assets_dir, training_yaml=None) -> "Spec":
        """The Spec of a Qwen3-VL config directory (rhinovla/assets/qwen3_vl_processor) and a
        training config's action_expert section (configs/training/demo_full_finetune.yaml;
        default the V3 configs' values)."""
        c = json.loads((Path(assets_dir) / "config.json").read_text())
        v, t = c["vision_config"], c["text_config"]
        kw = dict(v_hidden=v["hidden_size"], v_heads=v["num_heads"], v_ffn=v["intermediate_size"],
                  v_layers=v["depth"], patch=v["patch_size"], merge=v["spatial_merge_size"],
                  v_pos_grid=int(round(v["num_position_embeddings"] ** 0.5)),
                  deepstack=tuple(v["deepstack_visual_indexes"]),
                  hidden=t["hidden_size"], layers=t["num_hidden_layers"],
                  ffn=t["intermediate_size"], n_q=t["num_attention_heads"],
                  n_kv=t["num_key_value_heads"], head_dim=t["head_dim"], eps=t["rms_norm_eps"],
                  theta=float(t["rope_theta"]), mrope=tuple(t["rope_scaling"]["mrope_section"]))
        if v["out_hidden_size"] != t["hidden_size"] or v["hidden_act"] != "gelu_pytorch_tanh" \
                or t["hidden_act"] != "silu" or v["temporal_patch_size"] != 2:
            raise ValueError("not the Qwen3-VL layout this port lays out")
        if training_yaml is not None:
            import yaml
            y = yaml.safe_load(Path(training_yaml).read_text())
            a = y["framework"]["action_expert"]
            img = y["datasets"]["vla_data"].get("image_size", [256, 256])
            kw.update(e_hidden=a["width"], e_layers=a["depth"], e_ffn=a["mlp_dim"],
                      action_dim=a["action_dim"], state_dim=a["state_dim"],
                      chunk=a["action_horizon"],
                      offset=a.get("prefix_layer_offset", kw["layers"] - a["depth"]),
                      steps=a.get("num_inference_timesteps", 10),
                      mask_condition=bool(a.get("use_mask_condition", False)),
                      image=int(img[0]) if hasattr(img, "__iter__") else int(img),
                      views=len(y["datasets"]["vla_data"].get("camera_names", [0, 1, 2])))
            if not a.get("use_state_token", True) or not a.get("use_adarms", True):
                raise ValueError("only use_state_token and use_adarms (V3's) are laid out")
        return Spec(**kw)


class Weights:
    """The checkpoint's tensors by the model's names (fp32 numpy; bf16 widened), from a
    torch.save'd state dict (rhinovla_pretrain.ckpt, steps_<n>_pytorch_model.pt; a "module."
    prefix stripped) or a .safetensors; or wrap a dict of arrays (the tests')."""

    def __init__(self, path):
        import torch
        p = Path(path)
        if p.suffix == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(str(p))
        else:
            try:
                sd = torch.load(str(p), map_location="cpu", mmap=True, weights_only=True)
            except Exception:
                sd = torch.load(str(p), map_location="cpu", weights_only=False)
            for k in ("state_dict", "model", "module"):
                if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
        self._t = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[7:]
            self._t[k] = v

    def __contains__(self, k) -> bool:
        return k in self._t

    def keys(self):
        return self._t.keys()

    def __getitem__(self, k) -> np.ndarray:
        return self._t[k].float().numpy()

    def rows(self, k, idx) -> np.ndarray:
        """Rows idx of a matrix (the embedding table: no fp32 copy of the whole)."""
        t = self._t[k]
        return np.stack([t[int(i)].float().numpy() for i in idx])


def rows_of(W, k, idx) -> np.ndarray:
    if hasattr(W, "rows"):
        return W.rows(k, idx)
    return np.asarray(W[k], np.float32)[np.asarray(idx, int)]


# =============================================================================== host side
def window_order(side: int, merge: int = 2) -> np.ndarray:
    """The processor's token order of a side x side patch grid: (grid row, grid column) of
    token i: 2 x 2 windows row-major over the grid, the 4 patches of a window row-major."""
    g = side // merge
    bh, bw, ih, iw = np.meshgrid(np.arange(g), np.arange(g), np.arange(merge), np.arange(merge),
                                 indexing="ij")
    r = (bh * merge + ih).reshape(-1)
    c = (bw * merge + iw).reshape(-1)
    return np.stack([r, c], 1)


def patches(img: np.ndarray, patch: int = 16, merge: int = 2) -> np.ndarray:
    """[3, H, W] pixels -> [N, 3 p p] patch rows in the processor's window order, each
    flattened (channel, row, column): the Conv3d's weight summed over its temporal pair and
    reshaped to [C, 3pp] makes the patch embedding a matrix product (the processor feeds the
    image twice along the temporal axis)."""
    c, h, w = img.shape
    g = img.reshape(c, h // patch, patch, w // patch, patch).transpose(1, 3, 0, 2, 4)
    g = g.reshape(h // patch, w // patch, c * patch * patch)
    rc = window_order(h // patch, merge)
    return np.ascontiguousarray(g[rc[:, 0], rc[:, 1]], np.float32)


def patch_matrix(spec: Spec, W) -> tuple[np.ndarray, np.ndarray]:
    """The patch embedding as [C, 3 p p] (the Conv3d's two temporal slices summed, float64
    then fp32) and its bias."""
    w = np.asarray(W[VIS + "patch_embed.proj.weight"], np.float64)      # [C, 3, 2, p, p]
    return (w.sum(2).reshape(w.shape[0], -1).astype(np.float32),
            np.asarray(W[VIS + "patch_embed.proj.bias"], np.float32))


def pos_table(spec: Spec, W) -> np.ndarray:
    """The learned position table (v_pos_grid^2 rows) bilinearly interpolated to the side x
    side grid (Qwen3VLVisionModel.fast_pos_embed_interpolate: fp32 linspace corners and
    weights), in window order: [N, C]."""
    n, side = spec.v_pos_grid, spec.side
    E = np.asarray(W[VIS + "pos_embed.weight"], np.float32)
    idx = np.linspace(0, n - 1, side, dtype=np.float32)
    fl = idx.astype(np.int32)
    ce = np.minimum(fl + 1, n - 1)
    d = idx - fl.astype(np.float32)
    out = np.zeros((side, side, E.shape[1]), np.float32)
    for hi, hw in ((fl, 1 - d), (ce, d)):
        for wi, ww in ((fl, 1 - d), (ce, d)):
            wt = (hw[:, None] * ww[None, :]).astype(np.float32)
            out += E[hi[:, None] * n + wi[None, :]] * wt[:, :, None]
    rc = window_order(side, spec.merge)
    return np.ascontiguousarray(out[rc[:, 0], rc[:, 1]])


def vit_rope_table(spec: Spec) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin [N, d / 2] of the ViT's 2D RoPE in window order: the first d / 4 frequencies
    at the token's grid row, the next d / 4 at its column (Qwen3VLVisionRotaryEmbedding(d / 2)
    over rot_pos_emb's (row, column) ids; rotate-half over the head's d)."""
    d = spec.v_head_dim
    inv = (1.0 / (spec.v_theta ** (np.arange(0, d // 2, 2, dtype=np.float32) / (d // 2)))
           ).astype(np.float32)
    rc = window_order(spec.side, spec.merge).astype(np.float32)
    ang = np.concatenate([rc[:, 0:1] * inv[None, :], rc[:, 1:2] * inv[None, :]], 1)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


@dataclass
class Layout:
    """One prompt's prefix: its token ids (the image pads in place), where each view's rows
    start, every row's mRoPE positions [3, L] and the suffix's first position."""
    ids: np.ndarray
    starts: list
    pos: np.ndarray
    suffix_pos: int

    @property
    def L(self) -> int:
        return len(self.ids)

    def text(self):
        """(row, token id) of the text tokens."""
        return [(i, int(t)) for i, t in enumerate(self.ids) if t != IMAGE_PAD]


def layout(spec: Spec, ids) -> Layout:
    """The prefix layout of token ids (the processor's: n_tok image pads per view in place):
    Qwen3VLModel.get_rope_index for images of a (merge x side / merge) grid."""
    ids = np.asarray(ids, np.int64)
    g = spec.side // spec.merge
    pos = np.zeros((3, len(ids)), np.int64)
    starts, cur, i = [], 0, 0
    while i < len(ids):
        if ids[i] != IMAGE_PAD:
            pos[:, i] = cur
            cur += 1
            i += 1
            continue
        n = spec.n_tok
        if i + n > len(ids) or (ids[i:i + n] != IMAGE_PAD).any():
            raise ValueError(f"image pads at {i}: not {n} of them")
        starts.append(i)
        hh, ww = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        pos[0, i:i + n] = cur
        pos[1, i:i + n] = cur + hh.reshape(-1)
        pos[2, i:i + n] = cur + ww.reshape(-1)
        cur += g
        i += n
    if len(starts) != spec.views:
        raise ValueError(f"{len(starts)} views in the prompt, the spec has {spec.views}")
    return Layout(ids, starts, pos, int(pos.max()) + 1)


def text_rows(spec: Spec, W, lay: Layout) -> np.ndarray:
    """The text tokens' embedding rows [n_text, hidden] (no scaling)."""
    return rows_of(W, LM + "embed_tokens.weight", [t for _, t in lay.text()]).astype(np.float32)


def mrope_table(spec: Spec, pos3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin [L, head_dim / 2] of the interleaved mRoPE (Qwen3VLTextRotaryEmbedding) for
    positions pos3 [3, L] (T, H, W): frequency f takes the axis f mod 3 for f below 3 x the
    H / W sections, the T axis past them; fp32 angles."""
    d = spec.head_dim
    inv = (1.0 / (spec.theta ** (np.arange(0, d, 2, dtype=np.int64).astype(np.float32) / d))
           ).astype(np.float32)
    f = np.arange(d // 2)
    axis = np.zeros(d // 2, np.int64)
    for a in (1, 2):
        sel = (f % 3 == a) & (f < spec.mrope[a] * 3)
        axis[sel] = a
    p = np.asarray(pos3, np.float32)[axis, :].T            # [L, d/2]: each frequency's position
    ang = p * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope_tables(spec: Spec, lay: Layout, n_suffix: int | None = None):
    """The prefix's and the suffix's cos / sin as one table [L + rows, d / 2]: the suffix at
    positions suffix_pos .. on all three axes."""
    n = spec.rows if n_suffix is None else n_suffix
    sp = np.repeat(np.arange(lay.suffix_pos, lay.suffix_pos + n)[None, :], 3, 0)
    return mrope_table(spec, np.concatenate([lay.pos, sp], 1))


def times(spec: Spec) -> list[float]:
    """The Euler schedule as predict_action keeps it: t = 1, then t = max(t + dt, 0) in fp32."""
    dt = np.float32(-1.0 / spec.steps)
    t, out = np.float32(1.0), []
    for _ in range(spec.steps):
        out.append(float(t))
        t = np.float32(max(np.float32(t + dt), np.float32(0.0)))
    return out


def time_embedding(spec: Spec, t: float) -> np.ndarray:
    """create_sinusoidal_pos_embedding(t, e_hidden) on the CPU (fp32 throughout): [sin, cos]
    of t 2 pi / period, the periods geometric from min_period to max_period. In torch's
    arithmetic when it is there (numpy's fp32 pow and linspace differ by an ulp at the highest
    frequencies, 1e-4 of the row): the table is the host's."""
    half = spec.e_hidden // 2
    try:
        import torch
        frac = torch.linspace(0.0, 1.0, half, dtype=torch.float32)
        period = spec.min_period * (spec.max_period / spec.min_period) ** frac
        x = torch.tensor([t], dtype=torch.float32)[:, None] * (2 * math.pi / period)[None, :]
        return torch.cat([torch.sin(x), torch.cos(x)], -1)[0].numpy()
    except ImportError:                 # pragma: no cover
        frac = np.linspace(0.0, 1.0, half, dtype=np.float32)
        period = (np.float32(spec.min_period) * np.float32(spec.max_period / spec.min_period) ** frac
                  ).astype(np.float32)
        x = (np.float32(t) * (np.float32(2 * math.pi) / period)).astype(np.float32)
        return np.concatenate([np.sin(x), np.cos(x)]).astype(np.float32)


def modulations(spec: Spec, W, cond: np.ndarray) -> dict:
    """Every expert adaRMS norm's (gain, shift, gate) for one condition: gain = w (1 + scale)
    (the RMSNorm's weight folded), gate = tanh(gate); {(layer, "in" / "post"): (gain, shift,
    gate)} and "final": (gain, shift)."""
    out = {}

    def dense(p):
        m = (cond @ W[p + "cond.weight"].T + W[p + "cond.bias"]).astype(np.float32)
        sc, sh, g = np.split(m, 3)
        w = np.asarray(W[p + "norm.weight"], np.float32)
        return (w * (1 + sc)).astype(np.float32), sh, np.tanh(g).astype(np.float32)
    for i in range(spec.e_layers):
        p = f"{AET}layers.{i}."
        out[i, "in"] = dense(p + "input_layernorm.")
        out[i, "post"] = dense(p + "post_attention_layernorm.")
    out["final"] = dense(AET + "norm.")[:2]
    return out


def suffix_fixed(spec: Spec, W, state, state_mask=None, action_mask=None) -> dict:
    """What the suffix embedding does not owe to x_t, host fp32: "state" the state token's
    row (state_proj(state x state_mask) + state_mask_proj(state_mask)), "bias" the action
    rows' constant (action_in_proj's bias + action_mask_proj(action_mask)), "mask" the action
    mask [action_dim] (ones: every slot live)."""
    A, S = spec.action_dim, spec.state_dim
    am = np.ones(A, np.float32) if action_mask is None else np.asarray(action_mask, np.float32)
    sm = np.ones(S, np.float32) if state_mask is None else np.asarray(state_mask, np.float32)
    st = np.asarray(state, np.float32)[:S] * sm
    row = (st @ W[AEIO + "state_proj.weight"].T + W[AEIO + "state_proj.bias"]).astype(np.float32)
    bias = np.asarray(W[AEIO + "action_in_proj.bias"], np.float32)
    if spec.mask_condition:
        row = (row + sm @ W[AE + "state_mask_proj.weight"].T + W[AE + "state_mask_proj.bias"]
               ).astype(np.float32)
        bias = (bias + am @ W[AE + "action_mask_proj.weight"].T + W[AE + "action_mask_proj.bias"]
                ).astype(np.float32)
    return {"state": row, "bias": bias, "mask": am}


# =============================================================================== reference
def _ln(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    xc = x - m
    return (xc / np.sqrt((xc * xc).mean(-1, keepdims=True) + np.float32(eps)) * w + b
            ).astype(np.float32)


def gelu_tanh(x):
    return (0.5 * x * (1 + np.tanh(np.float32(math.sqrt(2 / math.pi))
                                   * (x + np.float32(0.044715) * x ** 3)))).astype(np.float32)


def gelu_erf(x):
    """nn.GELU(): 0.5 x (1 + erf(x / sqrt 2))."""
    try:
        import torch
        return torch.nn.functional.gelu(torch.from_numpy(np.ascontiguousarray(x, np.float32))
                                        ).numpy()
    except ImportError:                 # pragma: no cover
        from math import erf
        e = np.vectorize(erf)(x.astype(np.float64) / math.sqrt(2))
        return (0.5 * x * (1 + e)).astype(np.float32)


def silu(x):
    return (x / (np.float32(1) + np.exp(-x))).astype(np.float32)


def _rms(x, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(eps))


def _rmsn(x, w, eps):
    """Qwen3VLTextRMSNorm: w (x rsqrt(mean x^2 + eps))."""
    return (w * _rms(x, eps)).astype(np.float32)


def _softmax(s):
    s = np.exp(s - s.max(-1, keepdims=True))
    return s / s.sum(-1, keepdims=True)


def _rope(x, cos, sin):
    """x [T, h, d] at per-row cos / sin [T, d/2] (rotate-half)."""
    h = x.shape[-1] // 2
    c, s = cos[:, None, :], sin[:, None, :]
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1).astype(np.float32)


def _attn(q, k, v, scale, mask=None):
    """q [T, Hq, d] against k, v [U, Hkv, d] (query head h on KV head h // (Hq / Hkv)); mask
    [T, U] True where a query may see a key (None: all)."""
    G = q.shape[1] // k.shape[1]
    kk = np.repeat(k, G, 1)
    vv = np.repeat(v, G, 1)
    s = np.einsum("thd,uhd->htu", q, kk) * np.float32(scale)
    if mask is not None:
        s = np.where(mask[None], s, np.float32(-np.inf))
    return np.einsum("htu,uhd->thd", _softmax(s), vv).astype(np.float32)


def causal(n: int) -> np.ndarray:
    return np.tril(np.ones((n, n), bool))


class Exact:
    """The reference's arithmetic: fp32 matrix products (the numerics hook; Emulated quantizes
    where the card does). lin's kind: "v" the tower's, "p" the backbone's, "e" the expert's,
    "io" the action / state projections (fp32 on the card too)."""

    def lin(self, x, W, name: str, kind: str = "") -> np.ndarray:
        return (x @ W[name].T).astype(np.float32)

    def attn(self, q, k, v, scale, mask=None, kind: str = "") -> np.ndarray:
        return _attn(q, k, v, scale, mask)


class Emulated(Exact):
    """The card's quantization points in float64 (docs/rhinovla.md 4): "w" every weight
    matrix in its MXU format (quant.quantize_mxu, dequantized), "a" every streamed product's
    activation operand as QACT makes it (one row scale, power-of-two offsets per D/4 group:
    fp32.quantize_mx), "kv" the attention's int8 operands (q per row, K per token with its
    block offsets, V per token, P V-scaled per row of each 256-key block). parts may be
    restricted to kinds ("v", "p", "e") and a_only to names' substrings (one matrix group at
    W8A8 with the rest at W8A16: the precision map's sweep). Not bit exact (float64 sums, the
    softmax in one pass): a measurement of each point's size."""

    def __init__(self, parts=("w", "a", "kv"), kinds=("v", "p", "e"), fmt: str = "int8",
                 D: int = 128, a_only=None):
        self.parts, self.kinds, self.fmt, self.D = set(parts), set(kinds), fmt, D
        self.a_only, self._w = a_only, {}

    def _wq(self, W, name):
        if name not in self._w:
            from .. import quant as Q
            if len(self._w) > 64:
                self._w.clear()
            m = np.asarray(W[name], np.float32)
            K = _up(m.shape[1], self.D)
            mp = np.zeros((m.shape[0], K), np.float32)
            mp[:, :m.shape[1]] = m
            q, sc = Q.quantize_mxu(mp, self.fmt, self.D)
            self._w[name] = Q.dequantize_mxu(q, sc, self.fmt, K, self.D)[:, :m.shape[1]]
        return self._w[name]

    def _aq(self, x, g=None, row=False):
        from .. import fp32 as F
        x = np.asarray(x, np.float32)
        n = x.shape[-1]
        K = _up(n, self.D)
        xp = np.zeros(x.shape[:-1] + (K,), np.float32)
        xp[..., :n] = x
        flat = xp.reshape(-1, K)
        g = g or self.D // 4
        q, S, d = F.quantize_mx(flat, g, row)
        deq = q.astype(np.float64) * np.repeat(np.exp2(-d.astype(np.float64)), g, axis=1) \
            * S.astype(np.float64)[:, None]
        return deq.reshape(xp.shape)[..., :n]

    def lin(self, x, W, name: str, kind: str = "") -> np.ndarray:
        if kind == "io" or kind not in self.kinds:
            return (x @ W[name].T).astype(np.float32)
        w = self._wq(W, name) if "w" in self.parts else W[name]
        qa = "a" in self.parts and (self.a_only is None or any(o in name for o in self.a_only))
        a = self._aq(x) if qa else x
        return (a @ np.asarray(w).T).astype(np.float32)

    def attn(self, q, k, v, scale, mask=None, kind: str = "") -> np.ndarray:
        if "kv" not in self.parts or kind not in self.kinds:
            return _attn(q, k, v, scale, mask)
        T, H, d = q.shape
        G = H // k.shape[1]
        qd = self._aq(q.reshape(T * H, d) * np.float32(scale)).reshape(T, H, d)
        U = len(k)
        kd = self._aq(k.transpose(1, 0, 2).reshape(-1, d), g=min(self.D, d)).reshape(
            k.shape[1], U, d)                   # QST: groups of D, the token's row scale
        vd = self._aq(v.transpose(1, 0, 2).reshape(-1, d), row=True).reshape(k.shape[1], U, d)
        out = np.zeros((H, T, d))
        for h in range(H):
            j = h // G
            s = qd[:, h] @ kd[j].T
            if mask is not None:
                s = np.where(mask, s, -np.inf)
            p = np.exp(s - s.max(-1, keepdims=True))
            l = p.sum(-1, keepdims=True)
            for b0 in range(0, U, 256):         # P quantized per 256-key block
                out[h] += self._aq(p[:, b0:b0 + 256]) @ vd[j][b0:b0 + 256]
            out[h] /= l
        return out.transpose(1, 0, 2).astype(np.float32)


EXACT = Exact()


def _mlp(W, p, y, nq, kind):
    g = nq.lin(y, W, p + "mlp.gate_proj.weight", kind)
    u = nq.lin(y, W, p + "mlp.up_proj.weight", kind)
    return nq.lin(silu(g) * u, W, p + "mlp.down_proj.weight", kind)


def _merger(W, p, x4, nq, post_norm: bool) -> np.ndarray:
    """Qwen3VLVisionPatchMerger on window rows x4 [N4, 4C] (already LayerNorm'd per token
    unless post_norm: the deepstack mergers' LayerNorm over the 4C)."""
    if post_norm:
        x4 = _ln(x4, W[p + "norm.weight"], W[p + "norm.bias"], 1e-6)
    h = gelu_erf(nq.lin(x4, W, p + "linear_fc1.weight", "v") + W[p + "linear_fc1.bias"])
    return (nq.lin(h, W, p + "linear_fc2.weight", "v") + W[p + "linear_fc2.bias"]).astype(np.float32)


def reference_vision(spec: Spec, W, img: np.ndarray, capture=None, layers=None, nq=EXACT,
                     tables=None):
    """One view [3, image, image] (pixels in [-1, 1]) -> (its n_tok prefix rows [64, hidden],
    [the deepstack rows [64, hidden] of each deepstack layer]). capture: "embed", "layer<i>".
    layers: run only the first n blocks (a cut model; the merger still runs, a deepstack
    layer past the cut is skipped). tables: (pos_table, vit cos, vit sin) if precomputed."""
    cap = capture if capture is not None else {}
    C, H, d = spec.v_hidden, spec.v_heads, spec.v_head_dim
    wp, bp = patch_matrix(spec, W)
    E, cos, sin = tables if tables is not None else (pos_table(spec, W), *vit_rope_table(spec))
    x = (nq.lin(patches(img, spec.patch, spec.merge), {"wp": wp}, "wp", "v") + bp + E
         ).astype(np.float32)
    cap["embed"] = x
    T = len(x)
    deep = []
    for i in range(spec.v_layers if layers is None else layers):
        p = f"{VIS}blocks.{i}."
        y = _ln(x, W[p + "norm1.weight"], W[p + "norm1.bias"], spec.v_eps)
        qkv = (nq.lin(y, W, p + "attn.qkv.weight", "v") + W[p + "attn.qkv.bias"]).reshape(T, 3, H, d)
        q, k, v = _rope(qkv[:, 0], cos, sin), _rope(qkv[:, 1], cos, sin), qkv[:, 2]
        o = nq.attn(q, k, v, d ** -0.5, kind="v").reshape(T, C)
        x = x + nq.lin(o, W, p + "attn.proj.weight", "v") + W[p + "attn.proj.bias"]
        y = _ln(x, W[p + "norm2.weight"], W[p + "norm2.bias"], spec.v_eps)
        h = gelu_tanh(nq.lin(y, W, p + "mlp.linear_fc1.weight", "v") + W[p + "mlp.linear_fc1.bias"])
        x = (x + nq.lin(h, W, p + "mlp.linear_fc2.weight", "v") + W[p + "mlp.linear_fc2.bias"]
             ).astype(np.float32)
        cap[f"layer{i}"] = x
        if i in spec.deepstack:
            k_ = spec.deepstack.index(i)
            deep.append(_merger(W, f"{VIS}deepstack_merger_list.{k_}.", x.reshape(-1, 4 * C), nq,
                                True))
    m = VIS + "merger."
    y = _ln(x, W[m + "norm.weight"], W[m + "norm.bias"], 1e-6)
    cap["post"] = y
    return _merger(W, m, y.reshape(-1, 4 * C), nq, False), deep


def reference_prefix(spec: Spec, W, x: np.ndarray, lay: Layout, deep: list, capture=None,
                     layers=None, nq=EXACT) -> list:
    """The prefix rows x [L, hidden] (the views' merged rows at lay.starts, the text rows
    between) through the text model, causal, with the deepstack rows (deep[view][k]) added
    to the image rows after layers 0 .. len(deepstack) - 1: every layer's (K after k_norm
    and RoPE [L, n_kv, d], V [L, n_kv, d]). capture: "h<i>" the rows after layer i. layers:
    the first n only."""
    cap = capture if capture is not None else {}
    L, Hq, Hk, d = len(x), spec.n_q, spec.n_kv, spec.head_dim
    cos, sin = mrope_table(spec, lay.pos)
    mask = causal(L)
    kv = []
    n = spec.layers if layers is None else layers
    for i in range(n):
        p = f"{LM}layers.{i}."
        y = _rmsn(x, W[p + "input_layernorm.weight"], spec.eps)
        q = nq.lin(y, W, p + "self_attn.q_proj.weight", "p").reshape(L, Hq, d)
        k = nq.lin(y, W, p + "self_attn.k_proj.weight", "p").reshape(L, Hk, d)
        v = nq.lin(y, W, p + "self_attn.v_proj.weight", "p").reshape(L, Hk, d)
        q = _rope(_rmsn(q, W[p + "self_attn.q_norm.weight"], spec.eps), cos, sin)
        k = _rope(_rmsn(k, W[p + "self_attn.k_norm.weight"], spec.eps), cos, sin)
        kv.append((k.astype(np.float32), v.astype(np.float32)))
        if i == n - 1:
            break                               # the last layer's rest is dead
        o = nq.attn(q, k, v, d ** -0.5, mask, kind="p").reshape(L, Hq * d)
        x = x + nq.lin(o, W, p + "self_attn.o_proj.weight", "p")
        y = _rmsn(x, W[p + "post_attention_layernorm.weight"], spec.eps)
        x = (x + _mlp(W, p, y, nq, "p")).astype(np.float32)
        if i < len(spec.deepstack) and deep:
            x = x.copy()
            for vw, s0 in enumerate(lay.starts):
                x[s0:s0 + spec.n_tok] += deep[vw][i]
        cap[f"h{i}"] = x
    return kv


def reference_step(spec: Spec, W, kv: list, x_t: np.ndarray, fixed: dict, mods: dict,
                   cos: np.ndarray, sin: np.ndarray, capture=None, layers=None, nq=EXACT,
                   offset: int | None = None) -> np.ndarray:
    """One flow-matching velocity v_t [chunk, action_dim] of x_t: the expert over the prefix
    KV (kv: reference_prefix's, all layers; expert layer i reads kv[offset + i]), the step's
    adaRMS modulations (modulations()), the suffix's cos / sin [rows, d / 2] (rope_tables'
    last rows). capture: "in", "h<i>", "out". layers: the first n (a cut model)."""
    cap = capture if capture is not None else {}
    off = spec.offset if offset is None else offset
    T, Hq, Hk, d = spec.rows, spec.n_q, spec.n_kv, spec.head_dim
    a = nq.lin(x_t, W, AEIO + "action_in_proj.weight", "io") + fixed["bias"]
    x = np.concatenate([fixed["state"][None, :], a]).astype(np.float32)
    cap["in"] = x
    for i in range(spec.e_layers if layers is None else layers):
        p = f"{AET}layers.{i}."
        gn, sh, g = mods[i, "in"]
        y = (_rms(x, spec.eps) * gn + sh).astype(np.float32)
        q = nq.lin(y, W, p + "self_attn.q_proj.weight", "e").reshape(T, Hq, d)
        k = nq.lin(y, W, p + "self_attn.k_proj.weight", "e").reshape(T, Hk, d)
        v = nq.lin(y, W, p + "self_attn.v_proj.weight", "e").reshape(T, Hk, d)
        q = _rope(_rmsn(q, W[p + "self_attn.q_norm.weight"], spec.eps), cos, sin)
        k = _rope(_rmsn(k, W[p + "self_attn.k_norm.weight"], spec.eps), cos, sin)
        K = np.concatenate([kv[off + i][0], k])
        V = np.concatenate([kv[off + i][1], v])
        o = nq.attn(q, K, V, d ** -0.5, kind="e").reshape(T, Hq * d)
        x = x + nq.lin(o, W, p + "self_attn.o_proj.weight", "e") * g
        gn, sh, g = mods[i, "post"]
        y = (_rms(x, spec.eps) * gn + sh).astype(np.float32)
        x = (x + _mlp(W, p, y, nq, "e") * g).astype(np.float32)
        cap[f"h{i}"] = x
    gn, sh = mods["final"]
    y = (_rms(x, spec.eps) * gn + sh).astype(np.float32)
    cap["out"] = y
    return (nq.lin(y[-spec.chunk:], W, AEIO + "action_out_proj.weight", "io")
            + W[AEIO + "action_out_proj.bias"]).astype(np.float32)


def reference_actions(spec: Spec, W, images: list, lay: Layout, state, noise: np.ndarray,
                      state_mask=None, action_mask=None, capture=None, cut: dict | None = None,
                      nq=EXACT, enq=None, vis=None) -> np.ndarray:
    """The whole policy in fp32 numpy: the views' pixels ([3, image, image] each, in [-1,
    1]), the prompt's layout (layout()), the state [state_dim], noise [chunk, action_dim]
    (before the mask) -> x_t after the last step [chunk, action_dim] (predict_action's
    output, normalized slots). capture: "vis<v>", "deep<v>", "prefix", "kv", "x<s>", "v<s>".
    cut: {"vision": n, "prefix": n, "expert": n} layers (a cut model: the expert's layers
    read kv[offset + i], so a cut prefix needs offset + n layers; cut["offset"] overrides).
    nq: the tower's and prefix's numerics, enq the expert's; vis: [(rows, deep)] per view
    if known (skips the tower)."""
    cap = capture if capture is not None else {}
    cut = cut or {}
    if vis is None:
        tb = (pos_table(spec, W), *vit_rope_table(spec))
        vis = [reference_vision(spec, W, im, layers=cut.get("vision"), nq=nq, tables=tb)
               for im in images]
    for v, (r, dp) in enumerate(vis):
        cap[f"vis{v}"], cap[f"deep{v}"] = r, dp
    x = np.zeros((lay.L, spec.hidden), np.float32)
    x[[i for i, _ in lay.text()]] = text_rows(spec, W, lay)
    for v, s0 in enumerate(lay.starts):
        x[s0:s0 + spec.n_tok] = vis[v][0]
    cap["prefix"] = x
    kv = reference_prefix(spec, W, x, lay, [d for _, d in vis], layers=cut.get("prefix"), nq=nq)
    cap["kv"] = kv
    off = cut.get("offset", spec.offset)
    n = cut.get("expert")
    ne = spec.e_layers if n is None else n
    if off + ne > len(kv):
        raise ValueError(f"the expert's {ne} layers at offset {off} need {off + ne} prefix layers")
    enq = nq if enq is None else enq
    fixed = suffix_fixed(spec, W, state, state_mask, action_mask)
    cos, sin = rope_tables(spec, lay)
    cos, sin = cos[lay.L:], sin[lay.L:]
    xt = (np.asarray(noise, np.float32) * fixed["mask"]).astype(np.float32)
    dt = np.float32(-1.0 / spec.steps)
    for s, t in enumerate(times(spec)):
        cap[f"x{s}"] = xt
        v = reference_step(spec, W, kv, xt, fixed, modulations(spec, W, time_embedding(spec, t)),
                           cos, sin, layers=ne, nq=enq, offset=off)
        cap[f"v{s}"] = v
        xt = ((xt + dt * v) * fixed["mask"]).astype(np.float32)
    return xt


def _up(n: int, q: int) -> int:
    return -(-n // q) * q


# =============================================================================== tiny models
TINY = Spec(v_hidden=128, v_heads=2, v_ffn=256, v_layers=2, image=64, deepstack=(0,),
            hidden=256, layers=4, ffn=512, n_q=2, n_kv=1, e_hidden=128, e_layers=2, e_ffn=256,
            offset=2, chunk=10, steps=3)


def tiny_config(spec: Spec, base: dict) -> dict:
    """base (the assets' config.json) with spec's sizes: the HF config of a tiny model."""
    c = json.loads(json.dumps(base))
    v, t = c["vision_config"], c["text_config"]
    v.update(depth=spec.v_layers, hidden_size=spec.v_hidden, intermediate_size=spec.v_ffn,
             num_heads=spec.v_heads, deepstack_visual_indexes=list(spec.deepstack),
             out_hidden_size=spec.hidden, patch_size=spec.patch, spatial_merge_size=spec.merge,
             num_position_embeddings=spec.v_pos_grid ** 2)
    t.update(hidden_size=spec.hidden, intermediate_size=spec.ffn, num_hidden_layers=spec.layers,
             num_attention_heads=spec.n_q, num_key_value_heads=spec.n_kv, head_dim=spec.head_dim,
             rms_norm_eps=spec.eps, rope_theta=spec.theta)
    t["rope_scaling"]["mrope_section"] = list(spec.mrope)
    return c


def random_weights(spec: Spec, rng, vocab: int = 151936) -> dict:
    """Random tensors under the checkpoint's names, scaled like a trained model's (unit-ish
    activations through every layer): the tests' model, and tools/rhinovla_reference.py's
    when it builds the repo's PyTorch model at a tiny Spec."""
    W = {}

    def lin(name, n, k, bias=True, scale=1.0):
        W[name + ".weight"] = (scale * rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float32)
        if bias:
            W[name + ".bias"] = (0.1 * rng.standard_normal(n)).astype(np.float32)

    def vec(name, n, mu=1.0, sd=0.1):
        W[name] = (mu + sd * rng.standard_normal(n)).astype(np.float32)
    C, F, d = spec.v_hidden, spec.v_ffn, spec.head_dim
    for i in range(spec.v_layers):
        p = f"{VIS}blocks.{i}."
        vec(p + "norm1.weight", C)
        vec(p + "norm1.bias", C, 0.0)
        vec(p + "norm2.weight", C)
        vec(p + "norm2.bias", C, 0.0)
        lin(p + "attn.qkv", 3 * C, C)
        lin(p + "attn.proj", C, C, scale=0.5)
        lin(p + "mlp.linear_fc1", F, C)
        lin(p + "mlp.linear_fc2", C, F, scale=0.5)
    W[VIS + "patch_embed.proj.weight"] = (rng.standard_normal((C, 3, 2, spec.patch, spec.patch))
                                          / np.sqrt(6 * spec.patch ** 2)).astype(np.float32)
    vec(VIS + "patch_embed.proj.bias", C, 0.0)
    W[VIS + "pos_embed.weight"] = (0.5 * rng.standard_normal((spec.v_pos_grid ** 2, C))
                                   ).astype(np.float32)
    for p in [VIS + "merger."] + [f"{VIS}deepstack_merger_list.{k}." for k in
                                  range(len(spec.deepstack))]:
        post = "deepstack" in p
        vec(p + "norm.weight", 4 * C if post else C)
        vec(p + "norm.bias", 4 * C if post else C, 0.0)
        lin(p + "linear_fc1", 4 * C, 4 * C)
        lin(p + "linear_fc2", spec.hidden, 4 * C, scale=0.5)
    H = spec.hidden
    for i in range(spec.layers):
        p = f"{LM}layers.{i}."
        lin(p + "self_attn.q_proj", spec.e_q, H, False)
        lin(p + "self_attn.k_proj", spec.e_kv, H, False)
        lin(p + "self_attn.v_proj", spec.e_kv, H, False)
        lin(p + "self_attn.o_proj", H, spec.e_q, False, 0.5)
        vec(p + "self_attn.q_norm.weight", d)
        vec(p + "self_attn.k_norm.weight", d)
        lin(p + "mlp.gate_proj", spec.ffn, H, False)
        lin(p + "mlp.up_proj", spec.ffn, H, False)
        lin(p + "mlp.down_proj", H, spec.ffn, False, 0.5)
        vec(p + "input_layernorm.weight", H)
        vec(p + "post_attention_layernorm.weight", H)
    vec(LM + "norm.weight", H)
    W[LM + "embed_tokens.weight"] = rng.standard_normal((vocab, H)).astype(np.float32)
    E = spec.e_hidden
    for i in range(spec.e_layers):
        p = f"{AET}layers.{i}."
        lin(p + "self_attn.q_proj", spec.e_q, E, False)
        lin(p + "self_attn.k_proj", spec.e_kv, E, False)
        lin(p + "self_attn.v_proj", spec.e_kv, E, False)
        lin(p + "self_attn.o_proj", E, spec.e_q, False, 0.5)
        vec(p + "self_attn.q_norm.weight", d)
        vec(p + "self_attn.k_norm.weight", d)
        lin(p + "mlp.gate_proj", spec.e_ffn, E, False)
        lin(p + "mlp.up_proj", spec.e_ffn, E, False)
        lin(p + "mlp.down_proj", E, spec.e_ffn, False, 0.5)
        for n in ("input_layernorm", "post_attention_layernorm"):
            vec(p + n + ".norm.weight", E)
            lin(p + n + ".cond", 3 * E, E, scale=0.3)
    vec(AET + "norm.norm.weight", E)
    lin(AET + "norm.cond", 3 * E, E, scale=0.3)
    lin(AEIO + "action_in_proj", E, spec.action_dim)
    lin(AEIO + "action_out_proj", spec.action_dim, E)
    lin(AEIO + "state_proj", E, spec.state_dim)
    if spec.mask_condition:
        lin(AE + "state_mask_proj", E, spec.state_dim, scale=0.3)
        lin(AE + "action_mask_proj", E, spec.action_dim, scale=0.3)
    return W
