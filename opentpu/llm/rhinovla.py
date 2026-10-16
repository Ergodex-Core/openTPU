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
    restricted to kinds ("v", "p", "e") and a_only to names' substrings (a tuple: all of
    them in the name; one matrix group at W8A8 with the rest at W8A16: the precision map's
    sweep, tools/rhinovla_sensitivity.py). Not bit exact (float64 sums, the
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
        qa = "a" in self.parts and (self.a_only is None or any(
            all(t in name for t in (o if isinstance(o, tuple) else (o,))) for o in self.a_only))
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


# =============================================================================== on the card
class Host:
    """What the host computes for one observation, fp32 (docs/rhinovla.md 3): the prompt's
    text rows, the prefix's and suffix's RoPE tables, the causal mask table, the suffix's
    state row and action bias (suffix_fixed), action_out_proj's rows and bias times the
    action mask (so x_t += dt v_t keeps the dead slots at 0 with x_0 = noise x mask)."""

    def __init__(self, spec: Spec, W, lay: Layout, state, state_mask=None, action_mask=None):
        self.spec, self.lay = spec, lay
        self.text = text_rows(spec, W, lay)
        self.cos, self.sin = rope_tables(spec, lay)
        self.fixed = suffix_fixed(spec, W, state, state_mask, action_mask)
        m = self.fixed["mask"]
        self.woutf = (np.asarray(W[AEIO + "action_out_proj.weight"], np.float32) * m[:, None]
                      ).astype(np.float32)
        self.boutT = np.repeat((W[AEIO + "action_out_proj.bias"] * m).astype(np.float32),
                               spec.chunk)

    def mask_table(self, cap: int) -> np.ndarray:
        """[L, cap + 1]: row r is query row r's additive mask over the keys: -0 where key <=
        r, -inf past it (kernels/attention.Additive: the scores start as this tile)."""
        L = self.lay.L
        t = np.full((L, cap + 1), -np.inf, np.float32)
        for r in range(L):
            t[r, :r + 1] = -0.0
        return t


class _Block:
    """One layer's tensors at fixed offsets of a block; the layers' blocks `stride` bytes
    apart from `first`, so a layer index (an int, a run-time value, a hardware loop's
    counter) addresses them all (base = first + li * stride). (pi05's.)"""

    def __init__(self, D: int):
        self.D, self.next, self.vecs, self.mats = D, 0, {}, {}

    def _alloc(self, nbytes: int) -> int:
        a = self.next
        self.next = _up(a + nbytes, 128)
        return a

    def vec(self, name: str, n: int) -> None:
        self.vecs[name] = (self._alloc(4 * n), n)

    def mat(self, name: str, n: int, k: int, fmt: str = "int8") -> None:
        from .. import quant as Q
        d = self._alloc(n * Q.row_bytes(k, fmt, self.D))
        s = self._alloc(n * Q.scale_bytes(k, fmt, self.D))
        self.mats[name] = (d, s, n, k, fmt)

    def place(self, first: int, count: int) -> int:
        self.first, self.count = first, count
        self.stride = _up(self.next, 4096)
        return first + count * self.stride

    def at(self, li):
        from types import SimpleNamespace
        from ..compiler import Affine, QTensor, Tensor
        base = Affine(self.first) + Affine.of(li) * self.stride
        ns = {k: Tensor(base + o, (n,), (1,)) for k, (o, n) in self.vecs.items()}
        for k, (d, s, n, kk, fmt) in self.mats.items():
            ns[k] = QTensor.weight(base + d, base + s, n, kk, fmt, self.D)
        return SimpleNamespace(**ns)

    def put_mat(self, put, li: int, name: str, w: np.ndarray) -> None:
        from .. import quant as Q
        d, s, n, k, fmt = self.mats[name]
        if w.shape != (n, k):
            raise ValueError(f"{name}: {w.shape} for [{n}, {k}]")
        q, sc = Q.quantize_mxu(np.ascontiguousarray(w, np.float32), fmt, self.D)
        base = self.first + li * self.stride
        put(base + d, q)
        put(base + s, sc)

    def put_vec(self, put, li: int, name: str, v: np.ndarray) -> None:
        o, n = self.vecs[name]
        a = np.zeros(n, np.float32)
        v = np.asarray(v, np.float32).reshape(-1)
        a[:len(v)] = v
        put(self.first + li * self.stride + o, a)


def _interleave(wg: np.ndarray, wu: np.ndarray, chunk: int) -> np.ndarray:
    """[2F, H]: chunk c's gate rows, then its up rows (kernels/pi05.geglu's one MM a chunk)."""
    F, H = wg.shape
    return np.stack([wg.reshape(F // chunk, chunk, H), wu.reshape(F // chunk, chunk, H)],
                    1).reshape(2 * F, H)


def _pad(a, shape):
    out = np.zeros(shape, np.float32)
    out[tuple(slice(0, k) for k in a.shape)] = a
    return out


class Image:
    """RhinoVLA in one slice's DRAM (docs/rhinovla.md 3): the weights in an MXU format (int8
    by default), each part's layers in equal blocks (_Block), the vectors fp32; the work areas
    of one action chunk with a prefix of up to `np_max` rows.

    Vision (kernels/vit.py and kernels/qwen3vl_vit.py): per block wq, wk, wv (the fused qkv
    split), wo [C, C], w1 [F, C], w2 [C, F], the LayerNorms' gains, their biases folded into
    the next linear's (float64); the patch matrix [C, 768] (the Conv3d's temporal pair
    summed), its bias, the interpolated position rows, the 2D RoPE tables; the main merger
    (its per-token LayerNorm as the last block's `post`, w1 [4C, 4C], w2 [hidden, 4C], the
    biases) and the deepstack mergers (their LayerNorm over the 4C, w1, w2, biases).
    Prefix (Qwen3-VL text): per layer both norm weights, q_norm / k_norm, wq [2048, 2048],
    wk, wv [1024, 2048], wo, wgu [2 F, 2048] (gate | up by chunks), wd [2048, F]. Expert: the
    same shapes at 1024 / 3072 with q_norm / k_norm; action_in_proj^T by rows; the adaRMS
    table: per step and layer gain, shift, tanh(gate) of both norms, per step the final
    norm's gain and shift (host fp32, build_mods).

    Work: a view's patch rows P and residual X, Q / O scratch and KV scratch; the prefix's
    rows PX [np_max, 2048], Q / O scratch, the RoPE tables of every position (the host's: they
    follow the prompt), the mask table, every layer's KV cache (np_max + rows tokens: the
    expert appends its rows after the prefix's), the deepstack rows DS; the expert's rows EX
    [rows, 1024], Q / O scratch, x_t^T [96, chunk], v_t^T, the state row, the action bias,
    the masked action_out_proj rows and bias."""

    VCHUNK = 256        # the tower's MLP chunk
    MCHUNK = 256        # the mergers' MLP chunk
    PCHUNK = 256        # the text model's SwiGLU chunk
    ECHUNK = 256        # the expert's

    def __init__(self, spec: Spec, cfg, np_max: int = 256, fmt: str = "int8"):
        from ..compiler import k_scale_bytes
        from .qwen3 import _Bump
        self.spec, self.cfg, self.fmt = spec, cfg, fmt
        D = self.D = cfg.D
        s = spec
        C, F = s.v_hidden, _up(s.v_ffn, self.VCHUNK)
        N, H, d = s.n_patch, s.hidden, s.head_dim
        self.vF, self.KP = F, _up(3 * s.patch ** 2, D)
        self.vdp = _up(s.v_head_dim, D)
        self.pchunk = min(self.PCHUNK, s.ffn)
        self.echunk = min(self.ECHUNK, s.e_ffn)
        self.mchunk = min(self.MCHUNK, 4 * C)
        self.part = next(p for p in (512, 256, 128) if (4 * C) % p == 0)
        b = _Bump(4096)
        # ---- vision
        vb = _Block(D)
        for n in ("ln1w", "ln2w", "bq", "bk", "bv", "bo", "b2"):
            vb.vec(n, C)
        vb.vec("b1", F)
        for n in ("wq", "wk", "wv", "wo"):
            vb.mat(n, C, C, fmt)
        vb.mat("w1", F, C, fmt)
        vb.mat("w2", C, F, fmt)
        b.next = vb.place(_up(b.next, 4096), s.v_layers)
        self.vb = vb
        gb = _Block(D)                  # the tower's globals
        gb.vec("bp", C)
        gb.vec("post_w", C)
        gb.vec("post_b", C)
        gb.vec("E", N * C)
        gb.vec("rc", N * (s.v_head_dim // 2))
        gb.vec("rs", N * (s.v_head_dim // 2))
        gb.mat("wp", C, self.KP, fmt)
        b.next = gb.place(_up(b.next, 4096), 1)
        self.gb = gb
        mb = _Block(D)                  # the mergers: 0 the main one, 1 .. the deepstack's
        mb.vec("pre_w", 4 * C)
        mb.vec("pre_b", 4 * C)
        mb.vec("b1", 4 * C)
        mb.vec("b2", H)
        mb.mat("w1", 4 * C, 4 * C, fmt)
        mb.mat("w2", H, 4 * C, fmt)
        b.next = mb.place(_up(b.next, 4096), 1 + len(s.deepstack))
        self.mb = mb
        # ---- prefix
        pb = _Block(D)
        pb.vec("g1", H)
        pb.vec("g2", H)
        pb.vec("qn", d)
        pb.vec("kn", d)
        pb.mat("wq", s.e_q, H, fmt)
        pb.mat("wk", s.e_kv, H, fmt)
        pb.mat("wv", s.e_kv, H, fmt)
        pb.mat("wo", H, s.e_q, fmt)
        pb.mat("wgu", 2 * s.ffn, H, fmt)
        pb.mat("wd", H, s.ffn, fmt)
        b.next = pb.place(_up(b.next, 4096), s.layers)
        self.pb = pb
        # ---- expert
        E, Fe = s.e_hidden, s.e_ffn
        eb = _Block(D)
        eb.vec("qn", d)
        eb.vec("kn", d)
        eb.mat("wq", s.e_q, E, fmt)
        eb.mat("wk", s.e_kv, E, fmt)
        eb.mat("wv", s.e_kv, E, fmt)
        eb.mat("wo", E, s.e_q, fmt)
        eb.mat("wgu", 2 * Fe, E, fmt)
        eb.mat("wd", E, Fe, fmt)
        b.next = eb.place(_up(b.next, 4096), s.e_layers)
        self.eb = eb
        xb = _Block(D)                  # the expert's globals
        xb.vec("winT", s.action_dim * E)
        b.next = xb.place(_up(b.next, 4096), 1)
        self.xb = xb
        ab = _Block(D)                  # the adaRMS table: a block per (step, layer)
        for n in ("g_in", "s_in", "a_in", "g_post", "s_post", "a_post"):
            ab.vec(n, E)
        b.next = ab.place(_up(b.next, 4096), s.steps * s.e_layers)
        self.ab = ab
        fb = _Block(D)                  # the final norm's, a block per step
        fb.vec("g", E)
        fb.vec("s", E)
        b.next = fb.place(_up(b.next, 4096), s.steps)
        self.fb = fb
        self.weights_end = b.next
        # ---- work
        self.np_max = np_max
        T = s.rows
        self.cap = cap = _up(np_max + T, D)
        nds = len(s.deepstack)
        A = s.action_dim
        self.work = dict(P=b.alloc(4 * N * self.KP), X=b.alloc(4 * N * C),
                         Q=b.alloc(4 * N * s.v_heads * self.vdp), O=b.alloc(4 * N * C),
                         PX=b.alloc(4 * np_max * H), PQ=b.alloc(4 * cap * s.n_q * d),
                         PO=b.alloc(4 * cap * s.e_q), cos=b.alloc(4 * cap * d // 2),
                         sin=b.alloc(4 * cap * d // 2),
                         DS=b.alloc(4 * max(1, nds) * s.views * s.n_tok * H),
                         MASK=b.alloc(4 * np_max * (np_max + 1)),
                         EX=b.alloc(4 * T * E), EQ=b.alloc(4 * T * s.n_q * d),
                         EO=b.alloc(4 * T * s.e_q), XTt=b.alloc(4 * A * s.chunk),
                         Vt=b.alloc(4 * A * s.chunk), srow=b.alloc(4 * E), bin=b.alloc(4 * E),
                         woutf=b.alloc(4 * A * E), boutT=b.alloc(4 * A * s.chunk))
        nv = _up(N, D)
        self.vkv = {h: {"k": b.alloc(nv * self.vdp), "ks": b.alloc(4 * nv),
                        "vt": b.alloc(self.vdp * nv + D), "vs": b.alloc(4 * nv)}
                    for h in range(s.v_heads)}
        self.nv = nv
        kb = _Block(D)                  # a layer's KV cache: one block per layer, every head
        for h in range(s.n_kv):
            kb.vec(f"k{h}", cap * d // 4)
            kb.vec(f"ks{h}", cap * k_scale_bytes(d, D) // 4 + 1)
            kb.vec(f"vt{h}", (d * cap + D) // 4)
            kb.vec(f"vs{h}", cap)
        b.next = kb.place(_up(b.next, 4096), s.layers)
        self.kb = kb
        self.end = _up(b.next, 1 << 20)

    # ---- descriptors
    def _t(self, name, shape, off: int = 0):
        from .qwen3 import _tdesc
        return _tdesc(self.work[name] + off, shape)

    def vision_desc(self):
        """The vit phases' `v`."""
        from types import SimpleNamespace
        from ..compiler import Tensor
        from ..kernels.vit import VitKV
        s, N, C = self.spec, self.spec.n_patch, self.spec.v_hidden
        g = self.gb.at(0)
        hd = s.v_head_dim
        return SimpleNamespace(
            C=C, heads=s.v_heads, dp=self.vdp, dv=hd,
            P=self._t("P", (N, self.KP)), E=Tensor(g.E.base, (N, C), (C, 1)),
            rc=Tensor(g.rc.base, (N, hd // 2), (hd // 2, 1)),
            rs=Tensor(g.rs.base, (N, hd // 2), (hd // 2, 1)),
            X=self._t("X", (N, C)),
            Q=[self._t("Q", (N, self.vdp), 4 * h * N * self.vdp) for h in range(s.v_heads)],
            O=self._t("O", (N, C)),
            kv=VitKV(self.vkv, self.nv, self.vdp, self.D, 1, 0, dv=hd),
            emb=SimpleNamespace(wp=g.wp, bp=g.bp), post=(g.post_w, g.post_b))

    def kv(self, li, off: int = 0):
        """Layer off + li's KV cache (every KV head: K token-major, V^T rows cap wide)."""
        from ..compiler import Affine
        from ..kernels.vit import VitKV
        kb = self.kb
        base = Affine(kb.first + off * kb.stride) + Affine.of(li) * kb.stride
        heads = {h: {n: base + kb.vecs[f"{n}{h}"][0] for n in ("k", "ks", "vt", "vs")}
                 for h in range(self.spec.n_kv)}
        return VitKV(heads, self.cap, self.spec.head_dim, self.D, 1, 0)

    def text_desc(self, rows: str, n: int, li):
        """The prefix's (rows "P") or the expert's ("E") `g`: X, Q, O, RoPE tables, KV."""
        from types import SimpleNamespace
        s, d, cap = self.spec, self.spec.head_dim, self.cap
        X, Q, O = ("PX", "PQ", "PO") if rows == "P" else ("EX", "EQ", "EO")
        H = s.hidden if rows == "P" else s.e_hidden
        qrows = cap if rows == "P" else s.rows
        return SimpleNamespace(
            H=s.n_q, Hk=s.n_kv, d=d, X=self._t(X, (n, H)),
            Q=[self._t(Q, (qrows, d), 4 * h * qrows * d) for h in range(s.n_q)],
            O=self._t(O, (qrows, s.e_q)), cos=self._t("cos", (cap, d // 2)),
            sin=self._t("sin", (cap, d // 2)), kv=self.kv(li))

    def expert_desc(self):
        from types import SimpleNamespace
        from ..compiler import Tensor
        s = self.spec
        A, E = s.action_dim, s.e_hidden
        x = self.xb.at(0)
        e = SimpleNamespace(
            X=self._t("EX", (s.rows, E)), O=self._t("EO", (s.rows, s.e_q)),
            XTt=self._t("XTt", (A, s.chunk)), Vt=self._t("Vt", (A, s.chunk)),
            winT=Tensor(x.winT.base, (A, E), (E, 1)),
            woutf=self._t("woutf", (A, E)), boutT=self._t("boutT", (A, s.chunk)),
            srow=self._t("srow", (1, E)), bin=self._t("bin", (E,)),
            layer=self.eb.at, kv=lambda li: self.kv(li, s.offset),
            mods=lambda step, li: self.ab.at(step * s.e_layers + li), final=self.fb.at)
        e.g = self.text_desc("E", s.rows, 0)
        return e

    # ---- contents
    def build(self, W, put, log=None) -> None:
        """Write the weights (W: Weights, or a dict of the same names) with put(addr, array);
        the adaRMS table is the host's (build_mods). The prompt-dependent tables (text rows,
        RoPE, the mask) are run_chunk's."""
        s, C, H = self.spec, self.spec.v_hidden, self.spec.hidden
        f64 = np.float64

        def folded(w, bias, ln_b):
            return bias + np.asarray(w, f64) @ np.asarray(ln_b, f64)
        vb, F = self.vb, self.vF
        for li in range(s.v_layers):
            p = f"{VIS}blocks.{li}."
            qkv, bqkv = W[p + "attn.qkv.weight"], W[p + "attn.qkv.bias"]
            ln1b = W[p + "norm1.bias"]
            for i, n in enumerate("qkv"):
                vb.put_mat(put, li, "w" + n, qkv[i * C:(i + 1) * C])
                vb.put_vec(put, li, "b" + n, folded(qkv[i * C:(i + 1) * C], bqkv[i * C:(i + 1) * C],
                                                    ln1b))
            vb.put_mat(put, li, "wo", W[p + "attn.proj.weight"])
            vb.put_vec(put, li, "bo", W[p + "attn.proj.bias"])
            vb.put_mat(put, li, "w1", _pad(W[p + "mlp.linear_fc1.weight"], (F, C)))
            vb.put_vec(put, li, "b1", folded(W[p + "mlp.linear_fc1.weight"],
                                             W[p + "mlp.linear_fc1.bias"], W[p + "norm2.bias"]))
            vb.put_mat(put, li, "w2", _pad(W[p + "mlp.linear_fc2.weight"], (C, F)))
            vb.put_vec(put, li, "b2", W[p + "mlp.linear_fc2.bias"])
            vb.put_vec(put, li, "ln1w", W[p + "norm1.weight"])
            vb.put_vec(put, li, "ln2w", W[p + "norm2.weight"])
            if log:
                log(f"vision block {li}")
        gb = self.gb
        wp, bp = patch_matrix(s, W)
        gb.put_mat(put, 0, "wp", _pad(wp, (C, self.KP)))
        gb.put_vec(put, 0, "bp", bp)
        gb.put_vec(put, 0, "E", pos_table(s, W))
        rc, rs = vit_rope_table(s)
        gb.put_vec(put, 0, "rc", rc)
        gb.put_vec(put, 0, "rs", rs)
        m = VIS + "merger."
        gb.put_vec(put, 0, "post_w", W[m + "norm.weight"])
        gb.put_vec(put, 0, "post_b", W[m + "norm.bias"])
        mb = self.mb
        for k, p in enumerate([m] + [f"{VIS}deepstack_merger_list.{i}." for i in
                                     range(len(s.deepstack))]):
            mb.put_vec(put, k, "pre_w", np.ones(4 * C, np.float32) if k == 0 else W[p + "norm.weight"])
            mb.put_vec(put, k, "pre_b", np.zeros(4 * C, np.float32) if k == 0 else W[p + "norm.bias"])
            mb.put_mat(put, k, "w1", W[p + "linear_fc1.weight"])
            mb.put_vec(put, k, "b1", W[p + "linear_fc1.bias"])
            mb.put_mat(put, k, "w2", W[p + "linear_fc2.weight"])
            mb.put_vec(put, k, "b2", W[p + "linear_fc2.bias"])
        for blk, pre, n_l, ch in ((self.pb, LM, s.layers, self.pchunk),
                                  (self.eb, AET, s.e_layers, self.echunk)):
            for li in range(n_l):
                p = f"{pre}layers.{li}."
                for n in "qkvo":
                    blk.put_mat(put, li, "w" + n, W[p + f"self_attn.{n}_proj.weight"])
                blk.put_vec(put, li, "qn", W[p + "self_attn.q_norm.weight"])
                blk.put_vec(put, li, "kn", W[p + "self_attn.k_norm.weight"])
                blk.put_mat(put, li, "wgu", _interleave(W[p + "mlp.gate_proj.weight"],
                                                        W[p + "mlp.up_proj.weight"], ch))
                blk.put_mat(put, li, "wd", W[p + "mlp.down_proj.weight"])
                if pre == LM:
                    blk.put_vec(put, li, "g1", W[p + "input_layernorm.weight"])
                    blk.put_vec(put, li, "g2", W[p + "post_attention_layernorm.weight"])
                if log:
                    log(f"{'prefix' if pre == LM else 'expert'} layer {li}")
        self.xb.put_vec(put, 0, "winT", np.asarray(W[AEIO + "action_in_proj.weight"]).T.reshape(-1))
        self.build_mods(W, put)

    def build_mods(self, W, put) -> None:
        """The adaRMS table of every step: t's embedding and each norm's (gain, shift,
        tanh(gate)), host fp32 (time_embedding, modulations: the repo's arithmetic)."""
        s = self.spec
        for st, t in enumerate(times(s)):
            m = modulations(s, W, time_embedding(s, t))
            for li in range(s.e_layers):
                k = st * s.e_layers + li
                for which in ("in", "post"):
                    gn, sh, g = m[li, which]
                    self.ab.put_vec(put, k, "g_" + which, gn)
                    self.ab.put_vec(put, k, "s_" + which, sh)
                    self.ab.put_vec(put, k, "a_" + which, g)
            gn, sh = m["final"]
            self.fb.put_vec(put, st, "g", gn)
            self.fb.put_vec(put, st, "s", sh)

    # ---- the programs
    def programs(self, lay: Layout, R: int = 16, RP: int = 8, RE: int = 8, RM: int = 16,
                 block: int = 256) -> dict:
        """The phase programs of one chunk with the prompt's layout: {name: (program,
        run_args)}. Vision (per view; the block a run argument "layer", a merge's destination
        row "row"), the prefix's (the layer a run argument), the expert step (the Euler step a
        run argument "step"). R: the tower's row group, RP the prefix's, RE the expert's, RM
        the mergers' (windows)."""
        from ..compiler import Affine, RunVar, Tensor
        from ..kernels import qwen3vl_vit as QV
        from ..kernels import rhinovla as K
        from ..kernels import vit as VK
        s, cfg = self.spec, self.cfg
        Np = lay.L
        if Np > self.np_max:
            raise ValueError(f"{Np} prefix rows: the image holds {self.np_max}")
        N, H, M = s.n_patch, s.hidden, s.n_tok
        v = self.vision_desc()
        vlw = self.vb.at(RunVar("layer", bound=s.v_layers))
        vsc = 1.0 / math.log(2.0) / math.sqrt(s.v_head_dim)
        gsc = 1.0 / math.log(2.0) / math.sqrt(s.head_dim)

        def tr(kernel, **kw):
            b = kernel.trace(cfg, 0, kw)
            return b.finish(), list(b.run_args)
        row = RunVar("row", bound=max(self.np_max, len(s.deepstack) * s.views * M + 1))
        pxrow = Tensor(Affine(self.work["PX"]) + row * (4 * H), (M, H), (H, 1))
        dsrow = Tensor(Affine(self.work["DS"]) + row * (4 * H), (M, H), (H, 1))
        mk = RunVar("k", bound=len(s.deepstack) + 1)
        P = {"v_embed": tr(VK.vit_embed, v=v, N=N, R=R),
             "v_qkv": tr(QV.vit_qkv_rope, v=v, lw=vlw, src=v.X, N=N, R=R, eps=s.v_eps),
             "v_attend": tr(VK.vit_attend, v=v, N=N, scale=vsc, block=block),
             "v_out": tr(VK.vit_out_mlp, v=v, lw=vlw, dst=v.X, N=N, R=R, eps=s.v_eps,
                         chunk=self.VCHUNK),
             "v_last": tr(VK.vit_out_mlp, v=v, lw=vlw, dst=v.X, N=N, R=R, eps=s.v_eps,
                          chunk=self.VCHUNK, post=v.post),
             "v_merge": tr(VK.vit_merge, v=v, mw=self.mb.at(0), src=v.X, dst=pxrow, N4=M, R=RM,
                           eps=1e-6, chunk=self.mchunk, mean=False, part=self.part,
                           act=QV.gelu_erf, norm=False)}
        if s.deepstack:
            P["v_deep"] = tr(VK.vit_merge, v=v, mw=self.mb.at(mk), src=v.X, dst=dsrow, N4=M,
                             R=RM, eps=1e-6, chunk=self.mchunk, mean=False, part=self.part,
                             act=QV.gelu_erf, norm=True)
        li = RunVar("layer", bound=s.layers)
        pg = self.text_desc("P", Np, li)
        plw = self.pb.at(li)
        P["p_qkv"] = tr(K.prefix_qkv, g=pg, lw=plw, N=Np, R=RP, eps=s.eps)
        P["p_attend"] = tr(K.prefix_attend, g=pg, N=Np, scale=gsc, block=block,
                           mask=(self.work["MASK"], self.np_max + 1))
        P["p_out"] = tr(K.prefix_out, g=pg, lw=plw, N=Np, R=RP, eps=s.eps, chunk=self.pchunk)
        if s.deepstack:
            PX = Tensor(Affine(self.work["PX"]), (self.np_max, H), (H, 1))
            ds = Tensor(Affine(self.work["DS"]) + Affine.of(li) * (4 * s.views * M * H),
                        (s.views * M, H), (H, 1))
            P["p_deep"] = tr(QV.add_rows, dsts=[PX[s0:s0 + M, :] for s0 in lay.starts],
                             srcs=[ds[vw * M:(vw + 1) * M, :] for vw in range(s.views)],
                             N=M, R=RP)
        P["e_step"] = tr(K.expert_step, e=self.expert_desc(),
                         step=RunVar("step", bound=s.steps), layers=s.e_layers, Np=Np, R=RE,
                         eps=s.eps, scale=gsc, block=block, chunk=self.echunk,
                         dt=-1.0 / s.steps)
        return P

    def runs(self, lay: Layout) -> list:
        """The runs of one chunk in order: [(phase, {run argument: value})]; the host's
        writes go between ("write_patches", view), ("write_prompt",), ("write_suffix",)."""
        s = self.spec
        M = s.n_tok
        out = []
        for vw in range(s.views):
            out += [("write_patches", vw), ("v_embed", {})]
            for i in range(s.v_layers):
                last = i == s.v_layers - 1
                out += [("v_qkv", {"layer": i}), ("v_attend", {}),
                        ("v_last" if last else "v_out", {"layer": i})]
                if i in s.deepstack:
                    k = s.deepstack.index(i)
                    out.append(("v_deep", {"k": k + 1, "row": (k * s.views + vw) * M}))
            out.append(("v_merge", {"row": lay.starts[vw]}))
        out.append(("write_prompt",))
        for i in range(s.layers):
            out.append(("p_qkv", {"layer": i}))
            if i < s.layers - 1:
                out += [("p_attend", {"layer": i}), ("p_out", {"layer": i})]
                if i < len(s.deepstack):
                    out.append(("p_deep", {"layer": i}))
        out.append(("write_suffix",))
        for st in range(s.steps):
            out.append(("e_step", {"step": st}))
        return out


def run_chunk(image: Image, backend, programs: dict, images: list, host: Host,
              noise: np.ndarray, log=None, stop: str | None = None, after=None) -> np.ndarray:
    """One action chunk on `backend` (qwen3.IsaBackend: write, run, read): the views' pixels
    ([3, image, image] in [-1, 1]), the observation's Host, noise [chunk, action_dim] -> x_t
    after the last step [chunk, action_dim]. stop: "vision" returns (None) after the views,
    "prefix" after the prefix. log(name, args) before each run, after(name, args) after it."""
    from ..compiler import arg_words
    s, w, lay = image.spec, image.work, host.lay
    H = s.hidden
    for item in image.runs(lay):
        name = item[0]
        if name == "write_patches":
            p = np.zeros((s.n_patch, image.KP), np.float32)
            p[:, :3 * s.patch ** 2] = patches(images[item[1]], s.patch, s.merge)
            backend.write(0, w["P"], p)
            continue
        if name == "write_prompt":
            if stop == "vision":
                return None
            rows = host.text
            for j, (r, _) in enumerate(lay.text()):
                backend.write(0, w["PX"] + 4 * r * H, rows[j])
            backend.write(0, w["cos"], host.cos)
            backend.write(0, w["sin"], host.sin)
            backend.write(0, w["MASK"], host.mask_table(image.np_max))
            continue
        if name == "write_suffix":
            if stop == "prefix":
                return None
            f = host.fixed
            backend.write(0, w["srow"], f["state"])
            backend.write(0, w["bin"], f["bias"])
            backend.write(0, w["woutf"], host.woutf)
            backend.write(0, w["boutT"], host.boutT)
            x0 = (np.asarray(noise, np.float32) * f["mask"]).astype(np.float32)
            backend.write(0, w["XTt"], np.ascontiguousarray(x0.T))
            continue
        prog, ra = programs[name]
        if log:
            log(name, item[1])
        backend.run([prog], args=arg_words(ra, item[1]))
        if after:
            after(name, item[1])
    return read_xt(image, backend)


def read_prefix_rows(image: Image, backend, lay: Layout) -> np.ndarray:
    """The views' image rows of the prefix [views x n_tok, hidden] (after the merge; after
    the prefix's runs they hold the residual stream's rows)."""
    s, w = image.spec, image.work
    out = []
    for s0 in lay.starts:
        a = backend.read(0, w["PX"] + 4 * s0 * s.hidden, 4 * s.n_tok * s.hidden)
        out.append(a.view(np.float32).reshape(s.n_tok, s.hidden))
    return np.concatenate(out)


def read_xt(image: Image, backend, which: str = "x") -> np.ndarray:
    """x_t (which "x") or the last step's v_t ("v") [chunk, action_dim] from the card."""
    s, w = image.spec, image.work
    a = backend.read(0, w["XTt" if which == "x" else "Vt"], 4 * s.chunk * s.action_dim)
    return a.view(np.float32).reshape(s.action_dim, s.chunk).T.copy()


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
