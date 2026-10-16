"""pi0.5 (Physical Intelligence's vision-language-action model, lerobot's `pi05`) on openTPU.

The checkpoint ported is lerobot/pi05_libero_finetuned_v044 (docs/pi05.md): a PaliGemma
(SigLIP so400m/14 at 224 plus Gemma 2B) that reads two camera images and a prompt holding the
task and the discretized robot state, and a Gemma 300M "action expert" that turns 50 x 32
noise into an action chunk by flow matching (10 Euler steps from t = 1 to t = 0), attending
to the prefix's KV cache. lerobot's modeling_pi05.py is the golden reference
(tools/pi05_reference.py saves its tensors); this module holds:

  Spec              the dimensions (config.json's variants: gemma_2b, gemma_300m)
  Weights           the checkpoint's tensors, read lazily (bf16 / fp32 -> fp32 numpy)
  host side         the prompt's prefix layout, patches, RoPE tables, the time embedding and
                    every expert layer's adaRMS modulation for each Euler step (all of it
                    depends on t alone: a fixed schedule, computed once in fp32 as lerobot)
  reference_*       a plain numpy fp32 forward of the whole policy (vision, prefix, expert
                    steps), the device's numerics target
  Image / programs  the card's layout and phase programs (opentpu/kernels/pi05.py), run on
                    the ISA simulator through qwen3.IsaBackend

The model, as lerobot runs it (checked against its code, not openpi's paper):
  * Each 224 x 224 image (pixels in [-1, 1]) -> SigLIP: 16 x 16 patches of 14 (a Conv2d,
    i.e. [256, 588] @ [588, 1152]), plus a learned position row each; 27 pre-LayerNorm
    blocks (16 heads of 72, MLP 4304, gelu_tanh, biases everywhere, eps 1e-6), post
    LayerNorm; then PaliGemma's projector, one linear 1152 -> 2048 with a bias. No scaling of
    the image rows.
  * The prompt "Task: <task>, State: <8 bins 0..255>;\\nAction: " (PaliGemma tokenizer, BOS
    first, padded to 200) is embedded by the tied table times sqrt(2048) (fp32 45.254834).
  * The prefix is [camera 0, camera 1, camera 2, tokens]; the third camera is empty (-1
    pixels, pad mask 0) and the token padding has pad mask 0. A pad never attends nor is
    attended, and positions are cumsum(pad) - 1, so dropping the pads is exact: the card
    keeps the Np = 512 + n_tok valid rows, at positions 0 .. Np - 1.
  * Prefix attention is bidirectional over the valid prefix (prefix-LM with every prefix
    token in the first block). Gemma 2B: 18 layers, hidden 2048, 8 query heads and 1 KV head
    of 256, scale 1/16, RoPE theta 10000 (rotate-half), RMSNorm x * (1 + w) eps 1e-6, GeGLU
    16384 (gelu_tanh(gate) * up). The KV of every layer is kept; nothing else of the prefix
    is (the last layer's attention output and MLP are dead).
  * One Euler step: x_t [50, 32] -> action_in_proj (32 -> 1024, bias) -> 18 expert layers
    (hidden 1024, 8 heads of 256 on 1 KV head, GeGLU 4096) whose norms are adaRMS:
    normed = x rsqrt(mean x^2 + eps) (1 + scale) + shift and the residuals gated, x += y gate,
    with (scale, shift, gate) = dense(cond) per norm; cond = silu(time_mlp_out(silu(
    time_mlp_in(sincos(t))))). The 50 action rows attend to the whole valid prefix (its
    cached K, V of that layer) and to each other (bidirectionally), at positions Np .. Np+49.
    A final adaRMS norm (no gate), action_out_proj (1024 -> 32, bias): v_t; x_t += -0.1 v_t.
  * The output's first 7 dims, unnormalized (MEAN_STD), are the actions.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PG = "model.paligemma_with_expert.paligemma.model."
TOWER = PG + "vision_tower.vision_model."
LLM = PG + "language_model."
EXP = "model.paligemma_with_expert.gemma_expert.model."
TOP = "model."


@dataclass(frozen=True)
class Spec:
    # SigLIP so400m/14
    v_hidden: int = 1152
    v_heads: int = 16
    v_ffn: int = 4304
    v_layers: int = 27
    patch: int = 14
    image: int = 224
    v_eps: float = 1e-6
    # Gemma 2B (PaliGemma's language model) and the Gemma 300M expert: one head layout
    hidden: int = 2048
    layers: int = 18
    ffn: int = 16384
    e_hidden: int = 1024
    e_layers: int = 18
    e_ffn: int = 4096
    n_q: int = 8
    head_dim: int = 256
    eps: float = 1e-6
    theta: float = 10000.0
    # flow matching
    chunk: int = 50
    action_dim: int = 32
    out_dim: int = 7
    steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0
    cameras: int = 2

    @property
    def v_head_dim(self) -> int:
        return self.v_hidden // self.v_heads

    @property
    def side(self) -> int:
        return self.image // self.patch

    @property
    def n_img(self) -> int:
        """Image rows per camera (256)."""
        return self.side ** 2

    @staticmethod
    def from_ckpt(model_dir) -> "Spec":
        c = json.loads((Path(model_dir) / "config.json").read_text())
        if c.get("type") != "pi05":
            raise ValueError("not a pi05 checkpoint")
        if (c["paligemma_variant"], c["action_expert_variant"]) != ("gemma_2b", "gemma_300m"):
            raise ValueError("only gemma_2b + gemma_300m are laid out")
        if list(c["image_resolution"]) != [224, 224]:
            raise ValueError("only 224 x 224 images")
        cams = sum(1 for k, f in c["input_features"].items()
                   if f["type"] == "VISUAL" and "empty_camera" not in k)
        return Spec(chunk=c["chunk_size"], action_dim=c["max_action_dim"],
                    out_dim=c["output_features"]["action"]["shape"][0],
                    steps=c["num_inference_steps"], min_period=c["min_period"],
                    max_period=c["max_period"], cameras=cams)


class Weights:
    """The checkpoint's tensors by name, each read when asked (fp32 numpy; bf16 widened)."""

    def __init__(self, model_dir):
        from safetensors import safe_open
        self.dir = Path(model_dir)
        self._f = safe_open(str(self.dir / "model.safetensors"), "pt")
        self._keys = set(self._f.keys())

    def __contains__(self, k) -> bool:
        return k in self._keys

    def keys(self):
        return self._keys

    def __getitem__(self, k) -> np.ndarray:
        return self._f.get_tensor(k).float().numpy()

    def rows(self, k, idx) -> np.ndarray:
        """Rows idx of a matrix (the embedding table: no 2 GB fp32 copy)."""
        t = self._f.get_slice(k)
        return np.stack([t[int(i):int(i) + 1].float().numpy()[0] for i in idx])


# =============================================================================== host side
def patches(img: np.ndarray, patch: int = 14) -> np.ndarray:
    """[3, H, W] pixels -> [(H/p)(W/p), 3 p p] patch rows, row-major over the grid, each
    flattened (channel, row, column): the Conv2d's weight [C, 3, p, p] reshaped to [C, 3pp]
    makes the patch embedding a matrix product."""
    c, h, w = img.shape
    g = img.reshape(c, h // patch, patch, w // patch, patch).transpose(1, 3, 0, 2, 4)
    return np.ascontiguousarray(g.reshape(-1, c * patch * patch), np.float32)


def rope_table(spec: Spec, n: int) -> tuple[np.ndarray, np.ndarray]:
    """cos, sin [n, head_dim / 2] of positions 0 .. n - 1 (HF Gemma: inv_freq in fp32 from
    theta ** (arange(0, d, 2) / d), angles pos * inv_freq in fp32, rotate-half)."""
    d = spec.head_dim
    inv = (1.0 / (spec.theta ** (np.arange(0, d, 2, dtype=np.int64).astype(np.float32) / d)))
    inv = inv.astype(np.float32)
    ang = np.arange(n, dtype=np.float32)[:, None] * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def times(spec: Spec) -> list[float]:
    """The Euler schedule: t = 1 + s dt, dt = -1 / steps (Python doubles, as lerobot's
    euler_integrate, then fp32)."""
    dt = -1.0 / spec.steps
    return [float(np.float32(1.0 + s * dt)) for s in range(spec.steps)]


def time_embedding(spec: Spec, t: float) -> np.ndarray:
    """create_sinusoidal_pos_embedding of one fp32 t: [sin, cos] of t 2 pi / period, the
    periods geometric from min_period to max_period over e_hidden / 2 (float64, then fp32)."""
    half = spec.e_hidden // 2
    frac = np.linspace(0.0, 1.0, half, dtype=np.float64)
    period = spec.min_period * (spec.max_period / spec.min_period) ** frac
    x = (1.0 / period * 2 * math.pi) * np.float64(np.float32(t))
    return np.concatenate([np.sin(x), np.cos(x)]).astype(np.float32)


def _silu(x):
    return x / (np.float32(1) + np.exp(-x))


def time_condition(spec: Spec, W, t: float) -> np.ndarray:
    """adaRMS's condition for time t: silu(time_mlp_out(silu(time_mlp_in(sincos(t)))))."""
    e = time_embedding(spec, t)
    h = _silu(e @ W[TOP + "time_mlp_in.weight"].T + W[TOP + "time_mlp_in.bias"])
    return _silu(h @ W[TOP + "time_mlp_out.weight"].T + W[TOP + "time_mlp_out.bias"])


def modulations(spec: Spec, W, cond: np.ndarray) -> dict:
    """Every expert adaRMS norm's (scale, shift, gate) for one condition: {(layer, "in" /
    "post"): ([H], [H], [H])} and "final": (scale, shift) (its gate is unused)."""
    out = {}

    def dense(p):
        m = cond @ W[p + ".dense.weight"].T + W[p + ".dense.bias"]
        return np.split(m.astype(np.float32), 3)
    for i in range(spec.e_layers):
        p = f"{EXP}layers.{i}."
        out[i, "in"] = tuple(dense(p + "input_layernorm"))
        out[i, "post"] = tuple(dense(p + "post_attention_layernorm"))
    out["final"] = tuple(dense(EXP + "norm"))[:2]
    return out


def prompt_rows(spec: Spec, W, tokens, mask) -> np.ndarray:
    """The valid prompt tokens' rows [n, hidden]: the tied table's rows times sqrt(hidden)
    in fp32 (HF's GemmaScaledWordEmbedding; a host gather, as the other models' int8 tables
    on the host)."""
    ids = [int(t) for t, m in zip(tokens, mask) if m]
    e = W.rows(LLM + "embed_tokens.weight", ids) if LLM + "embed_tokens.weight" in W else \
        W.rows("model.paligemma_with_expert.paligemma.lm_head.weight", ids)
    return (e * np.float32(spec.hidden ** 0.5)).astype(np.float32)


# =============================================================================== reference
def _ln(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    xc = x - m
    return (xc / np.sqrt((xc * xc).mean(-1, keepdims=True) + np.float32(eps)) * w + b
            ).astype(np.float32)


def gelu_tanh(x):
    return (0.5 * x * (1 + np.tanh(np.float32(math.sqrt(2 / math.pi))
                                   * (x + np.float32(0.044715) * x ** 3)))).astype(np.float32)


def _rms(x, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(eps))


def _softmax(s):
    s = np.exp(s - s.max(-1, keepdims=True))
    return s / s.sum(-1, keepdims=True)


def _rope(x, cos, sin):
    """x [T, h, d] at per-row cos / sin [T, d/2] (rotate-half)."""
    h = x.shape[-1] // 2
    c, s = cos[:, None, :], sin[:, None, :]
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def vision_layer(spec: Spec, W, x, i):
    p = f"{TOWER}encoder.layers.{i}."
    lin = lambda y, n: y @ W[p + n + ".weight"].T + W[p + n + ".bias"]   # noqa: E731
    y = _ln(x, W[p + "layer_norm1.weight"], W[p + "layer_norm1.bias"], spec.v_eps)
    T, H, d = len(x), spec.v_heads, spec.v_head_dim
    q, k, v = (lin(y, f"self_attn.{n}_proj").reshape(T, H, d) for n in "qkv")
    s = np.einsum("thd,uhd->htu", q, k) * np.float32(d ** -0.5)
    o = np.einsum("htu,uhd->thd", _softmax(s), v).reshape(T, H * d)
    x = x + lin(o, "self_attn.out_proj")
    y = _ln(x, W[p + "layer_norm2.weight"], W[p + "layer_norm2.bias"], spec.v_eps)
    return (x + lin(gelu_tanh(lin(y, "mlp.fc1")), "mlp.fc2")).astype(np.float32)


def reference_vision(spec: Spec, W, img: np.ndarray, capture=None, layers=None) -> np.ndarray:
    """One image [3, 224, 224] (pixels in [-1, 1]) -> its 256 language-model rows [256, 2048].
    capture: gets "embed", "layer<i>", "post". layers: run only the first n (a cut model; the
    post LayerNorm and projector still run)."""
    cap = capture if capture is not None else {}
    C = spec.v_hidden
    wp = W[TOWER + "embeddings.patch_embedding.weight"].reshape(C, -1)
    x = (patches(img, spec.patch) @ wp.T + W[TOWER + "embeddings.patch_embedding.bias"]
         + W[TOWER + "embeddings.position_embedding.weight"]).astype(np.float32)
    cap["embed"] = x
    for i in range(spec.v_layers if layers is None else layers):
        x = vision_layer(spec, W, x, i)
        cap[f"layer{i}"] = x
    x = _ln(x, W[TOWER + "post_layernorm.weight"], W[TOWER + "post_layernorm.bias"], spec.v_eps)
    cap["post"] = x
    m = PG + "multi_modal_projector.linear."
    return (x @ W[m + "weight"].T + W[m + "bias"]).astype(np.float32)


def _attn(q, k, v, scale):
    """q [T, Hq, d] against one KV head k, v [U, d]: bidirectional, all U keys."""
    s = np.einsum("thd,ud->htu", q, k) * np.float32(scale)
    return np.einsum("htu,ud->thd", _softmax(s), v)


def _mlp(W, p, y):
    g = y @ W[p + "mlp.gate_proj.weight"].T
    u = y @ W[p + "mlp.up_proj.weight"].T
    return (gelu_tanh(g) * u) @ W[p + "mlp.down_proj.weight"].T


def reference_prefix(spec: Spec, W, x: np.ndarray, capture=None, layers=None) -> list:
    """The valid prefix rows x [Np, 2048] through the 2B (bidirectional, positions 0 ..
    Np - 1): every layer's (K after RoPE [Np, 256], V [Np, 256]). capture: "h<i>" the rows
    after layer i. layers: the first n only (a cut model)."""
    cap = capture if capture is not None else {}
    T, H, d = len(x), spec.n_q, spec.head_dim
    cos, sin = rope_table(spec, T)
    kv = []
    for i in range(spec.layers if layers is None else layers):
        p = f"{LLM}layers.{i}."
        y = _rms(x, spec.eps) * (1 + W[p + "input_layernorm.weight"])
        q = (y @ W[p + "self_attn.q_proj.weight"].T).reshape(T, H, d)
        k = (y @ W[p + "self_attn.k_proj.weight"].T).reshape(T, 1, d)
        v = y @ W[p + "self_attn.v_proj.weight"].T
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)[:, 0]
        kv.append((k.astype(np.float32), v.astype(np.float32)))
        o = _attn(q, k, v, d ** -0.5).reshape(T, H * d)
        x = x + o @ W[p + "self_attn.o_proj.weight"].T
        y = _rms(x, spec.eps) * (1 + W[p + "post_attention_layernorm.weight"])
        x = (x + _mlp(W, p, y)).astype(np.float32)
        cap[f"h{i}"] = x
    return kv


def reference_step(spec: Spec, W, kv: list, x_t: np.ndarray, mods: dict, capture=None,
                   layers=None) -> np.ndarray:
    """One flow-matching velocity v_t [50, 32] of x_t [50, 32]: the expert over the prefix
    KV (kv: reference_prefix's per layer), the step's adaRMS modulations (modulations()).
    capture: "in", "h<i>", "out". layers: the first n (a cut model, its kv as long)."""
    cap = capture if capture is not None else {}
    Np = len(kv[0][0])
    T, H, d = spec.chunk, spec.n_q, spec.head_dim
    cos, sin = rope_table(spec, Np + T)
    cos, sin = cos[Np:], sin[Np:]
    x = (x_t @ W[TOP + "action_in_proj.weight"].T + W[TOP + "action_in_proj.bias"]).astype(
        np.float32)
    cap["in"] = x
    for i in range(spec.e_layers if layers is None else layers):
        p = f"{EXP}layers.{i}."
        sc, sh, g = mods[i, "in"]
        y = _rms(x, spec.eps) * (1 + sc) + sh
        q = (y @ W[p + "self_attn.q_proj.weight"].T).reshape(T, H, d)
        k = (y @ W[p + "self_attn.k_proj.weight"].T).reshape(T, 1, d)
        v = y @ W[p + "self_attn.v_proj.weight"].T
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)[:, 0]
        K = np.concatenate([kv[i][0], k])
        V = np.concatenate([kv[i][1], v])
        o = _attn(q, K, V, d ** -0.5).reshape(T, H * d)
        x = x + (o @ W[p + "self_attn.o_proj.weight"].T) * g
        sc, sh, g = mods[i, "post"]
        y = _rms(x, spec.eps) * (1 + sc) + sh
        x = (x + _mlp(W, p, y) * g).astype(np.float32)
        cap[f"h{i}"] = x
    sc, sh = mods["final"]
    y = (_rms(x, spec.eps) * (1 + sc) + sh).astype(np.float32)
    cap["out"] = y
    return (y @ W[TOP + "action_out_proj.weight"].T + W[TOP + "action_out_proj.bias"]).astype(
        np.float32)


def reference_actions(spec: Spec, W, images: list, tokens, mask, noise: np.ndarray,
                      capture=None, cut: dict | None = None) -> np.ndarray:
    """The whole policy in fp32 numpy: images (the preprocessed [3, 224, 224] of each real
    camera), the 200 tokens and their mask, noise [50, 32] -> the model-space actions
    [50, 32] (sample_actions' output; the caller unnormalizes the first out_dim). capture:
    "vis<c>", "prefix", "kv", "cond<s>", "x<s>", "v<s>". cut: {"vision": n, "prefix": n,
    "expert": n} layers of each part (a cut model)."""
    cap = capture if capture is not None else {}
    cut = cut or {}
    vis = [reference_vision(spec, W, im, layers=cut.get("vision")) for im in images]
    for c, v in enumerate(vis):
        cap[f"vis{c}"] = v
    x = np.concatenate(vis + [prompt_rows(spec, W, tokens, mask)]).astype(np.float32)
    cap["prefix"] = x
    kv = reference_prefix(spec, W, x, layers=cut.get("prefix"))
    cap["kv"] = kv
    n = cut.get("expert")
    if n is not None and n > len(kv):
        raise ValueError("a cut expert needs as many prefix layers")
    xt = np.asarray(noise, np.float32)
    dt = np.float32(-1.0 / spec.steps)
    for s, t in enumerate(times(spec)):
        cond = time_condition(spec, W, t)
        cap[f"cond{s}"], cap[f"x{s}"] = cond, xt
        v = reference_step(spec, W, kv, xt, modulations(spec, W, cond),
                           layers=len(kv) if n is None else n)
        cap[f"v{s}"] = v
        xt = (xt + dt * v).astype(np.float32)
    return xt
