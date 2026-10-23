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
    name = LLM + "embed_tokens.weight"
    if name not in W:
        name = "model.paligemma_with_expert.paligemma.lm_head.weight"     # (tied)
    e = W.rows(name, ids) if hasattr(W, "rows") else np.asarray(W[name])[ids]
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


class Exact:
    """The reference's arithmetic: fp32 matrix products (the numerics hook of
    reference_prefix / reference_step; Emulated quantizes where the card does)."""

    def lin(self, x, W, name: str, kind: str = "") -> np.ndarray:
        return x @ W[name].T

    def attn(self, q, k, v, scale) -> np.ndarray:
        return _attn(q, k, v, scale)


class Emulated(Exact):
    """The card's quantization points in float64 (docs/pi05.md 4), to measure which of them
    costs the actions accuracy: "w" every weight matrix in its MXU format (quant.
    quantize_mxu, dequantized), "a" every streamed product's activation operand as QACT
    makes it (one row scale, power-of-two offsets per D/4 group: fp32.quantize_mx), "kv" the
    attention's int8 operands (q per row, K per token with its block offsets, V per token,
    P V-scaled per row of each 256-key block). io=False keeps action_in_proj and
    action_out_proj exact (the host's / the VPU's fp32). The emulation is not bit exact
    (float64 sums, softmax in one pass): a measurement of each point's size."""

    def __init__(self, parts=("w", "a", "kv"), io: bool = True, fmt: str = "int8",
                 D: int = 128, a_only=None):
        self.parts, self.io, self.fmt, self.D, self._w = set(parts), io, fmt, D, {}
        self.a_only = a_only        # activations quantized only into these matrices (names'
        #                             substrings), the others' exact

    def _wq(self, W, name):
        if name not in self._w:
            from .. import quant as Q
            if not name.startswith((EXP, TOP)):     # (the 2B's: not kept, 7 GB as fp32)
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
        if kind == "io" and not self.io:
            return x @ W[name].T
        w = self._wq(W, name) if "w" in self.parts else W[name]
        qa = "a" in self.parts and (self.a_only is None or any(o in name for o in self.a_only))
        a = self._aq(x) if qa else x
        return (a @ np.asarray(w).T).astype(np.float32)

    def attn(self, q, k, v, scale) -> np.ndarray:
        if "kv" not in self.parts:
            return _attn(q, k, v, scale)
        T, H, d = q.shape
        qd = self._aq(q.reshape(T * H, d) * np.float32(scale)).reshape(T, H, d)
        kd = self._aq(k, g=self.D)              # QST: groups of D, the token's row scale
        vd = self._aq(v, row=True)              # V^T's one scale per token (QST ROW)
        s = np.einsum("thd,ud->htu", qd, kd)
        p = np.exp(s - s.max(-1, keepdims=True))
        l = p.sum(-1, keepdims=True)
        out = np.zeros((H, T, d))
        for b0 in range(0, len(k), 256):        # P quantized per 256-key block
            pb = p[:, :, b0:b0 + 256].reshape(H * T, -1)
            out += (self._aq(pb).reshape(H, T, -1) @ vd[b0:b0 + 256])
        return (out / l).transpose(1, 0, 2).astype(np.float32)


EXACT = Exact()


def _mlp(W, p, y, nq=EXACT):
    g = nq.lin(y, W, p + "mlp.gate_proj.weight")
    u = nq.lin(y, W, p + "mlp.up_proj.weight")
    return nq.lin(gelu_tanh(g) * u, W, p + "mlp.down_proj.weight")


def reference_prefix(spec: Spec, W, x: np.ndarray, capture=None, layers=None,
                     nq=EXACT) -> list:
    """The valid prefix rows x [Np, 2048] through the 2B (bidirectional, positions 0 ..
    Np - 1): every layer's (K after RoPE [Np, 256], V [Np, 256]). capture: "h<i>" the rows
    after layer i. layers: the first n only (a cut model). nq: the numerics (Exact,
    Emulated)."""
    cap = capture if capture is not None else {}
    T, H, d = len(x), spec.n_q, spec.head_dim
    cos, sin = rope_table(spec, T)
    kv = []
    n = spec.layers if layers is None else layers
    for i in range(n):
        p = f"{LLM}layers.{i}."
        y = _rms(x, spec.eps) * (1 + W[p + "input_layernorm.weight"])
        q = nq.lin(y, W, p + "self_attn.q_proj.weight").reshape(T, H, d)
        k = nq.lin(y, W, p + "self_attn.k_proj.weight").reshape(T, 1, d)
        v = nq.lin(y, W, p + "self_attn.v_proj.weight")
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)[:, 0]
        kv.append((k.astype(np.float32), v.astype(np.float32)))
        if i == n - 1:
            break                               # the last layer's rest is dead
        o = nq.attn(q, k, v, d ** -0.5).reshape(T, H * d)
        x = x + nq.lin(o, W, p + "self_attn.o_proj.weight")
        y = _rms(x, spec.eps) * (1 + W[p + "post_attention_layernorm.weight"])
        x = (x + _mlp(W, p, y, nq)).astype(np.float32)
        cap[f"h{i}"] = x
    return kv


def reference_step(spec: Spec, W, kv: list, x_t: np.ndarray, mods: dict, capture=None,
                   layers=None, nq=EXACT) -> np.ndarray:
    """One flow-matching velocity v_t [50, 32] of x_t [50, 32]: the expert over the prefix
    KV (kv: reference_prefix's per layer), the step's adaRMS modulations (modulations()).
    capture: "in", "h<i>", "out". layers: the first n (a cut model, its kv as long). nq: the
    numerics."""
    cap = capture if capture is not None else {}
    Np = len(kv[0][0])
    T, H, d = spec.chunk, spec.n_q, spec.head_dim
    cos, sin = rope_table(spec, Np + T)
    cos, sin = cos[Np:], sin[Np:]
    x = (nq.lin(x_t, W, TOP + "action_in_proj.weight", "io") + W[TOP + "action_in_proj.bias"]
         ).astype(np.float32)
    cap["in"] = x
    for i in range(spec.e_layers if layers is None else layers):
        p = f"{EXP}layers.{i}."
        sc, sh, g = mods[i, "in"]
        y = _rms(x, spec.eps) * (1 + sc) + sh
        q = nq.lin(y, W, p + "self_attn.q_proj.weight").reshape(T, H, d)
        k = nq.lin(y, W, p + "self_attn.k_proj.weight").reshape(T, 1, d)
        v = nq.lin(y, W, p + "self_attn.v_proj.weight")
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)[:, 0]
        K = np.concatenate([kv[i][0], k])
        V = np.concatenate([kv[i][1], v])
        o = nq.attn(q, K, V, d ** -0.5).reshape(T, H * d)
        x = x + nq.lin(o, W, p + "self_attn.o_proj.weight") * g
        sc, sh, g = mods[i, "post"]
        y = _rms(x, spec.eps) * (1 + sc) + sh
        x = (x + _mlp(W, p, y, nq) * g).astype(np.float32)
        cap[f"h{i}"] = x
    sc, sh = mods["final"]
    y = (_rms(x, spec.eps) * (1 + sc) + sh).astype(np.float32)
    cap["out"] = y
    return (nq.lin(y, W, TOP + "action_out_proj.weight", "io") + W[TOP + "action_out_proj.bias"]
            ).astype(np.float32)


def reference_actions(spec: Spec, W, images: list, tokens, mask, noise: np.ndarray,
                      capture=None, cut: dict | None = None, nq=EXACT, enq=None,
                      vis=None) -> np.ndarray:
    """The whole policy in fp32 numpy: images (the preprocessed [3, 224, 224] of each real
    camera), the 200 tokens and their mask, noise [50, 32] -> the model-space actions
    [50, 32] (sample_actions' output; the caller unnormalizes the first out_dim). capture:
    "vis<c>", "prefix", "kv", "cond<s>", "x<s>", "v<s>". cut: {"vision": n, "prefix": n,
    "expert": n} layers of each part (a cut model). nq: the prefix's numerics, enq the
    expert's (default nq's); vis: the cameras' projector rows if known (skips the tower)."""
    cap = capture if capture is not None else {}
    cut = cut or {}
    if vis is None:
        vis = [reference_vision(spec, W, im, layers=cut.get("vision")) for im in images]
    for c, v in enumerate(vis):
        cap[f"vis{c}"] = v
    x = np.concatenate(vis + [prompt_rows(spec, W, tokens, mask)]).astype(np.float32)
    cap["prefix"] = x
    kv = reference_prefix(spec, W, x, layers=cut.get("prefix"), nq=nq)
    enq = nq if enq is None else enq
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
                           layers=len(kv) if n is None else n, nq=enq)
        cap[f"v{s}"] = v
        xt = (xt + dt * v).astype(np.float32)
    return xt


# =============================================================================== on the card
def _up(n: int, q: int) -> int:
    return -(-n // q) * q


class _Block:
    """One layer's tensors at fixed offsets of a block; the layers' blocks `stride` bytes
    apart from `first`, so a layer index (an int, a run-time value, a hardware loop's
    counter) addresses them all (base = first + li * stride)."""

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
        """The namespace of layer li's tensors: vectors as Tensors, matrices as QTensors."""
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


class Image:
    """pi0.5 in one slice's DRAM (docs/pi05.md 3): the weights in an MXU format (int8 by
    default: quant.quantize_mxu's rows of the checkpoint's fp32 values), each part's layers in
    equal blocks (_Block), the vectors fp32; the work areas of one action chunk with a prefix
    of up to `np_max` valid rows. Layers: spec's (a cut model: a Spec with fewer).

    Vision (SigLIP, kernels/vit.py's phases, minicpmv's conventions): per layer wq, wk, wv,
    wo [C, C], w1 [F, C], w2 [C, F] (F = 4304 padded to its 256-chunks with zeros), the
    LayerNorms' gains, their biases folded into the next linear's (float64); the patch
    embedding [C, 640] (588 padded), its bias, the position rows; the post LayerNorm; the
    projector [2048, C] and its bias. Prefix (Gemma 2B): per layer 1 + w of both norms, wq
    [2048, 2048], wk, wv [256, 2048], wo, wgu [2 F, 2048] (gate | up by chunks), wd
    [2048, F]. Expert: the same shapes at 1024 / 4096; action_in_proj's [1024, 128] (its 32
    columns padded), action_out_proj [32, 1024], their biases; the adaRMS table: per step
    and layer 1 + scale, shift, gate of both norms, per step the final norm's 1 + scale and
    shift (host fp32, build_mods).

    Work: the camera's patch rows P and residual X, Q / O scratch and KV scratch; the
    prefix's rows PX [np_max, 2048], its Q / O scratch, the RoPE tables of every position,
    every layer's KV cache (np_max + chunk tokens: the expert appends its rows after the
    prefix's); the expert's rows EX, Q / O scratch, x_t [chunk, 128] and v_t [chunk, 32]."""

    VCHUNK = 256        # the tower's MLP chunk (4304 -> 4352)
    PCHUNK = 256        # the 2B's GeGLU chunk
    ECHUNK = 256        # the expert's

    def __init__(self, spec: Spec, cfg, np_max: int = 640, fmt: str = "int8",
                 io32: bool = True):
        from ..compiler import k_scale_bytes
        from .qwen3 import _Bump
        self.spec, self.cfg, self.fmt, self.io32 = spec, cfg, fmt, io32
        D = self.D = cfg.D
        s = spec
        C, F = s.v_hidden, _up(s.v_ffn, self.VCHUNK)
        self.vF, self.KP = F, _up(3 * s.patch ** 2, D)
        self.vdp = _up(s.v_head_dim, D)
        self.pchunk = min(self.PCHUNK, s.ffn)
        self.echunk = min(self.ECHUNK, s.e_ffn)
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
        gb = _Block(D)                  # the tower's globals: one block
        gb.vec("bp", C)
        gb.vec("post_w", C)
        gb.vec("post_b", C)
        gb.vec("bproj", s.hidden)
        gb.vec("E", s.n_img * C)
        gb.mat("wp", C, self.KP, fmt)
        gb.mat("wproj", s.hidden, C, fmt)
        b.next = gb.place(_up(b.next, 4096), 1)
        self.gb = gb
        # ---- prefix
        H, d, Fp = s.hidden, s.head_dim, s.ffn
        pb = _Block(D)
        pb.vec("g1", H)
        pb.vec("g2", H)
        pb.mat("wq", s.n_q * d, H, fmt)
        pb.mat("wk", d, H, fmt)
        pb.mat("wv", d, H, fmt)
        pb.mat("wo", H, s.n_q * d, fmt)
        pb.mat("wgu", 2 * Fp, H, fmt)
        pb.mat("wd", H, Fp, fmt)
        b.next = pb.place(_up(b.next, 4096), s.layers)
        self.pb = pb
        # ---- expert
        E, Fe = s.e_hidden, s.e_ffn
        eb = _Block(D)
        eb.mat("wq", s.n_q * d, E, fmt)
        eb.mat("wk", d, E, fmt)
        eb.mat("wv", d, E, fmt)
        eb.mat("wo", E, s.n_q * d, fmt)
        eb.mat("wgu", 2 * Fe, E, fmt)
        eb.mat("wd", E, Fe, fmt)
        b.next = eb.place(_up(b.next, 4096), s.e_layers)
        self.eb = eb
        xb = _Block(D)                  # the expert's globals
        self.AP = _up(s.action_dim, D)
        xb.vec("bin", E)
        xb.vec("bout", s.action_dim)
        xb.mat("win", E, self.AP, fmt)
        xb.mat("wout", s.action_dim, E, fmt)
        # io32 (the default): both projections fp32 on the VPU instead, W_in^T and W_out by
        # rows, the output bias repeated over the chunk's rows (one LD a dim)
        xb.vec("winT", s.action_dim * E)
        xb.vec("woutf", s.action_dim * E)
        xb.vec("boutT", s.action_dim * s.chunk)
        b.next = xb.place(_up(b.next, 4096), 1)
        self.xb = xb
        mb = _Block(D)                  # the adaRMS table: a block per (step, layer)
        for n in ("g_in", "s_in", "a_in", "g_post", "s_post", "a_post"):
            mb.vec(n, E)
        b.next = mb.place(_up(b.next, 4096), s.steps * s.e_layers)
        self.mb = mb
        fb = _Block(D)                  # the final norm's, a block per step
        fb.vec("g", E)
        fb.vec("s", E)
        b.next = fb.place(_up(b.next, 4096), s.steps)
        self.fb = fb
        self.weights_end = b.next
        # ---- work
        self.np_max = np_max
        self.cap = cap = _up(np_max + s.chunk, D)
        n = s.n_img
        T = s.chunk
        self.work = dict(P=b.alloc(4 * n * self.KP), X=b.alloc(4 * n * C),
                         Q=b.alloc(4 * n * s.v_heads * self.vdp), O=b.alloc(4 * n * C),
                         PX=b.alloc(4 * np_max * H), PQ=b.alloc(4 * cap * s.n_q * d),
                         PO=b.alloc(4 * cap * s.n_q * d), cos=b.alloc(4 * cap * d // 2),
                         sin=b.alloc(4 * cap * d // 2), EX=b.alloc(4 * T * E),
                         EQ=b.alloc(4 * T * s.n_q * d), EO=b.alloc(4 * T * s.n_q * d),
                         XT=b.alloc(4 * T * self.AP), V=b.alloc(4 * T * s.action_dim),
                         XTt=b.alloc(4 * s.action_dim * T), Vt=b.alloc(4 * s.action_dim * T))
        nv = _up(n, D)
        self.vkv = {h: {"k": b.alloc(nv * self.vdp), "ks": b.alloc(4 * nv),
                        "vt": b.alloc(self.vdp * nv + D), "vs": b.alloc(4 * nv)}
                    for h in range(s.v_heads)}
        self.nv = nv
        kb = _Block(D)                  # a layer's KV cache: one block per layer
        kb.vec("k", cap * d // 4)
        kb.vec("ks", cap * k_scale_bytes(d, D) // 4 + 1)
        kb.vec("vt", (d * cap + D) // 4)
        kb.vec("vs", cap)
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
        s, n, C = self.spec, self.spec.n_img, self.spec.v_hidden
        g = self.gb.at(0)
        return SimpleNamespace(
            C=C, heads=s.v_heads, dp=self.vdp, dv=s.v_head_dim,
            P=self._t("P", (n, self.KP)), E=Tensor(g.E.base, (n, C), (C, 1)),
            X=self._t("X", (n, C)),
            Q=[self._t("Q", (n, self.vdp), 4 * h * n * self.vdp) for h in range(s.v_heads)],
            O=self._t("O", (n, C)),
            kv=VitKV(self.vkv, self.nv, self.vdp, self.D, 1, 0, dv=s.v_head_dim),
            emb=SimpleNamespace(wp=g.wp, bp=g.bp), post=(g.post_w, g.post_b))

    def kv(self, li):
        """Layer li's KV cache (one head: K token-major, V^T rows cap wide)."""
        from ..compiler import Affine
        from ..kernels.vit import VitKV
        base = Affine(self.kb.first) + Affine.of(li) * self.kb.stride
        h = {k: base + o for k, (o, _) in self.kb.vecs.items()}
        return VitKV({0: h}, self.cap, self.spec.head_dim, self.D, 1, 0)

    def gemma_desc(self, rows: str, n: int, li):
        """The prefix's (rows "P") or the expert's ("E") `g`: X, Q, O, RoPE tables, KV."""
        from types import SimpleNamespace
        s, d, cap = self.spec, self.spec.head_dim, self.cap
        X, Q, O = ("PX", "PQ", "PO") if rows == "P" else ("EX", "EQ", "EO")
        H = s.hidden if rows == "P" else s.e_hidden
        qrows = cap if rows == "P" else s.chunk
        return SimpleNamespace(
            H=s.n_q, d=d, X=self._t(X, (n, H)),
            Q=[self._t(Q, (qrows, d), 4 * h * qrows * d) for h in range(s.n_q)],
            O=self._t(O, (qrows, s.n_q * d)), cos=self._t("cos", (cap, d // 2)),
            sin=self._t("sin", (cap, d // 2)), kv=self.kv(li))

    def expert_desc(self):
        from types import SimpleNamespace
        from ..compiler import Tensor
        s = self.spec
        x = self.xb.at(0)
        e = SimpleNamespace(
            X=self._t("EX", (s.chunk, s.e_hidden)),
            O=self._t("EO", (s.chunk, s.n_q * s.head_dim)),
            XT=self._t("XT", (s.chunk, self.AP)), V=self._t("V", (s.chunk, s.action_dim)),
            win=x.win, bin=x.bin, wout=x.wout, bout=x.bout, layer=self.eb.at, kv=self.kv,
            XTt=self._t("XTt", (s.action_dim, s.chunk)), Vt=self._t("Vt", (s.action_dim, s.chunk)),
            winT=Tensor(x.winT.base, (s.action_dim, s.e_hidden), (s.e_hidden, 1)),
            woutf=Tensor(x.woutf.base, (s.action_dim, s.e_hidden), (s.e_hidden, 1)),
            boutT=Tensor(x.boutT.base, (s.action_dim, s.chunk), (s.chunk, 1)),
            mods=lambda step, li: self.mb.at(step * s.e_layers + li), final=self.fb.at)
        e.g = self.gemma_desc("E", s.chunk, 0)
        return e

    # ---- contents
    def build(self, W, put, log=None) -> None:
        """Write the weights (W: Weights, or a dict of the same names) and the RoPE tables
        with put(addr, array); the adaRMS table is the host's (build_mods)."""
        s, C = self.spec, self.spec.v_hidden
        f64 = np.float64

        def pad(a, shape):
            out = np.zeros(shape, np.float32)
            out[tuple(slice(0, k) for k in a.shape)] = a
            return out

        def folded(p, ln):
            return W[p + ".bias"] + W[p + ".weight"].astype(f64) @ W[ln + ".bias"]
        vb, F = self.vb, self.vF
        for li in range(s.v_layers):
            p = f"{TOWER}encoder.layers.{li}."
            for n in "qkv":
                vb.put_mat(put, li, "w" + n, W[p + f"self_attn.{n}_proj.weight"])
                vb.put_vec(put, li, "b" + n,
                           folded(p + f"self_attn.{n}_proj", p + "layer_norm1"))
            vb.put_mat(put, li, "wo", W[p + "self_attn.out_proj.weight"])
            vb.put_vec(put, li, "bo", W[p + "self_attn.out_proj.bias"])
            vb.put_mat(put, li, "w1", pad(W[p + "mlp.fc1.weight"], (F, C)))
            vb.put_vec(put, li, "b1", folded(p + "mlp.fc1", p + "layer_norm2"))
            vb.put_mat(put, li, "w2", pad(W[p + "mlp.fc2.weight"], (C, F)))
            vb.put_vec(put, li, "b2", W[p + "mlp.fc2.bias"])
            vb.put_vec(put, li, "ln1w", W[p + "layer_norm1.weight"])
            vb.put_vec(put, li, "ln2w", W[p + "layer_norm2.weight"])
            if log:
                log(f"vision layer {li}")
        gb = self.gb
        gb.put_mat(put, 0, "wp", pad(W[TOWER + "embeddings.patch_embedding.weight"]
                                     .reshape(C, -1), (C, self.KP)))
        gb.put_vec(put, 0, "bp", W[TOWER + "embeddings.patch_embedding.bias"])
        gb.put_vec(put, 0, "E", W[TOWER + "embeddings.position_embedding.weight"])
        gb.put_vec(put, 0, "post_w", W[TOWER + "post_layernorm.weight"])
        gb.put_vec(put, 0, "post_b", W[TOWER + "post_layernorm.bias"])
        m = PG + "multi_modal_projector.linear."
        gb.put_mat(put, 0, "wproj", W[m + "weight"])
        gb.put_vec(put, 0, "bproj", W[m + "bias"])
        for blk, pre, n_l, ch in ((self.pb, LLM, s.layers, self.pchunk),
                                  (self.eb, EXP, s.e_layers, self.echunk)):
            for li in range(n_l):
                p = f"{pre}layers.{li}."
                for n in "qkvo":
                    blk.put_mat(put, li, "w" + n, W[p + f"self_attn.{n}_proj.weight"])
                blk.put_mat(put, li, "wgu", _interleave(W[p + "mlp.gate_proj.weight"],
                                                        W[p + "mlp.up_proj.weight"], ch))
                blk.put_mat(put, li, "wd", W[p + "mlp.down_proj.weight"])
                if pre == LLM:
                    blk.put_vec(put, li, "g1", 1 + W[p + "input_layernorm.weight"])
                    blk.put_vec(put, li, "g2", 1 + W[p + "post_attention_layernorm.weight"])
                if log:
                    log(f"{'prefix' if pre == LLM else 'expert'} layer {li}")
        xb = self.xb
        xb.put_mat(put, 0, "win", pad(W[TOP + "action_in_proj.weight"],
                                      (s.e_hidden, self.AP)))
        xb.put_vec(put, 0, "bin", W[TOP + "action_in_proj.bias"])
        xb.put_mat(put, 0, "wout", W[TOP + "action_out_proj.weight"])
        xb.put_vec(put, 0, "bout", W[TOP + "action_out_proj.bias"])
        xb.put_vec(put, 0, "winT", W[TOP + "action_in_proj.weight"].T.reshape(-1))
        xb.put_vec(put, 0, "woutf", W[TOP + "action_out_proj.weight"].reshape(-1))
        xb.put_vec(put, 0, "boutT", np.repeat(W[TOP + "action_out_proj.bias"], s.chunk))
        cos, sin = rope_table(s, self.cap)
        put(self.work["cos"], cos)
        put(self.work["sin"], sin)
        self.build_mods(W, put)

    def build_mods(self, W, put) -> None:
        """The adaRMS table of every step: t's condition and each norm's (1 + scale, shift,
        gate), host fp32 (time_condition, modulations: lerobot's arithmetic)."""
        s = self.spec
        for st, t in enumerate(times(s)):
            m = modulations(s, W, time_condition(s, W, t))
            for li in range(s.e_layers):
                k = st * s.e_layers + li
                for which in ("in", "post"):
                    sc, sh, g = m[li, which]
                    self.mb.put_vec(put, k, "g_" + which, 1 + sc)
                    self.mb.put_vec(put, k, "s_" + which, sh)
                    self.mb.put_vec(put, k, "a_" + which, g)
            sc, sh = m["final"]
            self.fb.put_vec(put, st, "g", 1 + sc)
            self.fb.put_vec(put, st, "s", sh)

    # ---- the programs
    def programs(self, Np: int, R: int = 16, RP: int = 8, RE: int = 8,
                 block: int = 256) -> dict:
        """The phase programs of one chunk with Np valid prefix rows: {name: (program,
        run_args)}. Vision (per camera; the layer a run argument "layer", the projector's
        destination row "row"), the prefix's (the layer a run argument), the expert step
        (the Euler step a run argument "step"). R: the tower's row group, RP the prefix's,
        RE the expert's."""
        from ..compiler import Affine, RunVar, Tensor
        from ..kernels import pi05 as K
        from ..kernels import vit as VK
        s, cfg = self.spec, self.cfg
        if Np > self.np_max:
            raise ValueError(f"{Np} prefix rows: the image holds {self.np_max}")
        n = s.n_img
        v = self.vision_desc()
        vlw = self.vb.at(RunVar("layer", bound=s.v_layers))
        vsc = 1.0 / math.log(2.0) / math.sqrt(s.v_head_dim)
        gsc = 1.0 / math.log(2.0) / math.sqrt(s.head_dim)

        def tr(kernel, **kw):
            b = kernel.trace(cfg, 0, kw)
            return b.finish(), list(b.run_args)
        row = RunVar("row", bound=self.np_max)
        dst = Tensor(Affine(self.work["PX"]) + row * (4 * s.hidden), (n, s.hidden),
                     (s.hidden, 1))
        g = self.gb.at(0)
        P = {"v_embed": tr(VK.vit_embed, v=v, N=n, R=R),
             "v_qkv": tr(VK.vit_qkv, v=v, lw=vlw, src=v.X, N=n, R=R, eps=s.v_eps),
             "v_attend": tr(VK.vit_attend, v=v, N=n, scale=vsc, block=block),
             "v_out": tr(VK.vit_out_mlp, v=v, lw=vlw, dst=v.X, N=n, R=R, eps=s.v_eps,
                         chunk=self.VCHUNK),
             "v_last": tr(VK.vit_out_mlp, v=v, lw=vlw, dst=v.X, N=n, R=R, eps=s.v_eps,
                          chunk=self.VCHUNK, post=v.post),
             "v_proj": tr(K.project_rows, src=v.X, w=g.wproj, b=g.bproj, dst=dst, N=n, R=R)}
        li = RunVar("layer", bound=s.layers)
        pg = self.gemma_desc("P", Np, li)
        plw = self.pb.at(li)
        P["p_qkv"] = tr(K.prefix_qkv, g=pg, lw=plw, N=Np, R=RP, eps=s.eps)
        P["p_attend"] = tr(K.prefix_attend, g=pg, N=Np, scale=gsc, block=block)
        P["p_out"] = tr(K.prefix_out, g=pg, lw=plw, N=Np, R=RP, eps=s.eps, chunk=self.pchunk)
        P["e_step"] = tr(K.expert_step, e=self.expert_desc(),
                         step=RunVar("step", bound=s.steps), layers=s.e_layers, Np=Np, R=RE,
                         eps=s.eps, scale=gsc, block=block, chunk=self.echunk,
                         dt=-1.0 / s.steps, io32=self.io32)
        return P

    def runs(self, cams: int) -> list:
        """The runs of one chunk in order: [(phase, {run argument: value})]; the host's
        writes go between ("write_patches", camera), ("write_prompt",), ("write_noise",)."""
        s = self.spec
        out = []
        for c in range(cams):
            out += [("write_patches", c), ("v_embed", {})]
            for i in range(s.v_layers):
                last = i == s.v_layers - 1
                out += [("v_qkv", {"layer": i}), ("v_attend", {}),
                        ("v_last" if last else "v_out", {"layer": i})]
            out.append(("v_proj", {"row": c * s.n_img}))
        out.append(("write_prompt",))
        for i in range(s.layers):
            out.append(("p_qkv", {"layer": i}))
            if i < s.layers - 1:
                out += [("p_attend", {"layer": i}), ("p_out", {"layer": i})]
        out.append(("write_noise",))
        for st in range(s.steps):
            out.append(("e_step", {"step": st}))
        return out


def run_chunk(image: Image, backend, programs: dict, images: list, prompt: np.ndarray,
              noise: np.ndarray, log=None, stop: str | None = None,
              after=None) -> np.ndarray:
    """One action chunk on `backend` (qwen3.IsaBackend: write, run, read): the cameras'
    preprocessed pixels, the prompt's rows (prompt_rows), noise [chunk, action_dim] -> the
    model-space actions [chunk, action_dim] (x_t after the last step). stop: "vision"
    returns (None) after the cameras, "prefix" after the prefix (diagnostics). log(name,
    args) before each run, after(name, args) after it."""
    from ..compiler import arg_words
    s = image.spec
    w = image.work
    for item in image.runs(len(images)):
        name = item[0]
        if name == "write_patches":
            c = item[1]
            p = np.zeros((s.n_img, image.KP), np.float32)
            p[:, :3 * s.patch ** 2] = patches(images[c], s.patch)
            backend.write(0, w["P"], p)
            continue
        if name == "write_prompt":
            if stop == "vision":
                return None
            backend.write(0, w["PX"] + 4 * len(images) * s.n_img * s.hidden,
                          np.asarray(prompt, np.float32))
            continue
        if name == "write_noise":
            if stop == "prefix":
                return None
            if image.io32:                  # x_t^T [action_dim, chunk]
                backend.write(0, w["XTt"], np.ascontiguousarray(np.asarray(noise, np.float32).T))
            else:
                x = np.zeros((s.chunk, image.AP), np.float32)
                x[:, :s.action_dim] = noise
                backend.write(0, w["XT"], x)
            continue
        prog, ra = programs[name]
        if log:
            log(name, item[1])
        backend.run([prog], args=arg_words(ra, item[1]))
        if after:
            after(name, item[1])
    return read_xt(image, backend)


def read_xt(image: Image, backend, which: str = "x") -> np.ndarray:
    """x_t (which "x") or the last step's v_t ("v") [chunk, action_dim] from the card."""
    s, w = image.spec, image.work
    if image.io32:
        a = backend.read(0, w["XTt" if which == "x" else "Vt"], 4 * s.chunk * s.action_dim)
        return a.view(np.float32).reshape(s.action_dim, s.chunk).T.copy()
    if which == "v":
        return backend.read(0, w["V"], 4 * s.chunk * s.action_dim).view(np.float32).reshape(
            s.chunk, s.action_dim).copy()
    return backend.read(0, w["XT"], 4 * s.chunk * image.AP).view(np.float32).reshape(
        s.chunk, image.AP)[:, :s.action_dim].copy()
