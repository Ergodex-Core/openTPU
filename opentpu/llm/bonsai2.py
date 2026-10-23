"""Ternary-Bonsai 2 (prism-ml, Qwen3.5-2B / 4B / 9B with ternary weights) from its GGUF.

The checkpoints exist only as GGUFs for the PrismML llama.cpp fork. A model directory holds
one of them (PQ2_0, Q2_0 or PTQ1_0: the same weights, packed differently) beside the base
model's config.json and tokenizer (Qwen/Qwen3.5-<size>), and qwen35 runs it as any Qwen3.5.

What the GGUF stores (convert_hf_to_gguf's Qwen3_5TextModel, the fork's llama-model.cpp):
  * Every projection (q, k, v, qkv, z, o, out, gate, up, down) is ternary, {-1, 0, +1}, with
    an fp16 scale per group of 128 (PQ2_0) along the input dimension, in a rotated input
    basis: W_f = W R^T, where the activation's transform before the matmul is
    R x = H (s * x), s a fixed +-1 vector per input width (prism.hadamard.sign_values, one for
    2048 / hidden and one for the MLP's width) and H the normalized Sylvester Walsh-Hadamard
    matrix of 512 (prism.hadamard.block_size), block-diagonal over the input. So the model's
    own weight is W = W_f H diag(s) (H symmetric, orthonormal).
  * The token embedding is ternary too, its rows in the same latent basis (inverse after the
    lookup: e = s * (H z)), and tied: the LM head is the same table through the forward
    transform, so both are E = Z H diag(s), one table in the model's basis.
  * Norm gains of the zero-centered RMSNorms hold 1 + w (the linear attention's output norm
    holds w); A_log is stored as -exp(A_log); with fewer DeltaNet key than value heads (4B,
    9B) the value heads' rows (qkv's v, z, a, b, A_log, dt_bias, the convolution's v taps)
    are in llama.cpp's tiled order [rep, nk] and out_proj's columns, folded, keep HF's grouped
    order (prism.hadamard.gdn_v_grouped).

Weights gives HF's names and layout (qwen35's), the projections in the model's basis
(dense fp32: W_f H diag(s)), for the reference and for images that quantize them again.
Weights.ternary(name) gives the exact ternary parts of a projection in its latent basis
(row order HF's) and Weights.signs(K) the sign vector its activations take, for a ternary
image that applies the transform on the device (docs/bonsai2.md).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .gguf_file import GGUF, TERNARY

PROJ = {"attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z",
        "ssm_out": "linear_attn.out_proj", "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
NORM1 = {"attn_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm",
         "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}
NEXTN = {"nextn.eh_proj.weight": "mtp.fc.weight",              # [embedding, hidden] in
         "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
         "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
         "nextn.shared_head_norm.weight": "mtp.norm.weight"}
SMALL = {"ssm_alpha": "linear_attn.in_proj_a", "ssm_beta": "linear_attn.in_proj_b",
         "ssm_norm": "linear_attn.norm", "ssm_conv1d": "linear_attn.conv1d"}


def find_gguf(model_dir) -> Path | None:
    """The directory's language-model GGUF (not an mmproj / drafter), if it has one."""
    ps = [p for p in sorted(Path(model_dir).glob("*.gguf"))
          if "mmproj" not in p.name and "DFlash" not in p.name]
    if len(ps) > 1:
        raise ValueError(f"{model_dir}: more than one GGUF ({', '.join(p.name for p in ps)})")
    return ps[0] if ps else None


def is_bonsai2(model_dir) -> bool:
    p = find_gguf(model_dir)
    return p is not None and not any(Path(model_dir).glob("*.safetensors")) and \
        GGUF(p).meta.get("general.architecture") == "qwen35"


def spec(model_dir, ternary: bool = False, slices: int = 1):
    """qwen35's Spec from the base model's config.json, the embedding rows int8 (gathered
    from the tied int8 head: an fp32 table of 248320 rows would be 2-4 GB of DRAM). ternary:
    the image keeps the latent ternary weights and rotates on the device (Spec.rot; Weights
    with latent=True), its head the latent table (not tied to the embedding rows, which are
    the model's). A GGUF with its own output.weight (27B: prism.hadamard.version 1) is untied.
    slices: the MLP's width is padded with zero rows / columns to whole rotation blocks per
    slice (27B: 17408 = 17 blocks of 1024 -> 18432 on 2 slices; Weights(ffn=) pads the same)."""
    from dataclasses import replace
    from .qwen35 import Spec
    g = GGUF(find_gguf(model_dir))
    sp = replace(Spec.from_hf(model_dir), embed="int8")
    if "output.weight" in g:
        sp = replace(sp, tied=False)
    if ternary:
        B = int(g.meta["prism.hadamard.block_size"])
        sp = replace(sp, rot=B, tied=False, ffn=-(-sp.ffn // (slices * B)) * slices * B)
    return sp


def hadamard(n: int) -> np.ndarray:
    """The normalized Sylvester Walsh-Hadamard matrix of n (a power of 2): entry (r, c) is
    (-1)^popcount(r & c) / sqrt(n), as the fork builds it."""
    r = np.arange(n)
    x = r[:, None] & r[None, :]
    par = np.zeros_like(x)
    while x.any():
        par ^= x & 1
        x >>= 1
    return np.where(par == 1, -1.0, 1.0) / math.sqrt(n)


def untile(a: np.ndarray, nk: int, rep: int, hd: int, axis: int = 0) -> np.ndarray:
    """llama.cpp's tiled value-head order [rep, nk, hd] back to HF's grouped [nk, rep, hd]
    along `axis` (the inverse of _reorder_v_heads)."""
    a = np.moveaxis(a, axis, 0)
    sh = a.shape
    a = a.reshape(rep, nk, hd, *sh[1:]).swapaxes(0, 1).reshape(sh)
    return np.moveaxis(a, 0, axis)


class Weights(Mapping):
    """A Ternary-Bonsai 2 GGUF under qwen35's (HF Qwen3.5 text) names; fp32 arrays in the
    model's basis, computed when asked for. Small tensors are cached, as qwen3.Weights."""

    CACHE = 16 << 20

    def __init__(self, model_dir, mtp: bool = False, latent: bool = False,
                 ffn: int | None = None):
        p = find_gguf(model_dir)
        self.latent = latent        # the projections in their rotated basis, as stored (the
        #                             image rotates the activations: Spec.rot)
        if p is None:
            raise ValueError(f"{model_dir}: no GGUF")
        self.model_dir, self.g = Path(model_dir), GGUF(p)
        m = self.g.meta
        if m.get("general.architecture") != "qwen35":
            raise ValueError(f"{p}: architecture {m.get('general.architecture')!r}, not qwen35")
        a = "qwen35."
        self.nextn = int(m.get(a + "nextn_predict_layers", 0))   # the MTP layer(s), last
        self.layers = int(m[a + "block_count"]) - self.nextn
        self.nk = int(m[a + "ssm.group_count"])
        self.nv = int(m[a + "ssm.time_step_rank"])
        self.dk = int(m[a + "ssm.state_size"])
        self.dv = int(m[a + "ssm.inner_size"]) // self.nv
        self.rep = self.nv // self.nk
        self.block = int(m.get("prism.hadamard.block_size", 0))
        self.folded = set(m.get("prism.hadamard.weight_names", []))
        self.inverse = set(m.get("prism.hadamard.inverse_weight_names", []))
        self.grouped = bool(m.get("prism.hadamard.gdn_v_grouped", False))
        if self.block and m.get("prism.hadamard.sign_mode") != "explicit":
            raise ValueError(f"{p}: sign_mode {m.get('prism.hadamard.sign_mode')!r}")
        self._signs = {}
        if self.block:
            vals, o = np.asarray(m["prism.hadamard.sign_values"], np.float32), 0
            for w in m["prism.hadamard.sign_widths"]:
                self._signs[int(w)] = vals[o:o + int(w)]
                o += int(w)
            if o != len(vals):
                raise ValueError(f"{p}: sign_values length {len(vals)}, widths sum {o}")
            self.H = hadamard(self.block).astype(np.float32)
        self._where = {}                                # HF name -> (gguf name, kind)
        self._where["model.embed_tokens.weight"] = ("token_embd.weight", "embed")
        if "output.weight" in self.g:   # an untied head (27B)
            self._where["lm_head.weight"] = ("output.weight", "proj")
        elif latent:                # the head: the latent table (lm_head of an untied Spec)
            self._where["lm_head.weight"] = ("token_embd.weight", "proj")
        self.ffn0 = int(m[a + "feed_forward_length"])
        self.ffn = ffn or self.ffn0         # the MLP's width with zero padding (spec(slices=))
        if self.ffn > self.ffn0 and self.block:
            self._signs[self.ffn] = np.concatenate(
                [self._signs[self.ffn0], np.ones(self.ffn - self.ffn0, np.float32)])
        self._where["model.norm.weight"] = ("output_norm.weight", "norm1")
        for i in range(self.layers):
            b, h = f"blk.{i}.", f"model.layers.{i}."
            for k, v in PROJ.items():
                if b + k + ".weight" in self.g:
                    self._where[h + v + ".weight"] = (b + k + ".weight", "proj")
            for k, v in NORM1.items():
                if b + k + ".weight" in self.g:
                    self._where[h + v + ".weight"] = (b + k + ".weight", "norm1")
            for k, v in SMALL.items():
                if b + k + ".weight" in self.g:
                    self._where[h + v + ".weight"] = (b + k + ".weight", "small")
            if b + "ssm_a" in self.g:
                self._where[h + "linear_attn.A_log"] = (b + "ssm_a", "a_log")
                self._where[h + "linear_attn.dt_bias"] = (b + "ssm_dt.bias", "small")
        if mtp and self.nextn != 1:
            raise ValueError(f"{p}: no MTP layer (use the *-MTP-*.gguf of the model)")
        if mtp:                     # the -MTP GGUF's nextn layer, under HF's mtp.* names
            b, h = f"blk.{self.layers}.", "mtp.layers.0."
            for k, v in PROJ.items():
                if b + k + ".weight" in self.g:
                    self._where[h + v + ".weight"] = (b + k + ".weight", "proj")
            for k, v in NORM1.items():
                if b + k + ".weight" in self.g:
                    self._where[h + v + ".weight"] = (b + k + ".weight", "norm1")
            for k, v in NEXTN.items():
                self._where[v] = (b + k, "norm1" if k.endswith("norm.weight") else "proj")
        self._small: dict = {}
        self._big: tuple | None = None

    # ------------------------------------------------------------------ the transform
    def signs(self, K: int) -> np.ndarray:
        """The +-1 vector an activation of width K takes before H (float32 [K])."""
        return self._signs[K]

    def rotate(self, x: np.ndarray) -> np.ndarray:
        """The activation-side transform R x = H (s * x) over the last axis (blocks of 512)."""
        K = x.shape[-1]
        y = (x * self.signs(K)).reshape(*x.shape[:-1], K // self.block, self.block)
        return (y @ self.H).reshape(x.shape)

    def unfold(self, Wf: np.ndarray) -> np.ndarray:
        """A folded weight [N, K] in the model's basis: W_f H diag(s) (H symmetric)."""
        N, K = Wf.shape
        y = (Wf.reshape(N, K // self.block, self.block) @ self.H).reshape(N, K)
        return y * self.signs(K)

    # ------------------------------------------------------------------ row orders
    def _vrows(self, hf: str) -> tuple | None:
        """(start, hd) of the value-head rows in tiled order of a DeltaNet tensor, or None."""
        if self.rep == 1 or ".linear_attn." not in hf:
            return None
        if hf.endswith("in_proj_qkv.weight") or hf.endswith("conv1d.weight"):
            return 2 * self.nk * self.dk, self.dv
        if hf.endswith("in_proj_z.weight"):
            return 0, self.dv
        if hf.endswith(("in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias")):
            return 0, 1
        return None

    def _order(self, hf: str, a: np.ndarray) -> np.ndarray:
        """Rows (and out_proj's columns) in HF's order."""
        v = self._vrows(hf)
        if v is not None:
            s, hd = v
            a = a.copy()
            a[s:] = untile(a[s:], self.nk, self.rep, hd)
        if self.rep > 1 and hf.endswith("linear_attn.out_proj.weight") and not self.grouped:
            a = untile(a, self.nk, self.rep, self.dv, axis=1)
        return a

    # ------------------------------------------------------------------ Mapping
    def __getitem__(self, hf):
        if hf in self._small:
            return self._small[hf]
        if self._big is not None and self._big[0] == hf:
            return self._big[1]
        gn, kind = self._where[hf]
        t = self.g[gn]
        if kind == "embed":
            v = self.unfold(t.dequant()) if gn in self.inverse else t.dequant()
        elif kind == "proj":
            v = t.dequant()
            if not self.latent and (gn in self.folded or gn in self.inverse):
                v = self.unfold(v)
        elif kind == "norm1":
            v = t.array() - np.float32(1)
        elif kind == "a_log":
            v = np.log(-t.array())
        else:
            v = t.array()
        if hf.endswith("conv1d.weight"):
            v = v.reshape(v.shape[0], 1, -1)
        v = self._pad(hf, v)
        v = np.ascontiguousarray(self._order(hf, v), np.float32)
        if not np.isfinite(v).all():
            raise ValueError(f"{hf}: NaN or infinity")
        if v.nbytes <= self.CACHE:
            self._small[hf] = v
        else:
            self._big = (hf, v)
        return v

    def _pad(self, hf: str, v: np.ndarray) -> np.ndarray:
        """The MLP's projections with zero rows (gate, up) or columns (down) to self.ffn."""
        n = self.ffn - self.ffn0
        if n and hf.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
            return np.concatenate([v, np.zeros((n, v.shape[1]), v.dtype)])
        if n and hf.endswith("mlp.down_proj.weight"):
            return np.concatenate([v, np.zeros((v.shape[0], n), v.dtype)], axis=1)
        return v

    def rows(self, hf: str, idx) -> np.ndarray:
        """Rows of a projection or the embedding table (model basis), without the rest."""
        gn, kind = self._where[hf]
        if kind not in ("embed", "proj") or self._vrows(hf) is not None or "mlp." in hf:
            return self[hf][np.asarray(idx)]
        t = self.g[gn]
        v = t.dequant(rows=list(idx))
        if kind == "proj" and self.latent:
            return v
        return self.unfold(v) if gn in self.folded | self.inverse else v

    def __iter__(self):
        return iter(self._where)

    def __len__(self) -> int:
        return len(self._where)

    def __contains__(self, hf) -> bool:
        return hf in self._where

    # ------------------------------------------------------------------ ternary parts
    def is_ternary(self, hf: str) -> bool:
        gn, kind = self._where.get(hf, (None, None))
        return kind in ("embed", "proj") and self.g[gn].type in TERNARY

    def ternary(self, hf: str) -> tuple[np.ndarray, np.ndarray, int, bool]:
        """A projection's (or the table's) exact parts, latent basis, HF row order: codes int8
        [N, K] in {-1, 0, +1}, scales fp32 [N, K / g], g, and whether its input takes the
        transform (rotate) first."""
        gn, _ = self._where[hf]
        q, s, g = self.g[gn].ternary()
        q, s = self._order(hf, q), self._order(hf, s) if self._vrows(hf) else s
        if self.rep > 1 and hf.endswith("out_proj.weight") and not self.grouped:
            raise ValueError(f"{hf}: tiled columns cut across scale groups")
        return q, s, g, gn in self.folded or gn in self.inverse
