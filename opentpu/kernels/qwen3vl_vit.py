"""Qwen3-VL's vision tower on openTPU (opentpu/llm/rhinovla.py, docs/rhinovla.md 3): SigLIP's
phases (kernels/vit.py) with what Qwen3-VL adds. No new op.

  * 2D RoPE on q and k (vit_qkv_rope): the head's 64 columns rotated (rotate-half) with a
    per-row cos / sin table [N, 32] the host builds from the token's grid row and column
    (llm/rhinovla.vit_rope_table), the same table for every view and layer: four VOPs a half
    on the [R, 64] of each head before the Q store / the K append (V is not rotated).
  * The merger's GELU is nn.GELU(), the erf one (gelu_erf): 0.5 x + 0.5 |x| erf(|x| / sqrt
    2) with Abramowitz-Stegun 7.1.26 for erf (|error| < 1.5e-7; |x| erf(|x|) is even, so no
    sign), 19 VOPs; the blocks' MLP stays gelu_pytorch_tanh (lib.gelu_tanh, 8 VOPs).
  * The main merger LayerNorms each token (1024) before the 2 x 2 shuffle: the last block's
    rows are stored LayerNorm'd (vit_out_mlp's `post`), then vit_merge runs without its own
    norm (norm=False); the deepstack mergers LayerNorm the [4C] window row (vit_merge as it
    is), reading the residual rows after blocks 5 / 11 / 17 into their own destination.
  * The tokens are in the processor's 2 x 2 window order already (llm/rhinovla.patches), so
    a window is 4 consecutive rows, as vit_merge wants.
"""
from __future__ import annotations

import math

from .. import language as ol
from . import pieces as PC
from .lib import rope_rows
from .vit import layernorm_q, row_groups, vit_append

# erf's Abramowitz-Stegun 7.1.26 coefficients
_P = 0.3275911
_A1, _A2, _A3, _A4, _A5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
_RS2 = 1.0 / math.sqrt(2.0)


def gelu_erf(x):
    """nn.GELU(): 0.5 x (1 + erf(x / sqrt 2)) = 0.5 x + 0.5 |x| erf(|x| / sqrt 2); erf(a) =
    1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) exp(-a^2), t = 1 / (1 + p a), a >= 0
    (|error| < 1.5e-7): 19 VOPs."""
    ax = ol.abs(x)
    a = ax * _RS2
    t = ol.recip(a * _P + 1.0)
    poly = ((((t * _A5 + _A4) * t + _A3) * t + _A2) * t + _A1) * t
    e = ol.exp2((a * a) * -ol.LOG2E)
    erf = (poly * e) * -1.0 + 1.0
    return x * 0.5 + (ax * erf) * 0.5


@ol.jit
def vit_qkv_rope(v, lw, src, N: int, R: int, eps: float):
    """vit.vit_qkv with Qwen3-VL's 2D RoPE: LayerNorm (ln1) of src's N rows, then per head q
    (rotated) -> v.Q[h], k (rotated) and v -> the KV scratch; the biases (the LayerNorm's
    folded in) start each [R, dp] tile. v.rc / v.rs: the cos / sin table [N, dv / 2]."""
    H, dp, dh = v.heads, v.dp, v.dv
    lw = PC.bind(lw)
    w = ol.load(lw.ln1w)
    bs = ol.load(lw.bq), ol.load(lw.bk), ol.load(lw.bv)
    ws = lw.wq, lw.wk, lw.wv
    tiles = [[ol.zeros([R, dp]) for _ in range(3)] for _ in range(2)]

    def body(r0, n):
        x = ol.load(src[r0:r0 + n, :])
        xs = layernorm_q(x, w, eps, inplace=True)
        del x
        c = ol.load(v.rc[r0:r0 + n, :])
        s = ol.load(v.rs[r0:r0 + n, :])
        for h in range(H):
            q, k, vv = (t[0:n, :] for t in tiles[h % 2])
            for t, wt, b in zip((q, k, vv), ws, bs):
                t[:, 0:dh].set(b[h * dh:(h + 1) * dh][None, :])
                ol.dot(xs, wt[h * dh:(h + 1) * dh, :], acc=t[:, 0:dh])
            q[:, 0:dh].set(rope_rows(q[:, 0:dh], c, s))
            k[:, 0:dh].set(rope_rows(k[:, 0:dh], c, s))
            ol.store(v.Q[h][r0:r0 + n, :], q)
            vit_append(v.kv, h, r0, k, vv)
    row_groups(N, R, body)


@ol.jit
def add_rows(dsts, srcs, N: int, R: int):
    """dst rows += src rows for each (dst, src) pair: the deepstack rows of every view into
    the prefix's image rows (one program a deepstack layer, the layer a run argument in the
    sources' addresses)."""
    for dst, src in zip(dsts, srcs):
        def body(r0, n, dst=dst, src=src):
            x = ol.load(dst[r0:r0 + n, :])
            x.set(x + ol.load(src[r0:r0 + n, :]))
            ol.store(dst[r0:r0 + n, :], x)
        row_groups(N, R, body)
