"""RhinoVLA's Qwen3-VL text phases and its action expert on openTPU (opentpu/llm/rhinovla.py,
docs/rhinovla.md 3): pi0.5's Gemma phases (kernels/pi05.py) with what Qwen3 changes. No new
op: q_norm / k_norm are lib.rmsnorm on the [R, 128] of a head (RSSQ, RSQRT, two MULs), the
causal mask an Additive tile (kernels/attention.py: the scores start as the mask, q.K^T is
accumulated into them), GQA the query heads' KV head h // 2 in _attend_heads, SwiGLU
kernels/pi05.geglu with lib.silu.

  prefix_qkv     RMSNorm(x) w (QACT RSCALE / CSCALE); per query head q -> q_norm -> RoPE ->
                 the Q scratch; per KV head k -> k_norm -> RoPE and v appended to the layer's
                 cache at the row's token. The RoPE tables are per row (the mRoPE positions
                 the host computed: cos / sin [cap, 64] in DRAM).
  prefix_attend  every query head's rows, in groups of MCOLS, against the Np prefix keys of
                 its KV head with the causal mask: group r0's rows r0 .. r0 + G - 1 take rows
                 r0 .. of the host's mask table (-0 where key <= row, -inf past it; a tile
                 whose address is affine in the loop counter), so one hardware loop serves
                 every group and the masked keys weigh exactly 0.
  prefix_out     x += O Wo^T; x += SwiGLU(RMSNorm(x) w2); the deepstack rows are added by
                 qwen3vl_vit.add_rows after layers 0 .. 2.
  expert_step    one Euler step: the state row (host) and x_t -> action_in_proj (+ the mask
                 bias) as rows 0 and 1 .. chunk of the suffix -> 18 expert layers (a hardware
                 loop: each layer's weights and its adaRMS rows of this step at loop-affine
                 addresses, the step a run argument; layer li attends to backbone layer offset
                 + li's cache, its own k / v appended after the prefix's Np rows) -> the final
                 adaRMS -> action_out_proj (its rows pre-multiplied by the action mask, so x_t
                 += dt v_t keeps the dead slots at 0) in fp32 on the VPU, x_t in place.
"""
from __future__ import annotations

from .. import language as ol
from ..compiler import Affine
from .attention import Additive, Blocks, _attend_heads
from .lib import rmsnorm, rope_rows, silu
from .pi05 import _ada, geglu, rms_r
from .vit import row_groups, vit_append


def _qkv(xs, lw, g, r0, n, t0, eps: float) -> None:
    """The q heads (q_norm, RoPE'd, to g.Q[h] rows r0 ..) and each KV head's k (k_norm,
    RoPE'd) and v (appended to g.kv at token t0 ..) of n rows quantized in xs; their RoPE
    rows from g.cos / g.sin at t0."""
    d, Hq, Hk = g.d, g.H, g.Hk
    c = ol.load(g.cos[t0:t0 + n, :])
    s = ol.load(g.sin[t0:t0 + n, :])
    qn, kn = ol.load(lw.qn), ol.load(lw.kn)
    for h in range(Hq):
        q = ol.dot(xs, lw.wq[h * d:(h + 1) * d, :])
        ol.store(g.Q[h][r0:r0 + n, :], rope_rows(rmsnorm(q, qn, eps), c, s))
        del q
    for j in range(Hk):
        k = rope_rows(rmsnorm(ol.dot(xs, lw.wk[j * d:(j + 1) * d, :]), kn, eps), c, s)
        v = ol.dot(xs, lw.wv[j * d:(j + 1) * d, :])
        vit_append(g.kv, j, t0, k, v)
        del k, v


def _attend(g, nq: int, nkeys: int, scale: float, block: int, O, mask=None) -> None:
    """Every query head's rows g.Q[h][0:nq] against the first nkeys keys of its KV head (h //
    (Hq / Hk)), in groups of MCOLS rows; the normalized outputs, heads side by side, to O.
    mask: None (every row sees every key) or (the mask table's address, its row stride in
    words): row r of the table is query row r's additive mask over the keys (-0 / -inf), a
    pad word at the end; each group's blocks take the tile at the group's rows."""
    Hq, Hk, d = g.H, g.Hk, g.d
    G = ol.mxu_columns()
    Gq = Hq // Hk

    def body(r0, n):
        qs = [lambda h=h: ol.load(g.Q[h][r0:r0 + n, :]) for h in range(Hq)]
        o = ol.empty([n, Hq * d])
        if mask is None:
            seq = nkeys
        else:
            base, stride = mask
            items = []
            for t0 in range(0, nkeys, block):
                nb = min(block, nkeys - t0)
                items.append((t0, nb, Additive(Affine(base + 4 * t0) + Affine.of(r0) * (4 * stride),
                                               stride - 1)))
            seq = Blocks(items)

        def emit(i, acc, l):
            o[:, i * d:(i + 1) * d].set(acc * ol.recip(l)[:, None])
        _attend_heads(qs, g.kv, [h // Gq for h in range(Hq)], seq, block, scale, emit=emit)
        ol.store(O[r0:r0 + n, :], o)
    row_groups(nq, G, body)


# =============================================================================== prefix
@ol.jit
def prefix_qkv(g, lw, N: int, R: int, eps: float):
    """Layer lw's RMSNorm, q, k, v of the prefix's N rows (g.X), RoPE at the rows' positions."""
    g1 = ol.load(lw.g1)

    def body(r0, n):
        x = ol.load(g.X[r0:r0 + n, :])
        xs = ol.quantize(x, free=True, rscale=rms_r(x, eps), cscale=g1)
        del x
        _qkv(xs, lw, g, r0, n, r0, eps)
    row_groups(N, R, body)


@ol.jit
def prefix_attend(g, N: int, scale: float, block: int, mask):
    _attend(g, N, N, scale, block, g.O, mask)


@ol.jit
def prefix_out(g, lw, N: int, R: int, eps: float, chunk: int):
    """x += O Wo^T, then x += SwiGLU(RMSNorm(x) w2) for the prefix's N rows."""
    g2 = ol.load(lw.g2)
    C = g.X.shape[1]

    def body(r0, n):
        os = ol.quantize(ol.load(g.O[r0:r0 + n, :]), free=True)
        x = ol.load(g.X[r0:r0 + n, :], out=ol.empty([n, C]))
        ol.dot(os, lw.wo, acc=x)
        del os
        zs = ol.quantize(x, rscale=rms_r(x, eps), cscale=g2)
        geglu(zs, lw.wgu, lw.wd, chunk, acc=x, act=silu)
        ol.store(g.X[r0:r0 + n, :], x)
    row_groups(N, R, body)


# =============================================================================== expert
@ol.jit
def expert_step(e, step, layers: int, Np: int, R: int, eps: float, scale: float,
                block: int, chunk: int, dt: float):
    """One Euler step of the action chunk (e: llm/rhinovla.Image.expert_desc): the step's
    index a run argument (its adaRMS rows' table), x_t^T [A, chunk] in DRAM updated in place.
    The suffix's rows: 0 the state token (e.srow, the host's), 1 .. chunk the actions: x_t
    W_in^T + the mask bias in fp32 on the VPU (A OUTERs), the output A RDOTs of the final
    adaRMS rows against W_out's mask-scaled rows (e.woutf, e.boutT)."""
    T, C = e.X.shape
    A = e.XTt.shape[0]
    g = e.g
    bi = ol.load(e.bin)

    def in_body(r0, n):
        x = ol.empty([n, C])
        x.set(bi[None, :])
        for k in range(A):              # (OUTER takes up to 256 columns: in parts)
            xk = ol.load(e.XTt[k, r0:r0 + n])
            for c0 in range(0, C, 256):
                c1 = min(C, c0 + 256)
                ol.outer(xk, ol.load(e.winT[k, c0:c1]), acc=x[:, c0:c1])
        ol.store(e.X[1 + r0:1 + r0 + n, :], x)
    row_groups(T - 1, R, in_body)
    del bi
    ol.store(e.X[0:1, :], ol.load(e.srow))

    def layer(li):
        lw, m = e.layer(li), e.mods(step, li)
        g.kv = e.kv(li)
        gi, si = ol.load(m.g_in), ol.load(m.s_in)

        def qkv_body(r0, n):
            x = ol.load(e.X[r0:r0 + n, :])
            xs = _ada(x, eps, gi, si)
            del x
            _qkv(xs, lw, g, r0, n, Np + r0, eps)
        row_groups(T, R, qkv_body)
        del gi, si
        _attend(g, T, Np + T, scale, block, e.O)
        ga, gp, sp, gm = ol.load(m.a_in), ol.load(m.g_post), ol.load(m.s_post), \
            ol.load(m.a_post)

        def out_body(r0, n):
            os = ol.quantize(ol.load(e.O[r0:r0 + n, :]), free=True)
            x = ol.load(e.X[r0:r0 + n, :], out=ol.empty([n, C]))
            y = ol.dot(os, lw.wo)
            del os
            x.set(x + y * ga[None, :])
            del y
            zs = _ada(x, eps, gp, sp)
            y = geglu(zs, lw.wgu, lw.wd, chunk, act=silu)
            del zs
            x.set(x + y * gm[None, :])
            ol.store(e.X[r0:r0 + n, :], x)
        row_groups(T, R, out_body)

    if layers == 1:
        layer(0)
    else:
        for li in ol.range(layers):
            layer(li)

    f = e.final(step)
    gf, sf = ol.load(f.g), ol.load(f.s)

    def out32(r0, n):
        x = ol.load(e.X[1 + r0:1 + r0 + n, :])
        y = (x * rms_r(x, eps)[:, None]) * gf[None, :] + sf[None, :]
        del x
        for j in range(A):
            v = ol.sum(y * ol.load(e.woutf[j])[None, :], axis=1)     # one RDOT
            v.set(v + ol.load(e.boutT[j, r0:r0 + n]))
            ol.store(e.Vt[j, r0:r0 + n], v)
            xt = ol.load(e.XTt[j, r0:r0 + n])
            xt.set(xt + v * dt)
            ol.store(e.XTt[j, r0:r0 + n], xt)
    row_groups(T - 1, R, out32)
