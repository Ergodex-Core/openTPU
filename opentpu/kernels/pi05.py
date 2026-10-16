"""pi0.5's Gemma phases on openTPU (opentpu/llm/pi05.py, docs/pi05.md): the 2B prefix over
its valid rows, and one flow-matching step of the 300M action expert. No new op: RMSNorm is
RSSQ and RSQRT with QACT's row and column scales, adaRMS a VPU pass more (its shift), RoPE
four VOPs a half, GeGLU gelu_tanh (8 VOPs) and a MUL, attention kernels/attention.py's.

Every row runs in groups of R (vit.row_groups: a hardware loop over the whole groups, the
rest at a static start), so a program's size does not grow with the rows. Attention is
bidirectional everywhere (no mask: the prefix's pads are not on the card), so it is
vit_attend's loop over groups of MCOLS query rows against a length of keys, with the 8 query
heads all on the one KV head.

  prefix_qkv     RMSNorm(x)(1 + w) (QACT RSCALE / CSCALE), per head q -> RoPE -> the Q
                 scratch, k -> RoPE and v appended to the layer's KV cache at the row's token
  prefix_attend  every head's queries against the Np prefix keys, out rows to the O scratch
  prefix_out     x += O Wo^T; x += GeGLU(RMSNorm(x)(1 + w2))
  expert_step    one Euler step: x_t (in DRAM) -> action_in_proj -> 18 expert layers (a
                 hardware loop over the layers: their weights and their adaRMS rows of this
                 step at loop-affine addresses, the step a run argument) -> the final adaRMS,
                 action_out_proj -> x_t += dt v_t in place. Each layer: adaRMS (scale, shift
                 from the step's table) -> q, k, v, RoPE at tokens Np .., k and v appended to
                 the layer's prefix cache after its Np rows; attention over Np + chunk keys;
                 x += (O Wo^T) gate; adaRMS; x += GeGLU gate2.

The GeGLU's gate and up rows are interleaved by chunks at build time (chunk c's gate rows,
then its up rows): one MM gives a chunk's [R, 2 chunk] (gate | up).
"""
from __future__ import annotations

from .. import language as ol
from ..compiler import CompileError
from .attention import _attend_heads
from .lib import gelu_tanh, rope_rows
from .vit import row_groups, vit_append


def rms_r(x, eps: float):
    """1 / sqrt(mean(x^2) + eps) of each row of x [n, C] (RSSQ, one MUL-ADD, RSQRT)."""
    return ol.rsqrt(ol.sumsq(x) * (1.0 / x.cols) + eps)


def geglu(zs, wgu, wd, chunk: int, acc=None):
    """(gelu_tanh(z Wg^T) * (z Wu^T)) Wd^T for z quantized in zs, F in chunks: chunk c + 1's
    gate | up rows stream while the VPU works on chunk c, whose product (quantized on its own)
    multiplies Wd's columns of the chunk, added into `acc` (the residual) or a new tile."""
    F = wd.shape[1]
    if F % chunk or wgu.shape[0] != 2 * F:
        raise CompileError(f"GeGLU width {F} in chunks of {chunk}, gate | up rows {wgu.shape}")
    starts = list(range(0, F, chunk))

    def up(c0):
        return ol.dot(zs, wgu[2 * c0:2 * c0 + 2 * chunk, :])

    nxt = up(starts[0])
    for i, c0 in enumerate(starts):
        h = nxt
        if i + 1 < len(starts):
            nxt = up(starts[i + 1])
        a = gelu_tanh(h[:, 0:chunk]) * h[:, chunk:2 * chunk]
        del h
        if acc is None:
            acc = ol.dot(a, wd[:, c0:c0 + chunk])
        else:
            ol.dot(a, wd[:, c0:c0 + chunk], acc=acc)
    return acc


def _qkv(xs, lw, g, r0, n, t0) -> None:
    """The q heads (RoPE'd, to g.Q[h] rows r0 ..) and k, v (k RoPE'd; appended to g.kv at
    token t0 ..) of n rows quantized in xs; their RoPE rows from g.cos / g.sin at t0."""
    d, H = g.d, g.H
    c = ol.load(g.cos[t0:t0 + n, :])
    s = ol.load(g.sin[t0:t0 + n, :])
    for h in range(H):
        q = ol.dot(xs, lw.wq[h * d:(h + 1) * d, :])
        ol.store(g.Q[h][r0:r0 + n, :], rope_rows(q, c, s))
        del q
    k = rope_rows(ol.dot(xs, lw.wk), c, s)
    v = ol.dot(xs, lw.wv)
    vit_append(g.kv, 0, t0, k, v)


def _attend(g, nq: int, nkeys: int, scale: float, block: int, O) -> None:
    """Every head's queries g.Q[h][0:nq] against the first nkeys keys of g.kv (the one KV
    head), in groups of MCOLS rows; the normalized outputs, heads side by side, to O."""
    H, d = g.H, g.d
    G = ol.mxu_columns()

    def body(r0, n):
        qs = [lambda h=h: ol.load(g.Q[h][r0:r0 + n, :]) for h in range(H)]
        o = ol.empty([n, H * d])

        def emit(i, acc, l):
            o[:, i * d:(i + 1) * d].set(acc * ol.recip(l)[:, None])
        _attend_heads(qs, g.kv, [0] * H, nkeys, block, scale, emit=emit)
        ol.store(O[r0:r0 + n, :], o)
    row_groups(nq, G, body)


# =============================================================================== prefix
@ol.jit
def prefix_qkv(g, lw, N: int, R: int, eps: float):
    """Layer lw's RMSNorm, q, k, v of the prefix's N rows (g.X), RoPE at positions 0 .. N-1."""
    g1 = ol.load(lw.g1)

    def body(r0, n):
        x = ol.load(g.X[r0:r0 + n, :])
        xs = ol.quantize(x, free=True, rscale=rms_r(x, eps), cscale=g1)
        del x
        _qkv(xs, lw, g, r0, n, r0)
    row_groups(N, R, body)


@ol.jit
def prefix_attend(g, N: int, scale: float, block: int):
    _attend(g, N, N, scale, block, g.O)


@ol.jit
def prefix_out(g, lw, N: int, R: int, eps: float, chunk: int):
    """x += O Wo^T, then x += GeGLU(RMSNorm(x)(1 + w2)) for the prefix's N rows."""
    g2 = ol.load(lw.g2)
    C = g.X.shape[1]

    def body(r0, n):
        os = ol.quantize(ol.load(g.O[r0:r0 + n, :]), free=True)
        x = ol.load(g.X[r0:r0 + n, :], out=ol.empty([n, C]))
        ol.dot(os, lw.wo, acc=x)
        del os
        zs = ol.quantize(x, rscale=rms_r(x, eps), cscale=g2)
        geglu(zs, lw.wgu, lw.wd, chunk, acc=x)
        ol.store(g.X[r0:r0 + n, :], x)
    row_groups(N, R, body)


# =============================================================================== expert
def _ada(x, eps, gain, shift):
    """adaRMS's normalized rows x rsqrt(mean x^2 + eps) (1 + scale) + shift, quantized (gain
    = 1 + scale, precomputed per step on the host)."""
    t = (x * rms_r(x, eps)[:, None]) * gain[None, :] + shift[None, :]
    return ol.quantize(t, free=True)


@ol.jit
def expert_step(e, step, layers: int, Np: int, R: int, eps: float, scale: float,
                block: int, chunk: int, dt: float):
    """One Euler step of the action chunk (e: llm/pi05.Image.expert_desc): the step's index a
    run argument (its adaRMS rows' table), x_t [T, A] in DRAM updated in place."""
    T, C = e.X.shape
    g = e.g

    bi = ol.load(e.bin)

    def in_body(r0, n):
        xs = ol.quantize(ol.load(e.XT[r0:r0 + n, :]), free=True)
        x = ol.empty([n, C])
        x.set(bi[None, :])
        ol.dot(xs, e.win, acc=x)
        ol.store(e.X[r0:r0 + n, :], x)
    row_groups(T, R, in_body)
    del bi

    def layer(li):
        lw, m = e.layer(li), e.mods(step, li)
        g.kv = e.kv(li)
        gi, si = ol.load(m.g_in), ol.load(m.s_in)

        def qkv_body(r0, n):
            x = ol.load(e.X[r0:r0 + n, :])
            xs = _ada(x, eps, gi, si)
            del x
            _qkv(xs, lw, g, r0, n, Np + r0)
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
            y = geglu(zs, lw.wgu, lw.wd, chunk)
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
    gf, sf, bo = ol.load(f.g), ol.load(f.s), ol.load(e.bout)
    A = e.wout.shape[0]

    def out_body(r0, n):
        x = ol.load(e.X[r0:r0 + n, :])
        ys = _ada(x, eps, gf, sf)
        del x
        v = ol.empty([n, A])
        v.set(bo[None, :])
        ol.dot(ys, e.wout, acc=v)
        ol.store(e.V[r0:r0 + n, :], v)
        xt = ol.load(e.XT[r0:r0 + n, 0:A])
        xt.set(xt + v * dt)
        ol.store(e.XT[r0:r0 + n, 0:A], xt)
    row_groups(T, R, out_body)


@ol.jit
def project_rows(src, w, b, dst, N: int, R: int):
    """dst rows = src rows W^T + b (PaliGemma's projector: the tower's post-LayerNorm rows
    into the prefix's image rows)."""
    bv = ol.load(b)

    def body(r0, n):
        xs = ol.quantize(ol.load(src[r0:r0 + n, :]), free=True)
        y = ol.empty([n, w.shape[0]])
        y.set(bv[None, :])                  # the bias, then the MM accumulates into it
        ol.dot(xs, w, acc=y)
        ol.store(dst[r0:r0 + n, :], y)
    row_groups(N, R, body)
