"""Speculative decoding with prompt-lookup drafts, its loop on the card (docs/specdec.md).

Decode on a memory-bound card streams every weight once per token; a run of R token rows streams
them once for all R (qwen3_rows: each MM takes the R rows of its stationary operand). The loop
here drafts up to K tokens after the current one from the context itself, verifies them in one
run of K + 1 rows and emits the drafts the model agrees with plus its own next token: greedy,
bit for bit the tokens of plain greedy decode (Engine.generate_card), since every emitted token
is the argmax of a logits row the decode step would compute on the same context (the rows
kernel is bit-identical to decode steps) and a rejected row leaves nothing behind (the KV cache
is positional: the next run appends at the first position not accepted, before anything
attends to it).

The program of an attention bucket and kind, at the run-time position p = t0 + tpos of the
token t (the state block's words, generate.py):

    V (R = K + 1 rows: t at p, the drafts d_1 .. d_K at p + 1 ..), E (R = 1: t alone)
    rows       qwen3_rows over hist[p .. p + R - 1] (RunRows toks_at: the tokens from DRAM, no
               argument registers), one generate.Greedy sink per row: a_0 .. a_(R-1)
    accept     c_j = c_(j-1) * [d_j == a_(j-1)] * [a_(j-1) not a stop id] * [j < tokens left],
               n = sum c_j; a_0 -> out[p + 1], a_j -> out[p + 1 + j] when c_j (LOOP of count c_j)
    commit     tok = a_n, tpos += n + 1, left -= n + 1, every a_j -> hist[p + 1 + j]
    draft      prompt lookup over hist[0 .. p'] (p' = p + n + 1, its token a_n): the latest j < p'
               whose 3, else 2, else 1 tokens up to j equal those up to p' (VOPs over the
               bucket's positions, one ARGMAX), d_i = hist[j + i] -> hist[p' + i], i = 1 .. K
    chain      HALT CHAIN to the next program: the bucket of p' (the next one at a wrap), V when
               a draft was found and the K + 1 rows fit that bucket (and the cache), else E;
               HALT at a stop id, the host's stop word or no tokens left

hist is the context's token ids at their positions (the host writes the prompt and the tokens
fed so far before a run; the card appends the tokens it accepts). The host reads the tokens
from out[] as they land, 1 .. K + 1 an iteration (BoardBackend.run_generate as for the plain
loop). A draft is only a guess: a wrong one costs its row and nothing else, so the drafts the
card proposes past the history (fewer than K tokens after the match) need no care.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from ..compiler import CompileError, Tensor, current
from . import generate as G

# state block words of the loop (the MTP loop's beat, 32..47: not the host's stop word's)
S_ITER, S_ACC, S_VER = 40, 41, 42           # iterations, drafts accepted, V iterations
S_DV = 43                                   # a draft model's KV: the positions it holds
S_TPM1 = 44                                 # tpos - 1 (D2's rows' position)
KINDS = ("V", "E", "D", "D2")
NK = len(KINDS)
V, E, D, D2 = 0, 1, 2, 3
GMAX = 3                                    # the longest n-gram matched
TILED = True                                # V's rows by the tiled kernel where a form fits


def alloc(b, cap: int, block: int, R: int, imem_words: int = 1 << 15,
          waitc: bool = False) -> dict:
    """DRAM of the loop: hist (one fp32 id a position, room for the drafts past the cache's
    end), a ramp 0, 1, ... (the lookup's positions; the host writes it), a program slot per
    bucket and kind (generate.prog_bytes: with its wait entries on a WAITC device) and the chain
    table ([address, instructions] per bucket and kind). R: the verify run's rows (K + 1)."""
    if not 2 <= R <= 16:
        raise ValueError(f"spec_rows {R}: a verify run of 2 .. 16 rows")
    nb, slot = G.buckets(cap, block), G.prog_bytes(imem_words, waitc)
    n = cap + 2 * R + 8
    return {"R": R, "nb": nb, "slot": slot, "n": n, "hist": b.alloc(4 * n),
            "ramp": b.alloc(4 * cap), "progs": b.alloc(slot * NK * nb),
            "ptab": b.alloc(8 * NK * nb)}


def prog_addr(a: dict, blocks: int, kind: int) -> int:
    return a["progs"] + a["slot"] * (NK * (blocks - 1) + kind)


def ptab_addr(a: dict, blocks: int, kind: int = 0) -> int:
    return a["ptab"] + 8 * (NK * (blocks - 1) + kind)


def v_limit(cap: int, block: int, blocks: int, R: int) -> int:
    """The largest tpos at which a V of R rows fits bucket `blocks` (and the cache); < 0: none."""
    t0 = (blocks - 1) * block
    return min(block, cap - t0) - R


def ngram_drafts(hist, k: int, gmax: int = GMAX) -> list:
    """The card's lookup on the host (a run's first drafts): for g = gmax .. 1 the latest
    earlier occurrence of the history's last g tokens, ending at j; the k tokens after it,
    periodically (d_i = hist[j + 1 + (i mod L)], L = len(hist) - 1 - j: what the lookup would
    draft again after each draft, a continuation that runs into the present repeating the
    period). [] when none."""
    hist = list(hist)
    n = len(hist)
    for g in range(gmax, 0, -1):
        if n <= g:
            continue
        tail = hist[n - g:]
        for s in range(n - g - 1, -1, -1):
            if hist[s:s + g] == tail:
                j = s + g - 1
                return [hist[j + 1 + i % (n - 1 - j)] for i in range(k)]
    return []


def _flag(x):
    """1.0 where the small integer x is >= 1, else 0.0."""
    return ol.minimum(ol.maximum(x, 0.0), 1.0)


def _eq(x, t):
    """1.0 where the integer ids x equal t (a [1] tile), else 0.0."""
    return ol.minimum(ol.abs(x - t), 1.0) * -1.0 + 1.0


def _hit(st, tok):
    """1.0 when tok is one of the state block's stop ids."""
    return ol.max(_eq(st[G.S_STOP:G.S_STOP + G.N_STOP], tok))


def _copy_at(b, src, idx, n: int, what: str, at: int = 0):
    """A new [n] tile: the words src.base + at + idx .. + n - 1 (idx a [1] tile holding an
    integer)."""
    r = b.scratch()
    b.rld(r, idx)
    out = b.alloc((n,))
    b.emit(I.vop(I.V_COPY, out.base, src.base + at, 0, 1, n, 0, 0, 0, rb=r, comment=what))
    b.unscratch(r)
    return out


def _spec_gen(m, pos, block: int, R: int, K: int, cap: int, draft: bool = False,
              tile: dict | None = None):
    """The program of one bucket and kind (V: R = K + 1, E: R = 1); see the module. draft: the
    drafts are a draft model's (its D program next, in the bucket of p'), not the lookup's.
    tile: the rows by the tiled kernel in this form (tiled.qwen3_tile: TMEM bounded whatever
    the widths, each K / V stream once for MCOLS / G rows; a whole form, today's bits), else
    qwen3_rows."""
    from .mtp import _args, _load_state
    from .qwen3 import head_rows_chunk, qwen3_rows
    b = current()
    g, a = m.gen, m.sd
    hist = Tensor(a["hist"], (a["n"],), (1,))
    st, at = _load_state(b, g)
    words = {"tpos": st.base + G.S_TPOS}
    b.run_words = words
    if R > 1:                                   # the drafts the rows take (hist[p + 1 ..])
        dv = ol.load(hist[pos.pos + 1:pos.pos + R])
    m.lm_sinks = [G.Greedy(b, m.v_loc, head_rows_chunk(R)) for _ in range(R)]
    gen = m.gen
    m.gen = SimpleNamespace(**{**vars(gen), "out": hist})   # the rows' tokens from hist
    if tile is None:
        qwen3_rows.fn(m, pos, list(range(R)), block)
    else:
        from .tiled import qwen3_tile
        qwen3_tile.fn(m, pos, R, list(range(R)), block, **tile)
    m.gen = gen
    toks = [sk.token() for sk in m.lm_sinks]
    m.lm_sinks = None
    left, tpos = st[G.S_LEFT:G.S_LEFT + 1], st[G.S_TPOS:G.S_TPOS + 1]
    # accept: a_0 always, a_j while the drafts hold, no stop id before it and tokens left
    ol.store(g.out[pos.pos + 1:pos.pos + 2], toks[0])
    stop = _hit(st, toks[0])
    alive = stop * -1.0 + 1.0
    n = ol.zeros([1])
    tok = ol.empty([1])
    tok.set(toks[0])
    c = None
    for j in range(1, R):
        e = _eq(dv[j - 1:j], toks[j - 1]) * alive * _flag(left - float(j))
        if draft and j == K:        # not all K when p' would be the next bucket's first: the
            e = e * _flag(ol.abs(tpos + float(K + 1 - block)))   # draft's K / V of p' - 1 (its
        c = e if c is None else c * e                            # D catches up) in this one
        r = b.scratch()
        b.rld(r, c, comment=f"draft {j} accepted")
        lp = b.begin_loop(0, rcount=r)
        b.unscratch(r)
        ol.store(g.out[pos.pos + 1 + j:pos.pos + 2 + j], toks[j])
        b.end_loop(lp)
        h = _hit(st, toks[j])
        stop = ol.maximum(stop, h * c)
        alive = alive * (h * -1.0 + 1.0)
        tok.set(tok + c * (toks[j] - toks[j - 1]))
        n.set(n + c)
    for j in range(R):                          # the history: every row's token (those past
        ol.store(hist[pos.pos + 1 + j:pos.pos + 2 + j], toks[j])    # the accepted ones are
    st[G.S_TOK:G.S_TOK + 1].set(tok)                                # overwritten later)
    tnew = tpos + n + 1.0                       # p' = t0 + tnew
    left.set(left - n - 1.0)
    st[S_ITER:S_ITER + 1].set(st[S_ITER:S_ITER + 1] + 1.0)
    st[S_ACC:S_ACC + 1].set(st[S_ACC:S_ACC + 1] + n)
    if R > 1:
        st[S_VER:S_VER + 1].set(st[S_VER:S_VER + 1] + 1.0)
    end = ol.maximum(ol.maximum(stop, ol.load(g.state[G.S_HALT:G.S_HALT + 1])),
                     _flag(left * -1.0 + 1.0))
    if draft:                   # the draft model's K / V holds p .. p + min(n, K - 1) (its D
        p0 = tpos + float(pos.t0 + 1)       # fed t, d_1 .. d_(K-1)); E: p (D fed t)
        dvn = p0 if R == 1 else p0 + n - _flag(n - float(K - 1))
        st[S_DV:S_DV + 1].set(dvn)
        catch = _flag(tnew + float(pos.t0) - dvn)       # it lacks p' - 1 (all K accepted): D2
        wrap = _flag(tnew - float(block - 1))
        tpos.set(tnew - wrap * float(block))
        st[S_TPM1:S_TPM1 + 1].set(tpos - 1.0)
        _args(b, at, words)
        ol.store(g.state[:G.S_HALT], st[:G.S_HALT])
        ol.store(g.state[32:48], st[32:48])
        G._chain(b, ptab_addr(a, pos.blocks), cond=end * -1.0 + 1.0,
                 off=(wrap * float(NK) + catch + float(D)) * 8.0, what="the draft")
        b.emit(I.halt())
        return
    # draft: prompt lookup over hist[0 .. p'] (positions below the bucket's end + 1)
    t0 = pos.t0
    N = min(cap, t0 + block)                    # j < p' <= t0 + block
    Lh = N + K + 1
    hp = b.alloc((Lh + 2,))
    hp[0:2].set(-1.0)
    ol.load(hist[0:Lh], out=hp[2:Lh + 2])
    pn = tnew + float(t0)
    tail = _copy_at(b, hp, pn, 3, "the last three tokens")      # h[p' - 2], h[p' - 1], h[p']
    ramp = ol.load(Tensor(a["ramp"], (cap,), (1,))[0:N])
    e1 = _eq(hp[2:N + 2], tail[2:3])
    e2 = e1 * _eq(hp[1:N + 1], tail[1:2])
    score = e1 * (ramp + 1.0)
    if GMAX >= 3:
        e3 = e2 * _eq(hp[0:N], tail[0:1])
        e2 = e2 + e3
    score = (score + e2 * float(cap + 1)) * _flag(ramp * -1.0 + pn)   # j < p'
    del e1, e2
    best = b.argmax(score)                      # [score, j]: the latest of the longest
    found = _flag(best[0:1])
    # d_i = h[j + 1 + (i mod L)], L = p' - j (hp[j + 3 + o] = h[j + 1 + o]): i - L * #{k >= 1:
    # i >= k L}, i < K
    jj = best[1:2]
    L = jj * -1.0 + pn
    rk = ramp[0:K]
    cnt = ol.zeros([K])
    for k in range(1, K):
        cnt.set(cnt + _flag(rk - L * float(k) + 1.0))
    idx = rk - L * cnt + jj
    d = b.alloc((K,))
    r = b.scratch()
    for i in range(K):
        b.rld(r, idx[i:i + 1])
        b.emit(I.vop(I.V_COPY, d.base + i, hp.base + 3, 0, 1, 1, 0, 0, 0, rb=r,
                     comment=f"draft {i + 1}"))
    b.unscratch(r)
    rb = b.scratch()
    b.rld(rb, pn * 4.0, comment="p' bytes")
    b.emit(I.st(a["hist"] + 4, d.base, K, ra=rb, comment="the drafts -> hist[p' + 1 ..]"))
    b.unscratch(rb)
    # the next program: its bucket (a wrap past the last position), V or E
    wrap = _flag(tnew - float(block - 1))
    tw = tnew - wrap * float(block)
    tpos.set(tw)
    blocks = pos.blocks
    lc, ln = (v_limit(cap, block, blocks, K + 1), v_limit(cap, block, blocks + 1, K + 1))
    lim = wrap * float(ln - lc) + float(lc)
    use_e = ol.maximum(found * -1.0 + 1.0, _flag(tw - lim))
    off = (wrap * float(NK) + use_e) * 8.0
    _args(b, at, words)
    ol.store(g.state[:G.S_HALT], st[:G.S_HALT])
    ol.store(g.state[32:48], st[32:48])
    G._chain(b, ptab_addr(a, blocks), cond=end * -1.0 + 1.0, off=off, what="the next iteration")
    b.emit(I.halt())


def _draft_gen(m, md, pos, block: int, K: int, cap: int):
    """D: the draft model's decode steps at p' = t0 + tpos (the state's), a hardware loop of
    one-row steps (its resident decode step, qwen3_step at a RunPos of its own image md) over
    q = dv .. : the token hist[q] (positions dv .. p' - 1 it has not taken yet, then p'), then its
    drafts, each step's token to hist[q + 1] from q = p' on; up to p' + K - 1, within the bucket.
    Then V (K drafts and their rows fit the bucket) or E."""
    from .qwen3 import qwen3_step
    b = current()
    g, a = m.gen, m.sd
    hist = Tensor(a["hist"], (a["n"],), (1,))
    t0 = pos.t0
    st = ol.load(g.state)
    tw = st[G.S_TPOS:G.S_TPOS + 1]
    pn = tw + float(t0)
    dv = st[S_DV:S_DV + 1]
    var = b.alloc((4,))                         # the step's run-time words: tok, tpos, ring
    var.set(0.0)
    var[1:2].set(dv - float(t0))
    r = b.scratch()
    b.rld(r, dv * 4.0, comment="dv's bytes")
    b.emit(I.ld(a["hist"], var.base, 1, ra=r, comment="hist[dv]"))
    b.unscratch(r)
    count = ol.minimum(pn - dv + float(K), var[1:2] * -1.0 + float(block))
    r = b.scratch()
    b.rld(r, count, comment="draft steps")
    loop = b.begin_loop(0, rcount=r)
    b.unscratch(r)
    body = b.stack[-1]
    mark = len(body)
    sink = G.Greedy(b, md.v_loc, min(8192, b.cfg.TMEM_WORDS // 8))
    md.lm_sink, md.lm_keep = sink, False
    b.run_words = {name: var.base + slot for name, slot in G.SLOT.items()}
    qwen3_step.fn(m=md, pos=pos, block=block)
    d = sink.token()
    q = var[1:2] + float(t0)
    c = _flag(q - pn + 1.0)                     # q >= p': a draft, to hist[q + 1]
    rq = b.scratch()
    b.rld(rq, q * 4.0, comment="q's bytes")
    r = b.scratch()
    b.rld(r, c)
    lp = b.begin_loop(0, rcount=r)
    b.unscratch(r)
    b.emit(I.st(a["hist"] + 4, d.base, 1, ra=rq, comment="the draft -> hist[q + 1]"))
    b.end_loop(lp)
    nx = b.alloc((1,))
    b.emit(I.ld(a["hist"] + 4, nx.base, 1, ra=rq, comment="hist[q + 1]"))
    b.unscratch(rq)
    sel = _flag(pn - q)                         # q + 1 <= p': the context's token
    var[0:1].set(d + sel * (nx - d))
    var[1:2].set(var[1:2] + 1.0)
    G._arguments(b, body, mark, var, md.spec, block)
    b.end_loop(loop)
    md.lm_sink = None
    st[S_DV:S_DV + 1].set(pn + 1.0)
    ol.store(g.state[32:48], st[32:48])
    use_e = _flag(tw - float(v_limit(cap, block, pos.blocks, K + 1)))
    G._chain(b, ptab_addr(a, pos.blocks), off=use_e * 8.0, what="the verify")
    b.emit(I.halt())


def _draft2_gen(m, md, pos, block: int, K: int, cap: int):
    """D2, before a D whose draft model lacks p' - 1 (the verify accepted all K drafts, the last
    of which it has not taken): its rows p' - 1, p' in one run (hist's tokens: d_K, a_K; one
    weight stream), d_1 from row p' to hist[p' + 1]; then D from p' + 1 (dv)."""
    from .mtp import _args, _load_state
    from .qwen3 import head_rows_chunk, qwen3_rows
    b = current()
    g, a = m.gen, m.sd
    st, at = _load_state(b, g)
    words = {"tpos": st.base + S_TPM1}
    b.run_words = words
    sink = G.Greedy(b, md.v_loc, head_rows_chunk(1))
    md.lm_sinks = [sink]
    gen = md.gen
    md.gen = SimpleNamespace(**{**vars(gen), "out": Tensor(a["hist"], (a["n"],), (1,))})
    qwen3_rows.fn(md, pos, [1], block)
    md.gen, md.lm_sinks = gen, None
    d = sink.token()
    pn = st[G.S_TPOS:G.S_TPOS + 1] + float(pos.t0)
    r = b.scratch()
    b.rld(r, pn * 4.0, comment="p' bytes")
    b.emit(I.st(a["hist"] + 4, d.base, 1, ra=r, comment="d_1 -> hist[p' + 1]"))
    b.unscratch(r)
    st[S_DV:S_DV + 1].set(pn + 1.0)
    _args(b, at, words)
    ol.store(g.state[32:48], st[32:48])
    G._chain(b, ptab_addr(a, pos.blocks, D), what="the draft's steps")
    b.emit(I.halt())


def compile_gen(image, blocks: int, kind: int, block: int, draft=None) -> list | None:
    """Program KINDS[kind] of bucket `blocks`, one per slice; None when the bucket has no
    position for it (V: K + 1 rows past the cache's end; D without a draft model). draft: the
    draft model's image (draft_engine's; D its decode steps, V and E chain to D)."""
    from .qwen3 import RunPos, RunRows
    a = image.lookup.get("sd")
    if a is None:
        raise ValueError("speculative decoding needs an image with spec_rows (Engine spec_rows)")
    Rv = a["R"]
    t0 = (blocks - 1) * block
    if kind in (D, D2):
        if draft is None:
            return None
        if draft.cap != image.cap or draft.cfg.S != 1:
            raise ValueError("a draft model's image of the target's capacity, one slice")
        rp = RunPos(blocks, block, t0, draft.lookup["zmask"], draft.cap) if kind == D else \
            RunRows(blocks, block, t0, draft.lookup["zmask"], draft.cap, 2, toks_at=0)
        progs = []
        for s in range(image.cfg.S):
            m = image.descriptors(s)
            m.sd = a
            progs.append(ol.jit(_draft_gen if kind == D else _draft2_gen).trace(
                image.cfg, s, {"m": m, "md": draft.descriptors(s), "pos": rp, "block": block,
                               "K": Rv - 1, "cap": image.cap}).finish())
        return progs
    R = Rv if kind == V else 1
    if v_limit(image.cap, block, blocks, R) < 0:
        return None
    if kind == V and TILED and type(image).__module__.endswith(".qwen3") and image.cfg.S == 1 \
            and R <= image.cfg.act_rows:
        from . import tiled as TL
        from .qwen3 import _amask
        rp = RunRows(blocks, block, t0, image.lookup["zmask"], image.cap, R, 0,
                     _amask(image.lookup))

        def trace(kw, fit):
            out = []
            for s in range(image.cfg.S):
                m = image.descriptors(s)
                m.sd = a
                out.append(ol.jit(_spec_gen).trace(image.cfg, s, {
                    "m": m, "pos": rp, "block": block, "R": R, "K": Rv - 1, "cap": image.cap,
                    "draft": draft is not None, "tile": kw}, fit=fit))
            return out
        try:
            return TL._first_fit(trace, TL.variants("qwen3", R, image.cfg.MCOLS), image.cfg,
                                 levels=True)[0]
        except CompileError:
            pass                    # (today's rows kernel)
    rp = RunRows(blocks, block, t0, image.lookup["zmask"], image.cap, R, toks_at=0)
    progs = []
    for s in range(image.cfg.S):
        m = image.descriptors(s)
        m.sd = a
        progs.append(ol.jit(_spec_gen).trace(image.cfg, s, {
            "m": m, "pos": rp, "block": block, "R": R, "K": Rv - 1,
            "cap": image.cap, "draft": draft is not None}).finish())
    return progs


@dataclass
class SpecStats:
    """One generate_card: the tokens, the device's iterations, drafts accepted, V iterations."""
    tokens: list = field(default_factory=list)
    iterations: int = 0
    accepted: int = 0
    verifies: int = 0
    compile_s: float = 0.0
    run: dict | None = None


class SubBackend:
    """A draft model's backend: the target's (one device, one DRAM), its image [base, end)
    written there once (the target's image below it is not touched); runs and reads go to the
    target's backend."""

    def __init__(self, parent, images, base: int):
        self.parent, self.base = parent, base
        for s, img in enumerate(images):
            parent.write(s, base, img[base:])

    def __getattr__(self, name):
        return getattr(self.parent, name)


def draft_engine(target, model, wformat: str = "int4", head_format: str | None = None,
                 W=None, spec=None, **kw):
    """The draft model's Engine on the target's device: its image at the top of the target's
    DRAM (above the target's image, its program area and kvio's staging area, which must fit
    below: else MemoryError), its vocabulary cut to the target's (a Qwen3-0.6B's padding rows
    past the tokenizer's 151669 ids: the drafts are target ids), resident (its decode step is
    the D program's), its prompts in prompt runs (its context follows the target's:
    SpecDecoder). model: a checkpoint directory (or W and spec)."""
    from dataclasses import replace

    from . import load_spec, model_dir
    from .qwen3 import PREFILL_ROWS, Engine, load_weights
    if spec is None:
        path = model_dir(model)
        spec, W = load_spec(path), load_weights(path)
    V = target.spec.vocab
    if spec.vocab < V or spec.hidden % target.cfg.D:
        raise ValueError(f"a draft model of vocabulary {spec.vocab} for one of {V}")
    if spec.vocab > V:
        W = dict(W)
        for k in ("model.embed_tokens.weight", "lm_head.weight"):
            if k in W:
                W[k] = np.asarray(W[k])[:V]
        spec = replace(spec, vocab=V)
    cfg = target.cfg
    rows = kw.pop("rows", PREFILL_ROWS)
    size = spec.image(cfg, target.cap, 1, rows, wformat=wformat, head_format=head_format,
                      lookup=True).nbytes
    from .. import waits as WT
    tend = max(target.image.nbytes, (target.bat or {}).get("end", 0))   # (its batch area)
    staging = STAGING if getattr(target.backend, "prog_at", None) is not None else 0
    end = -(-tend // 4096) * 4096 + WT.area(cfg.IMEM_WORDS, cfg.SEQ_WC) + staging
    base = (cfg.DRAM_BYTES - size - WT.area(cfg.IMEM_WORDS, cfg.SEQ_WC)) // (1 << 20) * (1 << 20)
    if base < end:
        raise MemoryError(f"the draft model's image ({size / 2**20:.0f} MiB) does not fit above "
                          f"the target's ({end / 2**20:.0f} MiB with its programs and staging "
                          f"area) in {cfg.DRAM_BYTES / 2**20:.0f} MiB")
    kw.setdefault("prompt_runs", True)
    return Engine(spec, W, cap=target.cap, cfg=cfg, backend=lambda c, imgs: SubBackend(
        target.backend, imgs, base), block=target.block, rows=rows, wformat=wformat,
        head_format=head_format, resident=True, image_base=base, **kw)


STAGING = 256 << 20     # kvio's staging area past the target's programs (kvio.STAGING_MAX; a
                        # backend with a program area: the card's)


class SpecDecoder:
    """Greedy speculative decoding on an Engine made with spec_rows (K + 1 = spec_rows: up to K
    drafts an iteration), its loop on the card: the drafts by prompt lookup, or with `draft` (a
    draft_engine on the same device) the draft model's. generate_card is Engine.generate_card's
    greedy form, the same tokens."""

    def __init__(self, engine, draft=None):
        img = engine.image
        a = getattr(img, "lookup", {}).get("sd")
        if a is None:
            raise ValueError("SpecDecoder needs an Engine made with spec_rows")
        if engine.cfg.S != 1:
            raise ValueError("speculative decoding runs on one slice")
        if not engine.can_generate:
            raise ValueError("speculative decoding runs the loop on the device: a resident "
                             "engine on a backend that runs the generate loop")
        if draft is not None and (getattr(draft.backend, "parent", None) is not engine.backend
                                  or draft.block != engine.block):
            raise ValueError("the draft model's Engine runs on the target's device "
                             "(draft_engine)")
        self.eng, self.img, self.a, self.draft = engine, img, a, draft
        self.K = a["R"] - 1
        self.last: SpecStats | None = None
        self.dctx: list = []            # the tokens the draft model's K / V holds
        self._done = engine.__dict__.setdefault(
            "_sd_gen" if draft is None else "_sd_gen_draft", {})
        ramp = np.arange(img.cap, dtype=np.float32)
        for s in range(img.cfg.S):
            engine.backend.write(s, a["ramp"], ramp)

    def _programs(self, b0: int, b1: int, st: SpecStats) -> None:
        """Buckets b0 .. b1's programs in the chain area and their table entries (once per
        engine and drafter)."""
        eng, img, a = self.eng, self.img, self.a
        dimg = None if self.draft is None else self.draft.image
        for blk in range(b0, b1 + 1):
            if blk in self._done:
                continue
            t0 = time.perf_counter()
            tab = np.zeros(2 * NK, np.uint32)
            progs = {}
            for k in range(NK):
                what = ("specgen", blk, k, eng.block) + (
                    () if dimg is None else ("draft", self.draft.layout, dimg.base))
                p = eng.cached(what, lambda k=k: (compile_gen(img, blk, k, eng.block, dimg),
                                                  None))[0]
                if p is None:
                    continue
                if not G.fits(img, p):
                    raise CompileError(f"speculative loop program {KINDS[k]} of bucket {blk}: "
                                       f"{max(map(len, p))} instructions, IMEM "
                                       f"{img.cfg.IMEM_WORDS // 8}")
                prep = getattr(eng.backend, "prepare", None)
                if prep is not None:
                    prep(p)
                for s, prog in enumerate(p):
                    eng.backend.write(s, prog_addr(a, blk, k), G.slot_words(eng.backend, prog))
                tab[2 * k], tab[2 * k + 1] = prog_addr(a, blk, k), len(I.assemble(p[0])) // 8
                progs[k] = p
            for s in range(img.cfg.S):
                eng.backend.write(s, ptab_addr(a, blk), tab)
            self._done[blk] = progs
            st.compile_s += time.perf_counter() - t0

    def warm(self) -> None:
        """Every bucket's programs compiled and in the chain area now (a server's start: none
        compiles during a request's decode)."""
        self._programs(1, self.a["nb"], SpecStats())

    def _sync_draft(self, ctx: list, p: int) -> None:
        """The draft model's K / V to positions 0 .. p - 1 of ctx: back to the longest prefix it
        holds, the rest fed (its prefill)."""
        dr = self.draft
        c = 0
        for x, y in zip(self.dctx, ctx[:p]):
            if x != y:
                break
            c += 1
        c = min(c, dr.pos, p)
        if c < dr.pos:
            dr.rollback(c)
        if c < p:
            dr.prefill(ctx[c:p])
        self.dctx = list(ctx[:p])
        if dr.pos != p:
            raise RuntimeError(f"the draft model is at {dr.pos}, the context at {p}")

    def generate_card(self, tok: int, n: int, stop_ids=None, on_token=None, stop=None,
                      context=()) -> list:
        """Feed `tok` at Engine.pos and generate up to n tokens after it, greedy, on the device
        (the module's loop): the tokens of Engine.generate_card(tok, n, stop_ids) bit for bit.
        context: the ids of positions 0 .. Engine.pos (the last one tok; a shorter tail is
        aligned at the end), which the drafts are looked up in (a draft model's context: all
        of it). on_token(t) for each token as the host reads it; stop() (polled while the card
        runs) halts the loop after the iteration in flight. Returns the tokens; Engine.pos is
        then the position of the last one (fed next, unless it is a stop id). self.last: the
        run's SpecStats."""
        eng, img, a = self.eng, self.img, self.a
        K, block, cap = self.K, eng.block, img.cap
        ids = list(img.spec.eos if stop_ids is None else stop_ids)
        st = SpecStats()
        self.last = st
        p = eng.pos
        n = min(n, cap - p - 1)
        if n <= 0:
            return []
        ctx = [int(t) for t in context][-(p + 1):]
        if not ctx or ctx[-1] != int(tok):
            ctx = (ctx + [int(tok)])[-(p + 1):]
        if self.draft is not None and len(ctx) != p + 1:
            raise ValueError(f"a draft model needs the whole context: {len(ctx)} ids for "
                             f"position {p}")
        hist = np.zeros(p + 1 + K, np.float32)
        hist[p + 1 - len(ctx):p + 1] = ctx
        b0, b1 = p // block + 1, (p + n - 1) // block + 1
        self._programs(b0, b1, st)
        words = G.state_words(img.spec, tok, p, n, ids, block)
        if self.draft is None:
            d = ngram_drafts(ctx, K)
            hist[p + 1:p + 1 + len(d)] = d
            kind = V if d and p % block <= v_limit(cap, block, b0, K + 1) and \
                V in self._done[b0] else E
        else:                           # the run starts with the draft's steps at p
            t0 = time.perf_counter()
            self._sync_draft(ctx, p)
            st.compile_s += time.perf_counter() - t0
            words[S_DV] = p
            kind = D
        g = img.lookup["gen"]
        for s in range(img.cfg.S):
            eng.backend.write(s, a["hist"], hist)
            eng.backend.write(s, g["state"], words)
            eng.backend.write(s, g["out"] + 4 * (p + 1), np.full(n, G.OUT_MARK, np.uint32))
        first = self._done[b0][kind]
        run = getattr(eng.backend, "run_generate", None)
        if run is not None:
            stats, got = run(first, g["out"] + 4 * (p + 1), n, on_token, stop, g["state"])
        else:
            stats = eng.backend.run(first)
            w = eng.backend.read(0, g["out"] + 4 * (p + 1), 4 * n).view(np.uint32)
            k = int(np.argmax(w == G.OUT_MARK)) if (w == G.OUT_MARK).any() else n
            got = [int(x) for x in w[:k].view(np.float32)]
            if on_token is not None:
                for t in got:
                    on_token(t)
        eng.stats.append(stats)
        sw = eng.backend.read(0, g["state"], 4 * G.STATE_WORDS).view(np.float32)
        st.iterations, st.accepted, st.verifies = int(sw[S_ITER]), int(sw[S_ACC]), int(sw[S_VER])
        st.tokens, st.run = list(got), stats
        eng.pos = p + len(got)
        if self.draft is not None:      # what the draft's K / V holds (its D's, V's word)
            dv = min(int(sw[S_DV]), eng.pos)
            self.draft.pos = dv
            self.dctx = (ctx + list(got))[:dv]
        return got
