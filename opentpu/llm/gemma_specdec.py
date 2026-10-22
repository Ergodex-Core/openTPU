"""Greedy Gemma 3 prompt-lookup decoding on the resident device.

One E program processes the current token.  It then proposes one token by
finding the latest earlier occurrence of that token in the committed history.
A V program processes the current token and that proposal in two rows.  Only
model-selected tokens are published in generate.out; proposals stay in hist.
Both programs use the existing RLD, ARGMAX, VOP, and HALT CHAIN instructions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from numbers import Integral
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from ..compiler import CompileError, Tensor, current
from . import gemma3
from . import generate as G
from .qwen3 import RunRows, head_rows_chunk
from .gemma4 import Image, gemma4_step


V, E = 0, 1
KINDS = ("V", "E")
S_ITER, S_ACC, S_VER = 40, 41, 42  # device-owned state beat, not the host's halt beat


def _token_ids(values, vocab: int, what: str) -> list[int]:
    ids = list(values)
    if any(not isinstance(x, Integral) or isinstance(x, bool) or not 0 <= x < vocab
           for x in ids):
        raise ValueError(f"{what} must contain Gemma vocabulary IDs")
    return [int(x) for x in ids]


def alloc(b, cap: int, block: int, imem_words: int = 1 << 15,
          waitc: bool = False) -> dict:
    """Reserve history, lookup indices, and two chain programs per bucket.

    The extra history words hold an uncommitted proposal at hist[p + 1].
    Nothing in generate.out is used for a proposal.
    """
    if cap < 2 or block < 2 or cap >= 1 << 24:
        raise ValueError("Gemma prompt lookup needs 2 <= cap < 2^24 and block >= 2")
    nb, slot = G.buckets(cap, block), G.prog_bytes(imem_words, waitc)
    return {"n": cap + 2, "nb": nb, "slot": slot,
            "hist": b.alloc(4 * (cap + 2)), "ramp": b.alloc(4 * cap),
            "progs": b.alloc(slot * len(KINDS) * nb),
            "ptab": b.alloc(8 * len(KINDS) * (nb + 1))}


def prog_addr(a: dict, blocks: int, kind: int) -> int:
    return a["progs"] + a["slot"] * (len(KINDS) * (blocks - 1) + kind)


def ptab_addr(a: dict, blocks: int) -> int:
    return a["ptab"] + 8 * len(KINDS) * (blocks - 1)


def v_limit(cap: int, block: int, blocks: int) -> int:
    """Last within-bucket tpos that admits two input rows; negative if none."""
    return min(block, cap - (blocks - 1) * block) - 2


def _flag(x):
    return ol.minimum(ol.maximum(x, 0.0), 1.0)


def _eq(x, y):
    return ol.minimum(ol.abs(x - y), 1.0) * -1.0 + 1.0


def _stop(st, tok):
    return ol.max(_eq(st[G.S_STOP:G.S_STOP + G.N_STOP], tok))


def _copy_at(b, src, idx, what: str):
    """Read one dynamically indexed history word through the existing V_COPY."""
    out, r = b.alloc((1,)), b.scratch()
    b.rld(r, idx, comment=what)
    b.emit(I.vop(I.V_COPY, out.base, src.base, 0, 1, 1, 0, 0, 0,
                 rb=r, comment=what))
    b.unscratch(r)
    return out


@ol.jit
def _program(m, pos, block: int, R: int, cap: int):
    from .mtp import _args, _load_state

    b, g, a = current(), m.gen, m.gemma_sd
    hist = Tensor(a["hist"], (a["n"],), (1,))
    st, at = _load_state(b, g)
    words = {"tpos": st.base + G.S_TPOS}
    b.run_words = words
    draft = ol.load(hist[pos.pos + 1:pos.pos + 2]) if R == 2 else None

    m.lm_sinks = [G.Greedy(b, m.v_loc, head_rows_chunk(R)) for _ in range(R)]
    m.gen = SimpleNamespace(**{**vars(g), "out": hist})
    try:
        gemma4_step.fn(m, pos, list(range(R)), block)
    finally:
        m.gen = g
    picks = [sink.token() for sink in m.lm_sinks]
    m.lm_sinks = None

    p0 = picks[0]
    ol.store(g.out[pos.pos + 1:pos.pos + 2], p0)
    ol.store(hist[pos.pos + 1:pos.pos + 2], p0)
    left, tpos = st[G.S_LEFT:G.S_LEFT + 1], st[G.S_TPOS:G.S_TPOS + 1]
    stop = _stop(st, p0)
    accepted = ol.zeros([1])
    chosen = b.alloc((1,))
    chosen.set(p0)
    if R == 2:
        accepted.set(_eq(draft, p0) * (_flag(left - 1.0)) * (stop * -1.0 + 1.0))
        r = b.scratch()
        b.rld(r, accepted, comment="draft accepted")
        lp = b.begin_loop(0, rcount=r)
        b.unscratch(r)
        ol.store(g.out[pos.pos + 2:pos.pos + 3], picks[1])
        ol.store(hist[pos.pos + 2:pos.pos + 3], picks[1])
        b.end_loop(lp)
        stop = ol.maximum(stop, _stop(st, picks[1]) * accepted)
        chosen.set(p0 + accepted * (picks[1] - p0))

    st[G.S_TOK:G.S_TOK + 1].set(chosen)
    tnew = tpos + accepted + 1.0
    left.set(left - accepted - 1.0)
    st[S_ITER:S_ITER + 1].set(st[S_ITER:S_ITER + 1] + 1.0)
    st[S_ACC:S_ACC + 1].set(st[S_ACC:S_ACC + 1] + accepted)
    if R == 2:
        st[S_VER:S_VER + 1].set(st[S_VER:S_VER + 1] + 1.0)

    # The newest token is at pn.  Score matching earlier positions by j + 1,
    # so ARGMAX selects the latest j; hist[j + 1] is its one-token proposal.
    pn = tnew + float(pos.t0)
    N = min(cap, pos.t0 + block)
    hp = ol.load(hist[0:N + 1])
    tail = _copy_at(b, hp, pn, "the latest token")
    ramp = ol.load(Tensor(a["ramp"], (cap,), (1,))[0:N])
    score = _eq(hp[0:N], tail) * (ramp + 1.0) * _flag(pn - ramp)
    best = b.argmax(score)
    found = _flag(best[0:1])
    proposal = _copy_at(b, hp, best[1:2] + 1.0, "latest match continuation")
    r = b.scratch()
    b.rld(r, pn * 4.0, comment="next position in history")
    b.emit(I.st(a["hist"] + 4, proposal.base, 1, ra=r,
                comment="proposal -> hist[pn + 1]"))
    b.unscratch(r)

    # Even if no match is found, E can run; its extra history word is ignored.
    wrap = _flag(tnew - float(block - 1))
    tw = tnew - wrap * float(block)
    tpos.set(tw)
    lc, ln = v_limit(cap, block, pos.blocks), v_limit(cap, block, pos.blocks + 1)
    lim = float(lc) + wrap * float(ln - lc)
    use_e = ol.maximum(ol.maximum(found * -1.0 + 1.0, _flag(tw - lim)),
                       _flag(2.0 - left))
    end = ol.maximum(ol.maximum(stop, ol.load(g.state[G.S_HALT:G.S_HALT + 1])),
                     _flag(left * -1.0 + 1.0))
    _args(b, at, words)
    ol.store(g.state[:G.S_HALT], st[:G.S_HALT])
    ol.store(g.state[32:48], st[32:48])
    G._chain(b, ptab_addr(a, pos.blocks), cond=end * -1.0 + 1.0,
             off=(wrap * float(len(KINDS)) + use_e) * 8.0,
             what="next Gemma prompt-lookup iteration")
    b.emit(I.halt())


def compile_gen(image: Image, blocks: int, kind: int, block: int | None = None) -> list | None:
    """Compile the bucket's V or E program; no host-supplied proposal IDs."""
    if not isinstance(image, Image) or type(image.spec) is not gemma3.Spec:
        raise ValueError("Gemma prompt lookup requires a Gemma 3 image")
    if not image.lookup or "gemma_sd" not in image.lookup or image.cfg.S != 1 or \
            image.rows < 2 or image.spec.experts or image.spec.ple_dim or image.ple_host:
        raise ValueError("Gemma prompt lookup requires a dense one-slice, two-row resident image")
    if kind not in (V, E) or not 1 <= blocks <= G.buckets(image.cap, image.block):
        raise ValueError("invalid Gemma prompt-lookup bucket or program kind")
    if block is None:
        block = image.block
    if block != image.block:
        raise ValueError("the image and program attention blocks differ")
    R = 2 if kind == V else 1
    if v_limit(image.cap, block, blocks) < 0 and R == 2:
        return None
    t0 = (blocks - 1) * block
    # Gemma constructs its own per-row attention masks in _pos_rows.  Unlike
    # Qwen's image, its lookup area has no zmask table.
    pos = RunRows(blocks, block, t0, 0, image.cap, R, toks_at=0)
    m = image.descriptors(0)
    m.gemma_sd = image.lookup["gemma_sd"]
    return [_program.trace(image.cfg, 0, {"m": m, "pos": pos,
                                           "block": block, "R": R,
                                           "cap": image.cap}).finish()]


@dataclass
class SpecStats:
    tokens: list[int] = field(default_factory=list)
    iterations: int = 0
    accepted: int = 0
    verifies: int = 0
    compile_s: float = 0.0
    run: dict | None = None


class GemmaPromptLookupDecoder:
    """Full-device, greedy Gemma 3 one-draft decoder on a prepared image."""

    def __init__(self, engine):
        image = engine.image
        if not isinstance(image, Image) or type(image.spec) is not gemma3.Spec or \
                not image.lookup or "gemma_sd" not in image.lookup:
            raise ValueError("the engine needs a Gemma 3 prompt-lookup image")
        if image.cfg.S != 1 or image.rows < 2 or image.spec.experts or image.spec.ple_dim or \
                image.ple_host or not engine.device_inputs:
            raise ValueError("prompt lookup requires a dense one-slice, two-row device image")
        if not engine.can_generate or not getattr(engine.backend, "chains", False) or \
                getattr(engine.backend, "host", None) is not None:
            raise ValueError("prompt lookup requires resident device chains without host service")
        self.eng, self.img, self.a = engine, image, image.lookup["gemma_sd"]
        self._done = engine.__dict__.setdefault("_gemma_sd_gen", {})
        self.last: SpecStats | None = None
        engine.backend.write(0, self.a["ramp"], np.arange(image.cap, dtype=np.float32))

    def _programs(self, b0: int, b1: int, st: SpecStats) -> None:
        eng, image, a = self.eng, self.img, self.a
        for blk in range(b0, b1 + 1):
            if blk in self._done:
                continue
            start = time.perf_counter()
            tab = np.zeros(2 * len(KINDS), np.uint32)
            progs = {}
            for kind in (V, E):
                p = compile_gen(image, blk, kind)
                if p is None:
                    continue
                if not G.fits(image, p):
                    raise CompileError(f"Gemma prompt-lookup {KINDS[kind]} bucket {blk} "
                                       f"does not fit IMEM ({len(p[0])} instructions)")
                prep = getattr(eng.backend, "prepare", None)
                if prep is not None:
                    prep(p)
                eng.backend.write(0, prog_addr(a, blk, kind), G.slot_words(eng.backend, p[0]))
                tab[2 * kind], tab[2 * kind + 1] = (prog_addr(a, blk, kind),
                                                   len(I.assemble(p[0])) // 8)
                progs[kind] = p
            eng.backend.write(0, ptab_addr(a, blk), tab)
            self._done[blk] = progs
            st.compile_s += time.perf_counter() - start

    def warm(self) -> None:
        self._programs(1, self.a["nb"], SpecStats())

    def generate_card(self, tok: int, n: int, stop_ids=None, on_token=None, stop=None,
                      context=()) -> list[int]:
        """Process the unfed `tok` at Engine.pos; emit up to n greedy tokens.

        `context` must contain every token at positions 0..Engine.pos, including tok.
        The host supplies the prompt once and never supplies a draft or picks logits.
        A host stop request is observed after the current iteration, which may
        publish two accepted tokens. It does not cancel an in-flight verifier.
        """
        eng, image, a = self.eng, self.img, self.a
        p = eng.pos
        if not isinstance(tok, Integral) or isinstance(tok, bool) or not 0 <= tok < image.spec.vocab:
            raise ValueError("the seed must be a Gemma vocabulary ID")
        if not isinstance(n, Integral) or isinstance(n, bool) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        ctx = _token_ids(context, image.spec.vocab, "context")
        if len(ctx) != p + 1 or ctx[-1] != int(tok):
            raise ValueError("context must be the complete prefix through the unfed seed")
        n = min(n, image.cap - p)
        self.last = st = SpecStats()
        if n <= 0:
            return []
        ids = _token_ids(image.spec.eos if stop_ids is None else stop_ids,
                         image.spec.vocab, "stop IDs")
        words = G.state_words(image.spec, int(tok), p, n, ids, eng.block)
        if stop is not None and getattr(eng.backend, "run_generate", None) is None:
            raise ValueError("a host stop callback requires streaming run_generate")
        eng._drain()
        b0, b1 = p // eng.block + 1, (p + n - 1) // eng.block + 1
        self._programs(b0, b1, st)
        hist = np.zeros(a["n"], np.float32)
        hist[:p + 1] = ctx
        g = image.lookup["gen"]
        eng.backend.write(0, a["hist"], hist)
        eng.backend.write(0, g["state"], words)
        eng.backend.write(0, g["out"] + 4 * (p + 1), np.full(n, G.OUT_MARK, np.uint32))
        first = self._done[b0][E]  # first proposal is made by the E program on the FPGA
        runner = getattr(eng.backend, "run_generate", None)
        if runner is None:
            run = eng.backend.run(first)
            raw = eng.backend.read(0, g["out"] + 4 * (p + 1), 4 * n).view(np.uint32)
            count = int(np.argmax(raw == G.OUT_MARK)) if (raw == G.OUT_MARK).any() else n
            got = [int(x) for x in raw[:count].view(np.float32)]
            if on_token is not None:
                for x in got:
                    on_token(x)
        else:
            run, got = runner(first, g["out"] + 4 * (p + 1), n, on_token, stop, g["state"])
        eng.stats.append(run)
        state = eng.backend.read(0, g["state"], 4 * G.STATE_WORDS).view(np.float32)
        st.iterations, st.accepted, st.verifies = (int(state[i]) for i in (S_ITER, S_ACC, S_VER))
        st.tokens, st.run = list(got), run
        eng.pos = p + len(got)
        return list(got)

    def generate_from_unfed_prompt(self, prompt, n: int, **kwargs) -> list[int]:
        """Prefill all but the final prompt ID, then let E pick the first output."""
        if self.eng.pos != 0:
            raise ValueError("use generate_card with a complete context to continue a prefix")
        ids = _token_ids(prompt, self.img.spec.vocab, "prompt")
        if not ids:
            raise ValueError("prompt must contain in-vocabulary token IDs")
        if len(ids) > self.img.cap:
            raise ValueError("prompt exceeds the KV capacity")
        if not isinstance(n, Integral) or isinstance(n, bool) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        if n == 0:
            self.last = SpecStats()
            return []
        if len(ids) > 1:
            for _ in self.eng.prefill_chunks(ids[:-1], logits=False):
                pass
        return self.generate_card(ids[-1], n, context=ids, **kwargs)
