"""Bounded PersonaPlex temporal-to-greedy-token slice on the selected backend.

Captured summed embeddings are the inputs. RoPE coefficients, temporal computation,
INT8 text-head projection and greedy selection execute on device. This slice does
not implement token embeddings, sampling, the depth transformer or audio codecs.
"""
from __future__ import annotations

import time

import numpy as np

from .. import language as ol
from .. import qcache as QC
from .. import quant as Q
from .. import waits
from ..compiler import CompileError, QTensor, Tensor, arg_words, trace_fit
from . import personaplex_temporal as T
from . import qwen3 as G
from .generate import Greedy


@ol.jit
def _resident_temporal(m, pos, block):
    # Reuse the temporal kernel unchanged, substituting resident coefficient rows.
    m.cos, m.sin = m.cos_t[pos.pos, :], m.sin_t[pos.pos, :]
    T._step.fn(m, pos, block)


@ol.jit
def _text_head(m):
    # m.x already contains the final normalized hidden vector. Do not normalize twice.
    x = ol.quantize(ol.load(m.x))
    greedy = Greedy(ol.current(), m.token_vocab, m.head_chunk)
    for lo in range(0, m.token_vocab, m.head_chunk):
        hi = min(lo + m.head_chunk, m.token_vocab)
        logits = ol.dot(x, m.token_head[lo:hi, :])
        ol.store(m.token_logits[:, lo:hi], logits)
        greedy(logits, lo)
    ol.store(m.token_id, greedy.token())


class TokenImage(T.Image):
    """Temporal image plus resident RoPE, an INT8 head and diagnostic logits.

    Construction only computes layout. A callable ``text_head`` is not evaluated
    until build(), after TokenEngine has compiled every temporal/head program.
    The extra 2 MiB workspace is reserved conservatively; separate submissions
    also retain the normal driver's program area after the image.
    """

    def __init__(self, spec, cfg, cap=256, wformat="int4", formats="", *,
                 vocab=32000, text_head=None):
        if isinstance(vocab, bool) or not isinstance(vocab, int) or not 0 < vocab <= 1 << 24:
            raise ValueError("vocab must be an integer within 1..16777216")
        self.vocab, self._text_head = vocab, text_head
        super().__init__(spec, cfg, cap, wformat, formats=formats)
        self.head_chunk = min(G.HEAD_CHUNK, cfg.TMEM_WORDS // 8)
        if self.head_chunk < 1:
            raise ValueError("TMEM is too small for a text head")
        self.head_program = None
        self.head_format = "int8"
        # RunPos bounds cover the physical attention block, including its unused
        # tail. Keep the full compiled address domain inside these tables.
        self.rope_bytes = 8 * self.cap * (spec.head_dim // 2)
        self.head_bytes = vocab * (Q.row_bytes(spec.hidden, "int8", cfg.D) +
                                   Q.scale_bytes(spec.hidden, "int8", cfg.D))
        at = self.nbytes

        def alloc(size, alignment=64):
            nonlocal at
            start = -(-at // alignment) * alignment
            at = start + size
            return start

        self.cos_table = alloc(self.rope_bytes // 2)
        self.sin_table = alloc(self.rope_bytes // 2)
        self.text_head_offsets = (alloc(vocab * Q.row_bytes(spec.hidden, "int8", cfg.D)),
                                  alloc(vocab * Q.scale_bytes(spec.hidden, "int8", cfg.D)))
        self.io["token_logits"] = alloc(4 * vocab)
        self.io["token_id"] = alloc(4)
        self.program_workspace_bytes = 2 << 20
        self.program_workspace = alloc(self.program_workspace_bytes, 4096)
        self.nbytes = -(-at // 4096) * 4096
        self.required_bytes = self.nbytes + waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
        self.weight_bytes += self.head_bytes
        self.weight_code_bytes += vocab * Q.row_bytes(spec.hidden, "int8", cfg.D)
        if self.required_bytes > cfg.DRAM_BYTES:
            raise MemoryError(f"token image and program area need {self.required_bytes} bytes, "
                              f"device has {cfg.DRAM_BYTES}")

    def descriptors(self, sid):
        m = super().descriptors(sid)
        half = self.temporal_spec.head_dim // 2
        m.cos_t = Tensor(self.cos_table, (self.cap, half), (half, 1))
        m.sin_t = Tensor(self.sin_table, (self.cap, half), (half, 1))
        m.token_head = QTensor.weight(*self.text_head_offsets, self.vocab,
                                     self.temporal_spec.hidden, "int8", self.cfg.D)
        m.token_logits = Tensor(self.io["token_logits"], (1, self.vocab), (self.vocab, 1))
        m.token_id = Tensor(self.io["token_id"], (1,), (1,))
        m.token_vocab, m.head_chunk = self.vocab, self.head_chunk
        return m

    def compile_head(self):
        b = trace_fit(_text_head, self.cfg, 0, lambda: {"m": self.descriptors(0)})
        program = b.finish()
        if len(program) * 8 > self.cfg.IMEM_WORDS:
            raise CompileError(f"text head needs {len(program) * 8} instruction words; "
                               f"device has {self.cfg.IMEM_WORDS}")
        if b.run_args:
            raise CompileError("text head must not require host runtime arguments")
        return [program]

    def compile_bucket(self, bucket):
        lo = (bucket - 1) * self.block
        if not 0 <= lo < self.frame_limit:
            raise ValueError(f"attention bucket {bucket} is outside the frame limit")
        if self.head_program is None:
            self.head_program = self.compile_head()
        b = trace_fit(_resident_temporal, self.cfg, 0, lambda: {
            "m": self.descriptors(0),
            "pos": G.RunPos(bucket, self.block, lo, self.zmask, self.cap),
            "block": self.block})
        program = b.finish()
        if len(program) * 8 > self.cfg.IMEM_WORDS:
            raise CompileError(f"resident temporal step needs {len(program) * 8} instruction words; "
                               f"device has {self.cfg.IMEM_WORDS}")
        return [program], list(b.run_args)

    def build(self, layer_weights, final_norm):
        source = self._text_head() if callable(self._text_head) else self._text_head
        head = T._checked(source, (self.vocab, self.temporal_spec.hidden), "text_head")
        images = super().build(layer_weights, final_norm)
        data = images[0]

        def put(addr, value):
            value = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
            data[addr:addr + value.size] = value

        for pos in range(self.cap):
            cos, sin = G.rope_tables(self.spec, pos)
            half = self.temporal_spec.head_dim // 2
            put(self.cos_table + 4 * half * pos, T._checked(cos, (half,), "cos"))
            put(self.sin_table + 4 * half * pos, T._checked(sin, (half,), "sin"))
        qb = QC.Batch()
        da, sa = self.text_head_offsets
        qb.submit(head, "int8", self.cfg.D, lambda q, s: (put(da, q), put(sa, s)))
        qb.wait()
        return images


class TokenEngine(T.Engine):
    """Two device programs per frame; only input embeddings and a token cross the host.

    step() returns a greedy integer ID. read_hidden()/read_logits() are explicit
    diagnostic transfers and do not participate in the production step path.
    Backend qualification, reset, context and failed-frame rules follow Engine.
    """

    def __init__(self, spec, layer_weights, final_norm, text_head, *, vocab=32000, **kwargs):
        if kwargs.get("backend", "isa") == "board":
            raise ValueError("token board execution requires a qualified backend factory "
                             "that checks CAP_GEN before upload")
        self.vocab, self._text_head = vocab, text_head
        super().__init__(spec, layer_weights, final_norm, **kwargs)
        if not getattr(self.backend, "generates", False):
            self.close()
            raise ValueError("token backend must support RLD and ARGMAX (CAP_GEN)")

    def _make_image(self, spec, cfg, cap, wformat, formats):
        return TokenImage(spec, cfg, cap, wformat, formats, vocab=self.vocab,
                          text_head=self._text_head)

    def _run(self, programs, args=None):
        start = getattr(self.backend, "start", None)
        if start is not None:
            start(programs, args=args)
            return self.backend.wait() or {}
        return self.backend.run(programs, args=args) or {}

    def step(self, embedding):
        if self._closed:
            raise RuntimeError("engine is closed")
        if self._failed:
            raise RuntimeError("previous frame failed; reset the engine before continuing")
        if self.pos >= self.cap:
            raise RuntimeError(f"temporal context is full ({self.cap} frames); reset required; "
                               "sliding-window KV is not implemented")
        x = T._checked(embedding, (self.spec.hidden,), "embedding")
        programs, run_args = self._programs[self.pos // self.image.block + 1]
        started = time.perf_counter()
        try:
            self.backend.write(0, self.image.io["x"], x)
            args = arg_words(run_args, G.RunPos.values(0, self.pos, block=self.image.block))
            temporal = self._run(programs, args)
            head = self._run(self.image.head_program)
            raw = self.backend.read(0, self.image.io["token_id"], 4)
            token = np.asarray(raw, np.uint8).view(np.float32)
            if (token.shape != (1,) or not np.isfinite(token[0]) or
                    token[0] != int(token[0]) or not 0 <= token[0] < self.vocab):
                raise RuntimeError("backend returned an invalid token ID")
            result = int(token[0])
        except BaseException:
            self._failed = True
            raise
        self.poss[0] += 1
        self.last_stats = {"frame": self.pos - 1, "stages": {"temporal": temporal, "head": head},
                           "wall_seconds": time.perf_counter() - started}
        if all(type(stage.get("cycles")) is int for stage in (temporal, head)):
            self.last_stats["cycles"] = temporal["cycles"] + head["cycles"]
        return result

    def _read_vector(self, address, width, name):
        if self._closed:
            raise RuntimeError("engine is closed")
        if self._failed or self.pos == 0:
            raise RuntimeError("diagnostics require a successfully completed frame")
        raw = self.backend.read(0, address, 4 * width)
        out = np.asarray(raw, np.uint8).view(np.float32).copy()
        if out.shape != (width,) or not np.isfinite(out).all():
            raise RuntimeError(f"backend returned invalid {name}")
        return out

    def read_hidden(self):
        return self._read_vector(self.image.io["x"], self.spec.hidden, "hidden state")

    def read_logits(self):
        return self._read_vector(self.image.io["token_logits"], self.vocab, "text logits")
