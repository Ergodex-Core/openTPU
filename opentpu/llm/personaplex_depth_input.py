"""Standalone device depth conditioning: selected projection plus step embedding.

This implements LMModel.forward_depformer's input preparation only. It does not
execute the six depth layers, sampling, codec, or unified temporal/depth memory.
"""
from dataclasses import dataclass, replace
import operator
import time
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from .. import qcache as QC
from .. import quant as Q
from .. import waits
from ..compiler import Affine, CompileError, DevVar, QTensor, Tensor, trace_fit
from ..isasim import Config
from . import qwen3 as G
from .personaplex_audio_heads import _checked
from .personaplex_embeddings import _row


@dataclass(frozen=True)
class DepthInputSpec:
    hidden: int = 4096
    depth: int = 1024
    codebooks: int = 16
    text_vocab: int = 32001
    audio_vocab: int = 2049
    multi_linear: bool = True

    def __post_init__(self):
        for name in ("hidden", "depth", "codebooks", "text_vocab", "audio_vocab"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 1 << 24:
                raise ValueError(f"{name} must be a positive integer below2^24")
        if type(self.multi_linear) is not bool:
            raise ValueError("multi_linear must be boolean")
        if self.embedding_rows >= 1 << 24:
            raise ValueError("aggregate embedding rows must be below2^24 for exact device indexing")

    @property
    def projections(self):
        return self.codebooks if self.multi_linear else 1

    @property
    def embedding_rows(self):
        return self.text_vocab + 1 + (self.codebooks - 1) * (self.audio_vocab + 1)


@ol.jit
def _condition(m):
    b = ol.current()
    control = ol.load(m.control)
    codebook, token = control[0:1], control[1:2]
    if m.projections > 1:
        rq, rs = b.scratch(), b.scratch()
        b.emit(I.rld(rq, codebook.base, mul=m.data_stride,
                     bound=(0, (m.projections - 1) * m.data_stride), comment="depth projection data bank"))
        b.emit(I.rld(rs, codebook.base, mul=m.scale_stride,
                     bound=(0, (m.projections - 1) * m.scale_stride), comment="depth projection scale bank"))
        data = Affine(m.projection_offsets[0]) + DevVar("projection data", rq)
        scales = Affine(m.projection_offsets[1]) + DevVar("projection scales", rs)
    else:
        data, scales = m.projection_offsets
    matrix = QTensor.weight(data, scales, m.depth_width, m.hidden_width, "int8", b.cfg.D)
    projected = ol.dot(ol.quantize(ol.load(m.hidden)), matrix)
    ol.store(m.projected, projected)
    # Ragged tables are concatenated with one leading zero row per table.
    # cb0 selects text; cb1.. select the previous-audio table cb-1. All arithmetic
    # here is exact integer-valued FP32; the layout bounds the total below2^24.
    audio_index = ol.maximum(codebook - 1.0, 0.0)
    audio_present = ol.minimum(codebook, 1.0)
    row = audio_index * m.audio_rows + audio_present * m.text_rows + token + 1.0
    embedding = _row(m.embeddings, row)
    ol.store(m.output, projected + embedding)
    if m.projections > 1:
        b.unscratch(rq)
        b.unscratch(rs)


class DepthInputImage:
    cap = batch = 1
    kv_bytes = 0
    wformat = "int8"

    def __init__(self, spec, cfg):
        if cfg.S != 1 or spec.hidden % cfg.D:
            raise ValueError("depth conditioning requires one slice and hidden divisible by D")
        self.spec, self.cfg = spec, cfg
        self.data_stride = spec.depth * Q.row_bytes(spec.hidden, "int8", cfg.D)
        self.scale_stride = spec.depth * Q.scale_bytes(spec.hidden, "int8", cfg.D)
        self.projection_bytes = spec.projections * (self.data_stride + self.scale_stride)
        self.embedding_bytes = 4 * spec.depth * spec.embedding_rows
        self.weight_bytes = self.projection_bytes + self.embedding_bytes
        self.weight_code_bytes = spec.projections * self.data_stride
        at = 0
        def alloc(size, alignment=64):
            nonlocal at
            start = -(-at // alignment) * alignment
            at = start + size
            return start
        self.io = {"hidden": alloc(4 * spec.hidden), "control": alloc(8),
                   "projected": alloc(4 * spec.depth), "output": alloc(4 * spec.depth)}
        self.projection_offsets = (alloc(self.weight_code_bytes, max(64, cfg.D)),
                                   alloc(spec.projections * self.scale_stride))
        self.embeddings_at = alloc(self.embedding_bytes)
        self.nbytes = -(-at // 4096) * 4096
        self.required_bytes = self.nbytes + waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
        if self.required_bytes > min(cfg.DRAM_BYTES, 1 << 32):
            raise MemoryError(f"depth input needs {self.required_bytes} bytes; "
                              f"device/address capacity is {min(cfg.DRAM_BYTES, 1 << 32)}")

    def descriptors(self):
        s = self.spec
        return SimpleNamespace(
            hidden=Tensor(self.io["hidden"], (1, s.hidden), (s.hidden, 1)),
            control=Tensor(self.io["control"], (2,), (1,)),
            projected=Tensor(self.io["projected"], (1, s.depth), (s.depth, 1)),
            output=Tensor(self.io["output"], (1, s.depth), (s.depth, 1)),
            embeddings=Tensor(self.embeddings_at, (s.embedding_rows, s.depth), (s.depth, 1)),
            projections=s.projections, hidden_width=s.hidden, depth_width=s.depth,
            projection_offsets=self.projection_offsets, data_stride=self.data_stride,
            scale_stride=self.scale_stride, audio_rows=s.audio_vocab + 1, text_rows=s.text_vocab + 1)

    def compile(self):
        b = trace_fit(_condition, self.cfg, 0, lambda: {"m": self.descriptors()})
        program = b.finish()
        if len(program) * 8 > self.cfg.IMEM_WORDS:
            raise CompileError(f"depth input needs {len(program) * 8} instruction words; "
                               f"device has {self.cfg.IMEM_WORDS}")
        if b.run_args:
            raise CompileError("depth input must not use host row/bank address arguments")
        return [program]

    def build(self, projections, embeddings):
        s = self.spec
        projections = projections() if callable(projections) else projections
        bank = _checked(projections, (s.projections, s.depth, s.hidden), "depth projections")
        embeddings = embeddings() if callable(embeddings) else embeddings
        if len(embeddings) != s.codebooks:
            raise ValueError("expected text plus every previous-audio embedding table")
        tables = [_checked(t, (s.text_vocab if i == 0 else s.audio_vocab, s.depth), "depth embedding")
                  for i, t in enumerate(embeddings)]
        data = np.zeros(self.nbytes, np.uint8)
        def put(at, value):
            raw = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
            data[at:at + raw.size] = raw
        for index in range(s.projections):
            da, sa = (self.projection_offsets[0] + index * self.data_stride,
                      self.projection_offsets[1] + index * self.scale_stride)
            batch = QC.Batch()
            batch.submit(bank[index], "int8", self.cfg.D,
                         lambda q, scales, da=da, sa=sa: (put(da, q), put(sa, scales)))
            batch.wait()
        at = self.embeddings_at
        for table in tables:
            put(at + 4 * s.depth, table)
            at += 4 * s.depth * (len(table) + 1)
        return [data]


class DepthInputEngine:
    """Uploads a diagnostic input vector and IDs; output stays on selected backend."""
    def __init__(self, projections, embeddings, *, spec=None, cfg=None, backend="isa"):
        self.spec = spec or DepthInputSpec()
        self._closed = self._failed = self._ready = False
        self.poss, self.last_stats = [0], {}
        if backend == "board":
            raise ValueError("depth input board execution requires a qualified backend factory with CAP_GEN")
        if backend != "isa" and not callable(backend):
            raise ValueError("backend must be isa or a qualified backend factory")
        if cfg is None:
            if backend != "isa":
                raise ValueError("physical backend requires its measured device config")
            cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                         TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True, DRAM_BYTES=1 << 40)
            cfg = replace(cfg, DRAM_BYTES=DepthInputImage(self.spec, cfg).required_bytes)
        self.cfg, self.image = cfg, DepthInputImage(self.spec, cfg)
        self.program = self.image.compile()
        images = self.image.build(projections, embeddings)
        self.backend = G.IsaBackend(cfg, images, adopt=True) if backend == "isa" else backend(cfg, images)
        try:
            if not getattr(self.backend, "generates", False):
                raise ValueError("depth input requires indexed RLD execution (CAP_GEN)")
            attach = getattr(self.backend, "attach", None)
            if attach is not None:
                attach(self)
        except BaseException:
            self.close()
            raise

    def _check(self):
        if self._closed:
            raise RuntimeError("depth input engine is closed")
        if self._failed:
            raise RuntimeError("previous depth input failed; reset required")

    def step(self, hidden, codebook, previous_token):
        self._check()
        ids = []
        for value, label in ((codebook, "codebook"), (previous_token, "previous token")):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{label} must be integral")
            try:
                ids.append(operator.index(value))
            except TypeError as error:
                raise ValueError(f"{label} must be integral") from error
        cb, token = ids
        if not 0 <= cb < self.spec.codebooks:
            raise ValueError("codebook outside device bank")
        limit = self.spec.text_vocab if cb == 0 else self.spec.audio_vocab
        if not -1 <= token < limit:
            raise ValueError("previous token outside selected embedding table")
        value = _checked(hidden, (self.spec.hidden,), "temporal hidden")
        self._ready = False
        started = time.perf_counter()
        try:
            self.backend.write(0, self.image.io["hidden"], value)
            self.backend.write(0, self.image.io["control"], np.asarray(ids, np.float32))
            start = getattr(self.backend, "start", None)
            if start is not None:
                start(self.program)
                stats = self.backend.wait()
            else:
                stats = self.backend.run(self.program)
        except BaseException:
            self._failed = True
            raise
        self._ready = True
        self.poss[0] += 1
        self.last_stats = {**(stats or {}), "codebook": cb, "request": self.poss[0] - 1,
                           "wall_seconds": time.perf_counter() - started}

    @property
    def output_address(self):
        self._check()
        if not self._ready:
            raise RuntimeError("no completed depth input")
        return self.image.io["output"]

    def _read(self, key):
        _ = self.output_address
        raw = self.backend.read(0, self.image.io[key], 4 * self.spec.depth)
        value = np.asarray(raw, np.uint8).view(np.float32).copy()
        if value.shape != (self.spec.depth,) or not np.isfinite(value).all():
            raise RuntimeError("invalid device depth input output")
        return value

    def read_output(self):
        return self._read("output")

    def read_projected(self):
        return self._read("projected")

    def reset(self):
        if self._closed:
            raise RuntimeError("depth input engine is closed")
        board = getattr(self.backend, "board", None)
        if self._failed and board is not None:
            board.stop()
        self._failed = self._ready = False
        self.poss[0], self.last_stats = 0, {}

    def close(self):
        if not self._closed:
            self._closed = True
            close = getattr(getattr(self, "backend", None), "close", None)
            if close is not None:
                close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
