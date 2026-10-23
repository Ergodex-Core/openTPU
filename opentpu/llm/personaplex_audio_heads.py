"""Standalone PersonaPlex codebook-indexed INT8 audio projection and greedy ID.

The caller supplies the native depth-transformer output and a codebook index.
The device selects that codebook's resident head, projects without another norm,
and selects its greedy token. This does not implement the six-layer depth
transformer, stochastic sampling, conditioning, or a codec/full speech runtime.
"""
from __future__ import annotations

from dataclasses import replace
import time
from types import SimpleNamespace

import numpy as np

from .. import fp32 as F
from .. import isa as I
from .. import language as ol
from .. import qcache as QC
from .. import quant as Q
from .. import waits
from ..compiler import Affine, CompileError, DevVar, QTensor, Tensor, trace_fit
from ..isasim import Config
from . import qwen3 as G
from .generate import Greedy


def _positive_int(value, name, limit=None):
    if (isinstance(value, bool) or not isinstance(value, int) or value <= 0
            or limit is not None and value > limit):
        raise ValueError(f"{name} must be a positive integer" +
                         (f" no greater than {limit}" if limit is not None else ""))
    return value


def _checked(value, shape, name):
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.asarray(value, np.float32)
    if value.shape != tuple(shape):
        raise ValueError(f"{name}: shape {value.shape}, expected {tuple(shape)}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: non-finite values")
    return F.ftz(value)


@ol.jit
def _audio_head(m):
    b = ol.current()
    index = ol.load(m.codebook)
    rq, rs = b.scratch(), b.scratch()
    # Integer multiplication in RLD selects resident weight offsets. The host
    # uploads only the scalar index, never an offset or a selected head's weights.
    b.emit(I.rld(rq, index.base, mul=m.data_stride,
                 bound=(0, (m.codebooks - 1) * m.data_stride), comment="head bank data offset"))
    b.emit(I.rld(rs, index.base, mul=m.scale_stride,
                 bound=(0, (m.codebooks - 1) * m.scale_stride), comment="head bank scale offset"))
    head = QTensor.weight(Affine(m.head_offsets[0]) + DevVar("head data", rq),
                          Affine(m.head_offsets[1]) + DevVar("head scales", rs),
                          m.vocab, m.hidden_width, "int8", b.cfg.D)
    # The native linear consumes dep_output directly, without an extra head norm.
    hidden = ol.quantize(ol.load(m.hidden))
    greedy = Greedy(b, m.vocab, m.head_chunk)
    for lo in range(0, m.vocab, m.head_chunk):
        hi = min(lo + m.head_chunk, m.vocab)
        logits = ol.dot(hidden, head[lo:hi, :])
        ol.store(m.logits[:, lo:hi], logits)
        greedy(logits, lo)
    ol.store(m.token_id, greedy.token())
    b.unscratch(rq)
    b.unscratch(rs)


class AudioHeadImage:
    """Layout-only bank; build materializes [codebooks,vocab,hidden] once offline."""

    wformat = head_format = "int8"
    cap = batch = 1
    kv_bytes = 0

    def __init__(self, cfg, *, hidden=1024, codebooks=16, vocab=2048, head_bank=None):
        self.hidden = _positive_int(hidden, "hidden")
        self.codebooks = _positive_int(codebooks, "codebooks", 1 << 24)
        self.vocab = _positive_int(vocab, "vocab", 1 << 24)
        if cfg.S != 1:
            raise ValueError("audio heads currently support one FPGA slice")
        if hidden % cfg.D:
            raise ValueError(f"hidden must be a multiple of D={cfg.D}")
        self.cfg, self._head_bank = cfg, head_bank
        self.head_chunk = min(G.HEAD_CHUNK, cfg.TMEM_WORDS // 8)
        if self.head_chunk < 1:
            raise ValueError("TMEM is too small for an audio head")
        self.data_stride = vocab * Q.row_bytes(hidden, "int8", cfg.D)
        self.scale_stride = vocab * Q.scale_bytes(hidden, "int8", cfg.D)
        self.weight_code_bytes = codebooks * self.data_stride
        self.weight_bytes = codebooks * (self.data_stride + self.scale_stride)
        self.head_bytes = self.weight_bytes
        at = 0

        def alloc(size, alignment=64):
            nonlocal at
            start = -(-at // alignment) * alignment
            at = start + size
            return start

        self.io = {"hidden": alloc(4 * hidden), "codebook": alloc(4),
                   "logits": alloc(4 * vocab), "token_id": alloc(4)}
        # STREAM matrix rows require a D-byte-aligned base even when a small
        # vocabulary leaves the preceding FP32 I/O area only 64-byte aligned.
        self.head_offsets = (alloc(self.weight_code_bytes, max(64, cfg.D)),
                             alloc(codebooks * self.scale_stride))
        self.nbytes = -(-at // 4096) * 4096
        self.required_bytes = self.nbytes + waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
        # ISA addresses and RLD's integer offset multiplication are unsigned32.
        if self.required_bytes > min(cfg.DRAM_BYTES, 1 << 32):
            raise MemoryError(f"audio head image and program area need {self.required_bytes} bytes, "
                              f"device/address capacity is {min(cfg.DRAM_BYTES, 1 << 32)}")

    def descriptors(self):
        return SimpleNamespace(
            hidden=Tensor(self.io["hidden"], (1, self.hidden), (self.hidden, 1)),
            codebook=Tensor(self.io["codebook"], (1,), (1,)),
            logits=Tensor(self.io["logits"], (1, self.vocab), (self.vocab, 1)),
            token_id=Tensor(self.io["token_id"], (1,), (1,)),
            hidden_width=self.hidden, codebooks=self.codebooks, vocab=self.vocab,
            head_offsets=self.head_offsets, data_stride=self.data_stride,
            scale_stride=self.scale_stride, head_chunk=self.head_chunk)

    def compile(self):
        b = trace_fit(_audio_head, self.cfg, 0, lambda: {"m": self.descriptors()})
        program = b.finish()
        if len(program) * 8 > self.cfg.IMEM_WORDS:
            raise CompileError(f"audio head needs {len(program) * 8} instruction words; "
                               f"device has {self.cfg.IMEM_WORDS}")
        if b.run_args:
            raise CompileError("audio head index must come from the device scalar, not host arguments")
        return [program]

    def build(self):
        source = self._head_bank() if callable(self._head_bank) else self._head_bank
        bank = _checked(source, (self.codebooks, self.vocab, self.hidden), "head_bank")
        data = np.zeros(self.nbytes, np.uint8)

        def put(addr, value):
            value = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
            data[addr:addr + value.size] = value

        # Construction is offline. Runtime never indexes or materializes host heads.
        for index in range(self.codebooks):
            da = self.head_offsets[0] + index * self.data_stride
            sa = self.head_offsets[1] + index * self.scale_stride
            qb = QC.Batch()
            qb.submit(bank[index], "int8", self.cfg.D,
                      lambda q, s, da=da, sa=sa: (put(da, q), put(sa, s)))
            qb.wait()
        return [data]


class AudioHeadEngine:
    """Stateless head requests on one backend; repeated/arbitrary indices are allowed.

    step(hidden,codebook) uploads only the vector and index, and reads one token.
    Explicit diagnostic reads are separate. A physical backend must be supplied
    by a factory that qualifies CAP_GEN and the complete device/profile identity
    before any upload. There is no host projection/argmax fallback.
    """

    def __init__(self, head_bank, *, cfg=None, backend="isa", hidden=1024,
                 codebooks=16, vocab=2048):
        if backend == "board":
            raise ValueError("audio head board execution requires a qualified backend factory "
                             "that checks CAP_GEN before upload")
        if backend != "isa" and not callable(backend):
            raise ValueError("backend must be 'isa' or a qualified backend factory")
        self._closed = self._failed = False
        self.poss, self.last_stats = [0], {}
        if cfg is None:
            if backend != "isa":
                raise ValueError("the FPGA backend requires its measured device config")
            probe_cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                               TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                               DRAM_BYTES=1 << 40)
            probe = AudioHeadImage(probe_cfg, hidden=hidden, codebooks=codebooks, vocab=vocab)
            cfg = replace(probe_cfg, DRAM_BYTES=probe.required_bytes)
        self.cfg = cfg
        self.image = AudioHeadImage(cfg, hidden=hidden, codebooks=codebooks, vocab=vocab,
                                    head_bank=head_bank)
        self.program = self.image.compile()
        images = self.image.build()
        self.backend = (G.IsaBackend(cfg, images, adopt=True) if backend == "isa" else
                        backend(cfg, images))
        try:
            if not getattr(self.backend, "generates", False):
                raise ValueError("audio head backend must support RLD and ARGMAX (CAP_GEN)")
            attach = getattr(self.backend, "attach", None)
            if attach is not None:
                attach(self)
        except BaseException:
            self.close()
            raise

    @property
    def pos(self):
        return self.poss[0]

    def step(self, hidden, codebook):
        if self._closed:
            raise RuntimeError("engine is closed")
        if self._failed:
            raise RuntimeError("previous head failed; reset the engine before continuing")
        if (isinstance(codebook, (bool, np.bool_)) or not isinstance(codebook, (int, np.integer))
                or not 0 <= codebook < self.image.codebooks):
            raise ValueError(f"codebook must be an integer within 0..{self.image.codebooks - 1}")
        vector = _checked(hidden, (self.image.hidden,), "hidden")
        started = time.perf_counter()
        try:
            self.backend.write(0, self.image.io["hidden"], vector)
            self.backend.write(0, self.image.io["codebook"], np.array([codebook], np.float32))
            start = getattr(self.backend, "start", None)
            if start is not None:
                start(self.program)
                stats = self.backend.wait() or {}
            else:
                stats = self.backend.run(self.program) or {}
            raw = self.backend.read(0, self.image.io["token_id"], 4)
            token = np.asarray(raw, np.uint8).view(np.float32)
            if (token.shape != (1,) or not np.isfinite(token[0]) or
                    token[0] != int(token[0]) or not 0 <= token[0] < self.image.vocab):
                raise RuntimeError("backend returned an invalid audio token ID")
            result = int(token[0])
        except BaseException:
            self._failed = True
            raise
        self.poss[0] += 1
        self.last_stats = {**stats, "step": self.pos - 1, "codebook": int(codebook),
                           "wall_seconds": time.perf_counter() - started}
        return result

    def reset(self):
        if self._closed:
            raise RuntimeError("engine is closed")
        if self._failed:
            board = getattr(self.backend, "board", None)
            if board is not None:
                board.stop()
        self.poss[0], self.last_stats, self._failed = 0, {}, False

    def _read_vector(self, key, width):
        if self._closed:
            raise RuntimeError("engine is closed")
        if self._failed or self.pos == 0:
            raise RuntimeError("diagnostics require a successfully completed head request")
        raw = self.backend.read(0, self.image.io[key], 4 * width)
        values = np.asarray(raw, np.uint8).view(np.float32).copy()
        if values.shape != (width,) or not np.isfinite(values).all():
            raise RuntimeError(f"backend returned invalid {key}")
        return values

    def read_hidden(self):
        return self._read_vector("hidden", self.image.hidden)

    def read_logits(self):
        return self._read_vector("logits", self.image.vocab)

    def close(self):
        if not self._closed:
            close = getattr(getattr(self, "backend", None), "close", None)
            if close is not None:
                close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
