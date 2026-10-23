"""Device input-code embeddings and stored-voice replay, not a speech runtime.

Tables stay FP32 to preserve the pinned source's embedding values and addition
order. This standalone stage does not fit beside the current temporal image;
a shared residency/overlay scheduler is still required for a complete model.
"""
from dataclasses import dataclass, replace
from types import SimpleNamespace
import operator
import time

import numpy as np

from .. import isa as I
from .. import language as ol
from .. import waits
from ..compiler import Affine, CompileError, DevVar, Tensor, trace_fit
from ..isasim import Config
from . import personaplex_temporal as T
from . import qwen3 as G


@dataclass(frozen=True)
class EmbeddingSpec:
    hidden: int = 4096
    text_vocab: int = 32001
    audio_vocab: int = 2049
    audio_streams: int = 16
    voice_frames: int = 0

    def __post_init__(self):
        for name in ("hidden", "text_vocab", "audio_vocab", "audio_streams"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 1 << 24:
                raise ValueError(f"{name} must be a positive integer below 2^24")
        if type(self.voice_frames) is not int or not 0 <= self.voice_frames < 1 << 24:
            raise ValueError("voice_frames must be a nonnegative integer below 2^24")


def _row(table, token):
    b = ol.current()
    register = b.scratch()
    stride = 4 * table.strides[0]
    b.emit(I.rld(register, token.base, mul=stride,
                 bound=(0, (table.shape[0] - 1) * stride), comment="resident embedding row"))
    row = ol.load(Tensor(Affine.of(table.base) + DevVar("embedding row", register),
                         (1, table.shape[1]), table.strides))
    b.unscratch(register)
    return row


@ol.jit
def _embed_codes(m):
    codes = ol.load(m.codes)
    # Each table has a prepended +0 row: native zero_token_id -1 maps to row0.
    # Preserve upstream LMModel.embed_codes order: audio0, audio1.., text last.
    value = _row(m.audio[0], codes[1:2] + 1.0)
    for index in range(1, len(m.audio)):
        value = value + _row(m.audio[index], codes[index + 1:index + 2] + 1.0)
    value = value + _row(m.text, codes[0:1] + 1.0)
    ol.store(m.output, value)


@ol.jit
def _voice_frame(m):
    index = ol.load(m.voice_index)
    ol.store(m.output, _row(m.voice, index))


class EmbeddingImage:
    cap = batch = 1
    kv_bytes = 0

    def __init__(self, spec, cfg):
        if cfg.S != 1:
            raise ValueError("embedding stage requires one device slice")
        self.spec, self.cfg = spec, cfg
        at = 0

        def alloc(size):
            nonlocal at
            start = -(-at // 64) * 64
            at = start + size
            return start

        self.text_at = alloc(4 * (spec.text_vocab + 1) * spec.hidden)
        self.audio_at = tuple(alloc(4 * (spec.audio_vocab + 1) * spec.hidden)
                              for _ in range(spec.audio_streams))
        self.table_bytes = 4 * spec.hidden * (spec.text_vocab + 1 +
                                            spec.audio_streams * (spec.audio_vocab + 1))
        self.voice_bytes = 4 * spec.voice_frames * spec.hidden
        self.voice_at = alloc(self.voice_bytes)
        self.io = {"codes": alloc(4 * (spec.audio_streams + 1)), "voice_index": alloc(4),
                   "output": alloc(4 * spec.hidden)}
        self.nbytes = -(-at // 4096) * 4096
        self.required_bytes = self.nbytes + waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
        self.weight_bytes = self.table_bytes + self.voice_bytes
        if self.required_bytes > min(cfg.DRAM_BYTES, 1 << 32):
            raise MemoryError(f"embedding image needs {self.required_bytes} bytes; "
                              f"device/address capacity is {min(cfg.DRAM_BYTES, 1 << 32)}")

    def descriptors(self):
        s = self.spec
        table = lambda at, rows: Tensor(at, (rows, s.hidden), (s.hidden, 1))
        return SimpleNamespace(text=table(self.text_at, s.text_vocab + 1),
                               audio=tuple(table(at, s.audio_vocab + 1) for at in self.audio_at),
                               voice=table(self.voice_at, s.voice_frames),
                               codes=Tensor(self.io["codes"], (s.audio_streams + 1,), (1,)),
                               voice_index=Tensor(self.io["voice_index"], (1,), (1,)),
                               output=Tensor(self.io["output"], (1, s.hidden), (s.hidden, 1)))

    def compile(self):
        programs = {}
        for name, kernel in (("codes", _embed_codes), ("voice", _voice_frame)):
            if name == "voice" and self.spec.voice_frames == 0:
                continue
            b = trace_fit(kernel, self.cfg, 0, lambda: {"m": self.descriptors()})
            program = b.finish()
            if len(program) * 8 > self.cfg.IMEM_WORDS:
                raise CompileError(f"{name} embeddings need {len(program) * 8} instruction words; "
                                   f"device has {self.cfg.IMEM_WORDS}")
            if b.run_args:
                raise CompileError("embedding selection must not use host row-address arguments")
            programs[name] = [program]
        return programs

    def build(self, tables, voices=None):
        tables = tables() if callable(tables) else tables
        if len(tables) != self.spec.audio_streams + 1:
            raise ValueError("expected text table followed by every audio embedding table")
        sizes = [self.spec.text_vocab] + [self.spec.audio_vocab] * self.spec.audio_streams
        checked = [T._checked(table, (rows, self.spec.hidden), "embedding table")
                   for table, rows in zip(tables, sizes)]
        voice = None
        if self.spec.voice_frames:
            voices = voices() if callable(voices) else voices
            voice = T._checked(voices, (self.spec.voice_frames, self.spec.hidden), "voice embeddings")
        data = np.zeros(self.nbytes, np.uint8)
        for at, table in zip((self.text_at, *self.audio_at), checked):
            # Native -1 means a positive zero vector. All original rows retain
            # their exact FP32 bits; the device computes token+1 before lookup.
            raw = table.view(np.uint8).reshape(-1)
            start = at + 4 * self.spec.hidden
            data[start:start + raw.size] = raw
        if voice is not None:
            raw = voice.view(np.uint8).reshape(-1)
            data[self.voice_at:self.voice_at + raw.size] = raw
        return [data]


class EmbeddingEngine:
    """Resident output stage. step()/step_voice() never read activations on host."""

    def __init__(self, tables, *, voices=None, spec=None, cfg=None, backend="isa"):
        self.spec = spec or EmbeddingSpec()
        self._closed = self._failed = self._ready = False
        self.poss, self.last_stats = [0], {}
        if backend == "board":
            raise ValueError("embedding board execution requires a qualified backend factory with CAP_GEN")
        if backend != "isa" and not callable(backend):
            raise ValueError("backend must be isa or a qualified backend factory")
        if cfg is None:
            if backend != "isa":
                raise ValueError("a physical backend requires its measured device config")
            cfg = Config(S=1, D=128, MCOLS=4, DRAM_BYTES=1 << 40)
            cfg = replace(cfg, DRAM_BYTES=EmbeddingImage(self.spec, cfg).required_bytes)
        self.cfg, self.image = cfg, EmbeddingImage(self.spec, cfg)
        self.programs = self.image.compile()  # Capacity before lazy checkpoint reads.
        images = self.image.build(tables, voices)
        self.backend = G.IsaBackend(cfg, images, adopt=True) if backend == "isa" else backend(cfg, images)
        try:
            if not getattr(self.backend, "generates", False):
                raise ValueError("embedding backend requires indexed RLD execution (CAP_GEN)")
            attach = getattr(self.backend, "attach", None)
            if attach is not None:
                attach(self)
        except BaseException:
            self.close()
            raise

    def _check_ready(self):
        if self._closed:
            raise RuntimeError("embedding engine is closed")
        if self._failed:
            raise RuntimeError("previous embedding step failed; reset required")

    def _run(self, mode, value):
        self._check_ready()
        started = time.perf_counter()
        self._ready = False
        try:
            self.backend.write(0, self.image.io["codes" if mode == "codes" else "voice_index"], value)
            start = getattr(self.backend, "start", None)
            if start is not None:
                start(self.programs[mode])
                stats = self.backend.wait()
            else:
                stats = self.backend.run(self.programs[mode])
        except BaseException:
            self._failed = True
            raise
        self._ready = True
        self.poss[0] += 1
        self.last_stats = {**(stats or {}), "mode": mode, "frame": self.poss[0] - 1,
                           "wall_seconds": time.perf_counter() - started}

    def step(self, codes):
        self._check_ready()
        values = np.asarray(codes, dtype=object)
        if values.shape != (self.spec.audio_streams + 1,):
            raise ValueError("codes must contain text ID followed by every audio ID")
        result = []
        for index, value in enumerate(values):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("code IDs must be integers, not booleans")
            try:
                token = operator.index(value)
            except TypeError as error:
                raise ValueError("code IDs must be integral") from error
            limit = self.spec.text_vocab if index == 0 else self.spec.audio_vocab
            if not -1 <= token < limit:
                raise ValueError(f"code ID{index} outside -1..{limit - 1}")
            result.append(token)
        self._run("codes", np.asarray(result, np.float32))

    def step_voice(self, index):
        self._check_ready()
        if isinstance(index, (bool, np.bool_)):
            raise ValueError("voice index must be integral")
        try:
            index = operator.index(index)
        except TypeError as error:
            raise ValueError("voice index must be integral") from error
        if not 0 <= index < self.spec.voice_frames:
            raise ValueError("voice index outside resident prompt")
        self._run("voice", np.asarray([index], np.float32))

    @property
    def output_address(self):
        self._check_ready()
        if not self._ready:
            raise RuntimeError("no completed embedding frame")
        return self.image.io["output"]

    def read_output(self):
        raw = self.backend.read(0, self.output_address, 4 * self.spec.hidden)
        result = np.asarray(raw, np.uint8).view(np.float32).copy()
        if result.shape != (self.spec.hidden,) or not np.isfinite(result).all():
            raise RuntimeError("invalid device embedding output")
        return result

    def reset(self):
        if self._closed:
            raise RuntimeError("embedding engine is closed")
        board = getattr(self.backend, "board", None)
        if self._failed and board is not None:
            board.stop()
        self._failed = self._ready = False
        self.poss[0], self.last_stats = 0, {}

    def close(self):
        if not self._closed:
            self._closed = True
            close = getattr(self.backend, "close", None)
            if close is not None:
                close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
