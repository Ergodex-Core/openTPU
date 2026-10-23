"""Bounded PersonaPlex depth layers with device-selected weights and KV state.

Default native policy reproduces the pinned runtime's final-step ring-position
boundary (slot0 is masked on step15). The explicit causal policy is a diagnostic
correction, not native parity. Neither policy implements a full speech runtime.
"""
from dataclasses import dataclass, replace
import math
import time
from types import SimpleNamespace
import numpy as np
from .. import isa as I
from .. import language as ol
from .. import quant as Q
from .. import qcache as QC
from .. import waits
from ..compiler import (Affine, CompileError, KVDesc, QTensor, RunVar, Tensor,
                        arg_reg, k_scale_bytes, trace_fit)
from ..isasim import Config
from ..kernels.attention import Blocks, _attend_heads
from ..kernels.mlp import down_chunk, swiglu_down
from . import qwen3 as G
from .personaplex_audio_heads import _checked


@dataclass(frozen=True)
class DepthLayerSpec:
    hidden: int = 1024
    n_heads: int = 16
    ffn: int = 2816
    steps: int = 16
    layers: int = 1
    eps: float = 1e-8
    cache_policy: str = "native"

    def __post_init__(self):
        for name in ("hidden", "n_heads", "ffn", "steps", "layers"):
            v = getattr(self, name)
            if type(v) is not int or not 0 < v < 1 << 24:
                raise ValueError(f"{name} must be a positive integer below2^24")
        if self.hidden % self.n_heads:
            raise ValueError("hidden must divide into heads")
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        if self.cache_policy not in ("native", "causal"):
            raise ValueError("cache_policy must be native or causal")
        if self.cache_policy == "native" and self.steps < 2:
            raise ValueError("native ring boundary requires at least2 steps")

    @property
    def head_dim(self):
        return self.hidden // self.n_heads


def _norm(x, gamma, eps):
    # Preserve native RMSNorm multiplication order: x * (alpha * rsqrt(var)).
    r = ol.rsqrt(ol.sum(x * x, axis=1) * (1.0 / x.cols) + eps)
    return x * (gamma.reshape(1, x.cols) * r[:, None])


def _attention(x, w, position, seq, spec, block):
    h, d = spec.n_heads, spec.head_dim
    normalized = ol.quantize(_norm(x, ol.load(w.g_in), spec.eps))
    k, v = ol.dot(normalized, w.wk), ol.dot(normalized, w.wv)
    queries = [ol.dot(normalized, w.wq[i * d:(i + 1) * d, :]) for i in range(h)]
    kh, vh = G._padded(k.reshape(h, d)), G._padded(v.reshape(h, d))
    for i in range(h):
        ol.kv_append(w.kv, i, position, kh[i:i + 1, :], None)
    def query(i):
        def prepare():
            ol.kv_append(w.kv, i, position, None, vh[i:i + 1, :])
            return G._padded(queries[i])
        return prepare
    output = ol.empty((1, spec.hidden))
    heads = output.reshape(h, d)
    def emit(i, acc, total):
        heads[i:i + 1, :].set(acc / total[:, None])
    _attend_heads([query(i) for i in range(h)], w.kv, list(range(h)), seq,
                  block, ol.LOG2E / math.sqrt(d), depth=3, ahead=2, emit=emit)
    return x + ol.dot(output, w.wo)


@ol.jit
def _step(m):
    b = ol.current()
    counter = ol.load(m.state)
    body, mark = b.stack[-1], len(b.stack[-1])
    b.run_words = {"depth_step": counter.base}
    position = RunVar("depth_step", m.spec.steps)
    seq = Blocks([(0, m.cap, Tensor(Affine(m.mask_at) + position * (4 * m.cap),
                                  (m.cap,), (1,)))])
    x = ol.load(m.hidden)
    for index in range(m.spec.layers):
        w = m.layer(index, position)
        x.set(_attention(x, w, position, seq, m.spec, m.cap))
        normalized = ol.quantize(_norm(x, ol.load(w.g_post), m.spec.eps))
        x.set(x + swiglu_down(normalized, w.wg, w.wu, w.wd, loop=True))
    ol.store(m.output, x)
    ol.store(m.state, counter + 1.0)
    # Compiler RunVar addresses need several integer multiples of the step. Load
    # every argument register ON DEVICE from the persistent counter, exactly like
    # generate._arguments, with explicit bounds for WAITC. No host arguments.
    arguments = list(b.run_args)
    body[mark:mark] = [I.rld(arg_reg(k), counter.base, mul=int(c),
                           bound=(0, int(c) * (m.spec.steps - 1)),
                           comment=f"device depth step *{c}")
                       for k, (v, c) in enumerate(arguments)]
    if any(v.name != "depth_step" for v, _ in arguments):
        raise CompileError("unexpected host variable in depth program")
    b.run_words = None
    b.run_args.clear()


class DepthLayerImage:
    batch = 1
    wformat = "int8"

    def __init__(self, spec, cfg):
        if cfg.S != 1 or spec.hidden % cfg.D or spec.ffn % cfg.D:
            raise ValueError("depth layers need one slice and hidden/ffn divisible by D")
        if max(spec.hidden, spec.ffn) > cfg.ACT_BLOCKS * cfg.D:
            raise ValueError("depth inner dimension exceeds ACT RAM")
        self.spec, self.cfg = spec, cfg
        self.cap = -(-spec.steps // cfg.D) * cfg.D
        self.dpad = -(-spec.head_dim // cfg.D) * cfg.D
        self.down_chunk = down_chunk(spec.ffn, cfg.D, "int8")
        self.mats = {name: (spec.hidden, spec.hidden) for name in ("wq", "wk", "wv", "wo")}
        self.mats.update(wg=(spec.ffn, spec.hidden), wu=(spec.ffn, spec.hidden))
        at = 0
        def alloc(size, alignment=128):
            nonlocal at
            base = -(-at // alignment) * alignment
            at = base + size
            return base
        self.io = {"hidden": alloc(4 * spec.hidden), "output": alloc(4 * spec.hidden), "state": alloc(4)}
        self.masks_at = alloc(4 * spec.steps * self.cap)
        self.norm_offsets = [(alloc(4 * spec.hidden), alloc(4 * spec.hidden)) for _ in range(spec.layers)]
        self.kv_heads, self.cache_regions = [], []
        for _ in range(spec.layers):
            start = at
            heads = {}
            for head in range(spec.n_heads):
                heads[head] = {"k": alloc(self.cap * self.dpad),
                               "ks": alloc(self.cap * k_scale_bytes(self.dpad, cfg.D)),
                               "vt": alloc(self.dpad * self.cap), "vs": alloc(4 * self.cap)}
            self.kv_heads.append(heads)
            self.cache_regions.append((start, at - start))
        self.kv_bytes = sum(size for _, size in self.cache_regions)
        # Repeated bank layout; KV and shared norms deliberately live outside it.
        self.weight_start = -(-at // 4096) * 4096
        bank_at = 0
        def balloc(size):
            nonlocal bank_at
            start = -(-bank_at // 128) * 128
            bank_at = start + size
            return start
        self.offsets = {}
        for name, (rows, cols) in self.mats.items():
            self.offsets[name] = (balloc(rows * Q.row_bytes(cols, "int8", cfg.D)),
                                  balloc(rows * Q.scale_bytes(cols, "int8", cfg.D)))
        self.offsets["wd"] = [(balloc(spec.hidden * Q.row_bytes(self.down_chunk, "int8", cfg.D)),
                               balloc(spec.hidden * Q.scale_bytes(self.down_chunk, "int8", cfg.D)))
                              for _ in range(spec.ffn // self.down_chunk)]
        self.bank_stride = -(-bank_at // 128) * 128
        self.bank_offsets = [self.weight_start + i * spec.steps * self.bank_stride for i in range(spec.layers)]
        self.weight_bytes = spec.layers * (spec.steps * self.bank_stride + 8 * spec.hidden)
        self.nbytes = -(-(self.weight_start + spec.layers * spec.steps * self.bank_stride) // 4096) * 4096
        self.required_bytes = self.nbytes + waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
        if self.required_bytes > min(cfg.DRAM_BYTES, 1 << 32):
            raise MemoryError(f"depth layer image needs {self.required_bytes} bytes; "
                              f"device/address capacity is {min(cfg.DRAM_BYTES, 1 << 32)}")

    def descriptors(self):
        s, D = self.spec, self.cfg.D
        def layer(index, position):
            base = Affine(self.bank_offsets[index]) + position * self.bank_stride
            norms = self.norm_offsets[index]
            w = SimpleNamespace(g_in=Tensor(norms[0], (s.hidden,), (1,)),
                                g_post=Tensor(norms[1], (s.hidden,), (1,)))
            for name, (rows, cols) in self.mats.items():
                da, sa = self.offsets[name]
                setattr(w, name, QTensor.weight(base + da, base + sa, rows, cols, "int8", D))
            parts = tuple(QTensor.weight(base + da, base + sa, s.hidden, self.down_chunk, "int8", D)
                          for da, sa in self.offsets["wd"])
            w.wd = QTensor(parts[0].data, parts[0].scale, (s.hidden, s.ffn), parts[0].rs,
                          parts[0].srs, D, parts=parts, pw=self.down_chunk, wf=parts[0].wf)
            w.kv = KVDesc(self.kv_heads[index], self.cap, self.dpad, D, 1, 0, dv=s.head_dim)
            return w
        return SimpleNamespace(spec=s, layer=layer, cap=self.cap, mask_at=self.masks_at,
            hidden=Tensor(self.io["hidden"], (1, s.hidden), (s.hidden, 1)),
            output=Tensor(self.io["output"], (1, s.hidden), (s.hidden, 1)),
            state=Tensor(self.io["state"], (1,), (1,)))

    def compile(self):
        b = trace_fit(_step, self.cfg, 0, lambda: {"m": self.descriptors()})
        program = b.finish()
        if program.run_args:
            raise CompileError("depth program must not require host step/address arguments")
        if len(program) * 8 > self.cfg.IMEM_WORDS:
            raise CompileError(f"depth program needs {len(program) * 8} instruction words; device has {self.cfg.IMEM_WORDS}")
        return [program]

    def build(self, layers):
        s = self.spec
        layers = layers() if callable(layers) else layers
        if len(layers) != s.layers:
            raise ValueError("incorrect number of depth layers")
        image = np.zeros(self.nbytes, np.uint8)
        def put(at, value):
            raw = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
            image[at:at + raw.size] = raw
        masks = np.full((s.steps, self.cap), -np.inf, np.float32)
        for step in range(s.steps):
            masks[step, :step + 1] = np.inf
        if s.cache_policy == "native":
            masks[-1, 0] = -np.inf  # pinned RingKVCache's full-capacity position bug
        put(self.masks_at, masks)
        for index in range(s.layers):
            layer = layers[index]
            for name, at in zip(("norm1", "norm2"), self.norm_offsets[index]):
                put(at, _checked(layer[name], (s.hidden,), name))
            steps = layer["steps"]
            if len(steps) != s.steps:
                raise ValueError("incorrect number of depth step banks")
            for step in range(s.steps):
                bank = steps[step]
                base = self.bank_offsets[index] + step * self.bank_stride
                batch = QC.Batch()
                for name, shape in self.mats.items():
                    value = _checked(bank[name], shape, name)
                    da, sa = (base + a for a in self.offsets[name])
                    batch.submit(value, "int8", self.cfg.D,
                                 lambda q, scale, da=da, sa=sa: (put(da, q), put(sa, scale)))
                value = _checked(bank["wd"], (s.hidden, s.ffn), "wd")
                addresses = [(base + da, base + sa) for da, sa in self.offsets["wd"]]
                def down(q, scale, addresses=addresses):
                    for (da, sa), (codes, scales) in zip(addresses, Q.column_parts(
                            q, scale, "int8", s.ffn, self.down_chunk, self.cfg.D)):
                        put(da, codes); put(sa, scales)
                batch.submit(value, "int8", self.cfg.D, down)
                batch.wait()
        return [image]


class DepthLayerEngine:
    def __init__(self, layers, *, spec=None, cfg=None, backend="isa"):
        self.spec = spec or DepthLayerSpec()
        self._closed = self._failed = self._ready = False
        self.poss, self.last_stats = [0], {}
        if backend == "board":
            raise ValueError("depth layer board execution requires a qualified backend factory with CAP_GEN")
        if backend != "isa" and not callable(backend):
            raise ValueError("backend must be isa or a qualified backend factory")
        if cfg is None:
            if backend != "isa":
                raise ValueError("physical backend requires its measured device config")
            cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                         TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True, DRAM_BYTES=1 << 40)
            cfg = replace(cfg, DRAM_BYTES=DepthLayerImage(self.spec, cfg).required_bytes)
        self.cfg, self.image = cfg, DepthLayerImage(self.spec, cfg)
        self.program = self.image.compile()
        images = self.image.build(layers)
        self.backend = G.IsaBackend(cfg, images, adopt=True) if backend == "isa" else backend(cfg, images)
        try:
            if not getattr(self.backend, "generates", False):
                raise ValueError("depth layer requires RLD execution (CAP_GEN)")
            attach = getattr(self.backend, "attach", None)
            if attach is not None:
                attach(self)
        except BaseException:
            self.close(); raise

    @property
    def pos(self):
        return self.poss[0]

    def _check(self):
        if self._closed:
            raise RuntimeError("depth layer engine is closed")
        if self._failed:
            raise RuntimeError("previous depth layer failed; reset required")

    def step(self, hidden):
        self._check()
        if self.pos >= self.spec.steps:
            raise RuntimeError("depth pass is full; reset before next temporal frame")
        value = _checked(hidden, (self.spec.hidden,), "depth input")
        self._ready = False
        started = time.perf_counter()
        try:
            self.backend.write(0, self.image.io["hidden"], value)
            start = getattr(self.backend, "start", None)
            if start is not None:
                start(self.program); stats = self.backend.wait()
            else:
                stats = self.backend.run(self.program)
        except BaseException:
            self._failed = True; raise
        self._ready = True
        self.poss[0] += 1
        self.last_stats = {**(stats or {}), "step": self.pos - 1,
                           "wall_seconds": time.perf_counter() - started}

    @property
    def output_address(self):
        self._check()
        if not self._ready:
            raise RuntimeError("no completed depth layer request")
        return self.image.io["output"]

    def read_output(self):
        raw = self.backend.read(0, self.output_address, 4 * self.spec.hidden)
        value = np.asarray(raw, np.uint8).view(np.float32).copy()
        if value.shape != (self.spec.hidden,) or not np.isfinite(value).all():
            raise RuntimeError("invalid device depth output")
        return value

    def read_counter(self):
        self._check()
        raw = self.backend.read(0, self.image.io["state"], 4)
        value = np.asarray(raw, np.uint8).view(np.float32)
        if value.shape != (1,) or not np.isfinite(value[0]) or value[0] != int(value[0]) or not 0 <= value[0] <= self.spec.steps:
            raise RuntimeError("invalid device depth counter")
        return int(value[0])

    def read_cache(self, layer=0):
        self._check()
        if type(layer) is not int or not 0 <= layer < self.spec.layers:
            raise ValueError("invalid depth cache layer")
        at, size = self.image.cache_regions[layer]
        return np.asarray(self.backend.read(0, at, size), np.uint8).copy()

    def reset(self):
        if self._closed:
            raise RuntimeError("depth layer engine is closed")
        board = getattr(self.backend, "board", None)
        if self._failed and board is not None:
            board.stop()
        try:
            self.backend.write(0, self.image.io["state"], np.zeros(1, np.float32))
        except BaseException:
            self._failed = True; raise
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
