"""PersonaPlex's temporal transformer on the openTPU ISA / FPGA.

The host supplies the sum of text/audio embeddings, one frame at a time. All
temporal transformer layers, attention, KV updates and the final RMSNorm execute
on the selected backend. The returned hidden vector feeds the text head and
depformer on the host. This module does not implement the codec or depformer.

The implementation reuses the proven Llama-style layout and kernels with no text
vocabulary or embedding table. PersonaPlex rotates adjacent Q/K pairs; permuting
each Q/K head's projection rows to [even, odd] makes the existing rotate-half
kernel equivalent. V and the output projection are unchanged.

Context is deliberately bounded. ``step`` rejects a frame before it would exceed
``cap`` or the model's 3000-frame context; it never silently wraps or truncates KV.
This is a functional port, not a claim of real-time performance on burst200.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np

from .. import fp32 as F
from .. import language as ol
from .. import qcache as QC
from .. import quant as Q
from .. import waits
from ..compiler import CompileError, arg_words, trace_fit
from ..isasim import Config
from ..kernels.attention import _attend_heads
from ..kernels.lib import rmsnorm
from . import qwen3 as G


@dataclass(frozen=True)
class Spec:
    hidden: int = 4096
    layers: int = 32
    n_heads: int = 32
    ffn: int = 11264
    eps: float = 1e-8
    rope_base: float = 10000.0
    context: int = 3000

    def __post_init__(self):
        for name in ("hidden", "layers", "n_heads", "ffn", "context"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden % self.n_heads or self.head_dim % 2:
            raise ValueError("hidden must divide into heads with an even head_dim")
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not np.isfinite(self.rope_base) or self.rope_base <= 1:
            raise ValueError("rope_base must be finite and greater than one")

    @property
    def head_dim(self):
        return self.hidden // self.n_heads

    def _core(self):
        return G.Spec(self.hidden, self.layers, self.n_heads, self.n_heads,
                      self.head_dim, self.ffn, 0, eps=self.eps,
                      theta=self.rope_base, tied=False, qk_norm=False,
                      qk_smooth=False)


def _checked(a, shape, name):
    a = np.asarray(a, dtype=np.float32)
    if a.shape != tuple(shape):
        raise ValueError(f"{name}: shape {a.shape}, expected {tuple(shape)}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: non-finite values")
    return F.ftz(a)


def _split_pairs(w, spec):
    """Reorder projection output lanes, not input columns or entire head order."""
    a = w.reshape(spec.n_heads, spec.head_dim, spec.hidden)
    return np.concatenate((a[:, 0::2], a[:, 1::2]), axis=1).reshape(w.shape)


def _attention(x, lw, c, s, pos, spec, block):
    """MHA with completed heads emitted immediately to bound live TMEM.

    The general Llama helper keeps every head's softmax state until all heads
    finish. PersonaPlex has 32 KV heads, so release that state as each head is
    emitted. K, V and Q projections retain the existing overlapping schedule.
    """
    d, nh = spec.head_dim, spec.n_kv
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_in), spec.eps))
    k, v = ol.dot(xs, lw.wk), ol.dot(xs, lw.wv)
    queries = [ol.dot(xs, lw.wq[j * d:(j + 1) * d, :]) for j in range(nh)]
    kh = G._rope_padded(k.reshape(nh, d), c, s)
    vh = v.reshape(nh, d)
    kpos = pos.pos
    for j in range(nh):
        ol.kv_append(lw.kv, j, kpos, kh[j:j + 1, :], None)

    def query(j):
        def prepare():
            ol.kv_append(lw.kv, j, kpos, None, vh[j:j + 1, :])
            return G._rope_padded(queries[j], c, s)
        return prepare

    out = ol.empty([1, nh * d])
    heads = out.reshape(nh, d)

    def emit(j, acc, total):
        heads[j:j + 1, :].set(acc / total[:, None])

    _attend_heads([query(j) for j in range(nh)], lw.kv, list(range(nh)),
                  pos.bucket, block, ol.LOG2E / math.sqrt(d), depth=3,
                  ahead=2, emit=emit)
    return x + ol.dot(out, lw.wo)


@ol.jit
def _step(m, pos, block):
    # Host inputs also when pos is a RunPos: no text embedding lookup.
    x, c, s = ol.load(m.x), ol.load(m.cos), ol.load(m.sin)
    for lw in G._layers(m):
        lw.mlp_loop = True
        x.set(_attention(x, lw, c, s, pos, m.spec, block))
        x.set(G._mlp(x, lw, m.spec))
    ol.store(m.x, rmsnorm(x, ol.load(m.g_final), m.spec.eps))


class Image(G.Image):
    """Layout-only construction is cheap and validates fit before reading weights.

    ``nbytes`` includes weights, KV, I/O and a causal mask. ``required_bytes``
    additionally reserves the driver's program area. ``cap`` is the physical
    attention-block-aligned KV capacity; ``frame_limit`` is the logical limit.
    ``weight_code_bytes`` is the port-B matrix payload; ``weight_bytes`` also
    includes scales and norms read through the other memory port. ``formats``
    explicitly selects per-layer INT4/INT8 kinds; the default ignores OTPU_FORMATS.
    """

    def __init__(self, spec: Spec, cfg: Config, cap: int = 256,
                 wformat: str = "int4", formats: str = ""):
        if cfg.S != 1:
            raise ValueError("PersonaPlex temporal currently supports one FPGA slice")
        if wformat not in ("int4", "int8"):
            raise ValueError("wformat must be int4 or int8")
        if not isinstance(formats, str):
            raise ValueError("formats must be an explicit string; use '' for uniform weights")
        rules = G.FM.rules(formats, G.KINDS)
        if any(rule[-1] not in ("int4", "int8") for rule in rules):
            raise ValueError("PersonaPlex formats support only int4 and int8")
        if isinstance(cap, bool) or not isinstance(cap, int) or not 0 < cap <= spec.context:
            raise ValueError(f"cap must be within 1..{spec.context} frames")
        if cfg.D > 256 or 256 % cfg.D:
            raise ValueError("the attention block D must divide 256")
        self.temporal_spec, self.frame_limit = spec, cap
        aligned_cap = -(-cap // cfg.D) * cfg.D
        # Match the 256-token V^T tiles on longer contexts so the attention
        # kernel can loop over blocks. Half-tile blocks disable that loop and
        # would overflow IMEM when many heads attend over a long conversation.
        self.block = 256 if aligned_cap >= 256 else cfg.D
        aligned_cap = -(-cap // self.block) * self.block
        super().__init__(spec._core(), cfg, aligned_cap, wformat=wformat,
                         head_format=wformat, lookup=False, formats=formats)
        self.zmask = (self.nbytes + 63) // 64 * 64
        self.nbytes = (self.zmask + 4 * (self.cap + self.block) + 63) // 64 * 64
        self.required_bytes = (self.nbytes + 4095) // 4096 * 4096 + \
            waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
        self.weight_bytes = 4 * spec.hidden * (2 * spec.layers + 1)
        self.weight_code_bytes = 0
        # Count the exact matrix+scale layouts, excluding alignment, KV and I/O.
        for key in self.lf:
            mf = self.mats_formats(key)
            for name, (n, k) in self.mats.items():
                self.weight_code_bytes += n * Q.row_bytes(k, mf[name], cfg.D)
                self.weight_bytes += n * (
                    Q.row_bytes(k, mf[name], cfg.D) + Q.scale_bytes(k, mf[name], cfg.D))
            fd = mf["wd"]
            c = self.dchunks[fd]
            self.weight_code_bytes += (spec.ffn // c) * spec.hidden * Q.row_bytes(c, fd, cfg.D)
            self.weight_bytes += (spec.ffn // c) * spec.hidden * (
                Q.row_bytes(c, fd, cfg.D) + Q.scale_bytes(c, fd, cfg.D))
        if self.required_bytes > cfg.DRAM_BYTES:
            raise MemoryError(f"temporal image and program area need {self.required_bytes} "
                              f"bytes, device has {cfg.DRAM_BYTES}; reduce cap or use a "
                              "larger-memory FPGA image")

    def build(self, layer_weights, final_norm):
        """Quantize [out,in] weights; layer mappings may decode arrays lazily.

        Each layer has norm1, norm2, wq, wk, wv, wo, wg, wu, wd. A layer's
        quantization completes before the next layer is materialized. No MLX
        runtime or raw MLX-packed weight execution is used.
        """
        spec, D = self.temporal_spec, self.cfg.D
        if len(layer_weights) != spec.layers:
            raise ValueError(f"expected {spec.layers} weight layers, got {len(layer_weights)}")
        norm = _checked(final_norm, (spec.hidden,), "final_norm")
        data = np.zeros(self.nbytes, dtype=np.uint8)

        def put(addr, value):
            v = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
            data[addr:addr + v.size] = v

        put(self.io["gf"], norm)
        put(self.zmask, np.concatenate((np.full(self.cap, np.inf, np.float32),
                                       np.full(self.block, -np.inf, np.float32))))
        for i in range(spec.layers):
            lw = layer_weights[i]
            base, offsets = self._off(i).const, self.bofs[self.lf[i]]
            mf = self.mats_formats(self.lf[i])
            for src, dst in (("norm1", "g_in"), ("norm2", "g_post")):
                put(base + offsets[dst], _checked(lw[src], (spec.hidden,), f"layer {i} {src}"))
            qb = QC.Batch()
            for name, shape in self.mats.items():
                a = _checked(lw[name], shape, f"layer {i} {name}")
                if name in ("wq", "wk"):
                    a = _split_pairs(a, spec)
                da, sa = (base + v for v in offsets[name])
                qb.submit(a, mf[name], D,
                          lambda q, s, da=da, sa=sa: (put(da, q), put(sa, s)))
            a = _checked(lw["wd"], (spec.hidden, spec.ffn), f"layer {i} wd")
            addresses = [(base + da, base + sa) for da, sa in offsets["wd"]]
            fd = mf["wd"]
            chunk = self.dchunks[fd]

            def down(q, s, addresses=addresses, fd=fd, chunk=chunk):
                for (da, sa), (pq, ps) in zip(addresses, Q.column_parts(
                        q, s, fd, spec.ffn, chunk, D)):
                    put(da, pq)
                    put(sa, ps)

            qb.submit(a, fd, D, down)
            qb.wait()
        return [data]

    def compile_bucket(self, bucket):
        lo = (bucket - 1) * self.block
        if not 0 <= lo < self.frame_limit:
            raise ValueError(f"attention bucket {bucket} is outside the frame limit")
        b = trace_fit(_step, self.cfg, 0, lambda: {
            "m": self.descriptors(0),
            "pos": G.RunPos(bucket, self.block, lo, self.zmask, self.cap),
            "block": self.block})
        program = b.finish()
        if len(program) * 8 > self.cfg.IMEM_WORDS:
            raise CompileError(f"temporal step needs {len(program) * 8} instruction words; "
                               f"device has {self.cfg.IMEM_WORDS}")
        return [program], list(b.run_args)


class Engine:
    """Synchronous temporal frame execution, with persistent INT8 KV.

    ``backend`` is ``'isa'``, ``'board'`` or a factory ``(cfg, images) -> backend``.
    The board path requires the actual config obtained from Board.info(), and can
    receive the burst200 transport explicitly. It never substitutes a CPU model
    if the FPGA is unavailable. ISA timing is not hardware performance.
    """

    def __init__(self, spec: Spec, layer_weights, final_norm, *, cfg: Config | None = None,
                 backend="isa", cap: int = 256, wformat: str = "int4", transport=None,
                 formats: str = ""):
        if not (isinstance(backend, str) and backend in ("isa", "board")) and \
                not callable(backend):
            raise ValueError("backend must be 'isa', 'board', or a backend factory")
        if transport is not None and backend != "board":
            raise ValueError("transport is only valid with the board backend")
        self.spec, self.cap = spec, cap
        self.poss, self.last_stats = [0], {}
        self._failed = self._closed = False
        self._programs = {}
        if cfg is None:
            if backend != "isa":
                raise ValueError("the FPGA backend requires its measured device config")
            # A simulator configuration, never an assumed FPGA configuration.
            probe_cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                               TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                               DRAM_BYTES=1 << 40)
            probe = Image(spec, probe_cfg, cap, wformat, formats=formats)
            cfg = replace(probe_cfg, DRAM_BYTES=probe.required_bytes)
        self.cfg = cfg
        self.image = Image(spec, cfg, cap, wformat, formats=formats)
        # Catch hardware instruction/TMEM capacity errors before costly quantization.
        for bucket in range(1, (cap + self.image.block - 1) // self.image.block + 1):
            self._programs[bucket] = self.image.compile_bucket(bucket)
        images = self.image.build(layer_weights, final_norm)
        if backend == "isa":
            self.backend = G.IsaBackend(cfg, images, adopt=True)
        elif backend == "board":
            from ..host.board import BoardBackend
            self.backend = BoardBackend(cfg, images, transport=transport,
                                        model="PersonaPlex temporal")
        elif callable(backend):
            self.backend = backend(cfg, images)
        try:
            if not getattr(self.backend, "args", False):
                raise ValueError("temporal backend must support runtime program arguments")
            attach = getattr(self.backend, "attach", None)
            if attach is not None:
                attach(self)
        except BaseException:
            self.close()
            raise

    @property
    def pos(self):
        return self.poss[0]

    def reset(self):
        if self._closed:
            raise RuntimeError("engine is closed")
        if self._failed:
            board = getattr(self.backend, "board", None)
            if board is not None:
                # Complete/abort outstanding writes before a new conversation.
                # If this fails, retain the failed state instead of reusing KV.
                board.stop()
        # Future positions are causally masked, so previously written KV needs no
        # bulk clear. Every layer overwrites its current K/V before reading it.
        self.poss[0] = 0
        self._failed = False
        self.last_stats = {}

    def step(self, embedding):
        if self._closed:
            raise RuntimeError("engine is closed")
        if self._failed:
            raise RuntimeError("previous frame failed; reset the engine before continuing")
        if self.pos >= self.cap:
            raise RuntimeError(f"temporal context is full ({self.cap} frames); reset required; "
                               "sliding-window KV is not implemented")
        x = _checked(embedding, (self.spec.hidden,), "embedding")
        bucket = self.pos // self.image.block + 1
        if bucket not in self._programs:
            self._programs[bucket] = self.image.compile_bucket(bucket)
        programs, run_args = self._programs[bucket]
        cos, sin = G.rope_tables(self.image.spec, self.pos)
        t0 = time.perf_counter()
        try:
            for name, value in (("x", x), ("cos", cos), ("sin", sin)):
                self.backend.write(0, self.image.io[name], value)
            args = arg_words(run_args, G.RunPos.values(0, self.pos, block=self.image.block))
            start = getattr(self.backend, "start", None)
            if start is not None:
                # BoardBackend.run has no args keyword; its resident-program API
                # passes ARG registers through start and obtains counters at wait.
                start(programs, args=args)
                stats = self.backend.wait()
            else:
                stats = self.backend.run(programs, args=args)
            out = self.backend.read(0, self.image.io["x"], 4 * self.spec.hidden)
            y = np.asarray(out, np.uint8).view(np.float32).copy()
            if y.shape != (self.spec.hidden,) or not np.isfinite(y).all():
                raise RuntimeError("backend returned an invalid temporal hidden state")
        except BaseException:
            self._failed = True
            raise
        self.poss[0] += 1
        self.last_stats = {**(stats or {}), "frame": self.pos - 1,
                           "wall_seconds": time.perf_counter() - t0}
        return y

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
