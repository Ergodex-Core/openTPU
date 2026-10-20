"""Co-resident DSpark draft on the target's openTPU device.

The five-layer noncausal backbone, context K/V projections and shared target
INT8 output head run on the device. Optional device_fusion_markov also runs
INT8 context fc/FP32 hidden_norm and INT8 Markov_w2 on the device. Embedding
lookups, sequential argmax and confidence remain on the host. Draft weights
use openTPU INT4/W4A8 arithmetic; this is not the floating-point DSpark oracle.
Exact greedy target verification remains responsible for every output token.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .. import fp32 as F, language as ol, qcache as QC, quant as QT, waits
from ..compiler import CompileError, RunVar, Tensor, arg_words, trace_fit
from ..kernels.attention import Additive, TileSeq, _attend_heads
from ..kernels.lib import rmsnorm
from . import qwen3 as Q, tiled as TL
from . import dspark as DS
from .nanbeige import _norm_rows_inplace


@dataclass(frozen=True)
class ProjectedContext:
    """Committed CPU fc+hidden_norm results; query K/V never become context.

    Immutable arrays make cache-prefix comparisons reliable after append or
    truncate. Device K/V are generated lazily when this context is drafted.
    """
    positions: np.ndarray
    hidden: np.ndarray

    def __post_init__(self):
        p = np.array(self.positions, dtype=np.int64, copy=True)
        h = np.array(self.hidden, dtype=np.float32, copy=True)
        if p.ndim != 1 or h.ndim != 2 or len(p) != len(h):
            raise ValueError("DSpark projected context shape mismatch")
        if np.any(p < 0) or np.any(np.diff(p) <= 0) or not np.all(np.isfinite(h)):
            raise ValueError("Invalid DSpark projected context")
        p.setflags(write=False)
        h.setflags(write=False)
        object.__setattr__(self, "positions", p)
        object.__setattr__(self, "hidden", h)

    def append(self, other):
        if self.hidden.shape[1:] != other.hidden.shape[1:]:
            raise ValueError("DSpark projected context widths differ")
        if len(self.positions) and len(other.positions) and self.positions[-1] >= other.positions[0]:
            raise ValueError("DSpark context append must strictly advance positions")
        return ProjectedContext(np.concatenate((self.positions, other.positions)),
                                np.concatenate((self.hidden, other.hidden)))

    def truncate(self, position):
        n = int(np.searchsorted(self.positions, position))
        return ProjectedContext(self.positions[:n], self.hidden[:n])


class _KernelSpec(Q.Spec):
    def check(self, cfg):
        # The tiled attention explicitly splits a GQA group into MCOLS parts.
        super().check(replace(cfg, MCOLS=max(cfg.MCOLS, self.n_q // self.n_kv)))


def kernel_spec(spec):
    return _KernelSpec(hidden=spec.hidden, layers=spec.layers, n_q=spec.n_q,
                       n_kv=spec.n_kv, head_dim=spec.head_dim, ffn=spec.ffn,
                       vocab=spec.draft_vocab, eps=spec.eps, theta=spec.theta,
                       tied=False, qk_norm=True, qk_smooth=False)


class Image(Q.Image):
    """Draft-owned [base, nbytes), with the target head referenced read-only.

    ``cap`` includes temporary query slots past the target's last position.
    No private LM head, embedding table or generic 256 MiB staging reservation.
    """
    def __init__(self, spec, cfg, cap, rows=7, base=0, head=None,
                 device_fusion_markov=False):
        if cfg.S != 1 or cfg.D != 128:
            raise ValueError("DSpark FPGA currently requires one D=128 slice")
        if not spec.neox_rope or spec.draft_vocab != spec.vocab:
            raise ValueError("DSpark FPGA requires full vocabulary and rotate-half RoPE")
        if head is None or len(head) != 2:
            raise ValueError("DSpark FPGA requires an explicit shared target head")
        if not 1 <= rows <= cfg.act_rows:
            raise ValueError("DSpark query rows exceed ACT_ROWS")
        ks = kernel_spec(spec)
        # Q.Image lays out a private head last. Discard that allocation before
        # checking real capacity, then use the target's absolute head addresses.
        super().__init__(ks, replace(cfg, DRAM_BYTES=max(cfg.DRAM_BYTES, 1 << 40)),
                         cap, rows=rows, wformat="int4", head_format="int8", base=base)
        self.cfg, self.draft_spec = cfg, spec
        self.nbytes, self.head = self.head[0], tuple(head)
        self.backbone_nbytes = self.nbytes
        self.device_fusion_markov = bool(device_fusion_markov)
        self.extra = {}
        if self.device_fusion_markov:
            if spec.topk is not None:
                raise ValueError("Device Markov projection currently requires full-vocabulary logits")
            K = len(spec.target_layers) * spec.target_hidden
            if K % cfg.D or K > cfg.ACT_BLOCKS * cfg.D:
                raise ValueError("DSpark fusion width does not fit the device activation memory")
            self.fusion_rows = min(rows, 4, (cfg.TMEM_WORDS - 64) // K)
            if self.fusion_rows < 1:
                raise ValueError("DSpark fusion input does not fit TMEM")
            self.markov_width = (spec.markov_rank + cfg.D - 1) // cfg.D * cfg.D
            b = Q._Bump(self.nbytes)
            self.fc = (b.alloc(ks.hidden * QT.row_bytes(K, "int8", cfg.D)),
                       b.alloc(ks.hidden * QT.scale_bytes(K, "int8", cfg.D)))
            self.markov_w2 = (
                b.alloc(ks.vocab * QT.row_bytes(self.markov_width, "int8", cfg.D)),
                b.alloc(ks.vocab * QT.scale_bytes(self.markov_width, "int8", cfg.D)))
            self.extra = {
                "f_in": b.alloc(4 * self.fusion_rows * K),
                "f_out": b.alloc(4 * self.fusion_rows * ks.hidden),
                "hidden_norm": b.alloc(4 * ks.hidden),
                "markov_in": b.alloc(4 * self.markov_width),
                "markov_out": b.alloc(4 * ks.vocab),
            }
            self.nbytes = b.next
        # Seven query rows use only the first seven rows of ACT_ROWS' attention
        # scratch. Its unused tail holds a run-time mask without growing layout.
        self.mask_at = self.io["o"] + rows * ks.n_q * ks.head_dim * 4
        self.mask_bytes = (cfg.act_rows - rows) * ks.n_q * ks.head_dim * 4
        if self.nbytes > cfg.DRAM_BYTES:
            raise MemoryError(f"DSpark co-resident image ends at {self.nbytes}; "
                              f"device exposes {cfg.DRAM_BYTES}")

    def upload(self, ref, backend):
        """Quantize and DMA matrices without allocating a base-sized host array."""
        D = self.cfg.D
        qb = QC.Batch()

        def put(addr, data):
            backend.write(0, addr, np.ascontiguousarray(data))

        def matrix(pair, value):
            qb.submit(value, "int4", D,
                      lambda q, sc, a=pair: (put(a[0], q), put(a[1], sc)))

        put(self.io["gf"], F.ftz(ref._weight("norm.weight", (self.spec.hidden,))))
        H, d = self.spec.hidden, self.spec.head_dim
        for li in range(self.spec.layers):
            b, lo = self._off(li).const, self.bofs[self.lf[li]]
            # Run-time masks read a complete bucket, including future slots.
            # Those slots must have finite scales before first use.
            first = lo["kv"][0]["k"]
            last = lo["kv"][-1]["vs"] + self.cap * 4
            put(b + first, np.zeros(last - first, np.uint8))
            for key, name, size in (("g_in", "input_layernorm", H),
                                    ("g_post", "post_attention_layernorm", H),
                                    ("qn", "self_attn.q_norm", d),
                                    ("kn", "self_attn.k_norm", d)):
                put(b + lo[key], F.ftz(ref._layer(li, name, (size,))))
            names = {"wq": "self_attn.q_proj", "wk": "self_attn.k_proj",
                     "wv": "self_attn.v_proj", "wo": "self_attn.o_proj",
                     "wg": "mlp.gate_proj", "wu": "mlp.up_proj"}
            for key, name in names.items():
                shape = self.mats[key]
                matrix(tuple(b + a for a in lo[key]), ref._layer(li, name, shape))
            value = ref._layer(li, "mlp.down_proj", (H, self.spec.ffn))
            addrs = [(b + a, b + sc) for a, sc in lo["wd"]]

            def down(q, sc, addrs=addrs):
                for pair, (pq, ps) in zip(addrs, QT.column_parts(
                        q, sc, "int4", self.spec.ffn, self.dchunk, D)):
                    put(pair[0], pq)
                    put(pair[1], ps)
            qb.submit(value, "int4", D, down)
        if self.device_fusion_markov:
            K = len(self.draft_spec.target_layers) * self.draft_spec.target_hidden
            for pair, value, width in ((self.fc, ref._weight("fc.weight", (H, K)), K),
                                (self.markov_w2, ref._weight("markov_head.markov_w2.weight",
                                 (self.spec.vocab, self.draft_spec.markov_rank)), self.markov_width)):
                if value.shape[1] < width:
                    value = np.pad(value, ((0, 0), (0, width - value.shape[1])))
                qb.submit(value, "int8", D,
                          lambda q, sc, a=pair: (put(a[0], q), put(a[1], sc)))
            put(self.extra["hidden_norm"], F.ftz(ref._weight("hidden_norm.weight", (H,))))
        qb.wait()

    def auxiliary_descriptors(self):
        if not self.device_fusion_markov:
            raise ValueError("Device fusion/Markov is disabled")
        from types import SimpleNamespace
        H, V, D = self.spec.hidden, self.spec.vocab, self.cfg.D
        K = len(self.draft_spec.target_layers) * self.draft_spec.target_hidden
        return SimpleNamespace(
            fc=Q._qdesc(*self.fc, H, K, D),
            w2=Q._qdesc(*self.markov_w2, V, self.markov_width, D),
            f_in=Q._tdesc(self.extra["f_in"], (self.fusion_rows, K)),
            f_out=Q._tdesc(self.extra["f_out"], (self.fusion_rows, H)),
            norm=Q._tdesc(self.extra["hidden_norm"], (H,)),
            markov_in=Q._tdesc(self.extra["markov_in"], (1, self.markov_width)),
            markov_out=Q._tdesc(self.extra["markov_out"], (1, V)),
            logits=self.io["logits"], vocab=V, eps=self.spec.eps,
            scale=self.draft_spec.logit_scale)

    def compile_fusion(self, rows):
        if not self.device_fusion_markov or not 1 <= rows <= self.fusion_rows:
            raise ValueError("Invalid DSpark fusion row count")
        progs = [trace_fit(fusion_kernel, self.cfg, 0, lambda: {
            "m": self.auxiliary_descriptors(), "T": rows}).finish()]
        self._check_programs(progs)
        return progs

    def compile_markov(self, row=None):
        if row is not None and not 0 <= row < self.rows:
            raise ValueError("Invalid DSpark Markov logit row")
        pos = RunVar("row", bound=self.rows) if row is None else row
        builder = trace_fit(markov_kernel, self.cfg, 0, lambda: {
            "m": self.auxiliary_descriptors(), "row": pos})
        progs = [builder.finish()]
        self._check_programs(progs)
        return progs, list(builder.run_args)

    def compile_query(self, position, queries=7, predictions=3, block=Q.ATTN_BLOCK):
        if not 0 <= position or position + queries > self.cap:
            raise ValueError("DSpark query positions exceed draft cache capacity")
        if not 1 <= queries <= self.rows or not 0 <= predictions <= queries:
            raise ValueError("Invalid DSpark query/prediction rows")
        progs = [trace_fit(query_kernel, self.cfg, 0, lambda: {
            "m": self.descriptors(0), "position": position, "T": queries,
            "predictions": predictions, "block": block}).finish()]
        self._check_programs(progs)
        return progs

    def compile_context(self, position, rows):
        if not 0 <= position or position + rows > self.cap or not 1 <= rows <= self.rows:
            raise ValueError("Invalid DSpark context rows")
        progs = [trace_fit(context_kernel, self.cfg, 0, lambda: {
            "m": self.descriptors(0), "position": position, "T": rows}).finish()]
        self._check_programs(progs)
        return progs

    def runtime_ok(self, position, queries, block=Q.ATTN_BLOCK):
        # V^T storage is tiled in 256 positions. A run-time append cannot cross
        # that tile's boundary; rare crossing queries retain the static path.
        return (block == 256 and position // 256 == (position + queries - 1) // 256
                and self.mask_bytes >= 4 * self.cfg.MCOLS * (block + 1))

    def compile_query_runtime(self, base, queries=7, predictions=3, block=Q.ATTN_BLOCK):
        if base % 256 or not self.runtime_ok(base, queries, block):
            raise ValueError("DSpark run-time query must fit one 256-position tile")
        position = RunVar("position", bound=min(256, self.cap - base) - queries + 1) + base
        seq = TileSeq(base // block, [(base, min(block, self.cap - base),
                                      Additive(self.mask_at, block))])
        builder = trace_fit(query_kernel, self.cfg, 0, lambda: {
            "m": self.descriptors(0), "position": position, "T": queries,
            "predictions": predictions, "block": block, "sequence": seq})
        progs = [builder.finish()]
        self._check_programs(progs)
        return progs, list(builder.run_args)

    def compile_context_runtime(self, base, rows):
        if base % 256 or not 1 <= rows <= min(self.rows, self.cap - base, 256):
            raise ValueError("Invalid DSpark run-time context tile")
        position = RunVar("position", bound=min(256, self.cap - base) - rows + 1) + base
        builder = trace_fit(context_kernel, self.cfg, 0, lambda: {
            "m": self.descriptors(0), "position": position, "T": rows})
        progs = [builder.finish()]
        self._check_programs(progs)
        return progs, list(builder.run_args)

    def _check_programs(self, progs):
        if any(len(p) * 8 > self.cfg.IMEM_WORDS for p in progs):
            raise CompileError("DSpark program exceeds device IMEM")


def _attention(res, lw, qn, kn, c, sn, position, T, spec, block, parts, oscr, sequence=None):
    """Tiled Qwen attention with a single noncausal extent for all query rows."""
    d, G = spec.head_dim, spec.n_q // spec.n_kv
    xs = TL._norm_quantize(res.x, lw.g_in, spec.eps)
    res.spill()
    kv = lw.kv
    heads = list(kv.owned_heads(spec.n_kv))
    gs = min(G, ol.mxu_columns())
    while gs % 2 == 0 and T * gs * d > ol.tmem_words() // TL.QPART:
        gs //= 2
    P = TL.packing(ol.mxu_columns(), gs)
    tiles = [(r0, min(P, T - r0)) for r0 in range(0, T, P)]
    gparts = [(g0, min(G, g0 + gs)) for g0 in range(0, G, gs)]
    if sequence is None:
        full, tail = divmod(position + T, block)
        seq = TileSeq(full, [] if tail == 0 else [(full * block, tail, None)])
    else:
        seq = sequence
    oscr = Tensor(oscr.base, (T, spec.n_q * d), (spec.n_q * d, 1))
    made, outputs = {}, {}
    entries, queries, kvheads = [], [], []

    def part(j, g0, g1):
        if g0 == 0:
            k = ol.dot(xs, lw.wk[j * d:(j + 1) * d, :])
            v = ol.dot(xs, lw.wv[j * d:(j + 1) * d, :])
            k = Q._rope_rows_padded(Q._norm_heads(k, kn, spec.eps), c, sn)
            ol.kv_append(kv, heads[j], position, k, v)
            del k, v
        w = g1 - g0
        q = ol.dot(xs, lw.wq[(j * G + g0) * d:(j * G + g1) * d, :])
        out = ol.empty([T * w, d], dense=True)
        view = out.reshape(T, w * d)
        for g in range(w):
            Q._rope_rows_padded(Q._norm_heads(q[:, g * d:(g + 1) * d], qn, spec.eps),
                                c, sn, out=view[:, g * d:(g + 1) * d])
        return out

    def get_query(j, g0, g1, r0, n):
        def emit():
            if (j, g0) not in made:
                made[j, g0] = part(j, g0, g1)
            q = made[j, g0]
            if r0 + n == T:
                made.pop((j, g0))
            w = g1 - g0
            return q[r0 * w:(r0 + n) * w, :]
        return emit

    for j in range(len(heads)):
        for g0, g1 in gparts:
            for r0, n in tiles:
                entries.append((j, g0, g1, r0, n))
                queries.append(get_query(j, g0, g1, r0, n))
                kvheads.append(heads[j])

    def emit(i, acc, den):
        j, g0, g1, r0, n = entries[i]
        w = g1 - g0
        if (j, g0) not in outputs:
            outputs[j, g0] = ol.empty([T, w * d])
        out = outputs[j, g0]
        for a in range(n):
            out[r0 + a, :].reshape(w, d).set(
                acc[a * w:(a + 1) * w, :] / den[a * w:(a + 1) * w][:, None])
        if r0 + n == T:
            h0 = (j * G + g0) * d
            ol.store(oscr[:, h0:h0 + w * d], outputs.pop((j, g0)))

    _attend_heads(queries, kv, kvheads, [seq] * len(queries), block,
                  ol.LOG2E / math.sqrt(d), depth=Q.ATTN_DEPTH, emit=emit)
    xs = None
    # All outputs must share one activation scale per row before W_o.
    out = ol.load(oscr)
    stationary = ol.quantize(out, free=True)
    del out
    TL._add_parts(res.back(), [stationary], [lw.wo], parts)


@ol.jit
def query_kernel(m, position, T, predictions, block=Q.ATTN_BLOCK, sequence=None):
    # Small long-lived tiles below x avoid fragmenting TMEM when x spills and
    # all seven 6144-wide attention outputs are loaded together.
    c, sn = ol.load(m.cosr[:T, :]), ol.load(m.sinr[:T, :])
    qn, kn = ol.empty([m.spec.head_dim]), ol.empty([m.spec.head_dim])
    x = ol.load(m.xr[:T, :])
    res = TL.Res(x, m.xr[:T, :])
    del x
    parts, sub = TL._widths(m, T)
    for lw in Q._layers(m):
        ol.load(lw.qn, out=qn)
        ol.load(lw.kn, out=kn)
        _attention(res, lw, qn, kn, c, sn, position, T, m.spec, block, parts, m.oscr, sequence)
        TL._mlp_tile(res.x, lw, m.spec, parts, sub(lw))
    _norm_rows_inplace(res.x, m.g_final, m.spec.eps)
    ol.store(m.xr[:T, :], res.x)
    if predictions:
        xs = ol.quantize(res.x[:predictions, :])
        chunk = min(Q.HEAD_CHUNK, ol.tmem_words() // max(8, predictions * 4))
        chunk = max(ol.block_size(), chunk // ol.block_size() * ol.block_size())
        for col in range(0, m.v_loc, chunk):
            n = min(chunk, m.v_loc - col)
            logits = ol.dot(xs, m.head[col:col + n, :])
            ol.store(m.logitsr[:predictions, col:col + n], logits)
            del logits


@ol.jit
def context_kernel(m, position, T):
    c, sn = ol.load(m.cosr[:T, :]), ol.load(m.sinr[:T, :])
    xs = ol.quantize(ol.load(m.xr[:T, :]), free=True)
    d = m.spec.head_dim
    for lw in Q._layers(m):
        kn = ol.load(lw.kn)
        for j, head in enumerate(lw.kv.owned_heads(m.spec.n_kv)):
            # Context is already fc+hidden_norm. Do not apply input_layernorm.
            k = ol.dot(xs, lw.wk[j * d:(j + 1) * d, :])
            v = ol.dot(xs, lw.wv[j * d:(j + 1) * d, :])
            k = Q._rope_rows_padded(Q._norm_heads(k, kn, m.spec.eps), c, sn)
            ol.kv_append(lw.kv, head, position, k, v)
            del k, v


@ol.jit
def fusion_kernel(m, T):
    # Four 15360-wide rows occupy 61440 words. Release them into ACT RAM
    # before allocating the projection output and FP32 normalization products.
    x = ol.load(m.f_in[:T, :])
    xs = ol.quantize(x, free=True)
    del x
    y = ol.dot(xs, m.fc)
    y.set(rmsnorm(y, ol.load(m.norm), m.eps))
    ol.store(m.f_out[:T, :], y)


@ol.jit
def markov_kernel(m, row):
    xs = ol.quantize(ol.load(m.markov_in), free=True)
    for col in range(0, m.vocab, Q.HEAD_CHUNK):
        n = min(Q.HEAD_CHUNK, m.vocab - col)
        correction = ol.dot(xs, m.w2[col:col + n, :])
        base = ol.load(Tensor(m.logits + row * m.vocab * 4 + col * 4,
                              (1, n), (n, 1)))
        ol.store(m.markov_out[:, col:col + n],
                 base * m.scale + correction * m.scale)
        del correction, base


class Engine(DS.Reference):
    """DSpark Reference-compatible draft sharing an existing target backend.

    Construction uploads draft matrices only and never owns/closes the target.
    The host calls are synchronous; target and draft must not run concurrently.
    """
    precision = {
        "backbone": "FPGA openTPU INT4 weights, INT8 activations/KV, FP32 residual/norm/RoPE/softmax",
        "context": "CPU FP32 fc/hidden_norm; FPGA INT4 per-layer K/V projection and QK norm/RoPE",
        "head": "shared target FPGA INT8 LM head; CPU original Markov/confidence weights",
        "embedding": "target host embedding rows",
        "source_equivalent": False,
    }

    def __init__(self, target, spec, weights, *, target_weights=None, rows=7,
                 block=Q.ATTN_BLOCK, device_fusion_markov=False):
        if (target.cfg.S != 1 or target.batch != 1 or target.stateful
                or target.spec.hidden != spec.hidden or target.spec.vocab != spec.vocab
                or spec.draft_vocab != spec.vocab or target.image.head_format != "int8"):
            raise ValueError("DSpark requires a matching single-stream target with an INT8 shared head")
        self.target, self.spec, self.weights = target, spec, weights
        self.target_weights = target_weights if target_weights is not None else {}
        self.target_vocab, self.d2t = target.spec.vocab, np.arange(spec.vocab, dtype=np.int64)
        self.cache_weights, self._cache = False, {}
        self.device, self.cfg, self.block = target.backend, target.cfg, block
        self.device_fusion_markov = bool(device_fusion_markov)
        self.precision = dict(type(self).precision)
        self.stats, self._programs = [], {}
        self.backend = ("fpga" if type(self.device).__name__ == "BoardBackend" else "isa") + \
            "-int4-backbone-shared-int8-head-host-fusion-markov"
        if self.device_fusion_markov:
            self.backend = self.backend.replace("host-fusion-markov", "int8-fusion-markov-host-lookups")
            self.precision.update(
                context="FPGA INT8 fc, FP32 hidden_norm; INT4 per-layer K/V projection and QK norm/RoPE",
                head="shared FPGA INT8 LM head and INT8 Markov_w2; CPU w1 lookup, argmax and confidence")
        self.embed = target.embed
        probe = np.asarray(self.embed[[0]])
        if (probe.shape != (1, spec.hidden) or not np.issubdtype(probe.dtype, np.floating)
                or getattr(self.embed, "int8", False)):
            raise ValueError("DSpark requires the target's floating-point host embedding")
        # Drain target compilation/execution before touching shared device state.
        target._drain()
        reservation = waits.area(self.cfg.IMEM_WORDS, self.cfg.SEQ_WC)
        end = max(target.image.nbytes, (target.bat or {}).get("end", 0))
        end = max(end, getattr(self.device, "prog_at", (end + 4095) // 4096 * 4096)
                  + reservation)
        base = (end + 4095) // 4096 * 4096
        cap = (target.cap + rows - 1 + self.cfg.D - 1) // self.cfg.D * self.cfg.D
        self.image = Image(spec, self.cfg, cap, rows=rows, base=base, head=target.image.head,
                           device_fusion_markov=self.device_fusion_markov)
        self.image.upload(self, self.device)
        self.cache_weights = True
        self._cached = ProjectedContext(np.empty(0, np.int64), np.empty((0, spec.hidden), np.float32))

    def project_context(self, auxiliary_states, positions):
        p = np.asarray(positions, dtype=np.int64)
        s = self.spec
        x = np.asarray(auxiliary_states, dtype=np.float32)
        if p.ndim != 1:
            raise ValueError("DSpark context positions must be a vector")
        expected = (len(p), len(s.target_layers) * s.target_hidden)
        if x.shape == (len(p), len(s.target_layers), s.target_hidden):
            x = x.reshape(expected)
        if p.ndim != 1 or x.shape != expected or not np.all(np.isfinite(x)):
            raise ValueError("DSpark auxiliary state shape/values are invalid")
        if self.device_fusion_markov:
            self.target._drain()
            out = np.empty((len(p), s.hidden), np.float32)
            for at in range(0, len(p), self.image.fusion_rows):
                n = min(self.image.fusion_rows, len(p) - at)
                self.device.write(0, self.image.extra["f_in"], F.ftz(x[at:at + n]))
                self._run(("fusion", n), lambda n=n: self.image.compile_fusion(n))
                out[at:at + n] = self.device.read(0, self.image.extra["f_out"],
                                                 n * s.hidden * 4).view(np.float32).reshape(n, s.hidden)
            return ProjectedContext(p, out)
        x = x @ self._weight("fc.weight", (s.hidden, expected[1])).T
        x = DS._rms(x, self._weight("hidden_norm.weight", (s.hidden,)), s.eps)
        return ProjectedContext(p, x)

    def _inputs(self, x, positions):
        cs = [Q.rope_tables(self.image.spec, int(p)) for p in positions]
        self.device.write(0, self.image.io["x"], F.ftz(np.asarray(x, np.float32)))
        self.device.write(0, self.image.io["cos"], np.stack([c for c, _ in cs]))
        self.device.write(0, self.image.io["sin"], np.stack([s for _, s in cs]))

    def _run(self, key, compile_program, values=None):
        if key not in self._programs:
            self._programs[key] = compile_program()
        if values is None:
            st = self.device.run(self._programs[key])
        else:
            progs, run_args = self._programs[key]
            words = arg_words(run_args, values)
            # BoardBackend's synchronous run intentionally has no args keyword;
            # its asynchronous pair is the argument-aware execution interface.
            if getattr(self.device, "start", None) is not None:
                self.device.start(progs, args=words)
                st = self.device.wait()
            else:
                st = self.device.run(progs, args=words)
        self.stats.append({**st, "draft_kind": key[0]})

    def _sync_context(self, context):
        if not isinstance(context, ProjectedContext) or context.hidden.shape[1] != self.spec.hidden:
            raise ValueError("FPGA DSpark needs its own ProjectedContext")
        n = len(context.positions)
        if not np.array_equal(context.positions, np.arange(n)):
            raise ValueError("FPGA DSpark context must be contiguous from position zero")
        keep = 0
        for keep in range(min(n, len(self._cached.positions))):
            if not np.array_equal(context.hidden[keep], self._cached.hidden[keep]):
                break
        else:
            keep = min(n, len(self._cached.positions))
        # A shortened context drops cache validity even though stale DRAM remains.
        # The next query overwrites its tail; later commits regenerate that tail.
        for at in range(keep, n, self.image.rows):
            stop = min(n, at + self.image.rows)
            self._inputs(context.hidden[at:stop], context.positions[at:stop])
            if getattr(self.device, "args", False) and at // 256 == (stop - 1) // 256:
                base = at // 256 * 256
                self._run(("context", "runtime", base, stop - at),
                          lambda base=base, at=at, stop=stop:
                          self.image.compile_context_runtime(base, stop - at),
                          {"position": at - base})
            else:
                self._run(("context", at, stop - at),
                          lambda at=at, stop=stop: self.image.compile_context(at, stop - at))
        self._cached = context

    def _forward(self, token_ids, positions, context, predictions, *, read_logits=True):
        ids, p = np.asarray(token_ids, np.int64), np.asarray(positions, np.int64)
        if (ids.ndim != 1 or ids.size == 0 or p.shape != ids.shape
                or np.any(ids < 0) or np.any(ids >= self.target_vocab)
                or np.any(np.diff(p) != 1) or p[0] != len(context.positions)
                or ids.size > self.image.rows):
            raise ValueError("Invalid DSpark FPGA query IDs/positions")
        self.target._drain()
        self._sync_context(context)
        self._inputs(self.embed[ids], p)
        if getattr(self.device, "args", False) and self.image.runtime_ok(int(p[0]), len(ids), self.block):
            base = int(p[0]) // 256 * 256
            mask = np.full((self.cfg.MCOLS, self.block + 1), -np.inf, np.float32)
            mask[:, :int(p[-1]) + 1 - base] = -0.0
            self.device.write(0, self.image.mask_at, mask)
            self._run(("query", "runtime", base, len(ids), predictions), lambda:
                      self.image.compile_query_runtime(base, len(ids), predictions, self.block),
                      {"position": int(p[0]) - base})
        else:
            self._run(("query", int(p[0]), len(ids), predictions), lambda:
                      self.image.compile_query(int(p[0]), len(ids), predictions, self.block))
        hidden = self.device.read(0, self.image.io["x"], len(ids) * self.spec.hidden * 4)
        hidden = hidden.view(np.float32).reshape(len(ids), self.spec.hidden)
        if read_logits:
            logits = self.device.read(0, self.image.io["logits"], predictions * self.spec.draft_vocab * 4)
            logits = logits.view(np.float32).reshape(predictions, self.spec.draft_vocab)
        else:
            logits = np.empty((0, self.spec.draft_vocab), np.float32)
        if not np.all(np.isfinite(hidden)) or not np.all(np.isfinite(logits)):
            raise ValueError("DSpark FPGA returned nonfinite hidden states/logits")
        return hidden, logits

    def backbone(self, token_ids, positions, context):
        return self._forward(token_ids, positions, context, 0)[0]

    def _markov_logits(self, embedding, row):
        """One sequential Markov correction plus its device-resident base row."""
        if not self.device_fusion_markov or not 0 <= row < self.image.rows:
            raise ValueError("Invalid device Markov request")
        value = np.asarray(embedding, np.float32)
        if value.shape != (self.spec.markov_rank,) or not np.all(np.isfinite(value)):
            raise ValueError("Invalid DSpark Markov embedding")
        inp = np.zeros((1, self.image.markov_width), np.float32)
        inp[0, :len(value)] = value
        self.device.write(0, self.image.extra["markov_in"], F.ftz(inp))
        if getattr(self.device, "args", False):
            self._run(("markov", "runtime"), self.image.compile_markov, {"row": row})
        else:
            self._run(("markov", row), lambda: self.image.compile_markov(row)[0])
        return self.device.read(0, self.image.extra["markov_out"],
                                self.spec.draft_vocab * 4).view(np.float32)

    def draft(self, anchor_token, anchor_position, context, num_tokens=None, *, query_tokens=None):
        s = self.spec
        num_tokens = s.block_size if num_tokens is None else num_tokens
        query_tokens = num_tokens if query_tokens is None else query_tokens
        if (type(num_tokens) is not int or type(query_tokens) is not int
                or not 1 <= num_tokens <= query_tokens <= self.image.rows):
            raise ValueError("Invalid DSpark FPGA query/prediction count")
        ids = np.full(query_tokens, s.mask_token, np.int64)
        ids[0] = anchor_token
        hidden, base = self._forward(ids, np.arange(anchor_position, anchor_position + query_tokens),
                                     context, num_tokens, read_logits=not self.device_fusion_markov)
        hidden, base = hidden[:num_tokens], base * np.float32(s.logit_scale)
        w1 = self._weight("markov_head.markov_w1.weight", (s.vocab, s.markov_rank))
        w2 = None if self.device_fusion_markov else self._weight(
            "markov_head.markov_w2.weight", (s.draft_vocab, s.markov_rank))
        logits = np.empty((num_tokens, s.draft_vocab), np.float32)
        tokens = np.empty(num_tokens, np.int64)
        confidence = np.empty(num_tokens, np.float32) if s.confidence_head else None
        prev = int(anchor_token)
        for i in range(num_tokens):
            emb = w1[prev]
            if self.device_fusion_markov:
                logits[i] = self._markov_logits(emb, i)
            elif s.topk is None:
                logits[i] = base[i] + np.float32(s.logit_scale) * (w2 @ emb)
            else:
                ix = np.argsort(-base[i], kind="stable")[:s.topk]
                logits[i].fill(-np.inf)
                logits[i, ix] = base[i, ix] + np.float32(s.logit_scale) * (w2[ix] @ emb)
            if np.any(np.isnan(logits[i])) or np.any(np.isposinf(logits[i])) or not np.any(np.isfinite(logits[i])):
                raise ValueError("DSpark Markov head produced invalid logits")
            tokens[i] = self.d2t[int(np.argmax(logits[i]))]
            if confidence is not None:
                inp = np.concatenate((hidden[i], emb)) if s.confidence_with_markov else hidden[i]
                w = self._weight("confidence_head.proj.weight", (1, inp.size))
                b = self._weight("confidence_head.proj.bias", (1,))
                z = (w @ inp + b)[0]
                confidence[i] = np.exp(-np.logaddexp(np.float32(0), -z))
            prev = int(tokens[i])
        return DS.Proposal(tokens, logits, hidden, confidence)
