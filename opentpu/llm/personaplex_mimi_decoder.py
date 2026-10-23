"""Resident causal Mimi upsample and decoder transformer, CPU/ISA qualified only.

Two tokens per frame share a native ring-cache update before attention. This
preserves the pinned source's full-ring position defect. SEANet/PCM is absent.
"""
from dataclasses import dataclass
import hashlib
import math
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from .. import quant as Q
from .. import qcache as QC
from ..compiler import (Affine, CompileError, KVDesc, QTensor, RunVar, Tensor,
                        arg_reg, k_scale_bytes, trace_fit)
from ..kernels.attention import Additive, Blocks, _attend_heads
from ..kernels.lib import rope_rows
from . import qwen3 as G
from .personaplex_audio_heads import _checked
from .personaplex_mimi import _isa_exact_guard
from .personaplex_temporal import _split_pairs


@dataclass(frozen=True)
class MimiDecoderSpec:
    hidden: int = 512
    n_heads: int = 8
    ffn: int = 2048
    layers: int = 8
    context: int = 250
    max_frames: int = 128
    eps: float = 1e-5
    rope_base: float = 10000.0
    precision: str = "int8"

    def __post_init__(self):
        for name in ("hidden", "n_heads", "ffn", "layers", "context", "max_frames"):
            v = getattr(self, name)
            if type(v) is not int or not 0 < v < 1 << 24:
                raise ValueError(f"{name} must be a positive integer below 2^24")
        if self.hidden % self.n_heads or self.head_dim % 2 or self.context < 2:
            raise ValueError("decoder needs even head dimensions and at least two cache slots")
        if not math.isfinite(self.eps) or self.eps <= 0 or not math.isfinite(self.rope_base) or self.rope_base <= 1:
            raise ValueError("invalid decoder normalization or RoPE constant")
        if 2 * self.max_frames >= 1 << 24:
            raise ValueError("decoder positions must be exactly representable in FP32")
        if self.precision not in ("int8", "fp32"):
            raise ValueError("decoder precision must be int8 or fp32")

    @property
    def head_dim(self):
        return self.hidden // self.n_heads


def native_ring_positions(end, context):
    """Offline source-compatible ring metadata, including delta<=0 boundary bug."""
    slots = np.arange(context, dtype=np.int64)
    delta = slots - end % context
    positions = np.where(delta <= 0, end + delta, end + delta - context)
    return np.where(slots >= end, -1, positions)


def offline_rope(spec):
    """Frozen FP32 coefficients; inference only loads rows selected on device."""
    frequency = np.exp(np.arange(spec.head_dim // 2, dtype=np.float32)
                       * np.float32(-math.log(spec.rope_base) * 2 / spec.head_dim))
    angle = np.arange(2 * spec.max_frames, dtype=np.float32)[:, None] * frequency
    shape = (spec.max_frames, 2, spec.head_dim // 2)
    return np.cos(angle).astype(np.float32).reshape(shape), np.sin(angle).astype(np.float32).reshape(shape)


def _layer_norm(x, gamma, beta, eps):
    center = x - (ol.sum(x, axis=1) * (1.0 / x.cols))[:, None]
    inverse = ol.rsqrt(ol.sum(center * center, axis=1) * (1.0 / x.cols) + eps)
    return (center * inverse[:, None]) * gamma[None, :] + beta[None, :]


def _gelu(x):
    """Erf-form GELU approximation; not the distinct tanh-form GELU function.

    A&S 7.1.26 polynomial has ~1.5e-7 absolute erf approximation error before
    ISA FP32/EXP2 rounding. Numerical error is measured separately from INT8.
    """
    u = x * (1.0 / math.sqrt(2.0))
    t = ol.recip(1.0 + ol.abs(u) * 0.3275911)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (
        1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    sign = ol.minimum(ol.maximum(u * 1e20, -1.0), 1.0)
    erf = sign * (1.0 - poly * ol.exp2((u * u) * -ol.LOG2E))
    return (x * 0.5) * (1.0 + erf)


@ol.jit
def _upsample(m):
    x = ol.load(m.latent)
    partial = ol.load(m.overlap)
    weights = ol.load(m.weight)
    output = ol.empty((2, m.hidden))
    output[0:1, :].set(x * weights[0:1, :] + partial[0:1, :])
    output[1:2, :].set(x * weights[1:2, :] + partial[1:2, :])
    ol.store(m.upsampled, output)
    ol.store(m.output, output)
    new_partial = ol.empty((2, m.hidden))
    new_partial[0:1, :].set(x * weights[2:3, :])
    new_partial[1:2, :].set(x * weights[3:4, :])
    ol.store(m.overlap, new_partial)


def _linear_fp32(x, weights, output):
    """FP32 coefficient-row reductions into resident two-token scratch."""
    rows, cols = weights.shape
    for channel in ol.range(rows):
        coefficients = ol.load(Tensor(weights.base + channel * (4 * cols),
                                      (cols,), (1,)))
        reduced = ol.sum(x * coefficients[None, :], axis=1)
        column = ol.empty((2, 1))
        column.set(reduced[:, None])
        ol.store(output[:, channel:channel + 1], column)
    return ol.load(output)


def _decoder_attention_fp32(x, w, slots, cos, sin, masks, spec, cap):
    normalized = _layer_norm(x, ol.load(w.norm1_weight), ol.load(w.norm1_bias), spec.eps)
    q = _linear_fp32(normalized, w.wq, w.projection_output["wq"])
    k = _linear_fp32(normalized, w.wk, w.projection_output["wk"])
    v = _linear_fp32(normalized, w.wv, w.projection_output["wv"])
    del normalized
    d = spec.head_dim
    # Complete both tokens in each native ring before either query attends.
    # Real/imag Q/K row permutation is the same as the INT8 path.
    for head in range(spec.n_heads):
        keys = rope_rows(k[:, head * d:(head + 1) * d], cos, sin)
        values = v[:, head * d:(head + 1) * d]
        base = w.fp_heads[head]
        for token in range(2):
            ol.store(Tensor(Affine(base["k"]) + slots[token] * (4 * d),
                            (1, d), (d, 1)), keys[token:token + 1, :])
            ol.store(Tensor(Affine(base["vt"]) + slots[token] * 4,
                            (d, 1), (cap, 1)), values[token:token + 1, :].reshape(d, 1))
    del keys, values
    attended = ol.empty((2, spec.hidden))
    tile = math.gcd(cap, 32)
    for head in range(spec.n_heads):
        base = w.fp_heads[head]
        queries = rope_rows(q[:, head * d:(head + 1) * d], cos, sin)
        for token in range(2):
            query = queries[token, :]
            # Bound cache working sets independently of the full context.
            # Scores/probabilities are resident scratch, never host feedback.
            for block in ol.range(cap // tile):
                start = block * tile
                keys = ol.load(Tensor(base["k"] + start * (4 * d), (tile, d), (d, 1)))
                partial_scores = ol.sum(keys * query[None, :], axis=1).reshape(1, tile)
                ol.store(w.attention_scores[token:token + 1, start:start + tile], partial_scores)
            scores = ol.load(w.attention_scores[token:token + 1, :])
            scores = scores * (ol.LOG2E / math.sqrt(d)) + masks[token:token + 1, :cap] * ol.LOG2E
            probabilities = ol.exp2(scores - ol.max(scores, axis=1)[:, None])
            ol.store(w.attention_probabilities[token:token + 1, :], probabilities)
            total = ol.sum(probabilities, axis=1)
            weighted = ol.zeros((d,))
            for block in ol.range(cap // tile):
                start = block * tile
                values = ol.load(Tensor(base["vt"] + start * 4, (d, tile), (cap, 1)))
                part = ol.load(w.attention_probabilities[token:token + 1, start:start + tile])
                weighted.set(weighted + ol.sum(values * part[0, :][None, :], axis=1))
            result = weighted.reshape(1, d) / total[:, None]
            attended[token:token + 1, head * d:(head + 1) * d].set(result)
        # Static head expansion must release the preceding head's large cache
        # tiles before allocating the next head on the production profile.
        del keys, values, queries, query, scores, probabilities, total, weighted, result, partial_scores, part
    return x + _linear_fp32(attended, w.wo, w.projection_output["wo"]) * ol.load(w.scale1)[None, :]


def _decoder_attention(x, w, slots, cos, sin, seq, spec, cap):
    normalized = ol.quantize(_layer_norm(x, ol.load(w.norm1_weight), ol.load(w.norm1_bias), spec.eps))
    k, v = ol.dot(normalized, w.wk), ol.dot(normalized, w.wv)
    d = spec.head_dim
    queries = [ol.dot(normalized, w.wq[i * d:(i + 1) * d, :]) for i in range(spec.n_heads)]
    # Q/K projection rows were permuted offline from adjacent [real,imag] to
    # [all real,all imag]. V and output projection rows remain unchanged.
    for head in range(spec.n_heads):
        keys = G._padded(rope_rows(k[:, head * d:(head + 1) * d], cos, sin))
        values = G._padded(v[:, head * d:(head + 1) * d])
        for token in range(2):
            ol.kv_append(w.kv, head, slots[token], keys[token:token + 1, :], values[token:token + 1, :])
    output = ol.empty((2, spec.hidden))
    def query(head):
        return lambda: G._padded(rope_rows(queries[head], cos, sin))
    def emit(head, acc, total):
        output[:, head * d:(head + 1) * d].set(acc / total[:, None])
    _attend_heads([query(h) for h in range(spec.n_heads)], w.kv, list(range(spec.n_heads)),
                  seq, cap, ol.LOG2E / math.sqrt(d), depth=3, ahead=2, emit=emit)
    return x + ol.dot(output, w.wo) * ol.load(w.scale1)[None, :]


@ol.jit
def _decoder_layer(m):
    b = ol.current()
    counter = ol.load(m.state)
    body, mark = b.stack[-1], len(b.stack[-1])
    b.run_words = {"mimi_frame": counter.base}
    frame = RunVar("mimi_frame", m.spec.max_frames)
    slot_values = ol.load(Tensor(Affine(m.slots_at) + frame * 8, (2,), (1,)))
    slot_mark = len(body)
    slots = [RunVar("mimi_slot" + str(i), m.spec.context) for i in range(2)]
    b.run_words = {"mimi_frame": counter.base, "mimi_slot0": slot_values.base,
                   "mimi_slot1": slot_values.base + 1}
    half = m.spec.head_dim // 2
    cos = ol.load(Tensor(Affine(m.cos_at) + frame * (8 * half), (2, half), (half, 1)))
    sin = ol.load(Tensor(Affine(m.sin_at) + frame * (8 * half), (2, half), (half, 1)))
    seq = Blocks([(0, m.cap, Additive(Affine(m.masks_at) + frame * (8 * (m.cap + 1)), m.cap))])
    x = ol.load(m.output)
    if m.spec.precision == "fp32":
        masks = ol.load(Tensor(Affine(m.masks_at) + frame * (8 * (m.cap + 1)),
                               (2, m.cap + 1), (m.cap + 1, 1)))
        x.set(_decoder_attention_fp32(x, m.layer, slots, cos, sin, masks, m.spec, m.cap))
        normalized = _layer_norm(x, ol.load(m.layer.norm2_weight), ol.load(m.layer.norm2_bias), m.spec.eps)
        hidden = _gelu(_linear_fp32(normalized, m.layer.w1, m.layer.projection_output["w1"]))
        ffn = _linear_fp32(hidden, m.layer.w2, m.layer.projection_output["w2"])
    else:
        x.set(_decoder_attention(x, m.layer, slots, cos, sin, seq, m.spec, m.cap))
        normalized = ol.quantize(_layer_norm(x, ol.load(m.layer.norm2_weight), ol.load(m.layer.norm2_bias), m.spec.eps))
        ffn = ol.dot(ol.quantize(_gelu(ol.dot(normalized, m.layer.w1))), m.layer.w2)
    x.set(x + ffn * ol.load(m.layer.scale2)[None, :])
    ol.store(m.output, x)
    ol.store(m.layer_output, x)
    if m.last:
        ol.store(m.state, counter + 1.0)
    bounds = {"mimi_frame": m.spec.max_frames, "mimi_slot0": m.spec.context, "mimi_slot1": m.spec.context}
    arguments = list(b.run_args)
    loads = [(var.name, I.rld(arg_reg(index), b.run_words[var.name], mul=int(mult),
                           bound=(0, int(mult) * (bounds[var.name] - 1)),
                           comment=f"resident {var.name} *{mult}"))
                       for index, (var, mult) in enumerate(arguments)]
    # Frame registers are needed to load the slot table; slot registers may be
    # initialized only after that load has completed. Both remain device inputs.
    body[slot_mark:slot_mark] = [op for name, op in loads if name != "mimi_frame"]
    body[mark:mark] = [op for name, op in loads if name == "mimi_frame"]
    b.run_words = None
    b.run_args.clear()


class MimiDecoderImage:
    def __init__(self, spec, cfg, *, latent_at, state_base=0x5E900000, state_limit=0x60000000,
                 weight_base=0x60000000, weight_limit=0xB0000000):
        if cfg.S != 1 or spec.hidden % cfg.D or spec.ffn % cfg.D:
            raise ValueError("Mimi decoder needs one slice and hidden/FFN divisible by D")
        if max(spec.hidden, spec.ffn) > cfg.ACT_BLOCKS * cfg.D:
            raise ValueError("Mimi decoder inner dimension exceeds ACT RAM")
        capacity = min(cfg.DRAM_BYTES, 1 << 32)
        for lo, hi in ((state_base, state_limit), (weight_base, weight_limit)):
            if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= capacity or lo % 128:
                raise MemoryError("Invalid Mimi decoder reserved mapping")
        if not 0 <= latent_at < latent_at + spec.hidden * 4 <= capacity:
            raise MemoryError("Mimi latent input exceeds mapped capacity")
        if state_base < weight_limit and weight_base < state_limit:
            raise ValueError("Mimi decoder state and matrix regions overlap")
        self.spec, self.cfg, self.latent_at = spec, cfg, latent_at
        self.spans, self.payload_hashes, self.programs = [], {}, None
        cursor = {"decoder_state": state_base, "decoder_weights": weight_base}
        limits = {"decoder_state": state_limit, "decoder_weights": weight_limit}
        def alloc(name, size, region="decoder_state", kind="state", alignment=128):
            at = -(-cursor[region] // alignment) * alignment
            if size <= 0 or at + size > limits[region]:
                raise MemoryError(f"Mimi decoder {name} exceeds {region}")
            if at < latent_at + 4 * spec.hidden and latent_at < at + size:
                raise ValueError("Decoder span overlaps resident latent")
            self.spans.append({"name": "decoder_" + name, "base": at, "bytes": size,
                               "region": region, "kind": kind, "alignment": alignment})
            cursor[region] = at + size
            return at
        self.io = {"state": alloc("state", 4), "overlap": alloc("overlap", 8 * spec.hidden),
                   "upsampled": alloc("upsampled", 8 * spec.hidden), "output": alloc("output", 8 * spec.hidden)}
        self.upsample_at = alloc("upsample_weight", 16 * spec.hidden, kind="immutable")
        half = spec.head_dim // 2
        self.cos_at = alloc("cos", 8 * spec.max_frames * half, kind="immutable")
        self.sin_at = alloc("sin", 8 * spec.max_frames * half, kind="immutable")
        self.slots_at = alloc("slots", 8 * spec.max_frames, kind="immutable")
        self.cap = -(-spec.context // cfg.D) * cfg.D
        self.dpad = -(-spec.head_dim // cfg.D) * cfg.D
        self.masks_at = alloc("masks", 8 * spec.max_frames * (self.cap + 1), kind="immutable")
        self.norm_offsets, self.layer_outputs, self.kv_heads, self.cache_regions = [], [], [], []
        self.mats = {name: (spec.hidden, spec.hidden) for name in ("wq", "wk", "wv", "wo")}
        self.mats.update(w1=(spec.ffn, spec.hidden), w2=(spec.hidden, spec.ffn))
        self.weight_offsets, self.projection_outputs, self.attention_scratch = [], [], []
        for index in range(spec.layers):
            self.norm_offsets.append({name: alloc(f"layer{index}_{name}", 4 * spec.hidden, kind="immutable")
                for name in ("norm1_weight", "norm1_bias", "norm2_weight", "norm2_bias", "scale1", "scale2")})
            self.layer_outputs.append(alloc(f"layer{index}_output", 8 * spec.hidden))
            heads = {}; start = cursor["decoder_state"]
            for head in range(spec.n_heads):
                if spec.precision == "fp32":
                    heads[head] = {"k": alloc(f"kv{index}_{head}_k", 4 * self.cap * spec.head_dim),
                                   "vt": alloc(f"kv{index}_{head}_vt", 4 * spec.head_dim * self.cap)}
                else:
                    heads[head] = {"k": alloc(f"kv{index}_{head}_k", self.cap * self.dpad),
                                   "ks": alloc(f"kv{index}_{head}_ks", self.cap * k_scale_bytes(self.dpad, cfg.D)),
                                   "vt": alloc(f"kv{index}_{head}_vt", self.dpad * self.cap),
                                   "vs": alloc(f"kv{index}_{head}_vs", 4 * self.cap)}
            self.kv_heads.append(heads); self.cache_regions.append((start, cursor["decoder_state"] - start))
            if spec.precision == "fp32":
                self.weight_offsets.append({name: (alloc(f"layer{index}_{name}_fp32", 4 * rows * cols,
                    "decoder_weights", "immutable"), None) for name, (rows, cols) in self.mats.items()})
                self.projection_outputs.append({name: alloc(f"layer{index}_{name}_scratch", 8 * rows)
                                                for name, (rows, _) in self.mats.items()})
                self.attention_scratch.append({name: alloc(f"layer{index}_attention_{name}", 8 * self.cap)
                                               for name in ("scores", "probabilities")})
            else:
                self.weight_offsets.append({name: (
                    alloc(f"layer{index}_{name}_codes", rows * Q.row_bytes(cols, "int8", cfg.D), "decoder_weights", "immutable"),
                    alloc(f"layer{index}_{name}_scales", rows * Q.scale_bytes(cols, "int8", cfg.D), "decoder_weights", "immutable"))
                    for name, (rows, cols) in self.mats.items()})
        self.state_end = -(-cursor["decoder_state"] // 4096) * 4096
        self.weight_end = -(-cursor["decoder_weights"] // 4096) * 4096
        if self.state_end > state_limit or self.weight_end > weight_limit:
            raise MemoryError("Aligned Mimi decoder mapping exceeds capacity")
        self.required_bytes = max(self.state_end, self.weight_end)
        self.kv_bytes = sum(size for _, size in self.cache_regions)
        self.weight_bytes = 16 * spec.hidden + spec.layers * (24 * spec.hidden + sum(
            (4 * rows * cols if spec.precision == "fp32" else
             rows * (Q.row_bytes(cols, "int8", cfg.D) + Q.scale_bytes(cols, "int8", cfg.D)))
            for rows, cols in self.mats.values()))

    def layer_descriptor(self, index):
        s = self.spec
        value = SimpleNamespace(**{name: Tensor(at, (s.hidden,), (1,))
                                   for name, at in self.norm_offsets[index].items()})
        for name, (rows, cols) in self.mats.items():
            if s.precision == "fp32":
                setattr(value, name, Tensor(self.weight_offsets[index][name][0], (rows, cols), (cols, 1)))
            else:
                setattr(value, name, QTensor.weight(*self.weight_offsets[index][name], rows, cols, "int8", self.cfg.D))
        if s.precision == "fp32":
            value.fp_heads = self.kv_heads[index]
            value.projection_output = {name: Tensor(at, (2, self.mats[name][0]), (self.mats[name][0], 1))
                                       for name, at in self.projection_outputs[index].items()}
            for name, at in self.attention_scratch[index].items():
                setattr(value, "attention_" + name, Tensor(at, (2, self.cap), (self.cap, 1)))
        else:
            value.kv = KVDesc(self.kv_heads[index], self.cap, self.dpad, self.cfg.D, 1, 0, dv=s.head_dim)
        return value

    def compile(self):
        if self.programs is None:
            s = self.spec
            def matrix(at): return Tensor(at, (2, s.hidden), (s.hidden, 1))
            up = SimpleNamespace(latent=Tensor(self.latent_at, (1, s.hidden), (s.hidden, 1)),
                overlap=matrix(self.io["overlap"]), upsampled=matrix(self.io["upsampled"]), output=matrix(self.io["output"]),
                weight=Tensor(self.upsample_at, (4, s.hidden), (s.hidden, 1)), hidden=s.hidden)
            programs = [trace_fit(_upsample, self.cfg, 0, lambda: {"m": up}).finish()]
            for index in range(s.layers):
                m = SimpleNamespace(spec=s, cap=self.cap, state=Tensor(self.io["state"], (1,), (1,)),
                    output=matrix(self.io["output"]), layer_output=matrix(self.layer_outputs[index]),
                    layer=self.layer_descriptor(index), last=index == s.layers - 1,
                    cos_at=self.cos_at, sin_at=self.sin_at, slots_at=self.slots_at, masks_at=self.masks_at)
                programs.append(trace_fit(_decoder_layer, self.cfg, 0, lambda: {"m": m}).finish())
            for p in programs:
                if p.run_args or len(p) * 8 > self.cfg.IMEM_WORDS:
                    raise CompileError("Mimi decoder requires host arguments or exceeds instruction memory")
            self.programs = programs
        return self.programs

    def budget(self):
        return {"required_bytes": self.required_bytes, "state_end": self.state_end, "weight_end": self.weight_end,
                "occupied_bytes": sum(p["bytes"] for p in self.spans), "kv_bytes": self.kv_bytes,
                "weight_bytes": self.weight_bytes,
                "payload_bytes": sum(p["bytes"] for p in self.spans if p["kind"] == "immutable"),
                "program_words": [len(p) * 8 for p in self.compile()], "spans": self.spans,
                "payload_sha256": self.payload_hashes, "max_frames": self.spec.max_frames,
                "precision": self.spec.precision,
                "cache_policy": "native two-token complete before attention; delta<=0 position defect preserved",
                "gelu": "native erf form approximated by device polynomial, separately measured"}

    def upload(self, backend, upsample_weight, layers, rope=None):
        self.compile()
        s = self.spec
        upsample_weight = upsample_weight() if callable(upsample_weight) else upsample_weight
        layers = layers() if callable(layers) else layers
        weight = _checked(upsample_weight, (s.hidden, 4), "Mimi upsample weight")
        if len(layers) != s.layers:
            raise ValueError("Wrong number of decoder layers")
        def put(name, at, value):
            raw = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
            self.payload_hashes[name] = hashlib.sha256(memoryview(raw)).hexdigest()
            backend.write(0, at, raw)
        put("upsample", self.upsample_at, weight.T.copy())
        rope = rope() if callable(rope) else rope
        cos, sin = offline_rope(s) if rope is None else rope
        shape = (s.max_frames, 2, s.head_dim // 2)
        put("cos", self.cos_at, _checked(cos, shape, "RoPE cosine"))
        put("sin", self.sin_at, _checked(sin, shape, "RoPE sine"))
        slots = np.arange(2 * s.max_frames, dtype=np.int64).reshape(-1, 2) % s.context
        put("slots", self.slots_at, slots.astype(np.float32))
        masks = np.full((s.max_frames, 2, self.cap + 1), -np.inf, np.float32)
        for frame in range(s.max_frames):
            positions = native_ring_positions(2 * (frame + 1), s.context)
            for token in range(2):
                delta = 2 * frame + token - positions
                keep = (positions >= 0) & (delta >= 0) & (delta < s.context)
                masks[frame, token, :s.context][keep] = -0.0
        put("masks", self.masks_at, masks)
        for index, layer in enumerate(layers):
            for name, at in self.norm_offsets[index].items():
                put(f"layer{index}_{name}", at, _checked(layer[name], (s.hidden,), name))
            batch = QC.Batch()
            for name, shape in self.mats.items():
                value = _checked(layer[name], shape, name)
                if name in ("wq", "wk"):
                    value = _split_pairs(value, s)
                da, sa = self.weight_offsets[index][name]
                if s.precision == "fp32":
                    put(f"layer{index}_{name}_fp32", da, value)
                    continue
                batch.submit(value, "int8", self.cfg.D,
                    lambda q, scales, index=index, name=name, da=da, sa=sa:
                    (put(f"layer{index}_{name}_codes", da, q), put(f"layer{index}_{name}_scales", sa, scales)))
            batch.wait()
        if s.precision == "fp32":
            # Fresh inactive slots must be finite even before their masks are
            # applied. Logical reset later retains these caches as source does.
            for heads in self.kv_heads:
                for values in heads.values():
                    for at in values.values():
                        backend.write(0, at, np.zeros(self.cap * s.head_dim, np.float32))
        # Native partial=None skips the first overlap add. Adding -0 is the
        # FP32 identity even for a -0 product, unlike adding an initial +0.
        backend.write(0, self.io["overlap"], np.full((2, s.hidden), -0.0, np.float32))
