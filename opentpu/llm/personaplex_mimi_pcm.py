"""Resident causal SEANet decoder to raw FP32 PCM; CPU/ISA qualification only.

Offline packing converts native convolution weights to INT8 matrices. Streaming
history, overlap, activation, residuals and PCM generation execute on the device.
No clipping, normalization, PCM integer conversion or serving backend is added.
"""
from dataclasses import dataclass
import hashlib
import math
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from .. import qcache as QC
from .. import quant as Q
from ..compiler import CompileError, QTensor, Tensor, trace_fit
from .personaplex_audio_heads import _checked


@dataclass(frozen=True)
class MimiPCMSpec:
    hidden: int = 512
    n_filters: int = 64
    ratios: tuple = (8, 6, 5, 4)
    input_rows: int = 2
    final_precision: str = "int8"
    residual_precision: str = "int8"

    def __post_init__(self):
        for name in ("hidden", "n_filters", "input_rows"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 1 << 16:
                raise ValueError(f"{name} must be a positive bounded integer")
        if self.n_filters % 2 or len(self.ratios) != 4 or any(
                type(x) is not int or not 1 < x <= 16 for x in self.ratios):
            raise ValueError("PCM needs even filters and four bounded upsampling ratios")
        if self.samples > 1 << 20:
            raise ValueError("PCM frame is too large")
        if self.final_precision not in ("int8", "fp32"):
            raise ValueError("final_precision must be int8 or fp32")
        if self.residual_precision not in ("int8", "fp32"):
            raise ValueError("residual_precision must be int8 or fp32")

    @property
    def samples(self):
        return self.input_rows * math.prod(self.ratios)


def _elu(x):
    # Stable alpha=1 ELU: EXP2 only sees nonpositive inputs. Source has no
    # final tanh, clipping, peak/RMS normalization or int16 scaling.
    return ol.maximum(x, 0.0) + (ol.exp2(ol.minimum(x, 0.0) * ol.LOG2E) - 1.0)


@ol.jit
def _conv(m):
    for block in ol.range(m.rows // m.tile):
        start = block * m.tile
        x = ol.load(m.input[start:start + m.tile, :])
        if m.elu:
            x = _elu(x)
            ol.store(m.activated[start:start + m.tile, :], x)
        ol.store(m.prepared[start + m.history:start + m.history + m.tile, :], x)
    if m.precision != "fp32":
        bias = ol.load(m.bias)
    for block in ol.range(m.rows // m.tile):
        start = block * m.tile
        window = ol.zeros((m.tile, m.padded_k))
        for tap in range(m.kernel):
            window[:, tap * m.cin:(tap + 1) * m.cin].set(
                ol.load(m.prepared[start + tap:start + tap + m.tile, :]))
        if m.precision == "fp32" and m.cout == 1:
            # The mono output projection is small and cancellation-sensitive.
            # Keep its checkpoint coefficients and activations in FP32. RSUM
            # of a product lowers to the existing VPU RDOT, without QACT/MM.
            reduced = ol.sum(window * ol.load(m.fp_weight)[None, :], axis=1)
            y = ol.empty((m.tile, 1))
            y.set(reduced[:, None])
            y.set(y + ol.load(m.bias)[None, :])
            ol.store(m.output[start:start + m.tile, :], y)
        elif m.precision == "fp32":
            # Streaming a coefficient row per output channel bounds TMEM and
            # keeps the program fixed-size. No host multiply/reduction.
            for channel in ol.range(m.cout):
                coefficient = ol.load(Tensor(m.fp_data + channel * (4 * m.padded_k),
                                              (m.padded_k,), (1,)))
                reduced = ol.sum(window * coefficient[None, :], axis=1)
                column = ol.empty((m.tile, 1))
                column.set(reduced[:, None])
                column.set(column + ol.load(m.bias[channel:channel + 1])[None, :])
                if m.skip is not None:
                    column.set(column + ol.load(m.skip[start:start + m.tile, channel:channel + 1]))
                ol.store(m.output[start:start + m.tile, channel:channel + 1], column)
        else:
            y = ol.dot(window, m.weight) + bias[None, :]
            if m.skip is not None:
                y = y + ol.load(m.skip[start:start + m.tile, :])
            ol.store(m.output[start:start + m.tile, :], y)
    if m.history:
        # This load precedes the overlapping prefix store, including the
        # first K7 frame where the chunk is shorter than the six-row history.
        retained = ol.load(m.prepared[m.rows:m.rows + m.history, :])
        ol.store(m.prepared[:m.history, :], retained)
    if m.last:
        ol.store(m.frame, ol.load(m.frame) + 1.0)


@ol.jit
def _transpose(m):
    raw_bias = ol.load(m.raw_bias)
    for block in ol.range(m.rows // m.tile):
        start = block * m.tile
        x = _elu(ol.load(m.input[start:start + m.tile, :]))
        ol.store(m.activated[start:start + m.tile, :], x)
        window = ol.zeros((m.tile, m.padded_k))
        window[:, :m.cin].set(x)
        ol.store(m.raw[start:start + m.tile, :], ol.dot(window, m.weight) + raw_bias[None, :])
    bias = ol.load(m.bias)
    seen = ol.load(m.seen)
    b = ol.current()
    r = b.scratch()
    for index in ol.range(m.rows):
        first = ol.load(m.raw[index:index + 1, :m.stride * m.cout]).reshape(m.stride, m.cout)
        # Native partial=None skips the first overlap operation completely.
        # Subsequent rows add (saved raw biased tail - bias). This conditional
        # LOOP reads resident state; no host argument or state inspection.
        b.emit(I.rld(r, seen.base, bound=(0, 1), comment="SEANet partial is present"))
        conditional = b.begin_loop(0, rcount=r)
        first.set(first + (ol.load(m.partial) - bias[None, :]))
        b.end_loop(conditional)
        ol.store(m.output[index * m.stride:(index + 1) * m.stride, :], first)
        tail = ol.load(m.raw[index:index + 1, m.stride * m.cout:]).reshape(m.stride, m.cout)
        ol.store(m.partial, tail)
        seen.set(1.0)
    ol.store(m.seen, seen)
    b.unscratch(r)


class MimiPCMImage:
    """Four-stage SEANet with persistent convolution state and fixed programs."""
    def __init__(self, spec, cfg, *, input_at, state_base, weight_base,
                 state_limit=0x60000000, weight_limit=0xB0000000):
        capacity = min(cfg.DRAM_BYTES, 1 << 32)
        if cfg.S != 1:
            raise ValueError("SEANet PCM requires one slice")
        for lo, hi in ((state_base, state_limit), (weight_base, weight_limit)):
            if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= capacity or lo % 128:
                raise MemoryError("Invalid SEANet reserved mapping")
        if state_base < weight_limit and weight_base < state_limit:
            raise ValueError("SEANet state and weights overlap")
        if not 0 <= input_at < input_at + 4 * spec.input_rows * spec.hidden <= capacity:
            raise MemoryError("Resident SEANet input exceeds capacity")
        self.spec, self.cfg, self.input_at = spec, cfg, input_at
        self.spans, self.payload_hashes, self.ops = [], {}, []
        self.programs = None
        self.reset_regions, self.layer_outputs = [], [None] * 15
        cursor = {"pcm_state": state_base, "pcm_weights": weight_base}
        limits = {"pcm_state": state_limit, "pcm_weights": weight_limit}

        def alloc(name, size, region="pcm_state", kind="state"):
            at = -(-cursor[region] // 128) * 128
            if size <= 0 or at + size > limits[region]:
                raise MemoryError(f"SEANet {name} exceeds {region}")
            if at < input_at + 4 * spec.input_rows * spec.hidden and input_at < at + size:
                raise ValueError("SEANet span overlaps resident decoder input")
            self.spans.append(dict(name="pcm_" + name, base=at, bytes=size,
                                   region=region, kind=kind, alignment=128))
            cursor[region] = at + size
            return at

        def tensor(at, rows, cols):
            return Tensor(at, (rows, cols), (cols, 1))

        frame_at = alloc("frame", 4)
        self.reset_regions.append((frame_at, 4))
        self.io = {"frame": frame_at}

        def op(kind, model_index, subpath, source_at, rows, cin, cout, kernel,
               *, elu=True, stride=1, skip_at=None, last=False):
            index = len(self.ops)
            trans = kind == "transpose"
            columns = cin if trans else cin * kernel
            padded_k = -(-columns // cfg.D) * cfg.D
            if padded_k > cfg.ACT_BLOCKS * cfg.D:
                raise MemoryError("SEANet convolution exceeds ACT RAM")
            out_rows = rows * stride
            matrix_rows = kernel * cout if trans else cout
            output_at = alloc(f"conv{index}_output", 4 * out_rows * cout)
            m = SimpleNamespace(kind=kind, model_index=model_index, subpath=subpath,
                cin=cin, cout=cout, kernel=kernel, stride=stride, rows=rows,
                tile=math.gcd(rows, cfg.MCOLS), padded_k=padded_k,
                input=tensor(source_at, rows, cin), output=tensor(output_at, out_rows, cout),
                skip=tensor(skip_at, rows, cout) if skip_at is not None else None,
                elu=elu, last=last, frame=Tensor(frame_at, (1,), (1,)))
            m.precision = (spec.final_precision if last else
                           spec.residual_precision if subpath.startswith("block.") else "int8")
            m.matrix_shape = (matrix_rows, padded_k)
            if elu:
                m.activated = tensor(alloc(f"conv{index}_activated", 4 * rows * cin), rows, cin)
            bias_at = alloc(f"conv{index}_bias", 4 * cout, kind="immutable")
            m.bias = Tensor(bias_at, (cout,), (1,))
            if trans:
                m.raw = tensor(alloc(f"conv{index}_raw", 4 * rows * matrix_rows), rows, matrix_rows)
                m.partial = tensor(alloc(f"conv{index}_partial", 4 * stride * cout), stride, cout)
                m.seen = Tensor(alloc(f"conv{index}_seen", 4), (1,), (1,))
                m.raw_bias = Tensor(alloc(f"conv{index}_raw_bias", 4 * matrix_rows, kind="immutable"),
                                    (matrix_rows,), (1,))
                self.reset_regions.extend([(m.partial.base, 4 * stride * cout), (m.seen.base, 4)])
            else:
                m.history = kernel - 1
                m.prepared = tensor(alloc(f"conv{index}_prepared", 4 * (rows + m.history) * cin),
                                    rows + m.history, cin)
                if m.history:
                    self.reset_regions.append((m.prepared.base, 4 * m.history * cin))
            if m.precision == "fp32":
                m.fp_data = alloc(f"conv{index}_fp32", 4 * matrix_rows * padded_k,
                                  "pcm_weights", "immutable")
                m.fp_weight = Tensor(m.fp_data, (padded_k,), (1,)) if cout == 1 else None
            else:
                codes_at = alloc(f"conv{index}_codes", matrix_rows * Q.row_bytes(padded_k, "int8", cfg.D),
                                 "pcm_weights", "immutable")
                scales_at = alloc(f"conv{index}_scales", matrix_rows * Q.scale_bytes(padded_k, "int8", cfg.D),
                                  "pcm_weights", "immutable")
                m.weight = QTensor.weight(codes_at, scales_at, matrix_rows, padded_k, "int8", cfg.D)
                m.weight_offsets = (codes_at, scales_at)
            self.ops.append(m)
            return output_at, m

        channels = spec.n_filters * 16
        at, first = op("conv", 0, "conv.conv", input_at, spec.input_rows,
                       spec.hidden, channels, 7, elu=False)
        self.layer_outputs[0] = (at, spec.input_rows, channels)
        rows = spec.input_rows
        for stage, ratio in enumerate(spec.ratios):
            index = 1 + stage * 3
            at, trans = op("transpose", index + 1, "convtr.convtr", at, rows,
                           channels, channels // 2, 2 * ratio, stride=ratio)
            self.layer_outputs[index] = (trans.activated.base, rows, channels)
            rows *= ratio; channels //= 2
            self.layer_outputs[index + 1] = (at, rows, channels)
            skip = at
            at, _ = op("conv", index + 2, "block.1.conv.conv", at, rows,
                        channels, channels // 2, 3)
            at, _ = op("conv", index + 2, "block.3.conv.conv", at, rows,
                        channels // 2, channels, 1, skip_at=skip)
            self.layer_outputs[index + 2] = (at, rows, channels)
        at, final = op("conv", 14, "conv.conv", at, rows, channels, 1, 3, last=True)
        self.layer_outputs[13] = (final.activated.base, rows, channels)
        self.layer_outputs[14] = (at, rows, 1)
        self.io["output"] = at
        self.state_end = -(-cursor["pcm_state"] // 4096) * 4096
        self.weight_end = -(-cursor["pcm_weights"] // 4096) * 4096
        if self.state_end > state_limit or self.weight_end > weight_limit:
            raise MemoryError("Aligned SEANet mapping exceeds capacity")
        self.required_bytes = max(self.state_end, self.weight_end)
        self.weight_bytes = sum(p["bytes"] for p in self.spans if p["region"] == "pcm_weights")

    def compile(self):
        if self.programs is None:
            programs = [trace_fit(_transpose if m.kind == "transpose" else _conv,
                                  self.cfg, 0, lambda m=m: {"m": m}).finish() for m in self.ops]
            if any(p.run_args or len(p) * 8 > self.cfg.IMEM_WORDS for p in programs):
                raise CompileError("SEANet requires host arguments or exceeds instruction capacity")
            self.programs = programs
        return self.programs

    def reset(self, backend):
        for at, size in self.reset_regions:
            backend.write(0, at, np.zeros(size, np.uint8))

    def upload(self, backend, weights):
        self.compile()  # reject capacity before touching lazy payloads
        weights = weights() if callable(weights) else weights
        if len(weights) != len(self.ops):
            raise ValueError("SEANet needs fourteen ordered convolution weights")
        prepared = []
        for m, record in zip(self.ops, weights):
            if any(record.get(key) != getattr(m, key) for key in ("kind", "model_index", "subpath")):
                raise ValueError("SEANet convolution order differs from native source")
            shape = ((m.cin, m.cout, m.kernel) if m.kind == "transpose"
                     else (m.cout, m.cin, m.kernel))
            value = _checked(record["weight"], shape, "SEANet convolution weight")
            bias = _checked(record["bias"], (m.cout,), "SEANet convolution bias")
            matrix = (value.transpose(2, 1, 0).reshape(m.kernel * m.cout, m.cin)
                      if m.kind == "transpose" else value.transpose(0, 2, 1).reshape(m.cout, -1))
            padded = np.zeros(m.matrix_shape, np.float32)
            padded[:, :matrix.shape[1]] = matrix
            prepared.append((m, padded, bias))

        def put(name, at, value):
            raw = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
            self.payload_hashes[name] = hashlib.sha256(memoryview(raw)).hexdigest()
            backend.write(0, at, raw)

        for index, (m, matrix, bias) in enumerate(prepared):
            put(f"conv{index}_bias", m.bias.base, bias)
            if m.kind == "transpose":
                put(f"conv{index}_raw_bias", m.raw_bias.base, np.tile(bias, m.kernel))
            if m.precision == "fp32":
                put(f"conv{index}_fp32", m.fp_data, matrix)
                continue
            codes, scales = m.weight_offsets
            batch = QC.Batch()
            batch.submit(matrix, "int8", self.cfg.D,
                lambda q, scales_value, index=index, codes=codes, scales=scales:
                (put(f"conv{index}_codes", codes, q), put(f"conv{index}_scales", scales, scales_value)))
            batch.wait()
        self.reset(backend)

    def budget(self):
        return dict(required_bytes=self.required_bytes, state_end=self.state_end, weight_end=self.weight_end,
            occupied_bytes=sum(p["bytes"] for p in self.spans), weight_bytes=self.weight_bytes,
            payload_bytes=sum(p["bytes"] for p in self.spans if p["kind"] == "immutable"),
            persistent_reset_bytes=sum(size for _, size in self.reset_regions),
            program_words=[len(p) * 8 for p in self.compile()], spans=self.spans,
            payload_sha256=self.payload_hashes, samples_per_frame=self.spec.samples,
            pcm_format="raw mono FP32; no clipping or scaling",
            final_precision=self.spec.final_precision,
            residual_precision=self.spec.residual_precision,
            precision=("FP32 final mono convolution; " if self.spec.final_precision == "fp32" else "") +
                      ("FP32 residual convolutions; " if self.spec.residual_precision == "fp32" else "") +
                      "other convolution weights/activations INT8; bias, history, overlap, ELU and residual FP32")
