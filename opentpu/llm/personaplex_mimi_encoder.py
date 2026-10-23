"""Resident FP32 causal SEANet input encoder; CPU/ISA qualification only.

Raw mono PCM, convolution history, ELU, residuals and strided convolutions stay
in device DRAM/programs. No quantization, clipping or input normalization is
performed. Encoder transformer, rate conversion and RVQ encoding are separate
components; this module alone does not implement the input codec.
"""
from dataclasses import dataclass
import hashlib
import math
from types import SimpleNamespace

import numpy as np

from .. import language as ol
from ..compiler import CompileError, Tensor, trace_fit
from .. import waits
from .personaplex_audio_heads import _checked
from .personaplex_mimi_pcm import _elu


@dataclass(frozen=True)
class MimiEncoderSpec:
    hidden: int = 512
    n_filters: int = 64
    ratios: tuple = (8, 6, 5, 4)
    output_rows: int = 2

    def __post_init__(self):
        for name in ("hidden", "n_filters", "output_rows"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 1 << 16:
                raise ValueError(f"{name} must be a positive bounded integer")
        if self.n_filters % 2 or len(self.ratios) != 4 or any(
                type(x) is not int or not 1 < x <= 16 for x in self.ratios):
            raise ValueError("Encoder needs even filters and four bounded stride ratios")
        if self.samples > 1 << 20:
            raise ValueError("Encoder PCM frame is too large")

    @property
    def samples(self):
        return self.output_rows * math.prod(self.ratios)


@ol.jit
def _encoder_conv(m):
    # Build a time-major [history, new frame] view. Initial history is constant
    # zero padding; later frames retain the source's unfinished convolution
    # input, including cases where a chunk is shorter than its history.
    for block in ol.range(m.rows // m.input_tile):
        start = block * m.input_tile
        x = ol.load(m.input[start:start + m.input_tile, :])
        if m.elu:
            x = _elu(x)
            ol.store(m.activated[start:start + m.input_tile, :], x)
        ol.store(m.prepared[start + m.history:start + m.history + m.input_tile, :], x)
    for block in ol.range(m.output_rows // m.tile):
        start = block * m.tile
        window = ol.zeros((m.tile, m.padded_k))
        for tap in range(m.kernel):
            # Rows within a strided convolution window are separated by the
            # native stride. The gather and FP32 multiply/reduction execute on
            # device; no im2col or activation feedback is supplied by the host.
            view = Tensor(m.prepared.base + (start * m.stride + tap) * (4 * m.cin),
                          (m.tile, m.cin), (m.stride * m.cin, 1))
            window[:, tap * m.cin:(tap + 1) * m.cin].set(ol.load(view))
        for channel in ol.range(m.cout):
            coefficient = ol.load(Tensor(m.fp_data + channel * (4 * m.padded_k),
                                          (m.padded_k,), (1,)))
            reduced = ol.sum(window * coefficient[None, :], axis=1)
            column = ol.empty((m.tile, 1))
            column.set(reduced[:, None])
            column.set(column + ol.load(m.bias[channel:channel + 1])[None, :])
            if m.skip is not None:
                column.set(ol.load(m.skip[start:start + m.tile, channel:channel + 1]) + column)
            ol.store(m.output[start:start + m.tile, channel:channel + 1], column)
    if m.history:
        retained = ol.load(m.prepared[m.rows:m.rows + m.history, :])
        ol.store(m.prepared[:m.history, :], retained)
    if m.last:
        ol.store(m.frame, ol.load(m.frame) + 1.0)


class MimiEncoderImage:
    """Four strided SEANet stages with fixed programs and persistent history."""
    @classmethod
    def after_output_codec(cls, output_image, spec=None):
        """Append a disjoint encoder layout within the existing mapped DRAM.

        The output codec's narrow state reservation is nearly full. This
        software layout places input state after its immutable weights, in the
        existing free interval before the depth matrices at 0xB0000000. It
        reserves 32 MiB for input state, without reallocating output addresses.
        Program storage appends within the existing 16 MiB instruction region.
        This describes residency; it does not add a serving or board engine.
        """
        if output_image.pcm is None:
            raise ValueError("A complete output-codec layout is required")
        spec = MimiEncoderSpec() if spec is None else spec
        input_at = output_image.pcm.weight_end
        limit = input_at + (32 << 20)
        image = cls(spec, output_image.cfg, input_at=input_at,
                    state_base=-(-(input_at + 4 * spec.samples) // 128) * 128,
                    state_limit=limit, weight_base=limit)
        raw = dict(name="encoder_raw_pcm", base=input_at, bytes=4 * spec.samples,
                   region="encoder_state", kind="input", alignment=128)
        slot_bytes = waits.area(output_image.cfg.IMEM_WORDS, output_image.cfg.SEQ_WC)
        next_slot = max(p["base"] + p["bytes"] for p in output_image.spans
                        if p["region"] == "program")
        image.program_slots = []
        for index in range(len(image.ops)):
            if next_slot + slot_bytes > 0x05000000:
                raise MemoryError("Combined encoder instructions exceed reserved program region")
            image.program_slots.append(next_slot)
            image.spans.append(dict(name=f"encoder_program{index}", base=next_slot,
                bytes=slot_bytes, region="program", kind="program", alignment=128))
            next_slot += slot_bytes
        image.spans.insert(0, raw)
        image.prefix_spans = list(output_image.spans)
        ordered = sorted(image.prefix_spans + image.spans, key=lambda p: p["base"])
        for first, second in zip(ordered, ordered[1:]):
            if first["base"] + first["bytes"] > second["base"]:
                raise MemoryError("Input encoder layout overlaps existing resident span: "
                                  + first["name"] + "/" + second["name"])
        return image

    def __init__(self, spec, cfg, *, input_at, state_base, weight_base,
                 state_limit=0x60000000, weight_limit=0xB0000000):
        capacity = min(cfg.DRAM_BYTES, 1 << 32)
        if cfg.S != 1:
            raise ValueError("SEANet encoder requires one slice")
        for lo, hi in ((state_base, state_limit), (weight_base, weight_limit)):
            if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= capacity or lo % 128:
                raise MemoryError("Invalid encoder reserved mapping")
        if state_base < weight_limit and weight_base < state_limit:
            raise ValueError("Encoder state and weights overlap")
        if (type(input_at) is not int or input_at % 128
                or not 0 <= input_at < input_at + 4 * spec.samples <= capacity):
            raise MemoryError("Resident raw PCM input exceeds capacity or alignment")
        self.spec, self.cfg, self.input_at = spec, cfg, input_at
        self.spans, self.payload_hashes, self.ops = [], {}, []
        self.programs = None
        self.reset_regions, self.layer_outputs = [], [None] * 15
        cursor = {"encoder_state": state_base, "encoder_weights": weight_base}
        limits = {"encoder_state": state_limit, "encoder_weights": weight_limit}

        def alloc(name, size, region="encoder_state", kind="state"):
            at = -(-cursor[region] // 128) * 128
            if size <= 0 or at + size > limits[region]:
                raise MemoryError(f"Encoder {name} exceeds {region}")
            if at < input_at + 4 * spec.samples and input_at < at + size:
                raise ValueError("Encoder span overlaps resident raw PCM input")
            self.spans.append(dict(name="encoder_" + name, base=at, bytes=size,
                                   region=region, kind=kind, alignment=128))
            cursor[region] = at + size
            return at

        def tensor(at, rows, cols):
            return Tensor(at, (rows, cols), (cols, 1))

        frame_at = alloc("frame", 4)
        self.reset_regions.append((frame_at, 4))
        self.io = {"frame": frame_at}

        def op(model_index, subpath, source_at, rows, cin, cout, kernel,
               *, elu=True, stride=1, skip_at=None, last=False):
            index = len(self.ops)
            if rows % stride:
                raise ValueError("Encoder frame must be aligned to every source stride")
            out_rows = rows // stride
            padded_k = -(-(cin * kernel) // cfg.D) * cfg.D
            if padded_k > cfg.ACT_BLOCKS * cfg.D:
                raise MemoryError("Encoder convolution exceeds ACT RAM")
            out = alloc(f"conv{index}_output", 4 * out_rows * cout)
            m = SimpleNamespace(kind="conv", model_index=model_index, subpath=subpath,
                cin=cin, cout=cout, kernel=kernel, stride=stride, rows=rows,
                output_rows=out_rows, tile=math.gcd(out_rows, cfg.MCOLS),
                input_tile=math.gcd(rows, cfg.MCOLS), history=kernel - stride,
                padded_k=padded_k, matrix_shape=(cout, padded_k), precision="fp32",
                input=tensor(source_at, rows, cin), output=tensor(out, out_rows, cout),
                skip=tensor(skip_at, out_rows, cout) if skip_at is not None else None,
                elu=elu, last=last, frame=Tensor(frame_at, (1,), (1,)))
            if elu:
                m.activated = tensor(alloc(f"conv{index}_activated", 4 * rows * cin), rows, cin)
            m.bias = Tensor(alloc(f"conv{index}_bias", 4 * cout, kind="immutable"), (cout,), (1,))
            m.prepared = tensor(alloc(f"conv{index}_prepared", 4 * (rows + m.history) * cin),
                                 rows + m.history, cin)
            if m.history:
                self.reset_regions.append((m.prepared.base, 4 * m.history * cin))
            m.fp_data = alloc(f"conv{index}_fp32", 4 * cout * padded_k,
                              "encoder_weights", "immutable")
            self.ops.append(m)
            return out, m

        channels, rows = spec.n_filters, spec.samples
        at, _ = op(0, "conv.conv", input_at, rows, 1, channels, 7, elu=False)
        self.layer_outputs[0] = (at, rows, channels)
        for stage, stride in enumerate(reversed(spec.ratios)):
            index = 1 + stage * 3
            skip = at
            at, _ = op(index, "block.1.conv.conv", at, rows, channels, channels // 2, 3)
            at, _ = op(index, "block.3.conv.conv", at, rows, channels // 2,
                        channels, 1, skip_at=skip)
            self.layer_outputs[index] = (at, rows, channels)
            at, down = op(index + 2, "conv.conv", at, rows, channels, channels * 2,
                           2 * stride, stride=stride)
            self.layer_outputs[index + 1] = (down.activated.base, rows, channels)
            rows //= stride
            channels *= 2
            self.layer_outputs[index + 2] = (at, rows, channels)
        at, final = op(14, "conv.conv", at, rows, channels, spec.hidden, 3, last=True)
        self.layer_outputs[13] = (final.activated.base, rows, channels)
        self.layer_outputs[14] = (at, rows, spec.hidden)
        self.io["output"] = at
        self.state_end = -(-cursor["encoder_state"] // 4096) * 4096
        self.weight_end = -(-cursor["encoder_weights"] // 4096) * 4096
        if self.state_end > state_limit or self.weight_end > weight_limit:
            raise MemoryError("Aligned encoder mapping exceeds capacity")
        self.required_bytes = max(self.state_end, self.weight_end)
        self.weight_bytes = sum(p["bytes"] for p in self.spans if p["region"] == "encoder_weights")

    def compile(self):
        if self.programs is None:
            programs = [trace_fit(_encoder_conv, self.cfg, 0, lambda m=m: {"m": m}).finish()
                        for m in self.ops]
            if any(p.run_args or len(p) * 8 > self.cfg.IMEM_WORDS for p in programs):
                raise CompileError("Encoder requires host arguments or exceeds instruction capacity")
            self.programs = programs
        return self.programs

    def reset(self, backend):
        for at, size in self.reset_regions:
            backend.write(0, at, np.zeros(size, np.uint8))

    def upload(self, backend, weights):
        self.compile()
        weights = weights() if callable(weights) else weights
        if len(weights) != len(self.ops):
            raise ValueError("Encoder needs fourteen ordered convolution weights")
        prepared = []
        for m, record in zip(self.ops, weights):
            if any(record.get(key) != getattr(m, key) for key in ("kind", "model_index", "subpath")):
                raise ValueError("Encoder convolution order differs from native source")
            value = _checked(record["weight"], (m.cout, m.cin, m.kernel), "Encoder convolution weight")
            bias = _checked(record["bias"], (m.cout,), "Encoder convolution bias")
            matrix = value.transpose(0, 2, 1).reshape(m.cout, -1)
            padded = np.zeros(m.matrix_shape, np.float32)
            padded[:, :matrix.shape[1]] = matrix
            prepared.append((m, padded, bias))

        def put(name, at, value):
            raw = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
            self.payload_hashes[name] = hashlib.sha256(memoryview(raw)).hexdigest()
            backend.write(0, at, raw)

        for index, (m, matrix, bias) in enumerate(prepared):
            put(f"conv{index}_bias", m.bias.base, bias)
            put(f"conv{index}_fp32", m.fp_data, matrix)
        self.reset(backend)

    def budget(self):
        budget = dict(required_bytes=self.required_bytes, state_end=self.state_end, weight_end=self.weight_end,
            occupied_bytes=sum(p["bytes"] for p in self.spans), weight_bytes=self.weight_bytes,
            payload_bytes=sum(p["bytes"] for p in self.spans if p["kind"] == "immutable"),
            persistent_reset_bytes=sum(size for _, size in self.reset_regions),
            program_words=[len(p) * 8 for p in self.compile()], spans=self.spans,
            payload_sha256=self.payload_hashes, samples_per_frame=self.spec.samples,
            output_rows=self.spec.output_rows, hidden=self.spec.hidden,
            input_format="raw mono FP32; no clipping, scaling or normalization",
            precision="FP32 coefficient rows, activations, bias, ELU, history and residuals",
            stream_policy="Full stride-aligned frames; zero initial kernel-stride history; device retained input")
        if hasattr(self, "prefix_spans"):
            joint = self.prefix_spans + self.spans
            budget.update(combined_occupied_bytes=sum(p["bytes"] for p in joint),
                          combined_required_bytes=-(-max(p["base"] + p["bytes"]
                                                        for p in joint) // 4096) * 4096,
                          program_slots=self.program_slots,
                          input_state_reservation_bytes=32 << 20,
                          input_state_slack_bytes=self.input_at + (32 << 20) - self.state_end,
                          input_weight_slack_bytes=0xB0000000 - self.weight_end)
        return budget
