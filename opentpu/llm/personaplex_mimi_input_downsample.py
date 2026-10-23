"""Resident learned Mimi input downsample; CPU/ISA qualification only.

One causal, stride-two FP32 Conv1d consumes each encoder-transformer pair and
emits one 512-wide row. Native replicate padding is initialized on the device
from the first incoming row. No host activation feedback, scaling, or RVQ is
part of this component.
"""
from dataclasses import dataclass
import hashlib
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from .. import waits
from ..compiler import CompileError, Tensor, trace_fit
from .personaplex_audio_heads import _checked


@dataclass(frozen=True)
class MimiInputDownsampleSpec:
    hidden: int = 512
    input_rows: int = 2
    output_rows: int = 1

    def __post_init__(self):
        if type(self.hidden) is not int or not 0 < self.hidden < 1 << 16:
            raise ValueError("hidden must be a positive bounded integer")
        if self.input_rows != 2 or type(self.input_rows) is not int:
            raise ValueError("input_rows must be two aligned source rows")
        if self.output_rows != 1 or type(self.output_rows) is not int:
            raise ValueError("output_rows must be one downsampled row")


@ol.jit
def _downsample(m):
    # StreamingConv1d uses replicate mode and its first left padding is two
    # copies of the first real input row. The initial flag is resident: one on
    # reset, zero after frame zero. This conditional has no host argument.
    x = ol.load(m.input)
    ol.store(m.prepared[2:4, :], x)
    builder = ol.current()
    count = builder.scratch()
    # RLD addresses TMEM and converts the FP32 word to an integer count.
    first = ol.load(m.first)
    builder.emit(I.rld(count, first.base, bound=(0, 1),
                       comment="native replicate padding pending"))
    conditional = builder.begin_loop(0, rcount=count)
    first_row = ol.load(m.input[0:1, :])
    ol.store(m.prepared[0:1, :], first_row)
    ol.store(m.prepared[1:2, :], first_row)
    builder.end_loop(conditional)
    builder.unscratch(count)

    window = ol.zeros((1, m.padded_k))
    for tap in range(4):
        window[:, tap * m.cin:(tap + 1) * m.cin].set(
            ol.load(m.prepared[tap:tap + 1, :]))
    for channel in ol.range(m.cout):
        coefficient = ol.load(Tensor(m.fp_data + channel * 4 * m.padded_k,
                                      (m.padded_k,), (1,)))
        reduced = ol.sum(window * coefficient[None, :], axis=1)
        column = ol.empty((1, 1))
        column.set(reduced[:, None])
        # The source convolution has bias=False. This immutable zero vector
        # satisfies the existing FP32-convolution shape without learned bias.
        column.set(column + ol.load(m.bias[channel:channel + 1])[None, :])
        ol.store(m.output[:, channel:channel + 1], column)

    retained = ol.load(m.prepared[2:4, :])
    ol.store(m.prepared[0:2, :], retained)
    ol.store(m.first, ol.zeros((1,)))
    ol.store(m.frame, ol.load(m.frame) + 1.0)


class MimiInputDownsampleImage:
    """Fixed one-program layout for the pinned learned stride-two convolution."""

    def __init__(self, spec, cfg, *, input_at, state_base, weight_base,
                 state_limit=0x60000000, weight_limit=0xB0000000):
        capacity = min(cfg.DRAM_BYTES, 1 << 32)
        if cfg.S != 1:
            raise ValueError("Input downsample requires one slice")
        for low, high in ((state_base, state_limit), (weight_base, weight_limit)):
            if (type(low) is not int or type(high) is not int or low % 128
                    or not 0 <= low < high <= capacity):
                raise MemoryError("Invalid input downsample reserved mapping")
        if state_base < weight_limit and weight_base < state_limit:
            raise ValueError("Input downsample state and weights overlap")
        if (type(input_at) is not int or input_at % 128
                or not 0 <= input_at < input_at + 8 * spec.hidden <= capacity):
            raise MemoryError("Resident input pair exceeds capacity or alignment")
        self.spec, self.cfg, self.input_at = spec, cfg, input_at
        self.spans, self.payload_hashes = [], {}
        self.programs = None
        cursor = {"input_downsample_state": state_base,
                  "input_downsample_weights": weight_base}
        limits = {"input_downsample_state": state_limit,
                  "input_downsample_weights": weight_limit}

        def alloc(name, size, region="input_downsample_state", kind="state"):
            address = -(-cursor[region] // 128) * 128
            if size <= 0 or address + size > limits[region]:
                raise MemoryError("Input downsample " + name + " exceeds " + region)
            if address < input_at + 8 * spec.hidden and input_at < address + size:
                raise ValueError("Input downsample allocation overlaps its input")
            self.spans.append(dict(name="input_downsample_" + name,
                                   base=address, bytes=size, region=region,
                                   kind=kind, alignment=128))
            cursor[region] = address + size
            return address

        hidden = spec.hidden
        padded_k = -(-(4 * hidden) // cfg.D) * cfg.D
        if padded_k > cfg.ACT_BLOCKS * cfg.D:
            raise MemoryError("Input downsample convolution exceeds ACT RAM")
        frame_at = alloc("frame", 4)
        first_at = alloc("first", 4)
        output_at = alloc("output", 4 * hidden)
        prepared_at = alloc("prepared", 4 * 4 * hidden)
        bias_at = alloc("zero_bias", 4 * hidden, kind="immutable")
        fp_data = alloc("fp32", 4 * hidden * padded_k,
                        "input_downsample_weights", "immutable")
        self.op = SimpleNamespace(kind="conv", cin=hidden, cout=hidden,
            rows=2, output_rows=1, kernel=4, stride=2, history=2,
            padded_k=padded_k, matrix_shape=(hidden, padded_k),
            precision="fp32", elu=False, skip=None,
            input=Tensor(input_at, (2, hidden), (hidden, 1)),
            output=Tensor(output_at, (1, hidden), (hidden, 1)),
            prepared=Tensor(prepared_at, (4, hidden), (hidden, 1)),
            bias=Tensor(bias_at, (hidden,), (1,)),
            fp_data=fp_data, first=Tensor(first_at, (1,), (1,)),
            frame=Tensor(frame_at, (1,), (1,)))
        self.io = dict(input=input_at, output=output_at, frame=frame_at,
                       first=first_at)
        self.reset_regions = [(frame_at, 4), (first_at, 4),
                              (prepared_at, 8 * hidden)]
        self.layer_outputs = [(output_at, 1, hidden)]
        self.state_end = -(-cursor["input_downsample_state"] // 4096) * 4096
        self.weight_end = -(-cursor["input_downsample_weights"] // 4096) * 4096
        if self.state_end > state_limit or self.weight_end > weight_limit:
            raise MemoryError("Aligned input downsample mapping exceeds capacity")
        self.required_bytes = max(self.state_end, self.weight_end)
        self.weight_bytes = sum(span["bytes"] for span in self.spans
                                if span["region"] == "input_downsample_weights")

    @classmethod
    def after_transformer(cls, transformer, spec=None):
        if not hasattr(transformer, "prefix_spans") or not hasattr(transformer, "input_state_limit"):
            raise ValueError("Use the input transformer appended to the resident encoder")
        spec = MimiInputDownsampleSpec() if spec is None else spec
        if transformer.spec.hidden != spec.hidden:
            raise ValueError("Input transformer and downsample hidden widths differ")
        image = cls(spec, transformer.cfg, input_at=transformer.io["output"],
                    state_base=transformer.state_end,
                    state_limit=transformer.input_state_limit,
                    weight_base=transformer.weight_end)
        image.input_state_limit = transformer.input_state_limit
        image.prefix_spans = list(transformer.prefix_spans) + list(transformer.spans)
        slot_bytes = waits.area(transformer.cfg.IMEM_WORDS,
                                transformer.cfg.SEQ_WC)
        next_slot = max(span["base"] + span["bytes"] for span in image.prefix_spans
                        if span["region"] == "program")
        if next_slot + slot_bytes > 0x05000000:
            raise MemoryError("Input downsample instructions exceed existing program region")
        image.program_slots = [next_slot]
        image.spans.append(dict(name="input_downsample_program0", base=next_slot,
                                bytes=slot_bytes, region="program", kind="program",
                                alignment=128))
        ordered = sorted(image.prefix_spans + image.spans,
                         key=lambda span: span["base"])
        for first, second in zip(ordered, ordered[1:]):
            if first["base"] + first["bytes"] > second["base"]:
                raise MemoryError("Input downsample overlaps resident span: "
                                  + first["name"] + "/" + second["name"])
        return image

    def compile(self):
        if self.programs is None:
            program = trace_fit(_downsample, self.cfg, 0,
                                lambda: {"m": self.op}).finish()
            if program.run_args or len(program) * 8 > self.cfg.IMEM_WORDS:
                raise CompileError("Input downsample needs host arguments or exceeds IMEM")
            self.programs = [program]
        return self.programs

    def upload(self, backend, weight):
        self.compile()
        value = _checked(weight, (self.spec.hidden, self.spec.hidden, 4),
                         "Mimi learned input downsample weight")
        matrix = value.transpose(0, 2, 1).reshape(self.spec.hidden, -1)
        packed = np.zeros(self.op.matrix_shape, np.float32)
        packed[:, :matrix.shape[1]] = matrix
        zero_bias = np.zeros(self.spec.hidden, np.float32)
        for name, at, payload in (("fp32", self.op.fp_data, packed),
                                  ("zero_bias", self.op.bias.base, zero_bias)):
            raw = np.ascontiguousarray(payload).view(np.uint8).reshape(-1)
            self.payload_hashes[name] = hashlib.sha256(memoryview(raw)).hexdigest()
            backend.write(0, at, raw)
        self.reset(backend)

    def reset(self, backend):
        backend.write(0, self.io["frame"], np.zeros(1, np.float32))
        backend.write(0, self.io["first"], np.ones(1, np.float32))
        backend.write(0, self.op.prepared.base,
                      np.zeros(2 * self.spec.hidden, np.float32))

    def budget(self):
        budget = dict(required_bytes=self.required_bytes, state_end=self.state_end,
            weight_end=self.weight_end, occupied_bytes=sum(s["bytes"] for s in self.spans),
            weight_bytes=self.weight_bytes,
            payload_bytes=sum(s["bytes"] for s in self.spans if s["kind"] == "immutable"),
            persistent_reset_bytes=sum(size for _, size in self.reset_regions),
            program_words=[len(program) * 8 for program in self.compile()],
            spans=self.spans, payload_sha256=self.payload_hashes,
            input_shape=[2, self.spec.hidden], output_shape=[1, self.spec.hidden],
            precision="FP32 learned ConvDownsample1d; no learned bias or scaling",
            stream_policy="First two history rows replicate initial input row; then retain previous pair")
        if hasattr(self, "prefix_spans"):
            joint = self.prefix_spans + self.spans
            budget.update(combined_occupied_bytes=sum(s["bytes"] for s in joint),
                combined_required_bytes=-(-max(s["base"] + s["bytes"]
                                               for s in joint) // 4096) * 4096,
                program_slots=self.program_slots,
                input_state_slack_bytes=self.input_state_limit - self.state_end,
                input_weight_slack_bytes=0xB0000000 - self.weight_end)
        return budget
