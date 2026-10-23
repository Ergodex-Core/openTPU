"""Resident two-token input transformer, reusing qualified FP32 Mimi kernels.

The source encoder pair is copied by a device program, without a causal
upsampler. Encoder weights and streaming state are independent of the decoder.
This is an offline CPU/ISA component, not a complete input codec or server.
"""
import numpy as np

from .. import language as ol
from .. import waits
from ..compiler import CompileError, Tensor, trace_fit
from .personaplex_mimi_decoder import MimiDecoderImage, MimiDecoderSpec


@ol.jit
def _input_pair(m):
    value = ol.load(m.source)
    ol.store(m.snapshot, value)
    ol.store(m.output, value)


class MimiInputTransformerImage:
    def __init__(self, spec, cfg, *, input_at, state_base, weight_base,
                 state_limit=0x60000000, weight_limit=0xB0000000):
        if spec.precision != "fp32":
            raise ValueError("Input transformer requires explicitly qualified FP32 precision")
        capacity = min(cfg.DRAM_BYTES, 1 << 32)
        if (type(input_at) is not int or input_at % 128
                or not 0 <= input_at < input_at + 8 * spec.hidden <= capacity):
            raise MemoryError("Resident input transformer pair exceeds mapped capacity or alignment")
        self.spec, self.cfg, self.input_at = spec, cfg, input_at
        self.transformer = MimiDecoderImage(spec, cfg, latent_at=input_at,
            state_base=state_base, state_limit=state_limit,
            weight_base=weight_base, weight_limit=weight_limit)
        # Reuse the validated layer allocator, matrix packing and masks without
        # changing any decoder source. Its small upsampler allocation is unused
        # compatibility storage and its program is never executed here.
        self.spans = [dict(span, name="input_transformer_" + span["name"].removeprefix("decoder_"))
                      for span in self.transformer.spans]
        for span in self.spans:
            if span["base"] < input_at + 8 * spec.hidden and input_at < span["base"] + span["bytes"]:
                raise ValueError("Input transformer allocation overlaps the resident encoder pair")
        self.io = dict(self.transformer.io)
        self.io["input_snapshot"] = self.io["upsampled"]
        for name in ("state_end", "weight_end", "required_bytes", "weight_bytes", "layer_outputs",
                     "kv_heads", "cache_regions", "cap", "dpad"):
            setattr(self, name, getattr(self.transformer, name))
        self.payload_hashes = self.transformer.payload_hashes
        self.programs = None

    @classmethod
    def after_encoder(cls, encoder, spec=None):
        if not hasattr(encoder, "prefix_spans"):
            raise ValueError("Use the encoder appended to the established output-codec layout")
        spec = MimiDecoderSpec(precision="fp32") if spec is None else spec
        if encoder.spec.output_rows != 2 or encoder.spec.hidden != spec.hidden:
            raise ValueError("Encoder output must be the transformer's resident two-row pair")
        image = cls(spec, encoder.cfg, input_at=encoder.io["output"],
            state_base=encoder.state_end, state_limit=encoder.input_at + (32 << 20),
            weight_base=encoder.weight_end)
        image.prefix_spans = list(encoder.prefix_spans) + list(encoder.spans)
        image.input_state_limit = encoder.input_at + (32 << 20)
        slot_bytes = waits.area(encoder.cfg.IMEM_WORDS, encoder.cfg.SEQ_WC)
        next_slot = max(span["base"] + span["bytes"] for span in image.prefix_spans
                        if span["region"] == "program")
        image.program_slots = []
        for index in range(1 + spec.layers):
            if next_slot + slot_bytes > 0x05000000:
                raise MemoryError("Input transformer instructions exceed the existing program region")
            image.program_slots.append(next_slot)
            image.spans.append(dict(name=f"input_transformer_program{index}", base=next_slot,
                bytes=slot_bytes, region="program", kind="program", alignment=128))
            next_slot += slot_bytes
        ordered = sorted(image.prefix_spans + image.spans, key=lambda span: span["base"])
        for first, second in zip(ordered, ordered[1:]):
            if first["base"] + first["bytes"] > second["base"]:
                raise MemoryError("Input transformer overlaps an existing resident span: "
                                  + first["name"] + "/" + second["name"])
        return image

    def compile(self):
        if self.programs is None:
            h = self.spec.hidden
            args = dict(source=Tensor(self.input_at, (2, h), (h, 1)),
                        snapshot=Tensor(self.io["input_snapshot"], (2, h), (h, 1)),
                        output=Tensor(self.io["output"], (2, h), (h, 1)))
            from types import SimpleNamespace
            pair = trace_fit(_input_pair, self.cfg, 0, lambda: {"m": SimpleNamespace(**args)}).finish()
            programs = [pair, *self.transformer.compile()[1:]]
            if any(program.run_args or len(program) * 8 > self.cfg.IMEM_WORDS for program in programs):
                raise CompileError("Input transformer requires host arguments or exceeds IMEM")
            self.programs = programs
        return self.programs

    def upload(self, backend, layers, rope=None):
        self.compile()
        # The inherited allocator reserves 8 KiB for an upsample kernel. Keep
        # those unused bytes explicitly zero; no upsampling operation runs and
        # no checkpoint upsample payload is loaded or claimed as encoder work.
        self.transformer.upload(backend, np.zeros((self.spec.hidden, 4), np.float32), layers, rope)
        self.reset(backend)

    def reset(self, backend):
        # Native fresh-stream positions start at zero; previously written KV
        # remains masked, matching the established decoder's logical reset.
        backend.write(0, self.io["state"], np.zeros(1, np.float32))

    def budget(self):
        budget = self.transformer.budget()
        budget.update(spans=self.spans, occupied_bytes=sum(span["bytes"] for span in self.spans),
            program_words=[len(program) * 8 for program in self.compile()],
            input_shape=[2, self.spec.hidden],
            scope="Resident encoder pair -> FP32 encoder transformer; no upsampler execution",
            unused_legacy_upsample_bytes=16 * self.spec.hidden,
            input_semantics="Unscaled FP32 encoder pair copied by device program",
            cache_policy="Native two-token append-before-attention; delta<=0 ring defect retained")
        if hasattr(self, "prefix_spans"):
            joint = self.prefix_spans + self.spans
            budget.update(combined_occupied_bytes=sum(span["bytes"] for span in joint),
                combined_required_bytes=-(-max(span["base"] + span["bytes"]
                                               for span in joint) // 4096) * 4096,
                program_slots=self.program_slots,
                input_state_slack_bytes=self.input_state_limit - self.state_end,
                input_weight_slack_bytes=0xB0000000 - self.weight_end)
        return budget
