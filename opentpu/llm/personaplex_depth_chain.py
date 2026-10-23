"""Software-qualified resident depth chain; no complete speech or physical backend.

Immutable payloads are packed offline into sparse final P/B spans. Per-frame
model operations and greedy token feedback execute through ISA programs only.
"""
from dataclasses import dataclass
import copy
import hashlib
import operator
import time
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from .. import waits
from ..compiler import Affine, CompileError, DevVar, Tensor, trace_fit
from ..isasim import Config
from . import qwen3 as G
from .personaplex_audio_heads import AudioHeadImage, _audio_head, _checked
from .personaplex_depth_input import DepthInputImage, DepthInputSpec
from .personaplex_depth_layer import DepthLayerImage, DepthLayerSpec
from .personaplex_mimi import MimiRVQImage, MimiRVQSpec
from .personaplex_mimi_decoder import MimiDecoderImage, MimiDecoderSpec
from .personaplex_mimi_pcm import MimiPCMImage, MimiPCMSpec


@dataclass(frozen=True)
class DepthChainSpec:
    hidden: int = 4096
    depth: int = 1024
    n_heads: int = 16
    ffn: int = 2816
    steps: int = 16
    layers: int = 6
    text_vocab: int = 32001
    audio_vocab: int = 2049
    head_vocab: int = 2048
    cache_policy: str = "native"

    def __post_init__(self):
        DepthInputSpec(self.hidden, self.depth, self.steps, self.text_vocab, self.audio_vocab)
        self.layer_spec()
        if type(self.head_vocab) is not int or not 0 < self.head_vocab <= self.audio_vocab:
            raise ValueError("head vocabulary must fit the previous-audio embedding table")

    def layer_spec(self):
        return DepthLayerSpec(hidden=self.depth, n_heads=self.n_heads, ffn=self.ffn,
                              steps=self.steps, layers=self.layers, cache_policy=self.cache_policy)


@ol.jit
def _resident_head(m):
    b = ol.current()
    index = ol.load(m.counter) - 1.0  # depth increments once after all layers
    token = _audio_head.fn(m.head, index=index)
    r = b.scratch()
    b.emit(I.rld(r, index.base, mul=4, bound=(0, 4 * (m.steps - 1)),
                 comment="resident audio sequence slot"))
    ol.store(Tensor(Affine(m.audio_ids) + DevVar("audio sequence offset", r),
                    (1,), (1,)), token)
    b.unscratch(r)


class DepthChainImage:
    """Final sparse P/B allocation and relocated component descriptors."""
    REGIONS = {
        "control": (0, 0x10000), "kv": (0x02110000, 0x02510000),
        "scratch": (0x02510000, 0x03510000), "program": (0x04000000, 0x05000000),
        "embeddings": (0x44498000, 0x539B8000),
        "heads": (0x5B7D1400, 0x5D831400), "B": (0xB0000000, 1 << 32),
    }

    def __init__(self, spec, cfg, *, mimi_spec=None, decoder_spec=None, pcm_spec=None):
        if cfg.S != 1 or cfg.DRAM_BYTES > 1 << 32:
            raise ValueError("resident depth chain needs one slice and unsigned32 mapped DRAM")
        self.spec, self.cfg = spec, cfg
        self.spans, self.payload_hashes = [], {}
        self.programs = None
        source_condition = DepthInputImage(DepthInputSpec(spec.hidden, spec.depth, spec.steps,
                                                        spec.text_vocab, spec.audio_vocab), cfg)
        source_depth = DepthLayerImage(spec.layer_spec(), cfg)
        source_head = AudioHeadImage(cfg, hidden=spec.depth, codebooks=spec.steps, vocab=spec.head_vocab)
        self.sources = {"condition": source_condition, "depth": source_depth, "head": source_head}
        cursor = {name: bounds[0] for name, bounds in self.REGIONS.items()}

        def alloc(name, size, region, alignment=128, source=None, source_at=None, kind="state"):
            lo, hi = self.REGIONS[region]
            at = -(-cursor[region] // alignment) * alignment
            end = at + size
            if size <= 0 or at < lo or end > hi or end > min(cfg.DRAM_BYTES, 1 << 32):
                raise MemoryError(f"{name}: span {at:#x}+{size} exceeds {region}/mapped capacity")
            if any(at < p["base"] + p["bytes"] and p["base"] < end for p in self.spans):
                raise ValueError(f"overlapping resident span: {name}")
            span = {"name": name, "region": region, "base": at, "bytes": size,
                    "alignment": alignment, "kind": kind}
            if source is not None:
                span.update(source=source, source_at=source_at)
            self.spans.append(span); cursor[region] = end
            return at

        control = alloc("control", 8, "control", 64)
        audio_ids = alloc("audio_ids", 4 * spec.steps, "control", 64)
        self.io = {"control": control, "counter": control, "previous_id": control + 4,
                   "audio_ids": audio_ids,
                   "hidden": alloc("hidden", 4 * spec.hidden, "scratch"),
                   "projected": alloc("projected", 4 * spec.depth, "scratch"),
                   "conditioned": alloc("conditioned", 4 * spec.depth, "scratch"),
                   "output": alloc("output", 4 * spec.depth, "scratch"),
                   "logits": alloc("logits", 4 * spec.head_vocab, "scratch")}
        self.condition = copy.copy(source_condition)
        self.condition.io = {"hidden": self.io["hidden"], "control": control,
                             "projected": self.io["projected"], "output": self.io["conditioned"]}
        self.condition.embeddings_at = alloc("step_embeddings", source_condition.embedding_bytes,
            "embeddings", source="condition", source_at=source_condition.embeddings_at, kind="immutable")
        self.depth = copy.copy(source_depth)
        self.depth.io = {"hidden": self.io["conditioned"], "output": self.io["output"], "state": control}
        self.depth.masks_at = alloc("depth_masks", 4 * spec.steps * source_depth.cap,
            "scratch", source="depth", source_at=source_depth.masks_at, kind="immutable")
        self.depth.norm_offsets = []
        for index, norms in enumerate(source_depth.norm_offsets):
            self.depth.norm_offsets.append(tuple(alloc(f"norm{index}_{part}", 4 * spec.depth,
                "scratch", source="depth", source_at=at, kind="immutable") for part, at in enumerate(norms)))
        self.depth.cache_regions, self.depth.kv_heads = [], []
        for index, (at, size) in enumerate(source_depth.cache_regions):
            new = alloc(f"kv{index}", size, "kv")
            self.depth.cache_regions.append((new, size))
            self.depth.kv_heads.append({head: {name: value + new - at for name, value in entries.items()}
                                       for head, entries in source_depth.kv_heads[index].items()})
        self.depth.bank_offsets = [alloc(f"depth_bank{index}", spec.steps * source_depth.bank_stride,
            "B", 4096, "depth", at, "immutable") for index, at in enumerate(source_depth.bank_offsets)]
        self.condition.projection_offsets = (
            alloc("projection_codes", source_condition.weight_code_bytes, "B", 4096,
                  "condition", source_condition.projection_offsets[0], "immutable"),
            alloc("projection_scales", spec.steps * source_condition.scale_stride, "B", 4096,
                  "condition", source_condition.projection_offsets[1], "immutable"))
        self.head = copy.copy(source_head)
        self.head.io = {"hidden": self.io["output"], "codebook": control,
                        "logits": self.io["logits"], "token_id": self.io["previous_id"]}
        self.head.head_offsets = (
            alloc("head_codes", source_head.weight_code_bytes, "heads", max(64, cfg.D),
                  "head", source_head.head_offsets[0], "immutable"),
            alloc("head_scales", spec.steps * source_head.scale_stride, "heads", 64,
                  "head", source_head.head_offsets[1], "immutable"))
        self.mimi = None
        if mimi_spec is not None:
            if mimi_spec.vocab != spec.head_vocab:
                raise ValueError("Mimi vocabulary must equal the device audio-head vocabulary")
            self.mimi = MimiRVQImage(mimi_spec, cfg, audio_ids_at=audio_ids,
                                     counter_at=control, expected_steps=spec.steps)
            for span in self.mimi.spans:
                if any(span["base"] < p["base"] + p["bytes"] and
                       p["base"] < span["base"] + span["bytes"] for p in self.spans):
                    raise ValueError("Mimi span overlaps a resident depth span")
            self.spans.extend(self.mimi.spans)
        self.decoder = None
        if decoder_spec is not None:
            if self.mimi is None or decoder_spec.hidden != self.mimi.spec.latent:
                raise ValueError("Mimi decoder needs a matching resident RVQ latent")
            self.decoder = MimiDecoderImage(decoder_spec, cfg, latent_at=self.mimi.io["output"],
                                             state_base=self.mimi.required_bytes)
            for span in self.decoder.spans:
                if any(span["base"] < p["base"] + p["bytes"] and
                       p["base"] < span["base"] + span["bytes"] for p in self.spans):
                    raise ValueError("Mimi decoder overlaps an existing resident span")
            self.spans.extend(self.decoder.spans)
        self.pcm = None
        if pcm_spec is not None:
            if self.decoder is None or pcm_spec.hidden != self.decoder.spec.hidden or pcm_spec.input_rows != 2:
                raise ValueError("SEANet PCM needs a matching resident two-token decoder")
            self.pcm = MimiPCMImage(pcm_spec, cfg, input_at=self.decoder.io["output"],
                state_base=self.decoder.state_end, weight_base=self.decoder.weight_end)
            for span in self.pcm.spans:
                if any(span["base"] < p["base"] + p["bytes"] and
                       p["base"] < span["base"] + span["bytes"] for p in self.spans):
                    raise ValueError("SEANet PCM overlaps an existing resident span")
            self.spans.extend(self.pcm.spans)
        program_count = ((4 if self.mimi else 3) + (1 + decoder_spec.layers if self.decoder else 0)
                         + (len(self.pcm.ops) if self.pcm else 0))
        self.program_slots = [alloc(f"program{index}", waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC),
                                    "program", 4096, kind="program")
                              for index in range(program_count)]
        self.required_bytes = -(-max(p["base"] + p["bytes"] for p in self.spans) // 4096) * 4096
        if self.required_bytes > min(cfg.DRAM_BYTES, 1 << 32):
            raise MemoryError("aligned resident mapping exceeds device capacity")
        self.B_used_bytes = cursor["B"] - self.REGIONS["B"][0]
        self.kv_bytes = source_depth.kv_bytes
        self.weight_bytes = (source_condition.weight_bytes + source_depth.weight_bytes + source_head.weight_bytes)
        if self.mimi is not None:
            self.weight_bytes += self.mimi.weight_bytes
        if self.decoder is not None:
            self.weight_bytes += self.decoder.weight_bytes
            self.kv_bytes += self.decoder.kv_bytes
        if self.pcm is not None:
            self.weight_bytes += self.pcm.weight_bytes

    def compile(self):
        if self.programs is None:
            condition = self.condition.compile()[0]
            depth = self.depth.compile()[0]
            m = SimpleNamespace(counter=Tensor(self.io["counter"], (1,), (1,)),
                head=self.head.descriptors(), audio_ids=self.io["audio_ids"], steps=self.spec.steps)
            b = trace_fit(_resident_head, self.cfg, 0, lambda: {"m": m})
            head = b.finish()
            self.programs = [condition, depth, head]
            if self.mimi is not None:
                self.programs.extend(self.mimi.compile())
            if self.decoder is not None:
                self.programs.extend(self.decoder.compile())
            if self.pcm is not None:
                self.programs.extend(self.pcm.compile())
            for program in self.programs:
                if program.run_args or len(program) * 8 > self.cfg.IMEM_WORDS:
                    raise CompileError("resident stage has host arguments or exceeds instruction capacity")
        return self.programs

    def budget(self):
        return {"mapped_bytes": self.cfg.DRAM_BYTES, "required_bytes": self.required_bytes,
                "occupied_bytes": sum(p["bytes"] for p in self.spans),
                "payload_bytes": sum(p["bytes"] for p in self.spans if p["kind"] == "immutable"),
                "weight_bytes": self.weight_bytes, "kv_bytes": self.kv_bytes,
                "B_used_bytes": self.B_used_bytes,
                "B_slack_bytes": self.REGIONS["B"][1] - self.REGIONS["B"][0] - self.B_used_bytes,
                "program_words": [len(p) * 8 for p in self.compile()],
                "program_slots": self.program_slots, "spans": self.spans,
                "payload_sha256": {**self.payload_hashes,
                    **({"mimi_" + k: v for k, v in self.mimi.payload_hashes.items()}
                       if self.mimi else {}),
                    **({"decoder_" + k: v for k, v in self.decoder.payload_hashes.items()}
                       if self.decoder else {}),
                    **({"pcm_" + k: v for k, v in self.pcm.payload_hashes.items()}
                       if self.pcm else {})},
                "mimi": self.mimi.budget() if self.mimi else None,
                "decoder": self.decoder.budget() if self.decoder else None,
                "pcm": self.pcm.budget() if self.pcm else None}

    def upload(self, backend, projections, embeddings, layers, head_bank):
        """Pack offline using established component builders; upload named spans only."""
        for stage in ("condition", "depth", "head"):
            source = self.sources[stage]
            if stage == "condition":
                image = source.build(projections, embeddings)[0]
            elif stage == "depth":
                image = source.build(layers)[0]
            else:
                source = copy.copy(source); source._head_bank = head_bank
                image = source.build()[0]
            for span in self.spans:
                if span.get("source") == stage:
                    value = image[span["source_at"]:span["source_at"] + span["bytes"]]
                    if value.size != span["bytes"]:
                        raise ValueError("sparse source span exceeds packed payload")
                    self.payload_hashes[span["name"]] = hashlib.sha256(memoryview(value)).hexdigest()
                    backend.write(0, span["base"], value)
            del image


class DepthChainEngine:
    """ISA-only runner; model compute/feedback remain device programs, not host math."""
    def __init__(self, projections, embeddings, layers, head_bank, *, spec=None, cfg=None,
                 backend="isa", mimi_tables=None, mimi_projections=None, mimi_spec=None,
                 decoder_weights=None, upsample_weight=None, decoder_spec=None, decoder_rope=None,
                 pcm_weights=None, pcm_spec=None):
        if backend != "isa":
            raise ValueError("resident depth chain is software/ISA only; no physical backend is qualified")
        self.spec = spec or DepthChainSpec()
        self._closed = self._failed = self._ready = self._active = False
        self._mimi_ready = False
        self._decoder_ready = self._decoder_consumed = False
        self._pcm_ready = self._pcm_consumed = False
        self.decoder_pos = 0
        if (mimi_tables is None) != (mimi_projections is None):
            raise ValueError("Mimi tables and projections must be supplied together")
        if mimi_spec is not None and mimi_tables is None:
            raise ValueError("Mimi specification requires its offline payloads")
        if (decoder_weights is None) != (upsample_weight is None):
            raise ValueError("Decoder layers and upsample weight must be supplied together")
        if decoder_weights is None and (decoder_spec is not None or decoder_rope is not None):
            raise ValueError("Decoder specification/coefficients require offline payloads")
        if decoder_weights is not None and mimi_tables is None:
            raise ValueError("Decoder requires resident Mimi RVQ")
        if pcm_weights is None and pcm_spec is not None:
            raise ValueError("SEANet PCM specification requires offline payloads")
        if pcm_weights is not None and decoder_weights is None:
            raise ValueError("SEANet PCM requires a resident Mimi decoder")
        self.poss, self.last_stats = [0], {}
        self.cfg = cfg or Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                                TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True, DRAM_BYTES=1 << 32)
        self.image = DepthChainImage(self.spec, self.cfg,
            mimi_spec=(mimi_spec or MimiRVQSpec()) if mimi_tables is not None else None,
            decoder_spec=(decoder_spec or MimiDecoderSpec()) if decoder_weights is not None else None,
            pcm_spec=(pcm_spec or MimiPCMSpec()) if pcm_weights is not None else None)
        self.programs = self.image.compile()  # capacity before any lazy payload reads
        self.backend = G.IsaBackend(self.cfg, [np.zeros(self.cfg.DRAM_BYTES, np.uint8)], adopt=True)
        try:
            self.image.upload(self.backend, projections, embeddings, layers, head_bank)
            if self.image.mimi is not None:
                self.image.mimi.upload(self.backend, mimi_tables, mimi_projections)
            if self.image.decoder is not None:
                self.image.decoder.upload(self.backend, upsample_weight, decoder_weights, decoder_rope)
            if self.image.pcm is not None:
                self.image.pcm.upload(self.backend, pcm_weights)
        except BaseException:
            self.close(); raise

    @property
    def pos(self):
        return self.poss[0]

    def _check(self):
        if self._closed:
            raise RuntimeError("resident depth chain is closed")
        if self._failed:
            raise RuntimeError("previous resident depth chain failed; reset required")

    def begin_frame(self, hidden, text_id):
        self._check()
        if self.image.decoder is not None:
            if self.decoder_pos >= self.image.decoder.spec.max_frames:
                raise RuntimeError("decoder frame limit reached; reset required")
            if self._active and not self._decoder_consumed:
                raise RuntimeError("complete the pending decoder frame or reset before a new frame")
        if self.image.pcm is not None and self._active and not self._pcm_consumed:
            raise RuntimeError("complete the pending PCM frame or reset before a new frame")
        value = _checked(hidden, (self.spec.hidden,), "temporal hidden")
        if isinstance(text_id, (bool, np.bool_)):
            raise ValueError("text ID must be integral")
        try:
            token = operator.index(text_id)
        except TypeError as error:
            raise ValueError("text ID must be integral") from error
        if not -1 <= token < self.spec.text_vocab:
            raise ValueError("text ID outside vocabulary")
        self._ready = self._active = self._mimi_ready = False
        self._decoder_ready = self._decoder_consumed = False
        self._pcm_ready = self._pcm_consumed = False
        try:
            self.backend.write(0, self.image.io["hidden"], value)
            self.backend.write(0, self.image.io["control"], np.array([0, token], np.float32))
        except BaseException:
            self._failed = True; raise
        self.poss[0] = 0; self._active = True

    def step(self):
        self._check()
        if not self._active:
            raise RuntimeError("begin a frame before resident depth steps")
        if self.pos >= self.spec.steps:
            raise RuntimeError("depth pass is full; reset/begin a new frame")
        self._ready = self._mimi_ready = False; started = time.perf_counter(); stages = []
        try:
            for program in self.programs[:3]:
                stages.append(self.backend.run([program]))
        except BaseException:
            self._failed = True; raise
        self.poss[0] += 1; self._ready = True
        self.last_stats = {"step": self.pos - 1, "stages": stages,
                           "cpu_wall_seconds": time.perf_counter() - started}

    def run_frame(self, hidden, text_id):
        self.begin_frame(hidden, text_id)
        for _ in range(self.spec.steps):
            self.step()

    def run_mimi(self):
        """Consume resident generated IDs without host readback or arguments."""
        self._check()
        if self.image.mimi is None:
            raise RuntimeError("Mimi RVQ was not configured")
        if self.image.decoder is not None and self._decoder_consumed:
            raise RuntimeError("decoder already consumed this frame; begin the next frame")
        if not self._ready or self.pos != self.spec.steps:
            raise RuntimeError("complete all depth steps before Mimi RVQ")
        self._mimi_ready = False
        try:
            self.mimi_stats = self.backend.run([self.programs[3]])
        except BaseException:
            self._failed = True; raise
        self._mimi_ready = True

    def run_decoder(self):
        """Consume resident RVQ latent; keep overlap, two-token KV and offsets resident."""
        self._check()
        image = self.image.decoder
        if image is None or not self._mimi_ready or self._decoder_consumed:
            raise RuntimeError("decoder needs an unconsumed completed resident RVQ frame")
        if self.decoder_pos >= image.spec.max_frames:
            raise RuntimeError("decoder frame limit reached; reset required")
        self._decoder_ready = False
        try:
            end = 4 + len(image.compile())
            self.decoder_stats = [self.backend.run([program]) for program in self.programs[4:end]]
        except BaseException:
            self._failed = True; raise
        self.decoder_pos += 1
        self._decoder_ready = self._decoder_consumed = True

    def run_pcm(self):
        """Generate raw float PCM from the resident decoder output, without host transfers."""
        self._check()
        image = self.image.pcm
        if image is None or not self._decoder_ready or self._pcm_consumed:
            raise RuntimeError("PCM needs an unconsumed completed resident decoder frame")
        self._pcm_ready = False
        try:
            start = 4 + len(self.image.decoder.compile())
            self.pcm_stats = [self.backend.run([program]) for program in self.programs[start:]]
        except BaseException:
            self._failed = True; raise
        self._pcm_ready = self._pcm_consumed = True

    def read_pcm_layer(self, layer):
        self._check()
        image = self.image.pcm
        if image is None or type(layer) is not int or not 0 <= layer < len(image.layer_outputs):
            raise ValueError("invalid SEANet stage")
        if not self._pcm_ready:
            raise RuntimeError("no completed resident PCM frame")
        at, rows, cols = image.layer_outputs[layer]
        result = np.asarray(self.backend.read(0, at, 4 * rows * cols), np.uint8).view(np.float32).copy()
        if result.shape != (rows * cols,) or not np.isfinite(result).all():
            self._failed = True
            raise RuntimeError("invalid resident SEANet output")
        return result.reshape(rows, cols)

    def read_pcm(self):
        return self.read_pcm_layer(14)[:, 0]

    def read_pcm_state(self):
        self._check()
        if self.image.pcm is None or not self._pcm_ready:
            raise RuntimeError("no completed resident PCM state")
        return [np.asarray(self.backend.read(0, at, size), np.uint8).copy()
                for at, size in self.image.pcm.reset_regions]

    def _read_decoder(self, address, width):
        self._check()
        if not self._decoder_ready:
            raise RuntimeError("no completed resident Mimi decoder frame")
        value = np.asarray(self.backend.read(0, address, width * 4), np.uint8).view(np.float32).copy()
        if value.shape != (width,) or not np.isfinite(value).all():
            self._failed = True
            raise RuntimeError("invalid resident decoder output")
        return value.reshape(2, width // 2)

    def read_decoder(self):
        image = self.image.decoder
        if image is None:
            raise RuntimeError("Mimi decoder was not configured")
        return self._read_decoder(image.io["output"], 2 * image.spec.hidden)

    def read_upsampled(self):
        image = self.image.decoder
        if image is None:
            raise RuntimeError("Mimi decoder was not configured")
        return self._read_decoder(image.io["upsampled"], 2 * image.spec.hidden)

    def read_decoder_layer(self, layer):
        image = self.image.decoder
        if image is None or type(layer) is not int or not 0 <= layer < image.spec.layers:
            raise ValueError("invalid Mimi decoder layer")
        return self._read_decoder(image.layer_outputs[layer], 2 * image.spec.hidden)

    def read_decoder_cache(self, layer):
        self._check()
        image = self.image.decoder
        if image is None or type(layer) is not int or not 0 <= layer < image.spec.layers:
            raise ValueError("invalid Mimi decoder cache layer")
        address, size = image.cache_regions[layer]
        return np.asarray(self.backend.read(0, address, size), np.uint8).copy()

    def read_mimi(self, name="output"):
        self._check()
        image = self.image.mimi
        if image is None or not self._mimi_ready:
            raise RuntimeError("no completed resident Mimi RVQ output")
        if name not in image.io:
            raise ValueError("unknown Mimi RVQ output")
        width = image.spec.dim if name in ("semantic", "acoustic") else image.spec.latent
        value = np.asarray(self.backend.read(0, image.io[name], 4 * width), np.uint8).view(np.float32).copy()
        if value.shape != (width,) or not np.isfinite(value).all():
            self._failed = True
            raise RuntimeError("invalid resident Mimi RVQ output")
        return value

    def _read(self, name, width, ready=True):
        self._check()
        if ready and not self._ready:
            raise RuntimeError("no completed resident depth step")
        value = np.asarray(self.backend.read(0, self.image.io[name], 4 * width), np.uint8).view(np.float32).copy()
        if value.shape != (width,) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid resident {name} output")
        return value

    def read_ids(self):
        self._check()
        if not self._ready or self.pos != self.spec.steps:
            raise RuntimeError("complete all depth steps before reading the ID sequence")
        values = self._read("audio_ids", self.spec.steps)
        if np.any(values != np.floor(values)) or np.any(values < 0) or np.any(values >= self.spec.head_vocab):
            raise RuntimeError("invalid device audio IDs")
        return values.astype(np.int64)

    def read_counter(self):
        value = float(self._read("counter", 1, False)[0])
        if value != int(value) or not 0 <= value <= self.spec.steps:
            raise RuntimeError("invalid device depth counter")
        return int(value)

    def read_previous_id(self):
        value = float(self._read("previous_id", 1, False)[0])
        if value != int(value):
            raise RuntimeError("invalid device feedback ID")
        return int(value)

    def read_projected(self):
        return self._read("projected", self.spec.depth)

    def read_conditioned(self):
        return self._read("conditioned", self.spec.depth)

    def read_output(self):
        return self._read("output", self.spec.depth)

    def read_logits(self):
        return self._read("logits", self.spec.head_vocab)

    def read_cache(self, layer=0):
        self._check()
        if type(layer) is not int or not 0 <= layer < self.spec.layers:
            raise ValueError("invalid resident depth cache layer")
        at, size = self.image.depth.cache_regions[layer]
        return np.asarray(self.backend.read(0, at, size), np.uint8).copy()

    def reset(self):
        if self._closed:
            raise RuntimeError("resident depth chain is closed")
        try:
            self.backend.write(0, self.image.io["control"], np.array([0, -1], np.float32))
            if self.image.decoder is not None:
                self.backend.write(0, self.image.decoder.io["state"], np.zeros(1, np.float32))
                self.backend.write(0, self.image.decoder.io["overlap"],
                                   np.full((2, self.image.decoder.spec.hidden), -0.0, np.float32))
            if self.image.pcm is not None:
                self.image.pcm.reset(self.backend)
        except BaseException:
            self._failed = True; raise
        self._failed = self._ready = self._active = self._mimi_ready = False
        self._decoder_ready = self._decoder_consumed = False
        self._pcm_ready = self._pcm_consumed = False
        self.decoder_pos = 0
        self.poss[0], self.last_stats = 0, {}

    def close(self):
        self._closed = True
        backend = getattr(self, "backend", None)
        if backend is not None:
            close = getattr(backend, "close", None)
            if close is not None:
                close()
            self.backend = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
