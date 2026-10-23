"""Checked, software-only ownership map for the shared PersonaPlex image.

This module allocates addresses and checks packet leases. It neither loads a
checkpoint nor executes a full-duplex conversation. A temporal page refill is
admissible only after the preceding program has halted and transport drained.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import quant as Q
from .. import target, waits
from .personaplex_depth_chain import DepthChainImage, DepthChainSpec
from .personaplex_embeddings import EmbeddingImage, EmbeddingSpec
from .personaplex_mimi import MimiRVQSpec
from .personaplex_mimi_decoder import MimiDecoderSpec
from .personaplex_mimi_encoder import MimiEncoderImage
from .personaplex_mimi_input_downsample import MimiInputDownsampleImage
from .personaplex_mimi_input_rvq import MimiInputRVQImage
from .personaplex_mimi_input_transformer import MimiInputTransformerImage
from .personaplex_mimi_pcm import MimiPCMSpec
from .personaplex_temporal import Spec as TemporalSpec
from .personaplex_temporal_pages import TemporalPagedImage


TARGET = "f2-hbm:p4rl8a64c150"
ARCH_HASH = "dd6b92a6"
CAPACITY = 1 << 32
TEMPORAL_KV_AT = 0x00010000
TEMPORAL_KV_END = 0x02110000
PAGE_LIMIT = 0xB0000000
PROGRAM_BASE = 0x04000000
PROGRAM_LIMIT = 0x05000000
PACKET_PLAN = ((0, 4, "int8"), (4, 12, "int4"), (12, 20, "int4"),
               (20, 28, "int4"), (28, 31, "int4"), (31, 32, "int8"))
TEMPORAL_ALIASES = ("temporal_x", "temporal_frame", "temporal_page",
                    "temporal_cos", "temporal_sin", "temporal_zmask",
                    "temporal_final_norm")
DUPLEX_PROGRAMS = ("schedule_prepare_program", "schedule_select_text_program",
                   "schedule_feedback_program", "schedule_commit_program",
                   "embed_codes_program", "text_head_program",
                   "depth_begin_program", "voice_replay_program")


def _align(value, alignment=128):
    return -(-value // alignment) * alignment


def check_spans(spans, capacity=CAPACITY):
    """Reject overlapping, unaligned or out-of-range physical reservations."""
    if type(capacity) is not int or not 0 < capacity <= CAPACITY:
        raise ValueError("invalid mapped capacity")
    names = set()
    ordered = []
    for span in spans:
        name, base, size = span["name"], span["base"], span["bytes"]
        alignment = span.get("alignment", 1)
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("duplicate or invalid resident span name")
        names.add(name)
        if (type(base) is not int or type(size) is not int or type(alignment) is not int or
                alignment <= 0 or base % alignment or size <= 0 or
                not 0 <= base < base + size <= capacity):
            raise MemoryError("invalid resident span: " + name)
        ordered.append(span)
    ordered.sort(key=lambda span: span["base"])
    for left, right in zip(ordered, ordered[1:]):
        if left["base"] + left["bytes"] > right["base"]:
            raise MemoryError("resident overlap: " + left["name"] + "/" + right["name"])
    return ordered


def check_temporal_aliases(resident_spans, temporal_spans):
    """Require temporal state and immutable aliases to match their owners exactly."""
    aliases = set(TEMPORAL_ALIASES)
    resident = [span for span in resident_spans if span["name"] in aliases]
    temporal = [span for span in temporal_spans if span["name"] in aliases]
    for spans in (resident, temporal):
        if (len(spans) != len(aliases) or {span["name"] for span in spans} != aliases or
                any(span["kind"] not in ("state", "immutable") for span in spans)):
            raise MemoryError("temporal aliases must have one state/immutable owner each")
    footprint = lambda spans: {span["name"]: (span["base"], span["bytes"])
                               for span in spans}
    if footprint(resident) != footprint(temporal):
        raise MemoryError("temporal alias base or byte count differs from its resident owner")


@dataclass(frozen=True)
class Packet:
    index: int
    start_layer: int
    end_layer: int
    format: str
    base: int
    bytes: int


class DuplexLayout:
    """Physical reservations and exclusive temporal page leases.

    Existing output and input codec spans are copied verbatim. The six
    temporal packets share one A-region lease, so only the largest packet is
    counted as resident. All other additions are disjoint P/state/program
    spans. This class does not claim a qualified full conversation scheduler.
    """

    @classmethod
    def production(cls):
        cfg = target.config(TARGET)
        if f"{target.arch_hash(cfg):08x}" != ARCH_HASH or cfg.DRAM_BYTES != CAPACITY:
            raise ValueError("unexpected shared F2 target/profile")
        output = DepthChainImage(DepthChainSpec(), cfg,
            mimi_spec=MimiRVQSpec(projection_precision="fp32"),
            decoder_spec=MimiDecoderSpec(precision="fp32"),
            pcm_spec=MimiPCMSpec(initial_precision="fp32", transpose_precision="fp32",
                                 residual_precision="fp32", final_precision="fp32"))
        encoder = MimiEncoderImage.after_output_codec(output)
        transformer = MimiInputTransformerImage.after_encoder(encoder)
        downsample = MimiInputDownsampleImage.after_transformer(transformer)
        rvq = MimiInputRVQImage.after_downsample(downsample)
        layout = cls(rvq)
        layout.output, layout.encoder = output, encoder
        layout.input_transformer, layout.input_downsample = transformer, downsample
        return layout

    def __init__(self, rvq):
        cfg = rvq.cfg
        if (cfg.S != 1 or cfg.DRAM_BYTES != CAPACITY or
                f"{target.arch_hash(cfg):08x}" != ARCH_HASH):
            raise ValueError("duplex layout requires the exact shared F2 profile")
        if rvq.weight_end != 1_960_284_160:
            raise MemoryError("input codec weight boundary has moved")
        self.cfg, self.input_rvq = cfg, rvq
        self.codec_spans = [dict(span) for span in rvq.prefix_spans + rvq.spans]
        check_spans(self.codec_spans)
        self.spans = list(self.codec_spans)
        self.io = {}

        def reserve(name, base, size, region, kind="state", alignment=128):
            span = dict(name=name, base=base, bytes=size, region=region,
                        kind=kind, alignment=alignment)
            check_spans(self.spans + [span])
            self.spans.append(span)
            return base

        def free(name, size, lo, hi, region, kind="state", alignment=128):
            cursor = _align(lo, alignment)
            for span in sorted(self.spans, key=lambda item: item["base"]):
                if span["base"] >= hi:
                    break
                if span["base"] + span["bytes"] <= cursor:
                    continue
                if cursor + size <= span["base"]:
                    break
                cursor = _align(span["base"] + span["bytes"], alignment)
            if cursor + size > hi:
                raise MemoryError(name + " does not fit reserved " + region)
            return reserve(name, cursor, size, region, kind, alignment)

        # The existing depth control/audio IDs retain their addresses. A
        # four-position ring matches the native audio delay. Presence bits
        # distinguish a supplied -1 mask from an absent stream. The frame
        # plan is immutable modulo/offset metadata, not host-computed tokens.
        for name, size in (("ring_tokens", 4 * 17 * 4),
                           ("ring_provided", 4 * 17 * 4),
                           ("incoming_tokens", 17 * 4),
                           ("incoming_present", 17 * 4),
                           ("target_tokens", 17 * 4),
                           ("target_provided", 17 * 4),
                           ("emitted", 17 * 4), ("frame_plan", 130 * 6 * 4),
                           ("offset", 4), ("position", 6 * 4), ("ready", 4),
                           ("current_text_id", 4), ("next_text_id", 4),
                           ("voice_index", 4), ("conversation_state", 64),
                           ("temporal_frame", 4), ("temporal_page", 4)):
            self.io[name] = free(name, size, 0, 0x10000, "control")
        self.io["input_queue"] = self.io["incoming_tokens"]
        self.io["output_queue"] = self.io["emitted"]
        self.io["text_token"] = self.io["current_text_id"]

        # 4096-wide FP32 handoff vectors are separate from all depth and codec
        # scratch. The temporal page kernel keeps its residual in temporal_x.
        for name in ("embedding_sum", "temporal_x", "temporal_normalized"):
            self.io[name] = free(name, 4 * 4096, 0x02510000, 0x03510000, "scratch")
        self.io["embedding"] = self.io["embedding_sum"]
        self.io["residual"] = self.io["temporal_x"]

        # The LM embedding tables are distinct from the already resident depth
        # step embeddings. EmbeddingImage adds exactly one zero row per table:
        # source checkpoint rows 32001/2049 are passed unprefixed to build().
        self.embedding_spec = EmbeddingSpec(voice_frames=4)
        self.embedding = EmbeddingImage(self.embedding_spec, cfg)
        table_base = 0x05000000
        row_bytes = 4 * self.embedding_spec.hidden
        self.io["lm_text_embedding"] = reserve("lm_text_embedding", table_base +
            self.embedding.text_at, (self.embedding_spec.text_vocab + 1) * row_bytes,
            "lm_embeddings", "immutable")
        self.io["lm_audio_embeddings"] = tuple(reserve(f"lm_audio_embedding{index}",
            table_base + address, (self.embedding_spec.audio_vocab + 1) * row_bytes,
            "lm_embeddings", "immutable") for index, address in enumerate(self.embedding.audio_at))
        self.io["voice_frames"] = reserve("voice_frames", table_base + self.embedding.voice_at,
            self.embedding.voice_bytes, "lm_embeddings", "immutable")
        if (table_base + self.embedding.voice_at + self.embedding.voice_bytes != 0x44498000 or
                self.embedding.table_bytes != 1_061_715_968):
            raise MemoryError("LM tables no longer end at the existing step embeddings")

        # The INT8 code and scale rows fill the exact documented gap before
        # the audio heads. Diagnostic FP32 logits instead use free scratch.
        text_vocab, hidden = 32000, 4096
        head_code_bytes = text_vocab * Q.row_bytes(hidden, "int8", cfg.D)
        head_scale_bytes = text_vocab * Q.scale_bytes(hidden, "int8", cfg.D)
        head_at = 0x539B8000
        self.io["text_head_codes"] = reserve("text_head_codes", head_at,
            head_code_bytes, "text_head", "immutable")
        self.io["text_head_scales"] = reserve("text_head_scales", head_at + head_code_bytes,
            head_scale_bytes, "text_head", "immutable")
        if self.io["text_head_scales"] + head_scale_bytes != 0x5B7D1400:
            raise MemoryError("text head no longer abuts the existing audio heads")
        self.io["text_logits"] = free("text_logits", 4 * text_vocab,
                                      0x02510000, 0x03510000, "scratch")

        self.io["temporal_cos"] = reserve("temporal_cos", 0x5D831400, 32_768,
            "temporal_static", "immutable")
        self.io["temporal_sin"] = reserve("temporal_sin", 0x5D839400, 32_768,
            "temporal_static", "immutable")
        self.io["temporal_final_norm"] = reserve("temporal_final_norm", 0x5D841400,
            16_384, "temporal_static", "immutable")
        self.io["temporal_zmask"] = reserve("temporal_zmask", 0x5D845400,
            1_024, "temporal_static", "immutable")

        # TemporalPagedImage validates KV and page sizes against its actual
        # descriptors, and appends six programs after the 60 codec slots.
        self.temporal = TemporalPagedImage(TemporalSpec(), cfg, cap=128,
            pages=PACKET_PLAN, kv_at=TEMPORAL_KV_AT,
            x_at=self.io["temporal_x"], frame_at=self.io["temporal_frame"],
            page_at=self.io["temporal_page"], cos_at=self.io["temporal_cos"],
            sin_at=self.io["temporal_sin"], zmask_at=self.io["temporal_zmask"],
            final_norm_at=self.io["temporal_final_norm"],
            weight_page_base=rvq.weight_end, weight_page_limit=PAGE_LIMIT,
            prefix_spans=self.codec_spans)
        if (self.temporal.kv_bytes != TEMPORAL_KV_END - TEMPORAL_KV_AT or
                self.temporal.page_bytes != 884_473_856):
            raise MemoryError("temporal KV or packet footprint changed")
        # The page object contains aliases for addresses already reserved
        # above. Compare their complete footprints before adding only its
        # unique physical KV, overlay and program spans.
        check_temporal_aliases(self.spans, self.temporal.spans)
        temporal_owned = {"temporal_kv", "temporal_weight_page"}
        temporal_owned.update(f"temporal_program{i}" for i in range(6))
        for span in self.temporal.spans:
            if span["name"] in temporal_owned:
                reserve(span["name"], span["base"], span["bytes"], span["region"],
                        span["kind"], span.get("alignment", 128))
            elif span["name"] not in TEMPORAL_ALIASES:
                raise MemoryError("unexpected temporal span: " + span["name"])

        slot_bytes = waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
        if slot_bytes != 163_840:
            raise MemoryError("shared F2 instruction slot size changed")
        if len([s for s in self.codec_spans if s["kind"] == "program"]) != 60:
            raise MemoryError("the established input/output path must own 60 slots")
        # Reserve each fixed scheduler/handoff program separately. Voice
        # replay remains a conservative slot even before prompt integration.
        for name in DUPLEX_PROGRAMS:
            self.io[name] = free(name, slot_bytes, PROGRAM_BASE, PROGRAM_LIMIT,
                                 "program", "program", 4096)
        programs = [span for span in self.spans if span["kind"] == "program"]
        self.program_slots = [span["base"] for span in programs]
        if (len(self.program_slots) != 74 or len(set(self.program_slots)) != 74 or
                any(span["bytes"] != slot_bytes or
                    not PROGRAM_BASE <= span["base"] < span["base"] + slot_bytes <= PROGRAM_LIMIT
                    for span in programs)):
            raise MemoryError("full duplex requires 74 exclusive shared F2 program slots")
        self.program_slot_bytes = slot_bytes
        self.program_reserved_bytes = sum(span["bytes"] for span in programs)
        self.packets = tuple(Packet(i, page.start, page.end, page.format,
                                    rvq.weight_end, page.weight_bytes)
                             for i, page in enumerate(self.temporal.pages))
        if tuple(packet.bytes for packet in self.packets) != (
                831_782_912, 884_473_856, 884_473_856, 884_473_856, 331_677_696, 207_945_728):
            raise MemoryError("temporal packet packing changed")
        self.spans = check_spans(self.spans)

    def check_refill(self, page_index, address, nbytes, *, halted, drained):
        """Preflight one complete immutable packet write into the exclusive A lease."""
        if type(page_index) is not int or not 0 <= page_index < len(self.packets):
            raise ValueError("invalid temporal packet")
        packet = self.packets[page_index]
        if halted is not True or drained is not True:
            raise RuntimeError("temporal page refill requires a halted and drained predecessor")
        if (type(address) is not int or type(nbytes) is not int or
                address != packet.base or nbytes != packet.bytes or
                address % 4096 or address + nbytes > PAGE_LIMIT):
            raise MemoryError("packet write escapes or only partially covers immutable page lease")
        return packet

    def check_page_block(self, page_index, layer_index, address, nbytes, *, halted, drained):
        """Preflight one of TemporalPagedImage.upload_page's contiguous layer writes."""
        if type(page_index) is not int or not 0 <= page_index < len(self.packets):
            raise ValueError("invalid temporal packet")
        packet = self.packets[page_index]
        count = packet.end_layer - packet.start_layer
        if type(layer_index) is not int or not 0 <= layer_index < count:
            raise ValueError("invalid packet-local layer index")
        if halted is not True or drained is not True:
            raise RuntimeError("temporal page refill requires a halted and drained predecessor")
        if packet.bytes % count:
            raise MemoryError("packet is not an integral set of layer blocks")
        stride = packet.bytes // count
        if (type(address) is not int or type(nbytes) is not int or
                address != packet.base + layer_index * stride or nbytes != stride or
                address % 128 or address + nbytes > packet.base + packet.bytes or
                address + nbytes > PAGE_LIMIT):
            raise MemoryError("layer write escapes immutable page lease")
        return packet

    def budget(self):
        return dict(target=TARGET, arch_hash=ARCH_HASH, mapped_bytes=CAPACITY,
            codec_spans=len(self.codec_spans), total_spans=len(self.spans),
            occupied_bytes=sum(span["bytes"] for span in self.spans),
            program_slots=tuple(sorted(self.program_slots)),
            program_slot_count=len(self.program_slots),
            program_slot_bytes=self.program_slot_bytes,
            program_reserved_bytes=self.program_reserved_bytes,
            program_remaining_bytes=PROGRAM_LIMIT - PROGRAM_BASE - self.program_reserved_bytes,
            program_slot_counts=dict(codec=60, temporal=6, scheduler=4,
                embedding=1, text_head=1, depth_begin=1, voice_replay=1),
            temporal_kv_bytes=self.temporal.kv_bytes,
            packet_bytes=tuple(packet.bytes for packet in self.packets),
            packet_window=(self.input_rvq.weight_end, PAGE_LIMIT),
            packet_slack_bytes=PAGE_LIMIT - self.input_rvq.weight_end -
                self.temporal.page_bytes,
            zero_prefix_rows_per_lm_table=1,
            scope="Checked software ownership and packet leases; no integrated scheduler or physical acceptance")
