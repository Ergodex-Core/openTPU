"""Four relocated device programs joining duplex LM, temporal and depth state.

This is a bounded image component, not a conversation engine. The last temporal
page leaves its final normalized vector in ``temporal_x``. Run ``text_head``,
then ``depth_begin``, then the scheduler's ``select_text`` before depth bank zero.
No codec, KV cache or conversation state is initialized by this component.
"""
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

import numpy as np

from .. import language as ol
from .. import qcache as QC
from .. import quant as Q
from ..compiler import CompileError, QTensor, Tensor, trace_fit
from . import personaplex_temporal as T
from . import qwen3 as G
from .personaplex_embeddings import EmbeddingSpec, _embed_codes, _voice_frame
from .personaplex_token import _text_head


@dataclass(frozen=True)
class DuplexHandoffSpec:
    hidden: int = 4096
    text_vocab: int = 32001
    audio_vocab: int = 2049
    audio_streams: int = 16
    head_vocab: int = 32000
    voice_frames: int = 4

    def __post_init__(self):
        EmbeddingSpec(self.hidden, self.text_vocab, self.audio_vocab,
                      self.audio_streams, self.voice_frames)
        if type(self.head_vocab) is not int or not 0 < self.head_vocab <= self.text_vocab:
            raise ValueError("text head vocabulary must fit the text embedding table")
        if self.voice_frames == 0:
            raise ValueError("resident voice replay needs at least one voice frame")


@ol.jit
def _depth_begin(m):
    # temporal_x has already received the final page's RMSNorm exactly once.
    ol.store(m.depth_hidden, ol.load(m.x))
    ol.store(m.depth_control[0:1], ol.zeros((1,)))
    ol.store(m.depth_control[1:2], ol.load(m.token_id))


class DuplexHandoffImage:
    """Relocatable LM embeddings, greedy text head and depth-start descriptors.

    Construction validates addresses without reading payloads. ``upload`` is an
    offline operation restricted to immutable spans; callers own all program
    submission and must keep the backend halted while uploading weights.
    """

    PROGRAM_NAMES = ("embed_codes", "voice_replay", "text_head", "depth_begin")
    cap = batch = 1
    kv_bytes = 0
    head_format = "int8"

    def __init__(self, spec, cfg, *, io, depth_hidden_at, depth_control_at):
        if not isinstance(spec, DuplexHandoffSpec):
            raise TypeError("duplex handoff requires a validated specification")
        if cfg.S != 1 or spec.hidden % cfg.D:
            raise ValueError("duplex handoff needs one slice and a D-aligned hidden width")
        self.spec, self.cfg = spec, cfg
        names = ("lm_text_embedding", "lm_audio_embeddings", "emitted", "temporal_x",
                 "current_text_id", "text_head_codes", "text_head_scales", "text_logits",
                 "voice_frames", "voice_index")
        addresses = {name: io[name] for name in names}
        addresses["lm_audio_embeddings"] = tuple(addresses["lm_audio_embeddings"])
        if len(addresses["lm_audio_embeddings"]) != spec.audio_streams:
            raise ValueError("expected every resident audio embedding address")
        self.io = MappingProxyType(addresses)
        self.depth_hidden_at, self.depth_control_at = depth_hidden_at, depth_control_at
        self.head_chunk = min(G.HEAD_CHUNK, cfg.TMEM_WORDS // 8)
        if self.head_chunk < 1:
            raise ValueError("TMEM is too small for the text head")
        row_bytes = 4 * spec.hidden
        self.table_bytes = row_bytes * (spec.text_vocab + 1 +
                                       spec.audio_streams * (spec.audio_vocab + 1))
        self.voice_bytes = row_bytes * spec.voice_frames
        self.weight_code_bytes = spec.head_vocab * Q.row_bytes(spec.hidden, "int8", cfg.D)
        self.head_scale_bytes = spec.head_vocab * Q.scale_bytes(spec.hidden, "int8", cfg.D)
        self.head_bytes = self.weight_code_bytes + self.head_scale_bytes
        self.weight_bytes = self.table_bytes + self.voice_bytes + self.head_bytes
        spans = [("lm_text_embedding", io["lm_text_embedding"], row_bytes * (spec.text_vocab + 1)),
                 *((f"lm_audio_embedding{i}", at, row_bytes * (spec.audio_vocab + 1))
                   for i, at in enumerate(self.io["lm_audio_embeddings"])),
                 ("voice_frames", io["voice_frames"], self.voice_bytes),
                 ("text_head_codes", io["text_head_codes"], self.weight_code_bytes),
                 ("text_head_scales", io["text_head_scales"], self.head_scale_bytes),
                 ("emitted", io["emitted"], 4 * (spec.audio_streams + 1)),
                 ("temporal_x", io["temporal_x"], row_bytes),
                 ("current_text_id", io["current_text_id"], 4),
                 ("text_logits", io["text_logits"], 4 * spec.head_vocab),
                 ("voice_index", io["voice_index"], 4),
                 ("depth_hidden", depth_hidden_at, row_bytes),
                 ("depth_control", depth_control_at, 8)]
        capacity = min(cfg.DRAM_BYTES, 1 << 32)
        for name, at, size in spans:
            if type(at) is not int or at % 4 or not 0 <= at < at + size <= capacity:
                raise MemoryError("invalid duplex handoff address: " + name)
        if io["text_head_codes"] % cfg.D:
            raise MemoryError("text head codes require D-byte alignment")
        ordered = sorted(spans, key=lambda span: span[1])
        for left, right in zip(ordered, ordered[1:]):
            if left[1] + left[2] > right[1]:
                raise MemoryError("overlapping duplex handoff buffers: " + left[0] + "/" + right[0])
        immutable = {"lm_text_embedding", "voice_frames", "text_head_codes", "text_head_scales"}
        self.spans = tuple(dict(name=name, base=at, bytes=size,
                               kind="immutable" if name in immutable or
                               name.startswith("lm_audio_embedding") else "state")
                           for name, at, size in ordered)
        self.required_bytes = max(at + size for _, at, size in spans)
        self.programs = None

    @classmethod
    def from_layout(cls, layout):
        """Bind the exact checked shared F2 ownership map; allocate nothing."""
        from .. import target
        from .personaplex_duplex_layout import ARCH_HASH, CAPACITY

        if (layout.cfg.S != 1 or layout.cfg.DRAM_BYTES != CAPACITY or
                f"{target.arch_hash(layout.cfg):08x}" != ARCH_HASH or
                len(layout.program_slots) != 74 or len(layout.temporal.pages) != 6):
            raise ValueError("duplex handoff requires the exact six-page shared F2 layout")
        e = layout.embedding_spec
        spec = DuplexHandoffSpec(hidden=e.hidden, text_vocab=e.text_vocab,
            audio_vocab=e.audio_vocab, audio_streams=e.audio_streams, voice_frames=e.voice_frames)
        output = layout.output
        if output.spec.hidden != spec.hidden or output.spec.steps != spec.audio_streams:
            raise ValueError("depth and LM handoff dimensions differ")
        image = cls(spec, layout.cfg, io=layout.io, depth_hidden_at=output.io["hidden"],
                    depth_control_at=output.io["control"])
        owners = {span["name"]: span for span in layout.spans}
        for span in image.spans:
            name = {"depth_hidden": "hidden", "depth_control": "control"}.get(span["name"], span["name"])
            owner = owners[name]
            if (owner["base"], owner["bytes"], owner["kind"]) != (
                    span["base"], span["bytes"], span["kind"]):
                raise MemoryError("duplex handoff differs from resident owner: " + name)
        image.program_slots = {name: layout.io[name + "_program"] for name in cls.PROGRAM_NAMES}
        return image

    def descriptors(self):
        s, io = self.spec, self.io
        table = lambda at, rows: Tensor(at, (rows, s.hidden), (s.hidden, 1))
        return SimpleNamespace(
            text=table(io["lm_text_embedding"], s.text_vocab + 1),
            audio=tuple(table(at, s.audio_vocab + 1) for at in io["lm_audio_embeddings"]),
            codes=Tensor(io["emitted"], (s.audio_streams + 1,), (1,)),
            output=table(io["temporal_x"], 1),
            voice=table(io["voice_frames"], s.voice_frames),
            voice_index=Tensor(io["voice_index"], (1,), (1,)),
            x=table(io["temporal_x"], 1),
            token_head=QTensor.weight(io["text_head_codes"], io["text_head_scales"],
                                      s.head_vocab, s.hidden, "int8", self.cfg.D),
            token_logits=Tensor(io["text_logits"], (1, s.head_vocab), (s.head_vocab, 1)),
            token_id=Tensor(io["current_text_id"], (1,), (1,)),
            token_vocab=s.head_vocab, head_chunk=self.head_chunk,
            depth_hidden=table(self.depth_hidden_at, 1),
            depth_control=Tensor(self.depth_control_at, (2,), (1,)))

    def compile(self):
        if self.programs is None:
            programs = {}
            for name, kernel in zip(self.PROGRAM_NAMES,
                                     (_embed_codes, _voice_frame, _text_head, _depth_begin)):
                program = trace_fit(kernel, self.cfg, 0, lambda: {"m": self.descriptors()}).finish()
                if program.run_args or len(program) * 8 > self.cfg.IMEM_WORDS:
                    raise CompileError("duplex " + name + " needs host args or exceeds IMEM")
                programs[name] = program
            self.programs = programs
        return self.programs

    def upload(self, backend, tables, text_head, voices=None):
        """Validate/pack offline, then write immutable payloads only.

        Embedding tables use native unprefixed rows; one positive-zero row is
        uploaded before each. Omitting voices initializes the reserved replay
        frames to positive zero. Runtime IDs and all mutable state are untouched.
        """
        self.compile()  # Program capacity is checked before lazy payload reads.
        s = self.spec
        tables = tables() if callable(tables) else tables
        if len(tables) != s.audio_streams + 1:
            raise ValueError("expected text table followed by every audio embedding table")
        sizes = (s.text_vocab,) + (s.audio_vocab,) * s.audio_streams
        checked = [T._checked(table, (rows, s.hidden), "embedding table")
                   for table, rows in zip(tables, sizes)]
        source = text_head() if callable(text_head) else text_head
        head = T._checked(source, (s.head_vocab, s.hidden), "text_head")
        voices = voices() if callable(voices) else voices
        voice = (np.zeros((s.voice_frames, s.hidden), np.float32) if voices is None else
                 T._checked(voices, (s.voice_frames, s.hidden), "voice embeddings"))
        packed = []
        qb = QC.Batch()
        qb.submit(head, "int8", self.cfg.D, lambda q, scale: packed.append((q, scale)))
        qb.wait()
        codes, scales = packed[0]
        if codes.nbytes != self.weight_code_bytes or scales.nbytes != self.head_scale_bytes:
            raise ValueError("packed text head differs from its resident footprint")
        zero = np.zeros(s.hidden, np.float32)
        for at, table in zip((self.io["lm_text_embedding"], *self.io["lm_audio_embeddings"]), checked):
            backend.write(0, at, zero)
            backend.write(0, at + zero.nbytes, table)
        backend.write(0, self.io["voice_frames"], voice)
        backend.write(0, self.io["text_head_codes"], codes)
        backend.write(0, self.io["text_head_scales"], scales)
