"""Bounded cold-duplex orchestration on one persistent software ISA machine.

Payload packing and upload happen before construction. During a conversation
the host supplies raw FP32 PCM, fixed forcing packets and immutable prepacked
temporal matrices, and reads only the final raw FP32 PCM. Every model operation,
token choice, delay gather and state update executes in an ISA program. This
runner is a software candidate: it supplies neither full-speech acceptance nor
a physical backend qualification.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import operator
from types import MappingProxyType

import numpy as np

from .. import target
from ..compiler import CompileError
from . import qwen3 as G
from .personaplex_duplex_handoff import DuplexHandoffImage
from .personaplex_duplex_layout import ARCH_HASH, CAPACITY, DuplexLayout
from .personaplex_duplex_schedule import DuplexScheduleImage, DuplexScheduleSpec


@dataclass(frozen=True)
class DuplexSubmission:
    """Optional supplied targets fixed before the first raw audio call."""

    text_token: int | None = None
    moshi_tokens: tuple | None = None

    def __post_init__(self):
        def checked(value, upper):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("supplied token must be integral")
            try:
                code = operator.index(value)
            except TypeError as error:
                raise ValueError("supplied token must be integral") from error
            if not -1 <= code < upper:
                raise ValueError("supplied token outside native range")
            return code

        if self.text_token is not None:
            object.__setattr__(self, "text_token", checked(self.text_token, 32001))
        if self.moshi_tokens is not None:
            values = tuple(self.moshi_tokens)
            if len(values) != 8:
                raise ValueError("supplied Moshi stream must have eight audio IDs")
            object.__setattr__(self, "moshi_tokens",
                               tuple(checked(value, 2049) for value in values))


@dataclass(frozen=True)
class DuplexPrograms:
    """Already compiled, argument-free programs and their resident Mimi view.

    ``output`` has the established depth-chain order: condition, six-layer
    depth, raw head, RVQ, decoder programs, then PCM programs. Its RVQ program
    must use ``mimi.audio_ids_at == emitted + 4``; raw depth-bank IDs have not
    yet passed through LMGen's output delay gather.
    """

    encoder: tuple
    input_transformer: tuple
    input_downsample: tuple
    input_rvq: tuple
    temporal: tuple
    schedule: object
    handoff: object
    output: tuple
    mimi: object

    def __post_init__(self):
        for name in ("encoder", "input_transformer", "input_downsample", "input_rvq",
                     "temporal", "output"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("schedule", "handoff"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @classmethod
    def from_layout(cls, layout, schedule, handoff):
        # A descriptor-only view reuses exactly the original output RVQ's
        # allocation and weights. Compile this view instead of duplicating its
        # tables/state or overwriting the depth chain's raw-ID program cache.
        mimi = copy.copy(layout.output.mimi)
        mimi.audio_ids_at = layout.io["emitted"] + 4
        mimi.programs = None
        output = copy.copy(layout.output)
        output.mimi, output.programs = mimi, None
        return cls(tuple(layout.encoder.compile()),
                   tuple(layout.input_transformer.compile()),
                   tuple(layout.input_downsample.compile()),
                   tuple(layout.input_rvq.compile()),
                   tuple(layout.temporal.compile()),
                   MappingProxyType(dict(schedule.compile())),
                   MappingProxyType(dict(handoff.compile())),
                   tuple(output.compile()), mimi)


class DuplexEngine:
    """One 128-call cold conversation; ``step(pcm)`` returns PCM or ``None``.

    The caller owns the uploaded ``IsaBackend`` exclusively for the engine's
    lifetime. Construction does not allocate 4 GiB, read a checkpoint or upload
    weights. The uploaded image must already include schedule.upload's fixed
    frame plan/ring initialization, handoff weights and temporal static tables,
    in addition to the input and output codec payloads. ``page_payloads``
    contains six immutable ``bytes`` objects, one
    complete prepacked temporal packet each, reused on every temporal frame.
    This intentionally keeps the offline packer outside the execution API.
    """

    full_speech_acceptance = False
    hardware_ready = False
    readiness = False

    def __init__(self, backend, layout, page_payloads, *, programs=None,
                 schedule=None, handoff=None, max_calls=128, submissions=(),
                 backend_kind="isa"):
        if backend_kind != "isa" or not isinstance(backend, G.IsaBackend):
            raise ValueError("duplex execution requires the software IsaBackend")
        if not isinstance(layout, DuplexLayout):
            raise TypeError("duplex execution requires the checked production layout")
        cfg = layout.cfg
        if (cfg.S != 1 or cfg.D != 128 or cfg.IMEM_WORDS != 32768 or
                cfg.DRAM_BYTES != CAPACITY or not cfg.SEQ_WC or
                f"{target.arch_hash(cfg):08x}" != ARCH_HASH or
                backend.machine.cfg != cfg):
            raise ValueError("duplex execution requires exact dd6b92a6 shared F2 ISA profile")
        if (type(max_calls) is not int or not 3 <= max_calls <= 128 or
                len(layout.program_slots) != 74 or len(set(layout.program_slots)) != 74 or
                len(layout.packets) != 6):
            raise ValueError("duplex execution requires six pages, 74 slots and 3..128 raw calls")
        self.backend, self.layout, self.cfg = backend, layout, cfg
        self.max_calls = max_calls
        self.schedule = schedule or DuplexScheduleImage(DuplexScheduleSpec(max_calls), cfg,
            io=layout.io, input_ids_at=layout.input_rvq.io["output_ids"],
            depth_control_at=layout.output.io["control"],
            depth_audio_ids_at=layout.output.io["audio_ids"])
        self.handoff = handoff or DuplexHandoffImage.from_layout(layout)
        self._check_seams()
        packets = tuple(page_payloads)
        if len(packets) != 6 or any(type(data) is not bytes or len(data) != packet.bytes
                                   for data, packet in zip(packets, layout.packets)):
            raise ValueError("provide six complete immutable prepacked byte packets")
        self.page_payloads = packets
        supplied = tuple(submissions)
        if len(supplied) > max_calls or any(not isinstance(s, DuplexSubmission) for s in supplied):
            raise ValueError("submissions must be predetermined DuplexSubmission values within the call bound")
        self.submissions = supplied + (DuplexSubmission(),) * (max_calls - len(supplied))
        self.programs = programs or DuplexPrograms.from_layout(layout, self.schedule, self.handoff)
        self._check_programs()
        self._closed = self._failed = self._active = False
        self._halted = self._drained = False
        self.raw_calls = self.temporal_frames = self.output_frames = 0
        self.last_stats = ()
        self.reset()

    def _check_seams(self):
        l, s, h = self.layout, self.schedule, self.handoff
        if (s.cfg != self.cfg or h.cfg != self.cfg or s.spec.max_calls != self.max_calls or
                s.io != {name: l.io[name] for name in s.REQUIRED_IO} or
                any(value != l.io[name] for name, value in h.io.items()) or
                (h.spec.hidden, h.spec.text_vocab, h.spec.audio_vocab,
                 h.spec.audio_streams, h.spec.head_vocab) != (4096, 32001, 2049, 16, 32000) or
                s.input_ids_at != l.input_rvq.io["output_ids"] or
                s.depth_control_at != l.output.io["control"] or
                s.depth_audio_ids_at != l.output.io["audio_ids"] or
                h.io["emitted"] != l.io["emitted"] or
                h.io["temporal_x"] != l.temporal.io["x"] or
                h.io["current_text_id"] != s.io["current_text_id"] or
                h.depth_hidden_at != l.output.io["hidden"] or
                h.depth_control_at != l.output.io["control"] or
                l.input_transformer.input_at != l.encoder.io["output"] or
                l.input_downsample.input_at != l.input_transformer.io["output"] or
                l.input_rvq.input_at != l.input_downsample.io["output"] or
                l.output.decoder.latent_at != l.output.mimi.io["output"] or
                l.output.pcm.input_at != l.output.decoder.io["output"] or
                self.max_calls > l.input_transformer.spec.max_frames or
                self.max_calls - 1 > l.temporal.frame_limit or
                self.max_calls - 2 > l.output.decoder.spec.max_frames):
            raise ValueError("duplex resident seam or frame bound does not match the layout")

    def _check_programs(self):
        p, l = self.programs, self.layout
        if not isinstance(p, DuplexPrograms):
            raise TypeError("duplex programs must be a precompiled DuplexPrograms bundle")
        expected = (len(l.encoder.program_slots), len(l.input_transformer.program_slots),
                    len(l.input_downsample.program_slots), len(l.input_rvq.program_slots), 6,
                    len(l.output.program_slots))
        groups = (p.encoder, p.input_transformer, p.input_downsample, p.input_rvq,
                  p.temporal, p.output)
        if (tuple(map(len, groups)) != expected or
                set(p.schedule) != {"prepare", "select_text", "feedback", "commit"} or
                set(p.handoff) != {"embed_codes", "voice_replay", "text_head", "depth_begin"}):
            raise CompileError("duplex program set does not cover the fixed 74 slots")
        mimi = p.mimi
        if (mimi.cfg != self.cfg or mimi.audio_ids_at != l.io["emitted"] + 4 or
                mimi.counter_at != l.output.io["counter"] or mimi.expected_steps != 16 or
                mimi.spec != l.output.mimi.spec or mimi.io != l.output.mimi.io or
                mimi.spans != l.output.mimi.spans or
                mimi.identity_at != l.output.mimi.identity_at or
                mimi.tables_at != l.output.mimi.tables_at or
                mimi.projection_offsets != l.output.mimi.projection_offsets):
            raise ValueError("output Mimi must gather emitted slots 1..8 using the existing allocation")
        all_programs = [program for group in groups for program in group]
        all_programs += list(p.schedule.values()) + list(p.handoff.values())
        if len(all_programs) != 74 or any(program.run_args or
                len(program) * 8 > self.cfg.IMEM_WORDS for program in all_programs):
            raise CompileError("duplex programs need host arguments or exceed the fixed instruction budget")

    def _check(self):
        if self._closed:
            raise RuntimeError("duplex engine is closed")
        if self._failed:
            raise RuntimeError("previous duplex call failed; reset the conversation")
        if self._active:
            raise RuntimeError("duplex call is already active")

    def _run(self, program, stats):
        self._halted = self._drained = False
        result = self.backend.run([program])
        # IsaBackend.run is synchronous, and its memory operations have no
        # asynchronous transport queue. Inspect the actual machine after HALT;
        # never infer completion merely from a backend return value.
        slices = self.backend.machine.slices
        self._halted = len(slices) == 1 and all(s.halted is True for s in slices)
        self._drained = self._halted and all(s.waiting is None and s.polling is None
                                          for s in slices)
        if not self._halted or not self._drained:
            raise RuntimeError("ISA predecessor has not halted and drained")
        stats.append(result)

    def _temporal(self, stats):
        for index, (packet, data, program) in enumerate(zip(
                self.layout.packets, self.page_payloads, self.programs.temporal)):
            self.layout.check_refill(index, packet.base, len(data),
                                    halted=self._halted, drained=self._drained)
            self.backend.write(0, packet.base, np.frombuffer(data, np.uint8))
            self._run(program, stats)

    def step(self, pcm):
        """Advance one raw 24 kHz frame; calls zero and one return no PCM.

        Call zero advances all input-codec history and primes LMGen at offset
        one without running the LM. Call one performs inference. Call two is
        the first to decode the undelayed agent stream into output PCM.
        """
        self._check()
        if self.raw_calls >= self.max_calls:
            raise RuntimeError("duplex raw-call limit reached; reset the conversation")
        value = np.asarray(pcm, dtype=np.float32)
        if value.shape != (self.layout.encoder.spec.samples,) or not np.isfinite(value).all():
            raise ValueError("input must be one finite raw FP32 PCM frame")
        value = np.ascontiguousarray(value)
        call, stats = self.raw_calls, []
        self._active = True
        try:
            self.backend.write(0, self.layout.encoder.input_at, value)
            submission = self.submissions[call]
            self.schedule.submit(self.backend, text_token=submission.text_token,
                                 moshi_tokens=submission.moshi_tokens)
            p = self.programs
            for group in (p.encoder, p.input_transformer, p.input_downsample, p.input_rvq):
                for program in group:
                    self._run(program, stats)
            self._run(p.schedule["prepare"], stats)
            result = None
            if call >= 1:
                self._run(p.handoff["embed_codes"], stats)
                self._temporal(stats)
                for program in (p.handoff["text_head"], p.handoff["depth_begin"],
                                p.schedule["select_text"]):
                    self._run(program, stats)
                for _ in range(16):
                    for program in (*p.output[:3], p.schedule["feedback"]):
                        self._run(program, stats)
                self._run(p.schedule["commit"], stats)
                if call >= 2:
                    for program in p.output[3:]:
                        self._run(program, stats)
                    at, rows, cols = self.layout.output.pcm.layer_outputs[14]
                    if rows != self.layout.output.pcm.spec.samples or cols != 1:
                        raise RuntimeError("final PCM descriptor is not the raw mono output")
                    result = np.asarray(self.backend.read(0, at, rows * 4),
                                        dtype=np.uint8).view(np.float32).copy()
                    if result.shape != (rows,) or not np.isfinite(result).all():
                        raise RuntimeError("invalid final resident PCM frame")
        except BaseException:
            self._failed = True
            raise
        finally:
            self._active = False
        self.raw_calls += 1
        self.temporal_frames += int(call >= 1)
        self.output_frames += int(call >= 2)
        self.last_stats = tuple(stats)
        return result

    def reset(self):
        """Start a cold conversation; never called between streaming frames."""
        if self._closed or self._active:
            raise RuntimeError("cannot reset a closed or active duplex engine")
        l, b = self.layout, self.backend
        try:
            for image in (l.encoder, l.input_transformer, l.input_downsample,
                          l.input_rvq, l.temporal, self.schedule):
                image.reset(b)
            # Depth uses logical bank position masking; its per-frame counter
            # reset is performed later by depth_begin, entirely on device.
            b.write(0, l.output.io["control"], np.array([0, -1], np.float32))
            b.write(0, l.output.decoder.io["state"], np.zeros(1, np.float32))
            b.write(0, l.output.decoder.io["overlap"],
                    np.full((2, l.output.decoder.spec.hidden), -0.0, np.float32))
            l.output.pcm.reset(b)
            b.write(0, l.io["voice_index"], np.zeros(1, np.float32))
            b.write(0, l.io["conversation_state"], np.zeros(64, np.uint8))
        except BaseException:
            self._failed = True
            raise
        self.raw_calls = self.temporal_frames = self.output_frames = 0
        self._failed = self._halted = self._drained = False
        self.last_stats = ()

    def budget(self):
        return dict(arch_hash=ARCH_HASH, mapped_bytes=CAPACITY, program_slots=74,
                    temporal_pages=6, max_raw_calls=self.max_calls,
                    max_temporal_frames=self.max_calls - 1,
                    max_emitted_frames=self.max_calls - 2,
                    raw_calls=self.raw_calls, temporal_frames=self.temporal_frames,
                    emitted_frames=self.output_frames,
                    full_speech_acceptance=False, hardware_ready=False, readiness=False,
                    backend="software ISA only", numerical_acceptance="separate gate")

    def close(self):
        self._closed = True  # Uploaded backend remains owned by the caller.

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()
