"""Resident LMGen delay ring and supplied-token routing (CPU/ISA only).

The input Mimi RVQ supplies the user eight audio IDs in device memory.  The
host may upload a fixed text/Moshi forcing packet, but never reads a generated
ID to decide the next program or token.  Four device programs prepare the
native ring, choose text for depth, override depth feedback when supplied, and
commit/undelay the completed frame.  The 17-slot ``emitted`` buffer is reused
as the current LM embedding input until commit replaces it with output codes.

The cold stream has no prompt prefix: call zero only primes LMGen, call one
performs inference without output, and call two first emits.  Reset follows
LMGenState.reset: it clears the offset and supplied flags, retaining ring
token bytes.  Prompt sequencing and saved voice embeddings are separate work.
"""
from dataclasses import dataclass
import operator
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from ..compiler import Affine, CompileError, DevVar, Tensor, trace_fit
from .personaplex_embeddings import _row


DELAYS = (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)
INITIAL_TEXT = 32000
INITIAL_AUDIO = 2048


@dataclass(frozen=True)
class DuplexScheduleSpec:
    max_calls: int = 128  # input codec's 128-frame bound; one call is LMGen priming
    text_vocab: int = 32001
    audio_vocab: int = 2049

    def __post_init__(self):
        if (type(self.max_calls) is not int or not 3 <= self.max_calls <= 128
                or type(self.text_vocab) is not int or self.text_vocab != INITIAL_TEXT + 1
                or type(self.audio_vocab) is not int or self.audio_vocab != INITIAL_AUDIO + 1):
            raise ValueError("duplex schedule needs bounded 17-stream native token ranges")

    @property
    def emitted_frames(self):
        return self.max_calls - 2


def _slot(base, register, index):
    return Tensor(Affine(base + 4 * index) + DevVar("duplex ring column", register),
                  (1,), (1,))


def _ring_register(b, position, index):
    register = b.scratch()
    b.emit(I.rld(register, position.base + index, mul=4 * 17,
                 bound=(0, 3 * 4 * 17), comment="native four-column delay ring"))
    return register


def _if_store(flag, destination, value):
    b = ol.current()
    register = b.scratch()
    b.emit(I.rld(register, flag.base, bound=(0, 1), comment="resident supplied-token flag"))
    branch = b.begin_loop(0, rcount=register)
    ol.store(destination, value)
    b.end_loop(branch)
    b.unscratch(register)


@ol.jit
def _prepare(m):
    b = ol.current()
    offset = ol.load(m.offset)
    plan = _row(m.frame_plan, offset)[0, :]
    ol.store(m.position, plan)
    target = _ring_register(b, plan, 1)
    future = _ring_register(b, plan, 2)
    model_input = _ring_register(b, plan, 0)

    # Fixed host forcing packets cover text and agent slots only.  User slots
    # 9..16 come directly from the resident input RVQ's eight output IDs.
    for index in range(9):
        flag = ol.load(m.incoming_present[index:index + 1])
        chosen = target if DELAYS[index] == 0 else future
        _if_store(flag, _slot(m.ring_tokens.base, chosen, index),
                  ol.load(m.incoming_tokens[index:index + 1]))
        _if_store(flag, _slot(m.ring_provided.base, chosen, index), ol.full((1,), 1.0))
    for index in range(9, 17):
        chosen = target if DELAYS[index] == 0 else future
        ol.store(_slot(m.ring_tokens.base, chosen, index),
                 ol.load(m.input_ids[index - 9:index - 8]))
        ol.store(_slot(m.ring_provided.base, chosen, index), ol.full((1,), 1.0))

    # Source startup overwrites even supplied values at offset zero.  At one,
    # only delay-one slots get the initial token in the current target column.
    prime = ol.maximum(1.0 - offset, 0.0)
    startup_delay_one = plan[5:6]
    for index in range(17):
        initial = INITIAL_TEXT if index == 0 else INITIAL_AUDIO
        _if_store(prime, _slot(m.ring_tokens.base, target, index),
                  ol.full((1,), initial))
        _if_store(prime, _slot(m.ring_provided.base, target, index), ol.full((1,), 1.0))
        if DELAYS[index]:
            _if_store(startup_delay_one, _slot(m.ring_tokens.base, target, index),
                      ol.full((1,), initial))
            _if_store(startup_delay_one, _slot(m.ring_provided.base, target, index),
                      ol.full((1,), 1.0))

    # Snapshot target/presence once.  The depth feedback program indexes these
    # contiguous arrays with its resident bank counter, never a host argument.
    for index in range(17):
        ol.store(m.emitted[index:index + 1],
                 ol.load(_slot(m.ring_tokens.base, model_input, index)))
        ol.store(m.target_tokens[index:index + 1],
                 ol.load(_slot(m.ring_tokens.base, target, index)))
        ol.store(m.target_provided[index:index + 1],
                 ol.load(_slot(m.ring_provided.base, target, index)))
    ol.store(m.ready, ol.zeros((1,)))
    ol.store(m.offset, offset + prime)
    b.unscratch(model_input)
    b.unscratch(future)
    b.unscratch(target)


@ol.jit
def _select_text(m):
    # Text head writes the raw sample to current_text_id.  A supplied target
    # controls depth bank zero, while the raw sample is retained for commit.
    selected = ol.load(m.current_text_id)
    provided = ol.load(m.target_provided[0:1])
    b = ol.current()
    register = b.scratch()
    b.emit(I.rld(register, provided.base, bound=(0, 1), comment="provided text target"))
    branch = b.begin_loop(0, rcount=register)
    selected.set(ol.load(m.target_tokens[0:1]))
    b.end_loop(branch)
    b.unscratch(register)
    ol.store(m.next_text_id, selected)
    ol.store(m.depth_control[0:1], ol.zeros((1,)))
    ol.store(m.depth_control[1:2], selected)


@ol.jit
def _feedback(m):
    # Run after each raw audio-head program.  That head has already stored its
    # sampled ID to audio_ids[s] and previous_id.  Only the latter is changed.
    bank = ol.load(m.depth_control[0:1])
    provided = _row(m.target_provided_rows, bank)
    target = _row(m.target_token_rows, bank)
    selected = ol.load(m.depth_control[1:2])
    b = ol.current()
    register = b.scratch()
    b.emit(I.rld(register, provided.base, bound=(0, 1), comment="provided audio target"))
    branch = b.begin_loop(0, rcount=register)
    selected.set(target[0, :])
    b.end_loop(branch)
    b.unscratch(register)
    ol.store(m.depth_control[1:2], selected)


@ol.jit
def _commit(m):
    b = ol.current()
    position = ol.load(m.position)
    model_input = _ring_register(b, position, 0)
    target = _ring_register(b, position, 1)
    output_zero = _ring_register(b, position, 3)
    output_one = _ring_register(b, position, 4)
    # Native clears supplied flags at model_input_position before committing
    # sampled values into target_position where no supplied target exists.
    for index in range(17):
        ol.store(_slot(m.ring_provided.base, model_input, index), ol.zeros((1,)))
    for index in range(17):
        flag = ol.load(m.target_provided[index:index + 1])
        missing = 1.0 - flag
        sample = (ol.load(m.current_text_id) if index == 0 else
                  ol.load(m.audio_ids[index - 1:index]))
        _if_store(missing, _slot(m.ring_tokens.base, target, index), sample)
    for index in range(17):
        column = output_zero if DELAYS[index] == 0 else output_one
        ol.store(m.emitted[index:index + 1],
                 ol.load(_slot(m.ring_tokens.base, column, index)))
    offset = ol.load(m.offset)
    # max_delay=1: the inference at offset one has no output.
    ol.store(m.ready, ol.minimum(ol.maximum(offset - 1.0, 0.0), 1.0))
    ol.store(m.offset, offset + 1.0)
    b.unscratch(output_one)
    b.unscratch(output_zero)
    b.unscratch(target)
    b.unscratch(model_input)


class DuplexScheduleImage:
    """Relocated fixed programs that implement the pinned 17-stream LMGen ring."""

    REQUIRED_IO = {"ring_tokens": 272, "ring_provided": 272,
                   "incoming_tokens": 68, "incoming_present": 68,
                   "target_tokens": 68, "target_provided": 68,
                   "emitted": 68, "frame_plan": 130 * 6 * 4,
                   "position": 24, "offset": 4, "ready": 4,
                   "current_text_id": 4, "next_text_id": 4}

    def __init__(self, spec, cfg, *, io, input_ids_at, depth_control_at,
                 depth_audio_ids_at):
        if cfg.S != 1:
            raise ValueError("duplex schedule needs one ISA slice")
        if not isinstance(spec, DuplexScheduleSpec):
            raise TypeError("duplex schedule requires a validated specification")
        self.spec, self.cfg = spec, cfg
        self.io = {name: io[name] for name in self.REQUIRED_IO}
        capacity = min(cfg.DRAM_BYTES, 1 << 32)
        spans = [(name, self.io[name], size) for name, size in self.REQUIRED_IO.items()]
        spans += [("input_ids", input_ids_at, 32),
                  ("depth_control", depth_control_at, 8),
                  ("depth_audio_ids", depth_audio_ids_at, 64)]
        for name, at, size in spans:
            if type(at) is not int or at % 4 or not 0 <= at < at + size <= capacity:
                raise MemoryError(f"invalid duplex address: {name}")
        ordered = sorted(spans, key=lambda item: item[1])
        for first, second in zip(ordered, ordered[1:]):
            if first[1] + first[2] > second[1]:
                raise ValueError("overlapping duplex buffers: " + first[0] + "/" + second[0])
        self.input_ids_at = input_ids_at
        self.depth_control_at = depth_control_at
        self.depth_audio_ids_at = depth_audio_ids_at
        self.programs = None

    def descriptors(self):
        io = self.io
        vector = lambda name, length: Tensor(io[name], (length,), (1,))
        return SimpleNamespace(
            ring_tokens=vector("ring_tokens", 4 * 17),
            ring_provided=vector("ring_provided", 4 * 17),
            incoming_tokens=vector("incoming_tokens", 17),
            incoming_present=vector("incoming_present", 17),
            target_tokens=vector("target_tokens", 17),
            target_provided=vector("target_provided", 17),
            target_token_rows=Tensor(io["target_tokens"], (17, 1), (1, 1)),
            target_provided_rows=Tensor(io["target_provided"], (17, 1), (1, 1)),
            emitted=vector("emitted", 17),
            # The reservation has spare rows, but the device lookup enforces
            # the admitted raw-call bound before accessing any ring state.
            frame_plan=Tensor(io["frame_plan"], (self.spec.max_calls, 6), (6, 1)),
            position=vector("position", 6),
            offset=vector("offset", 1), ready=vector("ready", 1),
            current_text_id=vector("current_text_id", 1),
            next_text_id=vector("next_text_id", 1),
            input_ids=Tensor(self.input_ids_at, (8,), (1,)),
            depth_control=Tensor(self.depth_control_at, (2,), (1,)),
            audio_ids=Tensor(self.depth_audio_ids_at, (16,), (1,)))

    def compile(self):
        if self.programs is None:
            programs = {}
            for name, kernel in (("prepare", _prepare), ("select_text", _select_text),
                                 ("feedback", _feedback), ("commit", _commit)):
                program = trace_fit(kernel, self.cfg, 0,
                                    lambda: {"m": self.descriptors()}).finish()
                if program.run_args or len(program) * 8 > self.cfg.IMEM_WORDS:
                    raise CompileError("duplex " + name + " needs host args or exceeds IMEM")
                programs[name] = program
            self.programs = programs
        return self.programs

    def upload(self, backend):
        self.compile()
        offsets = np.arange(130, dtype=np.float32)
        # Priming has no model input. Expose its initial column anyway, so
        # an ungenerated -2 can never reach the embedding input buffer.
        plan = np.stack((np.maximum(offsets - 1, 0) % 4, offsets % 4, (offsets + 1) % 4,
                         (offsets - 1) % 4, offsets % 4, (offsets == 1).astype(np.float32)),
                        axis=1).astype(np.float32)
        backend.write(0, self.io["frame_plan"], plan.reshape(-1))
        # Only initial construction fills the native ungenerated sentinel.
        # LMGenState.reset deliberately preserves the cache token bytes.
        backend.write(0, self.io["ring_tokens"], np.full(4 * 17, -2.0, np.float32))
        self.reset(backend)

    def reset(self, backend):
        # Source LMGenState.reset retains cache values; presence flags and the
        # offset govern validity.  Replaying a conversation overwrites its
        # reachable footprint before reading it.
        backend.write(0, self.io["ring_provided"], np.zeros(4 * 17, np.float32))
        backend.write(0, self.io["incoming_present"], np.zeros(17, np.float32))
        backend.write(0, self.io["offset"], np.zeros(1, np.float32))
        backend.write(0, self.io["ready"], np.zeros(1, np.float32))

    def submit(self, backend, *, moshi_tokens=None, text_token=None):
        """Upload fixed supplied targets; user audio comes from resident RVQ."""
        values = np.zeros(17, np.float32)
        present = np.zeros(17, np.float32)

        def token(value, upper, name):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(name + " must be an integer token")
            try:
                code = operator.index(value)
            except TypeError as error:
                raise ValueError(name + " must be an integer token") from error
            if not -1 <= code < upper:
                raise ValueError(name + " outside source token range")
            return code

        if text_token is not None:
            values[0] = token(text_token, self.spec.text_vocab, "supplied text")
            present[0] = 1.0
        if moshi_tokens is not None:
            if len(moshi_tokens) != 8:
                raise ValueError("supplied Moshi stream must have eight audio IDs")
            for index, value in enumerate(moshi_tokens):
                values[index + 1] = token(value, self.spec.audio_vocab,
                                          f"supplied Moshi audio{index}")
                present[index + 1] = 1.0
        backend.write(0, self.io["incoming_tokens"], values)
        backend.write(0, self.io["incoming_present"], present)

    def budget(self):
        return {"program_words": {name: len(program) * 8
                                  for name, program in self.compile().items()},
                "ring_bytes": 2 * 4 * 17 * 4, "max_raw_calls": self.spec.max_calls,
                "max_temporal_frames": self.spec.max_calls - 1,
                "max_emitted_frames": self.spec.emitted_frames,
                "source_delays": DELAYS, "startup_no_output_calls": 2,
                "backend": "CPU/ISA only; no physical qualification"}
