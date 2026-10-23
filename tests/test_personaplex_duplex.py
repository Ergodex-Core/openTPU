"""Duplex protocol checks. Recording programs do not establish numerical acceptance.

The real production addresses and component descriptors are retained. Only the
immutable packet lengths are reduced for these ordering/transfer tests; no
checkpoint, 4 GiB machine, model arithmetic or hardware backend is involved.
"""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu.compiler import CompileError
from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_duplex import DuplexEngine, DuplexPrograms, DuplexSubmission
from opentpu.llm.personaplex_duplex_handoff import DuplexHandoffImage
from opentpu.llm.personaplex_duplex_layout import DuplexLayout
from opentpu.llm.personaplex_duplex_schedule import DuplexScheduleImage, DuplexScheduleSpec


class Program:
    def __init__(self, name, *, words=8, run_args=()):
        self.name, self.words, self.run_args = name, words, run_args

    def __len__(self):
        return self.words // 8


class RecordingBackend(G.IsaBackend):
    """Exercises orchestration only; deliberately does not execute ISA programs."""

    def __init__(self, layout):
        self.machine = SimpleNamespace(cfg=layout.cfg, slices=[SimpleNamespace(
            halted=True, waiting=None, polling=None)])
        self.pcm_at, rows, cols = layout.output.pcm.layer_outputs[14]
        self.pcm_bytes = rows * cols * 4
        self.events = []
        self.incomplete = None

    def write(self, sid, address, data):
        assert sid == 0
        self.events.append(("write", address, data.nbytes))

    def read(self, sid, address, nbytes):
        assert sid == 0 and (address, nbytes) == (self.pcm_at, self.pcm_bytes)
        self.events.append(("read", address, nbytes))
        return np.zeros(nbytes, np.uint8)

    def run(self, programs, args=None):
        assert len(programs) == 1 and args is None
        program = programs[0]
        self.events.append(("run", program.name))
        state = self.machine.slices[0]
        state.halted, state.waiting, state.polling = True, None, None
        if program.name == self.incomplete:
            state.waiting = object()
        return {"recorded": program.name}


def bundle(layout):
    def group(prefix, n):
        return tuple(Program(f"{prefix}{i}") for i in range(n))

    mimi = copy.copy(layout.output.mimi)
    mimi.audio_ids_at = layout.io["emitted"] + 4
    decoder_count = 1 + layout.output.decoder.spec.layers
    output = tuple(Program(name) for name in ("condition", "depth", "raw_head", "output_rvq"))
    output += group("decoder", decoder_count) + group("pcm", len(layout.output.pcm.ops))
    return DuplexPrograms(group("encoder", len(layout.encoder.program_slots)),
        group("input_transformer", len(layout.input_transformer.program_slots)),
        group("downsample", len(layout.input_downsample.program_slots)),
        group("input_rvq", len(layout.input_rvq.program_slots)), group("temporal", 6),
        {name: Program(name) for name in ("prepare", "select_text", "feedback", "commit")},
        {name: Program(name) for name in ("embed_codes", "voice_replay", "text_head", "depth_begin")},
        output, mimi)


@pytest.fixture
def setup():
    layout = DuplexLayout.production()
    layout.packets = tuple(replace(packet, bytes=128) for packet in layout.packets)
    backend = RecordingBackend(layout)
    schedule = DuplexScheduleImage(DuplexScheduleSpec(3), layout.cfg, io=layout.io,
        input_ids_at=layout.input_rvq.io["output_ids"],
        depth_control_at=layout.output.io["control"],
        depth_audio_ids_at=layout.output.io["audio_ids"])
    handoff = DuplexHandoffImage.from_layout(layout)
    return layout, backend, schedule, handoff, bundle(layout)


def engine(setup, **kw):
    layout, backend, schedule, handoff, programs = setup
    return DuplexEngine(backend, layout, (bytes(128),) * 6, schedule=schedule,
                        handoff=handoff, programs=programs, max_calls=3, **kw)


def names(events):
    return [event[1] for event in events if event[0] == "run"]


def expected_call(programs, call):
    sequence = [p.name for group in (programs.encoder, programs.input_transformer,
                programs.input_downsample, programs.input_rvq) for p in group]
    sequence.append("prepare")
    if call:
        sequence += ["embed_codes", *(p.name for p in programs.temporal),
                     "text_head", "depth_begin", "select_text"]
        sequence += ["condition", "depth", "raw_head", "feedback"] * 16
        sequence.append("commit")
        if call >= 2:
            sequence += [p.name for p in programs.output[3:]]
    return sequence


def test_cold_priming_then_fixed_lm_depth_and_first_pcm_order(setup):
    runner = engine(setup)
    backend, programs = setup[1], setup[4]
    raw = np.zeros(runner.layout.encoder.spec.samples, np.float32)
    for call in range(3):
        backend.events.clear()
        result = runner.step(raw)
        assert names(backend.events) == expected_call(programs, call)
        assert (result is None) == (call < 2)
        assert runner.raw_calls == call + 1
        assert runner.temporal_frames == call
        assert runner.output_frames == max(call - 1, 0)
    assert result.shape == (runner.layout.output.pcm.spec.samples,)
    assert result.dtype == np.float32
    assert not runner.budget()["full_speech_acceptance"]
    assert not runner.budget()["hardware_ready"]
    assert not runner.budget()["readiness"]


def test_running_frames_transfer_only_pcm_fixed_targets_and_immutable_pages(setup):
    runner = engine(setup, submissions=(DuplexSubmission(text_token=10,
        moshi_tokens=(1,) * 8),))
    layout, backend = setup[:2]
    backend.events.clear()
    for _ in range(3):
        runner.step(np.zeros(layout.encoder.spec.samples, np.float32))
    writes = [event for event in backend.events if event[0] == "write"]
    allowed = {layout.encoder.input_at, layout.io["incoming_tokens"],
               layout.io["incoming_present"], layout.packets[0].base}
    assert {event[1] for event in writes} == allowed
    assert len([event for event in writes if event[1] == layout.packets[0].base]) == 12
    assert [event for event in backend.events if event[0] == "read"] == [
        ("read", backend.pcm_at, backend.pcm_bytes)]
    assert not any(event[0] == "write" and event[1] == layout.output.io["control"]
                   for event in backend.events)
    for index, event in enumerate(backend.events):
        if event[0] == "write" and event[1] == layout.packets[0].base:
            assert backend.events[index - 1][0] == "run"
    with pytest.raises(TypeError):
        runner.programs.schedule["prepare"] = Program("replacement")


def test_conversation_reset_is_explicit_and_replays_priming(setup):
    runner = engine(setup)
    layout, backend = setup[:2]
    reset_events = list(backend.events)
    raw = np.zeros(layout.encoder.spec.samples, np.float32)
    runner.step(raw)
    runner.step(raw)
    backend.events.clear()
    runner.reset()
    assert backend.events == reset_events
    assert (runner.raw_calls, runner.temporal_frames, runner.output_frames) == (0, 0, 0)
    backend.events.clear()
    assert runner.step(raw) is None
    assert names(backend.events) == expected_call(runner.programs, 0)


def test_four_call_protocol_emits_two_consecutive_pcm_frames_before_reset(setup):
    """Protocol-only continuity check; the recorder computes no PCM values."""
    layout, backend, _, handoff, programs = setup
    schedule = DuplexScheduleImage(DuplexScheduleSpec(4), layout.cfg, io=layout.io,
        input_ids_at=layout.input_rvq.io["output_ids"],
        depth_control_at=layout.output.io["control"],
        depth_audio_ids_at=layout.output.io["audio_ids"])
    runner = DuplexEngine(backend, layout, (bytes(128),) * 6, schedule=schedule,
                          handoff=handoff, programs=programs, max_calls=4)
    initial_reset = list(backend.events)
    raw = np.zeros(layout.encoder.spec.samples, np.float32)
    outputs = []
    for call in range(4):
        backend.events.clear()
        outputs.append(runner.step(raw))
        assert names(backend.events) == expected_call(programs, call)
        assert not any(event[0] == "write" and event[1] in {
            layout.output.io["control"], layout.output.decoder.io["state"],
            layout.output.decoder.io["overlap"], layout.temporal.io["frame"]}
            for event in backend.events)
    assert outputs[0] is outputs[1] is None
    assert all(value.shape == (layout.output.pcm.spec.samples,) for value in outputs[2:])
    assert outputs[2] is not outputs[3]
    assert (runner.raw_calls, runner.temporal_frames, runner.output_frames) == (4, 3, 2)
    backend.events.clear()
    runner.reset()
    assert backend.events == initial_reset
    backend.events.clear()
    assert runner.step(raw) is None
    assert names(backend.events) == expected_call(programs, 0)


def test_call_limit_and_invalid_pcm_are_guarded_before_io(setup):
    runner = engine(setup)
    backend = setup[1]
    backend.events.clear()
    with pytest.raises(ValueError, match="finite raw FP32"):
        runner.step(np.zeros(1, np.float32))
    assert backend.events == []
    raw = np.zeros(runner.layout.encoder.spec.samples, np.float32)
    for _ in range(3):
        runner.step(raw)
    before = list(backend.events)
    with pytest.raises(RuntimeError, match="raw-call limit"):
        runner.step(raw)
    assert backend.events == before
    runner.close()
    with pytest.raises(RuntimeError, match="closed"):
        runner.step(raw)


def test_incomplete_predecessor_blocks_refill_and_requires_conversation_reset(setup):
    runner = engine(setup)
    layout, backend = setup[:2]
    raw = np.zeros(layout.encoder.spec.samples, np.float32)
    runner.step(raw)
    backend.events.clear()
    backend.incomplete = "embed_codes"
    with pytest.raises(RuntimeError, match="halted and drained"):
        runner.step(raw)
    assert not any(event[0] == "write" and event[1] == layout.packets[0].base
                   for event in backend.events)
    with pytest.raises(RuntimeError, match="previous duplex call failed"):
        runner.step(raw)
    backend.incomplete = None
    runner.reset()
    assert runner.step(raw) is None


def test_refill_checks_actual_halt_and_drain_state(setup):
    runner = engine(setup)
    backend = setup[1]
    backend.events.clear()
    for halted, drained in ((False, False), (True, False), (False, True)):
        runner._halted, runner._drained = halted, drained
        with pytest.raises(RuntimeError, match="halted and drained"):
            runner._temporal([])
    assert backend.events == []


def test_preflight_rejects_mutable_packets_bad_profile_seams_and_host_arguments(setup):
    layout, backend, schedule, handoff, programs = setup
    with pytest.raises(ValueError, match="software IsaBackend"):
        engine(setup, backend_kind="hardware")
    with pytest.raises(ValueError, match="immutable prepacked"):
        DuplexEngine(backend, layout, (bytearray(128),) * 6, schedule=schedule,
                     handoff=handoff, programs=programs, max_calls=3)
    with pytest.raises(ValueError, match="3..128"):
        DuplexEngine(backend, layout, (bytes(128),) * 6, max_calls=129)
    bad = replace(programs, output=(Program("host", run_args=("argument",)),
                                   *programs.output[1:]))
    with pytest.raises(CompileError, match="host arguments"):
        DuplexEngine(backend, layout, (bytes(128),) * 6, schedule=schedule,
                     handoff=handoff, programs=bad, max_calls=3)
    original = handoff.depth_hidden_at
    handoff.depth_hidden_at += 128
    with pytest.raises(ValueError, match="resident seam"):
        engine(setup)
    handoff.depth_hidden_at = original
    assert backend.events == []


def test_raw_depth_ids_are_rejected_for_output_rvq(setup):
    programs = setup[4]
    programs.mimi.audio_ids_at = setup[0].output.io["audio_ids"]
    with pytest.raises(ValueError, match="emitted slots 1..8"):
        engine(setup)
    assert setup[1].events == []


@pytest.mark.parametrize("kwargs", [dict(text_token=True), dict(text_token=32001),
    dict(text_token=2.5), dict(moshi_tokens=(1,) * 7), dict(moshi_tokens=(2049,) * 8)])
def test_predetermined_targets_are_validated_before_execution(kwargs):
    with pytest.raises(ValueError):
        DuplexSubmission(**kwargs)


def test_compile_factory_rebases_only_mimi_descriptor_without_duplicate_state(setup, monkeypatch):
    layout, _, schedule, handoff, recorded = setup
    for image, programs in ((layout.encoder, recorded.encoder),
            (layout.input_transformer, recorded.input_transformer),
            (layout.input_downsample, recorded.input_downsample),
            (layout.input_rvq, recorded.input_rvq), (layout.temporal, recorded.temporal),
            (schedule, recorded.schedule), (handoff, recorded.handoff)):
        monkeypatch.setattr(image, "compile", lambda programs=programs: programs)
    original = layout.output.mimi
    original.programs = [Program("raw_depth_rvq")]
    monkeypatch.setattr(type(original), "compile", lambda self: [Program(
        "delayed_rvq" if self.audio_ids_at == layout.io["emitted"] + 4 else "raw_depth_rvq")])
    monkeypatch.setattr(layout.output.condition, "compile", lambda: [Program("condition")])
    monkeypatch.setattr(layout.output.depth, "compile", lambda: [Program("depth")])
    # The established head compiler would perform real ISA compilation. Stub
    # only the outer output compiler here to inspect which Mimi view it owns.
    def output_compile(image):
        return (*recorded.output[:3], *image.mimi.compile(), *recorded.output[4:])
    monkeypatch.setattr(type(layout.output), "compile", output_compile)
    compiled = DuplexPrograms.from_layout(layout, schedule, handoff)
    assert compiled.mimi is not original
    assert compiled.mimi.audio_ids_at == layout.io["emitted"] + 4
    assert original.audio_ids_at == layout.output.io["audio_ids"]
    assert original.programs[0].name == "raw_depth_rvq"
    assert compiled.mimi.spans is original.spans
    assert compiled.mimi.io is original.io
    assert compiled.mimi.projection_offsets is original.projection_offsets
    assert compiled.output[3].name == "delayed_rvq"


def test_production_factory_compiles_all_74_argument_free_programs_with_delayed_mimi():
    """Real compiler gate, separate from the recording-backend protocol tests.

    This compiles actual production descriptors without constructing a backend
    or allocating DRAM. Execution and end-to-end numerical quality remain a
    separate gate requiring the pinned checkpoint and raw PCM fixture.
    """
    layout = DuplexLayout.production()
    schedule = DuplexScheduleImage(DuplexScheduleSpec(), layout.cfg, io=layout.io,
        input_ids_at=layout.input_rvq.io["output_ids"],
        depth_control_at=layout.output.io["control"],
        depth_audio_ids_at=layout.output.io["audio_ids"])
    handoff = DuplexHandoffImage.from_layout(layout)
    original = layout.output.mimi
    programs = DuplexPrograms.from_layout(layout, schedule, handoff)
    all_programs = [program for group in (programs.encoder, programs.input_transformer,
        programs.input_downsample, programs.input_rvq, programs.temporal, programs.output)
        for program in group]
    all_programs += list(programs.schedule.values()) + list(programs.handoff.values())
    assert len(all_programs) == len(layout.program_slots) == 74
    assert all(not program.run_args and 0 < len(program) * 8 <= 32768
               for program in all_programs)
    assert len(programs.temporal) == 6
    assert programs.mimi.audio_ids_at == layout.io["emitted"] + 4
    assert original.audio_ids_at == layout.output.io["audio_ids"]
    assert programs.mimi.counter_at == original.counter_at
    assert programs.mimi.expected_steps == original.expected_steps == 16
    assert programs.mimi.spans is original.spans
    assert programs.mimi.io is original.io
    assert programs.mimi.tables_at == original.tables_at
    assert programs.mimi.projection_offsets is original.projection_offsets
