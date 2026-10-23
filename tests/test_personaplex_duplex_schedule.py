"""Pinned LMGen routing checks for the EC2 CPU/ISA builder, never the Mac.

The independent scalar oracle follows prepare_step_input and
process_transformer_output from the retained PersonaPlex LM source. Synthetic
head outputs make feedback, forced cache values and delayed output distinct.
"""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import target
from opentpu.isasim import SimError
from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_duplex_schedule import (
    DELAYS, DuplexScheduleImage, DuplexScheduleSpec,
)


SOURCE_DELAYS = (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)
INITIAL = np.asarray([32000] + [2048] * 16, np.float32)


def make_image(max_calls=6):
    cfg = target.config("f2-hbm:p4rl8a64c150")
    assert f"{target.arch_hash(cfg):08x}" == "dd6b92a6"
    io, at = {}, 0x1000
    for name, size in DuplexScheduleImage.REQUIRED_IO.items():
        io[name] = at
        at = (at + size + 127) // 128 * 128
    image = DuplexScheduleImage(DuplexScheduleSpec(max_calls=max_calls), cfg,
        io=io, input_ids_at=0x8000, depth_control_at=0x8100,
        depth_audio_ids_at=0x8200)
    # The compiled geometry is the exact shared profile. Routing only touches
    # low memory, so the synthetic simulator need not allocate four GiB.
    backend_cfg = replace(cfg, DRAM_BYTES=1 << 20)
    backend = G.IsaBackend(backend_cfg, [np.zeros(1 << 20, np.uint8)], adopt=True)
    image.upload(backend)
    return image, backend


def read(backend, address, count):
    return np.asarray(backend.read(0, address, 4 * count), np.uint8).view(np.float32).copy()


def state(image, backend, name, count):
    return read(backend, image.io[name], count)


class SourceRing:
    """Direct source equations with no scheduler plans or descriptor helpers."""

    def __init__(self):
        self.cache = np.full((4, 17), -2, np.float32)
        self.provided = np.zeros((4, 17), bool)
        self.offset = 0

    def reset(self):
        self.offset = 0
        self.provided[:] = False

    def prepare(self, user, moshi=None, text=None):
        for index, token in enumerate(user, start=9):
            at = (self.offset + SOURCE_DELAYS[index]) % 4
            self.cache[at, index] = token
            self.provided[at, index] = True
        if moshi is not None:
            for index, token in enumerate(moshi, start=1):
                at = (self.offset + SOURCE_DELAYS[index]) % 4
                self.cache[at, index] = token
                self.provided[at, index] = True
        if text is not None:
            self.cache[self.offset % 4, 0] = text
            self.provided[self.offset % 4, 0] = True
        for index, delay in enumerate(SOURCE_DELAYS):
            if self.offset <= delay:
                self.cache[self.offset % 4, index] = INITIAL[index]
                self.provided[self.offset % 4, index] = True
        if self.offset == 0:
            self.cache[0] = INITIAL
            self.offset += 1
            return None
        return (self.cache[(self.offset - 1) % 4].copy(),
                self.cache[self.offset % 4].copy(),
                self.provided[self.offset % 4].copy())

    def commit(self, raw):
        self.provided[(self.offset - 1) % 4] = False
        at = self.offset % 4
        missing = ~self.provided[at]
        self.cache[at, missing] = raw[missing]
        output = None
        if self.offset > 1:
            output = np.asarray([self.cache[(self.offset - 1 + delay) % 4, index]
                                 for index, delay in enumerate(SOURCE_DELAYS)], np.float32)
        self.offset += 1
        return output


def calls():
    forcing = [(7, list(range(30, 38))), (9, None),
               (None, list(range(101, 109))),
               (-1, [-1] + list(range(202, 209))), (None, None),
               (21, list(range(401, 409)))]
    for call, (text, moshi) in enumerate(forcing):
        user = np.arange(8, dtype=np.float32) + 600 + call * 8
        raw = np.asarray([111 + call] + list(range(300 + call * 17, 316 + call * 17)), np.float32)
        yield user, moshi, text, raw


def run_six_calls(image, backend, oracle):
    programs = image.compile()
    emitted, model_calls = [], 0
    for call, (user, moshi, text, raw) in enumerate(calls()):
        backend.write(0, image.input_ids_at, user)  # stand-in for resident RVQ output
        image.submit(backend, moshi_tokens=moshi, text_token=text)
        expected = oracle.prepare(user, moshi, text)
        backend.run([programs["prepare"]])
        np.testing.assert_array_equal(state(image, backend, "ring_tokens", 68), oracle.cache.ravel())
        np.testing.assert_array_equal(state(image, backend, "ring_provided", 68), oracle.provided.ravel())
        assert state(image, backend, "ready", 1)[0] == 0
        # No embedding input contains the ungenerated sentinel, including the
        # prime-only call which the execution schedule skips for LM inference.
        assert not np.any(state(image, backend, "emitted", 17) == -2)
        if expected is None:
            assert call == 0
            assert state(image, backend, "offset", 1)[0] == 1
            continue
        model_calls += 1
        input_codes, target_codes, provided = expected
        np.testing.assert_array_equal(state(image, backend, "emitted", 17), input_codes)
        np.testing.assert_array_equal(state(image, backend, "target_tokens", 17), target_codes)
        np.testing.assert_array_equal(state(image, backend, "target_provided", 17), provided)
        backend.write(0, image.io["current_text_id"], raw[:1])
        backend.write(0, image.depth_audio_ids_at, raw[1:])
        backend.run([programs["select_text"]])
        selected = target_codes[0] if provided[0] else raw[0]
        assert state(image, backend, "next_text_id", 1)[0] == selected
        np.testing.assert_array_equal(read(backend, image.depth_control_at, 2), [0, selected])
        # Depth increments its global bank counter before each head. Its
        # post-head value 1..16 indexes the matching global target slot.
        for bank in range(1, 17):
            backend.write(0, image.depth_control_at, np.asarray([bank, raw[bank]], np.float32))
            backend.run([programs["feedback"]])
            selected = target_codes[bank] if provided[bank] else raw[bank]
            np.testing.assert_array_equal(read(backend, image.depth_control_at, 2), [bank, selected])
        # Supplied IDs affect conditioning/cache, retaining all sixteen raw
        # predictions for reporting and the unsupplied commit positions.
        np.testing.assert_array_equal(read(backend, image.depth_audio_ids_at, 16), raw[1:])
        assert state(image, backend, "current_text_id", 1)[0] == raw[0]
        expected_output = oracle.commit(raw)
        backend.run([programs["commit"]])
        np.testing.assert_array_equal(state(image, backend, "ring_tokens", 68), oracle.cache.ravel())
        np.testing.assert_array_equal(state(image, backend, "ring_provided", 68), oracle.provided.ravel())
        assert state(image, backend, "offset", 1)[0] == oracle.offset
        assert state(image, backend, "ready", 1)[0] == (expected_output is not None)
        if expected_output is not None:
            output = state(image, backend, "emitted", 17)
            np.testing.assert_array_equal(output, expected_output)
            assert not np.any(output == -2)
            # Mimi receives delay-aligned committed agent slots 1..8.
            assert not np.array_equal(output[1:9], raw[1:9])
            emitted.append(output)
    assert model_calls == 5 and len(emitted) == 4
    return np.stack(emitted)


def test_six_call_native_ring_forcing_all_depth_feedback_and_reset_replay():
    image, backend = make_image()
    assert DELAYS == SOURCE_DELAYS
    np.testing.assert_array_equal(state(image, backend, "ring_tokens", 68), np.full(68, -2))
    oracle = SourceRing()
    first = run_six_calls(image, backend, oracle)
    before = state(image, backend, "ring_tokens", 68)
    image.reset(backend)
    oracle.reset()
    np.testing.assert_array_equal(state(image, backend, "ring_tokens", 68), before)
    np.testing.assert_array_equal(state(image, backend, "ring_provided", 68), np.zeros(68))
    np.testing.assert_array_equal(state(image, backend, "incoming_present", 17), np.zeros(17))
    second = run_six_calls(image, backend, oracle)
    np.testing.assert_array_equal(second, first)
    with pytest.raises(SimError, match="outside its bound"):
        backend.run([image.compile()["prepare"]])
    np.testing.assert_array_equal(state(image, backend, "ring_tokens", 68), oracle.cache.ravel())


def test_exact_shared_programs_are_fixed_and_bounded():
    image, _ = make_image(128)
    assert image.cfg.DRAM_BYTES == 1 << 32
    assert image.descriptors().frame_plan.shape == (128, 6)
    assert set(image.compile()) == {"prepare", "select_text", "feedback", "commit"}
    assert all(not p.run_args and 0 < len(p) * 8 <= 32768 for p in image.compile().values())
    budget = image.budget()
    assert budget["max_raw_calls"] == 128
    assert budget["max_temporal_frames"] == 127
    assert budget["max_emitted_frames"] == 126
    assert budget["startup_no_output_calls"] == 2


@pytest.mark.parametrize("token", [-2, 2049, 0.5, True, np.nan])
def test_invalid_audio_forcing_fails_before_write(token, monkeypatch):
    image, backend = make_image()
    monkeypatch.setattr(backend, "write", lambda *a: pytest.fail("invalid packet reached device"))
    with pytest.raises(ValueError):
        image.submit(backend, moshi_tokens=[token] + [0] * 7)


@pytest.mark.parametrize("token", [-2, 32001, 0.5, True, np.nan])
def test_invalid_text_forcing_fails_before_write(token, monkeypatch):
    image, backend = make_image()
    monkeypatch.setattr(backend, "write", lambda *a: pytest.fail("invalid packet reached device"))
    with pytest.raises(ValueError):
        image.submit(backend, text_token=token)


def test_minus_one_forcing_mask_and_fixed_packet_clear(monkeypatch):
    image, backend = make_image()
    reads = []
    original = backend.read
    monkeypatch.setattr(backend, "read", lambda *args: (reads.append(args), original(*args))[1])
    image.submit(backend, text_token=-1, moshi_tokens=[-1] * 8)
    assert not reads
    np.testing.assert_array_equal(state(image, backend, "incoming_present", 17), [1] * 9 + [0] * 8)
    np.testing.assert_array_equal(state(image, backend, "incoming_tokens", 17), [-1] * 9 + [0] * 8)
    image.submit(backend)
    np.testing.assert_array_equal(state(image, backend, "incoming_present", 17), np.zeros(17))
    with pytest.raises(ValueError, match="eight"):
        image.submit(backend, moshi_tokens=[0] * 7)


@pytest.mark.parametrize("value", [0, 2, 129, True, 3.0])
def test_raw_call_bound_cannot_exceed_resident_input_codec(value):
    with pytest.raises(ValueError):
        DuplexScheduleSpec(max_calls=value)
