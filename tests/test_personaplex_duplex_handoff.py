"""Small independent ISA checks; execute on the CPU builder, never on Mac."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu import quant as Q
from opentpu.compiler import Affine, CompileError
from opentpu.isasim import Config
from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_duplex_handoff import DuplexHandoffImage, DuplexHandoffSpec


def fixture(*, head_vocab=67):
    spec = DuplexHandoffSpec(hidden=32, text_vocab=head_vocab + 1, audio_vocab=5,
                             audio_streams=16, head_vocab=head_vocab, voice_frames=3)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=8192, IMEM_WORDS=32768, PAIR=True, DRAM_BYTES=1 << 20)
    cursor = 128

    def alloc(size):
        nonlocal cursor
        at = -(-cursor // 128) * 128
        cursor = at + size
        return at

    io = dict(lm_text_embedding=alloc(4 * (spec.text_vocab + 1) * spec.hidden),
        lm_audio_embeddings=tuple(alloc(4 * (spec.audio_vocab + 1) * spec.hidden)
                                  for _ in range(spec.audio_streams)),
        voice_frames=alloc(4 * spec.voice_frames * spec.hidden),
        text_head_codes=alloc(spec.head_vocab * Q.row_bytes(spec.hidden, "int8", cfg.D)),
        text_head_scales=alloc(spec.head_vocab * Q.scale_bytes(spec.hidden, "int8", cfg.D)),
        emitted=alloc(4 * (spec.audio_streams + 1)), temporal_x=alloc(4 * spec.hidden),
        current_text_id=alloc(4), text_logits=alloc(4 * spec.head_vocab), voice_index=alloc(4))
    image = DuplexHandoffImage(spec, cfg, io=io, depth_hidden_at=alloc(4 * spec.hidden),
                               depth_control_at=alloc(8))
    rng = np.random.default_rng(648)
    tables = (rng.normal(size=(spec.text_vocab, spec.hidden)).astype(np.float32),) + tuple(
        rng.normal(size=(spec.audio_vocab, spec.hidden)).astype(np.float32)
        for _ in range(spec.audio_streams))
    head = rng.normal(0, 0.13, (spec.head_vocab, spec.hidden)).astype(np.float32)
    voice = rng.normal(size=(spec.voice_frames, spec.hidden)).astype(np.float32)
    backend = G.IsaBackend(cfg, [np.full(cfg.DRAM_BYTES, 0xA5, np.uint8)], adopt=True)
    return image, backend, tables, head, voice


def read(backend, address, count):
    return backend.read(0, address, 4 * count).view(np.float32).copy()


def embedding_reference(spec, tables, codes):
    def row(index):
        return (np.zeros(spec.hidden, np.float32) if codes[index] == -1 else
                tables[index][codes[index]].copy())
    value = row(1)
    for index in range(2, len(tables)):
        value = np.add(value, row(index), dtype=np.float32)
    return np.add(value, row(0), dtype=np.float32)


def head_reference(hidden, head, width):
    # Preserve the MX activation/weight metadata and device window-drop rule.
    activation = Q.fake_act(hidden, width)
    weight = Q.fake_w(head, "int8", width)
    offsets, window, D, transposed = weight.mx
    assert D == width and not transposed
    logits = Q.mx_matmul(activation, np.asarray(weight), offsets, window, width)
    return logits, int(np.argmax(logits))


def test_resident_emitted_codes_match_independent_embedding_and_positive_zero():
    image, backend, tables, head, voice = fixture()
    image.upload(backend, tables, head, voice)
    programs = image.compile()
    spec = image.spec
    cases = [[-1] * 17, [spec.text_vocab - 1] + [spec.audio_vocab - 1] * 16,
             [3] + [i % spec.audio_vocab if i % 3 else -1 for i in range(16)]]
    for codes in cases:
        backend.write(0, image.io["emitted"], np.asarray(codes, np.float32))
        backend.run([programs["embed_codes"]])
        actual = read(backend, image.io["temporal_x"], spec.hidden)
        expected = embedding_reference(spec, tables, codes)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        if codes == cases[0]:
            assert not actual.view(np.uint32).any()


def test_embedding_addition_keeps_audio_order_and_text_last():
    image, backend, tables, head, voice = fixture()
    for table in tables:
        table.fill(0)
    tables[0].fill(1)
    tables[1].fill(1e8)
    tables[2].fill(1)
    tables[3].fill(-1e8)
    image.upload(backend, tables, head, voice)
    backend.write(0, image.io["emitted"], np.zeros(17, np.float32))
    backend.run([image.compile()["embed_codes"]])
    np.testing.assert_array_equal(read(backend, image.io["temporal_x"], 32),
                                  np.ones(32, np.float32))


def test_head_and_depth_begin_consume_final_normalized_x_without_host_transfers(monkeypatch):
    image, backend, tables, head, voice = fixture()
    image.upload(backend, tables, head, voice)
    # Model the final temporal page's nonuniform normalized output. Any second
    # RMSNorm, use of an old embedding_sum, or stale depth hidden is observable.
    source = np.linspace(-2.5, 1.5, image.spec.hidden, dtype=np.float32)
    gain = np.linspace(0.4, 1.7, image.spec.hidden, dtype=np.float32)
    normalized = (source / np.sqrt(np.mean(source * source) + 1e-8) * gain).astype(np.float32)
    expected_logits, expected_id = head_reference(normalized, head, image.cfg.D)
    backend.write(0, image.io["temporal_x"], normalized)
    backend.write(0, image.depth_hidden_at, np.full(32, -97.0, np.float32))
    backend.write(0, image.depth_control_at, np.asarray([15, -1], np.float32))
    before = backend.read(0, 0, image.cfg.DRAM_BYTES)
    programs = image.compile()
    actual_read = backend.read
    calls = []
    actual_run = backend.run

    def run(programs, *args, **kwargs):
        assert not args and not kwargs
        calls.append(programs[0])
        return actual_run(programs)

    def forbidden(*args, **kwargs):
        raise AssertionError("handoff performed a host activation/token transfer")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "read", forbidden)
        patch.setattr(backend, "write", forbidden)
        patch.setattr(backend, "run", run)
        backend.run([programs["text_head"]])
        backend.run([programs["depth_begin"]])
    assert calls == [programs["text_head"], programs["depth_begin"]]
    np.testing.assert_allclose(read(backend, image.io["text_logits"], image.spec.head_vocab),
                               expected_logits, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(read(backend, image.io["current_text_id"], 1), [expected_id])
    np.testing.assert_array_equal(read(backend, image.depth_control_at, 2), [0, expected_id])
    np.testing.assert_array_equal(read(backend, image.depth_hidden_at, 32).view(np.uint32),
                                  normalized.view(np.uint32))
    np.testing.assert_array_equal(read(backend, image.io["temporal_x"], 32).view(np.uint32),
                                  normalized.view(np.uint32))
    # A device handoff changes only its diagnostic logits, raw text ID, and
    # depth vector/control. All other bytes stand in for retained codec/KV state.
    after = actual_read(0, 0, image.cfg.DRAM_BYTES)
    for at, size in ((image.io["text_logits"], 4 * image.spec.head_vocab),
                     (image.io["current_text_id"], 4), (image.depth_hidden_at, 128),
                     (image.depth_control_at, 8)):
        before[at:at + size] = after[at:at + size]
    np.testing.assert_array_equal(before, after)


def test_voice_replay_selects_resident_rows_with_no_transfers(monkeypatch):
    image, backend, tables, head, voice = fixture()
    expected = voice.copy()
    image.upload(backend, tables, head, voice)
    voice.fill(np.nan)  # Device replay must not retain a host table dependency.
    program = image.compile()["voice_replay"]
    for index in (2, 0, 1, 2):
        backend.write(0, image.io["voice_index"], np.asarray([index], np.float32))
        with monkeypatch.context() as patch:
            patch.setattr(backend, "read", lambda *a: pytest.fail("unexpected host read"))
            patch.setattr(backend, "write", lambda *a: pytest.fail("unexpected host write"))
            backend.run([program])
        np.testing.assert_array_equal(read(backend, image.io["temporal_x"], 32), expected[index])


def test_text_head_first_ties_across_chunks_and_partial_tail():
    image, backend, tables, head, voice = fixture(head_vocab=1027)
    assert image.head_chunk == 1024
    head.fill(0)
    hidden = np.zeros(32, np.float32)
    hidden[0] = 1
    low, high = 7, image.head_chunk + 1
    head[low, 0] = head[high, 0] = 1
    image.upload(backend, tables, head, voice)
    backend.write(0, image.io["temporal_x"], hidden)
    backend.run([image.compile()["text_head"]])
    expected, token = head_reference(hidden, head, image.cfg.D)
    assert token == low and expected[low] == expected[high] > 0
    assert read(backend, image.io["current_text_id"], 1)[0] == low
    head.fill(0)
    image.upload(backend, tables, head, voice)
    backend.run([image.compile()["text_head"]])
    assert read(backend, image.io["current_text_id"], 1)[0] == 0


def test_upload_is_immutable_only_and_omitted_voices_are_positive_zero():
    image, backend, tables, head, _ = fixture()
    before = backend.read(0, 0, image.cfg.DRAM_BYTES)
    image.upload(backend, tables, head)
    after = backend.read(0, 0, image.cfg.DRAM_BYTES)
    for span in image.spans:
        at, size = span["base"], span["bytes"]
        if span["kind"] == "immutable":
            before[at:at + size] = after[at:at + size]
        else:
            np.testing.assert_array_equal(after[at:at + size], before[at:at + size])
    np.testing.assert_array_equal(before, after)
    for at in (image.io["lm_text_embedding"], *image.io["lm_audio_embeddings"]):
        assert not read(backend, at, 32).view(np.uint32).any()
    assert not read(backend, image.io["voice_frames"], 3 * 32).view(np.uint32).any()


def test_invalid_layout_payload_and_program_capacity_fail_before_upload(monkeypatch):
    image, backend, tables, head, voice = fixture()
    io = dict(image.io)
    io["temporal_x"] = image.depth_hidden_at
    with pytest.raises(MemoryError, match="overlapping"):
        DuplexHandoffImage(image.spec, image.cfg, io=io,
            depth_hidden_at=image.depth_hidden_at, depth_control_at=image.depth_control_at)
    with pytest.raises(MemoryError, match="address"):
        DuplexHandoffImage(image.spec, replace(image.cfg, DRAM_BYTES=image.required_bytes - 1),
            io=image.io, depth_hidden_at=image.depth_hidden_at, depth_control_at=image.depth_control_at)
    with pytest.raises(ValueError):
        DuplexHandoffSpec(head_vocab=32002)
    with pytest.raises(ValueError):
        DuplexHandoffSpec(voice_frames=0)
    monkeypatch.setattr(backend, "write", lambda *args: pytest.fail("invalid payload uploaded"))
    with pytest.raises(ValueError, match="shape"):
        image.upload(backend, tables, head[:-1], voice)
    broken = head.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        image.upload(backend, tables, broken, voice)

    def capacity_failure():
        raise CompileError("program capacity")

    monkeypatch.setattr(image, "compile", capacity_failure)
    unread = lambda: pytest.fail("lazy payload read before program capacity gate")
    with pytest.raises(CompileError, match="capacity"):
        image.upload(backend, unread, unread, unread)


def test_production_descriptors_compile_in_four_exclusive_shared_slots():
    from opentpu import target
    from opentpu.llm.personaplex_duplex_layout import DuplexLayout

    layout = DuplexLayout.production()
    image = DuplexHandoffImage.from_layout(layout)
    assert target.arch_hash(image.cfg) == 0xdd6b92a6
    assert len(layout.temporal.pages) == 6 and len(layout.program_slots) == 74
    assert image.spec == DuplexHandoffSpec()
    assert image.table_bytes == 1_061_715_968
    assert image.head_bytes == 132_224_000
    programs = image.compile()
    assert tuple(programs) == image.PROGRAM_NAMES
    assert image.compile() is programs
    assert all(not p.run_args and len(p) * 8 <= image.cfg.IMEM_WORDS for p in programs.values())
    assert len(set(image.program_slots.values())) == 4
    assert set(image.program_slots.values()) <= set(layout.program_slots)
    m = image.descriptors()
    assert Affine.of(m.codes.base).const == layout.io["emitted"]
    assert Affine.of(m.x.base).const == Affine.of(m.output.base).const == layout.io["temporal_x"]
    assert Affine.of(m.depth_hidden.base).const == layout.output.io["hidden"]
    assert Affine.of(m.depth_control.base).const == layout.output.io["control"]
    assert Affine.of(m.token_id.base).const == layout.io["current_text_id"]
    invalid = SimpleNamespace(**layout.__dict__)
    invalid.program_slots = layout.program_slots[:-1]
    with pytest.raises(ValueError, match="six-page"):
        DuplexHandoffImage.from_layout(invalid)
