"""Device embedding assembly versus native order, zero semantics and resident I/O."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu.compiler import CompileError
from opentpu.isasim import Config
from opentpu.llm.personaplex_embeddings import EmbeddingEngine, EmbeddingImage, EmbeddingSpec


def fixture():
    spec = EmbeddingSpec(hidden=32, text_vocab=9, audio_vocab=5, audio_streams=3, voice_frames=3)
    cfg = Config(S=1, D=32, MCOLS=4, TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=1 << 20)
    rng = np.random.default_rng(829)
    tables = (rng.normal(size=(spec.text_vocab, 32)).astype(np.float32),) + tuple(
        rng.normal(size=(spec.audio_vocab, 32)).astype(np.float32) for _ in range(spec.audio_streams))
    voices = rng.normal(size=(spec.voice_frames, 32)).astype(np.float32)
    return spec, cfg, tables, voices


def reference(spec, tables, codes):
    def row(index):
        return np.zeros(spec.hidden, np.float32) if codes[index] == -1 else tables[index][codes[index]].copy()
    result = row(1)
    for index in range(2, len(tables)):
        result = np.add(result, row(index), dtype=np.float32)
    return np.add(result, row(0), dtype=np.float32)


def test_codes_match_source_audio_first_order_and_zero_padding_ids():
    spec, cfg, tables, voices = fixture()
    cases = [[-1, -1, -1, -1], [8, 4, 4, 4], [0, 1, 2, 3], [3, -1, 0, 4]]
    with EmbeddingEngine(tables, voices=voices, spec=spec, cfg=cfg) as engine:
        for codes in cases:
            assert engine.step(codes) is None
            actual = engine.read_output()
            np.testing.assert_array_equal(actual.view(np.uint32), reference(spec, tables, codes).view(np.uint32))
        engine.step(cases[0])
        assert not engine.read_output().view(np.uint32).any()  # native where(...,0) means +0
        engine.reset()
        with pytest.raises(RuntimeError, match="no completed"):
            _ = engine.output_address


def test_nonassociative_fixture_detects_text_first_or_reordered_audio():
    spec, cfg, tables, voices = fixture()
    tables[0].fill(1)
    tables[1].fill(1e8)
    tables[2].fill(1)
    tables[3].fill(-1e8)
    with EmbeddingEngine(tables, voices=voices, spec=spec, cfg=cfg) as engine:
        engine.step([0, 0, 0, 0])
        np.testing.assert_array_equal(engine.read_output(), np.ones(spec.hidden, np.float32))


def test_native_scaled_embedding_and_embed_codes_contract():
    torch = pytest.importorskip("torch")
    from moshi.models.lm import LMModel, ScaledEmbedding
    spec, cfg, tables, voices = fixture()
    modules = []
    for table in tables:
        module = ScaledEmbedding(len(table), spec.hidden, norm=False, zero_idx=-1)
        module.weight = torch.nn.Parameter(torch.from_numpy(table.copy()), requires_grad=False)
        modules.append(module)
    native = SimpleNamespace(num_codebooks=4, num_audio_codebooks=3, audio_offset=1,
                             emb=modules[1:], text_emb=modules[0])
    with EmbeddingEngine(tables, voices=voices, spec=spec, cfg=cfg) as engine:
        for codes in ([0, 1, 2, 3], [-1, -1, -1, -1], [8, 4, 4, 4]):
            expected = LMModel.embed_codes(native, torch.tensor(codes).reshape(1, 4, 1)).detach().numpy().reshape(-1)
            engine.step(codes)
            np.testing.assert_array_equal(engine.read_output().view(np.uint32), expected.view(np.uint32))


def test_device_voice_lookup_is_exact_and_step_never_reads_or_selects_host_rows(monkeypatch):
    spec, cfg, tables, voices = fixture()
    expected = voices.copy()
    with EmbeddingEngine(tables, voices=voices, spec=spec, cfg=cfg) as engine:
        read, write, run = engine.backend.read, engine.backend.write, engine.backend.run
        reads, writes, programs = [], [], []
        monkeypatch.setattr(engine.backend, "read", lambda *args: (reads.append(args), read(*args))[1])
        monkeypatch.setattr(engine.backend, "write", lambda s, a, v: (writes.append((a, np.asarray(v).nbytes)), write(s, a, v))[1])
        monkeypatch.setattr(engine.backend, "run", lambda p: (programs.append(p), run(p))[1])
        for table in tables:
            table.fill(np.nan)
        voices.fill(np.nan)
        for index in (2, 0, 1, 2):
            assert engine.step_voice(index) is None
            assert not reads
        assert writes == [(engine.image.io["voice_index"], 4)] * 4
        assert all(program is engine.programs["voice"] for program in programs)
        np.testing.assert_array_equal(engine.read_output(), expected[2])
        reads.clear()
        engine.step([0, 1, 2, 3])
        assert not reads and writes[-1] == (engine.image.io["codes"], 16)


@pytest.mark.parametrize("codes", [[0, 0], [9, 0, 0, 0], [0, 5, 0, 0], [-2, 0, 0, 0],
                                   [0.5, 0, 0, 0], [True, 0, 0, 0], [np.nan, 0, 0, 0]])
def test_invalid_codes_fail_before_device_write(codes, monkeypatch):
    spec, cfg, tables, voices = fixture()
    with EmbeddingEngine(tables, voices=voices, spec=spec, cfg=cfg) as engine:
        def forbidden(*args):
            raise AssertionError("invalid input reached device")
        monkeypatch.setattr(engine.backend, "write", forbidden)
        with pytest.raises(ValueError):
            engine.step(codes)
        assert engine.poss == [0]


@pytest.mark.parametrize("index", [-1, 3, True, 0.5])
def test_invalid_voice_index_fails_before_write(index, monkeypatch):
    spec, cfg, tables, voices = fixture()
    with EmbeddingEngine(tables, voices=voices, spec=spec, cfg=cfg) as engine:
        monkeypatch.setattr(engine.backend, "write", lambda *args: pytest.fail("unexpected write"))
        with pytest.raises(ValueError):
            engine.step_voice(index)


def test_failure_requires_reset_and_close_is_idempotent(monkeypatch):
    spec, cfg, tables, voices = fixture()
    engine = EmbeddingEngine(tables, voices=voices, spec=spec, cfg=cfg)
    run = engine.backend.run
    monkeypatch.setattr(engine.backend, "run", lambda *args: (_ for _ in ()).throw(RuntimeError("injected")))
    with pytest.raises(RuntimeError, match="injected"):
        engine.step([0, 0, 0, 0])
    with pytest.raises(RuntimeError, match="reset"):
        engine.step_voice(0)
    engine.reset()
    monkeypatch.setattr(engine.backend, "run", run)
    engine.step([0, 0, 0, 0])
    np.testing.assert_array_equal(engine.read_output(), reference(spec, tables, [0, 0, 0, 0]))
    engine.close()
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.step([0, 0, 0, 0])


def test_layout_and_program_capacity_precede_lazy_reads(monkeypatch):
    spec, cfg, _, _ = fixture()
    def unread():
        raise AssertionError("capacity gate must precede payload reads")
    image = EmbeddingImage(spec, cfg)
    with pytest.raises(MemoryError):
        EmbeddingEngine(unread, spec=spec, cfg=replace(cfg, DRAM_BYTES=image.required_bytes - 1))
    def fail(self):
        raise CompileError("program capacity")
    monkeypatch.setattr(EmbeddingImage, "compile", fail)
    with pytest.raises(CompileError):
        EmbeddingEngine(unread, voices=unread, spec=spec, cfg=cfg)


def test_unqualified_board_is_rejected_before_any_payload_read():
    def unread():
        raise AssertionError("unqualified board must fail before payload")
    with pytest.raises(ValueError, match="qualified backend"):
        EmbeddingEngine(unread, backend="board")


def test_address_capacity_and_cap_gen_guards_precede_execution():
    spec = EmbeddingSpec(hidden=4096, text_vocab=300000, audio_vocab=1, audio_streams=1)
    with pytest.raises(MemoryError, match="address capacity"):
        EmbeddingImage(spec, Config(S=1, D=128, MCOLS=4, DRAM_BYTES=1 << 40))
    spec, cfg, tables, voices = fixture()
    # Cleanup preserves the original capability failure even if close is absent.
    with pytest.raises(ValueError, match="CAP_GEN"):
        EmbeddingEngine(tables, spec=spec, voices=voices, cfg=cfg,
                        backend=lambda c, images: SimpleNamespace(generates=False))


def test_actual_boardbackend_isatransport_embedding_and_voice_protocol():
    from opentpu.host.board import BoardBackend
    from opentpu.host.fake import _target_cfg
    from opentpu.host.isaboard import IsaTransport
    from opentpu.isasim import board_config
    from opentpu.target import arch_hash, current
    cfg = replace(board_config(), DRAM_BYTES=1 << 23,
                  SEQ_WC=bool(current().arch.get("SEQ_WC", False)))
    regcfg = _target_cfg(cfg.D, cfg.MCOLS, cfg.LANES, cfg.PAIR, cfg.DSTEP,
                         cfg.STREAM, cfg.act_rows, cfg.SEQ_WC)
    regcfg.update(DRAM_BYTES=cfg.DRAM_BYTES, channel_bytes=cfg.DRAM_BYTES // 2,
                  hash=arch_hash(cfg))
    transport = IsaTransport(cfg=regcfg)
    transport.cfg = cfg
    spec = EmbeddingSpec(hidden=128, text_vocab=9, audio_vocab=5, audio_streams=3, voice_frames=2)
    rng = np.random.default_rng(921)
    tables = (rng.normal(size=(9, 128)).astype(np.float32),) + tuple(
        rng.normal(size=(5, 128)).astype(np.float32) for _ in range(3))
    voices = rng.normal(size=(2, 128)).astype(np.float32)
    factory = lambda c, images: BoardBackend(c, images, transport=transport,
                                            own_transport=True, status=False)
    with EmbeddingEngine(tables, voices=voices, spec=spec, cfg=cfg, backend=factory) as engine:
        assert engine.backend.engine is engine
        assert engine.image.cap == engine.image.batch == 1 and engine.image.kv_bytes == 0
        for codes in ([8, 4, 4, 4], [-1, 1, -1, 3]):
            engine.step(codes)
            np.testing.assert_array_equal(engine.read_output(), reference(spec, tables, codes))
        for index in (1, 0, 1):
            engine.step_voice(index)
            np.testing.assert_array_equal(engine.read_output(), voices[index])
        assert transport.sim_runs == 5
