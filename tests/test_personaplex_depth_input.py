"""Device-selected depth projections and ragged previous-token embeddings.

Run on EC2 only. Synthetic fixtures and IsaTransport need no checkpoint or
physical FPGA. Source/reference arithmetic below belongs only to validation.
"""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import quant as Q
from opentpu.compiler import CompileError
from opentpu.isasim import Config
from opentpu.llm import personaplex_depth_input as D
from opentpu.llm.personaplex_depth_input import DepthInputEngine, DepthInputImage, DepthInputSpec


def fixture(*, multi_linear=True, codebooks=16):
    spec = DepthInputSpec(hidden=32, depth=32, codebooks=codebooks,
                          text_vocab=9, audio_vocab=5, multi_linear=multi_linear)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=8192, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 23)
    rng = np.random.default_rng(615)
    projections = rng.normal(0, 0.11, (codebooks if multi_linear else 1, 32, 32)).astype(np.float32)
    embeddings = (rng.normal(size=(9, 32)).astype(np.float32),) + tuple(
        rng.normal(size=(5, 32)).astype(np.float32) for _ in range(codebooks - 1))
    inputs = rng.normal(size=(20, 32)).astype(np.float32)
    inputs *= np.linspace(0.4, 5.0, len(inputs), dtype=np.float32)[:, None]
    return spec, cfg, projections, embeddings, inputs


def projection_reference(hidden, projection, width):
    activation = Q.fake_act(hidden, width)
    weight = Q.fake_w(projection, "int8", width)
    offsets, window, D_, transposed = weight.mx
    assert D_ == width and not transposed
    return Q.mx_matmul(activation, np.asarray(weight), offsets, window, width)


def embedding_reference(spec, embeddings, codebook, previous_token):
    return (np.zeros(spec.depth, np.float32) if previous_token == -1 else
            embeddings[codebook][previous_token])


def assert_result(engine, hidden, projections, embeddings, codebook, previous_token, spec, cfg):
    projected = engine.read_projected()
    source_projection = projections[codebook if spec.multi_linear else 0]
    expected = projection_reference(hidden, source_projection, cfg.D)
    np.testing.assert_allclose(projected, expected, rtol=2e-5, atol=2e-5)
    # Check the addition separately at the device's actual rounded projection.
    # This keeps matrix accumulation roundoff out of the FP32 embedding contract.
    added = np.add(projected, embedding_reference(spec, embeddings, codebook, previous_token),
                   dtype=np.float32)
    np.testing.assert_array_equal(engine.read_output().view(np.uint32), added.view(np.uint32))
    np.testing.assert_array_equal(engine.read_projected(), projected)


@pytest.mark.parametrize("multi_linear", [True, False])
def test_all_sixteen_indices_nonsequential_and_repeated_match_independent_reference(multi_linear):
    spec, cfg, projections, embeddings, inputs = fixture(multi_linear=multi_linear)
    order = [15, 0, 8, 3, 14, 1, 7, 12, 2, 9, 4, 13, 5, 11, 6, 10, 15, 0, 3, 15]
    with DepthInputEngine(projections, embeddings, spec=spec, cfg=cfg) as engine:
        for position, (hidden, codebook) in enumerate(zip(inputs, order)):
            previous = -1 if position % 5 == 0 else ((8 if codebook == 0 else 4) - position % 3)
            assert engine.step(hidden, np.int64(codebook), np.int64(previous)) is None
            assert_result(engine, hidden, projections, embeddings, codebook, previous, spec, cfg)
        assert engine.last_stats and engine.poss == [len(order)]


def test_shared_projection_is_identical_while_embedding_table_follows_codebook():
    spec, cfg, projections, embeddings, inputs = fixture(multi_linear=False)
    hidden = inputs[-1]
    with DepthInputEngine(projections, embeddings, spec=spec, cfg=cfg) as engine:
        projected = None
        for codebook in (15, 0, 1, 8, 15):
            engine.step(hidden, codebook, 0)
            if projected is None:
                projected = engine.read_projected()
            else:
                np.testing.assert_array_equal(engine.read_projected(), projected)
            assert_result(engine, hidden, projections, embeddings, codebook, 0, spec, cfg)


def test_projection_has_no_extra_normalization():
    spec, cfg, projections, embeddings, inputs = fixture(codebooks=1)
    hidden = inputs[-1]
    expected = projection_reference(hidden, projections[0], cfg.D)
    normalized = hidden / np.sqrt(np.mean(hidden.astype(np.float64) ** 2) + 1e-8)
    wrong = projection_reference(normalized, projections[0], cfg.D)
    assert np.linalg.norm(expected - wrong) > np.linalg.norm(expected) * 0.5
    with DepthInputEngine(projections, embeddings, spec=spec, cfg=cfg) as engine:
        engine.step(hidden, 0, -1)
        np.testing.assert_allclose(engine.read_projected(), expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(engine.read_output(), engine.read_projected())


def test_text_audio_padding_rows_and_negative_one_positive_zero():
    spec, cfg, projections, embeddings, inputs = fixture()
    projections.fill(0)
    for codebook, table in enumerate(embeddings):
        table[:] = (codebook * 100 + np.arange(len(table)) + 1)[:, None]
    cases = [(0, 0), (0, 8), (1, 0), (1, 4), (15, 4), (15, 0), (0, -1), (15, -1)]
    with DepthInputEngine(projections, embeddings, spec=spec, cfg=cfg) as engine:
        for codebook, previous in cases:
            engine.step(inputs[0], codebook, previous)
            expected = embedding_reference(spec, embeddings, codebook, previous)
            np.testing.assert_array_equal(engine.read_output().view(np.uint32), expected.view(np.uint32))
            if previous == -1:
                assert not engine.read_output().view(np.uint32).any()


def test_step_uses_only_resident_sources_and_transfers_hidden_plus_two_ids(monkeypatch):
    spec, cfg, projections, embeddings, inputs = fixture()
    order = [(15, 4), (0, 8), (9, -1), (15, 0)]
    expected = []
    for hidden, (codebook, previous) in zip(inputs, order):
        projected = projection_reference(hidden, projections[codebook], cfg.D).astype(np.float32)
        expected.append(np.add(projected, embedding_reference(spec, embeddings, codebook, previous),
                               dtype=np.float32))
    with DepthInputEngine(projections, embeddings, spec=spec, cfg=cfg) as engine:
        projections.fill(np.nan)
        for table in embeddings:
            table.fill(np.nan)
        read, write, run = engine.backend.read, engine.backend.write, engine.backend.run
        reads, writes, programs = [], [], []

        def traced_read(*args):
            reads.append(args)
            return read(*args)

        def traced_write(s, address, value):
            writes.append((s, address, np.asarray(value).copy()))
            return write(s, address, value)

        def traced_run(program, *args, **kwargs):
            programs.append(program)
            assert not args and kwargs.get("args") is None
            return run(program, *args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("step attempted host projection or weight quantization")

        monkeypatch.setattr(engine.backend, "read", traced_read)
        monkeypatch.setattr(engine.backend, "write", traced_write)
        monkeypatch.setattr(engine.backend, "run", traced_run)
        monkeypatch.setattr(Q, "fake_act", forbidden)
        monkeypatch.setattr(Q, "fake_w", forbidden)
        for index, (hidden, (codebook, previous)) in enumerate(zip(inputs, order)):
            before = len(writes)
            assert engine.step(hidden, codebook, previous) is None
            assert not reads
            batch = writes[before:]
            assert all(s == 0 and value.dtype == np.float32 for s, _, value in batch)
            assert sum(value.nbytes for _, _, value in batch) == 4 * spec.hidden + 8
            vector = [value for _, _, value in batch if value.nbytes == 4 * spec.hidden]
            assert len(vector) == 1
            np.testing.assert_array_equal(vector[0].reshape(-1), inputs[index])
            ids = np.concatenate([value.reshape(-1) for _, _, value in batch
                                  if value.nbytes != 4 * spec.hidden])
            np.testing.assert_array_equal(ids, np.array([codebook, previous], np.float32))
        assert len(programs) == len(order) and all(p is programs[0] for p in programs)
        np.testing.assert_allclose(engine.read_output(), expected[-1], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("bad", [-1, 16, True, np.bool_(False), 0.0, 1.5])
def test_invalid_codebook_fails_before_backend_write(bad, monkeypatch):
    spec, cfg, projections, embeddings, inputs = fixture()
    with DepthInputEngine(projections, embeddings, spec=spec, cfg=cfg) as engine:
        monkeypatch.setattr(engine.backend, "write", lambda *args: pytest.fail("unexpected write"))
        with pytest.raises(ValueError):
            engine.step(inputs[0], bad, 0)
        assert engine.poss == [0]


@pytest.mark.parametrize("codebook,previous", [(0, 9), (1, 5), (15, 5), (0, -2), (1, -2),
                                              (0, True), (1, np.bool_(False)), (0, 0.0),
                                              (1, 1.5), (0, np.nan)])
def test_previous_token_range_depends_on_selected_embedding_type(codebook, previous, monkeypatch):
    spec, cfg, projections, embeddings, inputs = fixture()
    with DepthInputEngine(projections, embeddings, spec=spec, cfg=cfg) as engine:
        monkeypatch.setattr(engine.backend, "write", lambda *args: pytest.fail("unexpected write"))
        with pytest.raises(ValueError):
            engine.step(inputs[0], codebook, previous)
        assert engine.poss == [0]


@pytest.mark.parametrize("bad", ["shape", "nan", "infinity", "overflow"])
def test_hidden_shape_finiteness_and_float32_overflow_are_checked_before_write(bad, monkeypatch):
    spec, cfg, projections, embeddings, inputs = fixture()
    hidden = (inputs[:1] if bad == "shape" else
              np.full(spec.hidden, {"nan": np.nan, "infinity": np.inf, "overflow": 1e300}[bad]))
    with DepthInputEngine(projections, embeddings, spec=spec, cfg=cfg) as engine:
        write = engine.backend.write
        monkeypatch.setattr(engine.backend, "write", lambda *args: pytest.fail("unexpected write"))
        with np.errstate(over="ignore"), pytest.raises(ValueError):
            engine.step(hidden, 0, 0)
        assert engine.poss == [0]
        monkeypatch.setattr(engine.backend, "write", write)
        engine.step(inputs[0], 0, 0)
        assert engine.poss == [1]


def test_failure_ready_reset_and_idempotent_close(monkeypatch):
    spec, cfg, projections, embeddings, inputs = fixture()
    engine = DepthInputEngine(projections, embeddings, spec=spec, cfg=cfg)
    for action in (lambda: engine.output_address, engine.read_output, engine.read_projected):
        with pytest.raises(RuntimeError):
            action()
    engine.step(inputs[0], 15, 4)
    first, projected = engine.read_output(), engine.read_projected()
    run = engine.backend.run

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic depth input failure")

    monkeypatch.setattr(engine.backend, "run", fail)
    with pytest.raises(RuntimeError, match="synthetic depth input"):
        engine.step(inputs[1], 0, 8)
    assert engine.poss == [1]
    monkeypatch.setattr(engine.backend, "run", run)
    for action in (lambda: engine.step(inputs[0], 0, 0), lambda: engine.output_address,
                   engine.read_output, engine.read_projected):
        with pytest.raises(RuntimeError):
            action()
    engine.reset()
    assert engine.poss == [0] and engine.last_stats == {}
    for action in (lambda: engine.output_address, engine.read_output, engine.read_projected):
        with pytest.raises(RuntimeError):
            action()
    engine.step(inputs[0], 15, 4)
    np.testing.assert_array_equal(engine.read_output(), first)
    np.testing.assert_array_equal(engine.read_projected(), projected)
    close = getattr(engine.backend, "close", lambda: None)
    closed = []

    def traced_close():
        closed.append(True)
        close()

    monkeypatch.setattr(engine.backend, "close", traced_close, raising=False)
    engine.close()
    engine.close()
    assert closed == [True]
    for action in (lambda: engine.step(inputs[0], 0, 0), engine.reset,
                   lambda: engine.output_address, engine.read_output, engine.read_projected):
        with pytest.raises(RuntimeError, match="closed"):
            action()


def unread():
    raise AssertionError("payload read before layout/program preflight")


def test_memory_imem_and_compile_preflight_precede_lazy_payload_reads(monkeypatch):
    spec, cfg, _, _, _ = fixture()
    layout = DepthInputImage(spec, cfg)
    with pytest.raises(MemoryError):
        DepthInputEngine(unread, unread, spec=spec,
                         cfg=replace(cfg, DRAM_BYTES=layout.required_bytes - 1))
    with pytest.raises((CompileError, MemoryError)):
        DepthInputEngine(unread, unread, spec=spec, cfg=replace(cfg, IMEM_WORDS=8))

    def fail(self):
        raise CompileError("synthetic depth-input capacity failure")

    monkeypatch.setattr(DepthInputImage, "compile", fail)
    with pytest.raises(CompileError, match="capacity"):
        DepthInputEngine(unread, unread, spec=spec, cfg=cfg)


def test_lazy_projection_and_embedding_sources_are_read_once_after_compile(monkeypatch):
    spec, cfg, projections, embeddings, inputs = fixture()
    compile_program = DepthInputImage.compile
    events = []

    def compile(self):
        result = compile_program(self)
        events.append("compile")
        return result

    def load(name, value):
        assert "compile" in events
        events.append(name)
        return value

    monkeypatch.setattr(DepthInputImage, "compile", compile)
    with DepthInputEngine(lambda: load("projections", projections),
                          lambda: load("embeddings", embeddings), spec=spec, cfg=cfg) as engine:
        engine.step(inputs[0], 15, 4)
        engine.reset()
        engine.step(inputs[0], 0, 8)
    assert events[0] == "compile"
    assert events.count("compile") == events.count("projections") == events.count("embeddings") == 1


@pytest.mark.parametrize("bad", ["projection_shape", "projection_nan", "projection_overflow",
                                  "embedding_count", "text_shape", "audio_shape", "embedding_nan",
                                  "embedding_overflow"])
def test_bad_projection_and_embedding_payloads_are_rejected(bad):
    spec, cfg, projections, embeddings, _ = fixture()
    tables = list(embeddings)
    if bad == "projection_shape":
        projections = projections[:-1]
    elif bad == "projection_nan":
        projections[1, 3, 4] = np.nan
    elif bad == "projection_overflow":
        projections = projections.astype(np.float64)
        projections[1, 3, 4] = 1e300
    elif bad == "embedding_count":
        tables.pop()
    elif bad == "text_shape":
        tables[0] = tables[0][:-1]
    elif bad == "audio_shape":
        tables[1] = tables[1][:, :-1]
    elif bad == "embedding_nan":
        tables[1][2, 3] = np.nan
    else:
        tables[1] = tables[1].astype(np.float64)
        tables[1][2, 3] = 1e300
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        DepthInputImage(spec, cfg).build(projections, tuple(tables))


@pytest.mark.parametrize("changes", [{"hidden": True}, {"depth": 0}, {"codebooks": 1.5},
                                      {"text_vocab": 0}, {"audio_vocab": True},
                                      {"multi_linear": 1}, {"hidden": 1 << 24},
                                      {"text_vocab": (1 << 24) - 1, "codebooks": 2, "audio_vocab": 2}])
def test_invalid_spec_and_inexact_aggregate_row_ids_are_rejected(changes):
    options = dict(hidden=32, depth=32, codebooks=16, text_vocab=9, audio_vocab=5,
                   multi_linear=True)
    options.update(changes)
    with pytest.raises(ValueError):
        DepthInputSpec(**options)


def test_isa_address_limit_and_misaligned_projection_dimensions_fail_before_payload_reads():
    spec, cfg, _, _, _ = fixture()
    with pytest.raises(ValueError):
        DepthInputImage(replace(spec, hidden=31), cfg)
    with pytest.raises(MemoryError):
        DepthInputImage(replace(spec, hidden=1 << 23), replace(cfg, DRAM_BYTES=1 << 40))


def test_backend_guards_and_missing_cap_gen_cleanup(monkeypatch):
    from opentpu.host import board
    spec, cfg, projections, embeddings, _ = fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("unqualified backend reached device or payload")

    monkeypatch.setattr(board, "open_transport", forbidden)
    monkeypatch.setattr(board, "BoardBackend", forbidden)
    with pytest.raises(ValueError, match="qualified backend"):
        DepthInputEngine(unread, unread, spec=spec, cfg=cfg, backend="board")
    with pytest.raises(ValueError, match="backend"):
        DepthInputEngine(unread, unread, spec=spec, cfg=cfg, backend="gpu")
    with pytest.raises(ValueError, match="measured.*config|config"):
        DepthInputEngine(unread, unread, spec=spec, backend=lambda c, i: None)
    closed = []

    class Unsupported:
        generates = False

        def close(self):
            closed.append(True)

    with pytest.raises(ValueError, match="CAP_GEN|RLD|indexed"):
        DepthInputEngine(projections, embeddings, spec=spec, cfg=cfg,
                         backend=lambda c, i: Unsupported())
    assert closed == [True]


def test_actual_boardbackend_protocol_small_vocab_and_d128_alignment():
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
    spec = DepthInputSpec(hidden=128, depth=128, codebooks=3, text_vocab=9, audio_vocab=5)
    rng = np.random.default_rng(485)
    projections = rng.normal(0, 0.1, (3, 128, 128)).astype(np.float32)
    embeddings = (rng.normal(size=(9, 128)).astype(np.float32),) + tuple(
        rng.normal(size=(5, 128)).astype(np.float32) for _ in range(2))
    inputs = rng.normal(size=(4, 128)).astype(np.float32)

    def backend(c, images):
        return BoardBackend(c, images, transport=transport, own_transport=True, status=False)

    with DepthInputEngine(projections, embeddings, spec=spec, cfg=cfg, backend=backend) as board:
        with DepthInputEngine(projections, embeddings, spec=spec, cfg=cfg) as isa:
            assert board.backend.engine is board
            assert board.image.cap == board.image.batch == 1 and board.image.kv_bytes == 0
            for hidden, (codebook, previous) in zip(inputs, ((2, 4), (0, 8), (1, -1), (2, 0))):
                board.step(hidden, codebook, previous)
                isa.step(hidden, codebook, previous)
                np.testing.assert_array_equal(board.read_output(), isa.read_output())
                np.testing.assert_array_equal(board.read_projected(), isa.read_projected())
            assert transport.sim_runs == 4
            assert transport.bytes_written >= board.image.nbytes and transport.bytes_read > 0
