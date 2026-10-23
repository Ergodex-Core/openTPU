"""Bounded audio-head projection and device-side codebook selection.

Execute on EC2 only. Synthetic fixtures need no checkpoint or real device.
Reference BLAS should use the runner's one-thread environment. IsaTransport
coverage exercises the actual BoardBackend protocol without claiming FPGA timing.
"""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import isa as I
from opentpu import quant as Q
from opentpu.compiler import CompileError
from opentpu.isasim import Config
from opentpu.llm import personaplex_audio_heads as A
from opentpu.llm.personaplex_audio_heads import AudioHeadEngine, AudioHeadImage


def fixture_bank(*, codebooks=16, vocab=67):
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=8192, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 23)
    rng = np.random.default_rng(921)
    bank = rng.normal(0, 0.13, (codebooks, vocab, 32)).astype(np.float32)
    hidden = rng.normal(size=(20, 32)).astype(np.float32)
    # A non-unit magnitude distinguishes the raw projection from extra RMSNorm.
    hidden *= np.linspace(0.4, 4.0, len(hidden), dtype=np.float32)[:, None]
    return cfg, bank, hidden


def engine(bank, cfg, **kwargs):
    c, v, h = bank.shape
    return AudioHeadEngine(bank, cfg=cfg, hidden=h, codebooks=c, vocab=v, **kwargs)


def reference(hidden, head, D):
    """Keep MX offsets and the accumulation window; never renormalize hidden."""
    activation = Q.fake_act(hidden, D)
    weights = Q.fake_w(head, "int8", D)
    offsets, window, width, transposed = weights.mx
    assert width == D and not transposed
    logits = Q.mx_matmul(activation, np.asarray(weights), offsets, window, width)
    return logits, int(np.argmax(logits))


def test_all_sixteen_heads_and_repeated_arbitrary_indices_match_quantized_reference():
    cfg, bank, inputs = fixture_bank()
    order = [15, 0, 8, 3, 14, 1, 7, 12, 2, 9, 4, 13, 5, 11, 6, 10, 15, 3, 0, 15]
    with engine(bank, cfg) as eng:
        for x, codebook in zip(inputs, order):
            expected_logits, expected_token = reference(x, bank[codebook], cfg.D)
            actual = eng.step(x, np.int64(codebook))
            assert type(actual) is int and actual == expected_token
            np.testing.assert_array_equal(eng.read_hidden(), x)
            np.testing.assert_allclose(eng.read_logits(), expected_logits,
                                       rtol=2e-5, atol=2e-5)
        assert eng.pos == len(order)
        assert eng.last_stats


def test_no_extra_normalization_changes_raw_projection_magnitude():
    cfg, bank, inputs = fixture_bank(codebooks=1)
    x = inputs[-1]
    expected, expected_token = reference(x, bank[0], cfg.D)
    normalized = x / np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-8)
    wrong, _ = reference(normalized, bank[0], cfg.D)
    assert np.linalg.norm(expected - wrong) > np.linalg.norm(expected) * 0.5
    with engine(bank, cfg) as eng:
        assert eng.step(x, 0) == expected_token
        np.testing.assert_allclose(eng.read_logits(), expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(eng.read_hidden(), x)


@pytest.mark.parametrize("positive", [False, True])
def test_single_column_ties_across_chunks_and_tail_choose_lowest_id(positive):
    cfg, _, _ = fixture_bank(codebooks=2)
    probe = AudioHeadImage(cfg, hidden=32, codebooks=2, vocab=1)
    vocab = probe.head_chunk + 3
    bank = np.zeros((2, vocab, 32), np.float32)
    x = np.zeros(32, np.float32)
    x[3] = 3
    if positive:
        for codebook, low in enumerate((7, 11)):
            bank[codebook, low, 3] = bank[codebook, probe.head_chunk + 1, 3] = 2
    with engine(bank, cfg) as eng:
        assert eng.image.head_chunk < vocab
        for codebook in (1, 0, 1):
            expected_logits, expected_token = reference(x, bank[codebook], cfg.D)
            assert expected_token == ((7, 11)[codebook] if positive else 0)
            assert eng.step(x, codebook) == expected_token
            logits = eng.read_logits()
            np.testing.assert_allclose(logits, expected_logits, rtol=2e-5, atol=2e-5)
            if positive:
                assert logits[expected_token] == logits[probe.head_chunk + 1]


def test_device_selects_bank_with_one_program_and_only_hidden_index_token_transfers(monkeypatch):
    cfg, bank, inputs = fixture_bank()
    order = (15, 0, 9, 15)
    expected = [reference(x, bank[codebook], cfg.D)[1]
                for x, codebook in zip(inputs, order)]
    with engine(bank, cfg) as eng:
        # Construction consumed the bank. Runtime must use its resident copy,
        # including when the host source becomes unusable between requests.
        bank.fill(np.nan)
        read, write, run = eng.backend.read, eng.backend.write, eng.backend.run
        reads, writes, runs = [], [], []

        def traced_read(s, addr, nbytes):
            reads.append((s, addr, nbytes))
            return read(s, addr, nbytes)

        def traced_write(s, addr, value):
            writes.append((s, addr, np.asarray(value).copy()))
            return write(s, addr, value)

        def traced_run(programs, *args, **kwargs):
            runs.append(programs)
            assert not args and kwargs.get("args") is None
            return run(programs, *args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("step attempted a diagnostic transfer or host head quantization")

        monkeypatch.setattr(eng.backend, "read", traced_read)
        monkeypatch.setattr(eng.backend, "write", traced_write)
        monkeypatch.setattr(eng.backend, "run", traced_run)
        monkeypatch.setattr(eng, "read_hidden", forbidden)
        monkeypatch.setattr(eng, "read_logits", forbidden)
        monkeypatch.setattr(Q, "fake_w", forbidden)
        monkeypatch.setattr(Q, "fake_act", forbidden)
        for x, codebook, token in zip(inputs, order, expected):
            actual = eng.step(x, codebook)
            assert type(actual) is int and actual == token
        assert reads == [(0, eng.image.io["token_id"], 4)] * 4
        assert len(writes) == 8 and len(runs) == 4
        assert all(program is eng.program for program in runs)
        for i, codebook in enumerate(order):
            pair = {addr: (s, value) for s, addr, value in writes[2 * i:2 * i + 2]}
            assert set(pair) == {eng.image.io["hidden"], eng.image.io["codebook"]}
            s, x = pair[eng.image.io["hidden"]]
            assert s == 0 and x.dtype == np.float32 and x.nbytes == 32 * 4
            np.testing.assert_array_equal(x.reshape(-1), inputs[i])
            s, cb = pair[eng.image.io["codebook"]]
            assert s == 0 and cb.dtype == np.float32 and cb.nbytes == 4
            assert cb.reshape(-1)[0] == codebook
        # RLD MUL derives separate resident matrix/scale offsets from the scalar.
        mul_loads = [ins for ins in eng.program[0]
                     if ins.op == I.RLD and ins.flags & I.F_MUL]
        assert len(mul_loads) >= 2


@pytest.mark.parametrize("invalid", [-1, 16, True, np.bool_(False), 1.0, 1.5])
def test_bad_codebook_does_not_advance_or_touch_backend(invalid, monkeypatch):
    cfg, bank, inputs = fixture_bank()
    with engine(bank, cfg) as eng:
        def forbidden(*args, **kwargs):
            raise AssertionError("invalid codebook reached the backend")

        monkeypatch.setattr(eng.backend, "write", forbidden)
        monkeypatch.setattr(eng.backend, "run", forbidden)
        with pytest.raises(ValueError, match="codebook"):
            eng.step(inputs[0], invalid)
        assert eng.pos == 0


@pytest.mark.parametrize("bad", ["shape", "nan", "infinity", "overflow"])
def test_bad_hidden_is_rejected_before_backend_and_does_not_poison_engine(bad, monkeypatch):
    cfg, bank, inputs = fixture_bank(codebooks=1)
    x = (inputs[:1] if bad == "shape" else
         np.full(32, {"nan": np.nan, "infinity": np.inf, "overflow": 1e300}[bad]))
    with engine(bank, cfg) as eng:
        write = eng.backend.write

        def forbidden(*args, **kwargs):
            raise AssertionError("invalid hidden reached the backend")

        monkeypatch.setattr(eng.backend, "write", forbidden)
        with np.errstate(over="ignore"), pytest.raises(ValueError, match="hidden"):
            eng.step(x, 0)
        assert eng.pos == 0
        monkeypatch.setattr(eng.backend, "write", write)
        assert type(eng.step(inputs[0], 0)) is int


def test_run_failure_latches_until_reset_and_reset_repeats_exactly(monkeypatch):
    cfg, bank, inputs = fixture_bank(codebooks=2)
    with engine(bank, cfg) as eng:
        initial = [eng.step(x, c) for x, c in zip(inputs[:3], (1, 0, 1))]
        hidden, logits = eng.read_hidden(), eng.read_logits()
        eng.reset()
        assert eng.pos == 0 and eng.last_stats == {}
        assert [eng.step(x, c) for x, c in zip(inputs[:3], (1, 0, 1))] == initial
        np.testing.assert_array_equal(eng.read_hidden(), hidden)
        np.testing.assert_array_equal(eng.read_logits(), logits)
        run = eng.backend.run

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic audio head failure")

        monkeypatch.setattr(eng.backend, "run", fail)
        with pytest.raises(RuntimeError, match="synthetic audio head failure"):
            eng.step(inputs[3], 0)
        assert eng.pos == 3
        monkeypatch.setattr(eng.backend, "run", run)
        with pytest.raises(RuntimeError, match="previous.*failed|reset"):
            eng.step(inputs[3], 0)
        eng.reset()
        assert eng.step(inputs[0], 1) == initial[0]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.0, 1.5, 67.0])
def test_bad_backend_token_requires_reset(invalid, monkeypatch):
    cfg, bank, inputs = fixture_bank(codebooks=1)
    with engine(bank, cfg) as eng:
        read = eng.backend.read

        def corrupt(s, addr, nbytes):
            if addr == eng.image.io["token_id"]:
                return np.array([invalid], np.float32).view(np.uint8)
            return read(s, addr, nbytes)

        monkeypatch.setattr(eng.backend, "read", corrupt)
        with pytest.raises(RuntimeError, match="token"):
            eng.step(inputs[0], 0)
        assert eng.pos == 0
        monkeypatch.setattr(eng.backend, "read", read)
        with pytest.raises(RuntimeError, match="previous.*failed|reset"):
            eng.step(inputs[0], 0)
        eng.reset()
        assert type(eng.step(inputs[0], 0)) is int


def test_diagnostics_require_successful_step_and_closed_operations_are_rejected(monkeypatch):
    cfg, bank, inputs = fixture_bank(codebooks=1)
    eng = engine(bank, cfg)
    for read in (eng.read_hidden, eng.read_logits):
        with pytest.raises(RuntimeError):
            read()
    eng.step(inputs[0], 0)
    eng.reset()
    for read in (eng.read_hidden, eng.read_logits):
        with pytest.raises(RuntimeError):
            read()
    close = getattr(eng.backend, "close", lambda: None)
    calls = []

    def traced_close():
        calls.append(True)
        close()

    monkeypatch.setattr(eng.backend, "close", traced_close, raising=False)
    eng.close()
    eng.close()
    assert calls == [True]
    for action in (lambda: eng.step(inputs[0], 0), eng.reset, eng.read_hidden, eng.read_logits):
        with pytest.raises(RuntimeError, match="closed"):
            action()


def unread_bank():
    raise AssertionError("audio head bank materialized before preflight")


def test_memory_and_compile_preflight_precede_lazy_bank_read(monkeypatch):
    cfg, _, _ = fixture_bank()
    image = AudioHeadImage(cfg, hidden=32, codebooks=16, vocab=67)
    with pytest.raises(MemoryError):
        AudioHeadEngine(unread_bank, cfg=replace(cfg, DRAM_BYTES=image.required_bytes - 1),
                        hidden=32, codebooks=16, vocab=67)
    with pytest.raises((CompileError, MemoryError)):
        AudioHeadEngine(unread_bank, cfg=replace(cfg, IMEM_WORDS=8),
                        hidden=32, codebooks=16, vocab=67)

    def fail(self):
        raise CompileError("synthetic audio head instruction capacity failure")

    monkeypatch.setattr(AudioHeadImage, "compile", fail)
    with pytest.raises(CompileError, match="instruction capacity"):
        AudioHeadEngine(unread_bank, cfg=cfg, hidden=32, codebooks=16, vocab=67)


def test_lazy_bank_is_loaded_once_after_compile_and_reset_does_not_reload(monkeypatch):
    cfg, bank, inputs = fixture_bank(codebooks=2)
    events = []
    compile_program = AudioHeadImage.compile

    def traced_compile(self):
        result = compile_program(self)
        events.append("compile")
        return result

    def load():
        assert events == ["compile"]
        events.append("bank")
        return bank

    monkeypatch.setattr(AudioHeadImage, "compile", traced_compile)
    with AudioHeadEngine(load, cfg=cfg, hidden=32, codebooks=2, vocab=67) as eng:
        eng.step(inputs[0], 1)
        eng.reset()
        eng.step(inputs[0], 0)
    assert events == ["compile", "bank"]


@pytest.mark.parametrize("bad", ["shape", "nan", "overflow"])
def test_bad_head_bank_shape_or_values_are_rejected(bad):
    cfg, bank, _ = fixture_bank(codebooks=2)
    if bad == "shape":
        bank = bank[:, :, :-1]
    elif bad == "nan":
        bank[1, 4, 5] = np.nan
    else:
        bank = bank.astype(np.float64)
        bank[1, 4, 5] = 1e300
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="head_bank|head bank"):
        AudioHeadImage(cfg, hidden=32, codebooks=2, vocab=67, head_bank=bank).build()


def test_layout_and_dimension_overflow_guards_do_not_read_bank():
    cfg, _, _ = fixture_bank()
    for kwargs in ({"hidden": 31}, {"hidden": True}, {"codebooks": 0},
                   {"codebooks": True}, {"vocab": 1.5}, {"vocab": (1 << 24) + 1}):
        options = {"hidden": 32, "codebooks": 16, "vocab": 67, **kwargs}
        with pytest.raises(ValueError):
            AudioHeadImage(cfg, head_bank=unread_bank, **options)
    with pytest.raises((MemoryError, ValueError)):
        AudioHeadImage(replace(cfg, DRAM_BYTES=1 << 40), hidden=32,
                       codebooks=1 << 24, vocab=67, head_bank=unread_bank)


def test_image_contains_disjoint_hidden_index_logits_and_every_int8_bank():
    cfg, bank, _ = fixture_bank(codebooks=2)
    image = AudioHeadImage(cfg, hidden=32, codebooks=2, vocab=67, head_bank=bank)
    data, = image.build()
    qaddr, saddr = image.head_offsets
    assert image.data_stride >= 67 * Q.row_bytes(32, "int8", cfg.D)
    assert image.scale_stride >= 67 * Q.scale_bytes(32, "int8", cfg.D)
    regions = [(image.io["hidden"], 4 * 32), (image.io["codebook"], 4),
               (image.io["logits"], 4 * 67), (image.io["token_id"], 4),
               (qaddr, 2 * image.data_stride), (saddr, 2 * image.scale_stride)]
    for i, (addr, size) in enumerate(regions):
        assert 0 <= addr < addr + size <= image.nbytes <= image.required_bytes
        for other, width in regions[i + 1:]:
            assert addr + size <= other or other + width <= addr
    for codebook in range(2):
        packed = Q.quantize_mxu(bank[codebook], "int8", cfg.D)
        for addr, stride, expected in zip(image.head_offsets,
                                          (image.data_stride, image.scale_stride), packed):
            raw = np.ascontiguousarray(expected).view(np.uint8).reshape(-1)
            start = addr + codebook * stride
            np.testing.assert_array_equal(data[start:start + raw.size], raw)


def test_backend_qualification_rejects_direct_board_and_missing_cap_gen(monkeypatch):
    from opentpu.host import board

    cfg, bank, _ = fixture_bank(codebooks=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("unqualified backend opened a board or read weights")

    monkeypatch.setattr(board, "BoardBackend", forbidden)
    monkeypatch.setattr(board, "open_transport", forbidden)
    with pytest.raises(ValueError, match="qualified backend factory"):
        AudioHeadEngine(unread_bank, cfg=cfg, hidden=32, codebooks=1, vocab=67,
                        backend="board")
    with pytest.raises(ValueError, match="backend"):
        AudioHeadEngine(unread_bank, cfg=cfg, hidden=32, codebooks=1, vocab=67,
                        backend="gpu")
    with pytest.raises(ValueError, match="measured.*config|config"):
        AudioHeadEngine(unread_bank, hidden=32, codebooks=1, vocab=67,
                        backend=lambda c, images: None)
    closed = []

    class UnsupportedBackend:
        generates = False

        def close(self):
            closed.append(True)

    with pytest.raises(ValueError, match="ARGMAX|CAP_GEN|RLD|generate"):
        engine(bank, cfg, backend=lambda c, images: UnsupportedBackend())
    assert closed == [True]


def test_actual_boardbackend_isatransport_matches_isa_for_nonsequential_codebooks():
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
    rng = np.random.default_rng(733)
    bank = rng.normal(0, 0.1, (3, 67, 128)).astype(np.float32)

    def backend(c, images):
        return BoardBackend(c, images, transport=transport, own_transport=True, status=False)

    with engine(bank, cfg, backend=backend) as board:
        assert board.image.head_offsets[0] % cfg.D == 0
        with engine(bank, cfg) as isa:
            assert board.backend.engine is board
            for codebook in (2, 0, 2, 1):
                x = rng.normal(size=128).astype(np.float32)
                assert board.step(x, codebook) == isa.step(x, codebook)
                np.testing.assert_array_equal(board.read_hidden(), isa.read_hidden())
                np.testing.assert_array_equal(board.read_logits(), isa.read_logits())
            assert transport.sim_runs == 4
            assert transport.bytes_written >= board.image.nbytes
            assert transport.bytes_read > 0
