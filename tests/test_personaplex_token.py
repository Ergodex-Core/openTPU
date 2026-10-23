"""Resident PersonaPlex text head, greedy selection, and host/device boundary.

Run these tests on EC2, as required by the PersonaPlex development handoff.
The fixtures are synthetic and small; neither checkpoints nor hardware are
needed. IsaTransport exercises BoardBackend's real protocol without an FPGA.
"""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import quant as Q
from opentpu.compiler import CompileError
from opentpu.isasim import Config
from opentpu.llm import personaplex_token as T
from opentpu.llm.personaplex_temporal import Engine, Spec
from opentpu.llm.personaplex_token import TokenEngine, TokenImage


def fixture_model(*, context=64, vocab=67):
    """D=32 makes frame 33 a bucket boundary without a long ISA regression."""
    spec = Spec(hidden=32, layers=1, n_heads=1, ffn=64, context=context)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 23)
    rng = np.random.default_rng(791)
    shapes = {"wq": (32, 32), "wk": (32, 32), "wv": (32, 32),
              "wo": (32, 32), "wg": (64, 32), "wu": (64, 32),
              "wd": (32, 64)}
    layer = {name: rng.normal(0, 0.14 if name in ("wq", "wk") else 0.08,
                              shape).astype(np.float32)
             for name, shape in shapes.items()}
    layer.update(norm1=rng.uniform(0.6, 1.4, 32).astype(np.float32),
                 norm2=rng.uniform(0.6, 1.4, 32).astype(np.float32))
    # Nonuniform final gain makes accidentally applying RMSNorm twice visible.
    norm = rng.uniform(0.4, 1.7, 32).astype(np.float32)
    head = rng.normal(0, 0.13, (vocab, 32)).astype(np.float32)
    inputs = rng.normal(0, 1, (34, 32)).astype(np.float32)
    return spec, cfg, [layer], norm, head, inputs


def head_reference(hidden, head, D):
    """Independent head math, retaining MX activation/weight offset metadata.

    A plain NumPy dot of dequantized arrays loses the MXU window-drop rule.
    The temporal Engine has already applied the final RMSNorm; do not repeat it.
    """
    activation = Q.fake_act(hidden, D)
    weight = Q.fake_w(head, "int8", D)
    offsets, window, width, transposed = weight.mx
    assert not transposed and width == D
    logits = Q.mx_matmul(activation, np.asarray(weight), offsets, window, width)
    return logits, int(np.argmax(logits))  # NumPy resolves ties to the lowest ID.


@pytest.mark.parametrize("wformat", ["int4", "int8"])
def test_multiframe_tokens_logits_and_hidden_match_quantized_head_reference(wformat):
    spec, cfg, weights, norm, head, inputs = fixture_model()
    with Engine(spec, weights, norm, cfg=cfg, cap=34, wformat=wformat) as temporal:
        expected_hidden = np.stack([temporal.step(x) for x in inputs])
    with TokenEngine(spec, weights, norm, head, vocab=len(head), cfg=cfg,
                     cap=34, wformat=wformat) as token:
        for x, hidden in zip(inputs, expected_hidden):
            expected_logits, expected_token = head_reference(hidden, head, cfg.D)
            actual = token.step(x)
            assert type(actual) is int and actual == expected_token
            # The head must consume and preserve the normalized temporal vector.
            np.testing.assert_array_equal(token.read_hidden(), hidden)
            np.testing.assert_allclose(token.read_logits(), expected_logits,
                                       rtol=2e-5, atol=2e-5)
        assert token.pos == 34
        assert len(token._programs) == 2
        assert token.image.head_format == "int8"
        assert token.last_stats["frame"] == 33


def test_step_uses_resident_rope_and_only_embedding_and_token_cross_host_boundary(monkeypatch):
    spec, cfg, weights, norm, head, inputs = fixture_model()
    with TokenEngine(spec, weights, norm, head, vocab=len(head), cfg=cfg,
                     cap=4) as eng:
        backend = eng.backend
        read, write, run = backend.read, backend.write, backend.run
        reads, writes, programs = [], [], []

        def traced_read(s, addr, nbytes):
            reads.append((s, addr, nbytes))
            return read(s, addr, nbytes)

        def traced_write(s, addr, value):
            writes.append((s, addr, np.asarray(value).nbytes))
            return write(s, addr, value)

        def traced_run(program, *args, **kwargs):
            programs.append(program)
            return run(program, *args, **kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("step used per-frame host RoPE or a diagnostic tensor read")

        monkeypatch.setattr(backend, "read", traced_read)
        monkeypatch.setattr(backend, "write", traced_write)
        monkeypatch.setattr(backend, "run", traced_run)
        monkeypatch.setattr(T.G, "rope_tables", forbidden)
        monkeypatch.setattr(eng, "read_hidden", forbidden)
        monkeypatch.setattr(eng, "read_logits", forbidden)
        for x in inputs[:4]:
            assert type(eng.step(x)) is int
        assert writes == [(0, eng.image.io["x"], 4 * spec.hidden)] * 4
        assert reads == [(0, eng.image.io["token_id"], 4)] * 4
        # Both programs execute on the attached backend; the vocabulary head
        # cannot silently fall back to a host dot/argmax or a second backend.
        assert len(programs) == 8
        assert programs[0] is not programs[1]
        assert all(programs[i] is programs[i % 2] for i in range(8))


def test_reset_is_exact_and_old_kv_is_masked():
    spec, cfg, weights, norm, head, inputs = fixture_model()
    with TokenEngine(spec, weights, norm, head, vocab=len(head), cfg=cfg,
                     cap=4) as eng:
        first = [eng.step(x) for x in inputs[:4]]
        first_hidden, first_logits = eng.read_hidden(), eng.read_logits()
        with pytest.raises(RuntimeError, match="context is full"):
            eng.step(inputs[4])
        assert eng.pos == 4
        eng.reset()
        assert eng.pos == 0 and eng.last_stats == {}
        assert [eng.step(x) for x in inputs[:4]] == first
        np.testing.assert_array_equal(eng.read_hidden(), first_hidden)
        np.testing.assert_array_equal(eng.read_logits(), first_logits)
        eng.reset()
        altered = eng.step(inputs[9])
        with TokenEngine(spec, weights, norm, head, vocab=len(head), cfg=cfg,
                         cap=4) as fresh:
            assert altered == fresh.step(inputs[9])
            np.testing.assert_array_equal(eng.read_hidden(), fresh.read_hidden())
            np.testing.assert_array_equal(eng.read_logits(), fresh.read_logits())
        assert eng.pos == 1
    with pytest.raises(RuntimeError, match="closed"):
        eng.step(inputs[0])
    with pytest.raises(RuntimeError, match="closed"):
        eng.reset()


def test_lowest_token_wins_ties_across_head_chunks_and_padded_tail():
    spec, cfg, weights, norm, _, inputs = fixture_model()
    probe = TokenImage(spec, cfg, cap=1, vocab=1)
    vocab = probe.head_chunk + 3
    head = np.zeros((vocab, spec.hidden), np.float32)
    with TokenEngine(spec, weights, norm, head, vocab=vocab, cfg=cfg,
                     cap=1) as eng:
        assert eng.image.head_chunk < vocab
        # All real and padding rows tie at zero. The first real ID must win.
        assert eng.step(inputs[0]) == 0
        np.testing.assert_array_equal(eng.read_logits(), np.zeros(vocab, np.float32))


def test_nonzero_tie_across_head_chunks_chooses_lowest_id():
    spec, cfg, weights, norm, _, inputs = fixture_model()
    probe = TokenImage(spec, cfg, cap=1, vocab=1)
    vocab = probe.head_chunk + 3
    with Engine(spec, weights, norm, cfg=cfg, cap=1) as temporal:
        hidden = temporal.step(inputs[0])
    head = np.zeros((vocab, spec.hidden), np.float32)
    low, high = 7, probe.head_chunk + 1
    # One nonzero column gives an exact tie, independent of a BLAS kernel's
    # reduction ordering for duplicated dense rows in different output chunks.
    column = int(np.argmax(np.abs(hidden)))
    head[low, column] = head[high, column] = np.sign(hidden[column])
    expected_logits, expected_token = head_reference(hidden, head, cfg.D)
    assert expected_token == low and expected_logits[low] > 0
    assert expected_logits[low] == expected_logits[high]
    with TokenEngine(spec, weights, norm, head, vocab=vocab, cfg=cfg,
                     cap=1) as eng:
        assert eng.step(inputs[0]) == low


def test_bad_embeddings_do_not_run_backend_or_advance_state(monkeypatch):
    spec, cfg, weights, norm, head, inputs = fixture_model()
    with TokenEngine(spec, weights, norm, head, vocab=len(head), cfg=cfg,
                     cap=2) as eng:
        run = eng.backend.run

        def forbidden(*args, **kwargs):
            raise AssertionError("invalid embedding reached the backend")

        monkeypatch.setattr(eng.backend, "run", forbidden)
        with pytest.raises(ValueError, match="embedding: shape"):
            eng.step(inputs[:1])
        for value in (np.nan, np.inf, -np.inf):
            with pytest.raises(ValueError, match="non-finite"):
                eng.step(np.full(spec.hidden, value, np.float32))
        assert eng.pos == 0
        monkeypatch.setattr(eng.backend, "run", run)
        assert type(eng.step(inputs[0])) is int and eng.pos == 1


@pytest.mark.parametrize("phase", ["temporal", "head"])
def test_either_program_failure_requires_reset(phase, monkeypatch):
    spec, cfg, weights, norm, head, inputs = fixture_model()
    with TokenEngine(spec, weights, norm, head, vocab=len(head), cfg=cfg,
                     cap=2) as eng:
        run, calls = eng.backend.run, []

        def fail(program, *args, **kwargs):
            calls.append(program)
            if len(calls) == (1 if phase == "temporal" else 2):
                raise RuntimeError("synthetic " + phase + " failure")
            return run(program, *args, **kwargs)

        monkeypatch.setattr(eng.backend, "run", fail)
        with pytest.raises(RuntimeError, match="synthetic " + phase + " failure"):
            eng.step(inputs[0])
        assert eng.pos == 0
        monkeypatch.setattr(eng.backend, "run", run)
        with pytest.raises(RuntimeError, match="previous frame failed"):
            eng.step(inputs[0])
        eng.reset()
        with TokenEngine(spec, weights, norm, head, vocab=len(head), cfg=cfg,
                         cap=2) as fresh:
            assert eng.step(inputs[0]) == fresh.step(inputs[0])
            np.testing.assert_array_equal(eng.read_hidden(), fresh.read_hidden())


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.0, 1.5, 67.0])
def test_invalid_backend_token_latches_failed_state(invalid, monkeypatch):
    spec, cfg, weights, norm, head, inputs = fixture_model()
    with TokenEngine(spec, weights, norm, head, vocab=len(head), cfg=cfg,
                     cap=1) as eng:
        read = eng.backend.read

        def corrupt(s, addr, nbytes):
            if addr == eng.image.io["token_id"]:
                return np.array([invalid], np.float32).view(np.uint8)
            return read(s, addr, nbytes)

        monkeypatch.setattr(eng.backend, "read", corrupt)
        with pytest.raises(RuntimeError, match="token"):
            eng.step(inputs[0])
        assert eng.pos == 0
        with pytest.raises(RuntimeError, match="previous frame failed"):
            eng.step(inputs[0])


class UnreadWeights:
    """Capacity/compile validation must complete before any model materializes."""

    def __len__(self):
        raise AssertionError("layer weights were read before preflight completed")

    def __getitem__(self, key):
        raise AssertionError("layer weights were read before preflight completed")


class UnreadNorm:
    def __array__(self, *args, **kwargs):
        raise AssertionError("final norm was read before preflight completed")


def unread_head():
    raise AssertionError("text head was read before preflight completed")


def test_memory_capacity_fails_before_any_weight_or_head_read():
    spec, cfg, _, _, _, _ = fixture_model()
    image = TokenImage(spec, cfg, cap=34, vocab=67)
    with pytest.raises(MemoryError):
        TokenEngine(spec, UnreadWeights(), UnreadNorm(), unread_head,
                    cfg=replace(cfg, DRAM_BYTES=image.required_bytes - 1),
                    cap=34, vocab=67)


def test_instruction_capacity_fails_before_any_weight_or_head_read():
    spec, cfg, _, _, _, _ = fixture_model()
    with pytest.raises((CompileError, MemoryError)):
        TokenEngine(spec, UnreadWeights(), UnreadNorm(), unread_head,
                    cfg=replace(cfg, IMEM_WORDS=8), cap=34, vocab=67)


def test_later_bucket_is_preflighted_before_any_weight_or_head_read(monkeypatch):
    spec, cfg, _, _, _, _ = fixture_model()
    calls = []
    compile_bucket = TokenImage.compile_bucket

    def fail_second(self, bucket):
        calls.append(bucket)
        if bucket == 2:
            raise CompileError("synthetic second bucket capacity failure")
        return compile_bucket(self, bucket)

    monkeypatch.setattr(TokenImage, "compile_bucket", fail_second)
    with pytest.raises(CompileError, match="second bucket"):
        TokenEngine(spec, UnreadWeights(), UnreadNorm(), unread_head,
                    cfg=cfg, cap=34, vocab=67)
    assert calls == [1, 2]


def test_head_program_is_preflighted_before_any_weight_or_head_read(monkeypatch):
    spec, cfg, _, _, _, _ = fixture_model()

    def fail(self):
        raise CompileError("synthetic text head capacity failure")

    monkeypatch.setattr(TokenImage, "compile_head", fail)
    with pytest.raises(CompileError, match="text head capacity"):
        TokenEngine(spec, UnreadWeights(), UnreadNorm(), unread_head,
                    cfg=cfg, cap=1, vocab=67)


def test_head_callable_is_materialized_once_after_all_programs_compile(monkeypatch):
    spec, cfg, weights, norm, head, inputs = fixture_model()
    events = []
    compile_bucket = TokenImage.compile_bucket
    compile_head = TokenImage.compile_head

    def traced_compile(self, bucket):
        result = compile_bucket(self, bucket)
        events.append(("bucket", bucket))
        return result

    def traced_head(self):
        result = compile_head(self)
        events.append(("head_program", None))
        return result

    def lazy_head():
        events.append(("head", None))
        assert ("bucket", 1) in events and ("bucket", 2) in events
        assert ("head_program", None) in events
        return head

    monkeypatch.setattr(TokenImage, "compile_bucket", traced_compile)
    monkeypatch.setattr(TokenImage, "compile_head", traced_head)
    with TokenEngine(spec, weights, norm, lazy_head, vocab=len(head),
                     cfg=cfg, cap=34) as eng:
        assert events[-1] == ("head", None)
        for x in inputs[:4]:
            assert type(eng.step(x)) is int
        eng.reset()
        assert type(eng.step(inputs[0])) is int
        assert events.count(("head", None)) == 1
        assert events.count(("head_program", None)) == 1


@pytest.mark.parametrize("vocab", [0, -1, True, 1.5, (1 << 24) + 1])
def test_invalid_vocab_is_rejected_before_reading_weights(vocab):
    spec, cfg, _, _, _, _ = fixture_model()
    with pytest.raises(ValueError, match="vocab"):
        TokenEngine(spec, UnreadWeights(), UnreadNorm(), unread_head,
                    cfg=cfg, cap=1, vocab=vocab)


@pytest.mark.parametrize("cap", [0, True, 65])
def test_invalid_context_is_rejected_before_reading_weights(cap):
    spec, cfg, _, _, _, _ = fixture_model()
    with pytest.raises(ValueError, match="cap must"):
        TokenEngine(spec, UnreadWeights(), UnreadNorm(), unread_head,
                    cfg=cfg, cap=cap, vocab=67)


@pytest.mark.parametrize("bad", ["shape", "nan", "infinity"])
def test_invalid_text_head_is_rejected(bad):
    spec, cfg, weights, norm, head, _ = fixture_model()
    if bad == "shape":
        head = head[:, :-1]
    else:
        head[3, 1] = np.nan if bad == "nan" else np.inf
    with pytest.raises(ValueError, match="text_head.*" + ("shape" if bad == "shape" else "non-finite")):
        TokenEngine(spec, weights, norm, head, vocab=67, cfg=cfg, cap=1)


def test_backend_arguments_are_validated_before_weights_are_read():
    spec, cfg, _, _, _, _ = fixture_model()
    with pytest.raises(ValueError, match="backend must be"):
        TokenEngine(spec, UnreadWeights(), UnreadNorm(), unread_head,
                    cfg=cfg, cap=1, vocab=67, backend="gpu")
    with pytest.raises(ValueError, match="transport is only valid"):
        TokenEngine(spec, UnreadWeights(), UnreadNorm(), unread_head,
                    cfg=cfg, cap=1, vocab=67, transport=object())
    with pytest.raises(ValueError, match="qualified backend factory"):
        TokenEngine(spec, UnreadWeights(), UnreadNorm(), unread_head,
                    cap=1, vocab=67, backend="board")
    with pytest.raises(ValueError, match="measured device config"):
        TokenEngine(spec, UnreadWeights(), UnreadNorm(), unread_head,
                    cap=1, vocab=67, backend=lambda c, images: None)


def test_direct_board_path_is_rejected_before_layout_build_or_transport(monkeypatch):
    from opentpu.host import board

    spec, cfg, _, _, _, _ = fixture_model()

    def forbidden(*args, **kwargs):
        raise AssertionError("unqualified direct board path reached layout, weights, or transport")

    monkeypatch.setattr(TokenImage, "build", forbidden)
    monkeypatch.setattr(T, "TokenImage", forbidden)
    monkeypatch.setattr(board, "BoardBackend", forbidden)
    monkeypatch.setattr(board, "open_transport", forbidden)
    with pytest.raises(ValueError, match="qualified backend factory"):
        TokenEngine(spec, UnreadWeights(), UnreadNorm(), unread_head,
                    cfg=cfg, cap=1, vocab=67, backend="board", transport=object())


def test_backend_without_generate_instructions_is_closed_and_rejected():
    spec, cfg, weights, norm, head, _ = fixture_model()
    closed, attached = [], []

    class UnsupportedBackend:
        args = True
        generates = False

        def attach(self, engine):
            attached.append(engine)

        def close(self):
            closed.append(True)

    with pytest.raises(ValueError, match="ARGMAX|generate|generates|CAP_GEN"):
        TokenEngine(spec, weights, norm, head, vocab=len(head), cfg=cfg,
                    cap=1, backend=lambda c, images: UnsupportedBackend())
    assert closed == [True] and len(attached) == 1
    attached[0].close()
    attached[0].close()
    assert closed == [True]


@pytest.mark.parametrize("bad", ["layer_shape", "layer_nan", "norm_shape", "norm_nan"])
def test_inherited_layer_and_norm_guards_apply_to_token_image(bad):
    spec, cfg, weights, norm, head, _ = fixture_model()
    if bad == "layer_shape":
        weights[0]["wq"] = weights[0]["wq"][:, :-1]
    elif bad == "layer_nan":
        weights[0]["wk"][0, 0] = np.nan
    elif bad == "norm_shape":
        norm = norm[:-1]
    else:
        norm[0] = np.nan
    with pytest.raises(ValueError, match="shape" if bad.endswith("shape") else "non-finite"):
        TokenImage(spec, cfg, cap=1, vocab=len(head), text_head=head).build(weights, norm)


def test_layout_accounts_for_resident_rope_and_head_without_materializing_weights():
    spec, cfg, _, _, _, _ = fixture_model()
    image = TokenImage(spec, cfg, cap=34, vocab=67)
    qaddr, saddr = image.text_head_offsets
    ranges = [(qaddr, 67 * Q.row_bytes(spec.hidden, "int8", cfg.D)),
              (saddr, 67 * Q.scale_bytes(spec.hidden, "int8", cfg.D)),
              (image.cos_table, image.cap * spec.head_dim // 2 * 4),
              (image.sin_table, image.cap * spec.head_dim // 2 * 4),
              (image.io["token_logits"], 67 * 4),
              (image.io["token_id"], 4)]
    assert image.required_bytes >= image.nbytes
    for addr, nbytes in ranges:
        assert 0 <= addr and nbytes > 0 and addr + nbytes <= image.nbytes
    for i, (a, n) in enumerate(ranges):
        for b, m in ranges[i + 1:]:
            assert a + n <= b or b + m <= a


def test_image_upload_contains_all_resident_rope_rows_and_int8_head():
    spec, cfg, weights, norm, head, _ = fixture_model()
    image = TokenImage(spec, cfg, cap=4, vocab=len(head), text_head=head)
    data, = image.build(weights, norm)
    half = spec.head_dim // 2
    assert image.frame_limit == 4 and image.cap == cfg.D
    assert image.rope_bytes == 8 * image.cap * half
    descriptors = image.descriptors(0)
    assert descriptors.cos_t.shape == descriptors.sin_t.shape == (image.cap, half)
    # Include the padded physical tail advertised by RunPos's argument bound.
    for pos in range(image.cap):
        expected_cos, expected_sin = T.G.rope_tables(image.spec, pos)
        for addr, expected in ((image.cos_table, expected_cos),
                               (image.sin_table, expected_sin)):
            lo = addr + 4 * half * pos
            np.testing.assert_array_equal(data[lo:lo + 4 * half].view(np.float32), expected)
    q, scales = Q.quantize_mxu(head, "int8", cfg.D)
    for addr, expected in zip(image.text_head_offsets, (q, scales)):
        expected = np.ascontiguousarray(expected).view(np.uint8).reshape(-1)
        np.testing.assert_array_equal(data[addr:addr + expected.size], expected)


def test_actual_boardbackend_protocol_matches_isa_tokens_hidden_and_logits():
    """Simulated registers/DRAM only; this test makes no FPGA timing claim."""
    from opentpu.host.board import BoardBackend
    from opentpu.host.fake import _target_cfg
    from opentpu.host.isaboard import IsaTransport
    from opentpu.isasim import board_config
    from opentpu.target import arch_hash, current

    base = board_config()
    cfg = replace(base, DRAM_BYTES=1 << 24,
                  SEQ_WC=bool(current().arch.get("SEQ_WC", False)))
    regcfg = _target_cfg(cfg.D, cfg.MCOLS, cfg.LANES, cfg.PAIR, cfg.DSTEP,
                         cfg.STREAM, cfg.act_rows, cfg.SEQ_WC)
    regcfg.update(DRAM_BYTES=cfg.DRAM_BYTES, channel_bytes=cfg.DRAM_BYTES // 2,
                  hash=arch_hash(cfg))
    transport = IsaTransport(cfg=regcfg)
    transport.cfg = cfg
    spec = Spec(hidden=128, layers=1, n_heads=1, ffn=256, context=256)
    rng = np.random.default_rng(985)
    shapes = {"wq": (128, 128), "wk": (128, 128), "wv": (128, 128),
              "wo": (128, 128), "wg": (256, 128), "wu": (256, 128),
              "wd": (128, 256)}
    layer = {name: rng.normal(0, 0.07, shape).astype(np.float32)
             for name, shape in shapes.items()}
    layer.update(norm1=np.ones(128, np.float32), norm2=np.ones(128, np.float32))
    norm = np.linspace(0.6, 1.4, 128, dtype=np.float32)
    head = rng.normal(0, 0.1, (131, 128)).astype(np.float32)

    def backend(c, images):
        return BoardBackend(c, images, transport=transport, status=False,
                            own_transport=True)

    with TokenEngine(spec, [layer], norm, head, vocab=len(head), cfg=cfg,
                     backend=backend, cap=4) as board:
        with TokenEngine(spec, [layer], norm, head, vocab=len(head), cfg=cfg,
                         cap=4) as isa:
            assert board.backend.engine is board
            for _ in range(4):
                x = rng.normal(size=128).astype(np.float32)
                assert board.step(x) == isa.step(x)
                np.testing.assert_array_equal(board.read_hidden(), isa.read_hidden())
                np.testing.assert_array_equal(board.read_logits(), isa.read_logits())
            layout = board.backend._layout()
            assert layout["kv_used"] > 0
            assert layout["kv_capacity"] == board.image.kv_bytes
            assert transport.sim_runs == 8  # temporal and head are separate runs
            assert transport.bytes_written >= board.image.nbytes
            assert transport.bytes_read > 0
