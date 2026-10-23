"""Bounded depth-layer validation; run on EC2, never on the Mac or FPGA.

Float64 source math below is diagnostic validation only. Production inference
uploads one hidden vector and selects all weight banks with a device counter.
"""
from dataclasses import replace

import numpy as np
import pytest

from opentpu.compiler import CompileError
from opentpu.isasim import Config
from opentpu.llm.personaplex_depth_layer import DepthLayerEngine, DepthLayerImage, DepthLayerSpec


NAMES = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")


def fixture(*, steps=4, layers=1, policy="native", padded=False):
    hidden, width = (128, 128) if padded else (64, 32)
    spec = DepthLayerSpec(hidden=hidden, n_heads=2, ffn=2 * hidden,
                          steps=steps, layers=layers, cache_policy=policy)
    cfg = Config(S=1, D=width, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True,
                 DRAM_BYTES=1 << 25)
    rng = np.random.default_rng(763)
    payload = []
    for _ in range(layers):
        banks = []
        for _ in range(steps):
            shapes = {name: (hidden, hidden) for name in NAMES[:4]}
            shapes.update(wg=(spec.ffn, hidden), wu=(spec.ffn, hidden), wd=(hidden, spec.ffn))
            banks.append({name: rng.normal(0, 0.035, shape).astype(np.float32)
                          for name, shape in shapes.items()})
        payload.append({"norm1": rng.uniform(0.7, 1.3, hidden).astype(np.float32),
                        "norm2": rng.uniform(0.7, 1.3, hidden).astype(np.float32),
                        "steps": banks})
    inputs = rng.normal(size=(steps, hidden)).astype(np.float32)
    return spec, cfg, payload, inputs


def source_reference(spec, layers, inputs):
    """Independent native pre-RMS, causal attention and SiLU-gate reference.

    No RoPE, q/k norm, biases, layer scales, or final normalization. Native's
    final cache-position bug is deliberately distinct from causal diagnostics.
    """
    histories = [[] for _ in layers]
    outputs = []
    for step, hidden in enumerate(inputs):
        x = hidden.astype(np.float64)
        for index, layer in enumerate(layers):
            weights = {key: value.astype(np.float64) for key, value in layer["steps"][step].items()}
            u = x * (layer["norm1"].astype(np.float64) /
                     np.sqrt(np.mean(x * x) + spec.eps))
            q, k, v = [(weights[key] @ u).reshape(spec.n_heads, spec.head_dim)
                       for key in ("wq", "wk", "wv")]
            histories[index].append((k, v))
            first = 1 if spec.cache_policy == "native" and step == spec.steps - 1 else 0
            keys, values = zip(*histories[index][first:])
            keys, values = np.stack(keys), np.stack(values)
            scores = np.einsum("hd,thd->ht", q, keys) / np.sqrt(spec.head_dim)
            probs = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            attention = np.einsum("ht,thd->hd", probs, values).reshape(-1)
            x = x + weights["wo"] @ attention
            u = x * (layer["norm2"].astype(np.float64) /
                     np.sqrt(np.mean(x * x) + spec.eps))
            gate, up = weights["wg"] @ u, weights["wu"] @ u
            x = x + weights["wd"] @ (gate / (1 + np.exp(-gate)) * up)
        outputs.append(x)
    return np.stack(outputs)


def bits(value):
    return np.asarray(value, np.float32).view(np.uint32)


@pytest.mark.parametrize("policy,padded", [("native", False), ("causal", False), ("native", True)])
def test_four_frames_match_independent_source_without_rope_or_final_norm(policy, padded):
    spec, cfg, layers, inputs = fixture(policy=policy, padded=padded)
    expected = source_reference(spec, layers, inputs)
    with DepthLayerEngine(layers, spec=spec, cfg=cfg) as engine:
        for step, hidden in enumerate(inputs):
            assert engine.step(hidden) is None
            output = engine.read_output()
            assert np.isfinite(output).all()
            # INT8 weights plus activation/KV rounding: synthetic regression,
            # not a model-quality acceptance threshold or FP64 parity claim.
            relative = np.linalg.norm(output - expected[step]) / np.linalg.norm(expected[step])
            assert relative < 0.025
            assert engine.pos == engine.read_counter() == step + 1
            assert engine.poss == [step + 1] and engine.last_stats["step"] == step
        assert len(engine.program) == 1 and not engine.program[0].run_args


def test_native_final_step_drops_only_first_slot_causal_keeps_full_prefix():
    spec, cfg, layers, inputs = fixture(steps=16)
    inputs.fill(1)
    layers[0]["norm1"].fill(1)
    layers[0]["norm2"].fill(1)
    for step, bank in enumerate(layers[0]["steps"]):
        for value in bank.values():
            value.fill(0)
        bank["wv"][:] = np.eye(spec.hidden, dtype=np.float32) * (step + 1)
        bank["wo"][:] = np.eye(spec.hidden, dtype=np.float32)
    causal_spec = replace(spec, cache_policy="causal")
    with DepthLayerEngine(layers, spec=spec, cfg=cfg) as native:
        with DepthLayerEngine(layers, spec=causal_spec, cfg=cfg) as causal:
            for step, hidden in enumerate(inputs):
                native.step(hidden)
                causal.step(hidden)
                n, c = native.read_output(), causal.read_output()
                if step < spec.steps - 1:
                    np.testing.assert_array_equal(bits(n), bits(c))
                else:
                    assert np.mean(n - c) > 0.35
                assert native.read_counter() == causal.read_counter() == step + 1
            np.testing.assert_allclose(n, source_reference(spec, layers, inputs)[-1], atol=0.13, rtol=0)
            np.testing.assert_allclose(c, source_reference(causal_spec, layers, inputs)[-1], atol=0.13, rtol=0)
            # Distinct masks live in the image, not in host-selected programs.
            masks_n = native.image.build(layers)[0][native.image.masks_at:
                native.image.masks_at + 4 * spec.steps * native.image.cap].view(np.float32)
            masks_c = causal.image.build(layers)[0][causal.image.masks_at:
                causal.image.masks_at + 4 * spec.steps * causal.image.cap].view(np.float32)
            different = np.flatnonzero(masks_n != masks_c)
            np.testing.assert_array_equal(different, [(spec.steps - 1) * native.image.cap])
            with pytest.raises(RuntimeError, match="full"):
                native.step(inputs[0])


def test_two_layers_equal_separate_quantized_layer_composition_and_reset_exact():
    spec, cfg, layers, inputs = fixture(layers=2)
    single = replace(spec, layers=1)
    with DepthLayerEngine(layers, spec=spec, cfg=cfg) as combined:
        with DepthLayerEngine(layers[:1], spec=single, cfg=cfg) as first:
            with DepthLayerEngine(layers[1:], spec=single, cfg=cfg) as second:
                outputs = []
                for hidden in inputs:
                    combined.step(hidden)
                    first.step(hidden)
                    second.step(first.read_output())
                    outputs.append(combined.read_output())
                    np.testing.assert_array_equal(bits(outputs[-1]), bits(second.read_output()))
                caches = [combined.read_cache(i) for i in range(spec.layers)]
                combined.reset()
                assert combined.read_counter() == combined.pos == 0
                assert combined.last_stats == {}
                with pytest.raises(RuntimeError, match="completed"):
                    combined.read_output()
                # Reset changes validity/counter only, with no full KV upload.
                for index in range(spec.layers):
                    np.testing.assert_array_equal(combined.read_cache(index), caches[index])
                for hidden, expected in zip(inputs, outputs):
                    combined.step(hidden)
                    np.testing.assert_array_equal(bits(combined.read_output()), bits(expected))
                for index in range(spec.layers):
                    np.testing.assert_array_equal(combined.read_cache(index), caches[index])
                assert combined.read_counter() == spec.steps


def test_device_bank_counter_needs_only_hidden_upload_and_no_host_reads(monkeypatch):
    from opentpu import quant as Q
    spec, cfg, layers, inputs = fixture()
    expected = source_reference(spec, layers, inputs)
    with DepthLayerEngine(layers, spec=spec, cfg=cfg) as engine:
        for layer in layers:
            layer["norm1"].fill(np.nan)
            layer["norm2"].fill(np.nan)
            for bank in layer["steps"]:
                for value in bank.values():
                    value.fill(np.nan)
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
            raise AssertionError("host model math during depth step")

        monkeypatch.setattr(engine.backend, "read", traced_read)
        monkeypatch.setattr(engine.backend, "write", traced_write)
        monkeypatch.setattr(engine.backend, "run", traced_run)
        monkeypatch.setattr(Q, "fake_act", forbidden)
        monkeypatch.setattr(Q, "fake_w", forbidden)
        for hidden in inputs:
            before = len(writes)
            engine.step(hidden)
            assert not reads
            assert len(writes) == before + 1
            s, address, value = writes[-1]
            assert s == 0 and address == engine.image.io["hidden"]
            assert value.dtype == np.float32 and value.nbytes == 4 * spec.hidden
            np.testing.assert_array_equal(value, hidden)
        assert all(program is engine.program for program in programs)
        np.testing.assert_allclose(engine.read_output(), expected[-1], atol=0.02, rtol=0.02)
        before = len(writes)
        engine.reset()
        assert len(writes) == before + 1
        assert writes[-1][1] == engine.image.io["state"] and writes[-1][2].nbytes == 4


def test_d128_half_heads_keep_padded_k_and_v_rows_zero():
    spec, cfg, layers, inputs = fixture(padded=True)
    with DepthLayerEngine(layers, spec=spec, cfg=cfg) as engine:
        assert spec.head_dim == 64 and engine.image.dpad == engine.image.cap == 128
        for hidden in inputs:
            engine.step(hidden)
        cache = engine.read_cache()
        start, size = engine.image.cache_regions[0]
        assert cache.dtype == np.uint8 and cache.shape == (size,)
        for head in engine.image.kv_heads[0].values():
            k = cache[head["k"] - start:head["k"] - start + engine.image.cap * engine.image.dpad]
            v = cache[head["vt"] - start:head["vt"] - start + engine.image.cap * engine.image.dpad]
            assert not k.reshape(engine.image.cap, engine.image.dpad)[:, spec.head_dim:].any()
            assert not v.reshape(engine.image.dpad, engine.image.cap)[spec.head_dim:].any()


@pytest.mark.parametrize("bad", ["shape", "nan", "inf", "overflow"])
def test_invalid_input_rejected_before_write_or_counter_advance(bad, monkeypatch):
    spec, cfg, layers, inputs = fixture()
    value = (inputs[:1] if bad == "shape" else
             np.full(spec.hidden, {"nan": np.nan, "inf": np.inf, "overflow": 1e300}[bad]))
    with DepthLayerEngine(layers, spec=spec, cfg=cfg) as engine:
        monkeypatch.setattr(engine.backend, "write", lambda *args: pytest.fail("unexpected write"))
        with np.errstate(over="ignore"), pytest.raises(ValueError):
            engine.step(value)
        assert engine.pos == 0


def test_failure_latch_reset_output_guard_and_idempotent_close(monkeypatch):
    spec, cfg, layers, inputs = fixture()
    engine = DepthLayerEngine(layers, spec=spec, cfg=cfg)
    assert engine.read_counter() == 0
    with pytest.raises(RuntimeError, match="completed"):
        engine.read_output()
    engine.step(inputs[0])
    first = engine.read_output()
    run = engine.backend.run

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic depth execution failure")

    monkeypatch.setattr(engine.backend, "run", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        engine.step(inputs[1])
    assert engine.pos == 1
    for action in (lambda: engine.step(inputs[0]), engine.read_output, engine.read_counter, engine.read_cache):
        with pytest.raises(RuntimeError, match="failed"):
            action()
    monkeypatch.setattr(engine.backend, "run", run)
    engine.reset()
    assert engine.read_counter() == 0
    engine.step(inputs[0])
    np.testing.assert_array_equal(bits(engine.read_output()), bits(first))
    closed = []
    close = getattr(engine.backend, "close", lambda: None)

    def traced_close():
        closed.append(True)
        close()

    monkeypatch.setattr(engine.backend, "close", traced_close, raising=False)
    engine.close()
    engine.close()
    assert closed == [True]
    for action in (lambda: engine.step(inputs[0]), engine.reset, engine.read_output,
                   engine.read_counter, engine.read_cache, lambda: engine.output_address):
        with pytest.raises(RuntimeError, match="closed"):
            action()


def unread():
    raise AssertionError("layer payload read before preflight")


def test_memory_instruction_and_compile_capacity_precede_lazy_payload(monkeypatch):
    spec, cfg, _, _ = fixture()
    layout = DepthLayerImage(spec, cfg)
    with pytest.raises(MemoryError):
        DepthLayerEngine(unread, spec=spec, cfg=replace(cfg, DRAM_BYTES=layout.required_bytes - 1))
    with pytest.raises((CompileError, MemoryError)):
        DepthLayerEngine(unread, spec=spec, cfg=replace(cfg, IMEM_WORDS=8))

    def fail(self):
        raise CompileError("synthetic depth program capacity failure")

    monkeypatch.setattr(DepthLayerImage, "compile", fail)
    with pytest.raises(CompileError, match="capacity"):
        DepthLayerEngine(unread, spec=spec, cfg=cfg)


def test_lazy_layers_load_once_after_compile(monkeypatch):
    spec, cfg, layers, inputs = fixture()
    compile_program = DepthLayerImage.compile
    events = []

    def compile(self):
        result = compile_program(self)
        events.append("compile")
        return result

    def load():
        assert events == ["compile"]
        events.append("load")
        return layers

    monkeypatch.setattr(DepthLayerImage, "compile", compile)
    with DepthLayerEngine(load, spec=spec, cfg=cfg) as engine:
        engine.step(inputs[0])
        engine.reset()
        engine.step(inputs[0])
    assert events == ["compile", "load"]


@pytest.mark.parametrize("bad", ["layer_count", "step_count", "norm_shape", "norm_nan",
                                  "matrix_shape", "matrix_nan", "matrix_overflow"])
def test_invalid_payload_shapes_and_nonfinite_values(bad):
    spec, cfg, layers, _ = fixture()
    if bad == "layer_count":
        layers = []
    elif bad == "step_count":
        layers[0]["steps"].pop()
    elif bad == "norm_shape":
        layers[0]["norm1"] = layers[0]["norm1"][None, :]
    elif bad == "norm_nan":
        layers[0]["norm2"][0] = np.nan
    elif bad == "matrix_shape":
        layers[0]["steps"][0]["wd"] = layers[0]["steps"][0]["wd"][:, :-1]
    elif bad == "matrix_nan":
        layers[0]["steps"][0]["wk"][0, 0] = np.nan
    else:
        value = layers[0]["steps"][0]["wg"].astype(np.float64)
        value[0, 0] = 1e300
        layers[0]["steps"][0]["wg"] = value
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        DepthLayerImage(spec, cfg).build(layers)


@pytest.mark.parametrize("changes", [{"hidden": True}, {"n_heads": 0}, {"ffn": 1.5},
                                      {"steps": True}, {"steps": 1}, {"layers": 0},
                                      {"hidden": 1 << 24}, {"n_heads": 3},
                                      {"eps": 0}, {"eps": np.nan}, {"cache_policy": "sliding8"}])
def test_invalid_specs(changes):
    options = dict(hidden=64, n_heads=2, ffn=128, steps=4)
    options.update(changes)
    with pytest.raises(ValueError):
        DepthLayerSpec(**options)


def test_causal_one_step_is_explicit_and_misaligned_layout_rejected():
    spec, cfg, layers, inputs = fixture(steps=1, policy="causal")
    with DepthLayerEngine(layers, spec=spec, cfg=cfg) as engine:
        engine.step(inputs[0])
        assert engine.read_counter() == 1 and np.isfinite(engine.read_output()).all()
    with pytest.raises(ValueError):
        DepthLayerImage(replace(spec, hidden=62), cfg)
    with pytest.raises(ValueError):
        DepthLayerImage(spec, replace(cfg, S=2))


def test_direct_board_factory_cap_gen_and_cleanup_guards(monkeypatch):
    from opentpu.host import board
    spec, cfg, layers, _ = fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("unqualified device or layer payload access")

    monkeypatch.setattr(board, "open_transport", forbidden)
    monkeypatch.setattr(board, "BoardBackend", forbidden)
    with pytest.raises(ValueError, match="qualified backend"):
        DepthLayerEngine(unread, spec=spec, cfg=cfg, backend="board")
    with pytest.raises(ValueError, match="backend"):
        DepthLayerEngine(unread, spec=spec, cfg=cfg, backend="gpu")
    with pytest.raises(ValueError, match="config"):
        DepthLayerEngine(unread, spec=spec, backend=lambda c, i: None)
    closed = []

    class Unsupported:
        generates = False

        def close(self):
            closed.append(True)

    with pytest.raises(ValueError, match="CAP_GEN|RLD"):
        DepthLayerEngine(layers, spec=spec, cfg=cfg, backend=lambda c, i: Unsupported())
    assert closed == [True]


def test_actual_boardbackend_over_isatransport_d128_padded_heads():
    from opentpu.host.board import BoardBackend
    from opentpu.host.fake import _target_cfg
    from opentpu.host.isaboard import IsaTransport
    from opentpu.isasim import board_config
    from opentpu.target import arch_hash, current
    spec, _, layers, inputs = fixture(padded=True)
    cfg = replace(board_config(), DRAM_BYTES=1 << 25,
                  SEQ_WC=bool(current().arch.get("SEQ_WC", False)))
    regcfg = _target_cfg(cfg.D, cfg.MCOLS, cfg.LANES, cfg.PAIR, cfg.DSTEP,
                         cfg.STREAM, cfg.act_rows, cfg.SEQ_WC)
    regcfg.update(DRAM_BYTES=cfg.DRAM_BYTES, channel_bytes=cfg.DRAM_BYTES // 2,
                  hash=arch_hash(cfg))
    transport = IsaTransport(cfg=regcfg)
    transport.cfg = cfg

    def backend(c, images):
        return BoardBackend(c, images, transport=transport, own_transport=True, status=False)

    with DepthLayerEngine(layers, spec=spec, cfg=cfg, backend=backend) as board:
        with DepthLayerEngine(layers, spec=spec, cfg=cfg) as isa:
            assert board.backend.engine is board
            assert board.image.batch == 1 and board.image.kv_bytes > 0
            for hidden in inputs:
                board.step(hidden)
                isa.step(hidden)
                np.testing.assert_array_equal(bits(board.read_output()), bits(isa.read_output()))
                assert board.read_counter() == isa.read_counter()
            np.testing.assert_array_equal(board.read_cache(), isa.read_cache())
            final = board.read_output()
            board.reset()
            for hidden in inputs:
                board.step(hidden)
            np.testing.assert_array_equal(bits(board.read_output()), bits(final))
            assert transport.sim_runs == 2 * spec.steps
            assert transport.bytes_written >= board.image.nbytes and transport.bytes_read > 0
