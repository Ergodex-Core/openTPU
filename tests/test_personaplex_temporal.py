"""Temporal FPGA math against an independent adjacent-pair NumPy transformer.

No checkpoint, torch, or FPGA is needed. The real ISA kernels execute persistent
INT8 KV over multiple frames; the reference uses unquantized float64 math.
"""
from dataclasses import replace

import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.llm.personaplex_temporal import Engine, Image, Spec


def fixture_model(layers=2, context=64):
    spec = Spec(hidden=64, layers=layers, n_heads=2, ffn=128, context=context)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True, DRAM_BYTES=1 << 22)
    rng = np.random.default_rng(192)
    weights = []
    for _ in range(layers):
        shapes = {"wq": (64, 64), "wk": (64, 64), "wv": (64, 64), "wo": (64, 64),
                  "wg": (128, 64), "wu": (128, 64), "wd": (64, 128)}
        layer = {k: rng.normal(0, 0.18 if k in ("wq", "wk") else 0.08, shape).astype(np.float32)
                 for k, shape in shapes.items()}
        layer.update(norm1=rng.uniform(0.8, 1.2, 64).astype(np.float32),
                     norm2=rng.uniform(0.8, 1.2, 64).astype(np.float32))
        weights.append(layer)
    norm = rng.uniform(0.8, 1.2, 64).astype(np.float32)
    inputs = rng.normal(0, 1, (34, 64)).astype(np.float32)
    return spec, cfg, weights, norm, inputs


def reference(spec, weights, final_norm, embeddings, *, adjacent=True):
    """Independent batched causal transformer; no production helpers/quantizers."""
    x = np.asarray(embeddings, np.float64)
    T, H = x.shape
    d = spec.head_dim
    angles = np.arange(T)[:, None, None] * spec.rope_base ** (-np.arange(0, d, 2) / d)
    cos, sin = np.cos(angles), np.sin(angles)

    def norm(v, gain):
        return v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + spec.eps) * gain

    def rotate(v):
        out = np.empty_like(v)
        if adjacent:
            even, odd = v[..., 0::2], v[..., 1::2]
            out[..., 0::2], out[..., 1::2] = even * cos - odd * sin, odd * cos + even * sin
        else:
            a, b = v[..., :d // 2], v[..., d // 2:]
            out[..., :d // 2], out[..., d // 2:] = a * cos - b * sin, b * cos + a * sin
        return out

    causal = np.where(np.arange(T)[None, :] <= np.arange(T)[:, None], 0., -np.inf)
    for w in weights:
        h = norm(x, w["norm1"])
        q = rotate((h @ w["wq"].T).reshape(T, spec.n_heads, d))
        k = rotate((h @ w["wk"].T).reshape(T, spec.n_heads, d))
        v = (h @ w["wv"].T).reshape(T, spec.n_heads, d)
        out = np.empty_like(v)
        for head in range(spec.n_heads):
            scores = q[:, head] @ k[:, head].T / np.sqrt(d) + causal
            p = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            p /= p.sum(axis=-1, keepdims=True)
            out[:, head] = p @ v[:, head]
        x = x + out.reshape(T, H) @ w["wo"].T
        h = norm(x, w["norm2"])
        gate = h @ w["wg"].T
        up = h @ w["wu"].T
        x = x + (gate / (1 + np.exp(-gate)) * up) @ w["wd"].T
    return norm(x, final_norm)


@pytest.mark.parametrize("fmt,relative_error", [("int8", 0.025), ("int4", 0.10)])
def test_multiframe_temporal_matches_independent_reference(fmt, relative_error):
    spec, cfg, weights, norm, inputs = fixture_model()
    expected = reference(spec, weights, norm, inputs)
    wrong_rope = reference(spec, weights, norm, inputs, adjacent=False)
    with Engine(spec, weights, norm, cfg=cfg, cap=34, wformat=fmt) as eng:
        actual = np.stack([eng.step(x) for x in inputs])
        assert eng.pos == 34
        assert len(eng._programs) == 2  # exercised a runtime attention-bucket boundary
        assert eng.last_stats["instructions"][0] > 0
    error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert error < relative_error
    # This fixture distinguishes the adjacent-pair convention; a plain Llama
    # import without permuting Q/K does not pass the comparison.
    assert np.linalg.norm(actual - expected) < np.linalg.norm(actual - wrong_rope) * 0.6


def test_context_bound_and_reset_mask_old_kv():
    spec, cfg, weights, norm, inputs = fixture_model(layers=1)
    eng = Engine(spec, weights, norm, cfg=cfg, cap=3)
    first = np.stack([eng.step(x) for x in inputs[:3]])
    with pytest.raises(RuntimeError, match="context is full"):
        eng.step(inputs[3])
    assert eng.pos == 3
    eng.reset()
    again = np.stack([eng.step(x) for x in inputs[:3]])
    np.testing.assert_array_equal(first.view(np.uint32), again.view(np.uint32))
    eng.reset()
    altered = eng.step(inputs[8])
    fresh = Engine(spec, weights, norm, cfg=cfg, cap=3)
    np.testing.assert_array_equal(altered, fresh.step(inputs[8]))
    assert eng.pos == fresh.pos == 1
    fresh.close()
    eng.close()
    with pytest.raises(RuntimeError, match="closed"):
        eng.step(inputs[0])


def test_bad_inputs_and_weights_fail_without_advancing_state():
    spec, cfg, weights, norm, inputs = fixture_model(layers=1)
    eng = Engine(spec, weights, norm, cfg=cfg, cap=3)
    with pytest.raises(ValueError, match="embedding: shape"):
        eng.step(inputs[:1])
    with pytest.raises(ValueError, match="non-finite"):
        eng.step(np.full(spec.hidden, np.nan))
    assert eng.pos == 0
    eng.close()
    bad = [{**weights[0], "wq": np.ones((63, 64))}]
    with pytest.raises(ValueError, match="wq: shape"):
        Image(spec, cfg, cap=3).build(bad, norm)
    with pytest.raises(ValueError, match="expected 1 weight layers"):
        Image(spec, cfg, cap=3).build([], norm)


def test_backend_failure_requires_explicit_reset():
    spec, cfg, weights, norm, inputs = fixture_model(layers=1)
    eng = Engine(spec, weights, norm, cfg=cfg, cap=3)
    run = eng.backend.run

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic hardware failure")

    eng.backend.run = fail
    with pytest.raises(RuntimeError, match="synthetic hardware failure"):
        eng.step(inputs[0])
    eng.backend.run = run
    with pytest.raises(RuntimeError, match="previous frame failed"):
        eng.step(inputs[0])
    eng.reset()
    assert np.isfinite(eng.step(inputs[0])).all()
    assert eng.pos == 1


def test_full_model_layout_and_program_fit_without_weight_allocation():
    cfg = Config(S=1, D=128, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=128, TMEM_WORDS=65536,
                 IMEM_WORDS=32768, PAIR=True, SEQ_WC=True, DRAM_BYTES=1 << 32)
    image = Image(Spec(), cfg, cap=256)
    assert image.nbytes == 3585903104
    assert image.required_bytes == 3586068480
    assert image.weight_bytes == 3516416000
    assert image.weight_code_bytes == 3288334336
    assert image.kv_bytes == 69206016
    long = Image(Spec(), cfg, cap=2816)
    for bucket in range(1, 12):
        programs, args = long.compile_bucket(bucket)
        assert 0 < len(programs[0]) * 8 <= cfg.IMEM_WORDS
        assert args  # actual position is runtime data; programs are reusable
    with pytest.raises(MemoryError):
        Image(Spec(), cfg, cap=3000)
    with pytest.raises(MemoryError, match="program area"):
        Image(Spec(), replace(cfg, DRAM_BYTES=image.nbytes), cap=256)
    with pytest.raises(ValueError, match="cap must"):
        Image(Spec(), cfg, cap=3001)
    with pytest.raises(ValueError, match="measured device config"):
        Engine(Spec(), None, None, backend="board")


def test_future_masked_kv_does_not_affect_current_output():
    spec, cfg, weights, norm, inputs = fixture_model(layers=1)
    eng = Engine(spec, weights, norm, cfg=cfg, cap=34)
    baseline = eng.step(inputs[0])
    # Fill a prior conversation through the second attention bucket, then reset.
    for x in inputs[1:]:
        eng.step(x)
    eng.reset()
    np.testing.assert_array_equal(eng.step(inputs[0]), baseline)


def test_nonaligned_cap_crosses_256_frame_boundary_and_stays_bounded():
    _, cfg, raw, norm, _ = fixture_model(layers=1)
    spec = Spec(hidden=32, layers=1, n_heads=1, ffn=64, context=300)
    shapes = {"norm1": (32,), "norm2": (32,), "wq": (32, 32), "wk": (32, 32),
              "wv": (32, 32), "wo": (32, 32), "wg": (64, 32), "wu": (64, 32),
              "wd": (32, 64)}
    weights = [{k: raw[0][k][tuple(slice(0, s) for s in shape)].copy()
                for k, shape in shapes.items()}]
    norm = norm[:32]
    inputs = np.random.default_rng(241).normal(0, 1, (257, 32)).astype(np.float32)
    eng = Engine(spec, weights, norm, cfg=cfg, cap=257, wformat="int8")
    assert eng.image.frame_limit == 257
    assert eng.image.cap == 512 and eng.image.block == 256
    assert len(eng._programs) == 2  # all requested buckets preflighted before inference
    expected = reference(spec, weights, norm, inputs)
    actual = np.stack([eng.step(x) for x in inputs])
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.025
    with pytest.raises(RuntimeError, match="context is full"):
        eng.step(inputs[0])
    # Padding is storage only; it does not extend the admitted conversation.
    assert eng.pos == 257
    eng.reset()
    np.testing.assert_array_equal(eng.step(inputs[0]), actual[0])
    eng.close()


def test_backend_arguments_are_validated_before_weights_are_read():
    spec, cfg, _, _, _ = fixture_model(layers=1)
    with pytest.raises(ValueError, match="backend must be"):
        Engine(spec, None, None, cfg=cfg, backend="gpu")
    with pytest.raises(ValueError, match="transport is only valid"):
        Engine(spec, None, None, cfg=cfg, backend="isa", transport=object())
    with pytest.raises(ValueError, match="transport is only valid"):
        Engine(spec, None, None, cfg=cfg, backend=lambda c, i: None, transport=object())


def test_actual_boardbackend_protocol_matches_isa_bits():
    """Exercise the real BoardBackend start/args/wait, attach and memory protocol.

    IsaTransport supplies simulated registers and executes the uploaded programs,
    so the test needs no device and cannot report measured FPGA timing.
    """
    from opentpu.host.board import BoardBackend
    from opentpu.host.fake import _target_cfg
    from opentpu.host.isaboard import IsaTransport
    from opentpu.isasim import board_config
    from opentpu.target import arch_hash, current

    base = board_config()
    cfg = replace(base, DRAM_BYTES=1 << 22,
                  SEQ_WC=bool(current().arch.get("SEQ_WC", False)))
    regcfg = _target_cfg(cfg.D, cfg.MCOLS, cfg.LANES, cfg.PAIR, cfg.DSTEP,
                         cfg.STREAM, cfg.act_rows, cfg.SEQ_WC)
    regcfg.update(DRAM_BYTES=cfg.DRAM_BYTES, channel_bytes=cfg.DRAM_BYTES // 2,
                  hash=arch_hash(cfg))
    transport = IsaTransport(cfg=regcfg)
    transport.cfg = cfg
    spec = Spec(hidden=128, layers=1, n_heads=1, ffn=256, context=256)
    rng = np.random.default_rng(875)
    shapes = {"wq": (128, 128), "wk": (128, 128), "wv": (128, 128), "wo": (128, 128),
              "wg": (256, 128), "wu": (256, 128), "wd": (128, 256)}
    layer = {k: rng.normal(0, 0.07, shape).astype(np.float32) for k, shape in shapes.items()}
    layer.update(norm1=np.ones(128, np.float32), norm2=np.ones(128, np.float32))
    norm = np.ones(128, np.float32)

    def backend(c, images):
        return BoardBackend(c, images, transport=transport, status=False, own_transport=True)

    eng = Engine(spec, [layer], norm, cfg=cfg, backend=backend, cap=3)
    ref = Engine(spec, [layer], norm, cfg=cfg, cap=3)
    assert eng.backend.engine is eng  # actual attach(), including capability checks
    for _ in range(3):
        x = rng.normal(size=128).astype(np.float32)
        np.testing.assert_array_equal(eng.step(x), ref.step(x))
    layout = eng.backend._layout()
    assert layout["kv_used"] > 0 and layout["kv_capacity"] == eng.image.kv_bytes
    assert transport.sim_runs == 3
    assert transport.bytes_written >= eng.image.nbytes and transport.bytes_read > 0
    eng.reset()
    ref.reset()
    np.testing.assert_array_equal(eng.step(x), ref.step(x))
    eng.close()
    ref.close()
