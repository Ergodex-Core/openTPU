"""Resident embeddings + complete layer tests; no upstream code or card access."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.isasim import Config
from opentpu.llm.qwen3_tts import FPGAOnlyUnavailable, TTSRequest, synthesize
from opentpu.llm.qwen3_tts_talker import ISASession, TalkerBlockImage, TalkerSpec
from qwen3_tts_talker_reference import Reference


FIXTURES = Path(__file__).parent / "fixtures"


def make_weights(image, seed=41):
    rng = np.random.default_rng(seed)
    weights = {}
    for name, (_, _, shape) in image.parameters.items():
        a = rng.standard_normal(shape)
        if "norm.weight" in name:
            a = 1 + 0.1 * a
        elif len(shape) == 2 and "embedding" not in name:
            a *= 0.6 / np.sqrt(shape[-1])
        elif name.endswith("bias"):
            a *= 0.1
        else:
            a *= 0.5
        weights[name] = a.astype(np.float32)
    return weights


def tiny(depth=32, capacity=64, attention_block=32):
    spec = TalkerSpec(text_vocab=17, codec_vocab=13, text_hidden=2 * depth,
                      hidden=2 * depth, n_q=4, n_kv=2, head_dim=depth, ffn=4 * depth)
    cfg = Config(S=1, D=depth, MCOLS=4, DRAM_BYTES=4 << 20)
    return TalkerBlockImage(spec, cfg, capacity, attention_block)


def read(session, name):
    desc = getattr(session.image.desc, name)
    n = int(np.prod(desc.shape))
    return session.machine.slices[0].dram[desc.base.const:desc.base.const + 4 * n].view(np.float32).copy().reshape(-1)


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-15))


@pytest.mark.parametrize("depth", [32, 128])
def test_full_slice_matches_source_derived_reference(depth, record_property):
    image = tiny(depth, capacity=2 * depth, attention_block=depth)
    weights = make_weights(image)
    session = ISASession(image, weights)
    reference = Reference(image.spec, weights)
    errors = {}
    for text_id, codec_id in [(1, 3), (16, 12), (0, 0), (8, 7)]:
        expected = reference.step(text_id, codec_id)
        got = session.step(text_id, codec_id)
        for name in ("projected", "assembled", "attention_residual", "output"):
            error = relative(read(session, name), expected[name])
            errors[name] = max(errors.get(name, 0), error)
            assert error < 0.035, name
        np.testing.assert_array_equal(got, read(session, "output"))
    record_property("max_relative_l2", json.dumps(errors))


def test_native_layer_widths_with_small_synthetic_vocabularies(record_property):
    # Exercise all 16/8 attention heads and native projection/MLP widths without
    # allocating or downloading the 1.24 GB FP32 text embedding table.
    spec = replace(TalkerSpec(), text_vocab=17, codec_vocab=13)
    image = TalkerBlockImage(spec, Config(D=128, DRAM_BYTES=64 << 20), capacity=256)
    weights = make_weights(image, 43)
    session, reference = ISASession(image, weights), Reference(spec, weights)
    errors = {}
    for text_id, codec_id in [(16, 12), (0, 0)]:
        expected = reference.step(text_id, codec_id)
        session.step(text_id, codec_id)
        for name in ("projected", "assembled", "attention_residual", "output"):
            error = relative(read(session, name), expected[name])
            errors[name] = max(errors.get(name, 0), error)
            assert error < 0.035, name
    record_property("max_relative_l2", json.dumps(errors))


def test_cache_is_resident_causal_and_crosses_attention_block_boundary():
    image = tiny()
    weights = make_weights(image, 7)
    a, b = ISASession(image, weights), ISASession(image, weights)
    reference = Reference(image.spec, weights)
    # Poison unwritten future KV slots. Causal attention must never consult them.
    rng = np.random.default_rng(8)
    for name, (address, size) in image.regions.items():
        if name.startswith("kv."):
            b.machine.slices[0].dram[address:address + size] = rng.integers(1, 100, size, dtype=np.uint8)
    for position in range(35):
        text_id, codec_id = (position * 7) % 17, (position * 5) % 13
        expected = reference.step(text_id, codec_id)["output"]
        out = a.step(text_id, codec_id)
        np.testing.assert_array_equal(out.view(np.uint32), b.step(text_id, codec_id).view(np.uint32))
        assert relative(out, expected) < 0.04
    # Equal current IDs with different earlier inputs must retain different state.
    fresh = ISASession(image, weights)
    assert not np.array_equal(out, fresh.step(text_id, codec_id))


def test_host_weights_can_disappear_after_packing_and_ids_are_the_only_input():
    image = tiny()
    weights = make_weights(image)
    a, b = ISASession(image, weights), ISASession(image, weights)
    for value in weights.values():
        value.fill(np.nan)
    for position, pair in enumerate([(2, 3), (7, 1), (16, 12)]):
        # B bypasses the session wrapper to show the wire input is exactly two IDs.
        programs, _ = image.compile(position)
        address = image.desc.ids.base.const
        b.machine.slices[0].dram[address:address + 8] = image.token_bytes(*pair)
        b.machine.load(programs).run()
        np.testing.assert_array_equal(a.step(*pair).view(np.uint32), read(b, "output").view(np.uint32))


def test_attention_hardware_loop_with_independent_resident_cache(record_property):
    image = tiny(capacity=1536, attention_block=256)
    weights = make_weights(image, 19)
    session, reference = ISASession(image, weights), Reference(image.spec, weights)
    position = 1283  # five full blocks plus a tail: takes the hardware-loop path
    rng = np.random.default_rng(20)
    s = image.spec
    keys = rng.integers(-100, 101, (s.n_kv, image.capacity, s.head_dim), dtype=np.int8)
    values = rng.integers(-80, 121, keys.shape, dtype=np.int8)
    kscale, vscale = np.float32(0.01), np.float32(0.02)

    def put(name, data):
        address, size = image.regions[name]
        raw = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        assert raw.size == size
        session.machine.slices[0].dram[address:address + size] = raw

    for head in range(s.n_kv):
        put(f"kv.{head}.k", keys[head])
        put(f"kv.{head}.ks", np.full(image.capacity, kscale, np.float32))
        # Independent 256-token V-transpose tile layout; no KV packing helper.
        tiled = values[head].reshape(-1, 256, s.head_dim).transpose(0, 2, 1)
        put(f"kv.{head}.vt", tiled)
        put(f"kv.{head}.vs", np.full(image.capacity, vscale, np.float32))
    reference.keys = list((keys[:, :position].astype(np.float64) * kscale).transpose(1, 0, 2))
    reference.values = list((values[:, :position].astype(np.float64) * vscale).transpose(1, 0, 2))
    session.position = position
    program, _ = image.compile(position)
    assert any(i.op == I.LOOP for i in program[0])
    expected = reference.step(5, 7)["output"]
    error = relative(session.step(5, 7), expected)
    assert error < 0.035
    record_property("max_relative_l2", json.dumps({"output": error}))


def test_native_weight_contract_matches_pinned_safetensors_header():
    config = json.loads((FIXTURES / "qwen3_tts_06b_config.json").read_text())
    contract = json.loads((FIXTURES / "qwen3_tts_talker_contract.json").read_text())
    spec = TalkerSpec.from_config(config)
    assert contract["source_revision"] == "022e286b98fbec7e1e916cb940cdf532cd9f488e"
    assert contract["model_revision"] == "85e237c12c027371202489a0ec509ded67b5e4b5"
    image = TalkerBlockImage(spec, Config(D=128, DRAM_BYTES=2 << 30), capacity=256)
    assert set(image.parameters) == set(contract["tensors"])
    for name, (_, _, shape) in image.parameters.items():
        assert list(shape) == contract["tensors"][name]["shape"]
        assert contract["tensors"][name]["dtype"] == "BF16"
    for field, value in [("attention_bias", True), ("num_attention_heads", 8),
                         ("sliding_window", 128), ("rms_norm_eps", 1e-5)]:
        changed = dict(config, talker_config=dict(config["talker_config"], **{field: value}))
        with pytest.raises(ValueError, match="configuration"):
            TalkerSpec.from_config(changed)
    changed = dict(config, talker_config=dict(config["talker_config"],
                   rope_scaling={"rope_type": "default", "interleaved": False, "mrope_section": [24, 20, 20]}))
    with pytest.raises(ValueError, match="rotary"):
        TalkerSpec.from_config(changed)


@pytest.mark.parametrize("position", [0, 255, 256, 32767])
def test_native_resident_shape_and_max_context_compile_without_allocating_weights(position, monkeypatch):
    from opentpu.target import load
    cfg = Config(**{k: v for k, v in load("f2").arch.items() if k in Config.__dataclass_fields__})
    image = TalkerBlockImage(TalkerSpec(), cfg, capacity=32768)
    def forbidden(*args, **kwargs):
        raise AssertionError("native descriptor check must not pack/download weights")
    monkeypatch.setattr(image, "build", forbidden)
    programs, budget = image.compile(position)
    assert budget["imem_words"] <= cfg.IMEM_WORDS
    assert budget["tmem_peak_words"] <= cfg.TMEM_WORDS
    assert image.nbytes < cfg.DRAM_BYTES
    assert sum(i.op == I.RLD for i in programs[0]) == 2
    assert all(i.op not in {I.WAITW, I.DSTEP, I.STREAM} for i in programs[0])


def test_zero_block_weights_preserve_the_assembled_residual():
    image = tiny()
    weights = make_weights(image)
    for name, a in weights.items():
        if name.startswith("talker.model.layers.0.") and a.ndim == 2:
            a.fill(0)
    session = ISASession(image, weights)
    result = session.step(5, 9)
    np.testing.assert_array_equal(result, read(session, "assembled"))


def test_rejections_precede_writes_and_no_full_tts_enablement():
    image = tiny()
    weights = make_weights(image)
    session = ISASession(image, weights)
    before = session.machine.slices[0].dram.copy()
    for pair in [(-1, 0), (17, 0), (0, 13), (0, 0.5), (True, 0)]:
        with pytest.raises(ValueError, match="token ID"):
            session.step(*pair)
        np.testing.assert_array_equal(before, session.machine.slices[0].dram)
    for position in [-1, image.capacity, True]:
        with pytest.raises(ValueError, match="position"):
            image.compile(position)
    with pytest.raises(MemoryError, match="no offload"):
        TalkerBlockImage(image.spec, replace(image.cfg, DRAM_BYTES=1024), 64, 32)
    with pytest.raises(MemoryError, match="packing budget"):
        image.build(weights, max_bytes=1024)
    with pytest.raises(ValueError, match="one FPGA slice"):
        TalkerBlockImage(image.spec, replace(image.cfg, S=2), 64, 32)
    with pytest.raises(FPGAOnlyUnavailable):
        synthesize(TTSRequest("Still not a complete TTS model."))


def test_checkpoint_shape_and_finite_guards():
    image = tiny()
    weights = make_weights(image)
    key = "talker.model.layers.0.self_attn.q_norm.weight"
    weights[key][0] = np.nan
    with pytest.raises(ValueError, match="expected finite"):
        image.build(weights)
    weights[key] = np.ones(31)
    with pytest.raises(ValueError, match="expected finite"):
        image.build(weights)
