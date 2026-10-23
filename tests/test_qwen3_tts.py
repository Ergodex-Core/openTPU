"""Independent affine/SiLU reference, existing-ISA execution, and strict serving gate.

Small synthetic tensors only. No upstream Python execution, checkpoint payload,
PyTorch, network, FPGA transport, or service is needed.
"""
import builtins
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.isasim import Config
from opentpu.llm.qwen3_tts import (FPGAOnlyUnavailable, MODEL_ID, ProjectionSpec,
                                 TTSRequest, portal_metadata, synthesize)
from opentpu.runtime import run


PREFIX = "talker.text_projection."


def fixture(spec, rows, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, spec.input_size)).astype(np.float32)
    w = {
        "linear_fc1.weight": rng.normal(size=(spec.intermediate_size, spec.input_size))
                              / np.sqrt(spec.input_size),
        "linear_fc1.bias": rng.normal(size=spec.intermediate_size) * 0.3,
        "linear_fc2.weight": rng.normal(size=(spec.output_size, spec.intermediate_size))
                              / np.sqrt(spec.intermediate_size),
        "linear_fc2.bias": rng.normal(size=spec.output_size) * 0.3,
    }
    return x, {PREFIX + k: v.astype(np.float32) for k, v in w.items()}


def reference(x, w):
    """Direct float64 mathematics; no compiler, ISA, or quantizer helpers."""
    first = x.astype(np.float64) @ w[PREFIX + "linear_fc1.weight"].astype(np.float64).T
    first += w[PREFIX + "linear_fc1.bias"]
    silu = first / (1.0 + np.exp(-first))
    return silu @ w[PREFIX + "linear_fc2.weight"].astype(np.float64).T + w[PREFIX + "linear_fc2.bias"]


@pytest.mark.parametrize("slices", [1, 2])
@pytest.mark.parametrize("depth", [32, 128])
@pytest.mark.parametrize("rows", [1, 4])
def test_projection_matches_independent_reference(slices, depth, rows):
    spec = ProjectionSpec(2 * depth, 2 * slices * depth, slices * depth)
    cfg = Config(S=slices, D=depth, MCOLS=4, DRAM_BYTES=2 << 20)
    x, weights = fixture(spec, rows)
    compiled, images = spec.compile(x, weights, cfg)
    result = run(compiled, images, backend="isa")
    expected = reference(x, weights)
    error = np.linalg.norm(result.outputs["out"] - expected) / np.linalg.norm(expected)
    assert error < 0.025, error  # INT8 device math is not the upstream BF16 computation.
    for program in compiled.programs:
        assert {i.op for i in program} <= {
            I.NOP, I.HALT, I.LI, I.ADDI, I.LOOP, I.BAR, I.LD, I.ST,
            I.MM, I.QACT, I.VOP, I.GATHER,
        }
        assert sum(i.op == I.MM for i in program) >= 2
        assert all((i.flags >> I.WF_SHIFT) & 3 == I.W8 for i in program if i.op == I.MM)
    # Replaying the same image cannot depend on hidden host state.
    replay = run(compiled, images, backend="isa")
    assert np.array_equal(result.outputs["out"].view(np.uint32),
                          replay.outputs["out"].view(np.uint32))


def test_biases_are_sharded_in_the_correct_order():
    spec = ProjectionSpec(64, 128, 64)
    x, weights = fixture(spec, 1)
    weights[PREFIX + "linear_fc1.weight"].fill(0)
    weights[PREFIX + "linear_fc1.bias"] = np.linspace(-3, 3, 128, dtype=np.float32)
    weights[PREFIX + "linear_fc2.bias"] = np.linspace(2, -2, 64, dtype=np.float32)
    cfg = Config(S=2)
    result = run(*spec.compile(x, weights, cfg), backend="isa").outputs["out"]
    assert np.linalg.norm(result - reference(x, weights)) / np.linalg.norm(reference(x, weights)) < 0.01


def test_native_dimensions_compile_with_existing_f2_isa():
    # Bound allocation to 8 MiB, not the full 4 GiB target memory window.
    from opentpu.target import load
    target = load("f2")
    cfg = Config(**{k: v for k, v in target.arch.items() if k in Config.__dataclass_fields__})
    cfg = replace(cfg, DRAM_BYTES=8 << 20)
    spec = ProjectionSpec()
    x = np.zeros((1, 2048), np.float32)
    weights = {PREFIX + "linear_fc1.weight": np.zeros((2048, 2048), np.float32),
               PREFIX + "linear_fc1.bias": np.zeros(2048, np.float32),
               PREFIX + "linear_fc2.weight": np.zeros((1024, 2048), np.float32),
               PREFIX + "linear_fc2.bias": np.zeros(1024, np.float32)}
    compiled, _ = spec.compile(x, weights, cfg)
    assert all(len(p) * 8 <= cfg.IMEM_WORDS for p in compiled.programs)
    assert all(i.op not in {I.DSTEP, I.STREAM, I.WAITW, I.RLD} for p in compiled.programs for i in p)


def test_configuration_is_exact_variant():
    path = Path(__file__).parent / "fixtures/qwen3_tts_06b_config.json"
    config = json.loads(path.read_text())
    assert ProjectionSpec.from_config(config) == ProjectionSpec()
    for key, value in [("tts_model_size", "1b7"), ("tts_model_type", "base"),
                       ("model_type", "qwen3"), ("tokenizer_type", "other")]:
        with pytest.raises(ValueError, match="0.6B"):
            ProjectionSpec.from_config(dict(config, **{key: value}))
    with pytest.raises(ValueError, match="activation"):
        ProjectionSpec.from_config(dict(config, talker_config=dict(config["talker_config"], hidden_act="gelu")))


def test_projection_refuses_bad_shapes_and_nonfinite_values():
    spec = ProjectionSpec(64, 64, 32)
    x, weights = fixture(spec, 1)
    with pytest.raises(ValueError, match="alignment"):
        ProjectionSpec(63, 64, 32).compile(x, weights, Config())
    with pytest.raises(ValueError, match="MCOLS"):
        spec.compile(np.zeros((9, 64)), weights, Config())
    with pytest.raises(ValueError, match="ACT RAM"):
        spec.compile(x, weights, Config(ACT_BLOCKS=1))
    bad = dict(weights)
    bad[PREFIX + "linear_fc1.bias"] = np.zeros(63)
    with pytest.raises(ValueError, match="expected"):
        spec.compile(x, bad, Config())
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite input"):
        spec.compile(x, weights, Config())
    x[0, 0] = 0
    weights[PREFIX + "linear_fc2.bias"][0] = np.inf
    with pytest.raises(ValueError, match="non-finite checkpoint"):
        spec.compile(x, weights, Config())


def test_portal_contract_and_no_silent_fallback(monkeypatch):
    metadata = portal_metadata()
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["id"] == MODEL_ID and metadata["enabled"] is False
    assert not metadata["style_instructions"] and not metadata["voice_cloning"]
    assert metadata["fallback"] is None and not metadata["streaming_available"]
    assert metadata["sample_rate_hz"] / metadata["samples_per_codec_frame"] == 12.5
    assert len(metadata["speakers"]) == 9 and len(metadata["languages"]) == 11  # includes Auto
    request = TTSRequest("Hello.", "rYaN", "english").validated()
    assert request.speaker == "Ryan" and request.language == "English"
    # Generation may not even import a runtime/model/backend, much less open one.
    def forbidden_import(*args, **kwargs):
        raise AssertionError("synthesis tried to import a backend")
    monkeypatch.setattr(builtins, "__import__", forbidden_import)
    for stream in (False, True):
        with pytest.raises(FPGAOnlyUnavailable, match="No host fallback"):
            synthesize(replace(request, stream=stream))


@pytest.mark.parametrize("kwargs", [{"text": " "}, {"speaker": "unknown"},
                                   {"language": "Dutch"}, {"instruct": "happy"},
                                   {"stream": "yes"}])
def test_request_rejects_unsupported_controls(kwargs):
    with pytest.raises(ValueError):
        TTSRequest(**dict({"text": "Hello"}, **kwargs)).validated()
