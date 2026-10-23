"""Bounded ISA-check guards and CPU ISA replay; no checkpoint or device required."""
import importlib.util
import os
from pathlib import Path

import pytest


@pytest.fixture
def isa_check(monkeypatch):
    # Importing the CLI sets its process environment for standalone execution.
    # Register those variables first so pytest restores the caller's settings.
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "OTPU_BUILD_JOBS", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
             "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    for name in names:
        monkeypatch.setenv(name, os.environ.get(name, ""))
    path = Path(__file__).resolve().parents[1] / "tools/personaplex_isa_check.py"
    spec = importlib.util.spec_from_file_location("personaplex_isa_check_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("protected", ["model", "reference", "source", "output"])
@pytest.mark.parametrize("relation", ["equal", "ancestor"])
def test_existing_cache_protections_reject_before_side_effects(isa_check, monkeypatch, tmp_path,
                                                             protected, relation):
    paths = {"model": tmp_path / "models" / "checkpoint",
             "reference": tmp_path / "references" / "baseline",
             "source": tmp_path / "sources" / "repo",
             "output": tmp_path / "results" / "new-check"}
    for name in ("model", "reference", "source"):
        paths[name].mkdir(parents=True)
    embeddings = tmp_path / "captures" / "input.npy"
    embeddings.parent.mkdir()
    embeddings.write_bytes(b"input payload must not be opened")
    monkeypatch.setattr(isa_check, "ROOT", paths["source"])
    cache = paths[protected] if relation == "equal" else paths[protected].parent

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid cache path must fail before conversion or cache management")

    monkeypatch.setattr(isa_check, "MLXWeights", forbidden)
    monkeypatch.setattr(isa_check, "Engine", forbidden)
    monkeypatch.setattr(isa_check.qcache, "_evict", forbidden)
    monkeypatch.setenv("OTPU_IMAGE_CACHE", "preserve-existing-cache-setting")
    with pytest.raises(ValueError, match="Cache must be an isolated directory"):
        isa_check.run(paths["model"], embeddings, paths["reference"], paths["output"], cache)
    assert not paths["output"].exists()
    assert embeddings.read_bytes() == b"input payload must not be opened"
    assert os.environ["OTPU_IMAGE_CACHE"] == "preserve-existing-cache-setting"


@pytest.mark.parametrize("location", ["file", "parent", "ancestor", "symlink_parent"])
def test_cache_cannot_manage_embedding_input_or_ancestors(isa_check, monkeypatch, tmp_path, location):
    model = tmp_path / "model"
    reference = tmp_path / "reference"
    output = tmp_path / "result"
    embeddings = tmp_path / "captures" / "session" / "input.npy"
    model.mkdir()
    reference.mkdir()
    embeddings.parent.mkdir(parents=True)
    # Deliberately invalid payload: isolation must be checked before data reads.
    embeddings.write_bytes(b"input payload must not be opened")
    sentinel = embeddings.parent / "other" / "preserved.npz"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"unrelated capture artifact")
    choices = {"file": embeddings, "parent": embeddings.parent,
               "ancestor": embeddings.parent.parent}
    if location == "symlink_parent":
        alias = tmp_path / "capture-alias"
        alias.symlink_to(embeddings.parent, target_is_directory=True)
        cache = alias
    else:
        cache = choices[location]

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid cache path must fail before conversion or cache management")

    monkeypatch.setattr(isa_check, "MLXWeights", forbidden)
    monkeypatch.setattr(isa_check, "Engine", forbidden)
    monkeypatch.setattr(isa_check.qcache, "_evict", forbidden)
    monkeypatch.setenv("OTPU_IMAGE_CACHE", "preserve-existing-cache-setting")
    with pytest.raises(ValueError, match="Cache must be an isolated directory"):
        isa_check.run(model, embeddings, reference, output, cache)
    assert not output.exists()
    assert embeddings.read_bytes() == b"input payload must not be opened"
    assert sentinel.read_bytes() == b"unrelated capture artifact"
    assert os.environ["OTPU_IMAGE_CACHE"] == "preserve-existing-cache-setting"


def test_progress_layers_preserves_four_frame_image_isa_replay_and_metrics(isa_check, monkeypatch):
    import numpy as np

    from opentpu.isasim import Config
    from tools.personaplex_verilator import reference, synthetic_model

    monkeypatch.setenv("OTPU_BUILD_JOBS", "1")
    monkeypatch.setenv("OTPU_IMAGE_CACHE", "0")
    spec = isa_check.Spec(hidden=64, layers=2, n_heads=2, ffn=128, context=4)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True, DRAM_BYTES=1 << 22)
    weights, norm, inputs = synthetic_model(spec, frames=4)
    assert np.unique(inputs, axis=0).shape[0] == 4
    expected = reference(spec, weights, norm, inputs)
    assert expected.shape == (4, spec.hidden) and expected.dtype == np.float64
    events = []

    def emit(stage, **details):
        events.append((stage, details))

    layers = isa_check.ProgressLayers(weights, emit)
    # Both constructors execute the real Image.build and quantization path.
    with isa_check.Engine(spec, weights, norm, cfg=cfg, backend="isa", cap=4) as plain, \
            isa_check.Engine(spec, layers, norm, cfg=cfg, backend="isa", cap=4) as wrapped:
        assert layers.next_index == spec.layers
        assert events == [
            ("converting_layer", {"layer_index": index, "layer_number": index + 1,
                                  "completed_layers": index, "total_layers": spec.layers})
            for index in range(spec.layers)]
        np.testing.assert_array_equal(
            wrapped.backend.read(0, 0, wrapped.image.nbytes),
            plain.backend.read(0, 0, plain.image.nbytes))
        plain_outputs = np.stack([plain.step(row) for row in inputs])
        outputs = np.stack([wrapped.step(row) for row in inputs])
        np.testing.assert_array_equal(outputs.view(np.uint32), plain_outputs.view(np.uint32))
        assert plain.pos == wrapped.pos == 4
        wrapped.reset()
        assert wrapped.pos == 0
        replay = np.stack([wrapped.step(row) for row in inputs])
        np.testing.assert_array_equal(replay.view(np.uint32), outputs.view(np.uint32))
        assert wrapped.pos == 4 and layers.next_index == spec.layers
        assert len(events) == spec.layers  # Execution/reset must not reread lazy weights.

    comparisons = [(outputs, expected), *zip(outputs, expected)]
    for actual, baseline in comparisons:
        metrics = isa_check.precision_metrics(actual, baseline)
        assert all(value is not None and np.isfinite(value) for value in metrics.values())
        assert metrics["reference_l2_norm"] > 0
        assert metrics["relative_l2"] >= 0
        assert -1 <= metrics["cosine_similarity"] <= 1
    assert isa_check.precision_metrics(outputs, expected) == isa_check.precision_metrics(
        plain_outputs, expected)


@pytest.mark.parametrize("formats", ["", "attn@0-3=int8,mlp@0-3=int8"])
def test_cli_passes_explicit_precision_rules_to_validation(isa_check, monkeypatch, tmp_path, formats):
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "completed", "reset_bit_exact": True,
                "precision_delta": {}, "accuracy_acceptance": "pending"}

    monkeypatch.setattr(isa_check, "run", run)
    argv = []
    for flag in ("model-dir", "embeddings", "reference-dir", "output-dir", "cache-dir"):
        argv.extend(["--" + flag, str(tmp_path / flag)])
    isa_check.main([*argv, "--formats", formats])
    assert len(calls) == 1
    assert calls[0][1] == {"cap": 1024, "formats": formats}
