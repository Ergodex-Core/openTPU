"""Bounded ISA-check input guards; no checkpoint, simulator or device required."""
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
