"""Guard saved temporal evidence and separate device selection from CPU diagnostics."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from tools import personaplex_token_check as tool


def test_head_comparison_uses_same_normalized_hidden_and_preserves_first_id_ties():
    hidden = np.array([[2, 0], [0, 3]], np.float32)
    head = np.array([[1, 0], [0, 1], [1, 1]], np.float32)
    logits = hidden @ head.T
    report = tool.compare_head(hidden, logits, np.array([0, 1]), head)
    assert report["source_head_top1_ids"] == [0, 1]
    assert report["all_source_head_top1_match"]
    assert report["logit_precision_delta"]["relative_l2"] == 0
    assert report["quality_accepted"] is False


def test_wrong_device_token_is_rejected_instead_of_replaced_by_host_selection():
    with pytest.raises(AssertionError, match="Device greedy IDs"):
        tool.compare_head(np.array([[2, 0]], np.float32), np.array([[2, 0]], np.float32),
                          np.array([1]), np.eye(2, dtype=np.float32))


def test_source_head_disagreement_is_reported_without_overwriting_device_result():
    report = tool.compare_head(np.array([[1, 0]], np.float32), np.array([[0, 2]], np.float32),
                               np.array([1]), np.eye(2, dtype=np.float32))
    assert report["source_head_top1_ids"] == [0]
    assert report["source_head_top1_matches"] == [False]
    assert report["all_source_head_top1_match"] is False


@pytest.fixture
def paths(tmp_path, monkeypatch):
    model, temporal, cache, output = (tmp_path / name for name in ("model", "temporal", "cache", "out"))
    model.mkdir()
    temporal.mkdir()
    inputs = tmp_path / "inputs.npy"
    np.save(inputs, np.zeros((2, 4096), np.float32))
    np.save(temporal / "isa_hidden.npy", np.ones((2, 4096), np.float32))
    baseline = {"status": "completed", "reset_bit_exact": True,
                "temporal_spec": asdict(tool.Spec()), "image": {"cap": 128},
                "precision": {"formats": ""}, "input": {"sha256": tool.sha256(inputs)},
                "output": {"sha256": tool.sha256(temporal / "isa_hidden.npy")}}
    (temporal / "isa_check.json").write_text(json.dumps(baseline))

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid saved evidence must fail before checkpoint access")

    monkeypatch.setattr(tool, "MLXWeights", forbidden)
    return model, inputs, temporal, output, cache, baseline


@pytest.mark.parametrize("fault", ["unfinished", "reset", "spec", "cap", "formats", "input", "hidden"])
def test_invalid_saved_baseline_fails_before_checkpoint_access_or_output(paths, fault):
    model, inputs, temporal, output, cache, baseline = paths
    if fault == "unfinished":
        baseline["status"] = "running"
    elif fault == "reset":
        baseline["reset_bit_exact"] = False
    elif fault == "spec":
        baseline["temporal_spec"]["layers"] = 1
    elif fault == "cap":
        baseline["image"]["cap"] = 1024
    elif fault == "formats":
        baseline["precision"]["formats"] = "attn=int8"
    elif fault == "input":
        np.save(inputs, np.ones((2, 4096), np.float32))
    else:
        np.save(temporal / "isa_hidden.npy", np.zeros((2, 4096), np.float32))
    (temporal / "isa_check.json").write_text(json.dumps(baseline))
    with pytest.raises(ValueError):
        tool.run(model, inputs, temporal, output, cache)
    assert not output.exists() and not cache.exists()


@pytest.mark.parametrize("protected", ["model", "inputs", "temporal", "output", "ancestor"])
def test_cache_cannot_manage_protected_evidence(paths, protected):
    model, inputs, temporal, output, cache, _ = paths
    cache = {"model": model, "inputs": inputs, "temporal": temporal,
             "output": output, "ancestor": model.parent}[protected]
    with pytest.raises(ValueError, match="Cache must be isolated"):
        tool.run(model, inputs, temporal, output, cache)
    assert not output.exists()


def test_existing_output_receipt_is_preserved(paths):
    model, inputs, temporal, output, cache, _ = paths
    output.mkdir()
    evidence = output / "token_check.json"
    evidence.write_text("existing run")
    with pytest.raises(FileExistsError):
        tool.run(model, inputs, temporal, output, cache)
    assert evidence.read_text() == "existing run" and not cache.exists()
