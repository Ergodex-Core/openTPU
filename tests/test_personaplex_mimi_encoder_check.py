"""Preflight and acceptance checks for the pinned input encoder experiment.

These tests run on the CPU ISA builder, never on a local Mac model runtime.
"""
import numpy as np
import pytest

from tools.personaplex_mimi_encoder_check import (
    _metrics_by_frame, numerical_acceptance, run, ENCODER_LIMITS)


def test_wrong_target_is_rejected_before_checkpoint_or_baseline_access():
    with pytest.raises(ValueError, match="exact shared target"):
        run("missing-codec", "missing-baseline", "missing-output",
            target_name="f2-hbm:burst200")


def test_existing_output_is_preserved_before_checkpoint_or_baseline_access(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "preserved-receipt.json"
    marker.write_text("leave intact")
    with pytest.raises(FileExistsError, match="must be new"):
        run(tmp_path / "missing-codec", tmp_path / "missing-baseline", output)
    assert marker.read_text() == "leave intact"


def test_encoder_acceptance_requires_aggregate_and_every_frame():
    reference = np.ones((2, 2, 512), np.float32)
    actual = reference.copy()
    actual[1, 0, 0] += np.float32(1.1e-4)
    metrics = _metrics_by_frame(actual, reference)
    assert metrics["aggregate"]["relative_l2"] < ENCODER_LIMITS["relative_l2"]
    assert metrics["aggregate"]["rmse"] < ENCODER_LIMITS["rmse"]
    assert metrics["aggregate"]["max_absolute_error"] > ENCODER_LIMITS["max_absolute_error"]
    assert numerical_acceptance(metrics)["passed"] is False

    # A frame-local RMS miss must fail even when the aggregate RMS passes.
    actual = reference.copy()
    actual[1, :, :] += np.float32(1.2e-5)
    metrics = _metrics_by_frame(actual, reference)
    assert metrics["aggregate"]["rmse"] < ENCODER_LIMITS["rmse"]
    assert metrics["per_frame"][1]["rmse"] > ENCODER_LIMITS["rmse"]
    assert numerical_acceptance(metrics)["passed"] is False
    assert numerical_acceptance(_metrics_by_frame(reference, reference))["passed"] is True
