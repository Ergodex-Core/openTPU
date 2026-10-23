"""Pinned input-transformer checker guards and frame-first diagnostic tests.

Run on the CPU ISA builder. Local Mac verification is AST-only.
"""
import numpy as np
import pytest

from opentpu.llm.personaplex_mimi_decoder import native_ring_positions
from tools.personaplex_mimi_input_transformer_check import (
    _check_native_positions, _metrics_by_frame, _transformer_layer_metrics,
    numerical_acceptance, ENCODER_LIMITS, run)


def test_wrong_target_rejected_before_baseline_or_checkpoint_access():
    with pytest.raises(ValueError, match="exact shared target"):
        run("missing-codec", "missing-baseline", "missing-output",
            target_name="f2-hbm:burst200")


def test_existing_output_preserved_before_baseline_or_checkpoint_access(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "previous-receipt.json"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError, match="must be new"):
        run(tmp_path / "missing-codec", tmp_path / "missing-baseline", output)
    assert marker.read_text() == "preserve"


def test_layer_errors_use_frame_then_layer_axes():
    reference = np.ones((2, 8, 2, 4), np.float32)
    actual = reference.copy()
    for layer in range(8):
        actual[0, layer] += np.float32((layer + 1) * 0.01)
        actual[1, layer] += np.float32((layer + 1) * 0.02)
    measured = _transformer_layer_metrics(actual, reference)
    assert [entry["layer"] for entry in measured] == list(range(8))
    for layer, entry in enumerate(measured):
        per_frame = entry["metrics"]["per_frame"]
        assert per_frame[0]["relative_l2"] == pytest.approx((layer + 1) * 0.01, abs=1e-6)
        assert per_frame[1]["relative_l2"] == pytest.approx((layer + 1) * 0.02, abs=1e-6)


def test_transformer_gate_requires_each_frame_as_well_as_aggregate():
    reference = np.ones((2, 2, 512), np.float32)
    actual = reference.copy()
    actual[1] += np.float32(1.2e-5)
    metrics = _metrics_by_frame(actual, reference)
    assert metrics["aggregate"]["rmse"] < ENCODER_LIMITS["rmse"]
    assert metrics["per_frame"][1]["rmse"] > ENCODER_LIMITS["rmse"]
    assert numerical_acceptance(metrics)["passed"] is False
    assert numerical_acceptance(_metrics_by_frame(reference, reference))["passed"] is True


def test_native_positions_check_all_layers_and_frames():
    positions = [[native_ring_positions(2 * (frame + 1), 4) for frame in range(2)]
                 for _ in range(8)]
    _check_native_positions({"positions": positions}, 4, 2)
    positions[7][1] = positions[7][1].copy()
    positions[7][1][0] = -99
    with pytest.raises(AssertionError, match="layer 7, frame 1"):
        _check_native_positions({"positions": positions}, 4, 2)
