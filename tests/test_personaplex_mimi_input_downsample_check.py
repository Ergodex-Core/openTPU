"""Checker guards and per-frame acceptance tests for the EC2 runner.

No checkpoint, model inference, or physical device is needed for these guards.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu.llm import qwen3 as G
from tools.personaplex_mimi_encoder_check import _metrics_by_frame
from tools.personaplex_mimi_input_downsample_check import (
    DOWNSAMPLE_LIMITS, _device_stream, _joint_budget, numerical_acceptance,
    require_downsample_acceptance, run)


def _forbid_isa_open(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("A rejected request opened an ISA backend")
    monkeypatch.setattr(G, "IsaBackend", forbidden)


def test_wrong_target_rejected_before_baseline_checkpoint_or_backend(monkeypatch):
    _forbid_isa_open(monkeypatch)
    with pytest.raises(ValueError, match="exact shared target"):
        run("missing-codec", "missing-baseline", "missing-output",
            target_name="f2-hbm:burst200")


def test_existing_output_preserved_before_baseline_checkpoint_or_backend(
        monkeypatch, tmp_path):
    _forbid_isa_open(monkeypatch)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "previous-receipt.json"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError, match="must be new"):
        run(tmp_path / "missing-codec", tmp_path / "missing-baseline", output)
    assert marker.read_text() == "preserve"


def test_downsample_gate_checks_each_frame_and_peak_not_only_aggregate():
    reference = np.ones((2, 1, 512), np.float32)
    actual = reference.copy()
    actual[1] += np.float32(1.2e-5)
    metrics = _metrics_by_frame(actual, reference)
    assert metrics["aggregate"]["rmse"] < DOWNSAMPLE_LIMITS["rmse"]
    assert metrics["per_frame"][1]["rmse"] > DOWNSAMPLE_LIMITS["rmse"]
    assert numerical_acceptance(metrics, DOWNSAMPLE_LIMITS)["passed"] is False

    actual = reference.copy()
    actual[1, 0, 0] += np.float32(1.2e-4)
    metrics = _metrics_by_frame(actual, reference)
    assert metrics["aggregate"]["rmse"] < DOWNSAMPLE_LIMITS["rmse"]
    assert metrics["per_frame"][1]["max_absolute_error"] > DOWNSAMPLE_LIMITS["max_absolute_error"]
    assert numerical_acceptance(metrics, DOWNSAMPLE_LIMITS)["passed"] is False
    assert numerical_acceptance(_metrics_by_frame(reference, reference),
                                DOWNSAMPLE_LIMITS)["passed"] is True


@pytest.mark.parametrize("accumulated,same_input,failed", [
    (False, True, "accumulated_acceptance"),
    (True, False, "same_isa_tf_input_acceptance"),
    (False, False, "accumulated_acceptance, same_isa_tf_input_acceptance"),
])
def test_downsample_requires_both_acceptance_gates(accumulated, same_input, failed):
    result = {"accumulated_acceptance": {"passed": accumulated},
              "same_isa_tf_input_acceptance": {"passed": same_input}}
    with pytest.raises(RuntimeError, match=failed):
        require_downsample_acceptance(result)
    result["accumulated_acceptance"]["passed"] = True
    result["same_isa_tf_input_acceptance"]["passed"] = True
    require_downsample_acceptance(result)


def test_joint_reservation_requires_51_disjoint_program_slots():
    cfg = SimpleNamespace(DRAM_BYTES=1 << 20)
    spans = [dict(name=f"program{i}", base=128 * i, bytes=128,
                  kind="program", region="program") for i in range(51)]
    image = SimpleNamespace(prefix_spans=spans[:-1], spans=spans[-1:],
                            budget=lambda: {"program_words": [128]})
    budget = _joint_budget(image, cfg)
    assert budget["program_slot_count"] == 51
    image.spans = []
    with pytest.raises(AssertionError, match="51"):
        _joint_budget(image, cfg)
    image.spans = [dict(spans[-1], base=spans[-2]["base"])]
    with pytest.raises(ValueError, match="overlap"):
        _joint_budget(image, cfg)


def test_24_program_stream_rejects_implicit_backend_feedback():
    program = SimpleNamespace(run_args=[])
    encoder = SimpleNamespace(input_at=0x1000, compile=lambda: [program] * 14)
    transformer = SimpleNamespace(compile=lambda: [program] * 9)
    downsample = SimpleNamespace(compile=lambda: [program])

    class FeedbackBackend:
        def write(self, *_args):
            pass
        def read(self, *_args):
            return np.zeros(4, np.uint8)
        def run(self, _program):
            self.read(0, 0, 4)

    receipt = {"frame_progress": []}
    with pytest.raises(AssertionError, match="interstage host transfer"):
        _device_stream(encoder, transformer, downsample, FeedbackBackend(),
                       np.zeros((2, 1920), np.float32), receipt, lambda: None,
                       "guard_probe")
