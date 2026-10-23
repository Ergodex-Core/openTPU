"""Input-RVQ checker guards for the pinned F2 runner; no model is loaded."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu.llm import qwen3 as G
from tools.personaplex_mimi_full_input_native import _exact as native_exact
from tools.personaplex_mimi_input_rvq_native import _base_contract
from tools.personaplex_mimi_encoder_check import _metrics_by_frame
from tools.personaplex_mimi_input_rvq_check import (
    INPUT_RVQ_LIMITS, _checked_ids, _device_stream, _distance_diagnostics,
    _id_agreement, _joint_budget, numerical_acceptance,
    require_input_rvq_acceptance, run)
from tools.personaplex_pcm_precision_check import DATA


def _forbid_isa_open(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("A rejected request opened an ISA backend")
    monkeypatch.setattr(G, "IsaBackend", forbidden)


def test_wrong_target_rejected_before_baseline_checkpoint_or_backend(monkeypatch):
    _forbid_isa_open(monkeypatch)
    with pytest.raises(ValueError, match="exact shared target"):
        run("missing-codec", "missing-baseline", "missing-output",
            target_name="f2-hbm:burst200")


def test_original_native_entrypoint_exact_boundary_guard_without_torch():
    reference = np.array([[1.0, 2.0]], np.float32)
    native_exact(reference.copy(), reference, "boundary")
    with pytest.raises(AssertionError, match="boundary"):
        native_exact(reference.astype(np.float64), reference, "boundary")
    with pytest.raises(AssertionError, match="boundary"):
        native_exact(reference.reshape(2, 1), reference, "boundary")
    changed = reference.copy()
    changed[0, 1] = np.nextafter(changed[0, 1], np.float32(3))
    with pytest.raises(AssertionError, match="boundary"):
        native_exact(changed, reference, "boundary")


def test_input_rvq_source_hashes_match_pinned_base_contract():
    contract = json.loads((DATA / "mimi_input_rvq_contract.json").read_text())
    base = _base_contract(contract)
    assert contract["installed_source_sha256"] == base["installed_source_sha256"]
    assert all(len(value) == 64 and set(value) <= set("0123456789abcdef")
               for value in contract["installed_source_sha256"].values())
    altered = dict(contract)
    altered["installed_source_sha256"] = dict(contract["installed_source_sha256"])
    altered["installed_source_sha256"]["quantization/core_vq.py"] = (
        altered["installed_source_sha256"]["quantization/core_vq.py"][1:])
    with pytest.raises(ValueError, match="pinned quantizer"):
        _base_contract(altered)


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


@pytest.mark.parametrize("corrupt", [
    lambda a: a.astype(np.int32),
    lambda a: a.reshape(1, 16),
    lambda a: np.where(np.indices(a.shape)[0] == 1, np.float32(1.5), a),
    lambda a: np.where(np.indices(a.shape)[0] == 1, np.float32(np.nan), a),
    lambda a: np.where(np.indices(a.shape)[0] == 1, np.float32(2048), a),
])
def test_resident_ids_reject_wrong_dtype_shape_fraction_nonfinite_and_range(corrupt):
    with pytest.raises(ValueError, match="finite integral FP32"):
        _checked_ids(corrupt(np.zeros((2, 8), np.float32)))
    assert _checked_ids(np.zeros((2, 8), np.float32)).dtype == np.int64


def test_exact_id_gate_is_independent_for_each_frame_and_reference_path():
    device = np.zeros((2, 8), np.int64)
    accumulated = device.copy()
    accumulated[1, 7] = 1
    same_input = device.copy()
    assert _id_agreement(device, accumulated, "accumulated") == dict(
        label="accumulated", passed=False, per_frame=[True, False],
        differing_ids_per_frame=[0, 1])
    assert _id_agreement(device, same_input, "same_input")["passed"] is True

    float_gates = {"projected_semantic": {"passed": True},
                   "residual_before_acoustic": {"passed": True}}
    accumulated_float_gates = {"projected_semantic": {"passed": True},
                               "residual_before_acoustic": {"passed": True}}
    result = dict(accumulated_ids=_id_agreement(device, accumulated, "accumulated"),
                  same_isa_downsample_ids=_id_agreement(device, same_input, "same_input"),
                  accumulated_float_acceptance=accumulated_float_gates,
                  same_input_float_acceptance=float_gates)
    with pytest.raises(RuntimeError, match="accumulated_ids"):
        require_input_rvq_acceptance(result)
    result["accumulated_ids"]["passed"] = True
    result["same_isa_downsample_ids"]["passed"] = False
    with pytest.raises(RuntimeError, match="same_isa_downsample_ids"):
        require_input_rvq_acceptance(result)
    result["same_isa_downsample_ids"]["passed"] = True
    float_gates["residual_before_acoustic"]["passed"] = False
    with pytest.raises(RuntimeError, match="residual_before_acoustic"):
        require_input_rvq_acceptance(result)
    float_gates["residual_before_acoustic"]["passed"] = True
    accumulated_float_gates["projected_semantic"]["passed"] = False
    with pytest.raises(RuntimeError, match="accumulated_float_acceptance.projected_semantic"):
        require_input_rvq_acceptance(result)


def test_float_gate_checks_each_frame_peak_and_both_branches():
    reference = np.ones((2, 2, 256), np.float32)
    actual = reference.copy()
    actual[1, 1] += np.float32(1.2e-5)
    metrics = _metrics_by_frame(actual[:, 1], reference[:, 1])
    assert metrics["aggregate"]["rmse"] < INPUT_RVQ_LIMITS["rmse"]
    assert metrics["per_frame"][1]["rmse"] > INPUT_RVQ_LIMITS["rmse"]
    assert numerical_acceptance(metrics, INPUT_RVQ_LIMITS)["passed"] is False
    actual = reference.copy()
    actual[1, 0, 0] += np.float32(1.2e-4)
    metrics = _metrics_by_frame(actual[:, 0], reference[:, 0])
    assert metrics["per_frame"][1]["max_absolute_error"] > INPUT_RVQ_LIMITS["max_absolute_error"]
    assert numerical_acceptance(metrics, INPUT_RVQ_LIMITS)["passed"] is False


def test_distance_diagnostics_are_nonnegative_and_descriptive_only():
    squared = np.full((2, 8, 2048), np.float32(4), np.float32)
    euclidean = np.full_like(squared, np.float32(2))
    margins = np.full((2, 8), np.float32(0.5), np.float32)
    report = _distance_diagnostics(squared, euclidean, margins)
    assert report["sqrt_device_minus_native_euclidean_max"] == 0
    assert report["native_winner_margin_per_frame"] == [0.5, 0.5]
    assert "diagnostic" in report["scope"]
    squared[0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        _distance_diagnostics(squared, euclidean, margins)


def test_joint_reservation_requires_60_slots_one_table_and_no_overlap():
    cfg = SimpleNamespace(DRAM_BYTES=1 << 20)
    slots = [dict(name=f"program{i}", base=128 * i, bytes=128,
                  kind="program", region="program") for i in range(60)]
    table = dict(name="mimi_tables", base=128 * 60, bytes=128,
                 kind="immutable", region="weights")
    image = SimpleNamespace(prefix_spans=slots[:-1] + [table], spans=slots[-1:],
                            tables_at=table["base"], budget=lambda: {"program_words": [128]})
    assert _joint_budget(image, cfg)["program_slot_count"] == 60
    image.prefix_spans = slots[:-1]
    with pytest.raises(AssertionError, match="60 joint slots and one shared"):
        _joint_budget(image, cfg)
    image.prefix_spans = slots[:-1] + [table, dict(table, base=table["base"] + 128)]
    with pytest.raises(AssertionError, match="60 joint slots and one shared"):
        _joint_budget(image, cfg)
    image.prefix_spans = slots[:-1] + [table]
    image.spans = [dict(slots[-1], base=slots[-2]["base"])]
    with pytest.raises(ValueError, match="overlap"):
        _joint_budget(image, cfg)


def test_33_program_stream_rejects_implicit_backend_feedback():
    program = SimpleNamespace(run_args=[])
    encoder = SimpleNamespace(input_at=0x1000, compile=lambda: [program] * 14)
    transformer = SimpleNamespace(compile=lambda: [program] * 9)
    downsample = SimpleNamespace(compile=lambda: [program])
    rvq = SimpleNamespace(compile=lambda: [program] * 9)

    class FeedbackBackend:
        def write(self, *_args):
            pass
        def read(self, *_args):
            return np.zeros(4, np.uint8)
        def run(self, _program):
            self.read(0, 0, 4)

    receipt = {"frame_progress": []}
    with pytest.raises(AssertionError, match="interstage host transfer"):
        _device_stream(encoder, transformer, downsample, rvq, FeedbackBackend(),
                       np.zeros((2, 1920), np.float32), receipt, lambda: None,
                       "guard_probe")


def test_original_native_geometry_guard_uses_real_profile_spec():
    from dataclasses import replace
    from opentpu.llm.personaplex_mimi_decoder import MimiDecoderSpec
    from tools.personaplex_mimi_full_input_native import _checked_transformer_spec
    spec = MimiDecoderSpec(precision="fp32")
    _checked_transformer_spec(spec)
    for field, value in (("hidden", 256), ("n_heads", 4), ("layers", 4), ("context", 40)):
        with pytest.raises(ValueError, match="geometry"):
            _checked_transformer_spec(replace(spec, **{field: value}))
