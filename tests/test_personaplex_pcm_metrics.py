"""Pure-array waveform diagnostic regressions; run on the EC2 CPU runner."""

import json

import numpy as np
import pytest

from tools.personaplex_pcm_metrics import waveform_metrics


def test_quiet_frame_has_large_relative_error_but_fixed_absolute_error():
    reference = np.stack((np.full(8, 1e-5, np.float32),
                          np.full(8, 0.5, np.float32)))
    actual = reference + np.float32(1e-3)
    result = waveform_metrics(actual, reference)
    quiet, loud = result["per_frame"]
    assert quiet["relative_l2"] > 90
    assert loud["relative_l2"] < 0.01
    assert quiet["error_rms_full_scale"] == pytest.approx(1e-3, rel=1e-4)
    assert loud["error_rms_full_scale"] == pytest.approx(1e-3, rel=1e-4)
    assert quiet["peak_absolute_error_full_scale"] == pytest.approx(1e-3, rel=1e-4)
    assert result["aggregate"]["samples"] == 16
    assert quiet["raw_rms_gain_ratio_diagnostic_only"] > 1


def test_perfect_match_and_zero_reference_have_finite_json():
    reference = np.array([[0.25, -0.25], [0.0, 0.0]], np.float32)
    actual = reference.copy()
    result = waveform_metrics(actual, reference)
    perfect, both_zero = result["per_frame"]
    assert perfect["zero_error_rms"] is True
    assert perfect["zero_reference_rms"] is False
    assert perfect["snr_db"] is None and perfect["error_rms_dbfs"] is None
    assert perfect["error_rms_full_scale"] == 0
    assert both_zero["zero_error_rms"] is True
    assert both_zero["zero_reference_rms"] is True
    assert both_zero["zero_actual_rms"] is True
    assert both_zero["signal_rms_dbfs"] is None
    assert both_zero["relative_l2"] is None
    assert both_zero["raw_rms_gain_ratio_diagnostic_only"] is None
    json.dumps(result, allow_nan=False)


def test_nonzero_error_against_zero_reference_reports_absolute_units():
    reference = np.zeros((1, 4), np.float32)
    actual = np.ones((1, 4), np.float32)
    item = waveform_metrics(actual, reference)["aggregate"]
    assert item["zero_reference_rms"] is True
    assert item["zero_error_rms"] is False
    assert item["snr_db"] is None and item["signal_rms_dbfs"] is None
    assert item["error_rms_full_scale"] == 1
    assert item["error_rms_dbfs"] == 0
    assert item["relative_l2"] is None


@pytest.mark.parametrize("actual,reference", [
    (np.ones((1, 4), np.float64), np.ones((1, 4), np.float32)),
    (np.ones((1, 4), np.float32), np.ones((1, 4), np.float64)),
    (np.ones((4,), np.float32), np.ones((4,), np.float32)),
    (np.ones((0, 4), np.float32), np.ones((0, 4), np.float32)),
    (np.ones((1, 0), np.float32), np.ones((1, 0), np.float32)),
    (np.ones((1, 4), np.float32), np.ones((2, 4), np.float32)),
    (np.array([[np.nan]], np.float32), np.ones((1, 1), np.float32)),
    (np.ones((1, 1), np.float32), np.array([[np.inf]], np.float32)),
])
def test_invalid_dtype_shape_or_nonfinite_input_rejected(actual, reference):
    with pytest.raises(ValueError, match="finite FP32 equal"):
        waveform_metrics(actual, reference)


def test_out_of_range_raw_samples_remain_unchanged():
    actual = np.array([[2.0, -3.0]], np.float32)
    reference = np.array([[0.5, -0.5]], np.float32)
    actual_before, reference_before = actual.copy(), reference.copy()
    item = waveform_metrics(actual, reference)["aggregate"]
    assert item["peak_absolute_error_full_scale"] == 2.5
    assert item["actual_rms_full_scale"] > 2
    assert item["raw_rms_gain_ratio_diagnostic_only"] > 5
    np.testing.assert_array_equal(actual, actual_before)
    np.testing.assert_array_equal(reference, reference_before)
