"""Diagnostic receipts must reject approximate, nonfinite or unsafe inputs."""
import numpy as np
import pytest

from tools.personaplex_components_check import require_exact, run


def test_reference_comparison_requires_fp32_bits_including_signed_zero():
    value = np.array([0, 1, -3], np.float32)
    require_exact(value, value.copy(), "same")
    changed = value.copy()
    changed[0] = -0.0
    with pytest.raises(AssertionError, match="bits differ"):
        require_exact(value, changed, "zero sign")
    changed = value.copy()
    changed[1] = np.nextafter(changed[1], np.float32(2))
    with pytest.raises(AssertionError):
        require_exact(value, changed, "one ULP")


@pytest.mark.parametrize("bad", [np.array([0, 1, -3], np.float64),
                                 np.array([0, np.nan, -3], np.float32),
                                 np.array([[0, 1, -3]], np.float32)])
def test_reference_comparison_rejects_invalid_diagnostic_arrays(bad):
    with pytest.raises(AssertionError):
        require_exact(np.array([0, 1, -3], np.float32), bad, "bad")


def test_checker_refuses_existing_output_before_checkpoint_reads(tmp_path):
    output = tmp_path / "prior_receipt"
    output.mkdir()
    preserved = output / "components_check.json"
    preserved.write_text("preserve this evidence")
    with pytest.raises(FileExistsError):
        run(tmp_path / "missing_model", tmp_path / "missing_voice.npy", output)
    assert preserved.read_text() == "preserve this evidence"
