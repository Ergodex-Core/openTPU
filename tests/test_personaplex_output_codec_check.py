"""Output-codec preflight guards; run only on the EC2 CPU environment."""
import pytest
import numpy as np

from tools.personaplex_output_codec_check import run, _decoder_layer_metrics


@pytest.mark.parametrize("mode", ["fp16", None])
def test_invalid_precision_rejected_before_touching_checkpoint(mode):
    with pytest.raises(ValueError, match="exact shared target"):
        run("missing-codec", "missing-baseline", "missing-output", decoder_precision=mode)


def test_other_image_profile_rejected_before_touching_checkpoint():
    with pytest.raises(ValueError, match="exact shared target"):
        run("missing-codec", "missing-baseline", "missing-output",
            decoder_precision="fp32", target_name="f2-hbm:burst200")


def test_completed_output_directory_is_never_reused(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "previous-receipt.json"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError, match="must be new"):
        run(tmp_path / "codec", tmp_path / "baseline", output)
    assert marker.read_text() == "preserve"


def test_native_decoder_layers_are_frame_first():
    reference = np.ones((2, 3, 2, 4), np.float32)
    actual = reference.copy()
    actual[:, 0] += 0.25
    actual[:, 1] += 0.5
    actual[:, 2] += 0.75
    result = _decoder_layer_metrics(actual, reference)
    assert len(result) == 3
    for layer, metrics in enumerate(result):
        expected = 0.25 * (layer + 1)
        assert metrics["aggregate"]["relative_l2"] == pytest.approx(expected)
        assert len(metrics["per_frame"]) == 2
        assert all(frame["relative_l2"] == pytest.approx(expected)
                   for frame in metrics["per_frame"])
