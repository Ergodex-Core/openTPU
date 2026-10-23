"""Small fake-receipt guards for the bounded Mimi PCM checkpoint checker.

Run only on the EC2 CPU test runner; these tests require no checkpoint payload.
"""
import hashlib
import json

import numpy as np
import pytest

from tools import personaplex_mimi_pcm_check as check


def _fake_decoder(tmp_path):
    folder = tmp_path / "completed-decoder"
    folder.mkdir()
    ancestry = tmp_path / "upstream-receipt.json"
    ancestry.write_text('{"status":"completed"}')
    arrays = {}
    for name in ("isa_ids", "isa_rvq", "isa_upsampled", "isa_output", "isa_layers"):
        value = np.arange(2, dtype=np.float32)
        path = folder / (name + ".npy")
        np.save(path, value, allow_pickle=False)
        arrays[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "shape": list(value.shape), "dtype": str(value.dtype)}
    receipt = {
        "status": "completed", "backend": "isa", "hardware_used": False,
        "reset_bit_exact": True, "final_decoder_raw_kv_reset_bit_exact": True,
        "final_depth_raw_kv_reset_bit_exact": True,
        "upsample_identical_latent_bit_exact": True,
        "checkpoint_revision": check.PINNED_REVISION,
        "execution_target": {"variant": "p4rl8a64c150", "arch_hash": "dd6b92a6"},
        "frames": 2, "arrays": arrays,
        "baseline_sha256": {str(ancestry): hashlib.sha256(ancestry.read_bytes()).hexdigest()},
    }
    receipt_path = folder / "mimi_decoder_check.json"
    receipt_path.write_text(json.dumps(receipt))
    return folder, receipt_path, ancestry, receipt


def test_decoder_baseline_checks_every_array_and_ancestry(tmp_path):
    folder, receipt_path, ancestry, receipt = _fake_decoder(tmp_path)
    actual, hashes = check.baseline_decoder(folder)
    assert actual == receipt
    assert hashes[str(receipt_path)] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert hashes[str(ancestry)] == hashlib.sha256(ancestry.read_bytes()).hexdigest()
    assert len(hashes) == 7
    saved = folder / "isa_output.npy"
    saved.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Saved decoder baseline changed: isa_output"):
        check.baseline_decoder(folder)


def test_decoder_baseline_rejects_changed_ancestry(tmp_path):
    folder, _, ancestry, _ = _fake_decoder(tmp_path)
    ancestry.write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="Decoder ancestry baseline changed"):
        check.baseline_decoder(folder)


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("backend", "board"), ("hardware_used", True),
    ("reset_bit_exact", False), ("final_decoder_raw_kv_reset_bit_exact", False),
    ("final_depth_raw_kv_reset_bit_exact", False),
    ("upsample_identical_latent_bit_exact", False),
    ("checkpoint_revision", "other"), ("frames", 1),
])
def test_decoder_baseline_requires_completed_software_and_reset(tmp_path, field, value):
    folder, receipt_path, _, receipt = _fake_decoder(tmp_path)
    receipt_path.write_text(json.dumps({**receipt, field: value}))
    with pytest.raises(ValueError, match="Completed exact-profile native-compared"):
        check.baseline_decoder(folder)


@pytest.mark.parametrize("field,value", [
    ("variant", "p4rl8a64c100"), ("arch_hash", "different-architecture"),
])
def test_decoder_baseline_requires_shared_target(tmp_path, field, value):
    folder, receipt_path, _, receipt = _fake_decoder(tmp_path)
    receipt["execution_target"][field] = value
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Completed exact-profile native-compared"):
        check.baseline_decoder(folder)


def test_decoder_baseline_requires_all_upstream_arrays(tmp_path):
    folder, receipt_path, _, receipt = _fake_decoder(tmp_path)
    del receipt["arrays"]["isa_layers"]
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="lacks required saved arrays"):
        check.baseline_decoder(folder)


def test_output_isolated_from_inputs_and_existing_paths(tmp_path):
    source = tmp_path / "baseline"
    source.mkdir()
    (source / "receipt.json").write_text("{}")
    assert check.isolated_output(tmp_path / "fresh", source) == (tmp_path / "fresh").resolve()
    with pytest.raises(ValueError, match="isolated"):
        check.isolated_output(source / "new", source)
    with pytest.raises(FileExistsError, match="must be new"):
        check.isolated_output(tmp_path, source)
    with pytest.raises(FileExistsError, match="must be new"):
        check.isolated_output(source, source)


def test_raw_pcm_statistics_count_unclipped_excursions():
    samples = np.zeros((2, 1920), np.float32)
    samples[0, 0] = 1.25
    samples[1, 0] = -1.5
    result = check.pcm_stats(samples)
    assert result["samples"] == 3840
    assert result["abs_gt_1_samples"] == 2
    assert [frame["abs_gt_1_samples"] for frame in result["per_frame"]] == [1, 1]
    assert result["min"] == -1.5 and result["max"] == 1.25
    with pytest.raises(ValueError, match="finite raw FP32"):
        check.pcm_stats(samples.astype(np.float64))
    samples[0, 1] = np.inf
    with pytest.raises(ValueError, match="finite raw FP32"):
        check.pcm_stats(samples)


@pytest.mark.parametrize("frames,target_name", [
    (1, check.COMMON_TARGET), (5, check.COMMON_TARGET),
    (True, check.COMMON_TARGET), (2.0, check.COMMON_TARGET),
    (2, "f2-hbm:p4rl8a64c100"),
])
def test_invalid_frame_or_target_rejected_before_checkpoint_access(tmp_path, frames, target_name):
    output = tmp_path / "new-output"
    with pytest.raises(ValueError, match="exact shared target and two to four"):
        check.run(tmp_path / "model", tmp_path / "codec", tmp_path / "tokens",
                  tmp_path / "conditioning", tmp_path / "stack", tmp_path / "rvq",
                  tmp_path / "decoder", output, frames=frames, target_name=target_name)
    assert not output.exists()
