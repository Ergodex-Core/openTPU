"""Fake-file preflight guards for the bounded decoder checkpoint checker.

Run on the EC2 CPU test runner; no checkpoint tensor or device is opened here.
"""
import hashlib
import json

import pytest

from tools import personaplex_mimi_decoder_check as check


def _fake_rvq(tmp_path):
    folder = tmp_path / "completed-rvq"
    folder.mkdir()
    saved = folder / "isa_output.npy"
    saved.write_bytes(b"synthetic saved latent")
    receipt = {
        "status": "completed", "reset_bit_exact": True,
        "source_lookup_sums_bit_exact": True,
        "execution_target": {"variant": "p4rl8a64c150", "arch_hash": "dd6b92a6"},
        "checkpoint_revision": "synthetic-pinned-revision",
        "arrays": {"isa_output": {"sha256": hashlib.sha256(saved.read_bytes()).hexdigest()}},
    }
    path = folder / "mimi_check.json"
    path.write_text(json.dumps(receipt))
    return folder, path, saved, receipt


def test_completed_exact_profile_rvq_baseline_hashes_receipt_and_saved_arrays(tmp_path):
    folder, path, saved, receipt = _fake_rvq(tmp_path)
    actual, hashes = check.baseline_rvq(folder)
    assert actual == receipt
    assert hashes == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest(),
        str(saved): hashlib.sha256(saved.read_bytes()).hexdigest(),
    }
    saved.write_bytes(b"tampered saved latent")
    with pytest.raises(ValueError, match="Saved RVQ baseline changed: isa_output"):
        check.baseline_rvq(folder)


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("reset_bit_exact", False),
    ("source_lookup_sums_bit_exact", False),
])
def test_decoder_baseline_requires_completed_source_compared_reset(tmp_path, field, value):
    folder, path, _, receipt = _fake_rvq(tmp_path)
    path.write_text(json.dumps({**receipt, field: value}))
    with pytest.raises(ValueError, match="Completed native-compared exact-profile"):
        check.baseline_rvq(folder)


@pytest.mark.parametrize("field,value", [
    ("variant", "p4rl8a64c100"), ("arch_hash", "different-architecture"),
])
def test_decoder_baseline_requires_exact_common_target(tmp_path, field, value):
    folder, path, _, receipt = _fake_rvq(tmp_path)
    receipt["execution_target"][field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Completed native-compared exact-profile"):
        check.baseline_rvq(folder)


@pytest.mark.parametrize("frames,target_name", [
    (1, check.COMMON_TARGET), (5, check.COMMON_TARGET),
    (True, check.COMMON_TARGET), (2.0, check.COMMON_TARGET),
    (2, "f2-hbm:p4rl8a64c100"),
])
def test_invalid_frame_count_or_target_rejected_before_baseline_and_payload(
        tmp_path, monkeypatch, frames, target_name):
    # Missing all receipts/contracts/checkpoints proves the argument guard fires
    # before metadata or model payload access. It also leaves no output directory.
    monkeypatch.setattr(check, "DATA", tmp_path / "absent-contracts")
    output = tmp_path / "new-output"
    with pytest.raises(ValueError, match="exact shared target and two to four"):
        check.run(tmp_path / "model", tmp_path / "codec", tmp_path / "tokens",
                  tmp_path / "conditioning", tmp_path / "stack", tmp_path / "rvq",
                  output, frames=frames, target_name=target_name)
    assert not output.exists()
