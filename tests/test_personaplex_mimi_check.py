"""Small preflight regressions for the bounded Mimi checkpoint checker.

The EC2 test runner uses synthetic metadata and never opens a real checkpoint.
"""
import hashlib
import json
import struct

import pytest

from tools import personaplex_mimi_check as check


def _fake_codec(tmp_path):
    header = json.dumps({"quantizer.first": {"dtype": "F32"},
                         "quantizer.rest": {"dtype": "F32"}}).encode()
    path = tmp_path / "small.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"small fake payload")
    contract = {"codec_header": {
        "file_bytes": path.stat().st_size,
        "header_bytes": len(header),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "tensor_count": 2,
    }}
    return path, contract


def test_codec_header_accepts_exact_fake_metadata_and_rejects_size_hash_count(tmp_path):
    path, contract = _fake_codec(tmp_path)
    stat = check.codec_header(path, contract)
    assert stat == {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}

    for field, altered in (("file_bytes", path.stat().st_size + 1),
                           ("header_bytes", contract["codec_header"]["header_bytes"] + 1),
                           ("header_sha256", "0" * 64),
                           ("tensor_count", 3)):
        bad = {"codec_header": {**contract["codec_header"], field: altered}}
        with pytest.raises(ValueError, match="Cached codec"):
            check.codec_header(path, bad)

    # Keep the file length and JSON tensor count unchanged: the header hash
    # still pins the exact source metadata before any tensor is loaded.
    raw = path.read_bytes().replace(b"first", b"other")
    assert len(raw) == path.stat().st_size
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="header hash differs"):
        check.codec_header(path, contract)


def _fake_chain_receipt(tmp_path):
    folder = tmp_path / "completed-chain"
    folder.mkdir()
    array_path = folder / "isa_ids.npy"
    array_path.write_bytes(b"synthetic saved IDs")
    receipt = {
        "status": "completed", "reset_bit_exact": True,
        "all_layer_kv_reset_bit_exact": True,
        "execution_target": {"variant": "p4rl8a64c150", "arch_hash": "dd6b92a6"},
        "isa_vs_native_original_id_matches": 16,
        "arrays": {"isa_ids": {"sha256": hashlib.sha256(array_path.read_bytes()).hexdigest()}},
        "frame_index": 0, "checkpoint_revision": "synthetic",
    }
    receipt_path = folder / "depth_chain_check.json"
    receipt_path.write_text(json.dumps(receipt))
    return folder, receipt_path, array_path, receipt


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("reset_bit_exact", False),
    ("all_layer_kv_reset_bit_exact", False),
    ("isa_vs_native_original_id_matches", 15),
])
def test_mimi_baseline_requires_completed_exact_chain(tmp_path, field, value):
    folder, path, _, receipt = _fake_chain_receipt(tmp_path)
    actual_folder, actual_path, actual_receipt = check._baseline_chain(folder)
    assert actual_folder == folder and actual_path == path
    assert actual_receipt == receipt
    path.write_text(json.dumps({**receipt, field: value}))
    with pytest.raises(ValueError, match="Completed exact common-profile"):
        check._baseline_chain(folder)


@pytest.mark.parametrize("field,value", [("variant", "p4rl8a64c100"),
                                          ("arch_hash", "wrong-architecture")])
def test_mimi_baseline_rejects_other_target_identity(tmp_path, field, value):
    folder, path, _, receipt = _fake_chain_receipt(tmp_path)
    receipt["execution_target"][field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Completed exact common-profile"):
        check._baseline_chain(folder)


def test_mimi_baseline_rechecks_saved_array_hash(tmp_path):
    folder, _, array_path, _ = _fake_chain_receipt(tmp_path)
    array_path.write_bytes(b"tampered saved IDs")
    with pytest.raises(ValueError, match="Saved depth baseline changed: isa_ids"):
        check._baseline_chain(folder)


def test_non_shared_target_is_rejected_before_contract_or_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(check, "DATA", tmp_path / "missing-contract-directory")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="explicit shared Gemma target"):
        check.run(tmp_path / "model", tmp_path / "codec", tmp_path / "tokens",
                  tmp_path / "conditioning", tmp_path / "stack", tmp_path / "chain",
                  output, target_name="f2-hbm:p4rl8a64c100")
    assert not output.exists()


def test_source_only_moshi_distribution_metadata_is_optional(monkeypatch):
    def version(name):
        if name == "moshi":
            raise check.importlib.metadata.PackageNotFoundError(name)
        return "existing-test-version"
    monkeypatch.setattr(check.importlib.metadata, "version", version)
    assert check.dependency_versions() == {
        "numpy": "existing-test-version", "torch": "existing-test-version",
        "safetensors": "existing-test-version", "moshi": None}
