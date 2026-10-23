"""Small data-reader and independent BF16-reference tests; no network/checkpoint."""
import hashlib
import json
import struct

import numpy as np
import pytest

from tools import qwen3_tts_checkpoint_acceptance as A


def fixture(tmp_path, monkeypatch, dtype="BF16"):
    data = np.asarray([0, 0x8000, 0x3f80, 0xbf00, 1], dtype="<u2")
    header = json.dumps({"weight": {"dtype": dtype, "shape": [5], "data_offsets": [0, 10]}}).encode().ljust(49224)
    path = tmp_path / "data.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + data.tobytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(A, "HEADER_SHA256", hashlib.sha256(header).hexdigest())
    monkeypatch.setattr(A, "FILE_SHA256", digest)
    monkeypatch.setattr(A, "FILE_BYTES", path.stat().st_size)
    manifest = {"model_id": A.MODEL_ID, "model_revision": A.MODEL_REVISION,
                "source_revision": A.SOURCE_REVISION, "verified_lfs_sha256": True,
                "file_sha256": digest, "file_bytes": path.stat().st_size}
    return path, manifest, data


def test_bf16_bit_extension_is_exact_and_preserves_signed_zero(tmp_path, monkeypatch):
    path, manifest, raw = fixture(tmp_path, monkeypatch)
    actual = A.BF16Checkpoint(path, manifest).tensor("weight", (5,))
    np.testing.assert_array_equal(actual.view(np.uint32), raw.astype(np.uint32) << 16)


def test_corrupt_or_unverified_checkpoint_fails_before_tensor_access(tmp_path, monkeypatch):
    path, manifest, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exact-model manifest"):
        A.BF16Checkpoint(path, dict(manifest, verified_lfs_sha256=False))
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        A.BF16Checkpoint(path, manifest)


def test_reader_rejects_wrong_shape_and_dtype(tmp_path, monkeypatch):
    path, manifest, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="invalid checkpoint tensor"):
        A.BF16Checkpoint(path, manifest).tensor("weight", (1, 5))
    path, manifest, _ = fixture(tmp_path, monkeypatch, dtype="F16")
    with pytest.raises(ValueError, match="invalid checkpoint tensor"):
        A.BF16Checkpoint(path, manifest).tensor("weight", (5,))


def test_locally_reviewed_bf16_math_against_independent_fp64():
    pytest.importorskip("torch")
    from qwen3_tts_talker_bf16_reference import BF16Reference
    from qwen3_tts_talker_reference import Reference
    from test_qwen3_tts_talker import tiny, make_weights
    image = tiny()
    weights = make_weights(image)
    # Both references start with the same BF16-rounded checkpoint values.
    for name, value in weights.items():
        bits = value.view(np.uint32)
        rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
        weights[name] = rounded.view(np.float32)
    a, b = BF16Reference(image.spec, weights), Reference(image.spec, weights)
    for pair in [(1, 3), (16, 12), (0, 0)]:
        actual, expected = a.step(*pair), b.step(*pair)
        for stage in A.STAGES:
            assert A.metrics(actual[stage], expected[stage])["relative_l2"] < 0.02
