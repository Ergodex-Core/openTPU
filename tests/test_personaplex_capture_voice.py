"""Stored voice-prefix capture using synthetic local tensors only."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools import personaplex_capture_voice as capture_voice


@pytest.fixture(autouse=True)
def preserve_numerical_environment(monkeypatch):
    # capture() sets standalone CPU limits; restore the surrounding test process.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, os.environ.get(name, ""))


@pytest.fixture
def voice(tmp_path):
    embeddings = (torch.arange(6 * 4096, dtype=torch.float32).reshape(6, 1, 1, 4096)
                  / 128 - 50).to(torch.bfloat16)
    path = tmp_path / "synthetic-voice.pt"
    torch.save({"embeddings": embeddings, "cache": torch.zeros(1, 17, 4, dtype=torch.long)}, path)
    return path, embeddings


@pytest.mark.parametrize("frames", [1, 2, 3, 4])
def test_bfloat16_prefix_preserves_order_bits_hashes_and_zero_kv_contract(voice, tmp_path,
                                                                      monkeypatch, frames):
    source, embeddings = voice
    before = source.read_bytes()
    output = tmp_path / "capture"
    original_load, loads = torch.load, []

    def load(path, **kwargs):
        loads.append((Path(path), kwargs))
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    report = capture_voice.capture(source, output, frames=frames)
    assert loads == [(source.resolve(), {"weights_only": True, "map_location": "cpu"})]
    actual = np.load(output / "input_embeddings.npy", allow_pickle=False)
    expected = embeddings[:frames, 0, 0, :].float().numpy()
    assert actual.shape == (frames, 4096) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert source.read_bytes() == before
    assert report["status"] == "completed"
    assert report["source"] == {
        "path": str(source.resolve()), "sha256": hashlib.sha256(before).hexdigest(),
        "bytes": len(before), "shape": [6, 1, 1, 4096], "dtype": "torch.bfloat16",
        "weights_only": True}
    assert report["frame_indices"] == list(range(frames))
    assert report["input_contract"] == {
        "frames": frames, "hidden": 4096, "initial_position": 0,
        "initial_kv": "empty", "dtype": "float32"}
    assert report["output"] == {
        "path": str((output / "input_embeddings.npy").resolve()),
        "sha256": hashlib.sha256((output / "input_embeddings.npy").read_bytes()).hexdigest(),
        "shape": [frames, 4096], "dtype": "float32", "all_finite": True}
    assert report["script_sha256"] == hashlib.sha256(Path(capture_voice.__file__).read_bytes()).hexdigest()
    assert report["temporal_executed"] is report["fpga_used"] is report["speech_quality_validated"] is False
    assert json.loads((output / "input.json").read_text()) == report
    assert sorted(path.name for path in output.iterdir()) == ["input.json", "input_embeddings.npy"]


def test_repeat_capture_refuses_before_loading_and_preserves_source_and_outputs(voice, tmp_path,
                                                                             monkeypatch):
    source, _ = voice
    output = tmp_path / "capture"
    capture_voice.capture(source, output)
    before = {path: path.read_bytes() for path in (source, *output.iterdir())}

    def forbidden(*args, **kwargs):
        raise AssertionError("Existing output must be refused before loading the voice")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(FileExistsError, match="Output directory must be new"):
        capture_voice.capture(source, output)
    assert {path: path.read_bytes() for path in before} == before
    assert len(list(output.iterdir())) == 2


@pytest.mark.parametrize("fault", ["not_dict", "missing", "not_tensor", "integer", "rank",
                                   "batch", "time", "width", "short", "nan", "inf",
                                   "negative_inf", "overflow", "negative_overflow"])
def test_invalid_payload_is_rejected_before_output_and_preserves_source(tmp_path, fault):
    embeddings = torch.zeros(4, 1, 1, 4096)
    payload = {"embeddings": embeddings}
    shapes = {"rank": (4, 4096), "batch": (4, 2, 1, 4096),
              "time": (4, 1, 2, 4096), "width": (4, 1, 1, 4095), "short": (3, 1, 1, 4096)}
    message = "Expected floating-point voice embeddings"
    if fault == "not_dict":
        payload, message = embeddings, "Voice payload must be a dictionary"
    elif fault == "missing":
        payload = {}
    elif fault == "not_tensor":
        payload = {"embeddings": []}
    elif fault == "integer":
        payload = {"embeddings": embeddings.long()}
    elif fault in shapes:
        payload = {"embeddings": torch.zeros(shapes[fault])}
    else:
        value = {"nan": float("nan"), "inf": float("inf"), "negative_inf": -float("inf"),
                 "overflow": 1e100, "negative_overflow": -1e100}[fault]
        embeddings = embeddings.double()
        embeddings[3, 0, 0, 42] = value  # Reject the whole prefix, including its final frame.
        payload = {"embeddings": embeddings}
        message = "remain finite in float32" if "overflow" in fault else "must be finite"
    source = tmp_path / "invalid-voice.pt"
    torch.save(payload, source)
    before = source.read_bytes()
    parent = tmp_path / "outputs"
    parent.mkdir()
    sentinel = parent / "preserved.bin"
    sentinel.write_bytes(b"existing output")
    output = parent / "capture"
    with pytest.raises(ValueError, match=message):
        capture_voice.capture(source, output)
    assert not output.exists()
    assert source.read_bytes() == before
    assert sentinel.read_bytes() == b"existing output"
    assert list(parent.iterdir()) == [sentinel]


@pytest.mark.parametrize("frames", [True, False, 1.0, 1.5, -1, 0, 5, None, "4"])
def test_invalid_frame_count_fails_before_loading_or_output(voice, tmp_path, monkeypatch, frames):
    source, _ = voice
    before = source.read_bytes()
    output = tmp_path / "capture"

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid frame count must be refused before loading the voice")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="frames must be an integer between 1 and 4"):
        capture_voice.capture(source, output, frames=frames)
    assert not output.exists()
    assert source.read_bytes() == before


def test_unselected_nonfinite_tail_does_not_change_finite_prefix(voice, tmp_path):
    source, embeddings = voice
    expected = embeddings[:4, 0, 0, :].float().numpy().copy()
    embeddings[5, 0, 0, 0] = float("nan")
    torch.save({"embeddings": embeddings}, source)
    before = source.read_bytes()
    output = tmp_path / "capture"
    capture_voice.capture(source, output)
    np.testing.assert_array_equal(np.load(output / "input_embeddings.npy", allow_pickle=False), expected)
    assert source.read_bytes() == before
