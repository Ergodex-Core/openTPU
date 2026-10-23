"""Generate small, auditable synthetic-audio golden vectors offline on CPU.

Run explicitly with --output-dir; never downloads weights or imports model code.
The ASR serving adapter does not import this module or offer a CPU fallback.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from opentpu.asr.phonon2 import MODEL_REVISION, BASE_REVISION
from tools.phonon2_reference import frontend_reference, slaney_mel
from tools.phonon2_torch_audit import frontend_torch


def cases():
    rng = np.random.default_rng(20261023)
    impulse = np.zeros(321, dtype="<i2")
    impulse[0], impulse[-1] = 32767, -32768
    t = np.arange(1601) / 16000
    tone = np.rint(20000 * np.sin(2 * np.pi * (400 * t + 5000 * t * t))).astype("<i2")
    return {"silence_min": np.zeros(320, "<i2"), "silence_100ms": np.zeros(1600, "<i2"),
            "boundary_impulses": impulse, "chirp_partial_hop": tone,
            "noise_exact_hop": rng.integers(-12000, 12001, 1600, dtype=np.int16).astype("<i2"),
            "noise_partial_hop": rng.integers(-20000, 20001, 3199, dtype=np.int16).astype("<i2")}


def generate(output_dir):
    import torch
    torch.set_num_threads(1)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    arrays, records = {"mel_bank": slaney_mel()}, []
    for name, pcm in cases().items():
        data = pcm.tobytes()
        golden = frontend_torch(data)
        independent = frontend_reference(data)
        arrays[name + ".pcm_s16le"] = pcm
        for stage in ("preemphasis", "log_mel", "features", "attention_mask"):
            arrays[name + "." + stage] = golden[stage]
        errors = {key: float(np.max(np.abs(golden[key] - independent[key])))
                  for key in ("preemphasis", "log_mel", "features")}
        assert np.array_equal(golden["attention_mask"], independent["attention_mask"])
        assert np.isfinite(golden["features"]).all()
        assert np.array_equal(golden["features"][-1], np.zeros(128))
        records.append({"name": name, "samples": len(pcm), "pcm_sha256": hashlib.sha256(data).hexdigest(),
                        "stored_frames": golden["stored_frames"], "valid_frames": golden["valid_frames"],
                        "numpy_max_abs_error": errors, "first_feature": float(golden["features"][0, 0])})
    path = root / "frontend_vectors.npz"
    np.savez_compressed(path, **arrays)
    report = {"model_revision": MODEL_REVISION, "base_revision": BASE_REVISION,
              "generator": "own audited equations using installed PyTorch CPU; no upstream model code executed",
              "torch": torch.__version__, "numpy": np.__version__, "machine": platform.machine(),
              "vector_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "vector_bytes": path.stat().st_size,
              "normalization": "FP32 masked mean; unbiased variance / (valid_frames - 1); std + 1e-5",
              "dither": 0.0, "dither_reason": "pinned Transformers path has no dither; NeMo archive says 1e-5",
              "input_policy": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1,
                               "min_samples": 320, "max_samples": 480000, "resampling": False},
              "librosa_algorithm_audit": "https://github.com/librosa/librosa/blob/0.11.0/librosa/filters.py#L121",
              "numerical_caveat": "Golden outputs identify this Torch CPU implementation; silence normalization is sensitive to FP32 reduction order. NumPy FFT/normalization is not a bitwise oracle or production path.",
              "hardware_validated": False, "records": records}
    (root / "frontend_vectors.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2))
