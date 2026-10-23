#!/usr/bin/env python3
"""Capture 1..4 initial temporal inputs from an existing PersonaPlex voice file.

The upstream voice-prompt loop replays embeddings in stored order after reset.
This tool only reads a small local voice payload; it loads no LM/codec, starts no
backend and downloads nothing. Outputs are new and must start with empty KV.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

MAX_VOICE_BYTES = 64 << 20


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(voice_file, output_dir, *, frames=4):
    if type(frames) is not int or not 1 <= frames <= 4:
        raise ValueError("frames must be an integer between 1 and 4")
    voice_file, output_dir = Path(voice_file).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    if not voice_file.is_file() or not 0 < voice_file.stat().st_size <= MAX_VOICE_BYTES:
        raise ValueError("Expected an existing voice payload no larger than 64 MiB")
    # Set before numerical imports; only the selected small voice is loaded.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    source_sha = sha256(voice_file)
    voice = torch.load(voice_file, weights_only=True, map_location="cpu")
    if not isinstance(voice, dict):
        raise ValueError("Voice payload must be a dictionary containing embeddings")
    embeddings = voice.get("embeddings")
    if (not isinstance(embeddings, torch.Tensor) or not embeddings.is_floating_point()
            or embeddings.ndim != 4 or tuple(embeddings.shape[1:]) != (1, 1, 4096)
            or embeddings.shape[0] < frames):
        raise ValueError("Expected floating-point voice embeddings [T,1,1,4096], T >= frames")
    selected = embeddings[:frames, 0, 0, :].detach()
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("Selected voice embeddings must be finite")
    converted = selected.to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(converted).all()):
        raise ValueError("Selected voice embeddings must remain finite in float32")
    if sha256(voice_file) != source_sha:
        raise RuntimeError("Voice payload changed during capture")
    report = {
        "status": "completed", "captured_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(voice_file), "sha256": source_sha,
                   "bytes": voice_file.stat().st_size, "shape": list(embeddings.shape),
                   "dtype": str(embeddings.dtype), "weights_only": True},
        "frame_indices": list(range(frames)),
        "selection": "voice['embeddings'][:frames,0,0,:].to(torch.float32)",
        "input_contract": {"frames": frames, "hidden": 4096, "initial_position": 0,
                           "initial_kv": "empty", "dtype": "float32"},
        "scope": "Initial stored voice-prompt temporal inputs, in upstream replay order",
        "script_sha256": sha256(Path(__file__)),
        "temporal_executed": False, "fpga_used": False,
        "speech_quality_validated": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    output_file = output_dir / "input_embeddings.npy"
    with output_file.open("xb") as handle:
        np.save(handle, converted.numpy(), allow_pickle=False)
    report["output"] = {"path": str(output_file), "sha256": sha256(output_file),
                        "shape": [frames, 4096], "dtype": "float32", "all_finite": True}
    with (output_dir / "input.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=4)
    args = parser.parse_args(argv)
    print(json.dumps(capture(args.voice_file, args.output_dir, frames=args.frames), allow_nan=False))


if __name__ == "__main__":
    main()
