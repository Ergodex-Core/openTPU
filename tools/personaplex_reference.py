#!/usr/bin/env python3
"""CPU FP64 temporal reference for 1..4 captured initial PersonaPlex embeddings.

Inputs must be consecutive frames starting at position zero with empty KV state.
This computes the unmodified, dequantized MLX checkpoint's mathematical baseline;
it does not run FPGA precision, evaluate conversation quality, or benchmark speed.
No model download, Engine construction, or device access occurs.
"""
from __future__ import annotations

# Set before NumPy, torch, or the imported reference module initializes BLAS.
import os
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ[_name] = "2"

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next((parent for parent in Path(__file__).resolve().parents
             if (parent / "opentpu/llm/personaplex_weights.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Run this tool from an extracted PersonaPlex source tree")
sys.path.insert(0, str(ROOT))
from opentpu.llm.personaplex_temporal import Spec
from opentpu.llm.personaplex_weights import MLXWeights
from tools.personaplex_verilator import reference

CHECKPOINT_REPO = "aufklarer/PersonaPlex-7B-MLX-4bit"
PINNED_REVISION = "8c686347007de51d9017706e5543d6f1757ecf5d"


def small_file_sha256(path):
    """Only called for source, config, and <=4-frame array artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_metadata(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def revision_evidence(model_dir, filenames):
    evidence = []
    if model_dir.parent.name == "snapshots" and len(model_dir.name) == 40:
        evidence.append({"source": "snapshot_directory_name", "revision": model_dir.name})
    for name in filenames:
        path = model_dir / ".cache" / "huggingface" / "download" / (name + ".metadata")
        if path.is_file():
            # huggingface_hub local-dir download metadata starts with commit SHA.
            with path.open() as handle:
                revision = handle.readline().strip()
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError(f"Invalid Hugging Face revision metadata: {path}")
            evidence.append({"source": str(path.resolve()), "revision": revision})
    if any(item["revision"] != PINNED_REVISION for item in evidence):
        raise ValueError("Local revision metadata does not match the pinned checkpoint")
    return evidence


def run(model_dir, embeddings_file, output_dir):
    model_dir, embeddings_file, output_dir = map(Path, (model_dir, embeddings_file, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new; refusing overwrite: {output_dir}")
    # mmap avoids allocating an unexpectedly large input before rejecting shape.
    embeddings = np.load(embeddings_file, allow_pickle=False, mmap_mode="r")
    if (not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2
            or not 1 <= embeddings.shape[0] <= 4 or embeddings.shape[1] != 4096):
        raise ValueError("Embeddings must have shape [frames, 4096] with 1 <= frames <= 4")
    if embeddings.dtype.kind != "f" or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must contain finite floating-point values")
    weights = MLXWeights(model_dir)
    if weights.spec != Spec():
        raise ValueError("Expected the complete 32-layer PersonaPlex-7B temporal specification")
    quantization = weights.config["quantization"]
    if quantization["bits"] != 4 or quantization["group_size"] != 64:
        raise ValueError("Expected the pinned MLX affine INT4, group-size-64 checkpoint")

    files = {name: file_metadata(model_dir / name) for name in
             ("config.json", "temporal.safetensors", "depformer.safetensors", "embeddings.safetensors")
             if (model_dir / name).is_file()}
    evidence = revision_evidence(model_dir, files)
    physical_dtypes = Counter(weights.temporal.handle.get_slice(key).get_dtype()
                              for key in weights.temporal.keys)
    source_paths = [Path(__file__), ROOT / "opentpu/llm/personaplex_weights.py",
                    ROOT / "opentpu/llm/personaplex_temporal.py",
                    ROOT / "tools/personaplex_verilator.py"]
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "safetensors")}
    receipt = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "Unmodified MLX affine INT4 weights dequantized to FP32; independent FP64 transformer arithmetic",
        "arithmetic": {"weight_dequantization": "FP32 code * scale + affine_bias",
                       "transformer": "FP64 RMSNorm, adjacent-pair RoPE, causal full MHA, SwiGLU, residuals and final RMSNorm",
                       "output_dtype": "float64", "weight_requantization": False,
                       "activation_quantization": False, "kv_quantization": False},
        "input_contract": {"frames": int(embeddings.shape[0]), "hidden": 4096,
                           "dtype": str(embeddings.dtype), "initial_position": 0,
                           "initial_kv": "empty",
                           "note": "Must be consecutive initial frames from a reset stream. Later frames require prior KV and are not represented by this baseline."},
        "input": {**file_metadata(embeddings_file), "sha256": small_file_sha256(embeddings_file)},
        "checkpoint": {"repository": CHECKPOINT_REPO, "expected_pinned_revision": PINNED_REVISION,
                       "local_revision_evidence": evidence, "local_revision_metadata_matches": bool(evidence),
                       "payload_hashes_verified": False,
                       "note": "Revision evidence is local metadata only; no multi-GB payload hashes are computed.",
                       "config": weights.config, "config_sha256": small_file_sha256(model_dir / "config.json"),
                       "files": files,
                       "temporal_physical_tensor_count": len(weights.temporal.keys),
                       "temporal_physical_dtype_counts": dict(sorted(physical_dtypes.items()))},
        "temporal_spec": asdict(weights.spec),
        "source_files": {str(path.resolve()): small_file_sha256(path) for path in source_paths},
        "dependency_versions": versions,
        "blas_thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
        "execution": "CPU only; lazy layer mappings; no Engine or backend constructed",
        "fpga_precision_validated": False, "quality_validated": False, "performance_measured": False,
        "limitations": ["Does not run the text head, depformer, token sampling or Mimi codec.",
                        "Is not a bit-exact NVIDIA PyTorch/MLX FP16 or BF16 inference reference.",
                        "Does not simulate FPGA weight/activation/INT8-KV rounding or FTZ.",
                        "Causal reference starts at zero; short inputs do not exercise sliding-window behavior."]}
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_file = output_dir / "reference.json"
    receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        # The reference requests one dequantized matrix at a time from lazy
        # mappings. NumPy promotes its FP32 weights into the FP64 matmuls.
        hidden = reference(weights.spec, weights.layers, weights.final_norm, embeddings)
        if hidden.shape != embeddings.shape or hidden.dtype != np.float64 or not np.isfinite(hidden).all():
            raise RuntimeError("Reference produced invalid hidden activations")
        for name, before in files.items():
            if file_metadata(model_dir / name) != before:
                raise RuntimeError(f"Checkpoint file changed during reference calculation: {name}")
        if small_file_sha256(embeddings_file) != receipt["input"]["sha256"]:
            raise RuntimeError("Embeddings file changed during reference calculation")
        output_file = output_dir / "reference_hidden.npy"
        with output_file.open("xb") as handle:
            np.save(handle, hidden, allow_pickle=False)
        receipt["status"] = "completed"
        receipt["output"] = {**file_metadata(output_file), "shape": list(hidden.shape),
                             "dtype": str(hidden.dtype), "sha256": small_file_sha256(output_file),
                             "all_finite": True}
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args.model_dir, args.embeddings, args.output_dir)
    print(json.dumps({"status": receipt["status"], "output_dir": str(args.output_dir.resolve()),
                      "frames": receipt["input_contract"]["frames"], "baseline": receipt["baseline"]}))


if __name__ == "__main__":
    main()
