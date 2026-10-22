#!/usr/bin/env python3
"""Convert the real PersonaPlex checkpoint and compare CPU ISA output with FP64.

This bounded check runs 1..4 initial frames, resets, and repeats those frames.
It never opens an FPGA, downloads a model, or measures hardware performance.
The output directory must be new. The explicitly selected, isolated quantization
cache may be reused after an interruption or by a later hardware run.
"""
from __future__ import annotations

# These limits must precede imports that initialize numerical libraries. Spawned
# qcache workers inherit them, and importing this script never selects a device.
import os
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ[_name] = "1"
os.environ["OTPU_BUILD_JOBS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opentpu import qcache, target
from opentpu.llm.personaplex_temporal import Engine, Image, Spec
from opentpu.llm.personaplex_weights import MLXWeights

PROFILE = ROOT / "tools/data/personaplex/burst200.toml"
CHECKPOINT_REPO = "aufklarer/PersonaPlex-7B-MLX-4bit"
PINNED_REVISION = "8c686347007de51d9017706e5543d6f1757ecf5d"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def small_sha256(path):
    # Sources and bounded input/reference arrays only, never checkpoint payloads.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_metadata(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_array(path, *, dtype=None):
    if path.stat().st_size > 1 << 20:
        raise ValueError(f"Expected a bounded 1..4-frame array, file exceeds 1 MiB: {path}")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if (not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1] != 4096
            or not 1 <= value.shape[0] <= 4 or value.dtype.kind != "f"
            or not np.isfinite(value).all()):
        raise ValueError(f"Expected finite floating-point [1..4,4096] array: {path}")
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise ValueError(f"Expected {dtype} array: {path}")
    return value


def revision_evidence(model_dir):
    evidence = []
    if model_dir.parent.name == "snapshots" and len(model_dir.name) == 40:
        evidence.append({"source": "snapshot_directory_name", "revision": model_dir.name})
    for name in ("config.json", "temporal.safetensors"):
        path = model_dir / ".cache/huggingface/download" / (name + ".metadata")
        if path.is_file():
            with path.open() as handle:
                revision = handle.readline().strip()
            evidence.append({"source": str(path), "revision": revision})
    if not evidence or any(item["revision"] != PINNED_REVISION for item in evidence):
        raise ValueError("Local checkpoint revision metadata must match the pinned revision")
    return evidence


def dependency_metadata():
    result = {}
    for name in ("numpy", "torch", "safetensors"):
        distribution = importlib.metadata.distribution(name)
        hashes = {}
        for filename in ("METADATA", "RECORD"):
            content = distribution.read_text(filename)
            if content is not None:
                hashes[filename] = hashlib.sha256(content.encode()).hexdigest()
        result[name] = {"version": distribution.version, "installed_metadata_sha256": hashes}
    return result


class ProgressLayers(Sequence):
    """Log Image.build's sequential lazy indexing without changing its kernels."""

    def __init__(self, layers, emit):
        self.layers, self.emit, self.next_index = layers, emit, 0

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        if index != self.next_index:
            raise RuntimeError("Image build no longer consumes layers once in ascending order")
        self.emit("converting_layer", layer_index=index, layer_number=index + 1,
                  completed_layers=index, total_layers=len(self))
        self.next_index += 1
        return self.layers[index]


def precision_metrics(actual, expected):
    a, b = np.asarray(actual, np.float64).reshape(-1), np.asarray(expected, np.float64).reshape(-1)
    delta = a - b
    anorm, bnorm, dnorm = (float(np.linalg.norm(v)) for v in (a, b, delta))
    result = {
        "relative_l2": dnorm / bnorm if bnorm else None,
        "cosine_similarity": float(np.clip(np.dot(a, b) / (anorm * bnorm), -1, 1))
                             if anorm and bnorm else None,
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mae": float(np.mean(np.abs(delta))),
        "max_absolute_error": float(np.max(np.abs(delta))),
        "isa_l2_norm": anorm, "reference_l2_norm": bnorm, "delta_l2_norm": dnorm,
    }
    if any(value is not None and not np.isfinite(value) for value in result.values()):
        raise RuntimeError("Non-finite precision comparison metrics")
    return result


def run(model_dir, embeddings_file, reference_dir, output_dir, cache_dir, *, cap=1024):
    model_dir, embeddings_file, reference_dir, output_dir, cache_dir = (
        Path(path).resolve() for path in
        (model_dir, embeddings_file, reference_dir, output_dir, cache_dir))
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new; refusing overwrite: {output_dir}")
    if not isinstance(cap, int) or isinstance(cap, bool) or not 4 <= cap <= Spec().context:
        raise ValueError("cap must be between 4 and the native 3000-frame context")
    # Prevent a mistaken cache root from managing entries in an input/source tree.
    for protected in (model_dir, reference_dir, ROOT, output_dir):
        if cache_dir == protected or cache_dir in protected.parents:
            raise ValueError(f"Cache must be an isolated directory, not an input/output ancestor: {cache_dir}")
    os.environ["OTPU_IMAGE_CACHE"] = str(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_file = output_dir / "isa_check.json"
    progress_file = output_dir / "progress.jsonl"
    progress_file.touch(exist_ok=False)
    cache_before = dict(qcache.stats)
    started = time.perf_counter()
    receipt = {
        "status": "running", "started_utc": utc_now(), "stage": "validating_inputs",
        "backend": "isa", "fpga_used": False, "network_used": False,
        "execution": "CPU-only full temporal checkpoint conversion and ISA simulation",
        "comparison": "Precision delta from saved unmodified dequantized MLX + FP64 math baseline",
        "accuracy_acceptance": "pending", "quality_validated": False,
        "hardware_performance_measured": False, "hardware_correctness_validated": False,
        "cache": {"path": str(cache_dir), "reused": cache_dir.exists(), "build_jobs": 2},
        "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
        "offline_environment": {name: os.environ[name] for name in
                                ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
                                 "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")},
        "timing_scope": "Host CPU wall time only; no FPGA cycles, throughput or end-to-end audio timing",
        "limitations": ["No quality threshold is inferred from synthetic tests.",
                        "Does not execute the text head, depformer, token sampling or Mimi codec.",
                        "A one-frame input cannot test nonzero RoPE positions or historical KV attention.",
                        "Short inputs do not test attention bucket transitions or context exhaustion.",
                        "Checkpoint revision/file metadata are checked, but payloads are not hashed.",
                        "Installed dependency metadata hashes do not verify every installed binary."]}

    def save():
        temporary = receipt_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(receipt_file)

    def emit(stage, **details):
        event = {"utc": utc_now(), "elapsed_seconds": time.perf_counter() - started,
                 "stage": stage, **details}
        receipt["stage"], receipt["progress"] = stage, event
        receipt["qcache_stats"] = {key: qcache.stats[key] - cache_before[key] for key in cache_before}
        with progress_file.open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
        save()
        print(json.dumps(event, allow_nan=False), flush=True)

    engine = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError("Received SIGTERM; conversion/simulation interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        emit("validating_inputs")
        cache_dir.mkdir(parents=True, exist_ok=True)
        reference_receipt_file = reference_dir / "reference.json"
        if reference_receipt_file.stat().st_size > 1 << 20:
            raise ValueError("Reference receipt exceeds 1 MiB")
        baseline = json.loads(reference_receipt_file.read_text())
        if baseline.get("status") != "completed" or not baseline.get("finished_utc"):
            raise ValueError("Reference receipt must describe a completed calculation")
        if baseline.get("temporal_spec") != asdict(Spec()):
            raise ValueError("Reference does not describe the complete production temporal specification")
        contract = baseline["input_contract"]
        if contract.get("initial_position") != 0 or contract.get("initial_kv") != "empty":
            raise ValueError("Reference must start at position zero with empty KV")
        embeddings = load_array(embeddings_file)
        reference_file = reference_dir / "reference_hidden.npy"
        expected = load_array(reference_file, dtype="float64")
        with np.errstate(over="ignore", invalid="ignore"):
            inputs = np.asarray(embeddings, np.float32)
        if not np.isfinite(inputs).all():
            raise ValueError("Embeddings must remain finite in float32")
        if (expected.shape != embeddings.shape or contract.get("frames") != len(inputs)
                or contract.get("hidden") != 4096 or contract.get("dtype") != str(embeddings.dtype)
                or baseline["output"].get("shape") != list(expected.shape)
                or baseline["output"].get("dtype") != "float64"):
            raise ValueError("Input/reference shapes or dtypes disagree with the reference receipt")
        input_sha, reference_sha = small_sha256(embeddings_file), small_sha256(reference_file)
        if input_sha != baseline["input"]["sha256"] or reference_sha != baseline["output"]["sha256"]:
            raise ValueError("Input/reference SHA256 does not match the completed reference receipt")
        receipt_sha = small_sha256(reference_receipt_file)
        receipt["input"] = {**file_metadata(embeddings_file), "sha256": input_sha,
                            "shape": list(inputs.shape), "source_dtype": str(embeddings.dtype)}
        receipt["reference"] = {**file_metadata(reference_file), "sha256": reference_sha,
                                "receipt_path": str(reference_receipt_file),
                                "receipt_sha256": receipt_sha, "baseline": baseline.get("baseline")}
        weights = MLXWeights(model_dir)
        if weights.spec != Spec() or weights.config["quantization"]["bits"] != 4 \
                or weights.config["quantization"]["group_size"] != 64:
            raise ValueError("Expected the complete production MLX affine INT4/group64 checkpoint")
        original_checkpoint = baseline["checkpoint"]
        config_sha = small_sha256(model_dir / "config.json")
        if (original_checkpoint.get("repository") != CHECKPOINT_REPO
                or original_checkpoint.get("expected_pinned_revision") != PINNED_REVISION
                or original_checkpoint.get("config_sha256") != config_sha
                or original_checkpoint.get("config") != weights.config):
            raise ValueError("Checkpoint identity/config differs from the completed reference")
        evidence = revision_evidence(model_dir)
        checkpoint_files = {name: file_metadata(model_dir / name)
                            for name in ("config.json", "temporal.safetensors")}
        for name, current in checkpoint_files.items():
            recorded = original_checkpoint["files"][name]
            if any(current[key] != recorded[key] for key in ("bytes", "mtime_ns")):
                raise ValueError(f"Checkpoint file metadata changed since reference calculation: {name}")
        receipt["checkpoint"] = {"repository": CHECKPOINT_REPO, "revision": PINNED_REVISION,
                                 "local_revision_evidence": evidence, "config": weights.config,
                                 "config_sha256": config_sha, "files": checkpoint_files,
                                 "payload_hashes_verified": False}
        profile = target.load(str(PROFILE))
        cfg = target.config(profile)
        image = Image(weights.spec, cfg, cap=cap, wformat="int4")
        receipt["temporal_spec"] = asdict(weights.spec)
        receipt["profile"] = {"path": str(PROFILE), "sha256": small_sha256(PROFILE),
                              "name": profile.name, "config": asdict(cfg)}
        receipt["image"] = {"cap": cap, "physical_cap": image.cap, "block": image.block,
                            "image_bytes": image.nbytes, "required_bytes": image.required_bytes,
                            "isa_dram_bytes": cfg.DRAM_BYTES, "weight_bytes": image.weight_bytes,
                            "weight_code_bytes": image.weight_code_bytes}
        receipt["precision"] = {"source_weights": "MLX affine INT4/group64 dequantized to FP32",
                                "isa_weights": "openTPU MX INT4 requantization",
                                "activations": "openTPU QACT INT8 with MX scales",
                                "kv_cache": "INT8", "arithmetic": "openTPU ISA FP32/FTZ and MX integer accumulation",
                                "output_dtype": "float32", "reference_dtype": "float64"}
        source_paths = sorted((ROOT / "opentpu").rglob("*.py")) + [Path(__file__), PROFILE,
                        ROOT / "tools/personaplex_reference.py", ROOT / "tools/personaplex_verilator.py"]
        receipt["source_files"] = {str(path.relative_to(ROOT)): small_sha256(path) for path in source_paths}
        receipt["dependencies"] = dependency_metadata()
        receipt["python"] = sys.version
        emit("building_image", completed_layers=0, total_layers=weights.spec.layers)
        build_started = time.perf_counter()
        layers = ProgressLayers(weights.layers, emit)
        engine = Engine(weights.spec, layers, weights.final_norm, cfg=cfg,
                        backend="isa", cap=cap, wformat="int4")
        receipt["image_build_cpu_wall_seconds"] = time.perf_counter() - build_started
        emit("image_ready", completed_layers=layers.next_index, total_layers=len(layers))
        outputs, receipt["isa_runs"] = [], []
        for index, row in enumerate(inputs):
            emit("simulating_frame", frame=index)
            outputs.append(engine.step(row))
            receipt["isa_runs"].append({"pass": "initial", **engine.last_stats})
            emit("frame_complete", frame=index, simulator=engine.last_stats)
        outputs = np.stack(outputs)
        output_file = output_dir / "isa_hidden.npy"
        with output_file.open("xb") as handle:
            np.save(handle, outputs, allow_pickle=False)
        receipt["output"] = {**file_metadata(output_file), "sha256": small_sha256(output_file),
                             "shape": list(outputs.shape), "dtype": str(outputs.dtype), "all_finite": True}
        receipt["precision_delta"] = precision_metrics(outputs, expected)
        receipt["precision_delta_per_frame"] = [precision_metrics(a, b) for a, b in zip(outputs, expected)]
        emit("checking_reset")
        engine.reset()
        receipt["reset_bit_exact"] = False
        for index, row in enumerate(inputs):
            emit("simulating_reset_frame", frame=index)
            repeated = engine.step(row)
            receipt["isa_runs"].append({"pass": "reset_repeat", **engine.last_stats})
            if not np.array_equal(repeated.view(np.uint32), outputs[index].view(np.uint32)):
                raise AssertionError(f"Reset replay differs bit-for-bit at frame {index}")
            emit("reset_frame_complete", frame=index, simulator=engine.last_stats)
        receipt["reset_bit_exact"] = True
        for name, before in checkpoint_files.items():
            if file_metadata(model_dir / name) != before:
                raise RuntimeError(f"Checkpoint file changed during calculation: {name}")
        if (small_sha256(embeddings_file) != input_sha or small_sha256(reference_file) != reference_sha
                or small_sha256(reference_receipt_file) != receipt_sha):
            raise RuntimeError("Input or reference changed during calculation")
        receipt["status"] = "completed"
        emit("completed")
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["traceback"] = traceback.format_exc()
        emit("failed")
        raise
    finally:
        if engine is not None:
            engine.close()
        signal.signal(signal.SIGTERM, previous_sigterm)
        receipt["finished_utc"] = utc_now()
        receipt["total_cpu_wall_seconds"] = time.perf_counter() - started
        receipt["qcache_stats"] = {key: qcache.stats[key] - cache_before[key] for key in cache_before}
        save()
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--cap", type=int, default=1024)
    args = parser.parse_args(argv)
    receipt = run(args.model_dir, args.embeddings, args.reference_dir,
                  args.output_dir, args.cache_dir, cap=args.cap)
    print(json.dumps({"status": receipt["status"], "output_dir": str(args.output_dir.resolve()),
                      "reset_bit_exact": receipt["reset_bit_exact"],
                      "precision_delta": receipt["precision_delta"],
                      "accuracy_acceptance": receipt["accuracy_acceptance"]}, allow_nan=False))


if __name__ == "__main__":
    main()
