#!/usr/bin/env python3
"""Bounded CPU ISA check of PersonaPlex temporal -> resident text head -> greedy ID.

Reuses a completed 1..4-frame temporal ISA receipt and checkpoint cache. This is
a diagnostic, not full speech inference. Never opens a physical device, downloads
weights or substitutes host token selection for the device's result.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.update(OTPU_BUILD_JOBS="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opentpu import target
from opentpu.llm.personaplex_temporal import Spec
from opentpu.llm.personaplex_token import TokenEngine, TokenImage
from opentpu.llm.personaplex_weights import MLXWeights
from tools.personaplex_isa_check import load_array, precision_metrics, PINNED_REVISION, PROFILE
# The temporal checker has a standalone two-worker default; this checker uses
# one worker. Its imported helpers do not construct a model or open a device.
os.environ["OTPU_BUILD_JOBS"] = "1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def compare_head(hidden, logits, tokens, source_head):
    """Offline diagnostic only; the production TokenEngine never calls this."""
    hidden, logits, source_head = (np.asarray(x) for x in (hidden, logits, source_head))
    tokens = np.asarray(tokens)
    if (hidden.ndim != 2 or source_head.ndim != 2 or logits.ndim != 2
            or logits.shape != (len(hidden), len(source_head))
            or hidden.shape[1] != source_head.shape[1] or tokens.shape != (len(hidden),)
            or tokens.dtype.kind not in "iu"
            or any(not np.isfinite(x).all() for x in (hidden, logits, source_head))):
        raise ValueError("Invalid bounded text-head diagnostic arrays")
    if not np.array_equal(tokens, np.argmax(logits, axis=1)):
        raise AssertionError("Device greedy IDs disagree with saved device logits")
    # Compare heads on identical normalized hidden states to isolate the head's
    # additional quantization from the already measured temporal precision loss.
    expected = np.asarray(hidden, np.float64) @ np.asarray(source_head, np.float64).T
    source_ids = np.argmax(expected, axis=1)
    return {"comparison": "Unmodified source head FP64 matmul on the SAME device hidden states",
            "device_greedy_matches_device_logits": True,
            "source_head_top1_ids": source_ids.tolist(),
            "source_head_top1_matches": (tokens == source_ids).tolist(),
            "all_source_head_top1_match": bool(np.array_equal(tokens, source_ids)),
            "logit_precision_delta": precision_metrics(logits, expected),
            "quality_accepted": False}


def run(model_dir, embeddings_file, temporal_dir, output_dir, cache_dir, *, cap=128, formats=""):
    model_dir, embeddings_file, temporal_dir, output_dir, cache_dir = (
        Path(p).resolve() for p in (model_dir, embeddings_file, temporal_dir, output_dir, cache_dir))
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    for protected in (model_dir, embeddings_file, temporal_dir, ROOT, output_dir):
        if cache_dir == protected or cache_dir in protected.parents:
            raise ValueError("Cache must be isolated from source, inputs and output")
    receipt_file = temporal_dir / "isa_check.json"
    if receipt_file.stat().st_size > 1 << 20:
        raise ValueError("Temporal receipt exceeds 1 MiB")
    baseline = json.loads(receipt_file.read_text())
    if (baseline.get("status") != "completed" or baseline.get("reset_bit_exact") is not True
            or baseline.get("temporal_spec") != asdict(Spec())
            or baseline.get("image", {}).get("cap") != cap
            or baseline.get("precision", {}).get("formats", "") != formats):
        raise ValueError("Completed temporal baseline must match the production spec, cap and formats")
    inputs = np.asarray(load_array(embeddings_file, dtype="float32"))
    hidden_file = temporal_dir / "isa_hidden.npy"
    expected_hidden = np.asarray(load_array(hidden_file, dtype="float32"))
    if (inputs.shape != expected_hidden.shape or sha256(embeddings_file) != baseline["input"]["sha256"]
            or sha256(hidden_file) != baseline["output"]["sha256"]):
        raise ValueError("Bounded input/temporal output does not match the baseline receipt")
    weights = MLXWeights(model_dir)
    if weights.spec != Spec() or weights.config != baseline["checkpoint"]["config"]:
        raise ValueError("Production checkpoint differs from the completed temporal baseline")
    files = {name: metadata(model_dir / name) for name in
             ("config.json", "temporal.safetensors", "embeddings.safetensors")}
    for name, value in files.items():
        revision_file = model_dir / ".cache/huggingface/download" / (name + ".metadata")
        with revision_file.open() as handle:
            revision = handle.readline().strip()
        if revision != PINNED_REVISION:
            raise ValueError(f"Checkpoint revision mismatch for {name}")
        if name in baseline["checkpoint"]["files"]:
            old = baseline["checkpoint"]["files"][name]
            if any(value[key] != old[key] for key in value):
                raise ValueError(f"Checkpoint metadata changed for {name}")
    head_metadata = weights.text_head_metadata()
    cfg = target.config(target.load(str(PROFILE)))
    # Layout/capacity is checked before materializing the new source head.
    layout = TokenImage(weights.spec, cfg, cap=cap, wformat="int4", formats=formats)
    os.environ["OTPU_IMAGE_CACHE"] = str(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "running", "stage": "building_image", "started_utc": utc_now(),
               "backend": "isa", "hardware_used": False, "hardware_performance_measured": False,
               "full_fpga_model_inference": False, "speech_generated": False, "quality_accepted": False,
               "scope": "Captured embeddings -> resident RoPE/temporal -> INT8 text head -> device greedy ID",
               "checkpoint_revision": PINNED_REVISION, "checkpoint_files": files,
               "input_sha256": baseline["input"]["sha256"], "baseline_receipt_sha256": sha256(receipt_file),
               "baseline_hidden_sha256": baseline["output"]["sha256"],
               "head_metadata": head_metadata, "cap": cap, "formats": formats,
               "required_bytes": layout.required_bytes, "dram_bytes": cfg.DRAM_BYTES,
               "frames": [], "reset_frames": [],
               "remaining_host_components": ["Mimi codec", "embeddings/conditioning", "depth transformer/audio heads",
                                              "native stochastic sampling/RNG"],
               "source_files": {str(p.relative_to(ROOT)): sha256(p) for p in
                                [Path(__file__), ROOT / "opentpu/llm/personaplex_token.py",
                                 ROOT / "opentpu/llm/personaplex_temporal.py",
                                 ROOT / "opentpu/llm/personaplex_weights.py"]}}
    result_file = output_dir / "token_check.json"

    def save():
        temporary = result_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(result_file)

    def interrupted(signum, frame):
        raise InterruptedError("Token diagnostic interrupted; inspect receipt before retrying")

    previous = signal.signal(signal.SIGTERM, interrupted)
    engine, started = None, time.perf_counter()
    save()
    try:
        build_start = time.perf_counter()
        engine = TokenEngine(weights.spec, weights.layers, weights.final_norm,
                             lambda: weights.text_head, cfg=cfg, backend="isa", cap=cap,
                             wformat="int4", formats=formats)
        receipt["image_build_cpu_seconds"] = time.perf_counter() - build_start
        tokens, hiddens, logits = [], [], []
        for index, row in enumerate(inputs):
            receipt.update(stage="simulating_frame", frame=index)
            save()
            tokens.append(engine.step(row))
            hidden, scores = engine.read_hidden(), engine.read_logits()
            if not np.array_equal(hidden.view(np.uint32), expected_hidden[index].view(np.uint32)):
                raise AssertionError(f"Resident RoPE temporal output differs from saved ISA frame{index}")
            hiddens.append(hidden)
            logits.append(scores)
            receipt["frames"].append({"index": index, "token_id": tokens[-1], "stats": engine.last_stats})
            save()
        hiddens, logits, tokens = np.stack(hiddens), np.stack(logits), np.asarray(tokens, np.int64)
        receipt["temporal_hidden_bit_exact"] = True
        receipt["head_comparison"] = compare_head(hiddens, logits, tokens, weights.text_head)
        for name, data in (("tokens", tokens), ("hidden", hiddens), ("logits", logits)):
            path = output_dir / (name + ".npy")
            with path.open("xb") as handle:
                np.save(handle, data, allow_pickle=False)
            receipt[name + "_output"] = {"sha256": sha256(path), "shape": list(data.shape), "dtype": str(data.dtype)}
        engine.reset()
        receipt["stage"] = "checking_reset"
        save()
        for index, row in enumerate(inputs):
            token = engine.step(row)
            if (token != int(tokens[index])
                    or not np.array_equal(engine.read_hidden().view(np.uint32), hiddens[index].view(np.uint32))
                    or not np.array_equal(engine.read_logits().view(np.uint32), logits[index].view(np.uint32))):
                raise AssertionError(f"Reset replay differs at frame{index}")
            receipt["reset_frames"].append({"index": index, "token_id": token, "stats": engine.last_stats})
            save()
        receipt["reset_bit_exact"] = True
        if (sha256(embeddings_file) != receipt["input_sha256"]
                or sha256(receipt_file) != receipt["baseline_receipt_sha256"]
                or sha256(hidden_file) != receipt["baseline_hidden_sha256"]
                or any(metadata(model_dir / name) != value for name, value in files.items())):
            raise RuntimeError("Inputs/checkpoint/baseline changed during the diagnostic")
        receipt.update(status="completed", stage="completed")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        try:
            if engine is not None:
                engine.close()
        finally:
            signal.signal(signal.SIGTERM, previous)
            receipt.update(finished_utc=utc_now(), cpu_wall_seconds=time.perf_counter() - started)
            save()
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "embeddings", "temporal-dir", "output-dir", "cache-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--cap", type=int, default=128)
    parser.add_argument("--formats", default="")
    args = parser.parse_args(argv)
    receipt = run(args.model_dir, args.embeddings, args.temporal_dir, args.output_dir,
                  args.cache_dir, cap=args.cap, formats=args.formats)
    print(json.dumps({key: receipt[key] for key in ("status", "temporal_hidden_bit_exact", "head_comparison",
                                                   "reset_bit_exact", "cpu_wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
