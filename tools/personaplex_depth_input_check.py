#!/usr/bin/env python3
"""Bounded cached-checkpoint CPU/ISA depth-input check; no depth layers or FPGA.

Uses saved temporal hidden states without resimulating temporal inference.
The native PyTorch method is intercepted immediately before the depth transformer
so the independent checkpoint reference measures only projection + embedding.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.update(OTPU_BUILD_JOBS="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opentpu import target
from opentpu.llm.personaplex_weights import MLXWeights
from opentpu.llm.personaplex_depth_input import DepthInputSpec, DepthInputImage, DepthInputEngine
from tools.personaplex_isa_check import load_array, precision_metrics, PINNED_REVISION, PROFILE
os.environ["OTPU_BUILD_JOBS"] = "1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if (a.dtype != np.float32 or b.dtype != np.float32 or a.shape != b.shape
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.array_equal(a.view(np.uint32), b.view(np.uint32))):
        raise AssertionError(f"{label}: FP32 bits differ")


def source_input(native, hidden, codebook, token):
    """Execute actual source forward_depformer only through the input-add stage."""
    import torch
    from moshi.models.lm import LMModel
    class Captured(Exception):
        pass
    def stop(value):
        raise Captured(value.detach().numpy().reshape(-1).copy())
    native.depformer = stop
    try:
        with torch.no_grad():
            LMModel.forward_depformer(native, codebook,
                torch.tensor([[[token]]], dtype=torch.long),
                torch.from_numpy(np.asarray(hidden, np.float32)).reshape(1, 1, -1))
    except Captured as captured:
        return captured.args[0]
    raise AssertionError("Native reference did not stop before depth transformer")


def run(model_dir, token_dir, output_dir):
    import torch
    from moshi.models.lm import LMModel, ScaledEmbedding
    torch.set_num_threads(1)
    model_dir, token_dir, output_dir = (Path(p).resolve() for p in (model_dir, token_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    for protected in (ROOT, model_dir, token_dir):
        if output_dir == protected or output_dir in protected.parents or protected in output_dir.parents:
            raise ValueError("Output must be isolated from source, checkpoint and saved baseline")
    baseline_path = token_dir / "token_check.json"
    if baseline_path.stat().st_size > 1 << 20:
        raise ValueError("Baseline receipt exceeds1MiB")
    baseline = json.loads(baseline_path.read_text())
    hidden_path, token_path = token_dir / "hidden.npy", token_dir / "tokens.npy"
    if (baseline.get("status") != "completed" or baseline.get("reset_bit_exact") is not True
            or baseline.get("checkpoint_revision") != PINNED_REVISION
            or sha256(hidden_path) != baseline["hidden_output"]["sha256"]
            or sha256(token_path) != baseline["tokens_output"]["sha256"]):
        raise ValueError("Completed saved token baseline or output hashes do not match")
    hidden = load_array(hidden_path, dtype="float32")
    tokens = np.load(token_path, allow_pickle=False)
    if hidden.shape != (4, 4096) or tokens.shape != (4,) or tokens.dtype.kind not in "iu":
        raise ValueError("Expected the saved four-frame temporal/token outputs")
    weights = MLXWeights(model_dir)
    if weights.spec.hidden != DepthInputSpec().hidden:
        raise ValueError("Expected production temporal width for depth conditioning")
    meta = weights.depth_conditioning_metadata()
    files = {}
    for name in ("config.json", "embeddings.safetensors", "depformer.safetensors"):
        path = model_dir / name
        stat = path.stat()
        files[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if name in baseline["checkpoint_files"] and files[name] != baseline["checkpoint_files"][name]:
            raise ValueError(f"Checkpoint metadata changed for {name}")
        revision = model_dir / ".cache/huggingface/download" / (name + ".metadata")
        with revision.open() as handle:
            if handle.readline().strip() != PINNED_REVISION:
                raise ValueError(f"Checkpoint revision mismatch: {name}")
    cfg = target.config(target.load(str(PROFILE)))
    spec = DepthInputSpec()
    layout = DepthInputImage(spec, cfg)
    program = layout.compile()
    source = Path(inspect.getfile(LMModel))
    receipt = {"status": "running", "stage": "building_image", "started_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "isa", "hardware_used": False, "hardware_performance_measured": False,
        "full_fpga_model_inference": False, "speech_generated": False, "quality_accepted": False,
        "scope": "Separate native depth-input projection + previous-token embedding; no depth layers",
        "input_scope": "Four saved real temporal ISA hidden states; bounded supplied previous-token IDs",
        "checkpoint_revision": PINNED_REVISION, "checkpoint_files": files, "component_metadata": meta,
        "payload_identity": "Revision metadata and size/mtime, not exhaustive checkpoint payload hashing",
        "baseline_receipt_sha256": sha256(baseline_path), "baseline_hidden_sha256": sha256(hidden_path),
        "baseline_tokens_sha256": sha256(token_path), "native_lm_source_sha256": sha256(source),
        "spec": asdict(spec), "required_bytes": layout.required_bytes,
        "projection_bytes": layout.projection_bytes, "embedding_bytes": layout.embedding_bytes,
        "program_words": len(program[0]) * 8, "dram_bytes": cfg.DRAM_BYTES, "imem_words": cfg.IMEM_WORDS,
        "cases": [], "source_files": {str(p.relative_to(ROOT)): sha256(p) for p in
            [Path(__file__), ROOT / "opentpu/llm/personaplex_depth_input.py", ROOT / "opentpu/llm/personaplex_weights.py",
             ROOT / "opentpu/llm/personaplex_embeddings.py", ROOT / "opentpu/llm/personaplex_audio_heads.py"]}}
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "depth_input_check.json"
    def save():
        temp = receipt_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n"); temp.replace(receipt_path)
    def interrupted(signum, frame):
        raise InterruptedError("Depth-input diagnostic interrupted; inspect receipt before retry")
    previous = signal.signal(signal.SIGTERM, interrupted)
    started, engine = time.perf_counter(), None
    save()
    try:
        build_started = time.perf_counter()
        projections, tables = weights.depth_input_projections, weights.depth_input_embeddings
        linears, embeddings = [], []
        for matrix in projections:
            linear = torch.nn.Linear(spec.hidden, spec.depth, bias=False)
            linear.weight = torch.nn.Parameter(torch.from_numpy(matrix), requires_grad=False)
            linears.append(linear)
        for table in tables:
            embedding = ScaledEmbedding(len(table), spec.depth, norm=False, zero_idx=-1)
            embedding.weight = torch.nn.Parameter(torch.from_numpy(table), requires_grad=False)
            embeddings.append(embedding)
        native = SimpleNamespace(depformer_multi_linear=True, depformer_in=linears,
            depformer_text_emb=embeddings[0], depformer_emb=embeddings[1:])
        engine = DepthInputEngine(projections, tables, spec=spec, cfg=cfg)
        receipt["image_build_cpu_seconds"] = time.perf_counter() - build_started
        requests = [(cb, i % 4, int(tokens[i % 4]) if cb == 0 else (cb * 127 + 23) % 2048)
                    for i, cb in enumerate((15, 0, 8, 3, 14, 1, 7, 12, 2, 9, 4, 13, 5, 11, 6, 10))]
        requests.extend([(0, 0, -1), (7, 1, -1), (0, 2, 32000), (15, 3, 2048)])
        actuals, projected, expected, source_projected = [], [], [], []
        for case, (cb, frame, token) in enumerate(requests):
            receipt.update(stage="checking_case", case=case); save()
            reference = source_input(native, hidden[frame], cb, token)
            with torch.no_grad():
                original_projection = linears[cb](torch.from_numpy(hidden[frame])).numpy()
                embedding = embeddings[cb](torch.tensor([token])).numpy().reshape(-1)
            engine.step(hidden[frame], cb, token)
            value, projection = engine.read_output(), engine.read_projected()
            # Isolate the addition from INT8 projection loss. This is a validation
            # reference only; production step never reads/changes the output.
            added = (torch.from_numpy(projection) + torch.from_numpy(embedding)).numpy()
            exact(value, added, f"case{case} embedding addition")
            actuals.append(value); projected.append(projection); expected.append(reference)
            source_projected.append(original_projection)
            receipt["cases"].append({"case": case, "codebook": cb, "frame": frame, "previous_token": token,
                "embedding_addition_fp32_bit_exact": True,
                "projection_precision": precision_metrics(projection, original_projection),
                "conditioned_precision": precision_metrics(value, reference), "stats": engine.last_stats})
            save()
        engine.reset()
        cb, frame, token = requests[0]
        engine.step(hidden[frame], cb, token)
        exact(engine.read_projected(), projected[0], "projection reset")
        exact(engine.read_output(), actuals[0], "conditioning reset")
        receipt.update(reset_bit_exact=True, projection_precision=precision_metrics(np.stack(projected), np.stack(source_projected)),
                       conditioned_precision=precision_metrics(np.stack(actuals), np.stack(expected)))
        for name, value in (("requests", np.asarray(requests, np.int64)), ("conditioned", np.stack(actuals)),
                            ("projected", np.stack(projected)), ("source_conditioned", np.stack(expected)),
                            ("source_projected", np.stack(source_projected))):
            path = output_dir / (name + ".npy")
            with path.open("xb") as handle:
                np.save(handle, value, allow_pickle=False)
            receipt[name + "_output"] = {"sha256": sha256(path), "shape": list(value.shape), "dtype": str(value.dtype)}
        if (sha256(baseline_path) != receipt["baseline_receipt_sha256"]
                or sha256(hidden_path) != receipt["baseline_hidden_sha256"]
                or sha256(token_path) != receipt["baseline_tokens_sha256"]
                or sha256(source) != receipt["native_lm_source_sha256"]):
            raise RuntimeError("Baseline or source changed during check")
        for name, before in files.items():
            stat = (model_dir / name).stat()
            if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != before:
                raise RuntimeError("Checkpoint changed during check")
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
            receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), cpu_wall_seconds=time.perf_counter() - started)
            save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True); parser.add_argument("--token-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    receipt = run(args.model_dir, args.token_dir, args.output_dir)
    print(json.dumps({k: receipt[k] for k in ("status", "reset_bit_exact", "projection_precision", "conditioned_precision", "cpu_wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
