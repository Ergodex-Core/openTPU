#!/usr/bin/env python3
"""Bounded real-checkpoint resident depth-chain check on the CPU ISA simulator.

One saved temporal hidden/text pair drives sixteen codebooks. Native PyTorch
references are diagnostic only; the ISA engine generates its own feedback IDs.
This script never opens a board, downloads a model, or generates PCM speech.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
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
from opentpu.llm.personaplex_depth_chain import DepthChainSpec, DepthChainImage, DepthChainEngine
from opentpu.llm.personaplex_depth_layer import DepthLayerSpec
from opentpu.llm.personaplex_weights import MLXWeights
from tools.personaplex_depth_input_check import exact, source_input
from tools.personaplex_depth_layer_check import (
    EXPECTED_NATIVE_SOURCE, check_native_positions, native_stack,
)
from tools.personaplex_isa_check import PINNED_REVISION, PROFILE, precision_metrics
# The imported historical ISA tool sets its own worker default; restore this bound.
os.environ["OTPU_BUILD_JOBS"] = "1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_sha256(array):
    """Hash array bytes, separately from the saved .npy container hash."""
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value).hexdigest()


def payload_sha256(parts):
    """Hash named, typed checkpoint arrays without allocating a second payload."""
    digest = hashlib.sha256()
    count = 0
    for name, array in parts:
        value = np.asarray(array)
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValueError(f"Invalid source payload {name}")
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)],
                                 separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value)
        count += value.nbytes
    return {"sha256": digest.hexdigest(), "raw_bytes": count}


def layer_parts(layers):
    for index, layer in enumerate(layers):
        for name in ("norm1", "norm2"):
            yield f"layer{index}.{name}", layer[name]
        for step, bank in enumerate(layer["steps"]):
            for name in ("wq", "wk", "wv", "wo", "wg", "wu", "wd"):
                yield f"layer{index}.step{step}.{name}", bank[name]


def exact_bytes(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise AssertionError(f"{label}: raw array bytes differ")


def _receipt(path):
    path = Path(path)
    if path.stat().st_size > 1 << 20:
        raise ValueError(f"Receipt exceeds 1MiB: {path}")
    return json.loads(path.read_text())


def _checked_output(folder, receipt, name):
    path = folder / (name + ".npy")
    if sha256(path) != receipt[name + "_output"]["sha256"]:
        raise ValueError(f"Baseline {name} array hash differs")
    return path


def preflight(model_dir, token_dir, conditioning_dir, stack_dir, output_dir, *, frame_index=0, target_name=None):
    """Require a saved pinned chain of receipts before touching large payloads."""
    if type(frame_index) is not int or not 0 <= frame_index < 4:
        raise ValueError("Frame index must select one of the four saved temporal inputs")
    model_dir, token_dir, conditioning_dir, stack_dir, output_dir = (
        Path(p).resolve() for p in
        (model_dir, token_dir, conditioning_dir, stack_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    for protected in (ROOT, model_dir, token_dir, conditioning_dir, stack_dir):
        if output_dir == protected or output_dir in protected.parents or protected in output_dir.parents:
            raise ValueError("Output must be isolated from source, checkpoint and baselines")

    token_path = token_dir / "token_check.json"
    input_path = conditioning_dir / "depth_input_check.json"
    stack_path = stack_dir / "depth_layer_check.json"
    token, conditioning, stack = (_receipt(p) for p in (token_path, input_path, stack_path))
    if (token.get("status") != "completed" or token.get("reset_bit_exact") is not True
            or token.get("checkpoint_revision") != PINNED_REVISION):
        raise ValueError("Completed pinned token baseline with reset is required")
    hidden_path = _checked_output(token_dir, token, "hidden")
    tokens_path = _checked_output(token_dir, token, "tokens")
    hidden = np.load(hidden_path, allow_pickle=False)
    tokens = np.load(tokens_path, allow_pickle=False)
    if (hidden.shape != (4, 4096) or hidden.dtype != np.float32
            or not np.isfinite(hidden).all() or tokens.shape != (4,)
            or tokens.dtype.kind not in "iu" or not 0 <= int(tokens[frame_index]) < 32001):
        raise ValueError("Expected four finite saved temporal frames and a valid selected text ID")
    if (conditioning.get("status") != "completed" or conditioning.get("reset_bit_exact") is not True
            or conditioning.get("checkpoint_revision") != PINNED_REVISION
            or conditioning.get("baseline_receipt_sha256") != sha256(token_path)
            or conditioning.get("baseline_hidden_sha256") != sha256(hidden_path)
            or conditioning.get("baseline_tokens_sha256") != sha256(tokens_path)):
        raise ValueError("Conditioning baseline does not descend from saved token baseline")
    baseline_arrays = [hidden_path, tokens_path]
    for name in ("requests", "conditioned", "projected", "source_conditioned", "source_projected"):
        baseline_arrays.append(_checked_output(conditioning_dir, conditioning, name))
    if (stack.get("status") != "completed" or stack.get("reset_bit_exact") is not True
            or stack.get("kv_reset_bit_exact") is not True
            or stack.get("native_reset_bit_exact") is not True
            or stack.get("spec", {}).get("layers") != 6
            or stack.get("checkpoint_revision") != PINNED_REVISION
            or stack.get("baseline_receipt_sha256") != sha256(input_path)):
        raise ValueError("Completed six-layer baseline from the same conditioning receipt is required")
    for name in ("inputs", "isa_hidden", "native_hidden", "isa_kv", "native_layer_hidden"):
        baseline_arrays.append(_checked_output(stack_dir, stack, name))

    files = {}
    for name in ("config.json", "embeddings.safetensors", "depformer.safetensors"):
        stat = (model_dir / name).stat()
        files[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        for previous in (token, conditioning, stack):
            if name in previous.get("checkpoint_files", {}) and files[name] != previous["checkpoint_files"][name]:
                raise ValueError(f"Checkpoint metadata differs from baseline: {name}")
        with (model_dir / ".cache/huggingface/download" / (name + ".metadata")).open() as handle:
            if handle.readline().strip() != PINNED_REVISION:
                raise ValueError(f"Checkpoint revision mismatch: {name}")

    from moshi.modules.transformer import StreamingTransformerLayer
    native_root = Path(inspect.getfile(StreamingTransformerLayer)).parents[1]
    for name, digest in EXPECTED_NATIVE_SOURCE.items():
        if sha256(native_root / name) != digest:
            raise ValueError(f"Native source contract changed: {name}")
    selected_target = target.load(target_name or str(PROFILE))
    source_paths = [Path(__file__), ROOT / "opentpu/llm/personaplex_depth_chain.py",
                    ROOT / "opentpu/llm/personaplex_weights.py",
                    ROOT / "opentpu/llm/personaplex_audio_heads.py",
                    ROOT / "opentpu/llm/personaplex_depth_input.py",
                    ROOT / "tools/personaplex_depth_input_check.py",
                    ROOT / "tools/personaplex_depth_layer_check.py", selected_target.path,
                    *[ROOT / ("opentpu/" + name + ".py") for name in
                      ("isa", "fp32", "quant", "compiler", "isasim", "target")]]
    for name in ("opentpu/llm/personaplex_depth_layer.py", "opentpu/llm/personaplex_weights.py"):
        if sha256(ROOT / name) != stack["source_files"][name]:
            raise ValueError(f"Core source differs from the completed six-layer baseline: {name}")
    return {"paths": {"token_receipt": token_path, "conditioning_receipt": input_path,
                      "stack_receipt": stack_path, "hidden": hidden_path, "tokens": tokens_path},
            "files": files, "native_root": native_root, "target": selected_target,
            "source_files": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
            "hidden": hidden[frame_index].copy(), "text_id": int(tokens[frame_index]),
            "baseline_hashes": {str(p): sha256(p) for p in
                                (token_path, input_path, stack_path, *baseline_arrays)},
            "native_source_sha256": EXPECTED_NATIVE_SOURCE}


def make_native_input(projections, embeddings, spec):
    """Use the original forward_depformer preparation with source FP32 arrays."""
    import torch
    from moshi.models.lm import ScaledEmbedding
    linears, tables = [], []
    for matrix in projections:
        linear = torch.nn.Linear(spec.hidden, spec.depth, bias=False)
        linear.weight = torch.nn.Parameter(torch.from_numpy(matrix), requires_grad=False)
        linears.append(linear)
    for table in embeddings:
        embedding = ScaledEmbedding(len(table), spec.depth, norm=False, zero_idx=-1)
        embedding.weight = torch.nn.Parameter(torch.from_numpy(table), requires_grad=False)
        tables.append(embedding)
    native = SimpleNamespace(depformer_multi_linear=True, depformer_in=linears,
                             depformer_text_emb=tables[0], depformer_emb=tables[1:])
    return native, linears


def native_chain_pass(model, source_input_model, linears, heads, hidden, text_id,
                      *, forced_ids=None):
    """Original FP32 stack/head, with its own or explicit diagnostic feedback."""
    import torch
    if forced_ids is not None and len(forced_ids) != len(heads):
        raise ValueError("Teacher-forced ID count must equal the number of codebooks")
    inputs, projections, depths, logits, ids = [], [], [], [], []
    positions = [[] for _ in model.layers]
    with torch.no_grad(), model.streaming(1):
        for index, layer in enumerate(model.layers):
            cache = layer.self_attn._streaming_state.kv_cache
            complete = cache.complete
            def record(k, v, complete=complete, index=index):
                result = complete(k, v)
                positions[index].append(result.positions.detach().cpu().tolist())
                return result
            cache.complete = record
        vector = torch.from_numpy(hidden)
        for step in range(len(heads)):
            previous = (text_id if step == 0 else
                        int(forced_ids[step - 1]) if forced_ids is not None else ids[-1])
            value = source_input(source_input_model, hidden, step, previous)
            projection = linears[step](vector).detach().numpy().reshape(-1).copy()
            depth = model(torch.from_numpy(value).reshape(1, 1, -1)).detach().numpy().reshape(-1).copy()
            scores = torch.nn.functional.linear(torch.from_numpy(depth),
                                                torch.from_numpy(heads[step])).detach().numpy().copy()
            inputs.append(value); projections.append(projection); depths.append(depth); logits.append(scores)
            ids.append(int(np.argmax(scores)))  # NumPy chooses the lowest ID on a tie.
    for per_layer in positions:
        check_native_positions(per_layer, len(heads))
    return {"projected": np.stack(projections), "conditioned": np.stack(inputs),
            "depth": np.stack(depths), "logits": np.stack(logits),
            "ids": np.asarray(ids, np.int32), "cache_positions": positions}


def _checked_fp32(value, shape, label):
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
        raise AssertionError(f"{label}: expected finite FP32 {shape}, got {array.shape} {array.dtype}")
    return array.copy()


def collect_isa_step(engine, step, spec):
    """Diagnostic reads occur after execution and never drive the next ISA step."""
    if engine.read_counter() != step + 1:
        raise AssertionError(f"ISA counter did not advance once at step {step}")
    return {"projected": _checked_fp32(engine.read_projected(), (spec.depth,), "projected"),
            "conditioned": _checked_fp32(engine.read_conditioned(), (spec.depth,), "conditioned"),
            "depth": _checked_fp32(engine.read_output(), (spec.depth,), "depth"),
            "logits": _checked_fp32(engine.read_logits(), (spec.head_vocab,), "logits")}


def _save_array(output_dir, receipt, name, value):
    array = np.ascontiguousarray(value)
    path = output_dir / (name + ".npy")
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    receipt["arrays"][name] = {"sha256": sha256(path), "raw_sha256": raw_sha256(array),
                               "shape": list(array.shape), "dtype": str(array.dtype)}


def run(model_dir, token_dir, conditioning_dir, stack_dir, output_dir, *, frame_index=0, target_name=None):
    import torch
    torch.set_num_threads(1)
    model_dir, output_dir = Path(model_dir).resolve(), Path(output_dir).resolve()
    guard = preflight(model_dir, token_dir, conditioning_dir, stack_dir, output_dir, frame_index=frame_index, target_name=target_name)
    weights = MLXWeights(model_dir)
    spec = DepthChainSpec()
    if (spec.hidden != 4096 or spec.depth != 1024 or spec.n_heads != 16
            or spec.ffn != 2816 or spec.steps != 16 or spec.layers != 6
            or spec.text_vocab != 32001 or spec.audio_vocab != 2049
            or spec.head_vocab != 2048 or spec.cache_policy != "native"
            or weights.config["depformer"]["num_layers"] != spec.layers):
        raise ValueError("Checker requires the pinned native six-layer production specification")
    component_metadata = {"conditioning": weights.depth_conditioning_metadata(),
                          "heads": weights.audio_head_metadata(),
                          "layers": [weights.depth_layer_metadata(i) for i in range(spec.layers)]}
    cfg = target.config(guard["target"])
    image = DepthChainImage(spec, cfg)
    programs = image.compile()
    if len(programs) != 3:
        raise AssertionError("Resident chain must compile exactly three fixed programs")
    budget = image.budget()
    receipt = {
        "status": "running", "stage": "materializing_source", "started_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "isa", "hardware_used": False, "hardware_performance_measured": False,
        "full_fpga_model_inference": False, "speech_generated": False, "quality_accepted": False,
        "scope": "One saved temporal frame, sixteen resident depth/codebook steps and full reset replay",
        "input_scope": f"Frame{frame_index} of four saved temporal ISA hidden/text outputs; no temporal resimulation",
        "checkpoint_revision": PINNED_REVISION, "checkpoint_files": guard["files"],
        "checkpoint_payload_hashes_verified": False,
        "payload_identity": "Pinned revision and size/mtime guard; materialized stage-array SHA-256 hashes",
        "baseline_sha256": guard["baseline_hashes"],
        "native_source_sha256": guard["native_source_sha256"],
        "source_files": guard["source_files"], "profile_sha256": sha256(guard["target"].path),
        "execution_target": guard["target"].to_dict(),
        "component_metadata": component_metadata, "spec": asdict(spec), "budget": budget,
        "program_words": [len(p) * 8 for p in programs],
        "frame_index": frame_index, "text_id": guard["text_id"],
        "precision": {"isa": "MX INT8 matrices/activations/KV and ISA FP32 elementwise",
                      "native": "Original dequantized checkpoint FP32 PyTorch reference"},
        "steps": [], "reset_steps": [], "arrays": {},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "depth_chain_check.json"
    def save():
        temp = receipt_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temp.replace(receipt_path)
    def interrupted(signum, frame):
        raise InterruptedError("Depth-chain check interrupted; inspect isolated receipt")
    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    engine = None
    started = time.perf_counter()
    save()
    try:
        materialize_started = time.perf_counter()
        projections = weights.depth_input_projections
        embeddings = weights.depth_input_embeddings
        heads = weights.audio_heads
        layers = []
        for index in range(spec.layers):
            receipt.update(stage="materializing_source_layer", source_layer=index); save()
            lazy = weights.depth_layer_weights(index)
            layers.append({"norm1": lazy["norm1"], "norm2": lazy["norm2"],
                           "steps": [{key: bank[key] for key in bank} for bank in lazy["steps"]]})
        receipt["payload_hashes"] = {
            "projections": payload_sha256((f"projection{i}", m) for i, m in enumerate(projections)),
            "embeddings": payload_sha256((f"embedding{i}", m) for i, m in enumerate(embeddings)),
            "heads": payload_sha256((f"head{i}", m) for i, m in enumerate(heads)),
            "layers": payload_sha256(layer_parts(layers)),
        }
        receipt["source_materialize_cpu_seconds"] = time.perf_counter() - materialize_started
        receipt.update(stage="native_original_greedy"); save()
        source_input_model, linears = make_native_input(projections, embeddings, spec)
        native = native_stack(layers, DepthLayerSpec(layers=spec.layers))
        original = native_chain_pass(native, source_input_model, linears, heads,
                                     guard["hidden"], guard["text_id"])
        receipt["native_cache_positions"] = original["cache_positions"]
        receipt["native_original_ids"] = original["ids"].tolist()
        receipt.update(stage="building_isa_image"); save()
        build_started = time.perf_counter()
        engine = DepthChainEngine(lambda: projections, lambda: embeddings,
                                  lambda: layers, lambda: heads, spec=spec, cfg=cfg, backend="isa")
        receipt["image_build_cpu_seconds"] = time.perf_counter() - build_started
        # Capture the executed image, not the separate layout-only preflight.
        receipt["budget"] = engine.image.budget()
        receipt.update(stage="simulating_frame"); save()
        engine.begin_frame(guard["hidden"], guard["text_id"])
        isa_parts = {name: [] for name in ("projected", "conditioned", "depth", "logits")}
        for step in range(spec.steps):
            receipt.update(stage="simulating_step", step=step); save()
            engine.step()
            data = collect_isa_step(engine, step, spec)
            for name in isa_parts:
                isa_parts[name].append(data[name])
            receipt["steps"].append({"step": step,
                                     "diagnostic_logit_argmax": int(np.argmax(data["logits"])),
                                     "native_original_id": int(original["ids"][step])})
            save()
        isa = {name: np.stack(items) for name, items in isa_parts.items()}
        ids = np.asarray(engine.read_ids())
        if ids.shape != (spec.steps,) or ids.dtype.kind not in "iu":
            raise AssertionError("ISA audio ID array must contain sixteen integers")
        if np.any(ids < 0) or np.any(ids >= spec.head_vocab):
            raise AssertionError("ISA audio ID outside head vocabulary")
        diagnostic_argmax = np.argmax(isa["logits"], axis=1)
        if not np.array_equal(ids, diagnostic_argmax):
            raise AssertionError("Resident audio IDs differ from independent logit argmax")
        if hasattr(engine, "read_previous_id") and engine.read_previous_id() != int(ids[-1]):
            raise AssertionError("Resident previous ID differs from the final audio ID")
        caches = [np.asarray(engine.read_cache(layer=i)).copy() for i in range(spec.layers)]
        receipt.update(stage="native_teacher_forced"); save()
        forced = native_chain_pass(native, source_input_model, linears, heads,
                                   guard["hidden"], guard["text_id"], forced_ids=ids)
        if original["cache_positions"] != forced["cache_positions"]:
            raise AssertionError("Native cache positions depend on feedback IDs")
        receipt["native_teacher_forced_ids"] = forced["ids"].tolist()
        receipt["isa_ids"] = ids.tolist()
        receipt["isa_vs_native_original_id_matches"] = int(np.count_nonzero(ids == original["ids"]))
        receipt["isa_vs_native_teacher_forced_id_matches"] = int(np.count_nonzero(ids == forced["ids"]))
        receipt["native_feedback_drift_id_matches"] = int(np.count_nonzero(forced["ids"] == original["ids"]))
        receipt["first_isa_vs_original_divergence"] = next(
            (i for i in range(spec.steps) if ids[i] != original["ids"][i]), None)
        receipt["precision_against_teacher_forced"] = {
            name: precision_metrics(isa[name], forced[name]) for name in isa}
        receipt["native_feedback_drift"] = {
            name: precision_metrics(forced[name], original[name]) for name in original
            if name in isa}
        receipt["precision_against_original_greedy"] = {
            name: precision_metrics(isa[name], original[name]) for name in isa}
        receipt.update(stage="reset_replay"); save()
        engine.reset()
        if engine.read_counter() != 0:
            raise AssertionError("ISA counter did not reset to zero")
        engine.begin_frame(guard["hidden"], guard["text_id"])
        for step in range(spec.steps):
            receipt.update(stage="reset_replay_step", step=step); save()
            engine.step()
            replay = collect_isa_step(engine, step, spec)
            for name in isa:
                exact(replay[name], isa[name][step], f"ISA reset {name} step{step}")
            receipt["reset_steps"].append({"step": step, "output_bit_exact": True,
                                           "logits_bit_exact": True,
                                           "conditioned_bit_exact": True,
                                           "projected_bit_exact": True})
            save()
        exact_bytes(np.asarray(engine.read_ids()), ids, "ISA reset all audio IDs")
        for index, expected_cache in enumerate(caches):
            exact_bytes(engine.read_cache(layer=index), expected_cache,
                        f"ISA reset layer{index} raw KV")
        receipt.update(reset_bit_exact=True, ids_reset_bit_exact=True,
                       all_layer_kv_reset_bit_exact=True)
        for name, value in (("temporal_hidden", guard["hidden"]),
                            ("text_id", np.asarray([guard["text_id"]], np.int32)),
                            ("isa_ids", ids), ("native_original_ids", original["ids"]),
                            ("native_teacher_forced_ids", forced["ids"])):
            _save_array(output_dir, receipt, name, value)
        for label, values in (("isa", isa), ("native_original", original),
                              ("native_teacher_forced", forced)):
            for name in ("projected", "conditioned", "depth", "logits"):
                _save_array(output_dir, receipt, label + "_" + name, values[name])
        for index, cache in enumerate(caches):
            _save_array(output_dir, receipt, f"isa_kv_layer{index}", cache)
        for path_str, digest in guard["baseline_hashes"].items():
            if sha256(path_str) != digest:
                raise RuntimeError(f"Baseline changed during check: {path_str}")
        for name, before in guard["files"].items():
            stat = (model_dir / name).stat()
            if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != before:
                raise RuntimeError(f"Checkpoint changed during check: {name}")
        for name, digest in guard["native_source_sha256"].items():
            if sha256(guard["native_root"] / name) != digest:
                raise RuntimeError(f"Native source changed during check: {name}")
        for name, digest in guard["source_files"].items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError(f"Checker/core/profile changed during check: {name}")
        receipt.update(status="completed", stage="completed")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}",
                       traceback=traceback.format_exc())
        raise
    finally:
        try:
            if engine is not None:
                engine.close()
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                           cpu_wall_seconds=time.perf_counter() - started)
            save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "token-dir", "conditioning-dir", "stack-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--frame-index", type=int, choices=range(4), default=0)
    parser.add_argument("--target", help="Explicit target/variant from the active runtime; no compatibility substitution")
    args = parser.parse_args()
    receipt = run(args.model_dir, args.token_dir, args.conditioning_dir,
                  args.stack_dir, args.output_dir, frame_index=args.frame_index, target_name=args.target)
    print(json.dumps({key: receipt[key] for key in
                      ("status", "isa_ids", "native_original_ids", "native_teacher_forced_ids",
                       "isa_vs_native_original_id_matches", "reset_bit_exact", "cpu_wall_seconds")},
                     indent=2))


if __name__ == "__main__":
    main()
