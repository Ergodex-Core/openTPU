#!/usr/bin/env python3
"""Bounded real-checkpoint resident depth -> Mimi RVQ CPU/ISA acceptance check.

Native PyTorch is an independent diagnostic reference. The inference programs
consume their own resident generated IDs; no CPU/GPU serving fallback exists.
No PCM, physical board, model download, or complete speech inference is run.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.update(OTPU_BUILD_JOBS="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import signal
import struct
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opentpu import target
from opentpu.llm.personaplex_depth_chain import DepthChainSpec, DepthChainImage, DepthChainEngine
from opentpu.llm.personaplex_mimi import MimiRVQSpec, normalize_codebooks
from opentpu.llm.personaplex_weights import MLXWeights
from tools.personaplex_depth_chain_check import (
    preflight as depth_preflight, payload_sha256, layer_parts, exact_bytes,
    sha256, raw_sha256, _save_array,
)
from tools.personaplex_isa_check import precision_metrics
os.environ["OTPU_BUILD_JOBS"] = "1"

NAMES = ("semantic", "acoustic", "semantic_projected", "acoustic_projected", "output")
DATA = ROOT / "tools/data/personaplex"
COMMON_TARGET = "f2-hbm:p4rl8a64c150"


def dependency_versions():
    result = {}
    for name in ("numpy", "torch", "safetensors", "moshi"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            # The pre-existing Moshi source tree has no distribution metadata;
            # its independently pinned source hashes still remain mandatory.
            result[name] = None
    return result


def codec_header(path, contract):
    """Pin small metadata before selectively loading cached checkpoint tensors."""
    path = Path(path)
    before = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    expected = contract["codec_header"]
    if before["bytes"] != expected["file_bytes"]:
        raise ValueError("Cached codec file size differs from the source audit")
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if size != expected["header_bytes"] or size > 1 << 20:
            raise ValueError("Cached codec header size differs")
        header = handle.read(size)
    if hashlib.sha256(header).hexdigest() != expected["header_sha256"]:
        raise ValueError("Cached codec header hash differs")
    if len([name for name in json.loads(header) if name != "__metadata__"]) != expected["tensor_count"]:
        raise ValueError("Cached codec tensor count differs")
    return before


def native_quantizer(codec_path, contract):
    """Use the installed original source; strict load all32, then activate8."""
    import torch
    from safetensors import safe_open
    from moshi.quantization.vq import SplitResidualVectorQuantizer
    native_root = Path(inspect.getfile(SplitResidualVectorQuantizer)).parents[1]
    for name, digest in contract["installed_source_sha256"].items():
        if sha256(native_root / name) != digest:
            raise ValueError(f"Native codec source changed: {name}")
    native = SplitResidualVectorQuantizer(**contract["native_constructor"])
    with safe_open(codec_path, framework="pt", device="cpu") as handle:
        state = {name.removeprefix("quantizer."): handle.get_tensor(name)
                 for name in handle.keys() if name.startswith("quantizer.")}
    native.load_state_dict(state, strict=True)
    native.set_num_codebooks(contract["active_codebooks"])
    native.eval()
    del state
    layers = [native.rvq_first.vq.layers[0], *native.rvq_rest.vq.layers[:7]]
    sums = np.stack([p._codebook.embedding_sum.detach().numpy().copy() for p in layers])
    usages = np.stack([p._codebook.cluster_usage.detach().numpy().copy() for p in layers])
    tables = normalize_codebooks(sums, usages)
    for index, layer in enumerate(layers):
        exact_bytes(tables[index], layer._codebook.embedding.detach().numpy(),
                    f"offline normalization/native centroid table{index}")
    projections = np.stack([p.output_proj.weight.detach().numpy()[:, :, 0].copy()
                            for p in (native.rvq_first, native.rvq_rest)])
    return native, tables, projections, native_root


def native_pass(native, ids):
    """Capture original branch inputs/outputs during the real native decode call."""
    import torch
    codes = torch.from_numpy(np.asarray(ids[:8], np.int64).reshape(1, 8, 1))
    result, handles = {}, []
    for label, branch in (("semantic", native.rvq_first), ("acoustic", native.rvq_rest)):
        def before(module, inputs, label=label):
            result[label] = inputs[0].detach().numpy().reshape(-1).copy()
        def after(module, inputs, value, label=label):
            result[label + "_projected"] = value.detach().numpy().reshape(-1).copy()
        handles.extend((branch.output_proj.register_forward_pre_hook(before),
                        branch.output_proj.register_forward_hook(after)))
    try:
        with torch.no_grad():
            result["output"] = native.decode(codes).detach().numpy().reshape(-1).copy()
    finally:
        for handle in handles:
            handle.remove()
    return result


def _baseline_chain(folder):
    folder = Path(folder).resolve()
    path = folder / "depth_chain_check.json"
    receipt = json.loads(path.read_text())
    if (receipt.get("status") != "completed" or receipt.get("reset_bit_exact") is not True
            or receipt.get("all_layer_kv_reset_bit_exact") is not True
            or receipt.get("execution_target", {}).get("variant") != "p4rl8a64c150"
            or receipt.get("execution_target", {}).get("arch_hash") != "dd6b92a6"
            or receipt.get("isa_vs_native_original_id_matches") != 16):
        raise ValueError("Completed exact common-profile checkpoint chain baseline required")
    for name, metadata in receipt["arrays"].items():
        if sha256(folder / (name + ".npy")) != metadata["sha256"]:
            raise ValueError(f"Saved depth baseline changed: {name}")
    return folder, path, receipt


def run(model_dir, codec_path, token_dir, conditioning_dir, stack_dir,
        chain_dir, output_dir, *, target_name=COMMON_TARGET):
    import torch
    torch.set_num_threads(1)
    if target_name != COMMON_TARGET:
        raise ValueError("Mimi qualification requires the explicit shared Gemma target")
    contract_path = DATA / "mimi_rvq_contract.json"
    contract = json.loads(contract_path.read_text())
    common = json.loads((DATA / "shared_gemma150_contract.json").read_text())
    folder, baseline_path, baseline = _baseline_chain(chain_dir)
    guard = depth_preflight(model_dir, token_dir, conditioning_dir, stack_dir, output_dir,
                            frame_index=baseline["frame_index"], target_name=target_name)
    cfg = target.config(guard["target"])
    if (guard["target"].to_dict()["arch_hash"] != common["expected_arch_hash"]
            or sha256(guard["target"].path) != common["target_source_sha256"]):
        raise ValueError("Exact shared target configuration differs")
    for name, digest in common["pinned_runtime_source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Pinned common runtime dependency changed: {name}")
    codec_path, output_dir = Path(codec_path).resolve(), Path(output_dir).resolve()
    if output_dir in folder.parents or folder in output_dir.parents or output_dir == folder:
        raise ValueError("Output must be isolated from the completed depth baseline")
    codec_before = codec_header(codec_path, contract)
    spec, mimi_spec = DepthChainSpec(), MimiRVQSpec(**contract["device_spec"])
    image = DepthChainImage(spec, cfg, mimi_spec=mimi_spec)
    budget = image.budget()  # complete fit+compile before any large payload
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in
               (Path(__file__), ROOT / "opentpu/llm/personaplex_mimi.py",
                ROOT / "opentpu/llm/personaplex_depth_chain.py", contract_path,
                DATA / "shared_gemma150_contract.json")}
    receipt = {"status": "running", "stage": "materializing", "started_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "isa", "hardware_used": False, "hardware_performance_measured": False,
        "full_fpga_model_inference": False, "speech_generated": False, "quality_accepted": False,
        "scope": "One saved temporal frame ->16 device-generated depth IDs ->8-codebook Mimi RVQ latent; reset replay",
        "source_files": sources, "pinned_runtime_source_sha256": common["pinned_runtime_source_sha256"],
        "native_source_sha256": contract["installed_source_sha256"],
        "target_sha256": common["target_source_sha256"], "execution_target": guard["target"].to_dict(),
        "runtime_revision": common["runtime_revision"], "image_context": common["image"],
        "checkpoint_revision": baseline["checkpoint_revision"], "codec_file": codec_before,
        "codec_header_sha256": contract["codec_header"]["header_sha256"],
        "checkpoint_payload_hashes_verified": False, "baseline_sha256": {**guard["baseline_hashes"], str(baseline_path): sha256(baseline_path)},
        "frame_index": baseline["frame_index"], "budget": budget, "arrays": {},
        "precision": "FP32 centroids/sums, MX INT8 branch projections/activations, FP32 output add; original native FP32 reference",
        "range_guard_scope": "ISA simulator F_BOUND only; no physical range-protection claim",
        "package_versions": dependency_versions()}
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "mimi_check.json"
    def save():
        temp = receipt_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temp.replace(receipt_path)
    def interrupt(signum, frame):
        raise InterruptedError("Mimi check interrupted; inspect receipt before retry")
    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    engine, started = None, time.perf_counter()
    save()
    try:
        native, tables, matrices, native_root = native_quantizer(str(codec_path), contract)
        receipt["payload_hashes"] = {"mimi_tables": payload_sha256((str(i), t) for i, t in enumerate(tables)),
                                     "mimi_projections": payload_sha256((str(i), t) for i, t in enumerate(matrices))}
        weights = MLXWeights(model_dir)
        projections, embeddings, heads = weights.depth_input_projections, weights.depth_input_embeddings, weights.audio_heads
        layers = []
        for index in range(spec.layers):
            receipt.update(stage="materializing_depth_layer", source_layer=index); save()
            lazy = weights.depth_layer_weights(index)
            layers.append({"norm1": lazy["norm1"], "norm2": lazy["norm2"],
                           "steps": [{key: bank[key] for key in bank} for bank in lazy["steps"]]})
        receipt["payload_hashes"].update(
            depth_projections=payload_sha256((str(i), t) for i, t in enumerate(projections)),
            depth_embeddings=payload_sha256((str(i), t) for i, t in enumerate(embeddings)),
            depth_heads=payload_sha256((str(i), t) for i, t in enumerate(heads)),
            depth_layers=payload_sha256(layer_parts(layers)))
        receipt.update(stage="building_shared_image"); save()
        engine = DepthChainEngine(projections, embeddings, layers, heads, spec=spec, cfg=cfg,
                                  mimi_tables=tables, mimi_projections=matrices, mimi_spec=mimi_spec)
        receipt["budget"] = engine.image.budget()
        original_read, original_write = engine.backend.read, engine.backend.write
        transfers = {"reads": 0, "writes": 0, "written_bytes": 0}
        def tracked_read(*args, **kwargs):
            transfers["reads"] += 1
            return original_read(*args, **kwargs)
        def tracked_write(s, address, value):
            transfers["writes"] += 1; transfers["written_bytes"] += np.asarray(value).nbytes
            return original_write(s, address, value)
        engine.backend.read, engine.backend.write = tracked_read, tracked_write
        receipt.update(stage="resident_depth_then_rvq"); save()
        engine.run_frame(guard["hidden"], guard["text_id"])
        before_mimi = dict(transfers)
        engine.run_mimi()
        if transfers != before_mimi or transfers != {"reads": 0, "writes": 2, "written_bytes": 16392}:
            raise AssertionError("Unexpected interstage activation/ID/weight transfer")
        receipt["runtime_transfers_through_latent"] = dict(transfers)
        engine.backend.read, engine.backend.write = original_read, original_write
        isa = {name: engine.read_mimi(name) for name in NAMES}
        ids = engine.read_ids()
        exact_bytes(ids, np.load(folder / "isa_ids.npy", allow_pickle=False), "resident IDs versus prior same-target chain")
        for name, read in (("projected", engine.read_projected), ("conditioned", engine.read_conditioned),
                           ("depth", engine.read_output), ("logits", engine.read_logits)):
            exact_bytes(read(), np.load(folder / ("isa_" + name + ".npy"), allow_pickle=False)[-1], name)
        caches = [engine.read_cache(i) for i in range(spec.layers)]
        for index, cache in enumerate(caches):
            if raw_sha256(cache) != baseline["arrays"][f"isa_kv_layer{index}"]["raw_sha256"]:
                raise AssertionError("Depth KV differs after shared Mimi execution")
        receipt.update(stage="native_quantizer_reference", isa_ids=ids.tolist(),
                       native_original_ids=baseline["native_original_ids"], prior_chain_ids_bit_exact=True,
                       prior_chain_last_vectors_and_all_kv_bit_exact=True); save()
        reference = native_pass(native, ids)
        for name in ("semantic", "acoustic"):
            exact_bytes(isa[name], reference[name], "native source lookup/sum " + name)
        receipt["source_lookup_sums_bit_exact"] = True
        receipt["precision_against_native"] = {name: precision_metrics(isa[name], reference[name]) for name in NAMES}
        replay_native = native_pass(native, ids)
        for name in NAMES:
            exact_bytes(replay_native[name], reference[name], "native reset replay " + name)
        receipt.update(stage="reset_replay"); save()
        engine.reset()
        try:
            engine.read_mimi()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Reset did not invalidate Mimi output")
        engine.run_frame(guard["hidden"], guard["text_id"])
        engine.run_mimi()
        for name in NAMES:
            exact_bytes(engine.read_mimi(name), isa[name], "ISA reset replay " + name)
        exact_bytes(engine.read_ids(), ids, "all resident IDs reset replay")
        for index, cache in enumerate(caches):
            exact_bytes(engine.read_cache(index), cache, "all depth KV reset replay")
        receipt.update(reset_bit_exact=True, ids_reset_bit_exact=True, all_layer_kv_reset_bit_exact=True,
                       native_replay_bit_exact=True)
        _save_array(output_dir, receipt, "isa_ids", ids)
        for name in NAMES:
            _save_array(output_dir, receipt, "isa_" + name, isa[name])
            _save_array(output_dir, receipt, "native_" + name, reference[name])
        for path, digest in receipt["baseline_sha256"].items():
            if sha256(path) != digest:
                raise RuntimeError("Saved baseline changed during Mimi validation")
        if codec_header(codec_path, contract) != codec_before:
            raise RuntimeError("Codec checkpoint metadata changed during check")
        for name, digest in sources.items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Mimi source/contract changed during check")
        for name, digest in common["pinned_runtime_source_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Pinned runtime changed during check")
        for name, digest in contract["installed_source_sha256"].items():
            if sha256(native_root / name) != digest:
                raise RuntimeError("Native codec source changed during check")
        receipt.update(status="completed", stage="completed")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        try:
            if engine is not None:
                engine.close()
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), cpu_wall_seconds=time.perf_counter() - started)
            save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "codec-path", "token-dir", "conditioning-dir", "stack-dir", "chain-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--target", default=COMMON_TARGET)
    args = parser.parse_args()
    receipt = run(**{key: value for key, value in vars(args).items() if key != "target"}, target_name=args.target)
    print(json.dumps({key: receipt[key] for key in ("status", "source_lookup_sums_bit_exact", "reset_bit_exact", "precision_against_native", "cpu_wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
