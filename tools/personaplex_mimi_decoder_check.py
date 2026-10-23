#!/usr/bin/env python3
"""Bounded resident depth/RVQ/upsample/decoder streaming check on EC2 CPU ISA.

Original native CPU modules are diagnostic references only. No hardware, PCM,
complete speech inference, downloads or serving fallback is implemented here.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.update(OTPU_BUILD_JOBS="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")

import argparse
from datetime import datetime, timezone
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
from opentpu.llm.personaplex_depth_chain import DepthChainImage, DepthChainEngine, DepthChainSpec
from opentpu.llm.personaplex_mimi import MimiRVQSpec
from opentpu.llm.personaplex_mimi_decoder import MimiDecoderSpec, native_ring_positions
from opentpu.llm.personaplex_weights import MLXWeights
from tools.personaplex_depth_chain_check import (preflight as depth_preflight, sha256, raw_sha256,
    exact_bytes, payload_sha256, layer_parts, _save_array)
from tools.personaplex_mimi_check import (native_quantizer, native_pass, codec_header,
                                         dependency_versions, COMMON_TARGET, DATA)
from tools.personaplex_mimi_decoder_native import native_modules, collect_native
from tools.personaplex_isa_check import precision_metrics
os.environ["OTPU_BUILD_JOBS"] = "1"


def baseline_rvq(folder):
    folder = Path(folder).resolve()
    receipt_path = folder / "mimi_check.json"
    q = json.loads(receipt_path.read_text())
    if (q.get("status") != "completed" or q.get("reset_bit_exact") is not True
            or q.get("source_lookup_sums_bit_exact") is not True
            or q.get("execution_target", {}).get("variant") != "p4rl8a64c150"
            or q.get("execution_target", {}).get("arch_hash") != "dd6b92a6"):
        raise ValueError("Completed native-compared exact-profile RVQ baseline required")
    paths = [receipt_path]
    for name, metadata in q["arrays"].items():
        path = folder / (name + ".npy")
        if sha256(path) != metadata["sha256"]:
            raise ValueError("Saved RVQ baseline changed: " + name)
        paths.append(path)
    return q, {str(p): sha256(p) for p in paths}


def run(model_dir, codec_path, token_dir, conditioning_dir, stack_dir, rvq_dir,
        output_dir, *, frames=2, target_name=COMMON_TARGET):
    import torch
    torch.set_num_threads(1)
    if target_name != COMMON_TARGET or type(frames) is not int or not 2 <= frames <= 4:
        raise ValueError("Require exact shared target and two to four captured temporal frames")
    output_dir, codec_path = Path(output_dir).resolve(), Path(codec_path).resolve()
    rvq_dir = Path(rvq_dir).resolve()
    if output_dir == rvq_dir or output_dir in rvq_dir.parents or rvq_dir in output_dir.parents:
        raise ValueError("Output must be isolated from the completed RVQ baseline")
    baseline, rvq_hashes = baseline_rvq(rvq_dir)
    guard = depth_preflight(model_dir, token_dir, conditioning_dir, stack_dir, output_dir,
                            frame_index=0, target_name=target_name)
    common = json.loads((DATA / "shared_gemma150_contract.json").read_text())
    contract_path = DATA / "mimi_decoder_contract.json"
    contract = json.loads(contract_path.read_text())
    rvq_contract = json.loads((DATA / "mimi_rvq_contract.json").read_text())
    if sha256(guard["target"].path) != common["target_source_sha256"]:
        raise ValueError("Shared target source changed")
    if guard["target"].to_dict()["arch_hash"] != common["expected_arch_hash"]:
        raise ValueError("Shared architecture differs")
    for name, digest in common["pinned_runtime_source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Pinned shared runtime changed: " + name)
    before_codec = codec_header(codec_path, rvq_contract)
    spec, rvq_spec = DepthChainSpec(), MimiRVQSpec()
    decoder_spec = MimiDecoderSpec(**contract["device_spec"])
    cfg = target.config(guard["target"])
    image = DepthChainImage(spec, cfg, mimi_spec=rvq_spec, decoder_spec=decoder_spec)
    budget = image.budget()  # fit and argument-free compilation precede payloads
    source_names = ["tools/personaplex_mimi_decoder_check.py", "tools/personaplex_mimi_decoder_native.py",
                    "opentpu/llm/personaplex_mimi_decoder.py", "opentpu/llm/personaplex_depth_chain.py",
                    "opentpu/llm/personaplex_mimi.py", "tools/data/personaplex/mimi_decoder_contract.json"]
    sources = {name: sha256(ROOT / name) for name in source_names}
    hidden = np.load(guard["paths"]["hidden"], allow_pickle=False)[:frames]
    text_ids = np.load(guard["paths"]["tokens"], allow_pickle=False)[:frames]
    receipt = {"status": "running", "stage": "materializing_native", "started_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "isa", "hardware_used": False, "hardware_performance_measured": False,
        "full_fpga_model_inference": False, "speech_generated": False, "quality_accepted": False,
        "scope": "Captured temporal frames ->resident generated depth IDs ->RVQ ->causal2xupsample ->8layer two-token decoder; full reset replay",
        "reference_scope": "Original cached native codec uses the same generated IDs; accumulated error includes RVQ and decoder, not native LM token-distribution quality",
        "frames": frames, "spec": contract["device_spec"], "source_files": sources,
        "native_source_sha256": contract["installed_source_sha256"], "pinned_runtime_source_sha256": common["pinned_runtime_source_sha256"],
        "runtime_revision": common["runtime_revision"], "execution_target": guard["target"].to_dict(),
        "image_context": common["image"], "baseline_sha256": {**guard["baseline_hashes"], **rvq_hashes},
        "checkpoint_revision": baseline["checkpoint_revision"], "codec_file": before_codec,
        "checkpoint_payload_hashes_verified": False, "budget": budget, "arrays": {},
        "package_versions": dependency_versions(), "frame_progress": [],
        "reset_policy": "Reset offsets/partial; retain masked old KV as original native source. Compare final whole cache after identical replay footprint.",
        "precision": "FP32 upsample/LayerNorm/RoPE/layerscale; INT8 matrices/activations/KV; device erf-polynomial GELU approximation"}
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "mimi_decoder_check.json"
    def save():
        temp = path.with_suffix(".json.tmp"); temp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n"); temp.replace(path)
    def interrupt(signum, frame):
        raise InterruptedError("Mimi decoder check interrupted; inspect receipt before retry")
    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    engine, started = None, time.perf_counter()
    save()
    try:
        quantizer, tables, rvq_matrices, native_root = native_quantizer(str(codec_path), rvq_contract)
        up, tf, kernel, decoder_layers, rope = native_modules(codec_path, decoder_spec, contract)
        receipt["payload_hashes"] = {
            "upsample": payload_sha256((("kernel", kernel),)),
            "decoder_layers": payload_sha256((f"layer{i}.{name}", value)
                for i, layer in enumerate(decoder_layers) for name, value in layer.items()),
            "rope": payload_sha256((("cos", rope[0]), ("sin", rope[1])))}
        weights = MLXWeights(model_dir)
        projections, embeddings, heads = weights.depth_input_projections, weights.depth_input_embeddings, weights.audio_heads
        depth_layers = []
        for index in range(spec.layers):
            receipt.update(stage="materializing_depth", source_layer=index); save()
            lazy = weights.depth_layer_weights(index)
            depth_layers.append({"norm1": lazy["norm1"], "norm2": lazy["norm2"],
                                 "steps": [{key: bank[key] for key in bank} for bank in lazy["steps"]]})
        receipt["payload_hashes"]["depth_layers"] = payload_sha256(layer_parts(depth_layers))
        receipt.update(stage="building_resident_image"); save()
        engine = DepthChainEngine(projections, embeddings, depth_layers, heads, spec=spec, cfg=cfg,
            mimi_tables=tables, mimi_projections=rvq_matrices, mimi_spec=rvq_spec,
            decoder_weights=decoder_layers, upsample_weight=kernel, decoder_spec=decoder_spec, decoder_rope=rope)
        receipt["budget"] = engine.image.budget()
        isa = {name: [] for name in ("ids", "rvq", "upsampled", "output", "layers")}
        for frame in range(frames):
            receipt.update(stage="resident_stream", frame=frame); save()
            original_read, original_write = engine.backend.read, engine.backend.write
            transfers = {"reads": 0, "writes": 0, "written_bytes": 0}
            def read(*args, **kwargs):
                transfers["reads"] += 1; return original_read(*args, **kwargs)
            def write(s, at, value):
                transfers["writes"] += 1; transfers["written_bytes"] += np.asarray(value).nbytes
                return original_write(s, at, value)
            engine.backend.read, engine.backend.write = read, write
            try:
                engine.run_frame(hidden[frame], int(text_ids[frame]))
                engine.run_mimi(); engine.run_decoder()
                if transfers != {"reads": 0, "writes": 2, "written_bytes": 16392}:
                    raise AssertionError("Unexpected resident interstage host transfers")
            finally:
                engine.backend.read, engine.backend.write = original_read, original_write
            isa["ids"].append(engine.read_ids()); isa["rvq"].append(engine.read_mimi())
            isa["upsampled"].append(engine.read_upsampled()); isa["output"].append(engine.read_decoder())
            isa["layers"].append(np.stack([engine.read_decoder_layer(i) for i in range(decoder_spec.layers)]))
            receipt["frame_progress"].append({"frame": frame, "transfers": transfers,
                                               "resident_decoder_frame": engine.decoder_pos}); save()
        isa = {name: np.stack(value) for name, value in isa.items()}
        exact_bytes(isa["ids"][0], np.load(rvq_dir / "isa_ids.npy", allow_pickle=False), "prior RVQ IDs")
        exact_bytes(isa["rvq"][0], np.load(rvq_dir / "isa_output.npy", allow_pickle=False), "prior RVQ latent")
        decoder_caches = [engine.read_decoder_cache(i) for i in range(decoder_spec.layers)]
        depth_caches = [engine.read_cache(i) for i in range(spec.layers)]
        receipt.update(stage="native_stream_comparison"); save()
        original_latents = np.stack([native_pass(quantizer, ids)["output"] for ids in isa["ids"]])
        original = collect_native(up, tf, original_latents)
        forced = collect_native(up, tf, isa["rvq"])
        exact_bytes(isa["upsampled"], forced["upsampled"], "FP32 upsample on identical ISA latent")
        for layer in range(decoder_spec.layers):
            for frame in range(frames):
                if not np.array_equal(forced["positions"][layer][frame], native_ring_positions(2 * (frame + 1), decoder_spec.context)):
                    raise AssertionError("Native two-token ring position contract changed")
        receipt["native_cache_positions"] = [[p.tolist() for p in per_layer] for per_layer in forced["positions"]]
        receipt["upsample_identical_latent_bit_exact"] = True
        receipt["accumulated_error"] = {"rvq": precision_metrics(isa["rvq"], original_latents),
            **{name: precision_metrics(isa[name], original[name]) for name in ("upsampled", "output")}}
        receipt["decoder_error_on_identical_rvq_latents"] = precision_metrics(isa["output"], forced["output"])
        receipt["accumulated_per_layer"] = [precision_metrics(isa["layers"][:, i], original["layers"][:, i]) for i in range(decoder_spec.layers)]
        receipt["decoder_per_layer_on_identical_rvq_latents"] = [precision_metrics(isa["layers"][:, i], forced["layers"][:, i]) for i in range(decoder_spec.layers)]
        native_replay = collect_native(up, tf, original_latents)
        for name in ("upsampled", "output", "layers"):
            exact_bytes(native_replay[name], original[name], "native fresh-stream replay " + name)
        receipt.update(stage="reset_stream_replay"); save()
        engine.reset()
        for frame in range(frames):
            receipt.update(stage="reset_frame", frame=frame); save()
            engine.run_frame(hidden[frame], int(text_ids[frame])); engine.run_mimi(); engine.run_decoder()
            for name, actual in (("ids", engine.read_ids()), ("rvq", engine.read_mimi()),
                                 ("upsampled", engine.read_upsampled()), ("output", engine.read_decoder())):
                exact_bytes(actual, isa[name][frame], "ISA reset " + name)
            for layer in range(decoder_spec.layers):
                exact_bytes(engine.read_decoder_layer(layer), isa["layers"][frame, layer], "ISA reset decoder layer")
        for layer, cache in enumerate(decoder_caches):
            exact_bytes(engine.read_decoder_cache(layer), cache, "final decoder raw KV replay")
        for layer, cache in enumerate(depth_caches):
            exact_bytes(engine.read_cache(layer), cache, "final depth raw KV replay")
        receipt.update(reset_bit_exact=True, final_decoder_raw_kv_reset_bit_exact=True,
                       final_depth_raw_kv_reset_bit_exact=True, native_fresh_stream_replay_bit_exact=True)
        for name, value in isa.items(): _save_array(output_dir, receipt, "isa_" + name, value)
        _save_array(output_dir, receipt, "native_rvq", original_latents)
        for label, values in (("native_original", original), ("native_same_rvq", forced)):
            for name in ("upsampled", "output", "layers"): _save_array(output_dir, receipt, label + "_" + name, values[name])
        for i, cache in enumerate(decoder_caches): _save_array(output_dir, receipt, f"decoder_kv_layer{i}", cache)
        for name, digest in receipt["baseline_sha256"].items():
            if sha256(name) != digest: raise RuntimeError("Completed baseline changed during check")
        if codec_header(codec_path, rvq_contract) != before_codec: raise RuntimeError("Codec metadata changed")
        for name, digest in sources.items():
            if sha256(ROOT / name) != digest: raise RuntimeError("Checker/decoder source changed")
        for name, digest in contract["installed_source_sha256"].items():
            if sha256(native_root / name) != digest: raise RuntimeError("Native source changed")
        for name, digest in common["pinned_runtime_source_sha256"].items():
            if sha256(ROOT / name) != digest: raise RuntimeError("Pinned runtime changed")
        receipt.update(status="completed", stage="completed")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()); raise
    finally:
        try:
            if engine is not None: engine.close()
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), cpu_wall_seconds=time.perf_counter() - started); save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "codec-path", "token-dir", "conditioning-dir", "stack-dir", "rvq-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--frames", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--target", default=COMMON_TARGET)
    args = parser.parse_args()
    result = run(**{k: v for k, v in vars(args).items() if k != "target"}, target_name=args.target)
    print(json.dumps({k: result[k] for k in ("status", "frames", "accumulated_error", "decoder_error_on_identical_rvq_latents", "reset_bit_exact", "cpu_wall_seconds")}, indent=2))


if __name__ == "__main__": main()
