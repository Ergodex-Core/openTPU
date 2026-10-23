#!/usr/bin/env python3
"""Bounded original-checkpoint SEANet PCM check on the exact shared CPU ISA.

The resident chain generates IDs, RVQ latents, decoder tokens and raw PCM. The
installed native codec is an offline diagnostic reference only. No board,
serving fallback, clipping, int16 conversion or speech-quality claim is made.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.update(OTPU_BUILD_JOBS="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", NO_TORCH_COMPILE="1")

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
from opentpu.llm.personaplex_mimi_pcm import MimiPCMSpec
from opentpu.llm.personaplex_weights import MLXWeights
from tools.personaplex_depth_chain_check import (preflight as depth_preflight, sha256,
    exact_bytes, payload_sha256, layer_parts, _save_array)
from tools.personaplex_mimi_check import (native_quantizer, native_pass, codec_header,
                                         dependency_versions, COMMON_TARGET, DATA)
from tools.personaplex_mimi_decoder_check import baseline_rvq
from tools.personaplex_mimi_decoder_native import native_modules, collect_native
from tools.personaplex_mimi_pcm_native import native_pcm_module, collect_native_pcm
from tools.personaplex_isa_check import PINNED_REVISION, precision_metrics
os.environ["OTPU_BUILD_JOBS"] = "1"


def baseline_decoder(folder):
    """Require the completed two-frame upstream check and hash every saved array."""
    folder = Path(folder).resolve()
    receipt_path = folder / "mimi_decoder_check.json"
    receipt = json.loads(receipt_path.read_text())
    target_info = receipt.get("execution_target", {})
    if (receipt.get("status") != "completed" or receipt.get("backend") != "isa"
            or receipt.get("hardware_used") is not False
            or receipt.get("reset_bit_exact") is not True
            or receipt.get("final_decoder_raw_kv_reset_bit_exact") is not True
            or receipt.get("final_depth_raw_kv_reset_bit_exact") is not True
            or receipt.get("upsample_identical_latent_bit_exact") is not True
            or receipt.get("checkpoint_revision") != PINNED_REVISION
            or target_info.get("variant") != "p4rl8a64c150"
            or target_info.get("arch_hash") != "dd6b92a6"
            or type(receipt.get("frames")) is not int or not 2 <= receipt["frames"] <= 4):
        raise ValueError("Completed exact-profile native-compared Mimi decoder baseline required")
    arrays = receipt.get("arrays")
    required = {"isa_ids", "isa_rvq", "isa_upsampled", "isa_output", "isa_layers"}
    if not isinstance(arrays, dict) or not required <= arrays.keys():
        raise ValueError("Completed decoder baseline lacks required saved arrays")
    paths = [receipt_path]
    for name, metadata in arrays.items():
        path = folder / (name + ".npy")
        if (sha256(path) != metadata["sha256"]
                or np.load(path, allow_pickle=False).shape != tuple(metadata["shape"])
                or str(np.load(path, mmap_mode="r", allow_pickle=False).dtype) != metadata["dtype"]):
            raise ValueError("Saved decoder baseline changed: " + name)
        paths.append(path)
    for name, digest in receipt.get("baseline_sha256", {}).items():
        if sha256(name) != digest:
            raise ValueError("Decoder ancestry baseline changed: " + name)
        paths.append(Path(name))
    return receipt, {str(path): sha256(path) for path in paths}


def isolated_output(output_dir, *inputs):
    """Keep the new receipt away from all immutable baseline/source trees."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("PCM output directory must be new")
    for input_path in inputs:
        protected = Path(input_path).resolve()
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("PCM output must be isolated from every baseline and source")
    return output


def pcm_stats(value):
    """Describe raw FP32 samples, including excursions outside [-1,1]."""
    array = np.asarray(value)
    if (array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 1920
            or array.shape[0] == 0 or not np.isfinite(array).all()):
        raise ValueError("PCM statistics require finite raw FP32 [frames,1920]")
    wide = array.astype(np.float64)
    return {"min": float(wide.min()), "max": float(wide.max()),
            "mean": float(wide.mean()), "rms": float(np.sqrt(np.mean(wide * wide))),
            "abs_gt_1_samples": int(np.count_nonzero(np.abs(wide) > 1)),
            "samples": int(array.size),
            "per_frame": [{"min": float(frame.min()), "max": float(frame.max()),
                           "mean": float(frame.mean()),
                           "rms": float(np.sqrt(np.mean(frame * frame))),
                           "abs_gt_1_samples": int(np.count_nonzero(np.abs(frame) > 1))}
                          for frame in wide]}


def _metrics_by_frame(actual, reference):
    if (np.asarray(actual).shape != np.asarray(reference).shape
            or np.asarray(actual).ndim < 1 or np.asarray(actual).shape[0] == 0):
        raise ValueError("Diagnostic frame shapes differ")
    return {"aggregate": precision_metrics(actual, reference),
            "per_frame": [precision_metrics(a, b) for a, b in zip(actual, reference)]}


def _load_prior(folder, receipt, name, frames):
    value = np.load(folder / (name + ".npy"), allow_pickle=False)
    if value.shape[0] < frames or str(value.dtype) != receipt["arrays"][name]["dtype"]:
        raise ValueError("Prior decoder array has insufficient frames: " + name)
    return np.ascontiguousarray(value[:frames])


def run(model_dir, codec_path, token_dir, conditioning_dir, stack_dir, rvq_dir,
        decoder_dir, output_dir, *, frames=2, target_name=COMMON_TARGET):
    if target_name != COMMON_TARGET or type(frames) is not int or not 2 <= frames <= 4:
        raise ValueError("Require exact shared target and two to four captured temporal frames")
    output_dir = isolated_output(output_dir, ROOT, model_dir, codec_path, token_dir,
                                 conditioning_dir, stack_dir, rvq_dir, decoder_dir)
    decoder_dir, rvq_dir, codec_path = (Path(p).resolve() for p in (decoder_dir, rvq_dir, codec_path))
    baseline, decoder_hashes = baseline_decoder(decoder_dir)
    if frames > baseline["frames"]:
        raise ValueError("Requested frames exceed completed decoder baseline")
    rvq_baseline, rvq_hashes = baseline_rvq(rvq_dir)
    if baseline["checkpoint_revision"] != rvq_baseline["checkpoint_revision"]:
        raise ValueError("Mimi baselines use different checkpoint revisions")
    guard = depth_preflight(model_dir, token_dir, conditioning_dir, stack_dir, output_dir,
                            frame_index=0, target_name=target_name)
    common = json.loads((DATA / "shared_gemma150_contract.json").read_text())
    rvq_contract = json.loads((DATA / "mimi_rvq_contract.json").read_text())
    decoder_contract = json.loads((DATA / "mimi_decoder_contract.json").read_text())
    pcm_contract = json.loads((DATA / "mimi_pcm_contract.json").read_text())
    if (common["image"] != "agfi-0001de1dd3df8e21e"
            or common["runtime_revision"] != "18fb978033799f0f21258f0e6e085c0360322d96"
            or common["expected_arch_hash"] != "dd6b92a6"
            or common["target"] != COMMON_TARGET
            or guard["target"].to_dict()["arch_hash"] != common["expected_arch_hash"]
            or sha256(guard["target"].path) != common["target_source_sha256"]
            or baseline.get("runtime_revision") != common["runtime_revision"]
            or baseline.get("image_context") != common["image"]):
        raise ValueError("Exact pinned shared-image/runtime target differs")
    for name, digest in common["pinned_runtime_source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Pinned shared runtime changed: " + name)
    before_codec = codec_header(codec_path, rvq_contract)
    spec, rvq_spec = DepthChainSpec(), MimiRVQSpec(**rvq_contract["device_spec"])
    decoder_spec = MimiDecoderSpec(**decoder_contract["device_spec"])
    pcm_spec = MimiPCMSpec(**pcm_contract["device_spec"])
    cfg = target.config(guard["target"])
    image = DepthChainImage(spec, cfg, mimi_spec=rvq_spec, decoder_spec=decoder_spec,
                            pcm_spec=pcm_spec)
    budget = image.budget()  # all programs and capacity checked before payload reads
    source_names = ["tools/personaplex_mimi_pcm_check.py", "tools/personaplex_mimi_pcm_native.py",
        "tools/personaplex_mimi_decoder_check.py", "tools/personaplex_mimi_decoder_native.py",
        "opentpu/llm/personaplex_mimi_pcm.py", "opentpu/llm/personaplex_mimi_decoder.py",
        "opentpu/llm/personaplex_mimi.py", "opentpu/llm/personaplex_depth_chain.py",
        "tools/data/personaplex/mimi_pcm_contract.json", "tools/data/personaplex/mimi_decoder_contract.json",
        "tools/data/personaplex/mimi_rvq_contract.json", "tools/data/personaplex/shared_gemma150_contract.json"]
    sources = {name: sha256(ROOT / name) for name in source_names}
    hidden = np.load(guard["paths"]["hidden"], allow_pickle=False)[:frames]
    text_ids = np.load(guard["paths"]["tokens"], allow_pickle=False)[:frames]
    receipt = {"status": "running", "stage": "materializing_native",
        "started_utc": datetime.now(timezone.utc).isoformat(), "backend": "isa",
        "hardware_used": False, "hardware_performance_measured": False,
        "pcm_generated": False, "speech_generated": False, "full_fpga_model_inference": False,
        "full_model_inference": False, "quality_accepted": False,
        "scope": "Captured temporal frames -> resident depth IDs -> RVQ -> causal decoder -> raw SEANet PCM",
        "reference_scope": "Original cached native RVQ, transformer, and SEANet on identical generated IDs; separate native SEANet on identical ISA decoder tokens",
        "frames": frames, "spec": pcm_contract["device_spec"], "source_files": sources,
        "native_source_sha256": pcm_contract["installed_source_sha256"],
        "pinned_runtime_source_sha256": common["pinned_runtime_source_sha256"],
        "runtime_revision": common["runtime_revision"], "execution_target": guard["target"].to_dict(),
        "image_context": common["image"], "baseline_sha256": {
            **guard["baseline_hashes"], **rvq_hashes, **decoder_hashes},
        "checkpoint_revision": baseline["checkpoint_revision"], "codec_file": before_codec,
        "checkpoint_payload_hashes_verified": False, "budget": budget, "arrays": {},
        "package_versions": dependency_versions(), "frame_progress": [],
        "pcm_format": "raw mono FP32, 24000Hz source, 1920 samples per 12.5Hz code frame; no clipping, scaling or int16 conversion",
        "performance_scope": "CPU wall time only; no device throughput or real-time claim",
        "native_diagnostic_mode": "Installed original codec in eager CPU mode (NO_TORCH_COMPILE=1); never serves inference",
        "precision": "FP32 native reference; resident INT8 convolution weights/activations and FP32 bias/history/overlap/ELU/residual"}
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "mimi_pcm_check.json"
    def save():
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temp.replace(path)
    def interrupt(signum, frame):
        raise InterruptedError("Mimi PCM check interrupted; inspect receipt before retry")
    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    engine, started = None, time.perf_counter()
    save()
    try:
        import torch
        torch.set_num_threads(1)
        quantizer, tables, rvq_matrices, native_root = native_quantizer(str(codec_path), rvq_contract)
        up, tf, kernel, decoder_layers, rope = native_modules(codec_path, decoder_spec, decoder_contract)
        seanet, pcm_weights = native_pcm_module(codec_path, pcm_contract)
        receipt["payload_hashes"] = {
            "mimi_tables": payload_sha256((str(i), value) for i, value in enumerate(tables)),
            "mimi_projections": payload_sha256((str(i), value) for i, value in enumerate(rvq_matrices)),
            "upsample": payload_sha256((("kernel", kernel),)),
            "decoder_layers": payload_sha256((f"layer{i}.{name}", value)
                for i, layer in enumerate(decoder_layers) for name, value in layer.items()),
            "rope": payload_sha256((("cos", rope[0]), ("sin", rope[1]))),
            "pcm_weights": payload_sha256((f"model{row['model_index']}.{row['subpath']}.{name}", row[name])
                for row in pcm_weights for name in ("weight", "bias"))}
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
            decoder_weights=decoder_layers, upsample_weight=kernel, decoder_spec=decoder_spec,
            decoder_rope=rope, pcm_weights=pcm_weights, pcm_spec=pcm_spec)
        receipt["budget"] = engine.image.budget()
        isa = {name: [] for name in ("ids", "rvq", "upsampled", "output", "layers", "pcm")}
        pcm_layers = [[] for _ in range(15)]
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
                engine.run_mimi(); engine.run_decoder(); engine.run_pcm()
                if transfers != {"reads": 0, "writes": 2, "written_bytes": 16392}:
                    raise AssertionError("Unexpected resident interstage host transfers")
            finally:
                engine.backend.read, engine.backend.write = original_read, original_write
            isa["ids"].append(engine.read_ids()); isa["rvq"].append(engine.read_mimi())
            isa["upsampled"].append(engine.read_upsampled()); isa["output"].append(engine.read_decoder())
            isa["layers"].append(np.stack([engine.read_decoder_layer(i) for i in range(decoder_spec.layers)]))
            isa["pcm"].append(engine.read_pcm())
            for index in range(15): pcm_layers[index].append(engine.read_pcm_layer(index))
            receipt["frame_progress"].append({"frame": frame, "transfers": transfers,
                "resident_decoder_frame": engine.decoder_pos, "raw_pcm_samples": pcm_spec.samples}); save()
        isa = {name: np.stack(value) for name, value in isa.items()}
        pcm_layers = [np.stack(layer) for layer in pcm_layers]
        for name in ("ids", "rvq", "upsampled", "output", "layers"):
            exact_bytes(isa[name], _load_prior(decoder_dir, baseline, "isa_" + name, frames),
                        "completed decoder baseline " + name)
        receipt["decoder_baseline_arrays_bit_exact"] = True
        decoder_caches = [engine.read_decoder_cache(i) for i in range(decoder_spec.layers)]
        depth_caches = [engine.read_cache(i) for i in range(spec.layers)]
        pcm_states = engine.read_pcm_state()
        receipt.update(stage="native_stream_comparison"); save()
        original_latents = np.stack([native_pass(quantizer, ids)["output"] for ids in isa["ids"]])
        original_decoder = collect_native(up, tf, original_latents)
        same_rvq_decoder = collect_native(up, tf, isa["rvq"])
        exact_bytes(isa["upsampled"], same_rvq_decoder["upsampled"],
                    "FP32 upsample on identical ISA latent")
        for layer in range(decoder_spec.layers):
            for frame in range(frames):
                if not np.array_equal(same_rvq_decoder["positions"][layer][frame],
                                      native_ring_positions(2 * (frame + 1), decoder_spec.context)):
                    raise AssertionError("Native two-token ring position contract changed")
        native_accumulated = collect_native_pcm(seanet, original_decoder["output"])
        native_same_input = collect_native_pcm(seanet, isa["output"])
        receipt["native_cache_positions"] = [[p.tolist() for p in per_layer]
                                             for per_layer in same_rvq_decoder["positions"]]
        receipt["upsample_identical_latent_bit_exact"] = True
        receipt["upstream_accumulated_error"] = {
            "rvq": _metrics_by_frame(isa["rvq"], original_latents),
            "decoder": _metrics_by_frame(isa["output"], original_decoder["output"])}
        receipt["pcm_accumulated_error"] = _metrics_by_frame(isa["pcm"], native_accumulated["pcm"])
        receipt["pcm_same_input_error"] = _metrics_by_frame(isa["pcm"], native_same_input["pcm"])
        receipt["pcm_stage_accumulated_error"] = [
            _metrics_by_frame(isa_layer, native_layer) for isa_layer, native_layer
            in zip(pcm_layers, native_accumulated["layers"])]
        receipt["pcm_stage_same_input_error"] = [
            _metrics_by_frame(isa_layer, native_layer) for isa_layer, native_layer
            in zip(pcm_layers, native_same_input["layers"])]
        receipt["raw_pcm_stats"] = {"isa": pcm_stats(isa["pcm"]),
            "native_accumulated": pcm_stats(native_accumulated["pcm"]),
            "native_same_input": pcm_stats(native_same_input["pcm"])}
        native_replay = collect_native_pcm(seanet, original_decoder["output"])
        exact_bytes(native_replay["pcm"], native_accumulated["pcm"], "native fresh-stream PCM replay")
        for index in range(15):
            exact_bytes(native_replay["layers"][index], native_accumulated["layers"][index],
                        f"native fresh-stream stage {index}")
        receipt.update(stage="reset_stream_replay"); save()
        engine.reset()
        for frame in range(frames):
            receipt.update(stage="reset_frame", frame=frame); save()
            engine.run_frame(hidden[frame], int(text_ids[frame]))
            engine.run_mimi(); engine.run_decoder(); engine.run_pcm()
            for name, actual in (("ids", engine.read_ids()), ("rvq", engine.read_mimi()),
                                 ("upsampled", engine.read_upsampled()),
                                 ("output", engine.read_decoder()), ("pcm", engine.read_pcm())):
                exact_bytes(actual, isa[name][frame], "ISA reset " + name)
            for layer in range(decoder_spec.layers):
                exact_bytes(engine.read_decoder_layer(layer), isa["layers"][frame, layer],
                            "ISA reset decoder layer")
            for index in range(15):
                exact_bytes(engine.read_pcm_layer(index), pcm_layers[index][frame],
                            f"ISA reset SEANet stage {index}")
        replay_pcm_states = engine.read_pcm_state()
        if len(replay_pcm_states) != len(pcm_states):
            raise AssertionError("Final PCM state region count differs")
        for index, state in enumerate(pcm_states):
            exact_bytes(replay_pcm_states[index], state, f"final PCM state {index}")
        for layer, cache in enumerate(decoder_caches):
            exact_bytes(engine.read_decoder_cache(layer), cache, "final decoder raw KV replay")
        for layer, cache in enumerate(depth_caches):
            exact_bytes(engine.read_cache(layer), cache, "final depth raw KV replay")
        receipt.update(reset_bit_exact=True, final_pcm_state_reset_bit_exact=True,
            final_decoder_raw_kv_reset_bit_exact=True, final_depth_raw_kv_reset_bit_exact=True,
            native_fresh_stream_replay_bit_exact=True)
        for name, value in isa.items(): _save_array(output_dir, receipt, "isa_" + name, value)
        for index, value in enumerate(pcm_layers):
            _save_array(output_dir, receipt, f"isa_pcm_stage{index}", value)
        _save_array(output_dir, receipt, "native_rvq", original_latents)
        for label, values in (("native_original", original_decoder), ("native_same_rvq", same_rvq_decoder)):
            for name in ("upsampled", "output", "layers"):
                _save_array(output_dir, receipt, label + "_" + name, values[name])
        for label, values in (("native_accumulated", native_accumulated),
                              ("native_same_input", native_same_input)):
            _save_array(output_dir, receipt, label + "_pcm", values["pcm"])
            for index, value in enumerate(values["layers"]):
                _save_array(output_dir, receipt, f"{label}_pcm_stage{index}", value)
        for index, state in enumerate(pcm_states):
            _save_array(output_dir, receipt, f"pcm_state{index}", state)
        for index, cache in enumerate(decoder_caches):
            _save_array(output_dir, receipt, f"decoder_kv_layer{index}", cache)
        for index, cache in enumerate(depth_caches):
            _save_array(output_dir, receipt, f"depth_kv_layer{index}", cache)
        for name, digest in receipt["baseline_sha256"].items():
            if sha256(name) != digest: raise RuntimeError("Completed baseline changed during check")
        if codec_header(codec_path, rvq_contract) != before_codec:
            raise RuntimeError("Codec metadata changed")
        for name, digest in sources.items():
            if sha256(ROOT / name) != digest: raise RuntimeError("Checker/PCM source changed")
        for name, digest in pcm_contract["installed_source_sha256"].items():
            if sha256(native_root / name) != digest: raise RuntimeError("Native codec source changed")
        for name, digest in common["pinned_runtime_source_sha256"].items():
            if sha256(ROOT / name) != digest: raise RuntimeError("Pinned runtime changed")
        receipt.update(status="completed", stage="completed", pcm_generated=True)
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}",
                       traceback=traceback.format_exc()); raise
    finally:
        try:
            if engine is not None: engine.close()
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                           cpu_wall_seconds=time.perf_counter() - started); save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "codec-path", "token-dir", "conditioning-dir", "stack-dir",
                 "rvq-dir", "decoder-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--frames", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--target", default=COMMON_TARGET)
    args = parser.parse_args()
    result = run(**{key: value for key, value in vars(args).items() if key != "target"},
                 target_name=args.target)
    print(json.dumps({key: result[key] for key in ("status", "frames", "pcm_accumulated_error",
        "pcm_same_input_error", "reset_bit_exact", "cpu_wall_seconds")}, indent=2))


if __name__ == "__main__": main()
