#!/usr/bin/env python3
"""Saved-ID output codec precision check on the pinned CPU/ISA F2 profile.

Only resident RVQ, decoder, and SEANet programs execute. The original CPU
codec is an independent numerical reference; this tool never accesses a board.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.update(OTPU_BUILD_JOBS="1", NO_TORCH_COMPILE="1", CUDA_VISIBLE_DEVICES="",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")

import argparse
from datetime import datetime, timezone
import gc
import inspect
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opentpu import target
from opentpu.llm import qwen3 as G
from opentpu.llm.personaplex_depth_chain import DepthChainImage, DepthChainSpec
from opentpu.llm.personaplex_mimi import MimiRVQSpec
from opentpu.llm.personaplex_mimi_decoder import MimiDecoderSpec, native_ring_positions
from opentpu.llm.personaplex_mimi_pcm import MimiPCMSpec
from tools.personaplex_depth_chain_check import exact_bytes, payload_sha256, sha256, _save_array
from tools.personaplex_isa_check import precision_metrics
from tools.personaplex_mimi_check import codec_header, native_pass, native_quantizer
from tools.personaplex_mimi_decoder_native import native_modules, collect_native
from tools.personaplex_mimi_pcm_check import isolated_output, pcm_stats
from tools.personaplex_mimi_pcm_native import native_pcm_module, collect_native_pcm
from tools.personaplex_pcm_metrics import waveform_metrics
from tools.personaplex_pcm_precision_check import (baseline_pcm, numerical_acceptance,
                                                  CODEC_LIMITS, SEANET_LIMITS, COMMON_TARGET, DATA)


def _metrics_by_frame(actual, reference):
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape or a.ndim < 1 or a.shape[0] != 2:
        raise ValueError("Expected equal two-frame diagnostic arrays")
    return {"aggregate": precision_metrics(a, b),
            "per_frame": [precision_metrics(x, y) for x, y in zip(a, b)]}


def _decoder_layer_metrics(actual, reference):
    """Native collect_native returns [frame,layer,token,hidden], not a list."""
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape or a.ndim != 4 or a.shape[0] != 2:
        raise ValueError("Expected matching frame-first decoder layers")
    return [_metrics_by_frame(a[:, layer], b[:, layer]) for layer in range(a.shape[1])]


def _fp32(backend, address, shape):
    size = int(np.prod(shape))
    raw = np.asarray(backend.read(0, address, 4 * size), np.uint8)
    value = raw.view(np.float32).copy().reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError("Resident output is nonfinite")
    return value


def _capture(image, backend):
    mimi, decoder, pcm = image.mimi, image.decoder, image.pcm
    return {
        "rvq": {name: _fp32(backend, at, (mimi.spec.dim if name in ("semantic", "acoustic")
                                                   else mimi.spec.latent,))
                for name, at in mimi.io.items()},
        "upsampled": _fp32(backend, decoder.io["upsampled"], (2, decoder.spec.hidden)),
        "output": _fp32(backend, decoder.io["output"], (2, decoder.spec.hidden)),
        "decoder_layers": [_fp32(backend, at, (2, decoder.spec.hidden))
                           for at in decoder.layer_outputs],
        "pcm_layers": [_fp32(backend, at, (rows, cols))
                       for at, rows, cols in pcm.layer_outputs],
    }


def _state(image, backend):
    return {
        "decoder": [np.asarray(backend.read(0, at, size), np.uint8).copy()
                    for at, size in image.decoder.cache_regions],
        "decoder_control": [np.asarray(backend.read(0, at, size), np.uint8).copy()
                            for at, size in ((image.decoder.io["state"], 4),
                                             (image.decoder.io["overlap"], 8 * image.decoder.spec.hidden))],
        "pcm": [np.asarray(backend.read(0, at, size), np.uint8).copy()
                for at, size in image.pcm.reset_regions],
    }


def _reset(image, backend):
    # Match DepthChainEngine's output-codec reset. KV bytes are retained and
    # masked by the decoder position, just as they are in the native stream.
    decoder = image.decoder
    backend.write(0, decoder.io["state"], np.zeros(1, np.float32))
    backend.write(0, decoder.io["overlap"], np.full((2, decoder.spec.hidden), -0.0, np.float32))
    image.pcm.reset(backend)


def _replay(image, backend, ids, receipt, save, label):
    """Run component programs without CPU activation or weight transfers."""
    frames = []
    for frame, codes in enumerate(ids):
        # The immutable fixture is the only frame input. The complete depth
        # pass would have left this FP32 counter and these FP32 IDs resident.
        backend.write(0, image.io["audio_ids"], codes.astype(np.float32))
        backend.write(0, image.io["counter"], np.array([image.spec.steps], np.float32))
        receipt.update(stage=label, frame=frame, component="fixture_inputs"); save()
        read, write = backend.read, backend.write
        transfers = {"reads": 0, "writes": 0}
        def guarded_read(*args, **kwargs):
            transfers["reads"] += 1
            return read(*args, **kwargs)
        def guarded_write(*args, **kwargs):
            transfers["writes"] += 1
            return write(*args, **kwargs)
        backend.read, backend.write = guarded_read, guarded_write
        try:
            for component in ("mimi", "decoder", "pcm"):
                programs = getattr(image, component).compile()
                for index, program in enumerate(programs):
                    if program.run_args:
                        raise AssertionError("Output codec program requires host arguments")
                    backend.run([program])
                    receipt.update(stage=label, frame=frame, component=component,
                                   program=index + 1, programs=len(programs))
                    save()
        finally:
            backend.read, backend.write = read, write
        if transfers != {"reads": 0, "writes": 0}:
            raise AssertionError("Output codec performed an intercomponent host transfer")
        frames.append(_capture(image, backend))
        receipt["frame_progress"].append({"pass": label, "frame": frame,
                                          "intercomponent_transfers": transfers,
                                          "pcm_samples": image.pcm.spec.samples})
        save()
    return frames, _state(image, backend)


def _stack(frames):
    return {
        "rvq": {name: np.stack([frame["rvq"][name] for frame in frames])
                for name in frames[0]["rvq"]},
        "upsampled": np.stack([frame["upsampled"] for frame in frames]),
        "output": np.stack([frame["output"] for frame in frames]),
        "decoder_layers": np.stack([np.stack(frame["decoder_layers"]) for frame in frames]),
        "pcm_layers": [np.stack([frame["pcm_layers"][index] for frame in frames])
                       for index in range(15)],
        "pcm": np.stack([frame["pcm_layers"][-1][:, 0] for frame in frames]),
    }


def run(codec_path, baseline_dir, output_dir, *, decoder_precision="int8",
        target_name=COMMON_TARGET):
    if decoder_precision not in ("int8", "fp32") or target_name != COMMON_TARGET:
        raise ValueError("Require int8/fp32 decoder on exact shared target")
    codec_path, baseline_dir = Path(codec_path).resolve(), Path(baseline_dir).resolve()
    output_dir = isolated_output(output_dir, ROOT, codec_path, baseline_dir)
    baseline, baseline_hashes = baseline_pcm(baseline_dir)
    common = json.loads((DATA / "shared_gemma150_contract.json").read_text())
    contracts = {name: json.loads((DATA / f"mimi_{name}_contract.json").read_text())
                 for name in ("rvq", "decoder", "pcm")}
    selected = target.load(target_name)
    if (common["target"] != target_name or selected.to_dict()["arch_hash"] != common["expected_arch_hash"]
            or sha256(selected.path) != common["target_source_sha256"]
            or baseline["runtime_revision"] != common["runtime_revision"]
            or baseline["image_context"] != common["image"]):
        raise ValueError("Pinned target, runtime, or image context changed")
    for name, digest in common["pinned_runtime_source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Pinned runtime source changed: " + name)
    codec_before = codec_header(codec_path, contracts["rvq"])
    ids = np.load(baseline_dir / "isa_ids.npy", allow_pickle=False)
    if (ids.shape != (2, 16) or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids >= 2048)):
        raise ValueError("Preserved two-frame generated audio IDs differ")
    spec = DepthChainSpec()
    rvq_spec = MimiRVQSpec(**contracts["rvq"]["device_spec"], projection_precision="fp32")
    decoder_spec = MimiDecoderSpec(**contracts["decoder"]["device_spec"], precision=decoder_precision)
    pcm_spec = MimiPCMSpec(**contracts["pcm"]["device_spec"], final_precision="fp32",
                          residual_precision="fp32", initial_precision="fp32",
                          transpose_precision="fp32")
    cfg = target.config(selected)
    image = DepthChainImage(spec, cfg, mimi_spec=rvq_spec, decoder_spec=decoder_spec,
                            pcm_spec=pcm_spec)
    budget = image.budget()  # Full depth-chain fit and compile before payload reads.
    names = ["tools/personaplex_output_codec_check.py", "tools/personaplex_pcm_precision_check.py",
             "tools/personaplex_pcm_metrics.py", "tools/personaplex_mimi_check.py",
             "tools/personaplex_mimi_decoder_native.py", "tools/personaplex_mimi_pcm_native.py",
             "opentpu/llm/personaplex_mimi.py", "opentpu/llm/personaplex_mimi_decoder.py",
             "opentpu/llm/personaplex_mimi_pcm.py", "opentpu/llm/personaplex_depth_chain.py",
             *[f"tools/data/personaplex/mimi_{name}_contract.json" for name in contracts],
             "tools/data/personaplex/shared_gemma150_contract.json"]
    sources = {name: sha256(ROOT / name) for name in names}
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "output_codec_check.json"
    result = dict(status="running", stage="native_materialization", started_utc=datetime.now(timezone.utc).isoformat(),
        backend="isa", hardware_used=False, hardware_performance_measured=False,
        quality_accepted=False, full_speech_enabled=False, full_model_inference=False,
        checkpoint_payload_hashes_verified=False, frames=2, decoder_precision=decoder_precision,
        rvq_projection_precision="fp32", pcm_precision={"final": "fp32", "residual": "fp32",
            "initial": "fp32", "transpose": "fp32"},
        scope="Saved generated IDs -> resident RVQ -> decoder -> raw PCM; no LM/depth execution",
        pcm_semantics="raw mono FP32, 1920 samples/frame; no clipping/scaling/normalization/int16 conversion",
        performance_scope="CPU/ISA wall time and resource budget only; no FPGA latency or realtime claim",
        execution_target=selected.to_dict(), runtime_revision=common["runtime_revision"],
        image_context=common["image"], checkpoint_revision=baseline["checkpoint_revision"],
        baseline_sha256=baseline_hashes, source_files=sources, codec_file=codec_before,
        native_source_sha256={name: contract["installed_source_sha256"] for name, contract in contracts.items()},
        budget=budget, arrays={}, frame_progress=[])
    def save():
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temp.replace(path)
    started = time.perf_counter()
    backend = None
    save()
    try:
        import torch
        torch.set_num_threads(1)
        quantizer, tables, projections, native_root = native_quantizer(str(codec_path), contracts["rvq"])
        up, tf, kernel, layers, rope = native_modules(codec_path, decoder_spec, contracts["decoder"])
        seanet, pcm_weights = native_pcm_module(codec_path, contracts["pcm"])
        result["payload_hashes"] = {
            "mimi_tables": payload_sha256((str(i), value) for i, value in enumerate(tables)),
            "mimi_projections": payload_sha256((str(i), value) for i, value in enumerate(projections)),
            "upsample": payload_sha256((("kernel", kernel),)),
            "decoder_layers": payload_sha256((f"layer{i}.{name}", value)
                for i, layer in enumerate(layers) for name, value in layer.items()),
            "rope": payload_sha256((("cos", rope[0]), ("sin", rope[1]))),
            "pcm_weights": payload_sha256((f"model{row['model_index']}.{row['subpath']}.{name}", row[name])
                for row in pcm_weights for name in ("weight", "bias")),
        }
        native_latent = np.stack([native_pass(quantizer, codes)["output"] for codes in ids])
        native_original = collect_native(up, tf, native_latent)
        native_accumulated = collect_native_pcm(seanet, native_original["output"])
        exact_bytes(native_accumulated["pcm"],
                    np.load(baseline_dir / "native_accumulated_pcm.npy", allow_pickle=False),
                    "preserved native accumulated PCM")
        _save_array(output_dir, result, "native_accumulated_pcm", native_accumulated["pcm"])
        save()
        native_again = collect_native(up, tf, native_latent)
        for name in ("upsampled", "output", "layers"):
            exact_bytes(native_again[name], native_original[name], "native decoder replay " + name)
        native_pcm_again = collect_native_pcm(seanet, native_original["output"])
        exact_bytes(native_pcm_again["pcm"], native_accumulated["pcm"], "native PCM replay")
        for index, (left, right) in enumerate(zip(native_pcm_again["layers"],
                                                  native_accumulated["layers"])):
            exact_bytes(left, right, f"native SEANet stage {index} replay")
        result.update(stage="device_upload", native_replay_bit_exact=True); save()
        backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
        image.mimi.upload(backend, tables, projections)
        image.decoder.upload(backend, kernel, layers, rope)
        image.pcm.upload(backend, pcm_weights)
        _reset(image, backend)
        result["budget"] = image.budget()
        save()
        first_frames, first_state = _replay(image, backend, ids, result, save, "resident_stream")
        first = _stack(first_frames)
        # Preserve completed device evidence before reset or diagnostic
        # comparisons. A reporting exception must not lose an expensive replay.
        for name, value in first["rvq"].items():
            _save_array(output_dir, result, "isa_rvq_" + name, value)
        for name in ("upsampled", "output", "decoder_layers", "pcm"):
            _save_array(output_dir, result, "isa_" + name, first[name])
        for index, value in enumerate(first["pcm_layers"]):
            _save_array(output_dir, result, f"isa_pcm_stage{index}", value)
        for name, values in first_state.items():
            for index, value in enumerate(values):
                _save_array(output_dir, result, f"{name}_state{index}", value)
        save()
        _reset(image, backend)
        replay_frames, replay_state = _replay(image, backend, ids, result, save, "reset_stream")
        replay = _stack(replay_frames)
        for name in ("upsampled", "output", "decoder_layers", "pcm"):
            exact_bytes(replay[name], first[name], "output codec reset " + name)
        for name in first["rvq"]:
            exact_bytes(replay["rvq"][name], first["rvq"][name], "RVQ reset " + name)
        for index in range(15):
            exact_bytes(replay["pcm_layers"][index], first["pcm_layers"][index],
                        f"SEANet stage {index} reset")
        for name in first_state:
            if len(first_state[name]) != len(replay_state[name]):
                raise AssertionError("Output codec final state region count changed")
            for index, (left, right) in enumerate(zip(first_state[name], replay_state[name])):
                exact_bytes(right, left, f"final {name} raw state {index}")
        result.update(reset_bit_exact=True, final_decoder_raw_kv_reset_bit_exact=True,
                      final_decoder_control_reset_bit_exact=True,
                      final_pcm_state_reset_bit_exact=True, stage="native_isolation"); save()
        same_rvq = collect_native(up, tf, first["rvq"]["output"])
        same_decoder_pcm = collect_native_pcm(seanet, first["output"])
        for index in range(decoder_spec.layers):
            for frame in range(2):
                if not np.array_equal(same_rvq["positions"][index][frame],
                                      native_ring_positions(2 * (frame + 1), decoder_spec.context)):
                    raise AssertionError("Native ring position contract changed")
        exact_bytes(first["upsampled"], same_rvq["upsampled"], "identical-latent upsample")
        result["rvq_error"] = _metrics_by_frame(first["rvq"]["output"], native_latent)
        native_rvq_frames = [native_pass(quantizer, codes) for codes in ids]
        for name in ("semantic", "acoustic"):
            exact_bytes(first["rvq"][name], np.stack([frame[name] for frame in native_rvq_frames]),
                        "RVQ native centroid sums " + name)
        result["rvq_lookup_sums_bit_exact"] = True
        result["rvq_boundary_error"] = {name: _metrics_by_frame(first["rvq"][name],
            np.stack([frame[name] for frame in native_rvq_frames]))
            for name in first["rvq"]}
        result["decoder_error_on_identical_rvq"] = _metrics_by_frame(first["output"], same_rvq["output"])
        result["decoder_accumulated_error"] = _metrics_by_frame(first["output"], native_original["output"])
        result["upsampled_error"] = _metrics_by_frame(first["upsampled"], native_original["upsampled"])
        result["decoder_layer_error_on_identical_rvq"] = _decoder_layer_metrics(
            first["decoder_layers"], same_rvq["layers"])
        result["pcm_stage_error_on_identical_decoder"] = [
            _metrics_by_frame(first["pcm_layers"][i], same_decoder_pcm["layers"][i])
            for i in range(15)]
        result["seanet_on_identical_decoder"] = waveform_metrics(first["pcm"], same_decoder_pcm["pcm"])
        result["accumulated_output_codec"] = waveform_metrics(first["pcm"], native_accumulated["pcm"])
        result["seanet_acceptance"] = numerical_acceptance(result["seanet_on_identical_decoder"], SEANET_LIMITS)
        result["accumulated_codec_acceptance"] = numerical_acceptance(result["accumulated_output_codec"], CODEC_LIMITS)
        result["raw_pcm"] = pcm_stats(first["pcm"])
        for name, digest in baseline_hashes.items():
            if sha256(name) != digest: raise RuntimeError("Preserved baseline changed")
        for name, digest in sources.items():
            if sha256(ROOT / name) != digest: raise RuntimeError("Output codec source changed")
        for name, digest in common["pinned_runtime_source_sha256"].items():
            if sha256(ROOT / name) != digest: raise RuntimeError("Pinned runtime source changed")
        if sha256(selected.path) != common["target_source_sha256"]:
            raise RuntimeError("Target manifest changed")
        for name, digest in contracts["rvq"]["installed_source_sha256"].items():
            if sha256(native_root / name) != digest: raise RuntimeError("Native source changed")
        for component, module in (("decoder", up), ("pcm", seanet)):
            native_component_root = Path(inspect.getfile(type(module))).resolve().parents[1]
            for name, digest in contracts[component]["installed_source_sha256"].items():
                if sha256(native_component_root / name) != digest:
                    raise RuntimeError("Native " + component + " source changed: " + name)
        if codec_header(codec_path, contracts["rvq"]) != codec_before:
            raise RuntimeError("Codec changed during check")
        result.update(status="completed", stage="completed")
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            close = getattr(backend, "close", None)
            if close is not None: close()
            del backend
            gc.collect()
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      cpu_wall_seconds=time.perf_counter() - started,
                      peak_host_rss_kib=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                         if sys.platform.startswith("linux") else None))
        save()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("codec-path", "baseline-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--decoder-precision", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--target", default=COMMON_TARGET)
    args = parser.parse_args()
    receipt = run(args.codec_path, args.baseline_dir, args.output_dir,
                  decoder_precision=args.decoder_precision, target_name=args.target)
    print(json.dumps({key: receipt[key] for key in ("status", "frames", "decoder_precision",
                    "seanet_acceptance", "accumulated_codec_acceptance", "cpu_wall_seconds")}, indent=2))
