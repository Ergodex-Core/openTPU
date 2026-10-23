#!/usr/bin/env python3
"""Replay saved SEANet stages and compare optional ISA FP32 convolutions.

Existing checkpoint/cache only, EC2 CPU/ISA only. No LM rerun, board, serving,
clipping, scaling or perceptual-quality acceptance.
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
import hashlib
import json
from pathlib import Path
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
from opentpu.llm.personaplex_mimi_decoder import MimiDecoderSpec
from opentpu.llm.personaplex_mimi_pcm import MimiPCMSpec
from tools.personaplex_depth_chain_check import sha256, exact_bytes, _save_array
from tools.personaplex_isa_check import precision_metrics
from tools.personaplex_mimi_pcm_check import isolated_output, pcm_stats
from tools.personaplex_mimi_pcm_native import native_pcm_module, collect_native_pcm
from tools.personaplex_pcm_metrics import waveform_metrics

BASELINE_SHA256 = "eb43e9b555de027a95f7a811f4e3f7ddfb4767b421b44e4cd179f9f7892159b5"
COMMON_TARGET = "f2-hbm:p4rl8a64c150"
DATA = ROOT / "tools/data/personaplex"

# Engineering agreement on this saved two-frame fixture, never an audio-quality
# or full-speech claim. The accumulated codec gate includes unchanged upstream
# RVQ/transformer error and is deliberately separate from SEANet acceptance.
SEANET_LIMITS = dict(relative_l2=1e-4, error_rms_full_scale=1e-6,
                     peak_absolute_error_full_scale=1e-5)
CODEC_LIMITS = dict(relative_l2=1e-3, error_rms_full_scale=1e-5,
                   peak_absolute_error_full_scale=1e-4)


def numerical_acceptance(metrics, limits):
    entries = [metrics["aggregate"], *metrics["per_frame"]]
    return dict(limits=limits, passed=all(
        item[name] is not None and item[name] <= maximum
        for item in entries for name, maximum in limits.items()),
        scope="aggregate and each frame on the saved two-frame fixture only")


def baseline_pcm(folder):
    folder = Path(folder).resolve()
    path = folder / "mimi_pcm_check.json"
    if sha256(path) != BASELINE_SHA256:
        raise ValueError("Require the preserved18567fef two-frame PCM receipt")
    data = json.loads(path.read_text())
    if (data.get("status") != "completed" or data.get("hardware_used") is not False
            or data.get("reset_bit_exact") is not True or data.get("frames") != 2
            or data.get("execution_target", {}).get("variant") != "p4rl8a64c150"
            or data.get("execution_target", {}).get("arch_hash") != "dd6b92a6"):
        raise ValueError("Preserved exact-profile PCM baseline is incomplete")
    hashes = {str(path): BASELINE_SHA256}
    for name, metadata in data["arrays"].items():
        relative = Path(name + ".npy")
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("Baseline array path is not a local filename")
        p = folder / relative
        if sha256(p) != metadata["sha256"]:
            raise ValueError("Preserved PCM baseline changed: " + name)
        hashes[str(p)] = metadata["sha256"]
    for name, digest in data["baseline_sha256"].items():
        if sha256(name) != digest:
            raise ValueError("Preserved PCM ancestry changed: " + name)
        hashes[name] = digest
    return data, hashes


def forced_native_stages(decoder, source, stages):
    """Each original module receives identical saved ISA input and streaming history."""
    import torch
    result = [[] for _ in range(15)]
    with torch.no_grad(), decoder.streaming(1):
        for frame in range(len(source)):
            for index, module in enumerate(decoder.model):
                value = source[frame] if index == 0 else stages[index - 1][frame]
                x = torch.from_numpy(value.T.copy()[None, :, :])
                output = module(x).detach().numpy()[0].T.copy()
                if output.shape != stages[index][frame].shape or not np.isfinite(output).all():
                    raise ValueError("Native isolated stage shape/finiteness differs")
                result[index].append(output)
    return [np.stack(x) for x in result]


def device_replay(image, backend, inputs):
    outputs, layers, transfers = [], [[] for _ in range(15)], []
    programs = image.compile()
    for frame, value in enumerate(inputs):
        backend.write(0, image.input_at, value)
        read, write = backend.read, backend.write
        count = {"reads": 0, "writes": 0}
        def guarded_read(*args):
            count["reads"] += 1
            return read(*args)
        def guarded_write(*args):
            count["writes"] += 1
            return write(*args)
        backend.read, backend.write = guarded_read, guarded_write
        try:
            for program in programs:
                if program.run_args:
                    raise AssertionError("PCM precision program needs host arguments")
                backend.run([program])
        finally:
            backend.read, backend.write = read, write
        if count != {"reads": 0, "writes": 0}:
            raise AssertionError("Implicit SEANet interstage host transfer")
        for index, (at, rows, cols) in enumerate(image.layer_outputs):
            raw = np.asarray(read(0, at, 4 * rows * cols), np.uint8).view(np.float32).copy()
            if not np.isfinite(raw).all():
                raise AssertionError("Nonfinite device stage")
            layers[index].append(raw.reshape(rows, cols))
        outputs.append(layers[-1][-1][:, 0].copy())
        transfers.append(dict(frame=frame, fixture_input_write_bytes=value.nbytes,
                              interstage=count))
    state = [np.asarray(backend.read(0, at, size), np.uint8).copy()
             for at, size in image.reset_regions]
    return np.stack(outputs), [np.stack(x) for x in layers], state, transfers


def run(codec_path, baseline_dir, output_dir, *, target_name=COMMON_TARGET,
        all_convolutions=False):
    if target_name != COMMON_TARGET:
        raise ValueError("Require exact shared target")
    baseline_dir, codec_path = Path(baseline_dir).resolve(), Path(codec_path).resolve()
    output_dir = isolated_output(output_dir, ROOT, codec_path, baseline_dir)
    baseline, baseline_hashes = baseline_pcm(baseline_dir)
    common = json.loads((DATA / "shared_gemma150_contract.json").read_text())
    contract = json.loads((DATA / "mimi_pcm_contract.json").read_text())
    selected = target.load(target_name)
    if (selected.to_dict()["arch_hash"] != "dd6b92a6" or sha256(selected.path) != common["target_source_sha256"]
            or baseline["runtime_revision"] != common["runtime_revision"]
            or baseline["image_context"] != common["image"]):
        raise ValueError("Pinned shared target/runtime/image context changed")
    for name, digest in common["pinned_runtime_source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Pinned runtime source changed: " + name)
    names = ["opentpu/llm/personaplex_mimi_pcm.py", "opentpu/llm/personaplex_depth_chain.py",
             "tools/personaplex_pcm_precision_check.py", "tools/personaplex_pcm_metrics.py",
             "tools/personaplex_mimi_pcm_native.py", "tools/data/personaplex/mimi_pcm_contract.json"]
    sources = {name: sha256(ROOT / name) for name in names}
    before = (codec_path.stat().st_size, codec_path.stat().st_mtime_ns)
    cfg = target.config(selected)
    cases = ("int8", "fp32_all") if all_convolutions else ("int8", "fp32", "fp32_residual")
    layouts = {precision: DepthChainImage(DepthChainSpec(), cfg, mimi_spec=MimiRVQSpec(),
        decoder_spec=MimiDecoderSpec(), pcm_spec=MimiPCMSpec(
            final_precision="int8" if precision == "int8" else "fp32",
            residual_precision="fp32" if precision in ("fp32_residual", "fp32_all") else "int8",
            initial_precision="fp32" if precision == "fp32_all" else "int8",
            transpose_precision="fp32" if precision == "fp32_all" else "int8"))
        for precision in cases}
    budgets = {precision: image.budget() for precision, image in layouts.items()}
    source = np.load(baseline_dir / "isa_output.npy", allow_pickle=False)
    old_pcm = np.load(baseline_dir / "isa_pcm.npy", allow_pickle=False)
    old_stages = [np.load(baseline_dir / f"isa_pcm_stage{i}.npy", allow_pickle=False) for i in range(15)]
    accumulated = np.load(baseline_dir / "native_accumulated_pcm.npy", allow_pickle=False)
    same_input = np.load(baseline_dir / "native_same_input_pcm.npy", allow_pickle=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "pcm_precision_check.json"
    result = dict(status="running", stage="native_isolation", started_utc=datetime.now(timezone.utc).isoformat(),
        backend="isa", hardware_used=False, hardware_performance_measured=False, quality_accepted=False,
        full_model_inference=False, full_speech_enabled=False, frames=2, baseline_sha256=baseline_hashes,
        source_files=sources, runtime_revision=common["runtime_revision"], image_context=common["image"],
        execution_target=selected.to_dict(), budgets=budgets, arrays={},
        scope="Saved resident ISA decoder outputs -> SEANet component replay; unchanged LM is not rerun",
        precision_remedy="Optional FP32 convolutions using existing VPU RDOT; all precision choices explicit and INT8 by default",
        pcm_semantics="rawFP32, no clipping/scaling/normalization/int16conversion",
        performance_scope="CPU/ISA wall time only, not measuredFPGA latency or realtime",
        checkpoint_payload_hashes_verified=False, candidates={})
    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    started = time.perf_counter()
    save()
    try:
        import torch
        torch.set_num_threads(1)
        decoder, weights = native_pcm_module(codec_path, contract)
        native_full = collect_native_pcm(decoder, source)
        exact_bytes(native_full["pcm"], same_input, "preserved native full SEANet replay")
        isolated = forced_native_stages(decoder, source, old_stages)
        isolated_again = forced_native_stages(decoder, source, old_stages)
        for index in range(15):
            exact_bytes(isolated[index], isolated_again[index], "native isolated reset replay")
        result["native_full_replay_bit_exact"] = result["native_isolated_replay_bit_exact"] = True
        result["isolated_stage_error"] = [dict(stage=index,
            metrics=precision_metrics(old_stages[index], value)) for index, value in enumerate(isolated)]
        result["historical_waveform_metrics"] = dict(
            accumulated=waveform_metrics(old_pcm, accumulated), same_input=waveform_metrics(old_pcm, same_input),
            final_convolution_isolated=waveform_metrics(old_pcm, isolated[-1][:, :, 0]))
        for precision, layout in layouts.items():
            result.update(stage="device_replay", precision=precision)
            save()
            image = layout.pcm
            backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
            try:
                image.upload(backend, weights)
                pcm, stages, state, transfers = device_replay(image, backend, source)
                unchanged = (15 if precision == "int8" else 14 if precision == "fp32"
                             else 3 if precision == "fp32_residual" else 0)
                for index in range(unchanged):
                    exact_bytes(stages[index], old_stages[index], "preserved ISA prefix stage")
                if precision == "int8":
                    exact_bytes(pcm, old_pcm, "historical INT8 PCM exact replay")
                if precision in ("int8", "fp32"):
                    for index, value in enumerate(state):
                        exact_bytes(value, np.load(baseline_dir / f"pcm_state{index}.npy", allow_pickle=False),
                                    "preserved streaming state")
                image.reset(backend)
                replay, replay_stages, replay_state, _ = device_replay(image, backend, source)
                exact_bytes(replay, pcm, "precision candidate PCM reset replay")
                for x, y in zip(stages, replay_stages): exact_bytes(x, y, "precision stage reset replay")
                for x, y in zip(state, replay_state): exact_bytes(x, y, "precision state reset replay")
                forced = forced_native_stages(decoder, source, stages)
                result["candidates"][precision] = dict(prefix_bit_exact=True, unchanged_stage_count=unchanged,
                    reset_bit_exact=True, historical_state_bit_exact=precision in ("int8", "fp32"),
                    historical_state_comparison=("bit exact" if precision in ("int8", "fp32")
                        else "not expected after changed convolution activations"),
                    transfers=transfers, budget=image.budget(),
                    accumulated=waveform_metrics(pcm, accumulated), same_input=waveform_metrics(pcm, same_input),
                    final_convolution_isolated=waveform_metrics(pcm, forced[-1][:, :, 0]), raw_pcm=pcm_stats(pcm),
                    isolated_stage_error=[dict(stage=index, metrics=precision_metrics(stages[index], value))
                                          for index, value in enumerate(forced)])
                _save_array(output_dir, result, precision + "_pcm", pcm)
                if precision == "fp32_all":
                    candidate = result["candidates"][precision]
                    candidate["seanet_acceptance"] = numerical_acceptance(
                        candidate["same_input"], SEANET_LIMITS)
                    candidate["accumulated_codec_acceptance"] = numerical_acceptance(
                        candidate["accumulated"], CODEC_LIMITS)
                    if not candidate["seanet_acceptance"]["passed"]:
                        save()
                        raise RuntimeError("FP32 SEANet missed declared numerical acceptance")
            finally:
                close = getattr(backend, "close", None)
                if close is not None: close()
                del backend
                gc.collect()
            save()
        _save_array(output_dir, result, "native_isolated_final", isolated[-1][:, :, 0])
        for name, digest in baseline_hashes.items():
            if sha256(name) != digest: raise RuntimeError("Preserved baseline changed")
        for name, digest in sources.items():
            if sha256(ROOT / name) != digest: raise RuntimeError("Precision source changed")
        if before != (codec_path.stat().st_size, codec_path.stat().st_mtime_ns):
            raise RuntimeError("Codec changed")
        result.update(status="completed", stage="completed")
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      cpu_wall_seconds=time.perf_counter() - started)
        save()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("codec-path", "baseline-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--target", default=COMMON_TARGET)
    parser.add_argument("--all-convolutions", action="store_true",
                        help="Replay preserved INT8 and optional all-FP32 SEANet only")
    args = parser.parse_args()
    result = run(args.codec_path, args.baseline_dir, args.output_dir, target_name=args.target,
                 all_convolutions=args.all_convolutions)
    print(json.dumps({k: result[k] for k in ("status", "frames", "cpu_wall_seconds")}, indent=2))
