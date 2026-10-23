#!/usr/bin/env python3
"""Saved raw-PCM input SEANet encoder check on the pinned CPU/ISA profile.

This executes only the fourteen resident encoder convolutions. The original
Moshi encoder is an independent numerical reference, never a serving path.
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
from opentpu.llm.personaplex_mimi_decoder import MimiDecoderSpec
from opentpu.llm.personaplex_mimi_pcm import MimiPCMSpec
from opentpu.llm.personaplex_mimi_encoder import MimiEncoderImage, MimiEncoderSpec
from tools.personaplex_depth_chain_check import exact_bytes, payload_sha256, sha256, _save_array
from tools.personaplex_isa_check import PINNED_REVISION, precision_metrics
from tools.personaplex_mimi_check import codec_header
from tools.personaplex_mimi_encoder_native import native_encoder_module, collect_native_encoder
from tools.personaplex_mimi_pcm_check import isolated_output
from tools.personaplex_pcm_precision_check import baseline_pcm, COMMON_TARGET, DATA


ENCODER_LIMITS = dict(relative_l2=1e-4, rmse=1e-5, max_absolute_error=1e-4)


def _metrics_by_frame(actual, reference):
    a, b = np.asarray(actual), np.asarray(reference)
    if (a.dtype != np.float32 or b.dtype != np.float32 or a.shape != b.shape
            or a.ndim < 1 or a.shape[0] != 2
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Encoder metrics require equal finite FP32 two-frame arrays")
    return {"aggregate": precision_metrics(a, b),
            "per_frame": [precision_metrics(x, y) for x, y in zip(a, b)]}


def numerical_acceptance(metrics, limits=ENCODER_LIMITS):
    entries = [metrics["aggregate"], *metrics["per_frame"]]
    return dict(limits=limits, passed=all(
        item[name] is not None and item[name] <= maximum
        for item in entries for name, maximum in limits.items()),
        scope="aggregate and each frame on the saved two-frame fixture only")


def _fp32(backend, address, shape):
    size = int(np.prod(shape))
    value = np.asarray(backend.read(0, address, 4 * size), np.uint8).view(np.float32).copy()
    value = value.reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError("Resident encoder output is nonfinite")
    return value


def _joint_budget(base, encoder, cfg, base_budget):
    spans = [*base.spans, *encoder.spans]
    ordered = sorted(spans, key=lambda span: span["base"])
    for previous, current in zip(ordered, ordered[1:]):
        if previous["base"] + previous["bytes"] > current["base"]:
            raise ValueError("Joint resident spans overlap: " + previous["name"] +
                             " and " + current["name"])
    high = max(span["base"] + span["bytes"] for span in spans)
    if high > min(cfg.DRAM_BYTES, 1 << 32):
        raise MemoryError("Joint resident image exceeds mapped capacity")
    encoder_budget = encoder.budget()
    return dict(mapped_bytes=cfg.DRAM_BYTES, required_bytes=encoder_budget["combined_required_bytes"],
                occupied_bytes=sum(span["bytes"] for span in spans),
                span_count=len(spans), spans=spans, overlap_checked=True,
                program_slots=encoder.program_slots,
                base=base_budget, encoder=encoder_budget,
                encoder_state_slack_bytes=encoder_budget["input_state_slack_bytes"],
                encoder_weight_slack_bytes=encoder_budget["input_weight_slack_bytes"])


def _device_stream(image, backend, inputs, receipt, save, label):
    """One raw frame write, fourteen resident programs, then diagnostic reads."""
    programs = image.compile()
    if len(programs) != 14 or any(p.run_args for p in programs):
        raise AssertionError("Encoder requires fourteen argument-free programs")
    outputs, stages, transfers = [], [[] for _ in range(15)], []
    for frame, pcm in enumerate(inputs):
        backend.write(0, image.input_at, pcm.reshape(-1, 1))
        receipt.update(stage=label, frame=frame, component="raw_pcm_input")
        save()
        read, write = backend.read, backend.write
        count = {"reads": 0, "writes": 0}

        def counted_read(*args, **kwargs):
            count["reads"] += 1
            return read(*args, **kwargs)

        def counted_write(*args, **kwargs):
            count["writes"] += 1
            return write(*args, **kwargs)

        backend.read, backend.write = counted_read, counted_write
        try:
            for index, program in enumerate(programs):
                backend.run([program])
                receipt.update(stage=label, frame=frame, component="encoder",
                               program=index + 1, programs=len(programs))
                save()
        finally:
            backend.read, backend.write = read, write
        if count != {"reads": 0, "writes": 0}:
            raise AssertionError("Encoder made an interstage host transfer")
        for index, (address, rows, channels) in enumerate(image.layer_outputs):
            stages[index].append(_fp32(backend, address, (rows, channels)))
        outputs.append(_fp32(backend, image.io["output"],
                             (image.spec.output_rows, image.spec.hidden)))
        if _fp32(backend, image.io["frame"], (1,))[0] != frame + 1:
            raise AssertionError("Resident encoder frame counter differs")
        transfers.append(dict(frame=frame, raw_pcm_write_bytes=pcm.nbytes,
                              interstage=count))
        receipt["frame_progress"].append(dict(pass_name=label, **transfers[-1]))
        save()
    state = [np.asarray(backend.read(0, address, size), np.uint8).copy()
             for address, size in image.reset_regions]
    return dict(output=np.stack(outputs), layers=[np.stack(value) for value in stages],
                state=state, transfers=transfers)


def run(codec_path, baseline_dir, output_dir, *, target_name=COMMON_TARGET):
    if target_name != COMMON_TARGET:
        raise ValueError("Require exact shared target")
    codec_path, baseline_dir = Path(codec_path).resolve(), Path(baseline_dir).resolve()
    output_dir = isolated_output(output_dir, ROOT, codec_path, baseline_dir)
    baseline, baseline_hashes = baseline_pcm(baseline_dir)
    common = json.loads((DATA / "shared_gemma150_contract.json").read_text())
    rvq_contract = json.loads((DATA / "mimi_rvq_contract.json").read_text())
    decoder_contract = json.loads((DATA / "mimi_decoder_contract.json").read_text())
    pcm_contract = json.loads((DATA / "mimi_pcm_contract.json").read_text())
    contract = json.loads((DATA / "mimi_encoder_contract.json").read_text())
    selected = target.load(target_name)
    if (common["target"] != COMMON_TARGET or common["expected_arch_hash"] != "dd6b92a6"
            or common["runtime_revision"] != "18fb978033799f0f21258f0e6e085c0360322d96"
            or selected.to_dict()["arch_hash"] != common["expected_arch_hash"]
            or sha256(selected.path) != common["target_source_sha256"]
            or baseline.get("checkpoint_revision") != PINNED_REVISION
            or baseline.get("runtime_revision") != common["runtime_revision"]
            or baseline.get("image_context") != common["image"]):
        raise ValueError("Pinned target, runtime, image, or checkpoint changed")
    for name, digest in common["pinned_runtime_source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Pinned runtime source changed: " + name)
    codec_before = codec_header(codec_path, rvq_contract)
    names = ["tools/personaplex_mimi_encoder_check.py",
             "tools/personaplex_mimi_encoder_native.py",
             "opentpu/llm/personaplex_mimi_encoder.py",
             "opentpu/llm/personaplex_mimi.py",
             "opentpu/llm/personaplex_mimi_decoder.py",
             "opentpu/llm/personaplex_mimi_pcm.py",
             "opentpu/llm/personaplex_depth_chain.py",
             "tools/personaplex_pcm_precision_check.py",
             "tools/data/personaplex/mimi_encoder_contract.json",
             "tools/data/personaplex/mimi_rvq_contract.json",
             "tools/data/personaplex/mimi_decoder_contract.json",
             "tools/data/personaplex/mimi_pcm_contract.json",
             "tools/data/personaplex/shared_gemma150_contract.json"]
    source_hashes = {name: sha256(ROOT / name) for name in names}
    inputs = np.load(baseline_dir / "native_accumulated_pcm.npy", allow_pickle=False)
    if inputs.dtype != np.float32 or inputs.shape != (2, 1920) or not np.isfinite(inputs).all():
        raise ValueError("Preserved native accumulated PCM must be finite raw FP32 [2,1920]")
    inputs = np.ascontiguousarray(inputs)

    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "mimi_encoder_check.json"
    result = dict(status="running", stage="layout", started_utc=datetime.now(timezone.utc).isoformat(),
        backend="isa", hardware_used=False, hardware_performance_measured=False,
        quality_accepted=False, full_speech_enabled=False, full_model_inference=False,
        checkpoint_payload_hashes_verified=False, frames=2, arrays={}, frame_progress=[],
        scope="Preserved raw native PCM -> resident causal SEANet input encoder only",
        pcm_semantics="raw mono FP32 [2,1920]; no normalization, clipping, scaling, or resampling",
        performance_scope="CPU/ISA wall time and mapped resource budget only; no physical performance claim",
        runtime_revision=common["runtime_revision"], image_context=common["image"],
        execution_target=selected.to_dict(), checkpoint_revision=baseline["checkpoint_revision"],
        baseline_sha256=baseline_hashes, source_files=source_hashes, codec_file=codec_before,
        native_source_sha256=contract["installed_source_sha256"],
        acceptance_limits=ENCODER_LIMITS)

    def save():
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(receipt_path)

    started = time.perf_counter()
    backend = None
    save()
    try:
        cfg = target.config(selected)
        base = DepthChainImage(DepthChainSpec(), cfg,
            mimi_spec=MimiRVQSpec(**rvq_contract["device_spec"], projection_precision="fp32"),
            decoder_spec=MimiDecoderSpec(**decoder_contract["device_spec"], precision="fp32"),
            pcm_spec=MimiPCMSpec(**pcm_contract["device_spec"], final_precision="fp32",
                residual_precision="fp32", initial_precision="fp32", transpose_precision="fp32"))
        base_budget = base.budget()  # Full established image and program fit.
        spec = MimiEncoderSpec(**contract["device_spec"])
        input_at = base.pcm.weight_end
        state_base = -(-(input_at + 4 * spec.samples) // 128) * 128
        weight_base = input_at + (32 << 20)
        result["layout_request"] = dict(input_at=input_at, input_bytes=4 * spec.samples,
            state_base=state_base, state_limit=weight_base,
            state_reservation_bytes=32 << 20,
            state_available_bytes=weight_base - state_base,
            weight_base=weight_base, weight_limit=0xB0000000,
            weight_available_bytes=0xB0000000 - weight_base,
            established_budget=base_budget)
        save()
        image = MimiEncoderImage.after_output_codec(base, spec)
        result["budget"] = _joint_budget(base, image, cfg, base_budget)
        result["encoder_program_words"] = [len(program) * 8 for program in image.compile()]
        result.update(stage="native_materialization"); save()

        import torch
        torch.set_num_threads(1)
        native, weights = native_encoder_module(codec_path, contract)
        result["consumed_encoder_payload_sha256"] = payload_sha256(
            (f"model{row['model_index']}.{row['subpath']}.{name}", row[name])
            for row in weights for name in ("weight", "bias"))
        result.update(stage="device_upload"); save()
        backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
        image.upload(backend, weights)
        result["budget"] = _joint_budget(base, image, cfg, base_budget)
        save()

        first = _device_stream(image, backend, inputs, result, save, "resident_stream")
        # Persist every first-pass boundary and raw reset region before reset
        # or any numerical comparison can fail.
        _save_array(output_dir, result, "isa_encoder_output", first["output"])
        for index, value in enumerate(first["layers"]):
            _save_array(output_dir, result, f"isa_encoder_stage{index}", value)
        for index, value in enumerate(first["state"]):
            _save_array(output_dir, result, f"isa_encoder_state{index}", value)
        result["first_pass_arrays_preserved"] = True
        save()

        image.reset(backend)
        replay = _device_stream(image, backend, inputs, result, save, "reset_stream")
        exact_bytes(replay["output"], first["output"], "encoder output reset replay")
        for index in range(15):
            exact_bytes(replay["layers"][index], first["layers"][index],
                        f"encoder stage {index} reset replay")
        for index, (left, right) in enumerate(zip(first["state"], replay["state"])):
            exact_bytes(right, left, f"encoder raw state {index} reset replay")
        result.update(reset_bit_exact=True, final_raw_state_reset_bit_exact=True,
                      stage="native_comparison"); save()

        native_full = collect_native_encoder(native, inputs)
        native_again = collect_native_encoder(native, inputs)
        # The diagnostic helper traverses modules to capture stages. Verify
        # that traversal against the actual source forward entry point too.
        direct = []
        with torch.no_grad(), native.streaming(1):
            for frame in inputs:
                source = torch.from_numpy(frame.reshape(1, 1, 1920).copy())
                direct.append(native(source).detach().cpu().numpy()[0].T.copy())
        exact_bytes(np.stack(direct), native_full["output"],
                    "native encoder forward versus captured module traversal")
        result["native_forward_entrypoint_bit_exact"] = True
        exact_bytes(native_again["output"], native_full["output"], "native encoder fresh stream")
        for index in range(15):
            exact_bytes(native_again["layers"][index], native_full["layers"][index],
                        f"native encoder stage {index} fresh stream")
        isolated = collect_native_encoder(native, inputs, forced_prefix=first["layers"][:14])
        result["native_fresh_stream_bit_exact"] = True
        result["accumulated_output_error"] = _metrics_by_frame(first["output"], native_full["output"])
        result["same_input_stage_output_error"] = _metrics_by_frame(first["output"], isolated["output"])
        result["accumulated_stage_error"] = [dict(stage=index, metrics=_metrics_by_frame(
            first["layers"][index], native_full["layers"][index])) for index in range(15)]
        result["same_input_stage_error"] = [dict(stage=index, metrics=_metrics_by_frame(
            first["layers"][index], isolated["layers"][index])) for index in range(15)]
        result["accumulated_acceptance"] = numerical_acceptance(result["accumulated_output_error"])
        result["same_input_stage_output_acceptance"] = numerical_acceptance(
            result["same_input_stage_output_error"])
        _save_array(output_dir, result, "native_encoder_output", native_full["output"])
        _save_array(output_dir, result, "same_input_stage_output", isolated["output"])
        for index in range(15):
            _save_array(output_dir, result, f"native_encoder_stage{index}", native_full["layers"][index])
            _save_array(output_dir, result, f"same_input_encoder_stage{index}", isolated["layers"][index])
        save()

        for name, digest in baseline_hashes.items():
            if sha256(name) != digest:
                raise RuntimeError("Preserved baseline changed: " + name)
        for name, digest in source_hashes.items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Input encoder check source changed: " + name)
        for name, digest in common["pinned_runtime_source_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Pinned runtime source changed: " + name)
        native_root = Path(inspect.getfile(type(native))).resolve().parents[1]
        for name, digest in contract["installed_source_sha256"].items():
            if sha256(native_root / name) != digest:
                raise RuntimeError("Installed native encoder source changed: " + name)
        if codec_header(codec_path, rvq_contract) != codec_before:
            raise RuntimeError("Cached codec header/file metadata changed")
        result.update(status="completed", stage="completed")
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            close = getattr(backend, "close", None)
            if close is not None:
                close()
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
    parser.add_argument("--target", default=COMMON_TARGET)
    args = parser.parse_args()
    receipt = run(args.codec_path, args.baseline_dir, args.output_dir,
                  target_name=args.target)
    print(json.dumps({name: receipt[name] for name in
                      ("status", "frames", "accumulated_acceptance", "cpu_wall_seconds")}, indent=2))
