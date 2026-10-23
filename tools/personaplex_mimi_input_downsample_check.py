#!/usr/bin/env python3
"""Saved raw-PCM input path through learned downsample on CPU/ISA.

The established output codec is reserved but not run. Original Moshi provides
offline references. This is a component check, not input RVQ, speech, serving,
or physical-device qualification.
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
from opentpu.llm.personaplex_mimi_input_transformer import MimiInputTransformerImage
from opentpu.llm.personaplex_mimi_input_downsample import (
    MimiInputDownsampleImage, MimiInputDownsampleSpec)
from tools.personaplex_depth_chain_check import exact_bytes, payload_sha256, sha256, _save_array
from tools.personaplex_isa_check import PINNED_REVISION
from tools.personaplex_mimi_check import codec_header
from tools.personaplex_mimi_encoder_check import _fp32, _metrics_by_frame, numerical_acceptance
from tools.personaplex_mimi_encoder_native import native_encoder_module, collect_native_encoder
from tools.personaplex_mimi_input_transformer_check import _check_native_positions
from tools.personaplex_mimi_input_transformer_native import (
    native_encoder_transformer, collect_native_encoder_transformer)
from tools.personaplex_mimi_input_downsample_native import (
    native_input_downsample, collect_native_input_downsample)
from tools.personaplex_mimi_pcm_check import isolated_output
from tools.personaplex_pcm_precision_check import baseline_pcm, COMMON_TARGET, DATA


DOWNSAMPLE_LIMITS = dict(relative_l2=1e-4, rmse=1e-5, max_absolute_error=1e-4)


def _joint_budget(downsample, cfg):
    spans = downsample.prefix_spans + downsample.spans
    ordered = sorted(spans, key=lambda span: span["base"])
    for first, second in zip(ordered, ordered[1:]):
        if first["base"] + first["bytes"] > second["base"]:
            raise ValueError("Joint resident spans overlap: " + first["name"] +
                             " and " + second["name"])
    high = max(span["base"] + span["bytes"] for span in spans)
    if high > min(cfg.DRAM_BYTES, 1 << 32):
        raise MemoryError("Joint input/output codec reservation exceeds mapped capacity")
    slots = [span for span in spans if span["kind"] == "program"]
    if len(slots) != 51:
        raise AssertionError("Expected 51 joint slots: 27 output, 14 encoder, 9 transformer, 1 downsample")
    return dict(mapped_bytes=cfg.DRAM_BYTES, required_bytes=-(-high // 4096) * 4096,
        occupied_bytes=sum(span["bytes"] for span in spans), span_count=len(spans),
        spans=spans, overlap_checked=True, program_slot_count=len(slots),
        program_slots=slots, downsample=downsample.budget())


def _device_stream(encoder, transformer, downsample, backend, inputs,
                   receipt, save, label):
    groups = (("encoder", encoder.compile()),
              ("input_transformer", transformer.compile()),
              ("input_downsample", downsample.compile()))
    if ([len(programs) for _, programs in groups] != [14, 9, 1]
            or any(program.run_args for _, programs in groups for program in programs)):
        raise AssertionError("Input path needs 24 argument-free resident programs")
    observed = {name: [] for name in ("encoder_output", "input_snapshot",
                                     "transformer_output", "output")}
    transfers = []
    for frame, pcm in enumerate(inputs):
        backend.write(0, encoder.input_at, pcm.reshape(-1, 1))
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
            for component, programs in groups:
                for index, program in enumerate(programs):
                    backend.run([program])
                    receipt.update(stage=label, frame=frame, component=component,
                                   program=index + 1, programs=len(programs))
                    save()
        finally:
            backend.read, backend.write = read, write
        if count != {"reads": 0, "writes": 0}:
            raise AssertionError("Input path made an interstage host transfer")
        observed["encoder_output"].append(_fp32(backend, encoder.io["output"], (2, 512)))
        observed["input_snapshot"].append(
            _fp32(backend, transformer.io["input_snapshot"], (2, 512)))
        observed["transformer_output"].append(
            _fp32(backend, transformer.io["output"], (2, 512)))
        observed["output"].append(_fp32(backend, downsample.io["output"], (1, 512)))
        if (_fp32(backend, encoder.io["frame"], (1,))[0] != frame + 1
                or _fp32(backend, transformer.io["state"], (1,))[0] != frame + 1
                or _fp32(backend, downsample.io["frame"], (1,))[0] != frame + 1
                or _fp32(backend, downsample.io["first"], (1,))[0] != 0):
            raise AssertionError("Resident input-path streaming counter or first flag differs")
        transfers.append(dict(frame=frame, raw_pcm_write_bytes=pcm.nbytes,
                              interstage=count, programs=24))
        receipt["frame_progress"].append(dict(pass_name=label, **transfers[-1]))
        save()
    state = dict(
        encoder=[np.asarray(backend.read(0, address, size), np.uint8).copy()
                 for address, size in encoder.reset_regions],
        transformer_kv=[np.asarray(backend.read(0, address, size), np.uint8).copy()
                        for address, size in transformer.cache_regions],
        transformer_counter=np.asarray(backend.read(0, transformer.io["state"], 4), np.uint8).copy(),
        downsample=[np.asarray(backend.read(0, address, size), np.uint8).copy()
                    for address, size in downsample.reset_regions])
    return dict(**{name: np.stack(values) for name, values in observed.items()},
                state=state, transfers=transfers)


def require_downsample_acceptance(result):
    failed = [name for name in ("accumulated_acceptance", "same_isa_tf_input_acceptance")
              if not result[name]["passed"]]
    if failed:
        raise RuntimeError("Learned input downsample missed declared numerical acceptance: "
                           + ", ".join(failed))


def _source_hashes():
    names = ["tools/personaplex_mimi_input_downsample_check.py",
             "tools/personaplex_mimi_input_downsample_native.py",
             "tools/personaplex_mimi_input_transformer_check.py",
             "tools/personaplex_mimi_input_transformer_native.py",
             "tools/personaplex_mimi_encoder_check.py",
             "tools/personaplex_mimi_encoder_native.py",
             "opentpu/llm/personaplex_mimi_input_downsample.py",
             "opentpu/llm/personaplex_mimi_input_transformer.py",
             "opentpu/llm/personaplex_mimi_encoder.py",
             "opentpu/llm/personaplex_mimi.py",
             "opentpu/llm/personaplex_mimi_decoder.py",
             "opentpu/llm/personaplex_mimi_pcm.py",
             "opentpu/llm/personaplex_depth_chain.py",
             "tools/personaplex_pcm_precision_check.py",
             *[f"tools/data/personaplex/mimi_{name}_contract.json" for name in
               ("input_downsample", "input_transformer", "encoder", "rvq", "decoder", "pcm")],
             "tools/data/personaplex/shared_gemma150_contract.json"]
    return {name: sha256(ROOT / name) for name in names}


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
    encoder_contract = json.loads((DATA / "mimi_encoder_contract.json").read_text())
    tf_contract = json.loads((DATA / "mimi_input_transformer_contract.json").read_text())
    down_contract = json.loads((DATA / "mimi_input_downsample_contract.json").read_text())
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
    sources = _source_hashes()
    inputs = np.load(baseline_dir / "native_accumulated_pcm.npy", allow_pickle=False)
    if inputs.dtype != np.float32 or inputs.shape != (2, 1920) or not np.isfinite(inputs).all():
        raise ValueError("Preserved native accumulated PCM must be finite raw FP32 [2,1920]")
    inputs = np.ascontiguousarray(inputs)

    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "mimi_input_downsample_check.json"
    result = dict(status="running", stage="layout", started_utc=datetime.now(timezone.utc).isoformat(),
        backend="isa", hardware_used=False, hardware_performance_measured=False,
        quality_accepted=False, full_speech_enabled=False, full_model_inference=False,
        checkpoint_payload_hashes_verified=False, frames=2, arrays={}, frame_progress=[],
        scope="Preserved raw native PCM -> resident encoder -> FP32 input transformer -> learned input downsample only",
        pcm_semantics="raw mono FP32 [2,1920]; no normalization, clipping, scaling, or resampling",
        performance_scope="CPU/ISA wall time and mapped budget only; no physical performance claim",
        runtime_revision=common["runtime_revision"], image_context=common["image"],
        execution_target=selected.to_dict(), checkpoint_revision=baseline["checkpoint_revision"],
        baseline_sha256=baseline_hashes, source_files=sources, codec_file=codec_before,
        native_source_sha256=dict(encoder=encoder_contract["installed_source_sha256"],
            input_transformer=tf_contract["installed_source_sha256"],
            input_downsample=down_contract["installed_source_sha256"]),
        acceptance_limits=DOWNSAMPLE_LIMITS)

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
        encoder = MimiEncoderImage.after_output_codec(
            base, MimiEncoderSpec(**encoder_contract["device_spec"]))
        tf_spec = MimiDecoderSpec(**tf_contract["device_spec"], precision="fp32")
        transformer = MimiInputTransformerImage.after_encoder(encoder, tf_spec)
        downsample = MimiInputDownsampleImage.after_transformer(
            transformer, MimiInputDownsampleSpec(**down_contract["device_spec"]))
        result["budget"] = _joint_budget(downsample, cfg)
        result["program_words"] = dict(
            encoder=[len(program) * 8 for program in encoder.compile()],
            input_transformer=[len(program) * 8 for program in transformer.compile()],
            input_downsample=[len(program) * 8 for program in downsample.compile()])
        result.update(stage="native_materialization"); save()

        import torch
        torch.set_num_threads(1)
        native_encoder, encoder_weights = native_encoder_module(codec_path, encoder_contract)
        native_tf, layers, rope = native_encoder_transformer(codec_path, tf_spec, tf_contract)
        native_down, down_weight = native_input_downsample(codec_path, downsample.spec, down_contract)
        result["consumed_payload_sha256"] = dict(
            encoder=payload_sha256((f"model{row['model_index']}.{row['subpath']}.{name}", row[name])
                for row in encoder_weights for name in ("weight", "bias")),
            input_transformer_layers=payload_sha256((f"layer{index}.{name}", value)
                for index, layer in enumerate(layers) for name, value in layer.items()),
            input_transformer_rope=payload_sha256((("cos", rope[0]), ("sin", rope[1]))),
            input_downsample=payload_sha256((("downsample.conv.conv.conv.weight", down_weight),)))
        result.update(stage="device_upload"); save()
        backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
        encoder.upload(backend, encoder_weights)
        transformer.upload(backend, layers, rope)
        downsample.upload(backend, down_weight)
        result["budget"] = _joint_budget(downsample, cfg)
        save()

        first = _device_stream(encoder, transformer, downsample, backend, inputs,
                               result, save, "resident_stream")
        for name in ("encoder_output", "input_snapshot", "transformer_output", "output"):
            _save_array(output_dir, result, "isa_input_" + name, first[name])
        for name in ("encoder", "transformer_kv", "downsample"):
            for index, value in enumerate(first["state"][name]):
                _save_array(output_dir, result, f"isa_input_{name}_state{index}", value)
        _save_array(output_dir, result, "isa_input_transformer_counter",
                    first["state"]["transformer_counter"])
        result["first_pass_arrays_preserved"] = True
        save()
        exact_bytes(first["input_snapshot"], first["encoder_output"],
                    "resident transformer input snapshot")
        encoder.reset(backend)
        transformer.reset(backend)
        downsample.reset(backend)
        replay = _device_stream(encoder, transformer, downsample, backend, inputs,
                                result, save, "reset_stream")
        for name in ("encoder_output", "input_snapshot", "transformer_output", "output"):
            exact_bytes(replay[name], first[name], "input path reset " + name)
        for name in ("encoder", "transformer_kv", "downsample"):
            if len(replay["state"][name]) != len(first["state"][name]):
                raise AssertionError("Input path raw state region count changed")
            for index, (left, right) in enumerate(zip(first["state"][name], replay["state"][name])):
                exact_bytes(right, left, f"input path {name} raw state {index} reset")
        exact_bytes(replay["state"]["transformer_counter"],
                    first["state"]["transformer_counter"], "input transformer counter reset")
        result.update(reset_bit_exact=True, final_raw_state_reset_bit_exact=True,
                      input_snapshot_bit_exact=True, stage="native_comparison"); save()

        source_encoder = collect_native_encoder(native_encoder, inputs)
        source_tf = collect_native_encoder_transformer(native_tf, source_encoder["output"])
        _check_native_positions(source_tf, tf_spec.context, 2)
        source_full = collect_native_input_downsample(native_down, source_tf["output"])
        source_again = collect_native_input_downsample(native_down, source_tf["output"])
        exact_bytes(source_again, source_full, "native downsample fresh stream")
        same_input = collect_native_input_downsample(native_down, first["transformer_output"])
        result["native_fresh_stream_bit_exact"] = True
        result["encoder_input_error"] = _metrics_by_frame(
            first["encoder_output"], source_encoder["output"])
        result["transformer_accumulated_error"] = _metrics_by_frame(
            first["transformer_output"], source_tf["output"])
        result["accumulated_output_error"] = _metrics_by_frame(first["output"], source_full)
        result["same_isa_tf_input_error"] = _metrics_by_frame(first["output"], same_input)
        result["accumulated_acceptance"] = numerical_acceptance(
            result["accumulated_output_error"], DOWNSAMPLE_LIMITS)
        result["same_isa_tf_input_acceptance"] = numerical_acceptance(
            result["same_isa_tf_input_error"], DOWNSAMPLE_LIMITS)
        _save_array(output_dir, result, "native_input_downsample_output", source_full)
        _save_array(output_dir, result, "native_same_isa_tf_input_downsample_output", same_input)
        save()
        require_downsample_acceptance(result)

        for name, digest in baseline_hashes.items():
            if sha256(name) != digest:
                raise RuntimeError("Preserved baseline changed: " + name)
        for name, digest in sources.items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Input downsample check source changed: " + name)
        for name, digest in common["pinned_runtime_source_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Pinned runtime source changed: " + name)
        for component, module, contract in (("encoder", native_encoder, encoder_contract),
                                            ("input_transformer", native_tf, tf_contract),
                                            ("input_downsample", native_down, down_contract)):
            native_root = Path(inspect.getfile(type(module))).resolve().parents[1]
            for name, digest in contract["installed_source_sha256"].items():
                if sha256(native_root / name) != digest:
                    raise RuntimeError("Installed native " + component + " source changed: " + name)
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
                      ("status", "frames", "same_isa_tf_input_acceptance", "cpu_wall_seconds")}, indent=2))
