#!/usr/bin/env python3
"""Saved raw-PCM encoder and input-transformer check on pinned CPU/ISA F2.

The established output codec is reserved but never run. Original Moshi modules
provide independent numerical references; this is not a full input codec.
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
from opentpu.llm.personaplex_mimi_decoder import MimiDecoderSpec, native_ring_positions
from opentpu.llm.personaplex_mimi_pcm import MimiPCMSpec
from opentpu.llm.personaplex_mimi_encoder import MimiEncoderImage, MimiEncoderSpec
from opentpu.llm.personaplex_mimi_input_transformer import MimiInputTransformerImage
from tools.personaplex_depth_chain_check import exact_bytes, payload_sha256, sha256, _save_array
from tools.personaplex_isa_check import PINNED_REVISION
from tools.personaplex_mimi_check import codec_header
from tools.personaplex_mimi_encoder_check import _fp32, _metrics_by_frame, numerical_acceptance, ENCODER_LIMITS
from tools.personaplex_mimi_encoder_native import native_encoder_module, collect_native_encoder
from tools.personaplex_mimi_input_transformer_native import (
    native_encoder_transformer, collect_native_encoder_transformer)
from tools.personaplex_mimi_pcm_check import isolated_output
from tools.personaplex_pcm_precision_check import baseline_pcm, COMMON_TARGET, DATA


def _transformer_layer_metrics(actual, reference):
    """Both operands have [frame,layer,row,hidden] order."""
    a, b = np.asarray(actual), np.asarray(reference)
    if (a.dtype != np.float32 or b.dtype != np.float32 or a.shape != b.shape
            or a.ndim != 4 or a.shape[:2] != (2, 8)):
        raise ValueError("Input transformer layers require matching [2,8,2,512] arrays")
    return [dict(layer=index, metrics=_metrics_by_frame(a[:, index], b[:, index]))
            for index in range(8)]


def _joint_budget(base, encoder, transformer, cfg):
    spans = transformer.prefix_spans + transformer.spans
    ordered = sorted(spans, key=lambda span: span["base"])
    for first, second in zip(ordered, ordered[1:]):
        if first["base"] + first["bytes"] > second["base"]:
            raise ValueError("Joint resident spans overlap: " + first["name"] +
                             " and " + second["name"])
    high = max(span["base"] + span["bytes"] for span in spans)
    if high > min(cfg.DRAM_BYTES, 1 << 32):
        raise MemoryError("Joint input/output codec reservation exceeds mapped capacity")
    programs = [span for span in spans if span["kind"] == "program"]
    if len(programs) != 50:
        raise AssertionError("Expected 27 output, 14 encoder, and 9 transformer program slots")
    return dict(mapped_bytes=cfg.DRAM_BYTES, required_bytes=-(-high // 4096) * 4096,
        occupied_bytes=sum(span["bytes"] for span in spans),
        span_count=len(spans), spans=spans, overlap_checked=True,
        program_slots=programs, program_slot_count=len(programs),
        base=base.budget(), encoder=encoder.budget(), transformer=transformer.budget())


def _device_stream(encoder, transformer, backend, inputs, receipt, save, label):
    encoder_programs, transformer_programs = encoder.compile(), transformer.compile()
    if (len(encoder_programs) != 14 or len(transformer_programs) != 9
            or any(program.run_args for program in encoder_programs + transformer_programs)):
        raise AssertionError("Input path needs 23 argument-free resident programs")
    outputs, snapshots, encoder_outputs = [], [], []
    layers = [[] for _ in range(8)]
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
            for component, programs in (("encoder", encoder_programs),
                                        ("input_transformer", transformer_programs)):
                for index, program in enumerate(programs):
                    backend.run([program])
                    receipt.update(stage=label, frame=frame, component=component,
                                   program=index + 1, programs=len(programs))
                    save()
        finally:
            backend.read, backend.write = read, write
        if count != {"reads": 0, "writes": 0}:
            raise AssertionError("Input encoder/transformer made an interstage host transfer")
        encoder_outputs.append(_fp32(backend, encoder.io["output"], (2, 512)))
        snapshots.append(_fp32(backend, transformer.io["input_snapshot"], (2, 512)))
        outputs.append(_fp32(backend, transformer.io["output"], (2, 512)))
        for index, address in enumerate(transformer.layer_outputs):
            layers[index].append(_fp32(backend, address, (2, 512)))
        if (_fp32(backend, encoder.io["frame"], (1,))[0] != frame + 1
                or _fp32(backend, transformer.io["state"], (1,))[0] != frame + 1):
            raise AssertionError("Resident input-path frame counter differs")
        transfers.append(dict(frame=frame, raw_pcm_write_bytes=pcm.nbytes,
                              interstage=count, programs=23))
        receipt["frame_progress"].append(dict(pass_name=label, **transfers[-1]))
        save()
    state = dict(encoder=[np.asarray(backend.read(0, address, size), np.uint8).copy()
                          for address, size in encoder.reset_regions],
                 transformer_kv=[np.asarray(backend.read(0, address, size), np.uint8).copy()
                                 for address, size in transformer.cache_regions],
                 transformer_counter=np.asarray(backend.read(0, transformer.io["state"], 4),
                                                np.uint8).copy())
    return dict(encoder_output=np.stack(encoder_outputs), input_snapshot=np.stack(snapshots),
                output=np.stack(outputs),
                layers=np.stack([np.stack(value) for value in layers], axis=1),
                state=state, transfers=transfers)


def _check_native_positions(native, context, frames):
    if len(native["positions"]) != 8:
        raise ValueError("Native input transformer position layer count changed")
    for layer, positions in enumerate(native["positions"]):
        if len(positions) != frames:
            raise ValueError(f"Native input transformer layer {layer} frame count changed")
        for frame, actual in enumerate(positions):
            expected = native_ring_positions(2 * (frame + 1), context)
            if not np.array_equal(actual, expected):
                raise AssertionError(f"Native input transformer ring positions differ at layer {layer}, frame {frame}")


def _native_boundary_probe(native_transformer, pairs, context):
    """Native-only 128-frame cache exercise; no 128-frame ISA acceptance."""
    repeated = np.tile(pairs, (64, 1, 1))
    if repeated.shape != (128, 2, 512):
        raise ValueError("Native ring probe requires two finite encoder pairs")
    captured = collect_native_encoder_transformer(native_transformer, repeated)
    _check_native_positions(captured, context, 128)
    boundaries = []
    for frame in (123, 124, 125, 127):
        actual = np.asarray(captured["positions"][0][frame], np.int64)
        boundaries.append(dict(frame=frame, end=2 * (frame + 1),
            actual_positions=actual.tolist(),
            positions_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
            layers_matching=8))
    return dict(scope="native-only 128-frame context-wrap probe; two-frame ISA gate unchanged",
                passed=True, frames=128, context=context, boundaries=boundaries)


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
    transformer_contract = json.loads((DATA / "mimi_input_transformer_contract.json").read_text())
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
    names = ["tools/personaplex_mimi_input_transformer_check.py",
             "tools/personaplex_mimi_input_transformer_native.py",
             "tools/personaplex_mimi_encoder_check.py",
             "tools/personaplex_mimi_encoder_native.py",
             "opentpu/llm/personaplex_mimi_input_transformer.py",
             "opentpu/llm/personaplex_mimi_encoder.py",
             "opentpu/llm/personaplex_mimi.py",
             "opentpu/llm/personaplex_mimi_decoder.py",
             "opentpu/llm/personaplex_mimi_pcm.py",
             "opentpu/llm/personaplex_depth_chain.py",
             "tools/personaplex_pcm_precision_check.py",
             *[f"tools/data/personaplex/mimi_{name}_contract.json" for name in
               ("input_transformer", "encoder", "rvq", "decoder", "pcm")],
             "tools/data/personaplex/shared_gemma150_contract.json"]
    sources = {name: sha256(ROOT / name) for name in names}
    inputs = np.load(baseline_dir / "native_accumulated_pcm.npy", allow_pickle=False)
    if inputs.dtype != np.float32 or inputs.shape != (2, 1920) or not np.isfinite(inputs).all():
        raise ValueError("Preserved native accumulated PCM must be finite raw FP32 [2,1920]")
    inputs = np.ascontiguousarray(inputs)

    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "mimi_input_transformer_check.json"
    result = dict(status="running", stage="layout", started_utc=datetime.now(timezone.utc).isoformat(),
        backend="isa", hardware_used=False, hardware_performance_measured=False,
        quality_accepted=False, full_speech_enabled=False, full_model_inference=False,
        checkpoint_payload_hashes_verified=False, frames=2, arrays={}, frame_progress=[],
        scope="Preserved raw native PCM -> resident encoder -> FP32 input transformer only",
        pcm_semantics="raw mono FP32 [2,1920]; no normalization, clipping, scaling, or resampling",
        performance_scope="CPU/ISA wall time and mapped resource budget only; no physical performance claim",
        runtime_revision=common["runtime_revision"], image_context=common["image"],
        execution_target=selected.to_dict(), checkpoint_revision=baseline["checkpoint_revision"],
        baseline_sha256=baseline_hashes, source_files=sources, codec_file=codec_before,
        native_source_sha256=dict(encoder=encoder_contract["installed_source_sha256"],
                                  input_transformer=transformer_contract["installed_source_sha256"]),
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
        encoder = MimiEncoderImage.after_output_codec(
            base, MimiEncoderSpec(**encoder_contract["device_spec"]))
        tf_spec = MimiDecoderSpec(**transformer_contract["device_spec"], precision="fp32")
        transformer = MimiInputTransformerImage.after_encoder(encoder, tf_spec)
        result["budget"] = _joint_budget(base, encoder, transformer, cfg)
        result["encoder_program_words"] = [len(program) * 8 for program in encoder.compile()]
        result["transformer_program_words"] = [len(program) * 8 for program in transformer.compile()]
        result.update(stage="native_materialization"); save()

        import torch
        torch.set_num_threads(1)
        native_encoder, encoder_weights = native_encoder_module(codec_path, encoder_contract)
        native_transformer, layers, rope = native_encoder_transformer(
            codec_path, tf_spec, transformer_contract)
        result["consumed_payload_sha256"] = dict(
            encoder=payload_sha256((f"model{row['model_index']}.{row['subpath']}.{name}", row[name])
                for row in encoder_weights for name in ("weight", "bias")),
            transformer_layers=payload_sha256((f"layer{index}.{name}", value)
                for index, layer in enumerate(layers) for name, value in layer.items()),
            transformer_rope=payload_sha256((("cos", rope[0]), ("sin", rope[1]))))
        result.update(stage="device_upload"); save()
        backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
        encoder.upload(backend, encoder_weights)
        transformer.upload(backend, layers, rope)
        unused = np.asarray(backend.read(0, transformer.transformer.upsample_at,
                                         16 * tf_spec.hidden), np.uint8)
        if np.any(unused):
            raise AssertionError("Unused input-transformer upsample reserve is nonzero")
        result["unused_upsample_reservation_zero"] = True
        result["budget"] = _joint_budget(base, encoder, transformer, cfg)
        save()

        first = _device_stream(encoder, transformer, backend, inputs, result, save,
                               "resident_stream")
        # Save all first-pass evidence before reset or native evaluation.
        for name in ("encoder_output", "input_snapshot", "output", "layers"):
            _save_array(output_dir, result, "isa_input_" + name, first[name])
        for index in range(8):
            _save_array(output_dir, result, f"isa_input_transformer_layer{index}",
                        first["layers"][:, index])
        for index, value in enumerate(first["state"]["encoder"]):
            _save_array(output_dir, result, f"isa_input_encoder_state{index}", value)
        for index, value in enumerate(first["state"]["transformer_kv"]):
            _save_array(output_dir, result, f"isa_input_transformer_kv{index}", value)
        _save_array(output_dir, result, "isa_input_transformer_counter",
                    first["state"]["transformer_counter"])
        result["first_pass_arrays_preserved"] = True
        save()
        exact_bytes(first["input_snapshot"], first["encoder_output"],
                    "resident input transformer snapshot of encoder pair")

        encoder.reset(backend)
        transformer.reset(backend)
        replay = _device_stream(encoder, transformer, backend, inputs, result, save,
                                "reset_stream")
        for name in ("encoder_output", "input_snapshot", "output", "layers"):
            exact_bytes(replay[name], first[name], "input path reset " + name)
        for name in ("encoder", "transformer_kv"):
            if len(replay["state"][name]) != len(first["state"][name]):
                raise AssertionError("Input path raw state region count changed")
            for index, (left, right) in enumerate(zip(first["state"][name], replay["state"][name])):
                exact_bytes(right, left, f"input path {name} raw state {index} reset")
        exact_bytes(replay["state"]["transformer_counter"],
                    first["state"]["transformer_counter"], "input transformer counter reset")
        result.update(reset_bit_exact=True, final_raw_state_reset_bit_exact=True,
                      input_snapshot_bit_exact=True, stage="native_comparison"); save()

        native_enc = collect_native_encoder(native_encoder, inputs)
        native_enc_again = collect_native_encoder(native_encoder, inputs)
        exact_bytes(native_enc_again["output"], native_enc["output"], "fresh native encoder")
        for index in range(15):
            exact_bytes(native_enc_again["layers"][index], native_enc["layers"][index],
                        f"fresh native encoder stage {index}")
        native_full = collect_native_encoder_transformer(native_transformer, native_enc["output"])
        native_again = collect_native_encoder_transformer(native_transformer, native_enc["output"])
        exact_bytes(native_again["output"], native_full["output"], "fresh native input transformer")
        exact_bytes(native_again["layers"], native_full["layers"], "fresh native input layers")
        _check_native_positions(native_full, tf_spec.context, 2)
        same_input = collect_native_encoder_transformer(native_transformer, first["encoder_output"])
        _check_native_positions(same_input, tf_spec.context, 2)
        result["native_fresh_stream_bit_exact"] = True
        result["native_two_frame_positions_match_source"] = True
        result["native_only_ring_boundary_probe"] = _native_boundary_probe(
            native_transformer, native_enc["output"], tf_spec.context)
        result["encoder_input_error"] = _metrics_by_frame(first["encoder_output"], native_enc["output"])
        result["accumulated_output_error"] = _metrics_by_frame(first["output"], native_full["output"])
        result["same_input_transformer_output_error"] = _metrics_by_frame(
            first["output"], same_input["output"])
        result["accumulated_layer_error"] = _transformer_layer_metrics(first["layers"], native_full["layers"])
        result["same_input_transformer_layer_error"] = _transformer_layer_metrics(
            first["layers"], same_input["layers"])
        result["accumulated_acceptance"] = numerical_acceptance(result["accumulated_output_error"])
        result["same_input_transformer_acceptance"] = numerical_acceptance(
            result["same_input_transformer_output_error"])
        _save_array(output_dir, result, "native_input_encoder_output", native_enc["output"])
        _save_array(output_dir, result, "native_input_transformer_output", native_full["output"])
        save()

        for name, digest in baseline_hashes.items():
            if sha256(name) != digest:
                raise RuntimeError("Preserved baseline changed: " + name)
        for name, digest in sources.items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Input transformer check source changed: " + name)
        for name, digest in common["pinned_runtime_source_sha256"].items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Pinned runtime source changed: " + name)
        for component, module, contract in (("encoder", native_encoder, encoder_contract),
                                            ("input_transformer", native_transformer, transformer_contract)):
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
                      ("status", "frames", "accumulated_acceptance", "cpu_wall_seconds")}, indent=2))
