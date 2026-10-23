#!/usr/bin/env python3
"""Pinned raw-PCM through input RVQ on the resident CPU/ISA image.

The output codec occupies its established reservation but does not run. Native
Moshi references are computed offline after the complete resident stream.
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
from opentpu.llm.personaplex_mimi_input_transformer import MimiInputTransformerImage
from opentpu.llm.personaplex_mimi_input_downsample import (
    MimiInputDownsampleImage, MimiInputDownsampleSpec)
from opentpu.llm.personaplex_mimi_input_rvq import MimiInputRVQImage, MimiInputRVQSpec
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
from tools.personaplex_mimi_input_rvq_native import (
    native_input_quantizer, collect_native_input_quantizer)
from tools.personaplex_mimi_full_input_native import check_native_full_input
from tools.personaplex_mimi_pcm_check import isolated_output
from tools.personaplex_pcm_precision_check import baseline_pcm, COMMON_TARGET, DATA


INPUT_RVQ_LIMITS = dict(relative_l2=1e-4, rmse=1e-5, max_absolute_error=1e-4)
_OBSERVED = ("encoder_output", "input_snapshot", "transformer_output", "downsample_output",
             "semantic_projected", "acoustic_projected", "semantic_residual",
             "acoustic_residual", "residual_before", "selected_centroids",
             "residual_after", "distances", "ids")


def _joint_budget(rvq, cfg):
    spans = rvq.prefix_spans + rvq.spans
    ordered = sorted(spans, key=lambda span: span["base"])
    for first, second in zip(ordered, ordered[1:]):
        if first["base"] + first["bytes"] > second["base"]:
            raise ValueError("Joint resident spans overlap: " + first["name"] +
                             " and " + second["name"])
    high = max(span["base"] + span["bytes"] for span in spans)
    if high > min(cfg.DRAM_BYTES, 1 << 32):
        raise MemoryError("Joint input/output codec reservation exceeds mapped capacity")
    slots = [span for span in spans if span["kind"] == "program"]
    tables = [span for span in spans if span["name"] == "mimi_tables"]
    if len(slots) != 60 or len(tables) != 1 or tables[0]["base"] != rvq.tables_at:
        raise AssertionError("Expected 60 joint slots and one shared frozen mimi_tables span")
    return dict(mapped_bytes=cfg.DRAM_BYTES, required_bytes=-(-high // 4096) * 4096,
        occupied_bytes=sum(span["bytes"] for span in spans), span_count=len(spans),
        spans=spans, overlap_checked=True, program_slot_count=len(slots),
        program_slots=slots, input_rvq=rvq.budget(), shared_table_owner="mimi_tables")


def _checked_ids(values, *, frames=2, codebooks=8, vocab=2048):
    """Validate resident FP32 ID words before their exact integer comparison."""
    ids = np.asarray(values)
    if (ids.dtype != np.float32 or ids.shape != (frames, codebooks)
            or not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all()
            or not ((ids >= 0) & (ids < vocab)).all()):
        raise ValueError("Resident input RVQ IDs must be finite integral FP32 in vocabulary")
    return ids.astype(np.int64)


def _id_agreement(actual, reference, label):
    left, right = np.asarray(actual), np.asarray(reference)
    if (left.dtype != np.int64 or right.dtype != np.int64 or left.shape != right.shape
            or left.ndim != 2 or left.shape[1] != 8):
        raise ValueError("Input RVQ ID comparison requires equal int64 [frames,8]")
    per_frame = [bool(np.array_equal(a, b)) for a, b in zip(left, right)]
    return dict(label=label, passed=all(per_frame), per_frame=per_frame,
                differing_ids_per_frame=[int(np.count_nonzero(a != b))
                                         for a, b in zip(left, right)])


def _distance_diagnostics(device_squared, native_euclidean, margins):
    """Offline diagnostic only: direct squared FP32 and torch.cdist can differ."""
    device = np.asarray(device_squared)
    source = np.asarray(native_euclidean)
    gap = np.asarray(margins)
    if (device.dtype != np.float32 or source.dtype != np.float32 or gap.dtype != np.float32
            or device.ndim != 3 or device.shape[1:] != (8, 2048)
            or source.shape != device.shape or gap.shape != device.shape[:2]
            or not np.isfinite(device).all() or not np.isfinite(source).all()
            or not np.isfinite(gap).all() or (device < 0).any()
            or (source < 0).any() or (gap < 0).any()):
        raise ValueError("Input RVQ distance diagnostic needs finite nonnegative FP32 arrays")
    difference = np.abs(np.sqrt(device) - source)
    return dict(native_winner_margin_min=float(np.min(gap)),
                native_winner_margin_per_frame=[float(np.min(row)) for row in gap],
                sqrt_device_minus_native_euclidean_max=float(np.max(difference)),
                sqrt_device_minus_native_euclidean_max_per_frame=[
                    float(np.max(row)) for row in difference],
                scope="Offline diagnostic; direct FP32 squared distance is not torch.cdist equivalence")


def _device_stream(encoder, transformer, downsample, rvq, backend, inputs,
                   receipt, save, label):
    groups = (("encoder", encoder.compile()),
              ("input_transformer", transformer.compile()),
              ("input_downsample", downsample.compile()),
              ("input_rvq", rvq.compile()))
    if ([len(programs) for _, programs in groups] != [14, 9, 1, 9]
            or any(program.run_args for _, programs in groups for program in programs)):
        raise AssertionError("Input path needs 33 argument-free resident programs")
    observed = {name: [] for name in _OBSERVED}
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
        addresses = (
            ("encoder_output", encoder.io["output"], (2, 512)),
            ("input_snapshot", transformer.io["input_snapshot"], (2, 512)),
            ("transformer_output", transformer.io["output"], (2, 512)),
            ("downsample_output", downsample.io["output"], (1, 512)),
            ("semantic_projected", rvq.io["semantic_projected"], (256,)),
            ("acoustic_projected", rvq.io["acoustic_projected"], (256,)),
            ("semantic_residual", rvq.io["semantic_residual"], (256,)),
            ("acoustic_residual", rvq.io["acoustic_residual"], (256,)),
            ("residual_before", rvq.io["residual_before"], (8, 256)),
            ("selected_centroids", rvq.io["selected_centroids"], (8, 256)),
            ("residual_after", rvq.io["residual_after"], (8, 256)),
            ("distances", rvq.io["distances"], (8, 2048)),
            ("ids", rvq.io["output_ids"], (8,)))
        for name, address, shape in addresses:
            observed[name].append(_fp32(backend, address, shape))
        if (_fp32(backend, encoder.io["frame"], (1,))[0] != frame + 1
                or _fp32(backend, transformer.io["state"], (1,))[0] != frame + 1
                or _fp32(backend, downsample.io["frame"], (1,))[0] != frame + 1
                or _fp32(backend, downsample.io["first"], (1,))[0] != 0
                or _fp32(backend, rvq.io["counter"], (1,))[0] != 8):
            raise AssertionError("Resident input-path streaming counter or first flag differs")
        transfers.append(dict(frame=frame, raw_pcm_write_bytes=pcm.nbytes,
                              interstage=count, programs=33))
        receipt["frame_progress"].append(dict(pass_name=label, **transfers[-1]))
        save()
    state = dict(
        encoder=[np.asarray(backend.read(0, address, size), np.uint8).copy()
                 for address, size in encoder.reset_regions],
        transformer_kv=[np.asarray(backend.read(0, address, size), np.uint8).copy()
                        for address, size in transformer.cache_regions],
        transformer_counter=np.asarray(backend.read(0, transformer.io["state"], 4), np.uint8).copy(),
        downsample=[np.asarray(backend.read(0, address, size), np.uint8).copy()
                    for address, size in downsample.reset_regions],
        input_rvq=[np.asarray(backend.read(0, address, size), np.uint8).copy()
                   for address, size in rvq.reset_regions])
    return dict(**{name: np.stack(values) for name, values in observed.items()},
                state=state, transfers=transfers)


def _branch_errors(device, native):
    projections = np.stack((device["semantic_projected"], device["acoustic_projected"]), axis=1)
    projected = native["projected"]
    residual_before = native["residual_before"]
    residual_after = np.subtract(residual_before, native["selected_centroids"])
    return dict(
        projected_all=_metrics_by_frame(projections, projected),
        projected_semantic=_metrics_by_frame(projections[:, 0], projected[:, 0]),
        projected_acoustic=_metrics_by_frame(projections[:, 1], projected[:, 1]),
        residual_before_all=_metrics_by_frame(device["residual_before"], residual_before),
        residual_before_semantic=_metrics_by_frame(device["residual_before"][:, 0], residual_before[:, 0]),
        residual_before_acoustic=_metrics_by_frame(device["residual_before"][:, 1:], residual_before[:, 1:]),
        residual_after_all=_metrics_by_frame(device["residual_after"], residual_after),
        residual_after_semantic=_metrics_by_frame(device["residual_after"][:, 0], residual_after[:, 0]),
        residual_after_acoustic=_metrics_by_frame(device["residual_after"][:, 1:], residual_after[:, 1:]))


def require_input_rvq_acceptance(result):
    failed = [name for name in ("accumulated_ids", "same_isa_downsample_ids")
              if not result[name]["passed"]]
    for path in ("accumulated_float_acceptance", "same_input_float_acceptance"):
        failed += [path + "." + name for name, value in result[path].items()
                   if not value["passed"]]
    if failed:
        raise RuntimeError("Input RVQ missed declared exact-ID/float acceptance: " + ", ".join(failed))


def _source_hashes():
    names = ["tools/personaplex_mimi_input_rvq_check.py",
             "tools/personaplex_mimi_input_rvq_native.py",
             "tools/personaplex_mimi_full_input_native.py",
             "tools/personaplex_mimi_input_downsample_check.py",
             "tools/personaplex_mimi_input_downsample_native.py",
             "tools/personaplex_mimi_input_transformer_check.py",
             "tools/personaplex_mimi_input_transformer_native.py",
             "tools/personaplex_mimi_encoder_check.py",
             "tools/personaplex_mimi_encoder_native.py",
             "opentpu/llm/personaplex_mimi_input_rvq.py",
             "opentpu/llm/personaplex_mimi_input_downsample.py",
             "opentpu/llm/personaplex_mimi_input_transformer.py",
             "opentpu/llm/personaplex_mimi_encoder.py",
             "opentpu/llm/personaplex_mimi.py",
             "opentpu/llm/personaplex_mimi_decoder.py",
             "opentpu/llm/personaplex_mimi_pcm.py",
             "opentpu/llm/personaplex_depth_chain.py",
             "tools/personaplex_pcm_precision_check.py",
             *[f"tools/data/personaplex/mimi_{name}_contract.json" for name in
               ("input_rvq", "input_downsample", "input_transformer", "encoder", "rvq", "decoder", "pcm")],
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
    input_rvq_contract = json.loads((DATA / "mimi_input_rvq_contract.json").read_text())
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
    receipt_path = output_dir / "mimi_input_rvq_check.json"
    result = dict(status="running", stage="layout", started_utc=datetime.now(timezone.utc).isoformat(),
        backend="isa", hardware_used=False, hardware_performance_measured=False,
        quality_accepted=False, full_speech_enabled=False, full_model_inference=False,
        checkpoint_payload_hashes_verified=False, frames=2, arrays={}, frame_progress=[],
        scope="Preserved raw PCM -> resident encoder -> FP32 input transformer -> learned downsample -> input RVQ",
        pcm_semantics="raw mono FP32 [2,1920]; no normalization, clipping, scaling, or resampling",
        performance_scope="CPU/ISA wall time and mapped budget only; no physical performance claim",
        runtime_revision=common["runtime_revision"], image_context=common["image"],
        execution_target=selected.to_dict(), checkpoint_revision=baseline["checkpoint_revision"],
        baseline_sha256=baseline_hashes, source_files=sources, codec_file=codec_before,
        native_source_sha256=dict(encoder=encoder_contract["installed_source_sha256"],
            input_transformer=tf_contract["installed_source_sha256"],
            input_downsample=down_contract["installed_source_sha256"],
            input_rvq=input_rvq_contract["installed_source_sha256"]),
        acceptance_limits=INPUT_RVQ_LIMITS,
        distance_policy="Device direct FP32 squared distance; native torch.cdist Euclidean, diagnostic only")

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
        rvq = MimiInputRVQImage.after_downsample(downsample, MimiInputRVQSpec())
        if (rvq.spec.hidden, rvq.spec.dim, rvq.spec.vocab, rvq.spec.codebooks) != (512, 256, 2048, 8):
            raise ValueError("Pinned input RVQ device dimensions changed")
        result["budget"] = _joint_budget(rvq, cfg)
        result["program_words"] = dict(
            encoder=[len(program) * 8 for program in encoder.compile()],
            input_transformer=[len(program) * 8 for program in transformer.compile()],
            input_downsample=[len(program) * 8 for program in downsample.compile()],
            input_rvq=[len(program) * 8 for program in rvq.compile()])
        result.update(stage="native_materialization"); save()

        import torch
        torch.set_num_threads(1)
        native_encoder, encoder_weights = native_encoder_module(codec_path, encoder_contract)
        native_tf, layers, rope = native_encoder_transformer(codec_path, tf_spec, tf_contract)
        native_down, down_weight = native_input_downsample(codec_path, downsample.spec, down_contract)
        native_q, tables, input_projections, native_rvq_root = native_input_quantizer(
            codec_path, input_rvq_contract)
        if (tables.dtype != np.float32 or tables.shape != (8, 2048, 256)
                or not tables.flags.c_contiguous or not np.isfinite(tables).all()
                or input_projections.shape != (2, 256, 512)):
            raise ValueError("Native input RVQ tables or projections differ")
        result["consumed_payload_sha256"] = dict(
            encoder=payload_sha256((f"model{row['model_index']}.{row['subpath']}.{name}", row[name])
                for row in encoder_weights for name in ("weight", "bias")),
            input_transformer_layers=payload_sha256((f"layer{index}.{name}", value)
                for index, layer in enumerate(layers) for name, value in layer.items()),
            input_transformer_rope=payload_sha256((("cos", rope[0]), ("sin", rope[1]))),
            input_downsample=payload_sha256((("downsample.conv.conv.conv.weight", down_weight),)),
            input_rvq_tables=payload_sha256((("normalized_native_tables", tables),)),
            input_rvq_projections=payload_sha256((("semantic", input_projections[0]),
                                                  ("acoustic", input_projections[1]))))
        result.update(stage="device_upload"); save()
        backend = G.IsaBackend(cfg, [np.zeros(cfg.DRAM_BYTES, np.uint8)], adopt=True)
        encoder.upload(backend, encoder_weights)
        transformer.upload(backend, layers, rope)
        downsample.upload(backend, down_weight)
        # The output codec's immutable table span is the sole table owner.
        backend.write(0, rvq.tables_at, tables.view(np.uint8).reshape(-1))
        rvq.upload(backend, input_projections)
        result["budget"] = _joint_budget(rvq, cfg)
        save()

        first = _device_stream(encoder, transformer, downsample, rvq, backend, inputs,
                               result, save, "resident_stream")
        for name in _OBSERVED:
            _save_array(output_dir, result, "isa_input_" + name, first[name])
        for name in ("encoder", "transformer_kv", "downsample", "input_rvq"):
            for index, value in enumerate(first["state"][name]):
                _save_array(output_dir, result, f"isa_input_{name}_state{index}", value)
        _save_array(output_dir, result, "isa_input_transformer_counter",
                    first["state"]["transformer_counter"])
        result["first_pass_arrays_preserved"] = True
        save()
        exact_bytes(first["input_snapshot"], first["encoder_output"],
                    "resident transformer input snapshot")
        device_ids = _checked_ids(first["ids"])
        for row in range(8):
            chosen = tables[row, device_ids[:, row]]
            exact_bytes(first["selected_centroids"][:, row], chosen,
                        f"resident selected centroid table row {row}")
        exact_bytes(first["semantic_residual"], first["residual_after"][:, 0],
                    "semantic final residual")
        exact_bytes(first["acoustic_residual"], first["residual_after"][:, -1],
                    "acoustic final residual")
        exact_bytes(first["residual_before"][:, 0], first["semantic_projected"],
                    "semantic projected search input")
        exact_bytes(first["residual_before"][:, 1], first["acoustic_projected"],
                    "acoustic projected search input")
        for row in range(2, 8):
            exact_bytes(first["residual_before"][:, row], first["residual_after"][:, row - 1],
                        f"acoustic resident residual link {row}")

        encoder.reset(backend)
        transformer.reset(backend)
        downsample.reset(backend)
        rvq.reset(backend)
        replay = _device_stream(encoder, transformer, downsample, rvq, backend, inputs,
                                result, save, "reset_stream")
        for name in _OBSERVED:
            exact_bytes(replay[name], first[name], "input RVQ path reset " + name)
        for name in ("encoder", "transformer_kv", "downsample", "input_rvq"):
            if len(replay["state"][name]) != len(first["state"][name]):
                raise AssertionError("Input RVQ path raw state region count changed")
            for index, (left, right) in enumerate(zip(first["state"][name], replay["state"][name])):
                exact_bytes(right, left, f"input RVQ path {name} raw state {index} reset")
        exact_bytes(replay["state"]["transformer_counter"],
                    first["state"]["transformer_counter"], "input transformer counter reset")
        result.update(reset_bit_exact=True, final_raw_state_reset_bit_exact=True,
                      input_snapshot_bit_exact=True, stage="native_comparison"); save()

        source_encoder = collect_native_encoder(native_encoder, inputs)
        source_tf = collect_native_encoder_transformer(native_tf, source_encoder["output"])
        _check_native_positions(source_tf, tf_spec.context, 2)
        source_down = collect_native_input_downsample(native_down, source_tf["output"])
        source_full = collect_native_input_quantizer(native_q, source_down)
        native_positions = np.stack([np.stack(row) for row in source_tf["positions"]])
        original_input = check_native_full_input(native_encoder, native_tf, native_down,
            native_q, inputs, dict(encoder=source_encoder["output"],
                transformer=source_tf["output"], latents=source_down,
                ids=source_full["ids"], positions=native_positions))
        qualification = dict(original_input["qualification"])
        qualification["graph_wrapper"] = original_input["graph_wrapper"]
        qualification["passed"] = all(qualification[name] for name in (
            "component_boundaries_bit_exact", "input_ids_bit_exact",
            "native_positions_bit_exact", "logical_reset_bit_exact",
            "fresh_stream_bit_exact"))
        result["original_native_input_entrypoint"] = qualification
        _save_array(output_dir, result, "original_native_input_ids", original_input["ids"])
        _save_array(output_dir, result, "original_native_input_latents", original_input["latents"])
        save()
        if not qualification["passed"]:
            raise AssertionError("Original MimiModel.encode input qualification did not pass")
        source_again = collect_native_input_quantizer(native_q, source_down)
        for name in source_full:
            exact_bytes(source_again[name], source_full[name], "native input RVQ fresh " + name)
        same_input = collect_native_input_quantizer(native_q, first["downsample_output"])
        result["native_fresh_stream_bit_exact"] = True
        result["encoder_input_error"] = _metrics_by_frame(
            first["encoder_output"], source_encoder["output"])
        result["transformer_accumulated_error"] = _metrics_by_frame(
            first["transformer_output"], source_tf["output"])
        result["downsample_accumulated_error"] = _metrics_by_frame(
            first["downsample_output"], source_down)
        result["accumulated_ids"] = _id_agreement(device_ids, source_full["ids"], "native raw PCM path")
        result["same_isa_downsample_ids"] = _id_agreement(
            device_ids, same_input["ids"], "native on identical resident downsample latents")
        result["accumulated_float_error"] = _branch_errors(first, source_full)
        result["same_input_float_error"] = _branch_errors(first, same_input)
        result["same_input_float_acceptance"] = {
            name: numerical_acceptance(metrics, INPUT_RVQ_LIMITS)
            for name, metrics in result["same_input_float_error"].items()}
        result["accumulated_float_acceptance"] = {
            name: numerical_acceptance(metrics, INPUT_RVQ_LIMITS)
            for name, metrics in result["accumulated_float_error"].items()}
        result["distance_diagnostics"] = dict(
            accumulated=_distance_diagnostics(first["distances"], source_full["native_distances"],
                                               source_full["winner_margin"]),
            same_input=_distance_diagnostics(first["distances"], same_input["native_distances"],
                                             same_input["winner_margin"]))
        result["same_input_selected_centroids_bit_exact"] = bool(np.array_equal(
            first["selected_centroids"], same_input["selected_centroids"]))
        result["native_residual_after_definition"] = (
            "Offline FP32 residual_before minus selected_centroids; native helper captures both operands")
        _save_array(output_dir, result, "native_input_rvq_accumulated_ids", source_full["ids"])
        _save_array(output_dir, result, "native_input_rvq_same_input_ids", same_input["ids"])
        for label, reference in (("accumulated", source_full), ("same_input", same_input)):
            for name in ("projected", "residual_before", "selected_centroids",
                         "native_distances", "winner_margin"):
                _save_array(output_dir, result, f"native_input_rvq_{label}_{name}", reference[name])
        save()

        for name, digest in baseline_hashes.items():
            if sha256(name) != digest:
                raise RuntimeError("Preserved baseline changed: " + name)
        for name, digest in sources.items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError("Input RVQ check source changed: " + name)
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
        for name, digest in input_rvq_contract["installed_source_sha256"].items():
            if sha256(native_rvq_root / name) != digest:
                raise RuntimeError("Installed native input RVQ source changed: " + name)
        if codec_header(codec_path, rvq_contract) != codec_before:
            raise RuntimeError("Cached codec header/file metadata changed")
        result["provenance_rechecked_after_execution"] = True
        save()
        require_input_rvq_acceptance(result)
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
                      ("status", "frames", "accumulated_ids", "same_isa_downsample_ids",
                       "cpu_wall_seconds")}, indent=2))
