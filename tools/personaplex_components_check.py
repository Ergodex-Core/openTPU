#!/usr/bin/env python3
"""Bounded EC2 CPU/ISA validation of embeddings and audio heads, not speech.

No physical backend, network download, temporal resimulation, or model fallback.
Source PyTorch math is used only as an independent validation reference.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
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
from opentpu.llm.personaplex_weights import MLXWeights
from opentpu.llm.personaplex_embeddings import EmbeddingEngine, EmbeddingImage, EmbeddingSpec
from opentpu.llm.personaplex_audio_heads import AudioHeadEngine, AudioHeadImage
from tools.personaplex_isa_check import load_array, precision_metrics, PINNED_REVISION, PROFILE
os.environ["OTPU_BUILD_JOBS"] = "1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def require_exact(actual, expected, context):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.dtype != np.float32 or expected.dtype != np.float32
            or actual.shape != expected.shape or not np.isfinite(actual).all()
            or not np.isfinite(expected).all()
            or not np.array_equal(actual.view(np.uint32), expected.view(np.uint32))):
        raise AssertionError(f"{context}: FP32 bits differ from source reference")


def run(model_dir, voice_file, output_dir):
    import torch
    from moshi.models.lm import LMModel, ScaledEmbedding
    torch.set_num_threads(1)
    model_dir, voice_file, output_dir = (Path(p).resolve() for p in (model_dir, voice_file, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    if (output_dir == model_dir or output_dir in model_dir.parents
            or model_dir in output_dir.parents or output_dir == ROOT
            or output_dir in ROOT.parents or ROOT in output_dir.parents
            or output_dir == voice_file or output_dir in voice_file.parents):
        raise ValueError("Output must be isolated from checkpoint, source and inputs")
    weights = MLXWeights(model_dir)
    files = {name: metadata(model_dir / name) for name in ("config.json", "embeddings.safetensors")}
    for name in files:
        revision_file = model_dir / ".cache/huggingface/download" / (name + ".metadata")
        with revision_file.open() as handle:
            if handle.readline().strip() != PINNED_REVISION:
                raise ValueError(f"Checkpoint revision mismatch for {name}")
    voices = load_array(voice_file, dtype="float32")
    if voices.shape != (4, 4096):
        raise ValueError("Expected the saved four-frame FP32 voice capture")
    spec = EmbeddingSpec(voice_frames=4)
    if (weights.spec.hidden != spec.hidden or len(weights.config["delays"]) != 17
            or weights.config["depformer"]["num_steps"] != 16):
        raise ValueError("Expected the pinned production PersonaPlex component dimensions")
    embedding_metadata = weights.input_embedding_metadata()
    head_metadata = weights.audio_head_metadata()
    cfg = target.config(target.load(str(PROFILE)))
    embedding_layout, audio_layout = EmbeddingImage(spec, cfg), AudioHeadImage(cfg)
    # Compile both default layouts before any large payload read.
    embedding_programs, audio_program = embedding_layout.compile(), audio_layout.compile()
    source = Path(inspect.getfile(LMModel))
    voice_sha = sha256(voice_file)
    receipt = {"status": "running", "stage": "building_embeddings",
               "started_utc": datetime.now(timezone.utc).isoformat(), "backend": "isa",
               "hardware_used": False, "hardware_performance_measured": False,
               "full_fpga_model_inference": False, "speech_generated": False, "quality_accepted": False,
               "sampling_mode": "greedy", "native_stochastic_mode_supported": False,
               "scope": "Separate resident input-embedding/voice and INT8 audio-head component programs",
               "checkpoint_revision": PINNED_REVISION, "checkpoint_files": files,
               "payload_identity": "Revision metadata and size/mtime; not exhaustive checkpoint payload hashing",
               "native_lm_source_sha256": sha256(source), "voice_input_sha256": voice_sha,
               "embedding_spec": asdict(spec), "embedding_metadata": embedding_metadata,
               "head_metadata": head_metadata, "embedding_required_bytes": embedding_layout.required_bytes,
               "embedding_table_bytes": embedding_layout.table_bytes,
               "voice_bytes": embedding_layout.voice_bytes,
               "embedding_program_words": {k: len(v[0]) * 8 for k, v in embedding_programs.items()},
               "audio_required_bytes": audio_layout.required_bytes, "audio_bank_bytes": audio_layout.head_bytes,
               "audio_program_words": len(audio_program[0]) * 8, "dram_bytes": cfg.DRAM_BYTES,
               "imem_words": cfg.IMEM_WORDS, "embedding_cases": [], "voice_cases": [], "audio_cases": [],
               "remaining_components": ["Depth input projections and per-codebook embeddings",
                   "Six-layer depth transformer and per-step state", "Mimi encoder/decoder and RVQ",
                   "Prompt cache/delay/stream scheduler", "Unified device residency and component chaining"],
               "source_files": {str(p.relative_to(ROOT)): sha256(p) for p in
                   [Path(__file__), ROOT / "opentpu/llm/personaplex_weights.py",
                    ROOT / "opentpu/llm/personaplex_embeddings.py", ROOT / "opentpu/llm/personaplex_audio_heads.py"]}}
    output_dir.mkdir(parents=True, exist_ok=False)
    result_file = output_dir / "components_check.json"
    def save():
        temp = result_file.with_suffix(".json.tmp")
        temp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temp.replace(result_file)
    def interrupted(signum, frame):
        raise InterruptedError("Component diagnostic interrupted; inspect receipt before retry")
    previous = signal.signal(signal.SIGTERM, interrupted)
    started, engine = time.perf_counter(), None
    save()
    try:
        build_start = time.perf_counter()
        tables = weights.input_embeddings
        modules = []
        for table in tables:
            module = ScaledEmbedding(len(table), spec.hidden, norm=False, zero_idx=-1)
            module.weight = torch.nn.Parameter(torch.from_numpy(table), requires_grad=False)
            modules.append(module)
        native = SimpleNamespace(num_codebooks=17, num_audio_codebooks=16, audio_offset=1,
                                 emb=modules[1:], text_emb=modules[0])
        engine = EmbeddingEngine(tables, voices=voices, spec=spec, cfg=cfg)
        receipt["embedding_build_cpu_seconds"] = time.perf_counter() - build_start
        codes = [[32000] + [2048] * 16, [0] + list(range(16)),
                 [128] + [((i * 127 + 23) % 2048) for i in range(16)], [-1] * 17,
                 [-1] + [(-1 if i % 2 else i) for i in range(16)]]
        outputs = []
        for index, ids in enumerate(codes):
            receipt.update(stage="checking_embeddings", case=index)
            save()
            with torch.no_grad():
                expected = LMModel.embed_codes(native, torch.tensor(ids).reshape(1, 17, 1)).numpy().reshape(-1)
            engine.step(ids)
            actual = engine.read_output()
            require_exact(actual, expected, f"embedding case{index}")
            outputs.append(actual)
            receipt["embedding_cases"].append({"ids": ids, "source_fp32_bit_exact": True, "stats": engine.last_stats})
            save()
        voice_outputs = []
        for index in (3, 0, 2, 1):
            engine.step_voice(index)
            actual = engine.read_output()
            require_exact(actual, voices[index], f"voice case{index}")
            voice_outputs.append(actual)
            receipt["voice_cases"].append({"index": index, "source_fp32_bit_exact": True, "stats": engine.last_stats})
        engine.reset()
        engine.step(codes[0]); require_exact(engine.read_output(), outputs[0], "embedding reset")
        engine.step_voice(3); require_exact(engine.read_output(), voice_outputs[0], "voice reset")
        engine.close(); engine = None
        del native, modules, tables
        receipt.update(stage="building_audio_heads", embedding_reset_bit_exact=True)
        save()
        build_start = time.perf_counter()
        bank = weights.audio_heads
        engine = AudioHeadEngine(bank, cfg=cfg)
        receipt["audio_build_cpu_seconds"] = time.perf_counter() - build_start
        # Explicitly synthetic normalized depth-width inputs, not generated by a
        # depth transformer. This validates only the selected projection stage.
        rng = np.random.default_rng(237)
        hidden = rng.normal(size=(16, 1024)).astype(np.float32)
        hidden /= np.sqrt(np.mean(hidden * hidden, axis=1, keepdims=True, dtype=np.float32))
        tokens, logits, expected_logits = [], [], []
        order = (15, 0, 8, 3, 14, 1, 7, 12, 2, 9, 4, 13, 5, 11, 6, 10)
        receipt["audio_output_codebooks"] = list(order)
        for index in order:
            receipt.update(stage="checking_audio_heads", codebook=index)
            save()
            token = engine.step(hidden[index], index)
            scores = engine.read_logits()
            if token != int(np.argmax(scores)):
                raise AssertionError(f"Audio codebook{index}: device greedy differs from device logits")
            with torch.no_grad():
                expected = torch.nn.functional.linear(torch.from_numpy(hidden[index]), torch.from_numpy(bank[index])).numpy()
            tokens.append(token); logits.append(scores); expected_logits.append(expected)
            receipt["audio_cases"].append({"codebook": index, "token_id": token,
                "source_fp32_top1_id": int(np.argmax(expected)),
                "source_fp32_top1_matches": token == int(np.argmax(expected)),
                "device_greedy_matches_device_logits": True,
                "logit_precision_delta": precision_metrics(scores[None], expected[None]),
                "stats": engine.last_stats})
            save()
        engine.reset()
        repeated = engine.step(hidden[15], 15)
        require_exact(engine.read_logits(), logits[0], "audio reset")
        if repeated != tokens[0]:
            raise AssertionError("Audio reset token changed")
        receipt.update(audio_reset_bit_exact=True,
                       audio_reference_inputs="Synthetic normalized 1024-wide vectors; depth transformer not executed",
                       all_audio_source_top1_match=all(c["source_fp32_top1_matches"] for c in receipt["audio_cases"]),
                       audio_aggregate_precision=precision_metrics(np.stack(logits), np.stack(expected_logits)))
        for name, value in (("embedding_outputs", np.stack(outputs)), ("voice_outputs", np.stack(voice_outputs)),
                            ("audio_inputs", hidden[list(order)]), ("audio_tokens", np.asarray(tokens, np.int64)),
                            ("audio_logits", np.stack(logits)), ("source_audio_logits", np.stack(expected_logits))):
            path = output_dir / (name + ".npy")
            with path.open("xb") as handle:
                np.save(handle, value, allow_pickle=False)
            receipt[name] = {"sha256": sha256(path), "shape": list(value.shape), "dtype": str(value.dtype)}
        if (sha256(voice_file) != voice_sha or sha256(source) != receipt["native_lm_source_sha256"]
                or any(metadata(model_dir / name) != value for name, value in files.items())):
            raise RuntimeError("Checkpoint, native source or input changed during component check")
        receipt.update(status="completed", stage="completed")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        try:
            if engine is not None:
                engine.close()
        finally:
            signal.signal(signal.SIGTERM, previous)
            receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), cpu_wall_seconds=time.perf_counter() - started)
            save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--voice-embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    receipt = run(args.model_dir, args.voice_embeddings, args.output_dir)
    print(json.dumps({key: receipt[key] for key in ("status", "embedding_reset_bit_exact", "audio_reset_bit_exact",
        "all_audio_source_top1_match", "audio_aggregate_precision", "cpu_wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
