#!/usr/bin/env python3
"""Validate real local PersonaPlex host assets on CPU; never run its temporal LM."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

# Set before importing numerical libraries or Hugging Face.
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  OPENBLAS_NUM_THREADS="2", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
for parent in Path(__file__).resolve().parents:
    if (parent / "opentpu/llm/personaplex_bridge.py").is_file():
        sys.path.insert(0, str(parent))
        break


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_info(path):
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def tensor_info(tensor):
    import torch
    require(isinstance(tensor, torch.Tensor), "Expected a tensor")
    require(tensor.device.type == "cpu", "Expected a CPU tensor")
    require(tensor.numel() > 0, "Empty tensor")
    require(bool(torch.isfinite(tensor).all()), "Non-finite tensor")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "finite": True, "min": float(tensor.min()), "max": float(tensor.max())}


def validate(args, report):
    import numpy as np
    import torch
    import sentencepiece
    from opentpu.llm.personaplex_assets import ServiceAssets, offline_hub
    from opentpu.llm.personaplex_bridge import (
        _assign_host_tensors, _meta_host_model, validate_host_checkpoint,
    )
    from opentpu.llm.personaplex_weights import MLXWeights

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(0)
    report["versions"] = {name: importlib.metadata.version(name)
                          for name in ("torch", "numpy", "moshi-personaplex", "sentencepiece", "safetensors")}
    report["threads"] = {"intraop": torch.get_num_threads(),
                         "interop": torch.get_num_interop_threads()}

    def stage(name):
        report["stage"] = name
        print(json.dumps({"stage": name, "status": "running"}), flush=True)

    stage("local_asset_structure")
    assets = ServiceAssets.resolve(args.assets_dir, args.model_dir)
    report["assets"] = {"config": file_info(assets.config),
                        "mimi": file_info(assets.mimi_weight),
                        "tokenizer": file_info(assets.tokenizer),
                        "browser_index": file_info(assets.static / "index.html"),
                        "browser_check": "index exists; browser execution not tested"}
    with offline_hub(assets), torch.inference_mode():
        stage("host_payload_load")
        weights = MLXWeights(args.model_dir)
        schema = validate_host_checkpoint(weights, weights.spec)
        lm = _meta_host_model(weights.spec, torch.float32)
        _assign_host_tensors(lm, weights.iter_host_tensors(), torch.float32)
        lm.eval()
        require(all(p.device.type == "cpu" for p in lm.parameters()),
                "Host parameters were not fully materialized on CPU")
        require(not any(name.startswith(("transformer.", "out_norm."))
                        for name in lm.state_dict()), "Unexpected temporal parameters")
        report["host"] = {"schema": schema, "payload_loaded": True,
                          "strict_assignment": True, "all_loaded_payloads_finite": True,
                          "device": "cpu", "dtype": "torch.float32",
                          "temporal_parameters_loaded": False,
                          "depth_transformer_executed": False, "text_head_executed": False}

        stage("native_initial_embeddings")
        initial = lm._get_initial_token()
        embeddings = lm.embed_codes(initial)
        require(tuple(initial.shape) == (1, lm.num_codebooks, 1), "Initial token shape mismatch")
        require(tuple(embeddings.shape) == (1, 1, weights.spec.hidden), "Embedding shape mismatch")
        require(embeddings.dtype == torch.float32, "Expected FP32 summed embeddings")
        info = tensor_info(embeddings)
        require(torch.equal(embeddings, lm.embed_codes(initial)), "Embedding repeat differs")
        output = args.output_dir / "input_embeddings.npy"
        with output.open("xb") as stream:
            np.save(stream, embeddings[0].numpy(), allow_pickle=False)
        report["input_embeddings"] = {**info, "saved_shape": [1, weights.spec.hidden],
            "path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "source": "native LMModel.embed_codes(LMModel._get_initial_token())",
            "initial_codes": initial.flatten().tolist(), "repeat_bit_exact": True,
            "is_temporal_output": False}
        voice_shape = (1, lm.num_codebooks, max(lm.delays) + 3)
        hidden, text_card, audio_card = weights.spec.hidden, lm.text_card, lm.card
        del lm, weights, initial, embeddings
        gc.collect()

        stage("sentencepiece")
        tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(assets.tokenizer))
        require(tokenizer.get_piece_size() == text_card, "Tokenizer vocabulary mismatch")
        tokens = tokenizer.encode("Hello.", out_type=int)
        require(tokens and all(0 <= token < text_card for token in tokens), "Tokenizer encode failed")
        decoded = tokenizer.decode(tokens)
        require(isinstance(decoded, str) and decoded, "Tokenizer decode failed")
        report["tokenizer"] = {"vocabulary_size": text_card, "sample_tokens": tokens,
                               "sample_decoded": decoded, "loaded": True}

        stage("voice_payloads")
        voices = []
        for path in assets.voices:
            voice = torch.load(path, weights_only=True, map_location="cpu")
            require(isinstance(voice, dict) and {"embeddings", "cache"} <= voice.keys(),
                    f"Missing voice payload fields: {path.name}")
            emb, cache = voice["embeddings"], voice["cache"]
            emb_info, cache_info = tensor_info(emb), tensor_info(cache)
            require(emb.is_floating_point() and emb.ndim == 4
                    and tuple(emb.shape[1:]) == (1, 1, hidden), f"Voice embedding shape: {path.name}")
            require(cache.dtype == torch.long and tuple(cache.shape) == voice_shape,
                    f"Voice cache shape/type: {path.name}")
            require(bool((cache >= -2).all()) and bool((cache[:, 0] <= text_card).all())
                    and bool((cache[:, 1:] <= audio_card).all()), f"Voice token range: {path.name}")
            voices.append({"name": path.name, **file_info(path),
                           "embeddings": emb_info, "cache": cache_info})
            del voice, emb, cache
        report["voices"] = {"count": len(voices), "weights_only": True,
                            "payloads": voices, "replayed_through_temporal_model": False}

        stage("mimi_silence_stream")
        from moshi.models.loaders import get_mimi
        mimi = get_mimi(str(assets.mimi_weight), device="cpu").eval()
        for name, value in mimi.state_dict().items():
            require(value.device.type == "cpu" and bool(torch.isfinite(value).all()),
                    f"Invalid Mimi tensor: {name}")
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        require(frame_size > 0 and frame_size * mimi.frame_rate == mimi.sample_rate,
                "Codec frame size is not integral")
        silence = torch.zeros(1, mimi.channels, frame_size, dtype=torch.float32, device="cpu")
        with mimi.streaming(1):
            codes = mimi.encode(silence)
            require(tuple(codes.shape) == (1, mimi.num_codebooks, 1), "Mimi code shape mismatch")
            require(codes.dtype == torch.long and bool((codes >= 0).all())
                    and bool((codes < mimi.cardinality).all()), "Mimi codes out of range")
            reconstructed = mimi.decode(codes)
        require(tuple(reconstructed.shape) == tuple(silence.shape), "Mimi decoded frame shape mismatch")
        report["mimi"] = {"loaded": True, "all_checkpoint_tensors_finite": True,
                          "device": "cpu", "streaming_frames": 1,
                          "sample_rate": mimi.sample_rate, "frame_rate": mimi.frame_rate,
                          "input": tensor_info(silence), "codes": tensor_info(codes),
                          "decoded": tensor_info(reconstructed), "speech_quality_tested": False}
    report["stage"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.output_dir / "host-assets-validation.json"
    require(not receipt.exists() and not (args.output_dir / "input_embeddings.npy").exists(),
            "Use a fresh output directory; existing results are not overwritten")
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "passed": False,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "stage": "imports", "scope": "real local host payloads and one CPU codec frame",
              "offline": True, "fpga_access": False, "temporal_execution": False,
              "temporal_numerics_validated": False, "hardware_validated": False,
              "end_to_end_inference": False, "speech_quality_validated": False}
    started = time.monotonic()
    try:
        validate(args, report)
        report["passed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["elapsed_seconds"] = time.monotonic() - started
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    with receipt.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "stage": report["stage"],
                      "receipt": str(receipt), "error": report.get("error")}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
