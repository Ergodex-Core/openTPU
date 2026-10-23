#!/usr/bin/env python3
"""Explicit, offline, CPU/ISA diagnostic using verified official tensor data.

No network, model-code import, pickle, device transport, or serving integration.
Run only on the authorized EC2 workflow: full FP32 tables require several GiB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from opentpu import isa as I
from opentpu.isasim import Config, Machine
from opentpu.llm.qwen3_tts import MODEL_ID, MODEL_REVISION, SOURCE_REVISION
from opentpu.llm.qwen3_tts_talker import TalkerBlockImage, TalkerSpec
from opentpu.target import load
from qwen3_tts_talker_reference import Reference


HEADER_SHA256 = "b125649e98d52f4ac605666a212eee68c6f4f5faf9b9a5e7b930565b578f96a1"
FILE_SHA256 = "bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb"
FILE_BYTES = 1811626576
STAGES = ("projected", "assembled", "attention_residual", "output")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class BF16Checkpoint:
    """Data-only reader; verifies the complete LFS digest before tensor access."""
    def __init__(self, path, manifest):
        self.path = Path(path)
        if (manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION
                or manifest.get("source_revision") != SOURCE_REVISION
                or manifest.get("verified_lfs_sha256") is not True
                or manifest.get("file_sha256") != FILE_SHA256
                or manifest.get("file_bytes") != FILE_BYTES):
            raise ValueError("expected verified exact-model manifest")
        if self.path.stat().st_size != manifest["file_bytes"]:
            raise ValueError("checkpoint file size mismatch")
        if sha256(self.path) != manifest["file_sha256"]:
            raise ValueError("checkpoint LFS checksum mismatch")
        with self.path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safetensors prefix")
            n, = struct.unpack("<Q", prefix)
            if n != 49224:
                raise ValueError("unexpected official safetensors header length")
            header = f.read(n)
        if hashlib.sha256(header).hexdigest() != HEADER_SHA256:
            raise ValueError("pinned safetensors header mismatch")
        self.header, self.offset = json.loads(header), n + 8

    def tensor(self, name, shape):
        entry = self.header[name]
        a, b = entry["data_offsets"]
        count = math.prod(shape)
        if (entry["dtype"] != "BF16" or tuple(entry["shape"]) != tuple(shape)
                or b - a != 2 * count or a < 0 or self.offset + b > self.path.stat().st_size):
            raise ValueError(f"invalid checkpoint tensor: {name}")
        raw = np.memmap(self.path, dtype="<u2", mode="r", offset=self.offset + a, shape=count)
        # BF16 -> FP32 is an exact bit extension, without a BF16 numerical plugin.
        return (np.asarray(raw, dtype=np.uint32) << 16).view(np.float32).reshape(shape)


def metrics(actual, reference):
    a, b = np.asarray(actual, np.float64).reshape(-1), np.asarray(reference, np.float64).reshape(-1)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite diagnostic output")
    delta = a - b
    return {"relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30)),
            "max_absolute": float(np.max(np.abs(delta))),
            "rms_error": float(np.sqrt(np.mean(delta * delta))),
            "reference_rms": float(np.sqrt(np.mean(b * b))),
            "cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))}


def read(machine, image, name):
    t = getattr(image.desc, name)
    n = 4 * math.prod(t.shape)
    return machine.slices[0].dram[t.base.const:t.base.const + n].view(np.float32).copy().reshape(-1)


def step(machine, image, position, pair):
    encoded = image.token_bytes(*pair)
    program, budget = image.compile(position)
    address = image.desc.ids.base.const
    machine.slices[0].dram[address:address + 8] = encoded
    machine.load(program).run()
    return {stage: read(machine, image, stage) for stage in STAGES}, budget


def reset_cache(machine, image):
    total = 0
    for name, (address, size) in image.regions.items():
        if name.startswith("kv."):
            machine.slices[0].dram[address:address + size] = 0
            total += size
    return total


def summarize(rows, reference):
    result = {}
    for stage in STAGES:
        samples = [(r["position"], r[reference][stage]) for r in rows if reference in r]
        if not samples:
            continue
        errors = np.asarray([m["relative_l2"] for _, m in samples])
        worst = max(samples, key=lambda pair: pair[1]["relative_l2"])
        result[stage] = {"samples": len(samples), "mean_relative_l2": float(errors.mean()),
                         "p95_relative_l2": float(np.percentile(errors, 95)),
                         "max_relative_l2": float(errors.max()), "worst_position": worst[0],
                         "min_cosine": min(m["cosine"] for _, m in samples),
                         "max_absolute": max(m["max_absolute"] for _, m in samples),
                         "samples_above_0_035": int(np.count_nonzero(errors >= 0.035))}
    return result


def run(args):
    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    config = json.loads((ROOT / "tests/fixtures/qwen3_tts_06b_config.json").read_text())
    reader = BF16Checkpoint(args.checkpoint, manifest)
    spec = TalkerSpec.from_config(config)
    arch = {k: v for k, v in load("f2").arch.items() if k in Config.__dataclass_fields__}
    cfg = Config(**arch)
    image = TalkerBlockImage(spec, cfg, capacity=512)
    weights = {name: reader.tensor(name, shape) for name, (_, _, shape) in image.parameters.items()}
    reference = Reference(spec, weights)
    bf16 = None
    if args.bf16_reference:
        from qwen3_tts_talker_bf16_reference import BF16Reference
        bf16 = BF16Reference(spec, weights)
    # This explicit budget is for authorized offline EC2 diagnostics only.
    packed = image.build(weights, max_bytes=2 << 30)
    del weights
    # Same f2 ISA dimensions; simulator DRAM allocation is bounded to this image.
    from dataclasses import replace
    sim_cfg = replace(cfg, DRAM_BYTES=image.nbytes)
    normal = Machine(sim_cfg, [[I.halt()]], [packed])
    poisoned = Machine(sim_cfg, [[I.halt()]], [packed])
    for name, (address, size) in image.regions.items():
        if name.startswith("kv."):
            poisoned.slices[0].dram[address:address + size] = 0x3f
    control = config["talker_config"]
    fixed = [(config["tts_bos_token_id"], control["codec_bos_id"]),
             (config["tts_pad_token_id"], control["spk_id"]["ryan"]),
             (config["tts_pad_token_id"], control["codec_language_id"]["english"]),
             (config["tts_pad_token_id"], control["codec_nothink_id"]),
             (151644, control["codec_pad_id"]), (0, 0),
             (spec.text_vocab - 1, spec.codec_vocab - 1),
             (config["tts_eos_token_id"], control["codec_eos_token_id"])]
    fixed += [(config["tts_pad_token_id"], value) for value in control["spk_id"].values()]
    rng = np.random.default_rng(2306)
    pairs = fixed + [(int(rng.integers(spec.text_vocab)), int(rng.integers(spec.codec_vocab)))
                     for _ in range(args.steps - len(fixed))]
    rows, budgets, mask_mismatches = [], [], []
    for position, pair in enumerate(pairs):
        expected = reference.step(*pair)
        actual, budget = step(normal, image, position, pair)
        masked, _ = step(poisoned, image, position, pair)
        if any(not np.array_equal(actual[k].view(np.uint32), masked[k].view(np.uint32)) for k in STAGES):
            mask_mismatches.append(position)
        row = {"position": position, "ids": list(pair),
               "fp64": {k: metrics(actual[k], expected[k]) for k in STAGES}}
        if bf16 is not None:
            expected_bf16 = bf16.step(*pair)
            row["bf16_eager"] = {k: metrics(actual[k], expected_bf16[k]) for k in STAGES}
            row["bf16_vs_fp64"] = {k: metrics(expected_bf16[k], expected[k]) for k in STAGES}
        rows.append(row)
        budgets.append(budget)
        if position % 32 == 0 or position + 1 == len(pairs):
            print(json.dumps({"progress": position + 1, "total": len(pairs),
                              "output_relative_l2": row["fp64"]["output"]["relative_l2"]}), flush=True)
    last = read(normal, image, "output")
    # An explicit counterfactual cache ablation at the same position; no extra model layer.
    reset_cache(poisoned, image)
    ablated, _ = step(poisoned, image, len(pairs) - 1, pairs[-1])
    cache_ablation = metrics(ablated["output"], last)
    del poisoned
    fresh = Machine(sim_cfg, [[I.halt()]], [packed])
    cleared = reset_cache(normal, image)
    reference.keys, reference.values = [], []
    reset_mismatches, reset_errors = [], []
    for position, pair in enumerate(pairs[:8]):
        actual, _ = step(normal, image, position, pair)
        pristine, _ = step(fresh, image, position, pair)
        expected = reference.step(*pair)
        if any(not np.array_equal(actual[k].view(np.uint32), pristine[k].view(np.uint32)) for k in STAGES):
            reset_mismatches.append(position)
        reset_errors.append(metrics(actual["output"], expected["output"]))
    result = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "source_revision": SOURCE_REVISION, "license": "Apache-2.0",
              "checkpoint_manifest": manifest, "checkpoint_manifest_sha256": sha256(args.manifest),
              "parameter_tensors": len(image.parameters),
              "parameter_elements": sum(math.prod(shape) for _, _, shape in image.parameters.values()),
              "input_semantics": "Deterministic aligned row probes, not a complete CustomVoice prompt",
              "sequence_seed": 2306, "positions": len(pairs), "cache_capacity": image.capacity,
              "full_embedding_tables_resident": True, "dram_bytes": image.nbytes,
              "reference_fp64": "Independent source-derived FP64 algebra on exact decoded BF16 tensors",
              "reference_bf16": "Locally reviewed eager BF16 CPU algebra; no upstream model import" if bf16 else None,
              "metrics_fp64": summarize(rows, "fp64"),
              "metrics_bf16_eager": summarize(rows, "bf16_eager"),
              "bf16_eager_vs_fp64": summarize(rows, "bf16_vs_fp64"),
              "inherited_diagnostic_threshold": 0.035,
              "threshold_is_speech_acceptance": False,
              "future_cache_bitwise_mismatch_positions": mask_mismatches,
              "cache_ablation_at_last_position": cache_ablation,
              "reset": {"cleared_kv_bytes": cleared, "positions": 8,
                        "bitwise_mismatch_positions": reset_mismatches,
                        "max_output_relative_l2_vs_fp64": max(r["relative_l2"] for r in reset_errors),
                        "scope": "Host-cleared layer-0 KV, position restarted at zero; no production reset API claim"},
              "greedy": {"logits_computed": False, "parity_established": False,
                         "reason": "One layer has no codec logits; remaining 27 layers/final norm/head and generation loop are absent. Hidden error cannot establish greedy parity."},
              "max_measured_imem_words": max(b["imem_words"] for b in budgets),
              "max_measured_tmem_words": max(b["tmem_peak_words"] for b in budgets),
              "seconds": time.monotonic() - started,
              "python": platform.python_version(), "numpy": np.__version__,
              "upstream_code_executed": False, "hardware_access": False,
              "synthesis": False, "tts_enabled": False,
              "selected_positions": [r for r in rows if r["position"] in {0, 1, 6, 16, 127, 128, 254, 255, 256, 257, 259}],
              "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                  [Path(__file__), ROOT / "tests/qwen3_tts_talker_reference.py",
                   ROOT / "tests/qwen3_tts_talker_bf16_reference.py",
                   ROOT / "opentpu/llm/qwen3_tts_talker.py", ROOT / "opentpu/kernels/tts_talker.py"]}}
    if bf16 is not None:
        import torch
        result["torch"] = torch.__version__
    result["diagnostic_numerical_threshold_passed"] = all(
        m["samples_above_0_035"] == 0 for group in
        (result["metrics_fp64"], result["metrics_bf16_eager"]) for m in group.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    details = args.output.with_suffix(".positions.json")
    details.write_text(json.dumps(rows, allow_nan=False) + "\n")
    result["positions_file_sha256"] = sha256(details)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "selected_positions"}, indent=2), flush=True)
    if mask_mismatches or reset_mismatches:
        return 1
    return 0 if result["diagnostic_numerical_threshold_passed"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--bf16-reference", action="store_true")
    args = parser.parse_args()
    if not 260 <= args.steps <= 512:
        parser.error("steps must be 260..512 to cover the native 256-token boundary")
    raise SystemExit(run(args))
