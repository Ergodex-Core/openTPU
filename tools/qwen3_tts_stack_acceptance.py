#!/usr/bin/env python3
"""Offline full-talker checkpoint/ISA diagnostic; no hardware, network or service."""
import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from opentpu.isasim import Config
from opentpu.target import load
from opentpu.llm.qwen3_tts import MODEL_ID, MODEL_REVISION, SOURCE_REVISION
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from opentpu.llm.qwen3_tts_stack import TalkerStackImage, StackISASession
from opentpu.llm.qwen3_tts_shared import shared_tts_config, profile_metadata
from tools.qwen3_tts_checkpoint_acceptance import BF16Checkpoint, metrics, sha256
from qwen3_tts_stack_reference import StackReference


def selection(logits, audio=2048, eos=2150):
    scores = np.asarray(logits, np.float64).copy()
    raw = int(np.argmax(scores))
    scores[audio:eos] = -np.inf
    scores[eos + 1:] = -np.inf
    return raw, int(np.argmax(scores))


def agreement(actual, reference, predicted, audio=2048, eos=2150):
    result = {}
    for index, policy in enumerate(("raw", "fixed_mask")):
        a, r = actual.copy(), reference.copy()
        if index:
            a[audio:eos] = r[audio:eos] = -np.inf
            a[eos + 1:] = r[eos + 1:] = -np.inf
        wanted = int(np.argmax(r))
        top = np.sort(r)[-2:]
        error_bound = float(np.max(np.abs(actual - reference)))
        result[policy] = {"device_id": predicted[index], "reference_id": wanted,
                          "agrees": predicted[index] == wanted,
                          "reference_margin": float(top[-1] - top[-2]),
                          "max_logit_error": error_bound,
                          "margin_gt_twice_error": bool(top[-1] - top[-2] > 2 * error_bound),
                          "reference_winner_rank_in_device": int(1 + np.count_nonzero(a > a[wanted]))}
    return result


def run(args):
    import torch
    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    reader = BF16Checkpoint(args.checkpoint, manifest)
    config = json.loads((ROOT / "tests/fixtures/qwen3_tts_06b_config.json").read_text())
    spec = TalkerSpec.from_config(config)
    cfg = shared_tts_config()
    image = TalkerStackImage(spec, cfg, 512)
    weights = {name: reader.tensor(name, shape) for name, (_, _, shape) in image.parameters.items()}
    references = {"fp64": StackReference(spec, weights, 28),
                  "bf16_eager": StackReference(spec, weights, 28, bf16=True)}
    packed = image.build(weights, max_bytes=2 << 30)
    del weights
    normal, poisoned = StackISASession(image, packed), StackISASession(image, packed)
    for address, size in image.cache_regions().values():
        poisoned.machine.slices[0].dram[address:address + size] = 0x3f
    c = config["talker_config"]
    pairs = [(config["tts_bos_token_id"], c["codec_bos_id"]),
             (config["tts_pad_token_id"], c["spk_id"]["ryan"]),
             (config["tts_pad_token_id"], c["codec_language_id"]["english"]),
             (config["tts_pad_token_id"], c["codec_nothink_id"]),
             (151644, c["codec_pad_id"]), (0, 0),
             (spec.text_vocab - 1, spec.codec_vocab - 1),
             (config["tts_eos_token_id"], c["codec_eos_token_id"])]
    rng = np.random.default_rng(2306)
    pairs += [(int(rng.integers(spec.text_vocab)), int(rng.integers(spec.codec_vocab))) for _ in range(args.steps - 8)]
    rows, mask_mismatches, greedy_invariants = [], [], []
    for position, pair in enumerate(pairs):
        expected = {name: ref.step(*pair) for name, ref in references.items()}
        outputs, other_outputs = [], []
        def collect(destination):
            return lambda name, s: destination.append(s.read("output")) if name.startswith("layer.") else None
        logits, raw, masked = normal.step(*pair, observe=collect(outputs))
        other, other_raw, other_masked = poisoned.step(*pair, observe=collect(other_outputs))
        if (not np.array_equal(logits.view(np.uint32), other.view(np.uint32))
                or (raw, masked) != (other_raw, other_masked)
                or any(not np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(outputs, other_outputs))):
            mask_mismatches.append(position)
        if (raw, masked) != selection(logits):
            greedy_invariants.append(position)
        row = {"position": position, "ids": list(pair)}
        for name, want in expected.items():
            row[name] = {"layers": [metrics(a, b) for a, b in zip(outputs, want["layers"])],
                         "normalized": metrics(normal.read("normalized"), want["normalized"]),
                         "logits": metrics(logits, want["logits"]),
                         "greedy": agreement(logits, want["logits"], (raw, masked))}
        row["reference_greedy_agreement"] = selection(expected["fp64"]["logits"]) == selection(expected["bf16_eager"]["logits"])
        rows.append(row)
        progress = {"completed_positions": position + 1, "total": len(pairs),
                    "logit_l2_vs_fp64": row["fp64"]["logits"]["relative_l2"],
                    "raw_agrees_fp64": row["fp64"]["greedy"]["raw"]["agrees"]}
        args.output.with_suffix(".progress.json").write_text(json.dumps(progress))
        print(json.dumps(progress), flush=True)
    del poisoned
    normal.reset()
    fresh = StackISASession(image, packed)
    for ref in references.values():
        ref.reset()
    reset_mismatches = []
    for position, pair in enumerate(pairs[:2]):
        a = normal.step(*pair)
        b = fresh.step(*pair)
        if not np.array_equal(a[0].view(np.uint32), b[0].view(np.uint32)) or a[1:] != b[1:]:
            reset_mismatches.append(position)
    budgets = []
    for capacity, position in ((512, 0), (512, 256), (32768, 32767)):
        descriptor = TalkerStackImage(spec, cfg, capacity)
        parts = descriptor.compile_parts(position)
        budgets.append({"capacity": capacity, "position": position, "dram_bytes": descriptor.nbytes,
                        "kv_bytes": sum(size for _, size in descriptor.cache_regions().values()),
                        "program_parts": len(parts), "total_imem_words": sum(p[2]["imem_words"] for p in parts),
                        "max_part_imem_words": max(p[2]["imem_words"] for p in parts),
                        "max_part_tmem_words": max(p[2]["tmem_peak_words"] for p in parts)})
    summary = {}
    for name in references:
        summary[name] = {"layer_max_relative_l2": [max(r[name]["layers"][index]["relative_l2"] for r in rows) for index in range(28)],
                         "max_normalized_relative_l2": max(r[name]["normalized"]["relative_l2"] for r in rows),
                         "max_logit_relative_l2": max(r[name]["logits"]["relative_l2"] for r in rows),
                         "mean_logit_relative_l2": float(np.mean([r[name]["logits"]["relative_l2"] for r in rows])),
                         "min_logit_cosine": min(r[name]["logits"]["cosine"] for r in rows),
                         "max_absolute_logit_error": max(r[name]["logits"]["max_absolute"] for r in rows),
                         "greedy_agreement_counts": {policy: sum(r[name]["greedy"][policy]["agrees"] for r in rows) for policy in ("raw", "fixed_mask")},
                         "greedy_mismatch_positions": {policy: [r["position"] for r in rows if not r[name]["greedy"][policy]["agrees"]] for policy in ("raw", "fixed_mask")}}
    parity = all(count == len(pairs) for s in summary.values() for count in s["greedy_agreement_counts"].values())
    result = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "source_revision": SOURCE_REVISION,
              "target_profile": profile_metadata(),
              "checkpoint_manifest": manifest, "layers": 28, "parameter_tensors": len(image.parameters),
              "parameter_elements": sum(math.prod(shape) for _, _, shape in image.parameters.values()),
              "positions": len(pairs), "input_semantics": "Common diagnostic aligned rows, not valid full CustomVoice prompts or free-running audio",
              "summary": summary, "greedy_parity_observed": parity,
              "reference_greedy_agreement_count": sum(r["reference_greedy_agreement"] for r in rows),
              "future_cache_bitwise_mismatch_positions": mask_mismatches,
              "device_greedy_invariant_mismatch_positions": greedy_invariants,
              "reset_bitwise_mismatch_positions": reset_mismatches, "reset_positions": 2,
              "budgets": budgets, "imem_word_limit": cfg.IMEM_WORDS, "dram_byte_limit": cfg.DRAM_BYTES,
              "policy_limit": "Device raw/fixed-suppression greedy only; no repetition penalty or stochastic sampler",
              "upstream_model_code_executed": False, "hardware_access": False, "tts_enabled": False,
              "seconds": time.monotonic() - started, "python": platform.python_version(),
              "numpy": np.__version__, "torch": torch.__version__,
              "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                  [Path(__file__), ROOT / "opentpu/llm/qwen3_tts_stack.py", ROOT / "opentpu/kernels/tts_stack.py",
                   ROOT / "tests/qwen3_tts_stack_reference.py", ROOT / "tools/qwen3_tts_checkpoint_acceptance.py"]}}
    details = args.output.with_suffix(".positions.json")
    details.write_text(json.dumps(rows, allow_nan=False) + "\n")
    result["positions_sha256"] = sha256(details)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 1 if mask_mismatches or reset_mismatches or greedy_invariants else (0 if parity else 2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=16)
    args = p.parse_args()
    if not 8 <= args.steps <= 32:
        p.error("bounded stack diagnostic supports 8..32 positions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(run(args))
