#!/usr/bin/env python3
"""Source-format prefill and quantization ablation; CPU/ISA diagnostics only."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from opentpu import quant as Q
from opentpu.llm.qwen3_tts_shared import shared_tts_config, profile_metadata
from opentpu.llm.qwen3_tts_prompt import custom_voice_prefill
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from opentpu.llm.qwen3_tts_stack import TalkerStackImage, StackISASession
from tools.qwen3_tts_checkpoint_acceptance import BF16Checkpoint, metrics, sha256
from tools.qwen3_tts_stack_acceptance import selection
from qwen3_tts_stack_reference import StackReference
from qwen3_tts_quant_ablation import QuantReference


def run(a):
    import torch
    started = time.monotonic()
    cfg = shared_tts_config()
    fixture = json.loads((ROOT / "tests/fixtures/qwen3_tts_prompts.json").read_text())
    config = json.loads((ROOT / "tests/fixtures/qwen3_tts_06b_config.json").read_text())
    reader = BF16Checkpoint(a.checkpoint, json.loads(a.manifest.read_text()))
    spec = TalkerSpec.from_config(config)
    image = TalkerStackImage(spec, cfg, 512)
    weights = {k: reader.tensor(k, shape) for k, (_, _, shape) in image.parameters.items()}
    qweights = {}
    for k, value in weights.items():
        if value.ndim == 2 and "embedding" not in k:
            raw, scales = Q.quantize_mxu(value, "int8", cfg.D)
            qweights[k] = Q.dequantize_mxu(raw, scales, "int8", value.shape[1], cfg.D).astype(np.float32)
        else:
            qweights[k] = value
    refs = {"fp64": StackReference(spec, weights, 28),
            "fp32": StackReference(spec, weights, 28, dtype=torch.float32),
            "bf16": StackReference(spec, weights, 28, bf16=True),
            "weight_int8": QuantReference(spec, qweights),
            "activation_int8": QuantReference(spec, weights, activation=True),
            "kv_int8": QuantReference(spec, weights, kv=True),
            "all_int8": QuantReference(spec, qweights, activation=True, kv=True, attention=True)}
    packed = image.build(weights, max_bytes=2 << 30)
    session = StackISASession(image, packed)
    rows, finals = [], []
    for prompt in fixture["prompts"]:
        schedule = custom_voice_prefill(prompt["input_ids"], config)
        session.reset()
        for ref in refs.values():
            ref.reset()
        for position, row in enumerate(schedule):
            expected = {name: ref.step(row.text_id, row.codec_id, input_kind=row.kind) for name, ref in refs.items()}
            actual_layers = []
            logits, raw, masked = session.step(row.text_id, row.codec_id,
                observe=lambda name, s: actual_layers.append(s.read("output")) if name.startswith("layer.") else None,
                input_kind=row.kind)
            base = expected["fp32"]
            record = {"text": prompt["text"], "position": position, "kind": row.kind,
                      "ids": [row.text_id, row.codec_id], "last_prefill_row": position + 1 == len(schedule),
                      "device_ids": [raw, masked], "references": {}}
            for name, want in expected.items():
                record["references"][name] = {"ids": list(selection(want["logits"])),
                    "logits_vs_fp32": metrics(want["logits"], base["logits"]),
                    "isa_logits_error": metrics(logits, want["logits"]),
                    "isa_layer_error": [metrics(x, y)["relative_l2"] for x, y in zip(actual_layers, want["layers"])],
                    "layer_error_vs_fp32": [metrics(x, y)["relative_l2"] for x, y in zip(want["layers"], base["layers"])]}
            rows.append(record)
            if record["last_prefill_row"]:
                finals.append(record)
            progress = {"text": prompt["text"], "position": position + 1, "total": len(schedule),
                        "isa_vs_fp32": record["references"]["fp32"]["isa_logits_error"]["relative_l2"],
                        "isa_vs_functional_int8": record["references"]["all_int8"]["isa_logits_error"]["relative_l2"]}
            a.output.with_suffix(".progress.json").write_text(json.dumps(progress))
            print(json.dumps(progress), flush=True)
    summary = {}
    for name in refs:
        summary[name] = {"max_logit_error_vs_fp32": max(r["references"][name]["logits_vs_fp32"]["relative_l2"] for r in rows),
                         "max_isa_logit_error": max(r["references"][name]["isa_logits_error"]["relative_l2"] for r in rows),
                         "all_row_greedy_agreement": {p: sum(r["device_ids"][i] == r["references"][name]["ids"][i] for r in rows) for i, p in enumerate(("raw", "fixed_mask"))},
                         "final_prefill_greedy_agreement": {p: sum(r["device_ids"][i] == r["references"][name]["ids"][i] for r in finals) for i, p in enumerate(("raw", "fixed_mask"))},
                         "max_layer_error_vs_fp32": [max(r["references"][name]["layer_error_vs_fp32"][i] for r in rows) for i in range(28)],
                         "max_isa_layer_error": [max(r["references"][name]["isa_layer_error"][i] for r in rows) for i in range(28)]}
    result = {"profile": profile_metadata(), "prompts": fixture, "rows": len(rows),
              "input_contract": "Source non_streaming_mode=True CustomVoice prefill; three text-only role rows; no generated audio",
              "summary": summary, "final_prefill_rows": finals, "seconds": time.monotonic() - started,
              "device_dram_bytes": image.nbytes, "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                 [Path(__file__), ROOT / "tests/qwen3_tts_stack_reference.py", ROOT / "tests/qwen3_tts_quant_ablation.py",
                  ROOT / "opentpu/kernels/tts_stack.py", ROOT / "opentpu/llm/qwen3_tts_stack.py",
                  ROOT / "opentpu/llm/qwen3_tts_prompt.py", ROOT / "opentpu/llm/qwen3_tts_shared.py"]},
              "hardware_access": False, "tts_enabled": False, "upstream_model_code_executed": False}
    details = a.output.with_suffix(".positions.json")
    details.write_text(json.dumps(rows, allow_nan=False) + "\n")
    result["positions_sha256"] = sha256(details)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("final_prefill_rows", "prompts")}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args)
