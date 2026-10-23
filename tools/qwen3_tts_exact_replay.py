#!/usr/bin/env python3
"""Independent operator-level numeric replay on a complete source prefill."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from opentpu.llm.qwen3_tts_shared import shared_tts_config, profile_metadata
from opentpu.llm.qwen3_tts_prompt import custom_voice_prefill
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from opentpu.llm.qwen3_tts_stack import TalkerStackImage, StackISASession
from tools.qwen3_tts_checkpoint_acceptance import BF16Checkpoint, metrics, sha256
from qwen3_tts_exact_mx import ExactMX


def run(a):
    started = time.monotonic()
    config = json.loads((ROOT / "tests/fixtures/qwen3_tts_06b_config.json").read_text())
    prompts = json.loads((ROOT / "tests/fixtures/qwen3_tts_prompts.json").read_text())["prompts"]
    image = TalkerStackImage(TalkerSpec.from_config(config), shared_tts_config(), 512)
    reader = BF16Checkpoint(a.checkpoint, json.loads(a.manifest.read_text()))
    weights = {name: reader.tensor(name, shape) for name, (_, _, shape) in image.parameters.items()}
    reference = ExactMX(image.spec, weights)
    session = StackISASession(image, image.build(weights, max_bytes=2 << 30))
    rows = []
    for position, row in enumerate(custom_voice_prefill(prompts[0]["input_ids"], config)):
        expected = reference.step(row.text_id, row.codec_id, input_kind=row.kind)
        layers = []
        logits, raw, masked = session.step(row.text_id, row.codec_id, input_kind=row.kind,
            observe=lambda name, s: layers.append(s.read("output")) if name.startswith("layer.") else None)
        layers_equal = [bool(np.array_equal(x.view(np.uint32), y.view(np.uint32))) for x, y in zip(layers, expected["layers"])]
        r = {"position": position, "layer_bitwise_equal": layers_equal,
             "layer_relative_l2": [metrics(x, y)["relative_l2"] for x, y in zip(layers, expected["layers"])],
             "logits_bitwise_equal": bool(np.array_equal(logits.view(np.uint32), expected["logits"].view(np.uint32))),
             "logit_metrics": metrics(logits, expected["logits"]), "device_ids": [raw, masked]}
        rows.append(r)
        print(json.dumps({"position": position, "layers_bitwise_equal": sum(layers_equal),
                          "logit_relative_l2": r["logit_metrics"]["relative_l2"]}), flush=True)
    result = {"profile": profile_metadata(), "prompt": prompts[0], "rows": rows,
              "all_layers_bitwise_equal": all(all(r["layer_bitwise_equal"]) for r in rows),
              "all_logits_bitwise_equal": all(r["logits_bitwise_equal"] for r in rows),
              "max_logit_relative_l2": max(r["logit_metrics"]["relative_l2"] for r in rows),
              "seconds": time.monotonic() - started, "hardware_access": False,
              "scope": "Operator-level replay sharing documented numeric primitives, not independent quality acceptance",
              "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / "tests/qwen3_tts_exact_mx.py"]}}
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args)
