"""Gemma 3 / FunctionGemma on the ISA simulator against the fp32 numpy reference.

  python tools/gemma3_check.py --model gemma3 --wformat int8 --steps 4
  python tools/gemma3_check.py --model functiongemma --wformat int4 --steps 4

The reference reads the same checkpoint (safetensors: Gemma 3 runs on Gemma 4's code,
opentpu/llm/gemma3.py, its weights folded by gemma4.Weights; a GGUF directory is not
supported here). The device uses openTPU weight and activation quantization. This reports
agreement, not a claim that those numerics reproduce a bf16 run exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None):
    from tokenizers import Tokenizer
    from opentpu.isasim import board_config
    from opentpu.llm import gemma3 as G, load_spec, model_dir
    from opentpu.llm.qwen3 import Engine, load_weights
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="functiongemma")
    ap.add_argument("--wformat", choices=("int8", "int4", "fp4"), default="int4")
    ap.add_argument("--head-format", choices=("int8", "int4", "fp4"))
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--emulated", action="store_true", help="also check quantization-aware emulation")
    ap.add_argument("--tools", help="JSON list of tools for the chat template")
    ap.add_argument("--chat", action="store_true", help="apply the checkpoint chat template")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--stop-token", type=int, action="append", default=[],
                    help="additional token ID to stop after (repeatable)")
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--json", help="save measured results")
    a = ap.parse_args(argv)
    path = model_dir(a.model)
    spec, W = load_spec(path), load_weights(path)      # (gemma4's Spec and folded Weights)
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    prompt = [spec.bos] + tok.encode(a.prompt, add_special_tokens=False).ids
    if a.chat:
        from opentpu.host.chat import load_tokenizer
        hf_tok = load_tokenizer(path)()
        prompt = hf_tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                            tokenize=True, add_generation_prompt=True, return_dict=False,
                                            **({"tools": json.loads(Path(a.tools).read_text())}
                                               if a.tools else {}))
    probe = spec.image(board_config(DRAM_BYTES=1 << 40), a.cap, rows=1,
                       wformat=a.wformat, head_format=a.head_format, lookup=True)
    cfg = board_config(DRAM_BYTES=1 << max(20, (probe.nbytes - 1).bit_length()))
    t0 = time.monotonic()
    eng = Engine(spec, W, cap=a.cap, cfg=cfg, wformat=a.wformat, head_format=a.head_format, resident=True, rows=1)
    print(f"image {eng.image.nbytes:,} bytes, build {time.monotonic() - t0:.1f}s", flush=True)
    logits = None
    for t in prompt:
        logits = eng.step(t)
    seq, records, device_logits = list(prompt), [], []
    for i in range(a.steps):
        ref = G.reference_logits(spec, W, seq)[-1]
        device_token, ref_token = int(logits.argmax()), int(ref.argmax())
        cos = float(np.dot(logits, ref) / np.linalg.norm(logits) / np.linalg.norm(ref))
        row = dict(step=i, device_token=device_token, reference_token=ref_token, cosine=cos,
                   text=tok.decode([device_token]), match=device_token == ref_token)
        records.append(row)
        device_logits.append(logits)
        print(json.dumps(row), flush=True)
        seq.append(device_token)
        if device_token in (*spec.eos, *a.stop_token) or i + 1 == a.steps:
            break
        logits = eng.step(device_token)
    if a.emulated:
        emu = G.emulated_logits(spec, W, seq[:-1], wformat=a.wformat,
                                head_format=a.head_format)[len(prompt) - 1:]
        for row, dev, ref in zip(records, device_logits, emu):
            row["emulated_token"] = int(ref.argmax())
            row["emulated_cosine"] = float(np.dot(dev, ref) / np.linalg.norm(dev) /
                                           np.linalg.norm(ref))
            print(json.dumps({k: row[k] for k in ("step", "device_token", "emulated_token",
                                                 "emulated_cosine")}), flush=True)
    result = dict(model=str(path), wformat=a.wformat, head_format=a.head_format, prompt=prompt, image_bytes=eng.image.nbytes,
                  seconds=time.monotonic() - t0, generated=seq[len(prompt):],
                  generated_text=tok.decode(seq[len(prompt):], skip_special_tokens=False),
                  stop_tokens=a.stop_token, results=records)
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=2) + "\n")
    print(f"teacher-forced argmax agreement: {sum(r['match'] for r in records)}/{len(records)}",
          flush=True)
    print(result["generated_text"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
