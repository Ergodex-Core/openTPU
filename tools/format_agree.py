#!/usr/bin/env python3
"""Greedy agreement of a weight-format choice with the Hugging Face fp32 model, over a prompt set
(docs/fpga_inference.md 5): the decode bytes a format saves against the tokens it keeps.

    python tools/format_agree.py ref MODEL --out ref.npz [--steps 32]
    python tools/format_agree.py run MODEL --ref ref.npz --wformat int8 --head-format int4 \
        [--formats 'mlp=int4'] [--json out.json]
    python tools/format_agree.py fit MODEL --wformat int4 --head-format int4 --cap 2048 --slots 8

`ref`: each prompt of PROMPTS in the checkpoint's chat template (thinking off), HF fp32 (eager,
use_cache) decoding greedily for --steps tokens or to an EOS; its tokens and its logits at every
generated position. `run`: the openTPU engine on the ISA simulator (the f2 target's arch, the
image's formats: --wformat / --head-format / --formats as OTPU_FORMATS, docs/formats.md) takes
each prompt as a prefill, then is teacher-forced along HF's tokens. Per prompt: the positions
whose argmax is HF's token (teacher-forced agreement), the first position where it is not (the
free-running greedy sequence matches HF's up to there: before it both histories are the same),
the logits' cosine with HF's at each position, the KL divergence of the device's next-token
distribution from HF's (nats, the mean over positions; docs/formats.md's measure), and HF's own top-1 / top-2 margin where they
differ (a near tie is a coin toss, not a defect).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from f2_bench import LONG, SHORT  # noqa: E402

PROMPTS = [
    SHORT,
    LONG,
    "What is 17 multiplied by 23? Show your work briefly.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Translate into French: The weather is beautiful today and I would like to go for a walk.",
    "List three causes of the French Revolution.",
    "Explain in simple terms how a vaccine trains the immune system.",
    "Write a haiku about autumn leaves.",
    "A train leaves at 3 pm and travels at 60 miles per hour. How far has it gone by 5:30 pm?",
    "Give me a JSON object with the fields name, age and city for a fictional person.",
]


def chat_ids(path, prompt: str) -> list:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                  add_generation_prompt=True, return_dict=False,
                                  enable_thinking=False)
    return [int(t) for t in ids]


def eos_ids(path) -> set:
    from opentpu.llm import load_spec
    return set(load_spec(path).eos)


def ref(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    from opentpu.llm import model_dir
    path = model_dir(a.model)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                                                 attn_implementation="eager",
                                                 local_files_only=True).eval()
    stops, out = eos_ids(path), {}
    for i, p in enumerate(PROMPTS):
        ids = chat_ids(path, p)
        t0 = time.monotonic()
        toks, lg = [], []
        with torch.inference_mode():
            o = model(torch.tensor([ids]), use_cache=True)
            for _ in range(a.steps):
                row = o.logits[0, -1].float().numpy().copy()
                lg.append(row)
                t = int(row.argmax())
                toks.append(t)
                if t in stops:
                    break
                o = model(torch.tensor([[t]]), past_key_values=o.past_key_values, use_cache=True)
        out[f"p{i}"] = np.asarray(ids, np.int64)
        out[f"t{i}"] = np.asarray(toks, np.int64)
        out[f"l{i}"] = np.asarray(lg, np.float32)
        print(f"prompt {i}: {len(ids)} + {len(toks)} tokens, {time.monotonic() - t0:.1f} s",
              flush=True)
    np.savez(a.out, **out)


def run(a) -> None:
    from opentpu import target
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import Engine, load_weights
    if a.formats is not None:
        os.environ["OTPU_FORMATS"] = a.formats
    path = model_dir(a.model)
    spec = load_spec(path)
    R = np.load(a.ref)
    n = sum(1 for k in R.files if k.startswith("p"))
    cfg = target.config(target.load("f2"))
    cfg = replace(cfg, DRAM_BYTES=1 << 32)              # the card's 4 GiB (the target's)
    t0 = time.monotonic()
    eng = Engine(spec, load_weights(path), cap=a.cap, cfg=cfg, wformat=a.wformat,
                 head_format=a.head_format, resident=True, pipeline=False)
    res = dict(model=path.name, wformat=a.wformat, head_format=a.head_format,
               formats=a.formats or "", image_bytes=int(eng.image.nbytes),
               build_s=round(time.monotonic() - t0, 1), prompts=[])
    print(f"{path.name} {a.wformat}/{a.head_format} formats={a.formats!r}: image "
          f"{eng.image.nbytes:,} B, build {res['build_s']} s", flush=True)
    for i in range(n):
        prompt, toks, hl = R[f"p{i}"], R[f"t{i}"], R[f"l{i}"]
        t1 = time.monotonic()
        eng.reset()
        logits = eng.prefill([int(t) for t in prompt])
        match, cos, kl, first, ties = [], [], [], None, []
        for j, t in enumerate(toks):
            d = np.asarray(logits, np.float64)
            h = hl[j].astype(np.float64)
            ok = int(d.argmax()) == int(t)
            match.append(ok)
            cos.append(float(d @ h / (np.linalg.norm(d) * np.linalg.norm(h))))
            lh, ld = h - h.max(), d - d.max()
            lh, ld = lh - np.log(np.exp(lh).sum()), ld - np.log(np.exp(ld).sum())
            kl.append(float(np.exp(lh) @ (lh - ld)))
            if not ok:
                top = np.sort(h)[-2:]
                ties.append(dict(pos=j, hf=int(t), dev=int(d.argmax()),
                                 hf_margin=float(top[1] - top[0]),
                                 hf_gap_to_dev=float(h[int(t)] - h[int(d.argmax())])))
                if first is None:
                    first = j
            if j + 1 < len(toks):
                logits = eng.step(int(t))
        r = dict(prompt=i, n=len(toks), agree=int(sum(match)),
                 first_diverge=first, seq_match=first is None,
                 cos_min=min(cos), cos_mean=float(np.mean(cos)), kl_mean=float(np.mean(kl)),
                 misses=ties,
                 seconds=round(time.monotonic() - t1, 1))
        res["prompts"].append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "misses"}), flush=True)
    P = res["prompts"]
    res["summary"] = dict(agree=sum(r["agree"] for r in P), n=sum(r["n"] for r in P),
                          seq_match=sum(r["seq_match"] for r in P), prompts=len(P),
                          prefix=sum(r["n"] if r["first_diverge"] is None else r["first_diverge"]
                                     for r in P),
                          cos_min=min(r["cos_min"] for r in P),
                          cos_mean=float(np.mean([r["cos_mean"] for r in P])),
                          kl_mean=float(sum(r["kl_mean"] * r["n"] for r in P) /
                                        sum(r["n"] for r in P)))
    print("summary", json.dumps(res["summary"]), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")


def fit(a) -> None:
    """Does the image fit the f2 slice's DRAM with otpu-serve's batched slots (`--slots` KV
    slots at context `--cap`, the I/O rows of its tiled prompt runs), the batch area and the
    program area after it (Engine(batch_area=True), BoardBackend)? No weights are read."""
    from opentpu import target
    from opentpu import waits as WC
    from opentpu.llm import batch as BA
    from opentpu.llm import load_spec, model_dir
    if a.formats is not None:
        os.environ["OTPU_FORMATS"] = a.formats
    spec = load_spec(model_dir(a.model))
    cfg = target.config(target.load("f2"))
    img = spec.image(replace(cfg, DRAM_BYTES=1 << 40), a.cap, a.slots, cfg.act_rows,
                     wformat=a.wformat, head_format=a.head_format, lookup=True)
    end = BA.alloc(img, cfg, a.slots, -(-img.nbytes // 4096) * 4096)["end"] if a.slots > 1 \
        else -(-img.nbytes // 4096) * 4096
    need = end + WC.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
    print(json.dumps(dict(model=Path(a.model).name, wformat=a.wformat, head=a.head_format,
                          formats=a.formats or "", cap=a.cap, slots=a.slots,
                          image_mib=round(img.nbytes / 2**20, 1), need_mib=round(need / 2**20, 1),
                          dram_mib=cfg.DRAM_BYTES >> 20, fits=need <= cfg.DRAM_BYTES)), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("ref", "run", "fit"))
    ap.add_argument("model")
    ap.add_argument("--out", help="ref: the HF reference's NPZ")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--ref", help="run: the HF reference's NPZ")
    ap.add_argument("--wformat", default="int8")
    ap.add_argument("--head-format")
    ap.add_argument("--formats", help="a formats string (docs/formats.md), as OTPU_FORMATS")
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--slots", type=int, default=8, help="fit: otpu-serve's KV slots")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "ref":
        if not a.out:
            ap.error("ref needs --out")
        ref(a)
    elif a.cmd == "fit":
        fit(a)
    else:
        if not a.ref:
            ap.error("run needs --ref")
        run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
