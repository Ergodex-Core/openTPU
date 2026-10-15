#!/usr/bin/env python3
"""Speculative decoding's acceptance, offline (docs/specdec.md): how many drafted tokens a greedy
verify accepts, for prompt-lookup (n-gram) drafts and for a draft model, on Hugging Face models
(no card, no simulator).

Per prompt (tools/f2_bench.py's SHORT and LONG, then tools/mtp_accept.py's nine) the target
generates up to --tokens tokens greedily. Each draft model then runs once, teacher-forced, over
the target's sequence: its argmax after every prefix. That is all the loop needs, greedy: a chain
of k drafts from position i is the draft model's own greedy continuation of the target's prefix,
and its accepted length is the run of drafts that equal the target's tokens, so while they match
the draft model's context is the target's sequence and its draft is the teacher-forced one.

The summary walks the loop exactly over each sequence for k = 1 .. 5 and 7 drafts an iteration: from
the token at position i, the drafter proposes up to k tokens for i + 1 ..; the target accepts the
longest prefix equal to its own tokens and emits it plus its next token. Per drafter it prints
the tokens per iteration and, with the costs of one decode step of the target (--t1, cycles), a
verify run's extra cost per row (--trow) and one draft-model step (--d1), the projected speedup
and tok/s at --mhz.

The n-gram drafter is the card's (opentpu/llm/specdec.py ngram_drafts): the longest of g = 3, 2,
1 whose last g tokens of the history occur earlier, at the latest such occurrence, proposing the
k tokens that followed it (periodically, where they run into the present).

    tools/specdec_accept.py --target Ternary-Bonsai-4B --drafts Qwen3-0.6B,Ternary-Bonsai-1.7B \\
        --out b4.json
    tools/specdec_accept.py --summary b4.json --t1 74.5e6 --trow 0.36e6 --d1 Qwen3-0.6B=19.9e6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def prompts() -> list:
    from f2_bench import LONG, SHORT
    from mtp_accept import PROMPTS
    return [("short", SHORT), ("long", LONG)] + list(PROMPTS)


from opentpu.llm.specdec import ngram_drafts  # noqa: E402 (the card's lookup)


def walk(seq, P: int, k: int, propose) -> dict:
    """The greedy loop over seq (the target's tokens; seq[P] the prefill's): iterations, those
    with drafts, tokens accepted. propose(i, k) -> the drafts for i + 1 .. (history seq[:i + 1])."""
    L = len(seq)
    i, it, drafted, acc, dr = P, 0, 0, 0, 0
    hist = []
    while i < L - 1:
        d = propose(i, k)
        n = 0
        while n < len(d) and i + 1 + n < L and d[n] == seq[i + 1 + n]:
            n += 1
        it += 1
        drafted += bool(d)
        dr += len(d)
        acc += n
        hist.append((len(d), n))
        i += n + 1
    return {"iters": it, "drafted": drafted, "drafts": dr, "accepted": acc, "tokens": L - 1 - P,
            "hist": hist}


def run(a) -> None:
    import torch
    import transformers
    torch.set_num_threads(a.threads)

    def path(m):
        return ROOT / "models" / m if not Path(m).is_dir() else Path(m)
    tp = path(a.target)
    tok = transformers.AutoTokenizer.from_pretrained(tp)
    model = transformers.AutoModelForCausalLM.from_pretrained(tp, dtype=torch.float32).eval()
    res = {"target": tp.name, "tokens": a.tokens, "prompts": []}
    for i, (kind, text) in enumerate(prompts()):
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, enable_thinking=False,
                                      tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        t0 = time.time()
        with torch.no_grad():
            seq = model.generate(torch.tensor([ids]), max_new_tokens=a.tokens, do_sample=False,
                                 repetition_penalty=1.0)[0].tolist()
        res["prompts"].append({"kind": kind, "P": len(ids), "ids": seq, "draft": {},
                               "seconds": round(time.time() - t0, 1)})
        print(f"{tp.name} #{i} {kind}: {len(ids)}+{len(seq) - len(ids)} tokens, "
              f"{time.time() - t0:.1f} s", flush=True)
    del model
    for dname in [d for d in (a.drafts or "").split(",") if d]:
        dp = path(dname)
        dm = transformers.AutoModelForCausalLM.from_pretrained(dp, dtype=torch.float32).eval()
        for r in res["prompts"]:
            with torch.no_grad():
                lg = dm(torch.tensor([r["ids"]])).logits[0]
            # d1[i]: the draft model's token for position i (its argmax after seq[:i])
            r["draft"][dp.name] = [None] + lg.argmax(-1).tolist()[:-1]
        print(f"draft {dp.name} done", flush=True)
        del dm
    Path(a.out).write_text(json.dumps(res))


def summary(files, a) -> None:
    d1 = {k: float(v) for k, v in (x.split("=") for x in (a.d1 or "").split(",") if x)}
    for f in files:
        r = json.loads(Path(f).read_text())
        names = ["ngram"] + list(r["prompts"][0]["draft"])
        print(f"\n== target {r['target']}: {len(r['prompts'])} prompts, "
              f"{sum(len(p['ids']) - p['P'] - 1 for p in r['prompts'])} tokens after the first")
        print(f"{'drafter':22s} {'k':>2s} {'kind':6s} {'tokens':>6s} {'iters':>5s} "
              f"{'drafted':>7s} {'tok/iter':>8s} {'accept':>6s} {'speedup':>7s} {'tok/s':>6s}")
        for name in names:
            for k in (1, 2, 3, 4, 5, 7):
                for kind in ("short", "long", "chat", "code", "summary", "all"):
                    ps = [p for p in r["prompts"] if kind in ("all", p["kind"])]
                    tot = {"iters": 0, "drafted": 0, "drafts": 0, "accepted": 0, "tokens": 0}
                    cost = 0.0
                    for p in ps:
                        seq, P = p["ids"], p["P"]
                        if name == "ngram":
                            def prop(i, k, seq=seq):
                                return ngram_drafts(seq[:i + 1], k)
                        else:
                            dd = p["draft"][name]

                            def prop(i, k, dd=dd, n=len(seq)):
                                return [dd[j] for j in range(i + 1, min(n, i + 1 + k))]
                        w = walk(seq, P, k, prop)
                        for x in tot:
                            tot[x] += w[x]
                        for nd, n in w["hist"]:
                            # the verify: a plain step without drafts, else k + 1 rows; a draft
                            # model's k steps (after all k accepted the first takes 2 rows, D2:
                            # the same weight stream)
                            cost += a.t1 + (nd and k * a.trow) + \
                                d1.get(name, 0.0) * k * (name != "ngram")
                    tpi = tot["tokens"] / max(1, tot["iters"])
                    sp = tot["tokens"] * a.t1 / cost if cost else 0.0
                    print(f"{name:22s} {k:2d} {kind:6s} {tot['tokens']:6d} {tot['iters']:5d} "
                          f"{tot['drafted'] / max(1, tot['iters']):7.2f} {tpi:8.2f} "
                          f"{tot['accepted'] / max(1, tot['drafts']):6.2f} {sp:7.2f} "
                          f"{sp * a.mhz * 1e6 / a.t1:6.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target")
    ap.add_argument("--drafts", default="")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out")
    ap.add_argument("--summary", nargs="+")
    ap.add_argument("--t1", type=float, default=1.0, help="a target decode step, cycles")
    ap.add_argument("--trow", type=float, default=0.0, help="a verify run's extra cycles a row")
    ap.add_argument("--d1", default="", help="draft model steps, cycles: NAME=c,...")
    ap.add_argument("--mhz", type=float, default=200.0)
    a = ap.parse_args()
    if a.summary:
        summary(a.summary, a)
    else:
        run(a)


if __name__ == "__main__":
    main()
