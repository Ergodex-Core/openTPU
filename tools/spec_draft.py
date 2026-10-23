#!/usr/bin/env python3
"""Speculative decoding with a draft model, phase 0 (docs/spec_draft.md): Gemma-3-1B-it (int8
weights, int8 head) drafted by gemma-3-270m-it, measured on the ISA simulator in the card's
configuration (targets/f2-hbm.toml variant p4rl8a64c150, CFG_HASH dd6b92a6: PORTB 4, row lanes,
MCOLS 8, ALINE 64; the ISA is the card's bit for bit, docs/wide_port.md "On the F2 card").

    spec_draft.py target   --out DIR [--prompts a,b|all] [--max-new 256]
        the target's plain greedy decode of each prompt (chat template), as the card's chat runs
        it: prompt runs (Engine.prefill_chunks), the first token picked from the prefill's
        logits, then Engine.generate_card (the resident generate loop, greedy on the device)
        up to an EOS id or --max-new; DIR/target/<prompt>.json
    spec_draft.py draft    --out DIR --tag T [--wformat int8] [--head-format int8]
        the draft model teacher-forced over each target sequence (prompt + reply): runs of
        --rows token rows (Engine.run_rows, every row's logits), per position its argmax and
        top-TOPK ids / logits (a truncated draft head's argmax is the first of them in its id
        set: the rows it keeps are the full head's, quantized row by row);
        DIR/draft-T/<prompt>.npz
    spec_draft.py rowcheck --model M [--wformat ...] [--rows 4,6,8]
        a rows run of R rows (every row's logits) against R decode steps (the resident decode
        program) from the same context: bit for bit (the verify run's premise)
    spec_draft.py accept   --out DIR [--tags T,...] [--k 1,2,3,4,5,6,7,8] [--json F]
        the greedy speculative loop walked over the target's tokens with each draft's
        predictions (a greedy draft chain's accepted run equals its teacher-forced predictions'
        while they match the target's tokens): tokens per verify E[n + 1], the acceptance of
        draft position i given 1 .. i - 1 accepted, the draft steps an iteration (K, K + 1 after
        an iteration that accepted all K), and with --costs the projected tok/s

The draft's K / V in the loop always holds the committed tokens (it catches up from the first
mismatch: positional caches are overwritten), so its predictions are the teacher-forced ones,
provided its rows runs are bit-identical to its decode steps (rowcheck).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

CARD = "f2-hbm:p4rl8a64c150"        # the card's image (417243f1): CFG_HASH dd6b92a6
TOPK = 64

SHORT = "Write one sentence about the Moon."          # tools/f2_bench.py's
EXTRA = [
    ("sky", "chat", "Why is the sky blue? Answer in one sentence."),
    ("transformer", "chat", "Explain how a transformer language model generates text, step "
                            "by step."),
    ("hello", "chat", "Hello"),
    ("tcp", "chat", "Explain the difference between TCP and UDP for a beginner, with an "
                    "example of when to use each."),
    ("list", "list", "List ten practical tips for writing clear technical documentation."),
    ("packing", "list", "Give me a packing list for a three-day hiking trip, grouped by "
                        "category."),
    ("palindrome", "code", "Write a Python function that checks whether a string is a "
                           "palindrome, ignoring case and punctuation. Include a docstring and "
                           "three examples."),
    ("linkedlist", "code", "Write a C function that reverses a singly linked list in place, "
                           "and explain how it works."),
    ("romeo", "summary", "Summarize the plot of Romeo and Juliet in one paragraph."),
    ("story", "chat", "Write a short story, about 150 words, about a robot who learns to "
                      "paint."),
]


def prompts() -> list:
    """(name, kind, text): f2_bench's SHORT and LONG, mtp_accept's nine (chat, code, summaries),
    the card runs' and a few more (explanations, lists, code, summaries)."""
    import f2_bench as FB
    import mtp_accept as MA
    out = [("short", "chat", SHORT), ("long", "summary", FB.LONG)]
    seen = {}
    for kind, text in MA.PROMPTS:
        seen[kind] = seen.get(kind, 0) + 1
        out.append((f"{kind}{seen[kind]}", kind, text))
    return out + EXTRA


def card_config(dram_bytes: int | None = None):
    """The card's Config (its CFG_HASH), with the DRAM cut to `dram_bytes` (the image layout and
    the programs do not depend on it: one model at the bottom of the slice's DRAM)."""
    from opentpu import target as T
    cfg = T.config(CARD)
    assert T.arch_hash(cfg) == 0xdd6b92a6, hex(T.arch_hash(cfg))
    return replace(cfg, DRAM_BYTES=dram_bytes) if dram_bytes else cfg


def model_path(name: str) -> Path:
    from opentpu.llm import model_dir
    return model_dir(name)


def engine(model: str, wformat: str, head_format: str, cap: int, rows: int | None = None,
           dram_bytes: int | None = None, prompt_runs: bool = True):
    """An Engine on the ISA simulator in the card's configuration, as otpu-chat makes it there
    (resident decode, prompt runs); rows: the image's I/O rows (run_rows' logits rows)."""
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import Engine, load_weights
    path = model_path(model)
    spec = load_spec(path)
    kw = {} if rows is None else {"rows": rows}
    return Engine(spec, load_weights(path), cap=cap, cfg=card_config(dram_bytes), backend="isa",
                  wformat=wformat, head_format=head_format, resident=True, prog_cache=False,
                  prompt_runs=prompt_runs, **kw)


def tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path(model))


def chat_ids(tok, text: str) -> list:
    ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


# ------------------------------------------------------------------------------------ target
def cmd_target(a) -> None:
    out = Path(a.out) / "target"
    out.mkdir(parents=True, exist_ok=True)
    tok = tokenizer(a.model)
    sel = prompts() if a.prompts == "all" else [p for p in prompts()
                                                 if p[0] in a.prompts.split(",")]
    t0 = time.time()
    eng = engine(a.model, a.wformat, a.head_format, a.cap, dram_bytes=a.dram)
    print(f"engine {time.time() - t0:.1f} s", flush=True)
    for name, kind, text in sel:
        f = out / f"{name}.json"
        if f.exists() and not a.force:
            print(f"{name}: done", flush=True)
            continue
        ids = chat_ids(tok, text)
        eng.reset()
        t1 = time.time()
        k0 = len(eng.stats)
        logits = eng.prefill(ids)
        a0 = int(np.argmax(logits))
        t2 = time.time()
        k1 = len(eng.stats)
        eos = list(eng.spec.eos)
        got = [] if a0 in eos else eng.generate_card(a0, a.max_new - 1, stop_ids=eos)
        t3 = time.time()
        gen = [a0] + got
        cyc = lambda st: sum((s or {}).get("instructions", [0])[0] for s in st)   # noqa: E731
        rec = {"name": name, "kind": kind, "text": text, "model": a.model,
               "wformat": a.wformat, "head_format": a.head_format, "config": CARD,
               "prompt": ids, "gen": gen, "eos": eos, "stopped": bool(gen and gen[-1] in eos),
               "reply": tok.decode([t for t in gen if t not in eos]),
               "prefill_s": t2 - t1, "decode_s": t3 - t2,
               "prefill_instructions": cyc(eng.stats[k0:k1]),
               "decode_instructions": cyc(eng.stats[k1:])}
        f.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"{name}: prompt {len(ids)}, generated {len(gen)} (prefill {t2 - t1:.0f} s, "
              f"decode {t3 - t2:.0f} s, {len(gen) / max(1e-9, t3 - t2):.2f} tok/s): "
              f"{rec['reply'][:100]!r}", flush=True)


# ------------------------------------------------------------------------------------- draft
def teacher_forced(eng, seq, R: int, topk: int = TOPK):
    """The engine (reset) fed `seq` in runs of R rows with every row's logits: per position q
    the argmax and the top-k (ids by descending logit, ties by id) of its logits (predicting
    seq[q + 1])."""
    n = len(seq)
    am = np.zeros(n, np.int64)
    ti = np.zeros((n, topk), np.int32)
    tv = np.zeros((n, topk), np.float32)
    eng.reset()
    p = 0
    while p < n:
        r = min(R, n - p)
        lg = eng.run_rows([(0, p + i) for i in range(r)], seq[p:p + r], list(range(r)))
        eng.pos = p + r
        for i in range(r):
            x = lg[i]
            am[p + i] = int(np.argmax(x))
            part = np.argpartition(-x, topk)[:topk]
            o = np.lexsort((part, -x[part]))
            ti[p + i], tv[p + i] = part[o], x[part[o]]
        p += r
    return am, ti, tv


def cmd_draft(a) -> None:
    src = Path(a.out) / "target"
    dst = Path(a.out) / f"draft-{a.tag}"
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.json"))
    if a.prompts != "all":
        files = [f for f in files if f.stem in a.prompts.split(",")]
    t0 = time.time()
    eng = engine(a.model, a.wformat, a.head_format, a.cap, rows=a.rows, dram_bytes=a.dram,
                 prompt_runs=False)
    print(f"engine {time.time() - t0:.1f} s", flush=True)
    meta = {"model": a.model, "wformat": a.wformat, "head_format": a.head_format,
            "config": CARD, "rows": a.rows, "topk": TOPK}
    (dst / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    for f in files:
        out = dst / f"{f.stem}.npz"
        if out.exists() and not a.force:
            continue
        rec = json.loads(f.read_text())
        seq = rec["prompt"] + rec["gen"]
        t1 = time.time()
        am, ti, tv = teacher_forced(eng, seq, a.rows)
        np.savez_compressed(out, argmax=am, topk_ids=ti, topk_vals=tv,
                            seq=np.asarray(seq, np.int64), prompt_len=len(rec["prompt"]))
        P = len(rec["prompt"])
        hit = np.mean(am[P - 1:len(seq) - 1] == np.asarray(seq[P:]))
        print(f"{f.stem}: {len(seq)} positions in {time.time() - t1:.0f} s, next-token "
              f"agreement with the target {hit:.3f}", flush=True)


# ---------------------------------------------------------------------------------- rowcheck
def cmd_rowcheck(a) -> None:
    """Decode steps against rows runs from the same context, logits bit for bit."""
    tok = tokenizer(a.model)
    Rs = [int(x) for x in a.rows_list.split(",")]
    eng = engine(a.model, a.wformat, a.head_format, a.cap, rows=max(Rs), dram_bytes=a.dram)
    ids = chat_ids(tok, a.text)
    res = []
    for R in Rs:
        eng.reset()
        lg = eng.prefill(ids)
        P = len(ids)
        toks = [int(np.argmax(lg))]
        steps = []
        for _ in range(R):                       # R decode steps: tokens toks[0 .. R - 1]
            x = eng.step(toks[-1])
            steps.append(x.copy())
            toks.append(int(np.argmax(x)))
        eng.pos = P                              # back to the prompt's end (positional KV)
        rows = eng.run_rows([(0, P + i) for i in range(R)], toks[:R], list(range(R)))
        eng.pos = P + R
        same = [bool(np.array_equal(steps[i].view(np.uint32), rows[i].view(np.uint32)))
                for i in range(R)]
        diff = [int(np.count_nonzero(steps[i].view(np.uint32) != rows[i].view(np.uint32)))
                for i in range(R)]
        res.append({"R": R, "positions": [P, P + R - 1], "bit_identical": same,
                    "unequal_words": diff, "tokens": toks[:R + 1]})
        print(f"R = {R} at positions {P} .. {P + R - 1}: rows run == decode steps: "
              f"{all(same)} (unequal words per row {diff})", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps({"model": a.model, "wformat": a.wformat,
                                            "head_format": a.head_format, "config": CARD,
                                            "text": a.text, "results": res}, indent=1) + "\n")


# ------------------------------------------------------------------------------------ accept
def subset_argmax(ti: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Per position: the first of the top-k ids that `keep` (a bool mask over the vocabulary)
    keeps, -1 when none of them is kept (counted as a miss)."""
    k = keep[ti]
    first = np.argmax(k, axis=1)
    out = ti[np.arange(len(ti)), first].astype(np.int64)
    out[~k.any(axis=1)] = -1
    return out


def walk(seq, P: int, pred, K: int) -> dict:
    """The greedy loop over the target's tokens seq (prompt of P, then the reply: seq[P] the
    prefill's token), drafts pred[q] for the token after position q (teacher-forced). An
    iteration at position p (its token seq[p] committed) drafts d_i = pred[p + i - 1] while they
    match; n = the accepted run (at most K), it emits n + 1 tokens (at most those left)."""
    n_tok = len(seq)
    p, it, emitted, steps, prev_all = P, 0, 0, 0, False
    reach = np.zeros(K + 1, np.int64)       # iterations where draft i was verified
    acc = np.zeros(K + 1, np.int64)         # ... and accepted
    hist = np.zeros(K + 2, np.int64)        # iterations by tokens emitted
    while p < n_tok - 1:
        left = n_tok - 1 - p
        n = 0
        while n < K and n < left - 1:
            reach[n + 1] += 1
            if pred[p + n] != seq[p + n + 1]:
                break
            acc[n + 1] += 1
            n += 1
        e = min(n + 1, left)
        steps += K + (1 if prev_all else 0)
        prev_all = n == K
        hist[e] += 1
        emitted += e
        it += 1
        p += e
    return {"iterations": it, "tokens": emitted, "draft_steps": steps, "reach": reach.tolist(),
            "accepted": acc.tolist(), "hist": hist.tolist()}


def cmd_accept(a) -> None:
    base = Path(a.out)
    targets = {f.stem: json.loads(f.read_text()) for f in sorted((base / "target").glob("*.json"))}
    tags = a.tags.split(",") if a.tags else sorted(d.name[6:] for d in base.glob("draft-*"))
    Ks = [int(x) for x in a.k.split(",")]
    V = 262144
    subsets = {"full": None}
    for n in (8192, 16384, 32768, 65536):
        m = np.zeros(V, bool)
        m[:n] = True
        subsets[f"low{n // 1024}K"] = m
    # frequency-ranked ids: every target reply's tokens counted, leave-one-prompt-out
    counts = {nm: np.bincount(np.asarray(r["gen"], np.int64), minlength=V)
              for nm, r in targets.items()}
    report = {"K": Ks, "drafts": {}}
    for tag in tags:
        d = base / f"draft-{tag}"
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        for sname, mask in subsets.items():
            tot = {K: None for K in Ks}
            per_prompt = {}
            for nm, rec in targets.items():
                z = d / f"{nm}.npz"
                if not z.exists():
                    continue
                npz = np.load(z)
                seq = [int(x) for x in npz["seq"]]
                P = int(npz["prompt_len"])
                pred = npz["argmax"] if mask is None else subset_argmax(npz["topk_ids"], mask)
                per_prompt[nm] = {}
                for K in Ks:
                    w = walk(seq, P, pred, K)
                    per_prompt[nm][K] = w
                    if tot[K] is None:
                        tot[K] = {k: (np.asarray(v) if isinstance(v, list) else v)
                                  for k, v in w.items()}
                    else:
                        for k, v in w.items():
                            tot[K][k] = tot[K][k] + (np.asarray(v) if isinstance(v, list) else v)
            if not per_prompt:
                continue
            summ = {}
            for K in Ks:
                t = tot[K]
                summ[K] = {"E_n1": float(t["tokens"] / t["iterations"]),
                           "iterations": int(t["iterations"]), "tokens": int(t["tokens"]),
                           "draft_steps_per_iter": float(t["draft_steps"] / t["iterations"]),
                           "accept_by_pos": [float(t["accepted"][i] / max(1, t["reach"][i]))
                                             for i in range(1, K + 1)],
                           "hist": [int(x) for x in t["hist"]]}
            report["drafts"][f"{tag}/{sname}"] = {"meta": meta, "summary": summ,
                                                   "per_prompt": {nm: {K: {
                                                       "E_n1": w["tokens"] / w["iterations"],
                                                       "iterations": w["iterations"],
                                                       "tokens": w["tokens"]}
                                                       for K, w in v.items()}
                                                       for nm, v in per_prompt.items()}}
    # the frequency-ranked heads (leave-one-prompt-out: a prompt's own reply is not counted)
    for tag in tags:
        d = base / f"draft-{tag}"
        for n in (8192, 16384, 32768):
            tot = {K: [0, 0] for K in Ks}
            for nm in targets:
                z = d / f"{nm}.npz"
                if not z.exists():
                    continue
                c = sum(v for k, v in counts.items() if k != nm)
                order = np.lexsort((np.arange(V), -c))
                m = np.zeros(V, bool)
                m[order[:n]] = True
                npz = np.load(z)
                seq = [int(x) for x in npz["seq"]]
                pred = subset_argmax(npz["topk_ids"], m)
                for K in Ks:
                    w = walk(seq, int(npz["prompt_len"]), pred, K)
                    tot[K][0] += w["tokens"]
                    tot[K][1] += w["iterations"]
            if tot[Ks[0]][1]:
                report["drafts"].setdefault(f"{tag}/freq{n // 1024}K-loo", {})["summary"] = {
                    K: {"E_n1": tot[K][0] / tot[K][1]} for K in Ks}
    print("| draft / head | " + " | ".join(f"K = {K}: E[n+1]" for K in Ks) + " |")
    print("|---|" + "---|" * len(Ks))
    for key, v in report["drafts"].items():
        s = v["summary"]
        print(f"| {key} | " + " | ".join(f"{s[K]['E_n1']:.3f}" for K in Ks) + " |")
    for key, v in report["drafts"].items():
        s = v["summary"]
        if "accept_by_pos" not in s[Ks[-1]]:
            continue
        print(f"{key}: acceptance by draft position (K = {Ks[-1]}): " +
              ", ".join(f"{x:.3f}" for x in s[Ks[-1]]["accept_by_pos"]))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("target", "draft", "rowcheck", "accept"):
        p = sub.add_parser(name)
        p.add_argument("--out", default="build/spec-draft")
        p.add_argument("--model", default="gemma3" if name != "draft" else
                       "models/gemma-3-270m-it")
        p.add_argument("--wformat", default="int8")
        p.add_argument("--head-format", default="int8")
        p.add_argument("--cap", type=int, default=2048)
        p.add_argument("--dram", type=int, default=None, help="DRAM bytes (default the card's)")
        p.add_argument("--prompts", default="all")
        p.add_argument("--force", action="store_true")
        p.add_argument("--json", default=None)
    sub.choices["target"].add_argument("--max-new", type=int, default=256)
    sub.choices["draft"].add_argument("--tag", required=True)
    sub.choices["draft"].add_argument("--rows", type=int, default=8)
    sub.choices["rowcheck"].add_argument("--rows-list", default="4,6,8")
    sub.choices["rowcheck"].add_argument("--text", default=SHORT)
    sub.choices["accept"].add_argument("--tags", default=None)
    sub.choices["accept"].add_argument("--k", default="1,2,3,4,5,6,7,8")
    a = ap.parse_args(argv)
    {"target": cmd_target, "draft": cmd_draft, "rowcheck": cmd_rowcheck,
     "accept": cmd_accept}[a.cmd](a)


if __name__ == "__main__":
    main()
