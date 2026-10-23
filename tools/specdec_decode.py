#!/usr/bin/env python3
"""Speculative decoding (opentpu/llm/specdec.py, docs/specdec.md) on a full checkpoint, on the ISA
simulator or the card's driver over it (--backend board-isa): plain greedy decode on the card
(Engine.generate_card) against the prompt-lookup loop, token for token, per prompt
(tools/specdec_accept.py's: f2_bench's SHORT and LONG, then mtp_accept's nine), with the loop's
iterations, drafts accepted and verify runs; and the DRAM bytes of the loop's programs (the ISA
ledger, tools/dataflow.py) at a position, for the memory law of docs/fpga_inference.md section 3:
cycles = 0.67 M + bytes / 32 B on the 200 MHz F2 image.

    tools/specdec_decode.py run --model qwen3 --wformat int8 --head-format int8 --k 3 \\
        --tokens 128 --json q06-isa.json
    tools/specdec_decode.py ledger --model qwen3 --wformat int8 --head-format int8 --k 3 --pos 300
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

LAW_FIXED, LAW_BPC = 0.67e6, 31.93      # docs/fpga_inference.md section 3: the five images' fit


def engine(a, spec_rows: int):
    """(target Engine, its checkpoint, the draft model's Engine or None)."""
    from dataclasses import replace

    from opentpu.host import chat as CH
    from opentpu.host.board import sim_config
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import Engine, load_weights
    from opentpu.llm.specdec import draft_engine
    path = model_dir(a.model)
    spec = load_spec(path)
    if a.embed:
        spec = replace(spec, embed=a.embed)
    backend, cfg = CH.make_backend(a.backend, spec, a.cap, None, path.name, lookup=True,
                                   wformat=a.wformat, head_format=a.head_format)
    if cfg is None:
        cfg = sim_config(spec, a.cap, board_config(PAIR=True, DSTEP=True, STREAM=True),
                         lookup=True, wformat=a.wformat, head_format=a.head_format)
        if a.draft:                 # the slice's 4 GiB: the draft model's image at its top
            cfg = replace(cfg, DRAM_BYTES=1 << 32)
    W = load_weights(path)
    eng = Engine(spec, W, cap=a.cap, cfg=cfg, backend=backend, wformat=a.wformat,
                 head_format=a.head_format, resident=True, prog_cache=False,
                 prompt_runs=True, spec_rows=spec_rows)
    dr = None
    if a.draft:
        dr = draft_engine(eng, a.draft, wformat=a.draft_wformat,
                          head_format=a.draft_head_format, prog_cache=False)
        print(f"draft {a.draft}: image at {dr.image.base / 2**20:.0f} .. "
              f"{dr.image.nbytes / 2**20:.0f} MiB (target {eng.image.nbytes / 2**20:.0f} MiB)",
              flush=True)
    return eng, path, dr


def cmd_run(a) -> None:
    from opentpu.host import chat as CH
    from opentpu.llm.specdec import SpecDecoder, ngram_drafts
    from specdec_accept import prompts, walk
    eng, path, dr = engine(a, a.k + 1)
    tok = CH.load_tokenizer(path)()
    dec = SpecDecoder(eng, draft=dr)
    eos = list(eng.spec.eos)
    res = {"model": path.name, "wformat": a.wformat, "head_format": a.head_format, "k": a.k,
           "embed": a.embed, "draft": a.draft, "draft_wformat": a.draft_wformat,
           "draft_head_format": a.draft_head_format,
           "tokens": a.tokens, "backend": a.backend, "image_bytes": int(eng.image.nbytes),
           "prompts": []}
    sel = prompts()
    if a.only:
        sel = [sel[i] for i in a.only]
    for kind, text in sel:
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, enable_thinking=False,
                                      tokenize=True)
        ids = [int(x) for x in (ids["input_ids"] if hasattr(ids, "keys") else ids)]
        eng.reset()
        t0 = time.time()
        a0 = int(np.argmax(eng.prefill(ids)))
        P = eng.pos
        t1 = time.time()
        ref = eng.generate_card(a0, a.tokens, stop_ids=eos) if not a.no_plain else None
        t2 = time.time()
        eng.rollback(P)
        got = dec.generate_card(a0, a.tokens, stop_ids=eos, context=ids + [a0])
        t3 = time.time()
        st = dec.last
        seq = ids + [a0] + got
        host = walk(seq, len(ids), a.k, lambda i, k: ngram_drafts(seq[:i + 1], k)) \
            if dr is None else {"iters": None, "accepted": None, "drafted": None}
        rec = {"kind": kind, "prompt": len(ids), "tokens": [a0] + got, "same": ref == got
               if ref is not None else None, "iterations": st.iterations,
               "accepted": st.accepted, "verifies": st.verifies,
               "host_walk": {k: host[k] for k in ("iters", "accepted", "drafted")},
               "prefill_s": round(t1 - t0, 1), "plain_s": round(t2 - t1, 1),
               "spec_s": round(t3 - t2, 1), "compile_s": round(st.compile_s, 1)}
        res["prompts"].append(rec)
        print(f"{path.name} {kind}: {len(ids)}+{len(got) + 1} tokens, same {rec['same']}, "
              f"{st.iterations} iterations ({st.verifies} verify), {st.accepted} accepted "
              f"({len(got) / max(1, st.iterations):.2f} tokens an iteration; host walk "
              f"{host['iters']} / {host['accepted']}); plain {t2 - t1:.0f} s, loop "
              f"{t3 - t2:.0f} s", flush=True)
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1))
    n = sum(len(p["tokens"]) - 1 for p in res["prompts"])
    it = sum(p["iterations"] for p in res["prompts"])
    print(f"all: {n} tokens in {it} iterations, {n / max(1, it):.3f} tokens an iteration; "
          f"identical: {all(p['same'] for p in res['prompts'])}")


def cmd_ledger(a) -> None:
    """The DRAM bytes of the loop's V (K + 1 rows) and E (one row) at position --pos, and the
    plain loop's token (its generate program for one token), with the law's F2 cycles."""
    import dataflow as DF
    from opentpu.llm import generate as G
    from opentpu.llm import specdec as SD
    from opentpu.llm.qwen3 import PREFILL_ROWS
    from opentpu.llm import load_spec, model_dir
    from opentpu.host.board import sim_config
    from opentpu.isasim import board_config
    path = model_dir(a.model)
    spec = load_spec(path)
    cfg = sim_config(spec, a.cap, board_config(PAIR=True, DSTEP=True, STREAM=True),
                     lookup=True, wformat=a.wformat, head_format=a.head_format)
    img = spec.image(cfg, a.cap, 1, PREFILL_ROWS, wformat=a.wformat, head_format=a.head_format,
                     lookup=True, spec_rows=a.k + 1)
    blocks, block = a.pos // 256 + 1, 256
    out = {"model": path.name, "pos": a.pos, "k": a.k, "image_bytes": int(img.nbytes)}

    def poke_state(tpos, left):
        def poke(s):
            w = G.state_words(spec, 0, a.pos, left, [], block)
            w[G.S_TPOS] = tpos
            s.dram[img.lookup["gen"]["state"]:img.lookup["gen"]["state"] + 4 * len(w)] = \
                w.view(np.uint8)
        return poke
    progs = {"V": SD.compile_gen(img, blocks, SD.V, block),
             "E": SD.compile_gen(img, blocks, SD.E, block),
             "plain": img.compile_generate(blocks, (blocks - 1) * block, block, chain=False)}
    rcfg = __import__("dataclasses").replace(cfg, DRAM_BYTES=-(-img.nbytes // (1 << 20)) << 20)
    for name, p in progs.items():
        r = DF.ledger(rcfg, p, poke=poke_state(a.pos % block, 1))
        t = r["total"]
        rd, wr = sum(t["dram_rd"].values()), sum(t["dram_wr"].values())
        cyc = LAW_FIXED + (rd + wr) / LAW_BPC
        out[name] = {"instructions": len(p[0]), "dram_rd": rd, "dram_wr": wr,
                     "f2_cycles": cyc, "rd": t["dram_rd"]}
        print(f"{name:5s} {len(p[0]):5d} instructions; DRAM {rd / 1e6:9.2f} MB read, "
              f"{wr / 1e6:7.2f} MB written; F2 law {cyc / 1e6:6.2f} M cycles", flush=True)
    print(f"V / plain: {out['V']['f2_cycles'] / out['plain']['f2_cycles']:.4f}, "
          f"E / plain: {out['E']['f2_cycles'] / out['plain']['f2_cycles']:.4f}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "ledger"])
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--wformat", default="int8")
    ap.add_argument("--head-format", default="int8")
    ap.add_argument("--backend", default="isa", choices=["isa", "board-isa"])
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--pos", type=int, default=300)
    ap.add_argument("--only", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--no-plain", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--draft", help="a draft model (its checkpoint) instead of prompt lookup")
    ap.add_argument("--draft-wformat", default="int4")
    ap.add_argument("--draft-head-format", default="int4")
    ap.add_argument("--embed", choices=["f32", "int8"],
                    help="the target's embedding rows (int8: the tied int8 head's)")
    a = ap.parse_args()
    {"run": cmd_run, "ledger": cmd_ledger}[a.cmd](a)


if __name__ == "__main__":
    main()
