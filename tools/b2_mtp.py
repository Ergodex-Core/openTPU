"""MTP speculative decode of a Ternary-Bonsai 2 -MTP GGUF on the ISA simulator, against plain
greedy decode (docs/ternary.md 8): the same tokens bit for bit (asserted), the acceptance rate,
and the cost of each run kind in unit cycles (tools/b2_step_costs.py's tally: opentpu.profile's
nominal cycles per executed instruction, the MMs at their advance counts), so the net tokens
per unit cycle against plain decode.

    python tools/b2_mtp.py MODEL_DIR [--target f2-hbm] [--wformat t2] [--head-format t2]
                           [--cap 2048] [--prompt short|long|TEXT] [--steps 32] [--rows 8]
                           [--portb 2] [--mcols 8] [--rowlanes] [--card] [--reference]
                           [--json OUT.jsonl]

The main model's projections stay the latent ternary weights (t2, the rotation on the device);
the -MTP GGUF's nextn layer (mtp.*: Q8_0, not ternary, not rotated) is int8 (the formats rule
attn@L=int8,mlp@L=int8 at L = the model's layer count: Spec.formats). The prompt goes through
the chunked prefill (--rows rows a run) in both decoders. --card runs the loop on the device
too (MTPDecoder.generate_card: one chained run, the serving path's) and checks its tokens.
--reference adds one teacher-forced fp32 pass of the GGUF reference over the history: the
device tokens' argmax agreement with the model's own, and the drafts'.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opentpu import isa as I  # noqa: E402
from opentpu import profile as P  # noqa: E402
from opentpu import target  # noqa: E402
from opentpu.isasim import Slice  # noqa: E402
from opentpu.llm import bonsai2  # noqa: E402
from opentpu.llm.mtp import MTPDecoder, MTPStats, mtp_engine  # noqa: E402
from opentpu.llm.qwen3 import Engine  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f2_bench import LONG, SHORT  # noqa: E402

FMT = ["int8", "int4", "fp4", "nvfp4", "t2"]


class Tally:
    """Unit cycles and port-B chunks of every instruction the ISA simulator executes, into the
    bucket named by `self.bucket` (a run kind), while installed."""

    def __init__(self):
        self.bucket = "other"
        self.t = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0, 0]))
        self.runs = collections.defaultdict(list)      # bucket -> [(cycles, chunks)] per run
        self._cur = None

    def __enter__(self):
        self._orig = Slice.execute
        tally = self

        def ex(slf, ins):
            name, _, cyc, pb, pa = P._describe(ins, slf.cfg)
            if ins.op == I.MM:
                wfm, pair, quad = I.mm_format(ins)
                Lr = I.mm_lanes(ins)
                if Lr:          # MM RL (docs/row_lanes.md): the profile's groups x advances
                    name = f"MM {FMT[wfm]} rl{Lr}"
                else:
                    N, KB, M = ins.w[3] & 0xFFFF, (ins.w[3] >> 16) & 0xFF, (ins.w[5] >> 16) & 0xFF
                    pb = N * -(-KB // I.blocks_per_chunk(wfm))
                    cyc = N * -(-KB // I.reuse(pair, quad)) * -(-M // slf.cfg.MCOLS)
                    name = f"MM {FMT[wfm]}{' quad' if quad else ' pair' if pair else ''}"
            b = tally.t[tally.bucket][name]
            b[0] += 1
            b[1] += cyc
            b[2] += pb
            b[3] += pa
            if tally._cur is not None:
                tally._cur[0] += cyc
                tally._cur[1] += pb
            return tally._orig(slf, ins)
        Slice.execute = ex
        return self

    def __exit__(self, *a):
        Slice.execute = self._orig

    def begin(self, bucket: str):
        self.bucket, self._cur = bucket, [0, 0]

    def end(self):
        if self._cur is not None:
            self.runs[self.bucket].append(tuple(self._cur))
        self._cur = None

    def total(self, bucket: str) -> tuple:
        return (sum(v[1] for v in self.t[bucket].values()),
                sum(v[2] for v in self.t[bucket].values()))


def chat_tokens(model_dir, text: str) -> list:
    """The chat template's ids (thinking off), as the serving path and the baselines' probe."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        return list(tok.apply_chat_template([dict(role="user", content=text)], tokenize=True,
                                            add_generation_prompt=True, return_dict=False,
                                            enable_thinking=False))
    except ImportError:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        s = (f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
             "<think>\n\n</think>\n\n")
        return tk.encode(s, add_special_tokens=False).ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--target", default="f2-hbm")
    ap.add_argument("--wformat", default="t2")
    ap.add_argument("--head-format", default=None)
    ap.add_argument("--portb", type=int, default=None,
                    help="chunks a cycle on port B (Config.PORTB, with PAIR; docs/wide_port.md)")
    ap.add_argument("--mcols", type=int, default=None,
                    help="MXU columns (Config.MCOLS; default the target's): 8 lets a verify's "
                         "two rows QUAD (docs/ternary.md 9)")
    ap.add_argument("--rowlanes", action="store_true",
                    help="row lanes (Config.ROWLANES, docs/row_lanes.md): the one-row MMs as "
                         "MM RL of the widest L for --portb / --mcols")
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--prompt", default="short")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--card", action="store_true", help="the loop on the device too")
    ap.add_argument("--no-plain", action="store_true", help="skip plain decode (no assert)")
    ap.add_argument("--plain-step-cycles", type=float, default=None,
                    help="with --no-plain: a plain step's unit cycles (a measured run's) for "
                         "the speedup")
    ap.add_argument("--compile-only", action="store_true",
                    help="no decode: time the MTP loop's programs' compile per bucket on this "
                         "host (MTPDecoder._gen_programs: what a reply's first loop in a bucket "
                         "waits for), then gen_warm.warm_mtp and the same again from the cache")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--mhz", type=float, default=200.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    mdir = Path(a.model)
    text = {"short": SHORT, "long": LONG}.get(a.prompt, a.prompt)
    prompt = chat_tokens(mdir, text)
    hf = a.head_format or a.wformat
    cfg0 = target.config(target.load(a.target))
    if a.portb:
        cfg0 = replace(cfg0, PORTB=a.portb, PAIR=True)
    if a.mcols:
        cfg0 = replace(cfg0, MCOLS=a.mcols)
    if a.rowlanes:
        cfg0 = replace(cfg0, ROWLANES=True)
    spec = bonsai2.spec(mdir, ternary=True, slices=cfg0.S)
    L = spec.layers
    spec = replace(spec, formats=f"attn@{L}={'int8'},mlp@{L}={'int8'}")   # (the MTP layer)
    W = bonsai2.Weights(mdir, mtp=True, latent=True, ffn=spec.ffn)
    out = {"model": mdir.name, "target": a.target, "portb": cfg0.PORTB, "mcols": cfg0.MCOLS,
           "rowlanes": bool(cfg0.ROWLANES), "wformat": a.wformat, "head_format": hf,
           "cap": a.cap, "prompt": a.prompt, "prompt_tokens": len(prompt), "steps": a.steps,
           "rows": a.rows}
    print(f"{mdir.name}: prompt {a.prompt!r} = {len(prompt)} tokens, {a.steps} steps, "
          f"{a.wformat}/{hf}, cap {a.cap}, PORTB {cfg0.PORTB} MCOLS {cfg0.MCOLS}"
          f"{' RL' if cfg0.ROWLANES else ''}", flush=True)
    kw = dict(backend="isa", wformat=a.wformat, head_format=hf, rows=a.rows, embed_host=True,
              prompt_runs=False)

    def sized(sp, mtp=False):
        img = sp.image(replace(cfg0, DRAM_BYTES=1 << 40), a.cap, rows=a.rows, lookup=True,
                       wformat=a.wformat, head_format=hf, embed_host=True)
        return replace(cfg0, DRAM_BYTES=-(-img.nbytes // 4096) * 4096), img.nbytes

    plain = None
    if not a.no_plain:
        cfg, nb = sized(spec)
        t0 = time.perf_counter()
        eng = Engine(spec, W, cap=a.cap, cfg=cfg, resident=True, **kw)
        out["plain_image_bytes"] = int(eng.image.nbytes)
        print(f"plain image {nb / 2**20:.0f} MiB built in {time.perf_counter() - t0:.0f} s",
              flush=True)
        with Tally() as ty:
            ty.begin("prefill")
            t0 = time.perf_counter()
            logits = eng.prefill(prompt)
            ty.end()
            pre_s = time.perf_counter() - t0
            plain, steps = [], []
            for _ in range(a.steps):
                t = int(np.argmax(logits))
                plain.append(t)
                if t in spec.eos or len(plain) >= a.steps:
                    break
                ty.begin("step")
                t0 = time.perf_counter()
                logits = eng.step(t)
                steps.append(time.perf_counter() - t0)
                ty.end()
        sc = ty.runs["step"]
        step_cyc = float(np.mean([c for c, _ in sc])) if sc else 0.0
        out["plain"] = {"tokens": plain, "prefill_cycles": ty.total("prefill")[0],
                        "prefill_s": pre_s, "step_cycles": [c for c, _ in sc],
                        "step_chunks": [b for _, b in sc], "step_cycles_mean": step_cyc,
                        "step_wall_s": float(np.mean(steps)) if steps else 0.0}
        print(f"plain: {len(plain)} tokens, prefill {ty.total('prefill')[0] / 1e6:.2f} M unit "
              f"cycles ({pre_s:.0f} s), a step {step_cyc / 1e6:.3f} M unit cycles "
              f"({out['plain']['step_wall_s']:.0f} s wall): {plain}", flush=True)
        del eng
        import gc
        gc.collect()

    # ---- the MTP decoder, the loop on the host (per-run costs)
    spec_m = replace(spec, mtp=True)
    cfg, nb = sized(spec_m, True)
    t0 = time.perf_counter()
    eng = mtp_engine(spec_m, W, cap=a.cap, cfg=cfg, **kw)
    out["mtp_image_bytes"] = int(eng.image.nbytes)
    print(f"MTP image {nb / 2**20:.0f} MiB (MTP layer formats "
          f"{eng.image.layouts[eng.image.keys[L][1]].mf['wq']}, fc {eng.image.mtp_fc}, draft "
          f"head fp4 x {eng.image.nd}) built in {time.perf_counter() - t0:.0f} s", flush=True)
    dec = MTPDecoder(eng)
    if a.compile_only:
        from opentpu import progcache as PC
        from opentpu.host import gen_warm as GW
        from opentpu.llm import generate as G
        nb = G.buckets(a.cap, eng.block)
        out["compile"] = {}
        for b in range(1, min(nb, 3) + 1):
            t0 = time.perf_counter()
            progs = dec._gen_programs(b, b)
            dt = time.perf_counter() - t0
            sizes = {k[1] if len(k) == 2 else f"{k[1]}.{k[2]}": max(map(len, v))
                     for k, v in progs.items() if k[0] == b and v is not None}
            out["compile"][b] = {"seconds": dt, "instructions": sizes}
            print(f"bucket {b}: the MTP loop's programs compiled in {dt:.1f} s on this host "
                  f"(instructions {sizes})", flush=True)
        if eng.prog_cache and PC.cache_dir() is not None:
            t0 = time.perf_counter()
            GW.warm_mtp(eng.image, eng.layout, eng.block, min(nb, 4))
            dt = time.perf_counter() - t0
            eng.__dict__.pop("_mtp_gen", None)
            t1 = time.perf_counter()
            dec._gen_programs(min(nb, 4), min(nb, 4))
            dt2 = time.perf_counter() - t1
            out["compile"]["warm"] = {"warm_seconds": dt, "after_warm_seconds": dt2}
            print(f"bucket {min(nb, 4)}: gen_warm.warm_mtp {dt:.1f} s, then _gen_programs from "
                  f"the cache {dt2:.2f} s", flush=True)
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(out) + "\n")
        return
    drafts = []                 # (position drafted for, draft id)
    orig_draft = dec._draft

    def _draft(p, toks, st, kind, h0=0):
        d = orig_draft(p, toks, st, kind, h0)
        for j, x in enumerate(d):
            drafts.append((p + j + 2, int(x)))
        return d
    dec._draft = _draft
    orig_run = dec._run
    ty = Tally()

    def _run(progs, kind, rows, st, tokens=None):
        ty.begin(kind)
        t0 = time.perf_counter()
        orig_run(progs, kind, rows, st, tokens)
        ty.runs[kind + " wall"].append((time.perf_counter() - t0, rows))
        ty.end()
    dec._run = _run
    with ty:
        t0 = time.perf_counter()
        st = dec.generate(prompt, max_new=a.steps)
        wall = time.perf_counter() - t0
    toks = list(st.tokens)
    acc = list(st.accepted)
    nv, nd = len(ty.runs["verify"]), len(ty.runs["mtp"])
    vc = [c for c, _ in ty.runs["verify"]]
    dcy = [c for c, _ in ty.runs["mtp"]]
    v_mean = float(np.mean(vc)) if vc else 0.0
    d_mean = float(np.mean(dcy)) if dcy else 0.0
    loop_cycles = sum(vc) + sum(dcy)
    same = plain is None or toks == plain
    res = {"tokens": toks, "accepted": acc, "acceptance": st.acceptance, "iterations": nv,
           "verify_cycles": vc, "draft_cycles": dcy, "verify_cycles_mean": v_mean,
           "draft_cycles_mean": d_mean, "loop_cycles": loop_cycles,
           "prefill_cycles": ty.total("prefill")[0] + ty.total("mtp prefill")[0],
           "mtp_prefill_cycles": ty.total("mtp prefill")[0],
           "compile_s": st.compile_s, "prefill_s": st.prefill_s, "wall_s": wall,
           "verify_wall_s": [w for w, _ in ty.runs["verify wall"]],
           "draft_wall_s": [w for w, _ in ty.runs["mtp wall"]],
           "drafts": drafts, "same_as_plain": same,
           "verify_kinds": {k: v for k, v in ty.t["verify"].items()},
           "draft_kinds": {k: v for k, v in ty.t["mtp"].items()}}
    out["mtp"] = res
    print(f"MTP (host loop): {len(toks)} tokens in {nv} verify + {nd} draft runs, acceptance "
          f"{st.acceptance:.2f}; verify {v_mean / 1e6:.3f} M unit cycles a run, draft "
          f"{d_mean / 1e6:.3f} M; prefill {res['prefill_cycles'] / 1e6:.2f} M (MTP layer "
          f"{res['mtp_prefill_cycles'] / 1e6:.2f} M); host compile {st.compile_s:.0f} s of "
          f"{wall:.0f} s; tokens {'EQUAL' if same else 'DIFFER'}: {toks}", flush=True)
    pc = out["plain"]["step_cycles_mean"] if plain is not None else a.plain_step_cycles
    if pc:
        n = len(toks) - 1                   # the tokens after the prefill's
        c2, cd = v_mean / pc, d_mean / pc
        sp = (n * pc) / loop_cycles if loop_cycles else 0.0
        out["speedup_cycles"] = sp
        out["c_verify"], out["c_draft"] = c2, cd
        out["tok_per_Mcycle"] = {"plain": 1e6 / pc, "mtp": n * 1e6 / loop_cycles}
        print(f"  c_verify {c2:.3f}, c_draft {cd:.3f} of a plain step; {n} tokens: plain "
              f"{n * pc / 1e6:.1f} M unit cycles, MTP {loop_cycles / 1e6:.1f} M: speedup "
              f"{sp:.3f}x; tok/s at {a.mhz:g} MHz plain {a.mhz * 1e6 / pc:.2f}, MTP "
              f"{a.mhz * 1e6 * n / loop_cycles:.2f}", flush=True)
    assert same, f"MTP tokens differ from plain greedy decode:\n{toks}\n{plain}"

    if a.card:                  # the loop on the device: one chained run, as otpu-serve runs it
        del dec, eng
        import gc
        gc.collect()
        eng = mtp_engine(spec_m, W, cap=a.cap, cfg=cfg, **kw)
        dec = MTPDecoder(eng)
        ty2 = Tally()
        with ty2:
            ty2.begin("card")
            t0 = time.perf_counter()
            st2 = dec.generate_card(prompt, max_new=a.steps)
            wall2 = time.perf_counter() - t0
            ty2.end()
        toks2 = list(st2.tokens)
        same2 = toks2 == toks
        out["card"] = {"tokens": toks2, "accepted": list(st2.accepted), "iterations":
                       st2.iterations, "acceptance": st2.acceptance,
                       "cycles_total": ty2.total("card")[0], "compile_s": st2.compile_s,
                       "prefill_s": st2.prefill_s, "wall_s": wall2, "same_as_host_loop": same2}
        print(f"MTP (loop on the device): {len(toks2)} tokens, {st2.iterations} iterations, "
              f"acceptance {st2.acceptance:.2f}, {ty2.total('card')[0] / 1e6:.1f} M unit cycles "
              f"in all (prefill + loop), compile {st2.compile_s:.0f} s of {wall2:.0f} s; tokens "
              f"{'EQUAL' if same2 else 'DIFFER'} to the host loop's", flush=True)
        assert same2, f"the device loop's tokens differ:\n{toks2}\n{toks}"

    if a.reference:             # one fp32 pass of the GGUF reference over the device's history
        from opentpu.llm import qwen35
        Wr, spr = bonsai2.Weights(mdir), bonsai2.spec(mdir)
        hist = prompt + toks[:-1]
        t0 = time.perf_counter()
        lg = qwen35.reference_logits(spr, Wr, hist)[len(prompt) - 1:]
        am = [int(x) for x in lg.argmax(1)]
        agree = sum(int(x == y) for x, y in zip(am, toks))
        dmap = dict(drafts)
        dref = [(dmap[p], am[p - len(prompt)]) for p in range(len(prompt), len(prompt) + len(toks))
                if p in dmap]
        dagree = sum(int(x == y) for x, y in dref)
        out["reference"] = {"argmax": am, "device_agree": agree, "of": len(toks),
                            "draft_agree": dagree, "drafts_of": len(dref),
                            "seconds": time.perf_counter() - t0}
        print(f"reference (fp32, {time.perf_counter() - t0:.0f} s): device tokens' argmax "
              f"agreement {agree} / {len(toks)}; drafts equal to the reference's argmax "
              f"{dagree} / {len(dref)}", flush=True)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
