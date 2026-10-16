"""The ISA-level cost of one pi0.5 action chunk (docs/pi05.md 5): every instruction the
chunk's runs execute (llm/pi05.Image.runs over its compiled programs; loop counts expanded,
no arithmetic), tallied per part (vision: both cameras, prefix, one expert step x 10) and per
unit at profile._describe's nominal rates:

  MXU     an MM's advances: N x KB x ceil(M / MCOLS) cycles (replayed rows: ACT_ROWS > MCOLS)
  port B  D-byte chunks: an MM's weight rows (N x KB, once per MM: a row group re-streams
          the matrix), LD / ST chunks; QST's bytes go to DRAM on port A's side (here: "QST")
  VPU     rows x ceil(cols / LANES) a VOP, split by kind: GELU (gelu_tanh), softmax (the
          attention's online softmax and P V scaling), norms (RMSNorm / adaRMS / LayerNorm),
          RoPE, the rest (bias adds, residual gates, ...)
  QACT    2 ceil(rows KB D / LANES), QST ceil(n / LANES) + n
  DMA     LD / ST: ceil(words / min(D / 4, LANES)) cycles

    python tools/pi05_cost.py [--np 558] [--mhz 200] [--alt] [--json out.json]

Seconds = cycles / MHz per unit; the chunk's time lies between the slowest unit's (every
unit overlapped) and their sum (none). --alt also tallies hypothetical configurations (ALTS:
ACT_ROWS 128, LANES 32 / 64 alone, then together with a larger TMEM and 16 / 64 MXU columns;
each with the row groups of the smallest bound that compile), the ISA's Config only, no RTL.
No weights are read: the image is laid out, its programs compiled, nothing run.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import isa as I  # noqa: E402
from opentpu import profile as PF  # noqa: E402
from opentpu.compiler import CompileError  # noqa: E402
from opentpu.llm import pi05 as P  # noqa: E402

VKINDS = (("gelu_tanh", "GELU"), ("_finish", "softmax"), ("emit", "softmax"),
          ("_attend_heads", "softmax"), ("_scores", "softmax"), ("rms_r", "norm"),
          ("_ada", "norm"), ("layernorm_q", "norm"), ("layernorm_", "norm"),
          ("rope_rows", "RoPE"))


def vkind(ins) -> str:
    fns = [f for _, _, f in ins.src]
    for fn, k in VKINDS:
        if fn in fns:
            return k
    return "rest"


def counts(prog) -> list:
    """How many times each instruction of a program runs (LOOP counts are immediates here:
    R[ra] + w2 with ra = 0; refused otherwise)."""
    n = [0] * len(prog)

    def walk(a, b, mult):
        i = a
        while i < b:
            ins = prog[i]
            if ins.op == I.LOOP:
                if ins.ra:
                    raise ValueError("a loop count from a register")
                body, cnt = ins.w[0], ins.w[1]
                walk(i + 1, i + 1 + body, mult * cnt)
                i += 1 + body
                continue
            if ins.op == I.HALT:
                return
            n[i] += mult
            i += 1
    walk(0, len(prog), 1)
    return n


def tally(prog, cfg) -> dict:
    t = collections.Counter()
    for ins, c in zip(prog, counts(prog)):
        if not c or ins.op in (I.LI, I.ADDI, I.NOP):
            continue
        name, _, cyc, pb, pa = PF._describe(ins, cfg)
        if ins.op == I.MM:
            N, KB, M = ins.w[3] & 0xFFFF, (ins.w[3] >> 16) & 0xFF, (ins.w[5] >> 16) & 0xFF
            t["MXU"] += c * N * KB * -(-M // cfg.MCOLS)
            t["portB"] += c * pb
            t["portB_MM"] += c * pb
            t["MACs"] += c * N * KB * cfg.D * M
            t["n_MM"] += c
        elif ins.op == I.VOP:
            t["VPU"] += c * cyc
            t["VPU." + vkind(ins)] += c * cyc
        elif ins.op == I.QACT:
            t["QACT"] += c * cyc
        elif ins.op == I.QST:
            t["QST"] += c * cyc
        elif ins.op in (I.LD, I.ST):
            t["DMA"] += c * cyc
            t["portB"] += c * pb
        t["instructions"] += c
    return t


def chunk_cost(spec, cfg, Np: int, R: int, RP: int, RE: int) -> dict:
    image = P.Image(spec, cfg, np_max=max(640, Np))
    progs = image.programs(Np, R=R, RP=RP, RE=RE)
    per = {k: tally(p, cfg) for k, (p, _) in progs.items()}
    parts = {"vision": collections.Counter(), "prefix": collections.Counter(),
             "expert_step": collections.Counter()}
    steps = 0
    for item in image.runs(spec.cameras):
        name = item[0]
        if name.startswith("write"):
            continue
        if name == "e_step":
            steps += 1
            if steps > 1:
                continue
        part = {"v": "vision", "p": "prefix", "e": "expert_step"}[name[0]]
        parts[part].update(per[name])
    return {k: dict(v) for k, v in parts.items()}, {k: len(p) for k, (p, _) in progs.items()}


# hypothetical configurations (the ISA's Config only; no RTL): one lever at a time from today's
# (ACT_ROWS, LANES, TMEM, MXU columns), then together, up to docs/vla_frequency_plan.md's
# f2-vla without its fused activation unit. ACT_ROWS 255 is the ISA's most (an MM's row count
# is an 8-bit field, isa.mm: 0 < m < 256) and reaches 256's pass count over the prefix, 3 at
# Np 558; its 255-row prefix group needs TMEM 1.5M words (at 1M: TMEM exhausted, and the
# group stays 128). docs/vla_frequency_plan.md 6: MCOLS 32 unpacked and 64 packed
ALTS = (("act128", dict(ACT_ROWS=128)),
        ("lanes32", dict(LANES=32)),
        ("lanes64", dict(LANES=64)),
        ("act128_lanes32", dict(ACT_ROWS=128, LANES=32)),
        ("act128_lanes32_tmem256k", dict(ACT_ROWS=128, LANES=32, TMEM_WORDS=262144)),
        ("mcols16_act128_lanes32_tmem256k", dict(MCOLS=16, ACT_ROWS=128, LANES=32,
                                                 TMEM_WORDS=262144)),
        ("mcols64_act128_lanes64_tmem1m", dict(MCOLS=64, ACT_ROWS=128, LANES=64,
                                               TMEM_WORDS=1 << 20)),
        ("mcols32_act255_lanes64_tmem1.5m", dict(MCOLS=32, ACT_ROWS=255, LANES=64,
                                                 TMEM_WORDS=3 << 19)),
        ("mcols64_act255_lanes64_tmem1.5m", dict(MCOLS=64, ACT_ROWS=255, LANES=64,
                                                 TMEM_WORDS=3 << 19)))

# the row groups tried for each configuration (tower R, prefix RP, expert RE): the one with
# the smallest bound among those that compile (TMEM holds the row groups' work areas); groups
# past the configuration's ACT_ROWS are skipped
ROW_GROUPS = ((128, 255, 64), (255, 255, 64), (128, 128, 64), (128, 64, 64), (64, 64, 64),
              (64, 32, 32), (32, 16, 16), (32, 16, 8), (16, 8, 8))

UNITS = ("MXU", "portB", "VPU", "QACT", "QST", "DMA")


def report(parts: dict, steps: int, mhz: float, title: str) -> dict:
    out = {}
    print(f"\n== {title}")
    tot = collections.Counter()
    for part, t in parts.items():
        k = steps if part == "expert_step" else 1
        sec = {u: t.get(u, 0) / (mhz * 1e6) for u in UNITS}
        bound = max(sec.values())
        out[part] = {"cycles": {u: int(t.get(u, 0)) for u in UNITS},
                     "vpu_split": {kk[4:]: int(v) for kk, v in t.items()
                                   if kk.startswith("VPU.")},
                     "MACs": int(t.get("MACs", 0)), "instructions": int(t.get("instructions", 0)),
                     "seconds_bound": bound, "seconds_serial": sum(sec.values())}
        for u in list(UNITS) + ["MACs"]:
            tot[u] += k * t.get(u, 0)
        for kk, v in t.items():
            if kk.startswith("VPU."):
                tot[kk] += k * v
        vs = ", ".join(f"{kk}={v / (mhz * 1e6):.3f}" for kk, v in
                       sorted(out[part]["vpu_split"].items(), key=lambda kv: -kv[1]))
        print(f"{part:12s}{' x%d' % k if k > 1 else '':4s} " +
              " ".join(f"{u}={sec[u]:.3f}s" for u in UNITS) +
              f"  bound={bound:.3f}s serial={sum(sec.values()):.3f}s  [VPU: {vs}]")
    sec = {u: tot[u] / (mhz * 1e6) for u in UNITS}
    out["chunk"] = {"seconds": sec, "bound": max(sec.values()), "serial": sum(sec.values()),
                    "MACs": int(tot["MACs"]),
                    "vpu_split_s": {kk[4:]: tot[kk] / (mhz * 1e6) for kk in tot
                                    if kk.startswith("VPU.")}}
    print("chunk       " + " ".join(f"{u}={sec[u]:.3f}s" for u in UNITS) +
          f"  bound={max(sec.values()):.3f}s serial={sum(sec.values()):.3f}s"
          f"  MACs={tot['MACs'] / 1e12:.3f}T")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=558)
    ap.add_argument("--mhz", type=float, default=200.0)
    ap.add_argument("--target", default="f2-hbm")
    ap.add_argument("--alt", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()
    from opentpu import target as TG
    cfg = TG.config(TG.load(args.target))
    spec = P.Spec()
    res = {}
    parts, sizes = chunk_cost(spec, cfg, args.np, 16, 8, 8)
    res["today"] = report(parts, spec.steps, args.mhz,
                          f"{args.target}: S={cfg.S} D={cfg.D} MCOLS={cfg.MCOLS} ACT_ROWS="
                          f"{cfg.ACT_ROWS} LANES={cfg.LANES} TMEM={cfg.TMEM_WORDS}, R 16/8/8")
    res["today"]["program_instructions"] = sizes
    if args.alt:
        import dataclasses
        for name, kw in ALTS:
            alt = dataclasses.replace(cfg, **kw)
            best = None
            for R in ROW_GROUPS:
                if max(R) > alt.ACT_ROWS:
                    continue
                try:
                    got = chunk_cost(spec, alt, args.np, *R)
                except CompileError:
                    continue
                b = max(sum((spec.steps if k == "expert_step" else 1) * t.get(u, 0)
                            for k, t in got[0].items()) for u in UNITS)
                if best is None or b < best[0]:
                    best = (b, R, got)
            _, best, (parts, sizes) = best
            res[name] = report(parts, spec.steps, args.mhz,
                               f"{name}: {kw}, R {best}")
            res[name]["R"] = best
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
