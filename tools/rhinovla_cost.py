"""The ISA-level cost of one RhinoVLA action chunk (docs/rhinovla.md 5), tools/pi05_cost.py's
method: every instruction the chunk's runs execute (llm/rhinovla.Image.runs over its compiled
programs; loop counts expanded, no arithmetic), tallied per part (vision: the 3 views, prefix,
one expert step x steps) and per unit at profile._describe's nominal rates:

  MXU     an MM's advances: N x KB x ceil(M / MCOLS) cycles (PACK: ceil(N / 2) x KB: two int8
          weight rows an advance), replayed rows at ACT_ROWS > MCOLS
  port B  D-byte chunks: an MM's weight rows (N x KB, once per row group), LD / ST chunks;
          seconds at PORTB chunks a cycle (--portb overrides the target's)
  VPU     rows x ceil(cols / LANES) a VOP, split by kind: GELU (gelu_tanh and the mergers' erf
          GELU), SiLU, softmax, norms, RoPE, the rest
  QACT    2 ceil(rows KB D / LANES); QST the K stores a word a cycle, the V^T stores a word of
          four rows (QST T); DMA LD / ST

    python tools/rhinovla_cost.py [--target f2-hbm | f2-vla] [--portb 2] [--mhz 200]
        [--text 41] [--json out.json] [--alt]

No weights are read: the image is laid out for the configured Spec (the V3 configs' shapes),
its programs compiled for a 3-view prompt of --text text tokens (41: the v2 template around
a 7-word task, 233 prefix rows), nothing run. The row groups (tower R, prefix RP, expert RE,
mergers RM) are the smallest bound's among those that compile (TMEM holds the work areas).
--alt also tallies the plan's points from today's target (ALTS: f2-vla's sizes, packed /
unpacked, PORTB 2) whatever --target says. Seconds = cycles / MHz per unit; a chunk's time
lies between the slowest unit's (every unit overlapped: "bound") and their sum ("serial").
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import isa as I  # noqa: E402
from opentpu import profile as PF  # noqa: E402
from opentpu.compiler import CompileError  # noqa: E402
from opentpu.llm import rhinovla as RV  # noqa: E402

VKINDS = (("gelu_tanh", "GELU"), ("gelu_erf", "GELU"), ("silu", "SiLU"), ("_finish", "softmax"),
          ("emit", "softmax"), ("_attend_heads", "softmax"), ("_scores", "softmax"),
          ("rms_r", "norm"), ("_ada", "norm"), ("rmsnorm", "norm"), ("layernorm_q", "norm"),
          ("layernorm_", "norm"), ("rope_rows", "RoPE"))


def vkind(ins) -> str:
    fns = [f for _, _, f in ins.src]
    for fn, k in VKINDS:
        if fn in fns:
            return k
    return "rest"


def counts(prog) -> list:
    """How many times each instruction of a program runs (LOOP counts are immediates)."""
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
            t["MXU"] += c * cyc
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


def prompt_ids(spec: RV.Spec, text: int = 41):
    """Placeholder token ids with the v2 template's shape: 9 text tokens, a view, 9, a view,
    9, a view, the rest of the text (14 of the 41; the ids themselves do not matter to the
    layout)."""
    heads = [9, 9, 9][:spec.views]
    tail = text - sum(heads)
    ids = []
    for h in heads:
        ids += [0] * h + [RV.IMAGE_PAD] * spec.n_tok
    return ids + [0] * tail


def chunk_cost(spec, cfg, lay, R: int, RP: int, RE: int, RM: int):
    image = RV.Image(spec, cfg, np_max=max(256, lay.L))
    progs = image.programs(lay, R=R, RP=RP, RE=RE, RM=RM)
    per = {k: tally(p, cfg) for k, (p, _) in progs.items()}
    parts = {"vision": collections.Counter(), "prefix": collections.Counter(),
             "expert_step": collections.Counter()}
    steps = 0
    for item in image.runs(lay):
        name = item[0]
        if name.startswith("write"):
            continue
        if name == "e_step":
            steps += 1
            if steps > 1:
                continue
        parts[{"v": "vision", "p": "prefix", "e": "expert_step"}[name[0]]].update(per[name])
    return {k: dict(v) for k, v in parts.items()}, {k: len(p) for k, (p, _) in progs.items()}


# the plan's points from today's target (docs/vla_frequency_plan.md 6): f2-vla's sizes
# unpacked and packed, at PORTB 1 and 2
ALTS = (("mcols32_act255_lanes64_tmem1.5m", dict(MCOLS=32, ACT_ROWS=255, LANES=64,
                                                  TMEM_WORDS=3 << 19, PACK=False)),
        ("mcols32_pack_act255_lanes64_tmem1.5m", dict(MCOLS=32, ACT_ROWS=255, LANES=64,
                                                       TMEM_WORDS=3 << 19, PACK=True)),
        ("mcols32_pack_act255_lanes64_tmem1.5m_portb2", dict(MCOLS=32, ACT_ROWS=255, LANES=64,
                                                              TMEM_WORDS=3 << 19, PACK=True,
                                                              PORTB=2)))

# the row groups tried (tower R, prefix RP, expert RE, mergers RM): the smallest bound among
# those that compile; groups past the configuration's ACT_ROWS are skipped
ROW_GROUPS = ((128, 255, 31, 64), (128, 128, 31, 64), (128, 64, 31, 32), (64, 64, 31, 32),
              (64, 32, 31, 16), (32, 16, 16, 16), (32, 16, 8, 8), (16, 8, 8, 8), (16, 8, 8, 4),
              (16, 8, 4, 4), (8, 8, 4, 2))

UNITS = ("MXU", "portB", "VPU", "QACT", "QST", "DMA")


def seconds(t, u, mhz, portb) -> float:
    return t.get(u, 0) / (mhz * 1e6) / (portb if u == "portB" else 1)


def report(parts: dict, steps: int, mhz: float, portb: int, title: str) -> dict:
    out = {}
    print(f"\n== {title}")
    tot = collections.Counter()
    for part, t in parts.items():
        k = steps if part == "expert_step" else 1
        sec = {u: seconds(t, u, mhz, portb) for u in UNITS}
        bound = max(sec.values())
        out[part] = {"cycles": {u: int(t.get(u, 0)) for u in UNITS},
                     "vpu_split": {kk[4:]: int(v) for kk, v in t.items()
                                   if kk.startswith("VPU.")},
                     "MACs": int(t.get("MACs", 0)), "instructions": int(t.get("instructions", 0)),
                     "seconds": sec, "seconds_bound": bound, "seconds_serial": sum(sec.values())}
        for u in list(UNITS) + ["MACs"]:
            tot[u] += k * t.get(u, 0)
        for kk, v in t.items():
            if kk.startswith("VPU."):
                tot[kk] += k * v
        vs = ", ".join(f"{kk}={v / (mhz * 1e6):.3f}" for kk, v in
                       sorted(out[part]["vpu_split"].items(), key=lambda kv: -kv[1]))
        print(f"{part:12s}{' x%d' % k if k > 1 else '':4s} " +
              " ".join(f"{u}={sec[u]:.3f}s" for u in UNITS) +
              f"  bound={bound:.3f}s serial={sum(sec.values()):.3f}s  MACs={t.get('MACs', 0) / 1e9:.1f}G"
              f"  [VPU: {vs}]")
    sec = {u: seconds(tot, u, mhz, portb) for u in UNITS}
    out["chunk"] = {"seconds": sec, "bound": max(sec.values()), "serial": sum(sec.values()),
                    "MACs": int(tot["MACs"]), "hz_bound": 1 / max(sec.values()),
                    "hz_serial": 1 / sum(sec.values()),
                    "vpu_split_s": {kk[4:]: tot[kk] / (mhz * 1e6) for kk in tot
                                    if kk.startswith("VPU.")}}
    print("chunk       " + " ".join(f"{u}={sec[u]:.3f}s" for u in UNITS) +
          f"  bound={max(sec.values()):.3f}s serial={sum(sec.values()):.3f}s"
          f"  MACs={tot['MACs'] / 1e12:.3f}T  {1 / sum(sec.values()):.2f}-{1 / max(sec.values()):.2f} Hz")
    return out


def best_rows(spec, cfg, lay):
    best = None
    for R in ROW_GROUPS:
        if max(R[:3]) > cfg.ACT_ROWS:
            continue
        try:
            got = chunk_cost(spec, cfg, lay, *R)
        except CompileError:
            continue
        portb = getattr(cfg, "PORTB", 1) or 1
        b = max(sum((spec.steps if k == "expert_step" else 1) * t.get(u, 0)
                    for k, t in got[0].items()) / (portb if u == "portB" else 1) for u in UNITS)
        if best is None or b < best[0]:
            best = (b, R, got)
    if best is None:
        raise SystemExit("no row group compiles")
    return best[1], best[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="f2-hbm")
    ap.add_argument("--portb", type=int, help="port B chunks a cycle (default the target's)")
    ap.add_argument("--mhz", type=float, default=200.0)
    ap.add_argument("--text", type=int, default=41, help="text tokens in the prompt")
    ap.add_argument("--rows", help="R,RP,RE,RM instead of the search")
    ap.add_argument("--alt", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()
    from opentpu import target as TG
    cfg = TG.config(TG.load(args.target))
    if args.portb:
        cfg = dataclasses.replace(cfg, PORTB=args.portb)
    spec = RV.Spec()
    lay = RV.layout(spec, prompt_ids(spec, args.text))
    res = {"prefix_rows": lay.L}

    def one(name, c, title):
        portb = getattr(c, "PORTB", 1) or 1
        if args.rows:
            R = tuple(int(x) for x in args.rows.split(","))
            parts, sizes = chunk_cost(spec, c, lay, *R)
        else:
            R, (parts, sizes) = best_rows(spec, c, lay)
        r = report(parts, spec.steps, args.mhz, portb, f"{title}, R {R}, PORTB {portb}")
        r["R"], r["program_instructions"], r["PORTB"] = R, sizes, portb
        res[name] = r
    one("target", cfg, f"{args.target}: S={cfg.S} D={cfg.D} MCOLS={cfg.MCOLS} PACK={int(bool(cfg.PACK))} "
        f"ACT_ROWS={cfg.ACT_ROWS} LANES={cfg.LANES} TMEM={cfg.TMEM_WORDS}; {lay.L} prefix rows")
    if args.alt:
        for name, kw in ALTS:
            one(name, dataclasses.replace(cfg, **kw), f"{name}: {kw}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
