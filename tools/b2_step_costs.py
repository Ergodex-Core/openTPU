"""The cost of one decode step of a Ternary-Bonsai 2 model on the ISA simulator, per instruction
kind, at a port width (docs/ternary.md 7; b2-scripts/step_costs.py's method): opentpu.profile's
nominal unit cycles and port-B D-byte chunks per executed instruction, the MMs at their real
chunk and advance counts (t2 QUAD a chunk an advance), tallied over one resident decode step
after four warm-up steps. --portb runs the same step at Config.PORTB 1, 2 and 4 (the compiler
takes the widest column reuse the port allows: docs/wide_port.md); --mcols 8 --rowlanes the
row-lane mode (docs/row_lanes.md: t2 RL 2 at PORTB 2 / MCOLS 8, the MMs at the profile's
ceil(N / L) x ceil(KB / bpc) advances).

    python tools/b2_step_costs.py MODEL_DIR [--target f2-hbm] [--wformat t2] [--head-format t2]
                                  [--rotate | --no-rotate] [--portb 1,2,4] [--mcols 8]
                                  [--rowlanes] [--cap 2048]
                                  [--json OUT.jsonl]

tok/s at --mhz is the port-bound figure: unit cycles a token at the core clock (the RTL runs
within a few percent of it when every instruction is port-bound; docs/ternary.md 6).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opentpu import isa as I  # noqa: E402
from opentpu import profile as P  # noqa: E402
from opentpu import target  # noqa: E402
from opentpu.isasim import Slice  # noqa: E402
from opentpu.llm import bonsai2  # noqa: E402
from opentpu.llm.qwen3 import Engine  # noqa: E402

WARM = [760, 6511, 314, 9338]       # the warm-up tokens (any: the costs do not depend on them)
STEP = 369
FMT = ["int8", "int4", "fp4", "nvfp4", "t2"]


def step_costs(mdir, cfg, wformat: str, head_format: str, rotate: bool, cap: int) -> dict:
    """One decode step's tally: {kind: [count, unit cycles, port-B chunks, port-A transfers]},
    the per-slice totals and the wall time of the step."""
    spec = bonsai2.spec(mdir, ternary=rotate, slices=cfg.S)
    W = bonsai2.Weights(mdir, latent=rotate, ffn=spec.ffn)
    eng = Engine(spec, W, cap=cap, cfg=cfg, backend="isa", wformat=wformat,
                 head_format=head_format, resident=True, rows=1, embed_host=rotate,
                 prompt_runs=False)
    for t in WARM:
        eng.step(t)
    tally = collections.defaultdict(lambda: [0, 0, 0, 0])
    per_slice = collections.defaultdict(lambda: [0, 0])
    orig = Slice.execute

    def ex(self, ins):
        name, _, cyc, pb, pa = P._describe(ins, self.cfg)
        if ins.op == I.MM:
            wfm, pair, quad = I.mm_format(ins)
            L = I.mm_lanes(ins)
            if L:               # MM RL (docs/row_lanes.md): the profile's ceil(N / L) groups
                name = f"MM {FMT[wfm]} rl{L}"      # of ceil(KB / bpc) advances, L chunks each
            else:
                N, KB, M = ins.w[3] & 0xFFFF, (ins.w[3] >> 16) & 0xFF, (ins.w[5] >> 16) & 0xFF
                ch = N * -(-KB // I.blocks_per_chunk(wfm))
                adv = N * -(-KB // I.reuse(pair, quad))
                name = f"MM {FMT[wfm]}{' quad' if quad else ' pair' if pair else ''}"
                cyc, pb = adv * -(-M // self.cfg.MCOLS), ch
        t = tally[name]
        t[0] += 1
        t[1] += cyc
        t[2] += pb
        t[3] += pa
        ps = per_slice[self.sid]
        ps[0] += cyc
        ps[1] += pb
        return orig(self, ins)
    Slice.execute = ex
    t0 = time.perf_counter()
    try:
        eng.step(STEP)
    finally:
        Slice.execute = orig
    wall = time.perf_counter() - t0
    return {"tally": dict(tally), "per_slice": dict(per_slice), "wall_s": wall,
            "image_bytes": int(eng.image.nbytes)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--target", default="f2-hbm")
    ap.add_argument("--wformat", default="t2")
    ap.add_argument("--head-format", default=None)
    ap.add_argument("--rotate", action="store_true",
                    help="the latent ternary weights, the rotation on the device (implied by t2)")
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--portb", default="1")
    ap.add_argument("--mcols", type=int, default=None,
                    help="MXU columns (Config.MCOLS; default the target's)")
    ap.add_argument("--rowlanes", action="store_true",
                    help="row lanes (Config.ROWLANES, docs/row_lanes.md): the image in row-lane "
                         "groups, MM RL of the widest L for --portb / --mcols")
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--mhz", type=float, default=200.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    cfg0 = target.config(target.load(a.target))
    if a.mcols:
        cfg0 = replace(cfg0, MCOLS=a.mcols)
    if a.rowlanes:
        cfg0 = replace(cfg0, ROWLANES=True)
    hf = a.head_format or a.wformat
    rot = (a.rotate or a.wformat == "t2" or hf == "t2") and not a.no_rotate
    for pb in [int(x) for x in a.portb.split(",")]:
        cfg = replace(cfg0, PORTB=pb, PAIR=cfg0.PAIR or pb > 1)
        r = step_costs(a.model, cfg, a.wformat, hf, rot, a.cap)
        tally = r["tally"]
        tot = [sum(v[i] for v in tally.values()) for i in range(4)]
        mm = [sum(v[i] for k, v in tally.items() if k.startswith("MM")) for i in range(4)]
        cyc = max(c for c, _ in r["per_slice"].values())        # the slices run in parallel
        toks = a.mhz * 1e6 / cyc
        print(f"{Path(a.model).name} {a.target} PORTB {pb} MCOLS {cfg.MCOLS}"
              f"{' RL' if cfg.ROWLANES else ''} {a.wformat}/{hf} rot={rot} cap {a.cap}: "
              f"port-B chunks {tot[2] / 1e6:.2f} M (MMs {mm[2] / 1e6:.2f} M), unit cycles "
              f"{tot[1] / 1e6:.2f} M (MMs {mm[1] / 1e6:.2f} M); a slice {cyc / 1e6:.2f} M "
              f"cycles a token = {toks:.1f} tok/s at {a.mhz:g} MHz; port B busy "
              f"{100 * max(b for _, b in r['per_slice'].values()) / pb / cyc:.0f}% "
              f"(image {r['image_bytes'] / 2**20:.0f} MiB, {r['wall_s']:.0f} s)")
        for sid, (c, b) in sorted(r["per_slice"].items()):
            print(f"  slice {sid}: port-B chunks {b / 1e6:.2f} M, unit cycles {c / 1e6:.2f} M")
        for k, v in sorted(tally.items(), key=lambda kv: -max(kv[1][1], kv[1][2]))[:14]:
            print(f"  {k:16s} n={v[0]:6d} cycles={v[1] / 1e6:7.3f}M portB={v[2] / 1e6:7.3f}M "
                  f"portA={v[3] / 1e6:7.3f}M")
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps({"model": Path(a.model).name, "target": a.target,
                                    "portb": pb, "mcols": cfg.MCOLS,
                                    "rowlanes": bool(cfg.ROWLANES),
                                    "wformat": a.wformat, "head_format": hf,
                                    "rotate": rot, "cap": a.cap, "cycles": cyc,
                                    "chunks": tot[2], "unit_cycles": tot[1],
                                    "mm_chunks": mm[2], "mm_cycles": mm[1],
                                    "tok_s": toks, "mhz": a.mhz,
                                    "image_bytes": r["image_bytes"],
                                    "tally": {k: v for k, v in tally.items()}}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
