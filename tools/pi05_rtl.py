"""One pi0.5 phase program of a cut model on the RTL (Verilator, the target's board uarch and
memory path, as tools/perf_qwen.py), its cycles against tools/pi05_cost.py's tally of the
same program: the calibration of the cost estimate (docs/pi05.md 5).

    OTPU_TARGET=f2-hbm python tools/pi05_rtl.py --ckpt DIR [--phase e_step] [--np 558]
                                                [--r 16 --rp 8 --re 8]

The cut model has one layer of each part (the real dimensions); the weights are the
checkpoint's first layers, the activations zeros (timing does not depend on the values:
no RLD, no data-dependent control). Prints the RTL's cycles, the tally's per-unit cycles
and the ratio of the cycles to the slowest unit's (1.0: every unit overlapped). --r / --rp
/ --re: the row groups (the MMs' M; Image.programs R / RP / RE), e.g. --rp 32 for an MM of
32 rows a pass. OTPU_MCOLS / OTPU_PACK / OTPU_ACT_ROWS pick the MXU (board_config): the
packed MXU (docs/mxu_packing.md 7) against the unpacked one of twice the columns.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from opentpu import rtlsim  # noqa: E402
from opentpu.compiler import arg_words  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import pi05 as P  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--phase", default="e_step")
    ap.add_argument("--np", type=int, default=558)
    ap.add_argument("--r", type=int, default=None, help="the tower's row group (Image.programs R)")
    ap.add_argument("--rp", type=int, default=None, help="the prefix's row group (RP)")
    ap.add_argument("--re", type=int, default=None, help="the expert's row group (RE)")
    ap.add_argument("--tmem", type=int, default=0,
                    help="TMEM words (default the target's; larger row groups need more)")
    ap.add_argument("--args", default="", help="run arguments, name=value,...")
    ap.add_argument("--rows", default="16,8,8",
                    help="the row groups R,RP,RE (tower, prefix, expert; tools/pi05_cost.py's "
                         "pick for the configuration: 16,8,8 at f2-hbm, 128,255,64 at f2-vla)")
    ap.add_argument("--no-qst-t", action="store_true",
                    help="the V^T appends a byte a cycle (no QST T): the measurement before it")
    ap.add_argument("--profile", action="store_true",
                    help="a trace run (opentpu.profile): the units' busy cycles, the MXU's "
                         "pops, starved and blocked cycles, the TMEM losses, the MXU gaps "
                         "blamed on their dependencies")
    a = ap.parse_args()
    import pi05_cost as PCO
    if a.no_qst_t:
        import opentpu.compiler as C
        C.QST_T = False
    spec = dataclasses.replace(P.Spec.from_ckpt(a.ckpt), v_layers=1, layers=1, e_layers=1,
                               steps=1)
    kw = {"TMEM_WORDS": a.tmem} if a.tmem else {}
    image = P.Image(spec, board_config(**kw))
    cfg = board_config(DRAM_BYTES=1 << (image.end - 1).bit_length(), **kw)
    image.cfg = cfg
    img = np.zeros(image.end, np.uint8)

    def put(addr, x):
        v = np.ascontiguousarray(x).view(np.uint8).reshape(-1)
        img[addr:addr + v.size] = v
    image.build(P.Weights(a.ckpt), put)
    R, RP, RE = (int(x) for x in a.rows.split(","))
    R, RP, RE = (a.r or R), (a.rp or RP), (a.re or RE)        # --r/--rp/--re override --rows
    print(f"{cfg}; row groups {R}/{RP}/{RE}; uarch {rtlsim.BOARD_UARCH}")
    prog, ra = image.programs(a.np, R=R, RP=RP, RE=RE)[a.phase]
    vals = {k: int(v) for k, v in (kv.split("=") for kv in a.args.split(",") if kv)}
    vals = {v.name: vals.get(v.name, 0) for v, _ in ra}
    t = PCO.tally(prog, cfg)
    print(f"{a.phase}: {len(prog)} instructions, tally " +
          " ".join(f"{u}={t.get(u, 0)}" for u in PCO.UNITS))
    t0 = time.time()
    _, _, st = rtlsim.run(cfg, [prog], [img], uarch=rtlsim.BOARD_UARCH, axi=True, boot=True,
                          max_cycles=1 << 40, args=arg_words(ra, vals), trace=a.profile)
    cyc = st["cycles"]
    slow = max(t.get(u, 0) for u in PCO.UNITS)
    print(f"RTL: {cyc} cycles ({time.time() - t0:.0f} s simulated); slowest unit {slow}: "
          f"ratio {cyc / slow:.3f}; serial sum {sum(t.get(u, 0) for u in PCO.UNITS)}")
    mx = t.get("MXU", 0)
    print(f"MXU utilization (tally MXU / RTL cycles): {100 * mx / cyc:.1f}%; the MXU's pops: "
          + ", ".join(f"{d['pop']} of {d['cyc']} cycles ({100 * d['pop'] / d['cyc']:.1f}%)"
                      for d in st["act_reads"]))
    if a.profile:
        from opentpu import profile as PF
        pr = PF.parse(st["trace"], cfg, [prog], a.phase)
        print(pr.summary())
        b = pr.buckets[0]
        tot = {k: sum(v) for k, v in b.items() if k not in ("c", "n")}
        print("  P/Q windows (cycles): " + " ".join(f"{k}={v}" for k, v in tot.items()))
        print("  (bm bd: port B busy for the MXU / DMA, am aq: port A for the MXU / QST, mx: "
              "MXU pops, fm fq fv fc: the MXU drain / QUANT / VPU / COLL lost to TMEM "
              "arbitration; bs as: port B / A requests waiting, ms mb: the MXU starved / "
              "blocked, ff: summed FIFO level, ld: loader)")
        g = pr.mxu_gaps(0)
        gaps = sum(e - s for s, e, _ in g["gaps"])
        print(f"  MXU gaps: prologue {g['prologue']}, epilogue {g['epilogue']}, "
              f"{len(g['gaps'])} gaps of {gaps} cycles; blamed on:")
        for (nm, pc), c in sorted(g["blame"].items(), key=lambda kv: -kv[1])[:8]:
            ins = prog[pc] if pc >= 0 else None
            print(f"    {c:9d}  {nm:12s} pc={pc} {ins.comment if ins else ''}")
        cls = pr.by_class(0)
        print("  by class (count, busy, work, wait_dep, wait_unit):")
        for k, v in sorted(cls.items(), key=lambda kv: -kv[1]["busy"])[:10]:
            print(f"    {k:14s} {v['count']:6d} {v['busy']:10d} {v['work']:10d} "
                  f"{v['wait_dep']:10d} {v['wait_unit']:10d}")
        for line in st["trace"].splitlines():
            if line.startswith("TMEMSTAT"):
                print("  " + line)


if __name__ == "__main__":
    main()
