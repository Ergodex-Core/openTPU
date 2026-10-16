"""One pi0.5 phase program of a cut model on the RTL (Verilator, the target's board uarch and
memory path, as tools/perf_qwen.py), its cycles against tools/pi05_cost.py's tally of the
same program: the calibration of the cost estimate (docs/pi05.md 5).

    OTPU_TARGET=f2-hbm python tools/pi05_rtl.py --ckpt DIR [--phase e_step] [--np 558]

The cut model has one layer of each part (the real dimensions); the weights are the
checkpoint's first layers, the activations zeros (timing does not depend on the values:
no RLD, no data-dependent control). Prints the RTL's cycles, the tally's per-unit cycles
and the ratio of the cycles to the slowest unit's (1.0: every unit overlapped).
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
    ap.add_argument("--args", default="", help="run arguments, name=value,...")
    ap.add_argument("--rows", default="16,8,8",
                    help="the row groups R,RP,RE (tower, prefix, expert; tools/pi05_cost.py's "
                         "pick for the configuration: 16,8,8 at f2-hbm, 128,255,64 at f2-vla)")
    a = ap.parse_args()
    import pi05_cost as PCO
    spec = dataclasses.replace(P.Spec.from_ckpt(a.ckpt), v_layers=1, layers=1, e_layers=1,
                               steps=1)
    image = P.Image(spec, board_config())
    cfg = board_config(DRAM_BYTES=1 << (image.end - 1).bit_length())
    image.cfg = cfg
    img = np.zeros(image.end, np.uint8)

    def put(addr, x):
        v = np.ascontiguousarray(x).view(np.uint8).reshape(-1)
        img[addr:addr + v.size] = v
    image.build(P.Weights(a.ckpt), put)
    R, RP, RE = (int(x) for x in a.rows.split(","))
    print(f"{cfg}; row groups {R}/{RP}/{RE}; uarch {rtlsim.BOARD_UARCH}")
    prog, ra = image.programs(a.np, R=R, RP=RP, RE=RE)[a.phase]
    vals = {k: int(v) for k, v in (kv.split("=") for kv in a.args.split(",") if kv)}
    vals = {v.name: vals.get(v.name, 0) for v, _ in ra}
    t = PCO.tally(prog, cfg)
    print(f"{a.phase}: {len(prog)} instructions, tally " +
          " ".join(f"{u}={t.get(u, 0)}" for u in PCO.UNITS))
    t0 = time.time()
    _, _, st = rtlsim.run(cfg, [prog], [img], uarch=rtlsim.BOARD_UARCH, axi=True, boot=True,
                          max_cycles=1 << 40, args=arg_words(ra, vals))
    cyc = st["cycles"]
    slow = max(t.get(u, 0) for u in PCO.UNITS)
    print(f"RTL: {cyc} cycles ({time.time() - t0:.0f} s simulated); slowest unit {slow}: "
          f"ratio {cyc / slow:.3f}; serial sum {sum(t.get(u, 0) for u in PCO.UNITS)}")


if __name__ == "__main__":
    main()
