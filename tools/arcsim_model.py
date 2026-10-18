#!/usr/bin/env python3
"""Build (or find) the simulator's model of a board configuration, as tools/perf_qwen.py and
tools/perf_bonsai2.py build it before a run, and print its path and build times: the one-command
build of docs/arcsim.md. The target is the environment's (OTPU_TARGET, OTPU_NCH, OTPU_SIM=arcsim
for an ArcSim model; e.g. OTPU_SIM=arcsim OTPU_TARGET=f2-hbm OTPU_NCH=8), the architecture knobs
are perf_qwen's (--mcols, --portb, --rowlanes, --aline), the memory path the board's (the native
model behind the adapter, boot).

  OTPU_SIM=arcsim OTPU_TARGET=f2-hbm OTPU_NCH=8 python tools/arcsim_model.py --mcols 8 --portb 4 --rowlanes --aline 64 --dram-bytes 1073741824
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu import rtlsim, target  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcols", type=int, default=None, help="MXU columns (default the target's)")
    ap.add_argument("--portb", type=int, default=None, choices=[1, 2, 4], help="port B chunks a cycle")
    ap.add_argument("--rowlanes", action="store_true", help="MM RL row lanes")
    ap.add_argument("--aline", type=int, default=None, choices=[32, 64], help="port A's line, bytes")
    ap.add_argument("--imem", type=int, default=None, help="IMEM words (a tool grows it for a long program)")
    ap.add_argument("--dram-bytes", type=int, default=None,
                    help="the simulated DRAM (default the target's); perf_qwen / perf_bonsai2 size it to "
                         "the image's next power of two (Gemma-3-1B, 4 layers, int8: 1073741824)")
    ap.add_argument("--lat", type=int, default=20, help="DRAM_LAT (timing of the behavioural DRAM only)")
    a = ap.parse_args()
    cfg = target.config()
    mk: dict = {}
    if a.mcols:
        mk["MCOLS"] = a.mcols
    if a.portb:
        mk.update(PORTB=a.portb, PAIR=True)
    if a.rowlanes:
        mk["ROWLANES"] = True
    if a.aline:
        mk["ALINE"] = a.aline
    if a.imem:
        mk["IMEM_WORDS"] = a.imem
    if a.dram_bytes:
        mk["DRAM_BYTES"] = a.dram_bytes
    cfg = dataclasses.replace(cfg, **mk)
    params = rtlsim.top_params(cfg, a.lat, rtlsim.BOARD_UARCH, True)
    print(f"simulator {rtlsim.SIM} ({rtlsim.arcsim_version() if rtlsim.SIM == 'arcsim' else rtlsim.verilator_version()})")
    print(f"target {target.current().name} params {params}")
    t = time.time()
    exe = rtlsim.build_top(cfg, a.lat, rtlsim.BOARD_UARCH, axi=True)
    print(f"model {exe} ({time.time() - t:.1f} s{' (cached)' if time.time() - t < 1 else ''})")
    times = exe.parent / "build_times.json"
    if times.exists():
        print("build times (s):", json.loads(times.read_text()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
