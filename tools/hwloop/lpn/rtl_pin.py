#!/usr/bin/env python3
"""The pinned RTL on an LPN input bundle: Verilator on the tree LPNDialect is validated against
(f237dab2 + its observer patch), the testbench's default micro-architecture and behavioral DRAM
(otpu_dram.sv, --dram-lat cycles): the model's own validated configuration, so the LPN cycles
of the same bundle are graded against it as upstream grades them. Every DRAM byte and TMEM
word is checked against the pin's ISA simulator, and the DRAM / TMEM hashes against the
export's (workload.json), so the two trees' ISA simulators agree on the program too.

Runs under the venv python with the pin tree on sys.path:

    python tools/hwloop/lpn/rtl_pin.py --tree PIN --input DIR --out DIR [--dram-lat 8] [--jobs 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dram-lat", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-trace", action="store_true", help="no instruction trace (faster; no command-PC check)")
    a = ap.parse_args(argv)
    tree = Path(a.tree).resolve()
    sys.path.insert(0, str(tree))
    import numpy as np
    import opentpu
    from opentpu import isa as I
    from opentpu import rtlsim
    from opentpu.isasim import Config, Machine
    if not Path(opentpu.__file__).resolve().is_relative_to(tree):
        raise SystemExit(f"opentpu imported from {opentpu.__file__}, not from {tree}")

    inp, out = Path(a.input), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((inp / "input.json").read_text())
    cfg = Config(**meta["config"])
    programs, images = [], []
    for s in range(cfg.S):
        words = [int(x, 16) for x in (inp / f"prog_{s}.hex").read_text().split()]
        n = meta["instruction_counts"][s] * 8
        words += [0] * (n - len(words))
        programs.append([I.Instr.decode(words[i:i + 8]) for i in range(0, n, 8)])
        images.append(np.fromfile(inp / f"dram_{s}.bin", dtype=">u4").astype("<u4").view(np.uint8))
    rtlsim.VFLAGS[rtlsim.VFLAGS.index("-j") + 1] = str(a.jobs)
    if "-O3" not in rtlsim.VFLAGS:
        rtlsim.VFLAGS += ["-CFLAGS", "-O3"]
    t0 = time.monotonic()
    exe = rtlsim.build_top(cfg, dram_lat=a.dram_lat)
    build_s = time.monotonic() - t0
    keep = out / "rtl"
    t1 = time.monotonic()
    drams, tmems, st = rtlsim.run(cfg, programs, images, dram_lat=a.dram_lat, trace=not a.no_trace, keep=keep,
                                  axi=False, max_cycles=1 << 40, args=meta.get("args") or None)
    run_s = time.monotonic() - t1
    (out / "rtl.log").write_text(st.get("trace", ""))
    same_inputs = {f: hashlib.sha256((keep / f).read_bytes()).hexdigest() == hashlib.sha256((inp / f).read_bytes()).hexdigest()
                   for f in (f"prog_{s}.hex" for s in range(cfg.S))}
    t2 = time.monotonic()
    golden = Machine(cfg, programs, [im.copy() for im in images], args=meta.get("args") or None)
    golden.run(max_steps=1 << 40)
    isa_s = time.monotonic() - t2
    exact = all(np.array_equal(drams[s], golden.slices[s].dram) and np.array_equal(tmems[s], golden.slices[s].tmem)
                for s in range(cfg.S))
    pin_hash = {"dram_sha256": hashlib.sha256(np.ascontiguousarray(golden.slices[0].dram, np.uint8)).hexdigest(),
                "tmem_sha256": hashlib.sha256(np.ascontiguousarray(golden.slices[0].tmem, np.uint32)).hexdigest()}
    export = {}
    if (inp / "workload.json").is_file():
        export = json.loads((inp / "workload.json").read_text()).get("isa", {})
    pcs = [int(x) for x in re.findall(r"T0 D c=\d+ s=\d+ pc=(\d+)", st.get("trace", ""))] if not a.no_trace else None
    rec = {"label": "measured", "kind": "verilator_rtl_pinned_config", "tree": str(tree),
           "revision": __import__("subprocess").check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip(),
           "verilator": rtlsim.verilator_version(), "vflags": rtlsim.VFLAGS, "dram_lat": a.dram_lat, "uarch": "tb_top default",
           "memory": "behavioral otpu_dram.sv, axi=False", "config": meta["config"], "cycles": st["cycles"],
           "instructions_dynamic": st.get("instructions"), "build_s": build_s, "run_s": run_s, "isa_s": isa_s,
           "rtl_matches_pin_isa_simulator": bool(exact), "identical_prog_hex": same_inputs,
           "pin_isa": pin_hash, "export_isa": {k: export.get(k) for k in ("dram_sha256", "tmem_sha256")},
           "pin_isa_matches_export_isa": (pin_hash["dram_sha256"] == export.get("dram_sha256")
                                          and pin_hash["tmem_sha256"] == export.get("tmem_sha256")) if export else None,
           "command_pcs_traced": len(pcs) if pcs is not None else None,
           "binary_sha256": hashlib.sha256(Path(exe).read_bytes()).hexdigest()}
    (out / "rtl_pin.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("vflags",)}, indent=2, default=str))


if __name__ == "__main__":
    main()
