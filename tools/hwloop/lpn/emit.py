#!/usr/bin/env python3
"""Build and compile the LPN model of an input bundle (LPNDialect examples/opentpu) with a
chosen Timing, and check its ISA frontend's numerical result against the bundle's reference.

Runs under the venv python with LPNDialect on PYTHONPATH (the clone is the library; this file
is the loop's driver, kept in this worktree) and the pinned openTPU tree as --opentpu-root:

    PYTHONPATH=$LPN python tools/hwloop/lpn/emit.py --opentpu-root PIN --input DIR --out DIR \
        --lpn-compiler $LPN/build/tools/lpn-compile/lpn-compile [--compact] [--batch-rows 8] \
        [--dram-latency 8] [--dram-bytes-per-cycle 128] [--matrix-engines 1]

What upstream's run.py does (use_checkout, load, CompactOpenTPU / OpenTPU, Timing, build,
lpn-compile), with the Timing's DRAM latency / bandwidth as arguments (upstream's validated
values are the defaults: anything else is outside the model's evidence, and the receipt says
so), the net statistics, the matrix batch sizes, and the frontend's DRAM / TMEM SHA-256 for
the check against the export's ISA simulator (workload.json).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opentpu-root", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lpn-compiler", required=True)
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--batch-rows", type=int, default=8)
    ap.add_argument("--dram-latency", type=int, default=8)
    ap.add_argument("--dram-bytes-per-cycle", type=int, default=128)
    ap.add_argument("--matrix-engines", type=int, default=1)
    a = ap.parse_args(argv)

    import numpy as np
    from examples.opentpu.model import OpenTPU, Timing
    from examples.opentpu.workload import load, use_checkout
    import examples.opentpu as pkg

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = use_checkout(a.opentpu_root)
    t0 = time.monotonic()
    workload = load(a.input)
    frontend_s = time.monotonic() - t0
    sl = workload.machine.slices
    frontend = {"wall_s": frontend_s, "dynamic_commands": [len(c) for c in workload.commands],
                "dram_sha256": [hashlib.sha256(np.ascontiguousarray(s.dram, np.uint8)).hexdigest() for s in sl],
                "tmem_sha256": [hashlib.sha256(np.ascontiguousarray(s.tmem, np.uint32)).hexdigest() for s in sl],
                "input_hashes": workload.hashes}
    kw = dict(batch_rows=a.batch_rows, dram_bytes_per_cycle=min(workload.config.D, a.dram_bytes_per_cycle),
              dram_latency=a.dram_latency)
    if a.matrix_engines != 1:
        kw["matrix_engines"] = a.matrix_engines
    timing = Timing(**kw)
    t1 = time.monotonic()
    if a.compact:
        from examples.opentpu.compact import CompactOpenTPU
        model = CompactOpenTPU(workload, timing)
    else:
        model = OpenTPU(workload, timing)
    ir = out / "model.mlir"
    ir.write_text(model.build())
    build_s = time.monotonic() - t1
    stats = model.statistics() if hasattr(model, "statistics") else None
    t2 = time.monotonic()
    comp = subprocess.run([str(Path(a.lpn_compiler).resolve()), str(ir), "-o", str(out / "opentpu")],
                          capture_output=True, text=True)
    (out / "compile.log").write_text(comp.stdout + comp.stderr)
    rec = {"opentpu_root": str(root), "revision": pkg.REVISION, "timing": dataclasses.asdict(timing),
           "compact": a.compact, "matrix_batch_rows": getattr(model, "matrix_batch_sizes", None),
           "net_statistics": stats, "frontend": frontend, "model_build_s": build_s,
           "compile_s": time.monotonic() - t2, "compile_exit": comp.returncode,
           "mlir_sha256": hashlib.sha256(ir.read_bytes()).hexdigest(),
           "binary_sha256": hashlib.sha256((out / "opentpu").read_bytes()).hexdigest() if comp.returncode == 0 else None}
    (out / "emit.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(json.dumps(rec, indent=2, default=str))
    sys.exit(comp.returncode)


if __name__ == "__main__":
    main()
