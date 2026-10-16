#!/usr/bin/env python3
"""Build and compile the LPN model of an input bundle (LPNDialect examples/opentpu) with a
chosen Timing, and check its ISA frontend's numerical result against the bundle's reference.

Runs under the venv python with LPNDialect on PYTHONPATH (the clone is the library; this file
is the loop's driver, kept in this worktree) and an openTPU tree as --opentpu-root:

    PYTHONPATH=$LPN python tools/hwloop/lpn/emit.py --opentpu-root PIN --input DIR --out DIR \
        --lpn-compiler $LPN/build/tools/lpn-compile/lpn-compile [--compact] [--batch-rows 8] \
        [--dram-latency 8] [--dram-bytes-per-cycle 128] [--matrix-engines 1]
    PYTHONPATH=$LPN python tools/hwloop/lpn/emit.py --opentpu-root EXPORT --frontend export --wide \
        --portb 2 --nch 8 --fifo-entries 1024 --dram-latency 60 --adapter-latency 14 ...

What upstream's run.py does (use_checkout, load, CompactOpenTPU / OpenTPU, Timing, build,
lpn-compile), with the Timing's DRAM latency / bandwidth as arguments (upstream's validated
values are the defaults: anything else is outside the model's evidence, and the receipt says
so), the net statistics, the matrix batch sizes, and the frontend's DRAM / TMEM SHA-256 for
the check against the export's ISA simulator (workload.json).

`--frontend pin` (default): upstream's workload.load on the pinned tree (its revision check).
`--frontend export`: lpn/wide.py's load_export on the export tree's own ISA simulator (PORTB,
int8 PAIR / QUAD, MM RL, MCOLS 8: the wide-port extensions' frontend), with `--wide` selecting
WideCompactOpenTPU and WideTiming (port_chunks, nch, fifo entries, channel depth, the path's
latency); memory.json then carries the static memory tally (wide / two-part reads, per-channel
beats, the rooflines).
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
    ap.add_argument("--frontend", choices=["pin", "export"], default="pin")
    ap.add_argument("--wide", action="store_true", help="lpn/wide.py's model (the wide port, the row lanes, the channels)")
    ap.add_argument("--portb", type=int, default=1)
    ap.add_argument("--nch", type=int, default=1)
    ap.add_argument("--no-chash", action="store_true")
    ap.add_argument("--fifo-entries", type=int, default=128)
    ap.add_argument("--channel-depth", type=int, default=64)
    ap.add_argument("--adapter-latency", type=int, default=4)
    ap.add_argument("--epilogue-lanes", type=int, default=2)
    a = ap.parse_args(argv)

    import numpy as np
    from examples.opentpu.model import OpenTPU, Timing
    import examples.opentpu as pkg

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if a.frontend == "pin":
        from examples.opentpu.workload import load, use_checkout
        root = use_checkout(a.opentpu_root)
        workload = load(a.input)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import wide
        root = Path(a.opentpu_root).resolve()
        workload = wide.load_export(a.input, root)
    frontend_s = time.monotonic() - t0
    sl = workload.machine.slices
    frontend = {"wall_s": frontend_s, "dynamic_commands": [len(c) for c in workload.commands],
                "dram_sha256": [hashlib.sha256(np.ascontiguousarray(s.dram, np.uint8)).hexdigest() for s in sl],
                "tmem_sha256": [hashlib.sha256(np.ascontiguousarray(s.tmem, np.uint32)).hexdigest() for s in sl],
                "input_hashes": workload.hashes, "frontend": a.frontend, "root": str(root)}
    kw = dict(batch_rows=a.batch_rows, dram_bytes_per_cycle=min(workload.config.D, a.dram_bytes_per_cycle),
              dram_latency=a.dram_latency)
    if a.matrix_engines != 1:
        kw["matrix_engines"] = a.matrix_engines
    t1 = time.monotonic()
    memory = None
    if a.wide:
        import wide
        from opentpu import isa as I
        timing = wide.WideTiming(port_chunks=a.portb, nch=a.nch, chash=not a.no_chash, fifo_chunks=a.fifo_entries,
                                 channel_depth=a.channel_depth, adapter_latency=a.adapter_latency,
                                 epilogue_lanes=a.epilogue_lanes, **kw)
        model = wide.WideCompactOpenTPU(workload, timing)
        memory = wide.memory_report(workload, timing, I)
        (out / "memory.json").write_text(json.dumps(memory, indent=2) + "\n")
    else:
        timing = Timing(**kw)
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
    rec = {"opentpu_root": str(root), "revision": pkg.REVISION if a.frontend == "pin" else None,
           "timing": dataclasses.asdict(timing), "model": type(model).__name__,
           "compact": a.compact or a.wide, "matrix_batch_rows": getattr(model, "matrix_batch_sizes", None),
           "net_statistics": stats, "frontend": frontend, "model_build_s": build_s,
           "compile_s": time.monotonic() - t2, "compile_exit": comp.returncode,
           "mlir_sha256": hashlib.sha256(ir.read_bytes()).hexdigest(),
           "binary_sha256": hashlib.sha256((out / "opentpu").read_bytes()).hexdigest() if comp.returncode == 0 else None,
           "memory": {k: v for k, v in memory.items() if k != "by_lanes"} | {"by_lanes": memory["by_lanes"]} if memory else None}
    (out / "emit.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(json.dumps(rec, indent=2, default=str))
    sys.exit(comp.returncode)


if __name__ == "__main__":
    main()
