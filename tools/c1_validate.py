#!/usr/bin/env python3
"""Validate a C1 workload against canonical ISA execution and export Lens policy profiles.

Run from the repository: python -m tools.c1_validate --output build/lens/c1
The CPU reference duration is separate from the explicitly analytic accelerator cycles.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from opentpu import Config, isa as I, quant as Q, lens
from opentpu import c1_rows as C
from opentpu.c1_profile import Timing, profile
from opentpu.isasim import Slice


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("build/lens/c1"))
    p.add_argument("--rows", type=int, default=257)
    p.add_argument("--blocks", type=int, default=9)
    p.add_argument("--seed", type=int, default=41)
    args = p.parse_args(argv)
    if not 1 <= args.rows <= 65535 or not 1 <= args.blocks <= 126:
        p.error("rows must be 1..65535 and blocks 1..126")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    w, scales = Q.quantize_mxu(rng.normal(size=(args.rows, args.blocks * 128)).astype(np.float32),
                              "int8", 128)
    wa = 128
    sa = (wa + w.size + 127) // 128 * 128
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128,
                 TMEM_WORDS=max(65536, args.rows + 20000),
                 DRAM_BYTES=sa + scales.size + 128)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[wa:wa + w.size] = w.reshape(-1)
    dram[sa:sa + scales.size] = scales.reshape(-1)
    source = Slice(cfg, 0, [], dram)
    x = rng.normal(size=args.blocks * 128).astype(np.float32)
    source.tput(np.arange(x.size), x)
    source.execute(I.qact(0, 1, 2, args.blocks, x.size))
    command = C.Command(0, args.rows, 0, args.blocks, 2, 18000)
    source.execute(I.mm(wa, sa, command.out, command.n, command.kb, w.shape[1], command.n,
                        1, command.ab, scales.shape[1]))
    expected = source.tmem[command.out:command.out + command.n].copy()
    profiles, validation = [], []
    for engines in (1, 4, 16, 32):
        image = C.pack(w, scales, engines=engines)
        ref = Slice(cfg, 0, [], dram)
        ref.act[:] = source.act
        ref.ascale[:] = source.ascale
        ref.adoff[:] = source.adoff
        begin = time.perf_counter()
        values = C.execute(image, command, ref)
        host_ms = (time.perf_counter() - begin) * 1000
        if not np.array_equal(values.view(np.uint32), expected):
            raise RuntimeError(f"C1 {engines} engines differs from canonical ISA")
        validation.append({"engines": engines, "outputs_bit_exact": command.n,
                           "cpu_reference_wall_ms": host_ms})
        item = profile(image.descriptor, command, cfg, name=f"C1 E{engines}; N={args.rows} KB={args.blocks}")
        item["c1_validation"] = validation[-1]
        profiles.append(item)
        if engines == 16:
            image.save(args.output / "validated-weights.c1rows")
            delayed = profile(image.descriptor, command, cfg,
                              name=f"C1 E16; PC0 +160 cycles/row; N={args.rows} KB={args.blocks}",
                              extra_pc_latency={0: 160})
            delayed["c1_validation"] = validation[-1]
            profiles.append(delayed)
    bundle = lens.save(profiles, args.output / "c1-validation.otpuprof")
    (args.output / "c1-validation.html").write_text(lens.render(lens.load(bundle)))
    report = {"status": "bit-exact reference validation; analytic timing only", "seed": args.seed,
              "rows": args.rows, "blocks": args.blocks, "validation": validation,
              "profiles": [{"name": x["name"], "kind": x["kind"], "cycles": x["cycles"]}
                           for x in profiles]}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Lens: {bundle}")


if __name__ == "__main__":
    main()
