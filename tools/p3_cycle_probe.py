"""v4 P3 (docs/v4/proposal-3-eight-lanes-packed.md): the cycle probe. A fixed set of RTL runs --
the mlp and attention kernels, two random programs, an MM-replay program, a random program at the
card's MCOLS 4 / D 128, two programs on the board's memory path and one on the card's uarch --
each recorded as its cycle count and the hashes of its DRAM and TMEM, as JSON. Run on two trees
(--root: the tree whose opentpu and tests/ helpers are imported) with the same --uarch: equal
records mean the change is cycle-exact there (P2's probe set, docs/v4/p2-status.md).

    python tools/p3_cycle_probe.py --root <tree> [--uarch MXU_PACK=1 ...] -o out.json
    python tools/p3_cycle_probe.py --compare a.json b.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def compare(a: Path, b: Path) -> int:
    ra, rb = json.loads(a.read_text()), json.loads(b.read_text())
    bad = 0
    for k in sorted(set(ra) | set(rb)):
        x, y = ra.get(k), rb.get(k)
        same = x == y
        state = x is not None and y is not None and (x["dram"], x["tmem"]) == (y["dram"], y["tmem"])
        bad += not same
        print(f"{k:14s} {'same' if same else 'DIFF'}  cycles {x and x['cycles']} / "
              f"{y and y['cycles']}  state {'same' if state else 'DIFF'}")
    print(f"{len(set(ra) | set(rb)) - bad} of {len(set(ra) | set(rb))} records equal")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="the tree to import (opentpu, tests/)")
    ap.add_argument("--uarch", action="append", default=[], metavar="K=V")
    ap.add_argument("-o", "--out")
    ap.add_argument("--compare", nargs=2, type=Path)
    a = ap.parse_args(argv)
    if a.compare:
        return compare(*a.compare)
    root = Path(a.root).resolve()
    sys.path[:0] = [str(root), str(root / "tests")]
    from opentpu import Config, rtlsim
    from opentpu.kernels import attention_decode, mlp
    from opentpu.runtime import compile_kernel
    import test_rtl as T
    from test_kernels import attn_args, mlp_args
    ua = {k: int(v) for k, v in (x.split("=") for x in a.uarch)}
    h = lambda xs: hashlib.sha1(b"".join(np.ascontiguousarray(x).tobytes() for x in xs)).hexdigest()[:16]
    rec = {}

    def go(name, cfg, progs, imgs, uarch=None, **kw):
        u = {**(uarch or {}), **ua}
        drams, tmems, st = rtlsim.run(cfg, progs, [i.copy() for i in imgs], uarch=u, **kw)
        rec[name] = {"cycles": int(st["cycles"]), "dram": h(drams), "tmem": h(tmems)}
        print(name, rec[name], flush=True)

    args, _ = mlp_args(np.random.default_rng(1), M=11)
    comp, imgs = compile_kernel(mlp, Config(S=1), **args)
    go("mlp", comp.cfg, comp.programs, imgs)
    args, _ = attn_args(np.random.default_rng(100), Hq=8, Hkv=2, T=100)
    comp, imgs = compile_kernel(attention_decode, Config(S=1), **args)
    go("attn", comp.cfg, comp.programs, imgs)
    for seed in (0, 1):
        rng = np.random.default_rng(7000 + seed)
        cfg = Config(S=1)
        go(f"fuzz{seed}", cfg, [T._random_program(rng, cfg)], T._images(rng, 1))
    rng = np.random.default_rng(3004)
    cfg = Config(S=1, MCOLS=4, ACT_ROWS=16)
    go("replay", cfg, [T._random_program(rng, cfg, n_ops=60)], T._images(rng, 1))
    rng = np.random.default_rng(6202)
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=16)
    go("mcols4", cfg, [T._random_program(rng, cfg)], T._images(rng, 1), uarch={"MXU_E": 4})
    for kind, seed, stall, lat in (("hazard", 0, 0, 20), ("random", 3, 40, 20)):
        rng = np.random.default_rng(6100 + seed)
        cfg = Config(S=1, D=128, ACT_BLOCKS=16)
        p = (T._hazard_program if kind == "hazard" else T._random_program)(rng, cfg)
        go(f"board_{kind}", cfg, [p], T._images(rng, 1), uarch={"MXU_E": 4}, axi=True,
           boot=True, stall=stall, seed=seed + 1, lat=lat)
    rng = np.random.default_rng(6101)
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    p = T._hazard_program(rng, cfg, gather=False)
    go("card", cfg, [p], T._images(rng, 1), uarch=dict(rtlsim.BOARD_UARCH), axi=True, boot=True,
       stall=50, seed=2)
    rng = np.random.default_rng(6103)
    cfg = Config(S=1, D=128, MCOLS=4, ACT_ROWS=16, ACT_BLOCKS=16)
    p = T._hazard_program(rng, cfg, gather=False)
    go("card_mcols4", cfg, [p], T._images(rng, 1), uarch=dict(rtlsim.BOARD_UARCH), axi=True,
       boot=True, stall=20, seed=3)
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
