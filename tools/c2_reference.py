#!/usr/bin/env python3
"""Run C2's exact reference and export executed events to openTPU Lens.

    PYTHONPATH=. python tools/c2_reference.py --out-dir build/lens/c2

These are synthetic int8 microbenchmarks using the repository quantizer/QACT,
not a Gemma/F2 measurement. Lens's time axis represents serialized reference
arithmetic slots; its display at 200 MHz is hypothetical.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from opentpu import c2, isa as I, lens, quant as Q
from opentpu.isasim import Config, Slice


def run_case(kb: int, rows: int, acc: bool, seed: int):
    rng = np.random.default_rng(seed)
    n0, k_begin, ab = 4, 3, 2
    full_rows, full_kb = n0 + rows, k_begin + kb
    raw = rng.standard_normal((full_rows, full_kb * 128)).astype(np.float32)
    raw *= np.exp2(np.arange(full_rows, dtype=np.float32)[:, None] % 7 - 3)
    weights, scales = Q.quantize_mxu(raw, "int8")
    x = rng.standard_normal((1, kb * 128)).astype(np.float32)
    sa, rs = 128, weights.shape[1]
    ssa = sa + weights.size
    dram_bytes = ((ssa + scales.size + 127) // 128) * 128
    cfg = Config(D=128, MCOLS=4, ACT_BLOCKS=ab + kb, DRAM_BYTES=dram_bytes,
                 TMEM_WORDS=max(65536, x.size + 128 + rows))
    dram = np.zeros(dram_bytes, np.uint8)
    dram[sa:sa + weights.size] = weights.reshape(-1)
    dram[ssa:ssa + scales.size] = scales.reshape(-1)
    sl = Slice(cfg, 0, [], dram)
    sl.tput(np.arange(x.size), x.reshape(-1))
    sl.execute(I.qact(0, 1, ab, kb, kb * 128))
    out = x.size + 128
    sl.tput(out + np.arange(rows), rng.standard_normal(rows).astype(np.float32))
    matrix = c2.Matrix(full_rows, full_kb, sa, ssa, rs, scales.shape[1])
    packed = c2.pack(dram, matrix, bases=tuple(128 * pc for pc in range(c2.PCS)))
    ins = I.mm(sa + n0 * rs + k_begin * 128, ssa + n0 * scales.shape[1], out,
               rows, kb, rs, rows, 1, ab, scales.shape[1], k0=k_begin, acc=acc)
    expected = copy.deepcopy(sl)
    expected._mm(ins)
    trace = c2.Trace()
    decision = c2.execute(sl, ins, packed, n0=n0, k_begin=k_begin, trace=trace)
    if decision.operation != "MM_ROWS" or not np.array_equal(sl.tmem, expected.tmem):
        raise RuntimeError(f"C2 correctness failure: N={rows}, KB={kb}, ACC={acc}")
    name = f"C2 N={rows} KB={kb} {'ACC' if acc else 'plain'}"
    profile = c2.lens_data(trace, cfg, name)
    evidence = {"name": name, "bit_exact": True, "seed": seed, "rows": rows, "kb": kb,
                "acc": acc, "n0": n0, "k_begin": k_begin, "ab": ab,
                "output_bits": sl.tmem[out:out + rows].tolist(),
                "tmem_sha256": hashlib.sha256(sl.tmem.tobytes()).hexdigest(),
                "descriptor": packed.descriptor(), "counts": trace.counts()}
    return profile, evidence


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("build/lens/c2"))
    ap.add_argument("--rows", type=int, default=7)
    ap.add_argument("--kb", type=int, nargs="+", default=[1, 9, 54, 255])
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args(argv)
    if not 1 <= args.rows < 65536 or any(not 1 <= k <= 255 for k in args.kb):
        ap.error("rows must be 1..65535 and each KB 1..255")
    profiles, cases = [], []
    for kb in args.kb:
        for acc in (False, True):
            profile, evidence = run_case(kb, args.rows, acc, args.seed + kb)
            profiles.append(profile)
            cases.append(evidence)
            count = evidence["counts"]
            print(f"{evidence['name']}: exact; {count['advance_slots']} advances, "
                  f"{count['commit_slots']} commits, {count['weight_bytes']} weight bytes")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = lens.save(profiles, args.out_dir / "c2-reference.otpuprof")
    (args.out_dir / "c2-reference.html").write_text(lens.render(profiles,
        title="C2 exact reference events — not RTL timing"))
    report = {"model": "C2 functional reference; serialized arithmetic slots, not hardware cycles",
              "workload": "synthetic int8 microbenchmarks; repository quantizer and QACT",
              "excluded": ["HBM timing", "scale cache transactions", "overlap", "QACT/vector timing",
                           "Gemma end-to-end", "F2 hardware"], "cases": cases}
    (args.out_dir / "c2-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Lens: {path}")


if __name__ == "__main__":
    main()
