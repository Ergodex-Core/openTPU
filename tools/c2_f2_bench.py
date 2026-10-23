#!/usr/bin/env python3
"""Prepare C2 inputs offline, or check an already-loaded isolated F2 benchmark.

Never configures an FPGA. Hardware runs require its explicit PCI BDF and BUILD_ID.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opentpu import c2_f2 as C, lens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare", help="generate and validate a deterministic input fixture without hardware")
    prepare.add_argument("--fixture", type=Path, required=True)
    prepare.add_argument("--rows", type=int, default=31)
    prepare.add_argument("--kb", type=int, default=9)
    prepare.add_argument("--seed", type=int, default=20261019)
    prepare.add_argument("--base-beat", type=int, default=16)
    run = sub.add_parser("run", help="run both modes on an already-loaded C2 benchmark image")
    run.add_argument("--fixture", type=Path, required=True)
    run.add_argument("--bdf", required=True)
    run.add_argument("--expected-build-id", type=lambda s: int(s, 0), required=True)
    run.add_argument("--build-manifest", type=Path, required=True, help="inputs.json from the isolated C2 image build")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.fixture.suffix != ".npz" or args.fixture.exists():
            raise ValueError("fixture path must be a new .npz file")
        f = C.make_fixture(args.rows, args.kb, args.seed, args.base_beat)
        bits, integers = f.reference()
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        f.save(args.fixture)
        manifest = dict(kind="offline-canonical-reference", rows=f.rows, kb=f.kb, seed=args.seed,
                        base_beat=f.base_beat, fixture_sha256=f.digest(), expected_bits=bits.tolist(),
                        expected_integers=integers, expected_pc_beats=f.expected_pc_beats())
        args.fixture.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")
        print(f"Prepared {args.fixture}; canonical reference checked, no hardware accessed")
        return
    if not 1 <= args.repeats <= 100 or args.timeout <= 0:
        raise ValueError("repeats must be 1..100 and timeout positive")
    fixture = C.Fixture.load(args.fixture)
    build_manifest = json.loads(args.build_manifest.read_text())
    if (build_manifest.get("kind") != "c2-hbm-kernel" or
            int(build_manifest.get("build_id", "-1"), 16) != args.expected_build_id or
            build_manifest.get("core_khz") != 200000):
        raise ValueError("build manifest must identify the expected C2 200 MHz benchmark image")
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    provenance = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip(),
        fixture_file_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        build_manifest_sha256=hashlib.sha256(args.build_manifest.read_bytes()).hexdigest(),
        build_manifest=build_manifest,
        sources={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 (root/"opentpu/c2_f2.py", root/"opentpu/host/c2_f2.py", Path(__file__).resolve())})
    (args.out/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    # Import only on the explicit run path; offline fixture generation never maps PCI.
    from opentpu.host.c2_f2 import C2F2Transport
    profiles, results = [], []
    with C2F2Transport(args.bdf, expected_build_id=args.expected_build_id) as transport:
        bench = C.Benchmark(transport, expected_build_id=args.expected_build_id)
        try:
            bench.preload(fixture)
            for repeat in range(args.repeats):
                # Alternate order to expose order/warm-state effects in the raw evidence.
                for mode in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    result = bench.run(mode, timeout=args.timeout)
                    result.update(repeat=repeat, source=provenance)
                    name = f"C2 {'decode' if mode else 'legacy M1'} N{fixture.rows} KB{fixture.kb} repeat{repeat}"
                    results.append(result)
                    (args.out/f"result-{repeat}-{mode}.json").write_text(json.dumps(result, indent=2)+"\n")
                    profiles.append(C.lens_data(result, name))
                    lens.save(profiles, args.out/"c2-f2.otpuprof")
                    print(f"{name}: {result['cycles']} device cycles, {result['weight_bytes']} native bytes, exact bits", flush=True)
        finally:
            if transport.reg_read(C.R_STATUS) & (C.BUSY | C.DRAINING):
                bench.abort()
    (args.out/"c2-f2.html").write_text(lens.render(profiles, title="C2 F2 matrix benchmark — hardware counters"))
    (args.out/"results.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
