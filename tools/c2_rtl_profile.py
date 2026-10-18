#!/usr/bin/env python3
"""Validate standalone C2 RTL and export observed cycles to openTPU Lens."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from opentpu import c2_rtl as C, lens


def decode_groups(kb, rows=31, acc=False):
    return [C.make_group(kb, min(4, rows-r), decode=True, acc=acc,
                         tag=r//4, seed=20261018+r+kb)
            for r in range(0, rows, 4)]


def legacy_groups(groups):
    """Same rows, weights, activations and initial values on legacy M=1."""
    result = []
    for group in groups:
        for c in range(group.count):
            g = copy.deepcopy(group)
            g.decode = False
            g.count = 1
            g.tag = len(result)
            g.weights[0] = group.weights[c]
            g.dw[0] = group.dw[c]
            g.sw[0] = group.sw[c]
            g.old[0] = group.old[c]
            g.addr = (1024 + g.tag * 32 + C.np.arange(4) * 7).astype(C.np.uint32)
            result.append(g)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("build/lens/c2/rtl"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.rtlsim.ROOT, text=True).strip()
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=C.rtlsim.ROOT).returncode != 0
    source = {"commit": revision, "dirty": dirty, "variant": "ping-pong",
              "sha256": {str(p.relative_to(C.rtlsim.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [C.rtlsim.ROOT / s for s in (
                             "rtl/mxu/otpu_c2_mxu.sv", "rtl/mxu/otpu_mxu.sv",
                             "rtl/tech/generic/otpu_tech_dsp.sv", "rtl/vpu/otpu_fp.sv",
                             "rtl/vpu/otpu_fpipe.sv", "sim/verilator/tb_c2_mxu.sv", "opentpu/c2_rtl.py")]}}
    cases = []
    for kb in (1, 9, 54):
        groups = decode_groups(kb)
        cases += [(f"decode-kb{kb}", groups, {}), (f"legacy-kb{kb}", legacy_groups(groups), {})]
    cases += [
        ("decode-kb1-slow-pc0", decode_groups(1), {"delay_pc0": 40}),
        ("decode-kb9-independent-stalls", decode_groups(9),
         {"wstall": 45, "sstall": 37, "astall": 33, "ostall": 61, "rstall": 51}),
        ("decode-kb1-acc-credit-pressure", decode_groups(1, rows=63, acc=True),
         {"old_hold": 300, "hold": 450, "ostall": 61, "rstall": 51}),
    ]
    mixed = [C.make_group(k, n, decode=mode, acc=acc, tag=i, seed=990+i, corners=True)
             for i, (k, n, mode, acc) in enumerate([
                 (9,4,False,False), (1,4,True,False), (1,3,True,True),
                 (54,2,False,True), (1,1,True,False), (9,4,False,True)])]
    cases.append(("mixed-mode-drains", mixed, {"wstall": 30, "sstall": 50, "ostall": 40, "rstall": 40}))
    profiles, reports = [], []
    for name, groups, service in cases:
        evidence = C.run(groups, out_dir=args.out_dir/name, seed=4401, **service)
        evidence["source"] = source
        profiles.append(C.lens_data(groups, evidence, "C2 "+name))
        profiles[-1]["notes"].append({"level": "info", "text":
            f"Ping-pong operand buffers. Source commit {revision}; dirty={dirty}. Full source hashes retained in c2_rtl.source."})
        reports.append({"name": name, **{k:v for k,v in evidence.items() if k != "events"}})
        print(name, json.dumps(evidence["counts"], sort_keys=True), flush=True)
    # Matched numerical workload, despite different grouping and simulated timing.
    for i in (0,2,4):
        assert [v for r in reports[i]["outputs"] for v in r["bits"]] == [v for r in reports[i+1]["outputs"] for v in r["bits"]]
    path = lens.save(profiles, args.out_dir/"c2-rtl.otpuprof")
    (args.out_dir/"c2-rtl.html").write_text(lens.render(profiles, title="C2 standalone RTL cycles — synthetic service"))
    (args.out_dir/"c2-rtl.json").write_text(json.dumps({
        "model": "Verilator execution of standalone dual-mode RTL with ping-pong buffers and synthetic per-channel services",
        "source": source,
        "clock": "200 MHz assumed display conversion; not synthesized or measured",
        "scope": "One D128/C4 product bank, operand/result queues, four epilogue lanes; no production/HBM integration",
        "cases": reports}, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
