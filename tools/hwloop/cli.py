#!/usr/bin/env python3
"""The per-change hardware loop: Verilator -> VisualSim -> (queued) nightly FPGA.

    python tools/hwloop/cli.py inventory                       # what is installed and what evidence exists
    python tools/hwloop/cli.py freeze  --rev <rev> --out DIR    # export the pinned tree, write the baseline manifest
    python tools/hwloop/cli.py run-rtl --tree DIR --out RUN [perf_qwen options]   # the Verilator timing run + trace
    python tools/hwloop/cli.py extract RUN                      # trace.json / mem_windows.csv from the run
    python tools/hwloop/cli.py build-vs RUN --profile NAME      # the VisualSim model for that run and memory profile
    python tools/hwloop/cli.py check-stale MODEL_DIR --tree DIR # refuse a model whose RTL fingerprint is stale
    python tools/hwloop/cli.py vs-run MODEL_DIR [--execute]     # prepare (default) or run the batch simulation
    python tools/hwloop/cli.py report --baseline RUN [--candidate RUN] --profile NAME --out DIR
    python tools/hwloop/cli.py nightly add|plan ...             # queue a candidate; print the (unactivated) plan
    python tools/hwloop/cli.py install RUN --profile NAME --dest VS_AR/ergodex_npu/hwloop/...   # the canonical copy

Only the standard library; perf_qwen runs under the venv given by --python (OTPU_PYTHON).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from tools.hwloop import nightly, report, trace, vslib, vsmodel  # noqa: E402

ROOT = HERE.parents[1]
DEFAULT_PY = os.environ.get("OTPU_PYTHON", sys.executable)


def _j(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_inventory(a):
    info = vslib.install_info()
    info["active_visualsim"] = vslib.active_visualsim()
    info["license_manager_up"] = vslib.license_manager_up()
    info["profiles"] = sorted(p.stem for p in (HERE / "profiles").glob("*.json"))
    info["verilator"] = shutil.which("verilator")
    _j(info)


def cmd_freeze(a):
    pin = trace.export_tree(Path(a.repo), a.rev, Path(a.out), Path(a.models) if a.models else None)
    fp = vslib.fingerprint_tree(Path(a.out), trace.RTL_GLOBS)
    baseline = {"pin": pin, "rtl_fingerprint": {k: v for k, v in fp.items() if k != "entries"},
                "rtl_fingerprint_entries": fp["entries"],
                "frozen": {"model": a.model, "layers": a.layers, "wformat": a.wformat, "head_format": a.head_format,
                           "pos": a.pos, "target": a.target, "nch": a.nch, "lat": a.lat, "mhz": a.mhz,
                           "memory_placement": "weights, KV and I/O in the slice's DRAM (the target's channels); programs resident",
                           "prompt": "perf_qwen's one decode token at --pos (the token's work does not depend on the prompt's values)",
                           "checkpoint": a.model, "quantization": f"weights {a.wformat}, head {a.head_format}, int8 activations, fp32 scales"}}
    ck = Path(a.out) / "models"
    if ck.exists():
        baseline["checkpoints_present"] = sorted(p.name for p in ck.iterdir())
    Path(a.out, "hwloop-baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    _j({k: v for k, v in baseline.items() if k != "rtl_fingerprint_entries"})


def cmd_run_rtl(a):
    r = trace.run_rtl(Path(a.tree), Path(a.out), python=a.python, model=a.model, layers=a.layers, wformat=a.wformat,
                      head_format=a.head_format, lat=a.lat, mhz=a.mhz, target=a.target, nch=a.nch, portb=a.portb,
                      extra=a.extra, check=a.check, dry=a.dry)
    _j(r)
    if r.get("status") == "OK":
        trace.extract(Path(a.out), window=a.window, nch=a.nch, lat=a.lat)


def cmd_extract(a):
    t = trace.extract(Path(a.run), window=a.window, nch=a.nch, lat=a.lat)
    _j({k: t.get(k) for k in ("derived", "memory_trace")})


def cmd_build_vs(a):
    run = Path(a.run)
    t = json.loads((run / "trace.json").read_text())
    prof = vsmodel.load_profile(a.profile)
    fp = None
    if a.tree:
        fp = vslib.fingerprint_tree(Path(a.tree), trace.RTL_GLOBS)
    elif (Path(t.get("source", {}).get("tree", "")) / "hwloop-baseline.json").is_file():
        b = json.loads((Path(t["source"]["tree"]) / "hwloop-baseline.json").read_text())
        fp = dict(b["rtl_fingerprint"], entries=b["rtl_fingerprint_entries"])
    out = run / "vs" / prof["name"]
    m = vsmodel.build(t, prof, out, name=a.name or f"openTPU_{prof['name']}_{a.variant}".replace("-", "_"),
                      rtl_fingerprint=fp, replay_fraction=a.fraction, variant=a.variant)
    _j({"model_dir": str(out), "static_validation": m["static_validation"], "label": m["label"], "phases": m["phases"],
        "memory_backend": m["memory_backend"]})


def cmd_check_stale(a):
    m = json.loads((Path(a.model_dir) / "manifest.json").read_text())
    st = vslib.check_stale(m, Path(a.tree))
    _j(st)
    sys.exit(2 if st["stale"] else 0)


def cmd_vs_run(a):
    md = Path(a.model_dir)
    manifest = json.loads((md / "manifest.json").read_text()) if (md / "manifest.json").is_file() else None
    import datetime as dt
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    r = vslib.run_batch(md / "model.xml", md / stamp, execute=a.execute, timeout_s=a.timeout, manifest=manifest,
                        stale_root=Path(a.tree) if a.tree else None)
    _j(r)
    if r["status"].startswith("REFUSED") or r["status"] == "FAILED":
        sys.exit(3)


def cmd_report(a):
    prof = vsmodel.load_profile(a.profile) if a.profile else None
    rep = report.compare(Path(a.baseline), Path(a.candidate) if a.candidate else None, prof, a.workload, Path(a.out), a.title)
    print((Path(a.out) / "report.md").read_text())


def cmd_nightly(a):
    if a.action == "add":
        e = nightly.add(a.name, a.commit, a.target, a.baseline_profile, json.loads(a.gates), json.loads(a.workload), a.run)
        _j(e)
        print(nightly.plan(e))
    elif a.action == "plan":
        q = nightly.load_queue()
        for e in q["candidates"]:
            if not a.name or e["name"] == a.name:
                print(nightly.plan(e))
    else:
        _j(nightly.load_queue())


def cmd_install(a):
    """Copy a run's trace, model(s), receipts and report into the canonical VS_AR/ergodex_npu tree,
    separating workload inputs, model XML, evidence, generated results and reports."""
    run = Path(a.run)
    dest = Path(a.dest)
    (dest / "workload").mkdir(parents=True, exist_ok=True)
    for f in ("trace.json", "mem_windows.csv", "perf.log", "perf.jsonl", "run_receipt.json", "command.sh"):
        if (run / f).is_file():
            shutil.copy2(run / f, dest / "workload" / f)
    for prof_dir in (run / "vs").glob("*"):
        if not prof_dir.is_dir():
            continue
        mdest = dest / "models" / prof_dir.name
        mdest.mkdir(parents=True, exist_ok=True)
        for f in ("model.xml", "manifest.json", "expected-counters.json", "static-validation.json", "scheduler.vm.txt"):
            if (prof_dir / f).is_file():
                shutil.copy2(prof_dir / f, mdest / f)
        for rd in prof_dir.glob("20*"):
            if rd.is_dir():
                rdest = dest / "results" / prof_dir.name / rd.name
                shutil.copytree(rd, rdest, dirs_exist_ok=True)
    if a.report:
        (dest / "reports").mkdir(exist_ok=True)
        for f in Path(a.report).glob("report.*"):
            shutil.copy2(f, dest / "reports" / f.name)
    prov = {"installed_from": str(run), "repo_root": str(ROOT), "profiles": sorted(p.name for p in (run / "vs").glob("*") if p.is_dir()),
            "note": "generated by tools/hwloop/cli.py install; the repository keeps the scripts and manifests, this tree the models and results"}
    (dest / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    _j(prov)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory").set_defaults(fn=cmd_inventory)

    p = sub.add_parser("freeze")
    p.add_argument("--repo", default=str(ROOT))
    p.add_argument("--rev", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models", help="models/ directory to link into the export")
    for k, d in (("model", "gemma3"), ("layers", 2), ("wformat", "int8"), ("head_format", "int8"), ("pos", 9),
                 ("target", "f2-hbm"), ("nch", 8), ("lat", 60), ("mhz", 200.0)):
        p.add_argument("--" + k.replace("_", "-"), default=d, type=type(d))
    p.set_defaults(fn=cmd_freeze)

    p = sub.add_parser("run-rtl")
    p.add_argument("--tree", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--python", default=DEFAULT_PY)
    for k, d in (("model", "gemma3"), ("layers", 2), ("wformat", "int8"), ("head_format", "int8"),
                 ("target", "f2-hbm"), ("nch", 8), ("lat", 60), ("mhz", 200.0), ("window", 1024)):
        p.add_argument("--" + k.replace("_", "-"), default=d, type=type(d))
    p.add_argument("--portb", type=int)
    p.add_argument("--check", action="store_true", help="compare with the ISA simulator (the independent reference)")
    p.add_argument("--dry", action="store_true")
    p.add_argument("extra", nargs="*")
    p.set_defaults(fn=cmd_run_rtl)

    p = sub.add_parser("extract")
    p.add_argument("run")
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--nch", type=int)
    p.add_argument("--lat", type=int)
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("build-vs")
    p.add_argument("run")
    p.add_argument("--profile", required=True)
    p.add_argument("--tree", help="the pinned source tree (fingerprint); default: the run's hwloop-baseline.json")
    p.add_argument("--name")
    p.add_argument("--variant", default="baseline")
    p.add_argument("--fraction", type=float, default=1.0, help="replay this fraction of the token's bytes")
    p.set_defaults(fn=cmd_build_vs)

    p = sub.add_parser("check-stale")
    p.add_argument("model_dir")
    p.add_argument("--tree", required=True)
    p.set_defaults(fn=cmd_check_stale)

    p = sub.add_parser("vs-run")
    p.add_argument("model_dir")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--tree", help="refuse if the model is stale against this tree")
    p.set_defaults(fn=cmd_vs_run)

    p = sub.add_parser("report")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate")
    p.add_argument("--profile")
    p.add_argument("--workload", default="gemma3-1b-int8")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="hwloop comparison")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("nightly")
    p.add_argument("action", choices=["add", "plan", "list"])
    p.add_argument("--name")
    p.add_argument("--commit")
    p.add_argument("--target", default="f2-hbm")
    p.add_argument("--baseline-profile", default="f2-hbm-burst200")
    p.add_argument("--gates", default="{}")
    p.add_argument("--workload", default='{"models": "gemma3-1b", "max_tokens": 128}')
    p.add_argument("--run")
    p.set_defaults(fn=cmd_nightly)

    p = sub.add_parser("install")
    p.add_argument("run")
    p.add_argument("--dest", required=True)
    p.add_argument("--report")
    p.set_defaults(fn=cmd_install)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
