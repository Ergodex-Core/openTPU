#!/usr/bin/env python3
"""Freeze and build only the standalone C2 kernel on F2's VU47P, without AWS/HDK.

Prepare locally, copy the entire output directory to a Vivado 2025.2 host, then run:
    python3 run.py --out . --run-prepared
The same source snapshot supports --stage route after a successful synthesis. This
is an out-of-context gate, not an AFI, shell qualification or hardware benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PART = "xcvu47p-fsvh2892-2-e"
SOURCES = ("rtl/vpu/otpu_fp.sv", "rtl/vpu/otpu_fpipe.sv", "rtl/top/otpu_pkg.sv",
           "rtl/tech/xilinxus/otpu_tech_dsp.sv", "rtl/mxu/otpu_mxu.sv",
           "rtl/mxu/otpu_c2_mxu.sv")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def prepare(out, *, stage="synth", threads=4):
    """Copy exact bytes, including dirty RTL, into a fresh relocatable run directory."""
    out = Path(out).resolve()
    if stage not in ("synth", "route") or threads not in range(1, 9):
        raise ValueError("stage must be synth/route and threads must be 1..8")
    out.mkdir(parents=True, exist_ok=False)
    for source in SOURCES:
        destination = out / "src" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
    (out / "flow").mkdir()
    shutil.copy2(ROOT / "tools/c2_ooc/ooc.tcl", out / "flow/ooc.tcl")
    shutil.copy2(ROOT / "tools/c2_ooc/constraints.xdc", out / "flow/constraints.xdc")
    shutil.copy2(Path(__file__), out / "run.py")
    info = {"schema": 1, "purpose": "C2 standalone FPGA out-of-context gate; no shell or hardware measurement",
            "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"), "top": "otpu_c2_mxu", "part": PART,
            "vivado_version": "2025.2", "period_ns": 5.0, "target_mhz": 200.0,
            "input_delay_max_ns": 0.5, "input_delay_min_ns": 0.0,
            "output_delay_max_ns": 0.5, "output_delay_min_ns": 0.0,
            "setup_uncertainty_ns": 0.1, "hold_uncertainty_ns": 0.05,
            "parameters": {"D": 128, "C": 4, "DEPTH": 256, "RF": 8},
            "stage": stage, "threads": threads, "sources": list(SOURCES)}
    info["files"] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "inputs.json", info)
    return out


def verify(out):
    out = Path(out).resolve()
    info = json.loads((out / "inputs.json").read_text())
    if info.get("schema") != 1 or info.get("part") != PART or info.get("top") != "otpu_c2_mxu":
        raise RuntimeError("not a supported C2 OOC snapshot")
    if info.get("sources") != list(SOURCES) or info.get("parameters") != {"D": 128, "C": 4, "DEPTH": 256, "RF": 8}:
        raise RuntimeError("source inventory or C2 parameters differ from the gate")
    required = {f"src/{name}" for name in SOURCES} | {"flow/ooc.tcl", "flow/constraints.xdc", "run.py"}
    if set(info.get("files", {})) != required:
        raise RuntimeError("snapshot inventory mismatch")
    for name, digest in info["files"].items():
        p = out / name
        if not p.is_file() or sha(p) != digest:
            raise RuntimeError(f"frozen build input modified: {name}")
    if info.get("threads") not in range(1, 9) or info.get("period_ns") != 5.0:
        raise RuntimeError("invalid bounded-thread/200MHz settings")
    return info


def check_version(output):
    if not re.search(r"\bvivado\s+v?2025\.2\b", output, re.I):
        raise RuntimeError("C2 OOC gate requires Vivado 2025.2")


def resume_checkpoint(out):
    """Only reuse a synthesis DCP tied to this exact frozen snapshot."""
    prior = json.loads((out / "outputs-synth.json").read_text())
    dcp = out / "checkpoints/post_synth.dcp"
    if (prior.get("tool_exit_code") != 0 or not prior.get("completed") or
            prior.get("inputs_sha256") != sha(out / "inputs.json") or
            prior.get("artifacts", {}).get("checkpoints/post_synth.dcp", {}).get("sha256") != sha(dcp)):
        raise RuntimeError("cannot resume an unverified or changed synthesis checkpoint")
    return dcp


def run(out, *, stage=None, vivado="vivado"):
    out = Path(out).resolve()
    info = verify(out)
    stage = stage or info["stage"]
    if stage not in ("synth", "route"):
        raise ValueError("invalid stage")
    if (out / f"outputs-{stage}.json").exists() or (out / f"console-{stage}.log").exists():
        raise RuntimeError(f"{stage} run already exists; preserve evidence and prepare a fresh snapshot")
    banner = subprocess.check_output([vivado, "-version"], text=True, stderr=subprocess.STDOUT)
    check_version(banner)
    resume = stage == "route" and (out / "outputs-synth.json").is_file()
    if resume:
        resume_checkpoint(out)
    invocation = [vivado, "-mode", "batch", "-nojournal", "-log", f"vivado-{stage}.log",
                  "-source", "flow/ooc.tcl", "-tclargs", ".", stage, str(info["threads"]), str(int(resume))]
    write_json(out / f"invocation-{stage}.json", {"command": invocation, "cwd": str(out), "vivado_banner": banner})
    with (out / f"console-{stage}.log").open("w") as log:
        result = subprocess.run(invocation, cwd=out, stdout=log, stderr=subprocess.STDOUT)
    verify(out)
    text = (out / f"console-{stage}.log").read_text()
    marker = "OTPU_C2_ROUTE_DONE" if stage == "route" else "OTPU_C2_SYNTH_DONE"
    checkpoint = f"checkpoints/post_{stage}.dcp"
    completed = result.returncode == 0 and marker in text and (out / checkpoint).is_file()
    metrics_path = out / f"reports/{stage}/metrics.tsv"
    metrics = dict(line.split("\t", 1) for line in metrics_path.read_text().splitlines()) if metrics_path.is_file() else {}
    artifacts = {str(p.relative_to(out)): {"sha256": sha(p), "bytes": p.stat().st_size}
                 for directory in (out / "reports" / stage, out / "checkpoints")
                 if directory.exists() for p in sorted(directory.rglob("*")) if p.is_file()}
    summary = {"stage": stage, "tool_exit_code": result.returncode, "completed": completed,
               "inputs_sha256": sha(out / "inputs.json"), "metrics": metrics, "artifacts": artifacts,
               "timing_claim": "post-synthesis estimate only" if stage == "synth" else
               "standalone routed OOC timing under explicit I/O constraints; not shell/F2 qualification"}
    write_json(out / f"outputs-{stage}.json", summary)
    if not completed:
        raise RuntimeError(f"C2 {stage} did not complete; inspect {out / f'console-{stage}.log'}")
    print(json.dumps(summary | {"artifacts": list(artifacts)}, indent=2))
    if stage == "route" and metrics.get("target_met") != "1":
        raise RuntimeError("OOC routing completed but 200MHz timing target was not met; reports/checkpoint retained")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="fresh output directory, or transferred snapshot for --run-prepared")
    parser.add_argument("--stage", choices=("synth", "route"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--vivado", default="vivado")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not args.run_prepared:
        prepare(out, stage=args.stage or "synth", threads=args.threads)
    if args.prepare_only:
        print(f"Frozen snapshot: {out}\nOn the Vivado host: python3 run.py --out . --run-prepared")
    else:
        run(out, stage=args.stage, vivado=args.vivado)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
