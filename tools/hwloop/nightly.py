"""The nightly FPGA step, prepared but never started from here.

A candidate that passed the day's gates (static checks, Verilator regressions against the ISA
simulator, the RTL timing run, a VisualSim replay when the seat allowed one) is queued with
its immutable commit. `plan` writes the exact commands the build host would run with the
existing authorized flow (boards/f2/build.py on the build instance, then tools/f2_bench.py on
the card against the same prompts as the baseline image) and the rollback (the known-good
AGFI). Nothing here launches an instance, programs the card, submits an AFI or activates a
schedule: that needs the owner's approval, and the plan says so in its first line.
"""
from __future__ import annotations

import datetime as _dt
import json
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUEUE = HERE / "nightly" / "queue.json"

KNOWN_GOOD = {
    "f2-hbm-burst200": {"agfi": "agfi-0d06cb48544a02106", "note": "Alex's burst200 image: the 2026-10-16 baselines (docs/ternary.md 5, docs/wide_port.md)"},
    "f2-ddr4-100mhz": {"agfi": "agfi-0014449e89d7fceee", "note": "build f9e0e5bf, the DDR image behind bench_ours100.json"},
}


def load_queue() -> dict:
    if QUEUE.is_file():
        return json.loads(QUEUE.read_text())
    return {"candidates": [], "note": "queued only; nothing scheduled or launched"}


def save_queue(q: dict) -> None:
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text(json.dumps(q, indent=2) + "\n")


def add(name: str, commit: str, target: str, baseline_profile: str, gates: dict, workload: dict, run_dir: str | None) -> dict:
    q = load_queue()
    entry = {"name": name, "commit": commit, "target": target, "baseline_profile": baseline_profile,
             "queued_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "gates": gates, "workload": workload,
             "run_dir": run_dir, "status": "QUEUED_AWAITING_APPROVAL"}
    q["candidates"] = [c for c in q["candidates"] if c["name"] != name] + [entry]
    save_queue(q)
    return entry


def plan(entry: dict, hdk: str = "/home/ubuntu/aws-fpga", build_root: str = "/home/ubuntu/opt/hwloop-nightly") -> str:
    """The commands, as text, for the build host and the card host. Not executed."""
    c = entry["commit"]
    t = entry["target"]
    out = f"{build_root}/{entry['name']}-{c[:8]}"
    kg = KNOWN_GOOD.get(entry["baseline_profile"], {})
    models = entry["workload"].get("models", "gemma3-1b")
    lines = [
        "# NOT ACTIVATED. Needs the F2 owner's approval before any line runs (shared instance, paid resources).",
        f"# candidate {entry['name']} at immutable commit {c}, target {t}; baseline image {kg.get('agfi', '?')} ({kg.get('note', '')})",
        "",
        "## build host (Vivado; tools/tourney/remote.py knows the hosts: OTPU_BUILD_HOSTS)",
        f"git -C ~/openTPU fetch origin && git -C ~/openTPU worktree add {out}/src {c}",
        f"cd {out}/src && python boards/f2/build.py --hdk {hdk} --target {shlex.quote(t)} --out {out}/synth --flow SynthCL --build-id {c[:8]}",
        f"cd {out}/src && python boards/f2/build.py --hdk {hdk} --target {shlex.quote(t)} --out {out}/full --flow BuildAll --package --build-id {c[:8]}",
        "#   -> Developer_CL tar; then `aws ec2 create-fpga-image` (approval: paid), `55_wait`-style polling for the AGFI",
        "",
        "## card host (after the AFI exists; only when no other session owns the slot: check the opt-* services first)",
        "#   load the candidate AGFI into the slot (approval: shared hardware)",
        f"python tools/f2_bench.py --dev f2:0 --json {out}/bench_candidate.json --models {models} --max-tokens {entry['workload'].get('max_tokens', 128)}",
        "#   then the baseline image, same prompts, same host, same day:",
        f"#   load {kg.get('agfi', '<baseline agfi>')}; python tools/f2_bench.py --dev f2:0 --json {out}/bench_baseline.json --models {models} --max-tokens {entry['workload'].get('max_tokens', 128)}",
        "",
        "## rollback",
        f"#   reload the known-good image {kg.get('agfi', '<baseline agfi>')}; keep {out}/synth and {out}/full for the record",
        "",
        "## feed back",
        f"#   copy {out}/bench_baseline.json and bench_candidate.json into VS_AR/ergodex_npu/hwloop/opentpu/<profile>/evidence/ and their numbers into",
        f"#   tools/hwloop/profiles/{entry['baseline_profile']}.json 'measured' (and a new profile pinned to the candidate's AGFI / commit), then",
        f"python tools/hwloop/cli.py report --baseline <baseline run> --candidate <candidate run> --profile {entry['baseline_profile']} --out {out}/report",
        "#   the report records the RTL-vs-board and VisualSim-vs-board errors per workload (calibration on the short prompt only; the rest held out)",
    ]
    return "\n".join(lines) + "\n"
