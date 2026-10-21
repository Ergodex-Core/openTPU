#!/usr/bin/env python3
"""Run the wide-port regression with durable per-test receipts.

The source fingerprint and each setup/call/teardown result are flushed to disk.
A disconnected executor therefore need not erase evidence of completed tests.
The output directory must be new; interrupted runs are never overwritten.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                      OTPU_TARGET="f2-hbm", OTPU_NCH="8")
    import pytest
    from opentpu import rtlsim

    rtlsim.VFLAGS[rtlsim.VFLAGS.index("-j") + 1] = "4"
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "rtl", "boards", "targets", "opentpu", "tests", "sim",
         "pyproject.toml"], cwd=root).split(b"\0")
    source = hashlib.sha256()
    for name in sorted(n for n in names if n):
        source.update(name + b"\0" + (root / os.fsdecode(name)).read_bytes() + b"\0")
    meta = {"started_utc": now(), "pid": os.getpid(), "root": str(root),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_fingerprint_sha256": source.hexdigest(), "python": sys.version,
            "verilator": rtlsim.verilator_version(), "vflags": rtlsim.VFLAGS,
            "environment": {k: os.environ[k] for k in
                            ("PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                             "OTPU_TARGET", "OTPU_NCH")},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (out / "reports.jsonl").open("x") as journal:
        class Receipt:
            def pytest_collection_finish(self, session):
                (out / "collected.json").write_text(json.dumps(
                    [item.nodeid for item in session.items], indent=2) + "\n")

            def pytest_runtest_logreport(self, report):
                row = {"nodeid": report.nodeid, "when": report.when,
                       "outcome": report.outcome, "duration": report.duration,
                       "recorded_utc": now()}
                if report.failed or report.skipped:
                    row["detail"] = str(report.longrepr)
                journal.write(json.dumps(row) + "\n")
                journal.flush()
                os.fsync(journal.fileno())

        code = pytest.main(["-q", "-p", "no:cacheprovider",
                            f"--basetemp={out / 'tmp'}",
                            f"--junitxml={out / 'junit.xml'}",
                            "tests/test_wide_port_rtl.py"], plugins=[Receipt()])
    (out / "completed.json").write_text(json.dumps(
        {"finished_utc": now(), "exit_code": int(code)}, indent=2) + "\n")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
