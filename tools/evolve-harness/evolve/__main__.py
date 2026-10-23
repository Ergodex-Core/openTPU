import argparse
import json
from pathlib import Path

from .common import GateError, read_json
from .loop import initialize, reconcile, rollback, run_loop, status


def main():
    p = argparse.ArgumentParser(description="OpenTPU bounded runtime optimization; no provider or FPGA build integration")
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--config", required=True)
    init.add_argument("--run", required=True)
    run = sub.add_parser("run")
    run.add_argument("--run", required=True)
    run.add_argument("--stop-after", type=int, help="total completed candidate limit; resume uses frozen full budget")
    st = sub.add_parser("status")
    st.add_argument("--run", required=True)
    rb = sub.add_parser("rollback")
    rb.add_argument("--run", required=True)
    rb.add_argument("--reason", required=True)
    rc = sub.add_parser("reconcile")
    rc.add_argument("--run", required=True)
    rc.add_argument("--trial", required=True)
    rc.add_argument("--receipt", required=True)
    el = sub.add_parser("eligibility")
    el.add_argument("--registry", default=str(Path(__file__).resolve().parents[1]/"configs/eligibility.json"))
    a = p.parse_args()
    try:
        if a.command == "init": result = initialize(a.config, a.run)
        elif a.command == "run": result = run_loop(a.run, a.stop_after)
        elif a.command == "status": result = status(a.run)
        elif a.command == "rollback": result = rollback(a.run, a.reason)
        elif a.command == "reconcile": result = reconcile(a.run,a.trial,a.receipt)
        else:
            registry = read_json(a.registry)
            from .common import file_hash, require
            for r in registry["sources"]:
                require(file_hash(Path(a.registry).resolve().parents[1]/r["snapshot"]) == r["sha256"],
                        "source receipt changed: "+r["snapshot"])
            result = registry
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (GateError, OSError, KeyError, TypeError, ValueError) as e:
        print(json.dumps({"status": "blocked", "reason": str(e), "exception": type(e).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
