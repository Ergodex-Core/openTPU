#!/usr/bin/env python3
"""Run ONLY after authorized transfer to the existing builder. No remote connection code.

Usage: python3 scripts/run_builder_checks.py --identity /path/to/current-owner-identity.json
The identity receipt is produced by an already authorized owner from current instance
metadata; this helper does not obtain credentials, probe networks or retry denied checks.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--identity",required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if sys.platform != "linux":
        raise SystemExit("Refused: tests must run on opentpu-m8a-build, never this Mac.")
    observed=Path("/sys/devices/virtual/dmi/id/board_asset_tag").read_text().strip()
    if observed != "i-0ec8d33362b3100be":
        raise SystemExit("Refused: local instance ID is not opentpu-m8a-build.")
    receipt=json.loads(Path(a.identity).read_text())
    now=dt.datetime.now(dt.timezone.utc)
    when=dt.datetime.fromisoformat(receipt["verified_at"].replace("Z","+00:00"))
    if not (receipt["instance_id"] == observed and receipt["instance_name"] == "opentpu-m8a-build"
            and receipt["iam_profile"] == "opentpu-f2-ssm" and receipt["access"] == "authorized"
            and when.tzinfo is not None and 0 <= (now-when).total_seconds() <= 120):
        raise SystemExit("Refused: fresh authorized instance/role evidence is missing.")
    stamp=now.strftime("%Y%m%dT%H%M%SZ")
    out=root/"runs"/("builder-checks-"+stamp)
    out.mkdir(parents=True,exist_ok=False)
    (out/"identity.json").write_text(json.dumps(receipt,indent=2)+"\n")
    python=sys.executable
    demo=out/"fixture-demo"
    commands=[
        [python,"-m","unittest","discover","-s","tests","-v"],
        [python,"-m","evolve","eligibility"],
        [python,"-m","evolve","init","--config","configs/fixture.json","--run",str(demo)],
        [python,"-m","evolve","run","--run",str(demo),"--stop-after","2"],
        [python,"-m","evolve","status","--run",str(demo)],
        [python,"-m","evolve","run","--run",str(demo)],
        [python,"-m","evolve","rollback","--run",str(demo),"--reason","Synthetic integration demonstration"],
    ]
    env={"PATH":os.defpath,"PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1"}
    results=[]
    for i,cmd in enumerate(commands):
        try:
            r=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=120)
            log=r.stdout+"\n"+r.stderr
            code=r.returncode
        except subprocess.TimeoutExpired:
            log="Builder check timed out; fixture-only process. Inspect this owned run before retry.\n"
            code=124
        logpath=out/f"{i:02d}.log";logpath.write_text(log)
        results.append({"argv":cmd,"returncode":code,"log":logpath.name,
                        "sha256":hashlib.sha256(log.encode()).hexdigest()})
        (out/"checks.json").write_text(json.dumps({"kind":"TEST_FIXTURE","results":results},indent=2)+"\n")
        print(f"check {i}: exit {code}; {logpath}")
        if code:return code
    print("Synthetic tests and dry-run commands completed; NO FPGA trial was run.")
    return 0


if __name__=="__main__":raise SystemExit(main())
