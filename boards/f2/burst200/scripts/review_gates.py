#!/usr/bin/env python3
"""The six packaging gates of a burst200 run, read from its audit stage's reports, as the review
JSON scripts/package_f2.py takes (checkpoint_sha256 and setup_hold_pulse_width, fully_routed,
drc, cdc, bus_skew, constraints, each with "pass" and its evidence as report:line text).

    python boards/f2/burst200/scripts/review_gates.py RUN [--out RUN/audit/review.json]
        [--reviewer NAME] [--no-clock-max 35]

The criteria are the ones builds 6 and 11 were signed off on (docs/v4/README.md): setup, hold
and pulse width met with no failing endpoint; every net routed, none with routing errors; no DRC
error and no critical warning; every critical CDC row under a timing exception (False Path,
asynchronous clock groups, max delay); no bus-skew violation; no unconstrained internal
endpoint, and no more no_clock pins than the shell's own (35 in builds 6 and 11: the small
shell's PCIe bufg_gt_sysclk trees), none inside the CL. A gate that fails says why. The JSON is
a reviewer's statement: it names who read the reports (--reviewer) and that the sign-off is
theirs; nothing here uploads, creates an AFI or loads a card."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lines(p: Path) -> list[str]:
    return p.read_text(errors="replace").splitlines() if p.is_file() else []


def ev(p: Path, i: int, text: str) -> str:
    return f"{p.name}:{i + 1}: {text.strip()}"


def timing_gate(rpt: Path) -> dict:
    """WNS / TNS / WHS / THS / WPWS / TPWS from the Design Timing Summary table."""
    L = lines(rpt)
    out = {"pass": False, "evidence": []}
    for i, l in enumerate(L):
        if re.match(r"\s*WNS\(ns\)\s+TNS\(ns\)", l):
            for j in range(i + 1, min(i + 5, len(L))):
                vals = L[j].split()
                if len(vals) >= 12 and re.match(r"-?\d+\.\d+$", vals[0]):
                    wns, tns, tnsf, _, whs, ths, thsf, _, wpws, tpws, tpwsf, _ = vals[:12]
                    out.update(wns=float(wns), tns=float(tns), tns_failing=int(tnsf),
                               whs=float(whs), ths=float(ths), ths_failing=int(thsf),
                               wpws=float(wpws), tpws=float(tpws), tpws_failing=int(tpwsf))
                    out["evidence"].append(ev(rpt, i, l))
                    out["evidence"].append(ev(rpt, j, L[j]))
                    break
            break
    met = [ev(rpt, i, l) for i, l in enumerate(L) if "All user specified timing constraints are met" in l]
    out["evidence"] += met[:1]
    out["pass"] = bool("wns" in out and out["wns"] >= 0 and out["whs"] >= 0 and out["wpws"] >= 0
                       and out["tns_failing"] == 0 and out["ths_failing"] == 0
                       and out["tpws_failing"] == 0 and met)
    if not out["pass"]:
        out["why"] = "timing not met (or the summary table / the met line not found)"
    return out


def route_gate(rpt: Path) -> dict:
    L = lines(rpt)
    out = {"pass": False, "evidence": []}
    num = {}
    for i, l in enumerate(L):
        m = re.match(r"\s*#\s*of\s+(.+?)\.*\s*:\s*(\d+)", l)
        if m:
            num[m.group(1).strip(" .")] = int(m.group(2))
            if any(k in m.group(1) for k in ("routing errors", "fully routed", "unrouted", "nets with some")):
                out["evidence"].append(ev(rpt, i, l))
    errors = next((v for k, v in num.items() if "routing errors" in k), None)
    unrouted = sum(v for k, v in num.items() if "unrouted" in k)
    out.update(routing_errors=errors, unrouted=unrouted,
               fully_routed=next((v for k, v in num.items() if "fully routed" in k), None))
    out["pass"] = bool(errors == 0 and unrouted == 0 and out["fully_routed"])
    if not out["pass"]:
        out["why"] = "routing errors or unrouted nets (or the counts not found)"
    return out


def drc_gate(rpt: Path) -> dict:
    """The rule table's rows by severity: | RULE-n | Severity | Description | Violations |."""
    L = lines(rpt)
    sev = {}
    rows = []
    for i, l in enumerate(L):
        m = re.match(r"\|\s*([A-Z0-9]+-\d+)\s*\|\s*(Error|Critical Warning|Warning|Advisory)\s*\|(.*)\|\s*(\d+)\s*\|", l)
        if m:
            sev[m.group(2)] = sev.get(m.group(2), 0) + int(m.group(4))
            if m.group(2) in ("Error", "Critical Warning"):
                rows.append(ev(rpt, i, l))
    out = {"errors": sev.get("Error", 0), "critical_warnings": sev.get("Critical Warning", 0),
           "warnings": sev.get("Warning", 0), "evidence": rows[:20]}
    found = [ev(rpt, i, l) for i, l in enumerate(L) if re.search(r"Violations found:\s*\d+", l)]
    out["evidence"] = found[:1] + out["evidence"]
    out["pass"] = bool(found and out["errors"] == 0 and out["critical_warnings"] == 0)
    if not out["pass"]:
        out["why"] = "DRC errors or critical warnings (listed), or the report not found"
    return out


def cdc_gate(rpt: Path) -> dict:
    """Critical CDC rows: each must carry a timing exception (False Path / Asynch Clock Groups /
    Max Delay) in its Exception column."""
    L = lines(rpt)
    crit, bare = 0, []
    summ = []
    for i, l in enumerate(L):
        m = re.match(r"^(CDC-\d+)\s+(Critical|Warning|Info)\s+(\d+)\s", l)
        if m:
            summ.append(ev(rpt, i, l))
            if m.group(2) == "Critical":
                crit += int(m.group(3))
        if re.match(r"^\s*\d+\s+CDC-\d+\s+Critical", l) and not re.search(
                r"False Path|Asynch Clock Groups|Max Delay|Clock Groups", l):
            bare.append(ev(rpt, i, l))
    out = {"critical": crit, "critical_without_exception": len(bare),
           "evidence": summ[:20] + bare[:20]}
    out["pass"] = bool(summ and not bare)
    if not out["pass"]:
        out["why"] = "critical CDC rows outside a timing exception (listed), or no summary found"
    return out


def bus_skew_gate(rpt: Path) -> dict:
    L = lines(rpt)
    rows, neg = 0, []
    on = False
    for i, l in enumerate(L):
        if l.startswith("1. Bus Skew Report Summary"):
            on = True
        elif l.startswith("2. Bus Skew Report Per Constraint"):
            on = False
        if on:
            m = re.match(r"^\s+(Slow|Fast)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", l)
            if m:
                rows += 1
                if float(m.group(4)) < 0:
                    neg.append(ev(rpt, i, l))
    found = [ev(rpt, i, l) for i, l in enumerate(L) if l.startswith("1. Bus Skew Report Summary")]
    out = {"rows": rows, "violated": len(neg), "evidence": found[:1] + neg[:20]}
    out["pass"] = bool(found and not neg)
    if not out["pass"]:
        out["why"] = "bus-skew violations (listed), or no summary found"
    return out


def constraints_gate(rpt: Path, no_clock_max: int) -> dict:
    """check_timing: no unconstrained internal endpoint; no_clock pins at most the shell's, none
    in the CL (their pins listed after the check's header)."""
    L = lines(rpt)
    counts, evid = {}, []
    for i, l in enumerate(L):
        m = re.match(r"^\s*\d+\.\s+checking\s+(\w+)\s+\((\d+)\)", l)
        if m:
            counts[m.group(1)] = int(m.group(2))
            evid.append(ev(rpt, i, l))
    # the no_clock section's pins: "WRAPPER/CL/..." would be the CL's own
    cl_pins = []
    sec = None
    for i, l in enumerate(L):
        m = re.match(r"^\s*\d+\.\s+checking\s+(\w+)", l)
        if m:
            sec = m.group(1)
            continue
        if sec == "no_clock" and "WRAPPER/CL/" in l:
            cl_pins.append(ev(rpt, i, l))
    out = {"checks": counts, "no_clock_in_cl": len(cl_pins), "evidence": evid[:16] + cl_pins[:10]}
    out["pass"] = bool(counts and counts.get("unconstrained_internal_endpoints", 1) == 0
                       and counts.get("no_clock", 1 << 30) <= no_clock_max and not cl_pins)
    if not out["pass"]:
        out["why"] = ("unconstrained internal endpoints, more no_clock pins than the shell's, or "
                      "no_clock pins in the CL (or check_timing not found)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="a burst200 build's output directory")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reviewer", default="unsigned")
    ap.add_argument("--no-clock-max", type=int, default=35)
    a = ap.parse_args()
    A = a.run / "audit"
    audit = dict(l.split("=", 1) for l in lines(A / "audit.txt") if "=" in l)
    ckpt = Path(audit.get("checkpoint", a.run / "refine/post_route.dcp"))
    review = {
        "run": str(a.run.resolve()), "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
        "reviewer": a.reviewer,
        "statement": "the six gates as read from the audit stage's reports (scripts/review_gates.py); "
                     "the sign-off is the reviewer's",
        "audit_txt": audit,
        "setup_hold_pulse_width": timing_gate(A / "timing.rpt"),
        "fully_routed": route_gate(A / "route_status.rpt"),
        "drc": drc_gate(A / "drc.rpt"),
        "cdc": cdc_gate(A / "cdc.rpt"),
        "bus_skew": bus_skew_gate(A / "bus_skew.rpt"),
        "constraints": constraints_gate(A / "timing.rpt", a.no_clock_max),
    }
    gates = ("setup_hold_pulse_width", "fully_routed", "drc", "cdc", "bus_skew", "constraints")
    review["all_pass"] = all(review[g]["pass"] for g in gates)
    out = a.out or A / "review.json"
    out.write_text(json.dumps(review, indent=1) + "\n")
    for g in gates:
        r = review[g]
        print(f"{g:24s} {'PASS' if r['pass'] else 'FAIL'}  " +
              ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("pass", "evidence", "why", "checks"))
              + (f"  ({r['why']})" if not r["pass"] else ""))
    print(f"checkpoint {ckpt} sha256 {review['checkpoint_sha256']}\nwrote {out}")
    return 0 if review["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
