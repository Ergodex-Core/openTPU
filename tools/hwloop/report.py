"""Baseline / candidate comparison reports: JSON, CSV and Markdown, every number labelled.

A run directory holds trace.json (Verilator, `measured` as an RTL simulation receipt; or an LPN
run's record, `lpn-estimate`: a model's cycles, never a measurement) and, optionally, vs/<profile>/<stamp>/receipt.json with the VisualSim counters (`unvalidated
scenario` until a run exists and has been compared with receipts), and the profile's board
numbers (`measured`). Missing evidence stays visible as `waiting`.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
from pathlib import Path

from . import vslib

LPN_ESTIMATE = "lpn-estimate"               # tools/hwloop/lpn: a model's cycles graded against a receipt

METRIC_NOTES = {
    "model_load_s": "model loading and initialization (weights upload, program compile): serving receipts only",
    "host_overhead": "host / runtime overhead: the difference between wall and device decode rates",
    "ttft_s": "client TTFT: request to first streamed token (prefill on the device + the first decode step + host)",
    "prefill_serving_tok_s": "prompt tokens / client TTFT: a serving metric, not device prefill throughput",
    "device_tok_s": "decode tokens/s from device cycles only (one token's cycles at the core clock; RTL: no host)",
    "decode_tok_s": "tokens after the first / their decode interval (client side)",
}


def _get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def rtl_row(run_dir: Path, name: str) -> dict:
    t = json.loads((run_dir / "trace.json").read_text())
    d = t.get("derived", {})
    s = t.get("summary", {})
    mt = t.get("memory_trace", {})
    src = t.get("source", {})
    label = t.get("label", vslib.MEASURED)
    evidence = ("LPN model (" + str(run_dir) + "; " + str(t.get("calibration")) + ")" if label == LPN_ESTIMATE
                else "Verilator RTL simulation (" + str(run_dir) + ")")
    return {
        "name": name, "label": label, "evidence": evidence, "calibration": t.get("calibration"),
        "source_rev": _get(src, "tree"), "command": _get(src, "shell"),
        "cycles": d.get("cycles"), "core_mhz": d.get("core_mhz"),
        "device_ms_per_token": d.get("device_ms_per_token"), "device_tok_s": d.get("device_tok_s"),
        "layers": s.get("layers"), "pos": s.get("pos"),
        "dram_bytes": d.get("dram_bytes"), "dram_read_bytes": d.get("dram_read_bytes"), "dram_write_bytes": d.get("dram_write_bytes"),
        "bytes_per_cycle": d.get("bytes_per_cycle"), "achieved_GBps": d.get("achieved_GBps"),
        "portb_utilization": d.get("portb_utilization"), "roofline_efficiency": d.get("roofline_efficiency"),
        "mxu_mac_pct": s.get("mxu_mac_pct"), "mxu_starved_cycles": s.get("mxu_starved_cycles"), "mxu_blocked_pct": s.get("mxu_blocked_pct"),
        "unit_busy_pct": s.get("unit_busy_pct"), "channels": s.get("channels"),
        "max_outstanding_reads_est": mt.get("max_outstanding_reads_est"), "idle_gaps_over_64": mt.get("idle_gaps_over_64"),
        "beats_per_cycle_active": mt.get("beats_per_cycle"), "sim_wall_s": s.get("sim_wall_s"),
        "isa_check": s.get("isa_check_line"), "phases": s.get("phases"),
    }


def vs_rows(run_dir: Path) -> list[dict]:
    rows = []
    for rec in sorted(run_dir.glob("vs/*/*/receipt.json")):
        r = json.loads(rec.read_text())
        prof = rec.parents[1].name
        row = {"name": f"visualsim:{prof}", "profile": prof, "status": r.get("status"), "label": r.get("label", vslib.WAITING),
               "run": str(rec.parent), "wall_s": r.get("wall_s"), "license_freed": r.get("license_freed")}
        if r.get("executed") and r.get("status") == "EXECUTED":
            c = vslib.read_counters(rec.parent)
            row.update(counters=c, device_tok_s=c.get("Device_Tok_s_Est"), achieved_GBps=c.get("Achieved_GBps"),
                       mean_latency_s=c.get("Mean_Memory_Latency_Seconds"), credit_blocked=c.get("Credit_Blocked_Events"))
        else:
            row["note"] = r.get("note") or "not executed"
        rows.append(row)
    return rows


def board_row(profile: dict, workload_key: str) -> dict | None:
    meas = (profile.get("measured") or {}).get(workload_key)
    if not meas:
        return None
    return {"name": f"board:{profile['name']}", "label": vslib.MEASURED, "profile": profile["name"],
            "pin": profile.get("pin"), "date": _get(profile, "measured", "date"), "tool": _get(profile, "measured", "tool"),
            **meas}


def rel_err(pred, meas):
    if pred is None or meas in (None, 0):
        return None
    return (pred - meas) / meas


def compare(baseline: Path, candidate: Path | None, profile: dict | None, workload_key: str, out_dir: Path,
            title: str = "hwloop comparison") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [rtl_row(baseline, "baseline")]
    if candidate:
        rows.append(rtl_row(candidate, "candidate"))
    vs = {"baseline": vs_rows(baseline), "candidate": vs_rows(candidate) if candidate else []}
    board = board_row(profile, workload_key) if profile else None
    b, c = rows[0], (rows[1] if candidate else None)
    delta = None
    if c and b.get("cycles") and c.get("cycles"):
        delta = {"cycles_ratio": c["cycles"] / b["cycles"], "speedup": b["cycles"] / c["cycles"],
                 "dram_bytes_ratio": (c.get("dram_bytes") or 0) / (b.get("dram_bytes") or 1),
                 "label": vslib.MEASURED if b["label"] == c["label"] == vslib.MEASURED else LPN_ESTIMATE,
                 "note": ("RTL cycles, same workload, same memory model: the candidate change's own effect in simulation"
                          if b["label"] == c["label"] == vslib.MEASURED else
                          "at least one side is an LPN estimate: a model's delta, graded only where a receipt exists")}
    calibration = None
    if board and b.get("device_tok_s"):
        meas = board.get("decode_tok_s_short") or board.get("decode_tok_s_device")
        calibration = {"rtl_device_tok_s": b["device_tok_s"], "board_decode_tok_s": meas,
                       "note": ("the RTL run covers layers=%s of the model: its tok/s is for the cut-down model and is not the board's "
                                "whole-model number; compare bytes/token and port utilization, or run --layers 0" % b.get("layers")),
                       "whole_model_comparable": str(b.get("layers")) in ("0", "all")}
        if calibration["whole_model_comparable"]:
            calibration["rel_error_rtl_vs_board"] = rel_err(b["device_tok_s"], meas)
    report = {"title": title, "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(), "workload": workload_key,
              "rows": rows, "delta": delta, "visualsim": vs, "board": board, "calibration": calibration,
              "metric_notes": METRIC_NOTES,
              "waiting": [x for x in [
                  None if any(r.get("status") == "EXECUTED" for r in vs["baseline"]) else "VisualSim baseline run (seat / execution)",
                  None if (not candidate or any(r.get("status") == "EXECUTED" for r in vs["candidate"])) else "VisualSim candidate run",
                  "nightly FPGA build + matched board bench of the candidate (needs approval)",
                  None if board else "board receipt for this profile / workload",
              ] if x]}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with open(out_dir / "report.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["name", "label", "cycles", "core_mhz", "device_ms_per_token", "device_tok_s", "dram_bytes", "achieved_GBps",
                "portb_utilization", "roofline_efficiency", "mxu_starved_cycles", "mxu_blocked_pct", "max_outstanding_reads_est", "sim_wall_s"]
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(k) for k in cols])
        for k in ("baseline", "candidate"):
            for r in vs[k]:
                w.writerow([r["name"] + f" ({k})", r.get("label"), None, None, None, r.get("device_tok_s"), None, r.get("achieved_GBps"), None, None, None, None, None, r.get("wall_s")])
        if board:
            w.writerow([board["name"], board["label"], None, None, None, board.get("decode_tok_s_short") or board.get("decode_tok_s_device"), None, None, None, None, None, None, None, None])
    (out_dir / "report.md").write_text(markdown(report))
    return report


def _f(v, fmt="{:.4g}"):
    return "-" if v is None else (fmt.format(v) if isinstance(v, (int, float)) else str(v))


def markdown(rep: dict) -> str:
    L = [f"# {rep['title']}", "", f"Generated {rep['generated_utc']}; workload `{rep['workload']}`.", "",
         "Labels: **measured** = a receipt (board, or an RTL simulation as stated); **calibrated estimate** = fitted to receipts, error stated; "
         "**unvalidated scenario** = no receipt checks it; **lpn-estimate** = the LPN model's cycles (its error against a Verilator receipt "
         "is in its run's lpn/compare.json; outside its validated configuration it is a projection); **waiting** = evidence does not exist yet.", "",
         "## RTL (Verilator) receipts and LPN estimates", "",
         "| run | label | layers | cycles | core MHz | ms/token (device) | tok/s (device, cut-down model) | DRAM bytes | GB/s achieved | port B util | roofline eff | MXU starved cyc | MXU blocked % | max outstanding reads (est) | sim wall s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["rows"]:
        L.append("| " + " | ".join(_f(r.get(k)) for k in ("name", "label", "layers", "cycles", "core_mhz", "device_ms_per_token", "device_tok_s", "dram_bytes",
                                                        "achieved_GBps", "portb_utilization", "roofline_efficiency", "mxu_starved_cycles", "mxu_blocked_pct",
                                                        "max_outstanding_reads_est", "sim_wall_s")) + " |")
    for r in rep["rows"]:
        if r.get("command"):
            L += ["", f"`{r['name']}`: `{r['command']}`"]
        if r.get("isa_check"):
            L.append(f"  ISA-simulator check: {r['isa_check']}")
        if r.get("label") == LPN_ESTIMATE:
            L.append(f"  LPN estimate: {r.get('calibration')}; evidence {r.get('evidence')}")
        if r.get("phases"):
            L += ["", f"Phases ({r['name']}): " + "; ".join(f"{k} {v['cycles']} cyc, {v['bytes']} B, VPU {v['vpu_busy']}" for k, v in r["phases"].items())]
    if rep.get("delta"):
        d = rep["delta"]
        L += ["", f"**Candidate vs baseline (RTL, {d['label']}):** cycles x{d['cycles_ratio']:.4f} (speedup {d['speedup']:.3f}x); DRAM bytes x{d['dram_bytes_ratio']:.3f}. {d['note']}"]
    L += ["", "## VisualSim", ""]
    any_vs = False
    for k in ("baseline", "candidate"):
        for r in rep["visualsim"][k]:
            any_vs = True
            L.append(f"- {k} `{r['profile']}`: status **{r['status']}** ({r['label']}); " +
                     (f"tok/s est {_f(r.get('device_tok_s'))}, {_f(r.get('achieved_GBps'))} GB/s, mean latency {_f(r.get('mean_latency_s'))} s, credit-blocked {_f(r.get('credit_blocked'))}, wall {_f(r.get('wall_s'))} s"
                      if r.get("status") == "EXECUTED" else str(r.get("note"))))
    if not any_vs:
        L.append("- no VisualSim run recorded (waiting)")
    L += ["", "## Board receipt", ""]
    b = rep.get("board")
    if b:
        L.append(f"- `{b['name']}` ({b['label']}, {b.get('date')}): " + ", ".join(f"{k} {v}" for k, v in b.items() if isinstance(v, (int, float, list)) and k not in ("pin",)))
        L.append(f"  pin: `{json.dumps(b.get('pin'))}`; tool: {b.get('tool')}")
    else:
        L.append("- none for this profile / workload (waiting)")
    if rep.get("calibration"):
        c = rep["calibration"]
        L += ["", "## Calibration", "", f"- RTL device tok/s {_f(c['rtl_device_tok_s'])} vs board decode tok/s {_f(c['board_decode_tok_s'])}; " +
              (f"relative error {c['rel_error_rtl_vs_board']:+.2%}" if c.get("rel_error_rtl_vs_board") is not None else "not comparable: " + c["note"])]
    L += ["", "## Waiting", ""] + [f"- {w}" for w in rep["waiting"]] + ["", "## Metric definitions", ""] + [f"- `{k}`: {v}" for k, v in rep["metric_notes"].items()] + [""]
    return "\n".join(L)
