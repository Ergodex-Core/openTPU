"""Verilator receipts of one decode token on openTPU's RTL, as the VisualSim side consumes them.

A run is `tools/perf_qwen.py` on an exported, pinned source tree (`run_rtl`): the board's
memory path (otpu_native_dram on OTPU_NCH channels in front of otpu_native_mem.sv) with the
real weights, `--json` for the structured summary and `+nat_trace=FILE` for every memory
command the channels take (core cycle, channel, write flag, beat). `extract` turns the run's
perf.log, perf.jsonl and nat_trace into trace.json (memory-inclusive cycles, bytes, per-channel
reads / writes, outstanding reads, queue stalls, the MXU's starved / blocked cycles, the
phases) and mem_windows.csv (per window of W core cycles: reads and writes per channel), the
workload the VisualSim model replays.

Everything here is labelled `measured` in the sense of an RTL simulation receipt: cycles of
the RTL under a memory model, not of a board.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from . import vslib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

RTL_GLOBS = ("rtl/**/*.sv", "rtl/**/*.v", "rtl/**/*.svh", "sim/verilator/*.sv", "sim/verilator/*.v",
             "opentpu/rtlsim.py", "opentpu/profile.py", "tools/perf_qwen.py")


# ----------------------------------------------------------------------------- running

def export_tree(repo: Path, rev: str, dest: Path, models_link: Path | None = None) -> dict:
    """`git archive rev | tar -x` into dest (a clean copy of the pinned source; never another
    worktree), with models/ linked in. Returns the pin (full SHA, describe)."""
    dest.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", rev], text=True).strip()
    if not any(dest.iterdir()):
        ar = subprocess.Popen(["git", "-C", str(repo), "archive", sha], stdout=subprocess.PIPE)
        subprocess.check_call(["tar", "-x", "-C", str(dest)], stdin=ar.stdout)
        ar.wait()
        if ar.returncode:
            raise RuntimeError("git archive failed")
    if models_link and not (dest / "models").exists():
        (dest / "models").symlink_to(models_link)
    return {"repo": str(repo), "rev": rev, "sha": sha, "export": str(dest)}


def run_rtl(tree: Path, out: Path, *, python: str, model: str = "gemma3", layers: int = 2,
            wformat: str = "int8", head_format: str = "int8", lat: int = 60, mhz: float = 200.0,
            target: str = "f2-hbm", nch: int = 8, portb: int | None = None, extra: list[str] | None = None,
            check: bool = False, env: dict | None = None, dry: bool = False) -> dict:
    """One perf_qwen run on the exported tree; returns the receipt (command, env, wall, exit)."""
    out.mkdir(parents=True, exist_ok=True)
    cmd = [python, "tools/perf_qwen.py", "--model", model, "--layers", str(layers), "--wformat", wformat,
           "--head-format", head_format, "--lat", str(lat), "--mhz", str(mhz),
           "--json", str(out / "perf.jsonl"), "--plus", f"+nat_trace={out / 'nat_trace.txt'}"]
    if portb is not None:
        cmd += ["--portb", str(portb)]
    if check:
        cmd.append("--check")
    cmd += extra or []
    e = {**os.environ, "OTPU_TARGET": target, "OTPU_NCH": str(nch), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8")}
    e.update(env or {})
    recorded_env = {k: e[k] for k in sorted(e) if k.startswith("OTPU_") or k == "OMP_NUM_THREADS"}
    receipt = {"tree": str(tree), "command": cmd, "env": recorded_env,
               "shell": "cd " + shlex.quote(str(tree)) + " && " + " ".join(f"{k}={v}" for k, v in recorded_env.items())
               + " " + " ".join(shlex.quote(c) for c in cmd)}
    (out / "command.sh").write_text("#!/bin/sh\n" + receipt["shell"] + "\n")
    if dry:
        receipt["status"] = "DRY"
        return receipt
    t0 = time.monotonic()
    with open(out / "perf.log", "w") as log:
        rc = subprocess.call(cmd, cwd=tree, env=e, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(exit_code=rc, wall_s=time.monotonic() - t0, status="OK" if rc == 0 else "FAILED")
    (out / "run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


# ----------------------------------------------------------------------------- parsing

_NUM = r"([\d,]+)"


def _i(s: str) -> int:
    return int(s.replace(",", ""))


def parse_perf_log(text: str) -> dict:
    """The numbers perf_qwen prints (the run's summary lines)."""
    d: dict = {}
    m = re.search(r"layers=(\S+) pos=(\d+) bw=(\d+)% lat=(\S+): (\d+) cycles \((\d+)s sim\), core-port roofline (\d+) cycles "
                  r"\(port B (\d+) chunks, port A (\d+), MXU (\d+)\), efficiency ([\d.]+)%", text)
    if m:
        d.update(layers=m.group(1), pos=int(m.group(2)), bw_pct=int(m.group(3)), lat=m.group(4),
                 cycles=int(m.group(5)), sim_wall_s=int(m.group(6)), roofline_cycles=int(m.group(7)),
                 portb_chunks=int(m.group(8)), porta_cycles=int(m.group(9)), mxu_cycles=int(m.group(10)),
                 roofline_efficiency=float(m.group(11)) / 100)
    m = re.search(r"DRAM read / write beats per channel: ([^;\n]+);", text)
    if m:
        pairs = re.findall(r"(\d+) / (\d+)", m.group(1))
        d["channels"] = [{"reads": int(a), "writes": int(b)} for a, b in pairs]
    m = re.search(r"MXU starved .*? (\d+) cycles = ([\d.]+)%; blocked \(chunks but no MAC\) ([\d.]+)%; MAC ([\d.]+)%; "
                  r"DRAM read beats/cycle/channel ([\d.]+)", text)
    if m:
        d.update(mxu_starved_cycles=int(m.group(1)), mxu_starved_pct=float(m.group(2)),
                 mxu_blocked_pct=float(m.group(3)), mxu_mac_pct=float(m.group(4)),
                 read_beats_per_cycle_per_channel=float(m.group(5)))
    for m in re.finditer(r"ch(\d+): port A reads (\d+), DDR3 row opens (\d+), read-modify-writes (\d+)", text):
        c = int(m.group(1))
        if "channels" in d and c < len(d["channels"]):
            d["channels"][c].update(porta_reads=int(m.group(2)), row_opens=int(m.group(3)), rmw=int(m.group(4)))
    m = re.search(r"useful bytes (\d+) -> (\d+) cycles at (\d+) B/cycle", text)
    if m:
        d.update(useful_bytes=int(m.group(1)), useful_roofline_cycles=int(m.group(2)), port_bytes_per_cycle=int(m.group(3)))
    m = re.search(r"at ([\d.]+) MHz: ([\d.]+) ms/token, ([\d.]+) tok/s", text)
    if m:
        d.update(core_mhz=float(m.group(1)), ms_per_token=float(m.group(2)), tok_s_device=float(m.group(3)))
    m = re.search(r"unit busy: DMA (\d+)%\s+MXU (\d+)%\s+QUANT (\d+)%\s+VPU (\d+)%", text)
    if m:
        d["unit_busy_pct"] = {"DMA": int(m.group(1)), "MXU": int(m.group(2)), "QUANT": int(m.group(3)), "VPU": int(m.group(4))}
    m = re.search(r"DRAM port B busy (\d+)% \(MXU (\d+), DMA (\d+)\)", text)
    if m:
        d.update(portb_busy_pct=int(m.group(1)), portb_mxu_chunks=int(m.group(2)), portb_dma_chunks=int(m.group(3)))
    phases = {}
    for m in re.finditer(r"^  (\w[\w ]*?)\s+(\d+)\s+([\d.]+)%\s+(\d+) B ->\s+(\d+) \(\s*([\d.]+)%\)\s+VPU\s+(\d+)\s+MXU\s+(\d+)", text, re.M):
        phases[m.group(1).strip()] = {"cycles": int(m.group(2)), "share_pct": float(m.group(3)), "bytes": int(m.group(4)),
                                      "roofline_cycles": int(m.group(5)), "vpu_busy": int(m.group(7)), "mxu_busy": int(m.group(8))}
    if phases:
        d["phases"] = phases
    m = re.search(r"MXU prologue (\d+) epilogue (\d+) gaps (\d+)", text)
    if m:
        d["mxu_gaps"] = {"prologue": int(m.group(1)), "epilogue": int(m.group(2)), "gaps": int(m.group(3))}
    m = re.search(r"wide reads[^\n]*", text)
    if m:
        d["wide_reads_line"] = m.group(0)
    m = re.search(r"ISA simulator.*?(match|mismatch|identical|differ)[^\n]*", text, re.I)
    if m:
        d["isa_check_line"] = m.group(0)
    return d


def parse_perf_jsonl(p: Path) -> dict | None:
    if not p.is_file():
        return None
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else None


def bin_nat_trace(p: Path, window: int, nch: int, lat: int) -> dict:
    """Bin the channels' command trace into windows of `window` core cycles: per channel reads
    and writes, and from the issue times an estimate of the reads outstanding (a read is
    outstanding from its command for `lat` cycles plus the jitter the model adds: the lower
    bound the trace supports). Lines: `cycle channel we beat` (whitespace or commas)."""
    rows: dict[int, list[int]] = {}
    total_r = total_w = 0
    first = last = None
    max_out = 0
    outstanding: list[int] = []           # issue cycles of reads not yet (lat) returned
    gaps: list[int] = []
    prev_cycle = None
    with open(p) as f:
        for ln in f:
            parts = re.split(r"[\s,]+", ln.strip())
            if len(parts) < 4:
                continue
            try:
                cyc, ch, we = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            first = cyc if first is None else first
            last = cyc
            w = cyc // window
            r = rows.setdefault(w, [0] * (2 * nch))
            if we:
                r[nch + ch] += 1
                total_w += 1
            else:
                r[ch] += 1
                total_r += 1
                outstanding.append(cyc)
                while outstanding and outstanding[0] + lat <= cyc:
                    outstanding.pop(0)
                if len(outstanding) > max_out:
                    max_out = len(outstanding)
            if prev_cycle is not None and cyc - prev_cycle > 64:
                gaps.append(cyc - prev_cycle)
            prev_cycle = cyc
    span = (last - first + 1) if first is not None else 0
    return {"window_cycles": window, "channels": nch, "reads": total_r, "writes": total_w,
            "first_cycle": first, "last_cycle": last, "active_span_cycles": span,
            "beats_per_cycle": (total_r + total_w) / span if span else 0.0,
            "max_outstanding_reads_est": max_out, "idle_gaps_over_64": len(gaps),
            "idle_cycles_in_gaps": sum(gaps), "rows": rows}


def write_windows_csv(binned: dict, dest: Path) -> int:
    nch = binned["channels"]
    with open(dest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "cycle_start"] + [f"rd_ch{c}" for c in range(nch)] + [f"wr_ch{c}" for c in range(nch)])
        n = 0
        for k in sorted(binned["rows"]):
            w.writerow([k, k * binned["window_cycles"]] + binned["rows"][k])
            n += 1
    return n


def extract(run_dir: Path, *, window: int = 1024, nch: int | None = None, lat: int | None = None,
            beat_bytes: int = 64, chunk_bytes: int = 128) -> dict:
    """perf.log + perf.jsonl + nat_trace.txt -> trace.json and mem_windows.csv in run_dir."""
    log = (run_dir / "perf.log").read_text(errors="replace") if (run_dir / "perf.log").is_file() else ""
    summary = parse_perf_log(log)
    js = parse_perf_jsonl(run_dir / "perf.jsonl")
    rr = json.loads((run_dir / "run_receipt.json").read_text()) if (run_dir / "run_receipt.json").is_file() else {}
    nch = nch or int(rr.get("env", {}).get("OTPU_NCH", len(summary.get("channels", [])) or 2))
    lat = lat or int(float(summary.get("lat") or 20))
    trace: dict = {"label": vslib.MEASURED, "kind": "verilator_rtl_decode_token", "run_dir": str(run_dir),
                   "source": rr, "summary": summary, "perf_json": js}
    nt = run_dir / "nat_trace.txt"
    if nt.is_file() and nt.stat().st_size:
        binned = bin_nat_trace(nt, window, nch, lat)
        n = write_windows_csv(binned, run_dir / "mem_windows.csv")
        binned.pop("rows")
        binned["csv_rows"] = n
        binned["csv"] = str(run_dir / "mem_windows.csv")
        binned["csv_sha256"] = vslib.sha256_file(run_dir / "mem_windows.csv")
        trace["memory_trace"] = binned
    cycles = summary.get("cycles") or (js or {}).get("cycles")
    if cycles:
        beats = sum(c["reads"] + c["writes"] for c in summary.get("channels", []))
        rbytes = sum(c["reads"] for c in summary.get("channels", [])) * beat_bytes
        wbytes = sum(c["writes"] for c in summary.get("channels", [])) * beat_bytes
        mhz = summary.get("core_mhz") or 0.0
        trace["derived"] = {
            "cycles": cycles, "core_mhz": mhz,
            "device_ms_per_token": cycles / mhz / 1e3 if mhz else None,
            "device_tok_s": mhz * 1e6 / cycles if mhz else None,
            "dram_read_bytes": rbytes, "dram_write_bytes": wbytes, "dram_bytes": rbytes + wbytes,
            "beats": beats, "bytes_per_cycle": (rbytes + wbytes) / cycles,
            "achieved_GBps": (rbytes + wbytes) / cycles * mhz * 1e6 / 1e9 if mhz else None,
            "portb_chunk_bytes": chunk_bytes,
            "portb_utilization": summary.get("portb_chunks", 0) / cycles if summary.get("portb_chunks") else None,
            "roofline_efficiency": summary.get("roofline_efficiency"),
            "mxu_starved_cycles": summary.get("mxu_starved_cycles"),
            "mxu_blocked_pct": summary.get("mxu_blocked_pct"),
            "note": "device cycles of one decode token at the run's position; no host, no sampling",
        }
    out = run_dir / "trace.json"
    out.write_text(json.dumps(trace, indent=2) + "\n")
    return trace


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="extract trace.json / mem_windows.csv from a perf_qwen run directory")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--nch", type=int)
    ap.add_argument("--lat", type=int)
    a = ap.parse_args(argv)
    t = extract(a.run_dir, window=a.window, nch=a.nch, lat=a.lat)
    print(json.dumps({k: t[k] for k in ("derived", "memory_trace") if k in t}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
