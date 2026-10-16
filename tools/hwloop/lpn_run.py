#!/usr/bin/env python3
"""The LPN tier of the hardware loop (`cli.py lpn-run`): the frozen export's decode token through
MJChku/LPNDialect's examples/opentpu model, graded against the Verilator receipts.

    python tools/hwloop/lpn_run.py --tree EXPORT --out RUN [--python VENV] [--lpn CLONE] [--pin PIN_TREE]
        [--llvm-bin DIR] [--model gemma3 --layers 2 --pos 9 --wformat int8 --head-format int8 --portb 1
         --target f2-hbm --nch 8 --lat 60 --mhz 200] [--dram-latency 8] [--batch-rows 8] [--expanded]
        [--compare RECEIPT_RUN_DIR] [--rtl-pin] [--uncalibrated] [--manifest-only]
        [--portb 2|4 | --rowlanes [--mcols 8] | --channels] [--grade]      # the wide model (lpn/wide.py)

Steps, each with its receipt in RUN/lpn/:
  1. manifest.json: the RTL fingerprint of the tree the model is validated against (upstream's
     pin f237dab2 with its observer patch applied: `--pin`), the LPN commit, the patches' and the
     compiler's SHA-256, the validated configuration. `check-stale RUN/lpn --tree EXPORT` reads it.
  2. staleness: the export's RTL against that fingerprint. An export whose RTL moved is REFUSED
     with the changed / added / removed files listed (the VisualSim rule), unless --uncalibrated,
     which runs anyway and labels the result "outside validated configuration" with the list.
  3. input/: lpn/inputs.py on the export (its compiler and ISA simulator; the same program and
     DRAM image perf_qwen hands the testbench; the ISA simulator's DRAM / TMEM hashes).
  4. model.mlir + opentpu: lpn/emit.py with LPNDialect on PYTHONPATH and the pin tree as
     --opentpu-root (upstream's frontend decodes and executes the program with the pin's ISA
     simulator; its hashes must equal the export's: the two trees agree on the program).
  5. the native run (--stats --wall-start=begin --wall-end=complete): cycles, zero rollbacks,
     program_completed.
  6. trace.json: the run-rtl / report record shape (cycles, bytes, port utilization, tok/s),
     label `lpn-estimate`; compare.json against --compare's trace.json (error, the 5% gate, the
     configuration deltas); optionally (--rtl-pin) the pinned RTL on the same bundle (lpn/rtl_pin.py:
     the model's own validated configuration) and its error.

Two models. PORTB 1 without --channels is upstream's CompactOpenTPU on the pinned tree's
frontend (steps 1-2 above: the pin's fingerprint, the staleness rule). PORTB 2 / 4, --rowlanes
and --channels (NCH > 1 at the receipt's latency) are lpn/wide.py's WideCompactOpenTPU on the
export's own frontend, and their rule is the graded-configuration registry
(tools/hwloop/lpn/validated.json): a configuration is accepted only with a graded point (an
LPN run compared with a Verilator receipt, `--compare RECEIPT --grade` writes it) and only on
the export whose RTL fingerprint that point was graded on; anything else is
REFUSED_NOT_MODELLED (no point) or REFUSED_STALE_MODEL (another export), unless --uncalibrated.
A graded point over the 5% gate stays in the registry labelled uncalibrated.

Only the standard library here; the helpers run under --python (OTPU_PYTHON) as subprocesses.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from tools.hwloop import trace as _trace, vslib  # noqa: E402
from tools.hwloop.lpn import LPN_GATE, LPN_LABEL, LPN_RTL_PIN, REJECTED, VALIDATED  # noqa: E402

ROOT = HERE.parents[1]
SCRATCH = Path(os.environ.get("HWLOOP_LPN_SCRATCH", str(Path.home() / "ergodex-03" / "lpn-scratch")))
DEFAULTS = {
    "python": os.environ.get("OTPU_PYTHON", str(Path.home() / "ergodex-03" / "bonsai2-venv" / "bin" / "python")),
    "lpn": os.environ.get("HWLOOP_LPN", str(SCRATCH / "LPNDialect")),
    "lpn_build": os.environ.get("HWLOOP_LPN_BUILD", str(SCRATCH / "lpn-build")),
    "pin": os.environ.get("HWLOOP_LPN_PIN", str(SCRATCH / "openTPU-lpn-pin")),
    "llvm_bin": os.environ.get("HWLOOP_LLVM20_BIN", str(SCRATCH / "llvm20-bin")),
}
PATCHES = ("examples/opentpu/verilator-observer.patch", "examples/opentpu/verilator-large-memory.patch")


def _git(repo: Path, *args) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------- the graded registry

REGISTRY = HERE / "lpn" / "validated.json"
KEY_FIELDS = ("model", "layers", "pos", "wformat", "head_format", "portb", "rowlanes", "mcols", "target", "nch", "lat")


def config_key(a) -> dict:
    return {"model": a.model, "layers": str(a.layers), "pos": int(a.pos), "wformat": a.wformat, "head_format": a.head_format,
            "portb": int(a.portb or 1), "rowlanes": bool(a.rowlanes), "mcols": int(a.mcols or 4), "target": a.target,
            "nch": int(a.nch), "lat": int(a.lat)}


def load_registry(path: Path = REGISTRY) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {"note": "the wide model's graded configurations: lpn_run --compare RECEIPT --grade writes them",
            "exports": {}, "configurations": []}


def find_graded(reg: dict, key: dict) -> dict | None:
    for e in reg.get("configurations", []):
        if all(e.get("key", {}).get(k) == key[k] for k in KEY_FIELDS):
            return e
    return None


def target_fifo_entries(tree: Path, target: str, default: int = 128) -> int:
    """The target's [uarch] FIFO_DEPTH (targets/<target>.toml; the MXU chunk FIFO's entries)."""
    name = target.split(":")[0]
    p = tree / "targets" / f"{name}.toml"
    if p.is_file():
        m = re.search(r"^FIFO_DEPTH\s*=\s*(\d+)", p.read_text(), re.M)
        if m:
            return int(m.group(1))
    return default


# ----------------------------------------------------------------------------- the manifest

def manifest(lpn: Path, lpn_build: Path, pin: Path, wide: dict | None = None) -> dict:
    """What the LPN model is: its commit, its patches, its compiler, and the RTL fingerprint of
    the pinned tree it is validated against (the observer patch applied there). `wide`: the
    wide model instead ({"export": sha, "fingerprint": ..., "graded": entry or None}): the
    fingerprint is the wide-port export's the configuration was graded on."""
    if wide is not None:
        fp = wide.get("fingerprint")
    else:
        fp = vslib.fingerprint_tree(pin, _trace.RTL_GLOBS) if pin.is_dir() else None
    compiler = lpn_build / "tools" / "lpn-compile" / "lpn-compile"
    patches = {}
    for rel in PATCHES:
        p = lpn / rel
        # applied: the patch reverses cleanly on the pin tree
        applied = (p.is_file() and pin.is_dir()
                   and subprocess.run(["git", "-C", str(pin), "apply", "--check", "--reverse", str(p)],
                                      capture_output=True).returncode == 0)
        patches[rel] = {"sha256": vslib.sha256_file(p) if p.is_file() else None, "applied_to_pin": applied}
    man = {"generator": "opentpu-hwloop-lpn/0.1", "label": LPN_LABEL, "model": "MJChku/LPNDialect examples/opentpu",
           "lpn_repo": str(lpn), "lpn_commit": _git(lpn, "rev-parse", "HEAD"), "lpn_branch": _git(lpn, "rev-parse", "--abbrev-ref", "HEAD"),
           "lpn_compiler": str(compiler), "lpn_compiler_sha256": vslib.sha256_file(compiler) if compiler.is_file() else None,
           "patches": patches, "validated_rtl_pin": LPN_RTL_PIN, "pin_tree": str(pin), "pin_head": _git(pin, "rev-parse", "HEAD"),
           "pin_tree_status": _git(pin, "status", "--short").splitlines(),
           "rtl_fingerprint": fp,
           "rtl_fingerprint_note": "the pinned tree with the observer patch applied: what upstream validated the model on; "
                                   "check-stale compares an export's rtl/**, sim/verilator/*, rtlsim.py, profile.py, perf_qwen.py against it",
           "validated_configuration": VALIDATED, "rejected_isa_features": list(REJECTED)}
    if wide is not None:
        wm = HERE / "lpn" / "wide.py"
        man.update(model="tools/hwloop/lpn/wide.py WideCompactOpenTPU over MJChku/LPNDialect examples/opentpu",
                   wide_model_sha256=vslib.sha256_file(wm) if wm.is_file() else None,
                   validated_rtl_pin=wide.get("export"), graded=wide.get("graded"),
                   rtl_fingerprint_note="the wide-port export the configuration was graded on (tools/hwloop/lpn/validated.json); "
                                        "lpn-run accepts that export and refuses others (REFUSED_STALE_MODEL)")
    return man


# ----------------------------------------------------------------------------- the record

NATIVE_RE = {"cycles": r"Simulation finished at t=([0-9.]+)", "wall_s": r"Wall window elapsed: ([0-9.]+) seconds",
             "rollbacks": r"rollbacks: (\d+)", "completed_tokens": r"place program_completed: (\d+) token",
             "guard_checks": r"guard checks: (\d+)", "firings": r"body executions: (\d+)"}


def parse_native(text: str) -> dict:
    """The native binary's --stats lines. The wide model's channel credits return at fractional
    cycles (a batch's beats at the channel's rate), so the finish time may be fractional: the
    cycle count is its ceiling (`cycles_exact` keeps the value)."""
    d: dict = {}
    for k, rx in NATIVE_RE.items():
        m = re.search(rx, text)
        if m:
            d[k] = float(m.group(1)) if k in ("cycles", "wall_s") else int(m.group(1))
    if "cycles" in d:
        d["cycles_exact"] = d["cycles"]
        d["cycles"] = float(math.ceil(d["cycles"]))
    d["complete"] = d.get("completed_tokens") == 1 and d.get("rollbacks") == 0 and "cycles" in d
    return d


def record(run: Path, args: dict, work: dict, emit: dict, native: dict, man: dict, stale: dict,
           calibration: str, mhz: float, D: int = 128, graded: dict | None = None) -> dict:
    cycles = native.get("cycles")
    isa = work.get("isa", {})
    rbytes, wbytes = isa.get("dram_read_bytes_footprint", 0), isa.get("dram_write_bytes_footprint", 0)
    chunks = isa.get("mm_weight_chunks")
    portb = args.get("portb") or 1
    derived = None
    if cycles:
        derived = {"cycles": int(cycles), "core_mhz": mhz,
                   "device_ms_per_token": cycles / mhz / 1e3 if mhz else None,
                   "device_tok_s": mhz * 1e6 / cycles if mhz else None,
                   "dram_read_bytes": rbytes, "dram_write_bytes": wbytes, "dram_bytes": rbytes + wbytes,
                   "bytes_per_cycle": (rbytes + wbytes) / cycles, "achieved_GBps": (rbytes + wbytes) / cycles * mhz * 1e6 / 1e9 if mhz else None,
                   "portb_chunk_bytes": D,
                   "portb_utilization": chunks / cycles / portb if chunks else None,
                   "roofline_efficiency": chunks / portb / cycles if chunks else None,
                   "mxu_starved_cycles": None, "mxu_blocked_pct": None,
                   "note": "LPN model cycles of one decode token (no RTL, no host); bytes are the ISA footprints' DRAM ranges, "
                           "port B utilization = the MM weight chunks / cycles / PORTB (the MXU stream's share of the port)"}
    summary = {"layers": args.get("layers"), "pos": args.get("pos"), "lat": str(emit.get("timing", {}).get("dram_latency")),
               "cycles": int(cycles) if cycles else None, "sim_wall_s": native.get("wall_s"),
               "roofline_cycles": (chunks // portb) if chunks else None, "portb_chunks": chunks,
               "core_mhz": mhz, "instructions_static": work.get("instructions_static"),
               "dynamic_commands": isa.get("dynamic_commands"), "opcodes": work.get("opcodes_static"),
               "rejected_features_static": work.get("rejected_features_static")}
    mem = emit.get("memory") or {}
    if derived and mem.get("roofline"):
        derived["lpn_roofline"] = mem["roofline"]
        derived["two_part_reads"] = {"wide_reads": mem.get("wide_reads"), "two_part": mem.get("two_part"), "share": mem.get("two_part_share")}
    return {"label": LPN_LABEL, "kind": "lpn_decode_token", "calibration": calibration, "run_dir": str(run),
            "source": {"tree": args.get("tree"), "command": args.get("command"), "env": args.get("env"), "shell": args.get("shell")},
            "summary": summary, "derived": derived,
            "lpn": {"commit": man.get("lpn_commit"), "branch": man.get("lpn_branch"), "compiler_sha256": man.get("lpn_compiler_sha256"),
                    "patches": man.get("patches"), "pin": man.get("validated_rtl_pin"), "pin_head": man.get("pin_head"),
                    "model": emit.get("model") or ("CompactOpenTPU" if emit.get("compact") else "OpenTPU"),
                    "wide_model_sha256": man.get("wide_model_sha256"), "graded": graded, "memory": mem or None,
                    "timing": emit.get("timing"), "compact": emit.get("compact"), "matrix_batch_rows": emit.get("matrix_batch_rows"),
                    "net_statistics": emit.get("net_statistics"), "frontend": emit.get("frontend"),
                    "model_build_s": emit.get("model_build_s"), "compile_s": emit.get("compile_s"), "native": native,
                    "numerical_check": {"export_isa": {k: isa.get(k) for k in ("dram_sha256", "tmem_sha256")},
                                        "pin_frontend": {"dram_sha256": (emit.get("frontend", {}).get("dram_sha256") or [None])[0],
                                                         "tmem_sha256": (emit.get("frontend", {}).get("tmem_sha256") or [None])[0]},
                                        "match": (isa.get("dram_sha256") == (emit.get("frontend", {}).get("dram_sha256") or [None])[0]
                                                  and isa.get("tmem_sha256") == (emit.get("frontend", {}).get("tmem_sha256") or [None])[0])}},
            "staleness": {k: v for k, v in stale.items() if k != "entries"},
            "validated_configuration": VALIDATED}


def compare(lpn_trace: dict, receipt_dir: Path, gate: float = LPN_GATE) -> dict:
    """LPN against a Verilator receipt (its trace.json): the error, the gate, what the two
    configurations share (the program: same bundle, same chunks) and where they differ."""
    r = json.loads((receipt_dir / "trace.json").read_text())
    rd, rs, rr = r.get("derived", {}), r.get("summary", {}), r.get("source", {})
    ld, ls = lpn_trace.get("derived") or {}, lpn_trace.get("summary", {})
    out = {"receipt": str(receipt_dir), "receipt_label": r.get("label"), "receipt_kind": r.get("kind"),
           "rtl_cycles": rd.get("cycles"), "lpn_cycles": ld.get("cycles"), "gate": gate}
    if rd.get("cycles") and ld.get("cycles"):
        out["rel_error"] = ld["cycles"] / rd["cycles"] - 1
        out["within_gate"] = abs(out["rel_error"]) <= gate
    out["rtl_portb_chunks"] = rs.get("portb_mxu_chunks") or rs.get("portb_chunks")
    out["lpn_portb_chunks"] = ls.get("portb_chunks")
    out["same_mxu_chunks"] = (out["rtl_portb_chunks"] == out["lpn_portb_chunks"]) if out["rtl_portb_chunks"] and out["lpn_portb_chunks"] else None
    out["rtl_dram_bytes_beats"] = rd.get("dram_bytes")
    out["lpn_dram_bytes_footprint"] = ld.get("dram_bytes")
    if rd.get("dram_bytes") and ld.get("dram_bytes"):
        out["bytes_rel"] = ld["dram_bytes"] / rd["dram_bytes"] - 1
    out["rtl_portb_utilization"] = rd.get("portb_utilization")
    out["lpn_portb_utilization"] = ld.get("portb_utilization")
    env = rr.get("env", {})
    cmd = rr.get("command") or []
    portb = int(cmd[cmd.index("--portb") + 1]) if "--portb" in cmd else 1
    rowlanes = "--rowlanes" in cmd
    mcols = int(cmd[cmd.index("--mcols") + 1]) if "--mcols" in cmd else 4
    out["receipt_configuration"] = {"portb": portb, "rowlanes": rowlanes, "mcols": mcols, "target": env.get("OTPU_TARGET"),
                                    "nch": env.get("OTPU_NCH"), "lat": rs.get("lat"), "layers": rs.get("layers")}
    deltas = []
    v = VALIDATED
    timing = (lpn_trace.get("lpn") or {}).get("timing") or {}
    wide = (lpn_trace.get("lpn") or {}).get("model", "").startswith("Wide")
    if wide:
        tp, tn = timing.get("port_chunks"), timing.get("nch")
        same = (tp == portb and str(tn) == str(env.get("OTPU_NCH") or 1) and str(timing.get("dram_latency")) == str(rs.get("lat"))
                and bool(lpn_trace.get("lpn", {}).get("memory", {}).get("mm_rl")) == rowlanes)
        out["configuration_deltas"] = [] if same else [
            f"the LPN run's configuration (PORTB {tp}, NCH {tn}, lat {timing.get('dram_latency')}, row lanes "
            f"{bool(lpn_trace.get('lpn', {}).get('memory', {}).get('mm_rl'))}) is not the receipt's (PORTB {portb}, NCH {env.get('OTPU_NCH')}, "
            f"lat {rs.get('lat')}, row lanes {rowlanes})"]
        out["overlap"] = ("the same program and DRAM image (the export's compiler), the same MM weight chunks" if out.get("same_mxu_chunks")
                          else "the same workload arguments; chunk counts differ or are missing")
        st = lpn_trace.get("staleness", {})
        if same and st.get("stale"):
            out["configuration_deltas"].append(f"the receipt's export is not the one the configuration was graded on: "
                                               f"{len(st.get('changed', []))} changed, {len(st.get('added', []))} added, {len(st.get('removed', []))} removed")
        out["verdict"] = ("not comparable: " + out["configuration_deltas"][0] if out["configuration_deltas"] else
                          ("graded configuration (the wide model): inside the %d%% gate" % round(gate * 100) if out.get("within_gate")
                           else "graded configuration (the wide model): over the %d%% gate, uncalibrated" % round(gate * 100)))
        return out
    if env.get("OTPU_TARGET") and env.get("OTPU_TARGET") not in ("ypcb-00338",):
        deltas.append(f"receipt target {env.get('OTPU_TARGET')} with the board micro-architecture (rtlsim.BOARD_UARCH: MXU_E 4, "
                      f"one-copy pumped TMEM, WIN 16) vs the model's validated testbench default ({v['epilogue_lanes']} epilogue lanes, replicated TMEM)")
    nch = env.get("OTPU_NCH")
    if nch and int(nch) != 1:
        deltas.append(f"receipt memory: otpu_native_dram over NCH {nch} channels at --lat {rs.get('lat')} cycles vs the model's "
                      f"{v['memory']}" + (f" (this run's Timing.dram_latency {ls.get('lat')})" if ls.get("lat") else ""))
    elif rs.get("lat") and str(rs.get("lat")) != str(v["dram_latency"]):
        deltas.append(f"receipt --lat {rs.get('lat')} vs the model's validated {v['dram_latency']}")
    if portb != v["PORTB"] or rowlanes:
        deltas.append(f"receipt PORTB {portb}{' with row lanes' if rowlanes else ''}: not this model (one port-B credit, one chunk a cycle); "
                      f"the wide model (lpn-run --portb / --rowlanes) is the one graded on it")
    if rr.get("tree") and lpn_trace.get("staleness", {}).get("stale"):
        st = lpn_trace["staleness"]
        deltas.append(f"receipt RTL tree {rr.get('tree')} is not the model's pin {LPN_RTL_PIN[:8]}: "
                      f"{len(st.get('changed', []))} changed, {len(st.get('added', []))} added, {len(st.get('removed', []))} removed fingerprinted files")
    out["configuration_deltas"] = deltas
    out["overlap"] = ("the same program and DRAM image (the export's compiler), the same MM weight chunks" if out.get("same_mxu_chunks")
                      else "the same workload arguments; chunk counts differ or are missing")
    out["verdict"] = ("not comparable: PORTB %d%s is outside this model" % (portb, " with row lanes" if rowlanes else "") if (portb != v["PORTB"] or rowlanes) else
                      ("outside the validated configuration (%d deltas): the error is recorded, not a calibration" % len(deltas) if deltas
                       else "validated configuration"))
    return out


# ----------------------------------------------------------------------------- the run

def run(a) -> dict:
    out = Path(a.out).resolve()
    lp = out / "lpn"
    lp.mkdir(parents=True, exist_ok=True)
    tree, lpn, lpn_build, pin = Path(a.tree).resolve(), Path(a.lpn).resolve(), Path(a.lpn_build).resolve(), Path(a.pin).resolve()
    receipt: dict = {"status": "STARTED", "started_utc": _now(), "label": LPN_LABEL, "tree": str(tree), "out": str(out),
                     "args": {k: v for k, v in vars(a).items() if k != "fn"}}
    wide = (a.portb or 1) != 1 or bool(a.rowlanes) or bool(a.channels)
    key = config_key(a)
    reg = load_registry()
    graded = find_graded(reg, key) if wide else None
    receipt["model"] = "wide" if wide else "pin"
    receipt["configuration"] = key
    wide_info = None
    tree_fp = None
    if wide:
        export_sha = graded["export"] if graded else None
        fp = reg.get("exports", {}).get(export_sha, {}).get("rtl_fingerprint") if export_sha else None
        if fp is None:
            tree_fp = vslib.fingerprint_tree(tree, _trace.RTL_GLOBS)
            export_sha = _git(tree, "rev-parse", "HEAD") or (json.loads((tree / "hwloop-baseline.json").read_text())["pin"]["sha"]
                                                             if (tree / "hwloop-baseline.json").is_file() else tree_fp["sha256"])
            fp = tree_fp
        wide_info = {"export": export_sha, "fingerprint": fp, "graded": graded}
    man = manifest(lpn, lpn_build, pin, wide=wide_info)
    _write(lp / "manifest.json", man)
    receipt["manifest"] = str(lp / "manifest.json")
    if not wide and man["pin_head"] and not man["pin_head"].startswith(LPN_RTL_PIN[:12]):
        receipt.update(status="REFUSED_PIN_MISMATCH", note=f"--pin {pin} is at {man['pin_head'][:12]}, the model's pin is {LPN_RTL_PIN[:12]}")
        _write(lp / "receipt.json", receipt)
        return receipt
    if a.manifest_only:
        receipt["status"] = "MANIFEST_ONLY"
        _write(lp / "receipt.json", receipt)
        return receipt
    # 2. the rule: the pin's staleness, or the graded registry
    if wide:
        if graded is None and not (a.grade or a.uncalibrated):
            receipt.update(status="REFUSED_NOT_MODELLED",
                           note=f"no graded LPN point for this configuration ({', '.join(f'{k}={v}' for k, v in key.items())}): "
                                f"the wide model is accepted only where it was graded against a Verilator receipt "
                                f"(tools/hwloop/lpn/validated.json). Grade it: --compare RECEIPT --grade; or --uncalibrated to run unlabelled.")
            _write(lp / "receipt.json", receipt)
            return receipt
        stale = vslib.check_stale(man, tree)
        if graded is None:
            stale["note"] = "no graded point: the tree's own fingerprint (this run grades it)" if a.grade else "no graded point: --uncalibrated"
        receipt["staleness"] = {k: v for k, v in stale.items() if k != "entries"}
        _write(lp / "staleness.json", stale)
        if stale["stale"] and not a.uncalibrated:
            receipt.update(status="REFUSED_STALE_MODEL",
                           note=f"the export's RTL is not the wide-port export this configuration was graded on ({graded['export'][:8]}): "
                                f"{len(stale['changed'])} changed, {len(stale['added'])} added, {len(stale['removed'])} removed "
                                f"(lpn/staleness.json). Re-grade it on this export (--compare RECEIPT --grade) or pass --uncalibrated.")
            _write(lp / "receipt.json", receipt)
            return receipt
        if graded is None:
            calibration = "grading: no point yet" if a.grade else "uncalibrated: no graded point for this configuration"
        elif stale["stale"]:
            calibration = (f"uncalibrated: RTL moved from the graded export {graded['export'][:8]} ({len(stale['changed'])} changed, "
                           f"{len(stale['added'])} added, {len(stale['removed'])} removed fingerprinted files)")
        elif graded.get("within_gate"):
            calibration = f"graded: {graded['rel_error']:+.3%} vs {graded['receipt']['cycles']:,} (Verilator, {graded['receipt']['dir']})"
        else:
            calibration = f"uncalibrated: {graded['rel_error']:+.3%} vs {graded['receipt']['cycles']:,} is over the {LPN_GATE:.0%} gate ({graded.get('cause', '')})"
    else:
        stale = vslib.check_stale(man, tree) if man.get("rtl_fingerprint") else {"stale": True, "reason": "no pin tree", "changed": [], "added": [], "removed": []}
        receipt["staleness"] = {k: v for k, v in stale.items() if k != "entries"}
        _write(lp / "staleness.json", stale)
        calibration = "validated configuration"
        if stale["stale"]:
            if not a.uncalibrated:
                receipt.update(status="REFUSED_STALE_MODEL",
                               note=f"the export's RTL is not the tree the LPN model is validated against ({LPN_RTL_PIN[:8]}): "
                                    f"{len(stale['changed'])} changed, {len(stale['added'])} added, {len(stale['removed'])} removed "
                                    f"(lpn/staleness.json). Pass --uncalibrated to run anyway, labelled so.")
                _write(lp / "receipt.json", receipt)
                return receipt
            calibration = (f"outside validated configuration: RTL moved ({len(stale['changed'])} changed, {len(stale['added'])} added, "
                           f"{len(stale['removed'])} removed fingerprinted files)")
        if a.dram_latency is not None and a.dram_latency != VALIDATED["dram_latency"]:
            calibration += f"; Timing.dram_latency {a.dram_latency} (validated {VALIDATED['dram_latency']})"
    # the timing: the pin's, or the receipt's memory for the wide model
    dram_latency = a.dram_latency if a.dram_latency is not None else (a.lat if wide else VALIDATED["dram_latency"])
    timing_args = {}
    if wide:
        nch = int(a.nch) if a.channels or (a.portb or 1) != 1 or a.rowlanes else 1
        timing_args = {"portb": int(a.portb or 1), "nch": nch,
                       "fifo_entries": a.fifo_entries or target_fifo_entries(tree, a.target),
                       "channel_depth": a.channel_depth or (64 if nch > 2 else 128),
                       "adapter_latency": a.adapter_latency if a.adapter_latency is not None else (14 if nch > 1 else 4),
                       "epilogue_lanes": a.epilogue_lanes, "chash": not a.no_chash}
        receipt["timing"] = dict(timing_args, dram_latency=dram_latency, batch_rows=a.batch_rows)
    env = {**os.environ, "OTPU_TARGET": a.target, "OTPU_NCH": str(a.nch), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    recorded_env = {k: env[k] for k in sorted(env) if k.startswith("OTPU_") or k == "OMP_NUM_THREADS"}
    # 3. inputs
    inp = lp / "input"
    cmd_in = [a.python, str(HERE / "lpn" / "inputs.py"), "--tree", str(tree), "--out", str(inp), "--revision", LPN_RTL_PIN,
              "--model", a.model, "--layers", str(a.layers), "--pos", str(a.pos), "--wformat", a.wformat, "--head-format", a.head_format]
    if a.portb:
        cmd_in += ["--portb", str(a.portb)]
    if a.mcols:
        cmd_in += ["--mcols", str(a.mcols)]
    if a.rowlanes:
        cmd_in.append("--rowlanes")
    steps: dict = {}
    if (inp / "workload.json").is_file() and (inp / "input.json").is_file() and a.reuse_inputs:
        steps["inputs"] = {"reused": True}
    else:
        t0 = time.monotonic()
        with open(lp / "inputs.log", "w") as log:
            rc = subprocess.call(cmd_in, cwd=lp, env=env, stdout=log, stderr=subprocess.STDOUT)
        steps["inputs"] = {"command": cmd_in, "exit": rc, "wall_s": time.monotonic() - t0}
        if rc:
            receipt.update(status="FAILED_INPUTS", steps=steps, log=str(lp / "inputs.log"))
            _write(lp / "receipt.json", receipt)
            return receipt
    work = json.loads((inp / "workload.json").read_text())
    if work.get("rejected_features_static"):
        receipt.update(status="REFUSED_REJECTED_ISA", steps=steps, rejected=work["rejected_features_static"],
                       note="the program uses ISA features upstream's timing backend rejects; it stays on Verilator")
        _write(lp / "receipt.json", receipt)
        return receipt
    # 4. the model
    cmd_emit = [a.python, str(HERE / "lpn" / "emit.py"), "--opentpu-root", str(tree if wide else pin), "--input", str(inp), "--out", str(lp),
                "--lpn-compiler", man["lpn_compiler"], "--batch-rows", str(a.batch_rows), "--dram-latency", str(dram_latency)]
    if wide:
        cmd_emit += ["--frontend", "export", "--wide", "--portb", str(timing_args["portb"]), "--nch", str(timing_args["nch"]),
                     "--fifo-entries", str(timing_args["fifo_entries"]), "--channel-depth", str(timing_args["channel_depth"]),
                     "--adapter-latency", str(timing_args["adapter_latency"]), "--epilogue-lanes", str(timing_args["epilogue_lanes"])]
        if timing_args.get("chash") is False:
            cmd_emit.append("--no-chash")
    elif not a.expanded:
        cmd_emit.append("--compact")
    emit_env = {**env, "PYTHONPATH": str(lpn), "PATH": f"{a.llvm_bin}:{env.get('PATH', '')}"}
    t0 = time.monotonic()
    with open(lp / "emit.log", "w") as log:
        rc = subprocess.call(cmd_emit, cwd=lp, env=emit_env, stdout=log, stderr=subprocess.STDOUT)
    steps["emit"] = {"command": cmd_emit, "exit": rc, "wall_s": time.monotonic() - t0, "PYTHONPATH": str(lpn), "PATH_prefix": a.llvm_bin}
    if rc:
        receipt.update(status="FAILED_MODEL", steps=steps, log=str(lp / "emit.log"))
        _write(lp / "receipt.json", receipt)
        return receipt
    emit = json.loads((lp / "emit.json").read_text())
    # 5. the native run
    cmd_nat = [str(lp / "opentpu"), "--stats", "--wall-start=begin", "--wall-end=complete", "--max-time=1000000000000"]
    t0 = time.monotonic()
    p = subprocess.run(cmd_nat, cwd=lp, capture_output=True, text=True)
    (lp / "native.log").write_text(p.stdout + p.stderr)
    native = parse_native(p.stdout)
    native.update(exit=p.returncode, process_wall_s=time.monotonic() - t0, command=cmd_nat)
    steps["native"] = native
    if p.returncode or not native["complete"]:
        receipt.update(status="FAILED_NATIVE", steps=steps, log=str(lp / "native.log"))
        _write(lp / "receipt.json", receipt)
        return receipt
    # 6. the record
    args = {"tree": str(tree), "layers": a.layers, "pos": a.pos, "portb": a.portb or 1, "env": recorded_env,
            "command": cmd_in[1:] + ["|", "emit.py"] + cmd_emit[2:] + ["|"] + cmd_nat,
            "shell": " && ".join(" ".join(shlex.quote(c) for c in cmd) for cmd in (cmd_in, cmd_emit, cmd_nat))}
    tr = record(out, args, work, emit, native, man, stale, calibration, a.mhz, D=int(work["config_pin_fields"]["D"]), graded=graded)
    tr["configuration"] = key
    tr["steps"] = steps
    _write(lp / "trace.json", tr)
    if not (out / "trace.json").is_file() or a.as_run_trace:
        _write(out / "trace.json", tr)            # a run directory of its own: `report --baseline RUN` reads it
    receipt["numerical_check_match"] = tr["lpn"]["numerical_check"]["match"]
    comps = []
    for rd in a.compare or []:
        c = compare(tr, Path(rd))
        comps.append(c)
    if comps:
        _write(lp / "compare.json", comps)
        receipt["compare"] = [{k: c.get(k) for k in ("receipt", "rtl_cycles", "lpn_cycles", "rel_error", "within_gate", "verdict")} for c in comps]
    # 6b. grading: the registry entry (the wide model only)
    if wide and a.grade and comps and comps[0].get("rel_error") is not None and not comps[0]["configuration_deltas"]:
        c = comps[0]
        fp_now = tree_fp or vslib.fingerprint_tree(tree, _trace.RTL_GLOBS)
        sha = wide_info["export"]
        reg.setdefault("exports", {})[sha] = {"rtl_fingerprint": fp_now, "tree": str(tree), "frozen_utc": _now()}
        entry = {"key": key, "export": sha, "model": "WideCompactOpenTPU", "wide_model_sha256": man.get("wide_model_sha256"),
                 "timing": emit.get("timing"), "lpn_cycles": tr["derived"]["cycles"],
                 "receipt": {"dir": str(Path(c["receipt"]).resolve()), "cycles": c["rtl_cycles"], "label": c.get("receipt_label")},
                 "rel_error": c["rel_error"], "within_gate": c["within_gate"], "gate": LPN_GATE, "graded_utc": _now(),
                 "cause": a.cause or ("" if c["within_gate"] else "see tools/hwloop/lpn/README.md"),
                 "memory": {k: tr["lpn"]["memory"].get(k) for k in ("wide_reads", "two_part", "two_part_share", "roofline", "channel_rate")} if tr["lpn"].get("memory") else None,
                 "run": str(lp)}
        reg["configurations"] = [e for e in reg.get("configurations", []) if not all(e.get("key", {}).get(k) == key[k] for k in KEY_FIELDS)] + [entry]
        _write(REGISTRY, reg)
        receipt["graded"] = entry
        calibration = (f"graded: {c['rel_error']:+.3%} vs {c['rtl_cycles']:,} (Verilator, {c['receipt']})" if c["within_gate"] else
                       f"uncalibrated: {c['rel_error']:+.3%} vs {c['rtl_cycles']:,} is over the {LPN_GATE:.0%} gate ({entry['cause']})")
        tr["calibration"] = calibration
        tr["lpn"]["graded"] = entry
        _write(lp / "trace.json", tr)
        if a.as_run_trace or json.loads((out / "trace.json").read_text()).get("kind") == "lpn_decode_token":
            _write(out / "trace.json", tr)
    # 7. optionally the pinned RTL on the same bundle
    if a.rtl_pin and not wide:
        cmd_rtl = [a.python, str(HERE / "lpn" / "rtl_pin.py"), "--tree", str(pin), "--input", str(inp), "--out", str(lp / "rtl-pin"),
                   "--dram-lat", str(dram_latency), "--jobs", str(a.jobs)]
        t0 = time.monotonic()
        with open(lp / "rtl-pin.log", "w") as log:
            rc = subprocess.call(cmd_rtl, cwd=lp, env={**env, "OTPU_TARGET": "ypcb-00338"}, stdout=log, stderr=subprocess.STDOUT)
        steps["rtl_pin"] = {"command": cmd_rtl, "exit": rc, "wall_s": time.monotonic() - t0}
        if rc == 0:
            rp = json.loads((lp / "rtl-pin" / "rtl_pin.json").read_text())
            err = tr["derived"]["cycles"] / rp["cycles"] - 1
            receipt["rtl_pin"] = {"cycles": rp["cycles"], "lpn_cycles": tr["derived"]["cycles"], "rel_error": err,
                                  "within_gate": abs(err) <= LPN_GATE, "rtl_matches_pin_isa": rp["rtl_matches_pin_isa_simulator"],
                                  "pin_isa_matches_export_isa": rp["pin_isa_matches_export_isa"], "verilator": rp["verilator"],
                                  "label": "measured (Verilator, the model's validated configuration: behavioral DRAM, lat %d, tb_top default uarch)" % dram_latency}
            tr["rtl_pin"] = receipt["rtl_pin"]
            _write(lp / "trace.json", tr)
            if a.as_run_trace or not (out / "trace.json").is_file():
                _write(out / "trace.json", tr)
    receipt.update(status="OK", steps=steps, cycles=tr["derived"]["cycles"], calibration=calibration, ended_utc=_now())
    _write(lp / "receipt.json", receipt)
    return receipt


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tree", required=True, help="the frozen export (git archive) whose program and image are modelled")
    p.add_argument("--out", required=True, help="the run directory (lpn/ inside it; trace.json if none exists)")
    p.add_argument("--python", default=DEFAULTS["python"])
    p.add_argument("--lpn", default=DEFAULTS["lpn"], help="the LPNDialect clone")
    p.add_argument("--lpn-build", default=DEFAULTS["lpn_build"], help="its build directory (tools/lpn-compile/lpn-compile)")
    p.add_argument("--pin", default=DEFAULTS["pin"], help="the pinned openTPU tree (f237dab2 + observer patch)")
    p.add_argument("--llvm-bin", default=DEFAULTS["llvm_bin"], help="directory with opt-20, llc-20, clang++-20")
    for k, d in (("model", "gemma3"), ("layers", "2"), ("pos", 9), ("wformat", "int8"), ("head_format", "int8"),
                 ("target", "f2-hbm"), ("nch", 8), ("lat", 60), ("mhz", 200.0), ("batch_rows", 8), ("jobs", 8), ("epilogue_lanes", 2)):
        p.add_argument("--" + k.replace("_", "-"), default=d, type=type(d))
    p.add_argument("--dram-latency", type=int, default=None, help="Timing.dram_latency (default: the pin's 8; the wide model: --lat)")
    p.add_argument("--portb", type=int)
    p.add_argument("--mcols", type=int)
    p.add_argument("--rowlanes", action="store_true", help="the row-lane image and MM RL (the wide model)")
    p.add_argument("--channels", action="store_true", help="the wide model at PORTB 1: NCH channels at --lat (the native memory path)")
    p.add_argument("--fifo-entries", type=int, help="the MXU chunk FIFO's entries (default: the target's FIFO_DEPTH)")
    p.add_argument("--channel-depth", type=int, help="B read beats in flight a channel (default: 64 at NCH > 2, else 128)")
    p.add_argument("--adapter-latency", type=int, help="cycles over --lat on a read's round trip (default: 14 on the native path, 4 behavioral)")
    p.add_argument("--no-chash", action="store_true")
    p.add_argument("--grade", action="store_true", help="with --compare: write the graded point into tools/hwloop/lpn/validated.json")
    p.add_argument("--cause", help="with --grade: the suspected cause when the point is over the gate")
    p.add_argument("--expanded", action="store_true", help="upstream's expanded net instead of --compact")
    p.add_argument("--compare", action="append", help="a Verilator run directory (trace.json) to grade against; repeatable")
    p.add_argument("--rtl-pin", action="store_true", help="also run the pinned RTL on the same bundle (Verilator, ~40 min)")
    p.add_argument("--uncalibrated", action="store_true", help="run on an export whose RTL moved from the pin / the graded export, labelled so")
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--reuse-inputs", action="store_true", help="keep an existing lpn/input bundle")
    p.add_argument("--as-run-trace", action="store_true", help="overwrite RUN/trace.json with the LPN record")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    a = ap.parse_args(argv)
    r = run(a)
    print(json.dumps(r, indent=2, default=str))
    sys.exit(0 if r["status"] in ("OK", "MANIFEST_ONLY") else 2 if r["status"].startswith("REFUSED") else 3)


if __name__ == "__main__":
    main()
