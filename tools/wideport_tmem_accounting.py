"""Verified RTL replay accounting for the wide-port LPN's validation reports.

These are overlapping event counts, NOT additive command latencies. The present
LPN has no bank arbiter/replay pipeline; retaining that limitation is preferable
to fitting a correction from the elapsed time being used for validation.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PORTS = ("MXU", "DMA", "QUANT", "VPU_A", "VPU_B", "QUANT3_GUEST", "QUANT2", "COLL")
UNITS = ("DMA", "MXU", "QUANT", "VPU", "COLL")
FIELDS = {"lose": 4, "n": 7, "d": 9, "dw": 10, "rd": 8, "rpl": 8}


def parse_tmemstat(line):
    """Decode the single-slice final RTL counters, rejecting partial/malformed data."""
    match = re.fullmatch(r"TMEMSTAT sid=(\d+) cyc=(\d+)\s+(.*)", line.strip())
    if not match:
        raise ValueError("invalid TMEMSTAT header")
    result = {"sid": int(match[1]), "cycles": int(match[2])}
    pairs = re.findall(r"(\w+)=\s*([0-9]+(?:,[0-9]+)*)", match[3])
    if len(pairs) != len(FIELDS) or {k for k, _ in pairs} != FIELDS.keys():
        raise ValueError("incomplete or duplicate TMEMSTAT counters")
    if re.sub(r"\w+=\s*[0-9]+(?:,[0-9]+)*", "", match[3]).strip():
        raise ValueError("invalid TMEMSTAT counter data")
    for key, value in pairs:
        result[key] = [int(x) for x in value.split(",")]
        if len(result[key]) != FIELDS[key]:
            raise ValueError(f"TMEMSTAT {key}: expected {FIELDS[key]} counters")
    if result["sid"] != 0 or result["cycles"] <= 0:
        raise ValueError("only a completed single-slice reference is supported")
    for key in ("n", "d", "dw"):
        if sum(result[key]) != result["cycles"]:
            raise ValueError(f"TMEMSTAT {key}: histogram does not cover all cycles")
    return result


def replay_scope(uarch):
    # Grant arbitration is still absent when replay itself is disabled.
    missing = uarch.get("TMEM_IMPL") == 3
    return {"tmem_pmp": uarch.get("TMEM_PMP"), "replay_timing_modeled": not missing,
            "cycle_correction": 0, "calibration_eligible": not missing,
            "limitation": ("Bank arbitration, replay holds and downstream backpressure are not "
                           "modeled. RTL counters overlap; no additive correction is applied."
                           if missing else None)}


def throughput_result(cycles, mhz, phase_error, scope):
    error_gate = phase_error is not None and phase_error <= 0.10
    calibrated = error_gate and scope["calibration_eligible"]
    return {"phase_error_gate_passed": error_gate, "calibrated": calibrated,
            "tok_s": mhz * 1e6 / cycles if calibrated else None}


def load_reference(directory, pmp, expected_parameters=None):
    """Load a preserved comparison run; status parameters override stale Lens uarch.

    No reference value is injected into the LPN. The runner log hash, correctness
    receipt, final counters and Lens elapsed time must agree before reporting it.
    """
    directory = Path(directory)
    status = json.loads((directory / "status.json").read_text())
    runs = [r for r in status["runs"] if r["pmp"] == pmp]
    if status["state"] != "passed" or len(runs) != 1:
        raise ValueError("reference must contain one passed run for the selected PMP")
    run = runs[0]
    if run["returncode"] != 0 or run["bit_exact_isa"] is not True:
        raise ValueError("reference did not pass ISA correctness")
    paths = {key: directory / f"pmp{pmp}{suffix}" for key, suffix in
             (("log", ".log"), ("lens", "-lens.json"), ("counters", "-tmemstat.log"), ("receipt", ".jsonl"))}
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    if hashes["log"] != run["log_sha256"]:
        raise ValueError("reference log hash mismatch")
    params = run["parameters"]
    if params["S"] != 1 or params["TMEM_PMP"] != pmp:
        raise ValueError("reference slice count or PMP mismatch")
    for key, value in (expected_parameters or {}).items():
        if params.get(key) != value:
            raise ValueError(f"reference {key}: expected {value}, found {params.get(key)}")
    lines = [line.strip() for line in paths["counters"].read_text().splitlines() if line.startswith("TMEMSTAT ")]
    if lines != run["tmem_counters"] or len(lines) != 1:
        raise ValueError("reference final counters disagree with status")
    counters = parse_tmemstat(lines[0])
    lens = json.loads(paths["lens"].read_text())
    receipt = json.loads([line for line in paths["receipt"].read_text().splitlines() if line.strip()][-1])
    if any(value != run["cycles"] for value in (counters["cycles"], lens["cycles"], receipt["cycles"])):
        raise ValueError("reference cycle counts disagree")
    phases = receipt["phases"]
    # Profile.phases charges unit completions; the final scalar HALT takes two
    # further cycles in these fixtures. Keep that tail visible, not in a phase.
    unattributed = run["cycles"] - sum(p["cycles"] for p in phases.values())
    if not 0 <= unattributed <= 2:
        raise ValueError("reference phases disagree with total beyond the final HALT tail")
    freezes = {name: {"instructions": 0, "frz": 0} for name in UNITS}
    for instruction in lens["instrs"]:
        sid, unit, start, end, stats = instruction[0], instruction[3], instruction[9], instruction[10], instruction[12]
        if sid != 0 or not 0 <= unit < len(UNITS) or not 0 <= start <= end <= run["cycles"]:
            raise ValueError("invalid Lens instruction interval/unit")
        counts = freezes[UNITS[unit]]
        counts["instructions"] += 1
        for key, value in stats.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError("invalid Lens instruction counter")
            counts[key] = counts.get(key, 0) + value
    report = {"directory": str(directory), "revision": status["revision"], "sha256": hashes,
              "integrity": {"log": "matches original runner SHA256",
                            "counters": "matches original runner counter text",
                            "lens_and_phases": "consistency checked and hashed now; original runner did not pin their digests"},
              "parameters_source": "status.runs[].parameters (Lens uarch is not authoritative)",
              "parameters": params, "rtl_cycles": run["cycles"], "phases": phases,
              "unattributed_tail_cycles": unattributed, "counters": counters,
              "per_port": {name: {"read_attempts": counters["rd"][i], "replay_loss_cycles": counters["rpl"][i]}
                           for i, name in enumerate(PORTS)},
              "replay_loss_port_cycles": sum(counters["rpl"]), "instruction_counters": freezes,
              "interpretation": "Overlapping diagnostics, not elapsed or exposed critical-path cycles.",
              "cycle_correction": 0}
    return report, lens, run


def verify_program(lens, programs, commands, unit_of):
    """Require identical static code and dynamic unit dispatch order to the RTL fixture."""
    if len(programs) != 1 or len(lens["programs"]) != 1:
        raise ValueError("only single-slice programs are supported")
    reference = lens["programs"][0]
    if len(reference) != len(programs[0]):
        raise ValueError("reference program length mismatch")
    for pc, (saved, instruction) in enumerate(zip(reference, programs[0])):
        registers = [instruction.ra, instruction.rb, instruction.rc, instruction.rd]
        if (saved[0] != pc or saved[5] != [int(x) for x in instruction.w] or
                saved[6] != registers or saved[7] != instruction.flags or saved[8] != instruction.op):
            raise ValueError(f"reference program mismatch at PC {pc}")
    sequence = [c.pc for c in commands if unit_of.get(c.instruction.op) is not None]
    if sequence != [i[2] for i in lens["instrs"]]:
        raise ValueError("reference dynamic unit dispatch order mismatch")
