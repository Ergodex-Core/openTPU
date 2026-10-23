"""Standalone C3 Verilator harness and measured-event Lens adapter.

The DUT is synthesizable RTL; its input service is deliberately synthetic, with
one header service cycle and 128-byte ready/valid blocks. It is not an HBM model,
an AXI adapter, a board command, or a measured FPGA clock/performance prediction.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from .. import rtlsim
from . import vocab_engine as C

SOURCES = ["rtl/vpu/otpu_fp.sv", "rtl/vpu/otpu_fpipe.sv",
           "rtl/c3/otpu_c3_epilogue.sv", "rtl/c3/otpu_c3_engine.sv",
           "rtl/c3/otpu_c3_reduce.sv", "rtl/c3/otpu_c3_cluster.sv",
           "sim/verilator/tb_c3.sv"]


def build_aux(top, sources, params=None):
    # Three proposal worktrees may compile concurrently. Include the bounded job
    # count in the existing build cache rather than changing the shared builder.
    old = rtlsim.VFLAGS
    try:
        rtlsim.VFLAGS = old.copy()
        rtlsim.VFLAGS[rtlsim.VFLAGS.index("-j") + 1] = "4"
        return rtlsim.build(top, [rtlsim.ROOT / p for p in sources], params)
    finally:
        rtlsim.VFLAGS = old


def build(engines=4):
    if engines not in (1, 4):
        raise ValueError("validation supports one or four engines")
    return build_aux("tb_c3", SOURCES, {"ENGINES": engines})


def write_vectors(path, sidecar, activation):
    sidecar.verify_payload()
    layout = sidecar.layout
    if layout.vocab > 8192 or len(activation.values) != layout.hidden:
        raise ValueError("harness supports <=8192 rows and matching activation shape")
    lines = [f"{layout.vocab} {layout.kb} {activation.scale_bits:08x}"]
    for k in range(layout.kb):
        values = int.from_bytes(activation.values[128*k:128*(k+1)], "little")
        da = sum(activation.offsets[4*k+g] << (5*g) for g in range(4))
        lines.append(f"{values:0256x} {da:05x}")
    for row in range(layout.vocab):
        data = sidecar.row(row).tobytes()
        lines.append(f"{int.from_bytes(data[:4], 'little'):08x}")
        for k in range(layout.kb):
            values = int.from_bytes(data[32+128*k:32+128*(k+1)], "little")
            lines.append(f"{values:0256x} {data[4+k]:02x}")
    Path(path).write_text("\n".join(lines) + "\n")


@dataclass(frozen=True)
class Simulation:
    layout: C.Layout
    engines: int
    events: tuple[dict, ...]
    logits_bits: tuple[np.ndarray, ...]
    winners: tuple[C.Winner, ...]
    settings: dict
    stdout: str

    @property
    def completed_epochs(self):
        return tuple(e["epoch"] for e in self.events if e["event"] == "R")


def run(sidecar, activation, *, engines=4, seed=1, stalls=0, output_stalls=0,
        latency=0, reset_after=-1, repeats=1, slow_engine=-1, result_hold=0, output_hold=0):
    for name, value, low, high in (("seed", seed, 1, 2**32-1), ("stalls", stalls, 0, 95),
                                 ("output_stalls", output_stalls, 0, 95),
                                 ("latency", latency, 0, 10000), ("reset_after", reset_after, -1, 100000),
                                 ("repeats", repeats, 1, 10), ("slow_engine", slow_engine, -1, engines-1),
                                 ("result_hold", result_hold, 0, 10000), ("output_hold", output_hold, 0, 10000)):
        C._integer(value, low, high, name)
    exe = build(engines)
    settings = dict(seed=seed, stalls=stalls, output_stalls=output_stalls, latency=latency,
                    reset_after=reset_after, repeats=repeats, slow_engine=slow_engine,
                    result_hold=result_hold, output_hold=output_hold)
    with tempfile.TemporaryDirectory(prefix="otpu-c3-") as tmp:
        path = Path(tmp)
        write_vectors(path / "input.txt", sidecar, activation)
        result = rtlsim.run_sim([exe, f"+input={path/'input.txt'}", f"+events={path/'events.txt'}",
                                *[f"+{k}={v}" for k, v in settings.items()]], timeout=180)
        if result.returncode or "C3_RTL_PASS" not in result.stdout:
            raise RuntimeError(f"C3 simulation failed:\n{result.stdout}\n{result.stderr}")
        events = []
        for line in (path / "events.txt").read_text().splitlines():
            cycle, epoch, event, engine, token, block, bits = line.split()
            events.append(dict(cycle=int(cycle), epoch=int(epoch), event=event,
                               engine=int(engine), token=int(token), block=int(block), bits=int(bits, 16)))
    outputs, winners = [], []
    for final in (e for e in events if e["event"] == "R"):
        logs = [e for e in events if e["epoch"] == final["epoch"] and e["event"] == "L"]
        if sorted(e["token"] for e in logs) != list(range(sidecar.layout.vocab)):
            raise AssertionError("RTL did not retire every token exactly once")
        values = np.empty(sidecar.layout.vocab, np.uint32)
        for e in logs:
            values[e["token"]] = e["bits"]
        values.flags.writeable = False
        outputs.append(values)
        winners.append(C.Winner(final["bits"], final["token"]))
    if len(outputs) != repeats:
        raise AssertionError("missing or duplicate command completion")
    return Simulation(sidecar.layout, engines, tuple(events), tuple(outputs), tuple(winners), settings, result.stdout)


def lens_profile(sim, *, epoch=None, mhz=200.0):
    """Build a profile from accepted DUT events, preserving real stage overlap."""
    from ..profile import UNITS
    epoch = sim.completed_epochs[-1] if epoch is None else epoch
    if epoch not in sim.completed_epochs:
        raise ValueError("only completed commands can be profiled")
    events = [e for e in sim.events if e["epoch"] == epoch]
    start = min(e["cycle"] for e in events if e["event"] == "A")
    final = next(e for e in events if e["event"] == "R")
    core_done = min(e["cycle"] for e in events if e["event"] in ("F", "R")) + 1 - start
    cycles = final["cycle"] + 1 - start
    instrs, programs, slices = [], [[] for _ in range(sim.engines)], []
    busy = [Counter() for _ in range(sim.engines)]

    def interval(engine, name, unit, lo, hi, detail, pb=0):
        pc = len(programs[engine])
        lo, hi = lo-start, hi-start
        programs[engine].append([pc, name, detail, "C3 RTL event", -1, [0]*8, [0]*4, 0, 0])
        instrs.append([engine, pc, pc, unit, name, detail, lo, lo, -1, lo, hi,
                       -1, {}, hi-lo, pb, 0])
        # Union intervals below; pipelined rows can occupy the same functional unit.
        busy[engine][UNITS[unit]] += 0

    for engine in range(sim.engines):
        act = [e["cycle"] for e in events if e["event"] == "A"]
        interval(engine, "activation snapshot", 0, min(act), max(act)+1,
                 "128 activation bytes + four offsets per accepted snapshot write; broadcast")
        rows = sorted({e["token"] for e in events if e["event"] == "L" and e["engine"] == engine})
        by_row = {}
        for e in events:
            if e["engine"] == engine:
                by_row.setdefault(e["token"], {}).setdefault(e["event"], []).append(e["cycle"])
        for row in rows:
            r = by_row[row]
            interval(engine, "sidecar service", 0, r["H"][0], r["W"][-1]+1,
                     f"token {row}; synthetic header/beat service, including ready/valid waits", sim.layout.pitch/128)
            interval(engine, "integer pipeline", 1, r["W"][0]+1, r["D"][-1]+1,
                     f"token {row}; actual 128-product, reduction and signed48 accumulation stages")
            interval(engine, "fp32 epilogue", 3, r["E"][0], r["L"][0]+1,
                     f"token {row}; i2f48, Sw, Sa*2^-16, diagnostic ready/valid waits")
            interval(engine, "local ARGMAX", 4, r["L"][0], r["L"][0]+1,
                     f"token {row}; exact fkey, lowest global ID ties")
    shard_cycles = [e["cycle"] for e in events if e["event"] == "S"]
    interval(0, "global ARGMAX", 4, min(shard_cycles), core_done+start,
             "actual completion collection and exact reducer; wait for independent shards included")
    for engine in range(sim.engines):
        for unit in range(len(UNITS)):
            spans = sorted((r[9], r[10]) for r in instrs if r[0] == engine and r[3] == unit)
            end, total = -1, 0
            for lo, hi in spans:
                total += max(0, hi-max(lo, end)); end = max(end, hi)
            busy[engine][UNITS[unit]] = total
        slices.append({"ports": {}, "busy": dict(busy[engine]), "lose": {}, "buckets": {}})
    # Work and overlap counts are directly observed, not throughput arithmetic.
    counts = Counter(e["event"] for e in events)
    overlap = 0
    ep_overlap = 0
    for engine in range(sim.engines):
        writes = {e["cycle"] for e in events if e["engine"] == engine and e["event"] == "W"}
        dots = {e["cycle"] for e in events if e["engine"] == engine and e["event"] == "D"}
        overlap += len(writes & dots)
        spans = [(r[9]+start, r[10]+start) for r in instrs if r[0] == engine and r[3] == 3]
        ep_overlap += sum(any(lo <= t < hi for lo, hi in spans) for t in writes)
    head_rows = (sim.layout.vocab + sim.engines-1)//sim.engines
    bound = head_rows * (sim.layout.kb+1)  # each model header occupies a whole service slot
    per_engine_rows = [max(0, min(head_rows, sim.layout.vocab-e*head_rows)) for e in range(sim.engines)]
    return {"kind": "rtl", "name": f"C3 {sim.engines}-engine RTL — synthetic row service",
            "cycles": cycles, "clock_mhz": mhz,
            "config": {"S": sim.engines, "D": 128, "MCOLS": 1, "LANES": 1, "CL": 1,
                       "C": 1, "ACT_BLOCKS": sim.layout.kb, "TMEM_WORDS": 0,
                       "MEM": "synthetic ready/valid row service", "model": "C3 standalone RTL"},
            "roofline": {"bound": bound, "efficiency": bound/cycles,
                         "per_slice": [{"portb": n*sim.layout.pitch/128, "porta": 0,
                                        "bound": n*(sim.layout.kb+1)} for n in per_engine_rows],
                         "note": "Ideal slots for this synthetic header/beat service; not HBM bandwidth"},
            "macs": sim.layout.vocab*sim.layout.hidden, "bytes": sim.layout.vocab*sim.layout.pitch,
            "peak_macs": 128*sim.engines, "peak_bytes": 128*sim.engines,
            "units": UNITS, "slices": slices, "instrs": instrs, "programs": programs,
            "sources": [], "mxu_gaps": {},
            "notes": [{"level": "warn", "text": "Cycle counts are actual standalone Verilator RTL execution. "
                       "Memory is a synthetic header/128-byte row service, not an HBM/AXI/controller model. "
                       "200 MHz is only a display assumption; no achieved F2 timing, full-token speedup or board integration."},
                      {"level": "info", "text": "Input starts after QACT; includes activation snapshot loading, "
                       "real concurrent integer/fp32/reduction stages and requested diagnostic/result backpressure. "
                       "S denotes head engines, not complete TPU slices. Phase intervals include pipeline bubbles and overlap."}],
            "c3": {"settings": sim.settings, "completed_epoch": epoch,
                   "core_completion_cycles": core_done, "result_consumer_hold_cycles": cycles-core_done,
                   "event_counts": dict(counts), "read_and_dot_overlap_engine_cycles": overlap,
                   "read_and_epilogue_overlap_engine_cycles": ep_overlap,
                   "winner": {"bits": final["bits"], "token_id": final["token"]},
                   "stage_intervals_from_actual_events": True, "HBM_modeled": False}}
