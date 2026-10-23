"""A deliberately conservative C1 schedule experiment exported through openTPU Lens.

This is an analytic policy model, not RTL, an HBM controller simulation, or a model of
a complete token. One row is outstanding per engine; its result must retire before
the engine requests another row. There is no inter-row prefetch or double buffering.
"""
from dataclasses import asdict, dataclass

from . import c1_rows as C
from .profile import UNITS


@dataclass(frozen=True)
class Timing:
    mhz: int = 200
    read_latency: int = 80
    activation_bytes_per_cycle: int = 8
    epilogue_latency: int = 9
    retire_cycles: int = 1

    def __post_init__(self):
        if any(type(x) is not int for x in asdict(self).values()):
            raise ValueError("timing parameters must be integer cycle counts/rates")
        if min(self.mhz, self.activation_bytes_per_cycle, self.retire_cycles) <= 0:
            raise ValueError("frequency, activation bandwidth and retirement must be positive")
        if min(self.read_latency, self.epilogue_latency) < 0:
            raise ValueError("latencies must be nonnegative")


def profile(descriptor, command, cfg, *, timing=Timing(), extra_pc_latency=None, name="C1"):
    """Build Lens v1 intervals from individual engine dispatch/completion/retirement events.

    A row pays the stated first-response latency plus its whole header and selected
    weight beats, followed by the epilogue. Weight transfer and 32-product computation
    overlap exactly; AXI burst splitting, DRAM bank timing and arbitration are absent.
    ACC retirement is charged separately but TMEM contention is not modeled.
    """
    d, c, t = descriptor, command, timing
    C.validate_command(d, c, cfg)
    extra = dict(extra_pc_latency or {})
    if any(pc not in d.pc_ids or type(v) is not int or v < 0 for pc, v in extra.items()):
        raise ValueError("extra latency must name a selected PC with a nonnegative cycle count")
    # One selected ACT vector and its Sa/4-offset record for each block, multicast once.
    act_bytes = c.kb * (C.D + 8)
    copy_cycles = -(-act_bytes // t.activation_bytes_per_cycle)
    available = [copy_cycles] * d.engines
    retire = copy_cycles
    per_engine = [0] * d.engines
    instrs = [[0, 0, 0, 0, "C1_ACT_COPY", f"{act_bytes} B multicast snapshot", 0, 0,
               -1, 0, copy_cycles, -1, {}, copy_cycles, 0, 0]]
    events = []
    row_cycles = d.header // 32 + 4 * c.kb
    for idx, n in enumerate(range(c.n0, c.n0 + c.n), 1):
        e = n % d.engines
        pc = d.pc_ids[e]
        issue = available[e]
        first = issue + t.read_latency + extra.get(pc, 0)
        complete = first + row_cycles + t.epilogue_latency
        commit = max(retire, complete)
        # The reference only supports ACC/plain: ACC adds one assumed read slot here.
        end = commit + t.retire_cycles * (2 if c.flags else 1)
        available[e], retire = end, end
        per_engine[e] += 1
        events.append({"row": n, "pc": pc, "dispatch": issue, "first_response": first,
                       "result_ready": complete, "retire_start": commit, "retired": end,
                       "credit_wait": commit - complete})
        instrs.append([0, idx, 1, 1, "C1_ROW", f"PC {pc}; row {n}; KB {c.kb}",
                       issue, issue, -1, issue, end, -1, {}, 4 * c.kb, c.kb, 0])
    bound = max(per_engine) * row_cycles
    return {
        "kind": "analytic-c1", "name": name + " [one-row-credit analytic policy]",
        "cycles": retire, "clock_mhz": t.mhz,
        "config": {"S": 1, "D": C.D, "MCOLS": cfg.MCOLS, "LANES": cfg.LANES,
                   "TMEM_WORDS": cfg.TMEM_WORDS, "MEM": "hbm", "PCS": d.engines,
                   "C1_ENGINES": d.engines},
        "units": UNITS, "instrs": instrs,
        "programs": [[[0, "C1_ACT_COPY", "multicast activation snapshot", "model event", -1],
                      [1, "C1_ROW", "independent engine row", "model event, not ISA", -1]]],
        "sources": [], "mxu_gaps": {},
        "slices": [{"ports": {}, "busy": {"DMA": copy_cycles, "MXU": retire - copy_cycles,
                                             "VPU": 0, "QUANT": 0, "COLL": 0},
                    "lose": {}, "buckets": {}}],
        "roofline": {"bound": bound, "efficiency": bound / retire,
                     "per_slice": [{"bound": bound, "portb": c.n * c.kb, "porta": 0}]},
        "macs": c.n * c.kb * C.D, "bytes": c.n * (d.header + c.kb * C.D),
        "peak_macs": d.engines * 32, "peak_bytes": d.engines * 32,
        "notes": [{"level": "warn", "text":
                   "Analytic policy experiment, NOT RTL or F2 hardware timing and NOT a token "
                   "throughput forecast. Timeline entries are modeled row work items, not ISA "
                   "instructions. One outstanding row/result per engine; ordered retirement; "
                   "no next-row prefetch. All timing assumptions are stored in c1_model."},
                  {"level": "info", "text":
                   "Includes activation multicast, whole row headers, selected weight bytes, "
                   "first-response and epilogue latency, ordered retirement and a bounded "
                   "one-row result credit per engine. Omits burst/bank scheduling, HBM "
                   "contention, vector/fallback work, CL DMA, host and complete-token costs. "
                   "ACC charges a read slot without modeling TMEM arbitration. The generic "
                   "Lens floorplan represents the retained backend, not a physical C1 layout."}],
        "c1_model": {"policy_version": 1, "timing": asdict(t), "extra_pc_latency": extra,
                     "descriptor": d.to_dict(), "command": asdict(c), "events": events,
                     "activation_bytes": act_bytes, "rows_per_engine": per_engine,
                     "maximum_results_outstanding_per_engine": 1},
    }
