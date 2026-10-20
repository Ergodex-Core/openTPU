"""One decode token of the wide-port slice as a Latency Petri Net (LPN; MJChku/LPNDialect's
compiled nets, the hardware loop's LPN tier on the hwloop branch: tools/hwloop/lpn), per phase
against the Verilator receipts, then the whole model's tok/s (docs/wide_port.md "LPN
estimate", docs/ternary.md 8.2). An estimate between the ISA tally (tools/tally_decode.py,
tools/b2_step_costs.py: each unit alone) and the RTL: the ISA's dependency structure and the
slice's units as resources.

P1's copy (docs/v4/p1-status.md section 6; the wide-port-lpn branch's tool, its changes marked
"p1:"): P1's wide port is Config.CPC, not the wide-port line's PORTB: port B and the MXU take CPC
chunks a cycle, an int8 MM row is ceil(KB / CPC) advances of CPC chunks (the last a half pair at
an odd KB) with no PAIR flag on the instruction, 4-bit MMs one chunk an advance; the native
adapter runs 2 CPC channels, chunk m on pair m mod CPC with no pair hash (so a wide read is never
in two parts) and the halves by the chunk's parity; port A's line is 32 bytes; no row lanes.

    OTPU_TARGET=f2 python tools/wideport_lpn.py --model gemma3 --layers 0 --wformat int8 \\
        --head-format int8 --lat 30 --mhz 200 --measured RECEIPT.jsonl --json out.json \\
        --wide HWLOOP/tools/hwloop/lpn/wide.py --dialect LPNDialect --lpn-compile LPN_COMPILE \\
        --llvm-bin LLVM20_BIN

The model is the tier's wide model (the hwloop branch's tools/hwloop/lpn/wide.py: upstream's
CompactOpenTPU with port B as a credit of PORTB chunks a cycle plus the two-part reads counted
from the addresses, the row lanes as one row of the model per group of L, the NCH channels
each a credit held for its beats at RD / round-trip beats a cycle; graded +1 .. +4% on the
96a90c2f export), imported from that worktree (`--wide`; or `git show hwloop:...` into
--work) and extended here, the hwloop branch untouched (the changes marked "wp:"):

  * port A at ALINE bytes a line (Config.ALINE, docs/row_lanes.md "Port A at 64 bytes"): the
    group's Sw line is a request of its own only where the header and the first advance's
    scale data do not fit the line (quant.rl_sw_alone), a QUAD advance's two-line crossings at
    the line's boundaries;
  * t2 weights (docs/ternary.md): a t2 chunk is four blocks, QUAD one chunk an advance with its
    scale words in one 16-byte group (never two lines), RL 2 two chunks an advance;
  * DSTEP / STREAM (the DeltaNet state updates of Qwen3.5 / Bonsai-2, on the DMA unit:
    opentpu.profile's lanes and chunks, the chunks on port B, a read's round trip first) --
    upstream's frontend refuses them, so the command trace is recorded here with the same
    rule and without the refusal;
  * the units the RTL has: the board's MXU_E epilogue lanes, WIN window, FIFO_DEPTH entries
    (targets/<target>.toml), the native path's RD read beats in flight a channel (otpu_board
    NRD) and the round trip --lat + the adapter's path;
  * the phases: one transition walks the runs of same-phase commands (perf_qwen's _phase by
    kernel source: attention / MLP / DeltaNet / the head as "other") and fires when every
    command of a run is complete, so the native run's --trace gives each run's end and the
    phase cycles add up exactly as opentpu.profile.Profile.phases charges them (the running
    max of the ends in dispatch order).

Nothing here is fitted: every constant is the RTL's (otpu_mxu's row pipeline CL + MCOLS + 5
and lanes 9, otpu_vpu's slots, otpu_quant's 60, otpu_board's NRD 64, the adapter's 14 cycles of
path derived from its structure as the tier's README says) or upstream's validated model's.
`--adapter-latency` is the one sensitivity worth a sweep (each cycle ~1.3% of a channel-bound
point). Labels: every number here is `lpn-estimate`; the receipt's column is `measured`.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("OTPU_TARGET", "f2")                     # p1: P1's target (CPC 2)
SCRATCH = Path(os.environ.get("HWLOOP_LPN_SCRATCH", str(Path.home() / "ergodex-03" / "lpn-scratch")))
HWLOOP = Path(os.environ.get("HWLOOP_TREE", str(Path.home() / "ergodex-03" / "openTPU-hwloop")))
LABEL = "lpn-estimate"
GATE = 0.10          # the brief's rule: no tok/s from a calibration whose mean |error| is above ~10%
SMALL = 0.02         # a phase under 2% of the token's cycles is shown but kept out of the mean

NATIVE_RE = {"cycles": r"Simulation finished at t=([0-9.]+)", "wall_s": r"Wall window elapsed: ([0-9.]+)",
             "rollbacks": r"rollbacks: (\d+)", "completed": r"place program_completed: (\d+) token",
             "firings": r"body executions: (\d+)"}


# ------------------------------------------------------------------------------ the program
def build_workload(a):
    """The decode step's program and (zeroed) DRAM image for the run's Config, exactly as
    tools/perf_qwen.py (Gemma) / tools/perf_bonsai2.py (Bonsai-2) hand them to the RTL: the
    timing does not depend on the weights' values (tools/tally_decode.py runs the same way)."""
    import numpy as np
    from opentpu.isasim import board_config
    mk = {}
    if a.mcols:
        mk["MCOLS"] = a.mcols
    if a.cpc:                                                   # p1: CPC, not PORTB
        mk["CPC"] = a.cpc
    pos, R = a.pos, 1
    if a.bonsai:
        # perf_bonsai2's Config: the target's (its T2 on), the run's PORTB / MCOLS / ROWLANES
        from opentpu import target as TG
        from opentpu.llm import bonsai2
        # (SEQ_WC off: the programs do not change with it and the model is the range
        # scoreboard's; perf_bonsai2's RTL run keeps the target's wait-entry sequencer)
        base = dataclasses.replace(TG.config(TG.load(os.environ["OTPU_TARGET"])), SEQ_WC=False, **mk)

        def board_config(**kw):
            return dataclasses.replace(base, **{k: v for k, v in kw.items() if k not in mk})
        spec = bonsai2.spec(a.model, ternary=True, slices=1)
        if a.layers:
            spec = dataclasses.replace(spec, kinds=spec.kinds[:a.layers])
        hf = a.head_format or a.wformat
        cap = 256
        wkw = dict(wformat=a.wformat, head_format=hf)
        name = Path(a.model).name
    else:
        from opentpu.llm import load_spec, model_dir
        path = model_dir(a.model)
        spec = load_spec(path)
        if a.layers:
            spec = dataclasses.replace(spec, **({"kinds": spec.kinds[:a.layers]} if hasattr(spec, "kinds")
                                                else {"layers": a.layers}))
        cap = 256 * ((pos + R - 1) // 256 + 1)
        wkw = dict(wformat=a.wformat, head_format=a.head_format or a.wformat, rows=R)
        name = path.name
    need = spec.image(board_config(DRAM_BYTES=1 << 40, **mk), cap, **wkw).nbytes
    cfg = board_config(DRAM_BYTES=1 << max(20, (need - 1).bit_length()), **mk)
    img = spec.image(cfg, cap, **wkw)
    progs = img.compile_step(pos)
    if 8 * len(progs[0]) > cfg.IMEM_WORDS:
        cfg = dataclasses.replace(cfg, IMEM_WORDS=1 << (8 * len(progs[0]) - 1).bit_length())
    return name, spec, cfg, progs, np.zeros(need, np.uint8)


def execute(cfg, programs, images, args=None):
    """upstream's examples.opentpu.workload.execute (the ISA simulator's command trace with the
    resolved registers and the footprints), without its refusal of DSTEP / STREAM: wp."""
    from types import MethodType
    from opentpu import isa as I, waits as W
    from opentpu.isasim import Machine
    from examples.opentpu.workload import Command, Workload
    args = list(args or [])
    machine = Machine(cfg, programs, images, args=args)
    commands = [[] for _ in range(cfg.S)]

    def record_step(self):
        ins = self.prog[self.pc]
        if ins.op == I.HALT and ins.flags:
            raise ValueError("HALT CHAIN is not supported by the timing frontend")
        regs = [(self.reg(r),) * 2 for r in range(16)]
        resolved = I.Instr(ins.op, flags=ins.flags, w=[v[0] for v in W.resolve(ins, regs)])
        commands[self.sid].append(Command(self.pc, resolved, W.footprint(ins, regs, cfg.D, cfg.S)))
        original[self.sid]()

    original = [s.step for s in machine.slices]
    for s in machine.slices:
        s.step = MethodType(record_step, s)
    machine.run(max_steps=1 << 40)
    return Workload(cfg, programs, images, commands, machine, {}, args)


# ------------------------------------------------------------------------------ the model
def load_wide(a):
    """The hwloop branch's tools/hwloop/lpn/wide.py (the tier's wide model), from its worktree
    or from git; its sha256 and the branch's commit are in the record."""
    p = Path(a.wide)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    src = None
    if p.is_file():
        src = p.read_bytes()
    else:
        try:
            src = subprocess.check_output(["git", "-C", str(ROOT), "show", "hwloop:tools/hwloop/lpn/wide.py"])
            p = work / "wide.py"
            p.write_bytes(src)
        except subprocess.CalledProcessError:
            raise SystemExit(f"no wide.py at {a.wide} and `git show hwloop:tools/hwloop/lpn/wide.py` failed")
    sys.path.insert(0, str(p.parent))
    import wide
    commit = ""
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "hwloop"], text=True).strip()
    except subprocess.CalledProcessError:
        pass
    return wide, {"path": str(p), "sha256": hashlib.sha256(src).hexdigest(), "hwloop_commit": commit}


def model_classes(wide, cfg_isa):
    """The subclasses, built once the clone is on sys.path (the DSL reads their source)."""
    from math import ceil, gcd
    import numpy as np
    from examples.opentpu.model import DRAM
    from examples.opentpu.compact import dependencies
    from examples.opentpu.components import CommandInterface, ExecutionUnit
    from opentpu import profile as PF
    from opentpu import quant as Q
    I = cfg_isa
    SCALE = wide.SCALE
    MMPlan = wide.MMPlan

    def pair_of(m, nch, chash):
        """p1: otpu_native_dram's pair at CPC = nch / 2: chunk m on pair m mod CPC, no hash (a
        wide read's two chunks are always on two pairs)."""
        return np.asarray(m, np.int64) % max(1, nch // 2)

    def chan_of(byte_addr, nch, chash):
        """p1: a beat's channel: its chunk's pair, then the half (address bit 6, with CHASH XOR
        the chunk index's parity: otpu_native_dram chan_of)."""
        a = np.asarray(byte_addr, np.int64)
        m = a >> 7
        h = (a >> 6) & 1
        if chash:
            par = m.copy()
            for k in (16, 8, 4, 2, 1):
                par ^= par >> k
            h ^= par & 1
        return 2 * pair_of(m, nch, chash) + h if nch > 2 else h

    @dataclasses.dataclass(frozen=True)
    class WpTiming(wide.WideTiming):
        aline: int = 32            # wp: Config.ALINE, port A's line
        dstep_lanes: int = 8       # wp: I.DSTEP_LANES, the state words a cycle

    def plan_mm(ins, cfg, t, I):
        """wide.plan_mm with the ALINE line and the t2 scale layout (wp)."""
        w, fl = ins.w, ins.flags
        D = cfg.D
        LINE = t.aline
        n, kb, k0 = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, (w[3] >> 24) & 0xFF
        rs = w[4] & ~getattr(I, "RS_MASK", 0x3)
        wf = (fl >> I.WF_SHIFT) & 3
        t2 = wf == I.W4I and bool(w[4] & getattr(I, "RS_T2", 0))
        bpc = 1 if wf == I.W8 else 4 if t2 else 2
        pair = bool(fl & I.F_PAIR)
        quad = bool(w[4] & getattr(I, "RS_QUAD", 0))
        L = I.mm_lanes(ins) if hasattr(I, "mm_lanes") else 0      # p1: no row lanes
        reuse = 4 if quad else 2 if pair else 1
        scaled = int(not bool(fl & I.F_UNIT))
        rsc = bool(fl & getattr(I, "F_RSC", 0))
        bs = 1 if wf == I.W8 else 8 if wf == I.WNV else 4
        fmt = "t2" if t2 else ["int8", "int4", "fp4", "nvfp4"][wf]
        sa, ssa, srs = w[0], w[1], w[6]
        nch, chash = t.nch, t.chash
        if L:
            groups = ceil(n / L)
            adv = ceil(kb / bpc)
            c_req = L
            requests = steps = adv
            A = Q.rl_adv_bytes(fmt, L)
            H = Q.rl_header(fmt, L)
            sw_extra = 1 if (scaled and not rsc and Q.rl_sw_alone(fmt, L, LINE)) else 0    # wp: the line
            lines = (adv + sw_extra) if scaled else 0
            model_rows, chunks_row = groups, L * adv
            row_stride = rs
            issue_extra = sw_extra
            cross = 0
        else:
            chunks_row = ceil(kb / bpc)
            # p1: an int8 advance takes CPC chunks (the column groups: no PAIR flag on the
            # instruction; an odd KB's last advance a half pair); 4-bit one chunk an advance
            c_req = cfg.CPC if wf == I.W8 else max(reuse // bpc, 1)
            requests = ceil(chunks_row / c_req)
            steps = ceil(kb / c_req) if wf == I.W8 else ceil(kb / reuse)
            model_rows = n
            row_stride = rs
            cross = 0
            # a 4-bit / NVFP4 QUAD advance's scale group crossing a line: the first line alone
            # (wp: at the ALINE boundary; t2's QUAD reads one 16-byte group, never two lines)
            if scaled and not rsc and reuse > 1 and wf != I.W8 and not t2:
                base = 24 if wf == I.WNV else 12
                Ab = reuse * bs
                for j in range(steps):
                    so = base + (k0 + j * reuse) * bs
                    cross += int(so // LINE != (so + Ab - 1) // LINE)
            issue_extra = cross
            lines = (requests + cross) if scaled else 0
        rows = np.arange(model_rows, dtype=np.int64)
        starts = sa + rows * row_stride
        wide_reads = two_part = 0
        beats = np.zeros(max(nch, 1), np.int64)
        if nch > 1:
            for j in range(requests):
                q0 = j * c_req
                k = min(c_req, chunks_row - q0)
                addrs = starts[:, None] + (q0 + np.arange(k)) * D
                ms = addrs >> 7
                if k > 1:
                    wide_reads += model_rows
                    p = pair_of(ms, nch, chash)
                    p.sort(axis=1)
                    two_part += int(np.count_nonzero(np.any(p[:, 1:] == p[:, :-1], axis=1)))
                pr = pair_of(ms, nch, chash).ravel()
                cnt = np.bincount(pr, minlength=nch // 2)
                beats[0::2] += cnt
                beats[1::2] += cnt
            if lines and scaled:
                sbase = ssa + rows * srs
                if L:
                    offs = [H + q * A for q in range(adv)] + ([0] if sw_extra else [])
                else:
                    base = 4 if wf == I.W8 else 16 if t2 else 24 if wf == I.WNV else 12
                    offs = [base + (k0 + j * reuse) * bs for j in range(requests)]
                    if cross:
                        offs += [o + LINE for o in offs if o // LINE != (o + reuse * bs - 1) // LINE]
                bts = np.unique((sbase[:, None] + np.asarray(offs, np.int64)[None, :]) >> 6)
                beats += np.bincount(chan_of(bts << 6, nch, chash), minlength=nch)
        else:
            for j in range(requests):
                k = min(c_req, chunks_row - j * c_req)
                if k > 1:
                    wide_reads += model_rows
        return MMPlan(n=model_rows, rows_isa=n, lanes=L, kb=kb, bpc=bpc, reuse=reuse, chunks=chunks_row,
                      requests=requests, steps=steps, entries=requests, lines=lines, issue_extra=issue_extra,
                      wide_reads=wide_reads, two_part=two_part, chan_beats=beats.tolist() if nch > 1 else [],
                      scaled=scaled)

    class WpCompactOpenTPU(wide.WideCompactOpenTPU):
        """The wide model with the items of the header (wp), and the phase-run transition."""
        def __init__(self, workload, timing, runs, phase_src=None):
            self.runs = runs                  # [(phase, [idx, ...])] in dispatch order
            self.plans = {}
            self.tally = {}
            if timing.dram_bytes_per_cycle != workload.config.D:
                raise ValueError("the wide model's port credit is in chunks: dram_bytes_per_cycle must be D")
            super(wide.WideCompactOpenTPU, self).__init__(workload, timing)

        def _simple_timing(self, ins, memory, I):
            cfg, t, w = self.workload.config, self.timing, ins.w
            if ins.op in (I.DSTEP, I.STREAM):
                # wp: the DeltaNet state update on the DMA unit: DSTEP_LANES state words a
                # cycle, every state chunk read and written on port B (profile._describe), the
                # first chunk's round trip before the update starts
                _, _, cyc, pb, _ = PF._describe(ins, cfg)
                latency = max(cyc, pb) + t.dram_latency + t.adapter_latency
                return latency, memory.b, pb
            if ins.op == I.WAITW:
                return 1, None, 0
            return super()._simple_timing(ins, memory, I)

        def _slice(self, sid, commands, I, W):
            cfg, t, net = self.workload.config, self.timing, self.net
            prefix = f"s{sid}"
            ncommands = len(commands)
            if not ncommands:
                raise ValueError("empty command trace")
            memory = DRAM(net, prefix + "_dram", t)
            channels = [net.place(prefix + f"_dram_ch{c}", initial_tokens=1) for c in range(t.nch)] if t.nch > 1 else []
            nsig = ncommands
            banks = [net.place(prefix + f"_flags_{b}", initial_tokens=1) for b in range(nsig // 64 + 1)]
            signals = net.place(prefix + "_signals")
            finished = net.place(prefix + "_halted")

            @net.func(name=prefix + "_is_done", args=("i64",), results=("i1",))
            def is_done(idx):
                token = peek(array(banks)[divi(idx, 64)])
                mask = shli(1, remi(idx, 64))
                func_return(select(idx == nsig, True, andi(token.get("bits"), mask) != 0, typ="i1"))

            @net.func(name=prefix + "_publish", args=("i64", "f64"), results=())
            def publish(idx, latency):
                emit(signals, create().set("id", idx), delay=latency)
                func_return()

            slots = [net.place(prefix + f"_slot_{s}", initial_tokens=1) for s in range(t.window)]
            window = net.place(prefix + "_window")
            initialize = net.place(prefix + "_initialize", initial_tokens=1)
            cursor = net.place(prefix + "_cursor", initial_tokens=1)
            pipeline = net.place(prefix + "_pipeline")
            pending = [net.place(prefix + f"_pending_{u}") for u in range(6)]
            ports = [memory.b, memory.sw]
            deps, dram_deps = dependencies(commands, I, W, 1)
            kinds, latencies, services, port_ids = [], [], [], []
            batches, entries, scale_flags, kbs, ns, row_services, tails = [], [], [], [], [], [], []
            port_xs, lines, chan_xs = [], [], [[] for _ in range(t.nch)]
            barriers, dispatch_delays, pipeline_delays = [], [], []
            last_register = nsig
            tally = {"mm": 0, "chunks": 0, "requests": 0, "steps": 0, "lines": 0, "issue_extra": 0,
                     "wide_reads": 0, "two_part": 0, "chan_beats": [0] * t.nch, "dma_chunks": 0, "by_kind": {}}
            for idx, c in enumerate(commands):
                ins, w = c.instruction, c.instruction.w
                u = W.UNIT.get(ins.op, 5)
                kinds.append(u)
                barriers.append(last_register)
                dispatch_delays.append(2 if ins.op == I.LOOP else 1)
                pipeline_delays.append(5 if ins.op >= I.LD else 0)
                if ins.op == I.RLD:
                    last_register = idx
                latency, port, service = (0, None, 0) if u in (W.U_MXU, 5) else self._simple_timing(ins, memory, I)
                latencies.append(latency)
                port_ids.append(ports.index(port) if port is not None else -1)
                services.append(service)
                if port is memory.b:
                    tally["dma_chunks"] += service
                batch, entry, scaled, kb, n, row_service, tail = 1, 1, 0, 1, 1, 1, 0
                port_x, line = 0, 0
                chan_x = [0] * t.nch
                if ins.op == I.MM:
                    p = plan_mm(ins, cfg, t, I)
                    self.plans[(sid, idx)] = p
                    m = (w[5] >> 16) & 255
                    if p.entries > t.fifo_chunks:
                        raise ValueError("one matrix row must fit the prefetch FIFO")
                    n, kb, scaled, entry = p.n, p.kb, p.scaled, p.entries
                    batch = min(t.batch_rows, max(1, n // 64), max(1, t.fifo_chunks // (2 * entry)))
                    if p.lanes:
                        drain = ceil(p.lanes / min(t.epilogue_lanes, p.lanes))
                    else:
                        drain = ceil(m / min(t.epilogue_lanes, cfg.LANES // gcd(w[5] & 65535, cfg.LANES)))
                    row_service = max(p.steps * ceil(m / cfg.MCOLS), drain)
                    tail = min(cfg.D // 8, t.chain_length) + cfg.MCOLS + 5 + 9 + (6 if ins.flags & I.F_ACC else 0) + drain
                    port_x, line = p.port_x, p.lines
                    chan_x = [p.chan_x(ch, t.channel_rate) for ch in range(t.nch)]
                    self.matrix_batch_sizes.append(batch)
                    tally["mm"] += 1
                    tally["chunks"] += p.n * p.chunks
                    tally["requests"] += p.n * p.requests
                    tally["steps"] += p.n * p.steps * ceil(m / cfg.MCOLS)
                    tally["lines"] += p.n * p.lines
                    tally["issue_extra"] += p.n * p.issue_extra
                    tally["wide_reads"] += p.wide_reads
                    tally["two_part"] += p.two_part
                    if p.chan_beats:
                        tally["chan_beats"] = [x + y for x, y in zip(tally["chan_beats"], p.chan_beats)]
                    key = (f"L{p.lanes}" if p.lanes else "QUAD" if p.reuse == 4 else "PAIR" if p.reuse == 2 else "plain")
                    tally["by_kind"][key] = tally["by_kind"].get(key, 0) + 1
                batches.append(batch)
                entries.append(entry)
                scale_flags.append(scaled)
                kbs.append(kb)
                ns.append(n)
                row_services.append(row_service)
                tails.append(tail)
                port_xs.append(port_x)
                lines.append(line)
                for ch in range(t.nch):
                    chan_xs[ch].append(chan_x[ch])
            self.tally[sid] = tally
            unit_parameters = wide._UnitTiming(latencies, services, port_ids)
            parameters = {"rows": ns, "batches": batches, "entries": entries, "port_x": port_xs, "lines": lines,
                          "scaled": scale_flags, "row_services": row_services, "tails": tails, "chan_x": chan_xs}
            metadata = [ncommands, max(1, t.fifo_chunks // max(b * c for b, c in zip(batches, entries)))]

            def csr(rows):
                offsets, flat = [0], []
                for row in rows:
                    flat.extend(row)
                    offsets.append(len(flat))
                return offsets, flat or [nsig]

            starts = [dram_deps[i] if kinds[i] == W.U_MXU else deps[i] for i in range(ncommands)]
            offsets, indices = csr(starts)
            compute_offsets, compute_indices = csr(deps)
            begins, ends = offsets[:-1], offsets[1:]

            @net.func(name=prefix + "_select_ready", args=("i64",), results=("i64",))
            def select_ready(unit):
                slot_places = array(slots)
                kind, begin, end = array(kinds), array(begins), array(ends)
                dep_indices = array(indices)
                selected, oldest = subi(0, 1), addi(array(metadata)[0], 1)
                selected_enabled = const_i64(0) == 0
                for s in range(t.window):
                    token = peek(slot_places[s])
                    idx = token.get("id")
                    active = select(token.get("active") == 1, kind[idx] == unit, False, typ="i1")
                    enabled = active
                    for d in range(select(enabled, begin[idx], 0), select(enabled, end[idx], 0)):
                        enabled = select(enabled, is_done(dep_indices[d]), False, typ="i1")
                    candidate = select(unit == W.U_MXU, active, enabled, typ="i1")
                    eligible = select(candidate, idx < oldest, False, typ="i1")
                    selected = select(eligible, s, selected, typ="i64")
                    oldest = select(eligible, idx, oldest)
                    selected_enabled = select(eligible, enabled, selected_enabled, typ="i1")
                func_return(select(selected_enabled, selected, subi(0, 1), typ="i64"))

            @net.transition(prefix + "_initialize_window")
            def initialize_window():
                take(initialize)
                for s in range(t.window):
                    emit(window, create().set("slot", s))

            @net.transition(prefix + "_dispatch")
            def dispatch():
                guard(count(cursor) > 0)
                guard(count(window) > 0)
                idx = peek(cursor).get("id")
                guard(idx < array(metadata)[0])
                bs, ps = array(barriers), array(pipeline_delays)
                barrier = bs[idx]
                guard(is_done(barrier))
                take(cursor)
                token = take(window).set("id", idx).set("active", 1)
                emit(pipeline, token, delay=sitofp(ps[idx]))
                emit(cursor, create().set("id", idx + 1), delay=sitofp(array(dispatch_delays)[idx]))

            @net.transition(prefix + "_arrive")
            def arrive():
                guard(count(pipeline) > 0)
                token = take(pipeline)
                place = array(slots)[token.get("slot")]
                take(place)
                emit(place, token)
                emit(array(pending)[array(kinds)[token.get("id")]], create())

            interface = CommandInterface(slots, pending, window, select_ready, is_done, publish,
                                         compute_offsets, compute_indices)
            units = [ExecutionUnit(net, prefix + f"_unit_{u}", u, interface, unit_parameters, memory)
                     for u in (W.U_DMA, W.U_Q, W.U_VPU, W.U_COLL, 5)]
            matrices = [wide.WideMatrixEngine(net, prefix + "_mxu_0", cfg, t, interface, parameters, memory, W.U_MXU, channels)]
            self.components[sid] = {"memory": memory, "units": units, "matrix_engines": matrices, "channels": channels}

            @net.transition(prefix + "_mark_done")
            def mark_done():
                guard(count(signals) > 0)
                idx = peek(signals).get("id")
                bank = array(banks)[divi(idx, 64)]
                guard(count(bank) > 0)
                take(signals)
                token = take(bank)
                emit(bank, token.set("bits", ori(token.get("bits"), shli(1, remi(idx, 64)))))
                if idx == array(metadata)[0] - 1:
                    emit(finished, create())

            # wp: the phase runs. A cursor walks them in order; run k's transition fires when
            # every command of the run is complete, so its --trace timestamp is the run's end
            # under Profile.phases' running-max rule (the cursor keeps the order).
            runs = self.runs if sid == 0 else []
            run_lo, run_idx = [0], []
            for _, members in runs:
                run_idx.extend(members)
                run_lo.append(len(run_idx))
            run_idx = run_idx or [nsig]
            nruns = len(runs)
            pcur = net.place(prefix + "_phase_cursor", initial_tokens=1)
            pmark = net.place(prefix + "_phase_marks")

            @net.transition(prefix + "_phase_mark")
            def phase_mark():
                guard(count(pcur) > 0)
                k = peek(pcur).get("k")
                guard(k < nruns)
                lo_t, ri = array(run_lo), array(run_idx)
                enabled = const_i64(0) == 0
                for d in range(lo_t[k], lo_t[k + 1]):
                    enabled = select(enabled, is_done(ri[d]), False, typ="i1")
                guard(enabled)
                take(pcur)
                emit(pcur, create().set("k", k + 1))
                emit(pmark, create().set("k", k))

            self.slice_done.append(finished)

    return WpTiming, WpCompactOpenTPU, plan_mm


# ------------------------------------------------------------------------------ driving
def phase_runs(commands, progs, W, phase_of):
    """The runs of consecutive unit commands (the scalar ones have no record on the RTL) of
    one phase, in dispatch order: [(phase, [idx, ...])]."""
    runs = []
    for idx, c in enumerate(commands):
        if W.UNIT.get(c.instruction.op) is None:
            continue
        ph = phase_of(progs[0][c.pc])
        if runs and runs[-1][0] == ph:
            runs[-1][1].append(idx)
        else:
            runs.append((ph, [idx]))
    return runs


def target_uarch(target: str) -> dict:
    """The target's [uarch] values the model takes from the RTL (targets/<name>.toml)."""
    name = target.split(":")[0]
    p = ROOT / "targets" / f"{name}.toml"
    out = {}
    for k in ("FIFO_DEPTH", "WIN", "MXU_E", "CHASH"):
        m = re.search(rf"^{k}\s*=\s*(\w+)", p.read_text(), re.M) if p.is_file() else None
        if m:
            v = m.group(1)
            out[k] = (v == "true") if v in ("true", "false") else int(v)
    return out


def native(binary: Path, work: Path, trace: bool = True) -> tuple[dict, str]:
    cmd = [str(binary), "--stats", "--wall-start=begin", "--wall-end=complete", "--max-time=1000000000000"]
    if trace:
        cmd.append("--trace")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = {}
    for k, rx in NATIVE_RE.items():
        m = re.search(rx, r.stdout)
        if m:
            out[k] = float(m.group(1)) if k in ("cycles", "wall_s") else int(m.group(1))
    out["process_s"] = time.time() - t0
    (work / "native.log").write_text("\n".join(l for l in r.stdout.splitlines() if not l.startswith("[t=")) + r.stderr)
    if r.returncode or out.get("completed") != 1 or out.get("rollbacks"):
        raise RuntimeError(f"native run failed or incomplete: {work / 'native.log'}")
    out["cycles_exact"] = out["cycles"]
    out["cycles"] = int(math.ceil(out["cycles"]))
    return out, r.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma3", help="opentpu.llm.MODELS short name or a checkpoint directory")
    ap.add_argument("--bonsai", action="store_true", help="a Ternary-Bonsai 2 directory (GGUF beside config.json): t2 with the rotation")
    ap.add_argument("--layers", type=int, default=4, help="the first N layers and the whole head; 0: the whole model")
    ap.add_argument("--pos", type=int, default=9)
    ap.add_argument("--wformat", default="int8")
    ap.add_argument("--head-format", default=None)
    ap.add_argument("--cpc", type=int, default=None, choices=[1, 2],
                    help="p1: chunks a cycle (default the target's: 2)")
    ap.add_argument("--mcols", type=int, default=None)
    ap.add_argument("--nch", type=int, default=None, help="p1: native channels (default 2 CPC)")
    ap.add_argument("--lat", type=int, default=60, help="the receipt's --lat: the port's read latency")
    ap.add_argument("--adapter-latency", type=int, default=None, help="cycles of path over --lat (default 14 on the native path, 4 behavioral)")
    ap.add_argument("--channel-depth", type=int, default=None, help="B read beats in flight a channel (p1 default: otpu_native_dram RD 128, otpu_native_axi MAX_OUT 128)")
    ap.add_argument("--window", type=int, default=None, help="the dispatch window (default the target's WIN)")
    ap.add_argument("--epilogue-lanes", type=int, default=None, help="default the target's MXU_E")
    ap.add_argument("--batch-rows", type=int, default=8, help="upstream's validated rows a prefetch event")
    ap.add_argument("--mhz", type=float, default=200.0)
    ap.add_argument("--measured", help="a tools/perf_qwen.py / perf_bonsai2.py --json receipt (its last line): the phases to calibrate against")
    ap.add_argument("--rtl-cycles", type=int, default=None, help="the measured whole-token cycles when there is no receipt with phases")
    ap.add_argument("--wide", default=str(HWLOOP / "tools" / "hwloop" / "lpn" / "wide.py"), help="the hwloop branch's wide model")
    ap.add_argument("--dialect", default=os.environ.get("HWLOOP_LPN", str(SCRATCH / "LPNDialect")))
    ap.add_argument("--lpn-compile", default=str(SCRATCH / "lpn-build" / "tools" / "lpn-compile" / "lpn-compile"))
    ap.add_argument("--llvm-bin", default=str(SCRATCH / "llvm20-bin"))
    ap.add_argument("--work", default=os.environ.get("WP_LPN_WORK", "/tmp/wideport_lpn"))
    ap.add_argument("--json")
    ap.add_argument("--diag", action="store_true", help="per phase the plan's rooflines")
    a = ap.parse_args()
    sys.path.insert(1, a.dialect)
    import lpnlang_mlir  # noqa: F401
    import numpy as np
    from opentpu import isa as I, waits as W
    from perf_qwen import _phase
    wide, wide_info = load_wide(a)
    t0 = time.time()
    name, spec, cfg, progs, img = build_workload(a)
    work = execute(cfg, progs, [img])
    commands = work.commands[0]
    fe = time.time() - t0
    ops = {}
    for c in commands:
        ops[I.OPNAMES.get(c.instruction.op, hex(c.instruction.op))] = ops.get(I.OPNAMES.get(c.instruction.op, hex(c.instruction.op)), 0) + 1
    runs = phase_runs(commands, progs, W, _phase)
    ua = target_uarch(os.environ["OTPU_TARGET"])
    nch = a.nch or 2 * cfg.CPC                                  # p1: 2 CPC channels
    a.rowlanes = False                                          # p1: no row lanes
    WpTiming, WpCompact, plan_mm = model_classes(wide, I)
    timing = WpTiming(dram_latency=a.lat, dram_bytes_per_cycle=cfg.D,
                      window=a.window or ua.get("WIN", 32), fifo_chunks=ua.get("FIFO_DEPTH", 128),
                      chain_length=16, epilogue_lanes=a.epilogue_lanes or ua.get("MXU_E", 2),
                      quant_buffer_chunks=1024, batch_rows=a.batch_rows,
                      port_chunks=cfg.CPC, nch=nch, chash=ua.get("CHASH", True),  # p1: CPC
                      channel_depth=a.channel_depth or 128,
                      adapter_latency=a.adapter_latency if a.adapter_latency is not None else (14 if nch > 1 else 4),
                      aline=32, dstep_lanes=getattr(I, "DSTEP_LANES", 8))
    tag = f"{name}_l{a.layers}_{a.wformat}_{a.head_format or a.wformat}_c{cfg.CPC}_m{cfg.MCOLS}_n{nch}_al{timing.adapter_latency}"
    wd = Path(a.work) / tag
    wd.mkdir(parents=True, exist_ok=True)
    print(f"{name}: {len(progs[0])} instructions, {len(commands)} commands, {len(runs)} phase runs; "
          f"{cfg}; frontend {fe:.0f} s")
    print(f"timing {timing}")
    t1 = time.time()
    model = WpCompact(work, timing, runs)
    mlir, exe = wd / "model.mlir", wd / "opentpu"
    mlir.write_text(model.build())
    build_s = time.time() - t1
    env = dict(os.environ, PATH=f"{a.llvm_bin}:{os.environ.get('PATH', '')}")
    if sys.platform == "darwin" and "SDKROOT" not in env:
        env["SDKROOT"] = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    t2 = time.time()
    comp = subprocess.run([a.lpn_compile, str(mlir), "-o", str(exe)], capture_output=True, text=True, env=env)
    (wd / "compile.log").write_text(comp.stdout + comp.stderr)
    if comp.returncode:
        raise SystemExit(f"lpn-compile failed: {wd / 'compile.log'}")
    comp_s = time.time() - t2
    nat, log = native(exe, wd)
    cycles = nat["cycles"]
    # the runs' ends from the trace: the k-th firing of s0_phase_mark is run k's
    ends = [float(m) for m in re.findall(r"\[t=([0-9.]+)\] fired s0_phase_mark$", log, re.M)]
    if len(ends) != len(runs):
        raise RuntimeError(f"{len(ends)} phase marks for {len(runs)} runs")
    phases: dict = {}
    done = 0.0
    for (ph, members), e in zip(runs, ends):
        p = phases.setdefault(ph, {"cycles": 0.0, "commands": 0, "mm": 0})
        p["cycles"] += max(0.0, e - done)
        p["commands"] += len(members)
        p["mm"] += sum(1 for i in members if commands[i].instruction.op == I.MM)
        done = max(done, e)
    for p in phases.values():
        p["cycles"] = int(round(p["cycles"]))
    # the rooflines (the plan's static tally), per phase and whole
    tally = model.tally[0]
    rate = timing.channel_rate

    def roofline(members):
        rb = mx = pa = 0
        cb = [0] * nch
        for i in members:
            p = model.plans.get((0, i))
            if p is None:
                continue
            m = (commands[i].instruction.w[5] >> 16) & 255
            rb += p.n * (p.requests + p.issue_extra) + p.two_part
            mx += p.n * p.steps * math.ceil(m / cfg.MCOLS)
            pa += p.n * p.lines
            cb = [x + y for x, y in zip(cb, p.chan_beats)] if p.chan_beats else cb
        ch = math.ceil(max(cb) / rate) if any(cb) else 0
        bound = max(("port B", rb), ("MXU", mx), ("port A", pa), ("channel", ch), key=lambda kv: kv[1])
        return {"port_b": rb, "mxu": mx, "port_a": pa, "channel": ch, "binds": bound[0]}

    per_phase_members = {}
    for ph, members in runs:
        per_phase_members.setdefault(ph, []).extend(members)
    rl = {ph: roofline(m) for ph, m in per_phase_members.items()}
    rl_all = roofline(range(len(commands)))
    meas = None
    if a.measured:
        lines_ = [l for l in Path(a.measured).read_text().splitlines() if l.strip()]
        meas = json.loads(lines_[-1])
    rtl_cycles = (meas or {}).get("cycles") or a.rtl_cycles
    # ---- the table
    print(f"\n{name} {a.wformat}/{a.head_format or a.wformat}, {'all' if not a.layers else a.layers} layers + head, CPC {cfg.CPC}, "
          f"MCOLS {cfg.MCOLS}, port A line {timing.aline}, NCH {nch} at --lat {a.lat} + {timing.adapter_latency} adapter")
    print(f"LPN ({LABEL}): {cycles:,} cycles" + (f"; RTL (measured): {rtl_cycles:,}, error {100 * (cycles / rtl_cycles - 1):+.2f}%" if rtl_cycles else "")
          + f"; wall: frontend {fe:.0f} s, model {build_s:.0f} s, compile {comp_s:.0f} s, native {nat['process_s']:.1f} s")
    print(f"rooflines: port B {rl_all['port_b']:,} / MXU {rl_all['mxu']:,} / port A {rl_all['port_a']:,} / channel {rl_all['channel']:,} "
          f"(RD {timing.channel_depth} over {timing.round_trip}: {rate:.3f} beats a cycle): {rl_all['binds']} binds; "
          f"wide reads {tally['wide_reads']:,}, two-part {tally['two_part']:,}; MM kinds {tally['by_kind']}; DMA chunks {tally['dma_chunks']:,}")
    errs, small = [], []           # (phase, error, RTL cycles): the phases that carry the token, the rest
    print(f"{'phase':10s} {'RTL (measured)':>15s} {'LPN (estimate)':>15s} {'error':>8s}   binds (port B / MXU / port A / channel)")
    order = sorted(phases, key=lambda k: -phases[k]["cycles"])
    mphases = (meas or {}).get("phases") or {}
    for ph in order:
        lp = phases[ph]["cycles"]
        mc = mphases.get(ph, {}).get("cycles")
        err = (lp / mc - 1) if mc else None
        if err is not None:
            (errs if mc >= SMALL * rtl_cycles else small).append((ph, err, mc))
        r = rl[ph]
        print(f"{ph:10s} {mc if mc else '-':>15} {lp:>15,} {('%+.1f%%' % (100 * err)) if err is not None else '-':>8s}   "
              f"{r['binds']} ({r['port_b']:,} / {r['mxu']:,} / {r['port_a']:,} / {r['channel']:,})")
    for ph in mphases:
        if ph not in phases:
            print(f"{ph:10s} {mphases[ph]['cycles']:>15} {'-':>15s} {'-':>8s}   (no run of this phase in the model's trace)")
    mean_abs = sum(abs(e) for _, e, _ in errs) / len(errs) if errs else None
    wmean = sum(abs(e) * mc for _, e, mc in errs + small) / sum(mc for _, _, mc in errs + small) if errs or small else None
    calibrated = mean_abs is not None and mean_abs <= GATE
    tok_s = a.mhz * 1e6 / cycles
    if mean_abs is not None:
        print(f"mean |error| over the {len(errs)} phases carrying >= {100 * SMALL:.0f}% of the token: {100 * mean_abs:.1f}% "
              f"(cycle-weighted over all {len(errs) + len(small)}: {100 * wmean:.1f}%; gate {100 * GATE:.0f}%): "
              + ("calibrated" if calibrated else "NOT calibrated: no tok/s from this point")
              + (f"; left out of the mean: " + ", ".join(f"{ph} {100 * e:+.0f}% of {mc:,}" for ph, e, mc in small) if small else ""))
    line = f"tok/s at {a.mhz:g} MHz ({LABEL}): {tok_s:.1f}" + (f" (RTL measured {a.mhz * 1e6 / rtl_cycles:.1f})" if rtl_cycles else "")
    print(line if (calibrated or mean_abs is None) else f"({line}: withheld, see the gate)")
    res = {"label": LABEL, "model": name, "args": vars(a), "config": str(cfg), "timing": dataclasses.asdict(timing),
           "target_uarch": ua, "wide_model": wide_info, "tree": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                                                 capture_output=True, text=True).stdout.strip(),
           "commands": len(commands), "opcodes": ops, "runs": len(runs), "cycles": cycles, "cycles_exact": nat["cycles_exact"],
           "tok_s": tok_s, "rtl_cycles": rtl_cycles, "error": (cycles / rtl_cycles - 1) if rtl_cycles else None,
           "phases": {ph: dict(phases[ph], roofline=rl[ph], rtl=((meas or {}).get("phases") or {}).get(ph, {}).get("cycles"))
                      for ph in phases},
           "mean_abs_error": mean_abs, "weighted_mean_abs_error": wmean, "small_phases": small, "calibrated": calibrated, "roofline": rl_all, "tally": tally,
           "native": nat, "wall": {"frontend_s": fe, "model_s": build_s, "compile_s": comp_s},
           "work": str(wd), "measured": a.measured}
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    (wd / "result.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
