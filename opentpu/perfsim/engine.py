"""Cycle-stepped OpenTPU issue window, dependency scoreboard and resource arbitration.

Functional commits use the repository's ISA engine, independently of timing. This is
not a claim of an independent numerical implementation; validation also runs NumPy/RTL.
"""
from collections import Counter
from dataclasses import asdict
import time

from .. import isa as I, waits as W
from ..isasim import Slice
from .config import Architecture, MemoryConfig, Unsupported
from .memory import MemorySystem


class Banks:
    def __init__(self, cfg, arch):
        self.cfg, self.arch = cfg, arch
        self.reset()

    def reset(self):
        self.reads, self.writes = Counter(), Counter()

    def grant(self, reads=(), writes=()):
        r = Counter(a % self.cfg.LANES for a in reads)
        w = Counter(a % self.cfg.LANES for a in writes)
        if any(v > self.arch.tmem_read_ports_per_bank for v in r.values()) or any(
                v > self.arch.tmem_write_ports_per_bank for v in w.values()):
            raise Unsupported('an atomic lane request exceeds the configured bank ports')
        if any(self.reads[b] + n > self.arch.tmem_read_ports_per_bank for b, n in r.items()) or any(
                self.writes[b] + n > self.arch.tmem_write_ports_per_bank for b, n in w.items()):
            return False
        self.reads.update(r)
        self.writes.update(w)
        return True


class Command:
    def __init__(self, sim, ident, pc, ins, regs):
        self.sim, self.id, self.pc, self.ins, self.regs = sim, ident, pc, ins, regs
        self.w = [v[0] for v in W.resolve(ins, [(v, v) for v in regs])]
        self.fp = W.footprint(ins, [(v, v) for v in regs], sim.cfg.D, sim.cfg.S)
        self.unit = W.UNIT[ins.op]
        self.dispatch, self.start, self.release, self.end = sim.cycle, None, None, None
        self.stalls, self.work = Counter(), Counter()
        self.task = None
        self.deps, self.dram_deps = [], []
        self.label = sim.labels.get(pc, ins.comment or I.OPNAMES[ins.op])

    def complete(self):
        sl = self.sim.state
        old = sl.R
        sl.R = self.regs.copy()
        if self.ins.op != I.BAR:
            sl.execute(self.ins)
        value = sl.R[self.ins.rd]
        sl.R = old
        if self.ins.op == I.RLD and self.ins.rd:
            sl.R[self.ins.rd] = value
        self.end = self.sim.cycle

    def record(self):
        return dict(id=self.id, pc=self.pc, opcode=I.OPNAMES[self.ins.op], unit=self.unit,
                    label=self.label, dispatch=self.dispatch, start=self.start,
                    release=self.release, end=self.end,
                    cycles=self.end - self.dispatch if self.end is not None else None,
                    stalls=dict(self.stalls), work=dict(self.work),
                    memory=dict(self.sim.memory.by_owner.get(self.id, {})))


class Simulator:
    def __init__(self, cfg, program, image, *, architecture=None, memory=None, args=None, labels=None, memory_system=None):
        self.cfg = cfg
        self.arch = architecture or Architecture()
        self.memcfg = memory_system.cfg if memory_system is not None else memory or MemoryConfig()
        self.arch.validate(cfg)
        self.program = program
        self.labels = labels or {}
        if len(program)*8 > cfg.IMEM_WORDS:
            raise ValueError('program exceeds IMEM')
        self._validate_program()
        self.state = Slice(cfg, 0, program, image)
        for n, v in enumerate(args or []):
            if n >= 8:
                raise ValueError('at most eight runtime arguments')
            self.state.R[n+8] = I.u32(v)
        self.memory = memory_system or MemorySystem(self.memcfg)
        self.owns_memory = memory_system is None
        self.memory_offset = self.memory.cycle+1
        self.owner_base = self.memory.next_owner
        self.memory.run_peak = 0
        self.memory_baseline = self.memory.stats.copy()
        self.banks = Banks(cfg, self.arch)
        self.commands, self.window = [], []
        self.cycle, self.next_dispatch = 0, self.arch.dispatch_latency
        self.stalls, self.resources, self.busy = Counter(), set(), Counter()
        self.rld_pending = None
        self.halt_seen = False

    def _validate_program(self):
        supported = {I.NOP,I.HALT,I.LI,I.ADDI,I.LOOP,I.BAR,I.RLD,I.LD,I.ST,I.MM,I.QACT,I.QST,I.VOP}
        for pc, ins in enumerate(self.program):
            if ins.op not in supported or (ins.op == I.HALT and ins.flags):
                raise Unsupported(f'pc {pc}: {I.OPNAMES.get(ins.op, hex(ins.op))} flags {ins.flags:#x}')
            masks={I.NOP:0,I.HALT:0,I.LI:0,I.ADDI:0,I.LOOP:0,I.BAR:0,I.RLD:7,I.LD:0,I.ST:0,I.MM:255,I.QACT:15,I.QST:3,I.VOP:3 if ((ins.w[5]>>16)&255)==I.V_OUTER else 0}
            if ins.flags & ~masks[ins.op]:
                raise Unsupported(f'pc {pc}: unsupported flags {ins.flags:#x}')
            if ins.op == I.VOP and (ins.w[5] >> 16) & 255 not in I.VFUNCS:
                raise Unsupported(f'pc {pc}: unknown VOP function')
            if ins.op == I.MM:
                kb, m = (ins.w[3] >> 16) & 255, (ins.w[5] >> 16) & 255
                if not (ins.w[3] & 65535) or not kb or not m:
                    raise Unsupported(f'pc {pc}: zero-size MM')
                if kb > self.cfg.ACT_BLOCKS or m > self.cfg.act_rows:
                    raise Unsupported(f'pc {pc}: MM exceeds ACT RAM')
                if ins.flags & I.F_PAIR and not self.cfg.PAIR:
                    raise Unsupported(f'pc {pc}: PAIR instruction without PAIR hardware')

    def reserve(self, resource):
        if resource in self.resources:
            return False
        self.resources.add(resource)
        return True

    def dispatch(self):
        if self.halt_seen or self.cycle < self.next_dispatch:
            return
        if self.rld_pending is not None and self.rld_pending.end is None:
            self.stalls['scalar_dependency'] += 1
            return
        if len(self.window) >= self.arch.window:
            self.stalls['window_full'] += 1
            return
        sl = self.state
        if not 0 <= sl.pc < len(self.program):
            raise ValueError('program ran past IMEM without HALT')
        ins = self.program[sl.pc]
        self.next_dispatch = self.cycle + self.arch.issue_period
        if ins.op == I.HALT:
            self.halt_seen = True
        elif ins.op in (I.NOP, I.LI, I.ADDI, I.LOOP):
            sl.step()
        else:
            cmd = Command(self, self.owner_base+len(self.commands), sl.pc, ins, [sl.reg(r) for r in range(16)])
            for older in self.window:
                if W.conflict(cmd.fp, older.fp):
                    cmd.deps.append(older)
                if W.conflict(cmd.fp, older.fp, dram_only=True):
                    cmd.dram_deps.append(older)
            self.commands.append(cmd)
            self.memory.next_owner = cmd.id+1
            self.window.append(cmd)
            sl.advance()
            if ins.op == I.RLD:
                self.rld_pending = cmd

    def run(self):
        from .units import make_task
        wall = time.monotonic()
        try:
            while self.cycle < self.arch.max_cycles:
                self.memory.begin_cycle(self.memory_offset+self.cycle, self.arch.core_period_ps)
                self.banks.reset()
                self.resources.clear()
                for cmd in list(self.window):
                    ready = all(d.end is not None for d in cmd.deps)
                    if ready and cmd.release is None:
                        cmd.release = self.cycle
                    if cmd.task is None:
                        if self.cycle < cmd.dispatch + self.arch.dispatch_latency:
                            continue
                        older_unit = [x for x in self.window if x.id < cmd.id and x.unit == cmd.unit]
                        active = [x for x in self.window if x.unit == cmd.unit and x.task is not None]
                        capacity = self.arch.mxu_queue if cmd.unit == W.U_MXU else self.arch.vpu_slots if cmd.unit == W.U_VPU else 1
                        # In-order units cannot pass an older unstarted command; VPU oldest ready can.
                        if len(active) >= capacity or (cmd.unit != W.U_VPU and any(x.task is None for x in older_unit)):
                            cmd.stalls['unit_queue'] += 1
                            continue
                        stream_ready = all(d.end is not None for d in cmd.dram_deps)
                        if not (stream_ready if cmd.unit == W.U_MXU else ready):
                            cmd.stalls['dependency'] += 1
                            continue
                        cmd.start = self.cycle
                        cmd.task = make_task(cmd)
                    self.busy[str(cmd.unit)] += 1
                    if cmd.task.tick():
                        cmd.complete()
                        self.window.remove(cmd)
                self.dispatch()
                if self.halt_seen and not self.window and self.memory.idle:
                    break
                self.cycle += 1
            else:
                raise TimeoutError(f'cycle limit {self.arch.max_cycles}; active PCs {[c.pc for c in self.window]}')
            result = self.report()
            result['host_wall_seconds'] = time.monotonic() - wall
            return result
        finally:
            if self.owns_memory:
                self.memory.close()

    def report(self):
        cycles = self.cycle + 1
        rows = [c.record() for c in self.commands]
        layers = {}
        for r in rows:
            z = layers.setdefault(r['label'], dict(first_dispatch=r['dispatch'], last_completion=r['end'],
                                                  instructions=0, memory=Counter(), stalls=Counter()))
            z['first_dispatch'] = min(z['first_dispatch'], r['dispatch'])
            z['last_completion'] = max(z['last_completion'], r['end'])
            z['instructions'] += 1
            z['memory'].update(r['memory']); z['stalls'].update(r['stalls'])
        for z in layers.values():
            z['span_cycles'] = z['last_completion'] - z['first_dispatch'] + 1
            z['memory'], z['stalls'] = dict(z['memory']), dict(z['stalls'])
        util = {}
        for u in range(5):
            intervals = sorted((c.start,c.end) for c in self.commands if c.unit == u)
            total, end = 0, -1
            for a,b in intervals:
                total += max(0,b-max(a,end)); end=max(end,b)
            util[str(u)] = total/cycles
        mxsteps = sum(c.work['mxu_advances'] for c in self.commands)
        macs = sum(c.work['macs'] for c in self.commands)
        return dict(schema_version=1, label='opentpu-perfsim-estimate', timing_validation='unvalidated',
                    backend=self.memcfg.backend, cycles=cycles,
                    seconds=cycles*self.arch.core_period_ps*1e-12,
                    core_frequency_hz=1e12/self.arch.core_period_ps,
                    config=asdict(self.cfg), architecture=asdict(self.arch), memory_config=asdict(self.memcfg),
                    memory={k:(self.memory.run_peak if k=='peak_outstanding' else v-self.memory_baseline[k]) for k,v in self.memory.stats.items()}, frontend_stalls=dict(self.stalls),
                    unit_busy_fraction=util, mxu_advance_fraction=mxsteps/cycles,
                    useful_mac_fraction=macs/(cycles*self.cfg.D*self.cfg.MCOLS),
                    instructions=rows, layers=layers,
                    boundaries={'start':'core run, program resident; no host upload',
                                'end':'HALT and all write completions drained',
                                'ttft_seconds':None,'prefill_seconds':None,'decode_seconds':None,
                                'conditional_tokens_per_second':None})
