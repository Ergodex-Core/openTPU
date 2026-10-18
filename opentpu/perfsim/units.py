"""OpenTPU unit micro-operations and bounded pipelines (timing approximations).

Constants are traced to RTL in docs/perfsim/architecture.md. No sparse/ternary skipping.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
from .. import isa as I
from .config import Unsupported


@dataclass
class Action:
    kind: str
    reads: tuple = ()
    writes: tuple = ()
    latency: int = 0
    addr: int = 0
    size: int = 0
    write: bool = False
    port: str = ''
    value: object = None
    until: int | None = None


def pause(n):
    return Action('delay', latency=max(1,n))


def banks(reads=(), writes=(), latency=0):
    return Action('bank', tuple(reads), tuple(writes), latency)


def mem(addr, size, write=False, port='B'):
    return Action('memory', addr=addr, size=size, write=write, port=port)


def wait(ticket):
    return Action('wait', value=ticket)


def predicate(fn, reason):
    return Action('predicate', value=fn, port=reason)


def chunks(address, length, lanes):
    for c in range(0, length, lanes):
        yield tuple(range(address+c, address+min(length,c+lanes)))


def scattered(addresses, nbank, ports=1):
    """Drain scattered MXU columns in conflict-free groups; no padded rows counted as useful."""
    group, counts = [], Counter()
    for a in addresses:
        b=a%nbank
        if counts[b] == ports:
            yield tuple(group)
            group, counts = [], Counter()
        group.append(a); counts[b]+=1
    if group:
        yield tuple(group)


class Coroutine:
    def __init__(self, task, generator):
        self.task, self.gen, self.action, self.done = task, generator, None, False
        self.reply = None

    def tick(self):
        if self.done:
            return
        t, sim = self.task, self.task.sim
        # Only zero-work waits can collapse; at most one issuing action per coroutine/cycle.
        for _ in range(16):
            if self.action is None:
                try:
                    self.action = self.gen.send(self.reply)
                    self.reply = None
                except StopIteration:
                    self.done = True
                    return
            a = self.action
            if a.kind == 'delay':
                if a.until is None:
                    a.until = sim.cycle+a.latency
                if sim.cycle < a.until:
                    t.cmd.stalls['pipeline_latency'] += 1
                    return
                self.action=None
                continue
            if a.kind == 'predicate':
                if not a.value():
                    t.cmd.stalls[a.port] += 1
                    return
                self.action=None
                continue
            if a.kind == 'wait':
                if not a.value.done:
                    t.cmd.stalls['memory_completion'] += 1
                    return
                self.action=None
                continue
            if a.kind == 'memory':
                ticket = sim.memory.request(a.addr, a.size, a.write, a.port, t.cmd.id)
                if ticket is None:
                    t.cmd.stalls['memory_backpressure'] += 1
                    return
                self.reply=ticket
            elif a.kind == 'bank':
                if len(t.outputs) >= sim.arch.result_queue:
                    t.cmd.stalls['result_buffer_full'] += 1
                    return
                key = ('input',t.cmd.unit)
                if key in sim.resources:
                    t.cmd.stalls['unit_issue'] += 1
                    return
                if not sim.banks.grant(a.reads, () if a.latency else a.writes):
                    t.cmd.stalls['tmem_bank'] += 1
                    return
                sim.resources.add(key)
                t.cmd.work['tmem_reads'] += len(a.reads)
                if a.latency and a.writes:
                    t.outputs.append((sim.cycle+a.latency, a.writes))
                else:
                    t.cmd.work['tmem_writes'] += len(a.writes)
                t.cmd.work['lane_issues'] += 1
            elif a.kind == 'mxu':
                if not sim.reserve('mxu_advance'):
                    t.cmd.stalls['mxu_issue'] += 1
                    return
                t.cmd.work['mxu_advances'] += 1
                t.cmd.work['macs'] += a.size
            else:
                raise RuntimeError(a.kind)
            self.action=None
            return


class Task:
    def __init__(self, cmd):
        self.cmd, self.sim = cmd, cmd.sim
        self.outputs = deque()
        self.coroutines = []

    def tick(self):
        # An output stalled by a bank remains buffered, eventually stopping input issue.
        if self.outputs and self.outputs[0][0] <= self.sim.cycle:
            _, addresses = self.outputs[0]
            if self.sim.banks.grant(writes=addresses):
                self.outputs.popleft()
                self.cmd.work['tmem_writes'] += len(addresses)
            else:
                self.cmd.stalls['tmem_drain'] += 1
        for co in self.coroutines:
            co.tick()
        return all(co.done for co in self.coroutines) and not self.outputs


class SequentialTask(Task):
    def __init__(self, cmd, generator):
        super().__init__(cmd)
        self.coroutines=[Coroutine(self,generator(self))]


def dma(t):
    w, cfg = t.cmd.w, t.sim.cfg
    write = t.cmd.ins.op == I.ST
    # DMA has one gathered D-byte chunk in this model; the next chunk waits for it.
    start, size = w[0], 4*w[2]
    end, cursor = start+size, start
    while cursor < end:
        amount=min(end-cursor,cfg.D-cursor%cfg.D)
        target=w[1]+(cursor-start)//4
        if write:
            for xs in chunks(target,amount//4,cfg.LANES):
                yield banks(reads=xs)
            yield pause(t.sim.arch.tmem_read_latency)
        ticket=yield mem(cursor, amount, write)
        yield wait(ticket)
        if not write:
            for xs in chunks(target,amount//4,cfg.LANES):
                yield banks(writes=xs)
        cursor+=amount


def vector(t):
    w, fl, cfg, a = t.cmd.w, t.cmd.ins.flags, t.sim.cfg, t.sim.arch
    rows, cols = w[3]&65535,w[3]>>16
    f,bm=(w[5]>>16)&255,(w[5]>>24)&3
    drs,ars,brs=w[4]&65535,w[4]>>16,w[5]&65535
    slots={I.V_ADD:1,I.V_SUB:1,I.V_RSUB:1,I.V_MUL:1,I.V_OUTER:1,
           I.V_EXP2:9,I.V_EXP2SUB:9,I.V_RECIP:6,I.V_RSQRT:10,I.V_LOG2:10}.get(f,0)
    lanes=(a.vpu_composite_lanes or max(1,cfg.LANES//4)) if slots>1 else cfg.LANES
    latency=a.tmem_read_latency+1+7*slots
    if f == I.V_OUTER:
        # Capture column vectors before overwriting the in-place state.
        for cs in chunks(w[6],cols,cfg.LANES):
            yield banks(reads=cs)
        if not fl&I.F_DONE:
            for ds in chunks(w[1],1 if fl&I.F_DSCALAR else cols,cfg.LANES):
                yield banks(reads=ds)
        yield pause(a.tmem_read_latency)
    for r in range(rows):
        for c in range(0,cols,lanes):
            width=min(lanes,cols-c)
            reads=[]
            if f != I.V_FILL:
                base=w[0]+r*drs if f == I.V_OUTER else w[1]+r*ars
                reads.extend(range(base+c,base+c+width))
            if f in I.READS_B or f == I.V_OUTER:
                if bm==I.B_FULL:
                    reads.extend(range(w[2]+r*brs+c,w[2]+r*brs+c+width))
                elif bm==I.B_ROW:
                    reads.append(w[2]+r*brs)
                elif bm==I.B_COL:
                    reads.extend(range(w[2]+c,w[2]+c+width))
            writes=() if f in I.REDUCE else range(w[0]+r*drs+c,w[0]+r*drs+c+width)
            yield banks(reads,writes,latency)
            t.cmd.work['elements'] += width
        if f in I.REDUCE:
            # isum_64 partial loop then its tree; max/argmax use compare trees.
            tree=(6*4+7) if f in (I.V_RSUM,I.V_RSSQ,I.V_RDOT) else (cfg.LANES.bit_length()+4)
            yield pause(a.tmem_read_latency+tree)
            yield banks(writes=range(w[0]+r*drs,w[0]+r*drs+(2 if f==I.V_ARGMAX else 1)))


def quant(t):
    w, fl, cfg, a = t.cmd.w, t.cmd.ins.flags, t.sim.cfg, t.sim.arch
    isact=t.cmd.ins.op==I.QACT
    rows,kb=(w[1]&255,w[1]>>16) if isact else (w[3]&65535,w[3]>>16)
    stride=w[2] if isact else w[4]
    if isact and (rows*(2 if fl&I.F_DUP else 1)>cfg.act_rows or ((w[1]>>8)&255)+kb>cfg.ACT_BLOCKS):
        raise Unsupported('QACT exceeds ACT RAM capacity')
    n=kb*cfg.D
    if (n+cfg.LANES-1)//cfg.LANES > a.quant_row_buffer_chunks:
        raise Unsupported('quantizer row exceeds configured XD buffer; reread path not implemented')
    tickets=[]
    for r in range(rows):
        for offset in range(0,n,cfg.LANES):
            reads=list(range(w[0]+r*stride+offset,w[0]+r*stride+min(n,offset+cfg.LANES)))
            if isact and fl&I.F_CSCALE:
                reads+=list(range(w[3]+offset,w[3]+min(n,offset+cfg.LANES)))
            if isact and fl&I.F_RSCALE:
                reads.append(w[4]+r)
            yield banks(reads=reads)
        # Two Newton reciprocal steps and correction: 2*(LM+LA+LM)+2*(LM+LA)+LM = 30.
        yield pause(30+5+a.tmem_read_latency)
        if isact:
            for _ in range((n+cfg.LANES-1)//cfg.LANES):
                yield banks()  # buffered second pass, quantize/pack into ACT RAM
            t.cmd.work['packed_activation_bytes'] += n*(2 if fl&I.F_DUP else 1)
        else:
            nw=n//2 if fl&I.F_HALF else n
            # Port SW byte stores; coalescing is deliberately absent, worst-case RMW adapter.
            for c in range(nw):
                ticket=yield mem(w[1]+r*w[5]+c*w[6],1,True,'SW')
                tickets.append(ticket)
                if len(tickets)>=a.result_queue:
                    yield wait(tickets.pop(0))
            sb=4 if fl&I.F_ROW else 4*(1+(kb+3)//4)
            for off in range(0,sb,4):
                ticket=yield mem(w[2]+r*sb+off,4,True,'SW')
                tickets.append(ticket)
    for ticket in tickets:
        yield wait(ticket)


def collective(t):
    if t.cmd.ins.op==I.RLD:
        yield banks(reads=[t.cmd.w[0]])
        yield pause(t.sim.arch.tmem_read_latency+3)
    else:
        yield pause(1)


@dataclass
class Row:
    index: int
    cost: int
    tickets: list = field(default_factory=list)
    issued: bool = False

    def ready(self):
        return self.issued and all(t.done for t in self.tickets)


class MatrixTask(Task):
    def __init__(self, cmd):
        super().__init__(cmd)
        w,fl=cmd.w,cmd.ins.flags
        self.n,self.kb,self.k0,self.m=w[3]&65535,(w[3]>>16)&255,w[3]>>24,(w[5]>>16)&255
        self.wf=(fl>>I.WF_SHIFT)&3
        self.pair=bool(fl&I.F_PAIR)
        if self.pair and (self.wf==I.W8 or 2*self.m>self.sim.cfg.MCOLS):
            raise Unsupported('PAIR requires packed weights and 2*M <= MCOLS')
        self.rows=deque()
        self.used=0
        self.coroutines=[Coroutine(self,self.prefetch()),Coroutine(self,self.consume())]

    def prefetch(self):
        w,fl,cfg=self.cmd.w,self.cmd.ins.flags,self.sim.cfg
        # The scale-line holding register is one line, not an unbounded ideal cache.
        last_scale=None
        last_ticket=None
        for n in range(self.n):
            packed=self.kb*cfg.D if self.wf==I.W8 else (self.kb*cfg.D+1)//2
            cost=(packed+cfg.D-1)//cfg.D
            yield predicate(lambda:self.used+cost<=self.sim.arch.mxu_fifo_chunks,'mxu_fifo_full')
            row=Row(n,cost)
            self.rows.append(row); self.used+=cost
            for off in range(0,packed,cfg.D):
                row.tickets.append((yield mem(w[0]+n*w[4]+off,min(cfg.D,packed-off),False,'B')))
            if not fl&I.F_UNIT:
                base=w[1]+n*w[6]
                addresses=[base]
                if not fl&I.F_RSC:
                    offset,sz=(4,1) if self.wf==I.W8 else (16,8) if self.wf==I.WNV else (8,4)
                    addresses.extend(base+offset+(self.k0+k)*sz for k in range(self.kb))
                for addr in addresses:
                    line=addr//32*32
                    if line!=last_scale:
                        last_ticket=yield mem(line,32,False,'A')
                        last_scale=line
                    row.tickets.append(last_ticket)
            row.issued=True

    def consume(self):
        cfg,a,w,fl=self.sim.cfg,self.sim.arch,self.cmd.w,self.cmd.ins.flags
        ors=w[5]&65535
        replay=(self.m+cfg.MCOLS-1)//cfg.MCOLS
        advances=(self.kb+1)//2 if self.pair else self.kb
        latency=min(cfg.D//8,a.mxu_chain)+cfg.MCOLS+5
        el=a.mxu_epilogue_lanes or max(1,cfg.MCOLS//2)
        for n in range(self.n):
            yield predicate(lambda:self.cmd.release is not None,'activation_dependency')
            yield predicate(lambda:bool(self.rows) and self.rows[0].ready(),'mxu_weight_starvation')
            yield predicate(lambda:len(self.outputs)<min(a.result_queue,a.mxu_result_rows),'mxu_result_full')
            for rg in range(replay):
                useful=min(cfg.MCOLS,self.m-rg*cfg.MCOLS)
                for k in range(advances):
                    kb_count=min(2,self.kb-2*k) if self.pair else 1
                    yield Action('mxu',size=useful*cfg.D*kb_count)
            row=self.rows.popleft(); self.used-=row.cost
            for group in scattered([w[2]+r*ors+n for r in range(self.m)],cfg.LANES,a.tmem_write_ports_per_bank):
                reads=list(group) if fl&I.F_ACC else []
                if fl&I.F_ASCALE:
                    reads+=list(range(w[1],w[1]+self.m))
                # Shared epilogue, fp conversion/multiply and optional ACC before bank drain.
                yield banks(reads,group,latency+9+(self.m+el-1)//el+(4 if fl&I.F_ACC else 0))
        if fl&I.F_RMAX:
            yield predicate(lambda:not self.outputs,'mxu_rmax_drain')
            yield pause(4)
            for group in scattered([w[2]+self.m*ors+r for r in range(self.m)],cfg.LANES):
                yield banks(writes=group)


def make_task(cmd):
    if cmd.ins.op==I.MM:
        return MatrixTask(cmd)
    generators={I.LD:dma,I.ST:dma,I.VOP:vector,I.QACT:quant,I.QST:quant,I.RLD:collective,I.BAR:collective}
    return SequentialTask(cmd,generators[cmd.ins.op])
