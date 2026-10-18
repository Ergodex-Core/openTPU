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
        for _ in range(1_000_000):
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
            if a.kind == 'memory_pair':
                tickets=sim.memory.request_many(a.value,t.cmd.id)
                if tickets is None:
                    t.cmd.stalls['memory_backpressure']+=1
                    return
                self.reply=tickets
            elif a.kind == 'memory':
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
            elif a.kind == 'quant_pack':
                if not sim.reserve('quant_pack'): return
                t.cmd.work['packed_activation_bytes']+=a.size
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
        raise RuntimeError('too many zero-work coroutine transitions; refusing artificial timing')


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


class DMATask(Task):
    def __init__(self,cmd):
        super().__init__(cmd)
        self.rows=deque(); self.produced=False; self.drained=None
        self.is_store=cmd.ins.op==I.ST
        self.coroutines=[Coroutine(self,self.produce()),Coroutine(self,self.consume())]

    def segments(self):
        w,cfg=self.cmd.w,self.sim.cfg
        first,end=w[0],w[0]+4*w[2]
        width=min(cfg.D//4,cfg.LANES)*4
        for base in range(first//cfg.D*cfg.D,(end+cfg.D-1)//cfg.D*cfg.D,cfg.D):
            groups=[]
            for seg in range(base,base+cfg.D,width):
                lo,hi=max(seg,first),min(seg+width,end)
                if hi>lo: groups.append(tuple(range(w[1]+(lo-first)//4,w[1]+(hi-first)//4)))
            yield base,max(base,first),min(base+cfg.D,end)-max(base,first),groups

    def accepts_next(self):
        # LD busy drops on ld_last, before done passes through the sequencer.
        return not self.is_store and self.drained is not None and self.sim.cycle>self.drained

    def produce(self):
        yield pause(2)  # sequencer start register and DMA command setup
        for base,addr,amount,groups in self.segments():
            yield predicate(lambda:len(self.rows)<self.sim.arch.dma_buffer_chunks,'dma_buffer_full')
            if self.is_store:
                for xs in groups: yield banks(reads=xs)
                # pa0 -> pa -> st_pend, then the gathered chunk enters lb.
                self.rows.append((self.sim.cycle-1+3,addr,amount,groups,None))
            else:
                ticket=yield mem(base,self.sim.cfg.D,False,'B')
                self.rows.append((0,addr,amount,groups,ticket))
        self.produced=True

    def consume(self):
        if self.is_store:
            tickets=[]; last_request=None
            while not self.produced or self.rows:
                def ready():
                    n=sum(r[0]<=self.sim.cycle for r in self.rows)
                    return n>=self.sim.arch.dma_store_run or (self.produced and n==len(self.rows))
                yield predicate(ready,'dma_store_gather')
                if not self.rows: break
                boundary=self.sim.cycle-(1 if last_request is not None and self.sim.cycle==last_request+1 else 0)
                n=sum(r[0]<=boundary for r in self.rows)
                # st_fin/occ observed -> st_wq registered -> first B request.
                yield pause(1 if last_request is not None and self.sim.cycle==last_request+1 else 2)
                for _ in range(n):
                    _,addr,amount,_,_=self.rows.popleft()
                    ticket=yield mem(addr,amount,True,'B'); tickets.append(ticket)
                    last_request=self.sim.cycle-1
            for ticket in tickets: yield wait(ticket)
            yield predicate(lambda:self.sim.memory.write_idle,'global_write_drain')
            yield pause(1)  # ackw/wr_idle -> done -> sequencer observation
        else:
            while not self.produced or self.rows:
                yield predicate(lambda:(self.produced and not self.rows) or (bool(self.rows) and self.rows[0][4].done),'dma_read_response')
                if not self.rows: break
                _,_,_,groups,_=self.rows[0]
                for xs in groups: yield banks(writes=xs,latency=3)  # response visibility, lb read, registered TMEM write
                self.rows.popleft()

    def tick(self):
        done=super().tick()
        if self.is_store: return done
        if self.drained is None and all(co.done for co in self.coroutines) and (not self.outputs or self.outputs[-1][0]<=self.sim.cycle+1): self.drained=self.sim.cycle
        return self.drained is not None and self.sim.cycle>=self.drained+2


def vector_slots(f):
    return {I.V_ADD:1,I.V_SUB:1,I.V_RSUB:1,I.V_MUL:1,I.V_OUTER:1,
            I.V_EXP2:9,I.V_EXP2SUB:9,I.V_RECIP:6,I.V_RSQRT:10,I.V_LOG2:10}.get(f,0)


def reduction_tree_latency(lanes):
    # otpu_vtree tree_i / RD, including the LANES=8 two-adder lane fold.
    rl=64//lanes; lw=lanes.bit_length()-1
    issue=rl//2
    for j in range(1,rl.bit_length()-1):
        h=rl>>(j+1)
        issue=issue-h+rl*((4+2*h+rl-1)//rl)
    return issue+4*(1+lw)-(rl-1)+(2 if lanes==8 else 0)


class VectorTask(Task):
    def __init__(self,cmd):
        super().__init__(cmd)
        self.begun=False; self.begin_cycle=None; self.issue_finished=False; self.payload_done=None
        self.function=(cmd.w[5]>>16)&255
        self.slots=vector_slots(self.function)
        self.coroutines=[Coroutine(self,vector(self))]

    def tick(self):
        if 'vpu_preflight' not in self.sim.resources:
            self.sim.resources.add('vpu_preflight')
            active=[c.task for c in self.sim.window if c.unit==self.cmd.unit and c.task is not None]
            # A denied registered write freezes all instructions, including an older
            # command whose done is pending but whose own last write already left.
            denied=any(t.outputs and t.outputs[0][0]<=self.sim.cycle and
                       not self.sim.banks.grant(writes=t.outputs[0][1],probe=True) for t in active)
            if denied:
                for other in active:
                    other.outputs=deque((due+1,xs) for due,xs in other.outputs)
                    if other.payload_done is not None:other.payload_done+=1
                    for co in other.coroutines:
                        if co.action is not None and co.action.until is not None:co.action.until+=1
                self.sim.vpu_loop_busy={v+1 if v>=self.sim.cycle else v for v in self.sim.vpu_loop_busy}
                self.sim.resources.add('vpu_frozen')
        if 'vpu_frozen' in self.sim.resources:
            self.cmd.stalls['vpu_pipeline_frozen']+=1
            return False
        if not self.begun:
            if self.sim.cycle < self.cmd.start+2: return False
            prior=[c.task for c in self.sim.window if c.unit==self.cmd.unit and c.start is not None
                   and c.start < self.cmd.start and c.task is not None]
            if any(not t.begun or not t.issue_finished for t in prior): return False
            prior.sort(key=lambda t:t.cmd.start)
            live=[t for t in prior if t.payload_done is None or t.payload_done>=self.sim.cycle]
            if any(t.function in I.REDUCE for t in live): return False
            if live and (self.function in I.REDUCE or self.slots < live[-1].slots): return False
            self.begun=True; self.begin_cycle=self.sim.cycle
        drained=super().tick()
        # cw_en precedes the registered TMEM tw_en by one cycle. ew_n clears at
        # the arithmetic tail; sequencer completion still waits through the output register.
        if self.payload_done is None and all(co.done for co in self.coroutines) and (not self.outputs or self.outputs[-1][0]<=self.sim.cycle+1):
            self.payload_done=self.sim.cycle
        return self.payload_done is not None and self.sim.cycle>=self.payload_done+2


def vector(t):
    w,fl,cfg,a=t.cmd.w,t.cmd.ins.flags,t.sim.cfg,t.sim.arch
    rows,cols=w[3]&65535,w[3]>>16
    f,bm=(w[5]>>16)&255,(w[5]>>24)&3
    drs,ars,brs=w[4]&65535,w[4]>>16,w[5]&65535
    slots=vector_slots(f)
    c8=cfg.LANES==8 and cfg.D>=32  # otpu_slice HAS_DSTEP defaults on, regardless of op usage
    composite=slots>1
    lanes=cfg.LANES if c8 or not composite else (a.vpu_composite_lanes or max(1,cfg.LANES//4))
    passes=(slots+2)//3
    latency=3+(25*passes if c8 and composite else 7*slots+(3 if f in (I.V_EXP2,I.V_EXP2SUB,I.V_LOG2) else 0))
    summed=f in (I.V_RSUM,I.V_RSSQ,I.V_RDOT)
    if summed: latency=3+(7 if 64//cfg.LANES>=8 else 6)+reduction_tree_latency(cfg.LANES)
    elif f in I.REDUCE: latency=3+(cfg.LANES.bit_length()-1)+(4 if f==I.V_ARGMAX else 0)
    count=(cols+lanes-1)//lanes
    if summed:
        rl=64//cfg.LANES; count=(count+rl-1)//rl*rl
    buffered=f==I.V_RDOT and rows<=256 and drs==1
    yield pause(1)  # registered begin -> first chunk
    if f==I.V_OUTER:
        for c in range(max(2,count)):
            off=c*cfg.LANES; width=max(0,min(cfg.LANES,cols-off))
            reads=list(range(w[6]+off,w[6]+off+width))
            if not fl&I.F_DONE:
                reads+=([w[1]] if fl&I.F_DSCALAR and width else list(range(w[1]+off,w[1]+off+width)))
            yield banks(reads=reads)
    for r in range(rows):
        for chunk in range(count):
            c=chunk*lanes; width=max(0,min(lanes,cols-c)); reads=[]
            if f!=I.V_FILL:
                base=w[0]+r*drs if f==I.V_OUTER else w[1]+r*ars
                reads.extend(range(base+c,base+c+width))
            if f in I.READS_B or f==I.V_OUTER:
                if bm==I.B_FULL: reads.extend(range(w[2]+r*brs+c,w[2]+r*brs+c+width))
                elif bm==I.B_ROW and width: reads.append(w[2]+r*brs)
                elif bm==I.B_COL: reads.extend(range(w[2]+c,w[2]+c+width))
            writes=range(w[0]+r*drs+c,w[0]+r*drs+c+width)
            if f in I.REDUCE:
                writes=range(w[0]+r*drs,w[0]+r*drs+(2 if f==I.V_ARGMAX else 1)) if chunk==count-1 and not buffered else ()
            if c8 and composite:
                yield predicate(lambda:t.sim.cycle not in t.sim.vpu_loop_busy,'composite_recirculation')
            yield banks(reads,writes,latency+1)  # registered tw_en after the arithmetic tail
            issued=t.sim.cycle-1
            if c8 and composite:
                for p in range(1,passes): t.sim.vpu_loop_busy.add(issued+25*p)
            t.cmd.work['elements']+=width
            if buffered and r==rows-1 and chunk==count-1:
                for k,xs in enumerate(chunks(w[0],rows,cfg.LANES)):
                    t.outputs.append((issued+latency+2+k,xs))
    t.issue_finished=True


class QuantActTask(Task):
    def __init__(self,cmd):
        super().__init__(cmd)
        w,fl,cfg=cmd.w,cmd.ins.flags,self.sim.cfg
        self.nr,self.kb=w[1]&255,w[1]>>16
        if self.nr*(2 if fl&I.F_DUP else 1)>cfg.act_rows or ((w[1]>>8)&255)+self.kb>cfg.ACT_BLOCKS:
            raise Unsupported('QACT exceeds ACT RAM capacity')
        self.chunks=self.kb*cfg.D//cfg.LANES
        if self.chunks>self.sim.arch.quant_row_buffer_chunks:
            raise Unsupported('unbuffered QACT reread path not implemented')
        self.rows=deque(); self.credits=0; self.read_rows=0; self.packed_rows=0
        self.coroutines=[Coroutine(self,self.scan()),Coroutine(self,self.pack())]

    def scan(self):
        w,fl,cfg=self.cmd.w,self.cmd.ins.flags,self.sim.cfg
        yield pause(2)
        for row in range(self.nr):
            for c in range(self.chunks):
                yield predicate(lambda:self.read_rows-self.packed_rows<4 and self.credits<self.sim.arch.quant_row_buffer_chunks,'quant_row_credit')
                off=c*cfg.LANES
                reads=list(range(w[0]+row*w[2]+off,w[0]+row*w[2]+off+cfg.LANES))
                if fl&I.F_CSCALE: reads+=list(range(w[3]+off,w[3]+off+cfg.LANES))
                if fl&I.F_RSCALE: reads.append(w[4]+row)
                yield banks(reads=reads)
                self.credits+=1
            # Two read stages, QL=5 prescale, 2 max stages, qscale LAT=30 plus
            # input/valid stages (32), then the bank's registered scale-valid bit.
            last_read=self.sim.cycle-1
            self.rows.append(last_read+2+5+2+32+1)
            self.read_rows+=1

    def pack(self):
        for row in range(self.nr):
            yield predicate(lambda:bool(self.rows) and self.rows[0]<=self.sim.cycle,'quant_scale_pipeline')
            yield pause(1)  # p1a row-start register
            for _ in range(self.chunks):
                yield Action('quant_pack',size=self.sim.cfg.LANES*(2 if self.cmd.ins.flags&I.F_DUP else 1))
                self.credits-=1
            self.rows.popleft(); self.packed_rows+=1
        # mb0/mb1/mb2 (2), writer select (1), multiply (2), q8 (2), done_w/done/seq (3).
        yield pause(9)  # coroutine resumes one cycle after the final pop


class QuantStoreTask(Task):
    """Unbuffered two-pass QST: scale writes precede the second TMEM read pass."""
    def __init__(self,cmd):
        super().__init__(cmd)
        self.writes=deque(); self.produced=False; self.tickets=[]; self.ack_cycle=None
        self.coroutines=[Coroutine(self,self.scan())]

    def accepts_next(self):
        return self.ack_cycle is not None and self.sim.cycle>self.ack_cycle

    def scan(self):
        w,fl,cfg=self.cmd.w,self.cmd.ins.flags,self.sim.cfg
        rows,kb=w[3]&65535,w[3]>>16; n=kb*cfg.D
        sw=4 if fl&I.F_ROW else 4*(1+(kb+3)//4)
        yield pause(2)
        for row in range(rows):
            src=w[0]+row*w[4]; dst=w[1]+row*w[5]
            for xs in chunks(src,n,cfg.LANES): yield banks(reads=xs)
            # Last pass-0 read -> read2 + QL5 + max2 + qscale32 + scale phase1.
            yield pause(41)  # coroutine resumes one cycle after the last read
            for off in range(0,sw,4):
                ticket=yield mem(w[2]+row*sw+off,4,True,'SW'); self.tickets.append(ticket)
            nw=n//2 if fl&I.F_ROW and fl&I.F_HALF else n
            width=4 if w[6]==1 and dst%4==0 else 1
            for off in range(0,nw,width):
                yield banks(reads=range(src+off,src+off+min(width,nw-off)))
                # Unbuffered writer: read2 + QL5 + selection1 + multiply2 + q8two.
                self.writes.append((self.sim.cycle-1+12,dst+off*w[6],min(width,nw-off)))
        self.produced=True

    def tick(self):
        if self.writes and self.writes[0][0]<=self.sim.cycle:
            _,addr,size=self.writes[0]
            if not self.sim.memory.can_request(addr,size,True,'SW'):
                # otpu_quant en/gnt freezes reader, scale and writer pipelines together.
                self.writes=deque((due+1,a,n) for due,a,n in self.writes)
                for co in self.coroutines:
                    if co.action is not None and co.action.until is not None:co.action.until+=1
                self.cmd.stalls['quant_output_frozen']+=1
                return False
            ticket=self.sim.memory.request(addr,size,True,'SW',self.cmd.id)
            assert ticket is not None
            self.tickets.append(ticket);self.writes.popleft()
        for co in self.coroutines:co.tick()
        self.cmd.work['peak_qst_pipeline']=max(self.cmd.work['peak_qst_pipeline'],len(self.writes))
        if self.produced and not self.writes and all(t.done for t in self.tickets) and self.sim.memory.write_idle:
            if self.ack_cycle is None:self.ack_cycle=self.sim.cycle
        return self.ack_cycle is not None and self.sim.cycle>=self.ack_cycle+2


def collective(t):
    if t.cmd.ins.op==I.RLD:
        yield pause(2)  # command capture and rl_pend/rl_pq before TMEM read
        yield banks(reads=[t.cmd.w[0]])
        yield pause(t.sim.arch.tmem_read_latency+3)
    else:
        yield pause(3)  # coll_req register, singleton acknowledge, sequencer observation


@dataclass
class Row:
    index: int
    cost: int
    tickets: list = field(default_factory=list)
    issued: bool = False

    def ready(self,cycle):
        return self.issued and all(t.done and cycle>=t.complete_cycle+delay for t,delay in self.tickets)


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
        self.results=deque(); self.compute_done=False; self.prefetch_done=False; self.prefetch_end=None
        self.drained=None
        self.coroutines=[Coroutine(self,self.prefetch()),Coroutine(self,self.consume()),Coroutine(self,self.drain())]

    def tick(self):
        if getattr(self.sim,'capture_micro',False):
            self.sim.micro_events.append({'cycle':self.sim.cycle,'id':self.cmd.id,'out':self.cmd.w[2],
             'output':list(self.outputs[0]) if self.outputs else None,
             'row_ready':bool(self.rows) and self.rows[0].ready(self.sim.memory_offset+self.sim.cycle),
             'compute_action':self.coroutines[1].action.kind if self.coroutines[1].action else None,
             'result':self.results[0] if self.results else None,
             'drain_action':self.coroutines[2].action.kind if self.coroutines[2].action else None})
        if self.outputs and self.outputs[0][0]<=self.sim.cycle and not self.sim.banks.grant(writes=self.outputs[0][1],probe=True):
            # t_gnt freezes the epilogue and FIFO drain, while the array and issuer continue.
            self.outputs=deque((due+1,xs) for due,xs in self.outputs)
            co=self.coroutines[2]
            if co.action is not None and co.action.until is not None:co.action.until+=1
            self.cmd.stalls['mxu_epilogue_frozen']+=1
            for co in self.coroutines[:2]:co.tick()
            return False
        return super().tick()

    def prefetch(self):
        w,fl,cfg=self.cmd.w,self.cmd.ins.flags,self.sim.cfg
        yield pause(3)  # sequencer start pulse -> command latch -> i_ini address setup
        sw_from_previous=False
        for n in range(self.n):
            packed=self.kb*cfg.D if self.wf==I.W8 else (self.kb*cfg.D+1)//2
            cost=(packed+cfg.D-1)//cfg.D
            yield predicate(lambda:self.used+cost<=self.sim.arch.mxu_fifo_chunks,'mxu_fifo_full')
            row=Row(n,cost); self.rows.append(row); self.used+=cost
            base=w[1]+n*w[6]
            offset,sz=(4,1) if self.wf==I.W8 else (16,8) if self.wf==I.WNV else (8,4)
            first_scale=base+offset+self.k0*sz
            if not fl&I.F_UNIT and not fl&I.F_RSC and not sw_from_previous and base//32!=first_scale//32:
                row.tickets.append(((yield mem(base//32*32,32,False,'A')),2 if self.sim.memcfg.native_ports else 1))
            for k in range(0,self.kb,2 if self.pair else 1):
                need_b=self.wf==I.W8 or self.pair or k%2==0
                requests=[]
                if need_b:
                    off=k*cfg.D if self.wf==I.W8 else k*cfg.D//2
                    requests.append((w[0]+n*w[4]+off,cfg.D,False,'B'))
                scale=base if fl&I.F_RSC else base+offset+(self.k0+k)*sz
                if not fl&I.F_UNIT and (not fl&I.F_RSC or need_b):
                    requests.append((scale//32*32,32,False,'A'))
                if requests:
                    tickets=yield Action('memory_pair',value=requests)
                    # B response -> FIFO count; A additionally registers line/tag before scale count.
                    row.tickets.extend((ticket,2 if request[3]=='A' and self.sim.memcfg.native_ports else 1) for ticket,request in zip(tickets,requests))
                else: yield pause(1)
            sw_from_previous=scale//32==(base+w[6])//32
            row.issued=True
        self.prefetch_done=True; self.prefetch_end=self.sim.cycle

    def consume(self):
        cfg,a,w,fl=self.sim.cfg,self.sim.arch,self.cmd.w,self.cmd.ins.flags
        ors=w[5]&65535
        replay=(self.m+cfg.MCOLS-1)//cfg.MCOLS
        advances=(self.kb+1)//2 if self.pair else self.kb
        latency=min(cfg.D//8,a.mxu_chain)+cfg.MCOLS+5
        el=a.mxu_epilogue_lanes or max(1,cfg.MCOLS//2)
        yield predicate(lambda:self.cmd.release is not None,'activation_dependency')
        yield pause(2)  # release register and active-head/ACT read setup
        older=[c.task for c in self.sim.window if c.unit==self.cmd.unit and c.id<self.cmd.id and c.task is not None]
        if older:
            yield predicate(lambda:all(t.compute_done for t in older),'mxu_pop_head')
            yield pause(1)  # c_left==0 -> pn/pop_q register before the next command advances
        for n in range(self.n):
            yield predicate(lambda:bool(self.rows) and self.rows[0].ready(self.sim.memory_offset+self.sim.cycle),'mxu_weight_starvation')
            yield predicate(lambda:len(self.results)<a.mxu_result_rows,'mxu_result_full')
            for rg in range(replay):
                useful=min(cfg.MCOLS,self.m-rg*cfg.MCOLS)
                for k in range(advances):
                    kb_count=min(2,self.kb-2*k) if self.pair else 1
                    yield Action('mxu',size=useful*cfg.D*kb_count)
                # S0 -> LROW -> row register -> registered FIFO head.
                self.results.append((self.sim.cycle-1+latency+3,n,rg,useful))
            row=self.rows.popleft(); self.used-=row.cost
        self.compute_done=True

    def drain(self):
        cfg,a,w,fl=self.sim.cfg,self.sim.arch,self.cmd.w,self.cmd.ins.flags
        ors=w[5]&65535; el=a.mxu_epilogue_lanes or max(1,cfg.MCOLS//2)
        older=[c for c in self.sim.window if c.unit==self.cmd.unit and c.id<self.cmd.id]
        yield predicate(lambda:all(c.end is not None for c in older),'mxu_drain_head')
        while not self.compute_done or self.results:
            yield predicate(lambda:self.compute_done and not self.results or bool(self.results) and self.results[0][0]<=self.sim.cycle,'mxu_array_fill')
            if not self.results: break
            _,n,rg,useful=self.results.popleft()
            for base in range(0,useful,el):
                addresses=[w[2]+(rg*cfg.MCOLS+r)*ors+n for r in range(base,min(useful,base+el))]
                for group in scattered(addresses,cfg.LANES,a.tmem_write_ports_per_bank):
                    reads=list(group) if fl&I.F_ACC else []
                    if fl&I.F_ASCALE: reads+=list(range(w[1],w[1]+self.m))
                    yield banks(reads,group,9+(9 if fl&I.F_ACC else 0))
        yield predicate(lambda:not self.outputs,'mxu_epilogue_drain')
        if fl&I.F_RMAX:
            yield pause(2)  # registered rx compare followed by mx_q
            for group in scattered([w[2]+self.m*ors+r for r in range(self.m)],cfg.LANES):
                yield banks(writes=group)
        yield pause(2)  # c_drained -> done -> sequencer completion


def make_task(cmd):
    if cmd.ins.op in (I.LD,I.ST):
        return DMATask(cmd)
    if cmd.ins.op==I.QACT:
        return QuantActTask(cmd)
    if cmd.ins.op==I.QST:
        return QuantStoreTask(cmd)
    if cmd.ins.op==I.MM:
        return MatrixTask(cmd)
    if cmd.ins.op==I.VOP:
        return VectorTask(cmd)
    generators={I.RLD:collective,I.BAR:collective}
    return SequentialTask(cmd,generators[cmd.ins.op])
