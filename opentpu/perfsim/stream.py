"""DMA/SE stream timing for the baseline eight-lane hardware descriptor subset.

Control follows otpu_dma's bounded 64-chunk runs and registered pe; arithmetic commits
through the unchanged ISA only at completion. No latency depends on numeric values.
"""
from collections import deque
from .. import isa as I, waits as W
from .config import Unsupported


class StreamTask:
    def __init__(self,cmd):
        self.cmd,self.sim=cmd,cmd.sim
        self.outputs=deque();self.coroutines=[]
        self.phase='descriptor' if cmd.ins.op==I.STREAM else 'setup'
        self.due=self.sim.cycle+2;self.queue=deque();self.writes=[]
        self.rr=False;self.rc=0;self.wr=False;self.og=0;self.pos=0
        self.issued=self.written=self.taken=self.ycount=self.ocount=0
        self.pe=deque([(False,None)]*3);self.clock=0;self.y=deque();self.o=deque()
        self.request_cycle=None;self.fill=deque();self.out_index=0;self.done_at=None

    def accepts_next(self):return False

    def setup(self):
        w=self.cmd.w;cfg=self.sim.cfg
        if self.cmd.ins.op==I.DSTEP:
            src=dst=w[0];vec=w[1];x=w[2];rows=w[3]&65535;cols=w[3]>>16;k=w[4];out=w[5];ks=w[6]
            hw=I.stream_hw_cfg(I.gdn_desc(rows,cols),0)
        else:
            desc=I.StreamDesc.decode(self.sim.state.tmem[(w[0]&65535):(w[0]&65535)+8])
            hw=I.stream_hw_cfg(desc,self.cmd.ins.flags)
            src,dst,vec,x,k,out=w[1:];ks=w[0]>>16;rows,cols=desc.rows,desc.cols
        if hw is None or src%cfg.D or dst%cfg.D or (src!=dst and src<dst+rows*cols*4 and dst<src+rows*cols*4):
            raise Unsupported('STREAM requires the baseline DRAM descriptor subset and disjoint or identical aligned ranges')
        if cfg.LANES!=8 or cfg.D<32:raise Unsupported('stream engine requires eight lanes')
        if max(src,dst)+rows*cols*4>cfg.DRAM_BYTES:raise Unsupported('stream DRAM range exceeds capacity')
        self.src,self.dst,self.out,self.rows,self.cols=src,dst,out,rows,cols
        self.spc=cfg.D//32;self.ns=max(16,cols//8);self.pad=cols==64
        self.nseg=rows*self.ns;self.nchunks=rows*cols*4//cfg.D;self.zero=bool(self.cmd.ins.flags&I.F_DZERO)
        self.qen=hw['q_en'];self.ng=128*self.spc
        starts=([vec] if self.qen else [])+[vec+cols]
        if hw['a_en'] and hw['a_sel']:starts.append(vec+2*cols)
        if hw['g_src']==1:starts.append(vec+3*cols)
        for a in starts:
            for j in range(0,cols,8):self.fill.append(tuple(range(a+j,a+j+8)))
        for j in range(0,rows,8):self.fill.append(tuple(range(x+j,x+min(j+8,rows))))
        self.fill.extend([(k,),(k+ks,)])
        self.request_cycle=self.sim.cycle;self.sim.stream_request=self
        self.phase='grant';self.due=self.sim.cycle+3

    def tick(self):
        s=self.sim;c=s.cycle
        if self.done_at is not None:return c>=self.done_at
        if self.outputs and self.outputs[0][0]<=c:
            _,xs=self.outputs[0]
            if not s.banks.grant(writes=xs):raise Unsupported('unexpected higher-priority TMEM writer during DMA stream output')
            self.outputs.popleft();self.cmd.work['tmem_writes']+=len(xs)
        if self.phase=='descriptor':
            if c>=self.due:
                xs=tuple(range(self.cmd.w[0]&65535,(self.cmd.w[0]&65535)+8))
                if s.banks.grant(reads=xs):self.phase='setup';self.due=c+3;self.cmd.work['tmem_reads']+=8
            return False
        if self.phase=='setup':
            if c>=self.due:self.setup()
            return False
        # Old registered SE outputs are observed before this edge's state changes.
        enable,meta=self.pe.popleft();yv=ov=False;ym=None
        if enable:
            if self.y and self.y[0][0]==self.clock:_,ym=self.y.popleft();yv=True
            if self.o and self.o[0]==self.clock:self.o.popleft();ov=True
            if meta is not None:
                self.y.append((self.clock+105,meta)) # X register + T_U(97) + U(7)
                if self.qen and meta%self.ns==self.ns-1:self.o.append(self.clock+140) # Y + Q(6) + QD(1) + tree(28)
            self.clock+=1
        keep=yv and not(self.pad and ym%16>=8)
        flushed=self.ycount==self.nseg and (not self.qen or self.ocount==self.rows)
        oldog=self.og;oldrr=self.rr;oldwr=self.wr;oldleft=self.nseg-self.taken
        room=oldog+int(yv)+int(self.pe[-1][0])<=self.ng-2
        take=self.phase=='run' and self.taken<self.nseg and room and (self.zero or (self.pad and self.taken%16>=8) or bool(self.queue) and self.queue[0].done and self.queue[0].complete_cycle<s.memory_offset+c)
        drain=self.phase=='run' and self.taken==self.nseg and not flushed and room
        self.pe.append((bool(take or drain),self.taken if take else None))
        pop=take and not self.zero and not(self.pad and self.taken%16>=8) and self.pos==self.spc-1
        accepted_read=accepted_write=False
        if oldwr:
            ticket=s.memory.request(self.dst+self.written*s.cfg.D,s.cfg.D,True,'B',self.cmd.id)
            if ticket is not None:self.writes.append(ticket);self.written+=1;accepted_write=True
            else:self.cmd.stalls['memory_backpressure']+=1
        elif not self.zero and self.issued<self.nchunks and len(self.queue)<128 and (oldrr or len(self.queue)<=64):
            ticket=s.memory.request(self.src+self.issued*s.cfg.D,s.cfg.D,False,'B',self.cmd.id)
            if ticket is not None:self.queue.append(ticket);self.issued+=1;accepted_read=True
            else:self.cmd.stalls['memory_backpressure']+=1
        if pop:self.queue.popleft()
        if take:
            if not self.zero and not(self.pad and self.taken%16>=8):self.pos=(self.pos+1)%self.spc
            self.taken+=1;self.cmd.work['stream_input_segments']+=1
        elif self.phase=='run' and self.taken<self.nseg:self.cmd.stalls['stream_input_or_gather']+=1
        if yv:self.ycount+=1
        if ov:self.ocount+=1
        self.og+=int(keep)-int(accepted_write)*self.spc
        rr_next=oldrr and not(accepted_read and (self.rc==63 or self.issued==self.nchunks))
        self.wr=self.og>=self.spc and (oldwr or (not rr_next and (self.og>=64*self.spc or oldleft==0)))
        self.rr=rr_next or (accepted_read and not oldrr and self.issued<self.nchunks)
        if accepted_read:self.rc=(self.rc+1)%64
        if self.phase=='grant':
            active=any(x.unit==W.U_VPU and x.task is not None for x in s.window)
            if not active and c>=self.due:self.phase='fill'
            elif active:self.due=c+1;self.cmd.stalls['stream_vpu_drain']+=1
        elif self.phase=='fill':
            if self.fill:
                xs=self.fill[0]
                if s.banks.grant(reads=xs):self.fill.popleft();self.cmd.work['tmem_reads']+=len(xs)
                else:raise Unsupported('stream fill denied by unsupported TMEM read topology')
            else:self.phase='fill_tail';self.due=c+1
        elif self.phase=='fill_tail':
            if c>=self.due:self.phase='run'
        elif self.phase=='run' and flushed:
            if s.stream_request is self:s.stream_request=None;s.stream_release=c+2
            if oldog==0:self.phase='out' if self.qen else 'ack'
        elif self.phase=='out':
            xs=tuple(range(self.out+self.out_index,self.out+min(self.out_index+8,self.rows)))
            self.outputs.append((c+1,xs));self.out_index+=len(xs)
            if self.out_index==self.rows:self.phase='ack'
        elif self.phase=='ack' and s.memory.write_idle and not self.outputs:self.done_at=c+1
        if getattr(s,'capture_micro',False):s.micro_events.append({'cycle':c,'stream':self.cmd.id,'phase':self.phase,'take':bool(take),'y':yv,'o':ov,'og':self.og,'left':self.nseg-self.taken,'wr':self.wr,'rr':self.rr})
        return False
