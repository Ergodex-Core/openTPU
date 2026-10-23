"""Registered native ports shared by the performance model and isolated RTL co-simulation.

One elastic outgoing register per A/B/SW port; B has one additional input register.
The controller consumes offers B,A,SW, splits/RMWs using the existing MemorySystem,
and returns A/B reads in acceptance order (at most one response per port per edge).
"""
from collections import deque
from dataclasses import replace
from .memory import MemorySystem,Ticket


class PortController:
    def __init__(self,cfg):
        self.memory=MemorySystem(replace(cfg,native_ports=False))
        self.reads={'A':deque(),'B':deque()};self.writes=[]
        self.events=[]

    def step(self,cycle,period,offers):
        m=self.memory;m.begin_cycle(cycle,period)
        completions=[]
        for port,q in self.reads.items():
            if q and q[0][1].done:
                ident,t=q.popleft();completions.append((port,ident))
        pending=[]
        for port,ident,t in self.writes:
            if t.done:completions.append((port,ident))
            else:pending.append((port,ident,t))
        self.writes=pending
        accepted=[]
        for port in ('B','A','SW'):
            if port not in offers:continue
            ident,addr,size,write,owner=offers[port]
            ticket=m.request(addr,size,write,port,owner)
            if ticket is None:continue
            accepted.append(port)
            if write:self.writes.append((port,ident,ticket))
            else:self.reads[port].append((ident,ticket))
            if m.cfg.record_trace:self.events.append({'cycle':cycle,'event':'port_accept','port':port,'id':ident,'address':addr,'size':size,'write':write})
        if m.cfg.record_trace:
            self.events.extend({'cycle':cycle,'event':'port_complete','port':port,'id':ident} for port,ident in completions)
        return accepted,completions

    @property
    def read_idle(self):return not any(self.reads.values())
    @property
    def write_idle(self):return not self.writes
    @property
    def idle(self):return self.memory.idle and self.read_idle and self.write_idle


class NativeMemorySystem:
    """Core-side view; its exact controller is also imported by the RTL DPI driver."""
    def __init__(self,cfg):
        self.cfg=cfg;self.controller=PortController(cfg);self.registers={};self.tickets={}
        self.responses=[];self.next_ident=0;self.cycle=-1
        self.next_owner=1

    def __getattr__(self,key):return getattr(self.controller.memory,key)
    @property
    def run_peak(self):return self.controller.memory.run_peak
    @run_peak.setter
    def run_peak(self,value):self.controller.memory.run_peak=value

    def begin_cycle(self,cycle,period):
        accepted,complete=self.controller.step(cycle,period,self.registers)
        self.cycle=cycle
        for port in accepted:del self.registers[port]
        for port,ident in complete:
            t,write=self.tickets[ident]
            self.responses.append((cycle+(1 if port=='B' and not write else 0),ident))
        future=[]
        for due,ident in self.responses:
            if due<=cycle:
                t,_=self.tickets.pop(ident);t.done=True;t.pending=0;t.complete_cycle=cycle
            else:future.append((due,ident))
        self.responses=future

    def can_request(self,addr,size,write,port):return port not in self.registers
    @property
    def write_idle(self):return not any(write for ticket,write in self.tickets.values())

    def request(self,addr,size,write,port,owner):
        if size<=0 or addr<0 or port not in ('A','B','SW'):raise ValueError('invalid native request')
        if port in self.registers:return None
        ident=self.next_ident;self.next_ident+=1
        ticket=Ticket(1,accepted_cycle=self.cycle);self.tickets[ident]=(ticket,write)
        self.registers[port]=(ident,addr,size,write,owner)
        return ticket

    def request_many(self,requests,owner):
        ports=[r[3] for r in requests]
        if len(ports)!=len(set(ports)) or any(p in self.registers for p in ports):return None
        return [self.request(*r,owner) for r in requests]

    @property
    def idle(self):return not self.registers and not self.tickets and self.controller.idle
    def close(self):self.controller.memory.close()


def make_memory(cfg):
    return NativeMemorySystem(cfg) if cfg.native_ports else MemorySystem(cfg)
