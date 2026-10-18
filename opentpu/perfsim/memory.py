"""Closed-loop memory: bounded requests, transactional retry, callbacks and write drain.

A logical port request may span many physical transactions. Partial writes perform a
read/modify/write transaction when configured. No future request trace is precomputed.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
import ctypes as C
import heapq
from pathlib import Path


class SimpleMemory:
    """Deterministic test memory; deliberately not labelled DDR/HBM."""
    def __init__(self, cfg):
        self.cfg, self.tick_count = cfg, 0
        self.pending, self.depth, self.accepted = [], Counter(), set()

    def send(self, tag, addr, write):
        ch = (addr // self.cfg.transaction_bytes) % self.cfg.simple_channels
        if ch in self.accepted or self.depth[ch] >= self.cfg.simple_queue_per_channel:
            return False
        self.accepted.add(ch)
        self.depth[ch] += 1
        heapq.heappush(self.pending, (self.tick_count + self.cfg.simple_latency_ticks, tag, ch))
        return True

    def tick(self):
        self.tick_count += 1
        self.accepted.clear()
        done = []
        while self.pending and self.pending[0][0] <= self.tick_count:
            _, tag, ch = heapq.heappop(self.pending)
            self.depth[ch] -= 1
            done.append(tag)
        return done

    def close(self):
        pass


class RamulatorMemory:
    """ctypes bridge to pinned Ramulator 2.0a, with write completion notification patch."""
    def __init__(self, cfg):
        if not Path(cfg.bridge).is_file() or not Path(cfg.ramulator_config).is_file():
            raise FileNotFoundError('Ramulator bridge/config missing; refusing simple-memory substitution')
        self.lib = C.CDLL(str(Path(cfg.bridge).resolve()))
        signatures = {
            'otpu_create': ([C.c_char_p], C.c_void_p),
            'otpu_error': ([], C.c_char_p),
            'otpu_send': ([C.c_void_p, C.c_uint64, C.c_uint64, C.c_int], C.c_int),
            'otpu_tick': ([C.c_void_p], C.c_int),
            'otpu_poll': ([C.c_void_p, C.POINTER(C.c_uint64)], C.c_int),
            'otpu_tck_ps': ([C.c_void_p], C.c_int),
            'otpu_tx_bytes': ([C.c_void_p], C.c_int),
            'otpu_destroy': ([C.c_void_p], None),
        }
        for name, (args, result) in signatures.items():
            f = getattr(self.lib, name)
            f.argtypes, f.restype = args, result
        self.handle = self.lib.otpu_create(str(Path(cfg.ramulator_config).resolve()).encode())
        if not self.handle:
            raise RuntimeError(self.lib.otpu_error().decode())
        period = self.lib.otpu_tck_ps(self.handle)
        size = self.lib.otpu_tx_bytes(self.handle)
        if (period, size) != (cfg.memory_period_ps, cfg.transaction_bytes):
            self.close()
            raise ValueError(f'Ramulator tCK/transaction mismatch: library {period} ps/{size} B, config {cfg.memory_period_ps} ps/{cfg.transaction_bytes} B')

    def send(self, tag, addr, write):
        r = self.lib.otpu_send(self.handle, tag, addr, int(write))
        if r < 0:
            raise RuntimeError(self.lib.otpu_error().decode())
        return bool(r)

    def tick(self):
        if self.lib.otpu_tick(self.handle) < 0:
            raise RuntimeError(self.lib.otpu_error().decode())
        done, tag = [], C.c_uint64()
        while self.lib.otpu_poll(self.handle, C.byref(tag)):
            done.append(tag.value)
        return done

    def close(self):
        if self.handle:
            self.lib.otpu_destroy(self.handle)
            self.handle = None


@dataclass
class Ticket:
    pending: int = 0
    done: bool = False
    accepted_cycle: int = -1
    complete_cycle: int = -1


@dataclass
class Transaction:
    addr: int
    write: bool
    ticket: Ticket
    owner: int
    rmw_write: bool = False


class MemorySystem:
    def __init__(self, cfg):
        cfg.validate()
        self.cfg = cfg
        self.backend = RamulatorMemory(cfg) if cfg.backend == 'ramulator' else SimpleMemory(cfg)
        self.fixed_pending=[]
        self.queue, self.inflight = deque(), {}
        self.next_tag, self.next_tick_ps = 0, cfg.memory_period_ps
        self.line_writers, self.line_readers = {}, Counter()
        self.cycle = -1
        self.core_period_ps = None
        self.next_owner = 1
        self.run_peak = 0
        self.ports = set()
        self.stats = Counter()
        self.by_owner = {}
        self.trace = []

    def _trace(self, event):
        if self.cfg.record_trace:
            self.trace.append(event)

    def can_request(self,addr,size,write,port):
        if port in self.ports:return False
        if self.cfg.backend=='rtl_fixed':return True
        tx=self.cfg.transaction_bytes;count=(addr%tx+size+tx-1)//tx
        return len(self.queue)+len(self.inflight)+count<=self.cfg.outstanding

    @property
    def write_idle(self):
        return not any(r[5] for r in self.fixed_pending) and not any(t.write or t.rmw_write for t in [*self.queue,*self.inflight.values()])

    def request_many(self, requests, owner):
        """Atomically accept the coupled A/B ports of one MXU advance."""
        if any(size<=0 or addr<0 for addr,size,write,port in requests):
            raise ValueError('invalid coupled memory request')
        ports=[r[3] for r in requests]
        if len(ports)!=len(set(ports)) or any(p in self.ports for p in ports): return None
        if self.cfg.backend!='rtl_fixed':
            tx=self.cfg.transaction_bytes
            count=sum((addr%tx+size+tx-1)//tx for addr,size,write,port in requests)
            if count>self.cfg.outstanding: raise ValueError('coupled transfer exceeds adapter capacity')
            if len(self.queue)+len(self.inflight)+count>self.cfg.outstanding: return None
        tickets=[self.request(*r,owner) for r in requests]
        assert all(t is not None for t in tickets)
        return tickets

    def request(self, addr, size, write, port, owner):
        if size <= 0 or addr < 0:
            raise ValueError('invalid memory request')
        if self.cfg.backend=='rtl_fixed':
            return self._fixed_request(addr,size,write,port,owner)
        tx = self.cfg.transaction_bytes
        count = (addr % tx + size + tx - 1) // tx
        if count > self.cfg.outstanding:
            raise ValueError('one logical transfer exceeds adapter capacity')
        if port in self.ports or len(self.queue) + len(self.inflight) + count > self.cfg.outstanding:
            return None
        self.ports.add(port)
        ticket = Ticket(count, accepted_cycle=self.cycle)
        self.stats['logical_write_bytes' if write else 'logical_read_bytes'] += size
        for a in range(addr // tx * tx, (addr + size + tx - 1) // tx * tx, tx):
            partial = write and (addr > a or addr + size < a + tx) and self.cfg.rmw
            self.queue.append(Transaction(a, write and not partial, ticket, owner, partial))
        return ticket

    def _fixed_request(self,addr,size,write,port,owner):
        # Local slice port -> request register -> otpu_dram -> response register.
        # No RMW, channel arbitration or DRAM technology claim in this backend.
        if port in self.ports: return None
        self.ports.add(port)
        ticket=Ticket(1,accepted_cycle=self.cycle)
        tag=self.next_tag; self.next_tag+=1
        due=self.cycle+2+(0 if write else self.cfg.simple_latency_ticks)
        self.fixed_pending.append((due,ticket,tag,owner,addr,write))
        physical=self.cfg.transaction_bytes if port=='B' else 4 if write else 32
        key='write' if write else 'read'
        self.stats['logical_'+key+'_bytes']+=size
        self.stats[key+'_bytes']+=physical; self.stats[key+'_transactions']+=1
        counts=self.by_owner.setdefault(owner,Counter())
        counts[key+'_bytes']+=physical; counts[key+'_transactions']+=1
        self._trace({'event':'accept','id':tag,'time_ps':(self.cycle+1)*self.core_period_ps,'owner':owner,'address':addr,'write':write})
        self.run_peak=max(self.run_peak,len(self.fixed_pending))
        self.stats['peak_outstanding']=max(self.stats['peak_outstanding'],self.run_peak)
        return ticket

    def begin_cycle(self, cycle, period_ps):
        if cycle <= self.cycle:
            raise ValueError('memory clock must advance monotonically')
        if self.core_period_ps is not None and period_ps != self.core_period_ps:
            raise ValueError('changing core period within a persistent memory session is unsupported')
        self.core_period_ps = period_ps
        self.cycle = cycle
        self.ports.clear()
        if self.cfg.backend=='rtl_fixed':
            if period_ps!=self.cfg.memory_period_ps:
                raise ValueError('rtl_fixed memory clock must equal the core clock')
            future=[]
            for due,ticket,tag,owner,addr,write in self.fixed_pending:
                if due<=cycle:
                    ticket.done=True; ticket.pending=0; ticket.complete_cycle=cycle
                    self._trace({'event':'complete','id':tag,'time_ps':cycle*period_ps,'owner':owner,'address':addr,'write':write})
                else: future.append((due,ticket,tag,owner,addr,write))
            self.fixed_pending=future
            return
        now = cycle * period_ps
        while self.next_tick_ps <= now:
            for tag in self.backend.tick():
                tr = self.inflight.pop(tag)
                self._trace({'event': 'complete', 'id': tag, 'time_ps': self.next_tick_ps,
                                   'owner': tr.owner, 'address': tr.addr, 'write': tr.write})
                if not tr.write and not tr.rmw_write:
                    self.line_readers[tr.addr] -= 1
                if tr.rmw_write:
                    self.queue.appendleft(Transaction(tr.addr, True, tr.ticket, tr.owner))
                else:
                    if tr.write:
                        del self.line_writers[tr.addr]
                    tr.ticket.pending -= 1
                    if tr.ticket.pending == 0:
                        tr.ticket.done, tr.ticket.complete_cycle = True, cycle
            for _ in range(self.cfg.ingress_per_controller_tick):
                if not self.queue:
                    break
                tr = self.queue[0]
                tag = self.next_tag
                owner = self.line_writers.get(tr.addr)
                if (owner is not None and owner is not tr.ticket) or ((tr.write or tr.rmw_write) and self.line_readers[tr.addr]):
                    self.stats['line_order_stall_ticks'] += 1
                    break
                if not self.backend.send(tag, tr.addr, tr.write):
                    self.stats['backend_rejected_ticks'] += 1
                    break
                self.queue.popleft()
                if tr.write or tr.rmw_write:
                    self.line_writers[tr.addr] = tr.ticket
                else:
                    self.line_readers[tr.addr] += 1
                self.inflight[tag] = tr
                self.next_tag += 1
                key = 'write' if tr.write else 'read'
                self.stats[key + '_transactions'] += 1
                self.stats[key + '_bytes'] += self.cfg.transaction_bytes
                owner = self.by_owner.setdefault(tr.owner, Counter())
                owner[key + '_transactions'] += 1
                owner[key + '_bytes'] += self.cfg.transaction_bytes
                self._trace({'event': 'accept', 'id': tag, 'time_ps': self.next_tick_ps,
                                   'owner': tr.owner, 'address': tr.addr, 'write': tr.write})
            self.stats['peak_outstanding'] = max(self.stats['peak_outstanding'], len(self.queue) + len(self.inflight))
            self.run_peak = max(self.run_peak, len(self.queue) + len(self.inflight))
            self.next_tick_ps += self.cfg.memory_period_ps

    @property
    def idle(self):
        return not self.queue and not self.inflight and not self.fixed_pending

    def close(self):
        self.backend.close()
