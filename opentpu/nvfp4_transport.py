"""Bounded simulated-memory transport for the strict local W4A16 controller.

No Board/XDMA imports, device access, inference backend or binary ISA allocation.
Single-threaded simulation contract; context numbers model trusted caller identity,
not OS authentication. Arithmetic is supplied by a separately tested local device.
"""
from collections import deque
from dataclasses import dataclass,field
import json
import struct

from .mxu_job import W4A16Job,local_job_fields
from . import nvfp4 as N

U64=(1<<64)-1
MAX_DESCRIPTOR=4096


class TransportError(ValueError):pass
class DescriptorError(TransportError):pass
class AccessDenied(TransportError):pass
class Busy(TransportError):pass
class StaleResponse(TransportError):pass


def uint(value,bits=64):
    if type(value) is not int or not 0<=value<(1<<bits):
        raise TransportError(f'expected unsigned {bits}-bit integer')
    return value


def span(address,length):
    uint(address);uint(length)
    if not length or address+length>1<<64:raise AccessDenied('invalid address/span')


def encode_descriptor(job):
    if type(job) is not W4A16Job:raise DescriptorError('expected checked W4A16Job')
    body=json.dumps(job.to_mapping(),sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
    if not 0<len(body)<=MAX_DESCRIPTOR:raise DescriptorError('descriptor too large')
    return struct.pack('>I',len(body))+body


def decode_descriptor(frame):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise DescriptorError('duplicate descriptor field')
            d[k]=v
        return d
    def invalid_number(_):raise DescriptorError('non-integer descriptor number')
    if type(frame) is not bytes or len(frame)<4:raise DescriptorError('short frame header')
    n=struct.unpack('>I',frame[:4])[0]
    if not 0<n<=MAX_DESCRIPTOR or len(frame)!=n+4:raise DescriptorError('invalid frame length')
    try:
        d=json.loads(frame[4:].decode('utf8'),object_pairs_hook=pairs,
                     parse_float=invalid_number,parse_constant=invalid_number)
        return W4A16Job.from_mapping(d)
    except (ValueError,TypeError,UnicodeError,RecursionError) as e:
        raise DescriptorError(str(e)) from e


@dataclass(frozen=True)
class Identity:
    context:int
    generation:int
    request:int
    position:int
    epoch:int

    @classmethod
    def of(cls,job):return cls(job.context,job.generation,job.request,job.position,job.epoch)


@dataclass(frozen=True)
class Handle:
    namespace:object
    transport:int
    serial:int
    identity:Identity


@dataclass(frozen=True)
class Packet:
    handle:Handle
    slot:int
    index:int
    group:int
    mask:int
    first:bool
    last:bool
    activations:tuple
    codes:tuple
    scale:int
    scale_group:int
    scale_index:int
    scale_epoch:int


@dataclass(frozen=True)
class DeviceResult:
    identity:Identity
    profile:str
    bits:int
    faults:int=0


@dataclass(frozen=True)
class Completion:
    handle:Handle
    status:str
    faults:int=0
    reason:str=''


@dataclass(frozen=True)
class WriteTicket:
    namespace:object
    serial:int
    lease:int
    context:int
    generation:int


@dataclass
class _Region:
    address:int
    data:bytearray
    context:int
    generation:int
    permissions:frozenset
    revision:int
    content_revision:int=0
    leases:set=field(default_factory=set)


@dataclass(frozen=True)
class _Lease:
    namespace:object
    serial:int
    job:W4A16Job
    revisions:tuple


class SimMemory:
    """Sparse bounded byte windows, never a live memory/device mapping.

    A lease reserves complete mapped windows exclusively, conservatively blocking
    writes/rebind/unmap even to unused padding. Each device access is additionally
    bounded by the corresponding declared descriptor span.
    """
    MAX_WINDOW=4096
    MAX_REGIONS=128

    def __init__(self):
        # Local opaque namespace prevents equal counters in independent memories
        # from accepting each other's handles, leases or write acknowledgments.
        self._namespace=object()
        self._regions={};self._leases={};self._writes={}
        self._nonce=0;self._revision=0;self._transport=0
        self.audit=deque(maxlen=2048)

    def _next(self):
        if self._nonce==U64:raise TransportError('simulation nonce exhausted')
        self._nonce+=1;return self._nonce

    def _transport_id(self):
        if self._transport==U64:raise TransportError('transport identity exhausted')
        self._transport+=1;return self._transport

    @staticmethod
    def _permissions(permissions):
        try:p=frozenset(permissions)
        except TypeError as e:raise AccessDenied('invalid permissions') from e
        if not p or p-{'read','write'}:raise AccessDenied('invalid permissions')
        return p

    def map(self,address,data,*,context,generation,permissions=('read',)):
        if type(data) is not bytes or not 0<len(data)<=self.MAX_WINDOW:
            raise AccessDenied('bounded initial bytes required')
        span(address,len(data));uint(context);uint(generation)
        p=self._permissions(permissions)
        if len(self._regions)>=self.MAX_REGIONS:raise Busy('simulated region limit')
        if any(max(address,r.address)<min(address+len(data),r.address+len(r.data))
               for r in self._regions.values()):raise AccessDenied('overlapping mappings')
        self._revision+=1
        self._regions[address]=_Region(address,bytearray(data),context,generation,p,self._revision)

    def _locate(self,address,length,context,generation,permission):
        span(address,length);uint(context);uint(generation)
        for r in self._regions.values():
            if r.address<=address and address+length<=r.address+len(r.data):
                if r.context!=context or r.generation!=generation or permission not in r.permissions:
                    raise AccessDenied('mapping identity/permission mismatch')
                return r
        raise AccessDenied('unmapped or crossing access')

    def read(self,address,length,*,context,generation):
        r=self._locate(address,length,context,generation,'read')
        off=address-r.address
        return bytes(r.data[off:off+length])

    def write(self,address,data,*,context,generation):
        if type(data) is not bytes:raise AccessDenied('bytes required')
        r=self._locate(address,len(data),context,generation,'write')
        if r.leases:raise Busy('mapped window leased')
        off=address-r.address;r.data[off:off+len(data)]=data
        r.content_revision+=1
        self.audit.append(('HOST_WRITE',address,len(data)))

    def unmap(self,address,*,context,generation):
        uint(address);uint(context);uint(generation)
        r=self._regions.get(address)
        if r is None or r.context!=context or r.generation!=generation:
            raise AccessDenied('mapping identity mismatch')
        if r.leases:raise Busy('mapped window leased')
        del self._regions[address]

    def rebind(self,address,*,context,generation,new_generation,permissions=None):
        uint(address);uint(context);uint(generation);uint(new_generation)
        r=self._regions.get(address)
        if r is None or r.context!=context or r.generation!=generation:
            raise AccessDenied('mapping identity mismatch')
        if new_generation<=generation:raise AccessDenied('generation must advance without wrap')
        if r.leases:raise Busy('mapped window leased')
        p=r.permissions if permissions is None else self._permissions(permissions)
        self._revision+=1;r.revision=self._revision;r.generation=new_generation;r.permissions=p

    def _acquire(self,job):
        regions=[]
        for name in ('a','w','scales','y'):
            b=getattr(job,name)
            r=self._locate(b.address,b.span,job.context,job.generation,
                           'write' if name=='y' else 'read')
            if r.leases:raise Busy('mapped window already leased')
            regions.append(r)
        lease=_Lease(self._namespace,self._next(),job,tuple(sorted({(r.address,r.revision) for r in regions})))
        for r in regions:r.leases.add(lease.serial)
        self._leases[lease.serial]=lease
        return lease

    def input_snapshot(self,job):
        """Host simulation release snapshot, not a hardware/cache fence.

        Validate every allocation, bind mapping lifetime plus complete input
        window content versions. Callers must check and acquire a lease within
        the same single-threaded operation; a snapshot alone reserves nothing.
        """
        if type(job) is not W4A16Job:raise DescriptorError('checked job required')
        result=[]
        for role in ('a','w','scales','y'):
            b=getattr(job,role)
            r=self._locate(b.address,b.span,job.context,job.generation,
                           'write' if role=='y' else 'read')
            if r.leases:raise Busy('mapped window already leased')
            result.append((role,r.address,r.revision,r.content_revision if role!='y' else None))
        return tuple(result)

    def _check(self,lease):
        if self._leases.get(lease.serial)!=lease:raise StaleResponse('inactive memory lease')
        for address,revision in lease.revisions:
            r=self._regions.get(address)
            if r is None or r.revision!=revision or lease.serial not in r.leases:
                raise StaleResponse('changed memory lease')

    def read_input(self,lease,role,offset,length):
        self._check(lease)
        if role not in ('a','w','scales'):raise AccessDenied('invalid input role')
        uint(offset);uint(length)
        b=getattr(lease.job,role)
        if not length or offset+length>b.span:raise AccessDenied('read outside descriptor span')
        r=self._locate(b.address+offset,length,lease.job.context,lease.job.generation,'read')
        self.audit.append(('READ',lease.serial,role,b.address+offset,length))
        off=b.address+offset-r.address
        return bytes(r.data[off:off+length])

    def prepare_output(self,lease,data):
        self._check(lease)
        if type(data) is not bytes or len(data)!=4 or len(data)>lease.job.y.span:
            raise AccessDenied('one FP32 private output required')
        j=lease.job;self._locate(j.y.address,4,j.context,j.generation,'write')
        if any(v[0].lease==lease.serial for v in self._writes.values()):raise Busy('write already staged')
        t=WriteTicket(self._namespace,self._next(),lease.serial,j.context,j.generation)
        self._writes[t.serial]=(t,bytes(data))
        self.audit.append(('WRITE_STAGED',t.serial,j.y.address,4))
        return t

    def commit_output(self,lease,ticket):
        self._check(lease)
        pending=self._writes.get(ticket.serial)
        if not pending or pending[0]!=ticket or ticket.lease!=lease.serial:
            raise StaleResponse('stale/duplicate write acknowledgment')
        j=lease.job;r=self._locate(j.y.address,4,j.context,j.generation,'write')
        data=pending[1];off=j.y.address-r.address
        r.data[off:off+4]=data
        r.content_revision+=1
        del self._writes[ticket.serial]
        self.audit.append(('Y_VISIBLE',ticket.serial,j.y.address,4))

    def discard_output(self,ticket):
        pending=self._writes.get(ticket.serial)
        if pending and pending[0]==ticket:del self._writes[ticket.serial]

    def _release(self,lease):
        self._check(lease)
        if any(v[0].lease==lease.serial for v in self._writes.values()):
            raise Busy('cannot release pending write')
        for address,_ in lease.revisions:self._regions[address].leases.remove(lease.serial)
        del self._leases[lease.serial]


class LocalJobTransport:
    """One descriptor, one tile slot, one terminal completion credit.

    Prefetch/accept mirror a valid-ready boundary. receive_result accepts a full
    private core result; a successful completion is unavailable until the staged
    simulated output write is explicitly acknowledged. All public operations are
    synchronous, single-threaded simulation calls.
    """
    def __init__(self,memory,*,generation):
        if type(memory) is not SimMemory:raise TransportError('SimMemory only; no device transport')
        uint(generation)
        self.memory=memory;self.generation=generation;self.transport=memory._transport_id()
        self._serial=0;self._slot_serial=0;self._job=None;self._handle=None;self._lease=None
        self._state='IDLE';self._index=0;self._data=None;self._scale=None;self._packet=None
        self._private=None;self._ticket=None;self._completion=None
        self.events=deque(maxlen=2048);self.history=deque(maxlen=32)

    @property
    def state(self):return self._state

    @property
    def packet(self):return self._packet

    @property
    def completion(self):return self._completion

    def submit(self,frame):
        if self._state!='IDLE':raise Busy('job/completion credit occupied')
        j=decode_descriptor(frame)
        try:local_job_fields(j,self.generation)
        except ValueError as e:raise DescriptorError(str(e)) from e
        identity=Identity.of(j)
        if any(c.handle.identity==identity for c in self.history):
            raise DescriptorError('retained request identity already completed; query before retry')
        if self._serial==U64:raise Busy('handle sequence exhausted')
        lease=self.memory._acquire(j)
        self._serial+=1
        h=Handle(self.memory._namespace,self.transport,self._serial,identity)
        self._job=j;self._handle=h;self._lease=lease;self._state='FETCH'
        self._index=0;self._data=None;self._scale=None;self._packet=None
        self._private=None;self._ticket=None;self._completion=None
        self.events.append(('ACCEPTED',h))
        return h

    def _current(self,handle):
        if handle!=self._handle or self._state=='IDLE':raise StaleResponse('wrong/stale job handle')

    def prefetch(self,*,operand_ready=True,scale_ready=True):
        if self._state!='FETCH':return None
        if self._packet is not None:return self._packet
        j=self._job;i=self._index;n=min(16,j.k-16*i)
        try:
            if self._data is None and operand_ready:
                a=self.memory.read_input(self._lease,'a',32*i,2*n)
                w=self.memory.read_input(self._lease,'w',8*i,(n+1)//2)
                self._data=(tuple(struct.unpack('<'+'H'*n,a))+(0,)*(16-n),
                            tuple(N.unpack_codes(w,n))+(0,)*(16-n))
            if self._scale is None and scale_ready:
                self._scale=self.memory.read_input(self._lease,'scales',i,1)[0]
            if self._data is not None and self._scale is not None:
                if self._slot_serial==U64:raise TransportError('tile slot sequence exhausted')
                self._slot_serial+=1
                self._packet=Packet(self._handle,self._slot_serial,i,0,(1<<n)-1,i==0,
                                    i==(j.k+15)//16-1,*self._data,self._scale,0,i,j.epoch)
                self.events.append(('PACKET_READY',self._packet.slot,i,j.epoch))
        except TransportError as e:self._finish('FAILED',reason=str(e))
        return self._packet

    def accept_packet(self,packet):
        # Opaque ownership token: an edited/copied packet cannot free this slot.
        if self._state!='FETCH' or packet is None or packet is not self._packet:
            raise StaleResponse('packet/scale identity or tile ownership mismatch')
        self.events.append(('BUFFER_FREE',packet.slot,packet.group,packet.index,packet.handle.identity.epoch))
        self._packet=None;self._data=None;self._scale=None;self._index+=1
        if self._index==(self._job.k+15)//16:self._state='WAIT_RESULT'

    def receive_result(self,handle,result):
        self._current(handle)
        if self._state!='WAIT_RESULT' or handle.identity.generation!=self.generation:
            raise StaleResponse('result before drain or from canceled generation')
        if type(result) is not DeviceResult or result.identity!=handle.identity or result.profile!=self._job.profile:
            raise StaleResponse('result identity/profile mismatch')
        if type(result.identity) is not Identity:raise StaleResponse('malformed result identity')
        for v in vars(result.identity).values():uint(v)
        uint(result.bits,32);uint(result.faults,8)
        if result.faults and result.bits:raise TransportError('failed core result must be canonical zero')
        if not result.faults:
            try:N.fp32_value(result.bits)
            except ValueError as e:raise TransportError('nonfinite successful result') from e
        self.events.append(('PRIVATE_RESULT',handle,result.faults))
        if result.faults:self._finish('FAILED',faults=result.faults)
        else:self._private=struct.pack('<I',result.bits);self._state='WRITE_PENDING'

    def issue_write(self,*,ready=True):
        if self._state=='WRITE_ACK':return self._ticket
        if self._state!='WRITE_PENDING' or not ready:return None
        try:self._ticket=self.memory.prepare_output(self._lease,self._private)
        except TransportError as e:
            self._finish('FAILED',reason=str(e));return None
        self._state='WRITE_ACK';self.events.append(('WRITE_STAGED',self._ticket))
        return self._ticket

    def acknowledge_write(self,ticket):
        if self._state!='WRITE_ACK' or ticket!=self._ticket:
            raise StaleResponse('stale/wrong/duplicate output acknowledgment')
        self.memory.commit_output(self._lease,ticket)
        self.events.append(('Y_VISIBLE',ticket))
        self._ticket=None;self._finish('SUCCEEDED')

    def _finish(self,status,*,faults=0,reason=''):
        if self._ticket is not None:self.memory.discard_output(self._ticket)
        self._ticket=None;self._private=None;self._packet=None;self._data=None;self._scale=None
        self._completion=Completion(self._handle,status,faults,reason)
        self._state='COMPLETION';self.events.append(('COMPLETION_RELEASE',self._completion))

    def cancel(self,handle):
        self._current(handle)
        if self._state=='COMPLETION':raise Busy('terminal completion already published')
        self._finish('CANCELLED')

    def reset(self,new_generation):
        uint(new_generation)
        if new_generation<=self.generation:raise TransportError('generation must advance without wrap')
        if self._state not in ('IDLE','COMPLETION'):self._finish('LOST_ON_RESET')
        self.generation=new_generation;self.events.append(('RESET',new_generation))

    def take_completion(self,*,context):
        uint(context)
        c=self._completion
        if c is None:return None
        if c.handle.identity.context!=context:raise AccessDenied('completion owner mismatch')
        self.memory._release(self._lease)
        self.history.append(c);self._completion=None;self._handle=None;self._lease=None;self._job=None
        self._state='IDLE';self.events.append(('COMPLETION_ACQUIRE',c))
        return c

    def lookup(self,handle):
        if self._handle==handle:return self._completion or self._state
        return next((c for c in self.history if c.handle==handle),None)
