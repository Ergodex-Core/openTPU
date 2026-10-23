"""NV41 OCL mailbox ABI1.0; no instruction encoding or live access on import.

Exclusive cooperative owner: the F2 device lock and allocation lifetime must be
held through terminal ACK. This is not OS/IOMMU or malicious-owner isolation.
"""
from dataclasses import dataclass
import struct
from ..mxu_job import W4A16Job,PROFILE_DIGESTS,local_job_fields
from ..nvfp4 import STRICT_GRADUAL,STRICT_FTZ
from ..nvfp4_transport import uint,TransportError

BASE=0x400
ID=0x3134564e
VERSION=0x00010000
DESC_MAGIC=0x314a344e
DESC_BYTES=256
CAP=1<<6
LAUNCH,ACK,CANCEL,RESET,ARM=1,2,3,4,5
SUCCESS,FAILED,CANCELLED,LOST_ON_RESET=1,2,3,4
READ,WRITE=1,2


@dataclass(frozen=True)
class Window:
    handle:int
    base:int
    span:int
    permissions:int

    def validate(self,bank_base=0,bank_bytes=1<<29):
        for x in (self.handle,self.base,self.span):uint(x)
        if not self.handle or not self.span or self.base%64 or self.span%64:
            raise TransportError('nonzero handle and aligned full-line allocation required')
        if type(self.permissions) is not int or self.permissions not in (READ,WRITE,READ|WRITE):
            raise TransportError('invalid window permissions')
        if self.base<bank_base or self.base+self.span>bank_base+bank_bytes or self.base+self.span>1<<64:
            raise TransportError('window outside selected HBM bank')

    def words(self):
        return [self.handle&0xffffffff,self.handle>>32,self.base&0xffffffff,self.base>>32,
                self.span&0xffffffff,self.span>>32,self.permissions,0]


def encode(job,windows,*,bank_base=0,bank_bytes=1<<29):
    if type(job) is not W4A16Job:raise TransportError('checked job required')
    local_job_fields(job,job.generation)
    if len(windows)!=4 or any(type(w) is not Window for w in windows):raise TransportError('four windows required')
    for w in windows:w.validate(bank_base,bank_bytes)
    if len({w.handle for w in windows})!=4:raise TransportError('duplicate allocation handle')
    words=[DESC_MAGIC,VERSION,DESC_BYTES,0,job.k,int(job.profile==STRICT_FTZ),job.global_bits,0]
    digest=int(job.profile_digest,16)
    words.extend((digest>>(32*i))&0xffffffff for i in range(8))
    for v in (job.context,job.generation,job.request,job.position,job.epoch):words.extend((v&0xffffffff,v>>32))
    selected=[]
    for role in ('a','w','scales','y'):
        b=getattr(job,role)
        matches=[w for w in windows if w.base<=b.address and b.address+b.span<=w.base+w.span]
        if len(matches)!=1:raise TransportError('ambiguous/unregistered operand allocation')
        w=matches[0];offset=b.address-w.base
        if offset%64 or not w.permissions & (WRITE if role=='y' else READ):
            raise TransportError('operand offset alignment or permission')
        # Every native read is a full line, including authorized allocation padding.
        needed=4 if role=='y' else 2*job.k if role=='a' else (job.k+1)//2 if role=='w' else (job.k+15)//16
        if offset+((needed+63)//64)*64>w.span:raise TransportError('missing full-line padding rights')
        selected.append(w)
        for v in (w.handle,offset,b.span,b.stride):words.extend((v&0xffffffff,v>>32))
    y=selected[3]
    if any(max(y.base,w.base)<min(y.base+y.span,w.base+w.span) for w in selected[:3]):
        raise TransportError('output allocation aliases input allocation')
    words.extend([0]*6)
    assert len(words)==64
    return struct.pack('<64I',*words)


@dataclass(frozen=True)
class Handle:
    sequence:int
    context:int
    generation:int
    request:int
    position:int
    epoch:int


@dataclass(frozen=True)
class Completion:
    handle:Handle
    status:int
    faults:int
    bits:int
    output_visible:bool


class Mailbox:
    """Register-only host client; caller owns transport and immutable allocations.

    Construction only performs discovery reads after a valid exclusive lock.
    This class never opens/maps a device, loads an image, or transfers weights.
    A failed/uncertain launch must be queried rather than automatically retried.
    """
    def __init__(self,owner):
        # F2Transport supplies both; Board supplies the guard and owns its transport.
        self._own=owner._own
        self.t=owner if hasattr(owner,"reg_read") else owner.t
        self._own()
        self.pending=None;self.windows=None;self.uncertain=False;self.acking=False;self.ack_completion=None
        if self.t.reg_read(0x3c)<5 or not self.t.reg_read(0x5c)&CAP:
            raise TransportError('image has no NV41 mailbox capability')
        if [self.read(x) for x in (0,4,8)]!=[ID,VERSION,DESC_BYTES] or self.read(0xc)!=0x00030140:
            raise TransportError('unsupported complete mailbox ABI/subset')
        for offset,profile in ((0x80,STRICT_GRADUAL),(0xa0,STRICT_FTZ)):
            digest=sum(self.read(offset+4*i)<<(32*i) for i in range(8))
            if digest!=int(PROFILE_DIGESTS[profile],16):raise TransportError('profile mismatch')

    def read(self,offset):
        self._own();return self.t.reg_read(BASE+offset)
    def read64(self,offset):return self.read(offset)|(self.read(offset+4)<<32)
    def write(self,offset,value):self._own();self.t.reg_write(BASE+offset,value)
    def write64(self,offset,value):
        uint(value);self.write(offset,value&0xffffffff);self.write(offset+4,value>>32)

    def configure(self,context,generation,windows):
        uint(context);uint(generation)
        if self.pending is not None or self.ack_completion is not None or self.read(0x10)&1:raise TransportError('job/completion credit occupied')
        if len(windows)!=4:raise TransportError('four windows required')
        bank_base=self.read64(0xc0);bank_bytes=self.read64(0xc8)
        for w in windows:w.validate(bank_base,bank_bytes)
        self.windows=None  # Any uncertain table/ARM write invalidates cached admission.
        self.write64(0x28,generation);self.write64(0x30,context)
        for index,w in enumerate(windows):
            for n,v in enumerate(w.words()):self.write(0x100+32*index+4*n,v)
        self.write(0x14,ARM)
        if self.read(0x18):raise TransportError('session configuration rejected')
        self.context=context;self.generation=generation;self.windows=tuple(windows)

    def submit(self,job):
        if self.windows is None or self.pending is not None or self.ack_completion is not None:raise TransportError('configure/drain before submit')
        if (job.context,job.generation)!=(self.context,self.generation):raise TransportError('owner/session mismatch')
        wire=encode(job,self.windows,bank_base=self.read64(0xc0),bank_bytes=self.read64(0xc8))
        before=self.read64(0x1c)
        for i,v in enumerate(struct.unpack('<64I',wire)):self.write(0x200+4*i,v)
        if before==(1<<64)-1:raise TransportError('acceptance sequence exhausted')
        # Preserve identity BEFORE the irreversible doorbell, including I/O exceptions.
        self.pending=Handle(before+1,job.context,job.generation,job.request,job.position,job.epoch)
        self.uncertain=True
        self.write(0x14,LAUNCH)
        if self.resolve() is None:raise TransportError('launch unconfirmed; resolve before retry')
        return self.pending

    def resolve(self):
        """Query an uncertain launch/ACK; never replay a doorbell.

        Returns a checked accepted handle, or None. A nonaccepted attempt with no
        definitive rejection stays pending until a coordinated RESET is observed.
        """
        self._own()
        if self.pending is None:return None
        sequence=self.read64(0x1c);state=self.read(0x10)
        if self.acking and not state&1 and sequence==self.pending.sequence:
            self.pending=None;self.acking=False;return None
        if self.uncertain:
            if sequence==self.pending.sequence:self.uncertain=False
            elif sequence==self.pending.sequence-1:
                error=self.read(0x18)
                reset_idle=not state&1 and self.read64(0xd0)>self.pending.generation
                if error or reset_idle:
                    self.pending=None;self.uncertain=False
                    raise TransportError('launch was not accepted')
                return None
            else:raise TransportError('foreign acceptance sequence')
        return self.pending

    def poll(self):
        if self.resolve() is None:return None
        if not self.read(0x10)&2:return None
        h=Handle(self.read64(0x48),*(self.read64(0x50+8*i) for i in range(5)))
        if h!=self.pending:raise TransportError('stale/foreign completion')
        status=self.read(0x40)
        return Completion(h,status&255,(status>>8)&255,self.read(0x44),bool(status&(1<<16)))

    def acquire(self):
        self.resolve()
        if self.acking:return None  # ACK outcome still unknown; never replay it.
        if self.ack_completion is not None:
            c=self.ack_completion;self.ack_completion=None;return c
        c=self.poll()
        if c is not None:
            self.ack_completion=c  # Terminal outcome survives an uncertain ACK.
            self.acking=True
            self.write(0x14,ACK)
            if self.read(0x18):raise TransportError('completion ACK rejected')
            self.pending=None;self.acking=False;self.ack_completion=None
        return c

    def retry_ack(self):
        """Explicit recovery only after an ordered query proves old ACK unexecuted."""
        if not self.acking or self.ack_completion is None:
            raise TransportError('no uncertain ACK to recover')
        c=self.poll()
        if not self.acking:return  # Old ACK was confirmed; acquire cached outcome.
        if c!=self.ack_completion:
            raise TransportError('cannot prove same terminal record remains held')
        self.write(0x14,ACK)
        if self.read(0x18):raise TransportError('completion ACK rejected')
        self.pending=None;self.acking=False

    def cancel(self):
        if self.pending is None:raise TransportError('no pending job')
        self.write(0x14,CANCEL)
        return self.read(0x18)==0

    def reset(self):
        self.windows=None
        self.write(0x14,RESET)
        if self.read(0x18):raise TransportError('reset rejected')
        self.windows=None
