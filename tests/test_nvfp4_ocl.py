"""NV41 wire ABI, bounded allocations and lock-bound host recovery (no device IO)."""
from dataclasses import replace
import struct
from types import SimpleNamespace
import pytest
from opentpu.host import nvfp4_ocl as O
from opentpu.mxu_job import Buffer
from opentpu.nvfp4_transport import TransportError
from test_nvfp4_local_job import job


def fixture(k=17,mode=0,request=0):
    j=replace(job(k,mode,request),a=Buffer(0x1000,2*k,2*k))
    windows=tuple(O.Window(i+1,addr,128 if i==0 else 64,O.WRITE if i==3 else O.READ)
                  for i,addr in enumerate((0x1000,0x2000,0x3000,0x4000)))
    return j,windows


@pytest.mark.parametrize('mode',[0,1])
@pytest.mark.parametrize('k',range(1,65))
def test_wire_identity_lengths_and_profile(k,mode):
    j,w=fixture(k,mode,1<<40);x=struct.unpack('<64I',O.encode(j,w))
    assert x[:8]==(O.DESC_MAGIC,O.VERSION,256,0,k,mode,j.global_bits,0)
    assert sum(x[8+i]<<(32*i) for i in range(8))==int(j.profile_digest,16)
    assert tuple(x[16+2*i]|x[17+2*i]<<32 for i in range(5))==(j.context,j.generation,j.request,j.position,j.epoch)
    for i,role in enumerate(('a','w','scales','y')):
        b=getattr(j,role);base=26+8*i
        assert tuple(x[base+2*n]|x[base+2*n+1]<<32 for n in range(4))==(i+1,0,b.span,b.stride)
    assert x[58:]==(0,)*6


@pytest.mark.parametrize('change',[
    dict(handle=0),dict(handle=True),dict(base=0x1001),dict(span=63),dict(span=0),
    dict(base=(1<<29)-64,span=128),dict(base=(1<<64)-64,span=128),
    dict(permissions=0),dict(permissions=4),dict(permissions=True)])
def test_invalid_windows_fail_before_io(change):
    j,w=fixture();w=list(w);w[0]=replace(w[0],**change)
    with pytest.raises(TransportError):O.encode(j,w)


def test_alias_permissions_missing_padding_and_ambiguous_mapping():
    j,w=fixture(64)
    for bad in [w[:3],(w[0],w[0],w[2],w[3]),
                (replace(w[0],span=64),*w[1:]),
                (*w[:3],replace(w[3],permissions=O.READ)),
                (*w[:3],replace(w[3],base=0x1000))]:
        with pytest.raises(TransportError):O.encode(j,bad)
    for j2 in [replace(j,a=replace(j.a,address=0x1002)),
               replace(j,a=replace(j.a,address=(1<<64)-128))]:
        with pytest.raises((TransportError,ValueError)):O.encode(j2,w)


class Registers:
    """Minimal scripted bus. Only tests client IO/recovery, not RTL validation."""
    def __init__(self):
        self.values={0x3c:5,0x5c:O.CAP,O.BASE:O.ID,O.BASE+4:O.VERSION,
                     O.BASE+8:256,O.BASE+12:0x30140,O.BASE+0xc8:1<<29}
        for off,profile in ((0x80,O.STRICT_GRADUAL),(0xa0,O.STRICT_FTZ)):
            d=int(O.PROFILE_DIGESTS[profile],16)
            self.values.update({O.BASE+off+4*i:(d>>(32*i))&0xffffffff for i in range(8)})
        self.locked=True;self.writes=[];self.raise_at=None;self.accept=True
    def _own(self):
        if not self.locked:raise RuntimeError('lock lost')
    def reg_read(self,a):
        if self.raise_at==('read',a):self.raise_at=None;raise OSError('uncertain read')
        return self.values.get(a,0)
    def reg_write(self,a,v):
        self.writes.append((a,v));self.values[a]=v
        if a==O.BASE+0x14 and v==O.LAUNCH:
            if self.accept:
                self.values[O.BASE+0x1c]=self.values.get(O.BASE+0x1c,0)+1
                self.values[O.BASE+0x10]=1
            else:self.values[O.BASE+0x18]=4
        if a==O.BASE+0x14 and v==O.ACK:self.values[O.BASE+0x10]=0
        if self.raise_at==('write',a):self.raise_at=None;raise OSError('uncertain write')


def complete(r,h):
    r.values[O.BASE+0x10]=3;r.values[O.BASE+0x40]=0x10001;r.values[O.BASE+0x44]=0x3f800000
    for off,v in [(0x48,h.sequence),*[(0x50+8*i,getattr(h,n)) for i,n in enumerate(('context','generation','request','position','epoch'))]]:
        r.values[O.BASE+off]=v&0xffffffff;r.values[O.BASE+off+4]=v>>32


@pytest.mark.parametrize('board_like',[False,True])
def test_lock_guard_on_every_operation_and_board_adapter(board_like):
    r=Registers();owner=SimpleNamespace(t=r,_own=r._own) if board_like else r
    m=O.Mailbox(owner);j,w=fixture();m.configure(j.context,j.generation,w);h=m.submit(j)
    complete(r,h);assert m.poll().bits==0x3f800000
    r.locked=False
    for action in (m.poll,m.acquire,m.resolve,lambda:m.read(0),m.cancel,m.reset):
        with pytest.raises(RuntimeError,match='lock lost'):action()
    r.locked=True;assert m.acquire().handle==h and m.pending is None


@pytest.mark.parametrize('failure',['doorbell','sequence','error','state'])
def test_accepted_launch_with_uncertain_io_is_queryable_without_replay(failure):
    r=Registers();m=O.Mailbox(r);j,w=fixture();m.configure(j.context,j.generation,w)
    # The first sequence/bank reads are prelaunch; inject on the doorbell instead.
    original=r.reg_write
    def write(a,v):
        original(a,v)
        if a==O.BASE+0x14 and v==O.LAUNCH:
            if failure=='doorbell':raise OSError('accepted then failed')
            r.raise_at=('read',O.BASE+{'sequence':0x1c,'error':0x18,'state':0x10}[failure])
    r.reg_write=write
    # Error register is only read for rejection; an accepted launch does not need it.
    if failure=='error':
        h=m.submit(j);assert h==m.pending;return
    with pytest.raises(OSError):m.submit(j)
    assert m.pending is not None
    with pytest.raises(TransportError):m.submit(j)
    h=m.resolve();complete(r,h);assert m.acquire().handle==h
    assert sum(a==O.BASE+0x14 and v==O.LAUNCH for a,v in r.writes)==1


@pytest.mark.parametrize('status',[O.SUCCESS,O.FAILED,O.CANCELLED,O.LOST_ON_RESET])
def test_uncertain_ack_queries_instead_of_replaying(status):
    r=Registers();m=O.Mailbox(r);j,w=fixture();m.configure(j.context,j.generation,w);h=m.submit(j);complete(r,h)
    r.values[O.BASE+0x40]=status|(0x8000 if status==O.FAILED else 0)|(0x10000 if status==O.SUCCESS else 0)
    r.values[O.BASE+0x44]=0x3f800000 if status==O.SUCCESS else 0
    expected=m.poll()
    r.raise_at=('write',O.BASE+0x14)
    with pytest.raises(OSError):m.acquire()
    assert m.pending==h and m.acking
    assert m.resolve() is None and m.pending is None
    with pytest.raises(TransportError):m.submit(j)
    assert m.acquire()==expected
    assert m.acquire() is None
    assert sum(a==O.BASE+0x14 and v==O.ACK for a,v in r.writes)==1


def test_old_image_refused_without_staging_or_probes_above_legacy_map():
    r=Registers();r.values[0x5c]=0
    with pytest.raises(TransportError):O.Mailbox(r)
    assert not r.writes


@pytest.mark.parametrize('failure',['table','arm','confirmation'])
def test_uncertain_reconfiguration_invalidates_cached_windows(failure):
    r=Registers();m=O.Mailbox(r);j,w=fixture();m.configure(j.context,j.generation,w)
    moved=tuple(replace(x,base=x.base+0x10000) for x in w)
    original=r.reg_write
    def write(a,v):
        original(a,v)
        if failure=='table' and a==O.BASE+0x100:raise OSError('uncertain table write')
        if a==O.BASE+0x14 and v==O.ARM:
            if failure=='arm':raise OSError('ARM executed, response lost')
            if failure=='confirmation':r.raise_at=('read',O.BASE+0x18)
    r.reg_write=write
    with pytest.raises(OSError):m.configure(j.context,j.generation,moved)
    assert m.windows is None
    with pytest.raises(TransportError):m.submit(j)
    assert not any(a==O.BASE+0x14 and v==O.LAUNCH for a,v in r.writes)


def test_ack_failure_before_execution_has_explicit_proven_recovery():
    r=Registers();m=O.Mailbox(r);j,w=fixture();m.configure(j.context,j.generation,w);h=m.submit(j);complete(r,h)
    expected=m.poll();original=r.reg_write
    def write(a,v):
        if a==O.BASE+0x14 and v==O.ACK:raise OSError('ACK was not sent')
        original(a,v)
    r.reg_write=write
    with pytest.raises(OSError):m.acquire()
    assert m.resolve()==h and m.acquire() is None
    r.reg_write=original;m.retry_ack()
    assert m.acquire()==expected and m.pending is None
    with pytest.raises(TransportError):m.retry_ack()
