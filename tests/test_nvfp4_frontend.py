"""Opt-in host negotiation/input-release/completion with simulated memory only."""
from dataclasses import replace
import json
import struct

import pytest

from opentpu.host.nvfp4 import (SimEndpoint,ProtectedFrontend,CAPABILITIES,Capabilities,
    CapabilityMismatch)
from opentpu.nvfp4_transport import (TransportError,DescriptorError,AccessDenied,Busy,
    StaleResponse,encode_descriptor,U64)
from opentpu.mxu_job import Buffer
from test_nvfp4_transport import fixture,drain,deliver,output,SENTINEL,frame
from test_nvfp4_local_job import SESSION


def host_fixture(k=17,mode=0,request=0):
    j,m,_,a,c,s=fixture(k,mode,request)
    e=SimEndpoint(m,context=j.context,generation=SESSION)
    f=ProtectedFrontend(e,context=j.context,generation=SESSION)
    return j,m,e,f,a,c,s


def accepted(j,e,f):
    wire=encode_descriptor(j);release=f.release_inputs(wire)
    return f.submit(wire,release)


BAD_CAPS=[('protocol','opentpu'),('major',2),('major',True),('minor',1),
          ('framing','binary'),('profiles',()),
          ('profiles',((CAPABILITIES.profiles[0][0],'0'*64),)),
          ('tokens',2),('outputs',2),('max_k',65),('in_flight',2),('address_bits',32),
          ('fences','HALTED'),('cancellation','rollback-visible')]


@pytest.mark.parametrize('field,value',BAD_CAPS)
@pytest.mark.parametrize('side',['host','endpoint'])
def test_exact_negotiated_capability_tuple(field,value,side):
    j,m,e,f,*_=host_fixture()
    bad=replace(CAPABILITIES,**{field:value})
    if side=='endpoint':e._capabilities=bad
    with pytest.raises(CapabilityMismatch):
        ProtectedFrontend(e,context=j.context,generation=SESSION,
                          capabilities=bad if side=='host' else CAPABILITIES)
    assert not m._leases and e.transport.state=='IDLE'
    e._capabilities=CAPABILITIES
    # Failed negotiation did not revoke the existing good session.
    accepted(j,e,f);f.cancel(e.transport._handle);assert f.acquire_completion().status=='CANCELLED'


def test_legacy_transports_fail_closed_without_any_io():
    class Legacy:
        def __getattr__(self,name):raise AssertionError('legacy I/O attempted: '+name)
    for value in [Legacy(),{},CAPABILITIES,None]:
        with pytest.raises(CapabilityMismatch):ProtectedFrontend(value,context=0,generation=0)
    from opentpu.host.fake import FakeTransport
    from opentpu.host import regs as R
    legacy=FakeTransport(devname=None,regmap=5)
    caps=legacy.reg_read(R.R_CAPS);assert caps & R.CAP_MX and caps & R.CAP_W4
    legacy.reg_read=lambda *a:pytest.fail('negotiation touched legacy registers')
    with pytest.raises(CapabilityMismatch):ProtectedFrontend(legacy,context=0,generation=0)
    # Existing fallback only concerns GATHER; it cannot imply a new W4A16 feature.
    assert R.caps2(None)=={'gather':True} and 'w4a16' not in R.caps(caps)


def test_context_generation_and_opaque_session_fence_tokens():
    j,m,e,f,*_=host_fixture()
    wire=encode_descriptor(j)
    for ctx,gen in [(j.context+1,SESSION),(j.context,SESSION+1),(True,SESSION)]:
        with pytest.raises(TransportError):ProtectedFrontend(e,context=ctx,generation=gen)
    release=f.release_inputs(wire)
    j2,m2,e2,f2,*_=host_fixture()
    with pytest.raises(StaleResponse):f2.submit(wire,release)
    with pytest.raises(StaleResponse):e2.release_inputs(f.session,wire)
    with pytest.raises(StaleResponse):e.release_inputs(replace(f.session),wire)
    with pytest.raises(StaleResponse):f.submit(wire,replace(release))
    # A successful idle renegotiation revokes the prior session and its fence.
    fresh=ProtectedFrontend(e,context=j.context,generation=SESSION)
    with pytest.raises(StaleResponse):f.release_inputs(wire)
    with pytest.raises(StaleResponse):f.poll_completion()
    with pytest.raises(StaleResponse):fresh.submit(wire,release)
    release=fresh.release_inputs(wire);h=fresh.submit(wire,release)
    with pytest.raises(Busy):ProtectedFrontend(e,context=j.context,generation=SESSION)
    with pytest.raises(StaleResponse):f.acquire_completion()
    fresh.cancel(h);assert fresh.acquire_completion().status=='CANCELLED'


@pytest.mark.parametrize('role',['a','w','scales','y'])
@pytest.mark.parametrize('problem',['permission','owner','generation','short','unmapped'])
def test_release_checks_all_allocations_atomically(role,problem):
    j,m,e,f,*_=host_fixture()
    b=getattr(j,role);data=bytes(m._regions[b.address].data)
    m.unmap(b.address,context=j.context,generation=SESSION)
    if problem!='unmapped':
        m.map(b.address,data[:-1] if problem=='short' else data,
              context=j.context+(problem=='owner'),generation=SESSION+(problem=='generation'),
              permissions=('read',) if role=='y' and problem=='permission' else
                          ('write',) if problem=='permission' else ('read','write'))
    with pytest.raises(AccessDenied):f.release_inputs(encode_descriptor(j))
    assert not m._leases and e.transport.state=='IDLE' and e._last_epoch is None


@pytest.mark.parametrize('mutation',['same_bytes','different_bytes','padding','unmap_remap','rebind','permission'])
def test_input_change_after_release_requires_a_new_fence(mutation):
    j,m,e,f,*_=host_fixture(1);b=j.a
    # Larger mapping proves conservative whole-window content binding.
    data=m.read(b.address,b.span,context=j.context,generation=SESSION)+b'P'*30
    m.unmap(b.address,context=j.context,generation=SESSION)
    m.map(b.address,data,context=j.context,generation=SESSION,permissions=('read','write'))
    wire=encode_descriptor(j);release=f.release_inputs(wire)
    if mutation in ('same_bytes','different_bytes','padding'):
        addr=b.address+(16 if mutation=='padding' else 0)
        m.write(addr,b'P' if mutation=='padding' else data[:1] if mutation=='same_bytes' else b'X',
                context=j.context,generation=SESSION)
    elif mutation=='unmap_remap':
        m.unmap(b.address,context=j.context,generation=SESSION)
        m.map(b.address,data,context=j.context,generation=SESSION,permissions=('read','write'))
    else:m.rebind(b.address,context=j.context,generation=SESSION,new_generation=SESSION+1,
                  permissions=('write',) if mutation=='permission' else None)
    with pytest.raises(TransportError):f.submit(wire,release)
    assert e.transport.state=='IDLE' and e._last_epoch is None and not m._leases
    if mutation in ('same_bytes','different_bytes','padding','unmap_remap'):
        h=f.submit(wire,f.release_inputs(wire));f.cancel(h);f.acquire_completion()


def test_full_descriptor_binding_malformed_frames_and_consumed_fences():
    j,m,e,f,*_=host_fixture();wire=encode_descriptor(j);r=f.release_inputs(wire)
    for bad in [wire[:-1],wire+b'x',frame('{"x":1,"x":2}'),struct.pack('>I',4097),frame('null')]:
        with pytest.raises(DescriptorError):f.submit(bad,r)
    for field,value in [('request',j.request+1),('epoch',j.epoch+1),('global_bits',0),
                        ('position',j.position+1),('context',j.context+1)]:
        altered=encode_descriptor(replace(j,**{field:value}))
        with pytest.raises(TransportError):f.submit(altered,r)
    # A descriptor outside the bound context is refused even if its buffers exist.
    alien=replace(j,context=j.context+1)
    with pytest.raises(AccessDenied):f.release_inputs(encode_descriptor(alien))
    r2=f.release_inputs(wire)
    with pytest.raises(StaleResponse):f.submit(wire,r)
    h=f.submit(wire,r2)
    with pytest.raises(TransportError):f.submit(wire,r2)
    with pytest.raises(TransportError):f.release_inputs(wire)
    assert e._last_epoch==j.epoch
    f.cancel(h);f.acquire_completion()
    for epoch in (j.epoch,j.epoch-1,0):
        with pytest.raises(StaleResponse):f.release_inputs(encode_descriptor(replace(j,epoch=epoch,request=j.request+1)))
    next_job=replace(j,epoch=U64,request=j.request+1)
    h=accepted(next_job,e,f);f.cancel(h);f.acquire_completion()
    with pytest.raises(StaleResponse):f.release_inputs(encode_descriptor(replace(j,epoch=0,request=0)))


@pytest.mark.parametrize('phase',['released','partial','packet','wait','private','write','visible'])
def test_reset_pending_work_and_old_completion_ownership(phase):
    j,m,e,f,a,c,s=host_fixture();wire=encode_descriptor(j);r=f.release_inputs(wire)
    h=ticket=None;t=e.transport
    if phase!='released':h=f.submit(wire,r)
    if phase=='partial':t.prefetch(scale_ready=False)
    if phase=='packet':t.prefetch()
    if phase in ('wait','private','write','visible'):drain(t)
    if phase in ('private','write','visible'):deliver(j,t,a,c,s)
    if phase in ('write','visible'):ticket=t.issue_write()
    if phase=='visible':t.acknowledge_write(ticket)
    before=output(j,m);f.reset(SESSION+1)
    with pytest.raises(StaleResponse):f.submit(wire,r)
    if phase=='released':
        with pytest.raises(StaleResponse):f.acquire_completion()
    else:
        with pytest.raises(Busy):ProtectedFrontend(e,context=j.context,generation=SESSION+1)
        cpl=f.poll_completion()
        assert cpl.handle==h and cpl.status==('SUCCEEDED' if phase=='visible' else 'LOST_ON_RESET')
        assert output(j,m)==before and not m._writes
        if ticket:
            with pytest.raises(StaleResponse):t.acknowledge_write(ticket)
        assert f.acquire_completion()==cpl
    for role in ('a','w','scales','y'):
        m.rebind(getattr(j,role).address,context=j.context,generation=SESSION,new_generation=SESSION+1)
    fresh=ProtectedFrontend(e,context=j.context,generation=SESSION+1)
    j2=replace(j,generation=SESSION+1,epoch=0,request=j.request+1)
    h2=accepted(j2,e,fresh)
    with pytest.raises(StaleResponse):f.cancel(h2)
    fresh.cancel(h2);assert fresh.acquire_completion().status=='CANCELLED'


def test_fence_and_completion_order_with_backpressure_and_invalid_reset():
    j,m,e,f,a,c,s=host_fixture();wire=encode_descriptor(j)
    for g in (SESSION,SESSION-1,0,U64+1,True):
        with pytest.raises(TransportError):f.reset(g)
    m.write(j.a.address,struct.pack('<H',0x4000),context=j.context,generation=SESSION)
    a[0]=0x4000
    r=f.release_inputs(wire);h=f.submit(wire,r);t=e.transport
    assert f.poll_completion() is None and f.acquire_completion() is None
    drain(t);bits,_=deliver(j,t,a,c,s)
    assert f.poll_completion() is None and f.acquire_completion() is None and output(j,m)==SENTINEL
    assert t.issue_write(ready=False) is None
    ticket=t.issue_write()
    for _ in range(9):assert f.poll_completion() is None and output(j,m)==SENTINEL
    t.acknowledge_write(ticket)
    cpl=f.poll_completion();assert cpl.handle==h and cpl.status=='SUCCEEDED'
    assert output(j,m)==bits.to_bytes(4,'little')
    with pytest.raises(Busy):f.release_inputs(encode_descriptor(replace(j,epoch=j.epoch+1)))
    assert f.acquire_completion()==cpl and f.poll_completion() is None
    assert [x[0] for x in e.events]==['NEGOTIATED','INPUT_RELEASE','ACCEPTED','COMPLETION_ACQUIRE']
    events=[x[0] for x in t.events]
    assert events.index('Y_VISIBLE')<events.index('COMPLETION_RELEASE')<events.index('COMPLETION_ACQUIRE')


def test_output_commit_changes_shared_input_window_content_revision():
    j,m,_,a,c,s=fixture(1)
    # Input A and output Y have disjoint spans in the same authorized window.
    j=replace(j,a=Buffer(0x1000,2,2),y=Buffer(0x1004,4,4))
    m.map(0x1000,b'\x80\x3fXX'+SENTINEL,context=j.context,generation=SESSION,permissions=('read','write'))
    release_snapshot=m.input_snapshot(j)
    e=SimEndpoint(m,context=j.context,generation=SESSION)
    f=ProtectedFrontend(e,context=j.context,generation=SESSION)
    accepted(j,e,f);drain(e.transport);deliver(j,e.transport,a,c,s)
    ticket=e.transport.issue_write();e.transport.acknowledge_write(ticket);f.acquire_completion()
    assert m.input_snapshot(j)!=release_snapshot
    assert output(j,m)!=SENTINEL


def test_capability_revocation_and_counter_exhaustion_are_fail_closed():
    j,m,e,f,*_=host_fixture();wire=encode_descriptor(j);r=f.release_inputs(wire)
    e._capabilities=replace(CAPABILITIES,minor=1)
    with pytest.raises(CapabilityMismatch):f.submit(wire,r)
    assert e.transport.state=='IDLE' and not m._leases
    e._capabilities=CAPABILITIES;e._serial=U64
    with pytest.raises(Busy):f.release_inputs(wire)
    # Failed fence allocation did not replace a prior valid fence.
    h=f.submit(wire,r);f.cancel(h);assert f.acquire_completion().status=='CANCELLED'
    with pytest.raises(Busy):ProtectedFrontend(e,context=j.context,generation=SESSION)


def test_other_output_in_input_window_invalidates_host_release():
    from opentpu.nvfp4_transport import LocalJobTransport
    j,m,_,a,c,s=fixture(1)
    first=replace(j,a=Buffer(0x1000,2,2),y=Buffer(0x1004,4,4))
    m.map(0x1000,b'\x80\x3fXX'+SENTINEL*3,context=j.context,generation=SESSION,
          permissions=('read','write'))
    e=SimEndpoint(m,context=j.context,generation=SESSION)
    f=ProtectedFrontend(e,context=j.context,generation=SESSION)
    wire=encode_descriptor(first);r=f.release_inputs(wire)
    other=replace(j,y=Buffer(0x1008,4,4),request=j.request+1,epoch=j.epoch+1)
    device=LocalJobTransport(m,generation=SESSION)
    device.submit(encode_descriptor(other));drain(device);deliver(other,device,a,c,s)
    ticket=device.issue_write();device.acknowledge_write(ticket)
    device.take_completion(context=j.context)
    with pytest.raises(StaleResponse):f.submit(wire,r)
    h=f.submit(wire,f.release_inputs(wire));f.cancel(h);f.acquire_completion()


def test_capability_change_cannot_strand_an_accepted_completion():
    j,m,e,f,*_=host_fixture();h=accepted(j,e,f)
    e._capabilities=replace(CAPABILITIES,fences='unknown')
    # Accepted work still has a cancellation/drain route under its original token.
    f.cancel(h);assert f.poll_completion().status=='CANCELLED'
    assert f.acquire_completion().handle==h and not m._leases
    with pytest.raises(CapabilityMismatch):ProtectedFrontend(e,context=j.context,generation=SESSION)
    with pytest.raises(CapabilityMismatch):f.release_inputs(encode_descriptor(replace(j,epoch=j.epoch+1)))


@pytest.mark.parametrize('role',['a','w','scales','y'])
@pytest.mark.parametrize('change',['same_bytes','remap'])
def test_every_input_and_output_mapping_is_bound_after_release(role,change):
    j,m,e,f,*_=host_fixture();wire=encode_descriptor(j);r=f.release_inputs(wire)
    b=getattr(j,role);data=m.read(b.address,b.span,context=j.context,generation=SESSION)
    if change=='remap':
        m.unmap(b.address,context=j.context,generation=SESSION)
        m.map(b.address,data,context=j.context,generation=SESSION,permissions=('read','write'))
    else:m.write(b.address,data,context=j.context,generation=SESSION)
    if role=='y' and change=='same_bytes':
        # Output prior content is not an input dependency; its lifetime/permission is.
        h=f.submit(wire,r)
    else:
        with pytest.raises(StaleResponse):f.submit(wire,r)
        assert e._last_epoch is None and not m._leases
        newer=f.release_inputs(wire)
        with pytest.raises(StaleResponse):f.submit(wire,r)
        h=f.submit(wire,newer)
    f.cancel(h);f.acquire_completion()


def test_accepted_epoch_survives_renegotiation_and_failed_accept_does_not_advance():
    j,m,e,f,*_=host_fixture();wire=encode_descriptor(j);r=f.release_inputs(wire)
    e.transport._serial=U64
    with pytest.raises(Busy):f.submit(wire,r)
    assert e._last_epoch is None and not m._leases
    e.transport._serial=0
    newer=f.release_inputs(wire)
    with pytest.raises(StaleResponse):f.submit(wire,r)
    h=f.submit(wire,newer);f.cancel(h);f.acquire_completion()
    fresh=ProtectedFrontend(e,context=j.context,generation=SESSION)
    for epoch in (j.epoch-1,j.epoch):
        with pytest.raises(StaleResponse):fresh.release_inputs(encode_descriptor(replace(j,request=j.request+1,epoch=epoch)))
    h=accepted(replace(j,request=j.request+1,epoch=j.epoch+1),e,fresh)
    fresh.cancel(h);fresh.acquire_completion()
