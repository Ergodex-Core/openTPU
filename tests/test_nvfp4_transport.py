"""Simulated descriptor transport: permissions, ownership and visibility only."""
from dataclasses import replace
import json
import random
import struct

import pytest

from opentpu import nvfp4 as N
from opentpu.nvfp4_transport import (SimMemory,LocalJobTransport,Identity,DeviceResult,
    encode_descriptor,decode_descriptor,TransportError,DescriptorError,AccessDenied,
    Busy,StaleResponse,U64,MAX_DESCRIPTOR)
from test_nvfp4_local_job import job,SESSION

SENTINEL=b'\xde\xad\xbe\xef'


def fixture(k=17,mode=0,request=0,*,a=None,c=None,s=None):
    j=job(k,mode,request)
    a=[0x3f80]*k if a is None else a
    c=[2]*k if c is None else c
    s=[0x38]*((k+15)//16) if s is None else s
    m=SimMemory()
    for role,data in [('a',struct.pack('<'+'H'*k,*a)),('w',N.pack_codes(c)),
                      ('scales',bytes(s)),('y',SENTINEL)]:
        m.map(getattr(j,role).address,data,context=j.context,generation=j.generation,
              permissions=('read','write'))
    return j,m,LocalJobTransport(m,generation=j.generation),a,c,s


def output(j,m):return m.read(j.y.address,4,context=j.context,generation=j.generation)


def drain(t,rng=None):
    packets=[]
    for _ in range(1000):
        if t.state!='FETCH':break
        p=t.prefetch(operand_ready=rng is None or bool(rng.randrange(2)),
                     scale_ready=rng is None or bool(rng.randrange(3)==0))
        if p is not None and (rng is None or rng.randrange(2)):
            packets.append(p);t.accept_packet(p)
    assert t.state=='WAIT_RESULT'
    return packets


def deliver(j,t,a,c,s):
    try:
        bits=N.strict_row(a,N.pack_codes(c),s,j.global_bits,profile=j.profile)
        faults=0
    except N.NumericalFault:bits=0;faults=2
    result=DeviceResult(Identity.of(j),j.profile,bits,faults)
    t.receive_result(t._handle,result)
    return bits,faults


def frame(body):
    b=body.encode() if isinstance(body,str) else body
    return struct.pack('>I',len(b))+b


def test_frames_are_bounded_exact_and_strict():
    j,m,t,*_=fixture()
    encoded=encode_descriptor(j)
    assert decode_descriptor(encoded)==j
    invalid=[b'',b'\0\0\0',b'\0'*4,struct.pack('>I',MAX_DESCRIPTOR+1),
             encoded[:-1],encoded+b'x',frame(b'\xff'),frame('[]'),frame('null'),
             frame('{"x":1,"x":2}'),frame(encoded[4:].replace(b'"address":',b'"address":1,"address":',1)),
             frame(encoded[4:].replace(b'"k":17',b'"k":17.0')),
             frame(encoded[4:].replace(b'"k":17',b'"k":true')),
             frame(encoded[4:].replace(b'"k":17',b'"k":NaN')),
             frame(encoded[4:].replace(b'"k":17',b'"k":Infinity')),
             frame('['*1200+']'*1200)]
    for key,value in [('k',0),('k',65),('k',-1),('k',1<<32),('generation',SESSION+1),
                      ('transform','hadamard'),('profile',N.FAST_F24),('activation_format','GGUF'),
                      ('extra',1),('dependencies',[1]),('context',True)]:
        d=j.to_mapping();d[key]=value;invalid.append(frame(json.dumps(d)))
    d=j.to_mapping();del d['epoch'];invalid.append(frame(json.dumps(d)))
    for f in invalid:
        with pytest.raises(TransportError):t.submit(f)
        assert t.state=='IDLE' and not m._leases and not t.events and not m.audit
    # Rejection leaves all mappings usable; no partial allocation or credit loss.
    h=t.submit(encoded);t.cancel(h);assert t.take_completion(context=j.context).status=='CANCELLED'


@pytest.mark.parametrize('role',['a','w','scales','y'])
@pytest.mark.parametrize('problem',['absent','permission','owner','generation','short'])
def test_mapping_authorization_is_atomic(role,problem):
    j,m,t,*_=fixture()
    b=getattr(j,role);r=m._regions[b.address]
    data=bytes(r.data);m.unmap(b.address,context=j.context,generation=j.generation)
    if problem!='absent':
        m.map(b.address,data[:-1] if problem=='short' else data,
              context=j.context+(problem=='owner'),generation=j.generation+(problem=='generation'),
              permissions=('read',) if role=='y' and problem=='permission' else
                          ('write',) if problem=='permission' else ('read','write'))
    with pytest.raises(AccessDenied):t.submit(encode_descriptor(j))
    assert t.state=='IDLE' and not m._leases and all(not r.leases for r in m._regions.values())


def test_memory_limits_bounds_and_no_generation_wrap():
    m=SimMemory();kw=dict(context=1,generation=2)
    m.map(U64-3,b'abcd',**kw,permissions=('read','write'))
    assert m.read(U64-3,4,**kw)==b'abcd'
    for addr,n in [(U64-3,5),(U64,2),(U64-4,2),(U64-3,0),(-1,1),(True,1)]:
        with pytest.raises(TransportError):m.read(addr,n,**kw)
    for other in [dict(context=2,generation=2),dict(context=1,generation=3),dict(context=True,generation=2)]:
        with pytest.raises(TransportError):m.read(U64-3,1,**other)
    for meth in (m.unmap,m.rebind):
        with pytest.raises(TransportError):
            meth(U64-3,context=True,generation=2,**({'new_generation':3} if meth==m.rebind else {}))
    with pytest.raises(AccessDenied):m.map(U64,b'ab',**kw)
    with pytest.raises(AccessDenied):m.map(U64-2,b'a',**kw)
    with pytest.raises(AccessDenied):m.map(0,b'a'*4097,**kw)
    with pytest.raises(AccessDenied):m.map(0,b'a',**kw,permissions=('execute',))
    m.map(0,b'ab',**kw);m.map(2,b'cd',**kw)
    with pytest.raises(AccessDenied):m.read(0,4,**kw)
    with pytest.raises(AccessDenied):m.rebind(0,**kw,new_generation=1)
    m.rebind(0,**kw,new_generation=U64)
    with pytest.raises(AccessDenied):m.rebind(0,context=1,generation=U64,new_generation=0)
    for i in range(m.MAX_REGIONS-3):m.map(100+2*i,b'x',**kw)
    with pytest.raises(Busy):m.map(1000,b'x',**kw)


@pytest.mark.parametrize('k',[1,2,15,16,17,31,32,33,63,64])
def test_tile_snapshot_delayed_scale_tail_and_free_event(k):
    j,m,t,a,c,s=fixture(k)
    # Ignored high nibble of an odd final byte is deliberately nonzero.
    if k%2:m._regions[j.w.address].data[-1]|=0xf0
    h=t.submit(encode_descriptor(j))
    for i in range((k+15)//16):
        before=len(m.audit)
        for _ in range(5):assert t.prefetch(operand_ready=False,scale_ready=False) is None
        assert len(m.audit)==before
        assert t.prefetch(scale_ready=False) is None
        assert len(m.audit)==before+2
        for _ in range(4):assert t.prefetch(scale_ready=False) is None
        assert len(m.audit)==before+2
        p=t.prefetch(operand_ready=False)
        n=min(16,k-16*i)
        assert p.handle==h and p.index==i and p.group==0 and p.scale_index==i
        assert p.scale_epoch==j.epoch and p.mask==(1<<n)-1
        assert p.activations==tuple(a[16*i:16*i+n])+ (0,)*(16-n)
        assert p.codes==tuple(c[16*i:16*i+n])+ (0,)*(16-n)
        for _ in range(8):assert t.prefetch() is p
        assert len(m.audit)==before+3 and t.completion is None and output(j,m)==SENTINEL
        for bad in [replace(p,index=i+1),replace(p,scale_epoch=j.epoch^1),replace(p,slot=p.slot+1),replace(p)]:
            with pytest.raises(StaleResponse):t.accept_packet(bad)
        assert t.packet is p
        t.accept_packet(p)
        assert list(t.events)[-1]==('BUFFER_FREE',p.slot,0,i,j.epoch)
        with pytest.raises(StaleResponse):t.accept_packet(p)
    assert t.state=='WAIT_RESULT'
    with pytest.raises(Busy):m.write(j.y.address,b'1234',context=j.context,generation=j.generation)
    bits,_=deliver(j,t,a,c,s)
    assert t.issue_write(ready=False) is None and output(j,m)==SENTINEL and t.completion is None
    ticket=t.issue_write()
    for _ in range(5):assert t.issue_write() is ticket
    assert output(j,m)==SENTINEL and t.completion is None
    t.acknowledge_write(ticket)
    assert output(j,m)==struct.pack('<I',bits) and t.completion.status=='SUCCEEDED'
    assert [e[0] for e in t.events][-2:]==['Y_VISIBLE','COMPLETION_RELEASE']
    with pytest.raises(Busy):t.submit(encode_descriptor(replace(j,request=j.request+1)))
    with pytest.raises(AccessDenied):t.take_completion(context=j.context+1)
    with pytest.raises(Busy):m.unmap(j.a.address,context=j.context,generation=j.generation)
    cpl=t.take_completion(context=j.context)
    assert cpl.handle==h and t.lookup(h)==cpl and t.state=='IDLE'
    with pytest.raises(StaleResponse):t.acknowledge_write(ticket)
    with pytest.raises(DescriptorError):t.submit(encode_descriptor(j))
    m.write(j.y.address,SENTINEL,context=j.context,generation=j.generation)


@pytest.mark.parametrize('action',['cancel','reset'])
@pytest.mark.parametrize('phase',['empty','partial','packet','wait','private','write','complete'])
def test_cancel_reset_and_stale_reuse(action,phase):
    j,m,t,a,c,s=fixture(17)
    h=t.submit(encode_descriptor(j));oldpacket=None;ticket=None
    if phase=='partial':t.prefetch(scale_ready=False)
    if phase=='packet':oldpacket=t.prefetch()
    if phase in ('wait','private','write','complete'):drain(t)
    if phase in ('private','write','complete'):deliver(j,t,a,c,s)
    if phase in ('write','complete'):ticket=t.issue_write()
    if phase=='complete':t.acknowledge_write(ticket)
    before=output(j,m)
    if action=='reset':t.reset(SESSION+1)
    elif phase=='complete':
        with pytest.raises(Busy):t.cancel(h)
    else:t.cancel(h)
    status='SUCCEEDED' if phase=='complete' else 'LOST_ON_RESET' if action=='reset' else 'CANCELLED'
    assert t.completion.status==status and t.completion.handle.identity==Identity.of(j)
    assert output(j,m)==before and not m._writes
    with pytest.raises(StaleResponse):t.receive_result(h,DeviceResult(Identity.of(j),j.profile,0))
    if ticket:
        with pytest.raises(StaleResponse):t.acknowledge_write(ticket)
    if oldpacket:
        with pytest.raises(StaleResponse):t.accept_packet(oldpacket)
    t.take_completion(context=j.context)
    if action=='reset':
        for role in ('a','w','scales','y'):
            m.rebind(getattr(j,role).address,context=j.context,generation=SESSION,new_generation=SESSION+1)
    j2=replace(j,generation=t.generation,request=j.request+1,epoch=j.epoch+1)
    h2=t.submit(encode_descriptor(j2))
    assert h2!=h
    with pytest.raises(StaleResponse):t.receive_result(h,DeviceResult(Identity.of(j),j.profile,0))
    if ticket:
        with pytest.raises(StaleResponse):t.acknowledge_write(ticket)
    assert t.state=='FETCH'
    t.cancel(h2);t.take_completion(context=j.context)


def test_result_validation_leases_and_wrong_transport():
    j,m,t,a,c,s=fixture();h=t.submit(encode_descriptor(j))
    good=DeviceResult(Identity.of(j),j.profile,0)
    with pytest.raises(StaleResponse):t.receive_result(h,good)
    for role in ('a','w','scales','y'):
        addr=getattr(j,role).address
        with pytest.raises(Busy):m.write(addr,b'x',context=j.context,generation=j.generation)
        with pytest.raises(Busy):m.rebind(addr,context=j.context,generation=j.generation,new_generation=SESSION+1)
    t2=LocalJobTransport(m,generation=SESSION)
    with pytest.raises(Busy):t2.submit(encode_descriptor(j))
    drain(t)
    for bad in [replace(good,identity=replace(good.identity,epoch=j.epoch^1)),
                replace(good,profile=N.FAST_F24),replace(good,bits=0x7f800000),
                replace(good,bits=0x7fc12345),replace(good,bits=-1),
                replace(good,bits=True),replace(good,faults=256),replace(good,faults=1,bits=1)]:
        with pytest.raises(TransportError):t.receive_result(h,bad)
        assert t.state=='WAIT_RESULT' and output(j,m)==SENTINEL and t.completion is None
    t.receive_result(h,replace(good,faults=2))
    assert t.completion.status=='FAILED' and t.completion.faults==2 and output(j,m)==SENTINEL
    t.take_completion(context=j.context)
    h2=t2.submit(encode_descriptor(j));assert h2.transport!=h.transport
    with pytest.raises(StaleResponse):t2.cancel(h)
    t2.cancel(h2);t2.take_completion(context=j.context)
    for bad in [SESSION,SESSION-1,-1,True,1<<64]:
        with pytest.raises(TransportError):t.reset(bad)
    assert t.generation==SESSION
    t.reset(U64)
    with pytest.raises(TransportError):t.reset(0)
    with pytest.raises(TransportError):LocalJobTransport(object(),generation=SESSION)


@pytest.mark.parametrize('seed',[81,182,283])
def test_randomized_stalls_and_ordered_visibility(seed):
    rng=random.Random(seed)
    for request in range(90):
        k=rng.randrange(1,65);mode=request%2
        a=[rng.randrange(0x7f80)|(rng.randrange(2)<<15) for _ in range(k)]
        c=[rng.randrange(16) for _ in range(k)]
        s=[rng.randrange(128) for _ in range((k+15)//16)]
        j,m,t,_,_,_=fixture(k,mode,request,a=a,c=c,s=s)
        h=t.submit(encode_descriptor(j));packets=drain(t,rng)
        assert [p.index for p in packets]==list(range((k+15)//16))
        bits,faults=deliver(j,t,a,c,s)
        if not faults:
            for _ in range(rng.randrange(10)):assert t.issue_write(ready=False) is None
            ticket=t.issue_write()
            for _ in range(rng.randrange(10)):
                assert t.issue_write() is ticket and output(j,m)==SENTINEL and t.completion is None
            t.acknowledge_write(ticket)
            assert output(j,m)==struct.pack('<I',bits)
        else:assert output(j,m)==SENTINEL
        cpl=t.take_completion(context=j.context)
        assert cpl.handle==h and cpl.status==('FAILED' if faults else 'SUCCEEDED')
        events=[e[0] for e in t.events]
        assert events.count('COMPLETION_RELEASE')==1 and events.count('COMPLETION_ACQUIRE')==1
        if not faults:assert events.index('PRIVATE_RESULT')<events.index('Y_VISIBLE')<events.index('COMPLETION_RELEASE')
        assert not m._writes and not m._leases


def test_independent_memories_cannot_acknowledge_each_other():
    j,m,t,a,c,s=fixture();j2,m2,t2,*_=fixture()
    h=t.submit(encode_descriptor(j));h2=t2.submit(encode_descriptor(j2))
    assert h!=h2
    with pytest.raises(StaleResponse):t2.cancel(h)
    drain(t);drain(t2);deliver(j,t,a,c,s);deliver(j2,t2,a,c,s)
    ticket=t.issue_write();ticket2=t2.issue_write()
    assert ticket!=ticket2
    with pytest.raises(StaleResponse):t2.acknowledge_write(ticket)
    with pytest.raises(StaleResponse):t.acknowledge_write(ticket2)
    with pytest.raises(StaleResponse):m2.read_input(t._lease,'a',0,2)
    with pytest.raises(StaleResponse):m2.commit_output(t2._lease,ticket)
    assert output(j,m)==SENTINEL and output(j2,m2)==SENTINEL
    t.acknowledge_write(ticket);t2.acknowledge_write(ticket2)
    assert t.take_completion(context=j.context).status=='SUCCEEDED'
    assert t2.take_completion(context=j2.context).status=='SUCCEEDED'


def test_back_to_back_reuse_history_eviction_and_nonce_exhaustion():
    j,m,t,a,c,s=fixture(1)
    oldh=oldt=oldp=None
    for index in range(40):
        current=replace(j,request=j.request+index,epoch=j.epoch+index)
        h=t.submit(encode_descriptor(current));p=t.prefetch();t.accept_packet(p)
        deliver(current,t,a,c,s);ticket=t.issue_write()
        if index==0:oldh,oldt,oldp=h,ticket,p
        else:
            with pytest.raises(StaleResponse):t.receive_result(oldh,DeviceResult(Identity.of(j),j.profile,0))
            with pytest.raises(StaleResponse):t.acknowledge_write(oldt)
            with pytest.raises(StaleResponse):t.accept_packet(oldp)
        t.acknowledge_write(ticket);t.take_completion(context=j.context)
    assert len(t.history)==32 and t.lookup(oldh) is None
    # Reuse a fully evicted logical identity: opaque handles/tickets still differ.
    h=t.submit(encode_descriptor(j));assert h!=oldh
    drain(t);deliver(j,t,a,c,s);ticket=t.issue_write();assert ticket!=oldt
    with pytest.raises(StaleResponse):t.acknowledge_write(oldt)
    t.acknowledge_write(ticket);t.take_completion(context=j.context)
    # Exhaustion is terminal/rejected, never silently wrapped into a stale nonce.
    t._serial=U64
    with pytest.raises(Busy):t.submit(encode_descriptor(replace(j,request=0)))
    assert not m._leases
    t._serial=42;t._slot_serial=U64
    h=t.submit(encode_descriptor(replace(j,request=0)));assert t.prefetch() is None
    assert t.completion.status=='FAILED';t.take_completion(context=j.context)
    t._slot_serial=42
    h=t.submit(encode_descriptor(replace(j,request=1)));drain(t);deliver(replace(j,request=1),t,a,c,s)
    m._nonce=U64
    assert t.issue_write() is None and t.completion.status=='FAILED' and not m._writes
    t.take_completion(context=j.context)
    with pytest.raises(TransportError):t.submit(encode_descriptor(replace(j,request=2)))
    assert t.state=='IDLE' and not m._leases


def test_declared_span_bounds_inside_larger_mapped_window():
    j,m,t,a,c,s=fixture(1)
    m.unmap(j.a.address,context=j.context,generation=j.generation)
    m.map(j.a.address,b'\x80\x3f'+b'P'*62,context=j.context,generation=j.generation,
          permissions=('read','write'))
    h=t.submit(encode_descriptor(j));lease=t._lease
    assert m.read_input(lease,'a',0,2)==b'\x80\x3f'
    for role,offset,length in [('a',0,3),('a',2,1),('a',63,1),('a',0,0),
                               ('a',-1,1),('a',True,1),('y',0,1),('unknown',0,1)]:
        with pytest.raises(TransportError):m.read_input(lease,role,offset,length)
    # Even unused mapped padding is held until terminal completion acquisition.
    with pytest.raises(Busy):m.write(j.a.address+32,b'X',context=j.context,generation=j.generation)
    t.cancel(h)
    with pytest.raises(Busy):m.write(j.a.address+32,b'X',context=j.context,generation=j.generation)
    t.take_completion(context=j.context)
    m.write(j.a.address+32,b'X',context=j.context,generation=j.generation)
    with pytest.raises(StaleResponse):m.read_input(lease,'a',0,2)
