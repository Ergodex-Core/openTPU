"""Replay real simulated-memory tile packets through the strict RTL controller.

This is a bounded file-backed simulation bridge, not live DMA/co-simulation.
Actual RTL private results then drive the transport's staged write/completion API.
"""
import random
import re

import pytest

from opentpu import nvfp4 as N
from opentpu.mxu_job import local_job_fields
from opentpu.host.nvfp4 import SimEndpoint,ProtectedFrontend
from opentpu.nvfp4_transport import DeviceResult,Identity,encode_descriptor
from test_nvfp4_transport import fixture,drain,output,SENTINEL
from test_nvfp4_local_job import TB,desc_pack,packet_pack,expected
from test_nvfp4_scale_acc_rtl import simulate,packed,ROOT,FP,ACC
from test_nvfp4_strict_rtl import STRICT


def packet_fields(p):
    i=p.handle.identity
    return dict(a=p.activations,w=p.codes,mask=p.mask,scale=p.scale,
                first=int(p.first),last=int(p.last),group=p.group,k=p.index,
                scale_group=p.scale_group,scale_k=p.scale_index,
                context=i.context,generation=i.generation,request=i.request,
                position=i.position,epoch=i.epoch,scale_epoch=p.scale_epoch)


def bridge_cases():
    rng=random.Random(347)
    rows=[]
    for mode in (0,1):
        for k in range(1,65):
            a=[(rng.randrange(110,141)<<7)|rng.randrange(128)|(rng.randrange(2)<<15)
               for _ in range(k)]
            c=[rng.randrange(16) for _ in range(k)]
            s=[rng.randrange(127) for _ in range((k+15)//16)]
            rows.append(fixture(k,mode,len(rows),a=a,c=c,s=s))
        for a,c,s in [([0x7f80],[0],[0x38]),([0x7fc1],[2],[0x38]),
                      ([0x7f7f]*16,[7]*16,[0]),([0xff7f]*16,[7]*16,[0x38]),
                      ([0x3f80],[2],[0x7f]),([1]*16,[1]*16,[0x38]),
                      ([0x8000]*16,[2]*16,[0x38]),
                      ([0x4c00,0x3f80,0xcc00,0x3f80],[2]*4,[0x38])]:
            rows.append(fixture(len(a),mode,len(rows),a=a,c=c,s=s))
    return rows


def bridge_tb(count):
    # Reuse the existing interface/stability assertions; its separate reset and
    # generation phase sweep remains covered by test_nvfp4_local_job.py.
    start=TB.index('    // Sweep reset')
    end=TB.index('    for(integer j=0;j<COUNT;j=j+1)')
    tb=TB[:start]+'    rst=0;\n'+TB[end:]
    tb=tb.replace('        if(!result_success) failed=failed+1;',
                  '        $display("OBSERVED j=%0d bits=%08h fault=%02h",j,result_bits,result_fault);\n'
                  '        if(!result_success) failed=failed+1;')
    tb=tb.replace('if(rejected<25 || completed<100 || failed<10 || packet_wait<100 || result_stalls<100)',
                  'if(rejected!=0 || completed!=COUNT || failed<10 || packet_wait<100 || result_stalls<100)')
    tb=tb.replace('TEST_PASS local_jobs=', 'TEST_PASS transport_bridge_jobs=')
    tb=tb.replace('reset_phases=45 generation_phases=45 ', '')
    return tb.replace('@COUNT@',str(count))


@pytest.mark.parametrize('backend',['iverilog','verilator'])
def test_simulated_transport_rtl_bridge(tmp_path,backend):
    rows=bridge_cases();ds=[];ps=[];wants=[];hosts=[];active=[]
    rng=random.Random(498)
    for j,m,t,a,c,s in rows:
        endpoint=SimEndpoint(m,context=j.context,generation=j.generation)
        host=ProtectedFrontend(endpoint,context=j.context,generation=j.generation)
        wire=encode_descriptor(j)
        host.submit(wire,host.release_inputs(wire))
        t=endpoint.transport;hosts.append(host);active.append((j,m,t,a,c,s))
        packets=[packet_fields(p) for p in drain(t,rng)]
        d=local_job_fields(j,j.generation)
        accepted,bits,fault=expected(d,packets)
        assert accepted
        wants.append((bits,fault))
        ds.append(f"{packed([(desc_pack(d),1641),(1,1),(bits,32),(int(fault),1)]):0419x}")
        ps.extend(f'{packet_pack(p):0215x}' for p in packets)
        ps.extend('0'*215 for _ in range(4-len(packets)))
        assert output(j,m)==SENTINEL and t.completion is None
    (tmp_path/'descriptors.hex').write_text('\n'.join(ds)+'\n')
    (tmp_path/'packets.hex').write_text('\n'.join(ps)+'\n')
    sources=[FP,ROOT/'rtl/mxu/otpu_nvfp4_job.sv',STRICT,ACC,ROOT/'rtl/mxu/otpu_nvfp4_local_job.sv']
    simulate(tmp_path,backend,sources,bridge_tb(len(rows)))
    observed=re.findall(r'OBSERVED j=(\d+) bits=([0-9a-fA-F]+) fault=([0-9a-fA-F]+)',
                        (tmp_path/'simulation.log').read_text())
    assert len(observed)==len(rows)
    for index,(idx,bits,faults) in enumerate(observed):
        assert int(idx)==index
        bits=int(bits,16);faults=int(faults,16)
        j,m,t,a,c,s=active[index];host=hosts[index]
        assert (bits,bool(faults))==wants[index]
        t.receive_result(t._handle,DeviceResult(Identity.of(j),j.profile,bits,faults))
        if faults:
            assert t.completion.status=='FAILED' and output(j,m)==SENTINEL
        else:
            assert t.issue_write(ready=False) is None and host.poll_completion() is None
            ticket=t.issue_write()
            assert output(j,m)==SENTINEL and host.acquire_completion() is None
            t.acknowledge_write(ticket)
            assert output(j,m)==bits.to_bytes(4,'little') and t.completion.status=='SUCCEEDED'
        cpl=host.acquire_completion()
        assert cpl.handle.identity==Identity.of(j) and not m._leases and not m._writes
    print(f'TEST_PASS transport_memory_visibility={len(observed)} backend={backend}')
