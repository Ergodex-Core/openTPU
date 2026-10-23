"""Immutable observed RTL schedules; Ramulator tests explicitly require the pinned library."""
import base64
import hashlib
import json
import os
import zlib
from pathlib import Path
import numpy as np
import pytest
from opentpu import Config, isa as I
from opentpu.perfsim import Simulator, Architecture, MemoryConfig
from opentpu.perfsim.ports import NativeMemorySystem
from tools.perfsim.compare_native import port_schedule

ROOT=Path(__file__).resolve().parents[2]
FIX=Path(__file__).parent/'fixtures'
NATIVE=json.loads((FIX/'rtl_native_round3.json').read_text())
COVERAGE=json.loads((FIX/'rtl_coverage_round3.json').read_text())
BRIDGE=os.environ.get('OTPU_RAMULATOR_BRIDGE')


def unpack(case):
    raw=zlib.decompress(base64.b64decode(case['image_be_u32_zlib_base64']))
    assert hashlib.sha256(raw).hexdigest()==case['image_sha256']
    image=np.frombuffer(raw,dtype='>u4').astype('<u4').view(np.uint8)
    words=case['program_words']
    return Config(**case['input']['config']),[I.Instr.decode(words[k:k+8]) for k in range(0,len(words),8)],image


def check_trace(result,case):
    assert result['cycles']==case['rtl_cycles']
    keys=('pc','dispatch','start','end')
    assert [{k:r[k] for k in keys} for r in result['instructions']]==[{k:r[k] for k in keys} for r in case['rtl_trace']]


@pytest.mark.parametrize('case',COVERAGE['cases'],ids=lambda c:c['name'])
def test_remaining_opcode_and_shape_observations(case):
    cfg,program,image=unpack(case)
    memory=MemoryConfig(backend='rtl_fixed',controller_period_ps=5000,memory_period_ps=5000,transaction_bytes=32,simple_latency_ticks=8,rmw=False)
    check_trace(Simulator(cfg,program,image,memory=memory).run(),case)


@pytest.mark.skipif(not BRIDGE,reason='set OTPU_RAMULATOR_BRIDGE to the built pinned library; no emulation')
@pytest.mark.parametrize('case',NATIVE['cases'],ids=lambda c:c['name'])
def test_native_closed_loop_observed_schedule(case):
    cfg,program,image=unpack(NATIVE['bundles'][case['bundle']])
    directory=ROOT/'tools/perfsim/configs'
    profile=json.loads((directory/(case['profile']+'.json')).read_text())
    mc=profile['memory'];mc['bridge']=BRIDGE;mc['record_trace']=True
    mc['ramulator_config']=str(directory/mc['ramulator_config'])
    memory=NativeMemorySystem(MemoryConfig(**mc))
    try:
        result=Simulator(cfg,program,image,memory_system=memory,architecture=Architecture(**profile['architecture'])).run()
        check_trace(result,case)
        observed=hashlib.sha256(json.dumps(port_schedule(memory.controller.events),sort_keys=True).encode()).hexdigest()
        assert observed==case['port_schedule_sha256']
        for key in ('read_bytes','write_bytes','read_transactions','write_transactions'):
            assert memory.stats.get(key,0)==case['statistics'].get(key,0)
    finally:
        memory.close()


def test_rtl_driver_scale_line_alignment_and_mask_rejection(tmp_path):
    from tools.perfsim.rtl.memory_driver import Driver
    profile={'architecture':{'core_period_ps':1000},'memory':{'native_ports':True,'record_trace':True}}
    (tmp_path/'profile.json').write_text(json.dumps(profile))
    np.arange(32,dtype='>u4').tofile(tmp_path/'dram_0.bin')
    driver=Driver(tmp_path/'profile.json',tmp_path,32,64)
    empty=(0,0,0,0,0,bytes(128))
    try:
        ready,*_=driver.step(0,(1,0,12,0,0,bytes(128)),empty,empty)
        assert ready&1
        event=driver.controller.events[0]
        assert (event['address'],event['size'])==(0,32)
        for cycle in range(1,100):
            result=driver.step(cycle,empty,empty,empty)
            if result[1]:break
        assert result[1] and result[2]==np.arange(8,dtype='<u4').tobytes()
        with pytest.raises(ValueError,match='non-contiguous'):
            driver.step(101,empty,empty,(1,1,32,5,0,bytes(128)))
        with pytest.raises(ValueError,match='outside DRAM'):
            driver.step(102,empty,(1,0,256,0,0,bytes(128)),empty)
    finally:
        driver.close()


def test_layer_iteration_labels_do_not_merge_loop_visits():
    program=[I.loop(1,2),I.vop(I.V_FILL,0,0,0,1,8,8,8,0,I.B_SCALAR,2),I.halt()]
    sim=Simulator(Config(),program,np.zeros(4,np.uint8),labels={'1':'decoder.layer_{iteration}'})
    result=sim.run()
    assert set(result['layers'])=={'decoder.layer_0','decoder.layer_1'}
    assert [r['loop_context'][0]['iteration'] for r in result['instructions']]==[0,1]
    assert all(r['instructions']==1 for r in result['layers'].values())


@pytest.mark.parametrize('native',[False,True])
def test_persistent_launch_uses_global_completion_clock(native):
    from opentpu.perfsim.ports import make_memory
    cfg,program,image=unpack(NATIVE['bundles']['gemm'])
    memory=make_memory(MemoryConfig(native_ports=native))
    try:
        results=[Simulator(cfg,program,image,memory_system=memory).run() for _ in range(3)]
        # Simple backend has no row-state or refresh history: offset alone must not
        # change latency, or compare local core cycles against global completion cycles.
        assert len({r['cycles'] for r in results})==1
        assert memory.cycle>2*results[0]['cycles']
    finally:
        memory.close()
