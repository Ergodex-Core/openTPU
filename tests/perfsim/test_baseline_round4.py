"""Frozen timing observations; numerical suitability is independently reported by the tools."""
import base64,hashlib,json,os,zlib
from pathlib import Path
import numpy as np
import pytest
from opentpu import Config,isa as I
from opentpu.perfsim import Simulator,Architecture,MemoryConfig
from opentpu.perfsim.ports import NativeMemorySystem
from tools.perfsim.compare_native import port_schedule

ROOT=Path(__file__).resolve().parents[2];FIX=Path(__file__).parent/'fixtures'
SHAPES=json.loads((FIX/'rtl_shapes_round4.json').read_text())
PERSISTENT=json.loads((FIX/'rtl_persistent_round4.json').read_text())
BRIDGE=os.environ.get('OTPU_RAMULATOR_BRIDGE')
pytestmark=pytest.mark.skipif(not BRIDGE,reason='requires pinned Ramulator; no fallback')


def unpack(case):
    raw=zlib.decompress(base64.b64decode(case['image_be_u32_zlib_base64']))
    assert hashlib.sha256(raw).hexdigest()==case['image_sha256']
    words=case['program_words']
    return Config(**case['input']['config']),[I.Instr.decode(words[k:k+8]) for k in range(0,len(words),8)],np.frombuffer(raw,'>u4').astype('<u4').view(np.uint8)


def setup(profile):
    base=ROOT/'tools/perfsim/configs';p=json.loads((base/(profile+'.json')).read_text());mc=p['memory']
    mc.update(bridge=BRIDGE,record_trace=True,ramulator_config=str(base/mc['ramulator_config']))
    return Architecture(**p['architecture']),NativeMemorySystem(MemoryConfig(**mc))


def trace(result,case):
    assert result['cycles']==case['rtl_cycles']
    keys=('pc','dispatch','start','end')
    assert [{k:r[k] for k in keys} for r in result['instructions']]==[{k:r[k] for k in keys} for r in case['rtl_trace']]


def port_hash(memory):return hashlib.sha256(json.dumps(port_schedule(memory.controller.events),sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize('case',SHAPES['cases'],ids=lambda c:c['name'])
def test_frozen_operator_timing_not_numeric_suitability(case):
    cfg,program,image=unpack(case);arch,memory=setup(case['profile'])
    try:
        sim=Simulator(cfg,program,image,architecture=arch,memory_system=memory);result=sim.run();trace(result,case)
        assert port_hash(memory)==case['port_sha256']
        peaks=[r['work'].get('peak_weight_fifo_chunks',0) for r in result['instructions']]
        assert max(peaks)<=arch.mxu_fifo_chunks
        if case['name']=='d32_gemma_mlp_down':assert max(peaks)>864 # a partial fifth row must fit
    finally:memory.close()


@pytest.mark.parametrize('case',PERSISTENT['cases'],ids=lambda c:c['name'])
def test_frozen_persistent_launches_and_retained_state(case):
    arch,memory=setup(case['profile']);previous=None;start=0
    try:
        for index,stage in enumerate(case['stages']):
            cfg,program,image=unpack(stage);seed=image.copy() if previous is None else previous.dram.copy()
            if previous is not None:
                for addr,size in stage['input']['metadata'].get('host_inputs',[]):seed[addr:addr+size]=image[addr:addr+size]
            sim=Simulator(cfg,program,seed,architecture=arch,memory_system=memory,args=stage['input'].get('args'))
            if previous is not None:
                for key in ('tmem','act','ascale','adoff'):getattr(sim.state,key)[:]=getattr(previous,key)
            assert sim.memory_offset==start==stage['rtl_global_start']
            result=sim.run();trace(result,stage)
            assert hashlib.sha256(sim.state.dram.tobytes()).hexdigest()==stage['dram_sha256']
            previous=sim.state;start+=result['cycles']+8+case['gap']
            if index+1<len(case['stages']):
                for cycle in range(memory.cycle+1,start):memory.begin_cycle(cycle,arch.core_period_ps)
        assert port_hash(memory)==case['port_sha256']
    finally:memory.close()
