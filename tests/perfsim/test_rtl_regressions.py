"""Frozen observations from unmodified RTL; never regenerate expectations from the model."""
import json
from pathlib import Path
import numpy as np
import pytest
from opentpu import Config,isa as I
from opentpu.perfsim import Simulator,MemoryConfig,Unsupported
from opentpu.perfsim.memory import MemorySystem

FIXTURE=json.loads((Path(__file__).parent/'fixtures/rtl_fixed_d32_l8.json').read_text())

@pytest.mark.parametrize('case',FIXTURE['cases'],ids=lambda c:c['name'])
def test_frozen_rtl_cycle_and_instruction_trace(case):
    cfg=Config(**FIXTURE['configuration'])
    words=case['program_words']
    program=[I.Instr.decode(words[k:k+8]) for k in range(0,len(words),8)]
    memory=MemoryConfig(backend='rtl_fixed',controller_period_ps=5000,memory_period_ps=5000,
                        transaction_bytes=32,simple_latency_ticks=8,rmw=False)
    sim=Simulator(cfg,program,np.arange(1024,dtype=np.float32).view(np.uint8),memory=memory)
    result=sim.run()
    assert result['cycles']==case['rtl_cycles']
    keys=('pc','dispatch','start','end')
    assert [{k:r[k] for k in keys} for r in result['instructions']]==[{k:r[k] for k in keys} for r in case['rtl_trace']]


def test_coupled_mxu_ports_are_atomic_under_backpressure():
    m=MemorySystem(MemoryConfig(outstanding=4,transaction_bytes=16,record_trace=True))
    m.begin_cycle(0,5000)
    assert m.request(0,16,False,'SW',0) is not None
    # Each 32-byte port needs two slots. Neither request may be partially accepted.
    assert m.request_many([(32,32,False,'A'),(64,32,False,'B')],1) is None
    assert len(m.queue)==1 and m.ports=={'SW'}
    assert m.stats['logical_read_bytes']==16
    for cycle in range(1,30): m.begin_cycle(cycle,5000)
    pair=m.request_many([(32,32,False,'A'),(64,32,False,'B')],1)
    assert pair is not None and len(m.queue)==4
    for cycle in range(30,60): m.begin_cycle(cycle,5000)
    assert all(t.done for t in pair) and m.idle
    m.close()


def test_fixed_memory_rejects_incompatible_contracts():
    with pytest.raises(Unsupported,match='masked writes'):
        MemorySystem(MemoryConfig(backend='rtl_fixed'))
    with pytest.raises(Unsupported,match='equal D'):
        Simulator(Config(),[I.halt()],np.zeros(4,np.uint8),memory=MemoryConfig(backend='rtl_fixed',rmw=False))
