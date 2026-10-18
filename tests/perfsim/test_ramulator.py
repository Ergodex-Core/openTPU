"""Explicit opt-in integration tests: an absent library is skipped, never emulated."""
import os
from pathlib import Path
import pytest
from opentpu.perfsim import MemoryConfig
from opentpu.perfsim.memory import MemorySystem

BRIDGE=os.environ.get('OTPU_RAMULATOR_BRIDGE')
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.skipif(not BRIDGE,reason='set OTPU_RAMULATOR_BRIDGE to the built pinned library')
@pytest.mark.parametrize('name,period,tx',[('hbm2',1000,16),('ddr4',833,64)])
def test_real_ramulator_backpressure_read_write_refresh(name,period,tx):
    cfg=MemoryConfig(backend='ramulator',bridge=BRIDGE,
                     ramulator_config=str(ROOT/f'tools/perfsim/configs/{name}.yaml'),
                     controller_period_ps=period,memory_period_ps=period,transaction_bytes=tx,
                     outstanding=16,record_trace=True)
    m=MemorySystem(cfg)
    tickets=[]
    try:
        for cycle in range(20000):
            m.begin_cycle(cycle,period)
            if len(tickets)<200:
                # alternate row-local reads, writes and row changes. Run across refresh.
                i=len(tickets); ticket=m.request((i%4)*4096+(i//4)*tx,tx,i%3==0,'B',i)
                if ticket is not None: tickets.append(ticket)
            if len(tickets)==200 and all(t.done for t in tickets) and cycle>12000: break
        assert len(tickets)==200 and all(t.done for t in tickets) and m.idle
        accepted=[x['id'] for x in m.trace if x['event']=='accept']
        completed=[x['id'] for x in m.trace if x['event']=='complete']
        assert len(accepted)==len(set(accepted))==200
        assert sorted(accepted)==sorted(completed)
        assert m.stats['read_bytes']+m.stats['write_bytes']==200*tx
    finally:
        m.close()
