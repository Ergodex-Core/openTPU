"""Additional observations; the earlier golden fixtures are never overwritten."""
import base64,hashlib,json,zlib
from pathlib import Path
import numpy as np
import pytest
from opentpu import Config,isa as I
from opentpu.perfsim import Simulator,MemoryConfig
from opentpu.perfsim.ports import NativeMemorySystem
F=json.loads((Path(__file__).parent/'fixtures/rtl_fixed_round3.json').read_text())

@pytest.mark.parametrize('case',F['cases'],ids=lambda c:c['name'])
def test_qst_overlap_and_compiled_model_observed_traces(case):
 raw=zlib.decompress(base64.b64decode(case['image_be_u32_zlib_base64']))
 assert hashlib.sha256(raw).hexdigest()==case['image_sha256']
 image=np.frombuffer(raw,dtype='>u4').astype('<u4').view(np.uint8)
 words=case['program_words'];program=[I.Instr.decode(words[k:k+8]) for k in range(0,len(words),8)]
 cfg=MemoryConfig(backend='rtl_fixed',controller_period_ps=5000,memory_period_ps=5000,transaction_bytes=32,simple_latency_ticks=8,rmw=False)
 sim=Simulator(Config(**F['configuration']),program,image,memory=cfg);r=sim.run()
 assert r['cycles']==case['rtl_cycles']
 keys=('pc','dispatch','start','end')
 assert [{k:v[k] for k in keys} for v in r['instructions']]==[{k:v[k] for k in keys} for v in case['rtl_trace']]


def test_native_elastic_registers_and_ordered_responses_under_backpressure():
 m=NativeMemorySystem(MemoryConfig(native_ports=True,outstanding=2,transaction_bytes=16,simple_channels=1,simple_queue_per_channel=1,simple_latency_ticks=20,record_trace=True))
 m.begin_cycle(0,1000)
 first=m.request(0,32,False,'B',1)
 assert first is not None and m.request(32,32,False,'B',2) is None
 m.begin_cycle(1,1000)
 second=m.request(32,32,False,'B',2)
 assert second is not None # first moved to the controller; register refills despite downstream pressure
 m.begin_cycle(2,1000)
 assert m.request(64,32,False,'B',3) is None
 for cycle in range(3,200):
  m.begin_cycle(cycle,1000)
  if second.done:break
 assert first.done and second.done and first.complete_cycle<second.complete_cycle and m.idle
 assert m.stats['peak_outstanding']<=2
 assert m.stats['backend_rejected_ticks']>0
 assert m.stats['read_transactions']==4
 m.close()
