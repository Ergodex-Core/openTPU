"""Frozen RTL observations and independent numerical references; no synthetic timing goldens."""
import base64,hashlib,json,os,zlib,sys
from pathlib import Path
import numpy as np
import pytest
from opentpu import Config,isa as I
from opentpu.perfsim import Simulator,Architecture,MemoryConfig,Unsupported
from opentpu.perfsim.io import save_bundle
from opentpu.perfsim.ports import NativeMemorySystem
from tools.perfsim.compare_native import port_schedule
from tools.perfsim import diagnose_int4

ROOT=Path(__file__).resolve().parents[2];FIX=Path(__file__).parent/'fixtures'
NATIVE=json.loads((FIX/'rtl_stream_waitc_round5.json').read_text())['cases']
HYBRID=json.loads((FIX/'rtl_hybrid_round5.json').read_text())['cases']
INT4=json.loads((FIX/'int4_diagnosis_round5.json').read_text())['cases']
BRIDGE=os.environ.get('OTPU_RAMULATOR_BRIDGE');native=pytest.mark.skipif(not BRIDGE,reason='requires pinned Ramulator; no fallback')


def unpack(case):
 raw=zlib.decompress(base64.b64decode(case['image_be_u32_zlib_base64']));assert hashlib.sha256(raw).hexdigest()==case['image_sha256']
 words=case['program_words'];return Config(**case['input']['config']),[I.Instr.decode(words[k:k+8]) for k in range(0,len(words),8)],np.frombuffer(raw,'>u4').astype('<u4').view(np.uint8)


def setup(profile):
 base=ROOT/'tools/perfsim/configs';p=json.loads((base/(profile+'.json')).read_text());mc=p['memory'];mc.update(bridge=BRIDGE,record_trace=True,ramulator_config=str(base/mc['ramulator_config']))
 return Architecture(**p['architecture']),NativeMemorySystem(MemoryConfig(**mc))


def compare(sim,result,case):
 assert result['cycles']==case['rtl_cycles'];keys=('pc','dispatch','start','end')
 assert [{k:r[k] for k in keys} for r in result['instructions']]==[{k:r[k] for k in keys} for r in case['rtl_trace']]
 assert hashlib.sha256(sim.state.dram).hexdigest()==case['dram_sha256']
 if 'reference_logits' in case:
  md=case['input']['metadata'];a,n=md['logits_address'],md['logits_count'];got=sim.state.dram[a:a+4*n].view(np.float32).astype(np.float64);want=np.array(case['reference_logits'])
  assert np.linalg.norm(got-want)/np.linalg.norm(want)<.05
  assert got@want/(np.linalg.norm(got)*np.linalg.norm(want))>.995


def port_hash(memory):return hashlib.sha256(json.dumps(port_schedule(memory.controller.events),sort_keys=True).encode()).hexdigest()


@native
@pytest.mark.parametrize('case',NATIVE,ids=lambda c:c['name'])
def test_stream_waitc_observed_rtl(case):
 cfg,program,image=unpack(case);arch,memory=setup(case['profile'])
 try:
  sim=Simulator(cfg,program,image,architecture=arch,memory_system=memory,args=case['input'].get('args'));compare(sim,sim.run(),case);assert port_hash(memory)==case['port_sha256']
 finally:memory.close()


@native
@pytest.mark.parametrize('case',HYBRID,ids=lambda c:c['name'])
def test_complete_hybrid_batched_prefill_persistent_decode(case):
 arch,memory=setup(case['profile']);previous=None;start=0
 try:
  for index,stage in enumerate(case['stages']):
   cfg,program,image=unpack(stage);seed=image.copy() if previous is None else previous.dram.copy()
   if previous is not None:
    for addr,size in stage['input']['metadata']['host_inputs']:seed[addr:addr+size]=image[addr:addr+size]
   sim=Simulator(cfg,program,seed,architecture=arch,memory_system=memory,args=stage['input'].get('args'))
   if previous is not None:
    for key in ('tmem','act','ascale','adoff'):getattr(sim.state,key)[:]=getattr(previous,key)
   assert sim.memory_offset==start==stage['rtl_global_start'];result=sim.run();compare(sim,result,stage)
   previous=sim.state;start+=result['cycles']+8
   if index+1<len(case['stages']):
    for cycle in range(memory.cycle+1,start):memory.begin_cycle(cycle,arch.core_period_ps)
  assert port_hash(memory)==case['port_sha256']
 finally:memory.close()


@pytest.mark.parametrize('case',INT4,ids=lambda c:str(c['D']))
def test_independent_int4_diagnosis_preserves_float_failure(case,tmp_path,monkeypatch,capsys):
 cfg,program,image=unpack(case);b=tmp_path/'bundle';r=tmp_path/'rtl';r.mkdir();out=tmp_path/'diagnosis'
 save_bundle(b,cfg,program,image,metadata=case['input']['metadata']);np.save(b/'reference_logits.npy',np.array(case['reference_logits']))
 md=case['input']['metadata'];data=bytearray(md['logits_address']+4*md['logits_count']);data[md['logits_address']:]=base64.b64decode(case['rtl_logits_bytes_base64']);(r/'dram_out_0.bin').write_bytes(data)
 monkeypatch.setattr(sys,'argv',['diagnose_int4','--bundle',str(b),'--rtl-run',str(r),'--out',str(out)]);diagnose_int4.main()
 result=json.loads((out/'diagnosis.json').read_text());assert result==case['diagnosis'];assert not result['original_float_gate_passed']
 assert result['independent_packed_arithmetic_bit_exact_RTL'];assert result['independent_activation_matches_ISA']


def test_waitc_rejects_missing_or_invalid_entries():
 cfg=Config(SEQ_WC=True);program=[I.halt()]
 for waits in ([],[1<<63],[-1]):
  with pytest.raises(ValueError,match='WAITC'):Simulator(cfg,program,np.zeros(4,np.uint8),wait_entries=waits)


def test_stream_disabled_or_non_eight_lane_rejected():
 ins=I.dstep(0,0,0,1,64,0,1,0)
 with pytest.raises(Unsupported,match='hardware'):Simulator(Config(),[ins,I.halt()],np.zeros(4,np.uint8))
 with pytest.raises(Unsupported,match='LANES=8'):Simulator(Config(DSTEP=True,LANES=4),[ins,I.halt()],np.zeros(4,np.uint8))


@native
@pytest.mark.parametrize('profile',['native-hbm2-200mhz','native-ddr4-200mhz'])
def test_scalar_batched_ramulator_identical_refresh_rmw_backpressure(profile,monkeypatch):
 observed=[]
 for batched in (False,True):
  monkeypatch.setenv('OTPU_PERFSIM_MEMORY_BATCH','1' if batched else '0');arch,memory=setup(profile);tickets=[]
  try:
   if batched and not memory.backend.batched:pytest.skip('bridge predates optional batching ABI')
   for cycle in range(5000):
    memory.begin_cycle(cycle,arch.core_period_ps)
    if len(tickets)<160:
     i=len(tickets);t=memory.request((i%4)*4096+(i//4)*64,4 if i%3==0 else 32,i%3==0,'SW' if i%3==0 else 'B',i)
     if t is not None:tickets.append(t)
   assert len(tickets)==160 and all(t.done for t in tickets) and memory.idle
   observed.append((memory.trace,memory.controller.events,dict(memory.stats)))
  finally:memory.close()
 assert observed[0]==observed[1]
