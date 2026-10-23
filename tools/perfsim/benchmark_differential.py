#!/usr/bin/env python3
"""Compare actual Ramulator scalar and batched ticking without skipping core/memory cycles."""
import argparse,hashlib,json,os,platform,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from opentpu.perfsim import Simulator,Architecture,MemoryConfig
from opentpu.perfsim.io import load_bundle,write_json,manifest
from opentpu.perfsim.ports import NativeMemorySystem


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for k in ('profile','bridge','out'):ap.add_argument('--'+k,type=Path,required=True)
 ap.add_argument('--bundles',nargs='+',type=Path,required=True);ap.add_argument('--repeats',type=int,default=3);a=ap.parse_args()
 if a.out.exists():ap.error('output must not exist')
 if a.repeats<1:ap.error('repeats must be positive')
 a.out.mkdir(parents=True);profile=json.loads(a.profile.read_text());mc=profile['memory'].copy();mc.update(bridge=str(a.bridge.resolve()),ramulator_config=str((a.profile.resolve().parent/mc['ramulator_config']).resolve()))
 rows=[];inputs=[a.profile,a.bridge]
 for bundle in a.bundles:
  cfg,program,image,obj=load_bundle(bundle);baseline=None;observations=[]
  for record in (True,False):
   for repeat in range(1 if record else a.repeats):
    for batched in ([False,True] if repeat%2==0 else [True,False]):
     os.environ['OTPU_PERFSIM_MEMORY_BATCH']='1' if batched else '0';memory=NativeMemorySystem(MemoryConfig(**dict(mc,record_trace=record)))
     try:
      if batched and not memory.backend.batched:raise RuntimeError('new bridge with otpu_advance is required')
      started=time.perf_counter();sim=Simulator(cfg,program,image,architecture=Architecture(**profile['architecture']),memory_system=memory,args=obj.get('args'),labels=obj.get('labels'));result=sim.run();wall=time.perf_counter()-started
      payload={'cycles':result['cycles'],'instructions':result['instructions'],'statistics':dict(memory.stats),'dram_sha256':hashlib.sha256(sim.state.dram).hexdigest(),'tmem_sha256':hashlib.sha256(sim.state.tmem).hexdigest()}
      digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
      if record:payload.update(physical_trace=memory.trace,port_trace=memory.controller.events)
      if baseline is None:baseline=payload
      else:
       compare=baseline if record else {k:baseline[k] for k in payload}
       if payload!=compare:raise AssertionError('scalar/batched differential mismatch')
      observations.append({'batched':batched,'trace_enabled':record,'repeat':repeat,'host_wall_seconds':wall,'cycles_per_host_second':result['cycles']/wall,'cycle_instruction_state_statistics_sha256':digest,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)})
     finally:memory.close()
  rows.append({'bundle':str(bundle),'cycles':baseline['cycles'],'exact_scalar_batched':True,'comparison':'every physical and port acceptance/completion, instruction cycles/stalls/work, memory statistics and final DRAM/TMEM','observations':observations})
  inputs.extend(bundle/f for f in ('input.json','prog_0.hex','dram_0.bin'));print(bundle,flush=True)
 write_json(a.out/'summary.json',{'cases':rows,'host':platform.platform(),'python':platform.python_version(),'boundary':'simulator construction/run and ISA arithmetic, excludes memory backend construction, file loading, independent reference and serialization','core_cycles_skipped':0,'memory_ticks_skipped':0,'baseline':'same Python implementation; scalar versus batched FFI only','numeric_validation':'differential equivalence; independent correctness is checked by separate RTL/reference comparisons'})
 write_json(a.out/'manifest.json',manifest(inputs))
if __name__=='__main__':main()
