#!/usr/bin/env python3
"""Run the same program on RTL and model with independently generated closed-loop memory traffic."""
import argparse,json,re,subprocess,sys,sysconfig,time
from pathlib import Path
from dataclasses import asdict
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from opentpu.isasim import Machine
from opentpu.perfsim import Simulator,MemoryConfig,Architecture
from opentpu.perfsim.ports import NativeMemorySystem
from opentpu.perfsim.io import load_bundle,save_bundle,write_json,digest,manifest
from tools.perfsim.timing_suite import parse_trace
from opentpu.perfsim.baseline import verify_rtl_parameters


def port_schedule(events):
 """Compare semantic per-port transactions; IDs are independently allocated by each run."""
 accepted={e['id']:e for e in events if e['event']=='port_accept'}
 return {port:[(e['event'],e['cycle'],*(accepted[e['id']][k] for k in ('address','size','write')))
               for e in events if e['port']==port] for port in ('A','B','SW')}


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ('bundle','profile','executable','out'):ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--micro-trace',action='store_true')
 ap.add_argument('--bridge',type=Path)
 a=ap.parse_args();out=a.out.resolve()
 if out.exists():ap.error('output must not exist')
 out.mkdir(parents=True)
 cfg,prog,img,obj=load_bundle(a.bundle);params=json.loads(a.executable.resolve().parents[1].joinpath('build-receipt.json').read_text())
 profile=json.loads(a.profile.read_text())
 verify_rtl_parameters(params,cfg,Architecture(**profile['architecture']))
 if digest(a.executable)!=params['binary_sha256']:raise ValueError('RTL binary hash mismatch')
 mc=profile['memory'].copy()
 for k in ('bridge','ramulator_config'):
  if mc.get(k):mc[k]=str((a.profile.resolve().parent/mc[k]).resolve())
 if a.bridge:mc['bridge']=str(a.bridge.resolve())
 if not mc.get('native_ports'):raise ValueError('comparison profile must explicitly enable native_ports')
 mc['record_trace']=True;profile['memory']=mc;write_json(out/'profile.json',profile)
 save_bundle(out,cfg,prog,img,args=obj.get('args',()),labels=obj.get('labels'),metadata=obj.get('metadata'))
 args=[str(a.executable.resolve()),f'+dir={out}',f'+memory_profile={out}/profile.json',f'+python_root={ROOT}',f'+python_site={sysconfig.get_paths()["purelib"]}',f'+python_executable={sys.executable}','+max_cycles=1000000','+trace']
 args += [f'+arg{k}={v}' for k,v in enumerate(obj.get('args',()))]
 if cfg.SEQ_WC:
  from opentpu import waits as W
  waits=W.wait_counts(prog,cfg.D,cfg.S);W.sync_file(waits,out/'waits.txt');args.append(f'+hz_sync_s0={out}/waits.txt')
 if a.micro_trace:args.append('+uarch_trace')
 write_json(out/'command.json',args)
 started=time.perf_counter();run=subprocess.run(args,capture_output=True,text=True,timeout=180);rtl_wall=time.perf_counter()-started
 log=run.stdout+run.stderr;(out/'stdout.log').write_text(log)
 match=re.search(r'RESULT cycles=(\d+) halted=1 error=0',log)
 if run.returncode or not match:
  write_json(out/'failure.json',{'returncode':run.returncode,'reason':'RTL failed; see stdout.log'})
  raise RuntimeError(f'RTL failed: {log[-1500:]}')
 ref=Machine(cfg,[prog],[img],obj.get('args')).run().slices[0]
 dram=np.fromfile(out/'dram_out_0.bin',np.uint8);tmem=np.array([int(x,16) for x in (out/'tmem_0.hex').read_text().split()],np.uint32)
 numeric_rtl=np.array_equal(dram,ref.dram) and np.array_equal(tmem,ref.tmem)
 memory=NativeMemorySystem(MemoryConfig(**mc))
 sim=Simulator(cfg,prog,img,architecture=Architecture(**profile['architecture']),memory_system=memory,args=obj.get('args'),labels=obj.get('labels'))
 sim.capture_micro=a.micro_trace;sim.micro_events=[]
 result=sim.run()
 if a.micro_trace:write_json(out/'micro-model.json',sim.micro_events)
 numeric_sim=np.array_equal(sim.state.dram,ref.dram) and np.array_equal(sim.state.tmem,ref.tmem)
 write_json(out/'metrics.json',result)
 write_json(out/'memory-model.json',{'statistics':dict(memory.stats),'transactions':memory.trace,'ports':memory.controller.events})
 rtl_memory=json.loads((out/'memory-rtl.json').read_text())
 rt=parse_trace(log);pt=[{k:r[k] for k in ('pc','dispatch','start','end')} for r in result['instructions']]
 exact=[{k:r[k] for k in ('pc','dispatch','start','end')} for r in rt]==pt
 def shape(events):return Counter((r['port'],r['address'],r['size'],r['write']) for r in events if r['event']=='port_accept')
 traffic=shape(rtl_memory['ports'])==shape(memory.controller.events)
 port_match=port_schedule(rtl_memory['ports'])==port_schedule(memory.controller.events)
 keys=('read_bytes','write_bytes','read_transactions','write_transactions')
 physical=all(rtl_memory['statistics'].get(k,0)==memory.stats.get(k,0) for k in keys)
 numeric={'rtl_vs_isa_bit_exact':bool(numeric_rtl),'model_vs_isa_bit_exact':bool(numeric_sim),'independent_float_reference':None}
 md=obj.get('metadata',{})
 if (a.bundle/'reference_logits.npy').exists():
  addr,n=md['logits_address'],md['logits_count'];want=np.load(a.bundle/'reference_logits.npy').astype(np.float64)
  reports={}
  for name,data in [('rtl',dram),('model',sim.state.dram)]:
   got=data[addr:addr+4*n].view(np.float32).astype(np.float64);error=float(np.linalg.norm(got-want)/np.linalg.norm(want));cos=float(got@want/(np.linalg.norm(got)*np.linalg.norm(want))) if np.linalg.norm(got)>0 else None
   reports[name]={'relative_l2':error,'cosine':cos,'passed':error<0.05 and cos is not None and cos>0.995}
  numeric['independent_float_reference']=reports
 summary={'rtl_host_wall_seconds':rtl_wall,'model_host_wall_seconds':result['host_wall_seconds'],'backend':mc['backend'],'native_ports':True,'rtl_cycles':int(match[1]),'predicted_cycles':result['cycles'],'trace_match':exact,'traffic_multiset_match':traffic,'physical_traffic_match':physical,'port_arrival_completion_match':port_match,'numerical':numeric,'timing_passed':exact and int(match[1])==result['cycles'] and port_match,'same_backend_configuration':True,'request_arrivals':'independently generated; no trace replay or arrival-time forcing','rtl_trace':rt,'predicted_trace':pt,'clocks_ps':{'core':profile['architecture']['core_period_ps'],'controller':mc['controller_period_ps'],'memory':mc['memory_period_ps']}}
 write_json(out/'summary.json',summary)
 inputs=[a.executable,out/'profile.json',a.bundle/'input.json',a.bundle/'prog_0.hex',a.bundle/'dram_0.bin']
 if mc.get('bridge'):inputs.append(Path(mc['bridge']))
 if mc.get('ramulator_config'):inputs.append(Path(mc['ramulator_config']))
 write_json(out/'manifest.json',manifest(inputs));memory.close()
 print(json.dumps({k:v for k,v in summary.items() if k not in ('rtl_trace','predicted_trace')},indent=2))
 if not numeric_rtl or not numeric_sim or (numeric['independent_float_reference'] is not None and not all(r['passed'] for r in numeric['independent_float_reference'].values())):raise SystemExit(1)
 if not summary['timing_passed'] or not traffic or not physical:raise SystemExit(2)
if __name__=='__main__':main()
