#!/usr/bin/env python3
"""Measure host execution throughput on compiled operator programs, never a roofline substitute."""
import argparse,json,time,sys,resource,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from opentpu.perfsim import Simulator,Architecture,MemoryConfig
from opentpu.perfsim.io import load_bundle,write_json,manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('bundles','profile','bridge','out'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--select',default='');ap.add_argument('--max-cycles',type=int,default=10000000)
    a=ap.parse_args()
    if a.out.exists():ap.error('output must not exist')
    a.out.mkdir(parents=True)
    profile=json.loads(a.profile.read_text());mc=profile['memory'].copy();mc['bridge']=str(a.bridge.resolve());mc['record_trace']=False
    mc['ramulator_config']=str((a.profile.resolve().parent/mc['ramulator_config']).resolve())
    arch=Architecture(**dict(profile['architecture'],max_cycles=a.max_cycles));rows=[];inputs=[a.profile,a.bridge]
    for case in json.loads((a.bundles/'catalog.json').read_text())['cases']:
        if a.select and a.select not in case['case']:continue
        bundle=a.bundles/case['case'];cfg,program,image,obj=load_bundle(bundle)
        started=time.perf_counter();sim=Simulator(cfg,program,image,architecture=arch,memory=MemoryConfig(**mc));result=sim.run();wall=time.perf_counter()-started
        md=obj['metadata'];addr,n=md['logits_address'],md['logits_count']
        got=sim.state.dram[addr:addr+4*n].view(np.float32).astype(np.float64);want=np.load(bundle/'reference_logits.npy').astype(np.float64)
        rel=float(np.linalg.norm(got-want)/np.linalg.norm(want));cos=float(got@want/(np.linalg.norm(got)*np.linalg.norm(want)))
        numeric={'relative_l2':rel,'cosine':cos,'passed':rel<0.05 and cos>0.995,'reference':md['independent_reference']}
        result['numerical_check']=numeric;write_json(a.out/(case['case']+'.json'),result)
        row={'name':case['case'],'shape_M_K_N':md['shape_M_K_N'],'complete_operator_shape':md['complete_operator_shape'],'weight_format':md['weight_format'],'cycles':result['cycles'],'host_wall_seconds':wall,'simulated_cycles_per_host_second':result['cycles']/wall,'dynamic_instructions':len(result['instructions']),'physical_transactions':result['memory'].get('read_transactions',0)+result['memory'].get('write_transactions',0),'numeric':numeric,'timing_validation':'prediction; this full operator is not asserted RTL-validated by this benchmark','model_seconds':result['seconds'],'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
        rows.append(row);print(json.dumps(row),flush=True)
        inputs.extend(bundle/f for f in ('input.json','prog_0.hex','dram_0.bin','reference_logits.npy'))
    write_json(a.out/'summary.json',{'backend':mc['backend'],'cases':rows,'pretrained_model_validated':False,'host':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'trace_enabled':False,'wall_boundary':'Simulator construction and run, including ISA functional execution; excludes compiler, independent reference and file serialization','purpose':'measured host simulator throughput; model cycles remain predictions unless separately matched to RTL'})
    write_json(a.out/'manifest.json',manifest(inputs))
    if not all(r['numeric']['passed'] for r in rows):raise SystemExit(1)
if __name__=='__main__':main()
