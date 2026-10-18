#!/usr/bin/env python3
"""Run bounded teacher-forced model stages with persistent KV and memory-controller state."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from opentpu.perfsim import Simulator,Architecture,MemoryConfig
from opentpu.perfsim.ports import make_memory
from opentpu.perfsim.io import load_bundle,manifest,state_hash,write_json,digest
from opentpu.isasim import Machine


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundles',required=True)
    ap.add_argument('--profile',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--bridge')
    ap.add_argument('--launch-overhead-cycles',type=int,default=8,help='explicit idle/reset edges between launches; 8 matches the persistent RTL harness')
    a=ap.parse_args(); out=Path(a.out)
    if a.launch_overhead_cycles<0:ap.error('launch overhead must be nonnegative')
    if out.exists() and any(out.iterdir()): ap.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    path=Path(a.profile).resolve(); profile=json.loads(path.read_text())
    mc=profile['memory'].copy()
    for k in ('bridge','ramulator_config'):
        if mc.get(k): mc[k]=str((path.parent/mc[k]).resolve())
    if a.bridge: mc['bridge']=str(Path(a.bridge).resolve())
    arch=Architecture(**profile.get('architecture',{})); mem=make_memory(MemoryConfig(**mc))
    base=Path(a.bundles); seq=json.loads((base/'sequence.json').read_text())
    inputs=[path,base/'sequence.json']
    stages=[]; previous=None; golden_previous=None
    try:
        for index,name in enumerate(seq['programs']):
            bundle=base/name; cfg,prog,image,obj=load_bundle(bundle); md=obj['metadata']
            seed=image.copy()
            if previous is not None:
                seed=previous.dram.copy()
                for addr,n in md['host_inputs']: seed[addr:addr+n]=image[addr:addr+n]
            sim=Simulator(cfg,prog,seed,architecture=arch,memory_system=mem,args=obj.get('args'),labels=obj.get('labels'))
            golden_seed=image.copy() if golden_previous is None else golden_previous.dram.copy()
            if golden_previous is not None:
                for addr,n in md.get('host_inputs',[]):golden_seed[addr:addr+n]=image[addr:addr+n]
            gold_machine=Machine(cfg,[prog],[golden_seed],obj.get('args'))
            if previous is not None:
                for field in ('tmem','act','ascale','adoff'):
                    getattr(sim.state,field)[:]=getattr(previous,field)
                    getattr(gold_machine.slices[0],field)[:]=getattr(golden_previous,field)
            result=sim.run()
            golden=gold_machine.run().slices[0]
            exact=state_hash(sim.state)==state_hash(golden)
            addr,n=md['logits_address'],md['logits_count']
            got=sim.state.dram[addr:addr+4*n].view(np.float32).astype(np.float64)
            want=np.load(bundle/'reference_logits.npy').astype(np.float64)
            error=float(np.linalg.norm(got-want)/np.linalg.norm(want))
            cosine=float(got@want/(np.linalg.norm(got)*np.linalg.norm(want)))
            result['numerical_check']={'isa_bit_exact':exact,'independent_reference':'NumPy reference_logits, float forward',
                                       'relative_l2':error,'cosine':cosine,'relative_l2_limit':0.05,'cosine_minimum':0.995,
                                       'passed':exact and error<0.05 and cosine>0.995}
            result['phase']=md['phase']; result['position']=md['position']
            write_json(out/(name+'.json'),result)
            if not result['numerical_check']['passed']: raise AssertionError(f'{name}: numeric validation failed')
            stages.append({'name':name,'phase':md['phase'],'cycles':result['cycles'],
                           'seconds':result['seconds'],'numerical_check':result['numerical_check'],
                           'memory':result['memory']})
            previous=sim.state;golden_previous=golden
            if index+1<len(seq['programs']):
                for cycle in range(mem.cycle+1,mem.cycle+1+a.launch_overhead_cycles):mem.begin_cycle(cycle,arch.core_period_ps)
            inputs += [bundle/x for x in ('input.json','prog_0.hex','dram_0.bin','reference_logits.npy')]
        prefill=sum(s['seconds'] for s in stages if s['phase']=='prefill')
        decode=sum(s['seconds'] for s in stages if s['phase']=='decode')
        gap=a.launch_overhead_cycles*arch.core_period_ps*1e-12
        prefill_elapsed=prefill+max(0,seq['prefill_steps']-1)*gap
        decode_elapsed=decode+seq['decode_steps']*gap
        summary={'label':'opentpu-perfsim-estimate','backend':mc['backend'],'timing_validation':'unvalidated',
                 'profile':profile['name'],'stages':stages,'prefill_seconds':prefill,'decode_seconds':decode,
                 'device_ttft_seconds':prefill_elapsed,'conditional_decode_tokens_per_second':seq['decode_steps']/decode_elapsed,
                 'prefill_elapsed_seconds':prefill_elapsed,'decode_elapsed_seconds':decode_elapsed,'launch_overhead_cycles':a.launch_overhead_cycles,
                 'core_only_decode_tokens_per_second':seq['decode_steps']/decode,
                 'boundary':'program-resident core start through final write completion; serial prompt steps, no sampling, host, queueing or input-transfer time',
                 'memory_state':'persistent DRAM/TMEM/ACT/controller and KV; core pipeline/register reset; explicit idle/reset launch edges',
                 'full_pretrained_model':False,'hardware_measured':False,'fmax_verified':False}
        write_json(out/'summary.json',summary)
        receipt=manifest(inputs)
        if mc.get('bridge'): receipt['bridge_sha256']=digest(mc['bridge'])
        if mc.get('ramulator_config'): receipt['ramulator_config_sha256']=digest(mc['ramulator_config'])
        write_json(out/'manifest.json',receipt)
        print(json.dumps(summary))
    finally:
        mem.close()

if __name__=='__main__': main()
