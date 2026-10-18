#!/usr/bin/env python3
"""Independently execute multiple core launches with retained RTL/Ramulator memory state."""
import argparse,json,re,subprocess,sys,sysconfig,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from opentpu.isasim import Machine
from opentpu.perfsim import Simulator,Architecture,MemoryConfig
from opentpu.perfsim.ports import NativeMemorySystem
from opentpu.perfsim.io import load_bundle,save_bundle,write_json,manifest,digest
from tools.perfsim.compare_native import port_schedule
from tools.perfsim.timing_suite import parse_trace


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('bundles','profile','executable','out'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--bridge',type=Path,required=True);ap.add_argument('--host-gap',type=int,default=0)
    a=ap.parse_args();out=a.out.resolve()
    if out.exists():ap.error('output must not exist')
    if a.host_gap<0:ap.error('host gap must be nonnegative')
    out.mkdir(parents=True)
    sequence=json.loads((a.bundles/'sequence.json').read_text())
    bundles=[load_bundle(a.bundles/name) for name in sequence['programs']]
    receipt=json.loads((a.executable.resolve().parents[1]/'build-receipt.json').read_text())
    if not receipt.get('persistent') or digest(a.executable)!=receipt['binary_sha256']:raise ValueError('persistent build receipt mismatch')
    cfg=bundles[0][0]
    for key,value in {'S':cfg.S,'D':cfg.D,'MCOLS':cfg.MCOLS,'LANES':cfg.LANES,'ACT_BLOCKS':cfg.ACT_BLOCKS,'ACT_ROWS':cfg.act_rows,'TMEM_WORDS':cfg.TMEM_WORDS,'IMEM_WORDS':cfg.IMEM_WORDS,'DRAM_WORDS':cfg.DRAM_BYTES//4,'WIN':16,'RPB':32,'WPB':1,'FIFO_DEPTH':1024}.items():
        if receipt['parameters'].get(key)!=value:raise ValueError(f'RTL build mismatch: {key}')
    profile=json.loads(a.profile.read_text());mc=profile['memory'].copy()
    mc.update(bridge=str(a.bridge.resolve()),record_trace=True)
    mc['ramulator_config']=str((a.profile.resolve().parent/mc['ramulator_config']).resolve())
    if not mc.get('native_ports'):raise ValueError('native_ports must be explicit')
    profile['memory']=mc;write_json(out/'profile.json',profile)
    cmd=[str(a.executable.resolve()),f'+dir={out}',f'+memory_profile={out}/profile.json',f'+python_root={ROOT}',f'+python_site={sysconfig.get_paths()["purelib"]}',f'+python_executable={sys.executable}',f'+stages={len(bundles)}',f'+host_gap={a.host_gap}','+max_cycles=1000000','+trace']
    for index,(c,program,image,obj) in enumerate(bundles):
        if c!=cfg:raise ValueError('all launches must use the same hardware configuration')
        save_bundle(out/f'stage{index}',c,program,image,args=obj.get('args',()),labels=obj.get('labels'),metadata=obj.get('metadata'))
        if index==0:save_bundle(out,c,program,image)
        cmd.extend(f'+stage{index}_arg{k}={v}' for k,v in enumerate(obj.get('args',())))
    write_json(out/'command.json',cmd)
    wall=time.perf_counter();run=subprocess.run(cmd,capture_output=True,text=True,timeout=240);rtl_wall=time.perf_counter()-wall
    log=run.stdout+run.stderr;(out/'stdout.log').write_text(log)
    if run.returncode:raise RuntimeError(f'RTL sequence failed: {log[-1200:]}')
    parts=re.split(r'STAGE_START index=(\d+) global=(\d+)\n',log)
    measured={int(parts[k]):(int(parts[k+1]),parts[k+2]) for k in range(1,len(parts),3)}
    if len(measured)!=len(bundles):raise RuntimeError('missing RTL stage measurements')
    memory=NativeMemorySystem(MemoryConfig(**mc));arch=Architecture(**profile['architecture'])
    rows=[];previous=None;golden_previous=None;global_start=0
    try:
        for index,(c,program,image,obj) in enumerate(bundles):
            seed=image.copy() if previous is None else previous.dram.copy()
            if previous is not None:
                for addr,size in obj.get('metadata',{}).get('host_inputs',[]):seed[addr:addr+size]=image[addr:addr+size]
            sim=Simulator(c,program,seed,memory_system=memory,architecture=arch,args=obj.get('args'),labels=obj.get('labels'))
            gold_seed=image.copy() if golden_previous is None else golden_previous.dram.copy()
            if golden_previous is not None:
                for addr,size in obj.get('metadata',{}).get('host_inputs',[]):gold_seed[addr:addr+size]=image[addr:addr+size]
            golden=Machine(c,[program],[gold_seed],obj.get('args'))
            if previous is not None:
                for key in ('tmem','act','ascale','adoff'):
                    getattr(sim.state,key)[:]=getattr(previous,key)
                    getattr(golden.slices[0],key)[:]=getattr(golden_previous,key)
            assert sim.memory_offset==global_start
            result=sim.run();golden.run();gold=golden.slices[0]
            rtl_start,stage_log=measured[index];match=re.search(r'STAGE_RESULT index=\d+ cycles=(\d+) halted=1 error=0',stage_log)
            if not match:raise RuntimeError('stage did not halt')
            dram=np.fromfile(out/f'stage{index}/dram_out_0.bin',np.uint8)
            tmem=np.array([int(x,16) for x in (out/f'stage{index}/tmem_0.hex').read_text().split()],np.uint32)
            numeric=all(np.array_equal(x,y) for x,y in [(dram,gold.dram),(tmem,gold.tmem),(sim.state.dram,gold.dram),(sim.state.tmem,gold.tmem)])
            float_ref=None;reference=a.bundles/sequence['programs'][index]/'reference_logits.npy'
            if reference.exists():
                md=obj['metadata'];addr,n=md['logits_address'],md['logits_count'];want=np.load(reference).astype(np.float64)
                got=dram[addr:addr+4*n].view(np.float32).astype(np.float64)
                error=float(np.linalg.norm(got-want)/np.linalg.norm(want));cos=float(got@want/(np.linalg.norm(got)*np.linalg.norm(want)))
                float_ref={'relative_l2':error,'cosine':cos,'passed':error<0.05 and cos>0.995}
            keys=('pc','dispatch','start','end');rt=parse_trace(stage_log)
            pt=[{k:r[k] for k in keys} for r in result['instructions']]
            exact=pt==[{k:r[k] for k in keys} for r in rt]
            row={'stage':index,'name':sequence['programs'][index],'phase':obj.get('metadata',{}).get('phase'),'rtl_global_start':rtl_start,'model_global_start':global_start,'rtl_cycles':int(match[1]),'model_cycles':result['cycles'],'trace_match':exact,'numeric_bit_exact':bool(numeric),'independent_reference':float_ref,'rtl_trace':rt,'model_trace':pt}
            row['timing_passed']=exact and rtl_start==global_start and int(match[1])==result['cycles']
            rows.append(row);write_json(out/f'stage{index}/metrics.json',result)
            previous=sim.state;golden_previous=gold
            global_start+=result['cycles']+8+a.host_gap
            if index+1<len(bundles):
                for cycle in range(memory.cycle+1,global_start):memory.begin_cycle(cycle,arch.core_period_ps)
        rtl_memory=json.loads((out/f'stage{len(bundles)-1}/memory-rtl.json').read_text())
        port_match=port_schedule(rtl_memory['ports'])==port_schedule(memory.controller.events)
        physical=all(rtl_memory['statistics'].get(k,0)==memory.stats.get(k,0) for k in ('read_bytes','write_bytes','read_transactions','write_transactions'))
        summary={'backend':'ramulator','stages':rows,'host_gap_cycles':a.host_gap,'fixed_launch_overhead_cycles':8,'retained':'DRAM, TMEM, ACT RAM, controller/open rows/refresh; reset core registers, sequencer and unit pipelines','request_arrivals':'independently generated; explicit host gap, no forced request times','port_schedule_match':port_match,'physical_traffic_match':physical,'timing_passed':all(r['timing_passed'] for r in rows) and port_match and physical,'numeric_passed':all(r['numeric_bit_exact'] and (r['independent_reference'] is None or r['independent_reference']['passed']) for r in rows),'rtl_host_wall_seconds':rtl_wall,'model_host_wall_seconds':sum(json.loads((out/f'stage{i}/metrics.json').read_text())['host_wall_seconds'] for i in range(len(rows)))}
        write_json(out/'memory-model.json',{'statistics':dict(memory.stats),'ports':memory.controller.events,'transactions':memory.trace})
        write_json(out/'summary.json',summary);write_json(out/'manifest.json',manifest([a.executable,a.bridge,out/'profile.json',a.bundles/'sequence.json',*[a.bundles/n/f for n in sequence['programs'] for f in ('input.json','prog_0.hex','dram_0.bin')]]))
        print(json.dumps({k:v for k,v in summary.items() if k!='stages'},indent=2))
        if not summary['numeric_passed']:raise SystemExit(1)
        if not summary['timing_passed']:raise SystemExit(2)
    finally:memory.close()


if __name__=='__main__':main()
