#!/usr/bin/env python3
"""Bounded, independently executed RTL timing corpus. Never changes RTL or expectations."""
import argparse,json,re,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from opentpu import isa as I
from opentpu.isasim import Machine
from opentpu.perfsim import Simulator,MemoryConfig
from opentpu.perfsim.io import load_bundle,save_bundle,write_json,digest,manifest


def fill(dst,n=67,rows=1):
    return I.vop(I.V_FILL,dst,0,0,rows,n,n,n,0,I.B_SCALAR,2)


def cases():
    yield 'halt',[I.halt()]
    for n in (1,8,9,64,67,129):
        yield f'fill_{n}',[fill(0,n),I.halt()]
    for f in (I.V_MUL,I.V_EXP2,I.V_RECIP,I.V_RSQRT,I.V_LOG2,I.V_RSUM,I.V_RSSQ,I.V_RDOT,I.V_RMAX,I.V_ARGMAX):
        for n in (1,67):
            yield f'func_{f}_{n}',[fill(0,n),I.vop(f,4096,0,0,1,n,n,n,0,I.B_SCALAR,3),I.halt()]
    for f in (I.V_RSUM,I.V_RDOT,I.V_RMAX):
        yield f'multirow_{f}',[fill(0,19,3),I.vop(f,4096,0,0,3,19,1,19,0,I.B_SCALAR,3),I.halt()]
    yield 'initial_failure',[fill(0),I.vop(I.V_MUL,128,0,0,1,67,67,67,0,I.B_SCALAR,3),I.vop(I.V_RSUM,256,128,0,1,67,67,67,0,I.B_SCALAR,0),I.halt()]
    for a,b in ((I.V_EXP2,I.V_COPY),(I.V_COPY,I.V_EXP2),(I.V_EXP2,I.V_RSUM)):
        yield f'overlap_{a}_{b}',[fill(0,19),I.vop(a,128,0,0,1,19,19,19,0,I.B_SCALAR,0),I.vop(b,256,0,0,1,19,19,19,0,I.B_SCALAR,0),I.halt()]
    yield 'window_pressure',[*[fill(n*8,1) for n in range(40)],I.halt()]
    for n in (8,19,67):
        yield f'load_{n}',[I.ld(0,0,n),I.halt()]
        yield f'store_{n}',[fill(0,n),I.st(0,0,n),I.halt()]
        yield f'load_add_store_{n}',[I.ld(0,0,n),I.vop(I.V_ADD,128,0,0,1,n,n,n,0,I.B_SCALAR,1),I.st(512,128,n),I.halt()]

    # Held out from the first pipeline diagnosis; run after deriving the stage schedule.
    for f in (I.V_EXP2,I.V_RECIP,I.V_RSUM,I.V_RDOT):
        yield f'heldout_long_{f}',[fill(0,513),I.vop(f,4096,0,0,1,513,513,513,0,I.B_SCALAR,2),I.halt()]
    for f in (I.V_RSUM,I.V_RDOT,I.V_ARGMAX):
        yield f'heldout_rows_{f}',[fill(0,65,9),I.vop(f,4096,0,0,9,65,2,65,0,I.B_SCALAR,0),I.halt()]
    yield 'heldout_loop',[I.loop(2,3),fill(0,19),I.vop(I.V_ADD,128,0,0,1,19,19,19,0,I.B_SCALAR,1),I.halt()]
    for n in (5,17,70,2048):
        yield f'heldout_dma_{n}',[I.ld(4,3,n),I.st(16388,3,n),I.halt()]
    yield 'heldout_contention',[fill(0,513),I.vop(I.V_EXP2,4096,0,0,1,513,513,513,0,I.B_SCALAR,0),I.ld(0,2048,1024),I.halt()]
    yield 'heldout_dependency',[fill(0,67),I.vop(I.V_MUL,128,0,0,1,67,67,67,0,I.B_SCALAR,3),fill(0,67),I.vop(I.V_ADD,256,128,0,1,67,67,67,0,I.B_SCALAR,4),I.halt()]

    for rows,kb in ((1,1),(1,2),(2,1),(3,2),(4,8)):
        yield f'quant_{rows}_{kb}',[fill(0,32*kb,rows),I.qact(0,rows,0,kb,32*kb),I.halt()]

    # Independent partial-tile cases added after the compiled GEMM diagnosis.
    for m,n,kb in ((1,1,1),(1,7,2),(2,13,1),(3,17,2),(4,9,4),(3,1,4),(4,33,1)):
        yield f'mxu_partial_{m}_{n}_{kb}',[fill(0,32*kb,m),I.qact(0,m,0,kb,32*kb),I.mm(0,0,4096,n,kb,32*kb,n,m,0,0,unit=True),I.halt()]
    for n in (31,33,63,65,257):
        yield f'frozen_dma_{n}',[I.ld(12,1,n),I.st(16396,1,n),I.halt()]


def parse_trace(log):
    active={}; records=[]
    for line in log.splitlines():
        m=re.match(r'T0 ([DSE]) c=(\d+) s=(\d+)(.*)',line)
        if not m: continue
        kind,cycle,slot,tail=m.groups(); cycle,slot=int(cycle),int(slot)
        if kind=='D':
            row={'id':len(records),'pc':int(re.search(r'pc=(\d+)',tail)[1]),'dispatch':cycle}
            records.append(row);active[slot]=row
        elif kind=='S': active[slot]['start']=cycle
        else: active.pop(slot)['end']=cycle
    return records


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--config-bundle',type=Path,default=ROOT/'build/perfsim/bundles-v2/compute')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--select',default='')
    ap.add_argument('--compiled',action='store_true')
    ap.add_argument('--memory',choices=['simple','rtl_fixed'],default='rtl_fixed')
    a=ap.parse_args()
    if a.out.exists(): ap.error('output must not exist')
    a.out.mkdir(parents=True)
    cfg,_,_,_=load_bundle(a.config_bundle)
    exe=a.executable.resolve(); build=exe.with_name('Vtb_top__verFiles.dat')
    line=next(s for s in build.read_text().splitlines() if s.startswith('C '))
    got={k:int(v) for k,v in re.findall(r'-G(\w+)=(\d+)',line)}
    expected={'S':1,'D':cfg.D,'MCOLS':cfg.MCOLS,'LANES':cfg.LANES,'ACT_BLOCKS':cfg.ACT_BLOCKS,'ACT_ROWS':cfg.act_rows,'TMEM_WORDS':cfg.TMEM_WORDS,'IMEM_WORDS':cfg.IMEM_WORDS,'DRAM_WORDS':cfg.DRAM_BYTES//4,'DRAM_LAT':8,'AXI':0,'WIN':16,'RPB':32,'WPB':1,'MXU_IMPL':2,'FIFO_DEPTH':1024}
    if any(got.get(k)!=v for k,v in expected.items()): raise ValueError('RTL configuration mismatch')
    results=[]
    corpus=[(name,prog,None) for name,prog in cases()]
    if a.compiled:
        for name in ('gemm','tiny_qwen_0','tiny_qwen_1','tiny_qwen_2'):
            cc,pp,ii,_=load_bundle(a.config_bundle.parent/name)
            if cc!=cfg: raise ValueError('compiled config mismatch')
            corpus.append(('compiled_'+name,pp,ii))
    for name,prog,provided_image in corpus:
        if a.select and not re.search(a.select,name): continue
        directory=a.out/name
        image=np.arange(1024,dtype=np.float32).view(np.uint8) if provided_image is None else provided_image
        save_bundle(directory,cfg,prog,image)
        run=subprocess.run([str(exe),f'+dir={directory.resolve()}','+max_cycles=2000000','+trace'],capture_output=True,text=True,timeout=120)
        log=run.stdout+run.stderr; (directory/'stdout.log').write_text(log)
        m=re.search(r'RESULT cycles=(\d+) halted=1 error=0',log)
        if run.returncode or not m: raise RuntimeError(f'{name}: RTL failed')
        ref=Machine(cfg,[prog],[image]).run().slices[0]
        dram=np.fromfile(directory/'dram_out_0.bin',np.uint8)
        tmem=np.array([int(x,16) for x in (directory/'tmem_0.hex').read_text().split()],np.uint32)
        exact=np.array_equal(dram,ref.dram) and np.array_equal(tmem,ref.tmem)
        mc=MemoryConfig(backend='rtl_fixed',memory_period_ps=5000,controller_period_ps=5000,transaction_bytes=cfg.D,simple_latency_ticks=8,rmw=False) if a.memory=='rtl_fixed' else MemoryConfig()
        sim=Simulator(cfg,prog,image,memory=mc); pred=sim.run()
        numeric=exact and np.array_equal(sim.state.dram,ref.dram) and np.array_equal(sim.state.tmem,ref.tmem)
        idle=not any(i.op in (I.LD,I.ST,I.MM,I.QST) for i in prog)
        measured=int(m[1]); traces=parse_trace(log)
        predicted_trace=[{k:r[k] for k in ('pc','dispatch','start','end')} for r in pred['instructions']]
        trace_match=[{k:r[k] for k in ('pc','dispatch','start','end')} for r in traces]==predicted_trace
        row={'trace_match':trace_match,'aggregate_cycle_gate_passed':abs(pred['cycles']/measured-1)<=0.05 if idle or a.memory=='rtl_fixed' else None,'name':name,'numeric_passed':bool(numeric),'rtl_cycles':measured,'predicted_cycles':pred['cycles'],'memory_conditions_match':idle or a.memory=='rtl_fixed','memory_backend':a.memory,'error_percent':100*(pred['cycles']/measured-1) if idle or a.memory=='rtl_fixed' else None,'rtl_trace':traces,'predicted_trace':[{k:r[k] for k in ('pc','dispatch','start','end')} for r in pred['instructions']]}
        row['input_sha256']={x:digest(directory/x) for x in ('input.json','prog_0.hex','dram_0.bin')}
        row['rtl_stdout_sha256']=digest(directory/'stdout.log')
        row['output_sha256']={x:digest(directory/x) for x in ('dram_out_0.bin','tmem_0.hex')}
        write_json(directory/'comparison.json',row)
        write_json(directory/'metrics.json',pred)
        results.append(row)
        print(name,measured,pred['cycles'],numeric,flush=True)
        if not numeric: raise AssertionError(name)
    write_json(a.out/'summary.json',{'schema_version':1,'binary_sha256':digest(exe),'build_receipt_sha256':digest(build),'parameters':expected,'configuration':json.loads((a.config_bundle/'input.json').read_text())['config'],'timing_gate':'all comparable D/S/E traces and cycle totals must match exactly','timing_gate_passed':all(r['memory_conditions_match'] and r['trace_match'] and r['rtl_cycles']==r['predicted_cycles'] for r in results),'cases':results})
    write_json(a.out/'manifest.json',manifest([exe,build]))
    if not all(r['trace_match'] and r['rtl_cycles']==r['predicted_cycles'] and r['memory_conditions_match'] for r in results): raise SystemExit(2)

if __name__=='__main__': main()
