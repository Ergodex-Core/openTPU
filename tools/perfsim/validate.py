#!/usr/bin/env python3
"""Bounded validation. Never changes existing RTL, golden tests or the LPN registry."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import os
import hashlib
import re
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['OTPU_IMAGE_CACHE']='0'
os.environ['OTPU_IMG_CACHE']='0'
os.environ['OTPU_PROG_CACHE']='0'
import numpy as np
from opentpu import rtlsim
from opentpu.isasim import Machine
from opentpu.perfsim import Simulator,Architecture,MemoryConfig
from opentpu.perfsim.io import load_bundle,manifest,state_hash,write_json,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--rtl',action='store_true')
    p.add_argument('--rtl-executable',type=Path,help='reuse a locally built tb_top; validates parameters from its Verilator build receipt')
    a=p.parse_args(); out=Path(a.out)
    if out.exists() and any(out.iterdir()): p.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    cfg,prog,image,obj=load_bundle(a.bundle)
    # This comparison excludes active external memory; no fabricated equivalence between
    # Ramulator HBM and Verilator's fixed-latency behavioral DRAM.
    memops={0x10,0x11,0x20,0x22}
    memory_idle=not any(i.op in memops for i in prog)
    sim=Simulator(cfg,prog,image,args=obj.get('args'))
    pred=sim.run()
    gold=Machine(cfg,[prog],[image],obj.get('args')).run().slices[0]
    numerical={'isa_bit_exact':state_hash(sim.state)==state_hash(gold)}
    refpath=Path(a.bundle)/'reference_logits.npy'
    if refpath.exists():
        md=obj['metadata']; addr,n=md['logits_address'],md['logits_count']
        got=sim.state.dram[addr:addr+4*n].view(np.float32).astype(np.float64)
        want=np.load(refpath).astype(np.float64)
        err=float(np.linalg.norm(got-want)/np.linalg.norm(want))
        cosine=float(got@want/(np.linalg.norm(got)*np.linalg.norm(want)))
        numerical.update(independent_reference='opentpu.llm.qwen3.reference_logits: NumPy float forward, no ISA quantizer',
                         relative_l2=err,cosine=cosine,relative_l2_limit=0.05,cosine_minimum=0.995,
                         passed=err<0.05 and cosine>0.995)
    result={'prediction':{'cycles':pred['cycles'],'label':pred['label']},'numerical':numerical}
    if a.rtl:
        # New bounded run; no patch to the existing testbench or measurement boundary.
        # Use 4 build jobs on this shared host.
        flags=list(rtlsim.VFLAGS); flags[flags.index('-j')+1]='4'; rtlsim.VFLAGS=flags
        uarch={'WIN':16,'RPB':32,'WPB':1,'FIFO_DEPTH':1024}
        if a.rtl_executable:
            exe=a.rtl_executable.resolve()
            build_receipt=exe.with_name('Vtb_top__verFiles.dat')
            command=next(line for line in build_receipt.read_text().splitlines() if line.startswith('C '))
            built={k:int(v) for k,v in re.findall(r'-G(\w+)=(\d+)',command)}
            expected=rtlsim.top_params(cfg,8,uarch,False)
            if any(built.get(k)!=v for k,v in expected.items()):
                raise ValueError('cached RTL executable configuration differs from bundle/validation configuration')
            # Only bypass rebuilding, not execution, measurement or the golden comparator.
            rtlsim.build_top=lambda *args,**kw: exe
        drams,tmems,stats=rtlsim.run(cfg,[prog],[image],args=obj.get('args'),keep=out/'rtl',
                                  dram_lat=8,max_cycles=2_000_000,trace=True,uarch=uarch,axi=False,boot=False)
        bit_exact=np.array_equal(drams[0],gold.dram) and np.array_equal(tmems[0],gold.tmem)
        # stats['cycles'] is the unaltered RTL run-to-halt counter.
        measured=stats['cycles']; measured=max(measured) if isinstance(measured,list) else measured
        result['rtl']={'label':'measured-verilator','cycles':measured,'bit_exact':bool(bit_exact),
                       'version':rtlsim.verilator_version(),'uarch':uarch,'dram_lat':8,
                       'stats':stats,'memory_conditions_match':memory_idle,
                       'comparison_status':'same program, idle external memory' if memory_idle else
                       'NOT_COMPARABLE_TIMING: behavioral DRAM is not the simple adapter/Ramulator',
                       'timing_error_percent':100*(pred['cycles']/measured-1) if memory_idle else None,
                       'acceptance_gate_percent':5.0 if memory_idle else None,
                       'timing_gate_passed':abs(pred['cycles']/measured-1)<=0.05 if memory_idle else None}
        result['rtl']['binary_sha256']=digest(rtlsim.build_top(cfg,8,uarch))
        if a.rtl_executable:
            result['rtl']['build_receipt_sha256']=digest(build_receipt)
            result['rtl']['binary_source_freshness']='reused artifact; build receipt retained, no fresh source fingerprint'
            (out/'rtl-build.dat').write_text(build_receipt.read_text())
    files=[Path(a.bundle)/name for name in ('input.json','prog_0.hex','dram_0.bin')]
    if refpath.exists(): files.append(refpath)
    write_json(out/'manifest.json',manifest(files))
    write_json(out/'validation.json',result)
    write_json(out/'metrics.json',pred)
    print(json.dumps(result))
    if not (numerical['isa_bit_exact'] and numerical.get('passed',True) and result.get('rtl',{}).get('bit_exact',True)):
        return 1
    return 2 if result.get('rtl',{}).get('timing_gate_passed') is False else 0

if __name__=='__main__': raise SystemExit(main())
