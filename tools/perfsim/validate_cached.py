#!/usr/bin/env python3
"""Compare a bundle with a previously built, parameter-checked Verilator tb_top.

This does not rebuild or mutate RTL, its testbench, ISA reference, or measurement counters.
Binary/source freshness is a separate limitation recorded explicitly in the receipt.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from opentpu.isasim import Machine
from opentpu.perfsim import Simulator
from opentpu.perfsim.io import load_bundle,digest,manifest,state_hash,write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',required=True,type=Path)
    ap.add_argument('--executable',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()): ap.error('output must be empty')
    a.out.mkdir(parents=True,exist_ok=True)
    cfg,prog,image,obj=load_bundle(a.bundle)
    if obj.get('args'): ap.error('cached validator currently supports no runtime arguments')
    exe=a.executable.resolve(); build=exe.with_name('Vtb_top__verFiles.dat')
    build_text=build.read_text()
    cmdline=next(line for line in build_text.splitlines() if line.startswith('C '))
    actual={k:int(v) for k,v in re.findall(r'-G(\w+)=(\d+)',cmdline)}
    expected={'S':cfg.S,'D':cfg.D,'MCOLS':cfg.MCOLS,'ACT_BLOCKS':cfg.ACT_BLOCKS,
              'ACT_ROWS':cfg.act_rows,'TMEM_WORDS':cfg.TMEM_WORDS,'IMEM_WORDS':cfg.IMEM_WORDS,
              'DRAM_WORDS':cfg.DRAM_BYTES//4,'DRAM_LAT':8,'LANES':cfg.LANES,'AXI':0,
              'WIN':16,'RPB':32,'WPB':1,'MXU_IMPL':2,'FIFO_DEPTH':1024}
    if any(actual.get(k)!=v for k,v in expected.items()):
        raise ValueError(f'RTL configuration mismatch: expected {expected}, got {actual}')
    if cfg.PAIR or cfg.STREAM or cfg.DSTEP or cfg.SEQ_WC:
        raise ValueError('optional features require a separate checked build')
    for name in ('prog_0.hex','dram_0.bin'): shutil.copyfile(a.bundle/name,a.out/name)
    command=[str(exe),f'+dir={a.out.resolve()}','+max_cycles=2000000','+trace']
    run=subprocess.run(command,capture_output=True,text=True,timeout=120)
    log=run.stdout+run.stderr; (a.out/'stdout.log').write_text(log)
    match=re.search(r'RESULT cycles=(\d+) halted=(\d+) error=(\d+)',log)
    if run.returncode or not match or match[2]!='1' or match[3]!='0':
        raise RuntimeError('RTL failed; inspect stdout.log')
    measured=int(match[1])
    gold=Machine(cfg,[prog],[image]).run().slices[0]
    sim=Simulator(cfg,prog,image); prediction=sim.run()
    # tb_top writes raw DRAM bytes; rtlsim's unmodified reader uses uint8 too.
    dram=np.fromfile(a.out/'dram_out_0.bin',np.uint8)
    words=(a.out/'tmem_0.hex').read_text().split()
    tmem=np.array([int(t,16) for t in words],np.uint32)
    exact=np.array_equal(dram,gold.dram) and np.array_equal(tmem,gold.tmem)
    idle=not any(i.op in (0x10,0x11,0x20,0x22) for i in prog)
    result={'schema_version':1,'label':'measured-verilator','cycles':measured,
            'rtl_bit_exact_vs_isa':bool(exact),'simulator_bit_exact_vs_isa':state_hash(sim.state)==state_hash(gold),
            'predicted_cycles':prediction['cycles'],'prediction_label':prediction['label'],
            'memory_conditions_match':idle,'timing_error_percent':100*(prediction['cycles']/measured-1) if idle else None,
            'timing_gate_percent':5.0 if idle else None,
            'timing_gate_passed':abs(prediction['cycles']/measured-1)<=0.05 if idle else None,
            'timing_status':'same program, idle external memory' if idle else 'NOT_COMPARABLE: RTL fixed LAT8 vs simulator adapter',
            'binary_sha256':digest(exe),'build_receipt_sha256':digest(build),'parameters':expected,
            'source_freshness':'binary built locally during this implementation; reused with parameter check, no fresh complete RTL fingerprint',
            'command':command}
    refpath=a.bundle/'reference_logits.npy'
    if refpath.exists():
        md=obj['metadata']; addr,n=md['logits_address'],md['logits_count']
        got=dram[addr:addr+4*n].view(np.float32).astype(np.float64)
        ref=np.load(refpath).astype(np.float64)
        err=float(np.linalg.norm(got-ref)/np.linalg.norm(ref))
        result['independent_numpy_relative_l2']=err
        result['independent_numpy_passed']=err<0.05
    (a.out/'rtl-build.dat').write_text(build_text)
    write_json(a.out/'validation.json',result)
    inputs=[a.bundle/f for f in ('input.json','prog_0.hex','dram_0.bin')]
    if refpath.exists(): inputs.append(refpath)
    write_json(a.out/'manifest.json',manifest(inputs))
    print(json.dumps(result))
    if not (exact and result['simulator_bit_exact_vs_isa'] and result.get('independent_numpy_passed',True)):
        return 1
    return 2 if result['timing_gate_passed'] is False else 0

if __name__=='__main__': raise SystemExit(main())
