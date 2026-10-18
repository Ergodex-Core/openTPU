#!/usr/bin/env python3
"""Report the implemented ISA domain separately from the production board target."""
import argparse,json,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from opentpu import isa as I
from opentpu.perfsim.io import digest,load_bundle,write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bundles',type=Path,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():ap.error('output must not exist')
    implemented={'NOP','HALT','LI','ADDI','LOOP','BAR','RLD','LD','ST','MM','QACT','QST','VOP'}
    programs={}
    for base in a.bundles:
        for p in sorted(base.glob('*/input.json')):
            cfg,program,_,_=load_bundle(p.parent)
            ops=sorted({I.OPNAMES[i.op] for i in program})
            programs[str(p.parent)]={'opcodes':ops,'all_base_opcodes_implemented':set(ops)<=implemented,'D':cfg.D,'ACT_ROWS':cfg.act_rows,'PAIR':cfg.PAIR,'DSTEP':cfg.DSTEP,'STREAM':cfg.STREAM,'SEQ_WC':cfg.SEQ_WC}
    target=ROOT/'targets/ypcb-00338.toml';board=tomllib.loads(target.read_text())
    result={'baseline_commit':'3c542bd9cb98f11499729ea52703788f4588e4d2','implemented_base_opcodes':sorted(implemented),'baseline_ISA_opcodes_missing':sorted(set(I.OPNAMES.values())-implemented),
            'implemented_inference_path':'single slice, ordinary footprint scoreboard, replicated/count-arbitrated TMEM, dense/packed MXU, VPU DeltaNet fallback, explicit native Ramulator controller',
            'compiled_program_inventory':programs,
            'production_target':{'path':str(target.relative_to(ROOT)),'sha256':digest(target),'architecture':board['arch'],'uarch':board['uarch'],'platform':board['platform']},
            'production_target_timing_supported':False,
            'missing_production_paths':['SEQ_WC wait-entry issue','DSTEP DMA recurrence','STREAM engine','TMEM_IMPL=3 replay/guest/pumped dual-port topology','DDR3 LiteDRAM/native AXI shell and channel hashing','program-loader/HALT CHAIN and host WAITW polling'],
            'not_needed_by_measured_fallback_programs':['DSTEP/STREAM: measured DeltaNet head uses RDOT/OUTER VPU path','GATHER: S=1 compiler collectives reduce to local work','WAITW/CHAIN: explicit program-resident host launch boundary'],
            'frontends':{'gemma3_module_in_pinned_baseline':(ROOT/'opentpu/llm/gemma3.py').exists(),'bonsai_gguf_module_in_pinned_baseline':(ROOT/'opentpu/llm/bonsai2.py').exists(),'local_newer_exports':'not silently imported; T2/wide fields, streaming/wait-count targets remain incompatible'},
            'memory_contract':'DDR4 and HBM2 supplied profiles only; shared simulated controller, not claimed equivalent to board external-memory shell',
            'full_model_limits':['no complete pretrained Gemma3/Bonsai inference run','no production bitstream cycle equivalence','no Fmax/resource/power measurement'],
            'sources':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'opentpu/isa.py',ROOT/'opentpu/isasim.py',ROOT/'opentpu/perfsim/engine.py',ROOT/'opentpu/perfsim/units.py')}}
    write_json(a.out,result)
if __name__=='__main__':main()
