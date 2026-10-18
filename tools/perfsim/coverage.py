#!/usr/bin/env python3
"""Inventory observed programs and gates; this is a coverage report, not new measurement."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from opentpu import isa as I
from opentpu.perfsim.io import load_bundle,write_json,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('output must not exist')
    fixed=json.loads((args.fixed/'summary.json').read_text())
    native=json.loads((args.native/'summary.json').read_text())
    opcodes=Counter();functions=Counter();flags={};shapes=set();configs={}
    names={v:k for k,v in vars(I).items() if k.startswith('V_') and isinstance(v,int)}
    for case in fixed['cases']:
        cfg,program,_,_=load_bundle(args.fixed/case['name'])
        configs[str((cfg.S,cfg.D,cfg.LANES,cfg.MCOLS))]=True
        for ins in program:
            opcodes[I.OPNAMES[ins.op]]+=1
            flags.setdefault(I.OPNAMES[ins.op],set()).add(ins.flags)
            if ins.op==I.VOP:
                function=(ins.w[5]>>16)&255;functions[names.get(function,str(function))]+=1
                shapes.add(('VOP',ins.w[3]&65535,ins.w[3]>>16))
            elif ins.op==I.MM:
                shapes.add(('MM',ins.w[5]>>16&255,ins.w[3]&65535,ins.w[3]>>16))
    exact=lambda c:c['trace_match'] and c['rtl_cycles']==c['predicted_cycles']
    result={'schema_version':1,'boundary':'Observed finite corpus only. Counts are static encoded instructions, not dynamic executions.',
            'fixed_cases':len(fixed['cases']),'fixed_exact_trace_and_cycles':sum(exact(c) for c in fixed['cases']),
            'fixed_numeric_passed':sum(c['numeric_passed'] for c in fixed['cases']),
            'native_cases':len(native),'native_exact_trace_cycles_ports':sum(c['timing_passed'] for c in native),
            'native_numeric_passed':sum(c['numerical']['rtl_vs_isa_bit_exact'] and c['numerical']['model_vs_isa_bit_exact'] and
                (c['numerical']['independent_float_reference'] is None or all(v['passed'] for v in c['numerical']['independent_float_reference'].values())) for c in native),
            'native_profiles':sorted({c['profile'] for c in native}),
            'observed_configurations_S_D_LANES_MCOLS':sorted(configs),
            'static_opcodes':dict(opcodes),'static_vpu_functions':dict(functions),
            'observed_flags':{k:sorted(v) for k,v in flags.items()},'observed_shapes':sorted(shapes),
            'explicitly_unsupported':['S>1/GATHER','WAITW','WAITC/SEQ_WC','STREAM','DSTEP','HALT CHAIN','unknown functions/flags/configuration fields','divided memory-controller clock'],
            'not_established':['pretrained/full-size model timing','persistent multi-launch RTL controller timing','non-D32/LANES8/MCOLS4 parameter timing','PAIR/alternate packed scaling exhaustive coverage','WBUF/replicated/subbank/pumped TMEM topology','FPGA shell/PCIe/host scheduling/sampling','physical Fmax/resources/power'],
            'sources':{str(p):digest(p) for p in (args.fixed/'summary.json',args.native/'summary.json')}}
    write_json(args.out,result)
    print(json.dumps({k:result[k] for k in ('fixed_cases','fixed_exact_trace_and_cycles','native_cases','native_exact_trace_cycles_ports')}))


if __name__=='__main__':main()
