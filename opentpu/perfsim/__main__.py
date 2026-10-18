"""python -m opentpu.perfsim --bundle BUNDLE --profile PROFILE.json --out RUN"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from ..isasim import Machine
from . import Simulator, Architecture, MemoryConfig, Unsupported
from .io import load_bundle,manifest,state_hash,write_json,digest


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',required=True)
    ap.add_argument('--profile',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--bridge',help='override local Ramulator shared library path')
    ap.add_argument('--numeric-check',action='store_true',help='compare scheduled functional commits with sequential ISA engine (shared arithmetic)')
    ap.add_argument('--trace',action='store_true',help='write accepted/completed external memory events')
    args=ap.parse_args(argv)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        ap.error('output directory is not empty; refusing to overwrite evidence')
    out.mkdir(parents=True,exist_ok=True)
    try:
        cfg,prog,image,obj=load_bundle(args.bundle)
        path=Path(args.profile).resolve()
        profile=json.loads(path.read_text())
        if set(profile)-{'schema_version','name','description','architecture','memory'}:
            raise ValueError('unknown profile fields')
        mc=dict(profile['memory'])
        mc['record_trace']=args.trace
        for key in ('bridge','ramulator_config'):
            if mc.get(key) and not Path(mc[key]).is_absolute():
                mc[key]=str((path.parent/mc[key]).resolve())
        if args.bridge: mc['bridge']=str(Path(args.bridge).resolve())
        sim=Simulator(cfg,prog,image,architecture=Architecture(**profile.get('architecture',{})),
                      memory=MemoryConfig(**mc),args=obj.get('args'),
                      labels={int(k):v for k,v in obj.get('labels',{}).items()})
        files=[path,*[Path(args.bundle)/f for f in ('input.json','prog_0.hex','dram_0.bin')]]
        receipt=manifest(files)
        if mc.get('bridge'): receipt['bridge_sha256']=digest(mc['bridge'])
        if mc.get('ramulator_config'): receipt['ramulator_config_sha256']=digest(mc['ramulator_config'])
        write_json(out/'manifest.json',receipt)
        result=sim.run()
        result['profile']=profile['name']
        result['state_sha256']=state_hash(sim.state)
        if args.numeric_check:
            golden=Machine(cfg,[prog],[image],obj.get('args')).run().slices[0]
            result['numerical_check']={'reference':'sequential ISA; shared arithmetic, independent schedule',
                                       'bit_exact':state_hash(golden)==state_hash(sim.state)}
            if not result['numerical_check']['bit_exact']:
                raise AssertionError('scheduled numeric state differs from sequential ISA')
        write_json(out/'metrics.json',result)
        if args.trace:
            with (out/'memory.jsonl').open('w') as f:
                for row in sim.memory.trace: f.write(json.dumps(row)+'\n')
        sim.state.dram.tofile(out/'dram_final.bin')
        print(json.dumps({k:result[k] for k in ('label','backend','cycles','seconds','timing_validation')}))
        return 0
    except (ValueError,RuntimeError,OSError,AssertionError) as exc:
        write_json(out/'failure.json',{'status':'unsupported' if isinstance(exc,Unsupported) else 'failed',
                                      'reason':str(exc),'metrics_valid':False})
        print(str(exc),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
