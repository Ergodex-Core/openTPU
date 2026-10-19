#!/usr/bin/env python3
from pathlib import Path
import sys, json, re, argparse
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from opentpu import target
ap=argparse.ArgumentParser()
ap.add_argument('output',type=Path)
ap.add_argument('--target',type=Path,default=ROOT/'targets/f2.toml')
args=ap.parse_args()
out = args.output; out.mkdir(parents=True,exist_ok=True)
t = target.load(str(args.target.resolve()))
p = target.rtl_params(t)
allowed = set(re.findall(r'parameter\s+(?:int|bit|logic\s*\[[^]]+\])\s+(\w+)', (ROOT/'rtl/shell/otpu_board.sv').read_text()))
p = {k:v for k,v in p.items() if k in allowed}
p.update(HAS_I2C=0, HOSTCAL=0, CORE_KHZ=t.core_khz)
(out/'target.json').write_text(json.dumps(t.to_dict() | {'rtl_params':p},indent=2)+'\n')
(out/'config.tcl').write_text('set part {'+t.tech['part']+'}\nset core_khz '+str(t.core_khz)+'\nset generics {'+' '.join(k+'='+str(int(v)) for k,v in p.items())+'}\nset rtl_files {'+' '.join(target.sources(t))+'}\n')
print(json.dumps({'arch_hash':t.to_dict()['arch_hash'],'parameters':p},indent=2))
