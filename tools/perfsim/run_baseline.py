#!/usr/bin/env python3
"""Reproduce the bounded HBM2 baseline without downloads, credentials or board access.

Build an explicit compute-RTL variant or verify a supplied persistent RTL executable.
Outputs are immutable per invocation. --model-only is explicitly not RTL validation.
"""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from opentpu.perfsim.io import write_json,manifest


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--bridge',type=Path,required=True)
 ap.add_argument('--engine',choices=['dstep','stream'],default='stream');ap.add_argument('--epilogue',type=int,choices=[2,4],default=2)
 choice=ap.add_mutually_exclusive_group(required=True);choice.add_argument('--build-rtl',action='store_true');choice.add_argument('--rtl-executable',type=Path);choice.add_argument('--model-only',action='store_true')
 a=ap.parse_args();out=a.out.resolve();bridge=a.bridge.resolve()
 if out.exists():ap.error('output must not exist')
 if not bridge.is_file():ap.error('build the pinned Ramulator bridge first; no automatic fallback/download')
 out.mkdir(parents=True);commands=[]
 def run(script,*args):
  cmd=[sys.executable,str(ROOT/'tools/perfsim'/script),*map(str,args)];commands.append(cmd);write_json(out/'commands.json',commands)
  with (out/(script+'.log')).open('w') as log:p=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError(f'{script} failed with exit {p.returncode}; see {out/(script+".log")}')
 profile=ROOT/f'tools/perfsim/configs/native-hbm2-200mhz{"-e4" if a.epilogue==4 else ""}.json'
 flags=['--waitc','--chunk-prefill']+(['--stream'] if a.engine=='stream' else [])
 run('hybrid_example.py','--out',out/'bundles',*flags)
 if not a.model_only:
  exe=a.rtl_executable.resolve() if a.rtl_executable else out/'rtl/obj/Vtb_perfsim'
  if a.build_rtl:run('build_rtl_ramulator.py','--bundle',out/'bundles/hybrid_0','--profile',profile,'--persistent','--out',out/'rtl')
  run('compare_persistent.py','--bundles',out/'bundles','--profile',profile,'--bridge',bridge,'--executable',exe,'--out',out/'rtl-check')
 run('sequence.py','--bundles',out/'bundles','--profile',profile,'--bridge',bridge,'--out',out/'metrics')
 seq=json.loads((out/'metrics/summary.json').read_text());check=None if a.model_only else json.loads((out/'rtl-check/summary.json').read_text())
 report={'baseline':'opentpu-native-hbm2-program-resident-v1','engine':a.engine,'epilogue_lanes':a.epilogue,
         'validation':'model-only prediction' if a.model_only else 'strict RTL cycles/instructions/ports plus independent NumPy logits',
         'timing_passed':None if check is None else check['timing_passed'],'numeric_passed':all(s['numerical_check']['passed'] for s in seq['stages']),
         'cycles':[s['cycles'] for s in seq['stages']],'core_frequency_hz':200000000,'device_ttft_seconds':seq['device_ttft_seconds'],
         'conditional_decode_tokens_per_second':seq['conditional_decode_tokens_per_second'],'prefill_tokens':2,'decode_tokens':2,
         'full_pretrained_model':False,'physical_hardware_measured':False,'loading_chain_waitw_included':False,'tmem_impl':0,
         'limits':['tiny random complete hybrid topology only','program/input loading, sampling and host time excluded','pumped TMEM and production shell unsupported','physical Fmax and resources unverified'],
         'original_int4_quality_failures':'still failed at D32=5.77272%, D128=7.94663%; packed arithmetic is independently bit-exact; see validation-round5.md'}
 write_json(out/'report.json',report);write_json(out/'manifest.json',manifest([profile,bridge,out/'commands.json',out/'bundles/sequence.json']))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
