#!/usr/bin/env python3
"""Consolidate canonical retained observations; never sum overlapping historical corpora."""
import argparse,gzip,json,sys,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from opentpu.perfsim.io import digest,write_json


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--latest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--pytest-count',type=int,required=True);a=ap.parse_args()
 if a.out.exists():ap.error('output must not exist')
 log=(a.latest/'pytest.log').read_text();m=re.search(r'(\d+) passed',log)
 if not m or int(m[1])!=a.pytest_count:raise ValueError('pytest count must match the retained latest log')
 a.out.mkdir(parents=True);base=ROOT/'tools/perfsim/receipts';sources={};rows=[]
 def read(path):
  if not path.exists():path=path.with_suffix(path.suffix+'.gz')
  raw=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes();sources[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)]=digest(path);return json.loads(raw)
 def verdict(s):
  if 'numeric_bit_exact' in s:return bool(s['numeric_bit_exact']),s.get('independent_reference')
  n=s.get('numerical')
  if n:
   f=n.get('independent_float_reference');f=None if f is None else {'passed':all(v['passed'] for v in f.values())}
   return n['rtl_vs_isa_bit_exact'] and n['model_vs_isa_bit_exact'],f
  return bool(s.get('numeric_passed')),None
 def add(name,cases,timing,fresh=0,notes=''):
  v=[verdict(s) for s in cases];r={'family':name,'canonical_observations':len(cases),'exact_timing_passed':sum(timing(s) for s in cases),'bitwise_passed':sum(bool(b) for b,f in v),'independent_float_passed':sum(f is not None and f['passed'] for b,f in v),'independent_float_failed':sum(f is not None and not f['passed'] for b,f in v),'independent_float_not_recorded':sum(f is None for b,f in v),'fresh_latest_RTL_executions':fresh,'notes':notes};rows.append(r)
 fixed=read(base/'2026-10-19-round3/fixed/summary.json')['cases'];add('Fixed-memory kernel corpus',fixed,lambda s:s['trace_match'] and s['rtl_cycles']==s['predicted_cycles'],notes='112-case round3 corpus replaces overlapping round2 corpus; behavioral RAM, not HBM2')
 add('Native Ramulator original corpus',read(base/'2026-10-19-round3/native/summary.json'),lambda s:s['timing_passed'],notes='HBM2 and historical DDR4/stress profiles')
 shapes=[read(p/'summary.json') for p in sorted((base/'2026-10-19-round4').glob('operators-d*/*'))]+[read(base/'2026-10-19-round4'/n/'summary.json') for n in ('full-rtl-gemma-up','full-rtl-bonsai-down')]
 add('Model-sized operators',shapes,lambda s:s['timing_passed'],notes='Two original int4 float-quality failures retained; all packed arithmetic bitwise passes')
 native=[read(p/'summary.json') for group in ('native','waitc') for p in sorted((base/'2026-10-19-round5'/group).iterdir())]
 add('WAITC and stream-engine kernels',native,lambda s:s['timing_passed'],notes='Final nondegenerate cases only; zero-reference DOT diagnostics remain retained')
 add('Cold hybrid launches', [read(p/'summary.json') for p in sorted((base/'2026-10-19-round5/cold').iterdir())],lambda s:s['timing_passed'])
 old=[read(base/'2026-10-19-round4'/('persistent-'+n)/'summary.json') for n in ('hbm5','ddr5','stress5-gap256','retained')]
 add('Persistent original topology/state probes',[r for s in old for r in s['stages']],lambda s:s['timing_passed'],notes='18 launches, four retained-controller sequences')
 old=[read(p/'summary.json') for p in sorted((base/'2026-10-19-round5/persistent').iterdir())]
 add('Persistent E2 hybrid topology',[r for s in old for r in s['stages']],lambda s:s['timing_passed'],fresh=9,notes='Latest six HBM2 reruns plus three packaging-smoke reruns; duplicates not added to cumulative count')
 latest=read(a.latest/'run-index.json');latest_s={r['name']:read(a.latest/r['relative_summary']) for r in latest}
 add('Persistent E4 hybrid topology',[s for name in ('e4-dstep','e4-stream') for s in latest_s[name]['stages']],lambda s:s['timing_passed'],fresh=6,notes='New HBM2 compute variant; does not validate pumped TMEM or a board shell')
 add('E2/E4 full-row epilogue kernels',[s for name in ('epilogue-e2','epilogue-e4') for s in latest_s[name]['stages']],lambda s:s['timing_passed'],fresh=6,notes='Three/four-row compiled GEMMs; independent images, persistent controller')
 fresh=sum(len(s['stages']) for s in latest_s.values());assert fresh==sum(r['fresh_latest_RTL_executions'] for r in rows)
 report={'counting':'Canonical retained comparison entries, not unique workloads or independent statistical samples. Round3 fixed corpus supersedes overlapping round2. Latest reruns do not inflate cumulative count. Pytest counts are the current suite, never summed across rounds.',
         'rows':rows,'canonical_RTL_observations':sum(r['canonical_observations'] for r in rows),'canonical_exact_timing_passed':sum(r['exact_timing_passed'] for r in rows),'latest_fresh_RTL_executions':fresh,'latest_new_canonical_observations':12,'latest_repeat_executions':fresh-12,'latest_pytest_passed':a.pytest_count,
         'historical_pytest_counts':[31,108,170,200,283],'sources':sources,
         'int4':{'original_float_gate':'FAILED and unchanged','D32_relative_l2':0.057727200854166834,'D128_relative_l2':0.07946628603309096,'independent_packed_arithmetic_vs_RTL':'bit-exact PASS','scope':'two seeded operators; not full-model quality'},
         'historical_negative_results':'Initial timing failure, round2 three tiny-model timing failures and round5 superseded/zero-reference diagnostics remain in their original receipts; excluded from current timing-passing canonical corpus, never erased.',
         'full_pretrained_model_validated':False,'production_pumped_TMEM_validated':False,'physical_Fmax_resources_validated':False}
 write_json(a.out/'coverage.json',report)
 lines=['# Consolidated validation coverage','',report['counting'],'','| Evidence family | Cumulative canonical | Exact timing | Bitwise | Float pass / fail / unrecorded | Fresh RTL this run |','|---|---:|---:|---:|---|---:|']
 for r in rows:lines.append(f"| {r['family']} | {r['canonical_observations']} | {r['exact_timing_passed']} | {r['bitwise_passed']} | {r['independent_float_passed']} / {r['independent_float_failed']} / {r['independent_float_not_recorded']} | {r['fresh_latest_RTL_executions']} |")
 lines+=['',f"**{report['canonical_RTL_observations']} canonical observations; {fresh} fresh RTL launches this run (12 new, {fresh-12} reruns); {a.pytest_count} current pytest cases pass.**",'','Frozen pytest replays are not new RTL executions. Historical pytest counts are not additive.','', '**Int4 remains a float-quality failure:** D32 5.77272%, D128 7.94663%. Independent packed arithmetic matches RTL bitwise. No gate was relaxed.','',report['historical_negative_results'],'','Full pretrained inference, pumped TMEM and physical Fmax/resources remain unvalidated. Historical DDR4 results are retained; this run adds HBM2 only.','']
 lines += [f"- {r['family']}: {r['notes']}" for r in rows if r['notes']]
 (a.out/'coverage.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:report[k] for k in ('canonical_RTL_observations','latest_fresh_RTL_executions','latest_pytest_passed')}))
if __name__=='__main__':main()
