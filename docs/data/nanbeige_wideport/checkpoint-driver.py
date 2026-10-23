"""Bounded exact-model ISA acceptance on an isolated wide-port runtime. No FPGA use."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,time,hashlib,shutil,signal
root=Path('/home/ubuntu/nanbeige-wideport-20261023');source=root/'source';work=root/'checkpoint-002'
reg=json.loads((root/'regression-003.json').read_text())
assert reg['status']=='complete' and reg['exit_code']==0 and reg['physical_open_attempts']==0
assert not work.exists()
for name,digest in reg['source_files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert mem>40*2**30 and shutil.disk_usage(root).free>30*2**30 and os.getloadavg()[0]<8
os.nice(19);os.sched_setaffinity(0,{21})
for key in list(os.environ):
 if key.startswith('OTPU_'):del os.environ[key]
os.environ.update(OTPU_IMG_CACHE=str(root/'img-cache'),OTPU_IMAGE_CACHE='0',OTPU_BUILD_JOBS='1',OTPU_PROG_CACHE='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
work.mkdir();os.chdir(source);sys.path.insert(0,str(source))
from opentpu.host import board,f2
attempts=[]
def deny(*a,**kw):attempts.append(str(a[:1]));raise AssertionError('physical transport forbidden in ISA acceptance')
board.open_transport=deny;board.XdmaTransport.__init__=deny;f2.F2Transport.__init__=deny
from opentpu.llm.nanbeige import Spec,load_weights
from opentpu.llm.nanbeige_strict import StrictEngine
from opentpu.target import config,load,arch_hash
from opentpu import progcache,imgcache
from tools import nanbeige_strict as cli
from tools.nanbeige_board import validate_profile
model=Path('/home/ubuntu/rachit/nanbeige-burst200/models/Nanbeige4.2-3B-GPTQ-Int8');baseline=Path('/home/ubuntu/nanbeige-strict-20261023')
raw=json.loads((baseline/'checkpoint-003/isa.json').read_text());chat=json.loads((baseline/'checkpoint-004/isa.json').read_text());ref=json.loads((baseline/'source-compare-001/reference.json').read_text())
assert raw['status']==chat['status']==ref['status']=='complete'
assert raw['physical_open_attempts']==chat['physical_open_attempts']==ref['physical_open_attempts']==0
identity=json.loads(json.dumps(imgcache.checkpoint(model)))
assert identity==raw['checkpoint_identity']==chat['checkpoint_identity'], 'checkpoint file metadata/header identity changed'
revisions=json.loads((model.parents[1]/'model-revisions.json').read_text())['Nanbeige4.2-3B-GPTQ-Int8']
assert revisions==ref['model']=={'repo_id':'Nanbeige/Nanbeige4.2-3B-GPTQ-Int8','revision':'e8c6ccdb655bbcf410952686026d3fcbf4c07392'}
assert [raw['result']['tokens'],chat['result']['tokens']]==[x['reference_tokens'] for x in ref['comparisons']]
spec=Spec.from_hf(model);profile=load(cli.PROFILE);cfg=config(profile)
tokenizer,chat_ids,_=cli.encode_request({'prompt':'Hi','max_new_tokens':2},spec,128,model)
assert chat_ids==chat['prompt_tokens'] and len(chat_ids)==36
receipt={'status':'running','stage':'packing','hardware_used':False,'started_utc':datetime.now(timezone.utc).isoformat(),'runtime_base':'18fb978033799f0f21258f0e6e085c0360322d96','model':revisions,'checkpoint_identity':imgcache.checkpoint(model),'source_files':reg['source_files'],'source_hash':progcache.source(),'source_hash_algorithm':'blake2b-128','profile_contract':validate_profile(profile),'arch_hash':arch_hash(cfg),'baseline':{'strict_commit':'16205f3fd1a94043b9076039cd8a1139a699eef6','source_reference':'FP64 oracle with source weights; not official BF16 rounding','raw_tokens':raw['result']['tokens'],'chat_tokens':chat['result']['tokens']},'physical_open_attempts':0,'cases':[],'completed_backend_calls':0,'affinity':sorted(os.sched_getaffinity(0))}
def save():
 receipt['updated_utc']=datetime.now(timezone.utc).isoformat();(work/'progress.json').write_text(json.dumps(receipt,indent=2,default=cli._json_default)+'\n')
def timed_out(*a):raise TimeoutError('bounded ISA checkpoint gate exceeded 900 seconds')
signal.signal(signal.SIGALRM,timed_out);signal.alarm(900)
save();print('START',json.dumps(receipt),flush=True)
try:
 start=time.monotonic();eng=StrictEngine(spec,load_weights(model),cfg=cfg,backend='isa',cap=128)
 receipt.update(stage='forward',setup_seconds=time.monotonic()-start,layout=eng.strict_plan)
 assert eng.image.head is None and eng.image.head_rl==4 and not eng.image.embed_host
 read=eng.backend.read;out=eng.image.lookup['gen']['out'];read_calls=[]
 def tokens_only(sid,at,n):
  assert out<=at and at+n<=out+4*(eng.cap+1),(at,n)
  read_calls.append((at,n));return read(sid,at,n)
 eng.backend.read=tokens_only
 write=eng.backend.write;model_io={eng.image.io[k] for k in ('x','cos','sin')}
 def input_guard(sid,at,data):
  assert at not in model_io,('host model input',at)
  return write(sid,at,data)
 eng.backend.write=input_guard
 for method in ('run','run_chain'):
  if not hasattr(eng.backend,method):continue
  original=getattr(eng.backend,method)
  def track(*a,original=original,**kw):
   result=original(*a,**kw);receipt['completed_backend_calls']+=1;save();return result
  setattr(eng.backend,method,track)
 for label,request,expected in [
  ('raw',{'prompt_tokens':raw['prompt_tokens'],'max_new_tokens':2},raw['result']['tokens']),
  ('chat',{'prompt':'Hi','max_new_tokens':2},chat['result']['tokens']),
  ('raw_reset_replay',{'prompt_tokens':raw['prompt_tokens'],'max_new_tokens':2},raw['result']['tokens'])]:
  receipt['active_case']=label;save();print('CASE_START',label,flush=True)
  result=cli.run_request(eng,request,model,hardware=False);result['text']=tokenizer.decode(result['tokens'],skip_special_tokens=False)
  case={'name':label,'result':result,'expected_tokens':expected,'exact_tokens_match':result['tokens']==expected}
  receipt['cases'].append(case);save();print('CASE_DONE',json.dumps(case),flush=True)
  assert case['exact_tokens_match'],label
 receipt.update(status='complete',stage='complete',read_calls=read_calls,precision=cli.PRECISION,finished_utc=datetime.now(timezone.utc).isoformat(),physical_open_attempts=len(attempts))
 assert not attempts
except BaseException as exc:
 receipt.update(status='failed',stage='failed',error_type=type(exc).__name__,error=str(exc),physical_open_attempts=len(attempts));save();raise
else:
 save();(work/'isa.json').write_text(json.dumps(receipt,indent=2,default=cli._json_default)+'\n');print('FINAL',json.dumps(receipt),flush=True)
finally:
 signal.alarm(0)
