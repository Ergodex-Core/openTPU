"""Offline synthetic lifecycle tests; no AWS, FPGA utilities, locks or BARs."""
import copy,importlib.util,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def load(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
S=load('c1_f2_alex_session')
B=load('c1_f2_alex_bench')
from opentpu import c1_f2 as C
spec=importlib.util.spec_from_file_location('canonical_fake',ROOT/'tests/test_c1_f2_host.py');fake_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fake_module)

def baseline():
 return dict(schema=1,kind='c2-f2-restoration-baseline',instance_id=S.INSTANCE,agfi_id='agfi-00c2d61ec51bd89aa',afi_id='afi-07e20bf5679cf37d7',owner='438374459156',shell_version='0x10212415',identity=dict(id=1330925653,version=8389640,regmap=5,build_id=2118297126,platform=3,core_khz=200000))
def q():
 return dict(agfi_id='agfi-0123456789abcdef0',build_id='0d654453',identity=dict(id=C.ID,abi=C.ABI,build_id=int('0d654453',16),core_khz=200000,max_rows=512,depth=128,credits=32),baseline=baseline())
class Lock:
 def __init__(self,ops):self.ops=ops
 def release(self):self.ops.events.append('release')
class Ops:
 def __init__(self,fail=None):self.events=[];self.records=[];self.restoring=False;self.fail=fail;self.image=baseline()['agfi_id'];self.logging_errors=[]
 def instance_id(self):return S.INSTANCE
 def record(self,r):self.records.append(copy.deepcopy(r))
 def verify_image_binding(self,q):self.events.append('verify');return {'synthetic_test_binding':True}
 def acquire(self,name):self.events.append('acquire');return Lock(self)
 def no_foreign_handles(self,bdf):self.events.append('owner_check')
 def discover(self,allow_unconfigured=False):return S.INITIAL_BDF
 def describe(self):return dict(agfi=self.image,state='loaded',status_code=0,error='ok',error_code=0)
 def baseline(self,bdf,wait=False):
  r={**baseline()['identity'],'status':104}
  if wait and self.fail=='restore_identity':r['build_id']+=1
  return r
 def check_cancel(self):pass
 def load(self,agfi,restoring=False):
  self.events.append('restore' if restoring else 'load');self.image=agfi
  if not restoring and self.fail=='partial_load':raise RuntimeError('partial load')
  if restoring and self.fail=='restore_load':raise RuntimeError('restore failed')
 def rescan(self):self.events.append('rescan')
 def wait_terminal(self):self.events.append('wait_terminal')
 def candidate(self,bdf,build):
  if self.fail=='candidate':raise RuntimeError('wrong candidate')
  return {**q()['identity'],'status':0,'hbm_ready':1}
 def benchmark(self,*args):
  self.events.append('benchmark')
  if self.fail=='benchmark':raise TimeoutError('original timeout')

def test_success_restores_before_lock_release():
 o=Ops();r=S.session(o,q(),['synthetic_fixture'],repeats=1)
 assert r['phase']=='complete' and r['restored'] and r['original_image']==r['restored_image']
 assert r['completed_fixtures']==['synthetic_fixture'] and o.events.index('restore')<o.events.index('release')

@pytest.mark.parametrize('failure',['partial_load','candidate','benchmark','restore_identity','restore_load'])
def test_failures_restore_and_never_complete(failure):
 o=Ops(failure)
 with pytest.raises(RuntimeError):S.session(o,q(),['synthetic_fixture'],repeats=1)
 assert 'restore' in o.events and o.records[-1]['phase']=='failed' and o.events[-1]=='release'
 if failure not in ('restore_identity','restore_load'):assert o.records[-1]['restored']
 else:assert 'restore_error' in o.records[-1]

def test_preload_original_mismatch_never_loads():
 o=Ops();o.image='agfi-09999999999999999'
 with pytest.raises(RuntimeError):S.session(o,q(),[],repeats=1)
 assert 'load' not in o.events and 'restore' not in o.events

@pytest.mark.parametrize('missing_verifier',[False,True])
def test_unverified_request_stops_before_localops(tmp_path,monkeypatch,missing_verifier):
 monkeypatch.setattr(S,'source_snapshot',lambda revision:dict(revision=revision,files={},helpers={}))
 request=tmp_path/'artifact.json';request.write_text('{"route_qualified": true}')
 b=tmp_path/'baseline.json';b.write_text(json.dumps(baseline()))
 fixture=tmp_path/'fixture.npz';C.make_fixture(1,1,1).save(fixture)
 out=tmp_path/'session';plan=S.prepare_plan(out,revision='a'*40,artifact_request=request,artifact_request_sha256=S.sha(request),baseline=b,baseline_sha256=S.sha(b),fixtures=[fixture])
 assert plan['route_qualified'] is False and plan['qualification_status']=='pending_independent_C1_artifact_verification'
 monkeypatch.setattr(S,'LocalOps',lambda *a:pytest.fail('hardware operations instantiated'))
 if missing_verifier:monkeypatch.setattr(S,'ROOT',tmp_path)
 error=RuntimeError if missing_verifier else ValueError
 message='verifier unavailable' if missing_verifier else 'Wrong C1 runtime request schema or target'
 with pytest.raises(error,match=message):S.execute_prepared(out,S.sha(out/'plan.json'))
 assert not (out/'qualification.json').exists()
 (out/'fixtures/00.npz').write_bytes(b'changed')
 with pytest.raises(ValueError,match='Frozen'):S.verify_saved_inputs(out)

def test_timeout_snapshot_precedes_abort():
 f=fake_module.Fake();b=B.DiagnosticBenchmark(f,expected_build_id=1234);b.preload(C.make_fixture(1,1,1));f.hang=True
 with pytest.raises(TimeoutError,match='completion timeout'):b.run(timeout=.001)
 assert b.failure['snapshot']['raw_registers']['status']==C.BUSY
 assert b.failure['abort']=={'drained':True} and f.regs[C.R_STATUS]==C.DONE|C.ABORTED
 assert b.failure['snapshot']['csr_reads']<=100 and b.failure['profile'] is False

def test_snapshot_and_abort_errors_preserve_original(monkeypatch):
 f=fake_module.Fake();b=B.DiagnosticBenchmark(f,expected_build_id=1234);b.preload(C.make_fixture(1,1,1));f.hang=True
 write=f.reg_write
 def broken(reg,value):
  if reg==C.R_CTRL and value==C.ABORT:raise OSError('abort error')
  return write(reg,value)
 f.reg_write=broken
 monkeypatch.setattr(B,'failure_snapshot',lambda t:(_ for _ in ()).throw(OSError('snapshot error')))
 with pytest.raises(TimeoutError,match='completion timeout'):b.run(timeout=.001)
 assert 'snapshot error' in b.failure['snapshot_error'] and 'abort error' in b.failure['abort']['error']

def test_wrap_and_read_errors_are_explicit():
 f=fake_module.Fake();f.regs[C.R_CYCLES_HI]=2;read=f.reg_read
 def bad(reg):
  if reg==C.R_ERROR:raise OSError('read failed')
  return read(reg)
 f.reg_read=bad;r=B.failure_snapshot(f)
 assert r['counters_may_have_wrapped'] and r['cycles']==2<<32 and r['raw_registers']['error'] is None and 'error' in r['read_errors']

def test_child_requires_inherited_lock_before_transport(tmp_path,monkeypatch):
 monkeypatch.delenv('OTPU_LOCK_HELD',raising=False)
 monkeypatch.setattr(B,'C1F2Transport',lambda *a,**kw:pytest.fail('transport constructed'))
 with pytest.raises(RuntimeError,match='Inherited'):B.checked_child(S.INITIAL_BDF,1234,C.make_fixture(1,1),tmp_path/'child',1)


def test_child_rejects_transport_lock_fallback_before_preload(tmp_path,monkeypatch):
 from types import SimpleNamespace
 monkeypatch.setenv('OTPU_LOCK_HELD','f2-'+S.INITIAL_BDF);monkeypatch.setenv('OTPU_LOCK_PID','123')
 monkeypatch.setenv('OTPU_RUN_DIR','/tmp/otpu')
 class Fallback:
  _lock=SimpleNamespace(holder=None,lost=lambda:False)
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def reg_write(self,*args):pytest.fail('device write after lock fallback')
 monkeypatch.setattr(B,'C1F2Transport',lambda *a,**kw:Fallback())
 with pytest.raises(RuntimeError,match='not inherited'):B.checked_child(S.INITIAL_BDF,1234,C.make_fixture(1,1),tmp_path/'child',1)
 assert json.loads((tmp_path/'child/failure.json').read_text())['profile'] is False
