from pathlib import Path
import hashlib,json
import pytest
from boards.c1_f2_alex import structural_review as S

CHECKPOINT='1'*64;PARENT='2'*64

def record(data):return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

@pytest.fixture
def sample(tmp_path):
 def make(role='accounting',stage='link'):
  kind,raw_name,helper_names,raw_kind,source_tag=S.SPECS[role]
  helpers={n:S.file_record(S.HERE/n)['sha256'] for n in helper_names}
  header=f'header schema 1 kind {raw_kind} checkpoint_sha256 {CHECKPOINT} checkpoint /immutable/design.dcp route_qualified 0 hardware_measured 0'
  if source_tag:raw=(header+'\n'+source_tag+' helper_sha256 '+helpers[helper_names[0]]+' native_helper_sha256 '+helpers['capture_native.tcl']+'\nsummary data_only\n').encode()
  else:raw=(header+' helpers {'+' '.join(k+' '+v for k,v in helpers.items())+'}\nsummary data_only\n').encode()
  raw_path=tmp_path/'raw.tcldata';raw_path.write_bytes(raw)
  files={n:S.file_record(S.HERE/n) for n in helper_names};files[raw_name]=record(raw)
  files.update({n:record(b'bounded retained capture input') for n in ('run.tcl','driver.py','runner.sh','vivado.log')})
  executor=S.ANCESTRY_EXECUTORS[stage]
  receipt={'schema':1,'kind':kind,'stage':stage,'collector_revision':'3'*40,'hardware_revision':S.HARDWARE_REVISION,'ancestry_executor_revision':executor['revision'],'ancestry_executor_hashes':executor['hashes'],'checkpoint_sha256':CHECKPOINT,'parent_receipt_sha256':PARENT,'started_at':'2026-10-21T10:00:00+00:00','completed_at':'2026-10-21T10:05:00+00:00','returncode':0,'inputs_unchanged':True,'postcheck_error':None,'helpers':helpers,'files':files,'summary':['summary data_only'],'route_qualified':False,'hardware_measured':False,'hardware_actions':False}
  receipt_path=tmp_path/'receipt.json';receipt_path.write_text(json.dumps(receipt))
  return receipt,dict(role=role,raw_path=raw_path,receipt_path=receipt_path,checkpoint_sha256=CHECKPOINT,parent_receipt_sha256=PARENT,stage=stage)
 return make

@pytest.mark.parametrize('role,stage',[('accounting','link'),('accounting','refine'),('accounting_boundary','link'),('accounting_boundary','refine'),('structural','refine')])
def test_raw_receipt_binding_does_not_claim_semantics(sample,role,stage):
 receipt,kwargs=sample(role,stage);result=S.authenticate_capture(**kwargs)
 assert result['route_qualified'] is False and result['raw']==receipt['files'][S.SPECS[role][1]]
 assert 'structural_proven' not in result

@pytest.mark.parametrize('field,value,reason',[
 ('returncode',True,'collector failed'),('returncode',1,'collector failed'),('inputs_unchanged',False,'collector failed'),
 ('postcheck_error','changed','collector failed'),('route_qualified',True,'claimed qualification'),('hardware_actions',True,'claimed qualification'),
 ('stage','refine','role differs'),('checkpoint_sha256','4'*64,'ancestry differs'),('parent_receipt_sha256','4'*64,'ancestry differs'),
 ('hardware_revision','4'*40,'ancestry differs'),('ancestry_executor_revision','4'*40,'ancestry executor'),
 ('completed_at','2026-10-21T09:00:00+00:00','time interval'),('schema',True,'role differs'),
])
def test_receipt_mutations_reject(sample,field,value,reason):
 receipt,kwargs=sample();receipt[field]=value;kwargs['receipt_path'].write_text(json.dumps(receipt))
 with pytest.raises(ValueError,match=reason):S.authenticate_capture(**kwargs)

@pytest.mark.parametrize('attack',['raw_hash','raw_header','raw_helper','helper','helper_file','executor','missing_file','extra_file','duplicate_json'])
def test_nested_binding_mutations_reject(sample,attack):
 receipt,kwargs=sample();raw_name=S.SPECS['accounting'][1]
 if attack=='raw_hash':kwargs['raw_path'].write_bytes(kwargs['raw_path'].read_bytes()+b'changed')
 elif attack in ('raw_header','raw_helper'):
  raw=kwargs['raw_path'].read_text().replace(CHECKPOINT,'4'*64) if attack=='raw_header' else kwargs['raw_path'].read_text().replace(receipt['helpers']['capture_accounting.tcl'],'4'*64)
  kwargs['raw_path'].write_text(raw);receipt['files'][raw_name]=record(raw.encode())
 elif attack=='helper':receipt['helpers']['capture_accounting.tcl']='4'*64
 elif attack=='helper_file':receipt['files']['capture_accounting.tcl']['bytes']+=1
 elif attack=='executor':receipt['ancestry_executor_hashes']={**receipt['ancestry_executor_hashes'],'implement.py':'4'*64}
 elif attack=='missing_file':del receipt['files']['driver.py']
 elif attack=='extra_file':receipt['files']['unbound.tcl']=record(b'x')
 if attack=='duplicate_json':kwargs['receipt_path'].write_text(json.dumps(receipt)[:-1]+',"schema":1}')
 else:kwargs['receipt_path'].write_text(json.dumps(receipt))
 with pytest.raises(ValueError):S.authenticate_capture(**kwargs)

def test_structural_wrapper_never_accepts_authenticated_but_unreviewed_facts(sample,monkeypatch):
 monkeypatch.setattr(S,'authenticate_capture',lambda **kw:{'role':kw['role'],'route_qualified':False})
 auxiliary={name:Path('/unused') for name in ('accounting','accounting_receipt','accounting_linked','accounting_linked_receipt','accounting_boundary','accounting_boundary_receipt','accounting_boundary_linked','accounting_boundary_linked_receipt','structural','structural_receipt')}
 result=S.review_artifact(reports={},auxiliary=auxiliary,checkpoint_sha256=CHECKPOINT,linked_checkpoint_sha256='4'*64,prerequisites={'receipt_sha256':{'link':PARENT,'refine':'5'*64}})
 assert result['issues'][0]['reason']=='structural_semantics_pending'
 assert result['resolved_pending_reviews']==[] and result['route_qualified'] is False
