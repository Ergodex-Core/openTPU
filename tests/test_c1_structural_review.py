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


@pytest.mark.parametrize('role,stage',[('accounting','link'),('accounting','refine'),('accounting_boundary','link'),('accounting_boundary','refine'),('structural','refine')])
def test_exact_m8a_executor_authenticates_only_current_checkpoint_capture(sample,role,stage):
 receipt,kwargs=sample(role,stage)
 receipt['ancestry_executor_revision']=S.M8A_ANCESTRY_EXECUTOR['revision']
 receipt['ancestry_executor_hashes']=dict(S.M8A_ANCESTRY_EXECUTOR['hashes'])
 kwargs['receipt_path'].write_text(json.dumps(receipt));result=S.authenticate_capture(**kwargs)
 assert result['ancestry_executor_revision']==S.M8A_ANCESTRY_EXECUTOR['revision']
 assert result['checkpoint_sha256']==CHECKPOINT and result['parent_receipt_sha256']==PARENT
 assert result['route_qualified'] is False and 'structural_proven' not in result


@pytest.mark.parametrize('stage',('link','refine'))
@pytest.mark.parametrize('attack',('old_revision_new_hashes','new_revision_old_hashes','other_stage','changed_build','checkpoint','parent'))
def test_reviewed_executor_tuples_cannot_be_crossed_or_replace_ancestry(sample,stage,attack):
 receipt,kwargs=sample('accounting',stage)
 receipt['ancestry_executor_revision']=S.M8A_ANCESTRY_EXECUTOR['revision']
 receipt['ancestry_executor_hashes']=dict(S.M8A_ANCESTRY_EXECUTOR['hashes'])
 if attack=='old_revision_new_hashes':receipt['ancestry_executor_revision']=S.ANCESTRY_EXECUTORS[stage]['revision']
 elif attack=='new_revision_old_hashes':receipt['ancestry_executor_hashes']=dict(S.ANCESTRY_EXECUTORS[stage]['hashes'])
 elif attack=='other_stage':
  other=S.ANCESTRY_EXECUTORS['refine' if stage=='link' else 'link']
  receipt['ancestry_executor_revision']=other['revision'];receipt['ancestry_executor_hashes']=dict(other['hashes'])
 elif attack=='changed_build':receipt['ancestry_executor_hashes']['build.py']='4'*64
 elif attack=='checkpoint':receipt['checkpoint_sha256']='4'*64
 elif attack=='parent':receipt['parent_receipt_sha256']='4'*64
 kwargs['receipt_path'].write_text(json.dumps(receipt))
 with pytest.raises(ValueError):S.authenticate_capture(**kwargs)


@pytest.mark.parametrize('role', ('accounting','accounting_boundary','structural'))
def test_exact_reimplementation_refine_executor_preserves_pending_scope(sample,role):
 receipt,kwargs=sample(role,'refine');executor=S.M8A_REIMPLEMENTATION_ANCESTRY_EXECUTOR
 receipt['collector_revision']=S.M8A_ANCESTRY_EXECUTOR['revision']
 receipt['ancestry_executor_revision']=executor['revision']
 receipt['ancestry_executor_hashes']=dict(executor['hashes'])
 kwargs['receipt_path'].write_text(json.dumps(receipt));result=S.authenticate_capture(**kwargs)
 assert result['collector_revision']==S.M8A_ANCESTRY_EXECUTOR['revision']
 assert result['ancestry_executor_revision']==executor['revision']
 assert result['checkpoint_sha256']==CHECKPOINT and result['parent_receipt_sha256']==PARENT
 assert result['route_qualified'] is False and 'structural_proven' not in result


@pytest.mark.parametrize('attack', ('new_revision_old_hashes','old_revision_new_hashes',
    'wrong_revision','changed_build','wrong_stage','checkpoint','parent','changed_helper'))
def test_reimplementation_executor_never_relabels_link_or_loosens_capture_binding(sample,attack):
 receipt,kwargs=sample('accounting','refine');executor=S.M8A_REIMPLEMENTATION_ANCESTRY_EXECUTOR
 receipt['ancestry_executor_revision']=executor['revision']
 receipt['ancestry_executor_hashes']=dict(executor['hashes'])
 if attack=='new_revision_old_hashes':receipt['ancestry_executor_hashes']=dict(S.M8A_ANCESTRY_EXECUTOR['hashes'])
 elif attack=='old_revision_new_hashes':receipt['ancestry_executor_revision']=S.M8A_ANCESTRY_EXECUTOR['revision']
 elif attack=='wrong_revision':receipt['ancestry_executor_revision']='4'*40
 elif attack=='changed_build':receipt['ancestry_executor_hashes']['build.py']='4'*64
 elif attack=='wrong_stage':receipt['stage']=kwargs['stage']='link'
 elif attack=='checkpoint':receipt['checkpoint_sha256']='4'*64
 elif attack=='parent':receipt['parent_receipt_sha256']='4'*64
 elif attack=='changed_helper':receipt['helpers']['capture_accounting.tcl']='4'*64
 kwargs['receipt_path'].write_text(json.dumps(receipt))
 with pytest.raises(ValueError):S.authenticate_capture(**kwargs)

def test_structural_wrapper_never_accepts_authenticated_but_unreviewed_facts(sample,monkeypatch):
 monkeypatch.setattr(S,'authenticate_capture',lambda **kw:{'role':kw['role'],'route_qualified':False})
 auxiliary={name:Path('/unused') for name in ('accounting','accounting_receipt','accounting_linked','accounting_linked_receipt','accounting_boundary','accounting_boundary_receipt','accounting_boundary_linked','accounting_boundary_linked_receipt','structural','structural_receipt')}
 result=S.review_artifact(reports={},auxiliary=auxiliary,checkpoint_sha256=CHECKPOINT,linked_checkpoint_sha256='4'*64,prerequisites={'receipt_sha256':{'link':PARENT,'refine':'5'*64}})
 assert result['issues'][0]['reason']=='structural_semantics_pending'
 assert result['resolved_pending_reviews']==[] and result['route_qualified'] is False


def tcl(values):
 return ' '.join('{'+str(x)+'}' for x in values)


def original(tmp_path,state='unresolved'):
 observation={'state':state,'route_qualified':'0'}
 if state=='unresolved':observation.update(error='exact observed failure',error_info='exact observed failure\n    while executing\noriginal collector')
 else:observation['topology']='retained original topology'
 raw=tcl(sum(([k,v] for k,v in observation.items()),[]))
 summary=tmp_path/'summary.tcldata';objects=tmp_path/'objects.tcldata'
 summary.write_text(tcl(['kind','c1_final_raw_capture','checkpoint_sha256',CHECKPOINT,'route_qualified','0','structural_capture',raw])+'\n')
 objects.write_text(tcl(['checkpoint','/immutable/design.dcp','sha256',CHECKPOINT])+'\n'+tcl(['structural_capture',raw])+'\n')
 return {'summary.tcldata':summary,'objects.tcldata':objects}


@pytest.mark.parametrize('state',('observed','unresolved'))
def test_original_structural_state_is_preserved_not_waived(tmp_path,state):
 reports=original(tmp_path,state);result=S.original_structural_observation(reports,CHECKPOINT)
 assert result['observation']['state']==state and result['failed_scope_resolved'] is False
 assert result['route_qualified'] is False
 if state=='unresolved':assert 'while executing' in result['observation']['error_info']


@pytest.mark.parametrize('attack',('summary_dcp','object_dcp','object_error','object_missing','object_duplicate','state','stack','missing_stack'))
def test_original_structural_binding_mutations_reject(tmp_path,attack):
 reports=original(tmp_path);summary=reports['summary.tcldata'];objects=reports['objects.tcldata']
 if attack=='summary_dcp':summary.write_text(summary.read_text().replace(CHECKPOINT,'9'*64))
 elif attack=='object_dcp':objects.write_text(objects.read_text().replace(CHECKPOINT,'9'*64))
 elif attack=='object_error':objects.write_text(objects.read_text().replace('exact observed failure','different failure'))
 elif attack=='object_missing':objects.write_text(tcl(['checkpoint','/immutable/design.dcp','sha256',CHECKPOINT])+'\n')
 elif attack=='object_duplicate':objects.write_text(objects.read_text()+objects.read_text())
 else:
  for path in (summary,objects):
   raw=path.read_text()
   if attack=='state':raw=raw.replace('unresolved','accepted')
   elif attack=='stack':raw=raw.replace('exact observed failure\n','unrelated stack\n')
   elif attack=='missing_stack':raw=raw.replace('{error_info}','{unrecognized}')
   path.write_text(raw)
 with pytest.raises(ValueError):S.original_structural_observation(reports,CHECKPOINT)


def test_partial_replay_does_not_drop_failed_capture_or_final_obligations(monkeypatch):
 from boards.c1_f2_alex import structural_capture_review as C, structural_static_review as T, structural_flow_review as F, accounting_review as A
 def auth(**kw):
  if kw['role']=='accounting_boundary':raise ValueError('actual admission incomplete')
  return {'raw':{'sha256':'1'*64},'checkpoint_sha256':CHECKPOINT,'helpers':{'capture_accounting.tcl':'2'*64,'capture_native.tcl':'3'*64}}
 monkeypatch.setattr(S,'authenticate_capture',auth)
 monkeypatch.setattr(S,'original_structural_observation',lambda *args:{'failed_scope_resolved':False})
 monkeypatch.setattr(S,'replay_current_native',lambda *args:(None,[]))
 monkeypatch.setattr(C,'review_artifact',lambda **kw:{'issues':[],'capture_complete':True,'canonical_facts':{'steps':[]}})
 monkeypatch.setattr(T,'review',lambda *args:{'issues':[],'proven_static_steps':['general','clocks','unused_clocks']})
 monkeypatch.setattr(F,'review',lambda *args,**kw:{'issues':[],'proven_steps':[]})
 monkeypatch.setattr(A,'review_artifact',lambda **kw:{'issues':[],'arithmetic_proven':True,'pending_reviews':['admission pending']})
 auxiliary={key:'/unused' for role in S.SPECS for key in (role,role+'_receipt',role+'_linked',role+'_linked_receipt')};auxiliary['native']='/unused'
 result=S.review_artifact(reports={},auxiliary=auxiliary,checkpoint_sha256=CHECKPOINT,linked_checkpoint_sha256='9'*64,prerequisites={'receipt_sha256':{'link':PARENT,'refine':PARENT},'native':{}})
 assert len([issue for issue in result['issues'] if issue['reason'].endswith('_authentication')])==2
 assert result['canonical_facts']['accounting_arithmetic']['arithmetic_proven'] is True
 assert result['canonical_facts']['static']['issues']==[]
 assert result['resolved_pending_reviews']==[] and result['route_qualified'] is False
 assert result['issues'][0]['reason']=='structural_semantics_pending'


def semantic_values():
 from boards.c1_f2_alex import admission_transition_review as T, admission_equation_review as E
 source={'source_elided_bits_proven':True,'issues':[],'facts_sha256':'a'*64,
         'canonical_facts':{'configuration':dict(T.CONFIGURATION)}}
 transition={'one_step_equivalent_under_source_specialization':True,'issues':[],'facts_sha256':'b'*64,
             'canonical_facts':{'channels':{str(pc):{'configuration_required':dict(T.CONFIGURATION),
                'rtl_sha256':E.RTL_SHA256,'one_step_equivalent_under_source_specialization':True,
                'root':f'actual/pc{pc}/root/O'} for pc in range(8)}}}
 return source,transition


def test_source_prerequisite_is_reread_not_trusted(monkeypatch):
 from boards.c1_f2_alex import admission_specialization_review as A
 source,_=semantic_values();seen=[];binding={'source_paths':{'frozen':'/actual'},'synth_inputs_path':'/input',
                                          'synth_inputs_sha256':'c'*64,'generated_top_path':'/top'}
 def replay(**kw):seen.append(kw);return source
 monkeypatch.setattr(A,'review_sources',replay)
 assert S.replay_source_specialization({'source_binding':binding,'source_specialization':source})==source
 assert seen==[binding]
 with pytest.raises(ValueError,match='differs from prerequisite'):
  S.replay_source_specialization({'source_binding':binding,'source_specialization':{**source,'facts_sha256':'d'*64}})
 source['source_elided_bits_proven']=False
 with pytest.raises(ValueError,match='replay incomplete'):
  S.replay_source_specialization({'source_binding':binding,'source_specialization':source})


@pytest.mark.parametrize('attack',('source_config','transition_config','source_flag','transition_flag',
                                  'transition_issue','channel_missing','channel_flag','rtl','stage_missing'))
def test_source_transition_composition_rejects_unbound_obligations(attack):
 import copy
 source,transition=semantic_values();transitions={'link':transition,'refine':copy.deepcopy(transition)}
 channel=transitions['refine']['canonical_facts']['channels']['7']
 if attack=='source_config':source['canonical_facts']['configuration']['QD']=32
 elif attack=='transition_config':channel['configuration_required']['QD']=32
 elif attack=='source_flag':source['source_elided_bits_proven']=False
 elif attack=='transition_flag':transitions['refine']['one_step_equivalent_under_source_specialization']=False
 elif attack=='transition_issue':transitions['refine']['issues']=[{'reason':'unproved reset'}]
 elif attack=='channel_missing':del transitions['refine']['canonical_facts']['channels']['7']
 elif attack=='channel_flag':channel['one_step_equivalent_under_source_specialization']=False
 elif attack=='rtl':channel['rtl_sha256']='f'*64
 elif attack=='stage_missing':del transitions['refine']
 with pytest.raises(ValueError):S.admission_source_composition(source,transitions)


@pytest.mark.parametrize('failed_stage',(None,'link','refine'))
def test_authenticated_admission_public_replays_preserve_scope(monkeypatch,failed_stage):
 from boards.c1_f2_alex import admission_specialization_review as Z, admission_transition_review as T, admission_pair_review as P
 source,transition=semantic_values();seen=[];pairs=[]
 def auth(**kw):
  if kw['role']!='accounting_boundary':raise ValueError('unavailable unrelated capture')
  if kw['stage']==failed_stage:raise ValueError('actual raw receipt mismatch')
  return {'raw':{'sha256':('a' if kw['stage']=='link' else 'b')*64},
          'checkpoint_sha256':kw['checkpoint_sha256'],
          'helpers':{'capture_accounting_boundary.tcl':'c'*64,'capture_native.tcl':'d'*64}}
 def replay(**binding):seen.append(binding);return transition
 def pair(**binding):pairs.append(binding);return {'issues':[],'paired_preservation_proven':True,
       'interface_assumption':'external input producer equivalence remains unproved'}
 monkeypatch.setattr(S,'authenticate_capture',auth)
 monkeypatch.setattr(S,'original_structural_observation',lambda *args:{'failed_scope_resolved':False})
 monkeypatch.setattr(Z,'review_sources',lambda **kw:source)
 monkeypatch.setattr(T,'review_artifact',replay);monkeypatch.setattr(P,'review_pair',pair)
 auxiliary={key:'/raw/'+key for role in S.SPECS for key in (role,role+'_receipt',role+'_linked',role+'_linked_receipt')}
 result=S.review_artifact(reports={},auxiliary=auxiliary,checkpoint_sha256=CHECKPOINT,linked_checkpoint_sha256='9'*64,
    prerequisites={'receipt_sha256':{'link':PARENT,'refine':PARENT},'source_binding':{},'source_specialization':source})
 expected=[{'topology_path':'/raw/accounting_boundary'+suffix,'topology_sha256':sha*64,
            'checkpoint_sha256':checkpoint,'helper_sha256':'c'*64,'native_helper_sha256':'d'*64}
           for stage,suffix,sha,checkpoint in (('link','_linked','a','9'*64),('refine','','b',CHECKPOINT)) if stage!=failed_stage]
 assert seen==expected
 if failed_stage is None:
  assert pairs==[{'linked':expected[0],'routed':expected[1]}]
  composed=result['canonical_facts']['admission_source_composition']
  assert composed['configuration_bound'] is True and composed['whole_native_equivalence_proven'] is False
  assert composed['paired_preservation_proven'] is False and composed['route_qualified'] is False
  assert 'external_admission_input_producer_equivalence' in composed['pending_reviews']
 else:
  assert not pairs and 'admission_source_composition' not in result['canonical_facts']
 assert result['issues'][0]['obligations']==['original_structural_failure_scope',
     'paired_accounting_and_admission_preservation','ready_replica_scope_closure']
 assert result['resolved_pending_reviews']==[] and result['route_qualified'] is result['hardware_measured'] is False


def test_failed_source_replay_is_retained_without_composition(monkeypatch):
 from boards.c1_f2_alex import admission_specialization_review as A
 source,_=semantic_values()
 monkeypatch.setattr(A,'review_sources',lambda **kw:{**source,'issues':[{'reason':'changed frozen source'}]})
 with pytest.raises(ValueError,match='differs from prerequisite'):
  S.replay_source_specialization({'source_binding':{},'source_specialization':source})
