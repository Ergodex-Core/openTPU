"""Synthetic immutable-review predicates. These are not hardware evidence."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

r=load('final_review_test',ROOT/'boards/c2_f2_alex/final_review.py');g=r.module('gray_review')
f=load('schema2_fixture',ROOT/'tests/test_c2_alex_evidence.py')

def td(d):return ' '.join('{'+str(x)+'}' for kv in d.items() for x in kv)

def summary():
 totals=dict(wns='0.028',whs='0.010',wpws='0.000',tns='0.000',ths='0.000',tpws='0.000',setup_failing='0',hold_failing='0',pulse_failing='0',setup_total='10',hold_total='10',pulse_total='10')
 checks={k:'0' for k in 'no_clock constant_clock pulse_width_clock unconstrained_internal_endpoints no_input_delay no_output_delay multiple_clock generated_clocks loops partial_input_delay partial_output_delay latch_loops'.split()}
 checks.update(no_clock='35',pulse_width_clock='1')
 return dict(schema='1',stage='final_route',hardware_measured='0',route_qualified='0',blockers='',timing=td(dict(totals=td(totals),checks=td(checks),blockers='')),routing=td(dict(pass_='unused',**{'pass':'1','routable':'10','routed':'10','errors':'0'})),supplemental_evidence=td(dict(incomplete_or_error='0')))

def test_positive_physical_summary_is_only_one_predicate():
 assert r.physical(g,summary())['timing']['wns']=='0.028'

@pytest.mark.parametrize('field,value',[('wns','-0.001'),('whs','-0.001'),('pulse_failing','1'),('setup_total','0')])
def test_true_physical_failures_cannot_be_disposed(field,value):
 s=summary();timing=g.td(s['timing']);totals=g.td(timing['totals']);totals[field]=value;timing['totals']=td(totals);s['timing']=td(timing)
 with pytest.raises(ValueError):r.physical(g,s)

def disposition_fixture():
 records=[{'family':'methodology','record':{'CHECK':'SYNTH-6','NAME':'SYNTH-6#1'}}]
 ids=[r.canonical(x) for x in records];aux={'source':'a'*64}
 d={'schema':1,'kind':'c2-alex-artifact-dispositions','checkpoint_sha256':'b'*64,'collection_receipt_sha256':'c'*64,'record_inventory_sha256':r.canonical(sorted(ids)),'auxiliary_sha256':aux,'dispositions':[{'classification':'reviewed-memory-output-register-suggestion','family':'methodology','record_sha256':ids,'evidence':['source'],'rationale':'Synthetic test only; exact RAM optimization record.'}]}
 return d,records,aux

def test_exact_artifact_disposition_inventory():
 d,records,aux=disposition_fixture();out=r.review_dispositions(d,checkpoint='b'*64,receipt_sha='c'*64,records=records,auxiliary=aux)
 assert out['records']==1

@pytest.mark.parametrize('change',['checkpoint','receipt','record','auxiliary','missing','duplicate','unknown_class','blanket_flag'])
def test_artifact_disposition_tampering_rejects(change):
 d,records,aux=disposition_fixture()
 if change=='checkpoint':d['checkpoint_sha256']='d'*64
 elif change=='receipt':d['collection_receipt_sha256']='d'*64
 elif change=='record':records[0]['record']['NAME']='NEW'
 elif change=='auxiliary':aux={'source':'d'*64}
 elif change=='missing':d['dispositions']=[]
 elif change=='duplicate':d['dispositions']*=2
 elif change=='unknown_class':d['dispositions'][0]['classification']='accept-all'
 else:d['dispositions'][0]['accepted']=True
 with pytest.raises(ValueError):r.review_dispositions(d,checkpoint='b'*64,receipt_sha='c'*64,records=records,auxiliary=aux)

def test_unapproved_artifact_never_reaches_predicates(tmp_path,monkeypatch):
 policy=tmp_path/'policy';policy.write_text('{"schema":1,"artifacts":{}}');monkeypatch.setattr(r,'POLICY',policy)
 with pytest.raises(ValueError,match='not source-reviewed'):
  r.replay(reports={},auxiliary={},disposition_path=tmp_path/'absent',checkpoint='a'*64,collection_receipt_sha256='b'*64)


def make_schema2(tmp_path,monkeypatch):
 root,m,_=f.fixture_bundle(tmp_path,monkeypatch)
 m['schema']=2;m['roots']['review']={'original_absolute_path':'/unavailable/original/review','bundle_relative_path':'runs/review'}
 m['source_snapshots']['review']={'revision':'d'*40,'bundle_relative_path':'sources/review'}
 f.put(root/'sources/review/placeholder','synthetic review source')
 f.put(root/'runs/review/review-result.json',{'schema':1,'kind':'SYNTHETIC_review_for_control_flow_test'})
 q=root/'runs/qualify';receipt=f.read(q/'stage-result.json');receipt.update(status='pending-review',route_qualified=False);f.put(q/'stage-result.json',receipt)
 p=q/'audit-result.txt';f.put(p,p.read_text().replace('route_qualified=1','route_qualified=0'))
 p=q/'qualification/summary.tcldata';f.put(p,p.read_text().replace('route_qualified 1','route_qualified 0').replace('pending_reviews {}','pending_reviews {synthetic_pending}'))
 f.put(q/'qualification/NOT_QUALIFIED.txt','original pending marker retained')
 helper='boards/c2_f2_alex/routed_host_ready.tcl'
 source=f.source_copy(root,'qualify',[helper]);f.put(q/'routed_host_ready.tcl',(ROOT/helper).read_bytes())
 si=f.read(q/'stage-inputs.json');si['executor_sources'].update(source)
 si['prepared_files']['routed_host_ready.tcl']=f.rec(q/'routed_host_ready.tcl');f.put(q/'stage-inputs.json',si)
 f.update_last_receipt(root,m)
 m['receipts']['review']={'relative_path':'runs/review/review-result.json','sha256':f.e.sha(root/'runs/review/review-result.json')}
 f.put(root/f.e.MANIFEST,m)
 return root,m,f.e.sha(root/f.e.MANIFEST)


def test_schema2_cannot_qualify_without_replaying_separate_review(tmp_path,monkeypatch):
 root,m,digest=make_schema2(tmp_path,monkeypatch);calls=[];old=f.e.local_module
 def verify(x):
  calls.append(x);assert x.text('qualify','qualification/NOT_QUALIFIED.txt')=='original pending marker retained'
  raise ValueError('Synthetic semantic review unresolved')
 monkeypatch.setattr(f.e,'local_module',lambda name: SimpleNamespace(verify_review=verify) if name=='final_review' else old(name))
 with pytest.raises(ValueError,match='semantic review unresolved'):f.e.verify_bundle(root,digest)
 assert len(calls)==1


def test_schema2_rejects_removing_original_pending_marker(tmp_path,monkeypatch):
 root,m,digest=make_schema2(tmp_path,monkeypatch);(root/'runs/qualify/qualification/NOT_QUALIFIED.txt').unlink();f.update_last_receipt(root,m)
 m['receipts']['review']={'relative_path':'runs/review/review-result.json','sha256':f.e.sha(root/'runs/review/review-result.json')};f.put(root/f.e.MANIFEST,m)
 with pytest.raises(ValueError,match='pending collector evidence'):f.e.verify_bundle(root,f.e.sha(root/f.e.MANIFEST))


def test_failed_collection_cannot_be_finalized(tmp_path):
 roots={k:tmp_path/k for k in ('qualify','refine','simulation')}
 for p in roots.values():p.mkdir()
 receipts={'qualify':{'stage':'qualify','status':'failed','returncode':1,'route_qualified':False,'hardware_actions':False},'refine':{},'simulation':{}}
 hashes={}
 for role,value in receipts.items():
  path=roots[role]/('simulation-complete.json' if role=='simulation' else 'stage-result.json');f.put(path,value);hashes[role]=r.record(path)['sha256']
 request=dict(schema=1,kind='c2-alex-final-review-request',instance_id=f.e.INSTANCE,roots={k:str(v) for k,v in roots.items()},receipt_sha256=hashes,auxiliary={},disposition={},source_revision='d'*40)
 p=tmp_path/'request.json';f.put(p,request)
 with pytest.raises(ValueError,match='Failed/incomplete collector'):r.finalize(p,r.record(p)['sha256'],tmp_path/'review')
 assert not (tmp_path/'review').exists()


def successful_request(tmp_path, monkeypatch):
 """Synthetic only: replace semantic replay, exercise immutable file protocol."""
 roots={role:tmp_path/role for role in ('qualify','refine','simulation')}
 for p in roots.values():p.mkdir()
 f.put(roots['refine']/'post_route.dcp','SYNTHETIC DCP')
 checkpoint=r.record(roots['refine']/'post_route.dcp')['sha256']
 refine={'stage':'refine','status':'complete','returncode':0,'checkpoint_sha256':checkpoint}
 simulation={'state':'passed','hardware_actions':False,'inputs_sha256':'a'*64}
 receipts={'refine':refine,'simulation':simulation};hashes={}
 for role,body in receipts.items():
  name=f.e.RECEIPTS[role];f.put(roots[role]/name,body);hashes[role]=r.record(roots[role]/name)['sha256']
 stage={'previous_receipt_sha256':hashes['refine'],'simulation':{'run':str(roots['simulation']),'receipt_sha256':hashes['simulation'],'inputs_sha256':simulation['inputs_sha256']}}
 f.put(roots['qualify']/'stage-inputs.json',stage)
 for name in f.e.FINAL_REPORTS:f.put(roots['qualify']/'qualification'/name,'SYNTHETIC '+name)
 f.put(roots['qualify']/'qualification/NOT_QUALIFIED.txt','preserve original pending evidence')
 outputs={'qualification/'+p.name:r.record(p) for p in (roots['qualify']/'qualification').iterdir()}
 collection={'stage':'qualify','status':'pending-review','returncode':0,'route_qualified':False,'hardware_actions':False,'checkpoint_sha256':checkpoint,'checkpoint':str(roots['refine']/'post_route.dcp'),'inputs_sha256':r.record(roots['qualify']/'stage-inputs.json')['sha256'],'outputs':outputs}
 f.put(roots['qualify']/'stage-result.json',collection);hashes['qualify']=r.record(roots['qualify']/'stage-result.json')['sha256']
 raw=tmp_path/'raw.tcldata';f.put(raw,'SYNTHETIC raw supplemental')
 f.put(tmp_path/'link/post_link.dcp','SYNTHETIC LINK DCP')
 linked=tmp_path/'native-linked.tcldata';f.put(linked,'schema 1 kind linked_native_ready route_qualified 0 checkpoint_sha256 '+r.record(tmp_path/'link/post_link.dcp')['sha256']+'\n')
 d=tmp_path/'dispositions.json';f.put(d,{'synthetic':True})
 request={'schema':1,'kind':'c2-alex-final-review-request','instance_id':f.e.INSTANCE,'roots':{k:str(v) for k,v in roots.items()},'receipt_sha256':hashes,'auxiliary':{'synthetic_raw':{'path':str(raw),**r.record(raw)},'native_ready_linked':{'path':str(linked),**r.record(linked)}},'disposition':{'path':str(d),**r.record(d)},'source_revision':'d'*40}
 p=tmp_path/'request.json';f.put(p,request)
 monkeypatch.setattr(r,'source_inventory',lambda:{'boards/c2_f2_alex/evidence.py':r.record(ROOT/'boards/c2_f2_alex/evidence.py')})
 monkeypatch.setattr(r,'replay',lambda **kw:{'synthetic_predicate':'passed_for_protocol_test_only'})
 return p,roots


def test_finalize_writes_new_receipt_last_preserving_all_collection_bytes(tmp_path,monkeypatch):
 request,roots=successful_request(tmp_path,monkeypatch)
 before=f.inventory(roots['qualify']);out=tmp_path/'review'
 receipt=r.finalize(request,r.record(request)['sha256'],out)
 assert f.inventory(roots['qualify'])==before
 assert receipt['status']=='complete' and receipt['hardware_actions'] is False
 assert receipt['outputs']=={name:meta for name,meta in f.inventory(out).items() if name!='review-result.json'}
 assert f.read(out/'review-inputs.json')['receipt_sha256']==f.read(request)['receipt_sha256']
 assert (out/'auxiliary/synthetic_raw').read_text()=='SYNTHETIC raw supplemental'
 assert 'review-result.json' not in receipt['outputs']
 with pytest.raises(FileExistsError):r.finalize(request,r.record(request)['sha256'],out)


@pytest.mark.parametrize('change',['report','receipt','auxiliary','extra_output','symlink'])
def test_finalize_rejects_changed_raw_inputs_without_receipt(tmp_path,monkeypatch,change):
 request,roots=successful_request(tmp_path,monkeypatch)
 if change=='report':f.put(roots['qualify']/'qualification/objects.tcldata','modified')
 elif change=='receipt':f.put(roots['qualify']/'stage-result.json',{})
 elif change=='auxiliary':f.put(tmp_path/'raw.tcldata','modified')
 elif change=='extra_output':f.put(roots['qualify']/'unbound','new')
 else:(roots['qualify']/'link').symlink_to(tmp_path/'raw.tcldata')
 out=tmp_path/'review'
 with pytest.raises(ValueError):r.finalize(request,r.record(request)['sha256'],out)
 assert not out.exists()


def test_failed_predicate_never_writes_review_output(tmp_path,monkeypatch):
 request,roots=successful_request(tmp_path,monkeypatch)
 def fail(**kwargs):raise ValueError('Unresolved semantic predicate')
 monkeypatch.setattr(r,'replay',fail);out=tmp_path/'review'
 with pytest.raises(ValueError,match='Unresolved'):r.finalize(request,r.record(request)['sha256'],out)
 assert not out.exists()


def test_source_or_input_mutation_during_review_is_rejected(tmp_path,monkeypatch):
 request,roots=successful_request(tmp_path,monkeypatch)
 def mutate(**kwargs):
  f.put(roots['qualify']/'qualification/objects.tcldata','changed during replay')
  return {'synthetic':'passed'}
 monkeypatch.setattr(r,'replay',mutate);out=tmp_path/'review'
 with pytest.raises(ValueError,match='Input changed'):r.finalize(request,r.record(request)['sha256'],out)
 assert not out.exists()


def test_review_output_cannot_mutate_original_collection_tree(tmp_path,monkeypatch):
 request,roots=successful_request(tmp_path,monkeypatch);before=f.inventory(roots['qualify'])
 with pytest.raises(ValueError,match='outside immutable inputs'):
  r.finalize(request,r.record(request)['sha256'],roots['qualify']/'new-review')
 assert f.inventory(roots['qualify'])==before


def review_bundle(tmp_path,monkeypatch):
 request,roots=successful_request(tmp_path,monkeypatch);out=tmp_path/'review'
 receipt=r.finalize(request,r.record(request)['sha256'],out)
 for name in receipt['source_inputs']:f.put(tmp_path/'sources/review'/name,(ROOT/name).read_bytes())
 e=r.module('evidence');x=e.Bundle.__new__(e.Bundle);x.root=tmp_path
 x.manifest={'schema':2,'roots':{role:{'bundle_relative_path':role} for role in ('link',*roots,'review')},'source_snapshots':{'review':{'revision':'d'*40,'bundle_relative_path':'sources/review'}},'receipts':{}}
 x.files=f.inventory(tmp_path)
 for role in (*roots,'review'):
  name=e.schema_receipts(2)[role];x.manifest['receipts'][role]={'relative_path':role+'/'+name,'sha256':r.record(tmp_path/role/name)['sha256']}
 return x,out


def test_review_receipt_replays_raw_predicate_after_relocation(tmp_path,monkeypatch):
 x,out=review_bundle(tmp_path,monkeypatch)
 assert r.verify_review(x)=={'synthetic_predicate':'passed_for_protocol_test_only'}


def test_changed_replayed_decision_cannot_use_claimed_qualified_receipt(tmp_path,monkeypatch):
 x,out=review_bundle(tmp_path,monkeypatch)
 monkeypatch.setattr(r,'replay',lambda **kw:{'synthetic_predicate':'now_unresolved'})
 with pytest.raises(ValueError,match='differs from raw predicate replay'):r.verify_review(x)


def test_reviewed_source_change_rejects_old_review_receipt(tmp_path,monkeypatch):
 x,out=review_bundle(tmp_path,monkeypatch)
 monkeypatch.setattr(r,'source_inventory',lambda:{'boards/c2_f2_alex/evidence.py':{'sha256':'0'*64,'bytes':123}})
 with pytest.raises(ValueError,match='verifier source differs'):r.verify_review(x)


@pytest.mark.parametrize('classification,family,rule',[
 ('reviewed-current-cdc-summary','methodology','TIMING-9'),
 ('reviewed-unused-net-report','drc','RTSTAT-10'),
])
def test_current_cdc_summary_and_unused_net_dispositions_still_bind_exact_records(classification,family,rule):
 d,records,aux=disposition_fixture();records[0]={'family':family,'record':{'CHECK':rule,'NAME':rule+'#1'}}
 ids=[r.canonical(records[0])];d['record_inventory_sha256']=r.canonical(ids)
 d['dispositions'][0].update(classification=classification,family=family,record_sha256=ids)
 assert r.review_dispositions(d,checkpoint='b'*64,receipt_sha='c'*64,records=records,auxiliary=aux)['records']==1
 records[0]['record']['NAME']=rule+'#2'
 with pytest.raises(ValueError,match='differ from reviewed inventory'):
  r.review_dispositions(d,checkpoint='b'*64,receipt_sha='c'*64,records=records,auxiliary=aux)


@pytest.mark.parametrize('classification,family',[
 ('reviewed-current-cdc-summary','drc'),
 ('reviewed-unused-net-report','methodology'),
])
def test_new_disposition_classes_cannot_cross_finding_families(classification,family):
 d,records,aux=disposition_fixture();records[0]['family']=family
 ids=[r.canonical(records[0])];d['record_inventory_sha256']=r.canonical(ids)
 d['dispositions'][0].update(classification=classification,family=family,record_sha256=ids)
 with pytest.raises(ValueError,match='classification/family'):
  r.review_dispositions(d,checkpoint='b'*64,receipt_sha='c'*64,records=records,auxiliary=aux)


def test_native_link_header_must_match_the_successful_bundled_link_dcp(tmp_path,monkeypatch):
 x,out=review_bundle(tmp_path,monkeypatch)
 x.files['link/post_link.dcp']={'sha256':'f'*64,'bytes':123}
 with pytest.raises(ValueError,match='successful LINK DCP'):r.verify_review(x)


@pytest.mark.parametrize('header',[
 '',
 'schema 1 kind narrow_routed_structure route_qualified 0 checkpoint_sha256 '+'a'*64,
 'schema 1 kind linked_native_ready route_qualified 1 checkpoint_sha256 '+'a'*64,
 'schema 1 kind linked_native_ready route_qualified 0 checkpoint_sha256 incomplete',
])
def test_native_link_identity_cannot_be_guessed_or_qualified(header,tmp_path):
 path=tmp_path/'native';path.write_text(header+'\n')
 with pytest.raises(ValueError):r.linked_checkpoint({'native_ready_linked':path})


def test_historical_executor_is_accepted_only_for_completed_prequalification_roles():
 sha='812b64f8afab2c3c8ff2d5235d6cdcb0eaec38b486a4ff2f786360144e956295'
 e=r.module('evidence')
 for role in ('link','implement','refine'):assert sha in e.REVIEWED_HISTORICAL[(role,'boards/c2_f2_alex/implement.py')]
 assert sha not in e.REVIEWED_HISTORICAL.get(('qualify','boards/c2_f2_alex/implement.py'),set())


def pulse_text():
 return '''3. checking pulse_width_clock (1)
---------------------------------
 There is 1 register/latch pin which need pulse_width check. (LOW)

<hidden>/<hidden>/<hidden>/<hidden>

4. checking unconstrained_internal_endpoints (0)
Pulse Width Checks
--------------------------------------------------------------------------------------
Clock Name:         clk_125
Waveform(ns):       { 0.000 4.000 }
Period(ns):         8.000
Sources:            { WRAPPER/IO_SHIM/PLL/CLKOUT1 }

Check Type        Corner  Lib Pin   Reference Pin  Required(ns)  Actual(ns)  Slack(ns)  Location        Pin
Low Pulse Width   Slow    FDCE/C    n/a            0.275         4.000       3.725      SLICE_X119Y148  <hidden>/<hidden>/<hidden>/<hidden>
High Pulse Width  Slow    FDCE/C    n/a            0.275         4.000       3.725      SLICE_X119Y148  <hidden>/<hidden>/<hidden>/<hidden>

'''


def test_exact_protected_pulse_context_preserves_identity_limit(tmp_path):
 old=tmp_path/'old';new=tmp_path/'new';old.write_text(pulse_text());new.write_text(pulse_text())
 out=r.protected_pulse_context(reports={'timing.rpt':new},auxiliary={'shell_timing':old},physical_result=r.physical(g,summary()))
 assert out['current_rows']==2 and out['hidden_endpoint_identity_asserted'] is False
 assert out['scoped_report_exhaustiveness_asserted'] is False


@pytest.mark.parametrize('before,after',[
 ('There is 1 register','There are 2 registers'),
 ('<hidden>/<hidden>/<hidden>/<hidden>','WRAPPER/CL/unknown/C'),
 ('CLKOUT1','CLKOUT2'),('3.725','-0.001'),('3.725','3.700'),('3.725','NaN'),
 ('High Pulse Width','Unknown Check'),('Clock Name:','Wrong Header:'),
])
def test_changed_protected_pulse_evidence_rejected(tmp_path,before,after):
 old=tmp_path/'old';new=tmp_path/'new';old.write_text(pulse_text());new.write_text(pulse_text().replace(before,after))
 with pytest.raises(ValueError):r.protected_pulse_context(reports={'timing.rpt':new},auxiliary={'shell_timing':old},physical_result=r.physical(g,summary()))


def test_exact_handshake_class_only_applies_to_cdc():
 assert r.CLASSES['reviewed-vendor-handshake-cdc']=={'cdc'}
