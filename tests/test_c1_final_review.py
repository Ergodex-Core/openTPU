"""Synthetic C1 acceptance-boundary tests; no design is qualified here."""
import copy
import json
import pytest
from types import SimpleNamespace
from boards.c1_f2_alex import final_review as r
g=r.module('gray_review')

def td(d):return ' '.join('{'+str(x)+'}' for kv in d.items() for x in kv)

def summary():
 totals=dict(wns='0.028',whs='0.010',wpws='0.000',tns='0.000',ths='0.000',tpws='0.000',setup_failing='0',hold_failing='0',pulse_failing='0',setup_total='10',hold_total='10',pulse_total='10')
 checks={k:'0' for k in 'no_clock constant_clock pulse_width_clock unconstrained_internal_endpoints no_input_delay no_output_delay multiple_clock generated_clocks loops partial_input_delay partial_output_delay latch_loops'.split()}
 checks.update(no_clock='35',pulse_width_clock='1')
 return dict(schema='1',kind='c1_final_raw_capture',stage='final_raw',hardware_measured='0',route_qualified='0',blockers='',timing=td(dict(totals=td(totals),checks=td(checks),blockers='')),routing=td(dict(pass_='unused',**{'pass':'1','routable':'10','routed':'10','errors':'0'})),supplemental_evidence=td(dict(incomplete_or_error='0')))

def test_positive_physical_summary_is_only_one_predicate():
 assert r.physical(g,summary())['timing']['wns']=='0.028'

@pytest.mark.parametrize('field,value',[('wns','-0.001'),('whs','-0.001'),('pulse_failing','1'),('setup_total','0')])
def test_true_physical_failures_cannot_be_disposed(field,value):
 s=summary();timing=g.td(s['timing']);totals=g.td(timing['totals']);totals[field]=value;timing['totals']=td(totals);s['timing']=td(timing)
 with pytest.raises(ValueError):r.physical(g,s)

def disposition_fixture():
 records=[{'family':'methodology','record':{'CHECK':'SYNTH-6','NAME':'SYNTH-6#1'}}]
 ids=[r.canonical(x) for x in records];aux={'source':'a'*64}
 d={'schema':1,'kind':'c1-alex-artifact-dispositions','checkpoint_sha256':'b'*64,'collection_receipt_sha256':'c'*64,'record_inventory_sha256':r.canonical(sorted(ids)),'auxiliary_sha256':aux,'dispositions':[{'classification':'current-findings','family':'methodology','record_sha256':ids,'evidence':['source'],'rationale':'Synthetic test only; exact RAM optimization record.'}]}
 return d,records,aux

def test_exact_artifact_disposition_inventory():
 d,records,aux=disposition_fixture();out=r.review_dispositions(d,checkpoint='b'*64,capture_receipt='c'*64,records=records,auxiliary=aux)
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
 with pytest.raises(ValueError):r.review_dispositions(d,checkpoint='b'*64,capture_receipt='c'*64,records=records,auxiliary=aux)

def test_unapproved_artifact_never_reaches_predicates(tmp_path,monkeypatch):
 policy=tmp_path/'policy';policy.write_text('{"schema":1,"artifacts":{}}');monkeypatch.setattr(r,'POLICY',policy)
 with pytest.raises(ValueError,match='not been source-reviewed'):
  r.replay(reports={},auxiliary={},disposition_path=tmp_path/'absent',checkpoint='a'*64,linked_checkpoint='c'*64,capture_receipt_sha256='b'*64,capture_receipts={})



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



def protected_cdc_fixture(tmp_path,monkeypatch):
 text='CDC-1 Critical 5 Synthetic\nSource Clock: shell_src\nDestination Clock: shell_dst\nCDC Type: User Ignored\n'
 text+=''.join(f' {i} CDC-1 Critical Synthetic protected finding {i} False Path <hidden>/<hidden> <hidden>/<hidden> N\n' for i in range(5))
 report=tmp_path/'shell_cdc';report.write_text(text);complete=tmp_path/'complete';complete.write_text(td({'kind':'original_shell_reference_captured','checkpoint_sha256':r.module('build').SHELL_SHA256}))
 monkeypatch.setattr(r,'SHELL_CDC_RECORD',r.record(report))
 rows=r.module('exception_review').parse_cdc_report(text)['rows']
 return [dict(family='cdc',record=row) for row in rows],dict(shell_cdc=report,shell_complete=complete)


def test_protected_cdc_requires_full_signature_multiplicity(tmp_path,monkeypatch):
 records,aux=protected_cdc_fixture(tmp_path,monkeypatch);out=r.protected_cdc_context(g,records,aux)
 assert out['count']==5 and out['hidden_endpoint_identity_asserted'] is False


@pytest.mark.parametrize('kind',['depth','clock','waived','missing','duplicate','cl_endpoint','original_bytes','checkpoint'])
def test_protected_cdc_mutations_rejected(tmp_path,monkeypatch,kind):
 records,aux=protected_cdc_fixture(tmp_path,monkeypatch)
 if kind=='depth':records[0]['record']['depth']='99'
 elif kind=='clock':records[0]['record']['source_clock']='new'
 elif kind=='waived':records[0]['record']['waived']='Y'
 elif kind=='missing':records.pop()
 elif kind=='duplicate':records.append(copy.deepcopy(records[0]))
 elif kind=='cl_endpoint':records[0]['record']['to']='WRAPPER/CL/new/D'
 elif kind=='original_bytes':aux['shell_cdc'].write_text(aux['shell_cdc'].read_text()+'\n')
 else:aux['shell_complete'].write_text(td({'kind':'original_shell_reference_captured','checkpoint_sha256':'0'*64}))
 with pytest.raises(ValueError):r.protected_cdc_context(g,records,aux)


def isolated_replay(tmp_path, monkeypatch):
 """Synthetic control-flow fixture; raw predicates explicitly replaced here."""
 reports = {}; auxiliary = {}; values = {}
 for name in r.E.FINAL_REPORTS:
  path=tmp_path/('report-'+name);path.write_text('SYNTHETIC raw control-flow input');reports[name]=path
 for name in set(r.R.RAW_AUXILIARY)|r.R.REFERENCE_AUXILIARY:
  path=tmp_path/('aux-'+name);path.write_text('SYNTHETIC raw control-flow input');auxiliary[name]=path
 s=summary();s.update(checkpoint_sha256='b'*64,pending_reviews=' '.join(sorted(r.R.REQUIRED_CATEGORIES)))
 reports['summary.tcldata'].write_text(td(s))
 d,records,_=disposition_fixture();d['auxiliary_sha256']={k:r.record(p)['sha256'] for k,p in auxiliary.items()}
 d['dispositions'][0]['evidence']=['shell_objects']
 disposition=tmp_path/'disposition.json';disposition.write_bytes(r.encode(d))
 policy=tmp_path/'policy.json';policy.write_bytes(r.encode({'schema':1,'artifacts':{'b'*64:r.record(disposition)['sha256']}}))
 monkeypatch.setattr(r,'POLICY',policy);monkeypatch.setattr(r,'source_inventory',lambda:{'synthetic':'fixed'})
 def good(**kw): return {'issues':[],'canonical_facts':{'synthetic':True}}
 native_pending=['current_routed_native_ready_equivalence','current_host_wready_equivalence','current_reset_status','physical_timing_and_exceptions']
 values['native']={**good(),'pending_reviews':native_pending}
 values['ready_pair']={'issues':[],'observed_ready_equivalent':True,'pending_reviews':['renamed_replica_scope_closure','C1_final_artifact_qualification']}
 values['structural']={**good(),'resolved_pending_reviews':['current_routed_native_ready_equivalence','current_host_wready_equivalence','renamed_replica_scope_closure']}
 actual_gray=r.module('gray_review')
 monkeypatch.setattr(actual_gray,'review_final',lambda *args:{'gray_evidence_consistent':True})
 modules={'gray_review':actual_gray,
  'no_clock_review':SimpleNamespace(review_raw=lambda *args:{'predicate_satisfied':True,'pins':35}),
  'reset_status_review':SimpleNamespace(review_artifact=good),
  'exception_review':SimpleNamespace(review_artifact=good),
  'native_review':SimpleNamespace(review_artifact=lambda **kw:copy.deepcopy(values['native'])),
  'ready_pair_review':SimpleNamespace(review_pair=lambda **kw:copy.deepcopy(values['ready_pair'])),
  'structural_review':SimpleNamespace(review_artifact=lambda **kw:copy.deepcopy(values['structural']))}
 monkeypatch.setattr(r,'module',lambda name:modules[name])
 monkeypatch.setattr(r,'protected_pulse_context',lambda **kw:{'synthetic':'pulse'})
 monkeypatch.setattr(r,'protected_cdc_context',lambda *args:{'synthetic':'cdc'})
 monkeypatch.setattr(r,'artifact_records',lambda *args:records)
 receipts={role:{'files':{name:{'sha256':'1'*64} for name in ('capture_ready.tcl','capture_native.tcl')}} for role in r.E.RAW_ROLES}
 receipts['linked_native']['linked_receipt_sha256']='2'*64
 receipts['capture']['refine_receipt_sha256']='3'*64
 arguments=dict(reports=reports,auxiliary=auxiliary,disposition_path=disposition,checkpoint='b'*64,
  linked_checkpoint='a'*64,capture_receipt_sha256='c'*64,capture_receipts=receipts)
 return arguments,values


def test_isolated_orchestration_closes_exact_categories(tmp_path,monkeypatch):
 args,_=isolated_replay(tmp_path,monkeypatch)
 result=r.replay(**args)
 assert result['route_qualified'] is True and result['hardware_measured'] is False
 assert result['resolved_pending_reviews']==sorted(r.R.REQUIRED_CATEGORIES)


@pytest.mark.parametrize('mode',['native_issue','ready_issue','structural_issue','missing_closure','unknown_nested','extra_closure','duplicate_closure'])
def test_any_unresolved_raw_obligation_prevents_acceptance(tmp_path,monkeypatch,mode):
 args,values=isolated_replay(tmp_path,monkeypatch)
 if mode=='native_issue':values['native']['issues']=[{'reason':'actual raw mismatch'}]
 elif mode=='ready_issue':values['ready_pair']['observed_ready_equivalent']=False
 elif mode=='structural_issue':values['structural']['issues']=[{'reason':'replica scope incomplete'}]
 elif mode=='missing_closure':values['structural']['resolved_pending_reviews'].pop()
 elif mode=='unknown_nested':values['native']['pending_reviews'].append('new_unreviewed_obligation')
 elif mode=='extra_closure':values['structural']['resolved_pending_reviews'].append('inapplicable')
 else:values['structural']['resolved_pending_reviews']*=2
 with pytest.raises(ValueError):r.replay(**args)
