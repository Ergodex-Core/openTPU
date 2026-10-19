import copy
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ready_luts',ROOT/'boards/c2_f2_alex/native_ready_review.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def f(init=8,inputs=None,sinks=None):
 return {'init':init,'inputs':inputs or [('rst',0),('valid',0)],'sinks':sinks or ['sink/D'],'output_inverted':0}

def test_inverter_absorption():
 # ready & valid before = (!reset) & valid after
 before=f(8,[('rst',1),('valid',0)])
 after=f(4)
 assert m.compare_cell(before,after)['assignments']==4

def test_permuted_inputs():
 assert m.compare_cell(f(4),f(2,[('valid',0),('rst',0)]))

def test_truth_mismatch_fails():
 with pytest.raises(ValueError,match='Truth table'):m.compare_cell(f(4),f(8))

def test_unknown_driver_fails():
 with pytest.raises(ValueError,match='driver identity'):m.compare_cell(f(),f(inputs=[('unknown',0),('valid',0)]))

def test_output_identity_fails():
 with pytest.raises(ValueError,match='sink identity'):m.compare_cell(f(),f(sinks=['other/D']))

def test_variable_bound():
 x=f(inputs=[(str(i),0) for i in range(7)])
 with pytest.raises(ValueError,match='bound'):m.compare_cell(x,x)

def test_output_inversion_matters():
 a=f(7);b=f(8);a['output_inverted']=1
 assert m.compare_cell(a,b)

def test_init_index_lsb_is_i0():
 assert m.truth(f(2),{'rst':1,'valid':0})==1
 assert m.truth(f(2),{'rst':0,'valid':1})==0

def test_complete_file_binding(tmp_path):
 p=tmp_path/'capture';p.write_text('schema 1 kind linked_native_ready checkpoint_sha256 abc route_qualified 0\ncomplete cells 0 route_qualified 0\n')
 digest=m.file_record(p)['sha256']
 assert m.read(p,'abc','linked_native_ready',digest)['input']['sha256']==digest
 with pytest.raises(ValueError,match='SHA'):m.read(p,'abc','linked_native_ready','0'*64)
 with pytest.raises(ValueError,match='header'):m.read(p,'other','linked_native_ready',digest)

def test_incomplete_marker_fails(tmp_path):
 p=tmp_path/'capture';p.write_text('schema 1 kind linked_native_ready checkpoint_sha256 abc route_qualified 0\nincomplete reason cap\ncomplete cells 0 route_qualified 0\n')
 with pytest.raises(ValueError,match='incomplete'):m.read(p,'abc','linked_native_ready',m.file_record(p)['sha256'])

def test_count_not_success(tmp_path):
 p=tmp_path/'capture';p.write_text('schema 1 kind linked_native_ready checkpoint_sha256 abc route_qualified 0\ncomplete cells 80 route_qualified 0\n')
 with pytest.raises(ValueError,match='Completion'):m.read(p,'abc','linked_native_ready',m.file_record(p)['sha256'])

def test_truncated_inventory():
 with pytest.raises(ValueError):m.unique(['a','...'],'truncated')
 with pytest.raises(ValueError):m.unique(['a','a'],'duplicate')

def test_actual_refined_capture_parses():
 path=ROOT/'build/c2-alex-remote/refined-ready-topology.tcldata'
 if not path.exists():pytest.skip('local actual capture not present')
 data=m.read(path,'f30f57b94c4566fc70eb02678afbc47884fd2e088a72c4f9ee648005ac18bad2','narrow_routed_structure','9d96e070a78143d61f07c17ccba53a5676916183c15ac5034f41812234ad6ef1')
 selected=[name for name in data['cells'] if m.SCOPE.fullmatch(name)]
 assert len(selected)==80
 for name in selected:
  value=m.lut(data,name)
  assert any(driver==m.RESET for driver,polarity in value['inputs'])

def test_reset_anchor_actual_and_mutation():
 path=ROOT/'build/c2-alex-remote/refined-ready-topology.tcldata'
 if not path.exists():pytest.skip('local actual capture not present')
 data=m.read(path,'f30f57b94c4566fc70eb02678afbc47884fd2e088a72c4f9ee648005ac18bad2','narrow_routed_structure','9d96e070a78143d61f07c17ccba53a5676916183c15ac5034f41812234ad6ef1')
 assert m.reset_anchor(data,data)['REF_NAME']=='FDRE'
 changed=copy.deepcopy(data)
 changed['pins']['WRAPPER/CL/core_rst_reg/D']['drivers']='different/Q'
 with pytest.raises(ValueError,match='configuration changed'):m.reset_anchor(data,changed)

@pytest.mark.parametrize('extra_before,extra_after,passes',[(False,False,True),(True,False,False),(False,True,False)])
def test_scope_closure_rejects_extra_consumers(monkeypatch,extra_before,extra_after,passes):
 name='WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/s01_entry_pipeline/s01_mmu/inst/b_sreg/example'
 inv='reset_inverter';sink=name+'/I0'
 defs={name:f(inputs=[(m.RESET,0)]),inv:f(inputs=[(m.RESET,0)],sinks=[sink]+(['unreviewed/I0'] if extra_before else []))}
 monkeypatch.setattr(m,'lut',lambda data,cell:defs[cell])
 before={'pins':{sink:{'drivers':inv+'/O'}}}
 leaves=[m.RESET,sink]+([name+'_unreviewed/I0'] if extra_after else [])
 after={'pins':{sink:{'drivers':m.RESET}},'fanout':{m.RESET:{'names':' '.join(leaves),'directions':'OUT '+' '.join('IN' for _ in leaves[1:])}}}
 if passes:assert m.scope_closure(before,after,[name],{inv})['routed_bank_reset_sinks']==[sink]
 else:
  with pytest.raises(ValueError,match='Unreviewed'):m.scope_closure(before,after,[name],{inv})


def test_public_api_missing_evidence_fails_closed():
 result=m.review_artifact(reports={},auxiliary={},checkpoint_sha256='a'*64,linked_checkpoint_sha256='b'*64)
 assert result['issues'] and result['route_qualified'] is False


def test_public_api_keeps_actual_removed_sinks_unresolved():
 linked=ROOT/'build/c2-alex-remote/linked-ready-topology.tcldata'
 refined=ROOT/'build/c2-alex-remote/refined-ready-topology.tcldata'
 if not linked.exists() or not refined.exists():pytest.skip('local actual captures not present')
 result=m.review_artifact(reports={},auxiliary={'native_ready_linked':linked,'native_ready_refined':refined},checkpoint_sha256='f30f57b94c4566fc70eb02678afbc47884fd2e088a72c4f9ee648005ac18bad2',linked_checkpoint_sha256='b8f10ea1a56aa145b9cbfa8f42246e858282dc69a307a32e0b8ed1582095922c')
 assert len(result['issues'])==16
 assert {item['reason'] for item in result['issues']}=={'Output sink identity changed'}
 assert len(result['canonical_facts']['luts'])==64
 assert result['route_qualified'] is False
