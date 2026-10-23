import copy
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('no_clock',Path(__file__).resolve().parents[1] / 'boards/c1_f2_alex/no_clock_review.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def example():
    pin='SL/icap/reg[0]/C';owner=pin[:-2];root='SL/pcie/bufg/O';root_owner=root[:-2]
    return {pin:{'finding':{'CHECK':'TIMING-17','severity':'CRITICAL WARNING','waived':False},'pin_selection':[pin],'owners':[owner],'clocks':[],
                 'roots':[root],'root_selection':[root],'root_owners':[root_owner],'root_clocks':[],
                 'pin_properties':{'NAME':pin,'REF_PIN_NAME':'C','DIRECTION':'IN','IS_CLOCK':'1'},
                 'owner_properties':{'NAME':owner,'REF_NAME':'FDRE','IS_SEQUENTIAL':'1','LOC':'SLICE_X1Y2','BEL':'AFF'},
                 'root_properties':{'NAME':root,'REF_PIN_NAME':'O','DIRECTION':'OUT','IS_CLOCK':'0'},
                 'root_owner_properties':{'NAME':root_owner,'REF_NAME':'BUFG_GT','IS_SEQUENTIAL':'0','LOC':'BUFG_GT_X1Y2','BEL':'BUF'}}}

def test_explicit_typed_properties_need_no_redundant_class_literal():
    a=example();b=copy.deepcopy(a);next(iter(a.values()))['pin_properties'].update(CLASS='pin',REF_NAME='FDRE')
    r=m.check_no_clock(a,b);assert r['predicate_satisfied'] and not r['route_qualified']

@pytest.mark.parametrize('mutation,reason',[
    (lambda r:r.pop('root_selection'),'missing_or_ambiguous_root_selection'),
    (lambda r:r.update(root_owners=['a','b']),'root_owner_identity'),
    (lambda r:r.update(root_clocks=['different']),'different_relationship_or_property'),
    (lambda r:r['root_owner_properties'].pop('REF_NAME'),'missing_root_owner_properties'),
    (lambda r:r.update(clocks=['clock']),'pin_clock_not_empty'),
    (lambda r:r.update(incomplete_relations=['cap']),'incomplete_capture'),
    (lambda r:r['finding'].update(waived=None),'waiver_state_unproved'),
    (lambda r:r['pin_properties'].update(IS_CLOCK='0'),'not_clock_input'),
    (lambda r:r['owner_properties'].update(LOC='OTHER'),'different_relationship_or_property'),
])
def test_partial_or_changed_witness_fails(mutation,reason):
    a=example();b=copy.deepcopy(a);mutation(next(iter(b.values())));r=m.check_no_clock(a,b)
    assert not r['predicate_satisfied'] and any(reason in x for x in r['unresolved_reasons'])

def test_same_count_different_pin_fails():
    a=example();b=copy.deepcopy(a);b['other']=b.pop(next(iter(b)));r=m.check_no_clock(a,b)
    assert 'different_pin_sets' in r['unresolved_reasons']


spec2=importlib.util.spec_from_file_location('gray_no_clock_adapter',Path(__file__).resolve().parents[1] / 'boards/c1_f2_alex/gray_review.py');reader=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(reader)

def tl(words):return ' '.join('{'+str(x)+'}' for x in words)
def td(d):return tl(x for kv in d.items() for x in kv)
def serialized(tmp_path):
 data=example();pin,row=next(iter(data.items()));finding='TIMING-17#1';base=('methodology','finding',finding)
 rows=[tl(['finding','methodology','properties',td({'NAME':finding,'CHECK':'TIMING-17'}),'rule_properties',td({'SEVERITY':'Critical Warning'})])]
 tails={'pin_selection':('pin_selection',pin),'owners':('owning_cells',),'clocks':('associated_clocks',),'roots':('fanin_roots',),'root_selection':('root','pin_selection',row['roots'][0]),'root_owners':('root','owning_cells'),'root_clocks':('root','associated_clocks')}
 rows.append(tl(['query',tl(base+('related_pins',)),'status','observed','value',tl([pin])]))
 for key,tail in tails.items():rows.append(tl(['query',tl(base+tail),'status','observed','value',tl(row[key])]))
 for group,kind in [('pin','pin'),('owner','cell'),('root','pin'),('root_owner','cell')]:
  prop=row[group+'_properties'];rows.append(tl(['object_properties',tl(base+(kind,prop['NAME'])),td({k:td({'ok':'1','value':v}) for k,v in prop.items()})]))
 # A net may share its name with a pin but is not the same typed object.
 rows.append(tl(['object_properties',tl(base+('nets','0')),td({'NAME':td({'ok':'1','value':pin}),'IS_CLOCK':td({'ok':'1','value':'0'}),'CLASS':td({'ok':'1','value':'net'})})]))
 path=tmp_path/'objects';path.write_text('\n'.join(rows)+'\n');return path

def test_actual_shape_adapter_and_separate_pin_net_identity(tmp_path):
 p=serialized(tmp_path);e=m.parse_evidence(reader,[p]);normalized=m.normalize(reader,e,True)
 r=m.check_no_clock(normalized,normalized);assert r['predicate_satisfied'],r

def test_missing_root_selection_remains_unresolved(tmp_path):
 p=serialized(tmp_path);e=m.parse_evidence(reader,[p]);key=next(k for k in e['queries'] if 'root' in k and 'pin_selection' in k);e['queries'].pop(key)
 normalized=m.normalize(reader,e,True);r=m.check_no_clock(normalized,normalized);assert not r['predicate_satisfied']

def test_global_waiver_evidence_required(tmp_path):
 p=serialized(tmp_path);e=m.parse_evidence(reader,[p]);normalized=m.normalize(reader,e,False)
 assert not m.check_no_clock(normalized,normalized)['predicate_satisfied']
