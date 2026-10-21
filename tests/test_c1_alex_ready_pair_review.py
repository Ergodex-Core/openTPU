import copy
import pytest
from boards.c1_f2_alex import ready_review as R, ready_pair_review as P


def test_boolean_truth_is_exhaustive_and_pin_order_matters():
    b=P.BDD();a=b.var('a');c=b.var('c')
    assert b.lut([a,c],0b1000)==b.lut([c,a],0b1000)
    assert b.lut([a,c],0b0010)==b.ite(a,b.invert(c),0)
    assert b.lut([a,c],0b0010)!=b.lut([c,a],0b0010)
    assert b.invert(b.invert(a))==a
    assert b.lut([a,b.invert(a)],0b1000)==0


def test_transition_equivalence_handles_different_enable_encodings():
    b=P.BDD();q=b.var('q');d=b.var('d');r=b.var('r');ce=b.var('ce')
    original=b.ite(r,0,b.ite(ce,d,q))
    folded=b.ite(r,0,b.ite(1,b.ite(ce,d,q),q))
    assert original==folded
    assert original!=b.ite(r,0,b.ite(b.invert(ce),d,q))


@pytest.mark.parametrize('value,width',[
    ("2'b10",1),("1'h2",1),("1'bx",1),("1'b2",1),("64'h0",4),
])
def test_unknown_or_wrong_width_init_rejected(value,width):
    with pytest.raises(ValueError):P.literal(value,width)


def test_boolean_resource_limit_is_an_error_not_equivalence():
    b=P.BDD(max_nodes=3);b.var('x')
    with pytest.raises(ValueError,match='node bound'):b.var('y')


def test_duplicate_checkpoint_cannot_be_presented_as_paired_proof():
    result=P.review_pair(linked={'checkpoint_sha256':'a'*64},routed={'checkpoint_sha256':'a'*64})
    assert not result['observed_ready_equivalent']
    assert 'distinct' in result['issues'][0]['reason']
    assert result['route_qualified'] is False


def test_missing_raw_file_cannot_be_qualified(tmp_path):
    binding=dict(topology_path=tmp_path/'missing',topology_sha256='a'*64,
                 checkpoint_sha256='b'*64,helper_sha256='c'*64,native_helper_sha256='d'*64)
    result=P.review_pair(linked=binding,routed={**binding,'checkpoint_sha256':'e'*64})
    assert result['issues'] and not result['observed_ready_equivalent']


def simple():
    # Actual raw observation schema, independent of any C1/C2 artifact counts.
    e=R.Evidence();g='cl/g';out=g+'/G';sink='cl/l/I0'
    e.objects['cell',g]={'REF_NAME':'GND'}
    e.objects['pin',out]={'REF_PIN_NAME':'G','IS_INVERTED':'0','DIRECTION':'OUT'}
    e.objects['cell','cl/l']={'REF_NAME':'LUT1'}
    e.objects['pin',sink]={'REF_PIN_NAME':'I0','IS_INVERTED':'0','DIRECTION':'IN'}
    e.queries['pin_owner',sink]=['cl/l']
    net={'nets':['n'],'drivers':[out],'leaves':[out,sink],'ports':[]}
    f={'roots':{'root':net},'graphs':{'root':{'nodes':{
        out:{'output':out,'cell':g,'properties':e.objects['cell',g]}}}},
       'registers':[],'candidate_discovery':{}}
    return e,f


def test_same_observed_graph_has_no_issues_but_not_route_acceptance():
    e,f=simple();result=P.compare(e,f,copy.deepcopy(e),copy.deepcopy(f))
    assert result['issues']==[]
    assert 'Equal values' in result['boundary_assumption']


@pytest.mark.parametrize('mutation',[
    lambda e,f:f['roots']['root']['leaves'].remove('cl/l/I0'),
    lambda e,f:e.objects['pin','cl/l/I0'].__setitem__('IS_INVERTED','1'),
    lambda e,f:e.objects['cell','cl/l'].__setitem__('REF_NAME','LUT2'),
    lambda e,f:f['roots']['root']['ports'].append('unseen_port'),
    lambda e,f:f['candidate_discovery'].__setitem__('new',{'all':['replica']}),
])
def test_changed_consumers_or_replica_discovery_remain_unresolved(mutation):
    e,f=simple();other_e,other_f=copy.deepcopy(e),copy.deepcopy(f)
    mutation(other_e,other_f)
    assert P.compare(e,f,other_e,other_f)['issues']


def test_no_optional_polarity_default():
    e,f=simple();del e.objects['pin','cl/g/G']['IS_INVERTED']
    assert P.compare(e,f,e,f)['issues']


def test_omitted_downstream_type_is_reported_not_invented():
    e,f=simple();del e.objects['cell','cl/l']['REF_NAME']
    result=P.compare(e,f,e,f)
    assert result['issues']==[]
    assert result['omitted_sink_references']['linked']['count']==1
    assert 'downstream sink-cell functions are outside' in result['scope_limit']


def intermediate():
    e,f=simple();pin='cl/l/I0';driver='cl/g/G';out='cl/l/O';sink='cl/ext/D'
    e.objects['cell','cl/l']['INIT']="2'h2"
    e.objects['pin',pin]['NAME']=pin
    e.objects['pin',driver]['NAME']=driver
    e.objects['pin',out]={'NAME':out,'REF_PIN_NAME':'O','DIRECTION':'OUT','IS_INVERTED':'0'}
    e.objects['cell','cl/ext']={'REF_NAME':'FDRE'}
    e.objects['pin',sink]={'NAME':sink,'REF_PIN_NAME':'D','DIRECTION':'IN','IS_INVERTED':'0'}
    e.queries['pin_owner',sink]=['cl/ext']
    e.queries['pin_owner',driver]=['cl/g']
    e.queries['select','pin',pin]=[pin]
    e.queries['select','pin',driver]=[driver]
    e.queries['input_nets',pin]=['n']
    e.queries['input_drivers',pin,'n']=[driver]
    e.queries['input_ports',pin,'n']=[]
    edge={'scope':'selected_input_and_drivers','input':pin,'properties':{},
          'nets':['n'],'drivers':[driver],'ports':[]}
    net={'nets':['mid'],'drivers':[out],'leaves':[out,sink],'ports':[]}
    f['graphs']['root']['nodes'][out]={'output':out,'cell':'cl/l',
        'properties':e.objects['cell','cl/l'],'inputs':{pin:edge},'output_network':net}
    return e,f


@pytest.mark.parametrize('change',['drop','new','polarity','function','port'])
def test_intermediate_consumers_and_functions_cannot_escape_root_checks(change):
    e,f=intermediate();other_e,other_f=copy.deepcopy(e),copy.deepcopy(f)
    net=other_f['graphs']['root']['nodes']['cl/l/O']['output_network']
    if change=='drop':net['leaves'].remove('cl/ext/D')
    elif change=='new':
        other_e.objects['pin','cl/ext/CE']={'NAME':'cl/ext/CE','REF_PIN_NAME':'CE',
                                          'DIRECTION':'IN','IS_INVERTED':'0'}
        other_e.queries['pin_owner','cl/ext/CE']=['cl/ext']
        net['leaves'].append('cl/ext/CE')
    elif change=='polarity':other_e.objects['pin','cl/ext/D']['IS_INVERTED']='1'
    elif change=='function':other_e.objects['cell','cl/l']['INIT']="2'h1"
    elif change=='port':net['ports'].append('top_output')
    result=P.compare(e,f,other_e,other_f)
    assert any(x['scope']=='all_captured_boundary_deliveries' for x in result['issues'])
    assert 'root_function:root' in result['checked']
    assert 'root_sinks:root' in result['checked']


def test_known_cone_inputs_internal_but_external_intermediate_consumers_retained():
    e,f=intermediate();c=P.Circuit(e,f,P.BDD())
    delivery=c.boundary_deliveries()
    assert set(delivery)=={'pin:cl/ext/D'}
    assert delivery['pin:cl/ext/D']['function']==0
    assert P.compare(e,f,e,f)['issues']==[]


def test_intermediate_driver_names_not_part_of_boundary_identity():
    e,f=intermediate();c=P.Circuit(e,f,P.BDD())
    expected=c.boundary_deliveries()
    node=c.nodes.pop('cl/l/O')
    node['output_network']['drivers']=['cl/new_name/O']
    node['output_network']['leaves'][0]='cl/new_name/O'
    c.nodes['cl/new_name/O']=node
    e.objects['pin','cl/new_name/O']=e.objects['pin','cl/l/O']
    assert c.boundary_deliveries()==expected


def test_successful_observed_pair_preserves_independent_scope_and_final_review(monkeypatch):
    e,f=simple()
    monkeypatch.setattr(P,'load',lambda binding:(copy.deepcopy(e),copy.deepcopy(f),{'raw_sha256':binding['topology_sha256']}))
    result=P.review_pair(linked={'checkpoint_sha256':'a'*64,'topology_sha256':'c'*64},
                         routed={'checkpoint_sha256':'b'*64,'topology_sha256':'d'*64})
    assert result['observed_ready_equivalent'] is True
    assert result['issues']==[]
    assert result['route_qualified'] is False and result['hardware_measured'] is False
    assert result['pending_reviews']==['renamed_replica_scope_closure','C1_final_artifact_qualification']
    assert result['captures']['linked']['raw_sha256']!=result['captures']['routed']['raw_sha256']


def test_failed_observed_pair_retains_transform_and_scope_obligations(monkeypatch):
    e,f=simple()
    def load(binding):
        observed=copy.deepcopy(f)
        if binding['checkpoint_sha256']=='b'*64:
            observed['roots']['root']['ports'].append('unproven_consumer')
        return copy.deepcopy(e),observed,{}
    monkeypatch.setattr(P,'load',load)
    result=P.review_pair(linked={'checkpoint_sha256':'a'*64},routed={'checkpoint_sha256':'b'*64})
    assert result['observed_ready_equivalent'] is False and result['issues']
    assert result['route_qualified'] is False
    assert result['pending_reviews']==['actual_ready_transformation_evidence',
                                      'renamed_replica_scope_closure','C1_final_artifact_qualification']
