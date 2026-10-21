"""Independent state-alias/API adversaries; internal controls never qualify DCPs."""
from copy import deepcopy
import hashlib
import pytest
from boards.c1_f2_alex import admission_pair_review as P
from test_c1_admission_pair_review import fixture, raw_fixture, serialize


def clone_state(ch, original, replacement):
    """A coherent distinct state instance, not a malformed output-only rename."""
    old=ch['nodes'][original];new=deepcopy(old);oldcell=old['cell'];newcell=replacement.rsplit('/',1)[0]
    new['cell']=newcell;new['properties']['NAME']=newcell;new['output_properties']['NAME']=replacement
    new['inputs']={pin.replace(oldcell+'/',newcell+'/'):edge for pin,edge in new['inputs'].items()}
    ch['nodes'][replacement]=new
    ch['memories' if new['reference']=='RAMD32' else 'registers'].append(newcell)
    return new


@pytest.mark.parametrize('reference,source,sink_role', [('RAMD32','memory/O','I0'),('FDRE','state/Q','I1')])
def test_distinct_state_instances_do_not_alias_by_equal_configuration(reference,source,sink_role):
    f=fixture();ch=f['channels']['0'];prefix='WRAPPER/CL/u_native0/';original=prefix+source;replacement=prefix+'second_'+source
    clone_state(ch,original,replacement);g=deepcopy(f)
    g['channels']['0']['nodes'][prefix+'logic/O']['inputs'][prefix+'logic/'+sink_role]['driver']=replacement
    with pytest.raises(ValueError,match='root expression'):P.compare(f,g)


def test_coherently_renamed_memory_requires_explicit_state_relation():
    f=fixture();g=deepcopy(f);ch=g['channels']['0'];old='WRAPPER/CL/u_native0/memory/O';new='WRAPPER/CL/u_native0/renamed_memory/O'
    clone_state(ch,old,new);oldcell=ch['nodes'].pop(old)['cell'];ch['memories'].remove(oldcell)
    for node in ch['nodes'].values():
        for edge in node['inputs'].values():
            if edge['driver']==old:edge['driver']=new
    with pytest.raises(ValueError,match='renamed, removed or added'):P.compare(f,g)


def test_two_memory_asynchronous_address_cycle_rejected_even_if_identical():
    f=fixture();ch=f['channels']['0'];root='WRAPPER/CL/u_native0/';first=root+'memory/O';second=root+'second_memory/O'
    node=clone_state(ch,first,second)
    ch['nodes'][first]['inputs'][root+'memory/RADR0']['driver']=second
    node['inputs'][root+'second_memory/RADR0']['driver']=first
    with pytest.raises(ValueError,match='asynchronous'):P.compare(f,deepcopy(f))


def test_synchronous_memory_data_feedback_is_not_misclassified_as_async_cycle():
    f=fixture();root='WRAPPER/CL/u_native0/';memory=f['channels']['0']['nodes'][root+'memory/O']
    memory['inputs'][root+'memory/I']['driver']=root+'memory/O'
    assert P.compare(f,deepcopy(f))


@pytest.mark.parametrize('mutation',['ram_missing_init','ram_missing_clock_config','ram_missing_address','ram_bad_output','ram_clock_marked_data','fdre_missing_clock_marker','constant_polarity','lut_missing_init','duplicate_state_name'])
def test_identical_invalid_configuration_on_both_sides_is_not_proof(mutation):
    f=fixture();ch=f['channels']['0'];root='WRAPPER/CL/u_native0/';q=ch['nodes'][root+'state/Q'];ram=ch['nodes'][root+'memory/O']
    if mutation=='ram_missing_init':del ram['properties']['INIT']
    elif mutation=='ram_missing_clock_config':del ram['properties']['IS_CLK_INVERTED']
    elif mutation=='ram_missing_address':del ram['inputs'][root+'memory/RADR4']
    elif mutation=='ram_bad_output':ram['output_properties']['IS_INVERTED']='1'
    elif mutation=='ram_clock_marked_data':ram['inputs'][root+'memory/I']['properties']['IS_CLOCK']='1'
    elif mutation=='fdre_missing_clock_marker':q['inputs'][root+'state/C']['properties']['IS_CLOCK']='0'
    elif mutation=='constant_polarity':ch['nodes'][root+'zero/G']['output_properties']['IS_INVERTED']='1'
    elif mutation=='lut_missing_init':del ch['nodes'][root+'logic/O']['properties']['INIT']
    elif mutation=='duplicate_state_name':ch['nodes'][root+'alias/Q']=deepcopy(q)
    with pytest.raises((ValueError,KeyError,TypeError)):P.compare(f,deepcopy(f))


def test_external_producer_renaming_is_conditional_on_equal_logical_inputs():
    f=fixture();g=deepcopy(f);ch=g['channels']['0'];old='WRAPPER/CL/data0/Q';new='WRAPPER/CL/new_external_data/Q'
    node=ch['nodes'].pop(old);node['cell']=new.rsplit('/',1)[0];node['properties']['NAME']=node['cell'];node['output_properties']['NAME']=new;ch['nodes'][new]=node
    ch['interfaces']['data']['driver']=new
    for item in ch['nodes'].values():
        for edge in item['inputs'].values():
            if edge['driver']==old:edge['driver']=new
    assert P.compare(f,g)  # Producer identity is expressly outside the theorem.


def raw_bindings(tmp_path):
    e=raw_fixture()
    for pc in range(8):e.objects['pin',f'WRAPPER/CL/u_native{pc}/state/C']['IS_CLOCK']='1'
    paths=[tmp_path/'linked',tmp_path/'routed'];texts=[serialize(e),serialize(e).replace('a'*64,'d'*64)]
    result=[]
    for path,text,checkpoint in zip(paths,texts,('a'*64,'d'*64)):
        path.write_text(text);result.append(dict(topology_path=path,topology_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint_sha256=checkpoint,helper_sha256='b'*64,native_helper_sha256='c'*64))
    return result


@pytest.mark.parametrize('mutation',['wrong_raw_hash','wrong_helper','wrong_native_helper','wrong_checkpoint','truncated_envelope','clock_marker'])
def test_public_api_cannot_bypass_raw_and_state_checks(tmp_path,mutation):
    linked,routed=raw_bindings(tmp_path)
    if mutation=='wrong_raw_hash':routed['topology_sha256']='e'*64
    elif mutation=='wrong_helper':routed['helper_sha256']='e'*64
    elif mutation=='wrong_native_helper':routed['native_helper_sha256']='e'*64
    elif mutation=='wrong_checkpoint':routed['checkpoint_sha256']='e'*64
    else:
        path=routed['topology_path'];text=path.read_text()
        if mutation=='truncated_envelope':text='\n'.join(text.splitlines()[:-1])+'\n'
        else:
            target='{property} {pin} {WRAPPER/CL/u_native0/state/C} {IS_CLOCK} {status} {observed} {value} {1}'
            assert target in text;text=text.replace(target,target[:-2]+'0}')
        path.write_text(text);routed['topology_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    result=P.review_pair(linked=linked,routed=routed)
    assert result['issues'] and result['paired_preservation_proven'] is False
    assert result['route_qualified'] is result['source_equation_proven'] is result['hardware_measured'] is False
    assert len(result['pending_reviews'])==3


def test_same_checkpoint_public_gate_precedes_any_graph_replay(monkeypatch):
    def forbidden(**_):raise AssertionError('same-checkpoint pair reached graph replay')
    monkeypatch.setattr(P.R,'review_artifact',forbidden)
    result=P.review_pair(linked={'checkpoint_sha256':'a'*64},routed={'checkpoint_sha256':'a'*64})
    assert result['paired_preservation_proven'] is False and 'distinct' in result['issues'][0]['reason']
