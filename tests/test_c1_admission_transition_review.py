"""Integer source oracle plus mutations of the authenticated real LINK slice."""
import copy
import hashlib
import json
from pathlib import Path
import random

import pytest
from boards.c1_f2_alex import admission_transition_review as R, ready_pair_review as P

WIDTHS={'cn':5,'cp':4,'cr':4,'idle_count':3,'next_addr':25,'pending_addr':18,
        'pending_count':5,'pending_valid':1,'reads':8,'writes':8}
INPUTS=('rst','n_cvalid','m_bvalid','m_arready','m_rvalid')


def oracle(state,inputs,stored,take):
    """Direct integer interpretation of the source's ordered assignments."""
    result=dict(state)
    if inputs['rst']:return {key:0 for key in state}
    ar_fire=bool(state['pending_valid'] and inputs['m_arready'])
    if ar_fire:
        result['pending_valid']=0;result['pending_count']=0;result['idle_count']=0
    elif take:
        if state['pending_count']==0:result['pending_addr']=stored>>6
        result['next_addr']=(stored+1)&((1<<25)-1)
        result['pending_count']=(state['pending_count']+1)&31
        result['idle_count']=0
        if state['pending_count']==15 or stored&63==63 or state['reads']==127:result['pending_valid']=1
    elif state['pending_count'] and not state['pending_valid']:
        if state['cn'] or state['idle_count']==3 or state['reads']==128:result['pending_valid']=1
        else:result['idle_count']=(state['idle_count']+1)&7
    push=bool(inputs['n_cvalid'] and state['cn']<16)
    if push:result['cp']=(state['cp']+1)&15
    if take:result['cr']=(state['cr']+1)&15
    result['cn']=(state['cn']+push-take)&31
    result['reads']=(state['reads']+take-inputs['m_rvalid'])&255
    result['writes']=(state['writes']-inputs['m_bvalid'])&255
    return result


def test_symbolic_source_transition_matches_independent_integer_oracle():
    b=P.BDD();states={key:[b.var(f's-{key}-{i:02}') for i in range(width)] for key,width in WIDTHS.items()}
    inputs={key:b.var('i-'+key) for key in INPUTS};stored=[b.var(f'c-{i:02}') for i in range(24)];take=b.var('take')
    expected=R.expected_next(b,states,inputs,stored,take)
    def evaluate(node,assignment):
        while node>1:
            key,low,high=b.nodes[node];node=high if assignment[key] else low
        return node
    rng=random.Random(9417)
    cases=[]
    for _ in range(512):
        state={key:rng.randrange(1<<width) for key,width in WIDTHS.items()}
        ins={key:rng.randrange(2) for key in INPUTS};cases.append((state,ins,rng.randrange(1<<24),rng.randrange(2)))
    for reset in range(2):
        for arready in range(2):
            for taken in range(2):
                for count in (0,1,15,16,31):
                    for reads in (0,127,128,255):
                        state={key:(1<<width)-1 for key,width in WIDTHS.items()}
                        state.update(pending_count=count,reads=reads,idle_count=3,cn=0)
                        ins=dict(rst=reset,n_cvalid=1,m_bvalid=1,m_arready=arready,m_rvalid=1)
                        cases.append((state,ins,63,taken))
    for state,ins,command,taken in cases:
        assignment={f's-{key}-{i:02}':(value>>i)&1 for key,value in state.items() for i in range(WIDTHS[key])}
        assignment.update({'i-'+key:value for key,value in ins.items()})
        assignment.update({f'c-{i:02}':(command>>i)&1 for i in range(24)});assignment['take']=taken
        actual={key:sum(evaluate(node,assignment)<<i for i,node in enumerate(values)) for key,values in expected.items()}
        assert actual==oracle(state,ins,command,taken)


@pytest.fixture(scope='module')
def actual_link():
    base=Path(__file__).resolve().parents[1]/'build/c1-alex-remote'
    raw=base/'admission-link-bbc65a8d-980aaec9-download/files/capture/raw/admission.tcldata'
    if not raw.is_file():pytest.skip('optional retained actual LINK artifact is not installed')
    graph=R.A.review_artifact(topology_path=raw,topology_sha256='a2fc0f982395f1db8d966d669d1ebe72179e11f05098dd226a14b331dd8a5cd0',
        checkpoint_sha256='bbc65a8dbc15473663c15afc1948c197b41a89c21e6a91757c150746d9d3cc3a',
        helper_sha256='378391d2e5019005c98c51548a5a7f9f82d6a269ee50d24bcdc1d07b1a3552ff',
        native_helper_sha256='aa08405a798ac57218b46847b540b83da224e99ea07b3cb2c0a91b16e2c8d45e')
    assert graph['capture_complete'] and not graph['issues']
    return graph['canonical_facts']['channels']


def test_actual_link_all_eight_admission_state_slices(actual_link):
    for pc,channel in actual_link.items():
        result=R.review_channel(pc,channel)
        assert result['one_step_equivalent_under_source_specialization']
        assert len(result['checked_registers'])==81 and len(result['checked_memories'])==24
        assert result['configuration_required']==R.CONFIGURATION
        assert not result['source_specialization_verified']
        assert not result['whole_native_equivalence_proven'] and not result['paired_preservation_proven']


@pytest.mark.parametrize('family',list(WIDTHS))
@pytest.mark.parametrize('role',['R','CE','D'])
def test_actual_link_register_control_mutation_rejects(actual_link,family,role):
    channel=copy.deepcopy(actual_link['0']);reg=next(reg for reg in channel['registers'] if '/'+family+'_reg' in reg)
    node=channel['nodes'][reg+'/Q'];edge=node['inputs'][reg+'/'+role]
    if role=='CE':
        # Forced zero must hold a register whose source equation can advance.
        edge['driver']='WRAPPER/CL/u_native0/GND/G';edge['properties']['IS_INVERTED']='0'
    else:edge['properties']['IS_INVERTED']='1' if edge['properties']['IS_INVERTED']=='0' else '0'
    with pytest.raises(ValueError):R.review_channel('0',channel)


def test_public_transition_entrypoint_rejects_claimed_json_success(tmp_path):
    path=tmp_path/'raw';path.write_text(json.dumps({'capture_complete':True,'issues':[]}))
    result=R.review_artifact(topology_path=path,topology_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             checkpoint_sha256='a'*64,helper_sha256='b'*64,native_helper_sha256='c'*64)
    assert result['issues'] and not result['one_step_equivalent_under_source_specialization']
    assert not result['source_specialization_verified'] and not result['route_qualified']
    assert not result['hardware_measured'] and not result['whole_native_equivalence_proven']


@pytest.mark.parametrize('attack',['cell_clock','cell_reset','cell_data','clock_marker','data_clock_marker'])
def test_actual_effective_pin_properties_cannot_contradict_primitive_configuration(actual_link,attack):
    channel=copy.deepcopy(actual_link['0']);reg='WRAPPER/CL/u_native0/cn_reg[0]';node=channel['nodes'][reg+'/Q']
    if attack.startswith('cell_'):
        prop={'cell_clock':'IS_C_INVERTED','cell_reset':'IS_R_INVERTED','cell_data':'IS_D_INVERTED'}[attack]
        node['properties'][prop]="1'b1"
    else:
        role='C' if attack=='clock_marker' else 'D'
        node['inputs'][reg+'/'+role]['properties']['IS_CLOCK']='0' if role=='C' else '1'
    with pytest.raises(ValueError):R.review_channel('0',channel)
