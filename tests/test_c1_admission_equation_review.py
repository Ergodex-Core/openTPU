"""Independent gate encoding and boundary mutations for the bounded equation."""
import copy
import hashlib
import random

import pytest
from boards.c1_f2_alex import admission_equation_review as R, native_review as N, ready_pair_review as P


def fixture():
    nodes = {}; scope = N.ROOT + 'u_native0/'; ordinal = 0

    def cell(name, ref, inputs=None, output='O', init=None, kind='logic', **props):
        properties = dict(NAME=name, REF_NAME=ref, **props)
        if init is not None: properties['INIT'] = init
        edges = {name+'/'+role: dict(driver=driver, properties=dict(
            NAME=name+'/'+role, REF_PIN_NAME=role, DIRECTION='IN', IS_INVERTED='0'))
            for role, driver in (inputs or {}).items()}
        out = name+'/'+output
        nodes[out] = dict(cell=name, reference=ref, kind=kind, properties=properties,
                          output_properties=dict(NAME=out, REF_PIN_NAME=output, DIRECTION='OUT', IS_INVERTED='0'),
                          inputs=edges)
        return out

    def lut(inputs, table):
        nonlocal ordinal
        ordinal += 1
        return cell(scope+f'fixture_logic_{ordinal}', f'LUT{len(inputs)}',
                    {f'I{i}': v for i,v in enumerate(inputs)},
                    init=f"{1<<len(inputs)}'h{table:x}")

    zero = cell(scope+'ground', 'GND', output='G', kind='constant')
    one = cell(scope+'power', 'VCC', output='P', kind='constant')
    interfaces = {}
    for role in R.A.LIVE_ROLES:
        out = cell(N.ROOT+'external/'+role, 'FDRE', output='Q', kind='input')
        nodes[out]['roles'] = [role]
        interfaces[role] = dict(driver=out, inversion='0')
    for reg in sorted(R.register_names(0)):
        cell(reg, 'FDRE', {'C': interfaces['clk']['driver'], 'D': reg+'/Q', 'CE': one, 'R': zero},
             output='Q', init="1'b0", kind='state')
    q = lambda name: scope+name+'_reg/Q'
    bit = lambda name, i: scope+name+f'_reg[{i}]/Q'
    # Independent source encoding: reject on any blocked condition, using NOR
    # and OR reductions rather than the expected equality/conjunction formula.
    def reduce_or(values):
        values = list(values)
        if not values: return zero
        while len(values) > 1:
            group, rest = values[:6], values[6:]
            values = [lut(group, (1 << (1 << len(group))) - 2)] + rest
        return values[0]

    def negate(value): return lut([value], 1)
    push = lut([interfaces['n_cvalid']['driver'], interfaces['rst']['driver'], bit('cn', 4)], 2)
    memories = []; stored = []
    for i in range(24):
        memory = scope + f'opaque_storage_{(i*7)%24}'
        inputs = {'I': interfaces[f'n_caddr[{i}]']['driver'], 'CLK': interfaces['clk']['driver'], 'WE': push}
        for address, family in (('RADR','cr'),('WADR','cp')):
            inputs.update({address+str(k): bit(family,k) for k in range(4)}); inputs[address+'4'] = zero
        stored.append(cell(memory,'RAMD32',inputs,init="32'h00000000",kind='state',IS_CLK_INVERTED="1'b0"))
        memories.append(memory)
    differences = [lut([stored[i],bit('next_addr',i)],6) for i in range(24)]
    differences += [lut([stored[i],bit('pending_addr',i)],6) for i in range(6,24)]
    mismatch = reduce_or(differences+[bit('next_addr',24)])
    pending_nonzero = reduce_or(bit('pending_count',i) for i in range(5))
    blocked = [q('pending_valid'), negate(reduce_or(bit('cn',i) for i in range(5))),
               reduce_or(bit('writes',i) for i in range(8)),bit('reads',7),
               lut([pending_nonzero,mismatch],8)]
    root = negate(reduce_or(blocked))
    channel = dict(nodes=nodes,interfaces=interfaces,registers=sorted(R.register_names(0)),memories=memories,root=root)
    return channel


def test_independent_gate_encoding_proves_bounded_equation_without_semantic_acceptance():
    result = R.review_channel(0, fixture())
    assert result['equation_proven_for_all_current_state_assignments']
    assert len(result['state_output_assumptions']) == 81
    assert len(result['ram_read_output_assumptions']) == 24
    assert len(result['ram_bit_mapping']) == 24
    assert not result['source_elided_bits_proven']
    assert not result['state_transition_correspondence_proven']
    assert not result['paired_preservation_proven']


@pytest.mark.parametrize('change', ['equation','missing_register','register_init','register_clock',
    'register_polarity','missing_ram','duplicate_data_bit','data_inversion','read_pointer','write_pointer',
    'high_read_address','high_write_address','ram_init','ram_clock','ram_clock_invert',
    'missing_ram_clock_property','write_enable','missing_ram_input','ram_output_polarity',
    'ram_reference','interface_polarity','renamed_state','hidden_state','output_identity',
    'missing_logic_polarity','cycle','missing_node'])
def test_equation_and_boundary_mutations_reject(change):
    f=fixture(); nodes=f['nodes']; scope=N.ROOT+'u_native0/'
    memory=f['memories'][0]; ram=nodes[memory+'/O']; reg=scope+'reads_reg[7]'
    edge=lambda role:ram['inputs'][memory+'/'+role]
    zero=scope+'ground/G';one=scope+'power/P'
    if change=='equation': nodes[f['root']]['properties']['INIT']="2'h2"
    elif change=='missing_register':f['registers'].remove(reg)
    elif change=='register_init':nodes[reg+'/Q']['properties']['INIT']="1'b1"
    elif change=='register_clock':nodes[reg+'/Q']['inputs'][reg+'/C']['driver']=zero
    elif change=='register_polarity':nodes[reg+'/Q']['output_properties']['IS_INVERTED']='1'
    elif change=='missing_ram':f['memories'].pop()
    elif change=='duplicate_data_bit':edge('I')['driver']=f['interfaces']['n_caddr[1]']['driver']
    elif change=='data_inversion':edge('I')['properties']['IS_INVERTED']='1'
    elif change in ('read_pointer','write_pointer'):edge('RADR0' if change=='read_pointer' else 'WADR0')['driver']=scope+'cr_reg[1]/Q'
    elif change in ('high_read_address','high_write_address'):edge('RADR4' if change=='high_read_address' else 'WADR4')['driver']=one
    elif change=='ram_init':ram['properties']['INIT']="32'h00000001"
    elif change=='ram_clock':edge('CLK')['driver']=zero
    elif change=='ram_clock_invert':ram['properties']['IS_CLK_INVERTED']="1'b1"
    elif change=='missing_ram_clock_property':del ram['properties']['IS_CLK_INVERTED']
    elif change=='write_enable':edge('WE')['driver']=one
    elif change=='missing_ram_input':del ram['inputs'][memory+'/RADR4']
    elif change=='ram_output_polarity':ram['output_properties']['IS_INVERTED']='1'
    elif change=='ram_reference':ram['reference']=ram['properties']['REF_NAME']='RAMS32'
    elif change=='interface_polarity':f['interfaces']['n_cvalid']['inversion']='1'
    elif change=='renamed_state':f['registers'][0]+='_replica'
    elif change=='hidden_state':
        node=nodes[f['root']];node['reference']=node['properties']['REF_NAME']='FDRE'
    elif change=='output_identity':nodes[f['root']]['output_properties']['NAME']='different/O'
    elif change=='missing_logic_polarity':del nodes[f['root']]['output_properties']['IS_INVERTED']
    elif change=='cycle':nodes[f['root']]['inputs'][nodes[f['root']]['cell']+'/I0']['driver']=f['root']
    elif change=='missing_node':del nodes[f['root']]
    with pytest.raises((ValueError,KeyError)):R.review_channel(0,f)


def test_state_transitions_are_explicitly_outside_this_combinational_proof():
    f=fixture();reg=N.ROOT+'u_native0/pending_valid_reg'
    f['nodes'][reg+'/Q']['inputs'][reg+'/D']['driver']=N.ROOT+'u_native0/power/P'
    result=R.review_channel(0,f)
    assert result['equation_proven_for_all_current_state_assignments']
    assert not result['state_transition_correspondence_proven']


def test_carry_bdd_matches_integer_addition_and_dual_carry_boundary():
    b=P.BDD();nodes={};cuts={};name='carry'
    for i in range(8):
        for family in ('a','b'):
            output=f'{family}{i}/Q';cuts[output]=b.var(f'{i:02}{family}')
            nodes[output]=dict(cell=output[:-2],reference='FDRE',properties=dict(NAME=output[:-2],REF_NAME='FDRE'),inputs={},
                              output_properties=dict(NAME=output,REF_PIN_NAME='Q',DIRECTION='OUT',IS_INVERTED='0'))
    for value,ref,role in ((0,'GND','G'),(1,'VCC','P')):
        output=f'constant{value}/'+role
        nodes[output]=dict(cell=f'constant{value}',reference=ref,properties=dict(NAME=f'constant{value}',REF_NAME=ref),inputs={},
                          output_properties=dict(NAME=output,REF_PIN_NAME=role,DIRECTION='OUT',IS_INVERTED='0'))
    def inp(cell,role,driver):return dict(driver=driver,properties=dict(NAME=cell+'/'+role,REF_PIN_NAME=role,DIRECTION='IN',IS_INVERTED='0'))
    inputs={'carry/CI':inp('carry','CI','constant0/G'),'carry/CI_TOP':inp('carry','CI_TOP','constant0/G')}
    for i in range(8):
        cell=f'select{i}';output=cell+'/O'
        nodes[output]=dict(cell=cell,reference='LUT2',properties=dict(NAME=cell,REF_NAME='LUT2',INIT="4'h6"),
                          inputs={cell+'/I0':inp(cell,'I0',f'a{i}/Q'),cell+'/I1':inp(cell,'I1',f'b{i}/Q')},
                          output_properties=dict(NAME=output,REF_PIN_NAME='O',DIRECTION='OUT',IS_INVERTED='0'))
        inputs[f'carry/S[{i}]']=inp('carry',f'S[{i}]',output)
        inputs[f'carry/DI[{i}]']=inp('carry',f'DI[{i}]',f'a{i}/Q')
    for role in [f'{role}[{i}]' for role in ('O','CO') for i in range(8)]:
        output='carry/'+role
        nodes[output]=dict(cell='carry',reference='CARRY8',properties=dict(NAME='carry',REF_NAME='CARRY8',CARRY_TYPE='SINGLE_CY8'),inputs=inputs,
                          output_properties=dict(NAME=output,REF_PIN_NAME=role,DIRECTION='OUT',IS_INVERTED='0'))
    def evaluate(node,assignments):
        while node>1:
            variable,low,high=b.nodes[node];node=high if assignments[variable] else low
        return node
    randomizer=random.Random(712)
    pairs=[(a,c) for a in (0,1,127,128,254,255) for c in (0,1,127,128,254,255)]
    pairs += [(randomizer.randrange(256),randomizer.randrange(256)) for _ in range(1024)]
    for mode in ('SINGLE_CY8','DUAL_CY4'):
        for node in nodes.values():
            if node['reference']=='CARRY8':node['properties']['CARRY_TYPE']=mode
        inputs['carry/CI_TOP']['driver']='constant0/G' if mode=='SINGLE_CY8' else 'constant1/P'
        circuit=R.Circuit(nodes,cuts,b);outputs=[circuit.value(f'carry/O[{i}]') for i in range(8)]+[circuit.value('carry/CO[7]')]
        for a,c in pairs:
            assignment={f'{i:02}{family}':(value>>i)&1 for family,value in (('a',a),('b',c)) for i in range(8)}
            observed=sum(evaluate(node,assignment)<<i for i,node in enumerate(outputs))
            expected=a+c if mode=='SINGLE_CY8' else (((a&15)+(c&15))&15) | (((a>>4)+(c>>4)+1)<<4)
            assert observed==expected


def test_public_entrypoint_requires_actual_raw_binding(tmp_path):
    path=tmp_path/'untrusted.json';path.write_text('{"capture_complete":true,"issues":[]}')
    result=R.review_artifact(topology_path=path,topology_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             checkpoint_sha256='a'*64,helper_sha256='b'*64,native_helper_sha256='c'*64)
    assert result['issues'] and not result['observed_equation_proven']
    assert not result['source_semantics_proven'] and not result['paired_preservation_proven']
    assert not result['route_qualified'] and not result['hardware_measured']
    assert result['pending_reviews']==R.PENDING
