"""Independent bounded graph fixtures; no synthetic fixture qualifies hardware."""
import hashlib
import pytest
from boards.c1_f2_alex import admission_review as R, native_review as N


def seq(values): return ' '.join('{' + str(v) + '}' for v in values)
def td(values): return seq(x for pair in values.items() for x in pair)


def fixture():
    e = R.Evidence()
    def cell(name, ref, pins, sequential=False):
        e.queries['select','cell',name] = [name]
        e.objects['cell',name] = dict(NAME=name,REF_NAME=ref,IS_PRIMITIVE='1',IS_BLACKBOX='0',IS_SEQUENTIAL=str(int(sequential)),INIT="1'b0")
        e.cells[name] = [name+'/'+p for p in pins]; e.queries['cell_pins',name] = e.cells[name]
        for role, direction in pins.items():
            pin = name+'/'+role
            e.queries['select','pin',pin] = [pin]; e.queries['pin_owner',pin] = [name]
            e.objects['pin',pin] = dict(NAME=pin,REF_PIN_NAME=role,DIRECTION=direction,IS_INVERTED='0',IS_CLOCK='0')
    def input_network(pin, driver):
        net = pin + '_net'; props = e.objects['pin',pin]
        e.queries['input_nets',pin] = [net]; e.queries['input_drivers',pin,net] = [driver]; e.queries['input_ports',pin,net] = []
        return td(dict(scope='selected_input_and_drivers',input=pin,properties=td({k:props[k] for k in ('NAME','REF_PIN_NAME','DIRECTION')}),nets=seq([net]),drivers=seq([driver]),ports=''))
    constant = N.ROOT+'GND'; cell(constant,'GND',{'G':'OUT'})
    for pc in map(str,range(8)):
        interfaces = {}; cuts = {}
        for role in R.LIVE_ROLES:
            name = R.interface_net(pc,role); owner = N.ROOT+'external/'+('shared' if role in ('clk','rst') else pc)+'/'+role
            cell(owner,'FDRE',{'Q':'OUT'},True); output=owner+'/Q';interfaces[role]=output;cuts[output]=seq([role])
            e.queries['select','net',name]=[name];e.queries['admission_net_drivers',name]=[output];e.queries['admission_net_ports',name]=[]
            e.queries['admission_interface_clocks',pc,role]=['core'] if role=='clk' else []
            e.admission['interface'][pc,role] = dict(role=role,kind='net',name=name,network=td(dict(scope='selected_net_and_drivers',net=name,drivers=seq([output]),ports='')))
        for role in R.UNCUT_ROLES:
            e.queries['admission_uncut_selection',pc,role]=[]
            e.admission['uncut_role'][pc,role]=dict(role=role,kind='net',name=R.interface_net(pc,role),observed='',disposition='no_assumed_input_or_constant')
        e.admission['cut_inventory'][pc,]=dict(complete='1',outputs=td(cuts))
        owner=N.ROOT+'u_native'+pc+'/state';output=owner+'/Q'
        cell(owner,'FDRE',dict(C='IN',CE='IN',D='IN',R='IN',Q='OUT'),True)
        gate=N.ROOT+'u_mem/reads[7]_i_1'+('' if pc=='0' else '__'+str(int(pc)-1));cell(gate,'LUT2',{'I1':'IN'})
        e.admission['root'][pc,]=dict(gate=gate,input=gate+'/I1',network=input_network(gate+'/I1',output))
        inputs=[owner+'/'+role for role in ('C','CE','D','R')]
        e.queries['admission_inputs',pc,owner]=inputs;e.queries['admission_outputs',pc,owner]=[output]
        e.admission['cell'][pc,owner]=dict(cell=owner,reference='FDRE',sequential='1',inputs=seq(inputs),outputs=seq([output]))
        for role, driver in {'C':interfaces['clk'],'CE':constant+'/G','D':output,'R':interfaces['rst']}.items():
            pin=owner+'/'+role
            e.admission['edge'][pc,pin]=dict(cell=owner,input=pin,network=input_network(pin,driver))
        for role in ('clk','rst'):
            pin=interfaces[role];e.admission['cut'][pc,pin]=dict(output=pin,cell=pin.rsplit('/',1)[0],reference='FDRE',roles=seq([role]))
        e.admission['constant'][pc,constant+'/G']=dict(output=constant+'/G',cell=constant,reference='GND')
        net=output+'_net';leaves=[output,owner+'/D',gate+'/I1']
        e.queries['segments',output]=[net];e.queries['select','net',net]=[net];e.queries['leaves',net]=leaves;e.queries['ports',net]=[]
        network=dict(nets=[net],leaves=leaves,drivers=[output],ports=[]);e.networks['pin',output]=network
        e.admission['output'][pc,output]=dict(output=output,network=td({k:seq(v) for k,v in network.items()}))
        e.admission['walk'][pc,]=dict(root=output,outputs=seq([output,interfaces['clk'],interfaces['rst'],constant+'/G']),cells=seq([owner]),registers=seq([owner]),memories='')
    return e


def test_complete_observed_graph_has_only_declared_external_cuts():
    facts=R.facts(fixture())
    assert len(facts['channels'])==8
    assert all(len(c['interfaces'])==30 and len(c['registers'])==1 for c in facts['channels'].values())
    assert 'no values assigned' in facts['cut_scope']


@pytest.mark.parametrize('mutation', ['missing_role','extra_cut','absent_value','absent_present','internal_cut','driver_disagrees','missing_edge','missing_dependency','missing_state_output','state_summary','outgoing_delivery','root_delivery','wrong_pin_owner','unavailable_init'])
def test_graph_mutations_cannot_claim_complete_inventory(mutation):
    e=fixture();pc='0';owner=N.ROOT+'u_native0/state';output=owner+'/Q'
    if mutation=='missing_role':del e.admission['interface'][pc,'clk']
    elif mutation=='extra_cut':e.admission['cut_inventory'][pc,]['outputs'] += ' {arbitrary/Q} {clk}'
    elif mutation=='absent_value':e.admission['uncut_role'][pc,'n_cwe']['disposition']='assumed_zero'
    elif mutation=='absent_present':e.queries['admission_uncut_selection',pc,'n_cwe']=['retained']
    elif mutation=='internal_cut':
        row=e.admission['interface'][pc,'clk'];name=row['name'];e.queries['admission_net_drivers',name]=[output];row['network']=td(dict(scope='selected_net_and_drivers',net=name,drivers=seq([output]),ports=''))
    elif mutation=='driver_disagrees':e.queries['admission_net_drivers',R.interface_net(pc,'clk')]=['different/Q']
    elif mutation=='missing_edge':del e.admission['edge'][pc,owner+'/D']
    elif mutation=='missing_dependency':e.admission['walk'][pc,]['outputs']=seq([output,N.ROOT+'GND/G'])
    elif mutation=='missing_state_output':e.admission['cell'][pc,owner]['outputs']=''
    elif mutation=='state_summary':e.admission['walk'][pc,]['registers']=''
    elif mutation=='outgoing_delivery':
        network=e.networks['pin',output];network['leaves'].remove(owner+'/D');e.queries['leaves',output+'_net']=network['leaves'];e.admission['output'][pc,output]['network']=td({k:seq(v) for k,v in network.items()})
    elif mutation=='root_delivery':
        network=e.networks['pin',output];network['leaves'].remove(N.ROOT+'u_mem/reads[7]_i_1/I1');e.queries['leaves',output+'_net']=network['leaves'];e.admission['output'][pc,output]['network']=td({k:seq(v) for k,v in network.items()})
    elif mutation=='wrong_pin_owner':e.queries['pin_owner',owner+'/D']=[N.ROOT+'other_state']
    elif mutation=='unavailable_init':del e.objects['cell',owner]['INIT']
    with pytest.raises(ValueError):R.facts(e)


def test_disconnected_declared_component_cannot_enlarge_reachable_walk():
    e=fixture();pc='0';other=N.ROOT+'extra_constant';out=other+'/G'
    e.objects['cell',other]=dict(NAME=other,REF_NAME='GND',IS_PRIMITIVE='1',IS_BLACKBOX='0',IS_SEQUENTIAL='0')
    e.objects['pin',out]=dict(NAME=out,REF_PIN_NAME='G',DIRECTION='OUT',IS_INVERTED='0')
    e.queries['select','cell',other]=[other];e.queries['select','pin',out]=[out];e.queries['pin_owner',out]=[other]
    e.admission['constant'][pc,out]=dict(output=out,cell=other,reference='GND');e.admission['walk'][pc,]['outputs']+=' '+seq([out])
    with pytest.raises(ValueError,match='disconnected'):R.facts(e)


def test_unknown_or_outside_state_is_not_an_input_variable():
    e=fixture();owner=N.ROOT+'u_native0/state'
    e.objects['cell',owner]['REF_NAME']='RAMB36E2';e.admission['cell']['0',owner]['reference']='RAMB36E2'
    with pytest.raises(ValueError,match='unreviewed admission state'):R.facts(e)


def test_envelope_failure_cannot_claim_functional_or_paired_semantics(tmp_path):
    p=tmp_path/'raw';p.write_text('header schema 1 kind c1_accounting_boundary_topology checkpoint_sha256 '+('a'*64)+' checkpoint /observed route_qualified 0 hardware_measured 0\n')
    result=R.review_artifact(topology_path=p,topology_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),checkpoint_sha256='a'*64,helper_sha256='b'*64,native_helper_sha256='c'*64)
    assert not result['capture_complete'] and result['issues']
    assert result['route_qualified'] is result['functional_semantics_proven'] is result['paired_preservation_proven'] is False
    assert len(result['pending_reviews'])==3


def serialize(e):
    """Emit fixture observations through the actual parser, without Tcl evaluation."""
    rows = [seq(['header', 'schema', '1', 'kind', 'c1_accounting_boundary_topology',
                 'checkpoint_sha256', 'a'*64, 'checkpoint', '/observed',
                 'route_qualified', '0', 'hardware_measured', '0']),
            seq(['admission_sources', 'helper_sha256', 'b'*64, 'native_helper_sha256', 'c'*64])]
    for key, values in e.queries.items(): rows.append(seq(['query', seq(key), 'status', 'observed', 'total', str(len(values)), 'names', seq(values)]))
    for (kind, name), props in e.objects.items():
        for key, value in props.items(): rows.append(seq(['property', kind, name, key, 'status', 'observed', 'value', value]))
    for kind, name, key in sorted(e.unavailable): rows.append(seq(['property', kind, name, key, 'status', 'unavailable']))
    for name, pins in e.cells.items(): rows.append(seq(['cell_pins', name, 'names', seq(pins)]))
    for (kind, name), network in e.networks.items(): rows.append(seq(['network', kind, name, *[x for k,v in network.items() for x in (k,seq(v))]]))
    for tag, records in e.admission.items():
        for key, row in records.items(): rows.append(seq(['admission_'+tag, key[0], *[x for pair in row.items() for x in pair]]))
    rows.append(seq(['summary', 'queries', str(len(e.queries)), 'incomplete_or_error', '0', 'output_byte_cap', '0', 'route_qualified', '0', 'hardware_measured', '0']))
    return '\n'.join(rows)+'\n'


def replay(tmp_path, text):
    p=tmp_path/'capture';p.write_text(text)
    return R.review_artifact(topology_path=p,topology_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            checkpoint_sha256='a'*64,helper_sha256='b'*64,native_helper_sha256='c'*64)


def test_complete_parser_replay_never_implies_functional_or_hardware_acceptance(tmp_path):
    e=fixture();e.unavailable.add(('cell',N.ROOT+'u_native0/state','IS_C_INVERTED'))
    result=replay(tmp_path,serialize(e))
    assert result['capture_complete'] and not result['issues']
    node=result['canonical_facts']['channels']['0']['nodes'][N.ROOT+'u_native0/state/Q']
    assert node['unavailable_cell_properties']==['IS_C_INVERTED']
    assert result['route_qualified'] is result['functional_semantics_proven'] is result['paired_preservation_proven'] is False
    assert len(result['pending_reviews'])==3


@pytest.mark.parametrize('mutation',['helper','checkpoint','queries','gap','cap','claim','unknown','duplicate','after_summary','ambiguous','conflicting_property','missing_source'])
def test_parser_and_envelope_mutations_reject(tmp_path,mutation):
    text=serialize(fixture())
    if mutation=='helper': text=text.replace('b'*64,'d'*64)
    elif mutation=='checkpoint': text=text.replace('a'*64,'d'*64)
    elif mutation=='queries':text=text.replace('{summary} {queries}', '{query} {bogus}\n{summary} {queries}')
    elif mutation=='gap':text=text.replace('{incomplete_or_error} {0}','{incomplete_or_error} {1}')
    elif mutation=='cap':text=text.replace('{output_byte_cap} {0}','{output_byte_cap} {1}')
    elif mutation=='claim':text=text.replace('{route_qualified} {0}','{route_qualified} {1}')
    elif mutation=='unknown':text=text.replace('{admission_root}', '{admission_unreviewed}')
    elif mutation=='duplicate':text=text.replace('{summary}', next(x for x in text.splitlines() if x.startswith('{admission_root}'))+'\n{summary}')
    elif mutation=='after_summary':text+='admission_sources helper_sha256 '+('b'*64)+' native_helper_sha256 '+('c'*64)+'\n'
    elif mutation=='ambiguous':text=text.replace('{summary}', 'selection net missing status missing_or_ambiguous observed {}\n{summary}')
    elif mutation=='conflicting_property':text=text.replace('{summary}', 'property cell '+N.ROOT+'u_native0/state INIT status unavailable\n{summary}')
    elif mutation=='missing_source':text='\n'.join(x for x in text.splitlines() if not x.startswith('{admission_sources}'))+'\n'
    result=replay(tmp_path,text)
    assert result['issues'] and not result['capture_complete']
    assert result['functional_semantics_proven'] is result['paired_preservation_proven'] is result['route_qualified'] is False


def test_unknown_or_conflicting_extra_primitive_property_rejects():
    e=R.Evidence();e.add(N.tcl_list(serialize(fixture()).splitlines()[0]))
    e.add(['property','cell','x','CARRY_TYPE','status','observed','value','SINGLE_CY8'])
    with pytest.raises(ValueError,match='conflicting'):e.add(['property','cell','x','CARRY_TYPE','status','unavailable'])
    with pytest.raises(ValueError,match='unknown'):e.add(['property','cell','x','PRIVATE_UNKNOWN','status','observed','value','0'])
