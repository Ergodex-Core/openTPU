"""Independent flow topology mutations must fail before route acceptance."""
from copy import deepcopy
import pytest
from boards.c1_f2_alex import structural_flow_review as S
from boards.c1_f2_alex import structural_capture_review as C
from boards.c1_f2_alex import native_review as N


def tl(values):
    return ' '.join('{' + str(value) + '}' for value in values)


def td(values):
    return tl(sum(([k, v] for k, v in values.items()), []))


def row(tag, identity, **values):
    return [tag, identity, *sum(([key, str(value)] for key, value in values.items()), [])]


def ordinary(iface='M_H0', signal='arvalid', scalar=False):
    scope = S.flow_scope(iface, signal); anchor = scope['anchor'] + ('[0]' if scalar else '')
    driver = scope['source'] + '/actual_state/Q'
    receiver = scope['destination'] + '/actual_receiver/D'
    net = N.ROOT + 'actual_signal'
    records = [row('fabric_flow', iface, signal=signal, anchor=tl([anchor]), nets=tl([net]),
        driver=tl([driver]), source_owner=scope['source'], destination_owner=scope['destination'],
        receiver_count='1', leaf_pins=tl([driver, receiver]), leaf_directions='OUT IN'),
        row('flow_detail', iface, signal=signal, anchor_kind='pin', anchor=tl([anchor]),
        anchor_properties=td({'NAME': anchor, 'DIRECTION': scope['direction'], 'IS_INVERTED': '0'}),
        local_nets=tl([net]), nets=tl([net]), ports='', drivers=tl([driver]),
        driver_cells=td({driver: tl([td({'NAME': driver.rsplit('/', 1)[0], 'REF_NAME': 'FDRE', 'IS_BLACKBOX': '0'})])}))]
    return {'label': ['flow', iface, signal], 'result': {'state': 'observed'}, 'records': records}


def disabled(pc=0, channel='aw'):
    si = f's{pc+1:02d}'; anchor = N.SC + f'{si}_entry_pipeline/{si}_mmu/inst/{channel}_sreg/s_axi_{channel}valid'
    driver = N.ROOT + 'GND/G'; net = N.ROOT + 'ground'
    return {'label': ['disabled_write', str(pc), channel], 'result': {'state': 'observed'}, 'records': [
        row('disabled_network', str(pc), channel=channel, anchor=tl([anchor]), local_nets=tl([net]),
            segment_count='27', drivers=tl([driver]), ports='', sink_scope='anchor_only'),
        row('disabled_polarity', str(pc), channel=channel,
            anchor=td({'NAME': anchor, 'REF_PIN_NAME': 's_axi_' + channel + 'valid', 'DIRECTION': 'IN', 'IS_INVERTED': '0'}),
            driver=td({'NAME': driver, 'REF_PIN_NAME': 'G', 'DIRECTION': 'OUT', 'IS_INVERTED': '0'}),
            owner=tl([N.ROOT + 'GND']), reference='GND'),
        row('native_disabled_write', str(pc), channel=channel, pin=tl([anchor]), nets=tl([net]), driver=tl([driver]), primitive='GND')]}


def field(step, tag, key, value):
    item = S.one(step, tag); item[item.index(key) + 1] = value


def dictionary(step, tag, key, subkey, value):
    values = N.td(S.fields(step, tag)[key]); values[subkey] = value; field(step, tag, key, td(values))


@pytest.mark.parametrize('pc', range(8))
@pytest.mark.parametrize('signal', C.SIGNALS)
def test_all_hbm_active_flows(pc, signal):
    result = S.review_flow(ordinary(f'M_H{pc}', signal))
    assert result['interface'] == f'M_H{pc}' and result['receiver_count'] == 1
    assert 'route_qualified' not in result


@pytest.mark.parametrize('signal', [x for x in C.SIGNALS if x != 'wready'])
@pytest.mark.parametrize('scalar', [False, True])
def test_host_scalar_and_vector_pin_names(signal, scalar):
    assert S.review_flow(ordinary('S_HOST', signal, scalar))['signal'] == signal


def test_host_wready_still_needs_replica_and_pair_proof():
    with pytest.raises(ValueError, match='paired replica'): S.review_flow(ordinary('S_HOST', 'wready'))


@pytest.mark.parametrize('mutation', ['error', 'missing_detail', 'duplicate_flow', 'wrong_step', 'extra_payload',
    'anchor', 'anchor_kind', 'anchor_direction', 'anchor_inversion', 'local_net', 'no_net', 'port',
    'two_drivers', 'wrong_owner', 'constant', 'blackbox', 'source', 'destination', 'leaf_direction',
    'extra_output', 'outside_receiver', 'no_receiver', 'wrong_count', 'duplicate_leaf'])
def test_active_flow_mutations_reject(mutation):
    step = ordinary(); flow = S.fields(step, 'fabric_flow'); detail = S.fields(step, 'flow_detail')
    if mutation == 'error': step['result']['state'] = 'error'
    elif mutation == 'missing_detail': step['records'].pop()
    elif mutation == 'duplicate_flow': step['records'].append(deepcopy(step['records'][0]))
    elif mutation == 'wrong_step': step['label'][1] = 'M_H1'
    elif mutation == 'extra_payload': step['records'].append(['routed_host_wready', 'bad'])
    elif mutation == 'anchor': field(step, 'flow_detail', 'anchor', 'wrong')
    elif mutation == 'anchor_kind': field(step, 'flow_detail', 'anchor_kind', 'net')
    elif mutation == 'anchor_direction': dictionary(step, 'flow_detail', 'anchor_properties', 'DIRECTION', 'OUT')
    elif mutation == 'anchor_inversion': dictionary(step, 'flow_detail', 'anchor_properties', 'IS_INVERTED', '1')
    elif mutation == 'local_net': field(step, 'flow_detail', 'local_nets', 'wrong')
    elif mutation == 'no_net': field(step, 'flow_detail', 'nets', '')
    elif mutation == 'port': field(step, 'flow_detail', 'ports', 'top_port')
    elif mutation == 'two_drivers': field(step, 'flow_detail', 'drivers', detail['drivers'] + ' another/Q')
    elif mutation in ('wrong_owner', 'constant', 'blackbox'):
        driver = N.names(detail['drivers'])[0]
        owner = {'NAME': driver.rsplit('/',1)[0], 'REF_NAME': 'FDRE', 'IS_BLACKBOX': '0'}
        owner[{'wrong_owner':'NAME','constant':'REF_NAME','blackbox':'IS_BLACKBOX'}[mutation]] = {'wrong_owner':'outside','constant':'VCC','blackbox':'1'}[mutation]
        field(step, 'flow_detail', 'driver_cells', td({driver: tl([td(owner)])}))
    elif mutation in ('source', 'destination'): field(step, 'fabric_flow', mutation + '_owner', 'outside')
    elif mutation == 'leaf_direction': field(step, 'fabric_flow', 'leaf_directions', 'OUT INOUT')
    elif mutation == 'extra_output': field(step, 'fabric_flow', 'leaf_directions', 'OUT OUT')
    elif mutation == 'outside_receiver': field(step, 'fabric_flow', 'leaf_pins', tl([N.names(flow['driver'])[0], 'other/D']))
    elif mutation == 'no_receiver': field(step, 'fabric_flow', 'leaf_pins', flow['driver']); field(step, 'fabric_flow', 'leaf_directions', 'OUT')
    elif mutation == 'wrong_count': field(step, 'fabric_flow', 'receiver_count', '2')
    elif mutation == 'duplicate_leaf': field(step, 'fabric_flow', 'leaf_pins', flow['leaf_pins'] + ' ' + flow['driver'])
    else: raise AssertionError(mutation)
    with pytest.raises(ValueError): S.review_flow(step)


@pytest.mark.parametrize('pc', range(8))
@pytest.mark.parametrize('channel', ('aw', 'w'))
def test_all_16_disabled_write_inputs(pc, channel):
    result = S.review_disabled(disabled(pc, channel))
    assert result['pc'] == pc and result['channel'] == channel and result['value'] == 0


@pytest.mark.parametrize('mutation', ['failed', 'missing', 'duplicate', 'wrong_step', 'anchor', 'port', 'two_drivers',
    'zero_segments', 'wrong_net', 'scope', 'inverted_sink', 'inverted_driver', 'wrong_reference', 'wrong_owner',
    'wrong_direction', 'wrong_pin_ref', 'nonboolean'])
def test_disabled_write_mutations_reject(mutation):
    step = disabled()
    if mutation == 'failed': step['result']['state'] = 'error'
    elif mutation == 'missing': step['records'].pop()
    elif mutation == 'duplicate': step['records'].append(deepcopy(step['records'][0]))
    elif mutation == 'wrong_step': step['label'][1] = '1'
    elif mutation == 'anchor': field(step, 'disabled_network', 'anchor', 'other')
    elif mutation == 'port': field(step, 'disabled_network', 'ports', 'outside')
    elif mutation == 'two_drivers': field(step, 'disabled_network', 'drivers', 'x/G y/G')
    elif mutation == 'zero_segments': field(step, 'disabled_network', 'segment_count', '0')
    elif mutation == 'wrong_net': field(step, 'native_disabled_write', 'nets', 'other')
    elif mutation == 'scope': field(step, 'disabled_network', 'sink_scope', 'full')
    elif mutation == 'inverted_sink': dictionary(step, 'disabled_polarity', 'anchor', 'IS_INVERTED', '1')
    elif mutation == 'inverted_driver': dictionary(step, 'disabled_polarity', 'driver', 'IS_INVERTED', '1')
    elif mutation == 'wrong_reference': field(step, 'disabled_polarity', 'reference', 'VCC')
    elif mutation == 'wrong_owner': field(step, 'disabled_polarity', 'owner', 'other')
    elif mutation == 'wrong_direction': dictionary(step, 'disabled_polarity', 'driver', 'DIRECTION', 'IN')
    elif mutation == 'wrong_pin_ref': dictionary(step, 'disabled_polarity', 'anchor', 'REF_PIN_NAME', 'other')
    elif mutation == 'nonboolean': dictionary(step, 'disabled_polarity', 'driver', 'IS_INVERTED', 'maybe')
    else: raise AssertionError(mutation)
    with pytest.raises(ValueError): S.review_disabled(step)


def native_fixture(pc=0, packed=False):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('native_flow_fixture', Path(__file__).with_name('test_c1_native_review.py'))
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    original = fixture.packed_fixture() if packed else fixture.fixture()
    evidence = N.Evidence(); evidence.__dict__.update(original.__dict__)
    facts = N.receive_review(evidence); fact = facts[pc]
    iface = f'S_N{pc}'; signal = 'rvalid'; scope = S.flow_scope(iface, signal); anchor = scope['anchor']
    network = evidence.networks['net', anchor]; driver = network['drivers'][0]
    owner = driver.rsplit('/', 1)[0]; leaves = network['leaves']
    rows = [row('fabric_flow', iface, signal=signal, anchor=tl([anchor]), nets=tl([anchor]),
        driver=tl([driver]), source_owner=scope['source'], destination_owner=scope['destination'],
        receiver_count='1', leaf_pins=tl(leaves), leaf_directions=tl([evidence.prop('pin', p, 'DIRECTION') for p in leaves])),
        row('flow_detail', iface, signal=signal, anchor_kind='net', anchor=tl([anchor]), anchor_properties='',
        local_nets=tl([anchor]), nets=tl([anchor]), ports='', drivers=tl([driver]),
        driver_cells=td({driver: tl([td({k:evidence.prop('cell', owner, k) for k in ('NAME','REF_NAME','IS_BLACKBOX')})])})),
        row('native_receive_gate', str(pc), input=fact['gate']+'/I0', reference='LUT2', init="4'h2",
        reset_driver=tl([fact['reset_driver']]), output=tl([fact['gate']+'/O']),
        sink_families=td(fact['sink_families']), endpoints=tl(fact['endpoints']), cell_references=td(fact['cell_references']))]
    for engine, memory in fact['tag_memories'].items():
        if memory is None: continue
        for pin, control in memory['controls'].items():
            rows.append(['tag_ram_control', 'engine', engine, 'cell', pin.rsplit('/',1)[0],
                'reference', memory['leaves'][pin.rsplit('/',1)[0]], 'pin', pin,
                'drivers', tl([control['driver']]), 'clocks', tl(control['clocks'])])
        rows.append(['tag_ram_inventory', 'engine', engine, 'cells', tl(memory['inventory']),
            'write_enable_pins', tl(name+'/WE' for name in memory['leaves']),
            'clock_reference', N.ROOT+'core_rst_reg/C', 'clocks', tl(memory['clocks'])])
    return {'label':['flow',iface,signal], 'result':{'state':'observed'}, 'records':rows}, evidence, facts


@pytest.mark.parametrize('pc', range(8))
@pytest.mark.parametrize('packed', (False, True))
def test_native_receive_joins_typed_network_and_complete_tag_controls(pc, packed):
    step, evidence, facts = native_fixture(pc, packed)
    assert S.review_flow(step, native=evidence, receive_facts=facts)['receiver_count'] == 1


@pytest.mark.parametrize('mutation', ['native_missing', 'query_leaf_missing', 'network_leaf_missing',
    'typed_direction', 'receive_missing', 'receive_reset', 'receive_endpoint', 'receive_family',
    'tag_control_missing', 'tag_driver', 'tag_inventory', 'tag_clock', 'failed_step'])
def test_native_and_tag_cross_capture_mutations_reject(mutation):
    step, evidence, facts = native_fixture(0, True)
    anchor = S.flow_scope('S_N0', 'rvalid')['anchor']
    if mutation == 'native_missing': evidence = None
    elif mutation == 'query_leaf_missing': evidence.queries['leaves', anchor] = evidence.queries['leaves', anchor][:-1]
    elif mutation == 'network_leaf_missing': evidence.networks['net', anchor]['leaves'] = evidence.networks['net', anchor]['leaves'][:-1]
    elif mutation == 'typed_direction': evidence.objects['pin', facts[0]['gate']+'/I0']['DIRECTION'] = 'OUT'
    elif mutation == 'receive_missing': facts = []
    elif mutation == 'receive_reset': field(step, 'native_receive_gate', 'reset_driver', 'other/Q')
    elif mutation == 'receive_endpoint': field(step, 'native_receive_gate', 'endpoints', 'other/D')
    elif mutation == 'receive_family': field(step, 'native_receive_gate', 'sink_families', 'other 1')
    elif mutation == 'tag_control_missing': step['records'].remove(next(r for r in step['records'] if r[0]=='tag_ram_control'))
    elif mutation == 'tag_driver':
        r = next(r for r in step['records'] if r[0]=='tag_ram_control'); r[r.index('drivers')+1] = 'other/Q'
    elif mutation == 'tag_inventory':
        r = next(r for r in step['records'] if r[0]=='tag_ram_inventory'); r[r.index('cells')+1] = ''
    elif mutation == 'tag_clock':
        r = next(r for r in step['records'] if r[0]=='tag_ram_control'); r[r.index('clocks')+1] = 'other_clock'
    elif mutation == 'failed_step': step['result']['state'] = 'error'
    else: raise AssertionError(mutation)
    with pytest.raises(ValueError): S.review_flow(step, native=evidence, receive_facts=facts)


def test_partial_success_never_drops_unobserved_steps():
    result = S.review({'steps':[ordinary(), disabled()]})
    assert result['proven_steps'] == [['flow','M_H0','arvalid'],['disabled_write','0','aw']]
    assert len(result['pending_steps']) == len(result['issues']) == 136
    assert result['route_qualified'] is result['hardware_measured'] is False
