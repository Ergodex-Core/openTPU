"""Replay C1 flow and disabled-input facts without granting route acceptance.

The routed host-ready rewrite and failed legacy native checks need independent
paired/semantic witnesses. They are never accepted from a step's status alone.
"""
from __future__ import annotations
import re
from . import native_review as N
from . import structural_capture_review as C

FORWARD = {'awvalid', 'wvalid', 'arvalid', 'bready', 'rready'}


def inside(name, prefix):
    return name.startswith(prefix + '/')


def bit(value):
    N.require(isinstance(value, str) and value.lower() in ('0', '1', 'false', 'true'),
              'invalid observed Boolean')
    return value.lower() in ('1', 'true')


def one(step, tag):
    rows = [row for row in step['records'] if row[0] == tag]
    N.require(len(rows) == 1, 'missing/duplicate ' + tag)
    return rows[0]


def fields(step, tag):
    return N.pairs(one(step, tag)[2:], C.FIELDS[tag])


def flow_scope(iface, signal):
    match = re.fullmatch(r'(aw|ar|w|b|r)(valid|ready)', signal)
    N.require(match, 'unknown AXI handshake')
    channel = match[1]; forward = signal in FORWARD
    if iface == 'S_HOST':
        external = N.ROOT + 'u_pcis_pipe'
        internal = N.SC + f's00_entry_pipeline/s00_mmu/inst/{channel}_sreg'
        anchor = external + '/m_axi_' + signal
        kind = 'pin'; direction = 'OUT' if forward else 'IN'
    elif match := re.fullmatch('S_N([0-7])', iface):
        pc = int(match[1]); si = f's{pc+1:02d}'
        N.require(signal in ('bvalid', 'arvalid', 'arready', 'rvalid'), 'inactive native flow')
        external = N.ROOT + f'u_native{pc}'
        internal = N.SC + f'{si}_entry_pipeline/{si}_mmu/inst/{channel}_sreg'
        anchor = N.ROOT + f'n{pc}_{signal}'; kind = 'net'; direction = None
    elif match := re.fullmatch('M_H([0-7])', iface):
        pc = int(match[1]); mi = f'm{pc:02d}'
        external = N.ROOT + 'u_hbm'
        internal = N.SC + f'{mi}_exit_pipeline/{mi}_exit/inst/{channel}_reg'
        anchor = external + f'/{"i" if forward else "o"}_axi_{signal}[{pc}]'
        kind = 'pin'; direction = 'IN' if forward else 'OUT'; forward = not forward
    else:
        raise ValueError('unknown fabric interface')
    return {'external': external, 'internal': internal, 'anchor': anchor, 'kind': kind,
            'direction': direction, 'source': external if forward else internal,
            'destination': internal if forward else external}


def relocated(pc, signal):
    suffix = '' if pc == 0 else f'__{pc-1}'
    if signal == 'arready':
        return {N.ROOT + f'u_mem/pending_count[4]_i_1{suffix}/I0',
                N.ROOT + f'u_mem/n_err_i_4{suffix}/I4'}
    if signal == 'rvalid':
        return {N.ROOT + f'u_mem/reads[7]_i_1{suffix}/I0',
                N.ROOT + f'u_mem/n_err_i_1{suffix}/I1'}
    if signal == 'bvalid':
        suffix = '' if pc == 7 else f'__{6-pc}'
        return {N.ROOT + f'u_mem/n_err_i_2{suffix}/I1'}
    return set()


def review_flow(step, *, native=None, receive_facts=None):
    N.require(tuple(step['label']) in C.STEPS and step['label'][0] == 'flow', 'unknown flow step')
    _, iface, signal = step['label']
    N.require(step['result']['state'] == 'observed', 'failed flow needs separate semantic closure')
    N.require((iface, signal) != ('S_HOST', 'wready'), 'host ready needs paired replica proof')
    allowed = {'fabric_flow', 'flow_detail'}
    if iface.startswith('S_N') and signal == 'bvalid': allowed.add('native_b_response')
    if iface.startswith('S_N') and signal == 'rvalid':
        allowed |= {'native_receive_gate', 'tag_ram_control', 'tag_ram_inventory'}
    N.require(all(row[0] in allowed for row in step['records']), 'unexpected flow witness')
    for row in step['records']: C.payload(row, tuple(step['label']))
    scope = flow_scope(iface, signal); flow = fields(step, 'fabric_flow'); detail = fields(step, 'flow_detail')
    anchors = N.names(detail['anchor']); wanted = {scope['anchor']}
    if iface == 'S_HOST': wanted.add(scope['anchor'] + '[0]')
    N.require(len(anchors) == 1 and anchors[0] in wanted and
              N.names(flow['anchor']) == anchors and detail['anchor_kind'] == scope['kind'],
              'flow anchor identity differs')
    anchor = anchors[0]; props = N.td(detail['anchor_properties'])
    if scope['kind'] == 'pin':
        N.require(set(props) == {'NAME', 'DIRECTION', 'IS_INVERTED'} and props['NAME'] == anchor and
                  props['DIRECTION'] == scope['direction'] and not bit(props['IS_INVERTED']),
                  'flow anchor direction/polarity differs')
    else:
        N.require(not props, 'native net represented as a pin')
    nets = N.names(detail['nets']); local = N.names(detail['local_nets'])
    N.require(nets and local and set(local) <= set(nets) and N.names(flow['nets']) == nets and
              not N.names(detail['ports']), 'flow disconnected/inconsistent/escaping network')
    if scope['kind'] == 'net': N.require(nets == local == [anchor], 'native anchor network differs')
    drivers = N.names(detail['drivers'])
    N.require(len(drivers) == 1 and N.names(flow['driver']) == drivers, 'flow driver cardinality differs')
    driver = drivers[0]; driver_cells = N.td(detail['driver_cells'])
    N.require(set(driver_cells) == {driver}, 'flow typed driver inventory differs')
    owners = N.tcl_list(driver_cells[driver]); N.require(len(owners) == 1, 'flow driver owner ambiguity')
    owner = N.td(owners[0], ('NAME', 'REF_NAME', 'IS_BLACKBOX'))
    N.require(owner['NAME'] == driver.rsplit('/', 1)[0] and owner['REF_NAME'] and
              owner['REF_NAME'] not in ('GND', 'VCC') and not bit(owner['IS_BLACKBOX']),
              'constant/unresolved active flow driver')
    N.require(flow['source_owner'] == scope['source'] and flow['destination_owner'] == scope['destination'] and
              inside(driver, scope['source']), 'flow source/destination scope differs')
    leaves = N.names(flow['leaf_pins']); directions = N.tcl_list(flow['leaf_directions'])
    N.require(leaves and len(leaves) == len(directions) and set(directions) <= {'IN', 'OUT'} and
              [pin for pin, direction in zip(leaves, directions) if direction == 'OUT'] == drivers,
              'flow leaf direction/driver inventory differs')
    pc = int(iface[-1]) if iface.startswith('S_N') else None
    gate = None
    if pc is not None:
        N.require(isinstance(native, N.Evidence) and not native.issues and not native.incomplete,
                  'native typed evidence missing/incomplete')
        native.selected('net', anchor)
        network = native.networks.get(('net', anchor), {})
        N.require(set(network) == {'nets', 'leaves', 'drivers', 'ports'} and
                  all(set(network[key]) == set(value) for key, value in
                      {'nets': nets, 'leaves': leaves, 'drivers': drivers, 'ports': []}.items()) and
                  set(native.query('leaves', anchor)) == set(leaves) and not native.query('ports', anchor),
                  'native and structural network inventories differ')
        for pin, direction in zip(leaves, directions): native.pin(pin, direction)
        if signal == 'rvalid':
            matching = [fact for fact in (receive_facts or []) if fact['pc'] == pc]
            N.require(len(matching) == 1, 'native receive predicate missing')
            fact = matching[0]; gate = fact['gate'] + '/I0'
            row = fields(step, 'native_receive_gate')
            N.require(row['input'] == gate and row['reference'] == 'LUT2' and row['init'].lower() == "4'h2" and
                      N.names(row['reset_driver']) == [fact['reset_driver']] and
                      N.names(row['output']) == [fact['gate'] + '/O'] and
                      sorted(N.names(row['endpoints'])) == fact['endpoints'] and
                      N.td(row['cell_references']) == fact['cell_references'] and
                      {k: N.number(v) for k, v in N.td(row['sink_families']).items()} == fact['sink_families'],
                      'native receive witness disagrees with typed predicate')
            inventories = [N.pairs(r[1:], C.TAG_FIELDS['tag_ram_inventory']) for r in step['records'] if r[0] == 'tag_ram_inventory']
            controls = [N.pairs(r[1:], C.TAG_FIELDS['tag_ram_control']) for r in step['records'] if r[0] == 'tag_ram_control']
            expected_inventories = {}; expected_controls = {}
            for engine, memory in fact['tag_memories'].items():
                if memory is None: continue
                expected_inventories[engine] = memory
                for pin, control in memory['controls'].items():
                    expected_controls[pin] = (engine, memory['leaves'][pin.rsplit('/', 1)[0]], control)
            N.require(len(inventories) == len(expected_inventories) and
                      {r['engine'] for r in inventories} == set(expected_inventories),
                      'tag RAM inventory coverage differs')
            for inventory in inventories:
                memory = expected_inventories[inventory['engine']]
                N.require(set(N.names(inventory['cells'])) == set(memory['inventory']) and
                          set(N.names(inventory['write_enable_pins'])) == {name + '/WE' for name in memory['leaves']} and
                          inventory['clock_reference'] == N.ROOT + 'core_rst_reg/C' and
                          N.names(inventory['clocks']) == memory['clocks'], 'tag RAM inventory disagrees')
            N.require(len(controls) == len(expected_controls) and
                      {r['pin'] for r in controls} == set(expected_controls), 'tag RAM control coverage differs')
            for control in controls:
                engine, reference, expected = expected_controls[control['pin']]
                N.require(control['engine'] == engine and control['cell'] == control['pin'].rsplit('/', 1)[0] and
                          control['reference'] == reference and N.names(control['drivers']) == [expected['driver']] and
                          N.names(control['clocks']) == expected['clocks'], 'tag RAM control disagrees')
    receiver_count = 0; observed_gate = 0
    for pin, direction in zip(leaves, directions):
        if not inside(pin, scope['external']) and not inside(pin, scope['internal']):
            N.require(pc is not None and direction == 'IN' and
                      (pin in relocated(pc, signal) or pin == gate), 'flow crossed unreviewed owner')
            N.require(re.fullmatch('LUT[1-6]', native.prop('cell', pin.rsplit('/', 1)[0], 'REF_NAME')),
                      'relocated native input is not typed LUT')
            if pin == gate: observed_gate += 1
            if signal == 'bvalid': receiver_count += 1
        if direction == 'IN' and inside(pin, scope['destination']): receiver_count += 1
    N.require(receiver_count > 0 and N.number(flow['receiver_count']) == receiver_count,
              'flow receiver count differs')
    if pc is not None and signal == 'rvalid': N.require(observed_gate == 1, 'receive gate delivery missing')
    if pc is not None and signal == 'bvalid':
        row = fields(step, 'native_b_response'); endpoints = N.names(row['endpoints'])
        required = {N.ROOT + f'u_native{pc}/n_err_reg/D'} | {
            N.ROOT + f'u_native{pc}/writes_reg[{i}]/CE' for i in range(8)}
        N.require(N.names(row['driver']) == drivers and required <= set(endpoints) and
                  all(inside(pin, scope['external']) or inside(pin, scope['internal']) for pin in endpoints),
                  'native B response endpoint scope differs')
    return {'interface': iface, 'signal': signal, 'anchor': anchor, 'nets': nets,
            'driver': driver, 'receiver_count': receiver_count, 'leaf_pins': leaves,
            'leaf_directions': directions, 'source': scope['source'], 'destination': scope['destination']}


def review_disabled(step):
    N.require(tuple(step['label']) in C.STEPS and step['label'][0] == 'disabled_write',
              'unknown disabled-input step')
    N.require(step['result']['state'] == 'observed', 'failed disabled input needs separate closure')
    N.require(len(step['records']) == 3, 'disabled witness inventory differs')
    for row in step['records']: C.payload(row, tuple(step['label']))
    pc = int(step['label'][1]); channel = step['label'][2]; si = f's{pc+1:02d}'
    anchor = N.SC + f'{si}_entry_pipeline/{si}_mmu/inst/{channel}_sreg/s_axi_{channel}valid'
    network = fields(step, 'disabled_network'); polarity = fields(step, 'disabled_polarity')
    raw = fields(step, 'native_disabled_write')
    local = N.names(network['local_nets']); drivers = N.names(network['drivers'])
    N.require(N.names(network['anchor']) == N.names(raw['pin']) == [anchor] and
              local and local == N.names(raw['nets']) and 0 < N.number(network['segment_count']) <= 100000 and
              len(local) <= N.number(network['segment_count']) and len(drivers) == 1 and
              N.names(raw['driver']) == drivers and not N.names(network['ports']) and
              network['sink_scope'] == 'anchor_only', 'disabled input network differs')
    driver = drivers[0]; ap = N.td(polarity['anchor'], ('NAME', 'REF_PIN_NAME', 'DIRECTION', 'IS_INVERTED'))
    dp = N.td(polarity['driver'], ('NAME', 'REF_PIN_NAME', 'DIRECTION', 'IS_INVERTED'))
    N.require(ap['NAME'] == anchor and ap['REF_PIN_NAME'] == 's_axi_' + channel + 'valid' and
              ap['DIRECTION'] == 'IN' and not bit(ap['IS_INVERTED']) and dp['NAME'] == driver and
              dp['REF_PIN_NAME'] == 'G' and dp['DIRECTION'] == 'OUT' and not bit(dp['IS_INVERTED']) and
              N.names(polarity['owner']) == [driver.rsplit('/', 1)[0]] and
              polarity['reference'] == raw['primitive'] == 'GND', 'disabled input effective value is not zero')
    return {'pc': pc, 'channel': channel, 'anchor': anchor, 'driver': driver, 'value': 0}


def review(canonical_facts, *, native=None, receive_facts=None):
    """Review all flow/disabled steps; callers authenticate both raw inputs first."""
    labels = [label for label in C.STEPS if label[0] in ('flow', 'disabled_write')]
    result = {'kind': 'c1-structural-flow-review', 'schema': 1, 'facts': [],
              'proven_steps': [], 'pending_steps': [list(label) for label in labels],
              'issues': [], 'route_qualified': False, 'hardware_measured': False}
    for label in labels:
        try:
            steps = [step for step in canonical_facts['steps'] if step['label'] == list(label)]
            N.require(len(steps) == 1, 'missing/duplicate flow or disabled step')
            step = steps[0]
            facts = review_disabled(step) if label[0] == 'disabled_write' else review_flow(
                step, native=native, receive_facts=receive_facts)
            result['facts'].append(facts); result['proven_steps'].append(list(label))
            result['pending_steps'].remove(list(label))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            result['issues'].append({'step': list(label), 'reason': str(exc)})
    return result
