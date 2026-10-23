"""Replay the complete observed C1 admission state boundary, never qualification.

Exact live input nets are cuts. Absent write-side/address roles are not assigned
values. Every other dependency must close through observed constants, native
state or fully inventoried supported combinational cells. RAM outputs are never
silently admitted as unconstrained terminal variables.
"""
from __future__ import annotations
import hashlib
import json
import re
from . import native_review as N, accounting_review as A

LIVE_ROLES = ('clk', 'rst', 'n_cvalid', 'm_bvalid', 'm_arready', 'm_rvalid') + tuple(f'n_caddr[{i}]' for i in range(24))
UNCUT_ROLES = ('n_cwe', 'n_wvalid', 'm_awready', 'm_wready', 'n_caddr[24]')
FIELDS = {
    'interface': ('role', 'kind', 'name', 'network'),
    'uncut_role': ('role', 'kind', 'name', 'observed', 'disposition'),
    'cut_inventory': ('complete', 'outputs'),
    'constant': ('output', 'cell', 'reference'),
    'cut': ('output', 'cell', 'reference', 'roles'),
    'output': ('output', 'network'),
    'cell': ('cell', 'reference', 'sequential', 'inputs', 'outputs'),
    'edge': ('cell', 'input', 'network'),
    'root': ('gate', 'input', 'network'),
    'walk': ('root', 'outputs', 'cells', 'registers', 'memories'),
}


def interface_net(pc, role):
    if role in ('clk', 'rst'): return N.ROOT + 'core_' + role
    if role in ('n_cvalid', 'n_cwe', 'n_wvalid'): return f'{N.ROOT}{role}[{pc}]'
    if role.startswith('m_'): return f'{N.ROOT}n{pc}_{role[2:]}'
    match = re.fullmatch(r'n_caddr\[([0-9]+)\]', role)
    N.require(match is not None, 'unknown admission interface role')
    return f'{N.ROOT}n_caddr[{pc}][{match[1]}]'


class Evidence(N.Evidence):
    def __init__(self):
        super().__init__()
        self.sources = []
        self.admission = {tag: {} for tag in FIELDS}

    def pin(self, name, direction):
        super().pin(name, direction)
        N.require(self.prop('pin', name, 'REF_PIN_NAME') == name.rsplit('/', 1)[1], 'admission pin reference role differs')

    def add(self, w):
        if len(w) >= 4 and w[0] == 'property' and w[1] == 'cell' and w[3] in ('CARRY_TYPE', 'IS_CLK_INVERTED', 'IS_WCLK_INVERTED'):
            N.require(self.headers and not self.seen_summary, 'property outside admission capture')
            self.counts['property'] += 1
            row = N.pairs(w[4:]); token = ('cell', w[2], w[3])
            obj = self.objects.setdefault(('cell', w[2]), {})
            N.require(row == {'status': 'unavailable'} or set(row) == {'status', 'value'} and row['status'] == 'observed', 'extra property schema')
            if row['status'] == 'unavailable':
                N.require(w[3] not in obj, 'conflicting property availability'); self.unavailable.add(token)
            else:
                N.require(token not in self.unavailable, 'conflicting property availability')
                self.store(obj, w[3], row['value'])
            return
        if not w or not w[0].startswith('admission_'): return super().add(w)
        N.require(self.headers and not self.seen_summary, 'admission record outside capture')
        self.counts[w[0]] += 1
        tag = w[0][10:]
        if tag == 'sources':
            self.sources.append(N.pairs(w[1:], ('helper_sha256', 'native_helper_sha256'))); return
        N.require(tag in FIELDS and len(w) >= 2 and w[1] in {str(i) for i in range(8)}, 'unknown admission tag or channel')
        row = N.pairs(w[2:], FIELDS[tag]); key = (w[1],)
        extra = {'interface': 'role', 'uncut_role': 'role', 'constant': 'output', 'cut': 'output', 'output': 'output', 'cell': 'cell', 'edge': 'input'}
        if tag in extra: key += (row[extra[tag]],)
        N.require(key not in self.admission[tag], 'duplicate admission record')
        self.admission[tag][key] = row


def net_cut(e, pc, role, row):
    name = interface_net(pc, role)
    N.require(row['kind'] == 'net' and row['name'] == name, 'admission interface name/role differs')
    e.selected('net', name)
    n = N.td(row['network'], ('scope', 'net', 'drivers', 'ports'))
    N.require(n['scope'] == 'selected_net_and_drivers' and n['net'] == name, 'admission cut observation scope')
    drivers, ports = N.names(n['drivers']), N.names(n['ports'])
    N.require(drivers == e.query('admission_net_drivers', name) and len(drivers) == 1 and
              ports == e.query('admission_net_ports', name) == [], 'admission cut driver/port inventory')
    driver = drivers[0]; e.pin(driver, 'OUT'); owner = e.query('pin_owner', driver)[0]
    N.require(e.prop('pin', driver, 'REF_PIN_NAME') == driver.rsplit('/', 1)[1], 'admission interface driver reference role differs')
    ref = e.prop('cell', owner, 'REF_NAME'); e.primitive(owner, ref)
    N.require(not owner.startswith(f'{N.ROOT}u_native{pc}/') or ref in ('GND', 'VCC'), 'internal nonconstant admission cut')
    return {'net': name, 'driver': driver, 'cell': owner, 'reference': ref,
            'inversion': e.prop('pin', driver, 'IS_INVERTED'),
            'clocks': e.query('admission_interface_clocks', pc, role)}


def primitive_ports(reference):
    """Mandatory expanded primitive interfaces; no guessed RAMS32 role schema."""
    if reference == 'FDRE': return {'C', 'CE', 'D', 'R'}, {'Q'}
    if reference == 'RAMD32': return {'CLK', 'I', 'WE'} | {f'{role}{i}' for role in ('RADR', 'WADR') for i in range(5)}, {'O'}
    if re.fullmatch(r'LUT[1-6]', reference): return {f'I{i}' for i in range(int(reference[-1]))}, {'O'}
    if reference == 'LUT6_2': return {f'I{i}' for i in range(6)}, {'O5', 'O6'}
    if reference == 'CARRY8': return {'CI', 'CI_TOP'} | {f'{role}[{i}]' for role in ('DI', 'S') for i in range(8)}, {f'{role}[{i}]' for role in ('O', 'CO') for i in range(8)}
    if reference in ('MUXF7', 'MUXF8', 'MUXF9'): return {'I0', 'I1', 'S'}, {'O'}
    raise ValueError('unreviewed admission primitive interface: ' + reference)


def facts(e):
    a = e.admission; pcs = {str(i) for i in range(8)}
    N.require(set(a['interface']) == {(p, role) for p in pcs for role in LIVE_ROLES}, 'live admission role inventory')
    N.require(set(a['uncut_role']) == {(p, role) for p in pcs for role in UNCUT_ROLES}, 'uncut admission role inventory')
    N.require(set(a['root']) == set(a['walk']) == set(a['cut_inventory']) == {(p,) for p in pcs}, 'admission channel inventory')
    expected = {tag: set() for tag in ('edge', 'cell', 'output', 'constant', 'cut')}
    channels = {}
    for pc in sorted(pcs):
        scope = f'{N.ROOT}u_native{pc}/'
        interfaces = {role: net_cut(e, pc, role, a['interface'][pc, role]) for role in LIVE_ROLES}
        for role in UNCUT_ROLES:
            row = a['uncut_role'][pc, role]
            N.require(row['kind'] == 'net' and row['name'] == interface_net(pc, role) and
                      row['disposition'] == 'no_assumed_input_or_constant' and
                      N.names(row['observed']) == e.query('admission_uncut_selection', pc, role) == [], 'uncut role was retained or assigned a value')
        cuts = {}
        for role, info in interfaces.items(): cuts.setdefault(info['driver'], []).append(role)
        inventory = a['cut_inventory'][pc,]
        actual_cuts = {k: N.names(v) for k, v in N.td(inventory['outputs']).items()}
        N.require(inventory['complete'] == '1' and actual_cuts == cuts, 'admission cut inventory disagrees with exact roles')
        root = a['root'][pc,]; gate = N.ROOT + 'u_mem/reads[7]_i_1' + ('' if pc == '0' else '__' + str(int(pc)-1))
        N.require(root['gate'] == gate and root['input'] == gate+'/I1', 'admission root anchor differs')
        e.primitive(gate, 'LUT2'); driver = e.input(e.input_record(root['network']), gate+'/I1')
        e.pin(root['input'], 'IN')
        N.require(e.prop('pin', root['input'], 'REF_PIN_NAME') == 'I1', 'admission root reference role differs')
        walk = a['walk'][pc,]; observed = set(N.names(walk['outputs']))
        N.require(walk['root'] == driver and driver in observed and 0 < len(observed) <= 8192, 'admission walk root/node bound')
        nodes = {}; expanded = set(); registers = set(); memories = set()
        for output in sorted(observed):
            e.pin(output, 'OUT'); owner = e.query('pin_owner', output)[0]
            ref = e.prop('cell', owner, 'REF_NAME'); e.primitive(owner, ref)
            props = dict(e.objects['cell', owner]); pin_props = dict(e.objects['pin', output])
            node = {'cell': owner, 'reference': ref, 'properties': props, 'output_properties': pin_props, 'inputs': {},
                    'unavailable_cell_properties': sorted(key for kind, name, key in e.unavailable if kind == 'cell' and name == owner)}
            if ref in ('GND', 'VCC'):
                N.require(e.prop('pin', output, 'REF_PIN_NAME') == ('G' if ref == 'GND' else 'P') and e.prop('pin', output, 'IS_INVERTED') in ('0', 'false', 'FALSE'), 'admission constant polarity/role')
                expected['constant'].add((pc, output))
                N.require(a['constant'].get((pc, output)) == {'output': output, 'cell': owner, 'reference': ref}, 'missing constant observation')
                node['kind'] = 'constant'
            elif output in cuts:
                expected['cut'].add((pc, output)); record = a['cut'].get((pc, output))
                N.require(record is not None and {k: record[k] for k in ('output', 'cell', 'reference')} == {'output': output, 'cell': owner, 'reference': ref} and N.names(record['roles']) == cuts[output], 'missing exact cut observation')
                node.update(kind='input', roles=cuts[output])
            else:
                seq = e.prop('cell', owner, 'IS_SEQUENTIAL').lower()
                N.require(seq in ('0', '1', 'false', 'true'), 'unknown state classification')
                sequential = seq in ('1', 'true')
                if sequential:
                    N.require(owner.startswith(scope) and ref in ('FDRE', 'RAMD32'), 'unreviewed admission state')
                    (registers if ref == 'FDRE' else memories).add(owner)
                    N.require(e.prop('cell', owner, 'INIT'), 'missing admission state initialization')
                else:
                    N.require(re.fullmatch(r'LUT[1-6]', ref) or ref in ('LUT6_2', 'CARRY8', 'MUXF7', 'MUXF8', 'MUXF9'), 'unsupported admission logic')
                node['kind'] = 'state' if sequential else 'logic'
                expanded.add(owner); expected['cell'].add((pc, owner)); record = a['cell'].get((pc, owner))
                N.require(record is not None and record['reference'] == ref and record['sequential'] == str(int(sequential)), 'admission cell witness differs')
                inputs, outputs = N.names(record['inputs']), N.names(record['outputs'])
                N.require(inputs and outputs and inputs == e.query('admission_inputs', pc, owner) and outputs == e.query('admission_outputs', pc, owner), 'admission pin query inventory differs')
                N.require(set(inputs) | set(outputs) == set(e.cells[owner]) and set(inputs).isdisjoint(outputs) and e.cells[owner] == e.query('cell_pins', owner), 'admission full cell pin inventory differs')
                expected_inputs, expected_outputs = primitive_ports(ref)
                N.require(set(inputs) == {owner+'/'+p for p in expected_inputs} and set(outputs) == {owner+'/'+p for p in expected_outputs}, 'admission mandatory primitive interface differs')
                N.require(output in outputs, 'admission output absent from cell inventory')
                for direction, pins in (('IN', inputs), ('OUT', outputs)):
                    for pin in pins:
                        e.pin(pin, direction)
                        N.require(e.query('pin_owner', pin) == [owner], 'admission cell pin owner differs')
                        N.require(e.prop('pin', pin, 'REF_PIN_NAME') == pin[len(owner)+1:], 'admission pin reference role differs')
                N.require(not sequential or set(outputs) <= observed, 'admission state output omitted')
                for pin in inputs:
                    key = (pc, pin); expected['edge'].add(key); edge = a['edge'].get(key)
                    N.require(edge is not None and edge['cell'] == owner, 'admission input edge omitted')
                    network = e.input_record(edge['network']); source = e.input(network, pin)
                    N.require(source in observed, 'admission dependency omitted')
                    node['inputs'][pin] = {'driver': source, 'network': network, 'properties': dict(e.objects['pin', pin])}
                expected['output'].add((pc, output)); record = a['output'].get((pc, output))
                N.require(record is not None, 'admission output network omitted')
                network = {k: N.names(v) for k, v in N.td(record['network'], ('nets', 'leaves', 'drivers', 'ports')).items()}
                N.require(network == A.full_network(e, 'pin', output) and network['drivers'] == [output], 'admission output network differs')
                node['selected_output_network'] = network
            nodes[output] = node
        reached = set(); queue = [driver]
        while queue:
            output = queue.pop()
            if output in reached: continue
            reached.add(output); node = nodes[output]
            queue.extend(row['driver'] for row in node['inputs'].values())
            if node['kind'] == 'state': queue.extend(p for p in observed if nodes[p]['cell'] == node['cell'])
        N.require(reached == observed, 'disconnected admission walk observation')
        if nodes[driver]['kind'] in ('logic', 'state'):
            N.require(root['input'] in nodes[driver]['selected_output_network']['leaves'], 'admission root delivery missing from observed source network')
        for node in nodes.values():
            for pin, edge in node['inputs'].items():
                source = nodes[edge['driver']]
                if source['kind'] in ('logic', 'state'):
                    N.require(pin in source['selected_output_network']['leaves'], 'admission input delivery missing from observed source network')
        N.require(set(N.names(walk['cells'])) == expanded and set(N.names(walk['registers'])) == registers and set(N.names(walk['memories'])) == memories, 'admission walk state/cell summary differs')
        channels[pc] = {'interfaces': interfaces, 'root': driver, 'nodes': nodes, 'registers': sorted(registers), 'memories': sorted(memories)}
    for tag, keys in expected.items(): N.require(set(a[tag]) == keys, 'extra admission ' + tag + ' observations')
    return {'channels': channels, 'cut_scope': '30 exact observed live input nets; no values assigned to five absent roles', 'output_scope': 'leaf inventory of selected output nets; no unobserved -segments closure claim'}


def review_artifact(*, topology_path, topology_sha256, checkpoint_sha256, helper_sha256, native_helper_sha256):
    result = {'schema': 1, 'kind': 'c1-admission-state-capture-review', 'capture_complete': False,
              'route_qualified': False, 'hardware_measured': False, 'functional_semantics_proven': False,
              'paired_preservation_proven': False, 'capture_scope': 'graph inventory only; functional properties and paired state proof pending',
              'issues': [], 'canonical_facts': {},
              'pending_reviews': ['take_read_admission_semantics', 'linked_final_accounting_preservation', 'admission_ram_state_address_control_preservation']}
    try:
        bound = N.identity(topology_path, topology_sha256); e = Evidence()
        for record in N.records(topology_path): e.add(record)
        N.require(N.identity(topology_path, topology_sha256) == bound, 'admission capture changed')
        N.require(len(e.headers) == len(e.summaries) == len(e.sources) == 1, 'admission capture envelope')
        h, s = e.headers[0], e.summaries[0]
        N.require(all(re.fullmatch('[0-9a-f]{64}', v or '') for v in (checkpoint_sha256, helper_sha256, native_helper_sha256)), 'invalid trusted admission identity')
        N.require(h['schema'] == '1' and h['kind'] == 'c1_accounting_boundary_topology' and h['checkpoint_sha256'] == checkpoint_sha256, 'admission schema/checkpoint differs')
        N.require(h['route_qualified'] == h['hardware_measured'] == s['route_qualified'] == s['hardware_measured'] == '0', 'admission capture claimed acceptance')
        N.require(e.sources[0] == {'helper_sha256': helper_sha256, 'native_helper_sha256': native_helper_sha256}, 'admission helper identity differs')
        N.require(N.number(s['queries']) == e.counts['query'] and s['incomplete_or_error'] == s['output_byte_cap'] == '0' and not e.issues and not e.selections, 'admission capture incomplete')
        canonical = facts(e); result.update(capture=bound, capture_complete=True, canonical_facts=canonical)
        result['facts_sha256'] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    except (ValueError, OSError, KeyError, TypeError, IndexError, RecursionError) as exc:
        result['issues'].append({'reason': str(exc)})
    return result
