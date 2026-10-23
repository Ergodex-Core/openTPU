"""Replay the observed C1 eight-bit read-credit arithmetic, never qualification.

The exact gate-I1 driver is an explicit admission cut. Its RAM/state semantics
and LINK/final preservation remain separate obligations even when this proof
passes. Primitive equations follow public AMD UG974 CARRY8 and UG574 Carry Logic:
https://docs.amd.com/r/2023.2-English/ug974-vivado-ultrascale-libraries/CARRY8
https://docs.amd.com/r/en-US/ug574-ultrascale-clb/Carry-Logic
No installed vendor source is used or redistributed.
"""
from __future__ import annotations
import hashlib
import json
import re
from . import native_review as N, ready_pair_review as P


class Evidence(N.Evidence):
    def __init__(self):
        super().__init__()
        self.sources = []
        self.accounting = {k: {} for k in ('gate', 'gate_output', 'gate_input',
                          'registers', 'control', 'terminal', 'walk', 'edge')}

    def add(self, w):
        if len(w) >= 4 and w[0] == 'property' and w[1] == 'cell' and w[3] == 'CARRY_TYPE':
            N.require(self.headers and not self.seen_summary, 'property outside capture')
            self.counts['property'] += 1
            row = N.pairs(w[4:], ('status', 'value'))
            N.require(row['status'] == 'observed', 'carry type unavailable')
            self.store(self.objects.setdefault(('cell', w[2]), {}), 'CARRY_TYPE', row['value'])
            return
        if not w or not w[0].startswith('accounting_'):
            return super().add(w)
        N.require(self.headers and not self.seen_summary, 'accounting record outside capture')
        self.counts[w[0]] += 1
        tag = w[0][11:]
        if tag == 'sources':
            self.sources.append(N.pairs(w[1:], ('helper_sha256', 'native_helper_sha256')))
            return
        fields = {'gate': ('cell',), 'gate_output': ('network',),
                  'gate_input': ('role', 'network'), 'registers': ('cells',),
                  'control': ('cell', 'role', 'network'),
                  'terminal': ('output', 'cell', 'reference'), 'walk': ('outputs',),
                  'edge': ('cell', 'input', 'network')}
        N.require(tag in fields, 'unknown accounting record')
        row = N.pairs(w[2:], fields[tag]); key = w[1]
        if tag in ('gate_input', 'control', 'terminal', 'edge'):
            key = (key,) + tuple(row[x] for x in {
                'gate_input': ('role',), 'control': ('cell', 'role'),
                'terminal': ('cell', 'output'), 'edge': ('cell', 'input')}[tag])
        if 'network' in row:
            row['network'] = ({k: N.names(v) for k, v in N.td(row['network'],
                ('nets', 'leaves', 'drivers', 'ports')).items()} if tag == 'gate_output'
                else self.input_record(row['network']))
        for name in ('cells', 'outputs'):
            if name in row: row[name] = N.names(row[name])
        N.require(key not in self.accounting[tag], 'duplicate accounting record')
        self.accounting[tag][key] = row


def full_network(e, kind, name):
    """Authenticate every leaf and driver of one captured complete network."""
    e.selected(kind, name)
    row = e.networks.get((kind, name))
    N.require(row is not None and row['nets'], 'missing complete network')
    N.require(row['nets'] == ([name] if kind == 'net' else e.query('segments', name)),
              'network segment inventory differs')
    leaves, ports = set(), set()
    for net in row['nets']:
        e.selected('net', net)
        leaves.update(e.query('leaves', net)); ports.update(e.query('ports', net))
    N.require(set(row['leaves']) == leaves and set(row['ports']) == ports,
              'network leaf/port inventory differs')
    drivers = set()
    for pin in leaves:
        direction = e.prop('pin', pin, 'DIRECTION'); e.pin(pin, direction)
        N.require(direction in ('IN', 'OUT'), 'unknown leaf direction')
        if direction == 'OUT': drivers.add(pin)
    N.require(set(row['drivers']) == drivers and len(drivers) == 1 and not ports,
              'network has ambiguous driver or external port')
    return row


def facts(e):
    a = e.accounting; pcs = {str(i) for i in range(8)}
    N.require(set(a['gate']) == set(a['gate_output']) == set(a['registers']) == pcs,
              'missing/extra accounting PC inventory')
    N.require(set(a['gate_input']) == {(pc, p) for pc in pcs for p in ('I0', 'I1')},
              'missing/extra gate inputs')
    controls = set(); expected_walks = set(); channels = {}
    for pc in sorted(pcs):
        scope = N.ROOT + 'u_native' + pc
        regs = [scope + '/reads_reg[' + str(i) + ']' for i in range(8)]
        N.require(set(a['registers'][pc]['cells']) == set(regs) and
                  a['registers'][pc]['cells'] == e.query('accounting_register_inventory', pc),
                  'counter register inventory differs')
        gate = a['gate'][pc]['cell']; e.primitive(gate, 'LUT2')
        N.require(P.literal(e.prop('cell', gate, 'INIT'), 4) == 6, 'counter gate is not XOR')
        output = full_network(e, 'pin', gate + '/O')
        N.require(a['gate_output'][pc]['network'] == output and output['drivers'] == [gate + '/O'],
                  'gate output witness differs')
        N.require(set(output['leaves']) == {gate + '/O', *(r + '/CE' for r in regs)},
                  'counter gate has missing or unexplained consumer')
        roots = {}
        for role in ('I0', 'I1'):
            roots[role] = e.input(a['gate_input'][pc, role]['network'], gate + '/' + role)
            expected_walks.add(('gate', pc, role))
        valid = full_network(e, 'net', N.ROOT + 'n' + pc + '_rvalid')
        N.require(valid['drivers'] == [roots['I0']], 'gate I0 is not actual native RVALID')
        N.require(len(set(roots.values())) == 2, 'admission and RVALID cut alias')
        states = {}
        for reg in regs:
            e.primitive(reg, 'FDRE')
            N.require(P.literal(e.prop('cell', reg, 'INIT'), 1) == 0, 'counter INIT differs')
            e.pin(reg + '/Q', 'OUT')
            N.require(e.prop('pin', reg + '/Q', 'REF_PIN_NAME') == 'Q' and
                      P.bit(e.prop('pin', reg + '/Q', 'IS_INVERTED')) == 0, 'counter Q polarity')
            state = {}
            for role in ('C', 'CE', 'R', 'D'):
                key = (pc, reg, role); controls.add(key)
                N.require(key in a['control'], 'missing counter control')
                pin = reg + '/' + role
                driver = e.input(a['control'][key]['network'], pin)
                N.require(e.prop('pin', pin, 'REF_PIN_NAME') == role, 'counter control role')
                inversion = P.bit(e.prop('pin', pin, 'IS_INVERTED'))
                state[role] = {'driver': driver, 'inversion': inversion,
                               'clocks': e.query('pin_clock', pin)}
                if role != 'C': expected_walks.add(('control', pc, reg, role))
            N.require(state['C']['driver'] == N.ROOT + 'u_core_clk/inst/clkout1_buf/O' and
                      not state['C']['inversion'] and state['C']['clocks'], 'counter clock differs')
            N.require(state['R']['driver'] == N.ROOT + 'core_rst_reg/Q' and
                      not state['R']['inversion'], 'counter reset differs')
            N.require(state['CE']['driver'] == gate + '/O', 'counter CE disagrees with gate consumer inventory')
            states[reg] = state
        channels[pc] = {'registers': states, 'gate': gate, 'rvalid': roots['I0'],
                        'take_read_cut': roots['I1'], 'rvalid_network': valid,
                        'gate_output': output}
    N.require(set(a['control']) == controls, 'extra accounting controls')
    actual_walks = {tuple(N.tcl_list(label)): label for label in a['walk']}
    N.require(len(actual_walks) == len(a['walk']) and set(actual_walks) == expected_walks,
              'missing/extra accounting walks')
    nodes = {}; expected_edges = set(); expected_terminals = set()
    for label, walk in a['walk'].items():
        observed = set(walk['outputs']); N.require(0 < len(observed) <= 1024, 'walk node bound')
        parts = N.tcl_list(label)
        root = (a['gate_input'][parts[1], parts[2]] if parts[0] == 'gate' else
                a['control'][parts[1], parts[2], parts[3]])['network']['drivers'][0]
        N.require(root in observed, 'walk root omitted')
        for output in observed:
            e.pin(output, 'OUT'); cell = e.query('pin_owner', output)[0]
            ref = e.prop('cell', cell, 'REF_NAME'); e.primitive(cell, ref)
            N.require(e.cells.get(cell) == e.query('cell_pins', cell), 'cell pin inventory differs')
            node = {'cell': cell, 'reference': ref, 'inputs': {},
                    'properties': dict(e.objects['cell', cell]),
                    'output_properties': dict(e.objects['pin', output]),
                    'input_polarities': {}}
            if ref in ('GND', 'VCC') or e.prop('cell', cell, 'IS_SEQUENTIAL').lower() in ('1', 'true'):
                key = (label, cell, output); expected_terminals.add(key)
                N.require(a['terminal'].get(key) == {'output': output, 'cell': cell, 'reference': ref},
                          'missing terminal witness')
            else:
                N.require(re.fullmatch(r'LUT[1-6]', ref) or ref in ('CARRY8', 'LUT6_2', 'MUXF7', 'MUXF8', 'MUXF9'),
                          'unsupported observed combinational primitive')
                inputs = e.query('accounting_inputs', label, cell)
                N.require(set(inputs) == {p for p in e.cells[cell] if e.prop('pin', p, 'DIRECTION') == 'IN'},
                          'combinational input inventory differs')
                for pin in inputs:
                    key = (label, cell, pin); expected_edges.add(key)
                    N.require(key in a['edge'], 'missing accounting edge')
                    network = a['edge'][key]['network']; driver = e.input(network, pin)
                    N.require(driver in observed, 'walk driver omitted')
                    node['inputs'][pin] = network
                    node['input_polarities'][pin] = P.bit(e.prop('pin', pin, 'IS_INVERTED'))
            N.require(output not in nodes or nodes[output] == node, 'conflicting accounting graph')
            nodes[output] = node
    N.require(set(a['edge']) == expected_edges and set(a['terminal']) == expected_terminals,
              'extra accounting graph records')
    return {'channels': channels, 'nodes': nodes}


class Truth:
    """All 2^11 assignments: eight count bits, admission, RVALID and reset."""
    count = 1 << 11
    mask = (1 << count) - 1

    @classmethod
    def variable(cls, index):
        N.require(0 <= index < 11, 'truth variable bound')
        return sum(1 << row for row in range(cls.count) if (row >> index) & 1)

    @classmethod
    def choose(cls, select, yes, no):
        return (select & yes) | ((cls.mask ^ select) & no)

    @classmethod
    def lut(cls, inputs, init):
        N.require(1 <= len(inputs) <= 6 and 0 <= init < (1 << (1 << len(inputs))), 'LUT truth bound')
        result = 0
        for row in range(1 << len(inputs)):
            if not (init >> row) & 1: continue
            product = cls.mask
            for bit, value in enumerate(inputs): product &= value if (row >> bit) & 1 else cls.mask ^ value
            result |= product
        return result


def carry8(values, mode):
    N.require(mode in ('SINGLE_CY8', 'DUAL_CY4'), 'unknown carry mode')
    expected = {'CI', 'CI_TOP'} | {f'{role}[{i}]' for role in ('DI', 'S') for i in range(8)}
    N.require(set(values) == expected, 'carry input roles')
    N.require(mode != 'SINGLE_CY8' or values['CI_TOP'] == 0, 'single carry upper input is not GND')
    carry = values['CI']; outputs = {}
    for bit in range(8):
        if bit == 4 and mode == 'DUAL_CY4': carry = values['CI_TOP']
        select = values[f'S[{bit}]']
        outputs[f'O[{bit}]'] = select ^ carry
        carry = Truth.choose(select, carry, values[f'DI[{bit}]'])
        outputs[f'CO[{bit}]'] = carry
    return outputs


class Circuit:
    def __init__(self, e, nodes, cuts):
        self.e, self.nodes, self.cuts = e, nodes, cuts
        self.memo = {}; self.active = set(); self.reached_cuts = set()

    def value(self, output):
        if output in self.cuts:
            self.reached_cuts.add(output); return self.cuts[output]
        if output in self.memo: return self.memo[output]
        N.require(output in self.nodes, 'missing arithmetic node')
        N.require(output not in self.active and len(self.active) < 128, 'arithmetic cycle/depth bound')
        self.active.add(output)
        node = self.nodes[output]; cell = node['cell']; ref = node['reference']
        role = self.e.prop('pin', output, 'REF_PIN_NAME')
        if ref in ('GND', 'VCC'):
            N.require(role == ('G' if ref == 'GND' else 'P'), 'constant role')
            result = Truth.mask if ref == 'VCC' else 0
        else:
            # Only explicit cuts may be unconstrained state. In particular,
            # IS_SEQUENTIAL RAMD32 is never silently assigned a free variable.
            N.require(re.fullmatch(r'LUT[1-6]', ref) or ref == 'CARRY8',
                      'unsupported arithmetic primitive/boundary: ' + ref)
            inputs = {}
            for pin, network in node['inputs'].items():
                input_role = self.e.prop('pin', pin, 'REF_PIN_NAME')
                N.require(input_role not in inputs, 'duplicate primitive input role')
                value = self.value(self.e.input(network, pin))
                inputs[input_role] = Truth.mask ^ value if P.bit(self.e.prop('pin', pin, 'IS_INVERTED')) else value
            if ref == 'CARRY8':
                outputs = carry8(inputs, self.e.prop('cell', cell, 'CARRY_TYPE'))
                N.require(role in outputs, 'unknown carry output role'); result = outputs[role]
            else:
                count = int(ref[-1]); N.require(role == 'O', 'unsupported LUT output')
                N.require(set(inputs) == {f'I{i}' for i in range(count)}, 'LUT input roles')
                result = Truth.lut([inputs[f'I{i}'] for i in range(count)],
                                   P.literal(self.e.prop('cell', cell, 'INIT'), 1 << count))
        if P.bit(self.e.prop('pin', output, 'IS_INVERTED')): result ^= Truth.mask
        self.active.remove(output); self.memo[output] = result
        return result


def prove(e, canonical):
    proofs = []
    expected = [sum(1 << row for row in range(Truth.count)
                    if not (row >> 10) & 1 and (((row & 255) + ((row >> 8) & 1) -
                         ((row >> 9) & 1)) & 255) >> bit & 1) for bit in range(8)]
    for pc, channel in sorted(canonical['channels'].items()):
        regs = list(channel['registers'])
        cuts = {reg + '/Q': Truth.variable(bit) for bit, reg in enumerate(regs)}
        take, valid, reset = channel['take_read_cut'], channel['rvalid'], N.ROOT + 'core_rst_reg/Q'
        N.require(len(set(cuts) | {take, valid, reset}) == 11, 'arithmetic cuts alias')
        cuts.update({take: Truth.variable(8), valid: Truth.variable(9), reset: Truth.variable(10)})
        circuit = Circuit(e, canonical['nodes'], cuts)
        for bit, (reg, state) in enumerate(channel['registers'].items()):
            values = {}
            for role in ('D', 'CE', 'R'):
                value = circuit.value(state[role]['driver'])
                values[role] = Truth.mask ^ value if state[role]['inversion'] else value
            N.require(values['CE'] == cuts[take] ^ cuts[valid], 'counter enable arithmetic differs')
            actual = Truth.choose(values['R'], 0, Truth.choose(values['CE'], values['D'], cuts[reg + '/Q']))
            N.require(actual == expected[bit], 'counter next-state arithmetic differs: ' + reg)
        proofs.append({'pc': int(pc), 'assignments': Truth.count, 'counter_bits': 8,
                       'take_read_cut': take, 'rvalid_cut': valid, 'reset_cut': reset,
                       'equation': 'reset ? 0 : (reads + take_read - RVALID) modulo 256',
                       'evaluated_outputs': sorted(circuit.memo), 'cuts': sorted(cuts)})
    return proofs


def review_artifact(*, topology_path, topology_sha256, checkpoint_sha256,
                    helper_sha256, native_helper_sha256):
    result = {'schema': 1, 'kind': 'c1-read-accounting-arithmetic-review',
              'route_qualified': False, 'hardware_measured': False, 'arithmetic_proven': False,
              'issues': [], 'canonical_facts': {},
              'pending_reviews': ['take_read_admission_semantics', 'linked_final_accounting_preservation',
                                  'admission_ram_state_address_control_preservation']}
    try:
        bound = N.identity(topology_path, topology_sha256); e = Evidence()
        for record in N.records(topology_path): e.add(record)
        N.require(N.identity(topology_path, topology_sha256) == bound, 'capture changed')
        N.require(len(e.headers) == len(e.summaries) == len(e.sources) == 1, 'capture envelope')
        h, s = e.headers[0], e.summaries[0]
        N.require(all(re.fullmatch('[0-9a-f]{64}', x or '') for x in
                      (checkpoint_sha256, helper_sha256, native_helper_sha256)), 'invalid trusted identity')
        N.require(h['schema'] == '1' and h['kind'] == 'c1_read_accounting_topology' and
                  h['checkpoint_sha256'] == checkpoint_sha256, 'capture schema/checkpoint differs')
        N.require(h['route_qualified'] == h['hardware_measured'] == s['route_qualified'] ==
                  s['hardware_measured'] == '0', 'capture claimed acceptance')
        N.require(e.sources[0] == {'helper_sha256': helper_sha256, 'native_helper_sha256': native_helper_sha256},
                  'capture helper identity differs')
        N.require(N.number(s['queries']) == e.counts['query'] and s['incomplete_or_error'] == s['output_byte_cap'] == '0',
                  'capture incomplete')
        N.require(not e.issues and not e.selections, 'capture has conflicts/errors')
        canonical = facts(e); proofs = prove(e, canonical)
        result.update(capture=bound, canonical_facts=canonical, arithmetic_proven=True, proofs=proofs)
        result['facts_sha256'] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as exc:
        result['issues'].append({'reason': str(exc)})
    return result
