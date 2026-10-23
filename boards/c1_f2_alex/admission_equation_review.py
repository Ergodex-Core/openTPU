"""Prove the observed admission equation under explicit current-state cuts.

This is not a proof of all RTL state transitions. The 24 stored command bits
are identified by their actual RAM I connections to authenticated address nets,
not inferred from RAM names. Eliminated RTL high/write bits and LINK/final
preservation remain obligations even when the combinational equation passes.
The public CARRY8 equations are the same ones used by accounting_review.py.
"""
from __future__ import annotations
import hashlib
import json
import re
from . import admission_review as A, native_review as N, ready_pair_review as P

RTL_SHA256 = '9219ade00e16bf1b3a200167bd90a44283a606842a6477ce75f9b3eae7e14f01'
EQUATION = ('!pending_valid && cn!=0 && writes==0 && !reads[7] && '
            '(pending_count==0 || (cq[23:0]==next_addr[23:0] && '
            '!next_addr[24] && cq[23:6]==pending_addr[23:6]))')
PENDING = ['source_elided_command_high_write_and_pending_high_bits',
           'native_state_transition_correspondence',
           'linked_final_accounting_preservation',
           'admission_ram_state_address_control_preservation']


def roles(node):
    result = {}
    for pin, edge in node['inputs'].items():
        p = edge['properties']; role = p['REF_PIN_NAME']
        N.require(p['NAME'] == pin and p['DIRECTION'] == 'IN' and
                  pin == node['cell'] + '/' + role and role not in result,
                  'admission equation input identity')
        P.bit(p['IS_INVERTED']); result[role] = edge
    return result


def output_role(output, node):
    p = node['output_properties']; role = p['REF_PIN_NAME']
    N.require(p['NAME'] == output and p['DIRECTION'] == 'OUT' and
              output == node['cell'] + '/' + role and
              node['properties']['NAME'] == node['cell'] and
              node['properties']['REF_NAME'] == node['reference'],
              'admission equation output identity')
    P.bit(p['IS_INVERTED']); return role


def register_names(pc):
    scope = N.ROOT + 'u_native' + str(pc) + '/'
    families = {'cn': range(5), 'cp': range(4), 'cr': range(4),
                'idle_count': range(3), 'next_addr': range(25),
                'pending_addr': range(6, 24), 'pending_count': range(5),
                'reads': range(8), 'writes': range(8)}
    return {scope + family + '_reg[' + str(bit) + ']'
            for family, bits in families.items() for bit in bits} | {scope + 'pending_valid_reg'}


class Circuit:
    """Bounded Boolean replay; only explicitly authenticated state/input cuts."""
    def __init__(self, nodes, cuts, bdd):
        self.nodes, self.cuts, self.b = nodes, cuts, bdd
        self.memo, self.active, self.reached = {}, set(), set()

    def edge(self, edge):
        value = self.value(edge['driver'])
        return self.b.invert(value) if P.bit(edge['properties']['IS_INVERTED']) else value

    def value(self, output):
        N.require(output in self.nodes, 'missing admission equation node')
        node = self.nodes[output]; role = output_role(output, node)
        if output in self.cuts:
            N.require(not P.bit(node['output_properties']['IS_INVERTED']), 'inverted admission cut')
            self.reached.add(output); return self.cuts[output]
        if output in self.memo: return self.memo[output]
        N.require(output not in self.active and len(self.active) < 128,
                  'admission equation cycle/depth bound')
        self.active.add(output); ref = node['reference']
        if ref in ('GND', 'VCC'):
            N.require(role == ('G' if ref == 'GND' else 'P') and
                      not P.bit(node['output_properties']['IS_INVERTED']), 'constant output semantics')
            result = int(ref == 'VCC')
        else:
            N.require(re.fullmatch(r'LUT[1-6]', ref) or ref == 'CARRY8',
                      'unsupported admission equation primitive/boundary: ' + ref)
            inputs = {key: self.edge(edge) for key, edge in roles(node).items()}
            if ref == 'CARRY8':
                N.require(set(inputs) == {'CI', 'CI_TOP'} |
                          {f'{name}[{bit}]' for name in ('DI', 'S') for bit in range(8)},
                          'carry input roles')
                mode = node['properties'].get('CARRY_TYPE')
                N.require(mode in ('SINGLE_CY8', 'DUAL_CY4'), 'unknown carry mode')
                N.require(mode != 'SINGLE_CY8' or inputs['CI_TOP'] == 0,
                          'single carry upper input is not GND')
                carry = inputs['CI']; outputs = {}
                for bit in range(8):
                    if bit == 4 and mode == 'DUAL_CY4': carry = inputs['CI_TOP']
                    select = inputs[f'S[{bit}]']
                    outputs[f'O[{bit}]'] = self.b.ite(select, self.b.invert(carry), carry)
                    carry = self.b.ite(select, carry, inputs[f'DI[{bit}]'])
                    outputs[f'CO[{bit}]'] = carry
                N.require(role in outputs, 'unsupported carry output')
                result = outputs[role]
            else:
                count = int(ref[-1])
                N.require(role == 'O' and set(inputs) == {f'I{i}' for i in range(count)},
                          'LUT input/output roles')
                result = self.b.lut([inputs[f'I{i}'] for i in range(count)],
                                    P.literal(node['properties']['INIT'], 1 << count))
            if P.bit(node['output_properties']['IS_INVERTED']): result = self.b.invert(result)
        self.active.remove(output); self.memo[output] = result
        return result


def _all(bdd, values):
    result = 1
    for value in values: result = bdd.ite(value, result, 0)
    return result


def _zero(bdd, values): return _all(bdd, (bdd.invert(x) for x in values))


def _equal(bdd, left, right):
    N.require(len(left) == len(right), 'equation vector width differs')
    return _all(bdd, (bdd.ite(x, y, bdd.invert(y)) for x, y in zip(left, right)))


def review_channel(pc, channel):
    """Internal semantic replay of one already authenticated canonical channel.

    Public callers use review_artifact; this helper does not authenticate JSON.
    Current register values are explicit independent assumptions. No register
    transition, unreachable-state invariant or renamed state is accepted here.
    """
    N.require(str(pc) in {str(i) for i in range(8)}, 'unknown admission equation PC')
    pc = str(pc); scope = N.ROOT + 'u_native' + pc + '/'
    nodes, interfaces = channel['nodes'], channel['interfaces']
    N.require(set(channel['registers']) == register_names(pc), 'native state inventory differs')
    N.require(set(interfaces) == set(A.LIVE_ROLES), 'admission equation interface inventory')
    b = P.BDD(); cuts = {}; state = {}; ram_bits = {}; ram_facts = {}
    # Controls precede interleaved address bits, keeping equality BDDs bounded.
    for reg in sorted(channel['registers']):
        output = reg + '/Q'; node = nodes[output]
        N.require(node['reference'] == 'FDRE' and node['kind'] == 'state' and
                  output_role(output, node) == 'Q' and
                  not P.bit(node['output_properties']['IS_INVERTED']), 'native state output semantics')
        N.require(P.literal(node['properties']['INIT'], 1) == 0, 'native state INIT differs')
        controls = roles(node)
        N.require(set(controls) == {'C', 'CE', 'D', 'R'}, 'native state pin roles')
        N.require(controls['C']['driver'] == interfaces['clk']['driver'] and
                  not P.bit(controls['C']['properties']['IS_INVERTED']), 'native state clock differs')
        label = reg[len(scope):].removesuffix('_reg')
        match = re.fullmatch(r'(next_addr|pending_addr)_reg\[([0-9]+)\]', reg[len(scope):])
        variable = (f'10-address-{int(match[2]):02}-{match[1]}' if match else '00-state-' + label)
        cuts[output] = b.var(variable); state[reg[len(scope):]] = cuts[output]
    for role, info in interfaces.items():
        output = info['driver']
        N.require(not P.bit(info['inversion']), 'inverted admission interface cut')
        if output not in nodes: continue  # Roles unused by the complete root graph.
        node = nodes[output]
        if node['reference'] in ('GND', 'VCC'): continue
        N.require(node['kind'] == 'input' and role in node['roles'], 'interface cut classification')
        if output not in cuts: cuts[output] = b.var('20-input-' + min(node['roles']))
    # Map each observed 32x1 RAM via its data driver, and prove that all address
    # bits use the same 16-entry read/write pointer and write-enable predicate.
    for memory in channel['memories']:
        output = memory + '/O'; node = nodes[output]
        N.require(node['reference'] == 'RAMD32' and node['kind'] == 'state' and
                  output_role(output, node) == 'O' and
                  not P.bit(node['output_properties']['IS_INVERTED']), 'unsupported command RAM output')
        N.require(P.literal(node['properties']['INIT'], 32) == 0 and
                  P.literal(node['properties']['IS_CLK_INVERTED'], 1) == 0, 'command RAM initialization/clock')
        inputs = roles(node)
        N.require(set(inputs) == {'I', 'CLK', 'WE'} |
                  {f'{name}{bit}' for name in ('RADR', 'WADR') for bit in range(5)},
                  'command RAM full pin roles')
        N.require(all(not P.bit(edge['properties']['IS_INVERTED']) for edge in inputs.values()),
                  'command RAM input polarity')
        matches = [bit for bit in range(24) if
                   inputs['I']['driver'] == interfaces[f'n_caddr[{bit}]']['driver']]
        N.require(len(matches) == 1 and matches[0] not in ram_bits, 'command RAM data-to-bit mapping')
        bit = matches[0]; ram_bits[bit] = output
        for address, family in (('RADR', 'cr'), ('WADR', 'cp')):
            for index in range(4):
                N.require(inputs[f'{address}{index}']['driver'] == scope + family + f'_reg[{index}]/Q',
                          'command RAM pointer bit mapping')
        N.require(inputs['CLK']['driver'] == interfaces['clk']['driver'], 'command RAM clock differs')
        cuts[output] = b.var(f'10-address-{bit:02}-cq')
        ram_facts[str(bit)] = {'cell': memory, 'output': output,
                              'address_role': f'n_caddr[{bit}]',
                              'inputs': {role: row['driver'] for role, row in inputs.items()},
                              'init': 0, 'clock_inversion': 0}
    N.require(set(ram_bits) == set(range(24)) and len(channel['memories']) == 24,
              'command RAM bit inventory')
    c = Circuit(nodes, cuts, b)
    q = lambda name: state[name + '_reg']
    vector = lambda name, indexes: [state[f'{name}_reg[{bit}]'] for bit in indexes]
    input_value = lambda role: c.value(interfaces[role]['driver'])
    pushc = _all(b, [input_value('n_cvalid'), b.invert(input_value('rst')),
                     b.invert(vector('cn', [4])[0])])
    for bit, output in ram_bits.items():
        inputs = roles(nodes[output])
        N.require(c.edge(inputs['RADR4']) == c.edge(inputs['WADR4']) == 0,
                  'command RAM upper address is not observed GND')
        N.require(c.edge(inputs['WE']) == pushc, 'command RAM write-enable equation differs')
    stored = [cuts[ram_bits[i]] for i in range(24)]
    adjacent = _all(b, [_equal(b, stored, vector('next_addr', range(24))),
                       b.invert(vector('next_addr', [24])[0]),
                       _equal(b, stored[6:], vector('pending_addr', range(6, 24)))])
    expected = _all(b, [b.invert(q('pending_valid')),
                       b.invert(_zero(b, vector('cn', range(5)))),
                       _zero(b, vector('writes', range(8))),
                       b.invert(vector('reads', [7])[0]),
                       b.ite(_zero(b, vector('pending_count', range(5))), 1, adjacent)])
    actual = c.value(channel['root'])
    N.require(actual == expected, 'observed take_read equation differs')
    return {'root': channel['root'], 'equation': EQUATION,
            'equation_proven_for_all_current_state_assignments': True,
            'ram_write_enable_equation': 'n_cvalid && !rst && cn<16',
            'ram_bit_mapping': ram_facts,
            'state_output_assumptions': sorted(reg + '/Q' for reg in channel['registers']),
            'ram_read_output_assumptions': [ram_bits[i] for i in range(24)],
            'reached_cuts': sorted(c.reached), 'boolean_nodes': len(b.nodes),
            'boolean_variables': sorted(b.variables),
            'state_transition_correspondence_proven': False,
            'source_elided_bits_proven': False, 'paired_preservation_proven': False}


def review_artifact(**binding):
    """Authenticate raw bytes and independently derive the bounded equation."""
    result = {'schema': 1, 'kind': 'c1-admission-equation-review',
              'observed_equation_proven': False, 'source_semantics_proven': False,
              'paired_preservation_proven': False, 'route_qualified': False,
              'hardware_measured': False, 'issues': [], 'canonical_facts': {},
              'pending_reviews': list(PENDING)}
    try:
        graph = A.review_artifact(**binding)
        N.require(graph.get('capture_complete') is True and graph.get('issues') == [],
                  'authenticated admission graph incomplete: ' + str(graph.get('issues')))
        channels = graph['canonical_facts']['channels']
        N.require(set(channels) == {str(i) for i in range(8)}, 'admission equation channel inventory')
        canonical = {'rtl_sha256': RTL_SHA256, 'graph_facts_sha256': graph['facts_sha256'],
                     'channels': {pc: review_channel(pc, channels[pc]) for pc in sorted(channels)}}
        result.update(observed_equation_proven=True, capture=graph['capture'], canonical_facts=canonical)
        result['facts_sha256'] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    except (ValueError, OSError, KeyError, TypeError, IndexError, RecursionError) as exc:
        result['issues'].append({'reason': str(exc)})
    return result
