"""Conservative C1 admission-state correspondence from two authenticated captures.

This compares exact primitive expressions, not approximate Boolean similarity.
FDRE names propose a state relation; INIT and every control must also match.
RAM read symbols bind opaque matched memory state and recursively proved read
addresses, never arbitrary output terminals. Every RAM input is compared before
accepting that relation. Source-specialization and whole-artifact qualification
remain separate obligations. Unsupported rewrites fail closed.
"""
from __future__ import annotations
import hashlib
import json
import re
from . import admission_review as R, native_review as N, ready_pair_review as P


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def polarity(node, role, edge):
    """Observed effective pin polarity; empty cell properties are not defaults."""
    result = P.bit(edge['properties'].get('IS_INVERTED'))
    property_name = {'C': 'IS_C_INVERTED', 'D': 'IS_D_INVERTED', 'R': 'IS_R_INVERTED', 'CLK': 'IS_CLK_INVERTED'}.get(role)
    if property_name:
        value = node['properties'].get(property_name)
        if node['reference'] == 'RAMD32' and role == 'CLK':
            N.require(value not in (None, ''), 'incomplete functional RAM clock configuration')
        if value not in (None, ''):
            observed = P.literal(value, 1) if "'" in value else P.bit(value)
            N.require(observed == result, 'cell/pin polarity disagreement')
    return result


class Circuit:
    def __init__(self, channel):
        self.channel = channel
        self.nodes = channel['nodes']
        self.memo, self.active = {}, set()
        self.state = {}
        for output, node in self.nodes.items():
            if node['kind'] != 'state': continue
            name = node['cell']
            N.require(name not in self.state, 'unsupported multi-output admission state')
            self.state[name] = (output, node)
        N.require(set(self.state) == set(channel['registers']) | set(channel['memories']), 'paired state inventory differs')
        for name, (output, node) in self.state.items():
            ref = node['reference']; role = node['output_properties']['REF_PIN_NAME']
            N.require((ref, role) in (('FDRE', 'Q'), ('RAMD32', 'O')), 'unsupported paired state primitive/output')
            N.require(P.bit(node['output_properties'].get('IS_INVERTED')) == 0, 'unsupported paired state output inversion')
            self.inputs(node)  # Validate every role/polarity before symbolic cuts.
            P.literal(node['properties'].get('INIT', ''), 1 if ref == 'FDRE' else 32)

    def inputs(self, node):
        result = {}
        for pin, edge in node['inputs'].items():
            role = edge['properties'].get('REF_PIN_NAME')
            N.require(role and pin == node['cell']+'/'+role and role not in result, 'invalid paired input role')
            result[role] = (edge, polarity(node, role, edge))
        required, _ = R.primitive_ports(node['reference'])
        N.require(set(result) == required, 'incomplete paired primitive input interface')
        return result

    def input_value(self, node, role):
        edge, inversion = self.inputs(node)[role]
        value = self.value(edge['driver'])
        return digest(['invert', value]) if inversion else value

    def value(self, output):
        if output in self.memo: return self.memo[output]
        N.require(output in self.nodes, 'missing paired expression node')
        N.require(output not in self.active and len(self.active) < 128, 'asynchronous address/combinational cycle or depth bound')
        N.require(len(self.memo) < 8192, 'paired expression node bound')
        self.active.add(output)
        node = self.nodes[output]; ref = node['reference']; props = node['properties']
        role = node['output_properties']['REF_PIN_NAME']
        inverted = P.bit(node['output_properties'].get('IS_INVERTED'))
        if node['kind'] == 'constant':
            N.require((ref, role) in (('GND', 'G'), ('VCC', 'P')) and not inverted, 'paired constant role/polarity')
            result = digest(['constant', int(ref == 'VCC')])
        elif node['kind'] == 'input':
            # The variable denotes the wire at the observed interface driver.
            # Its role aliases, source polarity and clock identity are compared
            # separately, before any expression is accepted.
            result = digest(['interface', sorted(node['roles'])])
        elif ref == 'FDRE':
            N.require(node['kind'] == 'state' and role == 'Q' and not inverted, 'unmatched FDRE boundary')
            result = digest(['fdre_state', node['cell']])
        elif ref == 'RAMD32':
            N.require(node['kind'] == 'state' and role == 'O' and not inverted, 'unmatched RAM read boundary')
            # The read address must be an actual acyclic expression. Writes are
            # compared separately, using these same address-dependent reads.
            address = [self.input_value(node, f'RADR{i}') for i in range(5)]
            result = digest(['ram_read', node['cell'], address])
        else:
            N.require(node['kind'] == 'logic', 'unsupported expression boundary')
            if re.fullmatch(r'LUT[1-6]', ref):
                config = {'INIT': P.literal(props.get('INIT', ''), 1 << int(ref[-1]))}
            elif ref == 'LUT6_2':
                config = {'INIT': P.literal(props.get('INIT', ''), 64)}
            elif ref == 'CARRY8':
                mode = props.get('CARRY_TYPE')
                N.require(mode in ('SINGLE_CY8', 'DUAL_CY4'), 'incomplete functional carry configuration')
                config = {'CARRY_TYPE': mode}
            elif ref in ('MUXF7', 'MUXF8', 'MUXF9'): config = {}
            else: raise ValueError('unsupported exact primitive expression: '+ref)
            _, outputs = R.primitive_ports(ref)
            N.require(role in outputs, 'unsupported expression output role')
            inputs = [(name, self.input_value(node, name)) for name in sorted(self.inputs(node))]
            result = digest(['primitive', ref, role, config, inputs, inverted])
        self.active.remove(output); self.memo[output] = result
        return result

    def state_interface(self, name):
        output, node = self.state[name]; ref = node['reference']
        ports = self.inputs(node)
        clock_roles = sorted(role for role, (edge, _) in ports.items() if P.bit(edge['properties'].get('IS_CLOCK')))
        N.require(clock_roles == (['C'] if ref == 'FDRE' else ['CLK']), 'paired state clock role markers differ')
        return {'reference': ref, 'output_role': node['output_properties']['REF_PIN_NAME'],
                'output_inversion': P.bit(node['output_properties']['IS_INVERTED']),
                'init': P.literal(node['properties'].get('INIT', ''), 1 if ref == 'FDRE' else 32),
                'inputs': {role: self.input_value(node, role) for role in sorted(ports)},
                'clock_inversion': ports['C' if ref == 'FDRE' else 'CLK'][1],
                'clock_roles': clock_roles}


def interfaces(channel):
    result = {}; aliases = {}
    for role, row in channel['interfaces'].items(): aliases.setdefault(row['driver'], []).append(role)
    for role, row in channel['interfaces'].items():
        ref = row['reference']
        result[role] = {'aliases': sorted(aliases[row['driver']]), 'inversion': P.bit(row['inversion']),
                        'constant': int(ref == 'VCC') if ref in ('GND', 'VCC') else None,
                        'clocks': sorted(row['clocks'])}
    N.require(result['clk']['clocks'], 'missing paired interface clock identity')
    return result


def compare(linked_facts, routed_facts):
    """Internal operation on validated facts; review_pair authenticates raw input."""
    left, right = linked_facts['channels'], routed_facts['channels']
    N.require(set(left) == set(right) == {str(i) for i in range(8)}, 'paired admission channel inventory')
    checked = []
    for pc in sorted(left):
        a, b = Circuit(left[pc]), Circuit(right[pc])
        N.require(interfaces(left[pc]) == interfaces(right[pc]), 'paired admission external role/alias/polarity/clock differs')
        N.require(set(a.state) == set(b.state), 'renamed, removed or added admission state requires separate relation')
        signatures = {}
        for name in sorted(a.state):
            x, y = a.state_interface(name), b.state_interface(name)
            N.require(x == y, 'paired state initialization or complete input expression differs: '+name)
            signatures[name] = x
        x, y = a.value(left[pc]['root']), b.value(right[pc]['root'])
        N.require(x == y, 'paired admission root expression differs')
        # Do not leave apparently irrelevant, but captured, logic/configuration
        # unexamined. Both graph reviews already authenticated root reachability.
        for output in sorted(a.nodes): a.value(output)
        for output in sorted(b.nodes): b.value(output)
        checked.append({'pc': pc, 'registers': len(left[pc]['registers']), 'memories': len(left[pc]['memories']),
                        'state_interface_sha256': digest(signatures), 'root_expression_sha256': x,
                        'linked_expression_nodes': len(a.memo), 'routed_expression_nodes': len(b.memo)})
    return checked


def review_pair(*, linked, routed):
    result = {'schema': 1, 'kind': 'c1-admission-state-pair-review', 'route_qualified': False,
              'hardware_measured': False, 'paired_preservation_proven': False,
              'source_equation_proven': False, 'issues': [], 'checked': [],
              'interface_assumption': 'corresponding observed logical input wires have equal values; external producer equivalence remains a separate qualification obligation',
              'scope': 'exact matched FDRE state and opaque RAM state with read-address-dependent outputs; every state input and admission root expression compared',
              'pending_reviews': ['take_read_admission_semantics', 'linked_final_accounting_preservation', 'admission_ram_state_address_control_preservation']}
    try:
        N.require(linked['checkpoint_sha256'] != routed['checkpoint_sha256'], 'LINK and routed checkpoints must be distinct')
        l, r = R.review_artifact(**linked), R.review_artifact(**routed)
        result['linked'], result['routed'] = ({k:v for k,v in item.items() if k != 'canonical_facts'} for item in (l,r))
        N.require(l['capture_complete'] and r['capture_complete'] and not l['issues'] and not r['issues'], 'paired raw graph replay incomplete')
        result['checked'] = compare(l['canonical_facts'], r['canonical_facts'])
        result['paired_preservation_proven'] = True
        result['pending_reviews'].remove('admission_ram_state_address_control_preservation')
        result['facts_sha256'] = digest(result['checked'])
    except (ValueError, TypeError, KeyError, IndexError, RecursionError, OSError) as exc:
        result['issues'].append({'reason': str(exc)})
    return result
