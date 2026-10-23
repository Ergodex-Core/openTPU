"""C1 bounded raw LINK/routed ready comparison.

No route qualification. Exact observed sequential-pin identity is the boundary
assumption. Renamed state, changed sink identities and candidate inventory changes
remain unresolved; neither names nor counts establish replica equivalence.
"""
from __future__ import annotations
import hashlib
import json
import re
from boards.c1_f2_alex import native_review as N, ready_review as R


def bit(value):
    N.require(value in ('0', '1', 'false', 'true', 'FALSE', 'TRUE'), 'missing effective polarity')
    return int(value.lower() in ('1', 'true'))


def literal(value, width):
    match = re.fullmatch(r"([0-9]+)'([bBhH])([0-9a-fA-F]+)", value)
    N.require(match is not None and int(match[1]) == width, 'invalid INIT width')
    digits = match[3]
    N.require(match[2].lower() != 'b' or set(digits) <= {'0', '1'}, 'invalid binary INIT')
    number = int(digits, 2 if match[2].lower() == 'b' else 16)
    N.require(number < 1 << width, 'INIT exceeds width')
    return number


class BDD:
    """Reduced ordered Boolean DAG; exhaustive symbolic proof, bounded resources."""
    def __init__(self, max_nodes=100000):
        self.nodes = [None, None]
        self.unique = {}
        self.cache = {}
        self.variables = set()
        self.max_nodes = max_nodes

    def node(self, var, low, high):
        if low == high:
            return low
        key = (var, low, high)
        if key not in self.unique:
            N.require(len(self.nodes) < self.max_nodes, 'Boolean node bound')
            self.unique[key] = len(self.nodes)
            self.nodes.append(key)
        return self.unique[key]

    def var(self, name):
        self.variables.add(name)
        N.require(len(self.variables) <= 256, 'Boolean variable bound')
        return self.node(name, 0, 1)

    def ite(self, cond, yes, no):
        if cond == 1: return yes
        if cond == 0: return no
        if yes == no: return yes
        key = (cond, yes, no)
        if key in self.cache: return self.cache[key]
        N.require(len(self.cache) < self.max_nodes * 8, 'Boolean operation bound')
        var = min(self.nodes[n][0] for n in key if n > 1)
        def branch(n, high):
            if n > 1 and self.nodes[n][0] == var:
                return self.nodes[n][2 if high else 1]
            return n
        low = self.ite(*(branch(n, False) for n in key))
        high = self.ite(*(branch(n, True) for n in key))
        result = self.node(var, low, high)
        self.cache[key] = result
        return result

    def invert(self, value): return self.ite(value, 0, 1)

    def lut(self, inputs, init):
        def walk(index, offset):
            if index == len(inputs): return (init >> offset) & 1
            return self.ite(inputs[index], walk(index + 1, offset | (1 << index)),
                            walk(index + 1, offset))
        return walk(0, 0)


class Circuit:
    def __init__(self, evidence, facts, bdd):
        self.e, self.f, self.b = evidence, facts, bdd
        self.nodes, self.memo, self.active, self.boundaries = {}, {}, set(), {}
        self.omitted_sink_references = set()
        for graph in facts['graphs'].values():
            for output, node in graph['nodes'].items():
                N.require(output not in self.nodes or self.nodes[output] == node,
                          'conflicting circuit observations: ' + output)
                self.nodes[output] = node
        self.regs = {}
        for reg in facts['registers']:
            cell = reg['cell']
            normalized = {k: v for k, v in reg.items() if k != 'label'}
            N.require(cell not in self.regs or self.regs[cell] == normalized,
                      'conflicting register observations: ' + cell)
            self.regs[cell] = normalized

    def pin_bit(self, pin):
        return bit(self.e.prop('pin', pin, 'IS_INVERTED'))

    def value(self, output):
        if output in self.memo: return self.memo[output]
        N.require(output in self.nodes, 'missing Boolean node: ' + output)
        N.require(output not in self.active and len(self.active) < 128, 'Boolean cycle/depth bound')
        self.active.add(output)
        node = self.nodes[output]; cell = node['cell']
        ref = self.e.prop('cell', cell, 'REF_NAME')
        role = self.e.prop('pin', output, 'REF_PIN_NAME')
        if ref in ('GND', 'VCC'):
            N.require(role == ('G' if ref == 'GND' else 'P') and self.pin_bit(output) == 0,
                      'constant pin semantics')
            result = int(ref == 'VCC')
        elif re.fullmatch(r'LUT[1-6]', ref):
            N.require(role == 'O', 'unsupported LUT output')
            inputs = {}
            for pin, network in node['inputs'].items():
                role_in = self.e.prop('pin', pin, 'REF_PIN_NAME')
                N.require(role_in not in inputs, 'duplicate LUT input role')
                driver = self.e.input(network, pin)
                value = self.value(driver)
                inputs[role_in] = self.b.invert(value) if self.pin_bit(pin) else value
            count = int(ref[-1])
            N.require(set(inputs) == {f'I{k}' for k in range(count)}, 'LUT input roles')
            result = self.b.lut([inputs[f'I{k}'] for k in range(count)],
                                literal(self.e.prop('cell', cell, 'INIT'), 1 << count))
            if self.pin_bit(output): result = self.b.invert(result)
        else:
            N.require(self.e.prop('cell', cell, 'IS_SEQUENTIAL').lower() in ('1', 'true'),
                      'nonsequential Boolean boundary')
            self.boundaries[output] = {'reference': ref, 'role': role,
                                       'inversion': self.pin_bit(output)}
            result = self.b.var(output)
        self.active.remove(output); self.memo[output] = result
        return result

    def control(self, reg, role):
        record = self.regs[reg]['controls'][role]
        value = self.value(record['network']['drivers'][0])
        return self.b.invert(value) if bit(record['inversion']) else value

    def state(self, cell):
        reg = self.regs[cell]
        N.require(set(reg['outputs']) == {cell + '/Q'}, 'unsupported FDRE outputs')
        N.require(self.e.prop('pin', cell + '/Q', 'REF_PIN_NAME') == 'Q' and
                  self.pin_bit(cell + '/Q') == 0, 'FDRE output polarity')
        init = literal(self.e.prop('cell', cell, 'INIT'), 1)
        q = self.b.var(cell + '/Q')
        value = self.b.ite(self.control(cell, 'R'), 0,
                          self.b.ite(self.control(cell, 'CE'), self.control(cell, 'D'), q))
        clock = reg['controls']['C']
        N.require(clock['clocks'], 'missing FDRE clock identity')
        return {'init': init, 'next': value, 'clock_driver': clock['network']['drivers'],
                'clock_inversion': bit(clock['inversion']), 'clocks': sorted(clock['clocks'])}

    def sinks(self, network):
        # All complete observed consumers, including those outside the named bank.
        result = {}
        for pin in network['leaves']:
            if self.e.prop('pin', pin, 'DIRECTION') != 'IN': continue
            owner = self.e.query('pin_owner', pin)[0]
            reference = self.e.objects.get(('cell', owner), {}).get('REF_NAME')
            if not reference:
                self.omitted_sink_references.add(owner)
            # We compare signal delivery at exact typed input pins, not the
            # logic downstream of those pins. Optional owner REF_NAME is not
            # substituted with a guessed primitive from its synthesized name.
            result[pin] = {'owner': owner, 'observed_reference': reference,
                           'role': self.e.prop('pin', pin, 'REF_PIN_NAME'),
                           'inversion': self.pin_bit(pin)}
        return {'sinks': result, 'ports': sorted(network['ports'])}

    def boundary_deliveries(self):
        """Every captured outgoing edge not independently modeled in this cone."""
        internal = set()
        for node in self.nodes.values():
            internal.update(node.get('inputs', {}))
        for cell in self.regs:
            internal.update(cell + '/' + role for role in ('D', 'CE', 'R', 'C'))
        networks = list(self.f['roots'].values())
        networks += [node['output_network'] for node in self.nodes.values()
                     if 'output_network' in node]
        networks += [net for reg in self.regs.values() for net in reg['outputs'].values()]
        deliveries = {}
        for network in networks:
            N.require(len(network['drivers']) == 1, 'boundary driver ambiguity')
            function = self.value(network['drivers'][0])
            observed = self.sinks(network)
            for pin, identity in observed['sinks'].items():
                if pin in internal:
                    continue
                # Function is at the wire before the boundary pin's own inversion;
                # exact pin polarity is compared independently in identity.
                key = 'pin:' + pin
                row = {'identity': identity, 'function': function}
                N.require(key not in deliveries or deliveries[key] == row,
                          'conflicting captured boundary delivery: ' + pin)
                deliveries[key] = row
            for port in observed['ports']:
                key = 'port:' + port
                row = {'observed_port_name': port, 'function': function}
                N.require(key not in deliveries or deliveries[key] == row,
                          'conflicting captured boundary port: ' + port)
                deliveries[key] = row
        return deliveries


def compare(linked_e, linked_f, routed_e, routed_f):
    """Compare validated facts only; public review_pair supplies raw validation."""
    bdd = BDD()
    left, right = Circuit(linked_e, linked_f, bdd), Circuit(routed_e, routed_f, bdd)
    issues, checked = [], []
    def check(label, callback):
        try:
            callback(); checked.append(label)
        except (ValueError, KeyError, TypeError, RecursionError) as exc:
            issues.append({'scope': label, 'reason': str(exc)})
    def same(a, b, reason): N.require(a == b, reason)
    same(set(linked_f['roots']), set(routed_f['roots']), 'different root inventories')
    for label in sorted(linked_f['roots']):
        ln, rn = linked_f['roots'][label], routed_f['roots'][label]
        check('root_function:' + label, lambda ln=ln, rn=rn:
              same(left.value(ln['drivers'][0]), right.value(rn['drivers'][0]),
                   'ready root Boolean function differs'))
        check('root_sinks:' + label, lambda ln=ln, rn=rn:
              same(left.sinks(ln), right.sinks(rn), 'consumer identity/polarity changed; actual sink proof needed'))
    check('register_inventory', lambda:
          same(set(left.regs), set(right.regs), 'renamed/added/removed state requires transition and scope evidence'))
    for cell in sorted(set(left.regs) & set(right.regs)):
        check('transition:' + cell, lambda cell=cell:
              same(left.state(cell), right.state(cell), 'FDRE initial/next-state/clock differs'))
        check('register_sinks:' + cell, lambda cell=cell:
              same({p: left.sinks(n) for p,n in left.regs[cell]['outputs'].items()},
                   {p: right.sinks(n) for p,n in right.regs[cell]['outputs'].items()},
                   'register output consumers differ; replica partition proof needed'))
    check('candidate_scope', lambda:
          same(linked_f['candidate_discovery'], routed_f['candidate_discovery'],
               'scoped FDRE inventory changed; named candidate filter does not close renamed replicas'))
    check('all_captured_boundary_deliveries', lambda:
          same(left.boundary_deliveries(), right.boundary_deliveries(),
               'captured intermediate/root/state boundary consumers or delivered functions differ'))
    check('sequential_boundaries', lambda:
          same(left.boundaries, right.boundaries, 'external sequential boundary identity/polarity differs'))
    return {'issues': issues, 'checked': checked, 'boolean_nodes': len(bdd.nodes),
            'boundary_assumption': 'Equal values at exactly matching observed external sequential output pins.',
            'omitted_sink_references': {side: {'count': len(c.omitted_sink_references),
                'names_sha256': hashlib.sha256(json.dumps(sorted(c.omitted_sink_references)).encode()).hexdigest()}
                for side,c in (('linked', left), ('routed', right))},
            'scope_limit': 'Only observed ready cones and delivery to exact input pins; downstream sink-cell functions are outside this proof. Renamed sinks/state remain unresolved. No general design equivalence.'}


def load(binding):
    reviewed = R.review_artifact(**binding)
    N.require(not reviewed['issues'], 'raw replay failed: ' + json.dumps(reviewed['issues']))
    before = N.identity(binding['topology_path'], binding['topology_sha256'])
    evidence = R.Evidence()
    for record in N.records(binding['topology_path']): evidence.add(record)
    N.require(N.identity(binding['topology_path'], binding['topology_sha256']) == before,
              'raw capture changed during pair replay')
    return evidence, reviewed['canonical_facts'], reviewed['capture']


def review_pair(*, linked, routed):
    result = {'kind': 'c1-ready-pair-preparation', 'route_qualified': False,
              'hardware_measured': False, 'issues': [], 'pending_reviews': []}
    try:
        N.require(linked['checkpoint_sha256'] != routed['checkpoint_sha256'],
                  'paired artifact requires distinct LINK and routed checkpoints')
        le, lf, li = load(linked); re_, rf, ri = load(routed)
        result.update(compare(le, lf, re_, rf))
        result['captures'] = {'linked': li, 'routed': ri}
        result['checkpoint_sha256'] = {key: value['checkpoint_sha256']
                                      for key,value in (('linked',linked),('routed',routed))}
        result['observed_ready_equivalent'] = not result['issues']
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as exc:
        result['issues'].append({'scope': 'pair_envelope', 'reason': str(exc)})
        result['observed_ready_equivalent'] = False
    result['pending_reviews'] = (['actual_ready_transformation_evidence'] if result['issues'] else [])
    result['pending_reviews'] += ['renamed_replica_scope_closure', 'C1_final_artifact_qualification']
    result['result_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True,
                                           separators=(',', ':')).encode()).hexdigest()
    return result
