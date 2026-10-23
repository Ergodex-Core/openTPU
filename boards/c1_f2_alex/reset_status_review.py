"""Pure C1 reset/status raw-evidence replay; never grants route acceptance.

The caller must authenticate every input via the immutable collection manifest.
A zero-error collector summary is necessary, not a topology or CDC disposition.
"""
from __future__ import annotations
from collections import Counter
from decimal import Decimal
import hashlib
import importlib.util
from pathlib import Path
import re
import stat
import sys


def _parser():
    spec = importlib.util.spec_from_file_location('c1_reset_status_tcl_parser', Path(__file__).with_name('gray_review.py'))
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


g = _parser()
ROOTS = ('WRAPPER/CL/u_rst_core', 'WRAPPER/CL/u_rst_hbm', 'WRAPPER/CL/u_status_cdc')
MAX_FILE = 256 << 20


def file_record(path):
    path = Path(path)
    g.require(not any(p.is_symlink() for p in (path, *path.parents)), 'Linked reset/status evidence')
    s = path.stat()
    g.require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and 0 < s.st_size <= MAX_FILE, 'Unsafe reset/status evidence')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': s.st_size, 'sha256': digest}


class Evidence:
    def __init__(self):
        self.objects = {}; self.queries = {}; self.groups = {}; self.timing = {}
        self.headers = []; self.summaries = []; self.coverage = []; self.issues = []
        self.endpoints = []; self.pulse = {}; self.counts = Counter()
        self.incomplete = []; self.repeated_root_properties = []; self.query_errors = 0
        self.inversions = {}

    def issue(self, reason, where=''):
        g.require(len(self.issues) < 100000, 'Reset/status issue limit')
        self.issues.append({'reason': reason, 'where': str(where)})

    def add(self, words):
        g.require(words, 'Empty reset/status record')
        tag = words[0]; self.counts[tag] += 1
        if tag == 'schema':
            self.headers.append(g.pairs(words))
        elif tag == 'topology_summary':
            self.summaries.append(g.pairs(words[1:]))
        elif tag == 'query':
            label = g.label(words[1]); value = g.pairs(words[2:])
            if value.get('status') != 'observed':
                self.query_errors += 1; self.issue('query_error', label); return
            g.require('value' in value, 'Query value missing')
            if label[:1] == ('reset_root',) and label[-2:-1] == ('property',):
                self.repeated_root_properties.append([list(label), value['value']])
                return
            if label in self.queries and self.queries[label] != value['value']:
                self.issue('conflicting_query', label)
            self.queries[label] = value['value']
        elif tag == 'object_properties':
            data = g.pairs(words[2:])
            g.require(set(data) == {'kind', 'name', 'values', 'review'}, 'Unknown object property schema')
            kind, name = data['kind'], data['name']
            g.require(kind in ('cell', 'pin', 'net', 'clock', 'port') and name, 'Missing typed object identity')
            values = g.td(data['values']); observed = self.objects.setdefault((kind, name), {})
            for key, result in values.items():
                item = g.td(result)
                if item.get('ok') != '1' or 'value' not in item:
                    self.issue('property_error', (kind, name, key)); continue
                value = item['value']
                if key == 'NAME' and value != name: self.issue('property_identity_mismatch', (kind, name))
                if key in observed and observed[key] != value: self.issue('conflicting_property', (kind, name, key))
                observed[key] = value
        elif tag == 'net_segment_group':
            data = g.pairs(words[2:]); segments = g.tcl_list(data['segments'])
            if not segments or len(segments) != len(set(segments)): self.issue('invalid_net_segment_group', words[1])
            if words[1] in self.groups: self.issue('duplicate_net_segment_group', words[1])
            self.groups[words[1]] = segments
        elif tag == 'discovered_reset_async_endpoints':
            self.endpoints.append(g.tcl_list(words[1]))
        elif tag == 'reset_timing_path':
            data = g.pairs(words[2:]); key = (words[1], data['delay'], data['user_ignored'])
            rows = [{'properties': g.td(g.td(row)['properties']),
                     'omitted_properties': g.tcl_list(g.td(row)['omitted_properties'])}
                    for row in g.tcl_list(data['paths'])]
            if key in self.timing: self.issue('duplicate_timing_witness', key)
            self.timing[key] = rows
        elif tag == 'incomplete':
            self.incomplete.append(words[1:])
            self.issue('incomplete_capture', words[1:])
        elif tag == 'inventory_coverage':
            data = g.pairs(words[2:]); self.coverage.append([words[1], data])
            if int(data['selected']) > int(data['cap']): self.issue('capture_cap', words[1])
        elif tag == 'collection_serialization':
            data = g.pairs(words[2:])
            if int(data['actual_count']) != int(data['serialized_count']): self.issue('collection_serialization_count', words[1])
            if data.get('representation') != 'expanded_per_object_names': self.issue('unreviewed_collection_serialization', words[1])
        elif tag == 'cell_inversion_availability':
            data = g.pairs(words[2:])
            g.require(set(data) == {'available','unavailable','interpretation'}, 'Unexpected inversion inventory fields')
            available, unavailable = g.tcl_list(data['available']), g.tcl_list(data['unavailable'])
            wanted = {'IS_C_INVERTED','IS_CLR_INVERTED','IS_PRE_INVERTED','IS_R_INVERTED','IS_S_INVERTED','IS_D_INVERTED','IS_CE_INVERTED'}
            if len(available) != len(set(available)) or len(unavailable) != len(set(unavailable)) or set(available) & set(unavailable) or set(available) | set(unavailable) != wanted:
                self.issue('inversion_inventory', words[1])
            if data.get('interpretation') != 'effective_pin_polarity_required_for_primitive_semantics': self.issue('inversion_interpretation', words[1])
            if words[1] in self.inversions: self.issue('duplicate_inversion_inventory', words[1])
            self.inversions[words[1]] = data
        elif tag == 'pulse_coverage':
            self.pulse[words[1]] = g.pairs(words[2:])
        elif tag not in {'clock_pin_fanout', 'boundary', 'reset_timing_coverage', 'reset_timing_absence', 'interpretation'}:
            self.issue('unknown_record_kind', tag)

    def observed(self, *label):
        value = self.queries.get(tuple(label))
        if value is None:
            self.issue('missing_query', label); return []
        values = g.tcl_list(value)
        if '...' in values: self.issue('abbreviated_collection', label)
        return values

    def properties(self, kind, name, required=()):
        values = self.objects.get((kind, name), {})
        if not values: self.issue('missing_properties', (kind, name))
        for key in required:
            if key not in values or values[key] == '': self.issue('missing_property', (kind, name, key))
        return values

    def pin_map(self, cell):
        result = {}
        for pin in self.observed('cell', cell, 'pins'):
            props = self.properties('pin', pin, ('NAME', 'REF_PIN_NAME', 'DIRECTION', 'IS_CLOCK'))
            ref = props.get('REF_PIN_NAME')
            if ref in result: self.issue('duplicate_reference_pin', (cell, ref))
            result[ref] = pin
        return result

    def net_for(self, pin):
        segments = self.observed('pin', pin, 'nets')
        matches = [key for key, values in self.groups.items() if set(values) == set(segments)]
        if not segments or len(matches) != 1:
            self.issue('missing_or_ambiguous_pin_net', pin); return None
        return matches[0]

    def drivers(self, pin, require_leaf=True):
        group = self.net_for(pin)
        if group is None: return []
        leaves = self.observed('net', group, 'leaf_pins')
        if require_leaf and pin not in leaves: self.issue('pin_absent_from_leaf_inventory', pin)
        result = self.observed('net', group, 'typed_output_pins')
        if len(result) != 1: self.issue('missing_or_multiple_driver', pin)
        if any(driver not in leaves for driver in result): self.issue('driver_absent_from_leaf_inventory', pin)
        for driver in result:
            if self.properties('pin', driver, ('NAME','DIRECTION')).get('DIRECTION') != 'OUT': self.issue('driver_direction', driver)
        return result

    def clock(self, pin):
        clocks = self.observed('pin', pin, 'clocks')
        if len(clocks) != 1:
            self.issue('missing_or_multiple_clock', pin); return None
        props = self.properties('clock', clocks[0], ('NAME', 'PERIOD'))
        try:
            period = Decimal(props['PERIOD'])
            g.require(period.is_finite() and period > 0, 'Invalid clock period')
        except (KeyError, ArithmeticError, ValueError):
            self.issue('invalid_clock_period', clocks[0]); return None
        return {'name': clocks[0], 'period': str(period)}

    def registers(self, root):
        discovered = self.observed('hierarchy_discovery', root)
        if not discovered or len(discovered) != len(set(discovered)):
            self.issue('missing_or_duplicate_hierarchy', root)
        result = {}
        for cell in discovered:
            if cell != root and not cell.startswith(root + '/'):
                self.issue('hierarchy_scope_escape', cell); continue
            props = self.properties('cell', cell, ('NAME', 'REF_NAME', 'IS_SEQUENTIAL'))
            if props.get('IS_SEQUENTIAL', '').lower() not in ('1', 'true'): continue
            pins = self.pin_map(cell)
            for required in ('D', 'Q', 'C'):
                if required not in pins: self.issue('missing_register_pin', (cell, required))
            if not {'D', 'Q', 'C'} <= set(pins): continue
            result[cell] = {'properties': props, 'pins': pins, 'clock': self.clock(pins['C']), 'drivers': self.drivers(pins['D'])}
        return result

    def chains(self, root, source_clock, destination_clock, width, depth):
        regs = self.registers(root)
        source = {cell: row for cell, row in regs.items() if row['clock'] and row['clock']['name'] == source_clock}
        dest = {cell: row for cell, row in regs.items() if row['clock'] and row['clock']['name'] == destination_clock}
        if len(source) != width or len(dest) != width * depth or len(regs) != width * (depth + 1):
            self.issue('status_register_clock_inventory', root)
        chains = []; used = set()
        for cell, row in sorted(source.items()):
            chain = []; driver = row['pins']['Q']
            for stage in range(depth):
                next_cells = [name for name, value in dest.items() if value['drivers'] == [driver]]
                if len(next_cells) != 1:
                    self.issue('status_chain_missing_or_branching', (cell, stage)); break
                name = next_cells[0]
                if name in used: self.issue('status_chain_cross_bit_or_cycle', name)
                used.add(name); chain.append(name); driver = dest[name]['pins']['Q']
                if dest[name]['properties'].get('ASYNC_REG', '').lower() not in ('1', 'true'):
                    self.issue('status_missing_async_reg', name)
            chains.append({'source': cell, 'stages': chain, 'terminal_q': driver})
        if used != set(dest): self.issue('status_uncovered_destination_registers', root)
        return {'registers': regs, 'chains': chains}

    def constant(self, pin, reference):
        drivers = self.drivers(pin)
        if len(drivers) != 1: return False
        driver = drivers[0]; group = self.net_for(pin)
        owners = self.observed('net', group, 'driver_owners', driver)
        if len(owners) != 1 or self.properties('cell', owners[0], ('REF_NAME',)).get('REF_NAME') != reference:
            self.issue('wrong_constant_driver', (pin, reference)); return False
        return True

    def reset_semantics(self, root, regs):
        source = [name for name,row in regs.items() if not any(row['drivers'] == [other['pins']['Q']] for other in regs.values())]
        chain = []; current = source[0] if len(source) == 1 else None
        if current is None: self.issue('reset_chain_source', root)
        while current is not None and current not in chain and len(chain) <= 4:
            chain.append(current); row = regs[current]
            if len(chain) == 1: self.constant(row['pins']['D'], 'VCC')
            if row['properties'].get('REF_NAME') != 'FDCE' or row['properties'].get('INIT') != "1'b1": self.issue('reset_primitive_or_init', current)
            if 'CE' not in row['pins'] or not self.constant(row['pins']['CE'], 'VCC'): self.issue('reset_enable', current)
            following = [name for name,value in regs.items() if value['drivers'] == [row['pins']['Q']]]
            if len(following) > 1: self.issue('reset_chain_branch', current)
            current = following[0] if len(following) == 1 else None
        if len(chain) != 4 or set(chain) != set(regs): self.issue('reset_chain_coverage', root)
        clear_drivers = {tuple(self.drivers(row['pins']['CLR'])) for row in regs.values() if 'CLR' in row['pins']}
        if len(clear_drivers) != 1 or len(next(iter(clear_drivers), ())) != 1:
            self.issue('reset_common_clear_gate', root); return {'chain': chain}
        driver = next(iter(clear_drivers))[0]; gate = driver.rsplit('/', 1)[0]
        n = 3 if root == ROOTS[0] else 2
        props = self.properties('cell', gate, ('REF_NAME','INIT')); pins = self.pin_map(gate)
        if props.get('REF_NAME') != 'LUT'+str(n) or props.get('INIT') != {2:"4'h8",3:"8'h80"}[n] or set(pins) != {'O', *(f'I{i}' for i in range(n))} or pins.get('O') != driver:
            self.issue('reset_gate_not_positive_and', gate)
        main = self.drivers('WRAPPER/CL/rst_main_n', require_leaf=False)
        expected = set(main + ['WRAPPER/CL/u_hbm_clk/inst/mmcme4_adv_inst/LOCKED'] + (['WRAPPER/CL/u_core_clk/inst/mmcme4_adv_inst/LOCKED'] if n == 3 else []))
        inputs = {ref: self.drivers(pin) for ref,pin in pins.items() if ref.startswith('I')}
        if len(main) != 1 or any(len(values)!=1 for values in inputs.values()) or {v[0] for v in inputs.values() if v} != expected:
            self.issue('reset_gate_input_mapping', gate)
        for locked in expected - set(main):
            if self.properties('cell', locked.rsplit('/',1)[0], ('REF_NAME',)).get('REF_NAME') != 'MMCME4_ADV': self.issue('reset_lock_source_primitive', locked)
        if chain:
            out = regs[chain[-1]]['pins']['Q']
            if self.net_for(out) != self.net_for(root+'/dest_arst'): self.issue('reset_output_not_last_stage', root)
        for row in regs.values():
            roots = [g.td(value) for value in self.observed('reset_fanin', row['pins']['CLR'], 'roots')]
            expected_roots = (expected - set(main)) | {value.rsplit('/',1)[0]+'/C' for value in main}
            if len(roots) != len(expected_roots) or {value.get('name') for value in roots} != expected_roots or any(value.get('kind') != 'pin' for value in roots): self.issue('reset_exact_fanin_roots', root)
            if self.net_for(row['pins']['CLR']) != self.net_for(root+'/src_arst'): self.issue('reset_clear_not_source_gate', root)
        return {'chain': chain, 'registers': regs, 'gate': {'cell': gate, 'reference': props.get('REF_NAME'), 'init': props.get('INIT'), 'inputs': inputs},
                'policy': 'active_low_async_assert_four_stage_sync_release_requires_cell_inversion_evidence'}

    def status_semantics(self, status):
        regs = status['registers']; facts = []
        outputs = self.observed('status_boundary_pins')
        for cell,row in regs.items():
            if row['properties'].get('REF_NAME') != 'FDRE' or row['properties'].get('INIT') != "1'b0": self.issue('status_primitive_or_init', cell)
            for ref,constant in [('CE','VCC'),('R','GND')]:
                if ref not in row['pins'] or not self.constant(row['pins'][ref], constant): self.issue('status_control_constant', (cell,ref))
        for chain in status['chains']:
            match = re.fullmatch(re.escape(ROOTS[2])+r'/src_ff_reg\[([0-9]+)\]', chain['source'])
            if not match: self.issue('status_source_bit_identity', chain['source']); continue
            bit = int(match[1]); output = f'WRAPPER/CL/cl_sh_status_vled[{bit}]' if bit < 3 else f'WRAPPER/CL/cl_sh_status2[{bit-3}]'
            q = chain['terminal_q']; group = self.net_for(q)
            if output not in outputs or group != self.net_for(output): self.issue('status_output_bit_mapping', bit)
            leaves = self.observed('net', group, 'leaf_pins') if group else []
            expected_leaf = f'WRAPPER/RL_SHIM/PIPE_VLED/pipe_reg[0][{bit}]/D' if bit < 3 else f'WRAPPER/RL_SHIM/PIPE_STATUS2/pipe_reg[0][{bit-3}]/D'
            if set(leaves) != {q, expected_leaf}: self.issue('status_output_has_unreviewed_control_fanout', bit)
            source_row = regs[chain['source']]
            if bit >= 3 and source_row['drivers'] != [f'WRAPPER/CL/u_native{bit-3}/n_err_reg/Q']: self.issue('status_error_bit_source', bit)
            for previous,following in zip([chain['source'],*chain['stages']], chain['stages']):
                qpin=regs[previous]['pins']['Q']; net=self.net_for(qpin)
                loads=self.observed('net',net,'leaf_pins') if net else []
                if set(loads)!={qpin,regs[following]['pins']['D']}: self.issue('status_intermediate_fanout', previous)
            facts.append({**chain,'bit':bit,'output':output,'source_driver':source_row['drivers'],'shell_load':expected_leaf})
        if sorted(row['bit'] for row in facts)!=list(range(11)): self.issue('status_bit_coverage')
        return {'chains': facts, 'registers': regs, 'semantics': 'eight_independent_native_error_bits_and_three_observational_indicators_no_word_coherence_claim'}

    def structural_review(self, checkpoint):
        if len(self.headers) != 1 or self.headers[0].get('kind') != 'c1_reset_status_diagnostic' or self.headers[0].get('checkpoint_sha256') != checkpoint or self.headers[0].get('schema') != '1' or self.headers[0].get('route_qualified') != '0':
            self.issue('checkpoint_binding')
        else:
            header = self.headers[0]
            if set(header) != {'schema','kind','checkpoint_sha256','collector_sha256','vivado_version','route_qualified','hardware_measured'} or header.get('vivado_version') != '2025.2' or header.get('hardware_measured') != '0' or not re.fullmatch(r'[0-9a-f]{64}',header.get('collector_sha256','')):
                self.issue('collector_source_or_version_binding')
        if len(self.summaries) != 1:
            self.issue('missing_or_duplicate_summary')
        else:
            summary = self.summaries[0]
            if int(summary.get('incomplete_or_error', '-1')) != self.query_errors + len(self.incomplete): self.issue('collector_error_count_mismatch')
            if summary.get('route_qualified') != '0': self.issue('unexpected_collector_qualification')
            if int(summary.get('queries', '-1')) != self.counts['query']: self.issue('query_count_mismatch')
        status = self.chains(ROOTS[2], 'clk_out1_f2_core_clk', 'clk_main_a0', 11, 4)
        resets = {}
        for root, clock in zip(ROOTS[:2], ('clk_out1_f2_core_clk', 'clk_out1_f2_hbm_clk')):
            regs = self.registers(root)
            if len(regs) != 4: self.issue('reset_stage_count', root)
            for name, row in regs.items():
                if not row['clock'] or row['clock']['name'] != clock: self.issue('reset_destination_clock', name)
                if row['properties'].get('ASYNC_REG', '').lower() not in ('1', 'true'): self.issue('reset_missing_async_reg', name)
                if not {'CLR', 'PRE'} & set(row['pins']): self.issue('reset_missing_async_control', name)
            resets[root] = regs
        # Every error/cap remains pending; no inherited artifact-specific allowances.
        reset_facts = {root: self.reset_semantics(root, regs) for root, regs in resets.items()}
        status_facts = self.status_semantics(status)
        expected_endpoints = sorted(pin for regs in resets.values() for row in regs.values() for ref, pin in row['pins'].items() if ref == 'CLR')
        if len(self.endpoints) != 1 or sorted(self.endpoints[0]) != expected_endpoints:
            self.issue('reset_endpoint_inventory')
        if set(self.timing) != {(pin, delay, ignored) for pin in expected_endpoints for delay in ('max', 'min') for ignored in ('0', '1')}:
            self.issue('reset_timing_inventory')
        for root, regs in resets.items():
            for row in regs.values():
                endpoint = row['pins'].get('CLR', '')
                roots = [g.td(x)['name'] for x in self.observed('reset_fanin', endpoint, 'roots')]
                for delay in ('max', 'min'):
                    for ignored in ('0', '1'):
                        paths = self.timing.get((endpoint, delay, ignored), [])
                        if len(paths) != 1: self.issue('reset_worst_path_missing', (endpoint, delay, ignored))
                        for path in paths:
                            props = path['properties']
                            if props.get('ENDPOINT_PIN') != endpoint or props.get('ENDPOINT_CLOCK') != row['clock']['name'] or props.get('STARTPOINT_PIN') not in roots:
                                self.issue('reset_path_endpoint_or_root', endpoint)
                            # XPM asynchronous assertion is intentionally excepted.
                            # Empty slack is recorded, never interpreted as +infinity.
                            if props.get('EXCEPTION') != 'False Path' or props.get('SLACK') != '' or props.get('REQUIREMENT') != '':
                                self.issue('unreviewed_reset_path_semantics', endpoint)
                            try:
                                delay_value = Decimal(props['DATAPATH_DELAY'])
                                if not delay_value.is_finite() or delay_value < 0: raise ValueError('Invalid delay')
                            except (KeyError, ArithmeticError, ValueError): self.issue('reset_path_delay', endpoint)
        return {'status': status_facts, 'resets': reset_facts, 'clock_detail_omissions': [],
                'timing': [[list(key), value] for key, value in sorted(self.timing.items())],
                'capture_counts': dict(self.counts), 'coverage_sha256': g.digest(self.coverage),
                'repeated_root_property_queries_sha256': g.digest(self.repeated_root_properties),
                'reset_glitch_freedom_asserted': False, 'coherent_status_word_asserted': False,
                'reset_timing_interpretation': 'vendor_async_assertion_false_paths_not_synchronous_recovery_slack'}



def pulse_review(path, root, registers):
    """Exact positive clock-pulse coverage; does not claim reset-event pulse width."""
    before = file_record(path); text = Path(path).read_text(); issues = []; rows = {}; clock = None
    expected = {row['pins']['C']: row['clock']['name'] for row in registers.values()}
    if '| Design State      : Routed' not in text or f'get_cells -quiet {root}' not in text:
        issues.append({'reason': 'pulse_scope_or_state', 'where': root})
    for line in text.splitlines():
        if line.startswith('Clock Name:'): clock = line.split(':', 1)[1].strip()
        if not re.match(r'^(Min Period|Low Pulse Width|High Pulse Width)\s{2,}', line):
            if re.split(r'\s{2,}', line.strip())[-1].startswith(root+'/'):
                issues.append({'reason':'unreviewed_pulse_check','where':line})
            continue
        fields = re.split(r'\s{2,}', line.strip())
        if len(fields) != 9:
            issues.append({'reason': 'malformed_pulse_row', 'where': line}); continue
        check, corner, lib, reference, required, actual, slack, loc, pin = fields
        key = (pin, check, corner)
        if key in rows or pin not in expected or clock != expected.get(pin) or reference != 'n/a':
            issues.append({'reason': 'pulse_endpoint_clock_or_duplicate', 'where': key})
        try:
            values = [Decimal(value) for value in (required, actual, slack)]
            if any(not value.is_finite() for value in values) or min(values) < 0 or abs(values[1]-values[0]-values[2]) > Decimal('.0015'):
                raise ValueError('Negative/inconsistent pulse row')
        except (ArithmeticError, ValueError): issues.append({'reason': 'pulse_value_failure', 'where': key})
        rows[key] = {'clock': clock, 'required': required, 'actual': actual, 'slack': slack, 'lib_pin': lib, 'location': loc}
    wanted = {(pin, check, corner) for pin in expected for check, corner in [('Min Period','n/a'),('Low Pulse Width','Slow'),('Low Pulse Width','Fast'),('High Pulse Width','Slow'),('High Pulse Width','Fast')]}
    if set(rows) != wanted: issues.append({'reason': 'pulse_exact_coverage', 'where': root})
    if re.search(r'VIOLATED|<hidden>', text): issues.append({'reason': 'pulse_hidden_or_violated', 'where': root})
    g.require(file_record(path) == before, 'Pulse report changed')
    return issues, {'input': before, 'rows': [[list(key), row] for key,row in sorted(rows.items())], 'reset_event_pulse_width_asserted': False}


def polarity_review(evidence, registers):
    """Use the same raw capture, with explicit effective pin polarity mandatory."""
    expected = {name: row for regs in registers.values() for name,row in regs.items()}
    issues = []; effective = {}; clock_drivers = {}
    def issue(reason,where): issues.append({'reason':reason,'where':str(where)})
    def bit(value):
        if value in ('0','false','FALSE',"1'b0"): return 0
        if value in ('1','true','TRUE',"1'b1"): return 1
        return None
    if not set(expected) <= set(evidence.inversions): issue('missing_inversion_inventory', sorted(set(expected)-set(evidence.inversions)))
    for root,regs in registers.items():
        for name,row in regs.items():
            effective[name] = {}
            props = evidence.properties('cell',name)
            availability = evidence.inversions.get(name,{})
            available = g.tcl_list(availability.get('available',''))
            inventory = evidence.observed('cell',name,'property_inventory')
            if any(key not in inventory for key in available): issue('inversion_property_inventory_mismatch',name)
            for ref,wanted in {'C':0,'D':0,'CE':0,'Q':0,**({'R':0} if root==ROOTS[2] else {'CLR':1})}.items():
                pin = row['pins'].get(ref,''); pinprops = evidence.properties('pin',pin,('NAME','REF_PIN_NAME','IS_INVERTED'))
                key = 'IS_'+ref+'_INVERTED'; cell_literal = props.get(key); pin_literal = pinprops.get('IS_INVERTED')
                cell_bit, pin_bit = bit(cell_literal),bit(pin_literal)
                if key in available and key not in props: issue('missing_available_cell_polarity',(name,key))
                if cell_literal not in (None,'') and cell_bit is None: issue('invalid_cell_polarity',(name,key))
                if cell_bit is not None and cell_bit != pin_bit: issue('cell_pin_polarity_conflict',(name,ref))
                if pin_bit != wanted: issue('missing_or_wrong_effective_polarity',(name,ref))
                if pinprops.get('NAME')!=pin or pinprops.get('REF_PIN_NAME')!=ref: issue('polarity_pin_identity',pin)
                effective[name][ref] = {'value':pin_bit,'cell_literal':cell_literal,'pin_literal':pin_literal}
            pin = row['pins'].get('C','')
            segments = evidence.observed('clock_pin',pin,'nets'); drivers = evidence.observed('clock_pin',pin,'drivers')
            if not segments or len(segments)!=len(set(segments)) or len(drivers)!=1: issue('clock_driver_inventory',pin)
            for driver in drivers:
                dp = evidence.properties('pin',driver,('NAME','DIRECTION','IS_CLOCK'))
                owners = evidence.observed('clock_pin',pin,'owner',driver)
                if dp.get('DIRECTION')!='OUT' or dp.get('IS_CLOCK','').lower() not in ('1','true') or len(owners)!=1: issue('clock_driver_identity',pin)
                if len(owners)==1:
                    owner = evidence.properties('cell',owners[0],('NAME','REF_NAME'))
                    if driver.rsplit('/',1)[0]!=owners[0]: issue('clock_driver_owner',pin)
                if row.get('clock'):
                    clock = row['clock']['name']
                    if clock in clock_drivers and clock_drivers[clock]!=drivers: issue('clock_driver_conflict',clock)
                    clock_drivers[clock]=drivers
    return issues, {'effective_polarity':effective,'clock_drivers':clock_drivers,
                    'scope':'same_capture_effective_primitive_polarity_and_typed_clock_driver_identity'}


def review_artifact(*, reports, auxiliary, checkpoint_sha256):
    """Replay raw evidence. Caller authenticates paths/hashes before this call."""
    if 'reset_status' not in auxiliary:
        return {'issues': [{'reason': 'missing_reset_status_capture', 'where': ''}], 'canonical_facts': {}, 'route_qualified': False}
    path = Path(auxiliary['reset_status']); before = file_record(path); evidence = Evidence()
    try:
        for words in g.records(path): evidence.add(words)
        facts = evidence.structural_review(checkpoint_sha256)
        regs = {root: value['registers'] for root,value in facts['resets'].items()}
        regs[ROOTS[2]] = facts['status']['registers']
        facts['pulse'] = {}
        for root,key in zip(ROOTS, ('reset_status_core_pulse','reset_status_hbm_pulse','reset_status_status_pulse')):
            if key not in auxiliary:
                evidence.issue('missing_scoped_pulse_report', key); continue
            issues, pulse = pulse_review(auxiliary[key], root, regs[root]); evidence.issues.extend(issues); facts['pulse'][root] = pulse
        issues, polarity = polarity_review(evidence, regs)
        evidence.issues.extend(issues); facts['polarity'] = polarity
    except (ValueError, KeyError, IndexError, ArithmeticError) as error:
        evidence.issue('malformed_evidence', str(error)); facts = {}
    g.require(file_record(path) == before, 'Reset/status evidence changed during review')
    facts['input'] = before
    return {'issues': evidence.issues, 'canonical_facts': facts, 'route_qualified': False}
