"""Replay hierarchy and clock observations, without granting route acceptance.

The caller must authenticate the capture first. Producer success flags are only
prerequisites: each predicate is recalculated from the retained query results.
"""
from collections import Counter
from decimal import Decimal, InvalidOperation
import re
from . import native_review as N
from . import structural_capture_review as S

ROOT = N.ROOT
SC = ROOT + 'u_mem/f2_mem_i/memory_sc'
HIERARCHY = {
    ROOT[:-1]: 'cl_opentpu', ROOT+'u_board': 'otpu_c1_hbm_kernel',
    ROOT+'u_board/core': 'otpu_c1_top',
    **{ROOT+f'u_native{i}': 'otpu_native_axi' for i in range(8)},
    **{ROOT+k: v for k, v in {
        'u_mem': 'f2_mem_wrapper', 'u_hbm': 'cl_mem_hbm_wrapper',
        'u_ocl_cdc': 'f2_ocl_cdc', 'u_pcis_pipe': 'f2_pcis_pipe',
        'u_core_clk': 'f2_core_clk', 'u_hbm_clk': 'f2_hbm_clk'}.items()},
}
ANCHORS = {'host': ('clk_main_a0', '4'), 'core': ('u_core_clk/clk_out1', '5'),
           'hbm': ('u_hbm_clk/clk_out1', '2.222222222222'),
           'reference': ('clk_hbm_ref', '10')}
PROPS = {'NAME', 'PERIOD', 'MASTER_CLOCK', 'SOURCE_PINS', 'IS_GENERATED',
         'IS_PROPAGATED', 'IS_VIRTUAL'}
STATIC_STEPS = ('general', 'clocks', 'unused_clocks')


def rows(step, tag):
    return [row for row in step['records'] if row[0] == tag]


def one(step, tag):
    values = rows(step, tag)
    N.require(len(values) == 1, 'missing/duplicate ' + tag)
    return values[0]


def keyed(step, tag):
    result = {}
    for row in rows(step, tag):
        N.require(row[1] not in result, 'duplicate ' + tag)
        result[row[1]] = row
    return result


def decimal(value):
    N.require(isinstance(value, str) and re.fullmatch(r'-?[0-9]+(?:\.[0-9]+)?', value),
              'invalid clock period')
    result = Decimal(value)
    N.require(result.is_finite() and result > 0, 'nonpositive clock period')
    return result


def count_dict(value):
    return {k: N.number(v) for k, v in N.td(value).items()}


def general(step):
    objects = keyed(step, 'hierarchy_cell')
    N.require(set(objects) == set(HIERARCHY), 'hierarchy inventory differs')
    for name, reference in HIERARCHY.items():
        data = N.pairs(objects[name][2:], S.FIELDS['hierarchy_cell'])
        prop = N.td(data['properties'], ('NAME', 'REF_NAME', 'ORIG_REF_NAME', 'IS_BLACKBOX'))
        N.require(N.names(data['selected']) == [name] and prop['NAME'] == name and
                  data['expected_reference'] == reference and
                  (prop['ORIG_REF_NAME'] or prop['REF_NAME']) == reference and
                  prop['IS_BLACKBOX'].lower() in ('0', 'false', 'no', 'off'),
                  'hierarchy identity/reference/blackbox differs: ' + name)
    inventories = keyed(step, 'cell_inventory')
    N.require(set(inventories) == {'native', 'blackboxes', 'tmem'}, 'cell inventories differ')
    cells = {}
    for label, row in inventories.items():
        data = N.pairs(row[2:], S.FIELDS['cell_inventory'])
        cells[label] = N.names(data['names'])
        N.require(N.number(data['count']) == len(cells[label]), 'cell inventory count differs')
    N.require(set(cells['native']) == {ROOT+f'u_native{i}' for i in range(8)} and
              not cells['tmem'], 'native/TMEM topology differs')
    N.require(all(n != ROOT[:-1] and not n.startswith(ROOT) for n in cells['blackboxes']),
              'unresolved CL/IP')
    N.require(N.names(one(step, 'unresolved_shell_objects')[1]) == cells['blackboxes'],
              'opaque shell inventory differs')
    pins = N.names(one(step, 'memory_interface_pins')[1])
    N.require(pins and all(p.startswith(ROOT+'u_mem/') for p in pins), 'memory interface scope differs')
    data = N.pairs(one(step, 'smartconnect_properties')[1:], ('selected', 'properties'))
    prop = N.td(data['properties'], ('NAME', 'REF_NAME', 'IS_BLACKBOX'))
    N.require(N.names(data['selected']) == [SC] and prop['NAME'] == SC and prop['REF_NAME'] and
              prop['IS_BLACKBOX'].lower() in ('0', 'false', 'no', 'off'), 'SmartConnect identity differs')
    sc = one(step, 'smartconnect')
    data = N.pairs(sc[2:], S.FIELDS['smartconnect'])
    N.require(N.names(sc[1]) == [SC] and data == {
        'reference': prop['REF_NAME'], 'slave_interfaces': '9', 'master_interfaces': '8',
        'address_map_validation': 'executor_generated_BD_required'}, 'SmartConnect scope differs')
    return {'hierarchy_cells': len(objects), 'native_masters': 8, 'slave_interfaces': 9,
            'master_interfaces': 8, 'generated_BD_validation_required': True}


def clocks(step):
    S.register_inventory(step)
    header = N.pairs(one(step, 'register_clock_inventory')[1:], ('count', 'names'))
    names = N.names(header['names'])
    N.require(names and N.number(header['count']) == len(names), 'clock count differs')
    domains, anchors = keyed(step, 'clock'), keyed(step, 'clock_anchor')
    N.require(set(domains) == set(anchors) == set(ANCHORS), 'clock domain/anchor inventory differs')
    role = {}; domain_data = {}
    for label, (pin, period) in ANCHORS.items():
        data = N.td(domains[label][2])
        N.require({'NAME', 'PERIOD'} <= set(data) <= PROPS, 'clock properties differ')
        name = data['NAME']
        N.require(name in names and name not in role and
                  abs(decimal(data['PERIOD']) - Decimal(period)) <= Decimal('0.001'),
                  'clock identity/period differs: ' + label)
        anchor = N.pairs(anchors[label][2:], S.FIELDS['clock_anchor'])
        N.require(anchor['pin'] == ROOT+pin and N.names(anchor['selected']) == [ROOT+pin] and
                  N.names(anchor['clocks']) == [name], 'clock anchor mapping differs: ' + label)
        role[name] = label; domain_data[name] = data
    inventories = keyed(step, 'register_inventory')
    N.require(set(inventories) == set(names), 'clock register inventories differ')
    properties = {}; members = {}
    for name in names:
        data = N.pairs(inventories[name][2:], S.FIELDS['register_inventory'])
        raw_props = N.td(data['properties'], PROPS)
        prop = {}
        for key, value in raw_props.items():
            if value == 'unavailable': continue
            words = N.tcl_list(value)
            N.require(len(words) == 2 and words[0] == 'observed', 'invalid observed clock property')
            prop[key] = words[1]
        N.require({'NAME', 'PERIOD', 'SOURCE_PINS'} <= set(prop) and prop['NAME'] == name,
                  'missing clock identity/period/source inventory')
        decimal(prop['PERIOD']); N.names(prop['SOURCE_PINS'])
        if name in domain_data:
            N.require(prop == domain_data[name], 'clock properties disagree across queries')
        properties[name] = prop
        members[name] = [n for row in rows(step, 'register_chunk') if row[1] == name
                         for n in N.names(N.pairs(row[2:], S.FIELDS['register_chunk'])['names'])]
    summaries = keyed(step, 'fabric_clock')
    N.require(set(summaries) == set(names), 'fabric clock coverage differs')
    ports = Counter(); owner_by_role = {}
    for name in names:
        label = role.get(name, 'other')
        label = label if label != 'reference' else 'other'
        owners = Counter()
        for member in members[name]:
            if not member.startswith(ROOT):
                N.require(label != 'hbm', 'HBM-clock register outside CL')
                continue
            local = member[len(ROOT):]; owner = local.split('/')[0]; owners[owner] += 1
            N.require(label != 'hbm' or owner in ('u_mem', 'u_hbm', 'u_rst_hbm'),
                      'custom register on HBM clock')
            N.require(not (owner == 'u_board' or re.fullmatch(r'u_native[0-7]', owner)) or label == 'core',
                      'custom datapath on wrong clock')
            N.require(owner != 'u_pcis_pipe' or label == 'host', 'PCIS on wrong clock')
            N.require(owner != 'u_mem' or label in ('host', 'core', 'hbm'), 'fabric on unreviewed clock')
            match = re.match(r'u_mem/f2_mem_i/memory_sc/inst/((s([0-9]{2})_entry_pipeline)|(m([0-9]{2})_exit_pipeline))/', local)
            if match:
                node, si, mi = match[1], match[3], match[5]
                index = int(si if si is not None else mi)
                N.require(index < (9 if si is not None else 8), 'unexpected fabric port index')
                wanted = ('host' if index == 0 else 'core') if si is not None else 'hbm'
                N.require(label == wanted, 'fabric port on wrong clock')
                ports[node] += 1
        summary = N.pairs(summaries[name][2:], S.FIELDS['fabric_clock'])
        N.require(summary['label'] == label and N.number(summary['registers']) == len(members[name]) and
                  count_dict(summary['owner_counts']) == dict(owners), 'fabric owner summary differs')
        if label != 'other': owner_by_role[label] = owners
    domain_counts = keyed(step, 'clock_registers')
    N.require(set(domain_counts) == {'host', 'core', 'hbm'}, 'domain register counts differ')
    required = {'host': {'u_mem', 'u_pcis_pipe'}, 'hbm': {'u_mem', 'u_hbm'},
                'core': {'u_mem', 'u_board', *(f'u_native{i}' for i in range(8))}}
    for label, owners in required.items():
        name = next(n for n, r in role.items() if r == label)
        data = N.pairs(domain_counts[label][2:], ('count',))
        N.require(N.number(data['count']) == len(members[name]) > 0 and
                  all(owner_by_role[label][owner] > 0 for owner in owners), 'missing required domain consumer')
    expected_ports = {f's{i:02d}_entry_pipeline' for i in range(9)} | {f'm{i:02d}_exit_pipeline' for i in range(8)}
    N.require(set(ports) == expected_ports and
              count_dict(one(step, 'fabric_port_clock_consumers')[1]) == dict(ports), 'fabric port counts differ')
    return {'roles': role, 'properties': properties, 'members': members,
            'clock_count': len(names), 'port_consumers': dict(ports)}


def unused(step, clock_facts):
    inventory = keyed(step, 'clock_source_inventory')
    N.require(set(inventory) == set(clock_facts['properties']), 'clock source inventory differs')
    wanted = Counter()
    for name, row in inventory.items():
        sources = N.names(N.pairs(row[2:], ('sources',))['sources'])
        N.require(sources == N.names(clock_facts['properties'][name]['SOURCE_PINS']), 'clock sources disagree')
        for source in sources:
            if source.startswith(ROOT+'u_core_clk/') and clock_facts['roles'].get(name) != 'core':
                wanted[name] += 1
    actual = Counter()
    for row in rows(step, 'unused_clock'):
        name = row[1]; actual[name] += 1
        N.require(name in wanted, 'unexpected unused clock')
        data = N.pairs(row[2:], S.FIELDS['unused_clock'])
        N.require(decimal(data['period']) == decimal(clock_facts['properties'][name]['PERIOD']) and
                  N.number(data['consumers']) == 0 and not N.names(data['registers']) and
                  not clock_facts['members'][name], 'unused clock has consumers or wrong period')
    N.require(actual == wanted, 'missing/duplicate unused-clock observation')
    return {'unused_clock_count': len(wanted), 'unused_clocks': sorted(wanted)}


def review(canonical_facts):
    result = {'kind': 'c1-structural-static-review', 'schema': 1,
              'proven_static_steps': [], 'pending_static_steps': list(STATIC_STEPS),
              'facts': {}, 'issues': [], 'route_qualified': False, 'hardware_measured': False}
    clock_facts = None
    for label in STATIC_STEPS:
        try:
            candidates = [s for s in canonical_facts['steps'] if s['label'] == [label]]
            N.require(len(candidates) == 1, 'missing/duplicate static step')
            step = candidates[0]
            N.require(step['result']['state'] == 'observed', 'collector step failed')
            if label == 'general': facts = general(step)
            elif label == 'clocks': facts = clock_facts = clocks(step)
            else:
                N.require(clock_facts is not None, 'clock proof prerequisite missing')
                facts = unused(step, clock_facts)
            if label == 'clocks': facts = {k: v for k, v in facts.items() if k not in ('properties', 'members')}
            result['facts'][label] = facts
            result['proven_static_steps'].append(label); result['pending_static_steps'].remove(label)
        except (ValueError, KeyError, TypeError, IndexError, InvalidOperation) as error:
            result['issues'].append({'step': label, 'reason': str(error)})
    return result
