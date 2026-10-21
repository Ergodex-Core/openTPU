"""Authenticate the complete structural observation envelope, not its semantics.

Every observed and failed step is retained for the final structural reviewer.
An ``observed`` result never grants a pass here: all 141 steps remain semantic
obligations. This module only parses bounded Tcl list data; it never runs Tcl.
"""
from __future__ import annotations

import hashlib
import json
import re
from . import native_review as N


HELPERS = frozenset({'capture_structure.tcl', 'qualify_c1_alex.tcl',
                     'audit_c1_alex.tcl', 'routed_host_ready.tcl'})
SIGNALS = ('awvalid', 'awready', 'wvalid', 'wready', 'bvalid', 'bready',
           'arvalid', 'arready', 'rvalid', 'rready')


def expected_steps():
    steps = [('general',), ('clocks',)]
    steps.extend(('flow', 'S_HOST', signal) for signal in SIGNALS)
    for pc in range(8):
        steps.extend(('flow', f'S_N{pc}', signal)
                     for signal in ('bvalid', 'arvalid', 'arready', 'rvalid'))
        steps.extend(('flow', f'M_H{pc}', signal) for signal in SIGNALS)
        steps.extend(('disabled_write', str(pc), channel) for channel in ('aw', 'w'))
    return (*steps, ('unused_clocks',))


STEPS = expected_steps()
ALLOWED_TAGS = {
    'general': {'memory_interface_pins', 'smartconnect', 'unresolved_shell_objects',
                'hierarchy_cell', 'cell_inventory'},
    'clocks': {'clock', 'clock_registers', 'fabric_clock', 'fabric_port_clock_consumers',
               'register_clock_inventory', 'register_inventory', 'register_chunk'},
    'flow': {'fabric_flow', 'native_receive_gate', 'native_b_response', 'routed_host_wready',
             'flow_detail', 'tag_ram_control', 'tag_ram_inventory'},
    'disabled_write': {'disabled_network', 'disabled_polarity', 'native_disabled_write'},
    'unused_clocks': {'clock_source_inventory', 'unused_clock'},
}
# Fixed field schemas of the unchanged collectors. Values are retained verbatim
# for semantic replay; accepting their syntax is not accepting their assertions.
FIELDS = {
    'hierarchy_cell': ('expected_reference', 'selected', 'properties'),
    'cell_inventory': ('count', 'names'),
    'register_inventory': ('properties', 'count', 'chunk_size', 'chunks'),
    'register_chunk': ('offset', 'names'),
    'flow_detail': ('signal', 'anchor_kind', 'anchor', 'anchor_properties', 'local_nets',
                    'nets', 'ports', 'drivers', 'driver_cells'),
    'smartconnect': ('reference', 'slave_interfaces', 'master_interfaces', 'address_map_validation'),
    'clock_registers': ('count',),
    'fabric_clock': ('label', 'registers', 'owner_counts'),
    'fabric_flow': ('signal', 'anchor', 'nets', 'driver', 'source_owner',
                    'destination_owner', 'receiver_count', 'leaf_pins', 'leaf_directions'),
    'native_receive_gate': ('input', 'reference', 'init', 'reset_driver', 'output',
                            'sink_families', 'endpoints', 'cell_references'),
    'native_b_response': ('driver', 'endpoints'),
    'disabled_network': ('channel', 'anchor', 'local_nets', 'segment_count', 'drivers', 'ports', 'sink_scope'),
    'disabled_polarity': ('channel', 'anchor', 'driver', 'owner', 'reference'),
    'native_disabled_write': ('channel', 'pin', 'nets', 'driver', 'primitive'),
    'clock_source_inventory': ('sources',),
    'unused_clock': ('period', 'consumers', 'registers'),
}
LIST_RECORDS = {'memory_interface_pins', 'unresolved_shell_objects'}
DICT_RECORDS = {'fabric_port_clock_consumers', 'routed_host_wready'}
TAG_FIELDS = {
    'tag_ram_control': ('engine', 'cell', 'reference', 'pin', 'drivers', 'clocks'),
    'tag_ram_inventory': ('engine', 'cells', 'write_enable_pins', 'clock_reference', 'clocks'),
}


def payload(words, label):
    tag = words[0]
    N.require(tag in ALLOWED_TAGS[label[0]], 'unexpected structural payload in step')
    if tag in TAG_FIELDS:
        fields = N.pairs(words[1:], TAG_FIELDS[tag])
        N.require(len(label) == 3 and re.fullmatch(r'S_N[0-7]', label[1]) and label[2] == 'rvalid',
                  'tag RAM witness outside native RVALID step')
        engine = N.number(fields['engine'])
        N.require(engine // 2 == int(label[1][-1]), 'tag RAM witness belongs to another PC')
        prefix = N.ROOT + f'u_board/adapter/engines[{engine}].tag_q'
        N.names(fields['clocks'])
        if tag == 'tag_ram_control':
            N.require(fields['cell'].startswith(prefix) and fields['pin'] in
                      (fields['cell'] + '/WE', fields['cell'] + '/CLK'), 'tag RAM control scope differs')
            N.names(fields['drivers'])
        else:
            cells = N.names(fields['cells']); pins = N.names(fields['write_enable_pins'])
            N.require(0 < len(cells) <= 256 and all(name.startswith(prefix) for name in cells) and
                      all(name.startswith(prefix) and name.endswith('/WE') for name in pins) and
                      fields['clock_reference'] == N.ROOT + 'core_rst_reg/C', 'tag RAM inventory scope differs')
    elif tag in FIELDS:
        N.require(len(words) >= 2 and words[1], 'missing structural payload identity')
        fields = N.pairs(words[2:], FIELDS[tag])
        if tag in ('fabric_flow', 'flow_detail'):
            N.require((words[1], fields['signal']) == label[1:], 'flow witness belongs to another step')
        elif tag in ('native_receive_gate', 'native_b_response'):
            N.require(label[1:] == ('S_N' + words[1], 'rvalid' if tag == 'native_receive_gate' else 'bvalid'),
                      'native witness belongs to another step')
        elif tag.startswith('disabled_') or tag == 'native_disabled_write':
            N.require((words[1], fields['channel']) == label[1:], 'disabled witness belongs to another step')
    elif tag in LIST_RECORDS:
        N.require(len(words) == 2, 'malformed structural list payload')
        N.names(words[1])
    elif tag in DICT_RECORDS:
        N.require(len(words) == 2, 'malformed structural dictionary payload')
        N.td(words[1])
        if tag == 'routed_host_wready':
            N.require(label == ('flow', 'S_HOST', 'wready'), 'host-ready witness belongs to another step')
    elif tag == 'clock':
        N.require(len(words) == 3 and words[1] in ('host', 'core', 'hbm', 'reference'),
                  'malformed clock payload')
        N.td(words[2])
    elif tag == 'register_clock_inventory':
        fields = N.pairs(words[1:], ('count', 'names'))
        N.require(N.number(fields['count']) == len(N.names(fields['names'])) <= 256,
                  'clock inventory count differs')
    else:
        raise ValueError('unhandled structural payload tag')
    return words


def register_inventory(step):
    """Reject dropped/duplicated chunks in a claimed complete clock observation.

    This proves enumeration consistency only. Every register's clock/owner and
    every clock's semantic role still require the final structural reviewer.
    """
    records = step['records']
    headers = [N.pairs(row[1:], ('count', 'names')) for row in records if row[0] == 'register_clock_inventory']
    inventories = [row for row in records if row[0] == 'register_inventory']
    chunks = [row for row in records if row[0] == 'register_chunk']
    if not headers and not inventories and not chunks:
        return
    N.require(len(headers) == 1, 'missing/duplicate register clock inventory')
    clocks = N.names(headers[0]['names'])
    N.require(clocks and N.number(headers[0]['count']) == len(clocks), 'empty/inconsistent clock inventory')
    N.require([row[1] for row in inventories] == clocks, 'missing/reordered/duplicate register inventory')
    remaining = list(chunks)
    for row in inventories:
        meta = N.pairs(row[2:], FIELDS['register_inventory'])
        count = N.number(meta['count'])
        N.require(count <= 2000000 and meta['chunk_size'] == '256' and
                  N.number(meta['chunks']) == (count + 255) // 256, 'register chunk declaration differs')
        properties = N.td(meta['properties'])
        N.require(N.tcl_list(properties.get('NAME', '')) == ['observed', row[1]], 'register clock identity differs')
        observed = []
        for offset in range(0, count, 256):
            N.require(remaining and remaining[0][1] == row[1], 'missing/reordered register chunk')
            chunk = N.pairs(remaining.pop(0)[2:], FIELDS['register_chunk'])
            names = N.names(chunk['names'])
            N.require(N.number(chunk['offset']) == offset and len(names) == min(256, count-offset),
                      'register chunk offset/length differs')
            observed.extend(names)
        N.require(len(set(observed)) == count, 'duplicate register across chunks')
    N.require(not remaining, 'extra register chunks')


def parse(records, checkpoint_sha256, helpers):
    N.require(re.fullmatch('[0-9a-f]{64}', checkpoint_sha256 or ''), 'invalid trusted checkpoint')
    N.require(isinstance(helpers, dict) and set(helpers) == HELPERS and
              all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h) for h in helpers.values()),
              'invalid trusted collector inventory')
    header = None
    summary = None
    active = None
    steps = []
    for words in records:
        N.require(words and summary is None, 'empty record or data after structural summary')
        tag = words[0]
        if header is None:
            N.require(tag == 'header', 'structural header must be first')
            header = N.pairs(words[1:], ('schema', 'kind', 'checkpoint', 'checkpoint_sha256',
                                         'helpers', 'route_qualified', 'hardware_measured'))
            N.require(header['schema'] == '1' and header['kind'] == 'c1_structural_scope_capture',
                      'structural schema differs')
            N.require(header['checkpoint_sha256'] == checkpoint_sha256 and
                      N.td(header['helpers'], HELPERS) == helpers, 'structural input identity differs')
            N.require(header['route_qualified'] == header['hardware_measured'] == '0',
                      'raw structural capture claimed acceptance')
        elif tag == 'step_begin':
            N.require(len(words) == 2 and active is None and len(steps) < len(STEPS),
                      'overlapping/excess structural step')
            label = tuple(N.tcl_list(words[1]))
            N.require(label == STEPS[len(steps)], 'missing/reordered/duplicate structural step')
            active = {'label': list(label), 'records': []}
        elif tag == 'step_result':
            N.require(active is not None and len(words) >= 2 and
                      N.tcl_list(words[1]) == active['label'], 'structural result has no matching step')
            result = N.pairs(words[2:])
            N.require(result.get('state') in ('observed', 'error'), 'unknown structural step state')
            fields = {'state', 'route_qualified'} | ({'value'} if result['state'] == 'observed'
                                                    else {'message', 'options'})
            N.require(set(result) == fields and result['route_qualified'] == '0',
                      'malformed structural result or acceptance claim')
            if result['state'] == 'error':
                options = N.td(result['options'])
                N.require({'-code', '-level'} <= set(options), 'incomplete Tcl error options')
            active['result'] = result
            if active['label'] == ['clocks'] and result['state'] == 'observed':
                register_inventory(active)
            steps.append(active)
            active = None
        elif tag == 'summary':
            N.require(active is None and len(steps) == len(STEPS), 'incomplete structural step coverage')
            summary = N.pairs(words[1:], ('attempted', 'observed', 'failed', 'labels',
                                          'output_byte_cap', 'route_qualified', 'hardware_measured'))
            errors = sum(step['result']['state'] == 'error' for step in steps)
            N.require(N.number(summary['attempted']) == len(STEPS) and
                      N.number(summary['failed']) == errors and
                      N.number(summary['observed']) == len(STEPS) - errors, 'structural summary counts differ')
            N.require(tuple(tuple(N.tcl_list(x)) for x in N.tcl_list(summary['labels'])) == STEPS,
                      'structural summary inventory differs')
            N.require(summary['output_byte_cap'] == summary['route_qualified'] == summary['hardware_measured'] == '0',
                      'incomplete structural capture or acceptance claim')
        else:
            N.require(active is not None, 'structural payload outside step')
            N.require(len(active['records']) < 10000, 'structural per-step record cap')
            active['records'].append(payload(words, tuple(active['label'])))
    N.require(header is not None and summary is not None, 'missing structural envelope')
    return {'header': header, 'steps': steps, 'summary': summary}


def review_artifact(*, path, sha256, checkpoint_sha256, helpers):
    result = {'kind': 'c1-structural-capture-envelope-review', 'schema': 1,
              'capture_complete': False, 'structural_proven': False,
              'route_qualified': False, 'hardware_measured': False, 'issues': []}
    try:
        identity = N.identity(path, sha256)
        canonical = parse(N.records(path), checkpoint_sha256, helpers)
        N.require(N.identity(path, sha256) == identity, 'structural capture changed')
        result.update(capture_complete=True, capture=identity, canonical_facts=canonical,
                      facts_sha256=hashlib.sha256(json.dumps(canonical, sort_keys=True,
                          separators=(',', ':')).encode()).hexdigest(),
                      failed_steps=[step for step in canonical['steps'] if step['result']['state'] == 'error'],
                      pending_semantic_steps=[list(label) for label in STEPS])
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as exc:
        result['issues'].append({'reason': str(exc)})
    return result
