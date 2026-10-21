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
    'general': {'memory_interface_pins', 'smartconnect', 'unresolved_shell_objects'},
    'clocks': {'clock', 'clock_registers', 'fabric_clock', 'fabric_port_clock_consumers'},
    'flow': {'fabric_flow', 'native_receive_gate', 'native_b_response', 'routed_host_wready'},
    'disabled_write': {'disabled_network', 'disabled_polarity', 'native_disabled_write'},
    'unused_clocks': {'clock_source_inventory', 'unused_clock'},
}
# Fixed field schemas of the unchanged collectors. Values are retained verbatim
# for semantic replay; accepting their syntax is not accepting their assertions.
FIELDS = {
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


def payload(words, label):
    tag = words[0]
    N.require(tag in ALLOWED_TAGS[label[0]], 'unexpected structural payload in step')
    if tag in FIELDS:
        N.require(len(words) >= 2 and words[1], 'missing structural payload identity')
        fields = N.pairs(words[2:], FIELDS[tag])
        if tag == 'fabric_flow':
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
    else:
        raise ValueError('unhandled structural payload tag')
    return words


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
