"""Incomplete or asserted structural observations cannot become qualification."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import subprocess
import pytest
from boards.c1_f2_alex import structural_capture_review as S

HELPERS = {name: 'a' * 64 for name in S.HELPERS}
DCP = 'b' * 64


def label(value):
    return ' '.join(value)


def records(errors=()):
    result = [['header', 'schema', '1', 'kind', 'c1_structural_scope_capture',
               'checkpoint', '/bound/source.dcp', 'checkpoint_sha256', DCP,
               'helpers', ' '.join(f'{k} {v}' for k, v in HELPERS.items()),
               'route_qualified', '0', 'hardware_measured', '0']]
    for index, step in enumerate(S.STEPS):
        result.append(['step_begin', label(step)])
        if index in errors:
            result.append(['step_result', label(step), 'state', 'error', 'message',
                           'raw failure\n[never_execute]', 'options', '-code 1 -level 0 -errorinfo {raw stack}',
                           'route_qualified', '0'])
        else:
            result.append(['step_result', label(step), 'state', 'observed', 'value', '', 'route_qualified', '0'])
    result.append(['summary', 'attempted', '141', 'observed', str(141-len(errors)),
                   'failed', str(len(errors)), 'labels', ' '.join('{' + label(x) + '}' for x in S.STEPS),
                   'output_byte_cap', '0', 'route_qualified', '0', 'hardware_measured', '0'])
    return result


def parse(rows):
    return S.parse(iter(rows), DCP, HELPERS)


def replace_field(row, key, value):
    row[row.index(key)+1] = value


def test_all_141_steps_retained_in_order_including_early_failures():
    parsed = parse(records((0, 5, 16)))
    assert len(parsed['steps']) == 141
    assert [tuple(x['label']) for x in parsed['steps']] == list(S.STEPS)
    assert [i for i, x in enumerate(parsed['steps']) if x['result']['state'] == 'error'] == [0, 5, 16]
    assert parsed['steps'][-1]['label'] == ['unused_clocks']
    assert parsed['steps'][0]['result']['message'] == 'raw failure\n[never_execute]'


@pytest.mark.parametrize('mutation', [
    'missing_header', 'duplicate_header', 'wrong_kind', 'wrong_dcp', 'wrong_helper',
    'header_claim', 'hardware_claim', 'missing_step', 'duplicate_step', 'nested_step',
    'unmatched_result', 'unknown_state', 'step_claim', 'wrong_result_fields',
    'wrong_attempted', 'wrong_observed', 'wrong_failed', 'missing_summary_label',
    'reordered_summary_label', 'summary_claim', 'cap', 'trailing_data', 'capture_error',
    'unknown_payload', 'payload_outside', 'missing_summary', 'malformed_error',
])
def test_mutated_capture_is_rejected(mutation):
    rows = records((0,))
    if mutation == 'missing_header': rows.pop(0)
    elif mutation == 'duplicate_header': rows.insert(1, rows[0][:])
    elif mutation == 'wrong_kind': replace_field(rows[0], 'kind', 'another')
    elif mutation == 'wrong_dcp': replace_field(rows[0], 'checkpoint_sha256', 'c'*64)
    elif mutation == 'wrong_helper': replace_field(rows[0], 'helpers', rows[0][rows[0].index('helpers')+1].replace('a'*64, 'c'*64, 1))
    elif mutation == 'header_claim': replace_field(rows[0], 'route_qualified', '1')
    elif mutation == 'hardware_claim': replace_field(rows[0], 'hardware_measured', '1')
    elif mutation == 'missing_step': del rows[3:5]
    elif mutation == 'duplicate_step': rows[3:3] = deepcopy(rows[1:3])
    elif mutation == 'nested_step': rows.insert(2, ['step_begin', 'clocks'])
    elif mutation == 'unmatched_result': rows[2][1] = 'clocks'
    elif mutation == 'unknown_state': replace_field(rows[2], 'state', 'passed')
    elif mutation == 'step_claim': replace_field(rows[2], 'route_qualified', '1')
    elif mutation == 'wrong_result_fields': rows[2].extend(['extra', 'allowed'])
    elif mutation == 'wrong_attempted': replace_field(rows[-1], 'attempted', '140')
    elif mutation == 'wrong_observed': replace_field(rows[-1], 'observed', '141')
    elif mutation == 'wrong_failed': replace_field(rows[-1], 'failed', '0')
    elif mutation == 'missing_summary_label': replace_field(rows[-1], 'labels', ' '.join('{' + label(x) + '}' for x in S.STEPS[:-1]))
    elif mutation == 'reordered_summary_label': replace_field(rows[-1], 'labels', ' '.join('{' + label(x) + '}' for x in reversed(S.STEPS)))
    elif mutation == 'summary_claim': replace_field(rows[-1], 'route_qualified', '1')
    elif mutation == 'cap': replace_field(rows[-1], 'output_byte_cap', '1')
    elif mutation == 'trailing_data': rows.append(['step_begin', 'general'])
    elif mutation == 'capture_error': rows[-1] = ['capture_error', 'message', 'failure', 'options', '-code 1 -level 0', 'route_qualified', '0']
    elif mutation == 'unknown_payload': rows.insert(2, ['unknown', 'accept'])
    elif mutation == 'payload_outside': rows.insert(3, ['memory_interface_pins', 'pin'])
    elif mutation == 'missing_summary': rows.pop()
    elif mutation == 'malformed_error': replace_field(rows[2], 'options', '-code 1')
    else: raise AssertionError(mutation)
    with pytest.raises(ValueError): parse(rows)


def test_payload_identity_and_field_inventory_are_not_interchangeable():
    record = ['fabric_flow', 'M_H7', 'signal', 'rready', 'anchor', 'p', 'nets', 'n',
              'driver', 'd', 'source_owner', 's', 'destination_owner', 't',
              'receiver_count', '1', 'leaf_pins', 'd p', 'leaf_directions', 'OUT IN']
    assert S.payload(record, ('flow', 'M_H7', 'rready')) == record
    with pytest.raises(ValueError): S.payload(record, ('flow', 'M_H6', 'rready'))
    with pytest.raises(ValueError): S.payload(record + ['extra', 'x'], ('flow', 'M_H7', 'rready'))
    with pytest.raises(ValueError): S.payload(record, ('general',))
    with pytest.raises(ValueError): S.payload(['routed_host_wready', 'kind raw'], ('flow', 'S_N0', 'rvalid'))


def test_native_and_disabled_observations_bind_their_step():
    native = ['native_b_response', '3', 'driver', 'd', 'endpoints', 'p']
    assert S.payload(native, ('flow', 'S_N3', 'bvalid')) == native
    with pytest.raises(ValueError): S.payload(native, ('flow', 'S_N4', 'bvalid'))
    disabled = ['native_disabled_write', '7', 'channel', 'w', 'pin', 'p', 'nets', 'n', 'driver', 'd', 'primitive', 'GND']
    assert S.payload(disabled, ('disabled_write', '7', 'w')) == disabled
    with pytest.raises(ValueError): S.payload(disabled, ('disabled_write', '7', 'aw'))


@pytest.mark.parametrize('engine', range(16))
def test_tag_ram_partial_records_retain_exact_pc_engine_scope(engine):
    cell = f'WRAPPER/CL/u_board/adapter/engines[{engine}].tag_q_reg_0_31_0_3/RAMA'
    label = ('flow', f'S_N{engine//2}', 'rvalid')
    control = ['tag_ram_control', 'engine', str(engine), 'cell', cell, 'reference', 'RAMD32',
               'pin', cell+'/WE', 'drivers', 'gate/O', 'clocks', '']
    inventory = ['tag_ram_inventory', 'engine', str(engine), 'cells', cell,
                 'write_enable_pins', cell+'/WE', 'clock_reference', 'WRAPPER/CL/core_rst_reg/C', 'clocks', 'core']
    for row in (control, inventory):
        assert S.payload(row, label) == row
        with pytest.raises(ValueError): S.payload(row, ('flow', f'S_N{(engine//2+1)%8}', 'rvalid'))
        with pytest.raises(ValueError): S.payload(row, ('flow', label[1], 'arready'))
        with pytest.raises(ValueError): S.payload(row + ['extra', 'x'], label)
    index = S.STEPS.index(label)
    rows = records((index,))
    rows[2 + index*2:2 + index*2] = [control, inventory]
    parsed = parse(rows)
    assert parsed['steps'][index]['records'] == [control, inventory]
    assert parsed['steps'][index]['result']['state'] == 'error'


def tcl_word(text):
    # Fixtures contain balanced braces and no escaped newline; braces preserve
    # literal substitutions in this Tcl-list-data encoder.
    return '{' + text + '}'


def write_records(path, rows):
    path.write_text('\n'.join(' '.join(tcl_word(x) for x in row) for row in rows)+'\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bounded_file_replay_keeps_every_semantic_obligation(tmp_path):
    path = tmp_path/'raw'; digest = write_records(path, records((0, 5)))
    result = S.review_artifact(path=path, sha256=digest, checkpoint_sha256=DCP, helpers=HELPERS)
    assert result['capture_complete'] and not result['issues']
    assert not result['structural_proven'] and not result['route_qualified'] and not result['hardware_measured']
    assert len(result['pending_semantic_steps']) == 141 and len(result['failed_steps']) == 2
    assert result['capture']['sha256'] == digest
    assert not S.review_artifact(path=path, sha256='f'*64, checkpoint_sha256=DCP, helpers=HELPERS)['capture_complete']
    linked = tmp_path/'link'; linked.symlink_to(path)
    assert not S.review_artifact(path=linked, sha256=digest, checkpoint_sha256=DCP, helpers=HELPERS)['capture_complete']


@pytest.mark.skipif(shutil.which('tclsh') is None, reason='Tcl unavailable')
def test_actual_collector_multiline_errors_and_late_steps_parse(tmp_path):
    source = Path(S.__file__).with_name('capture_structure.tcl')
    checkpoint = tmp_path/'source.dcp'; checkpoint.write_bytes(b'fixture')
    destination = tmp_path/'capture'
    script = tmp_path/'fixture.tcl'
    script.write_text('source {' + str(source) + '}\n' + r'''
proc ::c1_alex_route::sha256 {path} {return [string repeat a 64]}
proc version {args} {return 2025.2}
proc get_designs {args} {return {}}
proc set_param {args} {}
proc open_checkpoint {args} {}
proc close_design {} {}
proc ::c1_structure_capture::general {audit} {error "general failure\nsecond line"}
proc ::c1_structure_capture::clocks {audit} {return {}}
proc ::c1_structure_capture::unused_clocks {audit} {
 puts $audit [list clock_source_inventory core sources {WRAPPER/CL/u_core_clk/clk_out1}]
 return {clocks core unused {}}
}
proc ::c1_structure_capture::disabled {audit pc channel} {return {}}
proc ::c1_structure_capture::flow_detail {args} {}
proc ::c1_alex_routed_ready::host_wready {audit} {error {missing replica}}
proc ::c1_alex::flow {args} {return {}}
::c1_structure_capture::capture {DEST} {DCP} DIGEST
'''.replace('DEST', str(destination)).replace('DCP', str(checkpoint)).replace('DIGEST', 'a'*64))
    run = subprocess.run([shutil.which('tclsh'), str(script)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    path = destination/'structural.tcldata'
    result = S.review_artifact(path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), checkpoint_sha256='a'*64, helpers=HELPERS)
    assert result['capture_complete'], result['issues']
    assert len(result['failed_steps']) == 2 and len(result['pending_semantic_steps']) == 141
    assert result['failed_steps'][0]['result']['message'] == 'general failure\nsecond line'
    assert result['canonical_facts']['steps'][-1]['records'][0][0] == 'clock_source_inventory'
    assert not result['structural_proven']
