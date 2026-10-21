import copy
import hashlib
from pathlib import Path

import pytest
from boards.c1_f2_alex import accounting_review as A, native_review as N


def counter_fixture():
    """Independent eight-bit adder encoding for increment/decrement/hold."""
    e = A.Evidence(); nodes = {}; root = N.ROOT + 'u_native0/'

    def pin(name, direction, role=None):
        e.objects['pin', name] = dict(NAME=name, DIRECTION=direction,
                                     REF_PIN_NAME=role or name.rsplit('/', 1)[1], IS_INVERTED='0')
        e.queries['select', 'pin', name] = [name]
        e.queries['pin_owner', name] = [name.rsplit('/', 1)[0]]

    def cell(name, ref, inputs=None, init=None, outputs=('O',)):
        e.objects['cell', name] = {'REF_NAME': ref}
        if init is not None: e.objects['cell', name]['INIT'] = init
        edges = {}
        for role, driver in (inputs or {}).items():
            target = name + '/' + role; pin(target, 'IN')
            net = 'net:' + driver
            e.queries['input_nets', target] = [net]
            e.queries['input_drivers', target, net] = [driver]
            e.queries['input_ports', target, net] = []
            edges[target] = dict(scope='selected_input_and_drivers', input=target,
                                 properties={}, nets=[net], drivers=[driver], ports=[])
        for role in outputs:
            out = name + '/' + role; pin(out, 'OUT')
            nodes[out] = dict(cell=name, reference=ref, inputs=edges)
        return name + '/' + outputs[0]

    regs = [root + f'reads_reg[{i}]' for i in range(8)]
    for reg in regs: cell(reg, 'FDRE', outputs=('Q',))
    take = cell(root + 'admission', 'LUT6', init="64'h8000000000000000")
    valid = cell(root + 'response', 'FDRE', outputs=('Q',))
    reset = cell(N.ROOT + 'core_rst_reg', 'FDRE', outputs=('Q',))
    zero = cell(root + 'zero', 'GND', outputs=('G',))
    one = cell(root + 'one', 'VCC', outputs=('P',))
    enable = cell(root + 'enable', 'LUT2', {'I0': take, 'I1': valid}, "4'h6")
    # Addend is 1 for increment, 255 for decrement; no RTL expression copied.
    carry_inputs = {'CI': zero, 'CI_TOP': zero}
    for bit, reg in enumerate(regs):
        carry_inputs[f'DI[{bit}]'] = reg + '/Q'
        carry_inputs[f'S[{bit}]'] = cell(root + f'sum_select_{bit}', 'LUT2',
            {'I0': reg + '/Q', 'I1': one if bit == 0 else valid}, "4'h6")
    carry = root + 'adder'
    cell(carry, 'CARRY8', carry_inputs,
         outputs=tuple(f'{role}[{bit}]' for role in ('O', 'CO') for bit in range(8)))
    e.objects['cell', carry]['CARRY_TYPE'] = 'SINGLE_CY8'
    states = {reg: {'D': {'driver': carry + f'/O[{bit}]', 'inversion': 0},
                    'CE': {'driver': enable, 'inversion': 0},
                    'R': {'driver': reset, 'inversion': 0}}
              for bit, reg in enumerate(regs)}
    f = {'nodes': nodes, 'channels': {'0': {'registers': states, 'gate': root + 'enable',
                                         'rvalid': valid, 'take_read_cut': take}}}
    return e, f


def test_exact_counter_proof_includes_overflow_underflow_cancel_and_reset():
    e, f = counter_fixture(); p = A.prove(e, f)
    assert len(p) == 1 and p[0]['assignments'] == 2048
    assert len(p[0]['cuts']) == 11
    assert 'admission/O' not in p[0]['evaluated_outputs']


@pytest.mark.parametrize('change', ['ce_and', 'direction', 'carry_di', 'carry_ci',
    'wrong_sum_bit', 'reset_invert', 'data_invert', 'enable_invert', 'extra_state',
    'ram_boundary', 'cut_alias', 'unknown_carry_mode', 'single_carry_top',
    'output_invert', 'missing_polarity', 'lut_unknown', 'missing_role'])
def test_counter_mutations_cannot_pass(change):
    e, f = counter_fixture(); root = N.ROOT + 'u_native0/'; ch = f['channels']['0']
    reg = root + 'reads_reg[3]'; carry = root + 'adder'
    if change == 'ce_and': e.objects['cell', root + 'enable']['INIT'] = "4'h8"
    elif change == 'direction': e.objects['cell', root + 'sum_select_5']['INIT'] = "4'h9"
    elif change == 'carry_di':
        target = carry + '/DI[4]'; edge = f['nodes'][carry + '/O[0]']['inputs'][target]
        edge['drivers'] = [root + 'zero/G']; e.queries['input_drivers', target, edge['nets'][0]] = edge['drivers']
    elif change == 'carry_ci':
        target = carry + '/CI'; edge = f['nodes'][carry + '/O[0]']['inputs'][target]
        edge['drivers'] = [root + 'one/P']; e.queries['input_drivers', target, edge['nets'][0]] = edge['drivers']
    elif change == 'wrong_sum_bit': ch['registers'][reg]['D']['driver'] = carry + '/O[4]'
    elif change in ('reset_invert', 'data_invert', 'enable_invert'):
        ch['registers'][reg][{'reset_invert':'R', 'data_invert':'D', 'enable_invert':'CE'}[change]]['inversion'] = 1
    elif change in ('extra_state', 'ram_boundary'):
        driver = root + 'one/P'; ref = 'FDRE' if change == 'extra_state' else 'RAMD32'
        f['nodes'][driver]['reference'] = ref; e.objects['cell', root + 'one']['REF_NAME'] = ref
    elif change == 'cut_alias': ch['take_read_cut'] = ch['rvalid']
    elif change == 'unknown_carry_mode': e.objects['cell', carry]['CARRY_TYPE'] = 'unknown'
    elif change == 'single_carry_top':
        target = carry + '/CI_TOP'; edge = f['nodes'][carry + '/O[0]']['inputs'][target]
        edge['drivers'] = [root + 'one/P']; e.queries['input_drivers', target, edge['nets'][0]] = edge['drivers']
    elif change == 'output_invert': e.objects['pin', carry + '/O[5]']['IS_INVERTED'] = '1'
    elif change == 'missing_polarity': del e.objects['pin', carry + '/S[3]']['IS_INVERTED']
    elif change == 'lut_unknown': e.objects['cell', root + 'enable']['INIT'] = "4'hx"
    elif change == 'missing_role': del f['nodes'][carry + '/O[0]']['inputs'][carry + '/DI[0]']
    with pytest.raises((ValueError, KeyError)):
        A.prove(e, f)


def test_admission_memory_is_explicitly_outside_arithmetic_proof():
    e, f = counter_fixture(); cut = f['channels']['0']['take_read_cut']
    # Replacing the cut's unexamined internals cannot establish admission proof.
    f['nodes'][cut]['reference'] = 'RAMD32'
    assert A.prove(e, f)[0]['take_read_cut'] == cut
    result = A.review_artifact(topology_path='/definitely/missing', topology_sha256='a'*64,
        checkpoint_sha256='b'*64, helper_sha256='c'*64, native_helper_sha256='d'*64)
    assert not result['arithmetic_proven'] and not result['route_qualified']
    assert result['pending_reviews'] == ['take_read_admission_semantics',
        'linked_final_accounting_preservation', 'admission_ram_state_address_control_preservation']


def test_carry_truth_matches_independent_integer_addition():
    # All pairs of eight-bit operands, represented as 32 batches of 2048 rows.
    for batch in range(32):
        values = {'CI': 0, 'CI_TOP': 0}
        operands = [(row & 255, (row >> 8) + batch * 8) for row in range(A.Truth.count)]
        for bit in range(8):
            values[f'S[{bit}]'] = sum(1 << row for row, (a, b) in enumerate(operands) if ((a ^ b) >> bit) & 1)
            values[f'DI[{bit}]'] = sum(1 << row for row, (a, _) in enumerate(operands) if (a >> bit) & 1)
        outputs = A.carry8(values, 'SINGLE_CY8')
        for bit in range(8):
            expected = sum(1 << row for row, (a, b) in enumerate(operands) if ((a + b) >> bit) & 1)
            assert outputs[f'O[{bit}]'] == expected
        assert outputs['CO[7]'] == sum(1 << row for row, (a, b) in enumerate(operands) if a + b > 255)


def test_dual_carry_has_independent_upper_initial_carry():
    values = {'CI': 0, 'CI_TOP': A.Truth.mask}
    values.update({f'{role}[{bit}]': A.Truth.mask if role == 'S' else 0
                   for role in ('DI', 'S') for bit in range(8)})
    result = A.carry8(values, 'DUAL_CY4')
    assert result['CO[3]'] == 0 and result['CO[7]'] == A.Truth.mask
    assert result['O[3]'] == A.Truth.mask and result['O[4]'] == 0


def test_raw_unknown_duplicate_and_out_of_envelope_records_rejected():
    e = A.Evidence()
    with pytest.raises(ValueError, match='outside'): e.add(['accounting_gate', '0', 'cell', 'x'])
    e.add(['header', 'schema','1','kind','c1_read_accounting_topology',
           'checkpoint_sha256','a'*64,'checkpoint','/observed','route_qualified','0','hardware_measured','0'])
    e.add(['accounting_gate', '0', 'cell', 'x'])
    with pytest.raises(ValueError, match='duplicate'): e.add(['accounting_gate', '0', 'cell', 'x'])
    with pytest.raises(ValueError, match='unknown'): e.add(['accounting_unreviewed', '0', 'cell', 'x'])
    with pytest.raises(ValueError, match='inventory'): A.facts(e)


@pytest.mark.parametrize('change', ['helper', 'checkpoint', 'query_count', 'incomplete', 'claim'])
def test_envelope_mutations_fail_before_any_semantic_claim(tmp_path, change):
    text = ('header schema 1 kind c1_read_accounting_topology checkpoint_sha256 ' + 'a'*64 +
            ' checkpoint /observed route_qualified 0 hardware_measured 0\n' +
            'accounting_sources helper_sha256 ' + 'b'*64 + ' native_helper_sha256 ' + 'c'*64 + '\n' +
            'summary queries 0 incomplete_or_error 0 output_byte_cap 0 route_qualified 0 hardware_measured 0\n')
    if change == 'helper': text = text.replace('helper_sha256 ' + 'b'*64, 'helper_sha256 ' + 'd'*64)
    elif change == 'checkpoint': text = text.replace('checkpoint_sha256 ' + 'a'*64, 'checkpoint_sha256 ' + 'd'*64)
    elif change == 'query_count': text = text.replace('queries 0', 'queries 1')
    elif change == 'incomplete': text = text.replace('incomplete_or_error 0', 'incomplete_or_error 1')
    elif change == 'claim': text = text.replace('route_qualified 0', 'route_qualified 1')
    raw = tmp_path / 'raw'; raw.write_text(text)
    result = A.review_artifact(topology_path=raw, topology_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        checkpoint_sha256='a'*64, helper_sha256='b'*64, native_helper_sha256='c'*64)
    assert result['issues'] and not result['arithmetic_proven'] and not result['route_qualified']


def test_counter_proof_cycle_and_missing_node_are_errors():
    for cycle in (False, True):
        e, f = counter_fixture(); driver = N.ROOT + 'u_native0/one/P'
        if not cycle: del f['nodes'][driver]
        else:
            node = f['nodes'][driver]; node['reference'] = 'LUT1'
            cell = driver.rsplit('/', 1)[0]; e.objects['cell', cell]['INIT'] = "2'h2"
            target = cell + '/I0'
            e.objects['pin', target] = {'NAME': target, 'DIRECTION': 'IN', 'REF_PIN_NAME': 'I0', 'IS_INVERTED': '0'}
            e.queries['select', 'pin', target] = [target]
            e.queries['input_nets', target] = ['cycle']
            e.queries['input_drivers', target, 'cycle'] = [driver]
            e.queries['input_ports', target, 'cycle'] = []
            node['inputs'] = {target: dict(scope='selected_input_and_drivers', input=target, properties={}, nets=['cycle'], drivers=[driver], ports=[])}
        with pytest.raises(ValueError): A.prove(e, f)
