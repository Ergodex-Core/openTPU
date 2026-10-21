"""Coherent inventory corruption must not enter a ready-function proof."""
import pytest
from boards.c1_f2_alex import ready_review as R


def primitive(reference):
    evidence = R.Evidence()
    cell = 'cl/ready'
    inputs = ['C', 'CE', 'D', 'R'] if reference == 'FDRE' else [f'I{i}' for i in range(int(reference[-1]))]
    output = 'Q' if reference == 'FDRE' else 'O'
    pins = [cell + '/' + role for role in inputs + [output]]
    evidence.cells[cell] = pins
    evidence.queries['cell_pins', cell] = pins.copy()
    for role in inputs + [output]:
        pin = cell + '/' + role
        evidence.queries['select', 'pin', pin] = [pin]
        evidence.queries['pin_owner', pin] = [cell]
        evidence.objects['pin', pin] = dict(NAME=pin, REF_PIN_NAME=role,
            DIRECTION='IN' if role in inputs else 'OUT')
    return evidence, cell


@pytest.mark.parametrize('reference', ['FDRE', 'LUT1', 'LUT2', 'LUT3', 'LUT4', 'LUT5', 'LUT6'])
def test_supported_complete_pin_inventory(reference):
    evidence, cell = primitive(reference)
    inputs, outputs = R.primitive_pins(evidence, cell, reference)
    assert inputs.isdisjoint(outputs)
    assert inputs | outputs == set(evidence.cells[cell])


@pytest.mark.parametrize('reference,role', [('FDRE', 'D'), ('FDRE', 'Q'), ('LUT2', 'I1'), ('LUT2', 'O')])
@pytest.mark.parametrize('mutation', ['missing', 'extra', 'role', 'direction', 'owner', 'query'])
def test_coherent_or_inconsistent_pin_corruption_rejected(reference, role, mutation):
    evidence, cell = primitive(reference)
    pin = cell + '/' + role
    if mutation == 'missing':
        evidence.cells[cell].remove(pin)
        evidence.queries['cell_pins', cell].remove(pin)
    elif mutation == 'extra':
        evidence.cells[cell].append(cell + '/UNKNOWN')
        evidence.queries['cell_pins', cell].append(cell + '/UNKNOWN')
    elif mutation == 'role':
        evidence.objects['pin', pin]['REF_PIN_NAME'] = 'UNKNOWN'
    elif mutation == 'direction':
        row = evidence.objects['pin', pin]
        row['DIRECTION'] = 'OUT' if row['DIRECTION'] == 'IN' else 'IN'
    elif mutation == 'owner':
        evidence.queries['pin_owner', pin] = ['cl/other']
    elif mutation == 'query':
        evidence.queries['cell_pins', cell].remove(pin)
    with pytest.raises(ValueError):
        R.primitive_pins(evidence, cell, reference)
