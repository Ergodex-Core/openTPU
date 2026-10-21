"""Execute the carry guard around simulated physical changes, not Vivado."""
import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('carry_design_model', ROOT / 'tests/test_c1_complete_carry_repair.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
m = model.m

EXTEND = r'''
# Remove both forward and reverse topology so mutation tests cannot rely on a
# stale mock edge to disguise an actually disconnected carry-chain endpoint.
proc disconnect_pin {pin} {
  if {[dict exists $::pins $pin NET]} {
    set net [dict get $::pins $pin NET]
    set members [dict get $::nets $net]
    set at [lsearch -exact $members $pin]
    if {$at<0} {error "Incoherent mock reverse edge"}
    dict set ::nets $net [lreplace $members $at $at]
    dict unset ::pins $pin NET
  }
}
proc attach_pin {pin net} {
  disconnect_pin $pin
  dict set ::pins $pin NET $net
  dict lappend ::nets $net $pin
}
# Complete the independent finite CARRY8 interface model to all34 primitive pins.
foreach name $all {
  # Give each cell a shared16-load input net. This retains shared-net queries
  # without making the finite stub scan thousands of unrelated loads per pin.
  set input_owner inputs/$name
  dict set cells $input_owner [dict create NAME $input_owner REF_NAME GND]
  set input_driver [addpin $input_owner G OUT]
  connect $input_owner [list $input_driver]
  foreach family {DI S CO O} {
    for {set bit 0} {$bit<8} {incr bit} {
      set suffix [format {%s[%d]} $family $bit];set p $name/$suffix
      set direction [expr {$family in {DI S} ? "IN" : "OUT"}]
      if {![dict exists $pins $p]} {addpin $name $suffix $direction}
      if {$direction eq "IN"} {
        attach_pin $p $input_owner
      } elseif {![dict exists $pins $p NET]} {connect own/$p [list $p]}
    }
  }
}
foreach pin @EMPTY_INPUTS@ {disconnect_pin $pin}
set witness_terminal {WRAPPER/CL/u_board/core/cluster/engines[15].engine/sums_reg[0][0][11]_i_18}
dict for {p properties} $pins {
  dict set pins $p IS_INVERTED 0
  dict set pins $p IS_CLOCK 0
}
'''


def execute(tmp_path, mutation='', before_mutation=''):
    complete = m.complete_carry_pblock_script()
    chains = complete.split('set chains ', 1)[1].split('\n', 1)[0]
    prior = complete.split('set prior ', 1)[1].split('\n', 1)[0]
    stub = model.STUB.replace('@CHAINS@', chains).replace('@PRIOR@', prior).replace('@ATTACK@', 'none')
    empty_inputs = '{' + ' '.join('{' + pin + '}' for pin in m.carry_observed_empty_inputs()) + '}'
    stub = stub.replace('proc get_property {property objects}',
        EXTEND.replace('@EMPTY_INPUTS@', empty_inputs) + '\nproc get_property {property objects}')
    stub = stub.replace('lappend result [dict get $::pins $p NET]',
        'if {[dict exists $::pins $p NET]} {lappend result [dict get $::pins $p NET]}')
    stub = stub.replace('proc list_property {object} {return [dict keys [dict get $::cells [lindex $object 0]]]}',
        'proc list_property {object} {foreach table {cells pins blocks} {if {[dict exists [set ::$table] [lindex $object 0]]} {return [dict keys [dict get [set ::$table] [lindex $object 0]]]}};error "unknown object"}')
    before, after = m.carry_physopt_preservation_scripts()
    script = 'set run {' + str(tmp_path) + '}\n' + stub + '\nset code [catch {\n' + complete
    script += '\n' + before_mutation + '\n' + before + '\n' + mutation + '\n' + after
    script += '\n} message options]\nif {$code} {puts stderr [dict get $options -errorinfo];exit 1}\n'
    path = tmp_path / 'audit.tcl'
    path.write_text(script)
    return subprocess.run([shutil.which('tclsh') or 'tclsh', str(path)], capture_output=True, text=True)


def test_all349_preserved_with_placement_movement(tmp_path):
    result = execute(tmp_path, 'dict set cells $target LOC SLICE_X10Y20\ndict set cells $target BEL CARRY8')
    assert result.returncode == 0, result.stderr
    raw = (tmp_path / 'physopt_carry_preservation.tcldata').read_text()
    assert raw.count('\nbefore ') == raw.count('\nafter ') == 349
    assert 'carry_topology_and_configuration_preserved 1' in raw
    assert raw.count('nets {} drivers {} carry_successors {}') == 2 * 1690
    assert 'baseline_complete {cells 349 observed_empty_inputs 1690 exact_empty_input_witness_match 1' in raw
    assert 'semantic_acceptance 0' in raw
    assert m.CARRY_EMPTY_INPUT_WITNESS['raw_sha256'] in raw
    assert m.CARRY_EMPTY_INPUT_WITNESS['stage_receipt_sha256'] in raw
    assert 'placement_movement_permitted 1 route_qualified 0 hardware_actions 0' in raw


@pytest.mark.parametrize('mutation', [
    'dict unset cells $target',
    'dict set cells $target REF_NAME LUT6',
    'dict set cells $target CARRY_TYPE DUAL_CY4',
    'dict set cells $target PBLOCK pblock_CL',
    'dict set cells $target PARENT WRAPPER/CL/changed',
    'dict set cells $target DONT_TOUCH 1',
    'dict set cells $target IS_LOC_FIXED 1',
    'dict set cells $target LOCK_PINS changed',
    'attach_pin $target/CI ground_top',
    'dict unset pins $target/S\\[0\\]',
    'dict set pins $target/S\\[0\\] REF_PIN_NAME {DI[0]}',
    'dict set pins $target/S\\[0\\] DIRECTION OUT',
    'dict set pins $target/S\\[0\\] IS_INVERTED 1',
    'dict set pins $target/S\\[0\\] IS_CLOCK 1',
    'dict unset pins $target/S\\[0\\] IS_INVERTED',
    'set net [dict get $pins $target/CO\\[7\\] NET];dict lappend nets $net $target/CI',
    'set net [dict get $pins $target/CO\\[7\\] NET];dict lappend nets $net WRAPPER/CL/GND/G',
    'dict set blocks pblock_CL_SLR0 GRID_RANGES changed',
    'dict set blocks pblock_CL_SLR0 IS_SOFT 1',
    'dict set blocks pblock_CL_SLR0 PARENT ROOT',
    'set extra WRAPPER/CL/engines_extra/carry;dict set cells $extra [dict replace [dict get $cells $target] NAME $extra];lappend promoted $extra',
])
def test_physical_pass_cannot_silently_rewrite_carry_graph_or_regions(tmp_path, mutation):
    result = execute(tmp_path, mutation)
    assert result.returncode != 0
    assert 'Physical ' in result.stderr, result.stderr
    raw = tmp_path / 'physopt_carry_preservation.tcldata'
    assert raw.exists() and '\ncomplete ' not in raw.read_text()


@pytest.mark.parametrize('mutation', [
    'dict unset pins $target/DI\\[7\\]',
    'dict unset pins $target/CI IS_CLOCK',
    'dict set pins $target/S\\[0\\] DIRECTION OUT',
])
def test_complete_baseline_interface_is_required_before_physical_pass(tmp_path, mutation):
    result = execute(tmp_path, before_mutation=mutation)
    assert result.returncode != 0
    assert 'Physical carry' in result.stderr


@pytest.mark.parametrize('when', ['before', 'after'])
@pytest.mark.parametrize(('mutation', 'error'), [
    # Empty is permitted only at the exact witnessed terminal DI/S bits.
    (r'disconnect_pin $witness_terminal/DI\[0\]', 'unexpectedly disconnected'),
    (r'disconnect_pin $target/DI\[7\]', 'unexpectedly disconnected'),
    (r'disconnect_pin $target/CI', 'unexpectedly disconnected'),
    (r'disconnect_pin $witness_terminal/CI_TOP', 'unexpectedly disconnected'),
    (r'attach_pin $witness_terminal/DI\[2\] inputs/$witness_terminal', 'became connected'),
    (r'attach_pin $witness_terminal/S\[7\] inputs/$witness_terminal', 'became connected'),
    # Both ends are disconnected; no stale mock successor can hide the break.
    (r'disconnect_pin $target/CO\[7\];disconnect_pin [lindex [lindex $model_chains 0] 1]/CI', 'Physical '),
    (r'dict unset pins $witness_terminal/S\[7\]', 'full pin inventory differs'),
    (r'dict unset pins $witness_terminal/S\[7\] IS_CLOCK', 'pin property unavailable'),
    (r'rename get_nets original_get_nets;proc get_nets {args} {error "injected net API failure"}', 'injected net API failure'),
])
def test_witnessed_empty_topology_is_exact_before_and_after(tmp_path, when, mutation, error):
    result = execute(tmp_path, before_mutation=mutation if when == 'before' else '',
                     mutation=mutation if when == 'after' else '')
    assert result.returncode != 0
    assert error in result.stderr, result.stderr
    raw = tmp_path / 'physopt_carry_preservation.tcldata'
    assert raw.exists() and '\ncomplete ' not in raw.read_text()
    assert ('\nbaseline_complete ' in raw.read_text()) == (when == 'after')


def test_exact_empty_inventory_is_bound_and_first_failed_terminal_matches_observation(monkeypatch):
    pins = m.carry_observed_empty_inputs()
    terminal = 'WRAPPER/CL/u_board/core/cluster/engines[15].engine/sums_reg[0][0][11]_i_18'
    assert [pin for pin in pins if pin.startswith(terminal + '/')] == sorted(
        f'{terminal}/{family}[{bit}]' for family in ('DI', 'S') for bit in range(2, 8))
    assert len(pins) == 1690
    assert sum('/DI[' in pin for pin in pins) == sum('/S[' in pin for pin in pins) == 845
    assert len({pin.rsplit('/', 1)[0] for pin in pins}) == 159
    # A syntactically valid broadened, narrowed, or relocated exception must fail.
    original = m.CARRY_EMPTY_INPUT_BITS
    for replacement in (
        ((original[0][0], (1, 2, 3, 4, 5, 6, 7)),) + original[1:],
        ((original[0][0], (3, 4, 5, 6, 7)),) + original[1:],
        ((original[0][0], original[1][1]), (original[1][0], original[0][1])) + original[2:],
        original[:-1],
    ):
        monkeypatch.setattr(m, 'CARRY_EMPTY_INPUT_BITS', replacement)
        with pytest.raises(RuntimeError, match='inventory'):
            m.carry_physopt_preservation_scripts()


def test_strategy_preserves_existing_optimizer_and_all_non_strategy_text():
    baseline = ('set_param general.maxThreads 4\nopt_design\n'
        'write_checkpoint -force $run/post_opt.dcp\nplace_design -directive Explore\n'
        '  phys_opt_design -directive AggressiveExplore\nwrite_checkpoint -force $run/post_phys_opt.dcp\n'
        'route_design -directive Explore -tns_cleanup\n')
    actual = m.physical_script(baseline, 'align-complete-carry-and-phys-opt', 8)
    before, after = m.carry_physopt_preservation_scripts()
    expected = baseline.replace('general.maxThreads 4', 'general.maxThreads 8')
    expected = expected.replace('write_checkpoint -force $run/post_opt.dcp\n',
        m.CARRY_PBLOCK_SCRIPT + m.complete_carry_pblock_script() + 'write_checkpoint -force $run/post_opt.dcp\n')
    expected = expected.replace('  phys_opt_design -directive AggressiveExplore\n',
        before + '  phys_opt_design -directive AggressiveExplore\n' + after)
    assert actual == expected
    assert 'C1_PRE_ROUTE_PHYS_OPT_SKIPPED' not in actual
    assert actual.index(before) < actual.index('  phys_opt_design') < actual.index(after) < actual.index('route_design -directive')


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong', 'wrong_order'])
def test_strategy_rejects_changed_optimizer_anchor(mutation):
    optimizer = '  phys_opt_design -directive AggressiveExplore\n'
    checkpoint = 'write_checkpoint -force $run/post_opt.dcp\n'
    body = checkpoint + optimizer
    if mutation == 'missing': body = checkpoint
    elif mutation == 'duplicate': body += optimizer
    elif mutation == 'wrong': body = body.replace('AggressiveExplore', 'Explore')
    elif mutation == 'wrong_order': body = optimizer + checkpoint
    with pytest.raises(RuntimeError, match='anchor|must follow'):
        m.physical_script('set_param general.maxThreads 4\n' + body, 'align-complete-carry-and-phys-opt')
