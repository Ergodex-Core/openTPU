"""The additive admission collector records state closure without acceptance."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / 'boards/c1_f2_alex/capture_accounting_boundary.tcl'
spec = importlib.util.spec_from_file_location('boundary_mock', ROOT / 'tests/test_c1_native_capture.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def run(tmp_path, body, mock=M.MOCK):
    path = tmp_path / 'test.tcl'; raw = tmp_path / 'raw'
    path.write_text('source {' + str(CAPTURE) + '}\n' + mock +
                   '\nset ::c1_native_capture::out [open {' + str(raw) + '} w]\n' + body +
                   '\nclose $::c1_native_capture::out\n')
    result = subprocess.run([shutil.which('tclsh'), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return raw.read_text()


def test_typed_complete_outputs_and_explicit_cut_are_recorded(tmp_path):
    text = run(tmp_path, r'''
::c1_accounting_boundary::walk 0 cl/gate/O {cl/reset/Q rst}
if {$::c1_native_capture::gaps} {error "unexpected typed capture gap"}
''')
    assert 'admission_output 0 output cl/gate/O network' in text
    assert 'admission_cut 0 output cl/reset/Q cell cl/reset reference FDRE roles rst' in text
    assert 'admission_constant 0 output cl/const/G' in text
    assert 'admission_cell 0 cell cl/gate reference LUT2 sequential 0' in text
    assert 'incomplete' not in text


STATE = 'WRAPPER/CL/u_native0/state'
STATE_MOCK = M.MOCK.replace('cl/reset', STATE)
EXTRA_PINS = r'''
foreach role {D CE R A0 A1 DPRA0 DPRA1 WE CLK} {lappend ::pin_names STATE/$role}
rename get_nets original_get_nets
proc get_nets {args} {
 set i [lsearch -exact $args -of_objects]
 if {$i>=0} {
  set pin [typed [lindex $args [expr {$i+1}]] pin]
  if {[file dirname $pin] eq "STATE" && [file tail $pin] ni {C Q}} {return [token net cl/data]}
 }
 return [original_get_nets {*}$args]
}
'''.replace('STATE', STATE)


def test_fdre_inputs_and_feedback_expand_instead_of_ending_at_state(tmp_path):
    text = run(tmp_path, EXTRA_PINS + r'''
::c1_accounting_boundary::walk 0 cl/gate/O {}
if {$::c1_native_capture::gaps} {error "unexpected state closure gap"}
''', STATE_MOCK)
    assert f'admission_cell 0 cell {STATE} reference FDRE sequential 1' in text
    for role in ('C', 'D', 'CE', 'R'):
        assert f'admission_edge 0 cell {STATE} input {STATE}/{role}' in text
    assert f'registers {STATE} memories {{}}' in text
    assert 'admission_cut 0 output ' + STATE + '/Q' not in text


@pytest.mark.parametrize('reference', ['RAMD32', 'RAMS32'])
def test_ram_read_address_write_address_data_enable_clock_are_all_followed(tmp_path, reference):
    body = EXTRA_PINS + r'''
rename get_property original_get_property
proc get_property {key object} {
 if {$object eq "@cell|STATE" && $key in {REF_NAME ORIG_REF_NAME}} {return REFERENCE}
 return [original_get_property $key $object]
}
::c1_accounting_boundary::walk 0 cl/gate/O {}
if {$::c1_native_capture::gaps} {error "unexpected RAM closure gap"}
'''.replace('STATE', STATE).replace('REFERENCE', reference)
    text = run(tmp_path, body, STATE_MOCK)
    assert f'admission_cell 0 cell {STATE} reference {reference} sequential 1' in text
    for role in ('A0', 'A1', 'DPRA0', 'DPRA1', 'D', 'WE', 'CLK'):
        assert f'admission_edge 0 cell {STATE} input {STATE}/{role}' in text
    assert f'registers {{}} memories {STATE}' in text
    assert f'property cell {STATE} IS_CLK_INVERTED status unavailable' in text


def test_unknown_external_state_is_a_gap_not_an_extra_implicit_input(tmp_path):
    text = run(tmp_path, '::c1_accounting_boundary::walk 0 cl/gate/O {}')
    assert 'reason state_outside_native_scope cell cl/reset reference FDRE' in text
    assert 'admission_cut 0 output cl/reset/Q' not in text


@pytest.mark.parametrize('reference', ['RAMB36E2', 'SRL16E', 'FDSE'])
def test_unreviewed_state_primitives_remain_gaps(tmp_path, reference):
    body = r'''
rename get_property original_get_property
proc get_property {key object} {
 if {$object eq "@cell|STATE" && $key in {REF_NAME ORIG_REF_NAME}} {return REFERENCE}
 return [original_get_property $key $object]
}
::c1_accounting_boundary::walk 0 cl/gate/O {}
'''.replace('STATE', STATE).replace('REFERENCE', reference)
    text = run(tmp_path, body, STATE_MOCK)
    assert f'reason unsupported_state cell {STATE} reference {reference}' in text


def test_missing_node_depth_and_count_limits_remain_gaps(tmp_path):
    text = run(tmp_path, r'''
::c1_accounting_boundary::walk 0 missing/O {}
set ::c1_accounting_boundary::max_nodes 1
::c1_accounting_boundary::walk 0 cl/gate/O {cl/reset/Q rst}
set ::c1_accounting_boundary::max_nodes 20
set ::c1_accounting_boundary::max_depth 0
::c1_accounting_boundary::walk 0 cl/gate/O {cl/reset/Q rst}
''')
    for reason in ('missing_output', 'node_cap', 'depth_cap'):
        assert 'reason ' + reason in text
    assert 'route_qualified 1' not in text


@pytest.mark.parametrize('bad', ['drivers {} ports {}', 'drivers {first/Q second/Q} ports {}',
                                 'drivers first/Q ports external', '{}'])
def test_edges_never_accept_missing_multiple_or_external_drivers(tmp_path, bad):
    text = run(tmp_path, r'''
proc ::c1_accounting_boundary::input_drivers {pin} {return {BAD}}
::c1_accounting_boundary::walk 0 cl/gate/O {}
'''.replace('BAD', bad))
    assert 'reason missing_ambiguous_or_external_input' in text


def test_interface_role_inventory_and_aliases_are_explicit(tmp_path):
    text = run(tmp_path, r'''
proc ::c1_accounting_boundary::net_drivers {net} {return {drivers observed/Q ports {}}}
proc ::c1_accounting_boundary::properties {args} {return {}}
proc ::c1_accounting_boundary::qnames {args} {return {}}
proc ::c1_accounting_boundary::owners {args} {return observed}
proc ::c1_accounting_boundary::cell {args} {return {REF_NAME FDRE}}
set interface [::c1_accounting_boundary::interfaces 0]
if {![dict get $interface complete] || [llength [dict get $interface cuts observed/Q]]!=30} {error "interface role inventory"}
''', '')
    assert text.count('admission_interface 0 role ') == 30
    assert 'n_caddr[24]' in text and 'm_arready' in text
    assert 'admission_cut_inventory 0 complete 1 outputs {observed/Q' in text


def test_constants_stay_constants_even_when_interface_roles_alias_them(tmp_path):
    text = run(tmp_path, '::c1_accounting_boundary::walk 0 cl/const/G {cl/const/G {n_cwe n_wvalid}}')
    assert 'admission_constant' in text and 'admission_cut 0' not in text


@pytest.mark.parametrize('reference, gap', [('FDRE', True), ('LUT1', True), ('GND', False)])
def test_interface_cannot_hide_internal_nonconstant_state_or_logic(tmp_path, reference, gap):
    body = r'''
proc ::c1_accounting_boundary::net_drivers {net} {return {drivers WRAPPER/CL/u_native0/internal/Q ports {}}}
proc ::c1_accounting_boundary::properties {args} {return {}}
proc ::c1_accounting_boundary::qnames {args} {return {}}
proc ::c1_accounting_boundary::owners {args} {return WRAPPER/CL/u_native0/internal}
proc ::c1_accounting_boundary::cell {args} {return {REF_NAME REFERENCE}}
::c1_accounting_boundary::interfaces 0
'''.replace('REFERENCE', reference)
    text = run(tmp_path, body, '')
    assert ('reason internal_nonconstant_interface_driver' in text) is gap


def test_empty_selected_cell_input_inventory_is_a_gap(tmp_path):
    text = run(tmp_path, r'''
rename get_pins original_get_pins
proc get_pins {args} {
 if {"-filter" in $args && "-of_objects" in $args && [lindex $args end] eq "DIRECTION == IN"} {return {}}
 return [original_get_pins {*}$args]
}
::c1_accounting_boundary::walk 0 cl/gate/O {}
''')
    assert 'reason missing_input_output_inventory cell cl/gate' in text


def test_wrong_checkpoint_hash_cannot_open_design_or_create_output(tmp_path):
    dcp = tmp_path / 'source.dcp'; dcp.write_bytes(b'fixture')
    text = run(tmp_path, r'''
proc ::c1_native_capture::hash {p} {return [string repeat a 64]}
proc open_checkpoint {p} {error "MUST NOT OPEN"}
if {![catch {::c1_accounting_boundary::capture {OUTPUT} {DCP} [string repeat b 64]} msg] || $msg ne "Checkpoint binding mismatch"} {error "binding guard"}
'''.replace('OUTPUT', str(tmp_path/'out')).replace('DCP', str(dcp)), '')
    assert text == '' and not (tmp_path/'out').exists()


def test_collector_has_no_constraint_or_checkpoint_mutation_commands():
    text = CAPTURE.read_text()
    for command in ('set_property', 'create_clock', 'create_pblock', 'resize_pblock',
                    'write_checkpoint', 'place_design', 'route_design', 'phys_opt_design'):
        assert command not in text


def test_missing_interface_inventory_does_not_begin_any_cone(tmp_path):
    text = run(tmp_path, r'''
proc ::c1_accounting_boundary::net_drivers {net} {gap $net missing_interface_net;return {}}
proc ::c1_accounting_boundary::qnames {args} {return {}}
proc ::c1_accounting_boundary::walk {args} {error "MUST NOT WALK INCOMPLETE BOUNDARY"}
::c1_accounting_boundary::body
''', '')
    assert text.count('reason incomplete_interface_inventory') == 8
    assert 'admission_root' not in text


def test_unsupported_clock_and_external_state_do_not_enumerate_output_fanout(tmp_path):
    text = run(tmp_path, r'''
rename ::c1_accounting_boundary::network original_network
proc ::c1_accounting_boundary::network {kind name args} {
 if {$name eq "cl/reset/Q"} {error "MUST NOT ENUMERATE EXTERNAL FANOUT"}
 return [original_network $kind $name {*}$args]
}
::c1_accounting_boundary::walk 0 cl/gate/O {}
rename get_property original_get_property
proc get_property {key object} {
 if {$object eq "@cell|cl/reset" && $key in {REF_NAME ORIG_REF_NAME}} {return BUFGCE}
 if {$object eq "@cell|cl/reset" && $key eq "IS_SEQUENTIAL"} {return 0}
 return [original_get_property $key $object]
}
array unset ::c1_native_capture::seen_cells
array unset ::c1_native_capture::seen_properties
::c1_accounting_boundary::walk 0 cl/reset/Q {}
''')
    assert 'reason state_outside_native_scope' in text
    assert 'reason unsupported_combinational cell cl/reset reference BUFGCE' in text
    assert 'admission_output 0 output cl/reset/Q' not in text


def test_exact_interface_net_names_follow_source_roles(tmp_path):
    text = run(tmp_path, r'''
foreach {role expected} {clk WRAPPER/CL/core_clk rst WRAPPER/CL/core_rst n_cvalid {WRAPPER/CL/n_cvalid[7]} n_cwe {WRAPPER/CL/n_cwe[7]} n_wvalid {WRAPPER/CL/n_wvalid[7]} {n_caddr[24]} {WRAPPER/CL/n_caddr[7][24]} m_arready WRAPPER/CL/n7_arready m_awready WRAPPER/CL/n7_awready m_wready WRAPPER/CL/n7_wready m_bvalid WRAPPER/CL/n7_bvalid m_rvalid WRAPPER/CL/n7_rvalid} {
 if {[::c1_accounting_boundary::interface_net 7 $role] ne $expected} {error "interface mapping changed"}
}
if {![catch {::c1_accounting_boundary::interface_net 7 other}]} {error "unrecognized role accepted"}
''', '')
    assert text == ''


def test_net_driver_query_is_typed_and_does_not_enumerate_consumers(tmp_path):
    text = run(tmp_path, r'''
rename get_pins original_get_pins
proc get_pins {args} {
 if {"-leaf" in $args && "-filter" ni $args} {error "MUST NOT ENUMERATE GLOBAL CONSUMERS"}
 return [original_get_pins {*}$args]
}
set n [::c1_accounting_boundary::net_drivers cl/rst]
if {[dict get $n drivers] ne {cl/reset/Q} || [dict get $n net] ne "cl/rst"} {error "missing exact net binding"}
::c1_accounting_boundary::net_drivers absent
''')
    assert 'reason missing_interface_net' in text
    assert 'query {admission_net_drivers cl/rst}' in text


def test_absent_roles_are_observed_but_never_added_as_input_or_constant_cuts(tmp_path):
    text = run(tmp_path, r'''
proc ::c1_accounting_boundary::net_drivers {net} {return {drivers observed/Q ports {}}}
proc ::c1_accounting_boundary::qnames {args} {return {}}
proc ::c1_accounting_boundary::properties {args} {return {}}
proc ::c1_accounting_boundary::owners {args} {return observed}
proc ::c1_accounting_boundary::cell {args} {return {REF_NAME FDRE}}
set boundary [::c1_accounting_boundary::interfaces 0]
if {![dict get $boundary complete] || [llength [dict get $boundary cuts observed/Q]]!=30} {error "live boundary"}
foreach absent {n_cwe n_wvalid m_awready m_wready {n_caddr[24]}} {
 if {$absent in [dict get $boundary cuts observed/Q]} {error "fabricated missing input"}
}
''', '')
    assert text.count('admission_uncut_role 0 role ') == 5
    assert text.count('observed {} disposition no_assumed_input_or_constant') == 5
    assert 'admission_constant' not in text


def test_unexpected_retained_uncut_role_stays_a_gap(tmp_path):
    text = run(tmp_path, r'''
proc ::c1_accounting_boundary::net_drivers {net} {return {drivers observed/Q ports {}}}
proc ::c1_accounting_boundary::qnames {label script} {
 if {[lindex $label 0] eq "admission_uncut_selection"} {return retained_net}
 return {}
}
proc ::c1_accounting_boundary::properties {args} {return {}}
proc ::c1_accounting_boundary::owners {args} {return observed}
proc ::c1_accounting_boundary::cell {args} {return {REF_NAME FDRE}}
set boundary [::c1_accounting_boundary::interfaces 0]
if {[dict get $boundary complete]} {error "retained uncut role accepted"}
''', '')
    assert text.count('reason unexpected_retained_uncut_role') == 5
