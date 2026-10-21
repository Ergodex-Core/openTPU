"""Read-only accounting collector boundaries with typed Vivado mock objects."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('accounting_capture_mock',ROOT/'tests/test_c1_native_capture.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def run(tmp_path,body,*,mock=True):
    script='source {'+str(ROOT/'boards/c1_f2_alex/capture_accounting.tcl')+'}\n'+(M.MOCK if mock else '')
    script+='\nset ::c1_native_capture::out [open {'+str(tmp_path/'raw')+'} w]\n'+body+'\nclose $::c1_native_capture::out\n'
    path=tmp_path/'test.tcl';path.write_text(script)
    result=subprocess.run([shutil.which('tclsh'),str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return (tmp_path/'raw').read_text()


def test_backward_accounting_cone_preserves_typed_inputs_without_global_fanout(tmp_path):
    text=run(tmp_path,r'''
rename get_nets original_get_nets
proc get_nets {args} {
 if {"-segments" in $args} {error "global segments forbidden"}
 return [original_get_nets {*}$args]
}
rename get_pins original_get_pins
proc get_pins {args} {
 if {"-leaf" in $args && "-filter" ni $args} {error "global sink enumeration forbidden"}
 return [original_get_pins {*}$args]
}
::c1_accounting_capture::walk accounting {cl/gate/O}
if {$::c1_native_capture::gaps} {error "unexpected typed query gap"}
''')
    assert 'accounting_edge accounting cell cl/gate input cl/gate/I0' in text
    assert 'accounting_edge accounting cell cl/gate input cl/gate/I1' in text
    assert 'accounting_terminal accounting output cl/reset/Q cell cl/reset reference FDRE' in text
    assert 'accounting_terminal accounting output cl/const/G cell cl/const reference GND' in text
    assert 'incomplete' not in text


def test_resource_and_missing_output_failures_remain_raw_gaps(tmp_path):
    text=run(tmp_path,r'''
set ::c1_accounting_capture::max_nodes 1
::c1_accounting_capture::walk node_bound {cl/gate/O}
set ::c1_accounting_capture::max_nodes 10
set ::c1_accounting_capture::max_depth 0
::c1_accounting_capture::walk depth_bound {cl/gate/O}
::c1_accounting_capture::walk missing {no/such/pin}
''')
    assert 'reason node_cap' in text
    assert 'reason depth_cap' in text
    assert 'reason missing_output' in text
    assert 'route_qualified 1' not in text


def test_ambiguous_or_top_port_inputs_are_never_silently_followed(tmp_path):
    text=run(tmp_path,r'''
proc ::c1_accounting_capture::input_drivers {name} {
 return [dict create drivers {cl/const/G cl/reset/Q} ports external]
}
::c1_accounting_capture::walk ambiguous {cl/gate/O}
''')
    assert 'missing_ambiguous_or_external_input' in text
    assert 'accounting_terminal' not in text


def test_checkpoint_mismatch_cannot_open_or_create_output(tmp_path):
    dcp=tmp_path/'source.dcp';dcp.write_bytes(b'fixture')
    text=run(tmp_path,r'''
proc ::c1_native_capture::hash {p} {return [string repeat a 64]}
proc open_checkpoint {p} {error "MUST NOT OPEN"}
if {![catch {::c1_accounting_capture::capture {'''+str(tmp_path/'out')+'} {'+str(dcp)+r'''} [string repeat b 64]} msg] || $msg ne "Checkpoint binding mismatch"} {error "wrong binding guard"}
''',mock=False)
    assert not (tmp_path/'out').exists() and text==''


@pytest.mark.parametrize('role',['I0','I1','C','CE','R','D'])
@pytest.mark.parametrize('bad',['drivers {} ports {}','drivers {first/Q second/Q} ports {}',
                                'drivers first/Q ports external'])
def test_every_top_level_input_requires_one_internal_driver(tmp_path,role,bad):
    body=r'''
proc ::c1_accounting_capture::cell {name} {return {}}
proc ::c1_accounting_capture::network {args} {return {}}
proc ::c1_accounting_capture::qnames {label script} {
 set regs {};set pc [lindex $label 1]
 for {set i 0} {$i<8} {incr i} {lappend regs WRAPPER/CL/u_native${pc}/reads_reg$i}
 return $regs
}
proc ::c1_accounting_capture::walk {label drivers} {emit followed $label}
proc ::c1_accounting_capture::input_drivers {name} {
 if {[file tail $name] eq "ROLE"} {return {BAD}}
 return {drivers upstream/Q ports {}}
}
::c1_accounting_capture::body
'''.replace('ROLE',role).replace('BAD',bad)
    text=run(tmp_path,body,mock=False)
    assert 'missing_ambiguous_or_external_input' in text
    if role in ('I0','I1'):
        assert f'incomplete {{accounting_gate_input 0 {role}}}' in text
        assert f'followed {{gate 0 {role}}}' not in text
    else:
        assert f'incomplete {{accounting_control 0 WRAPPER/CL/u_native0/reads_reg0 {role}}}' in text
        assert f'followed {{control 0 WRAPPER/CL/u_native0/reads_reg0 {role}}}' not in text
