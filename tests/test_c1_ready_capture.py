"""Typed-boundary and scope tests for raw C1 supplemental ready capture."""
import importlib.util
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_capture_mock',ROOT/'tests/test_c1_native_capture.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def run(tmp_path,body):
    text='source {'+str(ROOT/'boards/c1_f2_alex/capture_ready.tcl')+'}\n'+M.MOCK+r'''
lappend ::pin_names cl/reset/D cl/reset/CE cl/reset/R
rename get_nets original_get_nets
proc get_nets {args} {
    if {"-segments" in $args} {error "Global net segments forbidden"}
    set i [lsearch -exact $args -of_objects]
    if {$i>=0} {
        set pin [typed [lindex $args [expr {$i+1}]] pin]
        if {$pin in {cl/reset/D cl/reset/CE cl/reset/R}} {return [token net cl/data]}
    }
    return [original_get_nets {*}$args]
}
rename get_pins original_get_pins
proc get_pins {args} {
    if {"-leaf" in $args && "-filter" ni $args} {
        set i [lsearch -exact $args -of_objects]
        if {[typed [lindex $args [expr {$i+1}]] net] eq "cl/data"} {error "Global GND sinks forbidden"}
    }
    return [original_get_pins {*}$args]
}
'''
    text+='\nset ::c1_native_capture::out [open {'+str(tmp_path/'raw')+'} w]\n'+body+'\nclose $::c1_native_capture::out\n'
    script=tmp_path/'test.tcl';script.write_text(text)
    result=subprocess.run([shutil.which('tclsh'),str(script)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return (tmp_path/'raw').read_text()


def test_fdre_controls_clock_and_full_output_use_fresh_typed_handles(tmp_path):
    text=run(tmp_path,r'''
::c1_ready_capture::register host_wready cl/reset
if {$::c1_native_capture::gaps} {error "Unexpected collection gap"}
''')
    assert 'ready_clock host_wready cell cl/reset' in text
    for role in ('D','CE','R'):
        assert f'ready_control host_wready cell cl/reset role {role}' in text
        assert f'ready_walk {{control host_wready cl/reset {role}}}' in text
    assert 'network pin cl/reset/Q' in text
    assert 'clock_scope names_and_properties_only' in text
    assert 'property pin cl/reset/C IS_INVERTED status observed value 0' in text
    assert 'incomplete' not in text


def test_control_walk_stops_at_sequential_state_without_expanding_clock_sinks(tmp_path):
    text=run(tmp_path,'::c1_ready_capture::walk test {cl/gate/O} 0')
    assert 'ready_terminal test output cl/reset/Q' in text
    assert 'ready_register ' not in text
    assert 'ready_edge test cell cl/gate input cl/gate/I1' in text
    assert 'incomplete' not in text


def test_ready_scope_excludes_global_core_reset_register():
    text=(ROOT/'boards/c1_f2_alex/capture_ready.tcl').read_text()
    assert 'network pin $output 0' in text
    assert 'if {[state_scope $owner]}' in text
    assert 'name_contains_ready' in text


def test_explicit_bounds_and_missing_root_are_not_qualification(tmp_path):
    text=run(tmp_path,r'''
set ::c1_ready_capture::max_nodes 0
::c1_ready_capture::walk bounded {cl/gate/O}
::c1_ready_capture::root missing no/such/pin
''')
    assert 'node_cap' in text and 'missing_pin' in text
    assert 'route_qualified 1' not in text
