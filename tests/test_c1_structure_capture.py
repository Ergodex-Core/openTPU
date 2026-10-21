"""Independent raw steps cannot hide checks after a known topology error."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'boards/c1_f2_alex/capture_structure.tcl'
TCL = shutil.which('tclsh')
pytestmark = pytest.mark.skipif(TCL is None, reason='Tcl unavailable')


def run(tmp_path, body):
    script = tmp_path / 'run.tcl'
    script.write_text('source {' + str(SOURCE) + '}\n' + body)
    return subprocess.run([TCL, str(script)], capture_output=True, text=True, timeout=30)


def setup(tmp_path):
    return r'''
set ::c1_structure_capture::out [open {OUTPUT} w]
fconfigure $::c1_structure_capture::out -translation binary -buffering none
set ::c1_structure_capture::audit [chan create write ::c1_structure_capture::sink]
fconfigure $::c1_structure_capture::audit -translation lf -encoding utf-8 -buffering none
'''.replace('OUTPUT', str(tmp_path / 'raw'))


def mocks():
    return r'''
proc ::c1_structure_capture::general {audit} {puts $audit {general_witness current};return current}
proc ::c1_structure_capture::clocks {audit} {puts $audit {clock_witness current};return current}
proc ::c1_structure_capture::unused_clocks {audit} {puts $audit {unused_clock_witness current};return current}
proc ::c1_structure_capture::disabled {audit pc channel} {puts $audit [list disabled_witness $pc $channel];return current}
proc ::c1_alex_routed_ready::host_wready {audit} {error {exact host replica missing}}
proc ::c1_alex::flow {prefix interface signal audit} {
 if {$interface eq "S_N0" && $signal eq "rvalid"} {error {exact relocated native load}}
 puts $audit [list fabric_flow $interface signal $signal]
}
'''


def test_early_host_and_native_errors_preserve_every_later_step(tmp_path):
    result = run(tmp_path, setup(tmp_path) + mocks() + r'''
::c1_structure_capture::body
puts "attempted=$::c1_structure_capture::attempted failures=$::c1_structure_capture::failures"
close $::c1_structure_capture::audit
close $::c1_structure_capture::out
''')
    assert result.returncode == 0, result.stderr
    assert 'attempted=141 failures=2' in result.stdout
    text = (tmp_path / 'raw').read_text()
    assert text.count('step_begin ') == 141
    assert text.count('step_result ') == 141
    assert 'state error message {exact host replica missing}' in text
    assert 'state error message {exact relocated native load}' in text
    assert 'fabric_flow M_H7 signal rready' in text
    assert 'disabled_witness 7 w' in text and 'unused_clock_witness current' in text
    assert '-errorinfo' in text and 'route_qualified 1' not in text


def test_failure_of_general_and_clock_checks_does_not_erase_flow_data(tmp_path):
    result = run(tmp_path, setup(tmp_path) + mocks() + r'''
proc ::c1_structure_capture::general {audit} {error {wrong native topology}}
proc ::c1_structure_capture::clocks {audit} {error {wrong HBM clock}}
::c1_structure_capture::body
puts $::c1_structure_capture::failures
close $::c1_structure_capture::audit
close $::c1_structure_capture::out
''')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '4'
    text = (tmp_path / 'raw').read_text()
    assert 'wrong native topology' in text and 'wrong HBM clock' in text
    assert 'fabric_flow M_H7 signal rready' in text


def test_output_cap_applies_to_unchanged_helper_writes_and_stops_capture(tmp_path):
    result = run(tmp_path, setup(tmp_path) + r'''
set ::c1_structure_capture::max_bytes 100
proc noisy {audit} {puts $audit [string repeat x 1000]}
if {![catch {::c1_structure_capture::step noisy [list noisy $::c1_structure_capture::audit]} message]} {error {cap did not fail}}
puts "$message exhausted=$::c1_structure_capture::exhausted"
catch {close $::c1_structure_capture::audit}
close $::c1_structure_capture::out
''')
    assert result.returncode == 0, result.stderr
    assert 'Structural raw output cap exhausted=1' in result.stdout
    assert (tmp_path / 'raw').stat().st_size <= 100


def test_duplicate_steps_cannot_overwrite_failed_observations(tmp_path):
    result = run(tmp_path, setup(tmp_path) + r'''
::c1_structure_capture::step x {error original}
if {![catch {::c1_structure_capture::step x {return accepted}} message]} {error duplicate}
puts $message
close $::c1_structure_capture::audit
close $::c1_structure_capture::out
''')
    assert result.returncode == 0 and 'Duplicate structural step' in result.stdout
    assert (tmp_path / 'raw').read_text().count('step_result ') == 1


@pytest.mark.parametrize('case', ['bad_digest', 'wrong_version', 'open_design', 'existing_output'])
def test_guards_before_design_open_or_output_creation(tmp_path, case):
    checkpoint = tmp_path / 'source.dcp'; checkpoint.write_bytes(b'fixture')
    out = tmp_path / 'capture'
    if case == 'existing_output': out.mkdir()
    body = r'''
proc ::c1_alex_route::sha256 {path} {return [string repeat a 64]}
proc version {args} {return VERSION}
proc get_designs {args} {return DESIGNS}
proc open_checkpoint {args} {error {MUST NOT OPEN}}
if {![catch {::c1_structure_capture::capture {OUT} {DCP} DIGEST} message]} {error {guard missing}}
puts $message
'''.replace('VERSION', '2024.1' if case == 'wrong_version' else '2025.2').replace('DESIGNS', 'open' if case == 'open_design' else '{}').replace('OUT', str(out)).replace('DCP', str(checkpoint)).replace('DIGEST', 'b'*64 if case == 'bad_digest' else 'a'*64)
    result = run(tmp_path, body)
    assert result.returncode == 0 and 'MUST NOT OPEN' not in result.stdout
    assert not (out / 'structural.tcldata').exists()


@pytest.mark.parametrize('changed', ['checkpoint', 'helper'])
def test_changed_input_prevents_summary_after_otherwise_complete_capture(tmp_path, changed):
    checkpoint = tmp_path / 'source.dcp'; checkpoint.write_bytes(b'fixture')
    out = tmp_path / 'capture'
    body = r'''
set ::closed 0
proc ::c1_alex_route::sha256 {path} {
 if {$::closed && (("CHANGED" eq "checkpoint" && [file tail $path] eq "source.dcp") || ("CHANGED" eq "helper" && [file tail $path] eq "capture_structure.tcl"))} {return [string repeat b 64]}
 return [string repeat a 64]
}
proc version {args} {return 2025.2}
proc get_designs {args} {return {}}
proc set_param {args} {}
proc open_checkpoint {args} {}
proc close_design {} {set ::closed 1}
proc ::c1_structure_capture::body {} {set ::c1_structure_capture::attempted 141}
if {![catch {::c1_structure_capture::capture {OUT} {DCP} DIGEST} message]} {error {changed inputs accepted}}
puts $message
'''.replace('CHANGED', changed).replace('OUT', str(out)).replace('DCP', str(checkpoint)).replace('DIGEST', 'a'*64)
    result = run(tmp_path, body)
    assert result.returncode == 0 and 'changed' in result.stdout.lower()
    text = (out / 'structural.tcldata').read_text()
    assert 'capture_error' in text and '\nsummary ' not in text


@pytest.mark.parametrize('consumer', [False, True])
def test_unused_clock_consumer_is_observed_and_rejected(tmp_path, consumer):
    body = setup(tmp_path) + r'''
proc ::c1_alex_route::clock {args} {return core}
proc get_clocks {args} {return {core unused}}
proc list_property {obj} {return {SOURCE_PINS}}
proc get_property {key obj} {
 if {$key eq "NAME"} {return $obj}
 if {$key eq "PERIOD"} {return 2.5}
 if {$key eq "SOURCE_PINS"} {return WRAPPER/CL/u_core_clk/clk_out2}
 error {unknown property}
}
proc all_registers {args} {return MEMBERS}
::c1_structure_capture::step unused_clocks [list ::c1_structure_capture::unused_clocks $::c1_structure_capture::audit]
puts $::c1_structure_capture::failures
close $::c1_structure_capture::audit
close $::c1_structure_capture::out
'''.replace('MEMBERS', 'forbidden_register' if consumer else '{}')
    result = run(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ('1' if consumer else '0')
    text = (tmp_path / 'raw').read_text()
    assert 'unused_clock unused period 2.5 consumers ' + ('1' if consumer else '0') in text


def test_collector_has_no_constraint_or_design_mutation_commands():
    text = SOURCE.read_text()
    for token in ('set_property', 'set_false_path', 'set_max_delay', 'create_waiver', 'write_checkpoint', 'route_design', 'place_design'):
        assert token not in text


def test_complete_capture_retains_failed_checks_without_qualification(tmp_path):
    checkpoint = tmp_path / 'source.dcp'; checkpoint.write_bytes(b'fixture')
    out = tmp_path / 'capture'
    body = mocks() + r'''
proc ::c1_alex_route::sha256 {path} {return [string repeat a 64]}
proc version {args} {return 2025.2}
proc get_designs {args} {return {}}
proc set_param {args} {}
proc open_checkpoint {args} {}
proc close_design {} {}
puts [::c1_structure_capture::capture {OUT} {DCP} DIGEST]
'''.replace('OUT', str(out)).replace('DCP', str(checkpoint)).replace('DIGEST', 'a'*64)
    result = run(tmp_path, body)
    assert result.returncode == 0, result.stderr
    text = (out / 'structural.tcldata').read_text()
    assert 'kind c1_structural_scope_capture' in text
    assert 'summary attempted 141 observed 139 failed 2 labels' in text
    assert 'route_qualified 0 hardware_measured 0' in result.stdout
    assert 'route_qualified 1' not in text


@pytest.mark.parametrize('kind', ['gnd', 'vcc', 'multiple', 'escape', 'missing', 'anchor_inverted', 'driver_inverted', 'wrong_pin'])
def test_disabled_write_check_requires_exact_internal_ground_driver(tmp_path, kind):
    body = setup(tmp_path) + r'''
proc ::c1_alex::exact_pin {name} {return target}
proc get_property {key object} {
 if {$key eq "DIRECTION"} {return [expr {$object eq "target" ? "IN" : "OUT"}]}
 if {$key eq "NAME"} {return $object}
 if {$key eq "REF_NAME"} {return PRIMITIVE}
 if {$key eq "IS_INVERTED"} {return [expr {$object eq "target" ? ANCHOR_INVERTED : DRIVER_INVERTED}]}
 if {$key eq "REF_PIN_NAME"} {return [expr {$object eq "target" ? "s_axi_wvalid" : "DRIVER_PIN"}]}
 error unknown_property
}
proc get_nets {args} {return NETS}
proc get_ports {args} {return PORTS}
proc get_pins {args} {return DRIVERS}
proc get_cells {args} {return constant}
::c1_structure_capture::step disabled [list ::c1_structure_capture::disabled $::c1_structure_capture::audit 7 w]
puts $::c1_structure_capture::failures
close $::c1_structure_capture::audit
close $::c1_structure_capture::out
'''.replace('PRIMITIVE', 'VCC' if kind == 'vcc' else 'GND').replace('NETS', '{}' if kind == 'missing' else 'network').replace('PORTS', 'top_escape' if kind == 'escape' else '{}').replace('DRIVERS', '{first second}' if kind == 'multiple' else 'constant/G').replace('ANCHOR_INVERTED', '1' if kind == 'anchor_inverted' else '0').replace('DRIVER_INVERTED', '1' if kind == 'driver_inverted' else '0').replace('DRIVER_PIN', 'P' if kind == 'wrong_pin' else 'G')
    result = run(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ('0' if kind == 'gnd' else '1')
