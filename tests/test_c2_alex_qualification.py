"""Offline qualifier parser/control tests. Fixtures are not FPGA evidence."""
from pathlib import Path
import importlib.util
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUALIFIER = ROOT / 'boards/c2_f2_alex/qualify_c2_alex.tcl'
HISTORICAL = ROOT.parent / 'tpubench/build/alex-burst200/runs/20261015_67_burst_audit'
SC = 'WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/m00_nodes/fifo/gen_async_clocks.inst_cdc_addra_to_rd_clk'
CHECKS = ('no_clock constant_clock pulse_width_clock unconstrained_internal_endpoints '
          'no_input_delay no_output_delay multiple_clock generated_clocks loops '
          'partial_input_delay partial_output_delay latch_loops').split()
TIMING = ('WNS(ns) TNS(ns) TNS Failing Endpoints TNS Total Endpoints WHS(ns) THS(ns) '
          'THS Failing Endpoints THS Total Endpoints WPWS(ns) TPWS(ns) TPWS Failing Endpoints TPWS Total Endpoints\n'
          '------- ------- ----------------\n'
          '0.100 0.000 0 100 0.010 0.000 0 100 0.000 0.000 0 100\n'
          + ''.join(f'{i}. checking {name} (0)\n' for i, name in enumerate(CHECKS, 1)))
ROUTE = ('# of routable nets..................... : 100 :\n'
         '# of fully routed nets................. : 100 :\n'
         '# of nets with routing errors.......... : 0 :\n')
CDC = ('ID Severity Count Description\nCDC-15 Warning 1 Clock enable controlled CDC structure detected\n'
       'Source Clock: core\nDestination Clock: hbm\nCDC Type: Safely Timed\n'
       '1 CDC-15 Warning Clock enable controlled CDC structure detected 0 Max Delay Datapath Only '
       'WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/fifo/gen_xpm_memory_fifo.inst_fifo/memory/CLK '
       'WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/fifo/gen_xpm_memory_fifo.inst_fifo/read/D N\n')


def bus_report(slack='0.125', source_bit='1', destination_bit='1'):
    return ('1. Bus Skew Report Summary\n--------------------------\n'
            'Id Position From To Corner Requirement(ns) Actual(ns) Slack(ns)\n'
            '1 25 [get_cells [list {' + SC + '/src_gray_ff_reg[0]} {' + SC + '/src_gray_ff_reg[' + source_bit + ']}]]\n'
            ' [get_cells [list {' + SC + '/dest_graysync_ff_reg[0][0]} {' + SC + '/dest_graysync_ff_reg[0][' + destination_bit + ']}]]\n'
            f' Slow 2.222 2.097 {slack}\n'
            '2. Bus Skew Report Per Constraint\n---------------------------------\n')


def tcl(tmp_path, script, *arguments):
    executable = shutil.which('tclsh')
    if not executable:
        pytest.skip('Tcl unavailable')
    path = tmp_path / 'driver.tcl'
    path.write_text(script)
    return subprocess.run([executable, str(path), str(QUALIFIER), *map(str, arguments)],
                          text=True, capture_output=True, timeout=30)


def parser(tmp_path, procedure, text):
    data = tmp_path / 'input.txt'
    data.write_text(text)
    return tcl(tmp_path, '''source [lindex $argv 0]
if {[catch {::c2_alex_route::[lindex $argv 1] [::c2_alex_route::read_text [lindex $argv 2]]} result]} {puts stderr $result;exit 1}
puts $result
''', procedure, data)


@pytest.mark.parametrize(('procedure', 'text', 'expected'), [
    ('timing_summary', TIMING, 'blockers {}'),
    ('route_summary', ROUTE, 'pass 1'),
    ('bus_rows', bus_report(), 'slack 0.125'),
    ('cdc_rows', CDC, 'category vendor_fifo_candidate review pending'),
])
def test_current_report_parsers_accept_complete_synthetic_evidence(tmp_path, procedure, text, expected):
    result = parser(tmp_path, procedure, text)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


@pytest.mark.parametrize(('procedure', 'text', 'expected'), [
    ('timing_summary', TIMING.replace('0.100', '-0.001'), 'negative_wns'),
    ('timing_summary', TIMING.replace('0.010', '-0.001'), 'negative_whs'),
    ('timing_summary', TIMING.replace('0.000 0.000 0 100', '-0.001 -0.001 1 100'), 'negative_wpws'),
    ('timing_summary', TIMING.replace('unconstrained_internal_endpoints (0)', 'unconstrained_internal_endpoints (1)'), 'check_timing_unconstrained_internal_endpoints'),
    ('route_summary', ROUTE.replace('nets................. : 100', 'nets................. : 99'), 'pass 0'),
    ('route_summary', ROUTE.replace('errors.......... : 0', 'errors.......... : 1'), 'pass 0'),
])
def test_reported_physical_failures_remain_failures(tmp_path, procedure, text, expected):
    result = parser(tmp_path, procedure, text)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


@pytest.mark.parametrize(('procedure', 'text'), [
    ('timing_summary', TIMING.replace('0.100', 'NaN')),
    ('timing_summary', TIMING.replace('0 100', '0 0')),
    ('timing_summary', TIMING.replace('12. checking latch_loops (0)\n', '')),
    ('timing_summary', TIMING + '1. checking no_clock (1)\n'),
    ('route_summary', ROUTE.replace('# of routable nets', '# of unknown nets')),
    ('bus_rows', bus_report().replace('1 25', '2 25')),
    ('bus_rows', bus_report().replace('2. Bus Skew Report Per Constraint', 'lost detail boundary')),
    ('bus_rows', bus_report(source_bit='0')),
    ('bus_rows', bus_report().replace('slack', 'bad').replace('Slow 2.222 2.097 0.125', 'Slow 2.222 unknown 0.125')),
    ('bus_rows', bus_report().replace('[get_cells [list', '[exec [list', 1)),
    ('cdc_rows', CDC.replace('CDC-15 Warning 1', 'CDC-15 Warning 2')),
    ('cdc_rows', CDC.replace('Destination Clock: hbm\n', '')),
    ('cdc_rows', CDC.replace('/read/D N', '/read/D unknown')),
])
def test_missing_or_ambiguous_evidence_is_rejected(tmp_path, procedure, text):
    result = parser(tmp_path, procedure, text)
    assert result.returncode != 0
    assert result.stderr.strip()


@pytest.mark.parametrize(('procedure', 'filename', 'expected'), [
    ('timing_summary', 'timing.rpt', 'wns 0.032'),
    ('route_summary', 'route_status.rpt', 'routable 272732 routed 272732 errors 0 pass 1'),
    ('bus_rows', 'bus_skew.rpt', 'slack 11.136'),
    ('cdc_rows', 'cdc.rpt', 'vendor_fifo_candidate'),
])
def test_actual_historical_report_format_only_not_current_approval(tmp_path, procedure, filename, expected):
    source = HISTORICAL / filename
    if not source.is_file():
        pytest.skip('Optional historical report format fixture absent')
    result = parser(tmp_path, procedure, source.read_text())
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert 'route_qualified 1' not in result.stdout


def object_mock():
    spec = importlib.util.spec_from_file_location('synth_fixture', ROOT / 'tests/test_c2_alex_audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = module.MOCK[:module.MOCK.index('if {$fault eq "existing_directory"}')]
    return text + r'''
# Promote the synthetic OOC fixture to linked hierarchy; no vendor netlist here.
foreach table {cells pins pin_clocks} {
    set replaced {}
    dict for {name value} [set ::$table] {
        if {$table ne "pin_clocks"} {dict set value NAME WRAPPER/CL/$name}
        dict set replaced WRAPPER/CL/$name $value
    }
    set ::$table $replaced
}
set p {};foreach name $mem_pins {lappend p WRAPPER/CL/$name};set mem_pins $p
foreach name [dict keys $members] {
    set p {};foreach member [dict get $members $name] {lappend p WRAPPER/CL/$member};dict set members $name $p
}
dict set members clk_main_a0 {WRAPPER/CL/u_pcis_pipe/reg}
foreach name {core unused400 hbm} {
    if {[dict exists $clocks $name]} {dict set clocks $name SOURCE_PINS WRAPPER/CL/[dict get $clocks $name SOURCE_PINS]}
}
foreach pin {clk_main_a0 clk_hbm_ref} {
    dict set pins WRAPPER/CL/$pin [dict create NAME WRAPPER/CL/$pin]
    dict set pin_clocks WRAPPER/CL/$pin $pin
}
add_cell WRAPPER/CL cl_opentpu
add_cell WRAPPER/CL/u_mem/f2_mem_i/memory_sc f2_mem_memory_sc_0
set sc WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/m00_nodes/fifo/gen_async_clocks.inst_cdc_addra_to_rd_clk
for {set bit 0} {$bit<2} {incr bit} {
    add_cell [format {%s/src_gray_ff_reg[%d]} $sc $bit] FDRE
    add_cell [format {%s/dest_graysync_ff_reg[0][%d]} $sc $bit] FDRE
}
set cells_before_path $cells
add_cell mock_path timing_path
dict set cells mock_path SLACK 0.01
dict set cells mock_path EXCEPTION {MaxDelay Path 5.000ns -datapath_only}
dict set cells mock_path REQUIREMENT 5.000
foreach name {RULE_DRC RULE_METHODOLOGY} {
    add_cell $name rule
    dict set cells $name SEVERITY {Critical Warning}
    dict set cells $name IS_ENABLED 1
}
add_cell finding violation
dict set cells finding CHECK RULE_DRC
dict set cells finding SEVERITY Error
switch $fault {
    missing_gray {dict unset cells [format {%s/dest_graysync_ff_reg[0][1]} $sc]}
    hidden_ip {add_cell WRAPPER/CL/u_mem/f2_mem_i/hidden_ip unknown 1}
    extra_clock_consumer {dict set members unused400 WRAPPER/CL/u_board/unexpected}
    missing_sc {dict unset cells WRAPPER/CL/u_mem/f2_mem_i/memory_sc}
    unknown_severity {dict set cells finding SEVERITY unknown}
    disabled_rule {dict set cells RULE_DRC IS_ENABLED 0}
}
rename get_cells base_get_cells
proc get_cells {args} {
    set query [lindex $args end]
    if {"-filter" ni $args && "-regexp" ni $args && [llength $query]>1} {
        set result {};foreach name $query {if {[dict exists $::cells $name]} {lappend result $name}};return $result
    }
    return [base_get_cells {*}$args]
}
proc emit_report {args} {
    set path [lindex $args end];set source [file join $::run fixtures [file tail $path]]
    if {$::fault eq "missing_report" && [file tail $path] eq "cdc.rpt"} {return}
    if {[file exists $source]} {file copy $source $path} else {
        set f [open $path w];puts $f {SYNTHETIC TEST REPORT NOT PHYSICAL EVIDENCE};close $f
    }
}
proc report_timing {args} {emit_report {*}$args}
proc report_route_status {args} {emit_report {*}$args}
proc get_nets {args} {return [expr {$::fault eq "unresolved_nets" ? "WRAPPER/CL/unrouted" : ""}]}
proc get_timing_paths {args} {
    if {"-user_ignored" in $args} {return {}}
    if {"-slack_lesser_than" in $args} {return [expr {$::fault eq "negative_path" ? "mock_path" : ""}]}
    return mock_path
}
proc get_drc_checks {args} {return RULE_DRC}
proc get_methodology_checks {args} {return RULE_METHODOLOGY}
proc get_drc_violations {args} {return [expr {$::fault in {drc_error unknown_severity} ? "finding" : ""}]}
set action [lindex $argv 3]
if {[catch {
    if {$action eq "link"} {set result [::c2_alex_route::audit_linked $run]} else {set result [::c2_alex_route::qualify $run $run/input.dcp]}
} message]} {puts stderr $message;exit 1}
puts $result
'''


def run_mock(tmp_path, fault='none', action='qualify', overrides=None):
    fixtures = tmp_path / 'fixtures'
    fixtures.mkdir()
    reports = {'timing.rpt': TIMING, 'route_status.rpt': ROUTE, 'bus_skew.rpt': bus_report(),
               'cdc.rpt': CDC, 'waivers.rpt': 'No waivers found.\n'}
    reports.update(overrides or {})
    for name, text in reports.items():
        (fixtures / name).write_text(text)
    (tmp_path / 'input.dcp').write_bytes(b'SYNTHETIC TEST CHECKPOINT NOT FPGA DATA')
    return tcl(tmp_path, object_mock(), tmp_path, fault, action)


def test_linked_structure_can_pass_without_route_approval(tmp_path):
    result = run_mock(tmp_path, action='link')
    assert result.returncode == 0, result.stderr
    assert 'structural_pass 1 route_qualified 0' in result.stdout
    assert (tmp_path / 'linked_audit/summary.tcldata').is_file()
    assert not list(tmp_path.rglob('*.passed'))


def test_complete_physical_reports_still_require_current_review(tmp_path):
    result = run_mock(tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'route_qualified 0' in result.stdout
    assert 'reason review_required' in result.stdout
    assert 'blockers {}' in result.stdout
    assert 'gray_bus_groups 1 bus_skew_constraints 1 cdc_records 1' in result.stdout
    assert (tmp_path / 'qualification/NOT_QUALIFIED.txt').is_file()
    evidence = (tmp_path / 'qualification/objects.tcldata').read_text()
    assert 'gray_timing_worst_witness' in evidence
    assert 'rule_configuration drc' in evidence
    assert not list(tmp_path.rglob('*.passed'))
    assert 'ROUTE_QUALIFIED' not in result.stdout


@pytest.mark.parametrize('fault', ['wrong_kernel', 'opaque_kernel', 'opaque_mxu', 'missing_native',
                                  'extra_native', 'wrong_native', 'wrong_vendor', 'hidden_ip',
                                  'missing_clock', 'wrong_clock', 'wrong_hbm_clock', 'wrong_host_clock',
                                  'wrong_reference_clock', 'disconnected_core', 'disconnected_hbm',
                                  'disconnected_host', 'missing_clock_pin', 'custom_hclk', 'lookalike_hclk',
                                  'missing_interface', 'wrong_width', 'tmem_cell', 'tmem_exception',
                                  'missing_gray', 'missing_sc', 'extra_clock_consumer', 'missing_report'])
def test_linked_or_final_invalid_evidence_fails_closed(tmp_path, fault):
    result = run_mock(tmp_path, fault=fault)
    assert result.returncode != 0, result.stdout
    assert result.stderr.strip()
    assert not (tmp_path / 'qualification/summary.tcldata').exists()
    assert not list(tmp_path.rglob('*.passed'))


@pytest.mark.parametrize(('fault', 'overrides', 'blocker'), [
    ('none', {'timing.rpt': TIMING.replace('0.100', '-0.001')}, 'negative_wns'),
    ('none', {'timing.rpt': TIMING.replace('0.010', '-0.001')}, 'negative_whs'),
    ('none', {'timing.rpt': TIMING.replace('0.000 0.000 0 100', '-0.001 -0.001 1 100')}, 'negative_wpws'),
    ('none', {'bus_skew.rpt': bus_report(slack='-0.001')}, 'bus_skew:1'),
    ('none', {'waivers.rpt': 'Waivers are present\n'}, 'waiver_review_required'),
    ('none', {'cdc.rpt': CDC.replace('/read/D N', '/read/D Y')}, 'waived_cdc'),
    ('unresolved_nets', {}, 'unresolved_nets'),
    ('negative_path', {}, 'negative_setup'),
    ('drc_error', {}, 'drc:RULE_DRC'),
])
def test_final_blockers_are_preserved_in_not_qualified_receipt(tmp_path, fault, overrides, blocker):
    result = run_mock(tmp_path, fault, overrides=overrides)
    assert result.returncode == 0, result.stderr
    assert 'reason physical_checks_failed_and_review_required' in result.stdout
    assert blocker in result.stdout
    assert 'route_qualified 0' in result.stdout
    assert not list(tmp_path.rglob('*.passed'))


def test_gray_bus_coverage_is_exact_not_historical_count(tmp_path):
    result = run_mock(tmp_path, overrides={'bus_skew.rpt': bus_report(destination_bit='2')})
    assert result.returncode != 0
    assert 'lacks unique exact bus-skew coverage' in result.stderr


def test_unknown_rule_severity_is_not_treated_as_advisory(tmp_path):
    result = run_mock(tmp_path, fault='unknown_severity')
    assert result.returncode != 0
    assert 'unknown severity' in result.stderr


def test_disabled_rule_configuration_cannot_hide_in_empty_finding_list(tmp_path):
    result = run_mock(tmp_path, fault='disabled_rule')
    assert result.returncode == 0, result.stderr
    assert 'route_qualified 0' in result.stdout
    assert 'current_DRC_methodology_and_severity_configuration' in result.stdout
    assert 'IS_ENABLED 0' in (tmp_path / 'qualification/objects.tcldata').read_text()
