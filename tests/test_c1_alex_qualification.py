"""Offline qualifier parser/control tests. Fixtures are not FPGA evidence."""
from pathlib import Path
import re
import importlib.util
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
QUALIFIER = ROOT / 'boards/c1_f2_alex/qualify_c1_alex.tcl'
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
if {[catch {::c1_alex_route::[lindex $argv 1] [::c1_alex_route::read_text [lindex $argv 2]]} result]} {puts stderr $result;exit 1}
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
    spec = importlib.util.spec_from_file_location('synth_fixture', ROOT / 'tests/test_c1_alex_audit.py')
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
fabric_build WRAPPER/CL
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
    set selected [base_get_cells {*}$args]
    # At top scope a regexp search does not traverse WRAPPER/CL unless
    # -hierarchical is requested. Exact hierarchical names still resolve.
    if {"-regexp" in $args && "-hierarchical" ni $args} {
        set direct {};foreach name $selected {if {[string first / $name]<0} {lappend direct $name}}
        return $direct
    }
    return $selected
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
rename get_nets fabric_get_nets
proc get_nets {args} {
 if {"-filter" in $args} {return [expr {$::fault eq "unresolved_nets" ? "WRAPPER/CL/unrouted" : ""}]}
 return [fabric_get_nets {*}$args]
}
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
    if {$action eq "link"} {set result [::c1_alex_route::audit_linked $run]} else {set result [::c1_alex_route::qualify $run $run/input.dcp]}
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
    # This fixture models an unreplicated synthetic fabric. The real routed
    # replication helper has dedicated positive and adversarial tests.
    (tmp_path / 'routed_host_ready.tcl').write_text('''
namespace eval ::c1_alex_routed_ready {}
proc ::c1_alex_routed_ready::host_wready {audit} {
    ::c1_alex::flow WRAPPER/CL S_HOST wready $audit
    return [dict create structural_pass 1 route_qualified 0]
}
''')
    return tcl(tmp_path, object_mock(), tmp_path, fault, action)


def test_linked_structure_can_pass_without_route_approval(tmp_path):
    result = run_mock(tmp_path, action='link')
    assert result.returncode == 0, result.stderr
    assert 'structural_pass 1 route_qualified 0' in result.stdout
    assert (tmp_path / 'linked_audit/summary.tcldata').is_file()
    assert not list(tmp_path.rglob('*.passed'))


@pytest.mark.parametrize('action', ['link', 'qualify'])
@pytest.mark.parametrize('fault', ['none', 'missing_native', 'extra_native'])
def test_nested_native_inventory_is_complete_and_exact(tmp_path, action, fault):
    result = run_mock(tmp_path, action=action, fault=fault)
    if fault == 'none':
        assert result.returncode == 0, result.stderr
        assert 'route_qualified 0' in result.stdout
    else:
        assert result.returncode != 0
        assert result.stderr.strip()
        stage = 'linked_audit' if action == 'link' else 'qualification'
        assert not (tmp_path / stage / 'summary.tcldata').exists()


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


@pytest.mark.parametrize('fault', ['wrong_kernel', 'opaque_kernel', 'opaque_core', 'missing_native',
                                  'extra_native', 'wrong_native', 'wrong_vendor', 'hidden_ip',
                                  'missing_clock', 'wrong_clock', 'wrong_hbm_clock', 'wrong_host_clock',
                                  'wrong_reference_clock', 'disconnected_core', 'disconnected_hbm',
                                  'disconnected_host', 'missing_clock_pin', 'custom_hclk', 'lookalike_hclk',
                                  'sc_clock_swap', 'pcis_on_core', 'tmem_cell', 'tmem_exception',
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


@pytest.mark.parametrize(('finding_severity', 'rule_severity', 'expected', 'fails'), [
    (' Warning ', 'Error', 'drc/WARNING 1', False),
    ('ABSENT', 'Error', 'blockers drc:DRC_RULE', False),
    ('unexpected', 'Error', 'unknown severity UNEXPECTED', True),
    ('', 'Error', 'unknown severity', True),
    ('ABSENT', 'ABSENT', 'missing severity for DRC_RULE', True),
])
def test_finding_severity_uses_captured_properties_with_strict_fallback(
        tmp_path, finding_severity, rule_severity, expected, fails):
    script = r'''
source [lindex $argv 0]
set objects [dict create DRC_RULE [dict create NAME DRC_RULE] METH_RULE [dict create NAME METH_RULE SEVERITY Warning] finding [dict create NAME finding CHECK DRC_RULE]]
if {[lindex $argv 2] ne {ABSENT}} {dict set objects finding SEVERITY [lindex $argv 2]}
if {[lindex $argv 3] ne {ABSENT}} {dict set objects DRC_RULE SEVERITY [lindex $argv 3]}
proc get_drc_checks {args} {return DRC_RULE}
proc get_methodology_checks {args} {return METH_RULE}
proc get_drc_violations {args} {return finding}
proc get_methodology_violations {args} {return {}}
proc list_property {object} {return [dict keys [dict get $::objects $object]]}
proc get_property {key object} {return [dict get $::objects $object $key]}
rename ::c1_alex_route::prop ::c1_alex_route::original_prop
proc ::c1_alex_route::prop {object key} {
    # Severity is valid in the initial property snapshots; re-querying a
    # separately selected owner is not valid for the opaque Vivado finding.
    if {$key eq {SEVERITY}} {error {severity must use the captured object properties}}
    return [original_prop $object $key]
}
set f [open [lindex $argv 1] w]
set code [catch {::c1_alex_route::findings $f} result];close $f
if {$code} {puts stderr $result;exit 1}
puts $result
'''
    result = tcl(tmp_path, script, tmp_path / 'findings.tcldata', finding_severity, rule_severity)
    assert bool(result.returncode) is fails, result.stdout + result.stderr
    assert expected in (result.stderr if fails else result.stdout)
    if not fails and finding_severity != 'ABSENT':
        assert 'blockers {}' in result.stdout


def test_disabled_rule_configuration_cannot_hide_in_empty_finding_list(tmp_path):
    result = run_mock(tmp_path, fault='disabled_rule')
    assert result.returncode == 0, result.stderr
    assert 'route_qualified 0' in result.stdout
    assert 'current_DRC_methodology_and_severity_configuration' in result.stdout
    assert 'IS_ENABLED 0' in (tmp_path / 'qualification/objects.tcldata').read_text()


@pytest.mark.parametrize('fault',['pruned_boundary_interface','pruned_boundary_bit','pruned_boundary_handshake'])
def test_linked_optimized_boundary_inventory_is_not_declared_axi_width(tmp_path,fault):
    result=run_mock(tmp_path,fault=fault)
    assert result.returncode==0,result.stderr
    assert 'route_qualified 0' in result.stdout
    assert 'C1_ALEX_ROUTE_NOT_QUALIFIED' in result.stdout


SUPPLEMENTAL_MOCK = r'''
source [lindex $argv 0]
set out [lindex $argv 1];set fault [lindex $argv 2]
set root WRAPPER/CL/u_mem/f2_mem_i/memory_sc/fifo
set groups {};set stages {};set timing_queries 0
for {set bit 0} {$bit<2} {incr bit} {
 dict set groups $root source $bit [format {%s/src_gray_ff_reg[%d]} $root $bit]
 dict set groups $root destination $bit [format {%s/dest_graysync_ff_reg[0][%d]} $root $bit]
 foreach stage {0 1} {lappend stages [format {%s/dest_graysync_ff_reg[%d][%d]} $root $stage $bit]}
}
if {$fault eq {missing_stage_zero}} {set stages [lrange $stages 1 end]}
if {$fault eq {gapped_stage}} {lappend stages [format {%s/dest_graysync_ff_reg[3][0]} $root]}
if {$fault eq {unexpected_bit}} {lappend stages [format {%s/dest_graysync_ff_reg[0][9]} $root]}
if {$fault eq {duplicate_stage}} {lappend stages [lindex $stages 0]}
proc get_cells {args} {
 if {"-filter" in $args} {return $::stages}
 if {"-of_objects" in $args} {return [list [file dirname [lindex [lindex $args end] 0]]]}
 return [lindex $args end]
}
proc get_pins {args} {
 if {"-of_objects" ni $args} {return [lindex $args end]}
 set net [lindex [lindex $args end] 0]
 if {$net eq {}} {error {unscoped empty-net query}}
 if {$::fault eq {empty_leaves}} {return {}}
 set destination [string range $net 4 end]
 regexp {^(.*)/dest_graysync_ff_reg\[([0-9]+)\]\[([0-9]+)\]/D$} $destination -> root stage bit
 set source [expr {$stage==0 ? [format {%s/src_gray_ff_reg[%d]/Q} $root $bit] : [format {%s/dest_graysync_ff_reg[%d][%d]/Q} $root [expr {$stage-1}] $bit]}]
 return [list $source $destination]
}
proc get_nets {args} {
 if {$::fault eq {empty_nets}} {return {}}
 return [list net:[lindex [lindex $args end] 0]]
}
proc get_clocks {args} {
 set pin [lindex [lindex $args end] 0]
 if {$::fault eq {missing_clock} || ![string match */C $pin]} {return {}}
 return [expr {[string first src_gray $pin]>=0 ? {source_clock} : {destination_clock}}]
}
proc get_property {key objects} {
 set result {}
 foreach object $objects {
  switch $key {
   NAME {lappend result $object}
   DIRECTION {lappend result [expr {[string match */Q $object] ? {OUT} : {IN}}]}
   REF_PIN_NAME {lappend result [file tail $object]}
   IS_CLOCK {lappend result [expr {[string match */C $object]}]}
   REF_NAME - ORIG_REF_NAME {lappend result FDRE}
   IS_SEQUENTIAL - ASYNC_REG {lappend result 1}
   PERIOD {lappend result 5.000}
   WAVEFORM {lappend result {0 2.5}}
   MASTER_CLOCK - SOURCE_PINS {lappend result source_clock}
   LOC - BEL {lappend result {}}
   STARTPOINT_PIN {lappend result [file dirname $::last_from]/C}
   ENDPOINT_PIN {lappend result $::last_to}
   STARTPOINT_CLOCK {lappend result source_clock}
   ENDPOINT_CLOCK {lappend result destination_clock}
   REQUIREMENT {lappend result 5.0}
   EXCEPTION {lappend result {MaxDelay Path 5.000ns -datapath_only}}
   DATAPATH_DELAY {lappend result 1.0}
   SLACK {lappend result 4.0}
   CORNER {lappend result [expr {[string match *0 $object] || [string match *1 $object] ? {Slow} : {Fast}}]}
   STARTPOINT_RISE_FALL {lappend result [expr {[string match *0 $object] || [string match *2 $object] ? {RISE} : {FALL}}]}
   default {error "unexpected property $key"}
  }
 }
 return [expr {[llength $objects]==1 ? [lindex $result 0] : $result}]
}
proc list_property {object} {
 if {![string match path* $object]} {error {cell property enumeration forbidden}}
 if {$::fault eq {unsupported_path_properties}} {error {mock unsupported timing property inventory}}
 return {STARTPOINT_PIN ENDPOINT_PIN SLACK CORNER STARTPOINT_RISE_FALL}
}
proc get_timing_paths {args} {
 incr ::timing_queries
 if {[lindex $args [expr {[lsearch -exact $args -max_paths]+1}]]!=16 || [lindex $args [expr {[lsearch -exact $args -nworst]+1}]]!=16} {error {timing query not bounded}}
 set ::last_from [lindex [lindex $args [expr {[lsearch -exact $args -from]+1}]] 0]
 set ::last_to [lindex [lindex $args [expr {[lsearch -exact $args -to]+1}]] 0]
 if {$::fault eq {unsupported_paths}} {error {mock unsupported timing API}}
 if {"-user_ignored" in $args} {return {}}
 if {$::fault eq {missing_normal} && [regexp {\[1\]/D$} $::last_to]} {return {}}
 if {$::fault eq {capped}} {return [lrepeat 16 path0]}
 return {path0 path1 path2 path3}
}
set ::c1_alex_route::capture_queries 0;set ::c1_alex_route::capture_pending 0;set ::c1_alex_route::captured_cells {}
set f [open $out w]
::c1_alex_route::capture_gray $groups $f
puts $f [list test_summary timing_queries $timing_queries pending $::c1_alex_route::capture_pending route_qualified 0]
close $f
'''


@pytest.mark.parametrize(('fault', 'required'), [
    ('none', 'pending 0'),
    ('missing_normal', 'capped_or_missing_normal_paths'),
    ('capped', 'capped 1'),
    ('unsupported_paths', 'mock unsupported timing API'),
    ('unsupported_path_properties', 'mock unsupported timing property inventory'),
    ('empty_nets', 'missing_data_nets'),
    ('empty_leaves', 'missing_data_leaf_pins'),
    ('missing_clock', 'missing_required_clock'),
    ('missing_stage_zero', 'missing_or_mismatched_stage_zero'),
    ('gapped_stage', 'missing_synchronizer_stage'),
    ('unexpected_bit', 'unexpected_gray_bit'),
    ('duplicate_stage', 'duplicate_gray_stage'),
])
def test_gray_supplement_preserves_each_bit_and_truthful_incompleteness(tmp_path, fault, required):
    report = tmp_path / 'supplemental.tcldata'
    result = tcl(tmp_path, SUPPLEMENTAL_MOCK, report, fault)
    assert result.returncode == 0, result.stderr
    raw = report.read_text()
    assert required in raw
    assert 'timing_queries 4' in raw
    assert 'route_qualified 0' in raw
    assert 'unscoped empty-net query' not in raw
    if fault == 'none':
        assert raw.count('gray_bit_timing ') == 4
        assert raw.count('user_ignored 1 expected_source') == 2
        assert 'data_driver ' in raw and 'driver_count 1' in raw
        assert raw.count('gray_stage_inventory ') == 2
        assert 'max_paths 16 nworst 16 returned 4 capped 0 normal_path_observed 1' in raw
        assert raw.count('gray_path_endpoint_relation ') == 8
        assert 'startpoint_relation source_register_clock_pin' in raw
        assert 'data_driver_query ' in raw and re.search(r'/Q\}? source_clock_query ', raw)
        assert raw.count('gray_path_variant_sample ') == 1
        assert len(re.findall(r'^object_properties .* optional_variant_properties\}', raw, re.M)) == 4
        assert 'CORNER {ok 1 value Slow}' in raw and 'CORNER {ok 1 value Fast}' in raw
    else:
        assert not re.search(r'test_summary .*pending 0 ', raw)


@pytest.mark.parametrize('fault', ['none', 'unsupported_fanin'])
def test_no_clock_finding_preserves_exact_pin_cell_and_fanin_root(tmp_path, fault):
    report = tmp_path / 'relationships.tcldata'
    script = r'''
source [lindex $argv 0]
set fault [lindex $argv 2]
proc get_methodology_violations {args} {return finding17}
proc get_drc_violations {args} {return {}}
proc get_nets {args} {return {}}
proc get_ports {args} {return {}}
proc get_pins {args} {
 if {"-of_objects" in $args} {return shell/register/C}
 return [lindex $args end]
}
proc get_cells {args} {
 if {"-of_objects" in $args} {
  set input [lindex $args end]
  return [expr {$input eq {finding17} ? {shell/register} : [file dirname $input]}]
 }
 return [lindex $args end]
}
proc get_clocks {args} {return {}}
proc get_property {key object} {
 switch $key {
  NAME {return $object} CHECK {return TIMING-17}
  REF_NAME - ORIG_REF_NAME {return [expr {[string match *bufg* $object] ? {BUFG_GT} : {FDRE}}]}
  REF_PIN_NAME {return [file tail $object]}
  DIRECTION {return [expr {[string match */O $object] ? {OUT} : {IN}}]}
  IS_CLOCK - IS_SEQUENTIAL {return 1}
  default {return {}}
 }
}
proc all_fanin {args} {
 if {"-to" ni $args || [lindex $args end] ne {shell/register/C}} {error {fanin must use exact -to pin}}
 if {$::fault eq {unsupported_fanin}} {error {mock fanin unsupported}}
 return shell/bufg/O
}
proc list_property {args} {error {hierarchical enumeration forbidden}}
set ::c1_alex_route::capture_queries 0;set ::c1_alex_route::capture_pending 0;set ::c1_alex_route::captured_cells {}
set f [open [lindex $argv 1] w];::c1_alex_route::capture_findings $f
puts $f [list pending $::c1_alex_route::capture_pending route_qualified 0];close $f
'''
    result = tcl(tmp_path, script, report, fault)
    assert result.returncode == 0, result.stderr
    raw = report.read_text()
    assert 'shell/register/C' in raw and 'shell/register' in raw
    if fault == 'none':
        assert 'shell/bufg/O' in raw and 'BUFG_GT' in raw
        assert 'pending 0 route_qualified 0' in raw
    else:
        assert 'mock fanin unsupported' in raw
        assert 'pending 1 route_qualified 0' in raw


def test_supplemental_is_in_linked_and_final_report_inventories(tmp_path):
    for action in ('link', 'qualify'):
        directory = tmp_path / action
        directory.mkdir()
        result = run_mock(directory, action=action)
        assert result.returncode == 0, result.stderr
        stage = directory / ('linked_audit' if action == 'link' else 'qualification')
        assert 'supplemental.tcldata' in (stage / 'summary.tcldata').read_text()
        raw = (stage / 'supplemental.tcldata').read_text()
        assert 'route_qualified 0' in raw and 'supplemental_summary' in raw
        assert 'unsupported_live_waiver_inventory' in raw
        for name in ('methodology.rpt', 'timing.rpt', 'waivers.rpt'):
            assert (stage / name).is_file()


def test_live_waiver_relationship_errors_remain_pending_without_losing_other_sets(tmp_path):
    report = tmp_path / 'waiver.tcldata'
    script = r'''
source [lindex $argv 0]
proc get_waivers {args} {return waiver0}
proc help {args} {return {MOCK INSTALLED API HELP}}
proc list_property {object} {if {$object ne {waiver0}} {error {unexpected enumeration}};return {NAME FROM TO}}
proc get_property {key object} {switch $key {NAME {return $object} FROM {return source/Q} TO {return destination/D}}}
proc get_pins {args} {return {source/Q destination/D}}
proc get_cells {args} {error {mock unsupported waiver cell relationship}}
proc get_nets {args} {return actual_net}
proc get_clocks {args} {return {source_clock destination_clock}}
set ::c1_alex_route::capture_queries 0;set ::c1_alex_route::capture_pending 0
set f [open [lindex $argv 1] w];::c1_alex_route::capture_waivers $f
puts $f [list pending $::c1_alex_route::capture_pending route_qualified 0];close $f
'''
    result = tcl(tmp_path, script, report)
    assert result.returncode == 0, result.stderr
    raw = report.read_text()
    for field in ('waiver_inventory', 'waiver_property_names', 'selected_pins', 'selected_cells', 'selected_nets', 'selected_clocks'):
        assert field in raw
    assert 'mock unsupported waiver cell relationship' in raw
    assert 'pending 1 route_qualified 0' in raw


@pytest.mark.parametrize(('initial', 'count', 'captured', 'skipped'), [(0, 300, 256, 44), (4096, 1, 0, 1)])
def test_cell_detail_expansion_caps_preserve_exact_skipped_count(tmp_path, initial, count, captured, skipped):
    report = tmp_path / 'caps.tcldata'
    script = r'''
source [lindex $argv 0]
proc get_property {key objects} {return [expr {$key eq {NAME} ? $objects : {FDRE}}]}
proc get_cells {args} {return [lindex $args end]}
proc list_property {args} {error {hierarchical enumeration forbidden}}
set ::c1_alex_route::capture_queries 0;set ::c1_alex_route::capture_pending 0;set ::c1_alex_route::captured_cells {}
for {set i 0} {$i<[lindex $argv 2]} {incr i} {dict set ::c1_alex_route::captured_cells existing$i 1}
set cells {};for {set i 0} {$i<[lindex $argv 3]} {incr i} {lappend cells selected$i}
set f [open [lindex $argv 1] w];::c1_alex_route::capture_cells $f test $cells
puts $f [list pending $::c1_alex_route::capture_pending route_qualified 0];close $f
'''
    result = tcl(tmp_path, script, report, initial, count)
    assert result.returncode == 0, result.stderr
    raw = report.read_text()
    assert f'selected {count} captured {captured} cached 0 skipped {skipped}' in raw
    assert 'cell_detail_cap_reached' in raw
    assert 'pending 1 route_qualified 0' in raw


def test_gray_and_finding_cell_detail_budgets_are_independent(tmp_path):
    script = r'''
source [lindex $argv 0]
proc get_property {key objects} {return [expr {$key eq {NAME} ? $objects : {FDRE}}]}
proc get_cells {args} {return [lindex $args end]}
proc ::c1_alex_route::capture_gray_ancestors {groups audit} {}
proc ::c1_alex_route::capture_gray {groups audit} {
 variable captured_cells
 for {set i 0} {$i<4096} {incr i} {dict set captured_cells gray$i 1}
 capture_cells $audit exhausted_gray {extra_gray}
}
proc ::c1_alex_route::capture_findings {audit} {
 capture_cells $audit fresh_finding {new_finding_cell}
}
proc ::c1_alex_route::capture_waivers {audit} {}
proc ::c1_alex_route::capture_pulse {audit} {}
puts [::c1_alex_route::supplemental [lindex $argv 1] {}]
'''
    result = tcl(tmp_path, script, tmp_path)
    assert result.returncode == 0, result.stderr
    raw = (tmp_path / 'supplemental.tcldata').read_text()
    assert raw.index('cell_detail_budget section gray') < raw.index('cell_detail_budget section findings')
    assert 'cell_detail_coverage exhausted_gray selected 1 captured 0 cached 0 skipped 1' in raw
    assert 'cell_detail_coverage fresh_finding selected 1 captured 1 cached 0 skipped 0' in raw
    assert 'section_cap 4096 section gray' in raw and 'section_cap 4096 section findings' in raw
    assert 'incomplete_or_error 1' in result.stdout


@pytest.mark.parametrize(('fault', 'required'), [
    ('none', 'selected 4 attempted 4 skipped 0 cap 1024'),
    ('missing_fifo', 'missing_or_ambiguous_ancestor'),
    ('wrong_identity', 'missing_or_ambiguous_ancestor'),
    ('unknown_ancestry', 'unrecognized_xpm_ancestor_path'),
    ('capped', 'selected 1052 attempted 1024 skipped 28 cap 1024'),
])
def test_gray_ancestor_capture_is_exact_deduplicated_bounded_and_not_acceptance(tmp_path, fault, required):
    report = tmp_path / 'ancestors.tcldata'
    script = r'''
source [lindex $argv 0]
set fault [lindex $argv 2]
set prefix WRAPPER/CL/u_mem/f2_mem_i/memory_sc
set fifo $prefix/inst/fifo/gen_xpm_memory_fifo.inst_fifo
set groups {};set count 2
if {$fault eq {capped}} {set count 1050}
for {set i 0} {$i<$count} {incr i} {
 set root $fifo/gen_async_clocks.inst_cdc_gray_$i
 if {$fault eq {unknown_ancestry}} {set root $prefix/other_$i}
 dict set groups $root source 0 $root/src_gray_ff_reg0
 dict set groups $root destination 0 $root/dest_graysync_ff_reg0
}
proc get_cells {args} {
 set name [lindex [lindex $args end] 0]
 if {$name eq $::fifo} {
  if {$::fault eq {missing_fifo}} {return {}}
  if {$::fault eq {wrong_identity}} {return WRONG_CELL}
 }
 return [list $name]
}
proc get_property {key objects} {
 set object [lindex $objects 0]
 switch $key {
  NAME {return $object}
  REF_NAME - ORIG_REF_NAME {
   if {$object eq $::fifo} {return xpm_fifo_async}
   if {$object eq $::prefix} {return f2_mem_memory_sc_0}
   return xpm_cdc_gray
  }
  IS_BLACKBOX {return 0}
  default {error {unsupported property}}
 }
}
proc list_property {args} {error {hierarchical property enumeration forbidden}}
set ::c1_alex_route::capture_queries 0;set ::c1_alex_route::capture_pending 0
set f [open [lindex $argv 1] w];::c1_alex_route::capture_gray_ancestors $groups $f
puts $f [list pending $::c1_alex_route::capture_pending route_qualified 0];close $f
'''
    result = tcl(tmp_path, script, report, fault)
    assert result.returncode == 0, result.stderr
    raw = report.read_text()
    assert required in raw and 'route_qualified 0' in raw
    if fault == 'none':
        assert raw.count('gray_ancestor_relation ') == 2
        assert len(re.findall(r'^object_properties .*roles xpm_fifo', raw, re.M)) == 1
        assert 'xpm_fifo_async' in raw and 'xpm_cdc_gray' in raw
        assert 'pending 0 route_qualified 0' in raw
    else:
        assert 'pending 0 route_qualified 0' not in raw


@pytest.mark.parametrize(('fault', 'pending', 'required'), [
    ('none', 0, 'complete not_asserted'),
    ('unsupported', 1, 'unsupported_pulse_report'),
    ('api_error', 2, 'mock pulse API failure'),
    ('empty', 2, 'empty_pulse_report'),
    ('wrong_scope', 1, 'missing_or_ambiguous_custom_logic'),
])
def test_pulse_reports_use_observed_documented_flags_and_keep_limits_pending(tmp_path, fault, pending, required):
    report = tmp_path / 'pulse.tcldata'
    script = r'''
source [lindex $argv 0]
set fault [lindex $argv 2];set calls {}
proc get_cells {args} {
 if {$::fault eq {wrong_scope}} {return WRAPPER/OTHER}
 return WRAPPER/CL
}
proc get_property {key object} {if {$key ne {NAME}} {error {unexpected property}};return $object}
if {$fault ne {unsupported}} {
 proc report_pulse_width {args} {
  if {[lrange $args 0 2] ne {-return_string -limit 128}} {error {unexpected pulse query}}
  if {[llength $args]!=3 && $args ne {-return_string -limit 128 -cells WRAPPER/CL}} {error {unexpected pulse selection}}
  lappend ::calls $args
  if {$::fault eq {api_error}} {error {mock pulse API failure}}
  if {$::fault eq {empty}} {return {}}
  return {SYNTHETIC pulse report NOT FPGA EVIDENCE}
 }
}
set ::c1_alex_route::capture_queries 0;set ::c1_alex_route::capture_pending 0
set f [open [lindex $argv 1] w];::c1_alex_route::capture_pulse $f
puts $f [list pending $::c1_alex_route::capture_pending route_qualified 0];close $f
puts [list calls $calls]
'''
    result = tcl(tmp_path, script, report, fault)
    assert result.returncode == 0, result.stderr
    raw = report.read_text()
    assert required in raw
    assert f'pending {pending} route_qualified 0' in raw
    if fault == 'none':
        assert 'pulse_report whole_design' in raw and 'pulse_report custom_logic' in raw
        assert raw.count('limit_per_check_type_per_clock 128 checks default_all complete not_asserted') == 2
        assert '-return_string -limit 128 -cells WRAPPER/CL' in result.stdout


@pytest.mark.parametrize('api', [
    'get_cells', 'get_pins', 'get_nets', 'get_ports', 'get_clocks',
    'all_fanin', 'all_fanout', 'all_registers',
    'get_drc_violations', 'get_methodology_violations', 'get_waivers',
])
def test_capture_expands_over_500_objects_without_replacing_typed_return(tmp_path, api):
    report = tmp_path / 'collections.tcldata'
    script = r'''
source [lindex $argv 0]
set api [lindex $argv 2]
set handles {};for {set i 0} {$i<701} {incr i} {lappend handles handle$i}
proc selected {args} {return $::handles}
interp alias {} $api {} selected
set reads 0
proc get_property {key object} {
 if {$key ne "NAME" || [llength $object]!=1 || ![regexp {^handle([0-9]+)$} $object -> index]} {
  error {must read NAME from one original object, not a bulk or reconstructed name list}
 }
 incr ::reads
 return [format {scope/register[%d]} $index]
}
set ::c1_alex_route::capture_queries 0;set ::c1_alex_route::capture_pending 0
set f [open [lindex $argv 1] w]
set result [::c1_alex_route::capture $f all_objects [list $api -quiet]]
close $f
if {[dict get $result value] ne $handles || ![dict get $result ok]} {error {original collection replaced}}
if {$reads!=701} {error {collection NAME reads incomplete}}
set f [open [lindex $argv 1] r];set row [gets $f];close $f
set evidence [dict create {*}[lrange $row 2 end]]
if {[llength [dict get $evidence value]]!=701 || [lindex [dict get $evidence value] end] ne {scope/register[700]}} {error {serialized inventory incomplete}}
puts [list queries $::c1_alex_route::capture_queries pending $::c1_alex_route::capture_pending]
'''
    result = tcl(tmp_path, script, report, api)
    assert result.returncode == 0, result.stderr
    assert 'queries 1 pending 0' in result.stdout
    raw = report.read_text()
    assert 'scope/register[700]' in raw and 'handle700' not in raw
    assert '...' not in raw


@pytest.mark.parametrize('fault', ['error', 'empty'])
def test_collection_name_serialization_failure_remains_explicit(tmp_path, fault):
    report = tmp_path / 'collection_error.tcldata'
    script = r'''
source [lindex $argv 0]
set fault [lindex $argv 2]
proc get_cells {args} {return {typed0 typed1}}
proc get_property {key object} {
 if {$object eq "typed1"} {
  if {$::fault eq "error"} {error {name API failed}}
  return {}
 }
 return cell0
}
set ::c1_alex_route::capture_queries 0;set ::c1_alex_route::capture_pending 0
set f [open [lindex $argv 1] w]
set result [::c1_alex_route::capture $f cells {get_cells -quiet}]
close $f
if {[dict get $result ok] || [dict get $result value] ne {typed0 typed1}} {error {failure lost original return or claimed success}}
puts [list queries $::c1_alex_route::capture_queries pending $::c1_alex_route::capture_pending]
'''
    result = tcl(tmp_path, script, report, fault)
    assert result.returncode == 0, result.stderr
    assert 'queries 1 pending 1' in result.stdout
    assert 'status error stage collection_name_serialization' in report.read_text()
    assert 'status observed' not in report.read_text()


def test_capture_does_not_expand_literal_property_values_or_reports(tmp_path):
    report = tmp_path / 'literal.tcldata'
    script = r'''
source [lindex $argv 0]
proc get_property {key object} {
 if {$key eq "NAME"} {error {unexpected NAME lookup on literal}}
 return {literal value ... not a collection}
}
proc report_pulse_width {args} {return "literal report ...\nsecond line"}
set ::c1_alex_route::capture_queries 0;set ::c1_alex_route::capture_pending 0
set f [open [lindex $argv 1] w]
set prop [::c1_alex_route::capture $f property {get_property DESCRIPTION object}]
set report [::c1_alex_route::capture $f report {report_pulse_width -return_string}]
close $f
if {[dict get $prop value] ne {literal value ... not a collection} || [dict get $report value] ne "literal report ...\nsecond line"} {error {literal mutated}}
puts [list queries $::c1_alex_route::capture_queries pending $::c1_alex_route::capture_pending]
'''
    result = tcl(tmp_path, script, report)
    assert result.returncode == 0, result.stderr
    assert 'queries 2 pending 0' in result.stdout


def test_finding_related_nets_and_ports_are_complete_without_recursive_expansion(tmp_path):
    report = tmp_path / 'finding_relations.tcldata'
    script = r'''
source [lindex $argv 0]
proc get_drc_violations {args} {return finding10}
proc get_methodology_violations {args} {return {}}
proc get_cells {args} {return {}}
proc get_pins {args} {return {}}
proc get_nets {args} {return $::nets}
proc get_ports {args} {return {port_handle}}
set nets {};for {set i 0} {$i<731} {incr i} {lappend nets net_handle$i}
proc get_property {key object} {
 if {$key eq {CHECK} && $object eq {finding10}} {return RTSTAT-10}
 if {$key ne {NAME}} {error {unexpected detail expansion}}
 if {[regexp {^net_handle([0-9]+)$} $object -> i]} {return shell/net$i}
 if {$object eq {port_handle}} {return shell_port}
 return $object
}
proc list_property {args} {error {unexpected property inventory}}
set ::c1_alex_route::capture_queries 0;set ::c1_alex_route::capture_pending 0;set ::c1_alex_route::captured_cells {}
set f [open [lindex $argv 1] w];::c1_alex_route::capture_findings $f;close $f
puts [list pending $::c1_alex_route::capture_pending]
'''
    result = tcl(tmp_path, script, report)
    assert result.returncode == 0, result.stderr
    raw = report.read_text()
    assert 'pending 0' in result.stdout
    assert 'related_nets' in raw and 'shell/net730' in raw
    assert 'related_ports' in raw and 'shell_port' in raw
    assert 'object_properties' not in raw


def test_routed_fabric_preserves_153_flows_and_requires_native_semantic_review(tmp_path):
    result = tcl(tmp_path, '''source [lindex $argv 0]
set flows {};set host_count 0
proc ::c1_alex::fabric_clocks {prefix clocks opaque audit} {return {}}
proc ::c1_alex::flow {prefix iface signal audit} {lappend ::flows [list $iface $signal]}
namespace eval ::c1_alex_routed_ready {}
proc ::c1_alex_routed_ready::host_wready {audit} {incr ::host_count}
set out [open [lindex $argv 1] w]
set facts [::c1_alex_route::routed_fabric WRAPPER/CL {} $out]
close $out
if {[llength $flows]!=153 || [llength [lsort -unique $flows]]!=153 || $host_count!=1} {error "Flow inventory differs"}
if {{S_HOST wready} in $flows || [dict get $facts active_handshake_witnesses]!=154 || [dict get $facts shared_reset_ready_connections]!=0 || [dict get $facts state] ne "pending_native_ready_equivalence"} {error "Missing explicit native review"}
puts $facts
''', tmp_path/'routed.tcldata')
    assert result.returncode == 0, result.stderr
    assert 'pending_native_ready_equivalence' in result.stdout
    raw = (tmp_path/'routed.tcldata').read_text()
    assert 'current_routed_native_ready_equivalence' in raw
    assert 'route_qualified 0' in raw


def test_routed_fabric_does_not_swallow_host_replica_failure(tmp_path):
    result = tcl(tmp_path, '''source [lindex $argv 0]
proc ::c1_alex::fabric_clocks {prefix clocks opaque audit} {return {}}
proc ::c1_alex::flow {prefix iface signal audit} {return {}}
namespace eval ::c1_alex_routed_ready {}
proc ::c1_alex_routed_ready::host_wready {audit} {error "Changed replica controls"}
::c1_alex_route::routed_fabric WRAPPER/CL {} stdout
''')
    assert result.returncode != 0
    assert 'Changed replica controls' in result.stderr
