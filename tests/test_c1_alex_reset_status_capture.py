"""Read-only collector tests with strict typed-object Vivado stubs."""
from pathlib import Path
import shutil
import subprocess
import pytest
HERE=Path(__file__).resolve().parents[1]
COLLECTOR=HERE/'boards/c1_f2_alex/capture_reset_status.tcl'
ENTRY=COLLECTOR.read_text()
TCL=shutil.which('tclsh')
pytestmark=pytest.mark.skipif(TCL is None, reason='Tcl interpreter unavailable')
def run(tmp_path,body):
 out=tmp_path/'evidence'
 program='source {'+str(COLLECTOR)+'}\nset audit [open {'+str(out)+'} w]\n'
 program+=r'''
set ::selections 0
proc typed {kind name} {return "HANDLE:$kind:$name"}
proc untyped {object} {if {![regexp {^HANDLE:([^:]+):(.*)$} $object -> kind name]} {error "Plain-name object rejected: $object"};return [list $kind $name]}
proc select {kind args} {
 incr ::selections
 set of [lsearch -exact $args -of_objects]
 if {$of>=0} {foreach object [lindex $args [expr {$of+1}]] {untyped $object};return {}}
 if {[lsearch -exact $args -hierarchical]>=0} {return {}}
 set result {};foreach name [lindex $args end] {lappend result [typed $kind $name]};return $result
}
foreach kind {cell pin net port clock} {interp alias {} get_${kind}s {} select $kind}
proc get_property {key objects} {
 if {[llength $objects]!=1} {error "Bulk properties rejected"}
 lassign [untyped [lindex $objects 0]] kind name
 switch -- $key {
 NAME {return $name}
 CLASS {return $kind}
 REF_PIN_NAME {return [file tail $name]}
 IS_CLOCK {return [expr {[file tail $name] eq "C"}]}
 REF_NAME {return FDCE}
 default {return 0}
 }
}
proc list_property {object} {untyped [lindex $object 0];return {NAME IS_C_INVERTED IS_CLR_INVERTED}}
proc get_timing_paths {args} {untyped [lindex [lindex $args [expr {[lsearch -exact $args -to]+1}]] 0];return {}}
'''
 program+='if {[catch {\n'+body+'\n} message options]} {puts stderr "$message\n[dict get $options -errorinfo]";exit 1}\nclose $audit\n'
 result=subprocess.run([TCL],input=program,text=True,capture_output=True);assert result.returncode==0,result.stderr
 return out.read_text(),result.stdout

def test_script_complete_and_no_mutations():
 code='set f [open {'+str(COLLECTOR)+'}];puts [info complete [read $f]];close $f\n'
 assert subprocess.check_output([TCL],input=code,text=True).strip()=='1'
 for action in ('set_property','create_waiver','set_false_path','set_max_delay','read_xdc','write_checkpoint','route_design','opt_design'):assert action not in ENTRY
 assert 'c2_' not in ENTRY and 'route_qualified 1' not in ENTRY
 assert 'open_checkpoint' not in subprocess.check_output([TCL],input='source {'+str(COLLECTOR)+'}\nputs library_loaded\n',text=True)

def test_discovered_names_are_reselected_before_properties(tmp_path):
 report,_=run(tmp_path,r'''
set result [::c1_alex_reset_status::query $audit discovery {get_cells -quiet {WRAPPER/CL/u_rst_core}} cell]
set plain [lindex [dict get $result value] 0]
if {$plain ne "WRAPPER/CL/u_rst_core"} {error "Inventory must use plain observed names"}
::c1_alex_reset_status::cell $audit $plain
if {$::c1_alex_reset_status::capture_pending!=0} {error "Typed reselection failed"}
if {$::selections<10} {error "Properties reused escaped handles"}
''')
 assert 'status error' not in report

def test_bracketed_pin_discovers_async_endpoint(tmp_path):
 report,_=run(tmp_path,r'''
set pin {WRAPPER/CL/u_rst_core/arststages_ff_reg[0]/CLR}
::c1_alex_reset_status::pin $audit $pin
if {![dict exists $::c1_alex_reset_status::endpoints $pin] || $::c1_alex_reset_status::capture_pending!=0} {error "Literal pin identity lost"}
''')
 assert 'arststages_ff_reg[0]/CLR' in report

def test_clock_retains_typed_driver_without_global_sink_fanout(tmp_path):
 report,_=run(tmp_path,r'''
rename get_nets old_get_nets
proc get_nets {args} {
 if {[lsearch -exact $args -of_objects]<0} {return [old_get_nets {*}$args]}
 foreach obj [lindex $args [expr {[lsearch -exact $args -of_objects]+1}]] {untyped $obj}
 return [list [typed net clk]]
}
rename get_pins old_get_pins
proc get_pins {args} {
 if {[lsearch -exact $args -of_objects]<0} {return [old_get_pins {*}$args]}
 foreach obj [lindex $args [expr {[lsearch -exact $args -of_objects]+1}]] {untyped $obj}
 if {[lsearch -exact $args -filter]<0} {error "Global sinks must not expand"}
 return [list [typed pin WRAPPER/CL/bufg/O]]
}
::c1_alex_reset_status::pin $audit {WRAPPER/CL/u_status_cdc/src_ff_reg[0]/C} 1
if {$::c1_alex_reset_status::capture_pending!=0} {error "Clock relation failed"}
''')
 assert 'global_clock_fanout_not_needed' in report
 assert 'WRAPPER/CL/bufg/O' in report and 'status error' not in report

def test_protected_boundary_and_budget(tmp_path):
 report,_=run(tmp_path,r'''
rename get_pins old_get_pins
proc get_pins {args} {error "No protected interior"}
::c1_alex_reset_status::cell $audit {WRAPPER/RL/protected[0]}
if {$::c1_alex_reset_status::capture_pending!=0} {error "External boundary traversed"}
set ::c1_alex_reset_status::max_cells 0
::c1_alex_reset_status::cell $audit WRAPPER/CL/new
''')
 assert 'external_to_CL' in report and 'cell_budget_exceeded' in report

def test_empty_timing_paths_are_not_pass(tmp_path):
 report,_=run(tmp_path,'::c1_alex_reset_status::timing $audit {WRAPPER/CL/u_rst_core/actual[0]/CLR}')
 assert report.count('no_path_is_not_a_passing_recovery_or_removal_check')==4
 assert 'status error' not in report

def test_selector_identity_mismatch_and_wrong_kind_fail_closed(tmp_path):
 report,_=run(tmp_path,r'''
rename get_cells old_get_cells
proc get_cells {args} {return [list [typed cell wrong_name]]}
set result [::c1_alex_reset_status::query $audit bad {::c1_alex_reset_status::property cell wanted NAME}]
if {[dict get $result ok]} {error "Mismatch accepted"}
if {![catch {::c1_alex_reset_status::resolve ambiguous {wanted}}]} {error "Kind guessed"}
''')
 assert 'identity/cardinality mismatch' in report

def test_large_collection_is_expanded_and_plain_names_returned(tmp_path):
 report,_=run(tmp_path,r'''
rename get_pins old_get_pins
proc get_pins {args} {set r {};for {set i 0} {$i<601} {incr i} {lappend r [typed pin "actual_$i"]};return $r}
set result [::c1_alex_reset_status::query $audit large {get_pins -quiet *} pin]
if {![dict get $result ok] || [llength [dict get $result value]]!=601 || [lindex [dict get $result value] end] ne "actual_600"} {error "Serialization lost names"}
''')
 assert 'actual_count 601 serialized_count 601' in report and 'actual_600' in report

def test_net_group_reselects_all_segments_and_preserves_loads(tmp_path):
 report,_=run(tmp_path,r'''
rename get_pins old_get_pins
proc get_pins {args} {
 if {[lsearch -exact $args -of_objects]<0} {return [old_get_pins {*}$args]}
 foreach object [lindex $args [expr {[lsearch -exact $args -of_objects]+1}]] {untyped $object}
 if {[lsearch -exact $args -filter]>=0} {return [list [typed pin WRAPPER/CL/gate/O]]}
 return [list [typed pin WRAPPER/CL/gate/O] [typed pin {WRAPPER/CL/u_rst_core/actual[0]/CLR}] [typed pin {WRAPPER/CL/u_rst_core/actual[1]/CLR}]]
}
set segments {WRAPPER/CL/net[0] WRAPPER/CL/u_rst_core/net[0]}
::c1_alex_reset_status::net $audit {WRAPPER/CL/net[0]} $segments
::c1_alex_reset_status::net $audit {WRAPPER/CL/u_rst_core/net[0]} $segments
if {$::c1_alex_reset_status::edge_count!=3 || $::c1_alex_reset_status::capture_pending!=0} {error "Net graph incomplete"}
''')
 assert 'actual[0]/CLR' in report and 'actual[1]/CLR' in report

def test_mixed_fanin_roots_retain_observed_types(tmp_path):
 report,_=run(tmp_path,r'''
proc all_fanin {args} {return [list [typed pin WRAPPER/CL/clock/LOCKED] [typed port reset_port]]}
set result [::c1_alex_reset_status::query $audit roots {::c1_alex_reset_status::fanin WRAPPER/CL/reset/CLR root} root]
foreach root [dict get $result value] {::c1_alex_reset_status::props $audit root [dict get $root kind] [dict get $root name] {NAME CLASS DIRECTION}}
if {$::c1_alex_reset_status::capture_pending!=0} {error "Mixed root type lost"}
''')
 assert 'kind pin name WRAPPER/CL/clock/LOCKED' in report and 'kind port name reset_port' in report

def test_ephemeral_timing_paths_are_inspected_in_query_frame(tmp_path):
 report,_=run(tmp_path,r'''
rename get_timing_paths old_timing
proc get_timing_paths {args} {return [list [typed timing path1]]}
rename list_property old_list_property
proc list_property {path} {untyped $path;return {SLACK ENDPOINT_PIN}}
::c1_alex_reset_status::timing $audit WRAPPER/CL/reset/CLR
if {$::c1_alex_reset_status::capture_pending!=0} {error "Timing path handle escaped"}
''')
 assert 'reset_timing_path' in report and 'properties {ENDPOINT_PIN 0 SLACK 0}' in report


def test_missing_pin_polarity_remains_explicit_even_after_cached_pin(tmp_path):
 report,_=run(tmp_path,r'''
set name {WRAPPER/CL/u_rst_hbm/stage[0]/CLR}
::c1_alex_reset_status::pin $audit $name
rename get_property original_property
proc get_property {key obj} {
 if {$key eq "IS_INVERTED"} {untyped [lindex $obj 0];return {}}
 return [original_property $key $obj]
}
::c1_alex_reset_status::pin $audit $name 1
if {$::c1_alex_reset_status::capture_pending!=1} {error "Missing polarity accepted"}
''')
 assert 'missing_effective_pin_polarity' in report


def test_cell_inversion_inventory_retains_blank_and_absent_properties(tmp_path):
 report,_=run(tmp_path,r'''
rename get_property original_property
proc get_property {key obj} {
 if {$key eq "IS_CLR_INVERTED"} {untyped [lindex $obj 0];return {}}
 return [original_property $key $obj]
}
::c1_alex_reset_status::inversions $audit WRAPPER/CL/u_rst_hbm/stage
''')
 assert 'available {IS_C_INVERTED IS_CLR_INVERTED}' in report
 assert 'unavailable {IS_PRE_INVERTED IS_R_INVERTED' in report
 assert 'IS_CLR_INVERTED {ok 1 value {}}' in report


def test_query_and_output_caps_fail_closed(tmp_path):
 report,_=run(tmp_path,r'''
set ::c1_alex_reset_status::max_query_objects 1
set result [::c1_alex_reset_status::query $audit capped {get_cells -quiet {one two}} cell]
if {[dict get $result ok] || $::c1_alex_reset_status::capture_pending!=1} {error "Query cap accepted"}
set ::c1_alex_reset_status::max_record_bytes 8
if {![catch {::c1_alex_reset_status::emit $audit {this record exceeds eight bytes}}]} {error "Record cap accepted"}
set ::c1_alex_reset_status::max_record_bytes 1000
set ::c1_alex_reset_status::max_output_bytes [tell $audit]
if {![catch {::c1_alex_reset_status::emit $audit next}]} {error "Total cap accepted"}
''')
 assert 'Typed query object cap reached' in report


def test_capture_rejects_wrong_version_before_checkpoint_access(tmp_path):
 _,_=run(tmp_path,r'''
proc version {args} {return 2025.1}
if {![catch {::c1_alex_reset_status::capture nofile [string repeat a 64] nooutput} msg] || $msg ne "Unreviewed Vivado version"} {error "Wrong version accepted"}
''')


def test_capture_hash_binding_summary_and_read_only_lifecycle(tmp_path):
 # Exercise entry-point orchestration without a real Vivado checkpoint.
 report,_=run(tmp_path,r'''
proc version {args} {return 2025.2}
rename ::c1_alex_reset_status::hash ::c1_alex_reset_status::real_hash
proc ::c1_alex_reset_status::hash {path} {return [string repeat a 64]}
set checkpoint [file join [file dirname {TMP_FILE}] input.dcp]
set output [file join [file dirname $checkpoint] captured]
set f [open $checkpoint w];puts $f stub;close $f
set ::lifecycle {}
proc get_designs {args} {return {}}
proc set_param {key value} {if {$key ne "general.maxThreads" || $value!=4} {error "Unexpected parameter"};lappend ::lifecycle threads4}
proc open_checkpoint {path} {lappend ::lifecycle open}
proc close_design {} {lappend ::lifecycle close}
rename ::c1_alex_reset_status::collect ::c1_alex_reset_status::real_collect
proc ::c1_alex_reset_status::collect {audit output} {missing $audit selected absent_required_property}
set result [::c1_alex_reset_status::capture $checkpoint [string repeat a 64] $output]
if {[dict get $result route_qualified]!=0 || [dict get $result hardware_measured]!=0 || [dict get $result incomplete_or_error]!=1 || $::lifecycle ne {threads4 open close}} {error "Raw diagnostic claimed success"}
set f [open [file join $output topology.tcldata]];set raw [read $f];close $f
if {[string first c1_reset_status_diagnostic $raw]<0 || [string first {incomplete_or_error 1 route_qualified 0} $raw]<0} {error "Header/summary absent"}
if {![catch {::c1_alex_reset_status::capture $checkpoint [string repeat a 64] $output} msg] || [string first fresh $msg]<0} {error "Existing output reused"}
rename get_designs empty_designs
proc get_designs {args} {return already_open}
if {![catch {::c1_alex_reset_status::capture $checkpoint [string repeat a 64] ${output}2} msg] || [string first {fresh empty} $msg]<0} {error "Existing design accepted"}
if {![catch {::c1_alex_reset_status::capture $checkpoint [string repeat b 64] ${output}2} msg] || [string first hash/type $msg]<0} {error "Hash mismatch accepted"}
'''.replace('TMP_FILE',str(tmp_path/'entry')))


def test_clock_many_aliases_preserve_exact_driver_without_sink_walk(tmp_path):
 report,_=run(tmp_path,r'''
rename get_nets old_get_nets
proc get_nets {args} {
 if {[lsearch -exact $args -of_objects]<0} {return [old_get_nets {*}$args]}
 set result {};for {set i 0} {$i<100} {incr i} {lappend result [typed net clock_alias_$i]};return $result
}
rename get_pins old_get_pins
proc get_pins {args} {
 if {[lsearch -exact $args -of_objects]<0} {return [old_get_pins {*}$args]}
 if {[lsearch -exact $args -filter]<0} {error "Global sinks expanded"}
 set collection [lindex $args [expr {[lsearch -exact $args -of_objects]+1}]]
 if {[llength $collection]!=100} {error "Aliases truncated"}
 foreach object $collection {untyped $object}
 return [list [typed pin WRAPPER/CL/actual_bufg/O]]
}
::c1_alex_reset_status::clock_driver $audit WRAPPER/CL/stage/C
if {$::c1_alex_reset_status::capture_pending} {error "Bounded clock alias discovery failed"}
''')
 assert 'selected 100 cap 4096' in report and 'WRAPPER/CL/actual_bufg/O' in report
