"""Fail-closed C2 HBM methodology context tests; mocked queries are not timing proof."""
from pathlib import Path
import shutil
import subprocess
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "boards/c2_f2/check_scoped_clocks.tcl"
# Vivado export framing: repeated generated ID0 and multiline commands are data.
EXPORT_PREAMBLE = r"""
# id:0
set_false_path -setup -hold -to [get_cells [list {WRAPPER/RL/a[0]} \
          {WRAPPER/RL/a[1]}]]
# id:0
set_false_path -setup -hold -to [get_pins {WRAPPER/RL/b[0]/CE}]
current_instance WRAPPER/CL
# id:195
set _xlnx_shared_i1 [get_pins {u_unused/Q}]
set_false_path -through $_xlnx_shared_i1
current_instance -quiet
"""
MOCKS = r'''
set objs {}; set clocks {}; set pins {}; set members {}; set violations {}
proc add {name props} {
    set id @o[dict size $::objs]
    dict set ::objs $id [dict merge [dict create NAME $name] $props]
    return $id
}
proc prop {id key} {return [dict get $::objs $id $key]}
proc list_property {id} {return [dict keys [dict get $::objs $id]]}
proc get_property {key ids} {
    set result {}
    foreach id $ids {
        if {[dict exists $::objs $id $key]} {lappend result [prop $id $key]} \
        else {lappend result {}}
    }
    if {[llength $ids] > 1 && (($key eq "NAME" && $::fault eq "bulk_name_truncated") ||
        ($key eq "CLASS" && $::fault eq "bulk_class_truncated"))} {
        return [lrange $result 1 end]
    }
    if {[llength $ids] == 1} {return [lindex $result 0]}
    return $result
}
dict for {name props} [::otpu_scoped::clock_data] {
    set id [add $name [dict merge $props {CLASS clock IS_GENERATED 1 IS_VIRTUAL 0 IS_PROPAGATED 1}]]
    dict set clocks $name $id
}
foreach {pin clock} {
 WRAPPER/CL/clk_main_a0 clk_main_a0
 WRAPPER/CL/u_clk/O clk_raw
 WRAPPER/CL/u_hclk/O hclk_raw
} {dict set pins $pin [add $pin [dict create CLASS pin CLOCK $clock]]}
foreach {name clock} {
 WRAPPER/CL/main_reg clk_main_a0
 {WRAPPER/RL_SHIM/PIPE_RST_OUT_N/pipe_reg[0][0]} clk_main_a0
 WRAPPER/RL/other_reg clk_main_a0
 WRAPPER/CL/core_reg clk_raw
 WRAPPER/CL/hbm_reg hclk_raw
} {
    set id [add $name {CLASS cell}]
    dict set cells $name $id
    dict lappend members $clock $id
}
set positions {72 73 74 75 76 77}
if {$fault eq "renumber"} {set positions {310 1024 7 88 999 42}}
set i 0
foreach pos [list [lindex $positions 0] [lindex $positions 1] [lindex $positions 4] [lindex $positions 5]] {
    incr i
    set pair {clk_main_a0 and clk_raw}
    if {$i > 2} {set pair {clk_main_a0 and hclk_raw}}
    set description "A scoped Max Delay Datapath Only timing constraint is set between clocks $pair (see constraint position $pos in the Timing Constraint window in Vivado IDE). It is not recommended to define such scoped constraints between clocks as the constraint impacts timing paths outside of the scope."
    dict set violations TIMING-54#$i [add TIMING-54#$i [dict create CLASS drc_violation CHECK TIMING-54 SEVERITY {Critical Warning} IS_WAIVED 0 WAIVER {} DESCRIPTION $description]]
}
dict set violations TIMING-17#1 [add TIMING-17#1 {CHECK TIMING-17 SEVERITY {Critical Warning}}]
set supplied [dict keys $violations]
set v [dict get $violations TIMING-54#1]
set cc [dict get $clocks clk_raw]
switch $fault {
    caller_subset {set supplied [lrange $supplied 1 end]}
    caller_duplicate {lappend supplied TIMING-54#1}
    missing_finding {dict unset violations TIMING-54#3;set supplied [dict keys $violations]}
    wrong_check {dict set objs $v CHECK TIMING-17}
    missing_check {dict unset objs $v CHECK}
    severity {dict set objs $v SEVERITY Warning}
    fatal {dict set objs $v SEVERITY Fatal}
    class {dict set objs $v CLASS clock}
    waived {dict set objs $v IS_WAIVED 1}
    unknown_waiver {dict set objs $v IS_WAIVED unknown}
    missing_waiver {dict unset objs $v IS_WAIVED}
    waiver_reference {dict set objs $v WAIVER W1}
    missing_severity {dict unset objs $v SEVERITY}
    missing_description {dict unset objs $v DESCRIPTION}
    description {dict set objs $v DESCRIPTION unreviewed}
    wrong_position {dict set objs $v DESCRIPTION [string map {{position 72} {position 9999}} [prop $v DESCRIPTION]]}
    duplicate_position {set v3 [dict get $violations TIMING-54#3];dict set objs $v3 DESCRIPTION [prop [dict get $violations TIMING-54#2] DESCRIPTION]}
    period {dict set objs $cc PERIOD 11.000}
    master {dict set objs $cc MASTER_CLOCK unreviewed}
    source {dict set objs $cc SOURCE_PINS WRAPPER/CL/other/O}
    wrong_description_pair {dict set objs $v DESCRIPTION [string map {clk_raw hclk_raw} [prop $v DESCRIPTION]]}
    generated {dict set objs $cc IS_GENERATED 0}
    missing_clock_property {dict unset objs $cc SOURCE_PINS}
    clock_name {dict set objs $cc NAME unexpected}
    core_escape {dict lappend members clk_raw [add WRAPPER/RL/extra_core {CLASS cell}]}
    hbm_escape {dict lappend members hclk_raw [add WRAPPER/RL/hbm_escape {CLASS cell}]}
    hbm_period {dict set objs [dict get $clocks hclk_raw] PERIOD 2.500}
    empty_members {dict set members clk_raw {}}
    duplicate_members {dict lappend members clk_raw [lindex [dict get $members clk_raw] 0]}
    bulk_name_missing {dict unset objs [dict get $cells WRAPPER/CL/main_reg] NAME}
    bulk_name_empty {dict set objs [dict get $cells WRAPPER/CL/main_reg] NAME {}}
    bulk_class_missing {dict unset objs [dict get $cells WRAPPER/RL/other_reg] CLASS}
    bulk_class_empty {dict set objs [dict get $cells WRAPPER/RL/other_reg] CLASS {}}
    bulk_class_wrong {dict set objs [dict get $cells WRAPPER/RL/other_reg] CLASS clock}
    new_critical {
        dict set violations OTHER#1 [add OTHER#1 {CHECK OTHER SEVERITY Error}]
        lappend supplied OTHER#1
    }
}
proc current_instance {} {if {$::fault eq "scope"} {return WRAPPER/CL};return {}}
proc version {args} {if {$::fault eq "version"} {return 2025.1};return 2025.2}
proc get_methodology_violations {args} {
    if {![llength $args]} {return [dict values $::violations]}
    return [dict get $::violations [lindex $args 0]]
}
proc get_clocks {args} {
    if {[lindex $args 0] eq "-of_objects"} {
        if {$::fault eq "clock_pin"} {return [dict get $::clocks clk_raw]}
        return [dict get $::clocks [prop [lindex $args 1] CLOCK]]
    }
    return [dict get $::clocks [lindex $args 0]]
}
proc get_pins {args} {return [dict get $::pins [lindex $args 0]]}
proc get_cells {names} {
    set result {};foreach name $names {lappend result [dict get $::cells $name]};return $result
}
proc all_registers {args} {return [dict get $::members [prop [lindex $args 1] NAME]]}
proc all_inputs {} {return @inputs}
proc all_outputs {} {return @outputs}
proc write_xdc {args} {
    if {[lrange $args 0 end-1] ne {-type timing -constraints all -write_id}} {error "writer flags changed"}
    set f [open [lindex $args end] w]
    set fixture [open $::schema_fixture r];puts $f [read $fixture];close $fixture
    set scope WRAPPER/CL;if {$::fault eq "export_scope"} {set scope WRAPPER}
    puts $f "current_instance $scope"
    if {$::fault eq "quiet_scope"} {puts $f {current_instance -quiet}}
    set n 0
    set actual [::otpu_scoped::read_export $::actual_c2_fixture]
    set actual_commands {}
    foreach pos {72 73 74 75 76 77} {lappend actual_commands [lindex [::otpu_scoped::unique_export $actual $pos] 1]}
    foreach cmd $actual_commands pos $::positions {
        incr n
        if {$n==1 && $::fault eq "changed_bound"} {set cmd [string map {5.000 4.000} $cmd]}
        if {$n==3 && $::fault eq "changed_internal_bound"} {set cmd [string map {2.222 5.000} $cmd]}
        if {$n==1 && $::fault eq "changed_command"} {append cmd { -allow_paths}}
        if {$n==3 && $::fault eq "missing_export"} {continue}
        if {$n==1 && $::fault eq "trailing_command"} {append cmd {; set_false_path -from DUMMY}}
        puts $f "# id:$pos";puts $f $cmd
        if {$n==1 && $::fault eq "next_command"} {puts $f {set_false_path -from DUMMY}}
        if {$n==1 && $::fault eq "duplicate_export"} {puts $f "# id:900";puts $f $cmd}
        if {$n==1 && $::fault eq "duplicate_id"} {puts $f "# id:$pos";puts $f {set_false_path -from DUMMY}}
    }
    if {$::fault eq "empty_position"} {puts $f {# id:901}}
    if {$::fault eq "incomplete_command"} {puts $f "# id:901";puts $f {set_false_path -to [get_cells}}
    if {$::fault eq "unnumbered_duplicate"} {puts $f {current_instance WRAPPER/CL};puts $f [lindex [::otpu_scoped::commands] 0]}
    close $f
}
proc path {src dst sp ep exception requirement} {
    return [add PATH[dict size $::objs] [dict create STARTPOINT_CLOCK $src ENDPOINT_CLOCK $dst STARTPOINT_PIN $sp ENDPOINT_PIN $ep EXCEPTION $exception REQUIREMENT $requirement SLACK 1.000 DATAPATH_DELAY 2.000]]
}
proc resets {dst ignored} {
    set result {}
    set period [dict get [::otpu_scoped::clock_data] $dst PERIOD]
    set stem core_reset_n; if {$dst eq "hclk_raw"} {set stem hbm_reset_n}
    for {set k 0} {$k<3} {incr k} {
        if {$ignored || $::fault eq "reset_fix"} {set ex {False Path};set req {}} \
        else {set ex "MaxDelay Path ${period}ns -datapath_only";set req $period}
        set sp {WRAPPER/RL_SHIM/PIPE_RST_OUT_N/pipe_reg[0][0]/C}
        set ep [format {WRAPPER/CL/%s_reg[%d]/CLR} $stem $k]
        if {$::fault eq "wrong_reset_source"} {set sp WRAPPER/RL/other_reg/C}
        if {$::fault eq "wrong_reset_endpoint"} {set ep WRAPPER/CL/core_reg/D}
        lappend result [path clk_main_a0 $dst $sp $ep $ex $req]
    }
    return [concat $result $result]
}
proc report_timing {args} {
    set f [open [lindex $args end] w];puts $f "mock raw report";close $f
    # Pollute common globals as tool commands can; the production helper must
    # keep its own canonical selections and state procedure-local.
    set ::src CORRUPTED;set ::dst CORRUPTED;set ::name CORRUPTED
}
proc arg {args flag} {return [lindex $args [expr {[lsearch -exact $args $flag]+1}]]}
proc get_timing_paths {args} {
    set from [arg $args -from];set to [arg $args -to]
    set ignored [expr {"-user_ignored" in $args}]
    if {$from eq "@inputs" || $to eq "@outputs"} {
        if {$::fault eq "extra_port"} {return [path clk_main_a0 clk_raw PORT WRAPPER/CL/reg/D {False Path} {}]}
        return {}
    }
    set sk [prop [lindex $from 0] CLASS];set dk [prop [lindex $to 0] CLASS]
    if {$sk eq "cell" || $dk eq "cell"} {
        if {$sk eq "cell" && [prop $to NAME] in {clk_raw hclk_raw}} {
            if {$ignored && $::fault ni {reset_fix wrong_reset_source wrong_reset_endpoint}} {return {}}
            return [resets [prop $to NAME] $ignored]
        }
        if {$::fault eq "external_data"} {return [path clk_raw clk_main_a0 WRAPPER/CL/source/C WRAPPER/RL/external/D {MaxDelay Path 4.000ns -datapath_only} 4.000]}
        return {}
    }
    set src [prop $from NAME];set dst [prop $to NAME]
    if {$ignored} {
        if {$::fault eq "query_cap"} {return [lrepeat 10000 [path $src $dst X/C Y/D {False Path} {}]]}
        if {$::fault eq "ignored_data"} {return [path $src $dst WRAPPER/CL/source/C WRAPPER/CL/data/D {False Path} {}]}
        if {$::fault in {reset_fix wrong_reset_source wrong_reset_endpoint} && $src eq "clk_main_a0"} {return [resets $dst 1]}
        return {}
    }
    if {$::fault eq "no_data"} {return {}}
    if {$::fault eq "no_internal_data" && $src eq "clk_raw" && $dst eq "hclk_raw"} {return {}}
    set req [dict get [::otpu_scoped::clock_data] $dst PERIOD]
    set ex "MaxDelay Path ${req}ns -datapath_only"
    if {$::fault eq "missing_bound"} {set ex {}}
    if {$::fault eq "missing_datapath_flag"} {set ex "MaxDelay Path ${req}ns"}
    if {$::fault eq "unknown_bound_flag"} {append ex { -unknown}}
    if {$::fault eq "wrong_requirement"} {set req 10.000}
    if {$::fault eq "wrong_actual_pair"} {set dst clk_unreviewed}
    return [path $src $dst WRAPPER/CL/source/C WRAPPER/CL/data/D $ex $req]
}
'''

@pytest.mark.parametrize('fault,expected', [
    ('none', None), ('renumber', None), ('reset_fix', None), ('new_critical', None),
    ('caller_subset', 'inventory'), ('caller_duplicate', 'inventory'),
    ('missing_finding', 'exact four'), ('wrong_check', 'inconsistent'),
    ('missing_check', 'missing CHECK'), ('severity', 'identity/severity'),
    ('fatal', 'identity/severity'), ('class', 'identity/severity'),
    ('waived', 'waiver state'), ('unknown_waiver', 'waiver state'),
    ('missing_waiver', 'missing IS_WAIVED'), ('waiver_reference', 'identity/severity'),
    ('missing_severity', 'missing SEVERITY'), ('missing_description', 'missing DESCRIPTION'),
    ('description', 'description'), ('wrong_description_pair', 'clock pair differs'), ('wrong_position', 'position missing'),
    ('duplicate_position', 'uniquely'), ('period', 'PERIOD differs'),
    ('master', 'MASTER_CLOCK differs'), ('source', 'SOURCE_PINS differs'),
    ('generated', 'IS_GENERATED differs'), ('missing_clock_property', 'missing SOURCE_PINS'),
    ('clock_name', 'clock identity'), ('core_escape', 'core/HBM clock escapes'),
    ('hbm_escape', 'core/HBM clock escapes'), ('hbm_period', 'PERIOD differs'), ('empty_members', 'membership'),
    ('duplicate_members', 'membership'), ('scope', 'top-level'),
    ('bulk_name_missing', 'missing or empty NAME'), ('bulk_name_empty', 'missing or empty NAME'),
    ('bulk_name_truncated', 'NAME result cardinality'),
    ('bulk_class_missing', 'missing or empty CLASS'), ('bulk_class_empty', 'missing or empty CLASS'),
    ('bulk_class_wrong', 'cell class mismatch'), ('bulk_class_truncated', 'CLASS result cardinality'),
    ('version', 'version'), ('clock_pin', 'clock-pin resolution'),
    ('export_scope', 'wrong scope'), ('changed_bound', 'missing or changed'),
    ('changed_command', 'missing or changed'), ('missing_export', 'missing or changed'),
    ('duplicate_export', 'duplicate reviewed'), ('duplicate_id', 'ambiguous export position'),
    ('quiet_scope', 'wrong scope'), ('trailing_command', 'missing or changed'),
    ('next_command', 'ambiguous export command bundle'), ('empty_position', 'without a command'),
    ('incomplete_command', 'incomplete export command'), ('unnumbered_duplicate', 'ambiguous export position'),
    ('wrong_reset_source', 'outside-boundary'), ('wrong_reset_endpoint', 'outside-boundary'),
    ('external_data', 'outside-boundary'), ('extra_port', 'top-level I/O'),
    ('query_cap', 'query limit'), ('ignored_data', 'ignored data'),
    ('no_data', 'data-path witness'), ('missing_bound', 'datapath bound'),
    ('wrong_requirement', 'datapath bound'), ('wrong_actual_pair', 'clock pair changed'),
    ('no_internal_data', 'data-path witness'), ('changed_internal_bound', 'missing or changed'),
    ('missing_datapath_flag', 'datapath bound'), ('unknown_bound_flag', 'datapath bound'),
])
def test_scoped_clock_classifier(tmp_path, fault, expected):
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl unavailable')
    prefix = tmp_path / 'review'
    fixture = tmp_path / 'actual-schema.xdc'
    fixture.write_text(EXPORT_PREAMBLE)
    script = (f'source {{{SCRIPT}}}\nset fault {fault}\nset schema_fixture {{{fixture}}}\nset actual_c2_fixture {{{SCRIPT.parents[1].parent / "tests/fixtures/c2_hbm_scoped_post_opt.xdc"}}}\n' + MOCKS +
              f'\nset rc [catch {{set result [otpu_classify_scoped_clocks $supplied {{{prefix}}} stdout]}} message]\n' + r'''
if {$rc} {puts stderr $message;exit 1}
if {[dict get $result context] ne {TIMING-54#1 TIMING-54#2 TIMING-54#3 TIMING-54#4}} {puts stderr "wrong accepted set";exit 1}
set remaining [dict get $result remaining]
if {"TIMING-17#1" ni $remaining} {puts stderr "accepted TIMING17";exit 1}
if {$fault eq "new_critical" && "OTHER#1" ni $remaining} {puts stderr "accepted new finding";exit 1}
foreach name [dict get $result context] {
    set v [dict get $violations $name]
    if {[prop $v SEVERITY] ne "Critical Warning" || [prop $v IS_WAIVED]!=0} {puts stderr "mutated finding";exit 1}
}
puts CLASSIFICATION_CHECK_PASSED
''')
    driver = tmp_path / "driver.tcl"
    driver.write_text(script)
    result = subprocess.run([tclsh, str(driver)], text=True, capture_output=True, timeout=10)
    if expected:
        assert result.returncode != 0, result.stdout
        assert expected in result.stderr
        assert 'classification reviewed_scoped_clock_context' not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.count('classification reviewed_scoped_clock_context') == 4
        assert 'CLASSIFICATION_CHECK_PASSED' in result.stdout
        assert (tmp_path / 'review_scoped_clocks.xdc').exists()
        assert len(list(tmp_path.glob('review_*.rpt'))) == 12


@pytest.mark.parametrize('name', ['WRAPPER/CL/u[0]/Q', 'WRAPPER/CL/a b/Q', r'WRAPPER/CL/a\b/Q'])
def test_singleton_names_remain_canonical_lists(tmp_path, name):
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl unavailable')
    script = (f'source {{{SCRIPT}}}\nset expected {{{name}}}\n' + r"""
proc get_property {key objects} {
    if {$key ne "NAME" || $objects ne "@one"} {error "unexpected query"}
    return $::expected
}
set rc [catch {
    set result [::otpu_scoped::names {@one}]
    if {$result ne [list $expected]} {error "singleton name lost canonical list form"}
} message]
if {$rc} {puts stderr $message;exit 1}
puts SINGLETON_NAME_PASSED
""")
    driver = tmp_path / "driver.tcl"
    driver.write_text(script)
    result = subprocess.run([tclsh, str(driver)], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'SINGLETON_NAME_PASSED' in result.stdout


@pytest.mark.parametrize("fault", ["none", "boundary_data", "internal_bound", "ignored_reset_requirement"])
def test_captured_post_opt_objects(tmp_path, fault):
    """Replay captured values, not a netlist: the separate Vivado check is authoritative."""
    tclsh = shutil.which("tclsh")
    if not tclsh:
        pytest.skip("Tcl unavailable")
    fixture_dir = Path(__file__).parent / "fixtures"
    schema = tmp_path / "schema.xdc"
    schema.write_text(EXPORT_PREAMBLE)
    script = (f"source {{{SCRIPT}}}\nset fault none\nset schema_fixture {{{schema}}}\n"
              f"set actual_c2_fixture {{{fixture_dir / 'c2_hbm_scoped_post_opt.xdc'}}}\n" + MOCKS +
              f"\nset capture {{{fixture_dir / 'c2_hbm_scoped_post_opt.tcldata'}}}\nset capture_fault {fault}\n" + r'''
set recorded {}; set counts {}; set pending {}
set f [open $capture r]
while {[gets $f line] >= 0} {
    if {$pending eq "" && ([string trim $line] eq "" || [string index [string trim $line] 0] eq "#")} {continue}
    append pending $line \n
    if {![info complete $pending]} {continue}
    set record [string trim $pending]; set pending {}
    lassign $record kind tag data
    switch -- $kind {
        clock {dict set objs [dict get $clocks [dict get $tag NAME]] $tag}
        methodology {dict set objs [dict get $violations [dict get $tag NAME]] $tag}
        query {
            dict set counts $tag [dict get [lrange $record 2 end] count]
            if {[string match *_inputs $tag] || [string match *_outputs $tag]} {
                if {[dict get $counts $tag] != 0} {error "unexpected recorded I/O path"}
            }
        }
        path {
            if {$capture_fault eq "boundary_data" && $tag eq "clk_main_a0_to_hclk_raw_ignored0_source"} {
                dict set data ENDPOINT_PIN WRAPPER/CL/u_sys/u_mem/data_reg/D
            }
            if {$capture_fault eq "internal_bound" && $tag eq "hclk_raw_to_clk_raw_ignored0"} {
                dict set data EXCEPTION {MaxDelay Path 2.222ns -datapath_only}
            }
            if {$capture_fault eq "ignored_reset_requirement" && $tag eq "clk_main_a0_to_clk_raw_ignored1_source"} {
                dict set data REQUIREMENT 5.000
            }
            dict lappend recorded $tag [add RECORDED[dict size $objs] $data]
        }
        default {error "unknown captured record kind"}
    }
}
close $f
if {$pending ne ""} {error "incomplete captured record"}
proc get_timing_paths {args} {
    set from [arg $args -from]; set to [arg $args -to]
    if {$from eq "@inputs" || $to eq "@outputs"} {return {}}
    set ignored [expr {"-user_ignored" in $args}]; set side {}
    if {[prop [lindex $from 0] CLASS] eq "cell"} {set src clk_main_a0; set side _source} else {set src [prop $from NAME]}
    if {[prop [lindex $to 0] CLASS] eq "cell"} {set dst clk_main_a0; set side _destination} else {set dst [prop $to NAME]}
    set tag ${src}_to_${dst}_ignored${ignored}${side}
    set count [dict get $::counts $tag]; set paths {}
    if {[dict exists $::recorded $tag]} {set paths [dict get $::recorded $tag]}
    if {[llength $paths] != $count} {error "capture count mismatch: $tag"}
    return $paths
}
''' + f"\nset result [otpu_classify_scoped_clocks $supplied {{{tmp_path / 'capture'}}} stdout]\n"
              'puts "CAPTURE_CHECK_PASSED [llength [dict get $result context]]"\n')
    driver = tmp_path / "capture.tcl"
    driver.write_text(script)
    result = subprocess.run([tclsh, str(driver)], text=True, capture_output=True, timeout=10)
    if fault == "none":
        assert result.returncode == 0, result.stderr
        assert "CAPTURE_CHECK_PASSED 4" in result.stdout
    else:
        assert result.returncode != 0
        expected = {"boundary_data": "outside-boundary", "internal_bound": "datapath bound",
                    "ignored_reset_requirement": "reset entry exception"}[fault]
        assert expected in result.stderr
