"""Exercise scoped-clock context classification without waiving timing checks."""
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'boards/f2/build/scripts/check_scoped_clocks.tcl'

# Representative lines from Vivado 2025.2's actual route export (SHA256
# c36649dfee05aef744810a45ce90a5a7ff0f180ab628068d5c73e65a1cc7e059).
# The long generated object lists are shortened; repeated id:0, continuation,
# command bundles, and current_instance -quiet retain the writer's real schema.
EXPORT_PREAMBLE = r"""
# Vivado Generated XDC timing constraints
# id:0
set_false_path -setup -hold -to [get_cells [list {WRAPPER/RL_SHIM/SDA_REG_SLC/inst/r.r_pipe/m_payload_i_reg[32]} \
          {WRAPPER/RL_SHIM/SDA_REG_SLC/inst/r.r_pipe/m_payload_i_reg[33]}]]
# id:0
set_false_path -setup -hold -to [get_pins [list {WRAPPER/RL_SHIM/SDA_REG_SLC/inst/w.w_pipe/m_payload_i_reg[0]/CE} \
          {WRAPPER/RL_SHIM/DMA_PCIS_AXI_REG_SLC/inst/ar.ar_pipe/m_payload_i_reg[47]/D}]]
current_instance WRAPPER/CL
# id:195
set _xlnx_shared_i1 [get_pins -quiet -of_objects [get_cells -quiet -hierarchical -filter {ORIG_REF_NAME == otpu_pump_bank || REF_NAME == otpu_pump_bank}] -filter {REF_PIN_NAME =~ qa[*] || REF_PIN_NAME =~ qb[*]}]
set_multicycle_path -setup -start -from [get_clocks -quiet -of_objects [get_pins -quiet {u_sys/u_board/u_slice/u_tmem/g_one.g_bank[0].g_sub[0].g_pmp.u_pb/clk_f}]] -through $_xlnx_shared_i1 -to [get_clocks -quiet -of_objects [get_pins -quiet {u_sys/u_board/u_slice/u_tmem/g_one.g_bank[0].g_sub[0].g_pmp.u_pb/clk}]] 2
#revert back to original instance
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
# the CL ($kind: ddr or hbm) and its core period ($t_core, the clock_data's clk_raw): the main
# -> core bound is 4.000 on the DDR CL, the core period on the HBM CL (bounds)
set t_mc [dict get [::otpu_scoped::bounds $kind $t_core] {clk_main_a0 clk_raw}]
dict for {name props} [::otpu_scoped::clock_data $kind $t_core] {
    set id [add $name [dict merge $props {CLASS clock IS_GENERATED 1 IS_VIRTUAL 0 IS_PROPAGATED 1}]]
    dict set clocks $name $id
}
if {$fault eq "hbm_ui_clock"} {dict set clocks mmcm_clkout0 [add mmcm_clkout0 {CLASS clock PERIOD 3.752}]}
set ddr_pin [::otpu_scoped::ddr_pin]
foreach {pin clock} {
 WRAPPER/CL/clk_main_a0 clk_main_a0
 WRAPPER/CL/u_clk/O clk_raw
} {dict set pins $pin [add $pin [dict create CLASS pin CLOCK $clock]]}
if {$kind eq "ddr" || $fault eq "hbm_ddr_present"} {dict set pins $ddr_pin [add $ddr_pin {CLASS pin CLOCK mmcm_clkout0}]}
set cell_clocks {
 WRAPPER/CL/main_reg clk_main_a0
 {WRAPPER/RL_SHIM/PIPE_RST_OUT_N/pipe_reg[0][0]} clk_main_a0
 WRAPPER/RL/other_reg clk_main_a0
 WRAPPER/CL/core_reg clk_raw
}
if {$kind eq "ddr"} {lappend cell_clocks WRAPPER/CL/SH_DDR/ddr_reg mmcm_clkout0}
foreach {name clock} $cell_clocks {
    set id [add $name {CLASS cell}]
    dict set cells $name $id
    dict lappend members $clock $id
}
# the SmartConnect fabric: the HBM CL's marker cell (not a clock member here)
if {$kind eq "hbm" || $fault eq "ddr_with_sc"} {dict set cells WRAPPER/CL/u_sc [add WRAPPER/CL/u_sc {CLASS cell}]}
if {$kind eq "ddr"} {set positions {108 176 177}} else {set positions {612 613}}
if {$fault eq "renumber"} {if {$kind eq "ddr"} {set positions {310 1024 7}} else {set positions {7 1024}}}
set i 0
foreach {cmd role} [::otpu_scoped::commands $kind $t_core] pos $positions {
    incr i
    if {$role eq "group"} {set dkind {Clock Group};set pair {clk_main_a0 and mmcm_clkout0}} \
    else {set dkind {Max Delay Datapath Only};set pair {clk_main_a0 and clk_raw}}
    if {$fault eq "hbm_ddr_finding" && $i == 2} {set pair {clk_main_a0 and mmcm_clkout0}}
    set description "A scoped $dkind timing constraint is set between clocks $pair (see constraint position $pos in the Timing Constraint window in Vivado IDE). It is not recommended to define such scoped constraints between clocks as the constraint impacts timing paths outside of the scope."
    dict set violations TIMING-54#$i [add TIMING-54#$i [dict create CLASS drc_violation CHECK TIMING-54 SEVERITY {Critical Warning} IS_WAIVED 0 WAIVER {} DESCRIPTION $description]]
}
set nctx $i
if {$fault eq "hbm_third_scoped"} {
    # a third scoped constraint on the HBM CL (a clock group at position 700, exported below)
    set description "A scoped Clock Group timing constraint is set between clocks clk_main_a0 and clk_raw (see constraint position 700 in the Timing Constraint window in Vivado IDE). It is not recommended to define such scoped constraints between clocks as the constraint impacts timing paths outside of the scope."
    dict set violations TIMING-54#3 [add TIMING-54#3 [dict create CLASS drc_violation CHECK TIMING-54 SEVERITY {Critical Warning} IS_WAIVED 0 WAIVER {} DESCRIPTION $description]]
}
dict set violations TIMING-17#1 [add TIMING-17#1 {CHECK TIMING-17 SEVERITY {Critical Warning}}]
set supplied [dict keys $violations]
set v [dict get $violations TIMING-54#1]
set cc [dict get $clocks clk_raw]
switch $fault {
    caller_subset {set supplied [lrange $supplied 1 end]}
    caller_duplicate {lappend supplied TIMING-54#1}
    missing_finding {dict unset violations TIMING-54#$nctx;set supplied [dict keys $violations]}
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
    wrong_position {dict set objs $v DESCRIPTION [string map {{position 108} {position 109}} [prop $v DESCRIPTION]]}
    duplicate_position {set v3 [dict get $violations TIMING-54#$nctx];dict set objs $v3 DESCRIPTION [prop [dict get $violations TIMING-54#1] DESCRIPTION]}
    period {dict set objs $cc PERIOD 11.000}
    period_rendering {dict set objs $cc PERIOD [format %.5f $t_core]}
    master {dict set objs $cc MASTER_CLOCK unreviewed}
    source {dict set objs $cc SOURCE_PINS WRAPPER/CL/other/O}
    generated {dict set objs $cc IS_GENERATED 0}
    missing_clock_property {dict unset objs $cc SOURCE_PINS}
    clock_name {dict set objs $cc NAME unexpected}
    core_escape {dict lappend members clk_raw [add WRAPPER/RL/extra_core {CLASS cell}]}
    ddr_escape {dict lappend members mmcm_clkout0 [add WRAPPER/CL/not_ddr {CLASS cell}]}
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
    if {[lindex $args 0] eq "-quiet"} {
        set name [lindex $args 1]
        return [expr {[dict exists $::clocks $name] ? [dict get $::clocks $name] : {}}]
    }
    return [dict get $::clocks [lindex $args 0]]
}
proc get_pins {args} {
    if {"-hierarchical" in $args} {
        if {![dict exists $::pins $::ddr_pin]} {return {}}
        set result [dict get $::pins $::ddr_pin]
        if {$::fault eq "ddr_wildcard"} {lappend result [dict get $::pins WRAPPER/CL/u_clk/O]}
        if {$::fault eq "ddr_wildcard_name"} {set result [add WRAPPER/CL/SH_DDR/other/CLKOUT0 {CLASS pin CLOCK mmcm_clkout0}]}
        return $result
    }
    return [dict get $::pins [lindex $args 0]]
}
proc get_cells {args} {
    if {[lindex $args 0] eq "-quiet"} {
        set name [lindex $args 1]
        return [expr {[dict exists $::cells $name] ? [dict get $::cells $name] : {}}]
    }
    set result {};foreach name [lindex $args 0] {lappend result [dict get $::cells $name]};return $result
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
    set cmds [dict keys [::otpu_scoped::commands $::kind $::t_core]]
    set last [llength $cmds]
    foreach cmd $cmds pos $::positions {
        incr n
        if {$n==$last-1 && $::fault eq "changed_bound"} {set cmd [string map [list $::t_mc 5.000] $cmd]}
        if {$n==$last-1 && $::fault eq "hbm_bound_4000"} {set cmd [string map [list " $::t_mc" " 4.000"] $cmd]}
        if {$n==$last-1 && $::fault eq "bound_rendering"} {set cmd [string map [list " $::t_mc" " [format %.5f $::t_mc]"] $cmd]}
        if {$n==1 && $::fault eq "changed_command"} {append cmd { -allow_paths}}
        if {$n==$last && $::fault eq "missing_export"} {continue}
        if {$n==1 && $::fault eq "trailing_command"} {append cmd {; set_false_path -from DUMMY}}
        puts $f "# id:$pos";puts $f $cmd
        if {$n==1 && $::fault eq "next_command"} {puts $f {set_false_path -from DUMMY}}
        if {$n==1 && $::fault eq "duplicate_export"} {puts $f "# id:900";puts $f $cmd}
        if {$n==1 && $::fault eq "duplicate_id"} {puts $f "# id:$pos";puts $f {set_false_path -from DUMMY}}
    }
    if {$::fault eq "empty_position"} {puts $f {# id:901}}
    if {$::fault eq "incomplete_command"} {puts $f "# id:901";puts $f {set_false_path -to [get_cells}}
    if {$::fault eq "unnumbered_duplicate"} {puts $f {current_instance WRAPPER/CL};puts $f [lindex $cmds 0]}
    if {$::fault eq "hbm_third_scoped"} {puts $f "# id:700";puts $f {set_clock_groups -asynchronous -group [get_clocks clk_main_a0] -group [get_clocks -of_objects [get_pins u_clk/O]]}}
    close $f
}
proc path {src dst sp ep exception requirement} {
    return [add PATH[dict size $::objs] [dict create STARTPOINT_CLOCK $src ENDPOINT_CLOCK $dst STARTPOINT_PIN $sp ENDPOINT_PIN $ep EXCEPTION $exception REQUIREMENT $requirement SLACK 1.000 DATAPATH_DELAY 2.000]]
}
proc resets {ignored} {
    set result {}
    for {set k 0} {$k<3} {incr k} {
        if {$ignored || $::fault eq "reset_fix"} {set ex {False Path};set req {}} \
        else {set ex "MaxDelay Path ${::t_mc}ns -datapath_only";set req $::t_mc}
        set sp {WRAPPER/RL_SHIM/PIPE_RST_OUT_N/pipe_reg[0][0]/C}
        set ep [format {WRAPPER/CL/core_reset_n_reg[%d]/CLR} $k]
        if {$::fault eq "wrong_reset_source"} {set sp WRAPPER/RL/other_reg/C}
        if {$::fault eq "wrong_reset_endpoint"} {set ep WRAPPER/CL/core_reg/D}
        lappend result [path clk_main_a0 clk_raw $sp $ep $ex $req]
    }
    return $result
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
    if {$from eq "@inputs"} {return {}}
    if {$to eq "@outputs"} {
        set src [prop $from NAME]
        if {$src eq "mmcm_clkout0" && !$ignored} {
            set ep RST_DIMM_N;if {$::fault eq "extra_port"} {set ep OTHER_PORT}
            return [path $src {} {WRAPPER/CL/SH_DDR/genblk1.IS_DDR_PRESENT.DDR4_0/inst/u_ddr4_mem_intfc/u_ddr_cal_top/cal_RESET_n_reg[0]/C} $ep {MaxDelay Path 50.000ns} 50.000]
        }
        return {}
    }
    set sk [prop [lindex $from 0] CLASS];set dk [prop [lindex $to 0] CLASS]
    if {$sk eq "cell" || $dk eq "cell"} {
        if {$sk eq "cell" && [prop $to NAME] eq "clk_raw"} {
            if {$ignored && $::fault ni {reset_fix wrong_reset_source wrong_reset_endpoint}} {return {}}
            return [resets $ignored]
        }
        if {$::fault eq "external_ddr" && $sk eq "cell" && [prop $to NAME] eq "mmcm_clkout0"} {return [path clk_main_a0 mmcm_clkout0 SHELL/C WRAPPER/CL/SH_DDR/reg/D {Asynchronous Clock Groups} {}]}
        return {}
    }
    set src [prop $from NAME];set dst [prop $to NAME]
    if {$src eq "clk_raw" || $dst eq "clk_raw"} {
        if {$ignored} {
            if {$::fault eq "query_cap"} {return [lrepeat 10000 [path $src $dst X/C Y/D {False Path} {}]]}
            if {$::fault eq "ignored_data"} {return [path $src $dst WRAPPER/CL/source/C WRAPPER/CL/data/D {False Path} {}]}
            if {$::fault in {reset_fix wrong_reset_source wrong_reset_endpoint} && $src eq "clk_main_a0"} {return [resets 1]}
            return {}
        }
        if {$::fault eq "no_data"} {return {}}
        # the bound in this direction: main -> core the CL's (4.000 DDR, the core period HBM)
        if {$src eq "clk_main_a0"} {set req $::t_mc} else {set req 4.000}
        set ex "MaxDelay Path ${req}ns -datapath_only"
        if {$::fault eq "missing_bound"} {set ex {}}
        if {$::fault eq "wrong_requirement"} {set req 5.000}
        if {$::fault eq "hbm_path_4000" && $src eq "clk_main_a0"} {set ex {MaxDelay Path 4.000ns -datapath_only};set req 4.000}
        if {$::fault eq "wrong_actual_pair"} {set dst mmcm_clkout0}
        return [path $src $dst WRAPPER/CL/source/C WRAPPER/CL/data/D $ex $req]
    }
    if {$::fault eq "no_ddr_witness"} {return {}}
    set ex {Asynchronous Clock Groups};if {$::fault eq "ddr_exception"} {set ex {False Path}}
    return [path $src $dst WRAPPER/CL/SH_DDR/source/C WRAPPER/CL/SH_DDR/dest/D $ex {}]
}
'''


DDR_CASES = [
    ('none', None), ('renumber', None), ('reset_fix', None), ('new_critical', None),
    ('period_rendering', None), ('bound_rendering', None),
    ('caller_subset', 'inventory'), ('caller_duplicate', 'inventory'),
    ('missing_finding', 'exact 3'), ('wrong_check', 'inconsistent'),
    ('missing_check', 'missing CHECK'), ('severity', 'identity/severity'),
    ('fatal', 'identity/severity'), ('class', 'identity/severity'),
    ('waived', 'waiver state'), ('unknown_waiver', 'waiver state'),
    ('missing_waiver', 'missing IS_WAIVED'), ('waiver_reference', 'identity/severity'),
    ('missing_severity', 'missing SEVERITY'), ('missing_description', 'missing DESCRIPTION'),
    ('description', 'description'), ('wrong_position', 'position missing'),
    ('duplicate_position', 'uniquely'), ('period', 'differs from the reviewed'),
    ('master', 'MASTER_CLOCK differs'), ('source', 'SOURCE_PINS differs'),
    ('generated', 'IS_GENERATED differs'), ('missing_clock_property', 'missing SOURCE_PINS'),
    ('clock_name', 'clock identity'), ('core_escape', 'core clock escapes'),
    ('ddr_escape', 'DDR clock escapes'), ('empty_members', 'membership'),
    ('duplicate_members', 'membership'), ('scope', 'top-level'),
    ('bulk_name_missing', 'missing or empty NAME'), ('bulk_name_empty', 'missing or empty NAME'),
    ('bulk_name_truncated', 'NAME result cardinality'),
    ('bulk_class_missing', 'missing or empty CLASS'), ('bulk_class_empty', 'missing or empty CLASS'),
    ('bulk_class_wrong', 'cell class mismatch'), ('bulk_class_truncated', 'CLASS result cardinality'),
    ('version', 'version'), ('clock_pin', 'clock-pin resolution'),
    # (two DDR4 MMCM pins: the CL detection refuses the design before the wildcard check)
    ('ddr_wildcard', 'unreviewed CL'), ('ddr_wildcard_name', 'wildcard clock selection'),
    ('export_scope', 'wrong scope'), ('changed_bound', 'missing or changed'),
    ('changed_command', 'missing or changed'), ('missing_export', 'missing or changed'),
    ('duplicate_export', 'duplicate reviewed'), ('duplicate_id', 'ambiguous export position'),
    ('quiet_scope', 'wrong scope'), ('trailing_command', 'missing or changed'),
    ('next_command', 'ambiguous export command bundle'), ('empty_position', 'without a command'),
    ('incomplete_command', 'incomplete export command'), ('unnumbered_duplicate', 'ambiguous export position'),
    ('wrong_reset_source', 'outside-boundary'), ('wrong_reset_endpoint', 'outside-boundary'),
    ('external_ddr', 'outside-boundary'), ('extra_port', 'top-level I/O'),
    ('query_cap', 'query limit'), ('ignored_data', 'ignored main/core'),
    ('no_data', 'data-path witness'), ('missing_bound', 'datapath bound'),
    ('wrong_requirement', 'datapath bound'), ('wrong_actual_pair', 'clock pair changed'),
    ('no_ddr_witness', 'DDR clock-group witness'), ('ddr_exception', 'exception missing'),
    ('ddr_with_sc', 'unreviewed CL'),
]
# The HBM CL (cl_opentpu_hbm.sv: no DIMM, the SmartConnect fabric): two findings, the main /
# core pair with main -> core at the core period (150 / 175 / 200 MHz), core -> main 4.000;
# a wrong period, a bound not the period, a third scoped constraint, a DDR trace are refused
HBM_CASES = [
    ('none', None, '6.667'), ('none', None, '5.000'), ('none', None, '5.714'),
    ('renumber', None, '6.667'), ('reset_fix', None, '6.667'), ('new_critical', None, '6.667'),
    ('period_rendering', None, '6.667'), ('bound_rendering', None, '6.667'),
    ('none', 'differs from the reviewed', '7.000'), ('none', 'differs from the reviewed', '10.000'),
    ('period', 'differs from the reviewed', '6.667'),
    ('hbm_bound_4000', 'missing or changed', '6.667'), ('hbm_bound_4000', 'missing or changed', '5.000'),
    ('changed_bound', 'missing or changed', '6.667'),
    ('hbm_path_4000', 'datapath bound', '6.667'), ('wrong_requirement', 'datapath bound', '6.667'),
    ('missing_bound', 'datapath bound', '5.000'),
    ('hbm_third_scoped', 'uniquely', '6.667'), ('missing_finding', 'exact 2', '6.667'),
    ('hbm_ddr_finding', 'main/core finding mismatch', '6.667'),
    ('hbm_ddr_present', 'unreviewed CL', '6.667'), ('hbm_ui_clock', 'unreviewed CL', '6.667'),
    ('core_escape', 'core clock escapes', '6.667'), ('no_data', 'data-path witness', '6.667'),
    ('ignored_data', 'ignored main/core', '6.667'), ('missing_export', 'missing or changed', '6.667'),
    ('export_scope', 'wrong scope', '6.667'), ('version', 'version', '6.667'),
]


@pytest.mark.parametrize('kind,fault,expected,t_core',
                         [('ddr', f, e, '10.000') for f, e in DDR_CASES] +
                         [('hbm', f, e, t) for f, e, t in HBM_CASES])
def test_scoped_clock_classifier(tmp_path, kind, fault, expected, t_core):
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl unavailable')
    prefix = tmp_path / 'review'
    fixture = tmp_path / 'actual-schema.xdc'
    fixture.write_text(EXPORT_PREAMBLE)
    nctx = 3 if kind == 'ddr' else 2
    accepted = ' '.join(f'TIMING-54#{i}' for i in range(1, nctx + 1))
    script = (f'source {{{SCRIPT}}}\nset fault {fault}\nset kind {kind}\nset t_core {t_core}\n'
              f'set schema_fixture {{{fixture}}}\n' + MOCKS +
              f'\nset rc [catch {{set result [otpu_classify_scoped_clocks $supplied {{{prefix}}} stdout]}} message]\n' +
              f'set accepted {{{accepted}}}\n' + r'''
if {$rc} {puts stderr $message;exit 1}
if {[dict get $result context] ne $accepted} {puts stderr "wrong accepted set";exit 1}
if {[dict get $result kind] ne $kind || [dict get $result core_period] ne $t_core} {puts stderr "wrong CL / period";exit 1}
set remaining [dict get $result remaining]
if {"TIMING-17#1" ni $remaining} {puts stderr "accepted TIMING17";exit 1}
if {$fault eq "new_critical" && "OTHER#1" ni $remaining} {puts stderr "accepted new finding";exit 1}
foreach name [dict get $result context] {
    set v [dict get $violations $name]
    if {[prop $v SEVERITY] ne "Critical Warning" || [prop $v IS_WAIVED]!=0} {puts stderr "mutated finding";exit 1}
}
puts CLASSIFICATION_CHECK_PASSED
''')
    result = subprocess.run([tclsh], input=script, text=True, capture_output=True, timeout=10)
    if expected:
        assert result.returncode != 0, result.stdout
        assert expected in result.stderr
        assert 'classification reviewed_scoped_clock_context' not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.count('classification reviewed_scoped_clock_context') == nctx
        assert 'CLASSIFICATION_CHECK_PASSED' in result.stdout
        assert f'cl_kind {kind} core_period {t_core}' in result.stdout
        assert (tmp_path / 'review_scoped_clocks.xdc').exists()
        # the witness reports: both directions of every clock pair, timed and ignored
        assert len(list(tmp_path.glob('review_*.rpt'))) == (8 if kind == 'ddr' else 4)


def test_hbm_reviewed_commands_at_150_mhz():
    """The exact reviewed scoped pair of the HBM CL at 150 MHz (and the DDR CL's three)."""
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl unavailable')
    script = (f'source {{{SCRIPT}}}\n'
              'foreach {c r} [::otpu_scoped::commands hbm 6.667] {puts "HBM $r: $c"}\n'
              'foreach {c r} [::otpu_scoped::commands ddr 10.000] {puts "DDR $r: $c"}\n'
              'puts "CANON [::otpu_scoped::canonical {set_max_delay -datapath_only -from [get_clocks clk_main_a0] -to [get_clocks -of_objects [get_pins u_clk/O]] 6.66667}]"\n'
              'puts "KEEP [::otpu_scoped::canonical {set_max_delay -datapath_only -from [get_clocks clk_main_a0] -to [get_clocks -of_objects [get_pins u_clk/O]] 4.000 -allow_paths}]"\n')
    result = subprocess.run([tclsh], input=script, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        'HBM main_to_core: set_max_delay -datapath_only -from [get_clocks clk_main_a0] -to [get_clocks -of_objects [get_pins u_clk/O]] 6.667',
        'HBM core_to_main: set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins u_clk/O]] -to [get_clocks clk_main_a0] 4.000',
        'DDR group: set_clock_groups -asynchronous -group [get_clocks clk_main_a0] -group [get_clocks -of_objects [get_pins -hierarchical -filter NAME=~*SH_DDR/genblk1.IS_DDR_PRESENT.DDR4_0/inst/u_ddr4_infrastructure/gen_mmcme4.u_mmcme_adv_inst/CLKOUT0]]',
        'DDR main_to_core: set_max_delay -datapath_only -from [get_clocks clk_main_a0] -to [get_clocks -of_objects [get_pins u_clk/O]] 4.000',
        'DDR core_to_main: set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins u_clk/O]] -to [get_clocks clk_main_a0] 4.000',
        'CANON set_max_delay -datapath_only -from [get_clocks clk_main_a0] -to [get_clocks -of_objects [get_pins u_clk/O]] 6.667',
        'KEEP set_max_delay -datapath_only -from [get_clocks clk_main_a0] -to [get_clocks -of_objects [get_pins u_clk/O]] 4.000 -allow_paths']


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
    result = subprocess.run([tclsh], input=script, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'SINGLETON_NAME_PASSED' in result.stdout
