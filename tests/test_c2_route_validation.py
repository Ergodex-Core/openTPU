"""Fail-closed route qualifier control flow with mocked Vivado object/report APIs.

These tests do not stand in for a routed Vivado report, timing closure, or CDC proof.
The frozen methodology classifier is substituted only in this isolated Tcl fixture.
"""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "boards/c2_f2/validate_hbm_route.tcl"
SUPPORT = ROOT / "boards/c2_f2/support/boards/f2/build/scripts"

# Vivado 2025.2 post-opt report structure: Id/Position, wrapped source and sink
# lists, then a separate numeric line. Cell lists shortened from the captured
# 4f750912 report; this is a parser fixture, not timing or coverage evidence.
SKEW_ROW = r'''@ID@   @POSITION@        [get_cells [list {@SCOPE@/wgray_reg[0]} \
          {@SCOPE@/wgray_reg[1]} \
          {@SCOPE@/wbin_reg[0]} \
          {@SCOPE@/wbin_reg[1]}]]
                                              [get_cells [list {@SCOPE@/wgray_r1_reg[0]} \
          {@SCOPE@/wgray_r1_reg[1]}]]
                                                                              Fast              2.222      -0.294      2.516'''

MOCK = r'''
lassign $argv validator root scenario
set scripts_dir $root/scripts
set reports_dir $root/reports
set checkpoints_dir $root/checkpoints
set CL cl_opentpu
set TAG deadbeef
set ::env(HDK_SHELL_DIR) $root/shell
file mkdir $scripts_dir $reports_dir $checkpoints_dir
set criticals 0
proc report_route_status {args} {}
proc report_timing_summary {args} {
  if {$::scenario eq "timing_warning"} { incr ::criticals }
}
proc report_drc {args} {}
proc report_methodology {args} {}
proc report_cdc {args} {}
proc report_exceptions {args} {}
proc report_clock_interaction {args} {}
proc report_utilization {args} {}
proc get_msg_config {args} { return $::criticals }
proc report_timing {args} {
  set f [open [lindex $args [expr {[lsearch -exact $args -file]+1}]] w]
  puts $f "mock path witness; object checks run separately"
  close $f
}
proc get_pins {args} {
  set pin [lindex $args end]
  if {$::scenario eq "missing_clock_pin" && [string match *u_hclk* $pin]} { return {} }
  return $pin
}
proc get_clocks {args} {
  switch -glob [lindex $args end] {
    *clk_main_a0 {return main}
    *u_clk/O {return core}
    *u_hclk/O {return hbm}
    default {error "unexpected clock query"}
  }
}
proc list_property {object} {
  return {NAME PERIOD EXCEPTION REQUIREMENT STARTPOINT_CLOCK ENDPOINT_CLOCK SEVERITY}
}
proc get_property {key object} {
  if {$object eq "drc0" && $key eq "SEVERITY"} {
    if {$::scenario eq "unknown_severity"} {return MYSTERY}
    return ERROR
  }
  if {$key eq "NAME"} {return $object}
  if {$key eq "PERIOD"} {
    if {$::scenario eq "wrong_clock" && $object eq "core"} {return 10.0}
    return [dict get {main 4.0 core 5.0 hbm 2.222222222222} $object]
  }
  lassign [split $object :] unused src dst
  set period [get_property PERIOD $dst]
  switch -- $key {
    EXCEPTION {
      if {$::scenario eq "wrong_exception" && $src eq "hbm" && $dst eq "core"} {
        return {Asynchronous Clock Groups}
      }
      if {$::scenario eq "wrong_bound" && $src eq "hbm" && $dst eq "core"} {
        return {MaxDelay Path 2.222ns -datapath_only}
      }
      if {$::scenario eq "missing_datapath_flag"} {return [format "MaxDelay Path %.3fns" $period]}
      if {$::scenario eq "unknown_exception_flag"} {return [format "MaxDelay Path %.3fns -datapath_only -other" $period]}
      return [format "MaxDelay Path %.3fns -datapath_only" $period]
    }
    REQUIREMENT {
      if {$::scenario eq "wrong_requirement" && $src eq "main" && $dst eq "hbm"} {return 4.0}
      return $period
    }
    STARTPOINT_CLOCK {
      if {$::scenario eq "wrong_witness_clock"} {return another_clock}
      return $src
    }
    ENDPOINT_CLOCK {return $dst}
    default {error "unexpected property $key on $object"}
  }
}
proc get_timing_paths {args} {
  if {[lsearch -exact $args -setup]>=0} {
    return [expr {$::scenario eq "setup_violation" ? "bad_setup" : ""}]
  }
  if {[lsearch -exact $args -hold]>=0} {
    return [expr {$::scenario eq "hold_violation" ? "bad_hold" : ""}]
  }
  set src [lindex $args [expr {[lsearch -exact $args -from]+1}]]
  set dst [lindex $args [expr {[lsearch -exact $args -to]+1}]]
  if {$::scenario eq "missing_crossing" && $src eq "core" && $dst eq "hbm"} {return {}}
  return path:$src:$dst
}
proc get_nets {args} {return [expr {$::scenario eq "unrouted" ? "bad_net" : ""}]}
proc get_drc_violations {} {
  return [expr {$::scenario in {drc_violation unknown_severity} ? "drc0" : ""}]
}
proc report_bus_skew {args} {
  if {$::scenario eq "skew_warning"} {incr ::criticals}
  set f [open $::root/skew_row.txt r]; set row_template [read $f]; close $f
  set f [open [lindex $args [expr {[lsearch -exact $args -file]+1}]] w]
  puts $f "Table of Contents\n-----------------"
  puts $f "1. Bus Skew Report Summary\n2. Bus Skew Report Per Constraint\n"
  puts $f "1. Bus Skew Report Summary"
  if {$::scenario ne "missing_section_underline"} {puts $f "--------------------------"}
  puts $f "Id Position From To Corner Requirement(ns) Actual(ns) Slack(ns)"
  # Shell constraints do not contribute to the owned HBM fabric inventory.
  puts $f [string map [list @ID@ 0 @POSITION@ 77 @SCOPE@ WRAPPER/SHELL/gray] $row_template]
  set count [expr {$::scenario eq "missing_gray" ? 46 : 47}]
  if {$::scenario eq "old_host_only_gray"} {set count 39}
  for {set i 1} {$i<=$count} {incr i} {
    set id [expr {$::scenario eq "duplicate_gray" && $i==47 ? 46 : $i}]
    if {$::scenario eq "malformed_gray" && $i==47} {set id wrapped}
    set row [string map [list @ID@ $id @POSITION@ [expr {77+$i}] @SCOPE@ WRAPPER/CL/u_sys/u_mem/fifo${i}] $row_template]
    if {$::scenario eq "incomplete_gray" && $i==47} {set row [join [lrange [split $row \n] 0 end-1] \n]}
    puts $f $row
  }
  if {$::scenario eq "extra_gray"} {
    puts $f [string map [list @ID@ 48 @POSITION@ 125 @SCOPE@ WRAPPER/CL/u_sys/u_mem/extra] $row_template]
  }
  puts $f "2. Bus Skew Report Per Constraint"
  puts $f "-------------------------------"
  # Detail may repeat paths or be limited, and must not affect the count.
  puts $f "1 WRAPPER/CL/u_sys/u_mem/fifo1/wgray_reg duplicate_detail"
  close $f
}
if {$scenario eq "existing_marker"} {
  close [open $checkpoints_dir/${CL}.${TAG}.route_validation.passed w]
}
if {[catch {source $validator} error]} {
  puts stderr $error
  exit 1
}
'''


def run_mock(tmp_path, scenario):
    tclsh = shutil.which("tclsh")
    if not tclsh:
        pytest.skip("tclsh required")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(SUPPORT / "check_rule_severity.tcl", scripts)
    (scripts / "classify_methodology.tcl").write_text(r'''
proc otpu_qualify_methodology {shell prefix audit} {
  if {$::scenario eq "methodology_schema_error"} {error "unreviewed scoped-clock findings"}
  set bad [expr {$::scenario eq "methodology_violation" ? "bad_methodology" : ""}]
  return [dict create blockers $bad raw_blockers $bad inherited {} context {}]
}
''')
    driver = tmp_path / "mock.tcl"
    driver.write_text(MOCK)
    (tmp_path / "skew_row.txt").write_text(SKEW_ROW)
    return subprocess.run([tclsh, str(driver), str(VALIDATOR), str(tmp_path), scenario],
                          text=True, capture_output=True)


def test_hbm_route_checks_six_crossings_and_all_gray_constraints(tmp_path):
    result = run_mock(tmp_path, "pass")
    assert result.returncode == 0, result.stderr
    summary = (tmp_path / "reports/cl_opentpu.deadbeef.route_validation.txt").read_text()
    assert "gray_bus_constraints=47" in summary
    assert "cdc_crossing_witnesses=6" in summary
    marker = tmp_path / "checkpoints/cl_opentpu.deadbeef.route_validation.passed"
    assert marker.read_text().strip() == "OTPU_STAGE_ROUTE_VALIDATION_PASSED"


@pytest.mark.parametrize("scenario", [
    "missing_clock_pin", "wrong_clock", "missing_crossing", "wrong_exception",
    "wrong_bound", "wrong_requirement", "wrong_witness_clock", "missing_gray",
    "missing_datapath_flag", "unknown_exception_flag", "old_host_only_gray",
    "missing_section_underline", "incomplete_gray",
    "duplicate_gray", "malformed_gray", "extra_gray", "skew_warning", "timing_warning",
    "setup_violation", "hold_violation", "unrouted", "drc_violation",
    "unknown_severity", "methodology_violation", "methodology_schema_error",
])
def test_hbm_route_rejects_incomplete_or_failing_evidence(tmp_path, scenario):
    result = run_mock(tmp_path, scenario)
    assert result.returncode != 0, scenario
    assert result.stderr.strip(), scenario
    assert not (tmp_path / "checkpoints/cl_opentpu.deadbeef.route_validation.passed").exists()


def test_hbm_route_refuses_stale_pass_marker(tmp_path):
    result = run_mock(tmp_path, "existing_marker")
    assert result.returncode != 0
    assert "refusing to reuse" in result.stderr
