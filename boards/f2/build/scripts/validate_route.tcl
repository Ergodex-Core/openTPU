# Called on the shell-linked routed checkpoint. Reports survive failed checks;
# packaging requires the marker written only after every gate passes.
proc otpu_validate_route {scripts_dir reports_dir checkpoints_dir CL TAG shell_dcp} {
set prefix ${reports_dir}/${CL}.${TAG}.route
report_route_status -file ${prefix}_status.rpt
set timing_criticals [get_msg_config -count -severity {CRITICAL WARNING}]
report_timing_summary -delay_type min_max -report_unconstrained \
  -check_timing_verbose -warn_on_violation -file ${prefix}_timing.rpt
set timing_failed [expr {[get_msg_config -count -severity {CRITICAL WARNING}] > $timing_criticals}]
set skew_criticals [get_msg_config -count -severity {CRITICAL WARNING}]
report_bus_skew -warn_on_violation -file ${prefix}_bus_skew.rpt
set skew_failed [expr {[get_msg_config -count -severity {CRITICAL WARNING}] > $skew_criticals}]
report_drc -file ${prefix}_drc.rpt
report_methodology -file ${prefix}_methodology.rpt
source ${scripts_dir}/check_rule_severity.tcl
source ${scripts_dir}/classify_methodology.tcl
set severity_audit [open ${prefix}_rule_severities.rpt w]
set severity_failed [catch {
  set drc_bad {}
  foreach violation [otpu_blocking_violations drc [get_drc_violations] $severity_audit] {
    lappend drc_bad [get_property NAME $violation]
  }
  set methodology [otpu_qualify_methodology $shell_dcp $prefix $severity_audit]
  set methodology_bad [dict get $methodology blockers]
} severity_error]
close $severity_audit
if {$severity_failed} {error $severity_error}
report_cdc -details -file ${prefix}_cdc.rpt
report_exceptions -ignored -file ${prefix}_ignored_exceptions.rpt
report_clock_interaction -file ${prefix}_clock_interaction.rpt
report_utilization -hierarchical -file ${prefix}_utilization.rpt
# logic levels and congestion (reports, not gates: the AWS Shell Interface Specification's hygiene)
report_design_analysis -logic_level_distribution -file ${prefix}_logic_levels.rpt
report_design_analysis -congestion -file ${prefix}_congestion.rpt

set bad_nets [get_nets -hierarchical -filter {
  ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL ||
  ROUTE_STATUS == UNPLACED || ROUTE_STATUS == UNPLACED_ALL || ROUTE_STATUS == CONFLICTS ||
  ROUTE_STATUS == GAPS || ROUTE_STATUS == ANTENNAS || ROUTE_STATUS == NODRIVER ||
  ROUTE_STATUS == MULTI_DRIVER
}]
set setup_bad [get_timing_paths -setup -max_paths 1 -slack_lesser_than 0]
set hold_bad [get_timing_paths -hold -max_paths 1 -slack_lesser_than 0]
set c_main [get_clocks -of_objects [get_pins WRAPPER/CL/clk_main_a0]]
set c_core [get_clocks -of_objects [get_pins WRAPPER/CL/u_clk/O]]
if {[llength $c_main] != 1 || [llength $c_core] != 1} {
  error "openTPU: shell/core clocks missing after shell linking"
}
# The HBM CL (WRAPPER/CL/u_sc, the SmartConnect fabric): cl_timing_user_hbm.tcl bounds each
# direction at the destination's period (core -> main 4.000 ns, main -> core the core period:
# 5.000 at 200 MHz, 6.667 at 150), its own crossings have no gray buses (the SmartConnects carry
# their own constraints) and there is no DIMM firmware; the DDR CL: 4 ns both ways, 14 gray buses
set otpu_hbm [expr {[llength [get_cells -quiet WRAPPER/CL/u_sc]] == 1}]
set t_core [format %.3f [get_property PERIOD $c_core]]
set t_main_core [if {$otpu_hbm} {set t_core} {set _ 4.000}]
foreach {src dst direction want} [list $c_main $c_core main_to_core $t_main_core \
                                      $c_core $c_main core_to_main 4.000] {
  set crossing [report_timing -from $src -to $dst -max_paths 1 -return_string]
  set f [open ${prefix}_${direction}.rpt w]; puts $f $crossing; close $f
  if {![regexp "MaxDelay Path [string map {. \.} $want]ns" $crossing]} {
    error "openTPU: expected $want ns datapath constraint for $direction"
  }
}
# Require all 12 FIFO pointer constraints plus the two write-response counters.
set f [open ${prefix}_bus_skew.rpt r]; set skew_report [read $f]; close $f
set summary [string range $skew_report [string last "1. Bus Skew Report Summary" $skew_report] end]
set end_summary [string first "2. Bus Skew Report Per Constraint" $summary]
if {$end_summary >= 0} {set summary [string range $summary 0 [expr {$end_summary-1}]]}
set own_skew_count 0
foreach line [split $summary \n] {
  if {[regexp {^\s*[0-9]+\s+[0-9]+\s+.*u_sys/g_ch} $line]} {incr own_skew_count}
}
if {$otpu_hbm} {
  set ddr_cal_init none_hbm
} else {
  source ${scripts_dir}/inspect_ddr_init.tcl
}
set summary [open ${prefix}_validation.txt w]
foreach {key value} [list setup_failed [llength $setup_bad] hold_failed [llength $hold_bad] \
    timing_failed $timing_failed bus_skew_failed $skew_failed gray_bus_constraints $own_skew_count \
    unrouted_nets [llength $bad_nets] drc_blockers [llength $drc_bad] \
    methodology_raw_blockers [llength [dict get $methodology raw_blockers]] \
    methodology_inherited_shell [llength [dict get $methodology inherited]] \
    methodology_scoped_clock_context [llength [dict get $methodology context]] \
    methodology_blockers [llength $methodology_bad] ddr_init_2c $ddr_cal_init] {
  puts $summary "$key=$value"
}
close $summary
if {[llength $bad_nets] || [llength $setup_bad] || [llength $hold_bad] || $timing_failed ||
    $skew_failed || $own_skew_count != ($otpu_hbm ? 0 : 14) || [llength $drc_bad] ||
    [llength $methodology_bad]} {
  error "openTPU: routed design failed qualification; see ${prefix}_validation.txt and reports"
}
set passed [open ${checkpoints_dir}/${CL}.${TAG}.route_validation.passed w]
puts $passed "OTPU_STAGE_ROUTE_VALIDATION_PASSED"
close $passed
puts "OTPU_STAGE_ROUTE_VALIDATION_PASSED"
}
otpu_validate_route $scripts_dir $reports_dir $checkpoints_dir $CL $TAG \
  [file join $::env(HDK_SHELL_DIR) build checkpoints from_aws cl_bb_routed.small_shell.dcp]
