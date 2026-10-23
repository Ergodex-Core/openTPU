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
foreach {src dst direction} [list $c_main $c_core main_to_core $c_core $c_main core_to_main] {
  set crossing [report_timing -from $src -to $dst -max_paths 1 -return_string]
  set f [open ${prefix}_${direction}.rpt w]; puts $f $crossing; close $f
  if {![regexp {MaxDelay Path 4\.000ns} $crossing]} {
    error "openTPU: expected 4 ns datapath constraint for $direction"
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
source ${scripts_dir}/inspect_ddr_init.tcl
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
    $skew_failed || $own_skew_count != 14 || [llength $drc_bad] || [llength $methodology_bad]} {
  error "openTPU: routed design failed qualification; see ${prefix}_validation.txt and reports"
}
set passed [open ${checkpoints_dir}/${CL}.${TAG}.route_validation.passed w]
puts $passed "OTPU_STAGE_ROUTE_VALIDATION_PASSED"
close $passed
puts "OTPU_STAGE_ROUTE_VALIDATION_PASSED"
}
otpu_validate_route $scripts_dir $reports_dir $checkpoints_dir $CL $TAG \
  [file join $::env(HDK_SHELL_DIR) build checkpoints from_aws cl_bb_routed.small_shell.dcp]
