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
# The HBM CLs: the SmartConnect CL (WRAPPER/CL/u_sc, the fabric; cl_timing_user_hbm.tcl) and
# the engine CL (WRAPPER/CL/u_sys/u_mem/g_port[p].u_eng, an otpu_hbm_ch a port;
# cl_timing_user_hbm5.tcl) bound each direction at the destination's period (core -> main
# 4.000 ns, main -> core the core period: 5.000 at 200 MHz, 6.667 at 150; the engine CL core
# <-> hclk too: 2.222 and the core period), and there is no DIMM firmware; the DDR CL: 4 ns
# both ways. The gray buses the CL's own constraints name: the DDR CL's 14 (u_sys/g_ch: 12
# FIFO pointers, two write-response counters), the engine CL's eight an engine
# (u_sys/u_mem/g_port: two pointers x four FIFOs), the SmartConnect CL's none (the
# SmartConnects carry their own)
set otpu_sc [expr {[llength [get_cells -quiet WRAPPER/CL/u_sc]] == 1}]
set otpu_engines [get_cells -quiet -hier -filter {NAME =~ WRAPPER/CL/u_sys/u_mem/g_port*.u_eng}]
set otpu_hbm [expr {$otpu_sc || [llength $otpu_engines] > 0}]
set t_core [format %.3f [get_property PERIOD $c_core]]
set t_main_core [if {$otpu_hbm} {set t_core} {set _ 4.000}]
set crossings [list $c_main $c_core main_to_core $t_main_core $c_core $c_main core_to_main 4.000]
if {[llength $otpu_engines]} {
  set c_hbm [get_clocks -of_objects [get_pins WRAPPER/CL/u_hclk/O]]
  if {[llength $c_hbm] != 1} { error "openTPU: HBM AXI clock missing after shell linking" }
  set t_hbm [format %.3f [get_property PERIOD $c_hbm]]
  lappend crossings $c_core $c_hbm core_to_hbm $t_hbm $c_hbm $c_core hbm_to_core $t_core
}
foreach {src dst direction want} $crossings {
  set crossing [report_timing -from $src -to $dst -max_paths 1 -return_string]
  set f [open ${prefix}_${direction}.rpt w]; puts $f $crossing; close $f
  if {![regexp "MaxDelay Path [string map {. \.} $want]ns" $crossing]} {
    error "openTPU: expected $want ns datapath constraint for $direction"
  }
}
# Require every gray-bus constraint of the CL's own.
set own_pattern [expr {[llength $otpu_engines] ? "u_sys/u_mem/g_port" : "u_sys/g_ch"}]
set own_want [expr {[llength $otpu_engines] ? 8 * [llength $otpu_engines] : ($otpu_sc ? 0 : 14)}]
set f [open ${prefix}_bus_skew.rpt r]; set skew_report [read $f]; close $f
set summary [string range $skew_report [string last "1. Bus Skew Report Summary" $skew_report] end]
set end_summary [string first "2. Bus Skew Report Per Constraint" $summary]
if {$end_summary >= 0} {set summary [string range $summary 0 [expr {$end_summary-1}]]}
set own_skew_count 0
foreach line [split $summary \n] {
  if {[regexp "^\\s*\\d+\\s+\\d+\\s+.*$own_pattern" $line]} {incr own_skew_count}
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
    $skew_failed || $own_skew_count != $own_want || [llength $drc_bad] ||
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
