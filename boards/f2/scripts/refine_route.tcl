# Background post-route timing refinement. Preserve every candidate and all gates.
set run [file normalize [lindex $argv 0]]
set checkpoint [file normalize [lindex $argv 1]]
cd $run
set_param general.maxThreads 6
set_param route.optimizeLeafClkProgDlys false
set fp [open $run/source_checkpoint.sha256 w]
puts $fp [exec sha256sum $checkpoint]
close $fp
open_checkpoint $checkpoint
help phys_opt_design
help route_design
set passed 0
foreach directive {Explore AggressiveExplore} {
  puts "POST_ROUTE_REFINEMENT $directive"
  phys_opt_design -directive $directive
  route_design -directive Explore -tns_cleanup
  set_property INTERNAL_VREF 0.84 [get_iobanks 74]
  write_checkpoint -force $run/post_${directive}.dcp
  report_timing_summary -delay_type min_max -report_unconstrained -file $run/timing_${directive}.rpt
  set setup [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
  set hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
  set fp [open $run/progress.txt a]
  puts $fp "$directive setup=$setup hold=$hold"
  close $fp
  if {$setup >= 0 && $hold >= 0} {set passed 1; break}
}
write_checkpoint -force $run/post_route.dcp
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file $run/route_timing.rpt
report_timing -delay_type max -max_paths 50 -path_type full_clock_expanded -file $run/worst_paths.rpt
report_timing -delay_type min -max_paths 25 -path_type full_clock_expanded -file $run/hold_paths.rpt
report_route_status -file $run/route_status.rpt
report_drc -file $run/route_drc.rpt
report_cdc -details -file $run/cdc.rpt
report_bus_skew -file $run/bus_skew.rpt
report_clock_interaction -file $run/clock_interaction.rpt
report_methodology -file $run/methodology.rpt
report_design_analysis -congestion -file $run/route_congestion.rpt
set fp [open $run/implementation_result.txt w]
puts $fp "setup_slack_ns=$setup\nhold_slack_ns=$hold\nIndependent review required before deployment."
close $fp
if {!$passed} {error "Post-route refinement did not close timing"}
puts ROUTED_TIMING_PASS_PENDING_REPORT_REVIEW
