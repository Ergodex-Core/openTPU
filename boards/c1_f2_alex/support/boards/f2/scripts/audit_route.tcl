# Independent final report pass; never changes or programs the device.
set run [file normalize [lindex $argv 0]]
set checkpoint [file normalize [lindex $argv 1]]
cd $run
set_param general.maxThreads 4
open_checkpoint $checkpoint
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file $run/timing.rpt
report_timing -delay_type max -max_paths 50 -path_type full_clock_expanded -file $run/setup_paths.rpt
report_timing -delay_type min -max_paths 25 -path_type full_clock_expanded -file $run/hold_paths.rpt
# Keep accelerator paths visible even when a separate interface path dominates
# the global worst-path list. Both directions of the related 1x/2x clocks matter.
foreach {source_label source_pin} {core clk_out1 tmem clk_out2} {
  set source_clock [get_clocks -of_objects [get_pins WRAPPER/CL/u_core_clk/$source_pin]]
  if {[llength $source_clock] != 1} {error "Expected one $source_label clock"}
  foreach {dest_label dest_pin} {core clk_out1 tmem clk_out2} {
    set dest_clock [get_clocks -of_objects [get_pins WRAPPER/CL/u_core_clk/$dest_pin]]
    if {[llength $dest_clock] != 1} {error "Expected one $dest_label clock"}
    foreach delay {max min} {
      report_timing -from $source_clock -to $dest_clock -delay_type $delay \
        -max_paths 25 -path_type full_clock_expanded \
        -file $run/${source_label}_to_${dest_label}_${delay}.rpt
    }
  }
}
if {[llength $argv] > 2} {
  # Optional router table: preserve both setup and hold paths to its tight pins.
  set source [open [file normalize [lindex $argv 2]] r]
  set tight_names {}
  foreach line [split [read $source] \n] {
    if {[regexp {(WRAPPER/CL/[^ |]+)} $line -> pin]} {lappend tight_names $pin}
  }
  close $source
  set tight [get_pins -quiet $tight_names]
  if {![llength $tight]} {error "Router tight-pin report matched no implemented pins"}
  report_timing -to $tight -delay_type max -max_paths 80 -nworst 4 -path_type full_clock_expanded -file $run/tight_setup_paths.rpt
  report_timing -to $tight -delay_type min -max_paths 80 -nworst 4 -path_type full_clock_expanded -file $run/tight_hold_paths.rpt
}
report_route_status -file $run/route_status.rpt
report_drc -file $run/drc.rpt
report_cdc -details -show_waiver -all_checks_per_endpoint -file $run/cdc.rpt
report_bus_skew -file $run/bus_skew.rpt
report_clock_interaction -file $run/clock_interaction.rpt
report_exceptions -coverage -file $run/exceptions.rpt
report_methodology -file $run/methodology.rpt
report_waivers -file $run/waivers.rpt
write_debug_probes -no_partial_ltxfile -force $run/debug_probes.ltx
set fp [open $run/audit.txt w]
puts $fp "checkpoint=$checkpoint"
foreach delay {max min} {
  set paths [get_timing_paths -delay_type $delay -max_paths 1]
  if {![llength $paths]} {error "No $delay timing paths in routed design"}
  puts $fp "${delay}_slack_ns=[get_property SLACK $paths]"
}
foreach severity {Error {Critical Warning} Warning} {
  set count 0
  foreach violation [get_drc_violations -quiet] {
    if {[get_property SEVERITY $violation] eq $severity} {incr count}
  }
  puts $fp "drc_${severity}=$count"
}
close $fp
puts ROUTE_AUDIT_COMPLETE_REVIEW_REPORTS
