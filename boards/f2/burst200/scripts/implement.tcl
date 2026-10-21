# Resume implementation from a saved, real-shell-linked checkpoint.
set run [file normalize [lindex $argv 0]]
set checkpoint [file normalize [lindex $argv 1]]
set hdk [file normalize [lindex $argv 2]]
set shell $hdk/common/shell_stable
cd $run
set_param general.maxThreads 8
set_param sta.enableAutoGenClkNamePersistence 0
# The directives: the flow's own unless build.py's strategy.tcl names others (--place-directive,
# --phys-directive, --route-directive; --incremental-directive for read_checkpoint -incremental).
set place_directive Explore
set phys_directive AggressiveExplore
set route_directive Explore
set incremental_directive RuntimeOptimized
set place_net_delay_weight {}
if {[file exists $run/strategy.tcl]} {source $run/strategy.tcl}
# Optional later-stage file proves that the source checkpoint has finished writing.
if {[llength $argv] > 3} {
  set ready [file normalize [lindex $argv 3]]
  set attempts 0
  while {![file exists $ready]} {
    if {$attempts >= 720} {error "Source build did not reach readiness marker: $ready"}
    puts "WAITING_FOR_SOURCE $ready"
    after 10000
    incr attempts
  }
}
set provenance [open $run/source_checkpoint.sha256 w]
puts $provenance [exec sha256sum $checkpoint]
close $provenance
open_checkpoint $checkpoint
if {![llength [get_pblocks -quiet pblock_CL]]} {
  read_xdc $shell/build/constraints/small_shell_level_1_fp_cl.xdc
} else {
  puts "Retaining legal pblock_CL from the linked AWS shell checkpoint"
}
source $run/floorplan.tcl
source $run/core_timing.tcl
source $run/platform_constraints.tcl
if {[file exists $run/tuning.tcl]} {source $run/tuning.tcl}
opt_design
# Same AMD HBM workaround as the official HDK, using our instance name.
set hbm [get_cells -quiet WRAPPER/CL/u_hbm/HBM_CORE_I/inst]
if {[llength $hbm]} {set_property DONT_TOUCH false $hbm}
write_checkpoint -force $run/post_opt.dcp
report_utilization -hierarchical -file $run/opt_utilization.rpt
report_utilization -pblocks [get_pblocks pblock_CL_SLR0] -file $run/slr0_utilization.rpt
report_drc -file $run/opt_drc.rpt
set incremental [file exists $run/incremental_reference.txt]
if {$incremental} {
  set fp [open $run/incremental_reference.txt r]; set ref [string trim [read $fp]]; close $fp
  set fp [open $run/incremental_reference.sha256 r]; set expected [string trim [read $fp]]; close $fp
  set actual [lindex [exec sha256sum $ref] 0]
  if {$actual ne $expected} {error "Incremental reference checksum mismatch"}
  # RuntimeOptimized (Vivado's default) targets the reference's WNS and favours reuse of the
  # timing-closed reference; TimingClosure targets WNS 0 and rips up the failing paths; Quick is
  # not timing driven. No clock, shell, pblock or timing-exception changes are introduced. The
  # place, phys_opt and route steps keep their default directives (route_design allows only
  # Explore, Quick and Default with incremental reuse).
  puts "IMPLEMENTATION_STRATEGY incremental from $ref directive=$incremental_directive"
  read_checkpoint -incremental $ref -directive $incremental_directive
  report_incremental_reuse -file $run/incremental_read_reuse.rpt
  place_design -no_bufg_opt
} else {
  puts "IMPLEMENTATION_STRATEGY place=$place_directive phys_opt=$phys_directive route=$route_directive net_delay_weight=$place_net_delay_weight"
  if {$place_net_delay_weight ne ""} {
    place_design -directive $place_directive -no_bufg_opt -net_delay_weight $place_net_delay_weight
  } else {
    place_design -directive $place_directive -no_bufg_opt
  }
}
write_checkpoint -force $run/post_place.dcp
report_timing_summary -report_unconstrained -file $run/place_timing.rpt
report_design_analysis -congestion -file $run/place_congestion.rpt
if {$incremental} {
  phys_opt_design
  report_incremental_reuse -file $run/incremental_place_reuse.rpt
} else {
  phys_opt_design -directive $phys_directive
}
write_checkpoint -force $run/post_phys_opt.dcp
if {[llength $argv] > 4 && [lindex $argv 4] eq "phys_opt_only"} {
  set complete [open $run/phys_opt_done.txt w]; puts $complete PHYS_OPT_COMPLETE; close $complete
  report_timing_summary -delay_type min_max -report_unconstrained -file $run/phys_opt_timing.rpt
  puts "PHYS_OPT_COMPLETE_ROUTING_NOT_RUN"
  close_project
  exit
}
# E31: skip only the 2025.2 leaf-clock optimizer that segfaulted with this shell.
puts "Leaf clock optimization default=[get_param route.optimizeLeafClkProgDlys]"
set_param route.optimizeLeafClkProgDlys false
set_param general.maxThreads 6
if {$incremental} {
  route_design -tns_cleanup
  report_incremental_reuse -file $run/incremental_route_reuse.rpt
} else {
  route_design -directive $route_directive -tns_cleanup
}
# Restore the reference shell value after the placer's bank-74 Vref change.
set_property INTERNAL_VREF 0.84 [get_iobanks 74]
write_checkpoint -force $run/post_route.dcp
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file $run/route_timing.rpt
report_timing -max_paths 50 -path_type full_clock_expanded -file $run/worst_paths.rpt
report_route_status -file $run/route_status.rpt
report_drc -file $run/route_drc.rpt
report_cdc -details -file $run/cdc.rpt
report_bus_skew -file $run/bus_skew.rpt
report_clock_interaction -file $run/clock_interaction.rpt
report_utilization -hierarchical -file $run/route_utilization.rpt
report_design_analysis -congestion -file $run/route_congestion.rpt
set setup [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
set fp [open $run/implementation_result.txt w]
puts $fp "setup_slack_ns=$setup\nhold_slack_ns=$hold\nReview route/DRC/CDC/unconstrained reports before deployment."
close $fp
if {$setup < 0 || $hold < 0} {puts "CANDIDATE_NEEDS_REFINEMENT_NOT_DEPLOYMENT_READY"}
puts "IMPLEMENTATION_CANDIDATE_SAVED_REFINEMENT_REQUIRED"
