# Reopen an immutable routed parent; preserve its shell locks and all constraints.
# Do not link again, reread XDC, refresh firmware, or weaken a timing exception.
set_param general.maxThreads 8
set CL cl_opentpu
set TAG $::env(BUILD_TAG)
set scripts_dir [file dirname [file normalize [info script]]]
set build_dir [file dirname $scripts_dir]
set reports_dir [file join $build_dir reports]
set checkpoints_dir [file join $build_dir checkpoints]
set parent_dcp $::env(OTPU_PARENT_DCP)
if {[lindex [exec sha256sum -- $parent_dcp] 0] ne $::env(OTPU_PARENT_SHA256)} {
  error "openTPU: inherited routed DCP hash mismatch"
}
open_checkpoint $parent_dcp
source ${scripts_dir}/inspect_ddr_init.tcl
if {![string equal -nocase $ddr_cal_init $::env(OTPU_EXPECTED_DDR_INIT)]} {
  error "openTPU: inherited DDR firmware sentinel mismatch"
}
report_timing_summary -delay_type min_max -file ${reports_dir}/${CL}.${TAG}.before_route_opt.rpt
phys_opt_design -directive Explore

# Physical optimization can alter routes. Complete any resulting unfinished or
# erroneous routes with the supported router, then requalify all timing paths.
set incomplete [get_nets -hierarchical -filter {
  ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL || ROUTE_STATUS == UNPLACED ||
  ROUTE_STATUS == UNPLACED_ALL || ROUTE_STATUS == CONFLICTS || ROUTE_STATUS == GAPS ||
  ROUTE_STATUS == ANTENNAS || ROUTE_STATUS == NODRIVER || ROUTE_STATUS == MULTI_DRIVER
}]
if {[llength $incomplete]} {
  puts "OTPU_ROUTE_OPT_REROUTE_NETS=[llength $incomplete]"
  route_design -directive Explore -tns_cleanup
}
set candidate ${checkpoints_dir}/${CL}.${TAG}.route_opt.unvalidated.dcp
write_checkpoint $candidate
close_design
open_checkpoint $candidate
source ${scripts_dir}/inspect_ddr_init.tcl
if {![string equal -nocase $ddr_cal_init $::env(OTPU_EXPECTED_DDR_INIT)]} {
  error "openTPU: optimized DDR firmware sentinel changed"
}
source ${scripts_dir}/validate_route.tcl
report_timing -delay_type min_max -max_paths 10 -nworst 1 -sort_by group \
  -path_type full_clock_expanded -input_pins \
  -file ${reports_dir}/${CL}.${TAG}.post_route_timing.rpt
write_debug_probes -no_partial_ltxfile ${checkpoints_dir}/${TAG}.debug_probes.ltx
# Copy only the persisted checkpoint that was reopened and fully qualified.
file copy $candidate ${checkpoints_dir}/${CL}.${TAG}.post_route.dcp
close_design
if {[lindex [exec sha256sum -- $parent_dcp] 0] ne $::env(OTPU_PARENT_SHA256)} {
  error "openTPU: inherited routed DCP changed during recovery"
}
puts "OTPU_ROUTE_OPT_COMPLETED"
