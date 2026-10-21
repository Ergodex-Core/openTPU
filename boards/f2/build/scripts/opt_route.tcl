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
# The parent's CL (route_opt.py, from its frozen target): the DDR CL carries the DIMM's
# calibration firmware, its sentinel pinned by the launcher (OTPU_EXPECTED_DDR_INIT); the HBM CL
# (platform mem "hbm": no DIMM, the SmartConnect fabric WRAPPER/CL/u_sc) has none, and the
# launcher passes none_hbm. The checkpoint must be that CL.
set otpu_expect_hbm [expr {$::env(OTPU_EXPECTED_DDR_INIT) eq "none_hbm"}]
set otpu_sc_cells [llength [get_cells -quiet WRAPPER/CL/u_sc]]
if {($otpu_expect_hbm && $otpu_sc_cells != 1) || (!$otpu_expect_hbm && $otpu_sc_cells != 0)} {
  error "openTPU: the checkpoint's CL (SmartConnect fabric cells: $otpu_sc_cells) differs from the parent run's target"
}
if {$otpu_expect_hbm} {
  set ddr_cal_init none_hbm
} else {
  source ${scripts_dir}/inspect_ddr_init.tcl
  if {![string equal -nocase $ddr_cal_init $::env(OTPU_EXPECTED_DDR_INIT)]} {
    error "openTPU: inherited DDR firmware sentinel mismatch"
  }
}
# A legal core MMCM in place of an illegal one (route_opt.py --mmcm DIVCLK MULT): the same VCO,
# so every clock keeps the period the design was placed and routed for; it clears DRC AVAL-168
# (CLKFBOUT_MULT_F 4.8 is not a 0.125 step), which route qualification counts as a blocker
if {[info exists ::env(OTPU_MMCM_MULT)] && $::env(OTPU_MMCM_MULT) ne ""} {
  set otpu_mmcm [get_cells -quiet WRAPPER/CL/u_mmcm]
  if {[llength $otpu_mmcm] != 1} { error "openTPU: no single core MMCM WRAPPER/CL/u_mmcm" }
  set otpu_d0 [get_property DIVCLK_DIVIDE $otpu_mmcm]
  set otpu_m0 [get_property CLKFBOUT_MULT_F $otpu_mmcm]
  set otpu_d1 $::env(OTPU_MMCM_DIVCLK)
  set otpu_m1 $::env(OTPU_MMCM_MULT)
  if {abs(double($otpu_m0) / $otpu_d0 - double($otpu_m1) / $otpu_d1) > 1e-9} {
    error "openTPU: MMCM M/D $otpu_m1/$otpu_d1 changes the VCO (it was $otpu_m0/$otpu_d0)"
  }
  set_property -dict [list DIVCLK_DIVIDE $otpu_d1 CLKFBOUT_MULT_F $otpu_m1] $otpu_mmcm
  puts "OTPU_MMCM_SET DIVCLK_DIVIDE $otpu_d0 -> $otpu_d1 CLKFBOUT_MULT_F $otpu_m0 -> $otpu_m1"
}
report_timing_summary -delay_type min_max -file ${reports_dir}/${CL}.${TAG}.before_route_opt.rpt
# Post-route physical optimization: the directives in order (route_opt.py --phys-opt-directive,
# default Explore alone), each pass from the previous one's result, stopping once setup is met
set otpu_popt {Explore}
if {[info exists ::env(OTPU_PHYS_OPT_DIRECTIVES)] && $::env(OTPU_PHYS_OPT_DIRECTIVES) ne ""} {
  set otpu_popt $::env(OTPU_PHYS_OPT_DIRECTIVES)
}
set otpu_left [llength $otpu_popt]
foreach otpu_d $otpu_popt {
  phys_opt_design -directive $otpu_d
  incr otpu_left -1
  if {$otpu_left > 0} {
    set otpu_wns [get_property SLACK [get_timing_paths -setup -max_paths 1 -nworst 1]]
    puts "OTPU_ROUTE_OPT_PASS $otpu_d WNS=$otpu_wns"
    if {$otpu_wns >= 0} { break }
  }
}

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
if {!$otpu_expect_hbm} {
  source ${scripts_dir}/inspect_ddr_init.tcl
  if {![string equal -nocase $ddr_cal_init $::env(OTPU_EXPECTED_DDR_INIT)]} {
    error "openTPU: optimized DDR firmware sentinel changed"
  }
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
