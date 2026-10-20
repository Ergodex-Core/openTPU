# Preserve the pinned HDK implementation sequence, then qualify its checkpoint.
# Unlike standalone synthesis recovery, linking consumes the firmware image
# already stored in post_synth.dcp and never calls refresh_meminit.
set_param general.maxThreads 8
create_project -in_memory -part $DEVICE_TYPE -force
# The HDK reads ${SHELL_MODE}_cl_pnr_user.xdc as a managed XDC (every Tcl command that is not a
# constraint dropped with a CRITICAL WARNING), so openTPU's placement Tcl (cl_pnr_user.tcl: the
# linked reset exception, the HBM floorplan) is read unmanaged at the one point of the HDK's
# sequence after its constraints and before implementation: its opt_design call, wrapped once
# (otpu_clk2x.tcl, the pumped TMEM's multicycles, with it).
rename opt_design otpu_hdk_opt_design
proc opt_design {args} {
  rename opt_design {}
  rename otpu_hdk_opt_design opt_design
  read_xdc -unmanaged ${::constraints_dir}/cl_pnr_user.tcl
  # the pumped TMEM's multicycles again, on the linked design (read after synthesis too, where
  # they check the netlist; the ones in the out-of-context checkpoint did not survive the link:
  # round 4 placed the TMEM's read requests at the fast period). The file fails closed.
  read_xdc -unmanaged ${::constraints_dir}/otpu_clk2x.tcl
  opt_design {*}$args
}
# Incremental implementation (build.py --incremental-from; docs/wide_port.md "Incremental
# implementation"): a routed reference checkpoint's placement and routing applied to the cells of
# this design that match it, loaded once, immediately before the HDK's place_design (after its
# post_opt checkpoint and reports). The reference's bytes are pinned like the parent synthesis
# checkpoint's. Without OTPU_INCR_DCP the HDK sequence runs untouched.
set otpu_incremental [expr {[info exists ::env(OTPU_INCR_DCP)] && $::env(OTPU_INCR_DCP) ne ""}]
if {$otpu_incremental} {
  if {[lsearch -exact {RuntimeOptimized TimingClosure Quick} $::env(OTPU_INCR_DIRECTIVE)] < 0} {
    error "openTPU: unknown incremental directive $::env(OTPU_INCR_DIRECTIVE)"
  }
  rename place_design otpu_hdk_place_design
  proc place_design {args} {
    rename place_design {}
    rename otpu_hdk_place_design place_design
    set ref $::env(OTPU_INCR_DCP)
    if {[lindex [exec sha256sum -- $ref] 0] ne $::env(OTPU_INCR_SHA256)} {
      error "openTPU: incremental reference checkpoint hash mismatch"
    }
    set opts [list -incremental -directive $::env(OTPU_INCR_DIRECTIVE)]
    # -force_incr: past Vivado's own reuse / reference-quality criteria (otherwise it may fall
    # back to the default algorithms; the log says which ran)
    if {$::env(OTPU_INCR_FORCE) eq "1"} { lappend opts -force_incr }
    read_checkpoint {*}$opts $ref
    report_incremental_reuse -hierarchical -hierarchical_depth 4 \
      -file ${::reports_dir}/${::CL}.${::TAG}.incremental_reuse.rpt
    puts "OTPU_INCREMENTAL_REFERENCE [file tail $ref] directive=$::env(OTPU_INCR_DIRECTIVE) force=$::env(OTPU_INCR_FORCE)"
    place_design {*}$args
  }
}
source ${HDK_SHELL_DIR}/build/scripts/build_level_1_cl.tcl
if {[info commands otpu_hdk_opt_design] ne ""} { error "openTPU: the HDK sequence ran no opt_design" }
if {$otpu_incremental && [info commands otpu_hdk_place_design] ne ""} {
  error "openTPU: the HDK sequence ran no place_design (the incremental reference was never loaded)"
}
set otpu_routed ${checkpoints_dir}/${CL}.${TAG}.post_route.dcp
if {![file exists $otpu_routed]} {
  set otpu_routed ${checkpoints_dir}/${CL}.${TAG}.post_route.VIOLATED.dcp
}
if {![file exists $otpu_routed]} { error "openTPU: implementation produced no routed checkpoint" }
close_project
open_checkpoint $otpu_routed
source ${scripts_dir}/validate_route.tcl
close_design
