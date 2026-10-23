# Preserve the pinned HDK implementation sequence, then qualify its checkpoint.
# Unlike standalone synthesis recovery, linking consumes the firmware image
# already stored in post_synth.dcp and never calls refresh_meminit.
set_param general.maxThreads 8
create_project -in_memory -part $DEVICE_TYPE -force
# The HDK reads ${SHELL_MODE}_cl_pnr_user.xdc as a managed XDC (every Tcl command that is not a
# constraint dropped with a CRITICAL WARNING), so openTPU's placement Tcl (cl_pnr_user.tcl: the
# linked reset exception, the HBM floorplan) is read unmanaged at the one point of the HDK's
# sequence after its constraints and before implementation: its opt_design call, wrapped once
# (otpu_clk2x.tcl, the pumped TMEM's multicycles, with it: read after synthesis too, where they
# check the netlist; read again here so they are on the linked design whatever the checkpoint
# carried. docs/v4/f2-timing.md).
rename opt_design otpu_hdk_opt_design
proc opt_design {args} {
  rename opt_design {}
  rename otpu_hdk_opt_design opt_design
  read_xdc -unmanaged ${::constraints_dir}/cl_pnr_user.tcl
  read_xdc -unmanaged ${::constraints_dir}/otpu_clk2x.tcl
  opt_design {*}$args
  # the HDK's HBM workaround (its build_level_1_cl.tcl after opt_design, on cl_mem_perf's
  # instance; Alex's build.tcl on his): the HBM core's DONT_TOUCH off, under our instance name
  if {[info commands get_cells] ne ""} {
    set otpu_hbm_core [get_cells -quiet WRAPPER/CL/u_hbm/HBM_CORE_I/inst]
    if {[llength $otpu_hbm_core]} {
      set_property DONT_TOUCH 0 $otpu_hbm_core
      puts "OTPU_HBM_DONT_TOUCH_CLEARED"
    }
  }
}
# --retime (build.py, OTPU_RETIME=1): phys_opt_design -retime after the HDK's post-place
# phys_opt_design (Alex's 20261015_63_burst_retime tuning, docs/v4/f2-round5.md), once; the
# HDK's call is wrapped the same way (the sequence runs it once, after place_design)
set otpu_retime [expr {[info exists ::env(OTPU_RETIME)] && $::env(OTPU_RETIME) eq "1"}]
if {[info commands phys_opt_design] ne ""} {
  rename phys_opt_design otpu_hdk_phys_opt_design
  proc phys_opt_design {args} {
    otpu_hdk_phys_opt_design {*}$args
    if {$::otpu_retime} {
      puts "OTPU_RETIME phys_opt_design -retime"
      otpu_hdk_phys_opt_design -retime
      set ::otpu_retime 0
    }
  }
}
source ${HDK_SHELL_DIR}/build/scripts/build_level_1_cl.tcl
if {[info commands otpu_hdk_opt_design] ne ""} { error "openTPU: the HDK sequence ran no opt_design" }
set otpu_routed ${checkpoints_dir}/${CL}.${TAG}.post_route.dcp
if {![file exists $otpu_routed]} {
  set otpu_routed ${checkpoints_dir}/${CL}.${TAG}.post_route.VIOLATED.dcp
}
if {![file exists $otpu_routed]} { error "openTPU: implementation produced no routed checkpoint" }
close_project
open_checkpoint $otpu_routed
source ${scripts_dir}/validate_route.tcl
close_design
