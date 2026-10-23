# openTPU's placement-time constraints, read by build_level_1_cl.tcl with read_xdc -unmanaged
# right before the HDK's opt_design (after the HDK's small_shell_level_1_fp_cl.xdc and the
# managed small_shell_cl_pnr_user.xdc). Unmanaged because this is Tcl: a managed XDC drops
# proc / if / foreach / puts with a CRITICAL WARNING and keeps going, which is how the first
# three placed f2-hbm builds ran with the exception below applied to no pins and (round 3)
# pblocks with no cells. For an HBM target build.py appends cl_pnr_user_hbm.tcl, the floorplan.

# A top-level reset port becomes a hierarchical pin when the CL DCP is linked.
# Its synthesis-only -from exception is therefore rejected by link_design.
# Restore the asynchronous reset-entry exception using that boundary as a
# through-point, limited to the three three-stage reset synchronizers. Recovery
# and removal at their CLR inputs are exempted because release is synchronized.
# The synchronizers' Q -> D paths and reset-release fanout remain timed.
set otpu_reset_warning_count [get_msg_config -count -severity WARNING]
set otpu_reset_critical_count [get_msg_config -count -severity {CRITICAL WARNING}]
set otpu_shell_reset [get_pins -quiet WRAPPER/CL/rst_main_n]
if {[llength $otpu_shell_reset] != 1} {
  error "openTPU: expected one shell reset boundary pin"
}
set otpu_reset_clrs {}
foreach stem {main_reset_n bridge_reset_n core_reset_n} {
  set regs [get_cells -quiet -filter {IS_SEQUENTIAL} WRAPPER/CL/${stem}_reg*]
  set pins [get_pins -quiet -of_objects $regs -filter {REF_PIN_NAME == CLR}]
  if {[llength $regs] != 3 || [llength $pins] != 3} {
    error "openTPU: expected three asynchronous reset pins for $stem"
  }
  set otpu_reset_clrs [concat $otpu_reset_clrs $pins]
}
set_false_path -through $otpu_shell_reset -to $otpu_reset_clrs
if {[get_msg_config -count -severity WARNING] > $otpu_reset_warning_count ||
    [get_msg_config -count -severity {CRITICAL WARNING}] > $otpu_reset_critical_count} {
  error "openTPU: shell-linked reset exception produced a warning"
}
puts "OTPU_LINKED_RESET_EXCEPTION_PASSED endpoints=[llength $otpu_reset_clrs]"
