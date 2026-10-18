# Preserve the pinned HDK implementation sequence, then qualify its checkpoint.
# Unlike standalone synthesis recovery, linking consumes the firmware image
# already stored in post_synth.dcp and never calls refresh_meminit.
set_param general.maxThreads 8
create_project -in_memory -part $DEVICE_TYPE -force
source ${HDK_SHELL_DIR}/build/scripts/build_level_1_cl.tcl
set otpu_routed ${checkpoints_dir}/${CL}.${TAG}.post_route.dcp
if {![file exists $otpu_routed]} {
  set otpu_routed ${checkpoints_dir}/${CL}.${TAG}.post_route.VIOLATED.dcp
}
if {![file exists $otpu_routed]} { error "openTPU: implementation produced no routed checkpoint" }
close_project
open_checkpoint $otpu_routed
source ${scripts_dir}/validate_route.tcl
close_design
