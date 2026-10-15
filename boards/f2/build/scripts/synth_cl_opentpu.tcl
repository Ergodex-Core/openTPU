# Called by the pinned HDK build_all.tcl. Sources and target parameters are frozen
# by boards/f2/build.py in a fresh CL directory before Vivado starts.
set_param general.maxThreads 8
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_header.tcl
set_property verilog_define {SYNTHESIS XSDB_SLV_DIS} [current_fileset]
set_property include_dirs [list ${CL_DIR}/design ${HDK_SHELL_DESIGN_DIR}/interfaces] [current_fileset]
source ${CL_DIR}/design/sources.tcl
foreach f $otpu_sources { read_verilog -sv ${CL_DIR}/design/$f }
read_verilog -sv ${CL_DIR}/design/cl_opentpu.sv
# sh_ddr's protected implementation instantiates the AXI clock converter; the
# stock HDK CL examples read both full-AXI and AXI-Lite converter IPs. Keep that
# pair together even though our exposed OCL bridge is openTPU RTL.
read_ip [list \
  ${HDK_IP_SRC_DIR}/cl_ddr4/cl_ddr4.xci \
  ${HDK_IP_SRC_DIR}/cl_axi_clock_converter/cl_axi_clock_converter.xci \
  ${HDK_IP_SRC_DIR}/cl_axi_clock_converter_light/cl_axi_clock_converter_light.xci]
read_xdc ${constraints_dir}/cl_synth_user.xdc
update_compile_order -fileset sources_1
synth_design -mode out_of_context -top cl_opentpu -part $DEVICE_TYPE \
  -flatten_hierarchy rebuilt -keep_equivalent_registers
# Diagnostic recovery point only. The launcher accepts post_synth.dcp, which is
# written by the HDK footer only after timing topology and firmware checks pass.
write_checkpoint -force ${checkpoints_dir}/${CL}.${TAG}.unvalidated_synth.dcp
# Apply topology-dependent CDC constraints only after synthesis; they are saved
# in the CL checkpoint and carried into shell implementation.
set otpu_constraint_criticals [get_msg_config -count -severity {CRITICAL WARNING}]
set otpu_constraint_warnings [get_msg_config -count -severity WARNING]
read_xdc -unmanaged ${constraints_dir}/cl_timing_user.tcl
read_xdc -unmanaged ${constraints_dir}/otpu_clk2x.tcl
if {[get_msg_config -count -severity {CRITICAL WARNING}] > $otpu_constraint_criticals ||
    [get_msg_config -count -severity WARNING] > $otpu_constraint_warnings} {
  error "openTPU timing constraints produced a warning"
}
if {![llength $c2x_pb] || ![info exists c2x_rp] || ![llength $c2x_rp] ||
    ![info exists c2x_qp] || ![llength $c2x_qp]} {
  error "openTPU pumped TMEM timing constraints were not applied"
}
report_utilization -hierarchical -file ${reports_dir}/${CL}.${TAG}.synth_utilization.rpt
report_timing_summary -file ${reports_dir}/${CL}.${TAG}.synth_timing.rpt
report_cdc -details -file ${reports_dir}/${CL}.${TAG}.synth_cdc.rpt
report_bus_skew -file ${reports_dir}/${CL}.${TAG}.synth_bus_skew.rpt
report_exceptions -ignored -file ${reports_dir}/${CL}.${TAG}.synth_ignored_exceptions.rpt
report_clock_interaction -file ${reports_dir}/${CL}.${TAG}.synth_clock_interaction.rpt
source ${scripts_dir}/check_ddr_init.tcl
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
puts "OTPU_STAGE_SYNTHESIS_PASSED"
