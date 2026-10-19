# Called by the pinned HDK build_all.tcl for an HBM target (boards/f2/build.py puts this file in
# place of synth_cl_opentpu.tcl). Sources and target parameters are frozen by build.py in a fresh
# CL directory before Vivado starts; the HDK's cl_mem_hbm_wrapper is copied next to them, and the
# cl_hbm IP (both stacks, 32 AXI ports at 450 MHz, the global address switch on) is the HDK's
# generated one (hdk/common/ip, read here as cl_mem_perf reads it). The SmartConnect fabric (the
# memory path, Alex's burst200 design at NPORT channels) is the block design build.py generated
# before this run (create_hbm_sc.tcl into ${CL_DIR}/ip: its .bd with its out-of-context
# checkpoints, its wrapper, and otpu_hbm_sc_inst.vh, the CL's instance of it). docs/f2_hbm.md.
set_param general.maxThreads 6
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_header.tcl
set_property verilog_define {SYNTHESIS XSDB_SLV_DIS} [current_fileset]
set otpu_ip ${CL_DIR}/ip
if {![file exists ${otpu_ip}/ip_done.txt] || ![file exists ${otpu_ip}/otpu_hbm_sc_inst.vh]} {
  error "openTPU: the SmartConnect fabric was not generated (${otpu_ip}: create_hbm_sc.tcl)"
}
set_property include_dirs [list ${CL_DIR}/design ${otpu_ip} ${HDK_SHELL_DESIGN_DIR}/interfaces] [current_fileset]
source ${CL_DIR}/design/sources.tcl
foreach f $otpu_sources { read_verilog -sv ${CL_DIR}/design/$f }
read_verilog -sv ${CL_DIR}/design/cl_mem_hbm_wrapper.sv
# the SmartConnect fabric (as Alex's build_f2.tcl reads his f2_mem block design)
add_files ${otpu_ip}/project/otpu_hbm_sc.srcs/sources_1/bd/otpu_hbm_sc/otpu_hbm_sc.bd
read_verilog ${otpu_ip}/otpu_hbm_sc_wrapper.v
read_verilog -sv ${CL_DIR}/design/cl_opentpu_hbm.sv
# the HBM controller; sh_ddr (pins only here) keeps the converter pair the DDR build reads
read_ip [list \
  ${HDK_IP_SRC_DIR}/cl_hbm/cl_hbm.xci \
  ${HDK_IP_SRC_DIR}/cl_axi_clock_converter/cl_axi_clock_converter.xci \
  ${HDK_IP_SRC_DIR}/cl_axi_clock_converter_light/cl_axi_clock_converter_light.xci]
read_xdc ${constraints_dir}/cl_synth_user.xdc
update_compile_order -fileset sources_1
synth_design -mode out_of_context -top cl_opentpu -part $DEVICE_TYPE \
  -flatten_hierarchy rebuilt -keep_equivalent_registers
write_checkpoint -force ${checkpoints_dir}/${CL}.${TAG}.unvalidated_synth.dcp
set otpu_constraint_criticals [get_msg_config -count -severity {CRITICAL WARNING}]
set otpu_constraint_warnings [get_msg_config -count -severity WARNING]
read_xdc -unmanaged ${constraints_dir}/cl_timing_user.tcl
read_xdc -unmanaged ${constraints_dir}/otpu_clk2x.tcl
if {[get_msg_config -count -severity {CRITICAL WARNING}] > $otpu_constraint_criticals ||
    [get_msg_config -count -severity WARNING] > $otpu_constraint_warnings} {
  error "openTPU timing constraints produced a warning"
}
if {![llength $c2x_pb] || ![info exists c2x_ip] || ![llength $c2x_ip] ||
    ![info exists c2x_qp] || ![llength $c2x_qp]} {
  error "openTPU pumped TMEM timing constraints were not applied"
}
report_utilization -file ${reports_dir}/${CL}.${TAG}.synth_utilization.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file ${reports_dir}/${CL}.${TAG}.synth_utilization_hier.rpt
report_timing_summary -file ${reports_dir}/${CL}.${TAG}.synth_timing.rpt
report_clock_interaction -file ${reports_dir}/${CL}.${TAG}.synth_clock_interaction.rpt
report_cdc -details -file ${reports_dir}/${CL}.${TAG}.synth_cdc.rpt
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
puts "OTPU_STAGE_SYNTHESIS_PASSED"
