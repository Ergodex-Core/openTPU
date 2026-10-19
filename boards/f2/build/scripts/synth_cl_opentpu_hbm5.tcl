# Called by the pinned HDK build_all.tcl for a v5 full-HBM target (platform mem_path "engine";
# boards/f2/build.py puts this file in place of synth_cl_opentpu.tcl). Sources and target
# parameters are frozen by build.py in a fresh CL directory before Vivado starts; the HDK's
# cl_mem_hbm_wrapper is copied next to them, and the cl_hbm IP (both stacks, 32 AXI ports at 450
# MHz, the global address switch on) is the HDK's generated one. The memory path is openTPU's
# own (otpu_hbm_ch, a 450 MHz engine per port; docs/full_hbm.md): no SmartConnect block design
# is generated for the channels; the host's chain is AWS's cl_axi_sc_1x1 as shipped, read as
# cl_dram_hbm_dma's synth script reads it.
set_param general.maxThreads 6
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_header.tcl
set_property verilog_define {SYNTHESIS XSDB_SLV_DIS} [current_fileset]
set_property include_dirs [list ${CL_DIR}/design ${HDK_SHELL_DESIGN_DIR}/interfaces] [current_fileset]
source ${CL_DIR}/design/sources.tcl
foreach f $otpu_sources { read_verilog -sv ${CL_DIR}/design/$f }
read_verilog -sv ${CL_DIR}/design/cl_mem_hbm_wrapper.sv
# the host's chain (AWS's cl_dram_hbm_dma: its synth script reads these the same way): its
# cl_axi_sc_1x1 as shipped; the channels have no block design (otpu_hbm_ch)
add_files ${HDK_BD_SRC_DIR}/cl_axi_sc_1x1/cl_axi_sc_1x1.bd
read_verilog ${HDK_BD_GEN_DIR}/cl_axi_sc_1x1/hdl/cl_axi_sc_1x1_wrapper.v
read_verilog -sv ${CL_DIR}/design/cl_opentpu_hbm5.sv
# the HBM controller; sh_ddr (pins only here) keeps the converter pair the DDR build reads
read_ip [list \
  ${HDK_IP_SRC_DIR}/cl_hbm/cl_hbm.xci \
  ${HDK_IP_SRC_DIR}/axi_register_slice/axi_register_slice.xci \
  ${HDK_IP_SRC_DIR}/cl_axi3_256b_reg_slice/cl_axi3_256b_reg_slice.xci \
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
# logic levels and congestion (the AWS Shell Interface Specification's timing hygiene: paths
# above ~10 levels at 250 MHz show here; a report, not a gate)
report_design_analysis -logic_level_distribution -file ${reports_dir}/${CL}.${TAG}.synth_logic_levels.rpt
report_clock_interaction -file ${reports_dir}/${CL}.${TAG}.synth_clock_interaction.rpt
report_cdc -details -file ${reports_dir}/${CL}.${TAG}.synth_cdc.rpt
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
puts "OTPU_STAGE_SYNTHESIS_PASSED"
