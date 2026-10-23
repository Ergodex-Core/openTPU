# Full shell-linked build. Every artifact is written under the experiment directory.
set run [file normalize [lindex $argv 0]]
set iprun [file normalize [lindex $argv 1]]
set pcisrun [file normalize [lindex $argv 2]]
set root [file normalize [lindex $argv 3]]
set hdk [file normalize [lindex $argv 4]]
set clockrun [file normalize [lindex $argv 5]]
set shell $hdk/common/shell_stable
cd $run
set_param general.maxThreads 8
set_param sta.enableAutoGenClkNamePersistence 0
# The launch is allowed to precede IP completion, but no design is built with stubs.
foreach dep [list $iprun $pcisrun $clockrun] {
  set attempts 0
  while {![file exists $dep/ip_done.txt]} {
    if {$attempts > 360} {error "IP prerequisite did not complete: $dep"}
    puts "WAITING_FOR_IP $dep"
    after 10000
    incr attempts
  }
}
source $run/config.tcl
create_project -in_memory -part $part
set_property XPM_LIBRARIES {XPM_CDC XPM_MEMORY XPM_FIFO} [current_project]
set_property include_dirs [list $shell/design/interfaces $hdk/common/lib] [current_fileset]
read_verilog -sv $hdk/common/lib/interfaces.sv
read_verilog -sv $hdk/common/lib/lib_pipe.sv
foreach f {sync.v flop_ccf.sv ccf_ctl.v mgt_acc_axl.sv mgt_gen_axl.sv sh_ddr.sv} {
  read_verilog -sv $shell/design/sh_ddr/synth/$f
}
foreach f $rtl_files {read_verilog -sv $root/rtl/$f}
read_verilog -sv $hdk/cl/examples/cl_mem_perf/design/cl_mem_hbm_wrapper.sv
read_verilog -sv [glob $run/rtl/*.sv]
foreach ip {cl_hbm f2_ocl_cdc} {
  read_ip $iprun/ip_project/f2_ip.srcs/sources_1/ip/$ip/$ip.xci
}
foreach ip {f2_core_clk f2_hbm_clk} {
  read_ip $clockrun/ip_project/clocks.srcs/sources_1/ip/$ip/$ip.xci
}
read_ip $pcisrun/ip_project/pcis.srcs/sources_1/ip/f2_pcis_pipe/f2_pcis_pipe.xci
set bd $iprun/ip_project/f2_ip.srcs/sources_1/bd/f2_mem/f2_mem.bd
add_files $bd
read_verilog $iprun/ip_project/f2_ip.gen/sources_1/bd/f2_mem/hdl/f2_mem_wrapper.v
read_xdc $shell/build/constraints/cl_pins.xdc
set fp [open $run/ooc_clocks.xdc w]
puts $fp {create_clock -name clk_main_a0 -period 4.000 [get_ports clk_main_a0]}
puts $fp {create_clock -name clk_hbm_ref -period 10.000 [get_ports clk_hbm_ref]}
close $fp
read_xdc $run/ooc_clocks.xdc
set_property USED_IN {synthesis implementation OUT_OF_CONTEXT} [get_files ooc_clocks.xdc]
set_property PROCESSING_ORDER EARLY [get_files ooc_clocks.xdc]
update_compile_order -fileset sources_1
synth_design -mode out_of_context -top cl_opentpu -part $part -flatten_hierarchy rebuilt -keep_equivalent_registers
write_checkpoint -force $run/post_synth.dcp
report_utilization -hierarchical -file $run/synth_utilization.rpt
report_timing_summary -report_unconstrained -file $run/synth_timing.rpt
report_drc -file $run/synth_drc.rpt
# Linking the supplied static shell is mandatory; never substitute a shell stub.
close_project
create_project -in_memory -part $part
add_files $shell/build/checkpoints/from_aws/cl_bb_routed.small_shell.dcp
add_files $run/post_synth.dcp
set_property SCOPED_TO_CELLS {WRAPPER/CL} [get_files $run/post_synth.dcp]
link_design -mode default -reconfig_partitions {WRAPPER/CL} -top top
foreach {output factor} {clk_out1 1.0 clk_out2 2.0} {
  set clock [get_clocks -of_objects [get_pins WRAPPER/CL/u_core_clk/$output]]
  if {[llength $clock] != 1} {error "Expected one derived clock for $output"}
  set period [get_property PERIOD $clock]
  set expected [expr {1000000.0/($core_khz*$factor)}]
  if {abs($period-$expected) > 0.001} {error "Clock hardware / CORE_KHZ mismatch: $period versus $expected ns"}
  puts "VERIFIED_CORE_CLOCK $output period_ns=$period"
}
write_checkpoint -force $run/post_link.dcp
set complete [open $run/link_done.txt w]; puts $complete LINK_COMPLETE; close $complete
if {[llength $argv] > 6 && [lindex $argv 6] eq "link_only"} {
  puts "SHELL_LINK_COMPLETE_IMPLEMENTATION_NOT_RUN"
  close_project
  exit
}
if {![llength [get_pblocks -quiet pblock_CL]]} {
  read_xdc $shell/build/constraints/small_shell_level_1_fp_cl.xdc
} else {
  puts "Retaining legal pblock_CL from the linked AWS shell checkpoint"
}
source $run/floorplan.tcl
source $run/core_timing.tcl
source $run/platform_constraints.tcl
opt_design
# Same AMD HBM workaround as the official HDK, using our instance name.
set hbm [get_cells -quiet WRAPPER/CL/u_hbm/HBM_CORE_I/inst]
if {[llength $hbm]} {set_property DONT_TOUCH false $hbm}
write_checkpoint -force $run/post_opt.dcp
report_utilization -hierarchical -file $run/opt_utilization.rpt
report_utilization -pblocks [get_pblocks pblock_CL_SLR0] -file $run/slr0_utilization.rpt
report_drc -file $run/opt_drc.rpt
place_design -directive Explore -no_bufg_opt
write_checkpoint -force $run/post_place.dcp
report_timing_summary -report_unconstrained -file $run/place_timing.rpt
report_design_analysis -congestion -file $run/place_congestion.rpt
phys_opt_design -directive AggressiveExplore
write_checkpoint -force $run/post_phys_opt.dcp
# E31 isolated Vivado 2025.2 leaf-clock optimizer crash workaround. No timing
# exceptions change; all routed timing/DRC/CDC reports are still required.
puts "Leaf clock optimization default=[get_param route.optimizeLeafClkProgDlys]"
set_param route.optimizeLeafClkProgDlys false
route_design -directive Explore -tns_cleanup
# E34/E35: placement changes bank 74 to 0.60 V. Restore the untouched shell's
# 0.84 V value for its POD12_DCI disabled-DDR inputs; do not waive voltage DRC.
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
if {$setup < 0 || $hold < 0} {error "Routed checkpoint has timing violations; not deployment-ready"}
puts "ROUTED_TIMING_PASS_PENDING_REPORT_REVIEW"
