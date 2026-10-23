# Build private copies of all IP; never generate into the installed HDK.
set run [file normalize [lindex $argv 0]]
set hdk [file normalize [lindex $argv 1]]
cd $run
set_param general.maxThreads 8
create_project f2_ip $run/ip_project -part xcvu47p-fsvh2892-2-e -force
set_property XPM_LIBRARIES {XPM_CDC XPM_MEMORY XPM_FIFO} [current_project]
import_ip -files $hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_hbm/cl_hbm.xci
upgrade_ip [get_ips cl_hbm]
# Inspect all relevant properties in the log; preserve the reference default map.
report_property [get_ips cl_hbm]
create_ip -name clk_wiz -vendor xilinx.com -library ip -module_name f2_core_clk
set_property -dict [list CONFIG.PRIM_SOURCE {No_buffer} CONFIG.PRIM_IN_FREQ {250.000} CONFIG.NUM_OUT_CLKS {2} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {200.000} CONFIG.CLKOUT2_USED {true} CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {400.000} CONFIG.USE_RESET {true} CONFIG.RESET_TYPE {ACTIVE_HIGH} CONFIG.USE_LOCKED {true}] [get_ips f2_core_clk]
create_ip -name clk_wiz -vendor xilinx.com -library ip -module_name f2_hbm_clk
set_property -dict [list CONFIG.PRIM_SOURCE {No_buffer} CONFIG.PRIM_IN_FREQ {100.000} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {450.000} CONFIG.USE_RESET {true} CONFIG.RESET_TYPE {ACTIVE_HIGH} CONFIG.USE_LOCKED {true}] [get_ips f2_hbm_clk]
# Control CDC: standard AXI-Lite converter, separate reset per domain.
create_ip -name axi_clock_converter -vendor xilinx.com -library ip -module_name f2_ocl_cdc
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {32} CONFIG.DATA_WIDTH {32}] [get_ips f2_ocl_cdc]
create_bd_design f2_mem
foreach {name hz} {core_clk 200000000 host_clk 250000000 hbm_clk 450000000} {
  set p [create_bd_port -dir I -type clk $name]
  set_property CONFIG.FREQ_HZ $hz $p
}
set reset [create_bd_port -dir I -type rst aresetn]
set_property CONFIG.POLARITY ACTIVE_LOW $reset
set sc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* memory_sc]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {2} CONFIG.NUM_CLKS {3}] $sc
connect_bd_net [get_bd_ports core_clk] [get_bd_pins memory_sc/aclk]
connect_bd_net [get_bd_ports host_clk] [get_bd_pins memory_sc/aclk1]
connect_bd_net [get_bd_ports hbm_clk] [get_bd_pins memory_sc/aclk2]
connect_bd_net $reset [get_bd_pins memory_sc/aresetn]
foreach {name pin clk width aw id proto} {
  S_HOST S00_AXI host_clk 512 64 16 AXI4
  S_N0   S01_AXI core_clk 512 34 1 AXI4
  S_N1   S02_AXI core_clk 512 34 1 AXI4
  M_H0   M00_AXI hbm_clk 256 34 6 AXI3
  M_H4   M01_AXI hbm_clk 256 34 6 AXI3
} {
  set mode [expr {[string match S* $name] ? "Slave" : "Master"}]
  set p [create_bd_intf_port -mode $mode -vlnv xilinx.com:interface:aximm_rtl:1.0 $name]
  set_property -dict [list CONFIG.PROTOCOL $proto CONFIG.ADDR_WIDTH $aw CONFIG.DATA_WIDTH $width CONFIG.ID_WIDTH $id CONFIG.HAS_BURST {1} CONFIG.HAS_LOCK {1} CONFIG.HAS_PROT {1} CONFIG.HAS_CACHE {1} CONFIG.HAS_QOS {1} CONFIG.HAS_REGION {0} CONFIG.SUPPORTS_NARROW_BURST {1} CONFIG.MAX_BURST_LENGTH {16} CONFIG.NUM_READ_OUTSTANDING {128} CONFIG.NUM_WRITE_OUTSTANDING {32}] $p
  connect_bd_intf_net $p [get_bd_intf_pins memory_sc/$pin]
}
set_property CONFIG.ASSOCIATED_BUSIF {S_N0:S_N1} [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_BUSIF {S_HOST} [get_bd_ports host_clk]
set_property CONFIG.ASSOCIATED_BUSIF {M_H0:M_H4} [get_bd_ports hbm_clk]
set_property CONFIG.ASSOCIATED_RESET {aresetn} [get_bd_ports core_clk]
# Give host access to both channel windows; each core master only its own window.
foreach {slave master base} {S_HOST M_H0 0x00000000 S_HOST M_H4 0x80000000 S_N0 M_H0 0x00000000 S_N1 M_H4 0x80000000} {
  assign_bd_address -offset $base -range 0x80000000 -target_address_space [get_bd_addr_spaces $slave] [get_bd_addr_segs $master/Reg] -force
}
exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces S_N0] [get_bd_addr_segs M_H4/Reg]
exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces S_N1] [get_bd_addr_segs M_H0/Reg]
validate_bd_design
save_bd_design
set bd [get_files f2_mem.bd]
set standalone_ips [get_ips {cl_hbm f2_core_clk f2_hbm_clk f2_ocl_cdc}]
generate_target all $standalone_ips
generate_target all $bd
make_wrapper -files $bd -top
write_bd_tcl -force $run/f2_mem_generated.tcl
# OOC synth IP and BD, at bounded concurrency.
foreach ip $standalone_ips {
  set f [get_files [get_property IP_FILE $ip]]
  if {[get_property GENERATE_SYNTH_CHECKPOINT $f]} {create_ip_run $f}
}
create_ip_run $bd
set jobs [get_runs -filter {IS_SYNTHESIS && NAME != synth_1}]
launch_runs $jobs -jobs 3
foreach r $jobs { wait_on_run $r; if {[get_property PROGRESS $r] ne "100%"} {error "IP synthesis failed: $r"} }
set fp [open $run/ip_done.txt w]; puts $fp "IP_DONE"; close $fp
