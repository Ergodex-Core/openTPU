# C1 adaptation of Alex burst200 06c169cb's private vendor-IP recipe.
# Eight native 512 MiB windows map directly to physical HBM ports 0..7.
# This new topology requires its own simulation, implementation and qualification.
set run [file normalize [lindex $argv 0]]
set hdk [file normalize [lindex $argv 1]]
set jobs 1
if {[llength $argv] > 2} {set jobs [lindex $argv 2]}
if {![string is integer -strict $jobs] || $jobs < 1 || $jobs > 3} {error "C1 IP jobs must be 1..3"}
cd $run
set_param general.maxThreads 4
create_project f2_ip $run/ip_project -part xcvu47p-fsvh2892-2-e -force
set_property XPM_LIBRARIES {XPM_CDC XPM_MEMORY XPM_FIFO} [current_project]
import_ip -files $hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_hbm/cl_hbm.xci
upgrade_ip [get_ips cl_hbm]
report_property [get_ips cl_hbm]
create_ip -name clk_wiz -vendor xilinx.com -library ip -module_name f2_core_clk
set_property -dict [list CONFIG.PRIM_SOURCE {No_buffer} CONFIG.PRIM_IN_FREQ {250.000} CONFIG.NUM_OUT_CLKS {2} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {200.000} CONFIG.CLKOUT2_USED {true} CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {400.000} CONFIG.USE_RESET {true} CONFIG.RESET_TYPE {ACTIVE_HIGH} CONFIG.USE_LOCKED {true}] [get_ips f2_core_clk]
create_ip -name clk_wiz -vendor xilinx.com -library ip -module_name f2_hbm_clk
set_property -dict [list CONFIG.PRIM_SOURCE {No_buffer} CONFIG.PRIM_IN_FREQ {100.000} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {450.000} CONFIG.USE_RESET {true} CONFIG.RESET_TYPE {ACTIVE_HIGH} CONFIG.USE_LOCKED {true}] [get_ips f2_hbm_clk]
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
set_property -dict [list CONFIG.NUM_SI {9} CONFIG.NUM_MI {8} CONFIG.NUM_CLKS {3}] $sc
connect_bd_net [get_bd_ports core_clk] [get_bd_pins memory_sc/aclk]
connect_bd_net [get_bd_ports host_clk] [get_bd_pins memory_sc/aclk1]
connect_bd_net [get_bd_ports hbm_clk] [get_bd_pins memory_sc/aclk2]
connect_bd_net $reset [get_bd_pins memory_sc/aresetn]
set interfaces [list S_HOST S00_AXI host_clk 512 64 16 AXI4]
set native_names {}
set hbm_names {}
for {set p 0} {$p < 8} {incr p} {
  lappend interfaces S_N$p [format S%02d_AXI [expr {$p+1}]] core_clk 512 34 1 AXI4
  lappend interfaces M_H$p [format M%02d_AXI $p] hbm_clk 256 34 6 AXI3
  lappend native_names S_N$p
  lappend hbm_names M_H$p
}
foreach {name pin clk width aw id proto} $interfaces {
  set mode [expr {[string match S* $name] ? "Slave" : "Master"}]
  set p [create_bd_intf_port -mode $mode -vlnv xilinx.com:interface:aximm_rtl:1.0 $name]
  set_property -dict [list CONFIG.PROTOCOL $proto CONFIG.ADDR_WIDTH $aw CONFIG.DATA_WIDTH $width CONFIG.ID_WIDTH $id CONFIG.HAS_BURST {1} CONFIG.HAS_LOCK {1} CONFIG.HAS_PROT {1} CONFIG.HAS_CACHE {1} CONFIG.HAS_QOS {1} CONFIG.HAS_REGION {0} CONFIG.SUPPORTS_NARROW_BURST {1} CONFIG.MAX_BURST_LENGTH {16} CONFIG.NUM_READ_OUTSTANDING {128} CONFIG.NUM_WRITE_OUTSTANDING {32}] $p
  connect_bd_intf_net $p [get_bd_intf_pins memory_sc/$pin]
}
set_property CONFIG.ASSOCIATED_BUSIF [join $native_names :] [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_BUSIF S_HOST [get_bd_ports host_clk]
set_property CONFIG.ASSOCIATED_BUSIF [join $hbm_names :] [get_bd_ports hbm_clk]
set_property CONFIG.ASSOCIATED_RESET aresetn [get_bd_ports core_clk]
# Exactly 16 address assignments and 56 excluded native crosspoints.
# All byte addresses remain identical to C1's existing raw-PCIS sidecar layout.
for {set p 0} {$p < 8} {incr p} {
  set base [format 0x%08x [expr {wide($p) << 29}]]
  foreach source [list S_HOST S_N$p] {
    assign_bd_address -offset $base -range 0x20000000 -target_address_space [get_bd_addr_spaces $source] [get_bd_addr_segs M_H$p/Reg] -force
  }
  for {set q 0} {$q < 8} {incr q} {
    if {$q != $p} {exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces S_N$p] [get_bd_addr_segs M_H$q/Reg]}
  }
}
validate_bd_design
save_bd_design
set bd [get_files f2_mem.bd]
set standalone_ips [get_ips {cl_hbm f2_core_clk f2_hbm_clk f2_ocl_cdc}]
generate_target all $standalone_ips
generate_target all $bd
make_wrapper -files $bd -top
write_bd_tcl -force $run/f2_mem_generated.tcl
foreach ip $standalone_ips {
  set f [get_files [get_property IP_FILE $ip]]
  if {[get_property GENERATE_SYNTH_CHECKPOINT $f]} {create_ip_run $f}
}
create_ip_run $bd
set runs [get_runs -filter {IS_SYNTHESIS && NAME != synth_1}]
launch_runs $runs -jobs $jobs
foreach r $runs {wait_on_run $r; if {[get_property PROGRESS $r] ne "100%"} {error "IP synthesis failed: $r"}}
set fp [open $run/ip_done.txt w]; puts $fp "IP_DONE"; close $fp
