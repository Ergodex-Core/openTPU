# The F2 HBM build's SmartConnect fabric, generated on the build host before synthesis
# (boards/f2/build.py runs it for an HBM target: vivado -mode batch -source create_hbm_sc.tcl
# -tclargs IPDIR NPORT PART). Modelled on Alex's burst200 boards/f2/scripts/create_f2_ip.tcl
# (ergodex-openTPU alex-burst200, commit 06c169cb: one SmartConnect memory_sc, S_HOST 512 b @
# host_clk 250 + S_N0 / S_N1 512 b @ core_clk 200 onto M_H0 / M_H4, AXI3 256 b @ hbm_clk 450,
# the interface properties below), scaled to NPORT channels and structured per channel:
#   sc_c<p>   (p < NPORT): a SmartConnect with two slaves, S_N<p> (otpu_native_axi p, 512 b @
#             core_clk) and the host's branch (512 b @ host_clk), one master M_H<p> (AXI3, 256 b
#             @ hbm_clk: HBM port p); the clock and width conversion and the AXI4 -> AXI3 split
#             are SmartConnect's. S_N<p> reaches only M_H<p>, at p x 512 MiB (its pseudo-channel).
#   sc_host   a 1 x NPORT SmartConnect @ host_clk (NPORT > 16: sc_host_top 1 x NPORT / 16 into
#             1 x 16 groups sc_host<g>): S_HOST (the PCIS through otpu_f2_pcis_gate, 512 b, 64-bit
#             addresses) demuxed by address, M_H<p> at p x 512 MiB (BAR4 offset = HBM address).
# No AMD IP product is committed: this project, its generated files and the OOC checkpoints live
# in IPDIR (the run's cl_opentpu/ip). It writes IPDIR/otpu_hbm_sc_inst.vh (the CL's instance of
# the wrapper, from the wrapper's own ports: hbm_sc_inst.tcl) and IPDIR/ip_done.txt.
set ipdir [file normalize [lindex $argv 0]]
set nport [lindex $argv 1]
set part [lindex $argv 2]
if {![string is integer -strict $nport] || $nport < 1 || $nport > 32} { error "openTPU SmartConnect: NPORT $nport" }
source [file join [file dirname [file normalize [info script]]] hbm_sc_inst.tcl]
file mkdir $ipdir
cd $ipdir
set_param general.maxThreads 8
create_project otpu_hbm_sc $ipdir/project -part $part -force
set_property XPM_LIBRARIES {XPM_CDC XPM_MEMORY XPM_FIFO} [current_project]
create_bd_design otpu_hbm_sc
foreach {name hz} {core_clk 200000000 host_clk 250000000 hbm_clk 450000000} {
  set p [create_bd_port -dir I -type clk $name]
  set_property CONFIG.FREQ_HZ $hz $p
}
foreach name {aresetn host_aresetn} {
  set r [create_bd_port -dir I -type rst $name]
  set_property CONFIG.POLARITY ACTIVE_LOW $r
}
# Alex's interface properties (create_f2_ip.tcl), for every external interface port
proc otpu_intf {name mode proto width aw id} {
  set p [create_bd_intf_port -mode $mode -vlnv xilinx.com:interface:aximm_rtl:1.0 $name]
  set_property -dict [list CONFIG.PROTOCOL $proto CONFIG.ADDR_WIDTH $aw CONFIG.DATA_WIDTH $width \
    CONFIG.ID_WIDTH $id CONFIG.HAS_BURST {1} CONFIG.HAS_LOCK {1} CONFIG.HAS_PROT {1} \
    CONFIG.HAS_CACHE {1} CONFIG.HAS_QOS {1} CONFIG.HAS_REGION {0} CONFIG.SUPPORTS_NARROW_BURST {1} \
    CONFIG.MAX_BURST_LENGTH {16} CONFIG.NUM_READ_OUTSTANDING {128} CONFIG.NUM_WRITE_OUTSTANDING {32}] $p
  return $p
}
# the host's demux: one level up to 16 masters, else a 1 x G top over 1 x 16 groups
otpu_intf S_HOST Slave AXI4 512 64 16
set hostm {}
proc otpu_sc {name nsi nmi nclk} {
  set sc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* $name]
  set_property -dict [list CONFIG.NUM_SI $nsi CONFIG.NUM_MI $nmi CONFIG.NUM_CLKS $nclk] $sc
  return $sc
}
proc otpu_hostclk {name} {
  connect_bd_net [get_bd_ports host_clk] [get_bd_pins $name/aclk]
  connect_bd_net [get_bd_ports host_aresetn] [get_bd_pins $name/aresetn]
}
if {$nport <= 16} {
  otpu_sc sc_host 1 $nport 1
  otpu_hostclk sc_host
  connect_bd_intf_net [get_bd_intf_ports S_HOST] [get_bd_intf_pins sc_host/S00_AXI]
  for {set p 0} {$p < $nport} {incr p} { lappend hostm sc_host/M[format %02d $p]_AXI }
} else {
  if {$nport % 16} { error "openTPU SmartConnect: NPORT > 16 must be a multiple of 16" }
  set ng [expr {$nport / 16}]
  otpu_sc sc_host_top 1 $ng 1
  otpu_hostclk sc_host_top
  connect_bd_intf_net [get_bd_intf_ports S_HOST] [get_bd_intf_pins sc_host_top/S00_AXI]
  for {set g 0} {$g < $ng} {incr g} {
    otpu_sc sc_host$g 1 16 1
    otpu_hostclk sc_host$g
    connect_bd_intf_net [get_bd_intf_pins sc_host_top/M[format %02d $g]_AXI] [get_bd_intf_pins sc_host$g/S00_AXI]
    for {set j 0} {$j < 16} {incr j} { lappend hostm sc_host$g/M[format %02d $j]_AXI }
  }
}
# a SmartConnect per channel: S00 the channel @ core_clk, S01 the host's branch @ host_clk,
# M00 HBM port p @ hbm_clk (aclk / aclk1 / aclk2: SmartConnect takes each interface's clock
# from the clock its source is associated with, as in Alex's three-clock memory_sc)
set busn {}
set busm {}
for {set p 0} {$p < $nport} {incr p} {
  otpu_sc sc_c$p 2 1 3
  connect_bd_net [get_bd_ports core_clk] [get_bd_pins sc_c$p/aclk]
  connect_bd_net [get_bd_ports host_clk] [get_bd_pins sc_c$p/aclk1]
  connect_bd_net [get_bd_ports hbm_clk] [get_bd_pins sc_c$p/aclk2]
  connect_bd_net [get_bd_ports aresetn] [get_bd_pins sc_c$p/aresetn]
  otpu_intf S_N$p Slave AXI4 512 34 1
  otpu_intf M_H$p Master AXI3 256 34 6
  connect_bd_intf_net [get_bd_intf_ports S_N$p] [get_bd_intf_pins sc_c$p/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins [lindex $hostm $p]] [get_bd_intf_pins sc_c$p/S01_AXI]
  connect_bd_intf_net [get_bd_intf_pins sc_c$p/M00_AXI] [get_bd_intf_ports M_H$p]
  lappend busn S_N$p
  lappend busm M_H$p
}
set_property CONFIG.ASSOCIATED_BUSIF [join $busn :] [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_BUSIF {S_HOST} [get_bd_ports host_clk]
set_property CONFIG.ASSOCIATED_BUSIF [join $busm :] [get_bd_ports hbm_clk]
set_property CONFIG.ASSOCIATED_RESET {aresetn} [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_RESET {host_aresetn} [get_bd_ports host_clk]
# addresses: channel p's pseudo-channel at p x 512 MiB, for its own master and the host's
for {set p 0} {$p < $nport} {incr p} {
  set base [format 0x%09llX [expr {$p << 29}]]   ;# (ll: past 4 GiB)
  assign_bd_address -offset $base -range 0x20000000 -target_address_space [get_bd_addr_spaces S_HOST] [get_bd_addr_segs M_H$p/Reg] -force
  assign_bd_address -offset $base -range 0x20000000 -target_address_space [get_bd_addr_spaces S_N$p] [get_bd_addr_segs M_H$p/Reg] -force
}
validate_bd_design
save_bd_design
set bd [get_files otpu_hbm_sc.bd]
generate_target all $bd
make_wrapper -files $bd -top
write_bd_tcl -force $ipdir/otpu_hbm_sc_generated.tcl
# the CL's instance from the wrapper's ports (fail closed on any port the CL cannot map)
set wrappers [glob -nocomplain $ipdir/project/otpu_hbm_sc.gen/sources_1/bd/otpu_hbm_sc/hdl/otpu_hbm_sc_wrapper.v \
                               $ipdir/project/otpu_hbm_sc.srcs/sources_1/bd/otpu_hbm_sc/hdl/otpu_hbm_sc_wrapper.v]
if {[llength $wrappers] != 1} { error "openTPU SmartConnect: expected one generated wrapper, found $wrappers" }
set f [open [lindex $wrappers 0]]; set text [read $f]; close $f
set inst [otpu_sc_instance $text $nport]
set f [open $ipdir/otpu_hbm_sc_inst.vh w]; puts -nonewline $f $inst; close $f
file copy -force [lindex $wrappers 0] $ipdir/otpu_hbm_sc_wrapper.v
puts "OTPU_HBM_SC nport=$nport wrapper=[lindex $wrappers 0]"
# out-of-context synthesis of the block design, at bounded concurrency
create_ip_run $bd
set jobs [get_runs -filter {IS_SYNTHESIS && NAME != synth_1}]
launch_runs $jobs -jobs 4
foreach r $jobs { wait_on_run $r; if {[get_property PROGRESS $r] ne "100%"} {error "openTPU SmartConnect: OOC synthesis failed: $r"} }
set fp [open $ipdir/ip_done.txt w]; puts $fp "IP_DONE nport=$nport"; close $fp
puts "OTPU_HBM_SC_DONE"
