# The F2 HBM build's channel SmartConnects, generated on the build host before synthesis
# (boards/f2/build.py runs it for an HBM target: vivado -mode batch -source create_hbm_sc.tcl
# -tclargs IPDIR NPORT PART CORE_HZ HDK_BD_DIR). Since the first placed build (cd16e195: a
# 2 x 1 SmartConnect per channel with the host's branch and a 1 x NPORT host demux, -1.35 ns at
# 450 MHz and -1.61 at 250) the memory path has the shape of AWS's cl_dram_hbm_dma
# (hdk/cl/examples/cl_dram_hbm_dma/design/cl_hbm_axi4.sv): per channel axi_register_slice ->
# a 1 x 1 SmartConnect -> cl_axi3_256b_reg_slice, the register slices instantiated in
# cl_opentpu_hbm.sv from the HDK's prebuilt IP, the host through AWS's own cl_axi_sc_1x1 as
# shipped. This block design holds the channels' 1 x 1 SmartConnects only:
#   sc_c<p>   (p < NPORT): S_N<p> (512 b AXI4 @ core_clk, from channel p's register slice) onto
#             M_H<p> (256 b AXI3 @ hbm_clk, to its 450 MHz register slice and HBM port p); every
#             setting copied from the HDK's cl_axi_sc_1x1.bd (its SmartConnect cell's CONFIG and
#             its AXI4 / AXI3 interface ports' CONFIG), except the slave side's clock: the core's
#             (S00 at 250 MHz there). S_N<p> reaches M_H<p> only, at p x 512 MiB.
# Modelled on Alex's burst200 create_f2_ip.tcl (ergodex-openTPU 06c169cb). No AMD IP product is
# committed: this project, its generated files and the OOC checkpoints live in IPDIR (the run's
# cl_opentpu/ip). It writes IPDIR/otpu_hbm_sc_inst.vh (the CL's instance of the wrapper, from the
# wrapper's own ports: hbm_sc_inst.tcl) and IPDIR/ip_done.txt.
set ipdir [file normalize [lindex $argv 0]]
set nport [lindex $argv 1]
set part [lindex $argv 2]
set core_hz [lindex $argv 3]
set hdk_bd [file normalize [lindex $argv 4]]
if {![string is integer -strict $nport] || $nport < 1 || $nport > 31} { error "openTPU SmartConnect: NPORT $nport" }
if {$core_hz ni {200000000 175000000 150000000}} { error "openTPU SmartConnect: core clock $core_hz Hz" }
set ref_bd $hdk_bd/cl_axi_sc_1x1/cl_axi_sc_1x1.bd
if {![file exists $ref_bd]} { error "openTPU SmartConnect: the HDK's cl_axi_sc_1x1 block design is missing ($ref_bd)" }
source [file join [file dirname [file normalize [info script]]] hbm_sc_inst.tcl]
file mkdir $ipdir
cd $ipdir
set_param general.maxThreads 8
create_project otpu_hbm_sc $ipdir/project -part $part -force
set_property XPM_LIBRARIES {XPM_CDC XPM_MEMORY XPM_FIFO} [current_project]
# ---- the reference: AWS's cl_axi_sc_1x1, read from a copy (the HDK's files are not touched)
file mkdir $ipdir/ref
file copy -force $ref_bd $ipdir/ref/cl_axi_sc_1x1.bd
add_files -norecurse $ipdir/ref/cl_axi_sc_1x1.bd
open_bd_design [get_files cl_axi_sc_1x1.bd]
proc otpu_config {obj} {
  # an object's settable CONFIG.* values (read-only ones are skipped by set_property's catch)
  set d {}
  foreach k [list_property $obj CONFIG.*] { dict set d $k [get_property $k $obj] }
  return $d
}
set ref_sc [get_bd_cells -filter {VLNV =~ *:smartconnect:*}]
if {[llength $ref_sc] != 1} { error "openTPU SmartConnect: cl_axi_sc_1x1 holds [llength $ref_sc] SmartConnects, expected 1" }
set sc_cfg [otpu_config $ref_sc]
set ref_vlnv [get_property VLNV $ref_sc]   ;# (read while the reference is open: a closed design's object crashes get_property)
set ref_s [get_bd_intf_ports -filter {MODE == Slave}]
set ref_m [get_bd_intf_ports -filter {MODE == Master}]
if {[llength $ref_s] != 1 || [llength $ref_m] != 1} { error "openTPU SmartConnect: cl_axi_sc_1x1's interface ports: [llength $ref_s] slave, [llength $ref_m] master" }
set s_cfg [otpu_config $ref_s]
set m_cfg [otpu_config $ref_m]
foreach k {CONFIG.NUM_SI CONFIG.NUM_MI CONFIG.NUM_CLKS} want {1 1 2} {
  if {[dict get $sc_cfg $k] != $want} { error "openTPU SmartConnect: cl_axi_sc_1x1 $k [dict get $sc_cfg $k], expected $want" }
}
puts "OTPU_HBM_SC reference [get_property NAME $ref_sc]: $sc_cfg"
puts "OTPU_HBM_SC reference slave [get_property NAME $ref_s]: $s_cfg"
puts "OTPU_HBM_SC reference master [get_property NAME $ref_m]: $m_cfg"
close_bd_design [get_bd_designs cl_axi_sc_1x1]
remove_files [get_files cl_axi_sc_1x1.bd]
proc otpu_apply {obj cfg skip} {
  set n 0
  dict for {k v} $cfg {
    if {$k in $skip} { continue }
    if {![catch {set_property $k $v $obj}]} { incr n }
  }
  return $n
}
# ---- the channels' block design
create_bd_design otpu_hbm_sc
foreach {name hz} [list core_clk $core_hz hbm_clk 450000000] {
  set p [create_bd_port -dir I -type clk $name]
  set_property CONFIG.FREQ_HZ $hz $p
}
set r [create_bd_port -dir I -type rst aresetn]
set_property CONFIG.POLARITY ACTIVE_LOW $r
set busn {}
set busm {}
for {set p 0} {$p < $nport} {incr p} {
  set sc [create_bd_cell -type ip -vlnv $ref_vlnv sc_c$p]
  otpu_apply $sc $sc_cfg {CONFIG.Component_Name}
  connect_bd_net [get_bd_ports core_clk] [get_bd_pins sc_c$p/aclk]
  connect_bd_net [get_bd_ports hbm_clk] [get_bd_pins sc_c$p/aclk1]
  connect_bd_net [get_bd_ports aresetn] [get_bd_pins sc_c$p/aresetn]
  set sp [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_N$p]
  set mp [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_H$p]
  otpu_apply $sp $s_cfg {CONFIG.FREQ_HZ CONFIG.CLK_DOMAIN CONFIG.PHASE}
  otpu_apply $mp $m_cfg {CONFIG.FREQ_HZ CONFIG.CLK_DOMAIN CONFIG.PHASE}
  set_property CONFIG.FREQ_HZ $core_hz $sp
  set_property CONFIG.FREQ_HZ 450000000 $mp
  connect_bd_intf_net $sp [get_bd_intf_pins sc_c$p/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins sc_c$p/M00_AXI] $mp
  lappend busn S_N$p
  lappend busm M_H$p
}
set_property CONFIG.ASSOCIATED_BUSIF [join $busn :] [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_BUSIF [join $busm :] [get_bd_ports hbm_clk]
set_property CONFIG.ASSOCIATED_RESET {aresetn} [get_bd_ports core_clk]
# addresses: channel p's pseudo-channel at p x 512 MiB from its own slave
for {set p 0} {$p < $nport} {incr p} {
  set base [format 0x%09llX [expr {$p << 29}]]   ;# (ll: past 4 GiB)
  assign_bd_address -offset $base -range 0x20000000 -target_address_space [get_bd_addr_spaces S_N$p] [get_bd_addr_segs M_H$p/Reg] -force
}
validate_bd_design
# fail closed, after validation: every channel SmartConnect takes >= 32 transactions in flight
# each way and 8-beat (512-byte) bursts on its 512-bit side. cl_axi_sc_1x1's SmartConnect
# accepts 32 a direction (its S00 / M00), its AXI4 side bursts to 64 beats: with otpu_native_axi's
# 8-beat read bursts that is 32 x 512 B = 16 KB in flight a channel, against 14.4 GB/s x 280 ns
# = 4 KB (AWS cl_mem_perf's measured HBM read latency); a single-beat burst counts as one
for {set p 0} {$p < $nport} {incr p} {
  set pin [get_bd_intf_pins sc_c$p/S00_AXI]
  foreach {k floor} {CONFIG.NUM_READ_OUTSTANDING 32 CONFIG.NUM_WRITE_OUTSTANDING 32 CONFIG.MAX_BURST_LENGTH 8} {
    set v [get_property $k $pin]
    if {$p == 0} { puts "OTPU_HBM_SC sc_c0/S00_AXI $k $v" }
    if {![string is integer -strict $v] || $v < $floor} {
      error "openTPU SmartConnect: sc_c$p/S00_AXI $k is $v, the channel needs >= $floor"
    }
  }
}
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
puts "OTPU_HBM_SC nport=$nport core_hz=$core_hz wrapper=[lindex $wrappers 0]"
# out-of-context synthesis of the block design, at bounded concurrency
create_ip_run $bd
set jobs [get_runs -filter {IS_SYNTHESIS && NAME != synth_1}]
launch_runs $jobs -jobs 4
foreach r $jobs { wait_on_run $r; if {[get_property PROGRESS $r] ne "100%"} {error "openTPU SmartConnect: OOC synthesis failed: $r"} }
set fp [open $ipdir/ip_done.txt w]; puts $fp "IP_DONE nport=$nport core_hz=$core_hz"; close $fp
puts "OTPU_HBM_SC_DONE"
