# Post-synthesis CDC timing of the HBM build (boards/f2/design/cl_opentpu_hbm.sv; build.py puts
# this file in place of cl_timing_user.tcl for an HBM target). In openTPU's logic two clock groups
# meet: the core (u_clk: 200 MHz, and u_clk_f: 400 MHz, phase aligned, timed together) and the
# shell's clk_main_a0 (250 MHz: OCL, PCIS), through otpu_axil_cdc's held request / response words
# and synchronized levels (the OCL crossing, the board's LED and HBM-ready levels): bound each
# direction by the destination clock's period, datapath only (a word is read only after its
# request has gone through the destination's two synchronizer flip-flops). The HBM's AXI clock
# (u_hclk: 450 MHz) meets the others only inside the SmartConnect fabric (u_sc, Alex's burst200
# memory path: its clock converters carry their own scoped constraints) and the HDK's HBM wrapper
# (xpm_cdc), so no openTPU bound names it (the rounds 1-8 bridges' afifos and gray write counts,
# and their bus-skew and hclk bounds, are gone with otpu_f2_hbm_nch / _host). The clocks are not
# grouped asynchronously (that would override the IP's bounds and these).
proc otpu_required {objects description} {
  if {![llength $objects]} { error "openTPU CDC: missing $description" }
  return $objects
}
set c_core [otpu_required [get_clocks -of_objects [get_pins u_clk/O]] {core clock}]
set c_main [otpu_required [get_clocks clk_main_a0] {shell clock}]
set c_hbm  [otpu_required [get_clocks -of_objects [get_pins u_hclk/O]] {HBM AXI clock}]
if {[llength $c_core] != 1 || [llength $c_main] != 1 || [llength $c_hbm] != 1} {
  error "openTPU CDC: expected one core, shell and HBM AXI clock"
}
# the SmartConnect fabric must be in the netlist (its own CDC constraints time the HBM clock's
# crossings): fail closed if the block design is missing
otpu_required [get_cells -quiet u_sc] {SmartConnect fabric u_sc}
set t_core [get_property PERIOD $c_core]
set t_main [get_property PERIOD $c_main]
set t_hbm  [get_property PERIOD $c_hbm]
set_max_delay -datapath_only -from $c_main -to $c_core $t_core
set_max_delay -datapath_only -from $c_core -to $c_main $t_main
puts "OTPU_HBM_CDC core=$t_core main=$t_main hbm=$t_hbm (hbm crossings: SmartConnect's own)"
# Lock is asynchronous to the generated clocks: only its reset-control fanout is exempted
set_false_path -through [get_pins u_mmcm/LOCKED] -to [get_pins -hier -filter {REF_PIN_NAME == CLR || REF_PIN_NAME == PRE}]
set_false_path -through [get_pins u_hmmcm/LOCKED] -to [get_pins -hier -filter {REF_PIN_NAME == CLR || REF_PIN_NAME == PRE}]
set_false_path -from [get_ports rst_main_n]
