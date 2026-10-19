# Preserve the upstream pumped-TMEM timing contract. Only read-address/enable
# and read-data paths get the existing multicycles; writes and phase remain one fast cycle.
source $run/otpu_clk2x.tcl
if {![info exists c2x_pb] || [llength $c2x_pb] == 0 || ![info exists c2x_rp] || ![llength $c2x_rp] || ![llength $c2x_qp]} {
  error "Pumped TMEM timing constraints did not match the implemented hierarchy"
}
# The outputs come from the same MMCM with zero phase offset; balance their roots.
set core_clock_nets [get_nets -of_objects [get_pins {WRAPPER/CL/u_core_clk/clk_out1 WRAPPER/CL/u_core_clk/clk_out2}]]
if {[llength $core_clock_nets] != 2} {error "Expected two related core/TMEM clock nets"}
set_property CLOCK_DELAY_GROUP OTPU_CORE_TMEM $core_clock_nets
report_exceptions -file $run/tmem_exceptions.rpt
