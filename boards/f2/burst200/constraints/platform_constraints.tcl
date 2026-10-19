# Match AWS HLx opt_design_pre.tcl's disabled-DDR handling. The reference
# cl_pins.xdc leaves clock pair 0 to the DDR IP, which is absent in this port.
set ddr_clk_ports [get_ports {M_CLK_DP[0] M_CLK_DN[0]}]
if {[llength $ddr_clk_ports] != 2} {error "Missing disabled-DDR clock ports"}
set_property IOSTANDARD SSTL12_DCI $ddr_clk_ports
# AWS explicitly permits CL clock-wizard MMCMs on the shell's HBM reference.
set hbm_ref_net [get_nets -of_objects [get_pins WRAPPER/CL/clk_hbm_ref]]
if {[llength $hbm_ref_net] != 1} {error "Expected one HBM reference net"}
set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN $hbm_ref_net
