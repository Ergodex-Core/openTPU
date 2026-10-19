# Explicit standalone interface budget. These are NOT shell/HBM constraints.
create_clock -name c2_clk -period 5.000 [get_ports clk]
set_clock_uncertainty -setup 0.100 [get_clocks c2_clk]
set_clock_uncertainty -hold 0.050 [get_clocks c2_clk]
set data_inputs [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_input_delay -clock c2_clk -max 0.500 $data_inputs
set_input_delay -clock c2_clk -min 0.000 $data_inputs
set_output_delay -clock c2_clk -max 0.500 [all_outputs]
set_output_delay -clock c2_clk -min 0.000 [all_outputs]
# No false paths, multicycles, clock groups, or DRC severity suppression.
