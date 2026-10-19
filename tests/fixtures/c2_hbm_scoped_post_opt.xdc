# Exact C2 command subset from Vivado 2025.2 post-opt timing export.
# Full export SHA256 4a936f2645fad49648395d70903331ceb35a2fff79f94eaca46db887ff529854
current_instance WRAPPER/CL
# id:72
set_max_delay -datapath_only -from [get_clocks clk_main_a0] -to [get_clocks -of_objects [get_pins u_clk/O]] 5.000
# id:73
set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins u_clk/O]] -to [get_clocks clk_main_a0] 4.000
# id:74
set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins u_clk/O]] -to [get_clocks -of_objects [get_pins u_hclk/O]] 2.222
# id:75
set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins u_hclk/O]] -to [get_clocks -of_objects [get_pins u_clk/O]] 5.000
# id:76
set_max_delay -datapath_only -from [get_clocks clk_main_a0] -to [get_clocks -of_objects [get_pins u_hclk/O]] 2.222
# id:77
set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins u_hclk/O]] -to [get_clocks clk_main_a0] 4.000
current_instance -quiet
