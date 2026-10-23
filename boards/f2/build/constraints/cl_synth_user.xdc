# clk_main_a0 = 250 MHz comes from generated_cl_clocks_aws.xdc.
# Vivado derives 100 MHz and 200 MHz from u_mmcm. Keep their phase relationship.
# Only reset assertion and MMCM lock enter asynchronous reset pins.
set_false_path -from [get_ports rst_main_n]
