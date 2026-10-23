set run [file normalize [lindex $argv 0]]
cd $run
create_project clocks $run/ip_project -part xcvu47p-fsvh2892-2-e -force
set_param general.maxThreads 4
foreach {name freq} {f2_core_clk 250.000 f2_hbm_clk 100.000} {
  create_ip -name clk_wiz -vendor xilinx.com -library ip -module_name $name
  set_property -dict [list CONFIG.PRIM_IN_FREQ $freq CONFIG.PRIM_SOURCE {No_buffer} CONFIG.USE_RESET {true} CONFIG.RESET_TYPE {ACTIVE_HIGH} CONFIG.USE_LOCKED {true}] [get_ips $name]
}
set_property -dict [list CONFIG.NUM_OUT_CLKS {2} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {200.000} CONFIG.CLKOUT2_USED {true} CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {400.000}] [get_ips f2_core_clk]
set_property CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {450.000} [get_ips f2_hbm_clk]
generate_target all [get_ips]
foreach ip [get_ips] {create_ip_run [get_files [get_property IP_FILE $ip]]}
set jobs [get_runs -filter {IS_SYNTHESIS && NAME != synth_1}]
launch_runs $jobs -jobs 2
foreach r $jobs {wait_on_run $r; if {[get_property PROGRESS $r] ne "100%"} {error "Clock synthesis failed"}}
set fp [open ip_done.txt w];puts $fp IP_DONE;close $fp
