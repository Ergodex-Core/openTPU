set run [file normalize [lindex $argv 0]]
cd $run
# The core clock (MHz; the TMEM clock is twice it): the target's, written by build.py as
# core_clock.tcl (set otpu_core_mhz <MHz>); without the file Alex's 200 / 400 MHz.
set otpu_core_mhz 200.0
if {[file exists $run/core_clock.tcl]} {source $run/core_clock.tcl}
set core_freq [format %.3f $otpu_core_mhz]
set tmem_freq [format %.3f [expr {2.0 * $otpu_core_mhz}]]
puts "CORE_CLOCK_REQUESTED core=$core_freq tmem=$tmem_freq"
create_project clocks $run/ip_project -part xcvu47p-fsvh2892-2-e -force
set_param general.maxThreads 4
foreach {name freq} {f2_core_clk 250.000 f2_hbm_clk 100.000} {
  create_ip -name clk_wiz -vendor xilinx.com -library ip -module_name $name
  set_property -dict [list CONFIG.PRIM_IN_FREQ $freq CONFIG.PRIM_SOURCE {No_buffer} CONFIG.USE_RESET {true} CONFIG.RESET_TYPE {ACTIVE_HIGH} CONFIG.USE_LOCKED {true}] [get_ips $name]
}
set_property -dict [list CONFIG.NUM_OUT_CLKS {2} CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $core_freq CONFIG.CLKOUT2_USED {true} CONFIG.CLKOUT2_REQUESTED_OUT_FREQ $tmem_freq] [get_ips f2_core_clk]
# The link stage checks the derived clocks against CORE_KHZ to 0.001 ns (build_f2.tcl), so the
# MMCM must hit the request exactly: report its settings and stop here if it does not.
set ip [get_ips f2_core_clk]
set m [get_property -quiet CONFIG.MMCM_CLKFBOUT_MULT_F $ip]
set d [get_property -quiet CONFIG.MMCM_DIVCLK_DIVIDE $ip]
set o1 [get_property -quiet CONFIG.MMCM_CLKOUT0_DIVIDE_F $ip]
set o2 [get_property -quiet CONFIG.MMCM_CLKOUT1_DIVIDE $ip]
if {$m ne "" && $d ne "" && $o1 ne "" && $o2 ne ""} {
  set a1 [expr {250.0 * $m / $d / $o1}]
  set a2 [expr {250.0 * $m / $d / $o2}]
  puts "CORE_CLOCK_ACTUAL core=$a1 tmem=$a2 (M=$m D=$d O=$o1/$o2 VCO=[expr {250.0 * $m / $d}])"
  if {abs($a1 - $otpu_core_mhz) > 0.0005 || abs($a2 - 2.0 * $otpu_core_mhz) > 0.001} {
    error "clk_wiz cannot produce $otpu_core_mhz / [expr {2.0 * $otpu_core_mhz}] MHz exactly: $a1 / $a2"
  }
} else {
  puts "CORE_CLOCK_ACTUAL unknown (clk_wiz MMCM_* properties not exposed); the link stage checks the periods"
}
set_property CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {450.000} [get_ips f2_hbm_clk]
generate_target all [get_ips]
foreach ip [get_ips] {create_ip_run [get_files [get_property IP_FILE $ip]]}
set jobs [get_runs -filter {IS_SYNTHESIS && NAME != synth_1}]
launch_runs $jobs -jobs 2
foreach r $jobs {wait_on_run $r; if {[get_property PROGRESS $r] ne "100%"} {error "Clock synthesis failed"}}
set fp [open ip_done.txt w];puts $fp IP_DONE;close $fp
