# v4 P3 (docs/v4/proposal-3-eight-lanes-packed.md section 7, milestone 0): out-of-context synthesis,
# placement and routing of the production MXU (otpu_mxu_lanes: C lock-step lanes, SPACK packed PEs)
# on the F2's part, with the DSP48E2 cells (rtl/tech/xilinxus). The ports get input / output
# delays (IODELAY ns of the period outside), so they are timed but not the point: the reg-to-reg
# group is the lane's own timing. Vivado 2025.2 on the m8a:
#   vivado -mode batch -nojournal -source rtl/exp/p3_ooc/ooc_mxu.tcl -tclargs <out> <part> \
#     <period_ns> [GENERIC=VALUE ...]
# e.g. xcvu47p-fsvh2892-2-e 5.0 C=1 SPACK=1 E=1. Prints OTPU_* summary lines; reports go to <out>.
set out    [file normalize [lindex $argv 0]]
set part   [lindex $argv 1]
set period [lindex $argv 2]
set gens   [lrange $argv 3 end]
set root   [file normalize [file join [file dirname [info script]] ../../..]]
set iod    [expr {$period * 0.4}]
file mkdir $out
set_param general.maxThreads 16
create_project -in_memory -part $part
set_property source_mgmt_mode None [current_project]
read_verilog -sv [list $root/rtl/vpu/otpu_fp.sv $root/rtl/vpu/otpu_fpipe.sv $root/rtl/top/otpu_pkg.sv \
  $root/rtl/mxu/otpu_mxu.sv $root/rtl/tech/xilinxus/otpu_tech_dsp.sv \
  $root/rtl/tech/generic/otpu_tech_ram.sv]
# the target's lane: D 128, MCOLS 4, ACT_ROWS 16, FIFO_DEPTH 1024, ULANES 8, systolic, CL 16
set gargs [list -generic D=128 -generic MCOLS=4 -generic ROWS=16 -generic DEPTH=1024 \
  -generic LANES=8 -generic IMPL=2 -generic CL=16]
foreach g $gens { lappend gargs -generic $g }
set t0 [clock seconds]
synth_design -top otpu_mxu_lanes -part $part -mode out_of_context -flatten_hierarchy rebuilt \
  -verilog_define SYNTHESIS {*}$gargs
puts "OTPU_SYNTH_S [expr {[clock seconds] - $t0}]"
create_clock -period $period -name clk [get_ports clk]
set ins [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_input_delay -clock clk $iod $ins
set_output_delay -clock clk $iod [get_ports -filter {DIRECTION == OUT}]
report_utilization -file $out/util_synth.txt
set t0 [clock seconds]
opt_design
place_design
puts "OTPU_PLACED_WNS [get_property SLACK [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]]"
phys_opt_design
route_design
puts "OTPU_IMPL_S [expr {[clock seconds] - $t0}]"
write_checkpoint -force $out/routed.dcp
report_utilization -file $out/util.txt
report_utilization -hierarchical -hierarchical_depth 6 -file $out/util_hier.txt
report_timing_summary -max_paths 20 -file $out/timing_summary.txt
report_methodology -file $out/methodology.txt
# the groups: reg-to-reg (the lane), and the ports' (with the IODELAY budget outside)
set regs [all_registers]
set rr [lindex [get_timing_paths -quiet -from $regs -to $regs -setup -max_paths 1 -nworst 1] 0]
set rh [lindex [get_timing_paths -quiet -from $regs -to $regs -hold -max_paths 1 -nworst 1] 0]
if {[llength $rr]} {
  puts "OTPU_R2R setup [get_property SLACK $rr] levels [get_property LOGIC_LEVELS $rr] [get_property STARTPOINT_PIN $rr] -> [get_property ENDPOINT_PIN $rr] | hold [get_property SLACK $rh]"
  report_timing -from $regs -to $regs -max_paths 30 -nworst 1 -file $out/timing_r2r.txt
}
foreach {tag f t} [list in2reg $ins $regs reg2out $regs [get_ports -filter {DIRECTION == OUT}]] {
  set q [lindex [get_timing_paths -quiet -from $f -to $t -setup -max_paths 1 -nworst 1] 0]
  if {[llength $q]} { puts "OTPU_$tag setup [get_property SLACK $q] [get_property STARTPOINT_PIN $q] -> [get_property ENDPOINT_PIN $q]" }
}
# the packed column's paths: the chain ends' split into the column ends (A / D), the weight
# pick into the PEs' B, and the activations into their A / D
proc worst {tag pins} {
  if {![llength $pins]} { puts "OTPU_GRP $tag none"; return }
  set q [lindex [get_timing_paths -quiet -to $pins -setup -max_paths 1 -nworst 1] 0]
  if {[llength $q]} {
    puts "OTPU_GRP $tag setup [get_property SLACK $q] levels [get_property LOGIC_LEVELS $q] [get_property STARTPOINT_PIN $q] -> [get_property ENDPOINT_PIN $q]"
  } else { puts "OTPU_GRP $tag unconstrained" }
}
set ce [get_cells -quiet -hierarchical -filter {NAME =~ *u_ce/g_m*u_dsp && REF_NAME == DSP48E2}]
puts "OTPU_CE_DSPS [llength $ce]"
worst chain_end_to_ce_AD [get_pins -quiet -of_objects $ce -filter {REF_PIN_NAME =~ A[*] || REF_PIN_NAME =~ D[*]}]
worst shift_to_ce_B [get_pins -quiet -of_objects $ce -filter {REF_PIN_NAME =~ B[*]}]
set pk [get_cells -quiet -hierarchical -filter {NAME =~ *g_pkc*u_pe/u_dsp && REF_NAME == DSP48E2}]
set up [get_cells -quiet -hierarchical -filter {NAME =~ *g_ask*u_pe/u_dsp && REF_NAME == DSP48E2}]
puts "OTPU_PE_DSPS packed [llength $pk] unpacked [llength $up]"
set pe [concat $pk $up]
worst pe_B [get_pins -quiet -of_objects $pe -filter {REF_PIN_NAME =~ B[*]}]
worst pe_AD [get_pins -quiet -of_objects $pe -filter {REF_PIN_NAME =~ A[*] || REF_PIN_NAME =~ D[*]}]
worst pe_INMODE [get_pins -quiet -of_objects $pe -filter {REF_PIN_NAME =~ INMODE[*]}]
set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
set h [lindex [get_timing_paths -hold -max_paths 1 -nworst 1] 0]
puts "OTPU_WNS [get_property SLACK $p] OTPU_WHS [get_property SLACK $h]"
puts "OTPU_DONE"
