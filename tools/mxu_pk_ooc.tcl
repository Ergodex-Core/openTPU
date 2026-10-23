# Out-of-context Vivado run of one MXU physical column on the F2's part (docs/mxu_packing.md 5):
# rtl/mxu/otpu_mxu_pk_col.sv packed (PACK 1: 128 otpu_pe_pk on DSP48E2s, two products each, the
# lane unpack, two otpu_mxce column ends: 146 DSPs, two logical columns) or unpacked (PACK 0:
# the existing otpu_pe and one column end, 137 DSPs, one logical column), WREG 1 (a column past
# the first: the weights in the DSPs' A / D registers), at one clock period. Synthesis, opt,
# place, phys_opt and route, as tools/tourney/remote.py's ooc_tcl(); the summary lines
# (OTPU_WNS, OTPU_WHS, OTPU_PERIOD) and util.txt are what tools/tourney/synth.parse_ooc() reads,
# plus OTPU_DSP / OTPU_LUT / OTPU_FF from the routed utilization. Non-project mode, on a build
# host (never on the Mac; tools/mxu_pk_ooc.sh runs the four combinations):
#
#   vivado -mode batch -nojournal -log <outdir>/vivado.log -source tools/mxu_pk_ooc.tcl \
#     -tclargs <outdir> <PACK 0|1> <period_ns> [GENERIC=VALUE ...]
#
# Reports land in <outdir>: util.txt (report_utilization), util_hier.txt, timing.txt (the 30
# worst setup paths), timing_summary.txt, routed.dcp. The part is the F2's (targets/f2-hbm.toml);
# MXU_PK_PART in the environment overrides it (xcku5p-ffvb676-2-e: rtl/exp/mx_mxu's stand-in).
set out    [file normalize [lindex $argv 0]]
set pack   [lindex $argv 1]
set period [lindex $argv 2]
set gens   [lrange $argv 3 end]
set root   [file normalize [file join [file dirname [info script]] ..]]
proc env_or {name dflt} { if {[info exists ::env($name)]} { return $::env($name) }; return $dflt }
set part [env_or MXU_PK_PART xcvu47p-fsvh2892-2-e]
file mkdir $out

create_project -in_memory -part $part
set_property source_mgmt_mode None [current_project]
# otpu_mxu.sv for otpu_skew (the operand skews the column shares with the array)
read_verilog -sv [list $root/rtl/vpu/otpu_fp.sv $root/rtl/vpu/otpu_fpipe.sv \
  $root/rtl/top/otpu_pkg.sv $root/rtl/mxu/otpu_mxu.sv $root/rtl/tech/xilinxus/otpu_tech_dsp.sv \
  $root/rtl/mxu/otpu_mxu_pk_col.sv]
set gargs [list -generic PACK=$pack -generic WREG=1 -generic D=128 -generic CLS=16 -generic NCH=8]
foreach g $gens { lappend gargs -generic $g }
set t0 [clock seconds]
synth_design -top otpu_mxu_pk_col -part $part -mode out_of_context -flatten_hierarchy rebuilt \
  -verilog_define SYNTHESIS {*}$gargs
puts "OTPU_SYNTH_S [expr {[clock seconds] - $t0}]"
create_clock -period $period -name clk [get_ports clk]
# the ports carry no timing of their own: every operand enters a register (otpu_mxu's S0 is the
# ACT RAM's read register and the chunk register), the accumulators leave one
report_utilization -file $out/util_synth.txt
set t0 [clock seconds]
opt_design
place_design
phys_opt_design
route_design
puts "OTPU_IMPL_S [expr {[clock seconds] - $t0}]"
write_checkpoint -force $out/routed.dcp
report_utilization -file $out/util.txt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier.txt
report_timing_summary -max_paths 10 -file $out/timing_summary.txt
report_timing -max_paths 30 -nworst 1 -sort_by slack -file $out/timing.txt
set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
puts "OTPU_WNS [get_property SLACK $p]"
set h [lindex [get_timing_paths -hold -max_paths 1 -nworst 1] 0]
puts "OTPU_WHS [get_property SLACK $h]"
puts "OTPU_PERIOD $period"
puts "OTPU_PACK $pack"
puts "OTPU_DSP [llength [get_cells -hierarchical -filter {PRIMITIVE_SUBGROUP == DSP}]]"
puts "OTPU_LUT [llength [get_cells -hierarchical -filter {PRIMITIVE_SUBGROUP == LUT}]]"
puts "OTPU_FF [llength [get_cells -hierarchical -filter {PRIMITIVE_SUBGROUP == SDR}]]"
puts "OTPU_WORST setup [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p]"
puts "OTPU_DONE"
