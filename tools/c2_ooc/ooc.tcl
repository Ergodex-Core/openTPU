# Frozen C2-only Vivado 2025.2 flow. Arguments: snapshot synth|route threads resume.
if {$argc != 4} { error "usage: ooc.tcl snapshot synth|route threads resume" }
set root [file normalize [lindex $argv 0]]
set stage [lindex $argv 1]
set threads [lindex $argv 2]
set resume [lindex $argv 3]
if {$stage ni {synth route}} { error "invalid stage" }
if {![string is integer -strict $threads] || $threads < 1 || $threads > 8} { error "threads must be 1..8" }
if {[version -short] ne "2025.2"} { error "Vivado 2025.2 required" }
set_param general.maxThreads $threads
set part xcvu47p-fsvh2892-2-e
set top otpu_c2_mxu
file mkdir $root/checkpoints $root/reports/synth $root/reports/route

proc path_slack {kind args} {
    set paths [get_timing_paths -quiet -delay_type $kind -max_paths 1 -nworst 1 {*}$args]
    if {[llength $paths] == 0} { return "NA" }
    return [get_property SLACK [lindex $paths 0]]
}
proc reports {root stage} {
    set out $root/reports/$stage
    report_utilization -file $out/utilization.rpt
    report_utilization -hierarchical -hierarchical_depth 8 -file $out/utilization_hierarchical.rpt
    report_timing_summary -report_unconstrained -check_timing_verbose -max_paths 20 -file $out/timing_summary.rpt
    report_timing -delay_type max -max_paths 20 -nworst 1 -file $out/setup_paths.rpt
    report_timing -delay_type min -max_paths 20 -nworst 1 -file $out/hold_paths.rpt
    set q [all_registers -output_pins]
    set d [all_registers -data_pins]
    if {![llength $q] || ![llength $d]} { error "C2 has no register timing endpoints" }
    report_timing -from $q -to $d -delay_type max -max_paths 20 -nworst 1 -file $out/register_setup_paths.rpt
    report_timing -from $q -to $d -delay_type min -max_paths 20 -nworst 1 -file $out/register_hold_paths.rpt
    report_clocks -file $out/clocks.rpt
    report_clock_interaction -file $out/clock_interaction.rpt
    report_drc -file $out/drc.rpt
    report_methodology -file $out/methodology.rpt
    redirect -file $out/check_timing.rpt { check_timing -verbose }
    set setup [path_slack max]
    set hold [path_slack min]
    set rsetup [path_slack max -from $q -to $d]
    set rhold [path_slack min -from $q -to $d]
    set target_met 0
    if {$setup ne "NA" && $hold ne "NA" && $setup >= 0 && $hold >= 0} { set target_met 1 }
    set f [open $out/metrics.tsv w]
    foreach {key value} [list stage $stage period_ns 5.000 setup_slack_ns $setup hold_slack_ns $hold \
        register_setup_slack_ns $rsetup register_hold_slack_ns $rhold target_met $target_met \
        dsp48e2 [llength [get_cells -hierarchical -filter {REF_NAME == DSP48E2}]] \
        ramb36 [llength [get_cells -hierarchical -filter {REF_NAME =~ RAMB36*}]] \
        ramb18 [llength [get_cells -hierarchical -filter {REF_NAME =~ RAMB18*}]] \
        uram [llength [get_cells -hierarchical -filter {REF_NAME =~ URAM*}]] \
        lut [llength [get_cells -hierarchical -filter {REF_NAME =~ LUT*}]] \
        ff [llength [get_cells -hierarchical -filter {REF_NAME =~ FD*}]]] {
        puts $f "$key\t$value"
    }
    close $f
    puts "OTPU_C2_METRICS stage=$stage WNS=$setup WHS=$hold REG_WNS=$rsetup REG_WHS=$rhold target_met=$target_met"
}

if {$resume} {
    if {$stage ne "route"} { error "only route can resume synthesis" }
    open_checkpoint $root/checkpoints/post_synth.dcp
} else {
    set critical_before [get_msg_config -count -severity {CRITICAL WARNING}]
    create_project -in_memory -part $part
    set_property source_mgmt_mode None [current_project]
    set sources [list rtl/vpu/otpu_fp.sv rtl/vpu/otpu_fpipe.sv rtl/top/otpu_pkg.sv \
        rtl/tech/xilinxus/otpu_tech_dsp.sv rtl/mxu/otpu_mxu.sv rtl/mxu/otpu_c2_mxu.sv]
    foreach source $sources { read_verilog -sv $root/src/$source }
    read_xdc $root/flow/constraints.xdc
    set t0 [clock seconds]
    synth_design -top $top -part $part -mode out_of_context -flatten_hierarchy rebuilt \
        -verilog_define SYNTHESIS -generic D=128 -generic C=4 -generic DEPTH=256 -generic RF=8
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {
        error "unresolved black boxes in synthesized C2"
    }
    if {[llength [get_clocks c2_clk]] != 1 || [get_property PERIOD [get_clocks c2_clk]] != 5.0} {
        error "missing or incorrect C2 200MHz constraint"
    }
    if {[get_msg_config -count -severity {CRITICAL WARNING}] > $critical_before} {
        error "C2 synthesis or timing constraints produced a critical warning; checkpoint is not qualified"
    }
    write_checkpoint -force $root/checkpoints/post_synth.dcp
    reports $root synth
    puts "OTPU_C2_SYNTH_SECONDS [expr {[clock seconds]-$t0}]"
    puts "OTPU_C2_SYNTH_DONE"
}
if {$stage eq "route"} {
    set t0 [clock seconds]
    opt_design
    place_design
    phys_opt_design
    route_design
    write_checkpoint -force $root/checkpoints/post_route.dcp
    reports $root route
    report_route_status -file $root/reports/route/route_status.rpt
    puts "OTPU_C2_ROUTE_SECONDS [expr {[clock seconds]-$t0}]"
    puts "OTPU_C2_ROUTE_DONE"
}
