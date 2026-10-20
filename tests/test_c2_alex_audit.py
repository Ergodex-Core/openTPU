"""Synthesis audit rejection tests with mocked Vivado objects; no FPGA evidence."""
from pathlib import Path
import shutil
import subprocess

import pytest


AUDIT = Path(__file__).resolve().parents[1] / 'boards/c2_f2_alex/audit_c2_alex.tcl'
MOCK = r'''
lassign $argv script run fault
source $script
set cells {}; set clocks {}; set pins {}; set pin_clocks {}; set members {}
proc add_cell {name ref {opaque 0}} {
    dict set ::cells $name [dict create NAME $name REF_NAME ${ref}_synth ORIG_REF_NAME $ref IS_BLACKBOX $opaque]
}
foreach {name ref} {
 u_board otpu_c2_hbm_kernel u_board/u_mxu otpu_c2_mxu
 u_mem f2_mem_wrapper u_hbm cl_mem_hbm_wrapper u_ocl_cdc f2_ocl_cdc
 u_pcis_pipe f2_pcis_pipe u_core_clk f2_core_clk u_hbm_clk f2_hbm_clk
} {add_cell $name $ref}
for {set p 0} {$p < 8} {incr p} {add_cell u_native$p otpu_native_axi}
foreach {name period} {clk_main_a0 4.000 clk_hbm_ref 10.000 core 5.000 unused400 2.500 hbm 2.222} {
    dict set clocks $name [dict create NAME $name PERIOD $period IS_VIRTUAL 0 IS_PROPAGATED 1]
}
dict set clocks core SOURCE_PINS u_core_clk/inst/mmcm/CLKOUT0
dict set clocks unused400 SOURCE_PINS u_core_clk/inst/mmcm/CLKOUT1
dict set clocks hbm SOURCE_PINS u_hbm_clk/inst/mmcm/CLKOUT0
foreach {pin clock} {u_core_clk/clk_out1 core u_core_clk/clk_out2 unused400 u_hbm_clk/clk_out1 hbm u_mem/core_clk core u_mem/hbm_clk hbm u_mem/host_clk clk_main_a0} {
    dict set pins $pin [dict create NAME $pin]
    dict set pin_clocks $pin $clock
}
set ifaces {S_HOST}
for {set p 0} {$p < 8} {incr p} {lappend ifaces S_N$p M_H$p}
set mem_pins {}
foreach iface $ifaces {
    foreach signal {awvalid awready wvalid wready bvalid bready arvalid arready rvalid rready} {
        set name u_mem/${iface}_$signal
        lappend mem_pins $name; dict set pins $name [dict create NAME $name]
    }
    set aw [expr {$iface eq "S_HOST" ? 64 : 34}]
    set dw [expr {[string match M_H* $iface] ? 256 : 512}]
    foreach {signal width} [list awaddr $aw araddr $aw wdata $dw rdata $dw wstrb [expr {$dw/8}]] {
        for {set bit 0} {$bit < $width} {incr bit} {
            set name [format {u_mem/%s_%s[%d]} $iface $signal $bit]
            lappend mem_pins $name; dict set pins $name [dict create NAME $name]
        }
    }
}
dict set members hbm {u_mem/f2_mem_i/memory_sc/cdc_reg u_hbm/vendor_reg u_rst_hbm/sync_reg}
dict set members core {u_board/data_reg}
dict set members unused400 {}
switch $fault {
    ref_without_orig {
        foreach name [dict keys $cells] {
            dict set cells $name REF_NAME [dict get $cells $name ORIG_REF_NAME]
            dict unset cells $name ORIG_REF_NAME
        }
    }
    ref_empty_orig {
        foreach name [dict keys $cells] {
            dict set cells $name REF_NAME [dict get $cells $name ORIG_REF_NAME]
            dict set cells $name ORIG_REF_NAME {}
        }
    }
    pruned_unused_pin {dict unset pins u_core_clk/clk_out2;dict unset clocks unused400}
    retained_unused_clock {dict unset pins u_core_clk/clk_out2}
    pruned_pin_used_400 {dict unset pins u_core_clk/clk_out2;dict lappend members unused400 u_board/illegal_reg}
    pruned_pin_unknown_source {dict unset pins u_core_clk/clk_out2;dict set clocks unused400 SOURCE_PINS unrelated/clock}
    pruned_pin_no_source {dict unset pins u_core_clk/clk_out2;dict unset clocks unused400 SOURCE_PINS}
    wrong_kernel - audit_error_and_checkpoint_error {dict set cells u_board ORIG_REF_NAME otpu_board}
    missing_ref_property {dict unset cells u_board REF_NAME}
    missing_blackbox_property {dict unset cells u_board IS_BLACKBOX}
    empty_ref_property {dict set cells u_board REF_NAME {}}
    empty_blackbox_property {dict set cells u_board IS_BLACKBOX {}}
    invalid_blackbox_property {dict set cells u_board IS_BLACKBOX unknown}
    opaque_kernel {dict set cells u_board IS_BLACKBOX 1}
    opaque_mxu {dict set cells u_board/u_mxu IS_BLACKBOX 1}
    missing_native {dict unset cells u_native7}
    extra_native {add_cell u_native8 otpu_native_axi}
    wrong_native {dict set cells u_native0 ORIG_REF_NAME otpu_f2_hbm_nch}
    wrong_vendor {dict set cells u_mem ORIG_REF_NAME wrong_memory}
    vendor_opaque {add_cell u_mem/f2_mem_i/memory_sc vendor_smartconnect 1}
    unknown_opaque {add_cell custom_bad unknown 1}
    lookalike_opaque {add_cell u_mem_extra/custom bad 1}
    opaque_clock {
        add_cell u_core_clk/inst clock_model 1
        dict unset clocks core; dict unset clocks unused400
    }
    missing_clock {dict unset clocks core}
    wrong_clock {dict set clocks core PERIOD 10.000}
    wrong_hbm_clock {dict set clocks hbm PERIOD 2.500}
    wrong_host_clock {dict set clocks clk_main_a0 PERIOD 5.000}
    wrong_reference_clock {dict set clocks clk_hbm_ref PERIOD 8.000}
    disconnected_core {dict set pin_clocks u_mem/core_clk clk_main_a0}
    disconnected_hbm {dict set pin_clocks u_mem/hbm_clk core}
    disconnected_host {dict set pin_clocks u_mem/host_clk core}
    missing_clock_pin {dict unset pins u_core_clk/clk_out1}
    custom_hclk {dict lappend members hbm u_board/custom_reg}
    lookalike_hclk {dict lappend members hbm u_hbm_extra/custom_reg}
    used_400 {dict lappend members unused400 u_board/tmem_reg}
    pruned_boundary_interface {
        set remaining {}; foreach n $mem_pins {if {![string match u_mem/S_N7_* $n]} {lappend remaining $n}}
        set mem_pins $remaining
    }
    extra_interface {lappend mem_pins u_mem/S_N8_arvalid;dict set pins u_mem/S_N8_arvalid {NAME u_mem/S_N8_arvalid}}
    pruned_boundary_handshake {set at [lsearch -exact $mem_pins u_mem/M_H7_rready];set mem_pins [lreplace $mem_pins $at $at]}
    indexed_handshakes {
        set updated {}
        foreach name $mem_pins {
            if {[regexp {_(awvalid|awready|wvalid|wready|bvalid|bready|arvalid|arready|rvalid|rready)$} $name]} {
                set name [format {%s[0]} $name]
                dict set pins $name [dict create NAME $name]
            }
            lappend updated $name
        }
        set mem_pins $updated
    }
    duplicate_handshake {lappend mem_pins u_mem/S_HOST_awvalid}
    mixed_handshake {lappend mem_pins {u_mem/S_HOST_awvalid[0]}}
    duplicate_handshake_bit - handshake_extra_bit - handshake_nonzero_bit {
        set at [lsearch -exact $mem_pins u_mem/S_HOST_awvalid]
        set mem_pins [lreplace $mem_pins $at $at]
        if {$fault eq "duplicate_handshake_bit"} {
            lappend mem_pins {u_mem/S_HOST_awvalid[0]} {u_mem/S_HOST_awvalid[0]}
        } elseif {$fault eq "handshake_extra_bit"} {
            lappend mem_pins {u_mem/S_HOST_awvalid[0]} {u_mem/S_HOST_awvalid[1]}
        } else {lappend mem_pins {u_mem/S_HOST_awvalid[1]}}
    }
    pruned_boundary_bit {set at [lsearch -exact $mem_pins {u_mem/S_N7_awaddr[33]}];set mem_pins [lreplace $mem_pins $at $at]}
    duplicate_bit {lappend mem_pins {u_mem/M_H0_wdata[0]}}
    vector_scalar_duplicate {lappend mem_pins u_mem/S_HOST_awaddr}
}
proc version {args} {return [expr {$::fault eq "version" ? "2025.1" : "2025.2"}]}
proc current_instance {} {return [expr {$::fault eq "scope" ? "WRAPPER/CL" : ""}]}
proc list_property {object} {
    if {$::fault eq "no_cell_property_enumeration" && [dict exists $::cells $object]} {
        error "hierarchical property enumeration is forbidden: $object"
    }
    foreach table {cells clocks pins} {if {[dict exists [set ::$table] $object]} {return [dict keys [dict get [set ::$table] $object]]}}
    if {[string match */* $object]} {return NAME}
    error "unknown fixture object $object"
}
proc get_property {key objects} {
    set values {}
    foreach object $objects {
        if {$key eq "NAME"} {lappend values $object;continue}
        set found 0
        foreach table {cells clocks pins} {
            if {[dict exists [set ::$table] $object $key]} {
                lappend values [dict get [set ::$table] $object $key];set found 1;break
            }
        }
        if {!$found && $key in {REF_NAME ORIG_REF_NAME IS_BLACKBOX}} {
            # Vivado can return an empty value for an unavailable property.
            # The audit must reject required values without list_property.
            lappend values {};set found 1
        }
        if {!$found} {error "unknown fixture property $key $object"}
    }
    if {[llength $objects] == 1} {return [lindex $values 0]}
    return $values
}
proc get_cells {args} {
    set selected [lindex $args end]
    if {"-filter" in $args} {
        if {$selected eq "IS_BLACKBOX"} {
            set result {};dict for {name props} $::cells {if {[dict get $props IS_BLACKBOX]} {lappend result $name}}
            return $result
        }
        return [expr {$::fault eq "tmem_cell" ? "u_board/u_tmem" : ""}]
    }
    if {"-regexp" in $args} {
        set result {};foreach name [dict keys $::cells] {if {[regexp $selected $name]} {lappend result $name}}
        return $result
    }
    return [expr {[dict exists $::cells $selected] ? $selected : ""}]
}
proc get_pins {args} {
    if {"-of_objects" in $args} {return $::mem_pins}
    set name [lindex $args end]
    return [expr {[dict exists $::pins $name] ? $name : ""}]
}
proc get_clocks {args} {
    set name [lindex $args end]
    if {$name eq "*"} {return [dict keys $::clocks]}
    if {"-of_objects" in $args} {
        if {![dict exists $::pin_clocks $name]} {return {}}
        set name [dict get $::pin_clocks $name]
    }
    return [expr {[dict exists $::clocks $name] ? $name : ""}]
}
proc all_registers {args} {return [dict get $::members [lindex $args 1]]}
proc write_checkpoint {path} {
    if {$::fault in {checkpoint_error audit_error_and_checkpoint_error report_error_and_checkpoint_error}} {error "mock checkpoint write failure"}
    set ::checkpoint_written 1
    set f [open $path {WRONLY CREAT EXCL}]
    puts $f {MOCK_DIAGNOSTIC_NOT_QUALIFIED_NOT_FPGA_CONTENT};close $f
}
proc write_xdc {args} {
    set f [open [lindex $args end] w]
    puts $f {# mock export; never physical constraint evidence}
    switch $::fault {
        tmem_exception {puts $f {set_multicycle_path 2 -to [get_pins u_tmem/ADDRARDADDR]}}
        custom_multicycle {puts $f {set_multicycle_path 2 -from [get_clocks core] -to [get_clocks hbm]}}
        vendor_multicycle {puts $f {current_instance u_mem/f2_mem_i/memory_sc};puts $f {set_multicycle_path 2 -to [get_pins vendor_endpoint/D]}}
        incomplete_export {puts $f {set_false_path -to [get_pins}}
    }
    close $f
}
proc emit_report {args} {
    if {$::fault eq "checkpoint_mutates_graph" && [info exists ::checkpoint_written]} {error "report inspected graph after diagnostic serialization"}
    set path [lindex $args end]
    if {$::fault eq "missing_report" && [file tail $path] eq "cdc.rpt"} {return}
    if {$::fault in {report_error report_error_and_checkpoint_error} && [file tail $path] eq "cdc.rpt"} {error "mock CDC report failure"}
    set f [open $path w];puts $f {MOCK_REPORT_NOT_HARDWARE_EVIDENCE};close $f
}
foreach command {report_timing_summary report_drc report_cdc report_bus_skew report_clock_interaction report_exceptions report_methodology report_waivers report_utilization} {
    proc $command {args} {emit_report {*}$args}
}
proc get_drc_violations {args} {return {}}
proc get_methodology_violations {args} {return {}}
if {$fault eq "existing_directory"} {file mkdir [file join $run synth_audit]}
if {$fault eq "existing_checkpoint"} {
    set f [open [file join $run pre_audit.NOT_QUALIFIED.dcp] w]
    puts $f {PREEXISTING};close $f
}
if {[catch {::c2_alex::audit_synthesis $run} result]} {puts stderr $result;exit 1}
puts [list SUMMARY $result]
'''


FABRIC_MOCK = r'''
# Network fixture uses ordinary source/destination registers, not vendor proof.
set fabric_nets {};set fabric_leaf {};set fabric_anchor {};set fabric_net_ports {};set fabric_b_endpoints {}
proc fabric_add_pin {name direction} {dict set ::pins $name [dict create NAME $name DIRECTION $direction]}
proc fabric_build {{prefix {}}} {
    set root [expr {$prefix eq {} ? {} : "$prefix/"}]
    set ::fabric_nets {};set ::fabric_leaf {};set ::fabric_anchor {};set ::fabric_net_ports {};set ::fabric_b_endpoints {}
    set interfaces {S_HOST};for {set p 0} {$p<8} {incr p} {lappend interfaces S_N$p M_H$p}
    foreach iface $interfaces {
        set signals {awvalid awready wvalid wready bvalid bready arvalid arready rvalid rready}
        if {[string match S_N* $iface]} {set signals {awvalid awready wvalid wready bvalid arvalid arready rvalid}}
        foreach signal $signals {
            set fwd [expr {$signal in {awvalid wvalid arvalid bready rready}}]
            regexp {^(aw|ar|w|b|r)} $signal -> channel
            set sc ${root}u_mem/f2_mem_i/memory_sc/inst
            if {$iface eq {S_HOST}} {
                set ext ${root}u_pcis_pipe;set node $sc/s00_entry_pipeline/s00_mmu/inst/${channel}_sreg
                set anchor $ext/m_axi_$signal;set net ${root}pcis_$signal
                set direction [expr {$fwd ? {OUT} : {IN}}]
            } elseif {[regexp {S_N([0-7])} $iface -> p]} {
                set ext ${root}u_native$p;set si [format s%02d [expr {$p+1}]];set node $sc/${si}_entry_pipeline/${si}_mmu/inst/${channel}_sreg
                set net ${root}n${p}_$signal
                if {$signal eq {bvalid}} {set net ${root}u_mem/n${p}_bvalid}
                set anchor $net;set direction {}
            } else {
                regexp {M_H([0-7])} $iface -> p
                set ext ${root}u_hbm;set mi [format m%02d $p];set node $sc/${mi}_exit_pipeline/${mi}_exit/inst/${channel}_reg
                set anchor [format {%s/%s_axi_%s[%d]} $ext [expr {$fwd ? {i} : {o}}] $signal $p]
                set net [format {%sh_%s[%d]} $root $signal $p]
                set direction [expr {$fwd ? {IN} : {OUT}}];set fwd [expr {!$fwd}]
            }
            set src [expr {$fwd ? $ext : $node}];set dst [expr {$fwd ? $node : $ext}]
            set driver $src/${iface}_${signal}_reg/Q;set receiver $dst/${iface}_${signal}_reg/D
            add_cell [file dirname $driver] FDRE;add_cell [file dirname $receiver] FDRE
            fabric_add_pin $driver OUT;fabric_add_pin $receiver IN
            if {$direction ne {}} {fabric_add_pin $anchor $direction}
            dict set ::fabric_nets $net [dict create NAME $net]
            dict set ::fabric_anchor $anchor $net;dict set ::fabric_anchor $net $net
            dict set ::fabric_leaf $net [list $driver $receiver]
            if {[regexp {^S_N([0-7])$} $iface -> p] && $signal eq {bvalid}} {
                set relocated [::c2_alex::relocated_native_loads $root $p bvalid]
                foreach pin $relocated {fabric_add_pin $pin IN;add_cell [file dirname $pin] LUT6;dict set ::cells [file dirname $pin] REF_NAME LUT6}
                dict set ::fabric_leaf $net [linsert $relocated 0 $driver]
                set endpoints [list ${root}u_native$p/n_err_reg/D]
                for {set bit 0} {$bit<8} {incr bit} {lappend endpoints [format {%su_native%d/writes_reg[%d]/CE} $root $p $bit]}
                dict set ::fabric_b_endpoints $driver $endpoints
            }
        }
    }
    set ready ${root}n7_rready
    set driver ${root}u_mem/u_status_cdc_i_2/O
    set inverter [file dirname $driver]
    add_cell $inverter LUT1;dict set ::cells $inverter REF_NAME LUT1;dict set ::cells $inverter INIT {2'h1}
    fabric_add_pin $driver OUT
    set receivers {}
    for {set p 1} {$p<=8} {incr p} {
        set si [format s%02d $p]
        foreach channel {b r} {
            set pin ${root}u_mem/f2_mem_i/memory_sc/inst/${si}_entry_pipeline/${si}_mmu/inst/${channel}_sreg/ready_load/I0
            fabric_add_pin $pin IN;lappend receivers $pin
        }
    }
    dict set ::fabric_nets $ready [dict create NAME $ready]
    dict set ::fabric_anchor $ready $ready;dict set ::fabric_leaf $ready [linsert $receivers 0 $driver]
    set reset ${root}core_rst_reg;add_cell $reset FDRE;dict set ::cells $reset REF_NAME FDRE
    fabric_add_pin $reset/Q OUT;fabric_add_pin $reset/C IN;dict set ::pin_clocks $reset/C core
    fabric_add_pin $inverter/I0 IN
    set net ${root}core_rst;dict set ::fabric_nets $net [dict create NAME $net]
    dict set ::fabric_anchor $inverter/I0 $net;dict set ::fabric_anchor $net $net
    dict set ::fabric_leaf $net [list $reset/Q $inverter/I0]
}
fabric_build
# This fixture exposes no original u_mem clock pins, just actual register owners.
foreach pin {u_mem/core_clk u_mem/host_clk u_mem/hbm_clk} {dict unset pins $pin;dict unset pin_clocks $pin}
dict set members clk_main_a0 {u_mem/host_reg u_pcis_pipe/host_reg}
dict set members clk_hbm_ref {u_hbm/apb_reg}
dict set members core {u_board/data_reg u_mem/core_reg}
for {set p 0} {$p<8} {incr p} {dict lappend members core u_native$p/data_reg}
dict lappend members clk_main_a0 u_mem/f2_mem_i/memory_sc/inst/s00_entry_pipeline/state_reg
for {set p 0} {$p<8} {incr p} {
 dict lappend members core u_mem/f2_mem_i/memory_sc/inst/[format s%02d_entry_pipeline [expr {$p+1}]]/state_reg
 dict lappend members hbm u_mem/f2_mem_i/memory_sc/inst/[format m%02d_exit_pipeline $p]/state_reg
}
switch $fault {
 disconnected_core {dict set members core {u_board/data_reg};for {set p 0} {$p<8} {incr p} {dict lappend members core u_native$p/data_reg}}
 disconnected_hbm {dict set members hbm {u_hbm/vendor_reg u_rst_hbm/sync_reg}}
 disconnected_host {dict set members clk_main_a0 {u_pcis_pipe/host_reg}}
 ready_constant {dict set cells u_mem/u_status_cdc_i_2 INIT {2'h3}}
 ready_missing_channel {dict set fabric_leaf n7_rready [lrange [dict get $fabric_leaf n7_rready] 0 end-1]}
 ready_wrong_reset {fabric_add_pin u_board/other_reset/Q OUT;dict set fabric_leaf core_rst {u_board/other_reset/Q u_mem/u_status_cdc_i_2/I0}}
 ready_wrong_clock {dict set pin_clocks core_rst_reg/C clk_main_a0}
 ready_wrong_channel {
  set leaf [dict get $fabric_leaf n7_rready]
  set old [lindex $leaf 1];set new [string map {/b_sreg/ /w_sreg/} $old]
  fabric_add_pin $new IN;dict set fabric_leaf n7_rready [lreplace $leaf 1 1 $new]
 }
 b_missing_error {
  set driver [lindex [dict get $fabric_leaf u_mem/n0_bvalid] 0]
  dict set fabric_b_endpoints $driver [lrange [dict get $fabric_b_endpoints $driver] 1 end]
 }
 b_wrong_native {
  set driver [lindex [dict get $fabric_leaf u_mem/n0_bvalid] 0]
  dict lappend fabric_b_endpoints $driver u_native1/n_err_reg/D
 }
 sc_clock_swap {
  set h u_mem/f2_mem_i/memory_sc/inst/s00_entry_pipeline/state_reg
  set c u_mem/f2_mem_i/memory_sc/inst/s01_entry_pipeline/state_reg
  dict set members clk_main_a0 [string map [list $h $c] [dict get $members clk_main_a0]]
  dict set members core [string map [list $c $h] [dict get $members core]]
 }
 core_on_host {dict lappend members clk_main_a0 u_native0/wrong_reg}
 fabric_on_reference {dict lappend members clk_hbm_ref u_mem/wrong_reg}
 pcis_on_core {dict lappend members core u_pcis_pipe/wrong_reg}
 native_swap {
  set net n0_arvalid;set old [lindex [dict get $fabric_leaf $net] 0]
  set new [string map {u_native0/ u_native1/} $old]
  fabric_add_pin $new OUT;add_cell [file dirname $new] FDRE
  dict set fabric_leaf $net [lreplace [dict get $fabric_leaf $net] 0 0 $new]
 }
 sc_swap {
  set net n0_arvalid;set old [lindex [dict get $fabric_leaf $net] 1]
  set new [string map {s01_entry_pipeline s02_entry_pipeline} $old]
  fabric_add_pin $new IN;add_cell [file dirname $new] FDRE
  dict set fabric_leaf $net [lreplace [dict get $fabric_leaf $net] 1 1 $new]
 }
 channel_swap {
  set net n0_arvalid;set old [lindex [dict get $fabric_leaf $net] 1]
  set new [string map {/ar_sreg/ /aw_sreg/} $old]
  fabric_add_pin $new IN;add_cell [file dirname $new] FDRE
  dict set fabric_leaf $net [lreplace [dict get $fabric_leaf $net] 1 1 $new]
 }
 hbm_index {dict unset pins {u_hbm/i_axi_arvalid[0]}}
 missing_active_net {dict unset fabric_nets n0_arvalid}
 shorted_net {dict lappend fabric_leaf n0_arvalid {*}[dict get $fabric_leaf n1_arvalid]}
 constant_active {dict set cells [file dirname [lindex [dict get $fabric_leaf n0_arvalid] 0]] REF_NAME GND}
 wrong_flow_direction {dict set pins u_pcis_pipe/m_axi_awvalid DIRECTION IN}
 escaped_net {dict set fabric_net_ports n0_arvalid {unexpected_top_port}}
 missing_receiver {dict set fabric_leaf n0_arvalid [lrange [dict get $fabric_leaf n0_arvalid] 0 0]}
 unknown_leaf_direction {dict set pins [lindex [dict get $fabric_leaf n0_arvalid] 1] DIRECTION INOUT}
}
rename get_pins legacy_get_pins
proc get_pins {args} {
 if {"-of_objects" in $args && "-leaf" in $args} {
  set result {};foreach net [lindex $args end] {if {[dict exists $::fabric_leaf $net]} {lappend result {*}[dict get $::fabric_leaf $net]}}
  return $result
 }
 if {"-of_objects" ni $args} {
  set name [lindex [lindex $args end] 0]
  return [expr {[dict exists $::pins $name] ? $name : {}}]
 }
 return [legacy_get_pins {*}$args]
}
rename get_cells legacy_get_cells
proc get_cells {args} {
 if {"-of_objects" in $args} {
  set result {};foreach pin [lindex $args end] {set cell [file dirname $pin];if {[dict exists $::cells $cell]} {lappend result $cell}}
  return $result
 }
 return [legacy_get_cells {*}$args]
}
proc get_nets {args} {
 set query [lindex $args end]
 if {"-of_objects" in $args} {
  set result {};foreach anchor $query {
   if {[dict exists $::fabric_nets $anchor]} {error "Vivado does not accept a net as get_nets -of_objects input"}
   if {[dict exists $::fabric_anchor $anchor]} {lappend result [dict get $::fabric_anchor $anchor]}
  }
  return [lsort -unique $result]
 }
 return [expr {[dict exists $::fabric_nets $query] ? $query : {}}]
}
proc all_fanout {args} {return [dict get $::fabric_b_endpoints [lindex [lindex $args end] 0]]}
proc get_ports {args} {
 set result {};foreach net [lindex $args end] {if {[dict exists $::fabric_net_ports $net]} {lappend result {*}[dict get $::fabric_net_ports $net]}}
 return $result
}
rename get_property legacy_get_property
proc get_property {key objects} {
 if {$::fault eq "checkpoint_mutates_graph" && [info exists ::checkpoint_written]} {error "property inspected graph after diagnostic serialization"}
 if {$key eq {NAME}} {return [expr {[llength $objects]==1 ? [lindex $objects 0] : $objects}]}
 return [legacy_get_property $key $objects]
}
'''
MOCK = MOCK.replace('if {$fault eq "existing_directory"}', FABRIC_MOCK + '\nif {$fault eq "existing_directory"}')


def run_audit(tmp_path, fault):
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl unavailable')
    driver = tmp_path / 'mock.tcl'
    driver.write_text(MOCK)
    return subprocess.run([tclsh, str(driver), str(AUDIT), str(tmp_path), fault],
                          text=True, capture_output=True, timeout=20)


@pytest.mark.parametrize('fault', ['none', 'vendor_opaque', 'opaque_clock', 'vendor_multicycle',
                                 'ref_without_orig', 'ref_empty_orig', 'pruned_unused_pin',
                                 'retained_unused_clock', 'indexed_handshakes', 'no_cell_property_enumeration',
                                 'pruned_boundary_bit','pruned_boundary_interface','pruned_boundary_handshake','checkpoint_mutates_graph'])
def test_synthesis_records_limits_and_required_current_vendor_simulation(tmp_path, fault):
    result = run_audit(tmp_path, fault)
    assert result.returncode == 0, result.stderr
    assert 'C2_ALEX_SYNTH_AUDIT_RECORDED route_qualified=0' in result.stdout
    directory = tmp_path / 'synth_audit'
    summary = (directory / 'summary.tcldata').read_text()
    assert 'route_qualified 0 hardware_measured 0' in summary
    assert 'inspected_state live_design_before_diagnostic_serialization' in summary
    assert 'diagnostic_capture_order after_live_audit_checks' in summary
    assert 'slave_interfaces 9 master_interfaces 8' in summary
    assert 'actual_9SI_8MI_vendor_simulation' in summary
    assert 'full_shell_link_and_complete_clock_CDC_audit' in summary
    assert 'generated_BD_required' in summary
    assert len(list(directory.iterdir())) == 14
    assert 'memory_interface_pins ' in (directory / 'objects.tcldata').read_text()
    assert (tmp_path / 'pre_audit.NOT_QUALIFIED.dcp').is_file()
    assert not list(tmp_path.rglob('*post_synth*.dcp'))
    assert 'qualification NOT_QUALIFIED capture_order after_live_audit_checks' in (directory / 'objects.tcldata').read_text()
    assert not list(tmp_path.rglob('*.passed'))
    checklist = (directory / 'REQUIRED_NEXT_STEPS.txt').read_text()
    for required in ('CURRENT nine-SI/eight-MI', 'narrow byte strobes', 'independent IDs',
                     'two-port simulation is insufficient', 'full byte readback'):
        assert required in checklist
    if fault == 'opaque_clock':
        assert 'deferred_clocks {core tmem_unused}' in summary
        assert 'state deferred_opaque_ip' in (directory / 'objects.tcldata').read_text()
    if fault == 'vendor_multicycle':
        assert 'vendor_multicycles_pending_review 1' in summary
    if fault == 'pruned_unused_pin':
        assert 'pruned_unused_outputs tmem_unused' in summary
        assert 'deferred_clocks {}' in summary
        assert 'state pruned_unused_output' in (directory / 'objects.tcldata').read_text()


@pytest.mark.parametrize('fault', [
    'wrong_kernel', 'opaque_kernel', 'opaque_mxu', 'missing_native', 'extra_native',
    'wrong_native', 'wrong_vendor', 'unknown_opaque', 'lookalike_opaque',
    'missing_clock', 'wrong_clock', 'wrong_hbm_clock', 'wrong_host_clock',
    'wrong_reference_clock', 'disconnected_core', 'disconnected_hbm',
    'disconnected_host', 'missing_clock_pin', 'custom_hclk', 'lookalike_hclk',
    'used_400', 'duplicate_bit', 'version', 'scope', 'tmem_cell',
    'tmem_exception', 'custom_multicycle', 'incomplete_export', 'missing_report',
    'report_error', 'existing_directory',
    'pruned_pin_used_400', 'pruned_pin_unknown_source', 'pruned_pin_no_source',
    'missing_ref_property', 'missing_blackbox_property', 'empty_ref_property',
    'empty_blackbox_property', 'invalid_blackbox_property', 'duplicate_handshake',
    'duplicate_handshake_bit', 'sc_clock_swap', 'ready_constant', 'ready_missing_channel',
    'ready_wrong_reset','ready_wrong_clock','ready_wrong_channel','b_missing_error','b_wrong_native',
    'existing_checkpoint', 'checkpoint_error',
])
def test_synthesis_audit_rejects_wrong_or_incomplete_evidence(tmp_path, fault):
    result = run_audit(tmp_path, fault)
    assert result.returncode != 0, fault
    assert result.stderr.strip(), fault
    assert 'C2_ALEX_SYNTH_AUDIT_RECORDED' not in result.stdout
    assert not (tmp_path / 'synth_audit/summary.tcldata').exists()
    assert not list(tmp_path.rglob('*.passed'))
    assert not list(tmp_path.rglob('*post_synth*.dcp'))
    diagnostic = tmp_path / 'pre_audit.NOT_QUALIFIED.dcp'
    if fault == 'existing_checkpoint':
        assert diagnostic.read_text().strip() == 'PREEXISTING'
    elif fault not in ('version', 'scope', 'existing_directory', 'checkpoint_error'):
        assert diagnostic.read_text().strip() == 'MOCK_DIAGNOSTIC_NOT_QUALIFIED_NOT_FPGA_CONTENT'


@pytest.mark.parametrize('fault', ['none','native_swap','sc_swap','channel_swap','hbm_index','missing_active_net',
    'shorted_net','constant_active','wrong_flow_direction','escaped_net','missing_receiver','unknown_leaf_direction'])
def test_rebuilt_fabric_active_connections(tmp_path,fault):
    text=MOCK[:MOCK.index('if {$fault eq "existing_directory"}')]
    text+=r"""
set stream [open [file join $run flows.tcldata] w]
if {[catch {
 foreach signal {awvalid awready wvalid wready bvalid bready arvalid arready rvalid rready} {::c2_alex::flow {} S_HOST $signal $stream}
 for {set p 0} {$p<8} {incr p} {
  foreach signal {awvalid awready wvalid wready arvalid arready rvalid} {::c2_alex::flow {} S_N$p $signal $stream}
  foreach signal {awvalid awready wvalid wready bvalid bready arvalid arready rvalid rready} {::c2_alex::flow {} M_H$p $signal $stream}
 }
} message]} {close $stream;puts stderr $message;exit 1}
close $stream
"""
    path=tmp_path/'flows.tcl';path.write_text(text)
    result=subprocess.run([shutil.which('tclsh'),str(path),str(AUDIT),str(tmp_path),fault],capture_output=True,text=True,timeout=20)
    assert (result.returncode==0)==(fault=='none'),result.stderr
    if fault=='none':assert len((tmp_path/'flows.tcldata').read_text().splitlines())==146


@pytest.mark.parametrize('prefix',['','WRAPPER/CL'])
def test_actual_d1c_connection_records_replay_owner_rules(tmp_path,prefix):
    """Optional recorded-name regression; mocked properties are not tool proof."""
    raw=AUDIT.parents[2]/'build/c2-alex-remote/probe-d1c20797-v2-evidence/diagnostic/external_handshakes.tcldata'
    if not raw.is_file():pytest.skip('Optional diagnostic record absent')
    script=r'''
source [lindex $argv 0]
set prefix [lindex $argv 3];set root [expr {$prefix eq {} ? {} : "$prefix/"}]
set pins {};set pin_nets {};set nets {};set leaves {};set cells {};set todo {}
set f [open [lindex $argv 2] r]
while {[gets $f row]>=0} {
 if {![dict exists $row interface]} continue
 set iface [dict get $row interface];set signal [dict get $row signal]
 # This historical record lacks receive-gate INIT/reset/fanout/clock facts.
 # Current receive validation has a separate typed fixture and real replay.
 if {[string match S_N* $iface] && $signal in {bvalid bready rvalid rready}} continue
 set forward [expr {$signal in {awvalid wvalid arvalid bready rready}}]
 if {$iface eq {S_HOST}} {set direction [expr {$forward ? {OUT} : {IN}}]} else {set direction [expr {$forward ? {IN} : {OUT}}]}
 set ns {};foreach n [dict get $row nets] {lappend ns $root$n;dict set nets $root$n 1}
 foreach query [dict get $row exact_pin_queries] {
  foreach pin [dict get $query objects] {dict set pins $root$pin $direction;dict set pin_nets $root$pin $ns}
 }
 foreach n [dict get [dict get $row top_net_query] objects] {dict set pin_nets $root$n $ns}
 set ls {}
 foreach pair [dict get $row leaf_directions] {
  lassign $pair pin dir;set pin $root$pin;dict set pins $pin $dir;lappend ls $pin
  # The diagnostic did not query these primitive REF_NAMEs. Use an explicit
  # mock LUT type here; actual Vivado rerun remains the authoritative gate.
  dict set cells [file dirname $pin] LUT6
 }
 foreach net $ns {dict set leaves $net $ls;dict set pin_nets $net $ns}
 lappend todo [list $iface $signal]
}
close $f
proc get_pins {args} {
 if {"-of_objects" in $args} {set r {};foreach n [lindex $args end] {lappend r {*}[dict get $::leaves $n]};return [lsort -unique $r]}
 set p [lindex [lindex $args end] 0];return [expr {[dict exists $::pins $p] ? $p : {}}]
}
proc get_nets {args} {
 set q [lindex $args end]
 if {"-of_objects" in $args} {set r {};foreach p $q {lappend r {*}[dict get $::pin_nets $p]};return [lsort -unique $r]}
 set q [lindex $q 0];return [expr {[dict exists $::nets $q] ? $q : {}}]
}
proc get_cells {args} {return [list [file dirname [lindex [lindex $args end] 0]]]}
proc get_ports {args} {return {}}
proc get_property {key objects} {
 set result {};foreach o $objects {
  switch $key {NAME {lappend result $o} DIRECTION {lappend result [dict get $::pins $o]} REF_NAME {lappend result [dict get $::cells $o]} default {error "unexpected property $key"}}
 }
 return [expr {[llength $objects]==1 ? [lindex $result 0] : $result}]
}
set out [open [lindex $argv 1] w]
foreach item $todo {lassign $item iface signal;::c2_alex::flow $prefix $iface $signal $out}
close $out
puts [llength $todo]
'''
    path=tmp_path/'replay.tcl';path.write_text(script)
    result=subprocess.run([shutil.which('tclsh'),str(path),str(AUDIT),str(tmp_path/'flows.tcldata'),str(raw),prefix],
                          text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='138'


@pytest.mark.parametrize('fault',['none','sc_clock_swap','core_on_host','fabric_on_reference','pcis_on_core','custom_hclk','lookalike_hclk'])
def test_fabric_clocks_use_real_per_port_consumers(tmp_path,fault):
    text=MOCK[:MOCK.index('if {$fault eq "existing_directory"}')]
    text+=r"""
set out [open [file join $run clocks.tcldata] w]
if {[catch {::c2_alex::fabric_clocks {} {host clk_main_a0 core core hbm hbm} {} $out} message]} {close $out;puts stderr $message;exit 1}
close $out
"""
    path=tmp_path/'clocks.tcl';path.write_text(text)
    result=subprocess.run([shutil.which('tclsh'),str(path),str(AUDIT),str(tmp_path),fault],text=True,capture_output=True,timeout=20)
    assert (result.returncode==0)==(fault=='none'),result.stderr


@pytest.mark.parametrize('fault',['none','b_missing_error','b_wrong_native'])
def test_native_b_response_absorbed_logic_reaches_own_state_and_error(tmp_path,fault):
    text=MOCK[:MOCK.index('if {$fault eq "existing_directory"}')]
    text+=r"""
set out [open [file join $run bresponse.tcldata] w]
if {[catch {for {set p 0} {$p<8} {incr p} {::c2_alex::flow {} S_N$p bvalid $out}} message]} {close $out;puts stderr $message;exit 1}
close $out
"""
    path=tmp_path/'bvalid.tcl';path.write_text(text)
    result=subprocess.run([shutil.which('tclsh'),str(path),str(AUDIT),str(tmp_path),fault],text=True,capture_output=True,timeout=20)
    assert (result.returncode==0)==(fault=='none'),result.stderr


@pytest.mark.parametrize('fault',['none','ready_constant','ready_missing_channel','ready_wrong_reset','ready_wrong_clock','ready_wrong_channel'])
def test_native_ready_proves_reset_inversion_and_all_sixteen_channel_sinks(tmp_path,fault):
    text=MOCK[:MOCK.index('if {$fault eq "existing_directory"}')]
    text+=r"""
set out [open [file join $run ready.tcldata] w]
if {[catch {::c2_alex::native_shared_ready {} {core core} $out} message]} {close $out;puts stderr $message;exit 1}
close $out
"""
    path=tmp_path/'ready.tcl';path.write_text(text)
    result=subprocess.run([shutil.which('tclsh'),str(path),str(AUDIT),str(tmp_path),fault],text=True,capture_output=True,timeout=20)
    assert (result.returncode==0)==(fault=='none'),result.stderr


def test_actual_slim_native_b_and_ready_evidence_matches_reviewed_rules(tmp_path):
    raw=AUDIT.parents[2]/'build/c2-alex-remote/probe-d1c20797-write-slim-evidence/diagnostic/write_ready.tcldata'
    if not raw.is_file():pytest.skip('Optional focused diagnostic record absent')
    script=r'''
source [lindex $argv 0]
set f [open [lindex $argv 1] r];set records {};set props {};set fanouts {};set bdrivers {}
while {[gets $f row]>=0} {
 if {[dict exists $row label] && [dict exists $row leaf_directions]} {dict set records [dict get $row label] $row}
 if {[lindex $row 0] eq {specific_cell}} {set p [dict get [lrange $row 1 end] properties];dict set props [dict get $p NAME] $p}
 if {[dict exists $row fanout]} {dict set fanouts [dict get $row fanout] [dict get $row endpoints]}
}
close $f
proc get_property {key objects} {return [expr {[llength $objects]==1 ? [lindex $objects 0] : $objects}]}
proc all_fanout {args} {return $::expected_endpoints}
set out [open [lindex $argv 2] w]
for {set p 0} {$p<8} {incr p} {
 set row [dict get $records [list native$p valid_state]]
 set si [format s%02d [expr {$p+1}]]
 set node u_mem/f2_mem_i/memory_sc/inst/${si}_entry_pipeline/${si}_mmu/inst/b_sreg
 set extras {};set drivers {}
 foreach pair [dict get $row leaf_directions] {
  lassign $pair pin direction
  if {$direction eq {OUT}} {lappend drivers $pin}
  if {![::c2_alex::inside $pin $node]} {
   if {$direction ne {IN}} {error {unexpected B driver}}
   lappend extras $pin
  }
 }
 if {[lsort $extras] ne [lsort [::c2_alex::relocated_native_loads {} $p bvalid]]} {error {B relocated inputs differ}}
 if {[llength $drivers]!=1} {error {B driver count}}
 set expected_endpoints [dict get $fanouts native$p]
 ::c2_alex::native_b_endpoints {} $p $drivers $node $out
}
set inv [dict get $props u_mem/u_status_cdc_i_2]
if {[dict get $inv REF_NAME] ne {LUT1} || [dict get $inv INIT] ne {2'h1}} {error {reset inverter differs}}
set rst [dict get $props core_rst_reg]
if {[dict get $rst REF_NAME] ne {FDRE}} {error {reset primitive differs}}
set input [dict get $records {specific u_mem/u_status_cdc_i_2/I0}];set drivers {}
foreach pair [dict get $input leaf_directions] {if {[lindex $pair 1] eq {OUT}} {lappend drivers [lindex $pair 0]}}
if {$drivers ne {core_rst_reg/Q}} {error {ready reset source differs}}
set output [dict get $records {specific u_mem/u_status_cdc_i_2/O}];set channels {};set drivers {};set status 0
foreach pair [dict get $output leaf_directions] {
 lassign $pair pin direction
 if {$direction eq {OUT}} {lappend drivers $pin;continue}
 if {$pin eq {u_status_cdc/src_ff_reg[0]/D}} {incr status;continue}
 if {$direction ne {IN} || ![regexp {^u_mem/f2_mem_i/memory_sc/inst/(s0[1-8])_entry_pipeline/(s0[1-8])_mmu/inst/(b|r)_sreg/} $pin -> si mmu ch] || $si ne $mmu} {error {unexpected ready sink}}
 dict incr channels $si/$ch
}
if {$drivers ne {u_mem/u_status_cdc_i_2/O} || [dict size $channels]!=16 || $status!=1} {error {shared ready coverage differs}}
close $out
puts {eight B responses and sixteen reset-gated ready connections match}
'''
    path=tmp_path/'slim.tcl';path.write_text(script)
    result=subprocess.run([shutil.which('tclsh'),str(path),str(AUDIT),str(raw),str(tmp_path/'review.tcldata')],
                          text=True,capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert 'sixteen reset-gated ready connections match' in result.stdout


@pytest.mark.parametrize('reconstructed_kind', ['', 'pins', 'nets', 'clocks'])
def test_filtered_selections_preserve_vivado_collection_provenance(tmp_path, reconstructed_kind):
    """Tcl-object identity models Vivado handles; it is not physical evidence.

    Reconstructing the same displayed names into another Tcl list must not
    satisfy APIs requiring an object collection. Keep selected objects alive
    so recycled Tcl pointer addresses cannot accidentally satisfy the mock.
    """
    typed = r'''
if {![llength [info commands ::tcl::unsupported::representation]]} {exit 77}
set typed_objects {};set held_objects {}
proc object_id {object} {
 if {![regexp {object pointer at ([^, ]+)} [::tcl::unsupported::representation $object] -> id]} {error {Tcl object identity unavailable}}
 return $id
}
proc typed_result {objects} {
 if {![llength $objects]} {return {}}
 # Give every selector result an independent identity, then retain both the
 # collection and its elements (foreach over a Vivado collection is valid).
 set value [string range "x$objects" 1 end]
 dict set ::typed_objects [object_id $value] 1;lappend ::held_objects $value
 foreach object $value {dict set ::typed_objects [object_id $object] 1;lappend ::held_objects $object}
 return $value
}
proc require_objects {objects operation} {
 if {![llength $objects]} {return}
 if {![dict exists $::typed_objects [object_id $objects]]} {error "untyped reconstructed collection at $operation"}
}
foreach name {get_pins get_nets get_cells get_clocks all_registers all_fanout} {
 rename $name typed_original_$name
 proc $name {args} [format {
  foreach option {-of_objects -clock -from} {
   set at [lsearch -exact $args $option]
   if {$at>=0} {require_objects [lindex $args [expr {$at+1}]] %s/$option}
  }
  return [typed_result [typed_original_%s {*}$args]]
 } $name $name]
}
rename get_ports typed_original_get_ports
proc get_ports {args} {
 if {"-of_objects" in $args} {require_objects [lindex $args end] get_ports/-of_objects}
 return [typed_original_get_ports {*}$args]
}
rename get_property typed_original_get_property
proc get_property {key objects} {
 require_objects $objects get_property/$key
 return [typed_original_get_property $key $objects]
}
# Simulate the old lappend/expansion bug after an otherwise correct selector.
set reconstructed_kind [lindex $argv 3]
if {$reconstructed_kind ne {}} {
 rename ::c2_alex::selected_objects ::c2_alex::typed_selected_objects
 proc ::c2_alex::selected_objects {kind wanted} {
  set selected [typed_selected_objects $kind $wanted]
  if {$kind eq $::reconstructed_kind} {set rebuilt {};lappend rebuilt {*}$selected;return $rebuilt}
  return $selected
 }
}
'''
    text = MOCK.replace('if {$fault eq "existing_directory"}', typed + '\nif {$fault eq "existing_directory"}')
    path = tmp_path / 'typed.tcl'
    path.write_text(text)
    result = subprocess.run([shutil.which('tclsh'), str(path), str(AUDIT), str(tmp_path),
                             'retained_unused_clock', reconstructed_kind],
                            text=True, capture_output=True, timeout=20)
    if result.returncode == 77:
        pytest.skip('Tcl object representation unavailable')
    if reconstructed_kind:
        assert result.returncode != 0
        assert 'untyped reconstructed collection' in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert 'C2_ALEX_SYNTH_AUDIT_RECORDED route_qualified=0' in result.stdout


@pytest.mark.parametrize('fault,primary', [
    ('audit_error_and_checkpoint_error', 'unexpected reference'),
    ('report_error_and_checkpoint_error', 'mock CDC report failure'),
])
def test_failed_diagnostic_capture_preserves_primary_audit_error(tmp_path, fault, primary):
    result = run_audit(tmp_path, fault)
    assert result.returncode != 0
    assert primary in result.stderr
    assert 'mock checkpoint write failure' not in result.stderr
    assert 'C2_ALEX_SYNTH_AUDIT_RECORDED' not in result.stdout
    assert not (tmp_path / 'synth_audit/summary.tcldata').exists()
    raw = (tmp_path / 'synth_audit/objects.tcldata').read_text()
    assert 'audit_failure ' in raw and primary in raw
    assert 'diagnostic_capture_error {mock checkpoint write failure}' in raw


def test_diagnostic_write_failure_after_checks_prevents_audit_success(tmp_path):
    result = run_audit(tmp_path, 'checkpoint_error')
    assert result.returncode != 0
    assert 'mock checkpoint write failure' in result.stderr
    assert 'C2_ALEX_SYNTH_AUDIT_RECORDED' not in result.stdout
    directory = tmp_path / 'synth_audit'
    assert (directory / 'utilization.rpt').is_file()
    assert not (directory / 'summary.tcldata').exists()
    assert not list(tmp_path.rglob('*post_synth*.dcp'))


def test_failed_live_check_captures_diagnostic_only_after_failure(tmp_path):
    result = run_audit(tmp_path, 'wrong_kernel')
    assert result.returncode != 0
    raw = (tmp_path / 'synth_audit/objects.tcldata').read_text()
    assert raw.index('audit_failure ') < raw.index('diagnostic_checkpoint ')
    assert 'capture_order after_audit_failure' in raw
    assert 'inspected_state live_design_before_diagnostic_serialization' in raw
    assert (tmp_path / 'pre_audit.NOT_QUALIFIED.dcp').is_file()
