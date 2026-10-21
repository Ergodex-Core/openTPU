# Additive raw structural observations on one immutable final C1 checkpoint.
# Every flow is attempted independently. Failed checks remain explicit data;
# this helper never qualifies a design or changes timing/design constraints.
source [file join [file dirname [info script]] qualify_c1_alex.tcl]
source [file join [file dirname [info script]] routed_host_ready.tcl]
namespace eval ::c1_structure_capture {
    variable helper [file normalize [info script]]
    variable max_bytes 134217728
    variable written 0
    variable exhausted 0
    variable attempted 0
    variable failures 0
    variable labels {}
    variable out
    variable audit
}
# A reflected channel bounds even output written by the unchanged audit helpers.
proc ::c1_structure_capture::sink {operation channel args} {
    variable out;variable written;variable max_bytes;variable exhausted
    switch -- $operation {
        initialize {return {initialize finalize watch write}}
        finalize {return}
        watch {return}
        write {
            set data [lindex $args 0];set n [string length $data]
            if {$written+$n>$max_bytes} {set exhausted 1;error "Structural raw output cap"}
            puts -nonewline $out $data;incr written $n;return $n
        }
        default {error "Unexpected raw channel operation"}
    }
}
proc ::c1_structure_capture::emit {record} {variable audit;puts $audit $record}
proc ::c1_structure_capture::step {label command} {
    variable attempted;variable failures;variable labels;variable exhausted
    if {$label in $labels} {error "Duplicate structural step"}
    lappend labels $label;incr attempted
    emit [list step_begin $label]
    set code [catch {uplevel 1 $command} value options]
    if {$exhausted} {error "Structural raw output cap"}
    if {$code} {
        incr failures
        emit [list step_result $label state error message $value options $options route_qualified 0]
        return [dict create observed 0 value {}]
    }
    emit [list step_result $label state observed value $value route_qualified 0]
    return [dict create observed 1 value $value]
}
proc ::c1_structure_capture::general {audit} {
    set cl WRAPPER/CL
    ::c1_alex::cell $cl cl_opentpu 1
    ::c1_alex::cell $cl/u_board otpu_c1_hbm_kernel 1
    ::c1_alex::cell $cl/u_board/core otpu_c1_top 1
    set expected {}
    for {set p 0} {$p<8} {incr p} {lappend expected $cl/u_native$p;::c1_alex::cell $cl/u_native$p otpu_native_axi 1}
    if {[lsort [::c1_alex::names [get_cells -quiet -hierarchical -regexp {^WRAPPER/CL/u_native[0-9]+$}]]] ne $expected} {error "Native topology changed"}
    foreach {name ref} {u_mem f2_mem_wrapper u_hbm cl_mem_hbm_wrapper u_ocl_cdc f2_ocl_cdc u_pcis_pipe f2_pcis_pipe u_core_clk f2_core_clk u_hbm_clk f2_hbm_clk} {::c1_alex::cell $cl/$name $ref 1}
    set opaque {}
    foreach object [get_cells -quiet -hierarchical -filter {IS_BLACKBOX}] {
        set name [::c1_alex::property $object NAME]
        if {$name eq $cl || [string first "$cl/" $name]==0} {error "Unresolved CL/IP $name"}
        lappend opaque $name
    }
    if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ otpu_tmem* || ORIG_REF_NAME =~ otpu_tmem*}]]} {error "Unexpected TMEM implementation"}
    set pins [::c1_alex::names [get_pins -quiet -of_objects [get_cells $cl/u_mem]]]
    if {![llength $pins] || [llength $pins]!=[llength [lsort -unique $pins]]} {error "Empty/duplicate memory interface pins"}
    foreach pin $pins {if {[string first "$cl/" $pin]!=0} {error "Memory pin outside CL"}}
    puts $audit [list memory_interface_pins $pins]
    set sc [get_cells -quiet $cl/u_mem/f2_mem_i/memory_sc]
    if {[::c1_alex::names $sc] ne [list $cl/u_mem/f2_mem_i/memory_sc] || [::c1_alex::property $sc IS_BLACKBOX]} {error "Missing concrete SmartConnect"}
    puts $audit [list smartconnect [::c1_alex::names $sc] reference [::c1_alex::property $sc REF_NAME] slave_interfaces 9 master_interfaces 8 address_map_validation executor_generated_BD_required]
    puts $audit [list unresolved_shell_objects $opaque]
    return [dict create opaque_shell $opaque native_masters 8 slave_interfaces 9 master_interfaces 8]
}
proc ::c1_structure_capture::clocks {audit} {
    set cl WRAPPER/CL;set clocks {}
    foreach {label pin period} {host clk_main_a0 4.000 core u_core_clk/clk_out1 5.000 hbm u_hbm_clk/clk_out1 2.222222222222 reference clk_hbm_ref 10.000} {
        set clock [::c1_alex_route::clock $cl/$pin $period]
        dict set clocks $label $clock
        set data {};foreach key {NAME PERIOD MASTER_CLOCK SOURCE_PINS IS_GENERATED IS_PROPAGATED IS_VIRTUAL} {if {$key in [list_property $clock]} {dict set data $key [get_property $key $clock]}}
        puts $audit [list clock $label $data]
    }
    foreach label {host core hbm} {
        set registers [all_registers -clock [dict get $clocks $label] -cells]
        if {![llength $registers]} {error "Empty clock domain $label"}
        puts $audit [list clock_registers $label count [llength $registers]]
        if {$label eq "hbm"} {
            foreach name [::c1_alex::names $registers] {if {[::c1_alex_route::owner $name] ni {u_mem u_hbm u_rst_hbm}} {error "Unexpected HBM-clock register $name"}}
        }
    }
    ::c1_alex::fabric_clocks $cl $clocks {} $audit
    set names {};dict for {label objects} $clocks {dict set names $label [::c1_alex::names $objects]}
    return [dict create clocks $names]
}
proc ::c1_structure_capture::unused_clocks {audit} {
    set core [::c1_alex_route::clock WRAPPER/CL/u_core_clk/clk_out1 5.000]
    set inventory {};set unused {}
    foreach clock [get_clocks -quiet *] {
        set name [get_property NAME $clock]
        if {$name eq "" || $name in $inventory} {error "Empty/duplicate clock name"}
        lappend inventory $name
        if {"SOURCE_PINS" ni [list_property $clock]} {error "Missing clock source inventory $name"}
        set sources [get_property SOURCE_PINS $clock]
        puts $audit [list clock_source_inventory $name sources $sources]
        foreach source $sources {
            if {[string first "WRAPPER/CL/u_core_clk/" $source]==0 && $clock ne $core} {
                set members [::c1_alex::names [all_registers -clock $clock -cells]]
                puts $audit [list unused_clock $name period [get_property PERIOD $clock] consumers [llength $members] registers $members]
                if {[llength $members]} {error "Unexpected second core-wizard clock consumer"}
                if {$name ni $unused} {lappend unused $name}
            }
        }
    }
    if {![llength $inventory]} {error "Empty clock inventory"}
    return [dict create clocks $inventory unused $unused]
}
proc ::c1_structure_capture::disabled {audit p channel} {
    if {$p<0 || $p>7 || $channel ni {aw w}} {error "Invalid disabled-write scope"}
    set si [format s%02d [expr {$p+1}]]
    set name WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/${si}_entry_pipeline/${si}_mmu/inst/${channel}_sreg/s_axi_${channel}valid
    set pin [::c1_alex::exact_pin $name]
    if {[get_property DIRECTION $pin] ne "IN"} {error "Disabled write input direction"}
    set nets [get_nets -quiet -of_objects $pin]
    if {![llength $nets] || [llength $nets]>256} {error "Disabled write input disconnected/excess local nets"}
    # Enumerate only drivers/ports of the complete typed network. Listing every
    # shared GND leaf/alias repeatedly is unnecessary for constant-input proof.
    set segments [get_nets -quiet -segments -of_objects $pin]
    if {![llength $segments] || [llength $segments]>100000} {error "Disabled write segment inventory bound"}
    set ports [get_ports -quiet -of_objects $segments]
    set drivers [get_pins -quiet -leaf -filter {DIRECTION == OUT} -of_objects $segments]
    if {[llength $drivers]>256 || [llength $ports]>256} {error "Disabled write endpoint inventory bound"}
    puts $audit [list disabled_network $p channel $channel anchor [::c1_alex::names $pin] local_nets [::c1_alex::names $nets] segment_count [llength $segments] drivers [::c1_alex::names $drivers] ports [::c1_alex::names $ports] sink_scope anchor_only]
    if {[llength $ports]} {error "Disabled write input escaped"}
    if {[llength $drivers]!=1} {error "Disabled write input driver ambiguity"}
    set cell [get_cells -quiet -of_objects $drivers]
    if {[llength $cell]!=1 || [get_property REF_NAME $cell] ne "GND"} {error "Native write input is not tied low"}
    set anchor_properties {};set driver_properties {}
    foreach key {NAME REF_PIN_NAME DIRECTION IS_INVERTED} {
        dict set anchor_properties $key [get_property $key $pin]
        dict set driver_properties $key [get_property $key $drivers]
    }
    puts $audit [list disabled_polarity $p channel $channel anchor $anchor_properties driver $driver_properties owner [::c1_alex::names $cell] reference GND]
    if {[dict get $anchor_properties REF_PIN_NAME] ne "s_axi_${channel}valid" || [dict get $anchor_properties IS_INVERTED] ni {0 false FALSE} || [dict get $driver_properties IS_INVERTED] ni {0 false FALSE} || [dict get $driver_properties REF_PIN_NAME] ne "G" || [dict get $driver_properties DIRECTION] ne "OUT"} {error "Disabled write effective polarity differs"}
    puts $audit [list native_disabled_write $p channel $channel pin [::c1_alex::names $pin] nets [::c1_alex::names $nets] driver [::c1_alex::names $drivers] primitive GND]
    return [dict create pc $p channel $channel primitive GND]
}
proc ::c1_structure_capture::body {} {
    variable audit
    step general [list general $audit]
    step clocks [list clocks $audit]
    foreach signal {awvalid awready wvalid wready bvalid bready arvalid arready rvalid rready} {
        if {$signal eq "wready"} {
            step [list flow S_HOST $signal] [list ::c1_alex_routed_ready::host_wready $audit]
        } else {step [list flow S_HOST $signal] [list ::c1_alex::flow WRAPPER/CL S_HOST $signal $audit]}
    }
    for {set p 0} {$p<8} {incr p} {
        foreach signal {bvalid arvalid arready rvalid} {step [list flow S_N$p $signal] [list ::c1_alex::flow WRAPPER/CL S_N$p $signal $audit]}
        foreach signal {awvalid awready wvalid wready bvalid bready arvalid arready rvalid rready} {step [list flow M_H$p $signal] [list ::c1_alex::flow WRAPPER/CL M_H$p $signal $audit]}
        foreach channel {aw w} {step [list disabled_write $p $channel] [list disabled $audit $p $channel]}
    }
    step unused_clocks [list unused_clocks $audit]
}
proc ::c1_structure_capture::capture {run checkpoint expected} {
    variable helper;variable out;variable audit;variable written;variable exhausted
    variable attempted;variable failures;variable labels
    if {![regexp {^[0-9a-f]{64}$} $expected] || ![file isfile $checkpoint] || [file type $checkpoint] ne "file" || [::c1_alex_route::sha256 $checkpoint] ne $expected} {error "Exact regular checkpoint binding required"}
    if {[version -short] ne "2025.2" || [llength [get_designs -quiet]]} {error "Fresh Vivado 2025.2 process required"}
    if {[file exists $run] || ![file isdirectory [file dirname $run]]} {error "Fresh output with existing parent required"}
    set helpers {}
    foreach name {capture_structure.tcl qualify_c1_alex.tcl audit_c1_alex.tcl routed_host_ready.tcl} {dict set helpers $name [::c1_alex_route::sha256 [file join [file dirname $helper] $name]]}
    set written 0;set exhausted 0;set attempted 0;set failures 0;set labels {}
    file mkdir $run
    set out [open [file join $run structural.tcldata] {WRONLY CREAT EXCL}]
    fconfigure $out -translation binary -buffering none
    set audit [chan create write ::c1_structure_capture::sink]
    fconfigure $audit -translation lf -encoding utf-8 -buffering none
    set opened 0
    set code [catch {
        emit [list header schema 1 kind c1_structural_scope_capture checkpoint $checkpoint checkpoint_sha256 $expected helpers $helpers route_qualified 0 hardware_measured 0]
        set_param general.maxThreads 2
        open_checkpoint $checkpoint;set opened 1
        body
        if {$attempted!=141} {error "Incomplete structural step inventory"}
    } message options]
    if {$opened} {
        if {[catch {close_design} close_error] && !$code} {set code 1;set message $close_error;set options [dict create -code 1 -level 0]}
    }
    set unchanged [catch {
        if {[::c1_alex_route::sha256 $checkpoint] ne $expected} {error "Checkpoint changed"}
        dict for {name digest} $helpers {if {[::c1_alex_route::sha256 [file join [file dirname $helper] $name]] ne $digest} {error "Collector changed"}}
    } changed_message changed_options]
    if {$unchanged} {set code 1;set message $changed_message;set options $changed_options}
    if {!$code} {
        set code [catch {emit [list summary attempted $attempted observed [expr {$attempted-$failures}] failed $failures labels $labels output_byte_cap $exhausted route_qualified 0 hardware_measured 0]} message options]
    } elseif {!$exhausted} {catch {emit [list capture_error message $message options $options route_qualified 0]}}
    catch {close $audit};close $out
    if {$code} {return -options $options $message}
    return [dict create output [file join $run structural.tcldata] attempted $attempted failed $failures route_qualified 0 hardware_measured 0]
}
