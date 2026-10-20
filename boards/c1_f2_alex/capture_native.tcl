# Independent raw C1 topology capture. No constraints, waivers or acceptance.
# source this file, then ::c1_native_capture::capture RUN DCP EXPECTED_SHA256
namespace eval ::c1_native_capture {
    variable out; variable gaps 0; variable queries 0
    variable max_bytes 134217728; variable written 0; variable exhausted 0
    variable max_names 20000; variable max_cells 256; variable max_depth 12
    variable seen_cells; variable seen_details; variable seen_networks; variable seen_properties; variable seen_owners; variable seen_tag
}
proc ::c1_native_capture::emit {args} {
    variable out;variable max_bytes;variable written;variable exhausted
    if {$exhausted} {return}
    set size [expr {[string length [encoding convertto utf-8 $args]]+1}]
    if {$written+$size > $max_bytes-4096} {
        set exhausted 1
        puts $out [list incomplete capture reason output_byte_cap limit $max_bytes]
        error "C1 raw output byte cap reached"
    }
    puts $out $args;incr written $size
}
proc ::c1_native_capture::gap {label reason args} {
    variable gaps; incr gaps; emit incomplete $label reason $reason {*}$args
}
proc ::c1_native_capture::qnames {label script} {
    variable queries; variable max_names; incr queries
    if {[catch {uplevel 1 $script} objects opts]} {
        gap $label api_error message $objects; return {}
    }
    # Never serialize a Vivado collection or call bulk get_property NAME.
    set names {}; set total 0
    foreach object $objects {
        incr total
        if {[catch {get_property NAME $object} name]} {gap $label name_error message $name; continue}
        if {$name eq ""} {gap $label empty_name; continue}
        if {$total <= $max_names} {lappend names $name}
    }
    if {$total > $max_names} {gap $label name_cap total $total retained [llength $names]}
    if {[llength [lsort -unique $names]] != [llength $names]} {gap $label duplicate_names}
    emit query $label status observed total $total names $names
    return $names
}
proc ::c1_native_capture::selection {kind name} {
    switch $kind {
        pin {set names [qnames [list select $kind $name] [list get_pins -quiet [list $name]]]}
        cell {set names [qnames [list select $kind $name] [list get_cells -quiet [list $name]]]}
        net {set names [qnames [list select $kind $name] [list get_nets -quiet [list $name]]]}
        default {error "Unsupported explicit object kind $kind"}
    }
    if {$names ne [list $name]} {emit selection $kind $name status missing_or_ambiguous observed $names; return 0}
    return 1
}
proc ::c1_native_capture::properties {kind name keys} {
    variable seen_properties
    set cache [list $kind $name $keys]
    if {[info exists seen_properties($cache)]} {return $seen_properties($cache)}
    set values {}
    if {![selection $kind $name]} {gap [list properties $kind $name] absent_selection; return $values}
    # Re-resolve typed objects at EVERY operation boundary. Only names cross procs.
    # Do not reconstruct a command containing typed handles: run it directly.
    if {[catch {
        switch $kind {
            pin {set available [list_property [get_pins -quiet [list $name]]]}
            cell {set available [list_property [get_cells -quiet [list $name]]]}
            net {set available [list_property [get_nets -quiet [list $name]]]}
        }
    } message]} {gap [list properties $kind $name] api_error message $message; return $values}
    foreach key $keys {
        if {$key ni $available} {emit property $kind $name $key status unavailable; continue}
        if {[catch {
            switch $kind {
                pin {set value [get_property $key [get_pins -quiet [list $name]]]}
                cell {set value [get_property $key [get_cells -quiet [list $name]]]}
                net {set value [get_property $key [get_nets -quiet [list $name]]]}
            }
        } message]} {gap [list property $kind $name $key] api_error message $message; continue}
        dict set values $key $value; emit property $kind $name $key status observed value $value
    }
    set seen_properties($cache) $values
    return $values
}
proc ::c1_native_capture::cell {name {details 1}} {
    variable seen_cells;variable seen_details
    if {[info exists seen_cells($name)]} {set props $seen_cells($name)} else {
    set props [properties cell $name {NAME REF_NAME ORIG_REF_NAME IS_PRIMITIVE IS_SEQUENTIAL IS_BLACKBOX INIT ORIG_CELL_NAME PHYS_OPT_MODIFIED IS_C_INVERTED IS_D_INVERTED IS_R_INVERTED IS_CLR_INVERTED}]
    set seen_cells($name) $props
    }
    if {!$details || [info exists seen_details($name)]} {return $props}
    set seen_details($name) 1
    if {![selection cell $name]} {return $props}
    set pins [qnames [list cell_pins $name] {get_pins -quiet -of_objects [get_cells -quiet [list $name]]}]
    foreach pin $pins {
        properties pin $pin {NAME REF_PIN_NAME DIRECTION IS_CLOCK IS_INVERTED}
        owners $pin
        qnames [list pin_clock $pin] {get_clocks -quiet -of_objects [get_pins -quiet [list $pin]]}
    }
    emit cell_pins $name names $pins
    return $props
}
# Owner names are stable within this immutable open checkpoint. Cache names,
# never typed handles, and retain the first exact query as raw evidence.
proc ::c1_native_capture::owners {pin} {
    variable seen_owners
    if {![info exists seen_owners($pin)]} {
        set seen_owners($pin) [qnames [list pin_owner $pin] {get_cells -quiet -of_objects [get_pins -quiet [list $pin]]}]
    }
    return $seen_owners($pin)
}
proc ::c1_native_capture::network {kind name {details 1}} {
    variable seen_networks
    set cache [list $kind $name $details]
    if {[info exists seen_networks($cache)]} {return $seen_networks($cache)}
    if {![selection $kind $name]} {return {}}
    if {$kind eq "net"} {
        # get_nets -of_objects does not accept a net in this Vivado version.
        set nets [list $name]
    } elseif {$kind eq "pin"} {
        set nets [qnames [list segments $name] {get_nets -quiet -of_objects [get_pins -quiet [list $name]]}]
    } else {error "Network anchor must be explicitly pin or net"}
    set leaves {}; set drivers {}; set ports {}
    foreach net $nets {
        if {![selection net $net]} {gap [list network $kind $name] changed_net; continue}
        foreach pin [qnames [list leaves $net] {get_pins -quiet -leaf -of_objects [get_nets -quiet [list $net]]}] {dict set leaves $pin 1}
        foreach port [qnames [list ports $net] {get_ports -quiet -of_objects [get_nets -quiet [list $net]]}] {dict set ports $port 1}
    }
    foreach pin [dict keys $leaves] {
        set props [properties pin $pin {NAME REF_PIN_NAME DIRECTION IS_CLOCK IS_INVERTED}]
        if {![dict exists $props DIRECTION]} {gap [list network $kind $name $pin] missing_direction; continue}
        if {[dict get $props DIRECTION] eq "OUT"} {lappend drivers $pin}
        set owners [owners $pin]
        emit leaf $kind $name pin $pin properties $props owners $owners
        if {$details} {foreach owner $owners {cell $owner}}
    }
    set result [dict create nets $nets leaves [lsort [dict keys $leaves]] drivers [lsort $drivers] ports [lsort [dict keys $ports]]]
    emit network $kind $name {*}$result
    set seen_networks($cache) $result; return $result
}
# Backward cones need the selected input and all its drivers, not every sink
# of a global constant/reset net. Keep this scope explicit in every raw edge.
proc ::c1_native_capture::input_drivers {name} {
    if {![selection pin $name]} {gap [list input_drivers $name] missing_input;return {}}
    set props [properties pin $name {NAME REF_PIN_NAME DIRECTION}]
    set nets [qnames [list input_nets $name] {get_nets -quiet -of_objects [get_pins -quiet [list $name]]}]
    set drivers {};set ports {}
    foreach net $nets {
        foreach pin [qnames [list input_drivers $name $net] {get_pins -quiet -leaf -of_objects [get_nets -quiet [list $net]] -filter {DIRECTION == OUT}}] {dict set drivers $pin 1}
        foreach port [qnames [list input_ports $name $net] {get_ports -quiet -of_objects [get_nets -quiet [list $net]]}] {dict set ports $port 1}
    }
    foreach driver [dict keys $drivers] {
        properties pin $driver {NAME REF_PIN_NAME DIRECTION}
        foreach owner [owners $driver] {cell $owner 0}
    }
    return [dict create scope selected_input_and_drivers input $name properties $props nets $nets drivers [lsort [dict keys $drivers]] ports [lsort [dict keys $ports]]]
}
proc ::c1_native_capture::cone {label driver_names} {
    variable max_cells; variable max_depth
    set queue {};set visited {};set terminal {}
    foreach pin $driver_names {lappend queue [list $pin 0]}
    while {[llength $queue]} {
        lassign [lindex $queue 0] pin depth;set queue [lrange $queue 1 end]
        if {[dict exists $visited $pin]} {continue}
        if {[dict size $visited] >= $max_cells} {gap [list cone $label] cell_cap remaining [llength $queue] next $pin;break}
        dict set visited $pin $depth
        if {![selection pin $pin]} {gap [list cone $label] missing_driver pin $pin;continue}
        set owners [owners $pin]
        if {[llength $owners]!=1} {gap [list cone $label] ambiguous_owner pin $pin;continue}
        set owner [lindex $owners 0];set props [cell $owner]
        if {![dict exists $props REF_NAME] || ![dict exists $props IS_SEQUENTIAL]} {gap [list cone $label] missing_cell_properties cell $owner;continue}
        set ref [dict get $props REF_NAME];set seq [dict get $props IS_SEQUENTIAL]
        if {$ref in {GND VCC} || $seq in {1 true TRUE}} {
            dict set terminal $pin $ref;emit cone_terminal $label pin $pin cell $owner reference $ref;continue
        }
        if {![regexp {^LUT[1-6]$} $ref]} {gap [list cone $label] unsupported_combinational reference $ref cell $owner;continue}
        if {$depth >= $max_depth} {gap [list cone $label] depth_cap cell $owner depth $depth;continue}
        set inputs [qnames [list cone_inputs $label $owner] {get_pins -quiet -of_objects [get_cells -quiet [list $owner]] -filter {DIRECTION == IN}}]
        foreach input $inputs {
            set n [input_drivers $input];emit cone_edge $label cell $owner input $input network $n
            if {![dict exists $n drivers] || ![llength [dict get $n drivers]]} {gap [list cone $label] missing_input_driver pin $input;continue}
            foreach driver [dict get $n drivers] {lappend queue [list $driver [expr {$depth+1}]]}
        }
    }
    emit cone $label visited $visited terminals $terminal
}
proc ::c1_native_capture::anchor {label kind name {walk 0}} {
    emit anchor $label kind $kind name $name
    set n [network $kind $name]
    if {$walk && [dict exists $n drivers]} {cone $label [dict get $n drivers]}
    return $n
}
# Endpoint semantics require direction, exact owner and primitive reference.
# Iterate typed bulk selections directly instead of re-selecting each register
# for thirteen unrelated optional properties. Full gate/cone properties remain
# collected by cell()/cone(); the entire endpoint inventory is still retained.
proc ::c1_native_capture::endpoint_facts {pc output endpoints} {
    variable seen_owners
    set selected [qnames [list receive_endpoint_pins $pc $output] {get_pins -quiet $endpoints}]
    if {[lsort $selected] ne [lsort $endpoints]} {gap [list receive_endpoints $pc $output] unresolved_endpoint_selection}
    foreach pin [get_pins -quiet $endpoints] {
        set name [get_property NAME $pin]
        emit property pin $name DIRECTION status observed value [get_property DIRECTION $pin]
        if {![info exists seen_owners($name)]} {
            set seen_owners($name) [qnames [list pin_owner $name] {get_cells -quiet -of_objects $pin}]
        }
    }
    set selected_cells [qnames [list receive_endpoint_cells $pc $output] {get_cells -quiet -of_objects [get_pins -quiet $endpoints]}]
    foreach object [get_cells -quiet -of_objects [get_pins -quiet $endpoints]] {
        set name [get_property NAME $object]
        set ref [get_property REF_NAME $object]
        emit property cell $name NAME status observed value $name
        emit property cell $name REF_NAME status observed value $ref
        if {$ref in {RAMD32 RAMS32} && [regexp {^WRAPPER/CL/u_board/adapter/engines\[([0-9]+)\]\.tag_q} $name -> engine]} {
            tag_memory $engine
        }
    }
}
proc ::c1_native_capture::tag_memory {engine} {
    variable seen_tag
    if {[info exists seen_tag($engine)]} {return}
    set seen_tag($engine) 1
    if {$engine<0 || $engine>=16} {gap [list tag_memory $engine] outside_engine_scope;return}
    set pattern [format {^WRAPPER/CL/u_board/adapter/engines\[%d\]\.tag_q.*$} $engine]
    set cells [qnames [list tag_memory_inventory $engine] {get_cells -quiet -hierarchical -regexp -filter "NAME =~ \"$pattern\""}]
    if {![llength $cells] || [llength $cells]>256} {gap [list tag_memory $engine] inventory_bound;return}
    foreach name $cells {
        set cp [properties cell $name {NAME REF_NAME IS_PRIMITIVE}]
        if {![dict exists $cp REF_NAME]} {continue}
        set ref [dict get $cp REF_NAME]
        if {$ref in {RAM32M RAM32M16}} {continue}
        if {$ref ni {RAMD32 RAMS32}} {gap [list tag_memory $engine $name] unknown_tag_object reference $ref;continue}
        foreach terminal {WE CLK} {
            set pin $name/$terminal
            properties pin $pin {NAME REF_PIN_NAME DIRECTION IS_INVERTED}
            owners $pin
            emit cone_edge [list tag_memory $engine] cell $name input $pin network [input_drivers $pin]
            qnames [list tag_memory_clock $pin] {get_clocks -quiet -of_objects [get_pins -quiet [list $pin]]}
        }
    }
    qnames tag_memory_reference_clock {get_clocks -quiet -of_objects [get_pins -quiet WRAPPER/CL/core_rst_reg/C]}
}
# The exact relocated read-accounting LUT is already observed in the full
# native RVALID leaf inventory. Preserve its endpoint observation, but do not
# traverse its unrelated address/counter arithmetic as a receive/reset proof.
proc ::c1_native_capture::receive_cone {pc owner output} {
    set suffix [expr {$pc == 0 ? "" : "__[expr {$pc-1}]"}]
    set accounting [format {WRAPPER/CL/u_mem/reads[7]_i_1%s} $suffix]
    if {$pc >= 0 && $pc < 8 && $owner eq $accounting && $output eq "$owner/O"} {
        emit cone_scope $pc cell $owner output $output input $owner/I0 source_network WRAPPER/CL/n${pc}_rvalid direction backward disposition not_collected purpose relocated_read_accounting
    } else {
        cone [list receive $pc $owner] [list $output]
    }
}
proc ::c1_native_capture::receive {pc n} {
    if {![dict exists $n leaves]} {return}
    set candidates {}
    foreach pin [dict get $n leaves] {
        set props [properties pin $pin {DIRECTION}]
        if {![dict exists $props DIRECTION] || [dict get $props DIRECTION] ne "IN"} {continue}
        foreach owner [owners $pin] {
            set cp [cell $owner]
            if {[dict exists $cp REF_NAME] && [regexp {^LUT[1-6]$} [dict get $cp REF_NAME]]} {dict set candidates $owner 1}
        }
    }
    if {[dict size $candidates]>64} {gap [list receive $pc] candidate_cap total [dict size $candidates];return}
    foreach owner [dict keys $candidates] {
        set outputs [qnames [list receive_outputs $pc $owner] {get_pins -quiet -of_objects [get_cells -quiet [list $owner]] -filter {DIRECTION == OUT}}]
        foreach output $outputs {
            set endpoints [qnames [list receive_endpoints $pc $output] {all_fanout -flat -endpoints_only -from [get_pins -quiet [list $output]]}]
            endpoint_facts $pc $output $endpoints
            emit receive_candidate $pc cell $owner output $output endpoints $endpoints
            receive_cone $pc $owner $output
        }
    }
}
# C1 native writes are disabled. Inspect only each actual input and its
# drivers, without enumerating every sink of the shared global GND network.
# This is an explicitly scoped observation, not a complete constant-net cone.
proc ::c1_native_capture::disabled {pc channel name} {
    emit anchor [list disabled_write $pc $channel] kind pin name $name
    set n [input_drivers $name]
    emit disabled_write $pc channel $channel network $n
}
proc ::c1_native_capture::body {} {
    set root WRAPPER/CL;set sc $root/u_mem/f2_mem_i/memory_sc/inst
    for {set pc 0} {$pc<8} {incr pc} {
        foreach signal {arvalid arready rvalid bvalid} {
            set n [anchor [list native $pc $signal] net ${root}/n${pc}_$signal]
            if {$signal eq "rvalid"} {receive $pc $n}
        }
        set si [format s%02d [expr {$pc+1}]]
        foreach channel {aw w} {
            set name $sc/${si}_entry_pipeline/${si}_mmu/inst/${channel}_sreg/s_axi_${channel}valid
            disabled $pc $channel $name
        }
        foreach channel {b r} {
            anchor [list native_ready_alias $pc $channel] net ${root}/n${pc}_${channel}ready 1
            set scope $sc/${si}_entry_pipeline/${si}_mmu/inst/${channel}_sreg
            if {![selection cell $scope]} {continue}
            set pins [qnames [list ready_boundary $pc $channel] {get_pins -quiet -of_objects [get_cells -quiet [list $scope]]}]
            foreach pin $pins {if {[string match *ready* [file tail $pin]]} {anchor [list ready_boundary $pc $channel] pin $pin 1}}
        }
    }
    # Reset-Q can drive many unrelated C1 state registers. Capture only its
    # actual B/R-bank leaves here; the separate reset/status proof owns full
    # reset fanout. Do not expand all reset net aliases or all custom sinks.
    set reset $root/core_rst_reg/Q
    properties pin $reset {NAME REF_PIN_NAME DIRECTION}
    if {[selection pin $reset]} {
        set nets [qnames reset_ready_nets {get_nets -quiet -of_objects [get_pins -quiet [list $reset]]}]
        foreach net $nets {
            set pins [qnames reset_ready_bank_scope {get_pins -quiet -leaf -of_objects [get_nets -quiet [list $net]] -filter {NAME =~ WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/s0*_entry_pipeline/s0*_mmu/inst/*_sreg/*}}]
            foreach pin $pins {
                if {[regexp {^WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/s0[1-8]_entry_pipeline/s0[1-8]_mmu/inst/[br]_sreg/} $pin]} {
                    foreach owner [owners $pin] {
                        foreach output [qnames [list reset_bank_output $owner] {get_pins -quiet -of_objects [get_cells -quiet [list $owner]] -filter {DIRECTION == OUT}}] {
                            anchor [list ready_reset_bank $owner] pin $output 1
                        }
                    }
                }
            }
        }
    }
    foreach pin [list $root/u_pcis_pipe/m_axi_wready ${root}/u_pcis_pipe/m_axi_wready\[0\]] {
        if {[selection pin $pin]} {anchor host_wready pin $pin 1}
    }
}
proc ::c1_native_capture::hash {path} {
    set result [exec sha256sum -- $path]
    if {![regexp {^([0-9a-f]{64})\s} $result -> digest]} {error "Malformed checkpoint hash"}
    return $digest
}
proc ::c1_native_capture::capture {run checkpoint expected_sha256} {
    variable out;variable gaps;variable queries;variable seen_cells;variable seen_details;variable seen_networks;variable seen_properties;variable seen_owners;variable seen_tag;variable written;variable exhausted
    if {![regexp {^[0-9a-f]{64}$} $expected_sha256] || ![file isfile $checkpoint] || [hash $checkpoint] ne $expected_sha256} {error "Checkpoint binding mismatch"}
    if {[version -short] ne "2025.2" || [llength [get_designs -quiet]]} {error "Fresh Vivado 2025.2 process required"}
    if {[file exists $run]} {error "Refusing existing raw capture output"}
    file mkdir $run;set out [open [file join $run topology.tcldata] {WRONLY CREAT EXCL}]
    fconfigure $out -encoding utf-8 -translation lf
    set gaps 0;set queries 0;set written 0;set exhausted 0;array unset seen_cells;array unset seen_details;array unset seen_networks;array unset seen_properties;array unset seen_owners;array unset seen_tag
    emit header schema 1 kind c1_native_topology checkpoint_sha256 $expected_sha256 checkpoint $checkpoint route_qualified 0 hardware_measured 0
    set status [catch {set_param general.maxThreads 4;open_checkpoint $checkpoint;body} message options]
    if {$status} {gap capture api_error message $message}
    catch {close_design}
    if {[hash $checkpoint] ne $expected_sha256} {gap capture checkpoint_changed;set status 1;set message "Checkpoint changed"}
    puts $out [list summary queries $queries incomplete_or_error $gaps output_byte_cap $exhausted route_qualified 0 hardware_measured 0]
    close $out
    if {$status} {error $message}
    return [dict create output [file join $run topology.tcldata] incomplete_or_error $gaps route_qualified 0]
}
