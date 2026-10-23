# C1-only supplemental raw ready topology. No acceptance or constraint edits.
# source, then ::c1_ready_capture::capture RUN CHECKPOINT SHA256
source [file join [file dirname [info script]] capture_native.tcl]
namespace eval ::c1_ready_capture {
    variable helper [file normalize [info script]]
    variable max_nodes 512
    variable max_depth 16
    variable roots {}
}
foreach proc {emit gap qnames selection properties cell owners network input_drivers hash} {
    interp alias {} ::c1_ready_capture::$proc {} ::c1_native_capture::$proc
}
proc ::c1_ready_capture::state_scope {name} {
    return [expr {
        [regexp {^WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/s0[1-8]_entry_pipeline/s0[1-8]_mmu/inst/[br]_sreg/} $name] ||
        [string match {WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/s00_entry_pipeline/s00_mmu/inst/w_sreg/*} $name] ||
        [string match {WRAPPER/CL/u_pcis_pipe/*} $name]
    }]
}
proc ::c1_ready_capture::walk {label drivers {controls 1}} {
    variable max_nodes;variable max_depth
    set queue {};set visited {};set registers {}
    foreach pin $drivers {lappend queue [list $pin 0]}
    while {[llength $queue]} {
        lassign [lindex $queue 0] output depth;set queue [lrange $queue 1 end]
        if {[dict exists $visited $output]} {continue}
        if {[dict size $visited]>=$max_nodes} {gap [list ready_walk $label] node_cap;break}
        dict set visited $output $depth
        if {![selection pin $output]} {gap [list ready_walk $label] missing_output pin $output;continue}
        properties pin $output {NAME REF_PIN_NAME DIRECTION IS_INVERTED}
        set owner_list [owners $output]
        if {[llength $owner_list]!=1} {gap [list ready_walk $label] ambiguous_owner;continue}
        set owner [lindex $owner_list 0];set p [cell $owner]
        if {![dict exists $p REF_NAME]} {gap [list ready_walk $label] missing_reference;continue}
        set ref [dict get $p REF_NAME]
        if {$ref in {GND VCC}} {
            emit ready_terminal $label output $output cell $owner reference $ref
            continue
        }
        if {[dict exists $p IS_SEQUENTIAL] && [dict get $p IS_SEQUENTIAL] in {1 true TRUE}} {
            emit ready_terminal $label output $output cell $owner reference $ref
            if {[state_scope $owner]} {dict set registers $owner 1}
            continue
        }
        if {![regexp {^LUT[1-6]$} $ref]} {gap [list ready_walk $label] unsupported_combinational cell $owner reference $ref;continue}
        network pin $output 0
        if {$depth>=$max_depth} {gap [list ready_walk $label] depth_cap cell $owner;continue}
        set inputs [qnames [list ready_inputs $label $owner] {get_pins -quiet -of_objects [get_cells -quiet [list $owner]] -filter {DIRECTION == IN}}]
        foreach input $inputs {
            set n [input_drivers $input]
            emit ready_edge $label cell $owner input $input network $n
            if {![dict exists $n drivers] || ![llength [dict get $n drivers]]} {gap [list ready_walk $label] missing_driver pin $input;continue}
            foreach driver [dict get $n drivers] {lappend queue [list $driver [expr {$depth+1}]]}
        }
    }
    emit ready_walk $label outputs [dict keys $visited] registers [dict keys $registers] expand_controls $controls
    if {$controls} {foreach reg [dict keys $registers] {register $label $reg}}
}
proc ::c1_ready_capture::register {label reg} {
    set props [cell $reg]
    if {![dict exists $props REF_NAME] || [dict get $props REF_NAME] ne "FDRE"} {
        gap [list ready_register $label $reg] unsupported_register;return
    }
    # Capture every output's local leaf inventory. Never walk global constant
    # sinks or clock fanout. Control-input cones terminate at their own registers.
    set outputs [qnames [list ready_register_outputs $label $reg] {get_pins -quiet -of_objects [get_cells -quiet [list $reg]] -filter {DIRECTION == OUT}}]
    foreach output $outputs {network pin $output 0}
    set clock [input_drivers $reg/C]
    emit ready_clock $label cell $reg network $clock
    foreach role {D CE R} {
        set input $reg/$role;set n [input_drivers $input]
        emit ready_control $label cell $reg role $role network $n
        if {[dict exists $n drivers] && [llength [dict get $n drivers]]} {
            walk [list control $label $reg $role] [dict get $n drivers] 0
        } else {gap [list ready_control $label $reg $role] missing_driver}
    }
    emit ready_register $label cell $reg outputs $outputs clock_scope names_and_properties_only
}
proc ::c1_ready_capture::root {label pin} {
    variable roots
    if {[dict exists $roots $label]} {error "Duplicate ready root"}
    dict set roots $label $pin
    emit ready_root $label pin $pin
    if {![selection pin $pin]} {gap [list ready_root $label] missing_pin;return}
    set n [network pin $pin 0]
    if {[dict exists $n drivers] && [llength [dict get $n drivers]]} {
        walk $label [dict get $n drivers]
    } else {gap [list ready_root $label] missing_driver}
}
proc ::c1_ready_capture::body {} {
    set cl WRAPPER/CL;set sc $cl/u_mem/f2_mem_i/memory_sc/inst
    for {set pc 0} {$pc<8} {incr pc} {
        set si [format s%02d [expr {$pc+1}]]
        foreach channel {b r} {
            root [list native $pc $channel] $sc/${si}_entry_pipeline/${si}_mmu/inst/${channel}_sreg/s_axi_${channel}ready
        }
    }
    set candidates {}
    foreach pin [list $cl/u_pcis_pipe/m_axi_wready ${cl}/u_pcis_pipe/m_axi_wready\[0\]] {
        set found [qnames [list ready_host_candidate $pin] {get_pins -quiet [list $pin]}]
        if {$found eq [list $pin]} {lappend candidates $pin}
    }
    emit ready_host_candidates pins $candidates
    if {[llength $candidates]!=1} {gap host_wready missing_or_ambiguous_anchor;return}
    root host_wready [lindex $candidates 0]
    # Preserve the complete scoped FDRE name inventory and inspect ready-named
    # candidates, including physical replicas. Other named state is explicitly
    # outside this supplemental discovery, never silently accepted.
    foreach scope [list $sc/s00_entry_pipeline/s00_mmu/inst/w_sreg $cl/u_pcis_pipe] {
        set all [qnames [list ready_host_register_inventory $scope] {get_cells -quiet -hierarchical -filter "REF_NAME == FDRE && NAME =~ $scope/*"}]
        set selected {}
        foreach reg $all {if {[string match -nocase *ready* $reg]} {lappend selected $reg}}
        emit ready_candidates scope $scope all $all selected $selected selection name_contains_ready
        if {[llength $selected]>64} {gap [list ready_candidates $scope] candidate_cap;continue}
        foreach reg $selected {register [list host_candidate $reg] $reg}
    }
}
proc ::c1_ready_capture::capture {run checkpoint expected_sha256} {
    variable roots;variable helper
    if {![regexp {^[0-9a-f]{64}$} $expected_sha256] || ![file isfile $checkpoint] || [hash $checkpoint] ne $expected_sha256} {error "Checkpoint binding mismatch"}
    if {[version -short] ne "2025.2" || [llength [get_designs -quiet]]} {error "Fresh Vivado 2025.2 process required"}
    if {[file exists $run]} {error "Refusing existing raw capture output"}
    set helper_sha [hash $helper];set base_sha [hash [file join [file dirname $helper] capture_native.tcl]]
    file mkdir $run
    namespace eval ::c1_native_capture {
        set gaps 0;set queries 0;set written 0;set exhausted 0
        array unset seen_cells;array unset seen_details;array unset seen_networks
        array unset seen_properties;array unset seen_owners
    }
    set roots {};set ::c1_native_capture::out [open [file join $run ready.tcldata] {WRONLY CREAT EXCL}]
    fconfigure $::c1_native_capture::out -encoding utf-8 -translation lf
    emit header schema 1 kind c1_ready_topology checkpoint_sha256 $expected_sha256 checkpoint $checkpoint route_qualified 0 hardware_measured 0
    emit ready_sources helper_sha256 $helper_sha native_helper_sha256 $base_sha
    set status [catch {set_param general.maxThreads 4;open_checkpoint $checkpoint;body} message]
    if {$status} {gap ready_capture api_error message $message}
    catch {close_design}
    if {[hash $checkpoint] ne $expected_sha256} {gap ready_capture checkpoint_changed;set status 1;set message "Checkpoint changed"}
    puts $::c1_native_capture::out [list summary queries $::c1_native_capture::queries incomplete_or_error $::c1_native_capture::gaps output_byte_cap $::c1_native_capture::exhausted route_qualified 0 hardware_measured 0]
    close $::c1_native_capture::out
    if {$status} {error $message}
    return [dict create output [file join $run ready.tcldata] incomplete_or_error $::c1_native_capture::gaps route_qualified 0]
}
