# Additive raw admission/state closure, never a qualification or constraint edit.
# Native interface inputs are explicit cuts. RAM outputs are expanded through
# every read/write address, data, enable and clock input instead of being treated
# as unconstrained sequential variables. Unknown state and limits remain gaps.
source [file join [file dirname [info script]] capture_native.tcl]
namespace eval ::c1_accounting_boundary {
    variable helper [file normalize [info script]]
    variable max_nodes 8192
    variable max_depth 128
}
foreach proc {emit gap qnames selection properties cell owners network input_drivers hash} {
    interp alias {} ::c1_accounting_boundary::$proc {} ::c1_native_capture::$proc
}
proc ::c1_accounting_boundary::single {label pin} {
    set n [input_drivers $pin]
    if {![dict exists $n drivers] || [llength [dict get $n drivers]]!=1 || ![dict exists $n ports] || [llength [dict get $n ports]]} {
        gap $label missing_ambiguous_or_external_input pin $pin
        return {}
    }
    return $n
}
# These exact top-level nets follow the frozen generated CL connections. A
# missing/renamed net is a gap, never an inferred substitute or arbitrary cut.
proc ::c1_accounting_boundary::interface_net {pc role} {
    if {$role eq "clk"} {return WRAPPER/CL/core_clk}
    if {$role eq "rst"} {return WRAPPER/CL/core_rst}
    if {$role in {n_cvalid n_cwe n_wvalid}} {return [format {WRAPPER/CL/%s[%d]} $role $pc]}
    if {[regexp {^n_caddr\[([0-9]+)\]$} $role -> bit]} {return [format {WRAPPER/CL/n_caddr[%d][%d]} $pc $bit]}
    if {$role in {m_awready m_wready m_bvalid m_arready m_rvalid}} {return WRAPPER/CL/n${pc}_[string range $role 2 end]}
    error "Unsupported explicit admission interface role"
}
proc ::c1_accounting_boundary::net_drivers {name} {
    if {![selection net $name]} {gap [list admission_net $name] missing_interface_net;return {}}
    set drivers [qnames [list admission_net_drivers $name] {get_pins -quiet -leaf -of_objects [get_nets -quiet [list $name]] -filter {DIRECTION == OUT}}]
    set ports [qnames [list admission_net_ports $name] {get_ports -quiet -of_objects [get_nets -quiet [list $name]]}]
    if {[llength $drivers]!=1 || [llength $ports]} {gap [list admission_net $name] missing_ambiguous_or_external_input;return {}}
    return [dict create scope selected_net_and_drivers net $name drivers $drivers ports $ports]
}
proc ::c1_accounting_boundary::interfaces {pc} {
    set scope WRAPPER/CL/u_native$pc
    set roles {clk rst n_cvalid m_bvalid m_arready m_rvalid}
    for {set bit 0} {$bit<24} {incr bit} {lappend roles [format {n_caddr[%d]} $bit]}
    set cuts {};set before $::c1_native_capture::gaps
    # These five source-interface nets are absent in the observed LINK design.
    # They create neither free inputs nor assumed constant values. The actual
    # cone must close through typed constants, native state and the live cuts.
    foreach role {n_cwe n_wvalid m_awready m_wready {n_caddr[24]}} {
        set net [interface_net $pc $role]
        set names [qnames [list admission_uncut_selection $pc $role] {get_nets -quiet [list $net]}]
        emit admission_uncut_role $pc role $role kind net name $net observed $names disposition no_assumed_input_or_constant
        if {[llength $names]} {gap [list admission_interface $pc $role] unexpected_retained_uncut_role}
    }
    foreach role $roles {
        set net [interface_net $pc $role]
        set n [net_drivers $net]
        emit admission_interface $pc role $role kind net name $net network $n
        qnames [list admission_interface_clocks $pc $role] {get_clocks -quiet -of_objects [get_nets -quiet [list $net]]}
        if {[dict exists $n drivers]} {
            set output [lindex [dict get $n drivers] 0]
            dict lappend cuts $output $role
            properties pin $output {NAME REF_PIN_NAME DIRECTION IS_CLOCK IS_INVERTED}
            set os [owners $output]
            if {[llength $os]!=1} {gap [list admission_interface $pc $role] ambiguous_owner;continue}
            set owner [lindex $os 0];set cp [cell $owner 0]
            if {[string match $scope/* $owner] && (![dict exists $cp REF_NAME] || [dict get $cp REF_NAME] ni {GND VCC})} {
                gap [list admission_interface $pc $role] internal_nonconstant_interface_driver cell $owner
            }
        }
    }
    set complete [expr {$::c1_native_capture::gaps==$before}]
    emit admission_cut_inventory $pc complete $complete outputs $cuts
    return [dict create complete $complete cuts $cuts]
}
proc ::c1_accounting_boundary::walk {pc root cuts} {
    variable max_nodes;variable max_depth
    set scope WRAPPER/CL/u_native$pc
    set queue [list [list $root 0]];set visited {};set expanded {};set states {};set memory {}
    while {[llength $queue]} {
        lassign [lindex $queue 0] output depth;set queue [lrange $queue 1 end]
        if {[dict exists $visited $output]} {continue}
        if {[dict size $visited]>=$max_nodes} {gap [list admission_walk $pc] node_cap;break}
        dict set visited $output $depth
        if {![selection pin $output]} {gap [list admission_walk $pc] missing_output pin $output;continue}
        set os [owners $output]
        if {[llength $os]!=1} {gap [list admission_walk $pc] ambiguous_owner pin $output;continue}
        set owner [lindex $os 0];set props [cell $owner]
        if {![dict exists $props REF_NAME] || ![dict exists $props IS_SEQUENTIAL]} {gap [list admission_walk $pc] missing_properties;continue}
        set ref [dict get $props REF_NAME]
        if {$ref in {GND VCC}} {
            emit admission_constant $pc output $output cell $owner reference $ref
            continue
        }
        if {[dict exists $cuts $output]} {
            emit admission_cut $pc output $output cell $owner reference $ref roles [dict get $cuts $output]
            continue
        }
        set sequential [expr {[dict get $props IS_SEQUENTIAL] in {1 true TRUE}}]
        if {$sequential} {
            if {![string match $scope/* $owner]} {gap [list admission_walk $pc] state_outside_native_scope cell $owner reference $ref;continue}
            if {$ref ni {FDRE RAMD32 RAMS32}} {gap [list admission_walk $pc] unsupported_state cell $owner reference $ref;continue}
            if {$ref eq "FDRE"} {
                dict set states $owner 1
            } else {
                dict set memory $owner 1
                properties cell $owner {INIT IS_CLK_INVERTED IS_WCLK_INVERTED}
            }
        } elseif {![regexp {^LUT[1-6]$} $ref] && $ref ni {LUT6_2 CARRY8 MUXF7 MUXF8 MUXF9}} {
            gap [list admission_walk $pc] unsupported_combinational cell $owner reference $ref
            continue
        }
        # Classify before querying consumers: an unsupported clock buffer or
        # external state must remain a gap without expanding global fanout.
        # This captures leaves of selected output nets, not -segments closure.
        set output_network [network pin $output 0]
        emit admission_output $pc output $output network $output_network
        if {[dict exists $expanded $owner]} {continue}
        dict set expanded $owner 1
        if {$ref eq "CARRY8"} {properties cell $owner {CARRY_TYPE}}
        if {$depth>=$max_depth} {gap [list admission_walk $pc] depth_cap cell $owner;continue}
        set inputs [qnames [list admission_inputs $pc $owner] {get_pins -quiet -of_objects [get_cells -quiet [list $owner]] -filter {DIRECTION == IN}}]
        set outputs [qnames [list admission_outputs $pc $owner] {get_pins -quiet -of_objects [get_cells -quiet [list $owner]] -filter {DIRECTION == OUT}}]
        emit admission_cell $pc cell $owner reference $ref sequential $sequential inputs $inputs outputs $outputs
        if {![llength $inputs] || ![llength $outputs]} {gap [list admission_walk $pc] missing_input_output_inventory cell $owner;continue}
        # All outputs of an admitted state object are inventoried and delivered;
        # memories are never reduced to a name-only O-pin boundary.
        if {$sequential} {foreach p $outputs {lappend queue [list $p $depth]}}
        foreach input $inputs {
            set n [single [list admission_edge $pc $owner $input] $input]
            emit admission_edge $pc cell $owner input $input network $n
            if {[dict exists $n drivers]} {lappend queue [list [lindex [dict get $n drivers] 0] [expr {$depth+1}]]}
        }
    }
    emit admission_walk $pc root $root outputs [dict keys $visited] cells [dict keys $expanded] registers [dict keys $states] memories [dict keys $memory]
}
proc ::c1_accounting_boundary::body {} {
    # A bounded name-only diagnostic preserves actual top-level signal spelling
    # when synthesis has removed or renamed one of the exact interface anchors.
    qnames admission_interface_net_inventory {get_nets -quiet -hier -filter {NAME =~ WRAPPER/CL/n_c* || NAME =~ WRAPPER/CL/n_wvalid* || NAME == WRAPPER/CL/core_clk || NAME == WRAPPER/CL/core_rst}}
    for {set pc 0} {$pc<8} {incr pc} {
        set interface [interfaces $pc]
        if {![dict get $interface complete]} {
            gap [list admission_walk $pc] incomplete_interface_inventory
            continue
        }
        set cuts [dict get $interface cuts]
        set suffix [expr {$pc==0 ? "" : "__[expr {$pc-1}]"}]
        set gate [format {WRAPPER/CL/u_mem/reads[7]_i_1%s} $suffix]
        cell $gate
        set n [single [list admission_root $pc] $gate/I1]
        emit admission_root $pc gate $gate input $gate/I1 network $n
        if {[dict exists $n drivers]} {walk $pc [lindex [dict get $n drivers] 0] $cuts}
    }
}
proc ::c1_accounting_boundary::capture {run checkpoint expected} {
    variable helper
    if {![regexp {^[0-9a-f]{64}$} $expected] || ![file isfile $checkpoint] || [hash $checkpoint] ne $expected} {error "Checkpoint binding mismatch"}
    if {[version -short] ne "2025.2" || [llength [get_designs -quiet]]} {error "Fresh Vivado 2025.2 process required"}
    if {[file exists $run]} {error "Refusing existing raw capture output"}
    set helper_sha [hash $helper];set native [file join [file dirname $helper] capture_native.tcl];set native_sha [hash $native]
    namespace eval ::c1_native_capture {
        set gaps 0;set queries 0;set written 0;set exhausted 0
        array unset seen_cells;array unset seen_details;array unset seen_networks
        array unset seen_properties;array unset seen_owners;array unset seen_tag
    }
    file mkdir $run
    set ::c1_native_capture::out [open [file join $run admission.tcldata] {WRONLY CREAT EXCL}]
    fconfigure $::c1_native_capture::out -encoding utf-8 -translation lf
    emit header schema 1 kind c1_accounting_boundary_topology checkpoint_sha256 $expected checkpoint $checkpoint route_qualified 0 hardware_measured 0
    emit admission_sources helper_sha256 $helper_sha native_helper_sha256 $native_sha
    set status [catch {set_param general.maxThreads 2;open_checkpoint $checkpoint;body} message]
    if {$status} {gap admission_capture api_error message $message}
    catch {close_design}
    if {[hash $checkpoint] ne $expected || [hash $helper] ne $helper_sha || [hash $native] ne $native_sha} {gap admission_capture input_changed;set status 1;set message "Capture input changed"}
    puts $::c1_native_capture::out [list summary queries $::c1_native_capture::queries incomplete_or_error $::c1_native_capture::gaps output_byte_cap $::c1_native_capture::exhausted route_qualified 0 hardware_measured 0]
    close $::c1_native_capture::out
    if {$status} {error $message}
    return [dict create output [file join $run admission.tcldata] incomplete_or_error $::c1_native_capture::gaps route_qualified 0]
}
