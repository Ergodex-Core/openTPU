# Additive raw C1 read-credit accounting observations. No constraints or acceptance.
# source, then ::c1_accounting_capture::capture RUN DCP SHA256
source [file join [file dirname [info script]] capture_native.tcl]
namespace eval ::c1_accounting_capture {
    variable helper [file normalize [info script]]
    variable max_nodes 1024
    variable max_depth 48
}
foreach proc {emit gap qnames selection properties cell owners network input_drivers hash} {
    interp alias {} ::c1_accounting_capture::$proc {} ::c1_native_capture::$proc
}
proc ::c1_accounting_capture::walk {label drivers} {
    variable max_nodes; variable max_depth
    set queue {};set visited {}
    foreach pin $drivers {lappend queue [list $pin 0]}
    while {[llength $queue]} {
        lassign [lindex $queue 0] output depth;set queue [lrange $queue 1 end]
        if {[dict exists $visited $output]} {continue}
        if {[dict size $visited]>=$max_nodes} {gap [list accounting_walk $label] node_cap;break}
        dict set visited $output $depth
        if {![selection pin $output]} {gap [list accounting_walk $label] missing_output pin $output;continue}
        set os [owners $output]
        if {[llength $os]!=1} {gap [list accounting_walk $label] ambiguous_owner;continue}
        set owner [lindex $os 0];set cp [cell $owner]
        if {![dict exists $cp REF_NAME] || ![dict exists $cp IS_SEQUENTIAL]} {gap [list accounting_walk $label] missing_properties;continue}
        set ref [dict get $cp REF_NAME]
        if {$ref in {GND VCC} || [dict get $cp IS_SEQUENTIAL] in {1 true TRUE}} {
            emit accounting_terminal $label output $output cell $owner reference $ref
            continue
        }
        if {![regexp {^LUT[1-6]$} $ref] && $ref ni {LUT6_2 CARRY8 MUXF7 MUXF8 MUXF9}} {
            gap [list accounting_walk $label] unsupported_combinational cell $owner reference $ref;continue
        }
        if {$ref eq "CARRY8"} {properties cell $owner {CARRY_TYPE}}
        if {$depth>=$max_depth} {gap [list accounting_walk $label] depth_cap;continue}
        set inputs [qnames [list accounting_inputs $label $owner] {get_pins -quiet -of_objects [get_cells -quiet [list $owner]] -filter {DIRECTION == IN}}]
        foreach input $inputs {
            set n [input_drivers $input]
            emit accounting_edge $label cell $owner input $input network $n
            if {![dict exists $n drivers] || [llength [dict get $n drivers]]!=1 || [llength [dict get $n ports]]} {gap [list accounting_walk $label] missing_ambiguous_or_external_input pin $input;continue}
            lappend queue [list [lindex [dict get $n drivers] 0] [expr {$depth+1}]]
        }
    }
    emit accounting_walk $label outputs [dict keys $visited]
}
proc ::c1_accounting_capture::body {} {
    for {set pc 0} {$pc<8} {incr pc} {
        set suffix [expr {$pc==0 ? "" : "__[expr {$pc-1}]"}]
        set gate [format {WRAPPER/CL/u_mem/reads[7]_i_1%s} $suffix]
        emit accounting_gate $pc cell $gate
        cell $gate
        set output [network pin $gate/O 0]
        emit accounting_gate_output $pc network $output
        foreach role {I0 I1} {
            set n [input_drivers $gate/$role]
            emit accounting_gate_input $pc role $role network $n
            if {![dict exists $n drivers] || [llength [dict get $n drivers]]!=1 || ![dict exists $n ports] || [llength [dict get $n ports]]} {gap [list accounting_gate_input $pc $role] missing_ambiguous_or_external_input;continue}
            walk [list gate $pc $role] [dict get $n drivers]
        }
        set scope WRAPPER/CL/u_native$pc
        set regs [qnames [list accounting_register_inventory $pc] {get_cells -quiet -hierarchical -filter "IS_SEQUENTIAL && NAME =~ $scope/reads_reg*"}]
        emit accounting_registers $pc cells $regs
        if {[llength $regs]!=8} {gap [list accounting_registers $pc] count_differs;continue}
        foreach reg $regs {
            cell $reg
            foreach role {C CE R D} {
                set n [input_drivers $reg/$role]
                emit accounting_control $pc cell $reg role $role network $n
                if {![dict exists $n drivers] || [llength [dict get $n drivers]]!=1 || ![dict exists $n ports] || [llength [dict get $n ports]]} {gap [list accounting_control $pc $reg $role] missing_ambiguous_or_external_input;continue}
                if {$role ne "C"} {walk [list control $pc $reg $role] [dict get $n drivers]}
            }
        }
        network net WRAPPER/CL/n${pc}_rvalid 0
    }
}
proc ::c1_accounting_capture::capture {run checkpoint expected} {
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
    set ::c1_native_capture::out [open [file join $run accounting.tcldata] {WRONLY CREAT EXCL}]
    fconfigure $::c1_native_capture::out -encoding utf-8 -translation lf
    emit header schema 1 kind c1_read_accounting_topology checkpoint_sha256 $expected checkpoint $checkpoint route_qualified 0 hardware_measured 0
    emit accounting_sources helper_sha256 $helper_sha native_helper_sha256 $native_sha
    set status [catch {set_param general.maxThreads 2;open_checkpoint $checkpoint;body} message]
    if {$status} {gap accounting_capture api_error message $message}
    catch {close_design}
    if {[hash $checkpoint] ne $expected || [hash $helper] ne $helper_sha || [hash $native] ne $native_sha} {gap accounting_capture input_changed;set status 1;set message "Capture input changed"}
    puts $::c1_native_capture::out [list summary queries $::c1_native_capture::queries incomplete_or_error $::c1_native_capture::gaps output_byte_cap $::c1_native_capture::exhausted route_qualified 0 hardware_measured 0]
    close $::c1_native_capture::out
    if {$status} {error $message}
    return [dict create output [file join $run accounting.tcldata] incomplete_or_error $::c1_native_capture::gaps route_qualified 0]
}
