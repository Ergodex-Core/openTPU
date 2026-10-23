# C1 raw final-checkpoint collection. This cannot qualify or modify a design.
# Existing structural errors are retained while independent reports are saved.
source [file join [file dirname [info script]] qualify_c1_alex.tcl]
source [file join [file dirname [info script]] capture_unused.tcl]
namespace eval ::c1_final_capture {
    variable library_path [file normalize [info script]]
    variable max_file_bytes 536870912
    variable max_total_bytes 1073741824
}
proc ::c1_final_capture::collect {directory checkpoint digest} {
    set audit [open $directory/objects.tcldata {WRONLY CREAT EXCL}]
    fconfigure $audit -encoding utf-8 -translation lf
    set code [catch {
        puts $audit [list checkpoint $checkpoint sha256 $digest]
        set blockers {};set structural [dict create state unresolved route_qualified 0]
        source [file join [file dirname $::c1_final_capture::library_path] routed_host_ready.tcl]
        if {[catch {::c1_alex_route::structure $audit 1} topology options]} {
            lappend blockers structural_review_incomplete
            dict set structural error $topology
            dict set structural error_info [dict get $options -errorinfo]
        } else {
            dict set structural state observed
            dict set structural topology $topology
        }
        puts $audit [list structural_capture $structural]
        set groups [::c1_alex_route::gray_inventory $audit]
        ::c1_alex_route::gray_witnesses $groups $audit
        set inventory [::c1_alex_route::report_set $directory]
        set exceptions [::c1_alex_route::constraints $directory/timing_all_ids.xdc $audit]
        set timing [::c1_alex_route::timing_summary [::c1_alex_route::read_text $directory/timing.rpt]]
        set route [::c1_alex_route::route_summary [::c1_alex_route::read_text $directory/route_status.rpt]]
        set rows [::c1_alex_route::bus_rows [::c1_alex_route::read_text $directory/bus_skew.rpt]]
        ::c1_alex_route::gray_coverage $groups $rows $audit
        dict for {id row} $rows {puts $audit [list bus_skew $id $row review pending_ip_scope_and_bounds]}
        set cdc [::c1_alex_route::cdc_rows [::c1_alex_route::read_text $directory/cdc.rpt]]
        foreach row $cdc {puts $audit [list cdc $row]}
        set findings [::c1_alex_route::findings $audit]
        set supplemental [::c1_alex_route::supplemental $directory $groups]
        set unused [::c1_unused_capture::collect $directory/unused_outputs.tcldata $digest]
        set blockers [concat $blockers [dict get $timing blockers] [dict get $findings blockers]]
        if {![dict get $route pass]} {lappend blockers routing_incomplete}
        if {[dict get $supplemental incomplete_or_error]} {lappend blockers supplemental_incomplete}
        dict for {id row} $rows {
            if {[dict get $row slack]<0 || [dict get $row requirement]<=0} {lappend blockers bus_skew:$id}
        }
        if {![regexp {(^|\n)No waivers found\.\s*$} [::c1_alex_route::read_text $directory/waivers.rpt]]} {lappend blockers waiver_review_required}
        foreach row $cdc {if {[dict get $row waived] ne "N"} {lappend blockers waived_cdc}}
        lappend inventory objects.tcldata supplemental.tcldata unused_outputs.tcldata
        set pending {
            current_structural_and_ready_equivalence
            current_vendor_CDC_and_gray_max_delay_scope
            current_exception_coverage_and_ignored_paths
            current_DRC_methodology_and_severity_configuration
            exact_shell_inherited_witnesses
            current_9SI_8MI_vendor_simulation_and_generated_IP_binding
            current_reset_status
        }
        set result [dict create schema 1 kind c1_final_raw_capture stage final_raw \
            route_qualified 0 hardware_measured 0 checkpoint $checkpoint checkpoint_sha256 $digest \
            structural_capture $structural blockers [lsort -unique $blockers] pending_reviews $pending \
            timing $timing routing $route findings $findings unused_output_capture $unused supplemental_evidence $supplemental \
            gray_bus_groups [dict size $groups] bus_skew_constraints [dict size $rows] \
            cdc_records [llength $cdc] exception_counts [dict get $exceptions counts] report_inventory $inventory]
    } message options]
    if {$code} {puts $audit [list capture_error message $message options $options route_qualified 0]}
    close $audit
    if {$code} {return -options $options $message}
    ::c1_alex_route::write_data $directory/summary.tcldata $result
    ::c1_alex_route::write_data $directory/NOT_QUALIFIED.txt [list reason raw_collection_only blockers [dict get $result blockers] pending_reviews $pending]
    return $result
}
proc ::c1_final_capture::capture {checkpoint expected directory} {
    variable library_path;variable max_file_bytes;variable max_total_bytes
    if {![regexp {^[0-9a-f]{64}$} $expected] || ![file isfile $checkpoint] || [file type $checkpoint] ne "file"} {error "Expected regular checkpoint and exact SHA256"}
    if {[::c1_alex_route::sha256 $checkpoint] ne $expected} {error "Checkpoint binding mismatch"}
    if {[version -short] ne "2025.2" || [llength [get_designs -quiet]]} {error "Fresh Vivado 2025.2 session required"}
    if {[file exists $directory] || ![file isdirectory [file dirname $directory]]} {error "Fresh output with existing parent required"}
    set helpers {}
    foreach name {capture_final.tcl qualify_c1_alex.tcl audit_c1_alex.tcl routed_host_ready.tcl capture_unused.tcl} {
        dict set helpers $name [::c1_alex_route::sha256 [file join [file dirname $library_path] $name]]
    }
    file mkdir $directory
    set_param general.maxThreads 4
    set opened 0
    set code [catch {
        open_checkpoint $checkpoint;set opened 1
        set result [collect $directory $checkpoint $expected]
    } message options]
    if {$opened} {close_design}
    if {[::c1_alex_route::sha256 $checkpoint] ne $expected} {error "Checkpoint changed during collection"}
    dict for {name digest} $helpers {
        if {[::c1_alex_route::sha256 [file join [file dirname $library_path] $name]] ne $digest} {error "Collector input changed"}
    }
    set total 0
    foreach path [glob -nocomplain -directory $directory *] {
        if {![file isfile $path] || [file type $path] ne "file" || [file size $path]>$max_file_bytes} {error "Raw output type/size bound exceeded"}
        incr total [file size $path]
        if {$total>$max_total_bytes} {error "Raw cumulative output bound exceeded"}
    }
    if {$code} {return -options $options $message}
    return $result
}
