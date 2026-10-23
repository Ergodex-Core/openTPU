# C1-only read-only reset/status diagnostic. Source as a library, then call
# ::c1_alex_reset_status::capture checkpoint expected_sha256 fresh_output.
# Names cross procedures; every Vivado operation reselects typed handles.
# No checkpoint/policy from another design is accepted or embedded.
namespace eval ::c1_alex_reset_status {
    variable cells {};variable pins {};variable nets {};variable endpoints {};variable edge_count 0
    variable max_cells 512;variable max_pins 8192;variable max_nets 4096
    variable max_net_edges 100000;variable max_edges 500000
    variable max_query_objects 100000;variable max_record_bytes 16777216;variable max_output_bytes 134217728
    variable capture_queries 0;variable capture_pending 0
    variable library_path [file normalize [info script]]
}
proc ::c1_alex_reset_status::emit {audit record} {
    variable max_record_bytes;variable max_output_bytes
    set size [expr {[string length [encoding convertto utf-8 $record]]+1}]
    if {$size>$max_record_bytes || [tell $audit]+$size>$max_output_bytes} {error "Raw diagnostic output bound exceeded"}
    puts $audit $record
}
proc ::c1_alex_reset_status::missing {audit label reason} {
    variable capture_pending;incr capture_pending
    emit $audit [list incomplete $label reason $reason review pending]
}
proc ::c1_alex_reset_status::inside {name} {return [string match {WRAPPER/CL/*} $name]}
proc ::c1_alex_reset_status::bounded {audit label values limit} {
    emit $audit [list inventory_coverage $label selected [llength $values] cap $limit]
    if {[llength $values]>$limit} {missing $audit $label [list detail_cap_reached $limit];return 0};return 1
}
proc ::c1_alex_reset_status::resolve {kind wanted} {
    # Kind is supplied by the known relation, never guessed from an arbitrary name.
    if {$kind ni {cell pin net port clock} || ![llength $wanted]} {error "Invalid typed selection"}
    set command get_${kind}s
    set selected [$command -quiet $wanted]
    set actual {};foreach object $selected {lappend actual [get_property NAME $object]}
    if {[lsort $actual] ne [lsort $wanted] || [llength [lsort -unique $actual]]!=[llength $wanted]} {error "Typed selection identity/cardinality mismatch: $kind $wanted"}
    # Validation may stringify the first collection. Reselect immediately for use.
    return [$command -quiet $wanted]
}
proc ::c1_alex_reset_status::property {kind name key} {return [get_property $key [resolve $kind [list $name]]]}
proc ::c1_alex_reset_status::relation {target kind names flags} {
    switch -- $target {
        pin {return [get_pins -quiet {*}$flags -of_objects [resolve $kind $names]]}
        net {return [get_nets -quiet {*}$flags -of_objects [resolve $kind $names]]}
        port {return [get_ports -quiet {*}$flags -of_objects [resolve $kind $names]]}
        cell {return [get_cells -quiet {*}$flags -of_objects [resolve $kind $names]]}
        clock {return [get_clocks -quiet {*}$flags -of_objects [resolve $kind $names]]}
        default {error "Unsupported typed relation"}
    }
}
proc ::c1_alex_reset_status::fanin {name kind} {
    if {$kind eq "cell"} {return [all_fanin -flat -only_cells -trace_arcs all -to [resolve pin [list $name]]]}
    if {$kind eq "root"} {return [all_fanin -flat -startpoints_only -trace_arcs all -to [resolve pin [list $name]]]}
    error "Unsupported fanin kind"
}
proc ::c1_alex_reset_status::pulse {root path} {return [report_pulse_width -cells [resolve cell [list $root]] -limit 128 -file $path]}
proc ::c1_alex_reset_status::query {audit label command {kind scalar}} {
    incr ::c1_alex_reset_status::capture_queries
    set code [catch {
        set raw [uplevel 1 $command]
        set value $raw
        if {$kind ne "scalar"} {
            set value {};set count 0
            if {[llength $raw]>$::c1_alex_reset_status::max_query_objects} {error "Typed query object cap reached"}
            foreach object $raw {
                incr count
                set name [get_property NAME $object]
                if {$kind eq "root"} {
                    set class [get_property CLASS $object]
                    if {$class ni {pin port}} {error "Unexpected typed fanin root class: $class"}
                    lappend value [dict create kind $class name $name]
                } else {lappend value $name}
            }
            if {$count!=[llength $raw]} {error "Collection cardinality changed"}
            emit $audit [list collection_serialization $label kind $kind actual_count $count serialized_count [llength $value] representation expanded_per_object_names]
        }
    } message options]
    if {$code} {
        incr ::c1_alex_reset_status::capture_pending
        emit $audit [list query $label status error message $message options $options review pending]
        return [dict create ok 0 value {}]
    }
    emit $audit [list query $label status observed value $value review pending]
    return [dict create ok 1 value $value]
}
proc ::c1_alex_reset_status::related {audit label target kind names {flags {}}} {
    return [query $audit $label [list relation $target $kind $names $flags] $target]
}
proc ::c1_alex_reset_status::props {audit label kind name keys} {
    set data {}
    foreach key $keys {dict set data $key [query $audit [list $label property $key] [list property $kind $name $key]]}
    emit $audit [list object_properties $label kind $kind name $name values $data review pending]
    return $data
}
proc ::c1_alex_reset_status::available_properties {kind name} {return [list_property [resolve $kind [list $name]]]}
proc ::c1_alex_reset_status::inversions {audit name} {
    set available [query $audit [list cell $name property_inventory] [list available_properties cell $name]]
    if {![dict get $available ok]} {return}
    set keys {IS_C_INVERTED IS_CLR_INVERTED IS_PRE_INVERTED IS_R_INVERTED IS_S_INVERTED IS_D_INVERTED IS_CE_INVERTED}
    set present {};set absent {}
    foreach key $keys {
        if {$key in [dict get $available value]} {lappend present $key} else {lappend absent $key}
    }
    emit $audit [list cell_inversion_availability $name available $present unavailable $absent interpretation effective_pin_polarity_required_for_primitive_semantics]
    if {[llength $present]} {props $audit [list cell $name inversions] cell $name $present}
}
proc ::c1_alex_reset_status::clock_driver {audit name} {
    variable max_nets
    # Preserve clock driver identity without traversing its global sink fanout.
    set nets [related $audit [list clock_pin $name nets] net pin [list $name] {-segments}]
    if {![dict get $nets ok] || ![bounded $audit [list clock_pin $name nets] [dict get $nets value] $max_nets]} {return}
    set names [dict get $nets value]
    if {![llength $names]} {missing $audit [list clock_pin $name] missing_clock_net;return}
    set drivers [related $audit [list clock_pin $name drivers] pin net $names {-leaf -filter {DIRECTION == OUT}}]
    if {![dict get $drivers ok] || ![bounded $audit [list clock_pin $name drivers] [dict get $drivers value] 16]} {return}
    foreach driver [dict get $drivers value] {
        props $audit [list clock_pin $name driver $driver] pin $driver {NAME REF_PIN_NAME DIRECTION IS_CLOCK}
        set owners [related $audit [list clock_pin $name owner $driver] cell pin [list $driver]]
        if {[dict get $owners ok] && [bounded $audit [list clock_pin $name owner $driver] [dict get $owners value] 16]} {
            foreach owner [dict get $owners value] {props $audit [list clock_pin $name driver_owner $owner] cell $owner {NAME REF_NAME IS_SEQUENTIAL}}
        }
    }
}
proc ::c1_alex_reset_status::net {audit name segments} {
    variable nets;variable max_nets;variable max_net_edges;variable max_edges;variable edge_count
    if {[dict exists $nets $name]} {return}
    if {[dict size $nets]>=$max_nets} {missing $audit [list net $name] net_budget_exceeded;return}
    foreach segment $segments {dict set nets $segment 1}
    if {[dict size $nets]>$max_nets} {missing $audit [list net $name] net_budget_exceeded;return}
    emit $audit [list net_segment_group $name segments $segments]
    foreach segment $segments {props $audit [list net_segment $segment] net $segment {NAME TYPE IS_CLOCK IS_ROUTE_FIXED}}
    set leaves [related $audit [list net $name leaf_pins] pin net $segments {-leaf}]
    related $audit [list net $name ports] port net $segments
    if {![dict get $leaves ok]} {return}
    set all [dict get $leaves value];incr edge_count [llength $all]
    if {![bounded $audit [list net $name leaves] $all $max_net_edges] || $edge_count>$max_edges} {missing $audit [list net $name] edge_budget_exceeded;return}
    set drivers [related $audit [list net $name typed_output_pins] pin net $segments {-leaf -filter {DIRECTION == OUT}}]
    if {![dict get $drivers ok] || ![bounded $audit [list net $name drivers] [dict get $drivers value] 64]} {return}
    foreach driver [dict get $drivers value] {
        props $audit [list net $name driver $driver] pin $driver {NAME REF_PIN_NAME DIRECTION IS_CLOCK}
        set owners [related $audit [list net $name driver_owners $driver] cell pin [list $driver]]
        if {[dict get $owners ok]} {foreach owner [dict get $owners value] {props $audit [list net $name driver_owner $owner] cell $owner {NAME REF_NAME IS_SEQUENTIAL}}}
    }
}
proc ::c1_alex_reset_status::pin_polarity {audit name} {
    set polarity [props $audit [list pin $name polarity] pin $name {IS_INVERTED}]
    if {![dict get $polarity IS_INVERTED ok] || [dict get $polarity IS_INVERTED value] ni {0 1}} {missing $audit [list pin $name] missing_effective_pin_polarity}
}
proc ::c1_alex_reset_status::pin {audit name {require_polarity 0}} {
    variable pins;variable max_pins;variable endpoints
    if {[dict exists $pins $name]} {if {$require_polarity} {pin_polarity $audit $name};return}
    if {[dict size $pins]>=$max_pins} {missing $audit [list pin $name] pin_budget_exceeded;return}
    dict set pins $name 1
    props $audit [list pin $name] pin $name {NAME REF_PIN_NAME DIRECTION IS_CLOCK}
    if {$require_polarity} {
        pin_polarity $audit $name
    }
    set ref [query $audit [list pin $name reference] [list property pin $name REF_PIN_NAME]]
    set isclock [query $audit [list pin $name is_clock] [list property pin $name IS_CLOCK]]
    set clocks [related $audit [list pin $name clocks] clock pin [list $name]]
    if {[dict get $clocks ok] && [bounded $audit [list pin $name clocks] [dict get $clocks value] 16]} {foreach clock [dict get $clocks value] {props $audit [list pin $name clock $clock] clock $clock {NAME PERIOD WAVEFORM MASTER_CLOCK SOURCE_PINS}}}
    if {[dict get $ref ok] && [dict get $ref value] in {CLR PRE} && [regexp {^WRAPPER/CL/u_rst_(core|hbm)/} $name]} {dict set endpoints $name 1}
    if {![dict get $isclock ok]} {missing $audit [list pin $name] cannot_classify_clock_fanout;return}
    if {[dict get $isclock value] in {1 true TRUE}} {emit $audit [list clock_pin_fanout $name collection omitted reason global_clock_fanout_not_needed clock_relationships_captured 1];clock_driver $audit $name;return}
    set ns [related $audit [list pin $name nets] net pin [list $name] {-segments}]
    if {[dict get $ns ok] && [bounded $audit [list pin $name nets] [dict get $ns value] 64]} {foreach n [dict get $ns value] {net $audit $n [dict get $ns value]}}
}
proc ::c1_alex_reset_status::cell {audit name} {
    variable cells;variable max_cells
    if {[dict exists $cells $name]} {return}
    if {[dict size $cells]>=$max_cells} {missing $audit [list cell $name] cell_budget_exceeded;return}
    dict set cells $name 1
    if {![inside $name]} {
        props $audit [list protected_or_external_boundary $name] cell $name {NAME REF_NAME IS_SEQUENTIAL}
        emit $audit [list boundary $name kind external_to_CL review provenance_based_shell_classification_no_hidden_identity_assertion];return
    }
    props $audit [list cell $name] cell $name {NAME REF_NAME ORIG_REF_NAME IS_SEQUENTIAL ASYNC_REG LOC BEL}
    set ref [query $audit [list cell $name reference] [list property cell $name REF_NAME]]
    if {[dict get $ref ok] && [regexp {^(LUT[1-6]|FD[A-Z0-9]+)$} [dict get $ref value]]} {props $audit [list cell $name initialization] cell $name {INIT}}
    set sequential [expr {[dict get $ref ok] && [regexp {^FD[A-Z0-9]+$} [dict get $ref value]]}]
    if {$sequential} {inversions $audit $name}
    set ps [related $audit [list cell $name pins] pin cell [list $name]]
    if {[dict get $ps ok] && [bounded $audit [list cell $name pins] [dict get $ps value] 256]} {foreach p [dict get $ps value] {pin $audit $p $sequential}}
}
proc ::c1_alex_reset_status::timing_paths {audit name delay ignored} {
    # Timing handles are ephemeral: inspect in the same frame, before serializing.
    set flags {};if {$ignored} {lappend flags -user_ignored}
    set paths [get_timing_paths -quiet -delay_type $delay -to [resolve pin [list $name]] -max_paths 1 -nworst 1 {*}$flags]
    set rows {};set count 0
    foreach path $paths {
        incr count;set available [list_property $path];set data {};set absent {}
        foreach key {STARTPOINT_PIN ENDPOINT_PIN STARTPOINT_CLOCK ENDPOINT_CLOCK SLACK REQUIREMENT DATAPATH_DELAY EXCEPTION PATH_GROUP PATH_TYPE CHECK_TYPE IS_RECOVERY IS_REMOVAL} {
            if {$key in $available} {dict set data $key [get_property $key $path]} else {lappend absent $key}
        }
        lappend rows [dict create properties $data omitted_properties $absent]
    }
    emit $audit [list reset_timing_path $name delay $delay user_ignored $ignored paths $rows]
    emit $audit [list reset_timing_coverage $name delay $delay user_ignored $ignored returned $count max_paths 1 interpretation worst_endpoint_witness_not_exhaustive]
    if {!$count} {emit $audit [list reset_timing_absence $name delay $delay user_ignored $ignored interpretation no_path_is_not_a_passing_recovery_or_removal_check]}
    return $count
}
proc ::c1_alex_reset_status::timing {audit name} {
    foreach delay {max min} {foreach ignored {0 1} {query $audit [list reset_timing $name $delay $ignored] [list timing_paths $audit $name $delay $ignored]}}
}
proc ::c1_alex_reset_status::collect {audit output} {
    variable endpoints
    foreach root {WRAPPER/CL/u_rst_core WRAPPER/CL/u_rst_hbm WRAPPER/CL/u_status_cdc} {
        set pattern "^${root}(/.*)?\$"
        set selected [query $audit [list hierarchy_discovery $root] [list get_cells -quiet -hierarchical -regexp $pattern] cell]
        if {![dict get $selected ok]} {continue}
        set objects [dict get $selected value]
        if {![llength $objects]} {missing $audit [list hierarchy_discovery $root] missing_hierarchy;continue}
        if {[bounded $audit [list hierarchy_discovery $root] $objects 256]} {foreach object $objects {cell $audit $object}}
        query $audit [list pulse $root] [list pulse $root [file join $output [file tail $root]_pulse.rpt]]
        emit $audit [list pulse_coverage $root checks default_all limit_per_check_type_per_clock 128 completeness not_asserted]
    }
    set outputs [related $audit status_boundary_pins pin cell {WRAPPER/CL} {-filter {REF_PIN_NAME =~ cl_sh_status2* || REF_PIN_NAME =~ cl_sh_status_vled* || REF_PIN_NAME == rst_main_n}}]
    if {[dict get $outputs ok] && [bounded $audit status_boundary_pins [dict get $outputs value] 128]} {foreach object [dict get $outputs value] {pin $audit $object}}
    emit $audit [list discovered_reset_async_endpoints [dict keys $endpoints] expected_source_design four_stages_per_reset actual_pairing_review pending]
    if {![dict size $endpoints]} {missing $audit reset_endpoints none_discovered}
    foreach name [dict keys $endpoints] {
        set roots [query $audit [list reset_fanin $name roots] [list fanin $name root] root]
        if {[dict get $roots ok] && [bounded $audit [list reset_fanin $name roots] [dict get $roots value] 128]} {
            foreach root [dict get $roots value] {props $audit [list reset_root $name] [dict get $root kind] [dict get $root name] {NAME CLASS DIRECTION}}
        }
        set cone [query $audit [list reset_fanin $name cells] [list fanin $name cell] cell]
        if {[dict get $cone ok] && [bounded $audit [list reset_fanin $name cells] [dict get $cone value] 256]} {foreach item [dict get $cone value] {cell $audit $item}}
        timing $audit $name
    }
    query $audit current_timing_summary [list report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file $output/timing.rpt]
    emit $audit [list interpretation topology_and_worst_endpoint_timing_only reset_gate_glitch_freedom_not_asserted independent_status_bit_semantics_require_source_review route_qualified 0]
}
proc ::c1_alex_reset_status::hash {path} {return [lindex [exec sha256sum -- $path] 0]}
proc ::c1_alex_reset_status::capture {checkpoint expected output} {
    variable library_path;variable capture_queries;variable capture_pending
    foreach name {cells pins nets endpoints} {variable $name;set $name {}}
    variable edge_count;set edge_count 0;set capture_queries 0;set capture_pending 0
    if {![regexp {^[0-9a-f]{64}$} $expected]} {error "Expected exact checkpoint SHA256"}
    if {[version -short] ne "2025.2"} {error "Unreviewed Vivado version"}
    if {![file isfile $checkpoint] || [file type $checkpoint] ne "file" || [hash $checkpoint] ne $expected} {error "Checkpoint hash/type differs"}
    if {[file exists $output] || ![file isdirectory [file dirname $output]]} {error "Output must be fresh with existing parent"}
    if {[llength [get_designs -quiet]]} {error "Diagnostic requires a fresh empty design session"}
    set_param general.maxThreads 4
    set helper_sha [hash $library_path]
    file mkdir $output
    set audit [open [file join $output topology.tcldata] {WRONLY CREAT EXCL}]
    fconfigure $audit -encoding utf-8 -translation lf
    set opened 0
    set code [catch {
        open_checkpoint $checkpoint;set opened 1
        emit $audit [list schema 1 kind c1_reset_status_diagnostic checkpoint_sha256 $expected collector_sha256 $helper_sha vivado_version 2025.2 route_qualified 0 hardware_measured 0]
        collect $audit $output
    } message options]
    if {$code} {catch {missing $audit collector [list unexpected_error $message $options]}}
    set summary_code [catch {emit $audit [list topology_summary queries $capture_queries incomplete_or_error $capture_pending route_qualified 0 review pending]} summary_message]
    close $audit
    if {$opened} {close_design}
    if {[hash $checkpoint] ne $expected || [hash $library_path] ne $helper_sha} {error "Pinned input changed during diagnostic"}
    set total_bytes 0
    foreach path [glob -nocomplain -directory $output *] {
        if {![file isfile $path] || [file type $path] ne "file" || [file size $path]>$::c1_alex_reset_status::max_output_bytes} {error "Diagnostic output size/type bound exceeded"}
        incr total_bytes [file size $path]
        if {$total_bytes>2*$::c1_alex_reset_status::max_output_bytes} {error "Cumulative diagnostic output bound exceeded"}
    }
    if {$code} {return -options $options $message}
    if {$summary_code} {error $summary_message}
    return [dict create checkpoint_sha256 $expected collector_sha256 $helper_sha queries $capture_queries incomplete_or_error $capture_pending route_qualified 0 hardware_measured 0]
}
# Library only. An externally hash-bound wrapper chooses the current C1 checkpoint.
