# Independent C2/Alex linked and final-route evidence. No timing/waiver mutations.
# The executor must freshly open the hash-bound checkpoint before qualify and
# bind these reports, frozen inputs/generated BD, and current vendor simulation.
source [file join [file dirname [info script]] audit_c2_alex.tcl]
namespace eval ::c2_alex_route {variable cl WRAPPER/CL;variable capture_cell_section standalone}
proc ::c2_alex_route::prop {object key} {return [::c2_alex::property $object $key]}
proc ::c2_alex_route::names {objects} {return [::c2_alex::names $objects]}
proc ::c2_alex_route::read_text {path} {set f [open $path r];set text [read $f];close $f;return $text}
proc ::c2_alex_route::write_data {path data} {set f [open $path {WRONLY CREAT EXCL}];puts $f $data;close $f}
proc ::c2_alex_route::number {value} {
    if {![regexp {^-?[0-9]+(?:\.[0-9]+)?$} $value]} {error "C2 Alex route: invalid numeric evidence $value"}
    return $value
}
proc ::c2_alex_route::sha256 {path} {
    if {![file isfile $path]} {error "C2 Alex route: missing checkpoint $path"}
    set executable [auto_execok sha256sum]
    if {$executable ne ""} {set result [exec {*}$executable -- $path]} else {
        set executable [auto_execok shasum]
        if {$executable eq ""} {error "C2 Alex route: SHA256 tool unavailable"}
        set result [exec {*}$executable -a 256 -- $path]
    }
    if {![regexp {^([0-9a-f]{64})\s} $result -> digest]} {error "C2 Alex route: malformed SHA256 output"}
    return $digest
}
proc ::c2_alex_route::fresh {run name} {
    if {[version -short] ne "2025.2" || [current_instance] ne ""} {error "C2 Alex route: expected Vivado 2025.2 at top scope"}
    set directory [file join $run $name]
    if {[file exists $directory]} {error "C2 Alex route: refusing existing evidence $directory"}
    file mkdir $directory
    return $directory
}
proc ::c2_alex_route::owner {name} {
    variable cl
    if {[string first "$cl/" $name] != 0} {return shell_or_unknown}
    return [::c2_alex::vendor_owner [string range $name [expr {[string length $cl]+1}] end]]
}
proc ::c2_alex_route::clock {pin period} {
    if {[names [get_pins -quiet $pin]] ne [list $pin]} {error "C2 Alex route: missing clock pin $pin"}
    set clock [get_clocks -quiet -of_objects [get_pins -quiet $pin]]
    if {[llength $clock] != 1 || abs([number [prop $clock PERIOD]]-$period)>0.001} {error "C2 Alex route: wrong/missing clock at $pin"}
    return $clock
}
proc ::c2_alex_route::structure {audit} {
    variable cl
    ::c2_alex::cell $cl cl_opentpu 1
    ::c2_alex::cell $cl/u_board otpu_c2_hbm_kernel 1
    ::c2_alex::cell $cl/u_board/u_mxu otpu_c2_mxu 1
    set expected {}
    for {set p 0} {$p<8} {incr p} {lappend expected $cl/u_native$p;::c2_alex::cell $cl/u_native$p otpu_native_axi 1}
    if {[lsort [names [get_cells -quiet -hierarchical -regexp {^WRAPPER/CL/u_native[0-9]+$}]]] ne $expected} {error "C2 Alex route: native topology changed"}
    foreach {name ref} {u_mem f2_mem_wrapper u_hbm cl_mem_hbm_wrapper u_ocl_cdc f2_ocl_cdc u_pcis_pipe f2_pcis_pipe u_core_clk f2_core_clk u_hbm_clk f2_hbm_clk} {::c2_alex::cell $cl/$name $ref 1}
    set opaque_shell {}
    foreach object [get_cells -quiet -hierarchical -filter {IS_BLACKBOX}] {
        set name [prop $object NAME]
        if {$name eq $cl || [string first "$cl/" $name] == 0} {error "C2 Alex route: unresolved CL/IP $name"}
        lappend opaque_shell $name
    }
    if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ otpu_tmem* || ORIG_REF_NAME =~ otpu_tmem*}]]} {error "C2 Alex route: unexpected TMEM implementation"}
    set interface_pins {}
    foreach pin [names [get_pins -quiet -of_objects [get_cells $cl/u_mem]]] {
        if {[string first "$cl/" $pin] != 0} {error "C2 Alex route: interface pin outside CL"}
        lappend interface_pins [string range $pin [expr {[string length $cl]+1}] end]
    }
    puts $audit [list memory_interface_pins $interface_pins]
    set sc [get_cells -quiet $cl/u_mem/f2_mem_i/memory_sc]
    if {[names $sc] ne [list $cl/u_mem/f2_mem_i/memory_sc] || [prop $sc IS_BLACKBOX]} {error "C2 Alex route: missing concrete SmartConnect"}
    puts $audit [list smartconnect $sc reference [prop $sc REF_NAME] slave_interfaces 9 master_interfaces 8 address_map_validation executor_generated_BD_required]
    set clocks {}
    foreach {label pin period} {host clk_main_a0 4.000 core u_core_clk/clk_out1 5.000 hbm u_hbm_clk/clk_out1 2.222222222222 reference clk_hbm_ref 10.000} {
        set object [clock $cl/$pin $period]
        dict set clocks $label $object
        set data {};foreach key {NAME PERIOD MASTER_CLOCK SOURCE_PINS IS_GENERATED IS_PROPAGATED IS_VIRTUAL} {if {$key in [list_property $object]} {dict set data $key [get_property $key $object]}}
        puts $audit [list clock $label $data]
    }
    foreach label {host core hbm} {
        set registers [all_registers -clock [dict get $clocks $label] -cells]
        if {![llength $registers]} {error "C2 Alex route: empty $label clock domain"}
        puts $audit [list clock_registers $label count [llength $registers]]
        if {$label eq "hbm"} {
            foreach name [names $registers] {
                if {[owner $name] ni {u_mem u_hbm u_rst_hbm}} {error "C2 Alex route: unexpected HBM-clock register $name"}
            }
        }
    }
    set fabric [::c2_alex::fabric_topology $cl $clocks {} $audit]
    puts $audit [list interface_topology $fabric address_map_evidence generated_BD_required]
    # No actual consumer may hide behind a removed/renamed unused wizard pin.
    foreach object [get_clocks -quiet *] {
        if {"SOURCE_PINS" ni [list_property $object]} {continue}
        foreach source [get_property SOURCE_PINS $object] {
            if {[string first "$cl/u_core_clk/" $source] == 0 && $object ne [dict get $clocks core]} {
                if {[llength [all_registers -clock $object -cells]]} {error "C2 Alex route: unexpected second core-wizard clock consumer"}
                puts $audit [list unused_clock [prop $object NAME] period [prop $object PERIOD] consumers 0]
            }
        }
    }
    puts $audit [list unresolved_shell_objects $opaque_shell]
    return [dict create clocks $clocks opaque_shell $opaque_shell]
}
proc ::c2_alex_route::constraints {path audit} {
    set pending {};set scope {};set counts {};set commands {}
    foreach line [split [read_text $path] \n] {
        if {$pending eq "" && ([string trim $line] eq "" || [string index [string trim $line] 0] eq "#")} {continue}
        append pending $line \n
        if {[regexp {(^|[^\\])(\\\\)*\\$} $line] || ![info complete $pending]} {continue}
        set command [string trim $pending];set pending {}
        if {[regexp -nocase {u_tmem|otpu_clk2x|OTPU_CORE_TMEM} $command]} {error "C2 Alex route: TMEM constraint in export"}
        if {[regexp {^current_instance(?:\s|$)} $command]} {
            if {[catch {llength $command}]} {error "C2 Alex route: malformed XDC scope"}
            set args [lrange $command 1 end];if {[lindex $args 0] eq "-quiet"} {set args [lrange $args 1 end]}
            if {[llength $args]>1} {error "C2 Alex route: ambiguous XDC scope"}
            set scope [lindex $args 0]
        }
        if {[regexp {(^|[;\n])\s*(set_false_path|set_clock_groups|set_max_delay|set_min_delay|set_bus_skew|set_multicycle_path|set_disable_timing)(?:\s|$)} $command -> unused type]} {
            dict incr counts $type
            lappend commands [list scope $scope command $command]
            # Preserve every exception as review data; never execute the export.
            puts $audit [list exception scope $scope command $command review pending]
        }
    }
    if {$pending ne ""} {error "C2 Alex route: incomplete XDC export"}
    return [dict create counts $counts commands $commands]
}
proc ::c2_alex_route::gray_inventory {audit} {
    set groups {}
    foreach cell [get_cells -quiet -hierarchical -regexp {^WRAPPER/CL/u_mem/f2_mem_i/memory_sc/.*/(src_gray_ff_reg\[[0-9]+\]|dest_graysync_ff_reg\[0\]\[[0-9]+\])$}] {
        set name [prop $cell NAME]
        if {[regexp {^(.*)/src_gray_ff_reg\[([0-9]+)\]$} $name -> root bit]} {set side source} elseif {[regexp {^(.*)/dest_graysync_ff_reg\[0\]\[([0-9]+)\]$} $name -> root bit]} {set side destination} else {error "C2 Alex route: unexpected gray-register object"}
        if {[dict exists $groups $root $side $bit]} {error "C2 Alex route: duplicate gray-register bit"}
        dict set groups $root $side $bit $name
    }
    if {![dict size $groups]} {error "C2 Alex route: no current SmartConnect gray inventory"}
    dict for {root group} $groups {
        if {![dict exists $group source] || ![dict exists $group destination] || [lsort -integer [dict keys [dict get $group source]]] ne [lsort -integer [dict keys [dict get $group destination]]]} {error "C2 Alex route: incomplete gray bus $root"}
        puts $audit [list vendor_gray_bus $root $group review pending_constraint_and_topology_match]
    }
    return $groups
}
proc ::c2_alex_route::gray_witnesses {groups audit} {
    dict for {root group} $groups {
        set selections {}
        foreach side {source destination} {
            set wanted [lsort [dict values [dict get $group $side]]]
            set cells [get_cells -quiet $wanted]
            if {[lsort [names $cells]] ne $wanted} {error "C2 Alex route: ambiguous gray timing selection"}
            dict set selections $side $cells
        }
        foreach ignored {0 1} {
            set args [list -quiet -from [dict get $selections source] -to [dict get $selections destination] -max_paths 1 -nworst 1]
            if {$ignored} {lappend args -user_ignored}
            set paths [get_timing_paths {*}$args]
            set data {}
            foreach path $paths {
                set record {}
                foreach key {STARTPOINT_PIN ENDPOINT_PIN STARTPOINT_CLOCK ENDPOINT_CLOCK REQUIREMENT SLACK EXCEPTION DATAPATH_DELAY} {
                    if {$key in [list_property $path]} {dict set record $key [get_property $key $path]}
                }
                lappend data $record
            }
            puts $audit [list gray_timing_worst_witness $root user_ignored $ignored max_paths 1 paths $data review pending_not_exhaustive_endpoint_proof]
        }
    }
}
# Supplemental evidence only: unknown APIs and missing objects remain explicit
# pending data. These routines never change exceptions, severity, or acceptance.
proc ::c2_alex_route::capture {audit label command} {
    variable capture_queries;variable capture_pending
    incr capture_queries
    set code [catch {uplevel 1 $command} value options]
    if {$code} {
        incr capture_pending
        puts $audit [list query $label status error message $value options $options review pending]
        return [dict create ok 0 value {}]
    }
    puts $audit [list query $label status observed value $value review pending]
    return [dict create ok 1 value $value]
}
proc ::c2_alex_route::capture_missing {audit label detail} {
    variable capture_pending;incr capture_pending
    puts $audit [list incomplete $label detail $detail review pending]
}
proc ::c2_alex_route::capture_properties {audit kind object keys} {
    # Never list_property a cell: use only explicit known primitive properties.
    set data {}
    foreach key $keys {
        set result [capture $audit [list $kind property $key] [list get_property $key $object]]
        dict set data $key $result
    }
    puts $audit [list object_properties $kind $data review pending]
    return $data
}
proc ::c2_alex_route::capture_cells {audit label objects} {
    variable captured_cells;variable capture_cell_section
    set selected [names $objects];set captured 0;set cached 0;set skipped 0
    foreach name $selected {
        if {[dict exists $captured_cells $name]} {incr cached;continue}
        if {$captured>=256 || [dict size $captured_cells]>=4096} {incr skipped;continue}
        dict set captured_cells $name 1;incr captured
        set cell [get_cells -quiet [list $name]]
        capture_properties $audit [list $label cell $name] $cell {NAME REF_NAME ORIG_REF_NAME IS_SEQUENTIAL ASYNC_REG LOC BEL}
    }
    puts $audit [list cell_detail_coverage $label selected [llength $selected] captured $captured cached $cached skipped $skipped per_call_cap 256 section_cap 4096 section $capture_cell_section]
    if {$skipped} {capture_missing $audit $label cell_detail_cap_reached}
}
proc ::c2_alex_route::capture_pin {audit label name {fanin 0} {require_clock 0}} {
    set selected [capture $audit [list $label pin_selection $name] [list get_pins -quiet [list $name]]]
    if {![dict get $selected ok] || [names [dict get $selected value]] ne [list $name]} {
        capture_missing $audit $label [list missing_or_ambiguous_pin $name];return {}
    }
    set pin [dict get $selected value]
    capture_properties $audit [list $label pin $name] $pin {NAME REF_PIN_NAME DIRECTION IS_CLOCK}
    set cells [capture $audit [list $label owning_cells] [list get_cells -quiet -of_objects $pin]]
    if {[dict get $cells ok]} {capture_cells $audit $label [dict get $cells value]}
    set clocks [capture $audit [list $label associated_clocks] [list get_clocks -quiet -of_objects $pin]]
    if {[dict get $clocks ok]} {
        if {$require_clock && ![llength [dict get $clocks value]]} {capture_missing $audit $label missing_required_clock}
        foreach clock [dict get $clocks value] {
            capture_properties $audit [list $label clock] $clock {NAME PERIOD WAVEFORM MASTER_CLOCK SOURCE_PINS}
        }
    }
    if {$fanin} {
        set roots [capture $audit [list $label fanin_roots] [list all_fanin -flat -startpoints_only -trace_arcs all -to $pin]]
        if {[dict get $roots ok]} {
            set root_names [names [dict get $roots value]]
            puts $audit [list root_detail_coverage $label selected [llength $root_names] cap 64]
            foreach root [lrange $root_names 0 63] {capture_pin $audit [list $label root] $root 0}
            if {[llength $root_names]>64} {capture_missing $audit $label fanin_root_detail_cap_reached}
            if {![llength [dict get $roots value]]} {capture_missing $audit $label empty_fanin_roots}
        }
    }
    return $pin
}
proc ::c2_alex_route::capture_findings {audit} {
    foreach {kind command} {methodology get_methodology_violations drc get_drc_violations} {
        set result [capture $audit [list $kind finding_inventory] [list $command -quiet]]
        if {![dict get $result ok]} {continue}
        foreach finding [dict get $result value] {
            set rule [capture $audit [list $kind finding_check] [list get_property CHECK $finding]]
            set label [list $kind finding [names $finding]]
            foreach type {pins cells} {
                set selected [capture $audit [list $label related_$type] [list get_$type -quiet -of_objects $finding]]
                if {![dict get $selected ok]} {continue}
                if {$type eq "cells"} {capture_cells $audit $label [dict get $selected value]}
                if {$type eq "pins" && [dict get $rule ok] && [dict get $rule value] eq "TIMING-17"} {
                    if {![llength [dict get $selected value]]} {capture_missing $audit $label no_clock_finding_without_related_pins}
                    set pin_names [names [dict get $selected value]]
                    puts $audit [list no_clock_pin_detail_coverage $label selected [llength $pin_names] cap 128]
                    foreach pin [lrange $pin_names 0 127] {capture_pin $audit $label $pin 1}
                    if {[llength $pin_names]>128} {capture_missing $audit $label no_clock_pin_detail_cap_reached}
                }
            }
        }
    }
}
proc ::c2_alex_route::capture_data_driver {audit label pin} {
    if {![llength $pin]} {return}
    set nets [capture $audit [list $label data_nets] [list get_nets -quiet -segments -of_objects $pin]]
    if {![dict get $nets ok]} {return}
    if {![llength [dict get $nets value]]} {capture_missing $audit $label missing_data_nets;return}
    set leaves [capture $audit [list $label data_leaf_pins] [list get_pins -quiet -leaf -of_objects [dict get $nets value]]]
    if {![dict get $leaves ok]} {return}
    if {![llength [dict get $leaves value]]} {capture_missing $audit $label missing_data_leaf_pins;return}
    set drivers {}
    foreach leaf [dict get $leaves value] {
        set direction [capture $audit [list $label data_leaf_direction [names $leaf]] [list get_property DIRECTION $leaf]]
        if {[dict get $direction ok] && [dict get $direction value] eq "OUT"} {lappend drivers {*}[names $leaf]}
    }
    puts $audit [list data_driver $label drivers $drivers driver_count [llength $drivers] review pending_exact_bit_and_stage_pairing]
    if {[llength $drivers]!=1} {capture_missing $audit $label missing_or_multiple_data_drivers}
}
proc ::c2_alex_route::capture_gray {groups audit} {
    # Actual linked evidence returned four normal paths for every bit at the old
    # cap of four. A larger bounded query distinguishes those variants from an
    # incomplete result; reaching the new cap still leaves coverage pending.
    set path_limit 16;set path_sample_taken 0
    set all_stages [capture $audit gray_destination_stage_inventory [list get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER/CL/u_mem/f2_mem_i/memory_sc/*dest_graysync_ff_reg*}]]
    set stages {}
    if {[dict get $all_stages ok]} {
        foreach name [names [dict get $all_stages value]] {
            if {[regexp {^(.*)/dest_graysync_ff_reg\[([0-9]+)\]\[([0-9]+)\]$} $name -> root stage bit] && [dict exists $groups $root]} {
                if {![dict exists $groups $root source $bit]} {capture_missing $audit $name unexpected_gray_bit;continue}
                if {[dict exists $stages $root $bit $stage]} {capture_missing $audit $name duplicate_gray_stage;continue}
                dict set stages $root $bit $stage $name
            } else {capture_missing $audit $name unmatched_destination_stage_object}
        }
    }
    dict for {root group} $groups {
        foreach bit [lsort -integer [dict keys [dict get $group source]]] {
            set source [dict get $group source $bit];set destination [dict get $group destination $bit]
            set label [list gray $root bit $bit]
            set q [capture_pin $audit [list $label source] $source/Q]
            capture_pin $audit [list $label source_clock] $source/C 0 1
            set d [capture_pin $audit [list $label first_destination] $destination/D]
            capture_data_driver $audit [list $label first_destination] $d
            if {![dict exists $stages $root $bit]} {capture_missing $audit $label missing_synchronizer_stage_inventory} else {
                set actual_stages [dict get $stages $root $bit]
                if {![dict exists $actual_stages 0] || [dict get $actual_stages 0] ne $destination} {capture_missing $audit $label missing_or_mismatched_stage_zero}
                set highest [lindex [lsort -integer [dict keys $actual_stages]] end]
                for {set stage 0} {$stage<=$highest} {incr stage} {
                    if {![dict exists $actual_stages $stage]} {capture_missing $audit [list $label stage $stage] missing_synchronizer_stage}
                }
                puts $audit [list gray_stage_inventory $label stages $actual_stages review pending_contiguous_stage_and_same_clock_proof]
                foreach stage [lsort -integer [dict keys $actual_stages]] {
                    set cell [dict get $actual_stages $stage]
                    set stage_d [capture_pin $audit [list $label stage $stage data] $cell/D]
                    capture_data_driver $audit [list $label stage $stage] $stage_d
                    capture_pin $audit [list $label stage $stage output] $cell/Q
                    capture_pin $audit [list $label stage $stage clock] $cell/C 0 1
                }
            }
            foreach ignored {0 1} {
                if {![llength $q] || ![llength $d]} {capture_missing $audit [list $label timing $ignored] missing_timing_pin_selection;continue}
                set command [list get_timing_paths -quiet -setup -from $q -to $d -max_paths $path_limit -nworst $path_limit]
                if {$ignored} {lappend command -user_ignored}
                set result [capture $audit [list $label timing user_ignored $ignored] $command]
                if {![dict get $result ok]} {continue}
                set paths [dict get $result value];set count [llength $paths]
                puts $audit [list gray_bit_timing $label user_ignored $ignored expected_source $source/Q expected_destination $destination/D \
                    expected_source_clock_pin $source/C max_paths $path_limit nworst $path_limit returned $count capped [expr {$count>=$path_limit}] normal_path_observed [expr {!$ignored && $count>0}] review pending_effective_exception_and_vendor_bound]
                if {$count>=$path_limit || (!$ignored && !$count)} {capture_missing $audit [list $label timing $ignored] capped_or_missing_normal_paths}
                set sample_this_query [expr {!$path_sample_taken && $count>0}];set path_index 0;set sample_keys {}
                foreach path $paths {
                    set path_label [list $label path user_ignored $ignored index $path_index]
                    set properties [capture_properties $audit $path_label $path {STARTPOINT_PIN ENDPOINT_PIN STARTPOINT_CLOCK ENDPOINT_CLOCK REQUIREMENT EXCEPTION DATAPATH_DELAY SLACK}]
                    if {[dict get $properties STARTPOINT_PIN ok] && [dict get $properties ENDPOINT_PIN ok]} {
                        set actual_start [dict get $properties STARTPOINT_PIN value];set actual_end [dict get $properties ENDPOINT_PIN value]
                        set relation other
                        if {$actual_start eq "$source/C"} {set relation source_register_clock_pin}
                        if {$actual_start eq "$source/Q"} {set relation source_register_data_pin}
                        # Vivado reports the launch C pin for observed Q->D
                        # queries. Preserve both identities, not a rewritten Q.
                        puts $audit [list gray_path_endpoint_relation $path_label observed_startpoint $actual_start observed_endpoint $actual_end \
                            data_driver_query $source/Q source_clock_query $source/C destination_query $destination/D \
                            startpoint_relation $relation endpoint_matches_query [expr {$actual_end eq "$destination/D"}] review pending_topology_and_effective_exception]
                    }
                    if {$sample_this_query} {
                        if {$path_index==0} {
                            set keys [capture $audit [list $path_label supported_property_names] [list list_property $path]]
                            if {[dict get $keys ok]} {
                                foreach key [dict get $keys value] {
                                    if {[regexp -nocase {CORNER|EDGE|RISE|FALL|POLARITY|PATH_TYPE|DELAY_TYPE} $key]} {lappend sample_keys $key}
                                }
                            }
                            puts $audit [list gray_path_variant_sample $label user_ignored $ignored paths $count limit $path_limit properties $sample_keys review pending_not_exhaustive_design_inventory]
                        }
                        if {[llength $sample_keys]} {capture_properties $audit [list $path_label optional_variant_properties] $path $sample_keys}
                    }
                    incr path_index
                }
                if {$sample_this_query} {set path_sample_taken 1}
            }
        }
    }
}
proc ::c2_alex_route::capture_waivers {audit} {
    foreach command {get_waivers report_pulse_width} {
        set available [expr {[llength [info commands $command]]>0}]
        puts $audit [list api_available $command $available review pending]
        if {$available} {capture $audit [list installed_help $command] [list help $command]}
    }
    capture $audit installed_help_check_timing {help check_timing}
    capture $audit installed_help_get_timing_paths {help get_timing_paths}
    puts $audit [list installed_pulse_commands [info commands *pulse*] review pending_protected_endpoint_identity]
    if {![llength [info commands get_waivers]]} {capture_missing $audit waivers unsupported_live_waiver_inventory;return}
    set selected [capture $audit waiver_inventory {get_waivers -quiet}]
    if {![dict get $selected ok]} {return}
    foreach waiver [dict get $selected value] {
        set keys [capture $audit waiver_property_names [list list_property $waiver]]
        if {[dict get $keys ok]} {capture_properties $audit waiver $waiver [dict get $keys value]}
        foreach kind {pins cells nets clocks} {
            capture $audit [list waiver [names $waiver] selected_$kind] [list get_$kind -quiet -of_objects $waiver]
        }
    }
}
proc ::c2_alex_route::capture_pulse {audit} {
    # These flags are documented by the actual installed Vivado 2025.2 help
    # retained in the c072 linked diagnostic. Never use -quiet (hides errors)
    # or -all_violators (would omit positive-slack protected-shell witnesses).
    if {![llength [info commands report_pulse_width]]} {capture_missing $audit pulse unsupported_pulse_report;return}
    set limit 128
    foreach scope {whole_design custom_logic} {
        set command [list report_pulse_width -return_string -limit $limit]
        if {$scope eq "custom_logic"} {
            set cells [capture $audit pulse_custom_logic_selection {get_cells -quiet WRAPPER/CL}]
            if {![dict get $cells ok] || [names [dict get $cells value]] ne {WRAPPER/CL}} {
                capture_missing $audit pulse missing_or_ambiguous_custom_logic;continue
            }
            lappend command -cells [dict get $cells value]
        }
        set report [capture $audit [list pulse_report $scope] $command]
        if {[dict get $report ok] && [string trim [dict get $report value]] eq ""} {capture_missing $audit [list pulse_report $scope] empty_pulse_report}
        puts $audit [list pulse_report_coverage $scope limit_per_check_type_per_clock $limit checks default_all \
            complete not_asserted review pending_hidden_endpoints_and_report_limit]
    }
}
proc ::c2_alex_route::supplemental {directory groups} {
    variable capture_queries 0;variable capture_pending 0;variable captured_cells {};variable capture_cell_section
    set audit [open $directory/supplemental.tcldata {WRONLY CREAT EXCL}]
    puts $audit [list schema 1 kind current_checkpoint_supplemental_evidence route_qualified 0 review pending]
    # Separate bounded caches let findings neither consume gray's detail budget
    # nor silently reuse incomplete gray captures. The two cell-collecting
    # sections together visit at most 8192 cells, with explicit cap diagnostics.
    foreach {label command} [list gray [list capture_gray $groups $audit] findings [list capture_findings $audit] waivers [list capture_waivers $audit] pulse [list capture_pulse $audit]] {
        set captured_cells {};set capture_cell_section $label
        puts $audit [list cell_detail_budget section $label per_call_cap 256 section_cap 4096]
        # A collector/API defect must remain visible without discarding all the
        # other independent evidence or implying structural/physical acceptance.
        capture $audit [list collector $label] $command
    }
    set summary [dict create queries $capture_queries incomplete_or_error $capture_pending review pending]
    puts $audit [list supplemental_summary $summary];close $audit
    return $summary
}

proc ::c2_alex_route::report_set {directory} {
    report_methodology -file $directory/methodology.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file $directory/timing.rpt
    report_timing -delay_type max -max_paths 50 -path_type full_clock_expanded -file $directory/setup.rpt
    report_timing -delay_type min -max_paths 25 -path_type full_clock_expanded -file $directory/hold.rpt
    report_route_status -file $directory/route_status.rpt
    report_drc -file $directory/drc.rpt
    report_cdc -details -show_waiver -all_checks_per_endpoint -file $directory/cdc.rpt
    report_bus_skew -file $directory/bus_skew.rpt
    report_clock_interaction -file $directory/clock_interaction.rpt
    report_exceptions -coverage -file $directory/exception_coverage.rpt
    report_exceptions -ignored -file $directory/ignored_exceptions.rpt
    report_waivers -file $directory/waivers.rpt
    report_utilization -hierarchical -file $directory/utilization.rpt
    write_xdc -type timing -constraints all -write_id $directory/timing_all_ids.xdc
    set files {methodology.rpt timing.rpt setup.rpt hold.rpt route_status.rpt drc.rpt cdc.rpt bus_skew.rpt clock_interaction.rpt exception_coverage.rpt ignored_exceptions.rpt waivers.rpt utilization.rpt timing_all_ids.xdc}
    foreach name $files {if {![file isfile $directory/$name] || [file size $directory/$name]==0} {error "C2 Alex route: missing/empty report $name"}}
    return $files
}
proc ::c2_alex_route::timing_summary {text} {
    if {![regexp {WNS\(ns\)[^\n]*TPWS Total Endpoints[^\n]*\n[^\n]*\n\s*([^\n]+)} $text -> row]} {error "C2 Alex route: missing timing totals"}
    set values [regexp -all -inline {\S+} $row]
    if {[llength $values]!=12} {error "C2 Alex route: unexpected timing totals format"}
    set result {};set blockers {}
    foreach name {wns tns setup_failing setup_total whs ths hold_failing hold_total wpws tpws pulse_failing pulse_total} value $values {dict set result $name [number $value]}
    foreach key {wns whs wpws} {if {[dict get $result $key]<0} {lappend blockers negative_$key}}
    foreach key {tns ths tpws setup_failing hold_failing pulse_failing} {if {[dict get $result $key]!=0} {lappend blockers nonzero_$key}}
    foreach key {setup_total hold_total pulse_total} {if {![string is integer -strict [dict get $result $key]] || [dict get $result $key]<=0} {error "C2 Alex route: missing $key endpoints"}}
    set checks {}
    foreach line [split $text \n] {
        if {[regexp {^([0-9]+)\. checking ([a-z_]+) \(([0-9]+)\)$} $line -> id name count]} {
            if {[dict exists $checks $name] && [dict get $checks $name] != $count} {error "C2 Alex route: inconsistent check_timing count"}
            dict set checks $name $count
        }
    }
    set expected {no_clock constant_clock pulse_width_clock unconstrained_internal_endpoints no_input_delay no_output_delay multiple_clock generated_clocks loops partial_input_delay partial_output_delay latch_loops}
    if {[lsort [dict keys $checks]] ne [lsort $expected]} {error "C2 Alex route: incomplete check_timing inventory"}
    foreach name $expected {
        if {$name ni {no_clock pulse_width_clock} && [dict get $checks $name]!=0} {lappend blockers check_timing_$name}
    }
    return [dict create totals $result checks $checks blockers $blockers]
}
proc ::c2_alex_route::route_summary {text} {
    set result {}
    foreach {key label} {routable {routable nets} routed {fully routed nets} errors {nets with routing errors}} {
        set expression [format {# of %s\.+\s*:\s*([0-9]+)\s*:} $label]
        if {![regexp $expression $text -> count]} {error "C2 Alex route: incomplete route-status inventory"}
        dict set result $key $count
    }
    dict set result pass [expr {[dict get $result routable]>0 && [dict get $result routable]==[dict get $result routed] && [dict get $result errors]==0}]
    return $result
}
proc ::c2_alex_route::literal_cells {expression} {
    if {![regexp {^\[get_cells \[list (.*)\]\]$} [string trim $expression] -> literal]} {error "C2 Alex route: unsupported bus-skew selector"}
    # report_bus_skew writes OCL handshake lists with Tcl line continuations.
    # Normalize only that whitespace, without evaluating any printed command.
    regsub -all {\\\r?\n[ \t]*} $literal { } literal
    if {[catch {llength $literal}]} {error "C2 Alex route: malformed bus-skew selector list"}
    set result [lrange $literal 0 end]
    if {![llength $result] || [llength [lsort -unique $result]]!=[llength $result]} {error "C2 Alex route: empty/duplicate bus-skew selector"}
    foreach name $result {if {[regexp {[\s;$\\]} $name]} {error "C2 Alex route: unexpected bus-skew object syntax"}}
    return [lsort $result]
}
proc ::c2_alex_route::bus_rows {text} {
    if {![regexp -indices {(^|\n)1\. Bus Skew Report Summary\r?\n-+\r?\n} $text match]} {error "C2 Alex route: missing all-constraint bus-skew summary"}
    set start [expr {[lindex $match 1]+1}]
    if {![regexp -indices -start $start {(^|\n)2\. Bus Skew Report Per Constraint\r?\n-+\r?\n} $text match]} {error "C2 Alex route: missing bus-skew detail boundary"}
    set summary [string range $text $start [expr {[lindex $match 0]-1}]]
    set rows {};set id {}
    foreach line [split $summary \n] {
        if {[regexp {^\s*([0-9]+)\s+([0-9]+)\s+\[} $line -> next pos]} {
            if {[dict exists $rows $next]} {error "C2 Alex route: duplicate bus-skew row"}
            set id $next;dict set rows $id "$line\n"
        } elseif {$id ne ""} {dict append rows $id "$line\n"}
    }
    if {![dict size $rows]} {error "C2 Alex route: empty bus-skew inventory"}
    set parsed {};set next 1
    dict for {id row} $rows {
        if {$id != $next} {error "C2 Alex route: noncontiguous bus-skew inventory"};incr next
        if {![regexp {^\s*[0-9]+\s+([0-9]+)\s+(\[get_cells \[list .*?\]\])\s+(\[get_cells \[list .*?\]\])\s+(Slow|Fast)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s*$} $row -> position from to corner requirement actual slack]} {error "C2 Alex route: incomplete bus-skew row $id"}
        dict set parsed $id [dict create position $position from [literal_cells $from] to [literal_cells $to] corner $corner requirement [number $requirement] actual [number $actual] slack [number $slack]]
    }
    return $parsed
}
proc ::c2_alex_route::gray_coverage {groups rows audit} {
    dict for {root group} $groups {
        set from [lsort [dict values [dict get $group source]]];set to [lsort [dict values [dict get $group destination]]]
        set matches {}
        dict for {id row} $rows {if {[dict get $row from] eq $from && [dict get $row to] eq $to} {lappend matches $id}}
        if {[llength $matches]!=1} {error "C2 Alex route: gray bus lacks unique exact bus-skew coverage: $root"}
        puts $audit [list gray_bus_skew_match $root id [lindex $matches 0] review pending_max_delay_and_ip_scope]
    }
}
proc ::c2_alex_route::cdc_rows {text} {
    set expected {};set counts {};set rows {};set source {};set destination {};set type {}
    foreach line [split $text \n] {
        if {[regexp {^(CDC-[0-9]+)\s+(Critical|Warning|Info)\s+([0-9]+)\s+} $line -> rule severity count]} {
            if {[dict exists $expected $rule]} {error "C2 Alex route: duplicate CDC summary"}
            dict set expected $rule $count
        } elseif {[regexp {^Source Clock: (.+)$} $line -> value]} {set source $value
        } elseif {[regexp {^Destination Clock: (.+)$} $line -> value]} {set destination $value
        } elseif {[regexp {^CDC Type: (.+)$} $line -> value]} {set type $value
        } elseif {[regexp {^\s*[0-9]+\s+CDC-} $line]} {
            if {![regexp {^\s*([0-9]+)\s+(CDC-[0-9]+)\s+(Critical|Warning|Info)\s+(.+?)\s+([0-9]+)\s+(.+?)\s+(\S+)\s+(\S+)\s+([NY])\s*$} $line -> id rule severity description depth exception from to waived] || $source eq "" || $destination eq "" || $type eq ""} {error "C2 Alex route: malformed CDC detail"}
            dict incr counts $rule
            set category unreviewed
            set prefix WRAPPER/CL/u_mem/f2_mem_i/memory_sc/
            if {$rule eq "CDC-15" && $exception eq "Max Delay Datapath Only" && [string first $prefix $from]==0 && [string first $prefix $to]==0 && [string first gen_xpm_memory_fifo.inst_fifo/ $from]>=0 && [string first gen_xpm_memory_fifo.inst_fifo/ $to]>=0} {set category vendor_fifo_candidate}
            lappend rows [dict create source_clock $source destination_clock $destination cdc_type $type row $id rule $rule severity $severity description $description depth $depth exception $exception from $from to $to waived $waived category $category review pending]
        }
    }
    if {![dict size $expected] || [lsort [dict keys $counts]] ne [lsort [dict keys $expected]]} {error "C2 Alex route: incomplete CDC inventory"}
    dict for {rule count} $expected {if {[dict get $counts $rule]!=$count} {error "C2 Alex route: CDC summary/detail mismatch $rule"}}
    return $rows
}
proc ::c2_alex_route::findings {audit} {
    set counts {};set blockers {}
    foreach {kind command checks_command} {drc get_drc_violations get_drc_checks methodology get_methodology_violations get_methodology_checks} {
        # Capture all rule settings, including disabled rules that cannot emit
        # findings. Do not infer an unmodified severity policy from empty output.
        set rules [$checks_command -quiet *]
        if {![llength $rules]} {error "C2 Alex route: missing $kind rule inventory"}
        foreach rule $rules {
            set rule_data {};foreach key [list_property $rule] {dict set rule_data $key [get_property $key $rule]}
            puts $audit [list rule_configuration $kind $rule_data review pending]
        }
        dict set counts $kind/rules [llength $rules]
        foreach violation [$command -quiet] {
            set data {};foreach key [list_property $violation] {dict set data $key [get_property $key $violation]}
            set rule [prop $violation CHECK]
            set check [$checks_command $rule]
            if {[names $check] ne [list $rule]} {error "C2 Alex route: ambiguous rule identity $rule"}
            set rule_data {};foreach key [list_property $check] {dict set rule_data $key [get_property $key $check]}
            # Use properties already read from the original typed objects.
            # Selecting an object through expr can discard its Vivado type.
            if {[dict exists $data SEVERITY]} {
                set severity [dict get $data SEVERITY]
            } elseif {[dict exists $rule_data SEVERITY]} {
                set severity [dict get $rule_data SEVERITY]
            } else {error "C2 Alex route: missing severity for $rule"}
            set severity [string toupper [string trim $severity]]
            if {$severity ni {FATAL ERROR {CRITICAL WARNING} WARNING ADVISORY INFO}} {error "C2 Alex route: unknown severity $severity"}
            dict incr counts $kind/$severity
            puts $audit [list finding $kind properties $data rule_properties $rule_data review pending]
            if {$kind eq "drc" && $severity in {FATAL ERROR {CRITICAL WARNING}}} {lappend blockers drc:$rule}
        }
    }
    return [dict create counts $counts blockers $blockers]
}
proc ::c2_alex_route::audit_linked {run} {
    set directory [fresh $run linked_audit]
    set audit [open $directory/objects.tcldata {WRONLY CREAT EXCL}]
    set status [catch {
        set topology [structure $audit]
        set groups [gray_inventory $audit]
        gray_witnesses $groups $audit
        write_xdc -type timing -constraints all -write_id $directory/timing_all_ids.xdc
        set exceptions [constraints $directory/timing_all_ids.xdc $audit]
        report_methodology -file $directory/methodology.rpt
        report_drc -file $directory/drc.rpt
        report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file $directory/timing.rpt
        report_waivers -file $directory/waivers.rpt
        report_cdc -details -show_waiver -all_checks_per_endpoint -file $directory/cdc.rpt
        report_clock_interaction -file $directory/clock_interaction.rpt
        report_exceptions -coverage -file $directory/exception_coverage.rpt
        report_exceptions -ignored -file $directory/ignored_exceptions.rpt
        set linked_findings [findings $audit]
        set supplemental [supplemental $directory $groups]
        set inventory {objects.tcldata timing_all_ids.xdc methodology.rpt drc.rpt timing.rpt waivers.rpt cdc.rpt clock_interaction.rpt exception_coverage.rpt ignored_exceptions.rpt supplemental.tcldata}
        flush $audit
        foreach name $inventory {if {![file isfile $directory/$name] || [file size $directory/$name]==0} {error "C2 Alex route: missing linked evidence $name"}}
        set result [dict create schema 1 stage linked structural_pass 1 route_qualified 0 report_directory $directory \
            reason linked_structure_only_pending_current_vendor_CDC_and_physical_review supplemental_evidence $supplemental report_inventory $inventory findings $linked_findings gray_bus_groups [dict size $groups] exception_counts [dict get $exceptions counts]]
        write_data $directory/summary.tcldata $result
    } message options]
    close $audit
    if {$status} {return -options $options $message}
    return $result
}
proc ::c2_alex_route::qualify {run checkpoint} {
    set checkpoint [file normalize $checkpoint]
    set digest [sha256 $checkpoint]
    set directory [fresh $run qualification]
    set audit [open $directory/objects.tcldata {WRONLY CREAT EXCL}]
    set status [catch {
        puts $audit [list checkpoint $checkpoint sha256 $digest]
        set topology [structure $audit]
        set groups [gray_inventory $audit]
        gray_witnesses $groups $audit
        set inventory [report_set $directory]
        set exceptions [constraints $directory/timing_all_ids.xdc $audit]
        set timing [timing_summary [read_text $directory/timing.rpt]]
        set route [route_summary [read_text $directory/route_status.rpt]]
        set rows [bus_rows [read_text $directory/bus_skew.rpt]]
        gray_coverage $groups $rows $audit
        dict for {id row} $rows {puts $audit [list bus_skew $id $row review pending_ip_scope_and_bounds]}
        set cdc [cdc_rows [read_text $directory/cdc.rpt]]
        foreach row $cdc {puts $audit [list cdc $row]}
        set findings [findings $audit]
        set supplemental [supplemental $directory $groups]
        lappend inventory supplemental.tcldata
        set blockers [concat [dict get $timing blockers] [dict get $findings blockers]]
        if {![dict get $route pass]} {lappend blockers routing_incomplete}
        set bad_nets [get_nets -quiet -hierarchical -filter {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL || ROUTE_STATUS == UNPLACED || ROUTE_STATUS == UNPLACED_ALL || ROUTE_STATUS == CONFLICTS || ROUTE_STATUS == GAPS || ROUTE_STATUS == ANTENNAS || ROUTE_STATUS == NODRIVER || ROUTE_STATUS == MULTI_DRIVER}]
        if {[llength $bad_nets]} {lappend blockers unresolved_nets;puts $audit [list unresolved_nets [names $bad_nets]]}
        foreach {label option} {setup -setup hold -hold} {
            set paths [get_timing_paths -quiet $option -max_paths 1 -slack_lesser_than 0]
            if {[llength $paths]} {lappend blockers negative_$label;puts $audit [list failing_path $label [get_property SLACK $paths]]}
            set paths [get_timing_paths -quiet $option -max_paths 1]
            if {[llength $paths]!=1 || [number [prop $paths SLACK]]<0} {lappend blockers missing_or_negative_${label}_witness}
        }
        dict for {id row} $rows {if {[dict get $row slack]<0 || [dict get $row requirement]<=0} {lappend blockers bus_skew:$id}}
        if {![regexp {(^|\n)No waivers found\.\s*$} [read_text $directory/waivers.rpt]]} {lappend blockers waiver_review_required}
        foreach row $cdc {if {[dict get $row waived] ne "N"} {lappend blockers waived_cdc}}
        # Actual current-design identities/exception coverage are review inputs,
        # not a historical two-port whitelist. Until reviewed classifiers exist,
        # this stage deliberately cannot manufacture a qualified image.
        set pending {current_vendor_CDC_and_gray_max_delay_scope current_exception_coverage_and_ignored_paths current_DRC_methodology_and_severity_configuration exact_shell_inherited_witnesses current_9SI_8MI_vendor_simulation_and_generated_IP_binding}
        set result [dict create schema 1 stage final_route route_qualified 0 hardware_measured 0 checkpoint $checkpoint checkpoint_sha256 $digest \
            report_directory $directory reason [expr {[llength $blockers] ? "physical_checks_failed_and_review_required" : "review_required"}] \
            blockers [lsort -unique $blockers] pending_reviews $pending supplemental_evidence $supplemental timing $timing routing $route \
            gray_bus_groups [dict size $groups] bus_skew_constraints [dict size $rows] cdc_records [llength $cdc] findings $findings report_inventory $inventory]
        if {[sha256 $checkpoint] ne $digest} {error "C2 Alex route: checkpoint changed during audit"}
        write_data $directory/summary.tcldata $result
        write_data $directory/NOT_QUALIFIED.txt [list reason [dict get $result reason] blockers [dict get $result blockers] pending_reviews $pending]
    } message options]
    close $audit
    if {$status} {return -options $options $message}
    puts "C2_ALEX_ROUTE_NOT_QUALIFIED reason=[dict get $result reason] directory=$directory"
    return $result
}
