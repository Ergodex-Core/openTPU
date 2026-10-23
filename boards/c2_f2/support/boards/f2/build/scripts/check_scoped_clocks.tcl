# Optional TIMING-54 context classification, never a timing waiver. The caller
# must apply the normal severity policy to `remaining` and retain timing gates.
# Reviewed evidence: shell-methodology-evidence-001.json SHA256
# c95e4fd49e6d9428f2626ade7bb6cc85b24898d8f1b4ac11908be0cab808e6d2;
# clock-query-evidence-001.json SHA256
# a12cf9bd579fd0f01c4ef0c874424cd09612a67c109495f6aaa2e011138408b5.
namespace eval ::otpu_scoped {}

proc ::otpu_scoped::property {object key} {
    if {$key ni [list_property $object]} {error "scoped clocks: missing $key on $object"}
    return [format %s [get_property $key $object]]
}
proc ::otpu_scoped::values {objects key} {
    # UG835: get_property returns one value per object, including an empty
    # placeholder for an absent property. Batch the query to avoid expensive
    # per-cell schema enumeration across the complete register inventory.
    if {![llength $objects]} {return {}}
    set result [get_property $key $objects]
    # A single object's result is a scalar, which may contain Tcl list syntax.
    if {[llength $objects] == 1} {set result [list $result]}
    if {[llength $result] != [llength $objects]} {
        error "scoped clocks: $key result cardinality mismatch"
    }
    set checked {}
    foreach value $result {
        if {$value eq ""} {error "scoped clocks: missing or empty $key value"}
        lappend checked $value
    }
    return $checked
}
proc ::otpu_scoped::names {objects} {
    # Preserve order and duplicates for the caller's exact identity checks.
    return [values $objects NAME]
}
proc ::otpu_scoped::clock_data {} {
    return {
        clk_main_a0 {PERIOD 4.000 MASTER_CLOCK clk_fpga SOURCE_PINS WRAPPER/IO_SHIM/RL_CLKS/MMCM_CLK_MAIN_A/inst/CLK_CORE_DRP_I/clk_inst/mmcme4_adv_inst/CLKOUT0}
        clk_raw {PERIOD 10.000 MASTER_CLOCK clk_main_a0 SOURCE_PINS WRAPPER/CL/u_mmcm/CLKOUT0}
        mmcm_clkout0 {PERIOD 3.752 MASTER_CLOCK CLK_DIMM_DP SOURCE_PINS WRAPPER/CL/SH_DDR/genblk1.IS_DDR_PRESENT.DDR4_0/inst/u_ddr4_infrastructure/gen_mmcme4.u_mmcme_adv_inst/CLKOUT0}
    }
}
proc ::otpu_scoped::commands {} {
    return {
        {set_clock_groups -asynchronous -group [get_clocks clk_main_a0] -group [get_clocks -of_objects [get_pins -hierarchical -filter NAME=~*SH_DDR/genblk1.IS_DDR_PRESENT.DDR4_0/inst/u_ddr4_infrastructure/gen_mmcme4.u_mmcme_adv_inst/CLKOUT0]]}
        {set_max_delay -datapath_only -from [get_clocks clk_main_a0] -to [get_clocks -of_objects [get_pins u_clk/O]] 4.000}
        {set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins u_clk/O]] -to [get_clocks clk_main_a0] 4.000}
    }
}
proc ::otpu_scoped::export_record {entries position scope commands} {
    if {$position eq "" && ![llength $commands]} {return $entries}
    if {![llength $commands]} {error "scoped clocks: export position without a command"}
    # Vivado repeats id:0 for generated constraints, and can emit a temporary
    # variable assignment plus a constraint under one ID. Preserve all records
    # and all commands; only a reviewed/reference position must be unambiguous.
    dict lappend entries $position [list $scope $commands]
    return $entries
}
proc ::otpu_scoped::read_export {path} {
    # Parse as data, never source/eval/subst exported constraints. info complete
    # only checks Tcl syntax; it does not execute bracket/variable substitutions.
    set f [open $path r]; set text [read $f]; close $f
    set result {}; set scope {}; set position {}; set commands {}; set pending {}
    foreach raw [split $text \n] {
        set line [string trim $raw]
        if {$pending eq ""} {
            if {[regexp {^# id:([0-9]+)\s*$} $line -> id]} {
                set result [export_record $result $position $scope $commands]
                set position $id; set commands {}
                continue
            }
            if {$line eq "" || [string index $line 0] eq "#"} {continue}
        }
        append pending $raw \n
        # A trailing unescaped backslash continues the command even when Tcl's
        # syntax-completeness check accepts the whitespace continuation alone.
        if {[regexp {(^|[^\\])(\\\\)*\\$} $raw] || ![info complete $pending]} {continue}
        set command [string trim $pending]; set pending {}
        if {[regexp {^current_instance(?:\s|$)} $command]} {
            set result [export_record $result $position $scope $commands]
            set position {}; set commands {}
            if {[catch {llength $command} count]} {error "scoped clocks: malformed export scope"}
            set args [lrange $command 1 end]
            if {[lindex $args 0] eq "-quiet"} {set args [lrange $args 1 end]}
            if {[llength $args] > 1} {error "scoped clocks: malformed export scope"}
            set scope [lindex $args 0]
            # The pinned writer uses a literal hierarchy name or no argument
            # (`current_instance -quiet`) to restore the top-level scope.
            if {$scope ne "" && ![regexp {^[A-Za-z0-9_/.[\]-]+$} $scope]} {
                error "scoped clocks: malformed export scope"
            }
            continue
        }
        lappend commands $command
    }
    if {$pending ne ""} {error "scoped clocks: incomplete export command"}
    return [export_record $result $position $scope $commands]
}
proc ::otpu_scoped::unique_export {entries position} {
    set records [dict get $entries $position]
    if {$position eq "" || [llength $records] != 1} {error "scoped clocks: ambiguous export position"}
    lassign [lindex $records 0] scope commands
    if {[llength $commands] != 1} {error "scoped clocks: ambiguous export command bundle"}
    return [list $scope [lindex $commands 0]]
}
proc ::otpu_scoped::resolve {kind selection} {
    switch -- $kind {
        clock {
            set objects [get_clocks $selection]
            if {[names $objects] ne [list $selection]} {error "scoped clocks: clock identity mismatch: $selection"}
            if {[property [lindex $objects 0] CLASS] ne "clock"} {error "scoped clocks: clock class mismatch"}
        }
        cells {
            set objects [get_cells $selection]
            if {[lsort [names $objects]] ne [lsort $selection]} {error "scoped clocks: cell selection mismatch"}
            foreach class [values $objects CLASS] {
                if {$class ne "cell"} {error "scoped clocks: cell class mismatch"}
            }
        }
        inputs {set objects [all_inputs]}
        outputs {set objects [all_outputs]}
        default {error "scoped clocks: invalid query kind"}
    }
    return $objects
}
proc ::otpu_scoped::reset_path {path} {
    if {[dict get $path STARTPOINT_CLOCK] ne "clk_main_a0" || [dict get $path ENDPOINT_CLOCK] ne "clk_raw" ||
        [dict get $path STARTPOINT_PIN] ne {WRAPPER/RL_SHIM/PIPE_RST_OUT_N/pipe_reg[0][0]/C}} {return 0}
    return [expr {[dict get $path ENDPOINT_PIN] in {
        {WRAPPER/CL/core_reset_n_reg[0]/CLR}
        {WRAPPER/CL/core_reset_n_reg[1]/CLR}
        {WRAPPER/CL/core_reset_n_reg[2]/CLR}
    }}]
}
proc ::otpu_scoped::bound4 {path} {
    return [expr {[dict get $path EXCEPTION] eq "MaxDelay Path 4.000ns -datapath_only" &&
        [dict get $path REQUIREMENT] eq "4.000"}]
}
proc ::otpu_scoped::paths {audit tag sk sn dk dn ignored limit nworst} {
    set opt {}; if {$ignored} {set opt {-user_ignored}}
    # Resolve canonical strings immediately at the call. No object container is
    # retained across any report command (which may use tool-global Tcl state).
    set objects [get_timing_paths -from [resolve $sk $sn] -to [resolve $dk $dn] \
        -max_paths $limit -nworst $nworst {*}$opt]
    puts $audit [list query $tag count [llength $objects] limit $limit]
    if {$limit == 10000 && [llength $objects] >= $limit} {error "scoped clocks: query limit reached: $tag"}
    set result {}
    foreach object $objects {
        set path {}
        foreach key {STARTPOINT_PIN ENDPOINT_PIN STARTPOINT_CLOCK ENDPOINT_CLOCK EXCEPTION REQUIREMENT SLACK DATAPATH_DELAY} {
            dict set path $key [property $object $key]
        }
        puts $audit [list path $tag $path]
        lappend result $path
    }
    flush $audit
    return $result
}
proc ::otpu_scoped::main {violation_names prefix audit} {
    if {[version -short] ne "2025.2"} {error "scoped clocks: unreviewed Vivado version"}
    if {[current_instance] ne ""} {error "scoped clocks: queries require top-level current_instance"}
    set live_names [names [get_methodology_violations]]
    if {[llength [lsort -unique $live_names]] != [llength $live_names] ||
        [llength [lsort -unique $violation_names]] != [llength $violation_names] ||
        [lsort $violation_names] ne [lsort $live_names]} {error "scoped clocks: incomplete live violation inventory"}
    set export ${prefix}_scoped_clocks.xdc
    if {[file exists $export]} {error "scoped clocks: evidence export already exists"}
    write_xdc -type timing -constraints all -write_id $export
    set entries [read_export $export]
    set expected [commands]
    set exported_positions {}
    dict for {id records} $entries {
        foreach record $records {
            lassign $record scope commands
            foreach command $commands {
                if {$command in $expected} {
                    unique_export $entries $id
                    if {$scope ne "WRAPPER/CL" || [dict exists $exported_positions $command]} {
                        error "scoped clocks: wrong scope or duplicate reviewed command"
                    }
                    dict set exported_positions $command $id
                }
            }
        }
    }
    if {[dict size $exported_positions] != 3} {error "scoped clocks: reviewed commands missing or changed"}
    set context {}; set remaining {}; set matched {}
    foreach name [lsort $violation_names] {
        set finding [get_methodology_violations $name]
        if {[names $finding] ne [list $name]} {error "scoped clocks: violation identity mismatch"}
        set check [property $finding CHECK]
        if {$check ne "TIMING-54"} {
            if {[string match {TIMING-54#*} $name]} {error "scoped clocks: inconsistent TIMING-54 CHECK"}
            lappend remaining $name; continue
        }
        if {![regexp {^TIMING-54#[0-9]+$} $name] || [property $finding CLASS] ne "drc_violation" ||
            [property $finding SEVERITY] ne "Critical Warning" || [property $finding WAIVER] ne ""} {
            error "scoped clocks: unreviewed finding identity/severity/waiver"
        }
        set waived [property $finding IS_WAIVED]
        if {![string is boolean -strict $waived] || $waived} {error "scoped clocks: waived or unknown waiver state"}
        set description [property $finding DESCRIPTION]
        if {![regexp {^A scoped (Clock Group|Max Delay Datapath Only) timing constraint is set between clocks (clk_main_a0 and (?:mmcm_clkout0|clk_raw)) \(see constraint position ([0-9]+) in the Timing Constraint window in Vivado IDE\)\. It is not recommended to define such scoped constraints between clocks as the constraint impacts timing paths outside of the scope\.$} $description -> kind clocks position]} {
            error "scoped clocks: unreviewed TIMING-54 description"
        }
        if {![dict exists $entries $position]} {error "scoped clocks: finding export position missing"}
        lassign [unique_export $entries $position] scope command
        if {$scope ne "WRAPPER/CL" || $command ni $expected || $command in $matched} {
            error "scoped clocks: finding does not map uniquely to reviewed command"
        }
        if {$command eq [lindex $expected 0]} {
            if {$kind ne "Clock Group" || $clocks ne "clk_main_a0 and mmcm_clkout0"} {error "scoped clocks: DDR finding mismatch"}
        } elseif {$kind ne "Max Delay Datapath Only" || $clocks ne "clk_main_a0 and clk_raw"} {error "scoped clocks: main/core finding mismatch"}
        puts $audit [list scoped_constraint $name position $position scope $scope command $command]
        lappend matched $command; lappend context $name
    }
    if {[llength $context] != 3 || [lsort $matched] ne [lsort $expected]} {error "scoped clocks: expected exact three context findings"}
    set outside {}; set clock_data [clock_data]
    dict for {name props} $clock_data {
        set object [resolve clock $name]
        dict for {key value} [dict merge $props {IS_GENERATED 1 IS_VIRTUAL 0 IS_PROPAGATED 1}] {
            if {[property $object $key] ne $value} {error "scoped clocks: $name $key differs from reviewed clock"}
        }
        puts $audit [list clock $name properties $props]
        set member_names [names [all_registers -clock [resolve clock $name] -cells]]
        if {![llength $member_names] || [llength [lsort -unique $member_names]] != [llength $member_names]} {
            error "scoped clocks: missing or duplicated clock membership"
        }
        set external {}
        foreach cell $member_names {
            if {$name eq "clk_raw" && ![string match {WRAPPER/CL/*} $cell]} {error "scoped clocks: core clock escapes CL"}
            if {$name eq "mmcm_clkout0" && ![string match {WRAPPER/CL/SH_DDR/*} $cell]} {error "scoped clocks: DDR clock escapes SH_DDR"}
            if {![string match {WRAPPER/CL/*} $cell]} {lappend external $cell}
        }
        dict set outside $name $external
        puts $audit [list membership $name count [llength $member_names] outside_cl $external]
    }
    foreach {pin clock} {
        WRAPPER/CL/clk_main_a0 clk_main_a0
        WRAPPER/CL/u_clk/O clk_raw
        WRAPPER/CL/SH_DDR/genblk1.IS_DDR_PRESENT.DDR4_0/inst/u_ddr4_infrastructure/gen_mmcme4.u_mmcme_adv_inst/CLKOUT0 mmcm_clkout0
    } {
        set selected [get_pins $pin]
        if {[names $selected] ne [list $pin] || [names [get_clocks -of_objects $selected]] ne [list $clock]} {
            error "scoped clocks: constraint clock-pin resolution mismatch"
        }
    }
    # The exported DDR selector is a wildcard. Prove it resolves to only the
    # reviewed infrastructure pin, including against an accidental extra copy.
    set ddr_pins [get_pins -hierarchical -filter {NAME=~*SH_DDR/genblk1.IS_DDR_PRESENT.DDR4_0/inst/u_ddr4_infrastructure/gen_mmcme4.u_mmcme_adv_inst/CLKOUT0}]
    if {[names $ddr_pins] ne [list [dict get $clock_data mmcm_clkout0 SOURCE_PINS]] ||
        [names [get_clocks -of_objects $ddr_pins]] ne {mmcm_clkout0}} {
        error "scoped clocks: DDR wildcard clock selection changed"
    }
    foreach {src dst} {clk_main_a0 clk_raw clk_raw clk_main_a0 clk_main_a0 mmcm_clkout0 mmcm_clkout0 clk_main_a0} {
        set is_core [expr {$src eq "clk_raw" || $dst eq "clk_raw"}]
        foreach ignored {0 1} {
            set tag ${src}_to_${dst}_ignored${ignored}
            set opt {}; if {$ignored} {set opt {-user_ignored}}
            set report ${prefix}_${tag}.rpt
            if {[file exists $report]} {error "scoped clocks: evidence report already exists"}
            report_timing -from [resolve clock $src] -to [resolve clock $dst] \
                -max_paths 3 -path_type full_clock_expanded {*}$opt -file $report
            if {[current_instance] ne ""} {error "scoped clocks: report changed current_instance"}
            if {$is_core && $ignored} {set limit 10000; set nworst 10000} else {set limit 10; set nworst 1}
            set paths [paths $audit $tag clock $src clock $dst $ignored $limit $nworst]
            set data_count 0
            foreach path $paths {
                if {[dict get $path STARTPOINT_CLOCK] ne $src || [dict get $path ENDPOINT_CLOCK] ne $dst} {
                    error "scoped clocks: queried clock pair changed"
                }
                if {$is_core} {
                    if {$ignored} {
                        if {![reset_path $path] || [dict get $path EXCEPTION] ne "False Path"} {error "scoped clocks: unexpected ignored main/core data path"}
                    } else {
                        if {![bound4 $path]} {error "scoped clocks: main/core datapath bound missing"}
                        if {![reset_path $path]} {incr data_count}
                    }
                } elseif {[dict get $path EXCEPTION] ne "Asynchronous Clock Groups"} {
                    error "scoped clocks: DDR clock-group exception missing"
                }
            }
            if {$is_core && !$ignored && !$data_count} {error "scoped clocks: no main/core data-path witness"}
            if {!$is_core && ![llength $paths]} {error "scoped clocks: no DDR clock-group witness"}
            foreach side {source destination} {
                if {$side eq "source"} {set selected [dict get $outside $src]} else {set selected [dict get $outside $dst]}
                if {![llength $selected]} {continue}
                if {$side eq "source"} {
                    set paths [paths $audit ${tag}_${side} cells $selected clock $dst $ignored 10000 10000]
                } else {set paths [paths $audit ${tag}_${side} clock $src cells $selected $ignored 10000 10000]}
                foreach path $paths {
                    if {[dict get $path STARTPOINT_CLOCK] ne $src || [dict get $path ENDPOINT_CLOCK] ne $dst} {
                        error "scoped clocks: outside register query found another clock pair"
                    }
                    if {!$is_core || ![reset_path $path]} {error "scoped clocks: unexpected outside-boundary path"}
                    if {![bound4 $path] && [dict get $path EXCEPTION] ne "False Path"} {error "scoped clocks: reset entry exception changed"}
                }
            }
            foreach side {inputs outputs} {
                if {$side eq "inputs"} {
                    set paths [paths $audit ${tag}_${side} inputs {} clock $dst $ignored 10000 10000]
                } else {set paths [paths $audit ${tag}_${side} clock $src outputs {} $ignored 10000 10000]}
                foreach path $paths {
                    # The DDR reset port is not a member of the main/DDR pair:
                    # its endpoint has no clock and the stock 50 ns bound applies.
                    if {$side ne "outputs" || $is_core || $src ne "mmcm_clkout0" || $ignored ||
                        [dict get $path STARTPOINT_CLOCK] ne "mmcm_clkout0" || [dict get $path ENDPOINT_CLOCK] ne "" ||
                        [dict get $path STARTPOINT_PIN] ne {WRAPPER/CL/SH_DDR/genblk1.IS_DDR_PRESENT.DDR4_0/inst/u_ddr4_mem_intfc/u_ddr_cal_top/cal_RESET_n_reg[0]/C} ||
                        [dict get $path ENDPOINT_PIN] ne "RST_DIMM_N" ||
                        [dict get $path EXCEPTION] ne "MaxDelay Path 50.000ns" || [dict get $path REQUIREMENT] ne "50.000"} {
                        error "scoped clocks: unexpected top-level I/O path"
                    }
                }
            }
        }
    }
    foreach name $context {puts $audit [list methodology $name severity {Critical Warning} classification reviewed_scoped_clock_context]}
    flush $audit
    return [dict create context $context remaining $remaining export $export]
}
proc otpu_classify_scoped_clocks {violation_names evidence_prefix audit} {
    return [::otpu_scoped::main $violation_names $evidence_prefix $audit]
}
