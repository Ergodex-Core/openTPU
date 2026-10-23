# C2 HBM scoped-clock context classifier. This reads design/report objects only;
# it never changes timing exceptions, severities, waivers, or design state.
# Reviewed Vivado 2025.2 post-opt evidence: timing_all_ids.xdc SHA256
# 4a936f2645fad49648395d70903331ceb35a2fff79f94eaca46db887ff529854;
# scope_inventory.tcldata SHA256
# 3ba6a7ce75318bdc75164bb38e3b653089b494b86558ea569384214dfcb40a3b.
# This is context evidence, not a routed timing or CDC signoff.
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
        clk_raw {PERIOD 5.000 MASTER_CLOCK clk_main_a0 SOURCE_PINS WRAPPER/CL/u_mmcm/CLKOUT0}
        hclk_raw {PERIOD 2.222 MASTER_CLOCK clk_hbm_ref SOURCE_PINS WRAPPER/CL/u_hmmcm/CLKOUT0}
    }
}
proc ::otpu_scoped::pairs {} {
    return {clk_main_a0 clk_raw clk_raw clk_main_a0 clk_raw hclk_raw hclk_raw clk_raw clk_main_a0 hclk_raw hclk_raw clk_main_a0}
}
proc ::otpu_scoped::commands {} {
    return {
        {set_max_delay -datapath_only -from [get_clocks clk_main_a0] -to [get_clocks -of_objects [get_pins u_clk/O]] 5.000}
        {set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins u_clk/O]] -to [get_clocks clk_main_a0] 4.000}
        {set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins u_clk/O]] -to [get_clocks -of_objects [get_pins u_hclk/O]] 2.222}
        {set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins u_hclk/O]] -to [get_clocks -of_objects [get_pins u_clk/O]] 5.000}
        {set_max_delay -datapath_only -from [get_clocks clk_main_a0] -to [get_clocks -of_objects [get_pins u_hclk/O]] 2.222}
        {set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins u_hclk/O]] -to [get_clocks clk_main_a0] 4.000}
    }
}
proc ::otpu_scoped::command_pairs {} {
    set result {}; set i 0
    foreach {src dst} [pairs] {dict set result [lindex [commands] $i] [lsort [list $src $dst]]; incr i}
    return $result
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
    if {[dict get $path STARTPOINT_CLOCK] ne "clk_main_a0" ||
        [dict get $path STARTPOINT_PIN] ne {WRAPPER/RL_SHIM/PIPE_RST_OUT_N/pipe_reg[0][0]/C}} {return 0}
    switch -- [dict get $path ENDPOINT_CLOCK] {
        clk_raw {set stem core_reset_n}
        hclk_raw {set stem hbm_reset_n}
        default {return 0}
    }
    foreach bit {0 1 2} {
        if {[dict get $path ENDPOINT_PIN] eq [format {WRAPPER/CL/%s_reg[%d]/CLR} $stem $bit]} {return 1}
    }
    return 0
}
proc ::otpu_scoped::bound {path} {
    set dst [dict get $path ENDPOINT_CLOCK]
    if {![dict exists [clock_data] $dst]} {return 0}
    set period [dict get [clock_data] $dst PERIOD]
    return [expr {[dict get $path EXCEPTION] eq "MaxDelay Path ${period}ns -datapath_only" &&
        [dict get $path REQUIREMENT] eq $period}]
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
proc ::otpu_scoped::context_commands {} {
    # TIMING-54 applies to the four constraints involving the shell clock,
    # whose register membership extends outside WRAPPER/CL. The core/HBM pair
    # remains inside CL, but its two exported bounds are checked independently.
    set result {}
    foreach command [commands] {if {[string first {get_clocks clk_main_a0} $command] >= 0} {lappend result $command}}
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
    if {[dict size $exported_positions] != 6} {error "scoped clocks: reviewed commands missing or changed"}
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
        if {![regexp {^A scoped Max Delay Datapath Only timing constraint is set between clocks (clk_main_a0 and (?:hclk_raw|clk_raw)) \(see constraint position ([0-9]+) in the Timing Constraint window in Vivado IDE\)\. It is not recommended to define such scoped constraints between clocks as the constraint impacts timing paths outside of the scope\.$} $description -> clocks position]} {
            error "scoped clocks: unreviewed TIMING-54 description"
        }
        if {![dict exists $entries $position]} {error "scoped clocks: finding export position missing"}
        lassign [unique_export $entries $position] scope command
        if {$scope ne "WRAPPER/CL" || $command ni [context_commands] || $command in $matched} {
            error "scoped clocks: finding does not map uniquely to reviewed command"
        }
        set other [lindex $clocks 2]
        if {[dict get [command_pairs] $command] ne [lsort [list clk_main_a0 $other]]} {error "scoped clocks: finding clock pair differs from command"}
        puts $audit [list scoped_constraint $name position $position scope $scope command $command]
        lappend matched $command; lappend context $name
    }
    if {[llength $context] != 4 || [lsort $matched] ne [lsort [context_commands]]} {error "scoped clocks: expected exact four context findings"}
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
            if {$name ne "clk_main_a0" && ![string match {WRAPPER/CL/*} $cell]} {error "scoped clocks: core/HBM clock escapes CL"}
            if {![string match {WRAPPER/CL/*} $cell]} {lappend external $cell}
        }
        dict set outside $name $external
        puts $audit [list membership $name count [llength $member_names] outside_cl $external]
    }
    foreach {pin clock} {WRAPPER/CL/clk_main_a0 clk_main_a0 WRAPPER/CL/u_clk/O clk_raw WRAPPER/CL/u_hclk/O hclk_raw} {
        set selected [get_pins $pin]
        if {[names $selected] ne [list $pin] || [names [get_clocks -of_objects $selected]] ne [list $clock]} {
            error "scoped clocks: constraint clock-pin resolution mismatch"
        }
    }
    foreach {src dst} [pairs] {
        foreach ignored {0 1} {
            set tag ${src}_to_${dst}_ignored${ignored}
            set opt {}; if {$ignored} {set opt {-user_ignored}}
            set report ${prefix}_${tag}.rpt
            if {[file exists $report]} {error "scoped clocks: evidence report already exists"}
            report_timing -from [resolve clock $src] -to [resolve clock $dst] \
                -max_paths 3 -path_type full_clock_expanded {*}$opt -file $report
            if {[current_instance] ne ""} {error "scoped clocks: report changed current_instance"}
            if {$ignored} {set limit 10000; set nworst 10000} else {set limit 10; set nworst 1}
            set paths [paths $audit $tag clock $src clock $dst $ignored $limit $nworst]
            set data_count 0
            foreach path $paths {
                if {[dict get $path STARTPOINT_CLOCK] ne $src || [dict get $path ENDPOINT_CLOCK] ne $dst} {
                    error "scoped clocks: queried clock pair changed"
                }
                if {$ignored} {
                    if {![reset_path $path] || [dict get $path EXCEPTION] ne "False Path"} {error "scoped clocks: unexpected ignored data path"}
                } else {
                    if {![bound $path]} {error "scoped clocks: destination-period datapath bound missing"}
                    if {![reset_path $path]} {incr data_count}
                }
            }
            if {!$ignored && !$data_count} {error "scoped clocks: no data-path witness"}
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
                    if {![reset_path $path]} {error "scoped clocks: unexpected outside-boundary path"}
                    # Vivado's cell-to-clock queries return these exact reset
                    # False Paths both with and without -user_ignored. No data
                    # path or different reset endpoint is admitted here.
                    if {[dict get $path EXCEPTION] eq "False Path"} {
                        if {[dict get $path REQUIREMENT] ne ""} {error "scoped clocks: reset entry exception changed"}
                    } elseif {$ignored || ![bound $path]} {error "scoped clocks: reset entry exception changed"}
                }
            }
            foreach side {inputs outputs} {
                if {$side eq "inputs"} {
                    set paths [paths $audit ${tag}_${side} inputs {} clock $dst $ignored 10000 10000]
                } else {set paths [paths $audit ${tag}_${side} clock $src outputs {} $ignored 10000 10000]}
                if {[llength $paths]} {error "scoped clocks: unexpected top-level I/O path"}
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
