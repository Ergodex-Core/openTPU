# Independent C2/Alex linked and final-route evidence. No timing/waiver mutations.
# The executor must freshly open the hash-bound checkpoint before qualify and
# bind these reports, frozen inputs/generated BD, and current vendor simulation.
source [file join [file dirname [info script]] audit_c2_alex.tcl]
namespace eval ::c2_alex_route {variable cl WRAPPER/CL}
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
    if {[lsort [names [get_cells -quiet -regexp {^WRAPPER/CL/u_native[0-9]+$}]]] ne $expected} {error "C2 Alex route: native topology changed"}
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
    ::c2_alex::interfaces $interface_pins
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
        set connected [get_clocks -quiet -of_objects [get_pins -quiet $cl/u_mem/${label}_clk]]
        if {[names $connected] ne [names [dict get $clocks $label]]} {error "C2 Alex route: $label fabric clock disconnected"}
        set registers [all_registers -clock [dict get $clocks $label] -cells]
        if {![llength $registers]} {error "C2 Alex route: empty $label clock domain"}
        puts $audit [list clock_registers $label count [llength $registers]]
        if {$label eq "hbm"} {
            foreach name [names $registers] {
                if {[owner $name] ni {u_mem u_hbm u_rst_hbm}} {error "C2 Alex route: unexpected HBM-clock register $name"}
            }
        }
    }
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
            set owner [expr {"SEVERITY" in [list_property $violation] ? $violation : $check}]
            set severity [string toupper [string trim [prop $owner SEVERITY]]]
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
        report_cdc -details -show_waiver -all_checks_per_endpoint -file $directory/cdc.rpt
        report_clock_interaction -file $directory/clock_interaction.rpt
        report_exceptions -coverage -file $directory/exception_coverage.rpt
        report_exceptions -ignored -file $directory/ignored_exceptions.rpt
        foreach name {timing_all_ids.xdc cdc.rpt clock_interaction.rpt exception_coverage.rpt ignored_exceptions.rpt} {if {![file isfile $directory/$name] || [file size $directory/$name]==0} {error "C2 Alex route: missing linked evidence $name"}}
        set result [dict create schema 1 stage linked structural_pass 1 route_qualified 0 report_directory $directory \
            reason linked_structure_only_pending_current_vendor_CDC_and_physical_review gray_bus_groups [dict size $groups] exception_counts [dict get $exceptions counts]]
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
            blockers [lsort -unique $blockers] pending_reviews $pending timing $timing routing $route \
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
