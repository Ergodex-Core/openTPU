# C2/Alex synthesis evidence only. Source after synth_design at top-level scope,
# then call ::c2_alex::audit_synthesis $run. This never writes a route-pass marker,
# changes timing exceptions/severities/waivers, or accepts a physical image.
namespace eval ::c2_alex {}

proc ::c2_alex::property {object key} {
    if {$key ni [list_property $object]} {error "C2 Alex audit: missing $key on $object"}
    return [get_property $key $object]
}
proc ::c2_alex::names {objects} {
    if {![llength $objects]} {return {}}
    set result [get_property NAME $objects]
    if {[llength $objects] == 1} {set result [list $result]}
    if {[llength $result] != [llength $objects]} {error "C2 Alex audit: name cardinality mismatch"}
    foreach name $result {if {$name eq ""} {error "C2 Alex audit: empty object name"}}
    return $result
}
proc ::c2_alex::cell {name expected_ref concrete} {
    set object [get_cells -quiet $name]
    if {[names $object] ne [list $name]} {error "C2 Alex audit: missing/ambiguous cell $name"}
    set ref [property $object REF_NAME]
    if {"ORIG_REF_NAME" in [list_property $object] && [get_property ORIG_REF_NAME $object] ne ""} {
        set ref [get_property ORIG_REF_NAME $object]
    }
    if {$ref ne $expected_ref} {error "C2 Alex audit: $name has unexpected reference $ref"}
    set opaque [property $object IS_BLACKBOX]
    if {![string is boolean -strict $opaque] || ($concrete && $opaque)} {
        error "C2 Alex audit: $name is unresolved or has unknown black-box state"
    }
    return $object
}
proc ::c2_alex::vendor_owner {name} {
    foreach root {u_mem u_hbm u_ocl_cdc u_pcis_pipe u_core_clk u_hbm_clk u_rst_main u_rst_core u_rst_hbm u_ready_cdc u_status_cdc} {
        if {$name eq $root || [string first "$root/" $name] == 0} {return $root}
    }
    return {}
}
proc ::c2_alex::opaque_roots {audit} {
    set result {}
    foreach object [get_cells -quiet -hierarchical -filter {IS_BLACKBOX}] {
        set name [property $object NAME]
        set owner [vendor_owner $name]
        if {$owner eq ""} {error "C2 Alex audit: unexpected unresolved cell $name"}
        dict lappend result $owner $name
        puts $audit [list opaque_vendor_cell $name reference [property $object REF_NAME]]
    }
    return $result
}
proc ::c2_alex::interfaces {pins} {
    # Validate exposed netlist interface widths even when an OOC IP checkpoint
    # is opaque. Address segments inside SmartConnect must ALSO be checked in
    # the generated BD by the build driver; pin widths do not prove its map.
    set seen {}; set bits {}; set interfaces {}
    foreach name $pins {
        if {![regexp {^u_mem/((?:S_HOST|S_N[0-9]+|M_H[0-9]+))_([a-z0-9]+)(?:\[([0-9]+)\])?$} $name -> iface signal bit]} {continue}
        dict set interfaces $iface 1
        set key $iface/$signal
        if {$bit eq ""} {dict incr seen $key} else {dict lappend bits $key $bit}
    }
    set expected {S_HOST}
    for {set p 0} {$p < 8} {incr p} {lappend expected S_N$p M_H$p}
    if {[lsort [dict keys $interfaces]] ne [lsort $expected]} {error "C2 Alex audit: expected nine SI/eight MI interfaces"}
    foreach iface $expected {
        foreach signal {awvalid awready wvalid wready bvalid bready arvalid arready rvalid rready} {
            if {![dict exists $seen $iface/$signal] || [dict get $seen $iface/$signal] != 1} {
                error "C2 Alex audit: missing/duplicate scalar $iface/$signal"
            }
        }
        set host [expr {$iface eq "S_HOST"}]; set master [string match M_H* $iface]
        set address_width [expr {$host ? 64 : 34}]
        set data_width [expr {$master ? 256 : 512}]
        foreach {signal width} [list awaddr $address_width araddr $address_width wdata $data_width rdata $data_width wstrb [expr {$data_width/8}]] {
            if {![dict exists $bits $iface/$signal]} {error "C2 Alex audit: missing vector $iface/$signal"}
            set actual [lsort -integer [dict get $bits $iface/$signal]]
            set wanted {}; for {set bit 0} {$bit < $width} {incr bit} {lappend wanted $bit}
            if {$actual ne $wanted} {error "C2 Alex audit: wrong width/duplicate bit for $iface/$signal"}
        }
    }
    return [dict create slave_interfaces 9 master_interfaces 8]
}
proc ::c2_alex::check_constraints {path} {
    # Treat exported XDC as text/list data, never execute it. Custom multicycles
    # and the old TMEM exceptions have no place in this standalone C2 kernel.
    set file [open $path r]; set text [read $file]; close $file
    set pending {}; set scope {}; set vendor_multicycles 0
    foreach line [split $text \n] {
        if {$pending eq "" && ([string trim $line] eq "" || [string index [string trim $line] 0] eq "#")} {continue}
        append pending $line \n
        if {[regexp {(^|[^\\])(\\\\)*\\$} $line] || ![info complete $pending]} {continue}
        set command [string trim $pending]; set pending {}
        if {[regexp -nocase {u_tmem|otpu_clk2x|OTPU_CORE_TMEM} $command]} {error "C2 Alex audit: inapplicable TMEM timing constraint"}
        if {[regexp {^current_instance(?:\s|$)} $command]} {
            if {[catch {llength $command}]} {error "C2 Alex audit: malformed constraint scope"}
            set args [lrange $command 1 end]
            if {[lindex $args 0] eq "-quiet"} {set args [lrange $args 1 end]}
            if {[llength $args] > 1} {error "C2 Alex audit: ambiguous constraint scope"}
            set scope [lindex $args 0]
        }
        if {[regexp {(^|[;\n])\s*set_multicycle_path(?:\s|$)} $command]} {
            if {[vendor_owner $scope] eq ""} {error "C2 Alex audit: unreviewed custom multicycle constraint"}
            incr vendor_multicycles
        }
    }
    if {$pending ne ""} {error "C2 Alex audit: incomplete constraint export"}
    return $vendor_multicycles
}
proc ::c2_alex::clock {label pin period owner opaque audit} {
    set selected [get_pins -quiet $pin]
    if {$label eq "tmem_unused" && ![llength $selected]} {
        # OOC synthesis may remove this unconnected output pin. A retained
        # 400 MHz clock still has to pass the zero-consumer check below.
        set clocks {}
        foreach candidate [get_clocks -quiet *] {
            set actual [property $candidate PERIOD]
            if {abs($actual-$period)>0.001} {continue}
            set sources [property $candidate SOURCE_PINS]
            if {![llength $sources]} {error "C2 Alex audit: unattributed unused-output clock"}
            foreach source $sources {
                if {[string first "$owner/" $source] != 0} {error "C2 Alex audit: unexpected 400 MHz clock source $source"}
            }
            lappend clocks $candidate
        }
        if {![llength $clocks]} {
            set state [expr {[dict exists $opaque $owner] ? "deferred_opaque_ip" : "pruned_unused_output"}]
            puts $audit [list clock $label expected_period $period state $state]
            return [dict create objects {} state $state]
        }
    } else {
        if {[names $selected] ne [list $pin]} {error "C2 Alex audit: missing clock pin $pin"}
        set clocks [get_clocks -quiet -of_objects $selected]
    }
    if {![llength $clocks] && [dict exists $opaque $owner]} {
        puts $audit [list clock $label expected_period $period state deferred_opaque_ip]
        return [dict create objects {} state deferred_opaque_ip]
    }
    if {[llength $clocks] != 1} {error "C2 Alex audit: missing/ambiguous $label clock"}
    set actual [property $clocks PERIOD]
    if {![string is double -strict $actual] || abs($actual-$period)>0.001} {
        error "C2 Alex audit: $label period $actual differs from $period ns"
    }
    set data [dict create NAME [property $clocks NAME] PERIOD $actual]
    foreach key {MASTER_CLOCK SOURCE_PINS IS_GENERATED IS_PROPAGATED IS_VIRTUAL} {
        if {$key in [list_property $clocks]} {dict set data $key [get_property $key $clocks]}
    }
    puts $audit [list clock $label state observed properties $data]
    return [dict create objects $clocks state observed]
}
proc ::c2_alex::audit_synthesis {run} {
    if {[version -short] ne "2025.2"} {error "C2 Alex audit: unreviewed Vivado version"}
    if {[current_instance] ne ""} {error "C2 Alex audit: requires top-level synthesis scope"}
    set directory [file join $run synth_audit]
    if {[file exists $directory]} {error "C2 Alex audit: refusing existing evidence directory"}
    file mkdir $directory
    set audit [open [file join $directory objects.tcldata] {WRONLY CREAT EXCL}]
    set status [catch {
        puts $audit [list evidence kind synthesis_only route_qualified 0 hardware_measured 0 tool [version -short]]
        cell u_board otpu_c2_hbm_kernel 1
        cell u_board/u_mxu otpu_c2_mxu 1
        set natives [names [get_cells -quiet -regexp {^u_native[0-9]+$}]]
        set expected {}; for {set p 0} {$p < 8} {incr p} {lappend expected u_native$p; cell u_native$p otpu_native_axi 1}
        if {[lsort $natives] ne $expected} {error "C2 Alex audit: expected exactly eight native masters"}
        foreach {name ref} {u_mem f2_mem_wrapper u_hbm cl_mem_hbm_wrapper u_ocl_cdc f2_ocl_cdc u_pcis_pipe f2_pcis_pipe u_core_clk f2_core_clk u_hbm_clk f2_hbm_clk} {
            cell $name $ref 0
        }
        if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ otpu_tmem* || ORIG_REF_NAME =~ otpu_tmem*}]]} {
            error "C2 Alex audit: unexpected TMEM implementation"
        }
        set topology [interfaces [names [get_pins -quiet -of_objects [get_cells u_mem]]]]
        puts $audit [list interface_topology $topology address_map_evidence generated_BD_required]
        set opaque [opaque_roots $audit]
        foreach {name period} {clk_main_a0 4.000 clk_hbm_ref 10.000} {
            set clock [get_clocks -quiet $name]
            if {[names $clock] ne [list $name] || abs([property $clock PERIOD]-$period)>0.001} {error "C2 Alex audit: wrong input clock $name"}
            puts $audit [list input_clock $name period [property $clock PERIOD]]
        }
        set clocks {}; set deferred {}; set pruned {}
        foreach {label pin period owner} {core u_core_clk/clk_out1 5.000 u_core_clk tmem_unused u_core_clk/clk_out2 2.500 u_core_clk hbm u_hbm_clk/clk_out1 2.222222222222 u_hbm_clk} {
            set evidence [clock $label $pin $period $owner $opaque $audit]
            set selected [dict get $evidence objects]
            dict set clocks $label $selected
            if {[dict get $evidence state] eq "deferred_opaque_ip"} {lappend deferred $label}
            if {[dict get $evidence state] eq "pruned_unused_output"} {lappend pruned $label}
        }
        foreach {label pin} {core u_mem/core_clk hbm u_mem/hbm_clk} {
            set selected [get_clocks -quiet -of_objects [get_pins -quiet $pin]]
            if {[names $selected] ne [names [dict get $clocks $label]]} {error "C2 Alex audit: vendor $label clock disconnected"}
        }
        if {[names [get_clocks -quiet -of_objects [get_pins -quiet u_mem/host_clk]]] ne {clk_main_a0}} {error "C2 Alex audit: vendor host clock disconnected"}
        set hclk_visible 0
        if {[llength [dict get $clocks hbm]]} {
            foreach name [names [all_registers -clock [dict get $clocks hbm] -cells]] {
                if {[vendor_owner $name] ni {u_mem u_hbm u_rst_hbm}} {error "C2 Alex audit: custom HBM-clock register $name"}
                incr hclk_visible
            }
        }
        if {[llength [dict get $clocks tmem_unused]] && [llength [all_registers -clock [dict get $clocks tmem_unused] -cells]]} {
            error "C2 Alex audit: unexpected consumer on unused 400 MHz clock"
        }
        set export [file join $directory timing_all_ids.xdc]
        write_xdc -type timing -constraints all -write_id $export
        set vendor_multicycles [check_constraints $export]
        report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file [file join $directory timing.rpt]
        report_drc -file [file join $directory drc.rpt]
        report_cdc -details -show_waiver -all_checks_per_endpoint -file [file join $directory cdc.rpt]
        report_bus_skew -file [file join $directory bus_skew.rpt]
        report_clock_interaction -file [file join $directory clock_interaction.rpt]
        report_exceptions -coverage -file [file join $directory exception_coverage.rpt]
        report_exceptions -ignored -file [file join $directory ignored_exceptions.rpt]
        report_methodology -file [file join $directory methodology.rpt]
        report_waivers -file [file join $directory waivers.rpt]
        report_utilization -hierarchical -file [file join $directory utilization.rpt]
        foreach {kind objects} [list drc [get_drc_violations -quiet] methodology [get_methodology_violations -quiet]] {
            foreach object $objects {
                set data {}
                foreach key {NAME CHECK SEVERITY IS_WAIVED WAIVER DESCRIPTION} {
                    if {$key in [list_property $object]} {dict set data $key [get_property $key $object]}
                }
                puts $audit [list finding $kind $data]
            }
        }
        set reports {objects.tcldata timing_all_ids.xdc timing.rpt drc.rpt cdc.rpt bus_skew.rpt clock_interaction.rpt exception_coverage.rpt ignored_exceptions.rpt methodology.rpt waivers.rpt utilization.rpt}
        flush $audit
        foreach name $reports {
            set path [file join $directory $name]
            if {![file isfile $path] || [file size $path] == 0} {error "C2 Alex audit: missing/empty evidence $name"}
        }
        set summary [dict create schema 1 stage synthesis_only route_qualified 0 hardware_measured 0 \
            kernel otpu_c2_hbm_kernel native_masters 8 slave_interfaces 9 master_interfaces 8 \
            opaque_vendor_roots [dict keys $opaque] deferred_clocks $deferred pruned_unused_outputs $pruned visible_hclk_registers $hclk_visible \
            custom_hclk_visible_registers 0 vendor_multicycles_pending_review $vendor_multicycles \
            address_map_validation generated_BD_required report_inventory $reports \
            required_next_steps {generated_IP_hash_and_BD_map_validation actual_9SI_8MI_vendor_simulation full_shell_link_and_complete_clock_CDC_audit physical_route_qualification package_hash_binding exclusive_hardware_session}]
        set file [open [file join $directory summary.tcldata] {WRONLY CREAT EXCL}]; puts $file $summary; close $file
        set checklist [open [file join $directory REQUIRED_NEXT_STEPS.txt] {WRONLY CREAT EXCL}]
        puts $checklist {SYNTHESIS EVIDENCE ONLY. This report does not qualify routing, CDC, an AFI, or hardware performance.
1. Bind the private generated IP/XDC/DCP hashes and actual BD configuration: nine SI, eight MI, three clocks, host access to all eight 512 MiB windows, each native master limited to its own PC0-7 window.
2. Simulate the CURRENT nine-SI/eight-MI vendor fabric: ordering, narrow byte strobes, burst splitting/4 KiB boundaries, independent IDs, port/address windows and reset/backpressure. Alex's historical two-port simulation is insufficient.
3. Link the real shell and all IP checkpoints. Resolve every opaque IP/clock recorded here; validate actual 250/200/450 MHz clocks, unused400 output, vendor CDC paths/exception coverage, resets, and absence of custom HBM-clock logic/TMEM multicycles.
4. Independently require nonnegative setup/hold, no pulse-width/unconstrained failures, zero unresolved/unrouted nets, reviewed DRC/methodology/CDC/waivers and all applicable bus-skew constraints. No historical fixed49/47/7 skew count or R5 eight-SmartConnect topology is accepted.
5. Bind qualification evidence to the exact final DCP and official package before publication. Require exclusive device ownership, preload same-BAR flush plus full byte readback, identity checks, exact numerical outputs and restoration before hardware results are accepted.}
        close $checklist
    } message options]
    close $audit
    if {$status} {return -options $options $message}
    puts "C2_ALEX_SYNTH_AUDIT_RECORDED route_qualified=0 directory=$directory"
    return $summary
}
