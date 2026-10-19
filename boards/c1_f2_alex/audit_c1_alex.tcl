# C1/Alex synthesis evidence only. Source after synth_design at top-level scope,
# then call ::c1_alex::audit_synthesis $run. This never writes a route-pass marker,
# changes timing exceptions/severities/waivers, or accepts a physical image.
namespace eval ::c1_alex {}

proc ::c1_alex::property {object key} {
    # Read only the required property. Hierarchical list_property enumeration
    # materialized 87 boundary objects at u_mem in the pinned Vivado probe.
    set value [get_property $key $object]
    if {$value eq ""} {error "C1 Alex audit: missing/empty $key on $object"}
    return $value
}
proc ::c1_alex::names {objects} {
    if {![llength $objects]} {return {}}
    set result [get_property NAME $objects]
    if {[llength $objects] == 1} {set result [list $result]}
    if {[llength $result] != [llength $objects]} {error "C1 Alex audit: name cardinality mismatch"}
    foreach name $result {if {$name eq ""} {error "C1 Alex audit: empty object name"}}
    return $result
}
proc ::c1_alex::cell {name expected_ref concrete} {
    set object [get_cells -quiet $name]
    if {[names $object] ne [list $name]} {error "C1 Alex audit: missing/ambiguous cell $name"}
    set ref [property $object REF_NAME]
    # ORIG_REF_NAME can be absent/empty after optimization. An API error still
    # propagates; otherwise the required concrete REF_NAME is the fallback.
    set original [get_property ORIG_REF_NAME $object]
    if {$original ne ""} {set ref $original}
    if {$ref ne $expected_ref} {error "C1 Alex audit: $name has unexpected reference $ref"}
    set opaque [property $object IS_BLACKBOX]
    if {![string is boolean -strict $opaque] || ($concrete && $opaque)} {
        error "C1 Alex audit: $name is unresolved or has unknown black-box state"
    }
    return $object
}
proc ::c1_alex::vendor_owner {name} {
    foreach root {u_mem u_hbm u_ocl_cdc u_pcis_pipe u_core_clk u_hbm_clk u_rst_main u_rst_core u_rst_hbm u_ready_cdc u_status_cdc} {
        if {$name eq $root || [string first "$root/" $name] == 0} {return $root}
    }
    return {}
}
proc ::c1_alex::opaque_roots {audit} {
    set result {}
    foreach object [get_cells -quiet -hierarchical -filter {IS_BLACKBOX}] {
        set name [property $object NAME]
        set owner [vendor_owner $name]
        if {$owner eq ""} {error "C1 Alex audit: unexpected unresolved cell $name"}
        dict lappend result $owner $name
        puts $audit [list opaque_vendor_cell $name reference [property $object REF_NAME]]
    }
    return $result
}
# Declared AXI widths/protocols/windows are checked against the hash-bound
# generated BD by the executor. Rebuilt hierarchy can remove constant bits or
# rename wrapper inputs. Here the invariants are actual connected leaf owners,
# unique active drivers, exact port identities, and real clock consumers.
proc ::c1_alex::inside {name root} {return [expr {[string first "$root/" $name] == 0}]}
proc ::c1_alex::relocated_native_loads {root p signal} {
    # Exact additional input pins observed in the hash-bound d1c diagnostic.
    # Rebuilt hierarchy moved native counters/error and kernel return logic
    # under u_mem. These are optional loads, never alternative drivers or a
    # blanket permission for another PC/SmartConnect port.
    set suffix [expr {$p == 0 ? "" : "__[expr {$p-1}]"}]
    if {$signal eq "arready"} {
        return [list [format {%su_mem/pending_count[4]_i_1%s/I0} $root $suffix] ${root}u_mem/n_err_i_4${suffix}/I4]
    }
    if {$signal eq "rvalid"} {
        set lut [lindex {13 19 13 17 13 17 13 18} $p]
        return [list [format {%su_mem/g_return[%d].memory_reg_0_i_%d/I0} $root $p $lut] \
            [format {%su_mem/pc_returned[%d][0]_i_2/I0} $root $p] \
            [format {%su_mem/reads[7]_i_1%s/I0} $root $suffix] ${root}u_mem/n_err_i_1${suffix}/I1]
    }
    if {$signal eq "bvalid"} {
        return [list [format {%su_mem/writes[7]_i_1%s/I0} $root $suffix] ${root}u_mem/n_err_i_2${suffix}/I1]
    }
    return {}
}
proc ::c1_alex::native_b_endpoints {root p driver internal audit} {
    set endpoints [names [all_fanout -flat -endpoints_only -from $driver]]
    set required [list ${root}u_native$p/n_err_reg/D]
    for {set bit 0} {$bit < 8} {incr bit} {lappend required [format {%su_native%d/writes_reg[%d]/CE} $root $p $bit]}
    foreach endpoint $required {
        if {$endpoint ni $endpoints} {error "C1 Alex audit: native B response lost state/error sink $endpoint"}
    }
    foreach endpoint $endpoints {
        if {![inside $endpoint ${root}u_native$p] && ![inside $endpoint $internal]} {error "C1 Alex audit: native B response reaches wrong port/channel $endpoint"}
    }
    puts $audit [list native_b_response $p driver [names $driver] endpoints $endpoints]
}
# Vivado collections carry object type information. Tcl list construction can
# discard it even when the displayed names are unchanged. Re-resolve names
# after selection/filtering; never pass a reconstructed name list as objects.
proc ::c1_alex::selected_objects {kind wanted} {
    if {![llength $wanted] || [llength [lsort -unique $wanted]] != [llength $wanted]} {error "C1 Alex audit: empty/duplicate $kind selection"}
    switch $kind {
        pins {set selected [get_pins -quiet $wanted]}
        nets {set selected [get_nets -quiet $wanted]}
        clocks {set selected [get_clocks -quiet $wanted]}
        default {error "C1 Alex audit: unsupported object selection $kind"}
    }
    if {[lsort [names $selected]] ne [lsort $wanted]} {error "C1 Alex audit: changed/ambiguous $kind selection"}
    return $selected
}
proc ::c1_alex::exact_pin {name {scalar 0}} {
    set result {}
    set wanted [list $name]
    if {$scalar} {lappend wanted [format {%s[0]} $name]}
    foreach candidate $wanted {
        set objects [get_pins -quiet [list $candidate]]
        if {[llength $objects] && [names $objects] ne [list $candidate]} {error "C1 Alex audit: ambiguous pin $candidate"}
        lappend result {*}[names $objects]
    }
    if {[llength $result] != 1} {error "C1 Alex audit: missing/duplicate endpoint $name"}
    return [selected_objects pins $result]
}
proc ::c1_alex::flow {prefix iface signal audit} {
    set root [expr {$prefix eq "" ? "" : "$prefix/"}]
    set sc ${root}u_mem/f2_mem_i/memory_sc/inst
    if {![regexp {^(aw|ar|w|b|r)(valid|ready)$} $signal -> channel]} {error "C1 Alex audit: unknown AXI handshake $signal"}
    set forward [expr {$signal in {awvalid wvalid arvalid bready rready}}]
    if {$iface eq "S_HOST"} {
        set external ${root}u_pcis_pipe
        set internal $sc/s00_entry_pipeline/s00_mmu/inst/${channel}_sreg
        set anchor [exact_pin $external/m_axi_$signal 1]
        set expected_direction [expr {$forward ? "OUT" : "IN"}]
        set mode pin
    } elseif {[regexp {^S_N([0-7])$} $iface -> p]} {
        set external ${root}u_native$p
        set si [format s%02d [expr {$p+1}]]
        set internal $sc/${si}_entry_pipeline/${si}_mmu/inst/${channel}_sreg
        set wanted ${root}n${p}_$signal
        if {$signal eq "bvalid"} {set wanted ${root}u_mem/n${p}_bvalid}
        set anchor [get_nets -quiet [list $wanted]]
        if {[names $anchor] ne [list $wanted]} {error "C1 Alex audit: missing/ambiguous native net $wanted"}
        set mode net
    } elseif {[regexp {^M_H([0-7])$} $iface -> p]} {
        set external ${root}u_hbm
        set mi [format m%02d $p]
        set internal $sc/${mi}_exit_pipeline/${mi}_exit/inst/${channel}_reg
        set io [expr {$forward ? "i" : "o"}]
        set anchor [exact_pin [format {%s/%s_axi_%s[%d]} $external $io $signal $p]]
        set expected_direction [expr {$forward ? "IN" : "OUT"}]
        set forward [expr {!$forward}]
        set mode pin
    } else {error "C1 Alex audit: unknown fabric interface $iface"}
    if {$mode eq "pin" && [get_property DIRECTION $anchor] ne $expected_direction} {error "C1 Alex audit: wrong external direction $iface/$signal"}
    # Vivado -of_objects accepts pins here, not an already selected net.
    # The v2 diagnostic's leaf query on the native net itself includes the
    # rebuilt hierarchy; dropping that anchor would falsely report no sinks.
    if {$mode eq "net"} {set nets $anchor} else {set nets [get_nets -quiet -segments -of_objects $anchor]}
    if {![llength $nets]} {error "C1 Alex audit: disconnected fabric endpoint $iface/$signal"}
    if {[llength [get_ports -quiet -of_objects $nets]]} {error "C1 Alex audit: fabric endpoint escapes top $iface/$signal"}
    set leaf [get_pins -quiet -leaf -of_objects $nets]
    set leaf_names [names $leaf]; set directions [get_property DIRECTION $leaf]
    if {![llength $leaf] || [llength $leaf_names] != [llength $directions] ||
        [llength [lsort -unique $leaf_names]] != [llength $leaf_names]} {error "C1 Alex audit: malformed leaf evidence $iface/$signal"}
    set driver {}; set receiver_count 0; set relocated {}
    if {[regexp {^S_N([0-7])$} $iface -> p]} {set relocated [relocated_native_loads $root $p $signal]}
    set source [expr {$forward ? $external : $internal}]
    set destination [expr {$forward ? $internal : $external}]
    foreach pin $leaf name $leaf_names direction $directions {
        if {![inside $name $external] && ![inside $name $internal]} {
            if {$direction ne "IN" || $name ni $relocated} {error "C1 Alex audit: wrong port/owner on $iface/$signal: $name"}
            set cell [get_cells -quiet -of_objects [exact_pin $name]]
            if {[llength $cell] != 1 || ![regexp {^LUT[1-6]$} [get_property REF_NAME $cell]]} {error "C1 Alex audit: unreviewed relocated native load $name"}
            if {$signal eq "bvalid"} {incr receiver_count}
        }
        if {$direction eq "OUT"} {
            if {![inside $name $source]} {error "C1 Alex audit: wrong driver owner on $iface/$signal"}
            lappend driver $name
        } elseif {$direction eq "IN"} {
            if {[inside $name $destination]} {incr receiver_count}
        } else {error "C1 Alex audit: unknown leaf direction on $iface/$signal"}
    }
    if {[llength $driver] != 1 || !$receiver_count} {error "C1 Alex audit: missing/multiple drivers or receiver on $iface/$signal"}
    set driver [selected_objects pins $driver]
    set driver_cell [get_cells -quiet -of_objects $driver]
    if {[llength $driver_cell] != 1 || [get_property REF_NAME $driver_cell] in {GND VCC}} {error "C1 Alex audit: constant/unresolved active driver on $iface/$signal"}
    if {[regexp {^S_N([0-7])$} $iface -> p] && $signal eq "bvalid"} {native_b_endpoints $root $p $driver $internal $audit}
    puts $audit [list fabric_flow $iface signal $signal anchor [names $anchor] nets [names $nets] \
        driver [names $driver] source_owner $source destination_owner $destination receiver_count $receiver_count \
        leaf_pins $leaf_names leaf_directions $directions]
}
proc ::c1_alex::fabric_clocks {prefix clocks opaque audit} {
    set root [expr {$prefix eq "" ? "" : "$prefix/"}]
    set counts {}; set port_counts {}
    foreach clock [get_clocks -quiet *] {
        set label other
        foreach candidate {host core hbm} {
            if {[dict exists $clocks $candidate] && $clock in [dict get $clocks $candidate]} {set label $candidate}
        }
        set members [names [all_registers -clock $clock -cells]]
        set owners {}
        foreach member $members {
            set local $member
            if {$root ne ""} {
                if {[string first $root $member] != 0} {
                    if {$label eq "hbm"} {error "C1 Alex audit: HBM-clock register outside CL: $member"}
                    continue
                }
                set local [string range $member [string length $root] end]
            }
            set owner [lindex [split $local /] 0]
            dict incr owners $owner
            if {$label eq "hbm" && [vendor_owner $local] ni {u_mem u_hbm u_rst_hbm}} {error "C1 Alex audit: custom HBM-clock register $member"}
            if {($owner eq "u_board" || [regexp {^u_native[0-7]$} $owner]) && $label ne "core"} {error "C1 Alex audit: custom datapath register on wrong clock $member"}
            if {$owner eq "u_pcis_pipe" && $label ne "host"} {error "C1 Alex audit: PCIS register on wrong clock $member"}
            if {$owner eq "u_mem" && $label ni {host core hbm}} {error "C1 Alex audit: fabric register on unreviewed clock $member"}
            if {[regexp {^u_mem/f2_mem_i/memory_sc/inst/(s([0-9][0-9])_entry_pipeline|m([0-9][0-9])_exit_pipeline)/} $local -> node si mi]} {
                if {$si ne ""} {
                    scan $si %d index
                    if {$index > 8} {error "C1 Alex audit: unexpected SI clock consumer $member"}
                    set wanted [expr {$index == 0 ? "host" : "core"}]
                } else {
                    scan $mi %d index
                    if {$index > 7} {error "C1 Alex audit: unexpected MI clock consumer $member"}
                    set wanted hbm
                }
                if {$label ne $wanted} {error "C1 Alex audit: $node register on wrong clock $member"}
                dict incr port_counts $node
            }
        }
        puts $audit [list fabric_clock [names $clock] label $label registers [llength $members] owner_counts $owners]
        if {$label ne "other"} {dict set counts $label $owners}
    }
    foreach label {host core hbm} {
        if {![dict exists $clocks $label] || ![llength [dict get $clocks $label]]} {
            if {$label eq "host" || ![dict exists $opaque u_${label}_clk]} {error "C1 Alex audit: unobserved fabric clock $label"}
            puts $audit [list fabric_clock_domain $label state deferred_opaque_ip]
            continue
        }
        set required {u_mem}
        if {$label eq "host"} {lappend required u_pcis_pipe}
        if {$label eq "hbm"} {lappend required u_hbm}
        if {$label eq "core"} {
            lappend required u_board
            for {set p 0} {$p < 8} {incr p} {lappend required u_native$p}
        }
        foreach owner $required {
            if {[dict exists $opaque $owner]} {continue}
            if {![dict exists $counts $label $owner] || [dict get $counts $label $owner] == 0} {error "C1 Alex audit: no actual $label consumers in $owner"}
        }
    }
    if {![dict exists $opaque u_mem]} {
        for {set p 0} {$p < 9} {incr p} {
            set label [expr {$p == 0 ? "host" : "core"}]
            set node [format s%02d_entry_pipeline $p]
            if {[llength [dict get $clocks $label]] && ![dict exists $port_counts $node]} {error "C1 Alex audit: no actual $label consumers in $node"}
        }
        for {set p 0} {$p < 8} {incr p} {
            set node [format m%02d_exit_pipeline $p]
            if {[llength [dict get $clocks hbm]] && ![dict exists $port_counts $node]} {error "C1 Alex audit: no actual HBM consumers in $node"}
        }
    }
    puts $audit [list fabric_port_clock_consumers $port_counts]
    return $counts
}
proc ::c1_alex::native_shared_ready {prefix clocks audit {opaque {}}} {
    set root [expr {$prefix eq "" ? "" : "$prefix/"}]
    set anchors {}
    for {set p 0} {$p < 8} {incr p} {
        foreach signal {bready rready} {
            set name ${root}n${p}_$signal
            set objects [get_nets -quiet [list $name]]
            if {[llength $objects] && [names $objects] ne [list $name]} {error "C1 Alex audit: ambiguous native ready alias $name"}
            lappend anchors {*}[names $objects]
        }
    }
    if {![llength $anchors]} {error "C1 Alex audit: no observed native ready alias"}
    set anchors [selected_objects nets $anchors]
    set nets $anchors
    if {[llength [get_ports -quiet -of_objects $nets]]} {error "C1 Alex audit: native ready escapes top"}
    set leaf [get_pins -quiet -leaf -of_objects $nets]
    set drivers {};set channels {}
    foreach pin $leaf name [names $leaf] direction [get_property DIRECTION $leaf] {
        if {$direction eq "OUT"} {lappend drivers $name;continue}
        if {$direction ne "IN"} {error "C1 Alex audit: unknown native ready leaf direction"}
        if {$name eq [format {%su_status_cdc/src_ff_reg[0]/D} $root]} {continue}
        set found 0
        for {set p 1} {$p <= 8} {incr p} {
            set si [format s%02d $p]
            foreach channel {b r} {
                set owner ${root}u_mem/f2_mem_i/memory_sc/inst/${si}_entry_pipeline/${si}_mmu/inst/${channel}_sreg
                if {[inside $name $owner]} {dict incr channels $si/$channel;set found 1}
            }
        }
        if {!$found} {error "C1 Alex audit: native ready reaches unreviewed owner/channel $name"}
    }
    if {[llength $drivers] != 1 || [dict size $channels] != 16} {error "C1 Alex audit: incomplete/shared native ready driver or sixteen B/R channel sinks"}
    set driver [selected_objects pins $drivers];set cell [get_cells -quiet -of_objects $driver]
    if {[llength $cell] != 1 || ![inside [lindex [names $driver] 0] ${root}u_mem] ||
        [get_property REF_NAME $cell] ne "LUT1" || ![regexp -nocase {^2'h1$} [get_property INIT $cell]]} {
        error "C1 Alex audit: native ready is not the observed reset inverter"
    }
    set input [exact_pin [lindex [names $cell] 0]/I0]
    set input_nets [get_nets -quiet -segments -of_objects $input]
    set input_leaf [get_pins -quiet -leaf -of_objects $input_nets]
    set reset_drivers {}
    foreach name [names $input_leaf] direction [get_property DIRECTION $input_leaf] {if {$direction eq "OUT"} {lappend reset_drivers $name}}
    if {$reset_drivers ne [list ${root}core_rst_reg/Q]} {error "C1 Alex audit: native ready inversion is not driven by core_rst_reg"}
    set reset_drivers [selected_objects pins $reset_drivers]
    set reset [get_cells -quiet [list ${root}core_rst_reg]]
    if {[llength $reset] != 1 || [get_property REF_NAME $reset] ne "FDRE"} {error "C1 Alex audit: native reset register changed"}
    set reset_clock [get_clocks -quiet -of_objects [exact_pin ${root}core_rst_reg/C]]
    if {![llength $reset_clock] && ![llength [dict get $clocks core]] && [dict exists $opaque u_core_clk]} {
        puts $audit [list native_reset_clock state deferred_opaque_ip]
    } elseif {[names $reset_clock] ne [names [dict get $clocks core]] || ![llength $reset_clock]} {
        error "C1 Alex audit: native reset register has wrong clock"
    }
    puts $audit [list native_shared_ready aliases [names $anchors] driver [names $driver] \
        primitive [get_property REF_NAME $cell] init [get_property INIT $cell] \
        reset_driver [names $reset_drivers] reset_clock [names $reset_clock] channel_sinks $channels \
        logical_connections 16 source_semantics {!core_rst}]
    return 16
}
proc ::c1_alex::fabric_topology {prefix clocks opaque audit} {
    set counts [fabric_clocks $prefix $clocks $opaque $audit]
    if {[dict exists $opaque u_mem] || [dict exists $opaque u_hbm] || [dict exists $opaque u_pcis_pipe]} {
        puts $audit [list fabric_connectivity state deferred_opaque_ip roots [dict keys $opaque]]
        return [dict create slave_interfaces 9 master_interfaces 8 active_handshake_witnesses 0 state deferred_opaque_ip]
    }
    set n 0
    foreach signal {awvalid awready wvalid wready bvalid bready arvalid arready rvalid rready} {
        flow $prefix S_HOST $signal $audit; incr n
    }
    for {set p 0} {$p < 8} {incr p} {
        foreach signal {awvalid awready wvalid wready bvalid arvalid arready rvalid} {flow $prefix S_N$p $signal $audit; incr n}
        foreach signal {awvalid awready wvalid wready bvalid bready arvalid arready rvalid rready} {flow $prefix M_H$p $signal $audit; incr n}
    }
    set shared [native_shared_ready $prefix $clocks $audit $opaque]
    set state observed
    foreach label {core hbm} {if {![llength [dict get $clocks $label]]} {set state observed_connections_deferred_clock}}
    return [dict create slave_interfaces 9 master_interfaces 8 active_handshake_witnesses $n \
        shared_reset_ready_connections $shared state $state declared_width_authority generated_BD_required]
}
proc ::c1_alex::check_constraints {path} {
    # Treat exported XDC as text/list data, never execute it. Custom multicycles
    # and the old TMEM exceptions have no place in this standalone C1 kernel.
    set file [open $path r]; set text [read $file]; close $file
    set pending {}; set scope {}; set vendor_multicycles 0
    foreach line [split $text \n] {
        if {$pending eq "" && ([string trim $line] eq "" || [string index [string trim $line] 0] eq "#")} {continue}
        append pending $line \n
        if {[regexp {(^|[^\\])(\\\\)*\\$} $line] || ![info complete $pending]} {continue}
        set command [string trim $pending]; set pending {}
        if {[regexp -nocase {u_tmem|otpu_clk2x|OTPU_CORE_TMEM} $command]} {error "C1 Alex audit: inapplicable TMEM timing constraint"}
        if {[regexp {^current_instance(?:\s|$)} $command]} {
            if {[catch {llength $command}]} {error "C1 Alex audit: malformed constraint scope"}
            set args [lrange $command 1 end]
            if {[lindex $args 0] eq "-quiet"} {set args [lrange $args 1 end]}
            if {[llength $args] > 1} {error "C1 Alex audit: ambiguous constraint scope"}
            set scope [lindex $args 0]
        }
        if {[regexp {(^|[;\n])\s*set_multicycle_path(?:\s|$)} $command]} {
            if {[vendor_owner $scope] eq ""} {error "C1 Alex audit: unreviewed custom multicycle constraint"}
            incr vendor_multicycles
        }
    }
    if {$pending ne ""} {error "C1 Alex audit: incomplete constraint export"}
    return $vendor_multicycles
}
proc ::c1_alex::clock {label pin period owner opaque audit} {
    set selected [get_pins -quiet $pin]
    if {$label eq "tmem_unused" && ![llength $selected]} {
        # OOC synthesis may remove this unconnected output pin. A retained
        # 400 MHz clock still has to pass the zero-consumer check below.
        set clocks {}
        foreach candidate [get_clocks -quiet *] {
            set actual [property $candidate PERIOD]
            if {abs($actual-$period)>0.001} {continue}
            set sources [property $candidate SOURCE_PINS]
            if {![llength $sources]} {error "C1 Alex audit: unattributed unused-output clock"}
            foreach source $sources {
                if {[string first "$owner/" $source] != 0} {error "C1 Alex audit: unexpected 400 MHz clock source $source"}
            }
            lappend clocks [property $candidate NAME]
        }
        if {![llength $clocks]} {
            set state [expr {[dict exists $opaque $owner] ? "deferred_opaque_ip" : "pruned_unused_output"}]
            puts $audit [list clock $label expected_period $period state $state]
            return [dict create objects {} state $state]
        }
        set clocks [selected_objects clocks $clocks]
    } else {
        if {[names $selected] ne [list $pin]} {error "C1 Alex audit: missing clock pin $pin"}
        set clocks [get_clocks -quiet -of_objects $selected]
    }
    if {![llength $clocks] && [dict exists $opaque $owner]} {
        puts $audit [list clock $label expected_period $period state deferred_opaque_ip]
        return [dict create objects {} state deferred_opaque_ip]
    }
    if {[llength $clocks] != 1} {error "C1 Alex audit: missing/ambiguous $label clock"}
    set actual [property $clocks PERIOD]
    if {![string is double -strict $actual] || abs($actual-$period)>0.001} {
        error "C1 Alex audit: $label period $actual differs from $period ns"
    }
    set data [dict create NAME [property $clocks NAME] PERIOD $actual]
    foreach key {MASTER_CLOCK SOURCE_PINS IS_GENERATED IS_PROPAGATED IS_VIRTUAL} {
        if {$key in [list_property $clocks]} {dict set data $key [get_property $key $clocks]}
    }
    puts $audit [list clock $label state observed properties $data]
    return [dict create objects $clocks state observed]
}
proc ::c1_alex::diagnostic_checkpoint {path audit capture_order} {
    # Capture the diagnostic after live inspection. The legacy filename is
    # retained for collection compatibility; the record below states the actual
    # capture order. This ordering does not imply serialization changes topology.
    write_checkpoint $path
    if {![file isfile $path] || [file size $path] == 0} {error "C1 Alex audit: missing/empty diagnostic checkpoint"}
    puts $audit [list diagnostic_checkpoint $path qualification NOT_QUALIFIED \
        capture_order $capture_order inspected_state live_design_before_diagnostic_serialization]
    flush $audit
}
proc ::c1_alex::audit_synthesis {run} {
    if {[version -short] ne "2025.2"} {error "C1 Alex audit: unreviewed Vivado version"}
    if {[current_instance] ne ""} {error "C1 Alex audit: requires top-level synthesis scope"}
    set directory [file join $run synth_audit]
    if {[file exists $directory]} {error "C1 Alex audit: refusing existing evidence directory"}
    set diagnostic [file join $run pre_audit.NOT_QUALIFIED.dcp]
    if {[file exists $diagnostic]} {error "C1 Alex audit: refusing existing diagnostic checkpoint"}
    file mkdir $directory
    set audit [open [file join $directory objects.tcldata] {WRONLY CREAT EXCL}]
    set diagnostic_attempted 0
    set status [catch {
        puts $audit [list evidence kind synthesis_only route_qualified 0 hardware_measured 0 tool [version -short]]
        puts $audit [list inspection_state live_design_before_diagnostic_serialization diagnostic_capture pending]
        flush $audit
        cell u_board otpu_c1_hbm_kernel 1
        cell u_board/core otpu_c1_top 1
        set natives [names [get_cells -quiet -regexp {^u_native[0-9]+$}]]
        set expected {}; for {set p 0} {$p < 8} {incr p} {lappend expected u_native$p; cell u_native$p otpu_native_axi 1}
        if {[lsort $natives] ne $expected} {error "C1 Alex audit: expected exactly eight native masters"}
        foreach {name ref} {u_mem f2_mem_wrapper u_hbm cl_mem_hbm_wrapper u_ocl_cdc f2_ocl_cdc u_pcis_pipe f2_pcis_pipe u_core_clk f2_core_clk u_hbm_clk f2_hbm_clk} {
            cell $name $ref 0
        }
        if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ otpu_tmem* || ORIG_REF_NAME =~ otpu_tmem*}]]} {
            error "C1 Alex audit: unexpected TMEM implementation"
        }
        set interface_pins [names [get_pins -quiet -of_objects [get_cells u_mem]]]
        if {[llength $interface_pins] != [llength [lsort -unique $interface_pins]]} {error "C1 Alex audit: duplicate boundary object inventory"}
        puts $audit [list memory_interface_pins $interface_pins]
        flush $audit
        set opaque [opaque_roots $audit]
        foreach {name period} {clk_main_a0 4.000 clk_hbm_ref 10.000} {
            set clock [get_clocks -quiet $name]
            if {[names $clock] ne [list $name] || abs([property $clock PERIOD]-$period)>0.001} {error "C1 Alex audit: wrong input clock $name"}
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
        dict set clocks host [get_clocks -quiet clk_main_a0]
        set topology [fabric_topology {} $clocks $opaque $audit]
        puts $audit [list interface_topology $topology address_map_evidence generated_BD_required]
        set hclk_visible 0
        if {[llength [dict get $clocks hbm]]} {
            foreach name [names [all_registers -clock [dict get $clocks hbm] -cells]] {
                if {[vendor_owner $name] ni {u_mem u_hbm u_rst_hbm}} {error "C1 Alex audit: custom HBM-clock register $name"}
                incr hclk_visible
            }
        }
        if {[llength [dict get $clocks tmem_unused]] && [llength [all_registers -clock [dict get $clocks tmem_unused] -cells]]} {
            error "C1 Alex audit: unexpected consumer on unused 400 MHz clock"
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
            if {![file isfile $path] || [file size $path] == 0} {error "C1 Alex audit: missing/empty evidence $name"}
        }
        # All live queries and reports precede diagnostic serialization. The
        # builder separately writes post_synth.dcp only after this audit returns.
        set diagnostic_attempted 1
        diagnostic_checkpoint $diagnostic $audit after_live_audit_checks
        set summary [dict create schema 1 stage synthesis_only route_qualified 0 hardware_measured 0 \
            inspected_state live_design_before_diagnostic_serialization diagnostic_capture_order after_live_audit_checks \
            kernel otpu_c1_hbm_kernel native_masters 8 slave_interfaces 9 master_interfaces 8 \
            fabric_connectivity $topology \
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
    if {$status} {
        # Preserve the original failure even if best-effort diagnostic capture
        # also fails. Never retry a failed checkpoint write or inspect its
        # potentially changed in-memory graph afterward.
        catch {puts $audit [list audit_failure $message];flush $audit}
        if {!$diagnostic_attempted} {
            set diagnostic_attempted 1
            if {[catch {diagnostic_checkpoint $diagnostic $audit after_audit_failure} capture_error]} {
                catch {puts $audit [list diagnostic_capture_error $capture_error];flush $audit}
            }
        }
        catch {close $audit}
        return -options $options $message
    }
    close $audit
    puts "C1_ALEX_SYNTH_AUDIT_RECORDED route_qualified=0 directory=$directory"
    return $summary
}
