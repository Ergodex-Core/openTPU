# Read-only routed host WREADY replication witness. Source as a library.
# This proves only the observed FDRE replication/connectivity transformation;
# the caller must bind the checkpoint, implementation lineage and other checks.
namespace eval ::c1_alex_routed_ready {}
proc ::c1_alex_routed_ready::require {condition message} {
    if {!$condition} {error "C1 routed host ready: $message"}
}
proc ::c1_alex_routed_ready::names {objects} {
    set result {}
    foreach object $objects {lappend result [get_property NAME $object]}
    require [expr {[llength $result] == [llength [lsort -unique $result]]}] "duplicate object names"
    return $result
}
proc ::c1_alex_routed_ready::one {kind name} {
    set command [dict get {cell get_cells pin get_pins} $kind]
    set objects [$command -quiet [list $name]]
    require [expr {[names $objects] eq [list $name]}] "missing/ambiguous $kind $name"
    return $objects
}
proc ::c1_alex_routed_ready::properties {object keys} {
    set result {};set available [list_property $object]
    foreach key $keys {
        require [expr {$key in $available}] "missing property $key"
        dict set result $key [get_property $key $object]
    }
    return $result
}
proc ::c1_alex_routed_ready::pin {name ref direction owner} {
    set object [one pin $name]
    set p [properties $object {NAME REF_PIN_NAME DIRECTION IS_CLOCK IS_INVERTED}]
    require [expr {[dict get $p NAME] eq $name && [dict get $p REF_PIN_NAME] eq $ref && [dict get $p DIRECTION] eq $direction}] "pin identity/direction $name"
    require [expr {[dict get $p IS_INVERTED] in {0 false FALSE}}] "pin inversion $name"
    require [expr {[dict get $p IS_CLOCK] eq [expr {$ref eq "C" ? "1" : "0"}]}] "pin clock classification $name"
    require [expr {[names [get_cells -quiet -of_objects $object]] eq [list $owner]}] "pin owner $name"
    return $p
}
proc ::c1_alex_routed_ready::network {name {clock 0}} {
    set object [one pin $name]
    set nets [get_nets -quiet -segments -of_objects $object]
    require [expr {[llength $nets] > 0 && [llength $nets] <= 20000}] "missing/excess net segments $name"
    set drivers [names [get_pins -quiet -leaf -of_objects $nets -filter {DIRECTION == OUT}]]
    require [expr {[llength $drivers] == 1}] "missing/multiple driver $name"
    require [expr {![llength [get_ports -quiet -of_objects $nets]]}] "top port escape $name"
    set clocks [names [get_clocks -quiet -of_objects $object]]
    # Clock fanout can be global: exact clock and driver identity suffice here.
    # Avoid an irrelevant large leaf/net-name expansion for the clock branch.
    set result [dict create drivers $drivers clocks $clocks segment_count [llength $nets]]
    if {$clock} {return $result}
    require [expr {[llength $nets] <= 256}] "control/ready net segment cap $name"
    dict set result nets [lsort [names $nets]]
    return $result
}
proc ::c1_alex_routed_ready::output {name allowed_owner} {
    set result [network $name]
    require [expr {[dict get $result drivers] eq [list $name]}] "wrong ready output driver $name"
    set nets [get_nets -quiet -segments -of_objects [one pin $name]]
    set leaf [get_pins -quiet -leaf -of_objects $nets]
    require [expr {[llength $leaf] > 1 && [llength $leaf] <= 256}] "missing/excess ready leaf inventory $name"
    set all [names $leaf];set sinks {};set owners {}
    foreach object $leaf {
        set pname [get_property NAME $object]
        if {$pname eq $name} {continue}
        require [expr {[get_property DIRECTION $object] eq "IN"}] "extra ready output $pname"
        set cells [get_cells -quiet -of_objects $object]
        set cn [names $cells]
        require [expr {[llength $cn] == 1 && [lindex $cn 0] eq [file dirname $pname]}] "leaf owner identity $pname"
        set cname [lindex $cn 0]
        require [expr {[string first "$allowed_owner/" $cname] == 0}] "unexpected ready sink owner $cname"
        set reference [get_property REF_NAME $cells]
        require [expr {$reference ne "" && $reference ni {VCC GND}}] "invalid ready sink primitive $cname"
        lappend sinks $pname;dict set owners $pname [dict create cell $cname reference $reference]
    }
    require [expr {$name in $all && [llength $sinks] > 0}] "ready has no actual leaf receiver"
    dict set result leaf_pins [lsort $all];dict set result sinks [lsort $sinks];dict set result sink_owners $owners
    return $result
}
proc ::c1_alex_routed_ready::host_wready {audit} {
    set scope WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/s00_entry_pipeline/s00_mmu/inst/w_sreg
    set original $scope/state_reg\[s_ready_i\]
    set replica ${original}_replica
    set facts {};set controls {}
    foreach name [list $original $replica] {
        set cell [one cell $name]
        set p [properties $cell {NAME REF_NAME ORIG_REF_NAME INIT IS_SEQUENTIAL ORIG_CELL_NAME PHYS_OPT_MODIFIED}]
        require [expr {[dict get $p NAME] eq $name && [dict get $p REF_NAME] eq "FDRE" && [dict get $p ORIG_REF_NAME] eq "FDRE" && [dict get $p INIT] eq "1'b0" && [dict get $p IS_SEQUENTIAL] eq "1"}] "ready primitive/initial state $name"
        require [expr {[dict get $p PHYS_OPT_MODIFIED] eq "SLR_CROSSING_OPT"}] "unreviewed physical transformation $name"
        if {$name eq $replica} {
            require [expr {[dict get $p ORIG_CELL_NAME] eq [file tail $original]}] "replica origin differs"
        } else {
            require [expr {[dict get $p ORIG_CELL_NAME] eq ""}] "original is itself unreviewed replica"
        }
        set pins [names [get_pins -quiet -of_objects $cell]];set wanted {}
        foreach ref {C CE D Q R} {lappend wanted $name/$ref}
        require [expr {[lsort $pins] eq [lsort $wanted]}] "ready primitive pin inventory $name"
        set pinfacts {};set inputfacts {}
        foreach ref {C CE D Q R} {
            set direction IN;if {$ref eq "Q"} {set direction OUT}
            dict set pinfacts $ref [pin $name/$ref $ref $direction $name]
            if {$ref ne "Q"} {dict set inputfacts $ref [network $name/$ref [expr {$ref eq "C"}]]}
        }
        require [expr {[dict get $inputfacts C clocks] eq {clk_main_a0}}] "ready clock differs"
        foreach ref {CE D R} {require [expr {[dict get $inputfacts $ref clocks] eq {}}] "unexpected control clock $name/$ref"}
        dict set facts $name [dict create cell $p pins $pinfacts controls $inputfacts]
        dict set controls $name $inputfacts
    }
    foreach ref {C CE D R} {
        foreach property {drivers clocks} {
            require [expr {[dict get $controls $original $ref $property] eq [dict get $controls $replica $ref $property]}] "replica $ref $property differs"
        }
    }
    # These exact observed drivers bind the common controls to the audited
    # SmartConnect ready register, not merely two arbitrary matching registers.
    set expected [dict create C WRAPPER/IO_SHIM/RL_CLKS/BUFG_CLK_A_EXT/O CE $scope/VCC/P D $scope/state\[s_ready_i\]_i_1__3/O R [file dirname $scope]/areset_reg/Q]
    dict for {ref driver} $expected {require [expr {[dict get $controls $original $ref drivers] eq [list $driver]}] "unreviewed common $ref driver"}
    set original_output [output $original/Q $scope]
    set replica_output [output $replica/Q WRAPPER/CL/u_pcis_pipe/inst/w.w_pipe]
    set anchor WRAPPER/CL/u_pcis_pipe/m_axi_wready
    set ap [one pin $anchor]
    require [expr {[get_property DIRECTION $ap] eq "IN"}] "host anchor direction"
    set anchor_network [network $anchor]
    require [expr {[dict get $anchor_network drivers] eq [list $original/Q] && [dict get $anchor_network nets] eq [dict get $original_output nets]}] "host anchor no longer binds original ready network"
    set result [dict create kind routed_host_wready_replication structural_pass 1 route_qualified 0 original $original replica $replica registers $facts original_output $original_output replica_output $replica_output anchor $anchor anchor_network $anchor_network]
    puts $audit [list routed_host_wready $result]
    return $result
}
