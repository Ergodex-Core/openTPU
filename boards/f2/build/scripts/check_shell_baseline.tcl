# Exact inherited-shell classification for route qualification. This does
# not change severity, waive findings, or change any timing/design constraint.
# The caller must retain all severity reports and classify every returned
# `remaining` finding with the normal fail-closed severity policy.
source [file join [file dirname [info script]] shell_baseline_2025_2.tcl]

proc otpu_shell_property {object property} {
    if {$property ni [list_property $object]} {
        error "shell baseline: missing $property on $object"
    }
    return [get_property $property $object]
}

proc otpu_shell_names {objects} {
    set names {}
    foreach object $objects {lappend names [otpu_shell_property $object NAME]}
    return $names
}

proc otpu_shell_exact {kind name} {
    if {$name eq ""} {error "shell baseline: empty $kind name"}
    # Methodology finding names have no hierarchy. Pins and cells must instead
    # be obtained by their verified relationships below: a full hierarchical
    # regex does not resolve them from the top instance in Vivado 2025.2.
    regsub -all {([][\\.^$*+?(){}|])} $name {\\\1} literal
    set pattern "^${literal}$"
    switch -- $kind {
        methodology_violations {set objects [get_methodology_violations -quiet -regexp $pattern]}
        default {error "shell baseline: unsupported object kind $kind"}
    }
    if {[llength $objects] != 1} {
        error "shell baseline: $kind $name resolved to [llength $objects] objects"
    }
    set object [lindex $objects 0]
    if {[otpu_shell_property $object NAME] ne $name} {
        error "shell baseline: $kind identity mismatch for $name"
    }
    return $object
}

proc otpu_shell_properties_match {object expected} {
    dict for {property value} $expected {
        set actual [otpu_shell_property $object $property]
        if {$actual ne $value} {
            error "shell baseline: $property mismatch on [otpu_shell_property $object NAME]: expected '$value', got '$actual'"
        }
    }
}

proc otpu_classify_shell_methodology {shell_dcp violation_names audit} {
    # Precondition: the caller ran a fresh, complete report_methodology for the
    # currently open design and kept its original reports. Names are strings,
    # not a subset of findings or handles cached from a previous design.
    # All live objects and canonical names stay procedure-local. In particular,
    # do not retain Vivado object handles across checkpoint opens or in globals.
    set baseline [otpu_shell_baseline_data]
    if {![file isfile $shell_dcp]} {error "shell baseline: missing shell checkpoint"}
    set shell_sha [lindex [exec sha256sum -- [file normalize $shell_dcp]] 0]
    if {$shell_sha ne [dict get $baseline shell_sha256]} {
        error "shell baseline: pinned shell SHA256 mismatch"
    }
    if {[version -short] ne [dict get $baseline vivado_version]} {
        error "shell baseline: Vivado version differs from reviewed baseline"
    }
    set expected_pins [dict get $baseline pins]
    if {[dict size $expected_pins] != 35 || [dict size [dict get $baseline roots]] != 2} {
        error "shell baseline: invalid baseline inventory"
    }
    # Require the complete current finding set, not a caller-selected subset.
    set live_names [otpu_shell_names [get_methodology_violations]]
    if {[llength [lsort -unique $live_names]] != [llength $live_names] ||
        [llength [lsort -unique $violation_names]] != [llength $violation_names] ||
        [lsort $violation_names] ne [lsort $live_names]} {
        error "shell baseline: supplied findings do not exactly match the live inventory"
    }
    set inherited {}
    set remaining {}
    set seen_pins {}
    foreach name [lsort $violation_names] {
        set finding [otpu_shell_exact methodology_violations $name]
        set check [otpu_shell_property $finding CHECK]
        if {$check ne "TIMING-17"} {
            if {[string match {TIMING-17#*} $name]} {
                error "shell baseline: TIMING-17 finding has inconsistent CHECK"
            }
            lappend remaining $name
            continue
        }
        if {![string match {TIMING-17#*} $name]} {
            error "shell baseline: unexpected name for TIMING-17 finding"
        }
        otpu_shell_properties_match $finding {SEVERITY {Critical Warning} WAIVER {}}
        set waived [otpu_shell_property $finding IS_WAIVED]
        if {![string is boolean -strict $waived] || $waived} {
            error "shell baseline: finding is waived or waiver state is unknown: $name"
        }
        set pins [get_pins -of_objects $finding]
        set pin_names [otpu_shell_names $pins]
        if {[llength $pin_names] != 1} {
            error "shell baseline: $name must relate to exactly one pin"
        }
        set pin_name [lindex $pin_names 0]
        if {![dict exists $expected_pins $pin_name] || $pin_name in $seen_pins} {
            error "shell baseline: unexpected or duplicate TIMING-17 pin $pin_name"
        }
        set expected [dict get $expected_pins $pin_name]
        # Use the object whose relationship and exact canonical NAME were just
        # checked. No report/checkpoint operation occurs while these are live.
        set pin [lindex $pins 0]
        otpu_shell_properties_match $pin [dict get $expected pin_properties]
        set cell_name [dict get $expected cell]
        foreach owner [list $finding $pin] {
            set related_cells [get_cells -of_objects $owner]
            if {[otpu_shell_names $related_cells] ne [list $cell_name]} {
                error "shell baseline: related cell mismatch for $pin_name"
            }
            set cell [lindex $related_cells 0]
            otpu_shell_properties_match $cell [dict get $expected cell_properties]
        }
        if {[llength [get_clocks -of_objects $pin]] != 0} {
            error "shell baseline: previously unclocked pin now has a defined clock: $pin_name"
        }
        set root_objects [all_fanin -flat -startpoints_only -trace_arcs all $pin]
        set roots [otpu_shell_names $root_objects]
        set root_name [dict get $expected root]
        if {$roots ne [list $root_name]} {
            error "shell baseline: clock root mismatch for $pin_name"
        }
        set root [lindex $root_objects 0]
        otpu_shell_properties_match $root [dict get $baseline roots $root_name]
        lappend seen_pins $pin_name
        lappend inherited $name
    }
    if {[llength $inherited] != 35 || [lsort $seen_pins] ne [lsort [dict keys $expected_pins]]} {
        error "shell baseline: TIMING-17 pin set is not the exact 35-pin baseline"
    }
    # Emit acceptance only after the complete inventory has passed every check.
    # The original Critical Warning severity remains visible and unchanged.
    foreach name $inherited {
        puts $audit [list methodology $name severity {Critical Warning} classification inherited_shell_baseline]
    }
    puts $audit [list shell_baseline shell_sha256 $shell_sha inherited_count 35 remaining_count [llength $remaining]]
    return [dict create inherited $inherited remaining $remaining shell_sha256 $shell_sha]
}
