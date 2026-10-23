# Keep every raw severity visible. Only exact, independently checked inherited
# shell and scoped-clock findings may be excluded from the blocking set.
source [file join [file dirname [info script]] check_shell_baseline.tcl]
source [file join [file dirname [info script]] check_scoped_clocks.tcl]

proc otpu_qualify_methodology {shell_dcp evidence_prefix audit} {
    # The caller must first run a fresh complete report_methodology. Keep all
    # object queries in procedure scope and persist canonical NAME strings.
    set names [otpu_shell_names [get_methodology_violations]]
    set raw [otpu_shell_names [otpu_blocking_violations methodology \
        [get_methodology_violations] $audit]]
    set shell [otpu_classify_shell_methodology $shell_dcp $names $audit]
    set scoped [otpu_classify_scoped_clocks $names $evidence_prefix $audit]
    set inherited [dict get $shell inherited]
    set context [dict get $scoped context]
    # The context findings a CL has (check_scoped_clocks.tcl names the CL it found, fail
    # closed): the DDR CL three (the HDK's DDR UI clock group, the main / core pair), the HBM CL
    # two (the pair). The 35 inherited shell findings are the same exact match for both.
    if {![dict exists $scoped kind] || [dict get $scoped kind] ni {ddr hbm}} {
        error "openTPU: the scoped-clock classifier did not name the CL"
    }
    set kind [dict get $scoped kind]
    set want [dict get {ddr 3 hbm 2} $kind]
    set accepted [concat $inherited $context]
    if {[llength $inherited] != 35 || [llength $context] != $want ||
        [llength [lsort -unique $accepted]] != 35 + $want} {
        error "openTPU: unexpected methodology classification inventory ($kind CL: [llength $inherited] inherited, [llength $context] context)"
    }
    foreach name $accepted {
        if {$name ni $raw} {
            error "openTPU: classified methodology finding was not a raw blocker: $name"
        }
    }
    set remaining {}
    foreach name $raw {if {$name ni $accepted} {lappend remaining $name}}
    puts $audit [list methodology_summary cl $kind raw_blockers [llength $raw] \
        inherited_shell [llength $inherited] scoped_clock_context [llength $context] \
        remaining_blockers [llength $remaining]]
    return [dict create raw_blockers $raw inherited $inherited context $context blockers $remaining kind $kind]
}
