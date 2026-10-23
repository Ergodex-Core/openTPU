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
    set accepted [concat $inherited $context]
    if {[llength $inherited] != 35 || [llength $context] != 3 ||
        [llength [lsort -unique $accepted]] != 38} {
        error "openTPU: unexpected methodology classification inventory"
    }
    foreach name $accepted {
        if {$name ni $raw} {
            error "openTPU: classified methodology finding was not a raw blocker: $name"
        }
    }
    set remaining {}
    foreach name $raw {if {$name ni $accepted} {lappend remaining $name}}
    puts $audit [list methodology_summary raw_blockers [llength $raw] \
        inherited_shell [llength $inherited] scoped_clock_context [llength $context] \
        remaining_blockers [llength $remaining]]
    return [dict create raw_blockers $raw inherited $inherited context $context blockers $remaining]
}
