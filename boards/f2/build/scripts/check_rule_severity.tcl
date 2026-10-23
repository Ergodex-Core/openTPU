# Resolve severity only from available properties on a violation or its rule.
# Never filter violations on an unverified property: an unsupported schema must
# fail qualification, rather than silently produce an empty blocker list.
proc otpu_blocking_violations {kind violations audit} {
  switch -- $kind {
    drc {set checks_command get_drc_checks}
    methodology {set checks_command get_methodology_checks}
    default {error "openTPU: unknown violation kind $kind"}
  }
  set blockers {}
  foreach violation $violations {
    set properties [list_property $violation]
    set owner $violation
    if {"SEVERITY" ni $properties} {
      if {"CHECK" ni $properties} {
        error "openTPU: $kind violation $violation has neither SEVERITY nor CHECK"
      }
      set check_name [get_property CHECK $violation]
      if {$check_name eq ""} {
        error "openTPU: $kind violation $violation has an empty CHECK"
      }
      # Both rule queries support a name pattern, not a violation -of_objects
      # query. Verify exact identity so wildcard matches cannot select a rule.
      set checks [$checks_command $check_name]
      if {[llength $checks] != 1} {
        error "openTPU: $kind violation $violation CHECK $check_name resolved to [llength $checks] rules"
      }
      set owner [lindex $checks 0]
      set properties [list_property $owner]
      if {"NAME" ni $properties || "SEVERITY" ni $properties} {
        error "openTPU: $kind rule $owner lacks NAME or SEVERITY"
      }
      if {[get_property NAME $owner] ne $check_name} {
        error "openTPU: $kind violation $violation CHECK did not resolve exactly to $check_name"
      }
    }
    set severity [string toupper [string trim [get_property SEVERITY $owner]]]
    switch -- $severity {
      FATAL - ERROR - "CRITICAL WARNING" {lappend blockers $violation}
      WARNING - ADVISORY {}
      default {error "openTPU: $kind violation $violation has unknown severity '$severity' on $owner"}
    }
    puts $audit [list $kind $violation severity $severity owner $owner]
  }
  return $blockers
}
