# C2 read-only HBM kernel qualification, sourced on the shell-linked routed DCP.
# No DDR initialization or pumped-TMEM exceptions apply. The frozen severity and
# methodology classifiers remain fail-closed; this file does not waive findings.

proc c2_hbm_property {object key} {
  if {$key ni [list_property $object]} { error "C2 HBM: missing $key on $object" }
  set value [get_property $key $object]
  if {$value eq ""} { error "C2 HBM: empty $key on $object" }
  return $value
}

proc c2_hbm_crossings {prefix} {
  set clocks [dict create]
  foreach {name pin period} {
    main WRAPPER/CL/clk_main_a0 4.0
    core WRAPPER/CL/u_clk/O 5.0
    hbm  WRAPPER/CL/u_hclk/O 2.222222222222
  } {
    set selected [get_pins -quiet $pin]
    if {[llength $selected] != 1} { error "C2 HBM: missing or ambiguous $name clock pin" }
    set clock [get_clocks -quiet -of_objects $selected]
    if {[llength $clock] != 1} { error "C2 HBM: missing or ambiguous $name clock" }
    set actual [c2_hbm_property $clock PERIOD]
    if {![string is double -strict $actual] || abs($actual-$period)>0.001} {
      error "C2 HBM: unexpected $name clock period $actual; expected $period"
    }
    dict set clocks $name $clock
  }
  set count 0
  foreach {a b} {main core core main core hbm hbm core main hbm hbm main} {
    set src [dict get $clocks $a]; set dst [dict get $clocks $b]
    report_timing -from $src -to $dst -max_paths 1 -path_type full_clock_expanded \
      -file ${prefix}_${a}_to_${b}.rpt
    set paths [get_timing_paths -quiet -from $src -to $dst -max_paths 1 -nworst 1]
    if {[llength $paths] != 1} { error "C2 HBM: missing timed $a to $b CDC witness" }
    set path [lindex $paths 0]
    set exception [c2_hbm_property $path EXCEPTION]
    set requirement [c2_hbm_property $path REQUIREMENT]
    set target [c2_hbm_property $dst PERIOD]
    if {![regexp {^MaxDelay Path ([0-9]+(?:\.[0-9]+)?)ns -datapath_only$} $exception -> bound] ||
        ![string is double -strict $requirement] ||
        abs($bound-$target)>0.001 || abs($requirement-$target)>0.001} {
      error "C2 HBM: $a to $b lacks destination-period max-delay bound ($exception; $requirement)"
    }
    if {[c2_hbm_property $path STARTPOINT_CLOCK] ne [c2_hbm_property $src NAME] ||
        [c2_hbm_property $path ENDPOINT_CLOCK] ne [c2_hbm_property $dst NAME]} {
      error "C2 HBM: $a to $b CDC witness clocks changed"
    }
    incr count
  }
  return $count
}

proc c2_hbm_gray_count {text} {
  # UG906: the Summary lists ALL set_bus_skew constraints; the later per-
  # constraint detail can be limited and must not be counted instead.
  # https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Bus-Skew-Report-Summary-Section
  # The titles also appear in the table of contents. Select the underlined
  # sections, then parse Id + Position records; Vivado wraps From/To cell lists
  # across many lines. Printed Tcl expressions are data and are never evaluated.
  if {![regexp -indices {(^|\n)1\. Bus Skew Report Summary\r?\n-+\r?\n} $text heading]} {
    error "C2 HBM: unrecognized bus-skew summary format"
  }
  set begin [expr {[lindex $heading 1]+1}]
  if {![regexp -indices -start $begin {(^|\n)2\. Bus Skew Report Per Constraint\r?\n-+\r?\n} $text heading]} {
    error "C2 HBM: missing bus-skew detail boundary"
  }
  set summary [string range $text $begin [expr {[lindex $heading 0]-1}]]
  set rows [dict create]; set id {}
  foreach line [split $summary \n] {
    if {[regexp {^\s*([0-9]+)\s+([0-9]+)\s+\[} $line -> next_id position]} {
      if {[dict exists $rows $next_id]} {error "C2 HBM: duplicate bus-skew summary id"}
      set id $next_id
      dict set rows $id "$line\n"
    } elseif {$id ne ""} {
      if {[regexp {^\S} $line]} {error "C2 HBM: malformed bus-skew summary row"}
      dict append rows $id "$line\n"
    } elseif {[string match {*u_sys/u_mem/*} $line]} {
      error "C2 HBM: fabric entry without a bus-skew summary id"
    }
  }
  set ids [dict create]
  dict for {id row} $rows {
    if {![string match {*u_sys/u_mem/*} $row]} {continue}
    set last [string trim [lindex [split [string trim $row] \n] end]]
    if {![regexp {^(?:Fast|Slow)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)$} $last -> requirement actual slack] ||
        ![string is double -strict $requirement] || ![string is double -strict $actual] ||
        ![string is double -strict $slack]} {
      error "C2 HBM: incomplete fabric bus-skew summary row $id"
    }
    dict set ids $id 1
  }
  # Eight read-only native bridges: two FIFOs each; PCIS: three FIFOs.
  # Each FIFO has two pointer directions. Post-opt retains the eight native
  # write-completion counters as well as the host counter: 19*2 + 8 + 1 = 47.
  # Any different inventory needs review.
  if {[dict size $ids] != 47} {
    error "C2 HBM: expected 47 fabric gray-bus constraints; found [dict size $ids]"
  }
  return [dict size $ids]
}

proc otpu_validate_hbm_route {scripts_dir reports_dir checkpoints_dir CL TAG shell_dcp} {
  set prefix ${reports_dir}/${CL}.${TAG}.route
  set marker ${checkpoints_dir}/${CL}.${TAG}.route_validation.passed
  if {[file exists $marker]} { error "C2 HBM: refusing to reuse an existing validation marker" }
  report_route_status -file ${prefix}_status.rpt
  set timing_criticals [get_msg_config -count -severity {CRITICAL WARNING}]
  report_timing_summary -delay_type min_max -report_unconstrained \
    -check_timing_verbose -warn_on_violation -file ${prefix}_timing.rpt
  set timing_failed [expr {[get_msg_config -count -severity {CRITICAL WARNING}]>$timing_criticals}]
  set skew_criticals [get_msg_config -count -severity {CRITICAL WARNING}]
  report_bus_skew -warn_on_violation -file ${prefix}_bus_skew.rpt
  set skew_failed [expr {[get_msg_config -count -severity {CRITICAL WARNING}]>$skew_criticals}]
  report_drc -file ${prefix}_drc.rpt
  report_methodology -file ${prefix}_methodology.rpt
  source ${scripts_dir}/check_rule_severity.tcl
  source ${scripts_dir}/classify_methodology.tcl
  set audit [open ${prefix}_rule_severities.rpt w]
  set severity_failed [catch {
    set drc_bad [otpu_blocking_violations drc [get_drc_violations] $audit]
    set methodology [otpu_qualify_methodology $shell_dcp $prefix $audit]
    set methodology_bad [dict get $methodology blockers]
  } severity_error]
  close $audit
  if {$severity_failed} { error $severity_error }
  report_cdc -details -file ${prefix}_cdc.rpt
  report_exceptions -ignored -file ${prefix}_ignored_exceptions.rpt
  report_clock_interaction -file ${prefix}_clock_interaction.rpt
  report_utilization -hierarchical -file ${prefix}_utilization.rpt
  set bad_nets [get_nets -hierarchical -filter {
    ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL ||
    ROUTE_STATUS == UNPLACED || ROUTE_STATUS == UNPLACED_ALL || ROUTE_STATUS == CONFLICTS ||
    ROUTE_STATUS == GAPS || ROUTE_STATUS == ANTENNAS || ROUTE_STATUS == NODRIVER ||
    ROUTE_STATUS == MULTI_DRIVER
  }]
  set setup_bad [get_timing_paths -setup -max_paths 1 -slack_lesser_than 0]
  set hold_bad [get_timing_paths -hold -max_paths 1 -slack_lesser_than 0]
  set crossing_count [c2_hbm_crossings $prefix]
  set f [open ${prefix}_bus_skew.rpt r]; set skew_text [read $f]; close $f
  set gray_count [c2_hbm_gray_count $skew_text]
  set summary [open ${prefix}_validation.txt w]
  foreach {key value} [list setup_failed [llength $setup_bad] hold_failed [llength $hold_bad] \
      timing_failed $timing_failed bus_skew_failed $skew_failed gray_bus_constraints $gray_count \
      cdc_crossing_witnesses $crossing_count unrouted_nets [llength $bad_nets] \
      drc_blockers [llength $drc_bad] \
      methodology_raw_blockers [llength [dict get $methodology raw_blockers]] \
      methodology_inherited_shell [llength [dict get $methodology inherited]] \
      methodology_scoped_clock_context [llength [dict get $methodology context]] \
      methodology_blockers [llength $methodology_bad]] { puts $summary "$key=$value" }
  close $summary
  if {[llength $bad_nets] || [llength $setup_bad] || [llength $hold_bad] ||
      $timing_failed || $skew_failed || [llength $drc_bad] || [llength $methodology_bad]} {
    error "C2 HBM: routed design failed qualification; see ${prefix}_validation.txt and reports"
  }
  set passed [open $marker {WRONLY CREAT EXCL}]
  puts $passed "OTPU_STAGE_ROUTE_VALIDATION_PASSED"; close $passed
  puts "OTPU_STAGE_ROUTE_VALIDATION_PASSED"
}

otpu_validate_hbm_route $scripts_dir $reports_dir $checkpoints_dir $CL $TAG \
  [file join $::env(HDK_SHELL_DIR) build checkpoints from_aws cl_bb_routed.small_shell.dcp]
