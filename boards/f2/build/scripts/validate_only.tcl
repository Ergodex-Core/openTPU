# Qualify an immutable, already routed checkpoint with the current validator.
# No optimization, routing, relinking, constraint reads, or checkpoint writes.
proc otpu_validate_existing_route {scripts_dir} {
    set CL cl_opentpu
    set TAG $::env(BUILD_TAG)
    if {![regexp {^[0-9a-fA-F]{8}$} $TAG]} {error "openTPU: invalid routed build tag"}
    set reports_dir [file join [file dirname $scripts_dir] reports]
    set checkpoints_dir [file join [file dirname $scripts_dir] checkpoints]
    set parent_dcp $::env(OTPU_PARENT_DCP)
    set expected_sha $::env(OTPU_PARENT_SHA256)
    if {![regexp {^[0-9a-f]{64}$} $expected_sha]} {error "openTPU: invalid inherited checkpoint hash"}
    set accepted ${checkpoints_dir}/${CL}.${TAG}.post_route.dcp
    set marker ${checkpoints_dir}/${CL}.${TAG}.route_validation.passed
    foreach path [list $accepted $marker] {
        if {[file exists $path]} {error "openTPU: validation output already exists: $path"}
    }
    set opened 0
    set status [catch {
        if {[lindex [exec sha256sum -- $parent_dcp] 0] ne $expected_sha} {
            error "openTPU: inherited routed DCP hash mismatch"
        }
        set_param general.maxThreads 8
        open_checkpoint $parent_dcp
        set opened 1
        # The parent's CL (route_opt.py, from its frozen target): the DDR CL carries the DIMM's
        # calibration firmware, its sentinel pinned by the launcher (OTPU_EXPECTED_DDR_INIT); the HBM CL
        # (platform mem "hbm": no DIMM, the SmartConnect fabric WRAPPER/CL/u_sc) has none, and the
        # launcher passes none_hbm. The checkpoint must be that CL.
        set otpu_expect_hbm [expr {$::env(OTPU_EXPECTED_DDR_INIT) eq "none_hbm"}]
        set otpu_sc_cells [llength [get_cells -quiet WRAPPER/CL/u_sc]]
        if {($otpu_expect_hbm && $otpu_sc_cells != 1) || (!$otpu_expect_hbm && $otpu_sc_cells != 0)} {
          error "openTPU: the checkpoint's CL (SmartConnect fabric cells: $otpu_sc_cells) differs from the parent run's target"
        }
        if {$otpu_expect_hbm} {
          set ddr_cal_init none_hbm
        } else {
          source ${scripts_dir}/inspect_ddr_init.tcl
          if {![string equal -nocase $ddr_cal_init $::env(OTPU_EXPECTED_DDR_INIT)]} {
            error "openTPU: inherited DDR firmware sentinel mismatch"
          }
        }
        source ${scripts_dir}/validate_route.tcl
        set f [open $marker r]
        set marker_text [string trim [read $f]]
        close $f
        if {$marker_text ne "OTPU_STAGE_ROUTE_VALIDATION_PASSED"} {
            error "openTPU: route validator did not provide the exact qualification marker"
        }
        report_timing -delay_type min_max -max_paths 10 -nworst 1 -sort_by group \
            -path_type full_clock_expanded -input_pins \
            -file ${reports_dir}/${CL}.${TAG}.post_route_timing.rpt
        close_design
        set opened 0
        if {[lindex [exec sha256sum -- $parent_dcp] 0] ne $expected_sha} {
            error "openTPU: inherited routed DCP changed during validation"
        }
        # Copy the exact persisted bytes that were opened; never serialize the
        # in-memory design. The parent remains available as failure evidence.
        file copy $parent_dcp $accepted
        foreach path [list $parent_dcp $accepted] {
            if {[lindex [exec sha256sum -- $path] 0] ne $expected_sha} {
                error "openTPU: validation-only checkpoint copy hash mismatch"
            }
        }
    } message options]
    if {$opened} {catch {close_design}}
    if {$status} {
        foreach path [list $accepted $marker] {
            if {[file exists $path]} {file delete $path}
        }
        return -options $options $message
    }
    puts "OTPU_ROUTE_VALIDATE_ONLY_COMPLETED"
}
otpu_validate_existing_route [file dirname [file normalize [info script]]]
