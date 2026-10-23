"""The optional inherited-shell classifier must reject every unreviewed change."""
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'boards/f2/build/scripts'
SHELL_SHA = '04a5331321e86cffc7ef740fb6132b29f179e68e2d58ccae5031c989403ecca6'
PIN_SHA = '46a550f9c99dfb0a4e1ba9df9f3c112a4d669dbf47966be3bb4e30532411a14f'


def run_tcl(script):
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl is unavailable')
    return subprocess.run([tclsh], input=script, text=True, capture_output=True, timeout=10)


def test_baseline_data_has_verified_provenance_and_canonical_pin_inventory():
    result = run_tcl(f'source {{{SCRIPTS / "shell_baseline_2025_2.tcl"}}}\n' + r'''
set data [otpu_shell_baseline_data]
foreach key {shell_sha256 report_sha256 report_bytes pin_inventory_sha256} {
    puts "$key=[dict get $data $key]"
}
foreach pin [lsort [dict keys [dict get $data pins]]] {puts "PIN=$pin"}
set known [dict get $data pins SL/SH_ICAP/ICAP/CLK]
puts "ICAP=[dict get $known cell_properties]"
''')
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    pins = [line[4:] for line in lines if line.startswith('PIN=')]
    assert len(pins) == len(set(pins)) == 35
    assert not any('{' in pin or '}' in pin for pin in pins)
    assert hashlib.sha256(('\n'.join(pins) + '\n').encode()).hexdigest() == PIN_SHA
    assert f'shell_sha256={SHELL_SHA}' in lines
    assert 'report_sha256=644b93b6f3876cf46f5bfb23038e572bb4becbcc23ced3326817374211fd407c' in lines
    assert 'report_bytes=5520187' in lines
    assert 'REF_NAME {ICAPE3} LOC {CONFIG_SITE_X0Y0} BEL {CONFIG_SITE.ICAP_TOP}' in result.stdout


MOCKS = r'''
set objects [dict create]
set kinds [dict create pins {} cells {} methodology_violations {}]
set related_pins [dict create]
set related_cells [dict create]
set roots [dict create]
set names {}
proc add_object {kind name properties} {
    set id @opaque[dict size $::objects]
    dict set ::objects $id [dict merge $properties [dict create NAME $name]]
    dict lappend ::kinds $kind $id
    return $id
}
proc id_named {name} {
    dict for {id properties} $::objects {
        if {[dict get $properties NAME] eq $name} {return $id}
    }
    error "mock name absent: $name"
}
set data [otpu_shell_baseline_data]
dict for {root properties} [dict get $data roots] {add_object pins $root $properties}
set n 0
dict for {pin expected} [dict get $data pins] {
    set p [add_object pins $pin [dict get $expected pin_properties]]
    set c [add_object cells [dict get $expected cell] [dict get $expected cell_properties]]
    set name TIMING-17#[incr n]
    set v [add_object methodology_violations $name {CHECK TIMING-17 SEVERITY {Critical Warning} IS_WAIVED 0 WAIVER {}}]
    lappend names $name
    dict set related_pins $v [list $p]
    dict set related_cells $v [list $c]
    dict set related_cells $p [list $c]
    dict set roots $p [list [id_named [dict get $expected root]]]
}
add_object methodology_violations TIMING-54#1 {CHECK TIMING-54 SEVERITY {Critical Warning}}
add_object methodology_violations OTHER#1 {CHECK OTHER SEVERITY Warning}
lappend names TIMING-54#1 OTHER#1
set v0 [id_named TIMING-17#1]
set v1 [id_named TIMING-17#2]
set p0 [lindex [dict get $related_pins $v0] 0]
set p1 [lindex [dict get $related_pins $v1] 0]
set c0 [lindex [dict get $related_cells $v0] 0]
set c1 [lindex [dict get $related_cells $v1] 0]
set r0 [lindex [dict get $roots $p0] 0]
switch -- $fault {
    unrelated_neighbor {
        set near_name [string map {. X} [dict get $objects $p0 NAME]]
        add_object pins $near_name [dict get $objects $p0]
    }
    missing_finding {
        set names [lrange $names 1 end]
        dict set kinds methodology_violations [lrange [dict get $kinds methodology_violations] 1 end]
    }
    caller_subset {set names [lrange $names 1 end]}
    caller_duplicate {lappend names TIMING-17#1}
    live_duplicate {dict lappend kinds methodology_violations $v0}
    new_pin {dict set objects $p0 NAME UNREVIEWED/C}
    duplicate_pin {dict set related_pins $v1 [list $p0]}
    multiple_pins {dict set related_pins $v0 [list $p0 $p1]}
    no_related_pin {dict set related_pins $v0 {}}
    wrong_check {dict set objects $v0 CHECK TIMING-54}
    missing_check {dict unset objects $v0 CHECK}
    wrong_severity {dict set objects $v0 SEVERITY Error}
    missing_severity {dict unset objects $v0 SEVERITY}
    waived {dict set objects $v0 IS_WAIVED 1}
    unknown_waiver_state {dict set objects $v0 IS_WAIVED unknown}
    missing_waiver_state {dict unset objects $v0 IS_WAIVED}
    nonempty_waiver {dict set objects $v0 WAIVER WAIVER-1}
    missing_waiver {dict unset objects $v0 WAIVER}
    cell_ref {dict set objects $c0 REF_NAME FDCE}
    cell_loc {dict set objects $c0 LOC SLICE_X0Y0}
    cell_bel {dict set objects $c0 BEL SLICEL.AFF}
    missing_location {dict unset objects $c0 LOC}
    empty_location {dict set objects $c0 LOC {}}
    pin_ref {dict set objects $p0 REF_PIN_NAME D}
    root_ref {dict set objects $r0 REF_NAME BUFGCE}
    wrong_cell_relation {dict set related_cells $v0 [list $c1]}
    wrong_pin_owner {dict set related_cells $p0 [list $c1]}
    multiple_cells {dict set related_cells $v0 [list $c0 $c1]}
    no_related_cell {dict set related_cells $v0 {}}
    duplicate_cell {dict set related_cells $p0 [list $c0 $c0]}
    same_name_bad_cell {
        set bad [add_object cells [dict get $objects $c0 NAME] [dict replace [dict get $objects $c0] LOC SLICE_X0Y0]]
        dict set related_cells $p0 [list $bad]
    }
    wrong_root {dict set roots $p0 [list $p1]}
    multiple_roots {dict lappend roots $p0 $p1}
    missing_root {dict set roots $p0 {}}
    new_critical {
        add_object methodology_violations NEW#1 {CHECK NEW SEVERITY {Critical Warning}}
        lappend names NEW#1
    }
    unknown_severity {
        dict set objects [id_named OTHER#1] SEVERITY Unknown
    }
}
proc list_property {obj} {return [dict keys [dict get $::objects $obj]]}
proc get_property {property obj} {return [dict get $::objects $obj $property]}
proc query {kind args} {
    if {[llength $args] == 0} {return [lreverse [dict get $::kinds $kind]]}
    if {[lindex $args 0] eq "-of_objects"} {
        set obj [lindex $args 1]
        if {$kind eq "pins"} {return [dict get $::related_pins $obj]}
        if {$kind eq "cells"} {return [dict get $::related_cells $obj]}
        error "unexpected relationship query"
    }
    # Reproduce the live failure mode: hierarchical pin/cell name regexes must
    # not be used. Relationship queries above are the supported lookup route.
    if {$kind in {pins cells}} {error "hierarchical named lookup is unsupported"}
    if {[lrange $args 0 1] ne {-quiet -regexp} || [llength $args] != 3} {error "unexpected named query $args"}
    set result {}
    foreach id [dict get $::kinds $kind] {
        if {[regexp [lindex $args 2] [dict get $::objects $id NAME]]} {lappend result $id}
    }
    if {$::fault eq "ambiguous_resolution" && $kind eq "methodology_violations" && $result eq [list $::v0]} {
        lappend result $::v0
    }
    if {$::fault eq "wrong_resolution" && $kind eq "methodology_violations" && $result eq [list $::v0]} {
        return [list $::v1]
    }
    return $result
}
proc get_pins {args} {return [query pins {*}$args]}
proc get_cells {args} {return [query cells {*}$args]}
proc get_methodology_violations {args} {return [query methodology_violations {*}$args]}
proc get_clocks {args} {
    if {$::fault eq "new_clock" && [lindex $args 1] eq $::p0} {return NEW_CLOCK}
    return {}
}
proc all_fanin {args} {
    if {[lrange $args 0 end-1] ne {-flat -startpoints_only -trace_arcs all}} {error "wrong fanin query"}
    return [dict get $::roots [lindex $args end]]
}
proc version {args} {if {$::fault eq "version"} {return 2025.1}; return 2025.2}
rename exec real_exec
proc exec {args} {
    if {[lrange $args 0 1] ne {sha256sum --}} {error "unexpected external command"}
    if {$::fault eq "shell_hash"} {return [string repeat 0 64]}
    return [dict get [otpu_shell_baseline_data] shell_sha256]
}
'''


@pytest.mark.parametrize('fault,error', [
    ('none', None), ('new_critical', None), ('unrelated_neighbor', None),
    ('shell_hash', 'SHA256 mismatch'), ('version', 'Vivado version'),
    ('missing_finding', 'exact 35-pin baseline'),
    ('caller_subset', 'live inventory'), ('caller_duplicate', 'live inventory'),
    ('live_duplicate', 'live inventory'),
    ('new_pin', 'unexpected or duplicate'), ('duplicate_pin', 'unexpected or duplicate'),
    ('multiple_pins', 'exactly one pin'), ('no_related_pin', 'exactly one pin'),
    ('wrong_check', 'inconsistent CHECK'), ('missing_check', 'missing CHECK'),
    ('wrong_severity', 'SEVERITY mismatch'), ('missing_severity', 'missing SEVERITY'),
    ('waived', 'waiver state'), ('unknown_waiver_state', 'waiver state'),
    ('missing_waiver_state', 'missing IS_WAIVED'),
    ('nonempty_waiver', 'WAIVER mismatch'), ('missing_waiver', 'missing WAIVER'),
    ('cell_ref', 'REF_NAME mismatch'), ('cell_loc', 'LOC mismatch'),
    ('cell_bel', 'BEL mismatch'), ('missing_location', 'missing LOC'),
    ('empty_location', 'LOC mismatch'), ('pin_ref', 'REF_PIN_NAME mismatch'),
    ('root_ref', 'REF_NAME mismatch'),
    ('wrong_cell_relation', 'related cell mismatch'), ('wrong_pin_owner', 'related cell mismatch'),
    ('multiple_cells', 'related cell mismatch'), ('no_related_cell', 'related cell mismatch'),
    ('duplicate_cell', 'related cell mismatch'), ('same_name_bad_cell', 'LOC mismatch'),
    ('wrong_root', 'clock root mismatch'), ('multiple_roots', 'clock root mismatch'),
    ('missing_root', 'clock root mismatch'), ('new_clock', 'now has a defined clock'),
    ('ambiguous_resolution', 'resolved to 2 objects'), ('wrong_resolution', 'identity mismatch'),
    ('unknown_severity', 'unknown severity'),
])
def test_shell_baseline_is_exact_and_keeps_other_critical_findings_blocking(tmp_path, fault, error):
    checkpoint = tmp_path / 'shell.dcp'
    checkpoint.write_text('mock shell; digest command mocked\n')
    harness = (f'source {{{SCRIPTS / "check_shell_baseline.tcl"}}}\n'
               f'source {{{SCRIPTS / "check_rule_severity.tcl"}}}\n'
               f'set fault {fault}\n' + MOCKS +
               f'\nset rc [catch {{set result [otpu_classify_shell_methodology {{{checkpoint}}} $names stdout]}} message]\n' + r'''
if {$rc} {puts stderr $message; exit 1}
set inherited [dict get $result inherited]
if {[llength $inherited] != 35 || [lsearch -glob $inherited @*] >= 0} {error "wrong inherited result"}
set remaining [dict get $result remaining]
if {"TIMING-54#1" ni $remaining || "TIMING-54#1" in $inherited} {error "TIMING-54 was accepted"}
set objects_remaining {}
foreach name $remaining {lappend objects_remaining [otpu_shell_exact methodology_violations $name]}
if {[catch {set blockers [otpu_blocking_violations methodology $objects_remaining stdout]} message]} {
    puts stderr $message; exit 1
}
set names_blocking [lsort [otpu_shell_names $blockers]]
if {$fault eq "new_critical"} {
    if {$names_blocking ne {NEW#1 TIMING-54#1}} {error "new critical finding escaped"}
} elseif {$names_blocking ne {TIMING-54#1}} {error "TIMING-54 did not remain blocking"}
# Classification must never modify the live severity or waiver properties.
if {[get_property SEVERITY $v0] ne "Critical Warning" || [get_property IS_WAIVED $v0] != 0} {
    error "classification mutated finding"
}
puts "CLASSIFICATION_CHECK_PASSED"
''')
    result = run_tcl(harness)
    if error:
        assert result.returncode != 0, result.stdout
        assert error in result.stderr
        if fault != 'unknown_severity':
            assert 'classification inherited_shell_baseline' not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.count('classification inherited_shell_baseline') == 35
        assert 'CLASSIFICATION_CHECK_PASSED' in result.stdout
