"""Reject incorrect receive gating, port scope and BRAM clocks."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "boards/c2_f2_alex/audit_c2_alex.tcl"

FIXTURE = r"""
lassign $argv script out fault prefix
source $script
set root [expr {$prefix eq {} ? {} : "$prefix/"}]
set gate "${root}u_mem/g_return\[2\].memory_reg_0_i_987"
set reset "${root}core_rst_reg"
set memory "${root}u_board/g_return\[2\].memory_reg_0"
set endpoints [list $memory/WEBWE\[0\] \
    ${root}u_board/returned_reg\[2\]\[0\]/CE \
    ${root}u_board/pc_returned_reg\[2\]\[0\]/D \
    ${root}u_board/pc_latency_reg\[2\]\[0\]/D \
    ${root}u_board/read_returned_reg\[0\]/D \
    ${root}u_native2/lr_reg\[0\]/CE \
    ${root}u_native2/lengths_reg_0_63_0_4/RAMA/WE \
    ${root}u_board/state_reg\[0\]/D]
switch $fault {
    other_port {lappend endpoints ${root}u_board/g_return\[3\].memory_reg_0/WEBWE\[0\]}
    other_counter {lappend endpoints ${root}u_board/pc_returned_reg\[3\]\[0\]/D}
    other_native {lappend endpoints ${root}u_native3/lr_reg\[0\]/CE}
    unrelated_state {lappend endpoints ${root}u_board/cfg_rows_reg\[0\]/D}
    unknown_pin {lappend endpoints $memory/ADDRBWRADDR\[0\]}
    unknown_bank {lappend endpoints ${root}u_board/g_return\[2\].memory_reg_8/WEBWE\[0\]}
    narrow_bank_bit {lappend endpoints ${root}u_board/g_return\[2\].memory_reg_7/WEBWE\[7\]}
    unknown_lengths {lappend endpoints ${root}u_native2/lengths_reg_128_255_0_4/RAMA/WE}
    missing_ram {set endpoints [lrange $endpoints 1 end]}
    missing_counter {set endpoints [lreplace $endpoints 2 2]}
    duplicate {lappend endpoints [lindex $endpoints 0]}
    outside_prefix {lappend endpoints WRAPPER/OTHER/u_board/state_reg\[0\]/D}
}
proc get_pins {args} {
    if {"-of_objects" in $args} {
        return [list [expr {$::fault eq "wrong_reset" ? "${::root}other_reg/Q" : "$::reset/Q"}]]
    }
    return [lindex $args end]
}
proc get_nets {args} {return reset_net}
proc get_cells {args} {
    if {"-of_objects" in $args} {return [list [file dirname [lindex [lindex $args end] 0]]]}
    return [lindex $args end]
}
proc get_property {key objects} {
    if {$key eq "NAME"} {return [expr {[llength $objects] == 1 ? [lindex $objects 0] : $objects}]}
    set object [lindex $objects 0]
    if {$key eq "REF_NAME"} {
        if {$object eq $::gate} {return [expr {$::fault eq "wrong_gate" ? "LUT3" : "LUT2"}]}
        if {$object eq $::reset} {return [expr {$::fault eq "wrong_reset_type" ? "LUT1" : "FDRE"}]}
        if {$object eq $::memory} {return [expr {$::fault eq "wrong_storage" ? "LUT6" : "RAMB36E2"}]}
        if {[string match */RAM? $object]} {return [expr {$::fault eq "wrong_lengths_type" ? "FDRE" : "RAMD64E"}]}
        if {[string first "${::root}u_" $object] == 0} {return [expr {$::fault eq "wrong_state_type" ? "LUT6" : "FDRE"}]}
        error "unknown fixture cell $object"
    }
    if {$key eq "INIT"} {return [expr {$::fault eq "wrong_function" ? "4'h8" : "4'h2"}]}
    if {$key eq "REF_PIN_NAME"} {return [file tail $object]}
    if {$key eq "DIRECTION"} {return [expr {[file tail $object] eq "O" ? "OUT" : "IN"}]}
    if {$key eq "IS_INVERTED"} {return [expr {$::fault eq "inverted" ? 1 : 0}]}
    error "unknown fixture property $key"
}
proc get_clocks {args} {
    set pin [lindex [lindex $args end] 0]
    if {$::fault eq "wrong_clock" && $pin eq "$::memory/CLKBWRCLK"} {return host_clock}
    if {$::fault eq "missing_clock"} {return {}}
    return core_clock
}
proc all_fanout {args} {return $::endpoints}
set stream [open $out w]
if {[catch {::c2_alex::native_return_gate $root 2 $gate/I0 $stream} message]} {
    close $stream;puts stderr $message;exit 1
}
close $stream
"""


@pytest.mark.parametrize("prefix", ["", "WRAPPER/CL"])
@pytest.mark.parametrize("fault", [
    "none", "other_port", "other_counter", "other_native", "unrelated_state",
    "unknown_pin", "unknown_bank", "narrow_bank_bit", "unknown_lengths",
    "missing_ram", "missing_counter", "duplicate", "outside_prefix",
    "wrong_reset", "wrong_reset_type", "wrong_gate", "wrong_storage",
    "wrong_function", "inverted", "wrong_clock", "missing_clock",
    "wrong_lengths_type", "wrong_state_type",
])
def test_receive_gate_requires_reset_function_port_scope_and_ram_clock(tmp_path, prefix, fault):
    script = tmp_path / "check.tcl"
    script.write_text(FIXTURE)
    process = subprocess.run(
        [shutil.which("tclsh"), str(script), str(AUDIT), str(tmp_path / "evidence"),
         fault, prefix], text=True, capture_output=True, timeout=20)
    assert (process.returncode == 0) == (fault == "none"), process.stderr


def test_actual_all_eight_receive_endpoint_inventories_match_source_scope(tmp_path):
    record = ROOT / "build/c2-alex-remote/metadata-probe-50f15ee2-5ad001/receive-endpoint-name-summary.json"
    if not record.is_file():
        pytest.skip("Optional retained Vivado endpoint evidence absent")
    rows = json.loads(record.read_text())
    assert {row["pc"] for row in rows} == set(range(8))
    assertions = []
    for row in rows:
        for endpoint in row["full_endpoint_names"]:
            assert "}" not in endpoint and "\n" not in endpoint
            assertions.append(
                f"if {{[::c2_alex::native_receive_endpoint {{}} {row['pc']} {{{endpoint}}}] eq {{}}}} "
                f"{{error {{unrecognized endpoint {endpoint}}}}}")
    script = tmp_path / "actual.tcl"
    script.write_text("source [lindex $argv 0]\n" + "\n".join(assertions) + "\n")
    process = subprocess.run([shutil.which("tclsh"), str(script), str(AUDIT)],
                             text=True, capture_output=True, timeout=20)
    assert process.returncode == 0, process.stderr
