"""C1 raw collection cannot conceal structural/report failures or qualify."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "boards/c1_f2_alex/capture_final.tcl"
TCL = shutil.which("tclsh")
pytestmark = pytest.mark.skipif(TCL is None, reason="Tcl unavailable")


def replay(tmp_path, structural_error=True, report_error=False, timing_blockers=""):
    output = tmp_path / "raw"
    output.mkdir()
    # The ready helper is sourced during collection; mock the structural call,
    # never its successful/failed state in the collector itself.
    script = "source {" + str(SOURCE) + "}\n"
    mocks = {
        "structure": 'error "C1 observed host topology requires review"' if structural_error else "return {clocks current}",
        "gray_inventory": "return {fifo {source 0 destination 0}}",
        "gray_witnesses": "",
        "report_set": 'error "report generation failed"' if report_error else 'return {timing.rpt route_status.rpt bus_skew.rpt cdc.rpt waivers.rpt timing_all_ids.xdc}',
        "constraints": "return {counts {set_max_delay 1}}",
        "timing_summary": "return {blockers {" + timing_blockers + "}}",
        "route_summary": "return {routable 1 routed 1 errors 0 pass 1}",
        "bus_rows": "return {1 {slack 0.1 requirement 5.0}}",
        "gray_coverage": "",
        "cdc_rows": "return {{waived N}}",
        "findings": "return {counts {} blockers {}}",
        "supplemental": "return {queries 1 incomplete_or_error 0}",
    }
    for name, body in mocks.items():
        script += f"rename ::c1_alex_route::{name} ::c1_alex_route::original_{name}\n"
        script += f"proc ::c1_alex_route::{name} {{args}} {{{body}}}\n"
    for name in ("timing.rpt", "route_status.rpt", "bus_skew.rpt", "cdc.rpt", "timing_all_ids.xdc"):
        (output / name).write_text("placeholder\n")
    (output / "waivers.rpt").write_text("No waivers found.\n")
    script += f'if {{[catch {{::c1_final_capture::collect {{{output}}} /exact.dcp {"a"*64}}} result]}} {{puts stderr $result;exit 1}}\nputs $result\n'
    run = subprocess.run([TCL], input=script, text=True, capture_output=True, timeout=30)
    return run, output


def test_structural_error_retained_and_independent_reports_collected(tmp_path):
    run, output = replay(tmp_path)
    assert run.returncode == 0, run.stderr
    assert "structural_review_incomplete" in run.stdout
    assert "state unresolved" in run.stdout
    assert "C1 observed host topology requires review" in (output / "objects.tcldata").read_text()
    assert "route_qualified 0" in (output / "summary.tcldata").read_text()
    assert (output / "NOT_QUALIFIED.txt").is_file()


def test_successful_structure_and_timing_do_not_qualify(tmp_path):
    run, output = replay(tmp_path, structural_error=False)
    assert run.returncode == 0, run.stderr
    assert "state observed" in run.stdout and "route_qualified 1" not in run.stdout
    assert "current_structural_and_ready_equivalence" in run.stdout
    assert "raw_collection_only" in (output / "NOT_QUALIFIED.txt").read_text()


def test_physical_failure_remains_blocker(tmp_path):
    run, _ = replay(tmp_path, structural_error=False, timing_blockers="negative_wns")
    assert run.returncode == 0 and "blockers negative_wns" in run.stdout


def test_report_failure_preserved_without_success_summary(tmp_path):
    run, output = replay(tmp_path, report_error=True)
    assert run.returncode != 0 and "report generation failed" in run.stderr
    assert "capture_error" in (output / "objects.tcldata").read_text()
    assert not (output / "summary.tcldata").exists()


def test_input_binding_before_open_and_no_design_mutations(tmp_path):
    text = SOURCE.read_text()
    for token in ("set_property", "set_false_path", "set_max_delay", "create_waiver", "write_checkpoint", "route_design"):
        assert token not in text
    program = f"""
source {{{SOURCE}}}
proc open_checkpoint {{args}} {{error {{must not open}}}}
if {{![catch {{::c1_final_capture::capture /missing.dcp nope {{{tmp_path}/out}}}} result]}} {{exit 1}}
puts $result
"""
    run = subprocess.run([TCL], input=program, text=True, capture_output=True)
    assert run.returncode == 0 and "exact SHA256" in run.stdout
