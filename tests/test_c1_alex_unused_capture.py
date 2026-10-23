"""Unused-net capture discovers current findings and preserves typed objects."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'boards/c1_f2_alex/capture_unused.tcl'
TCL=shutil.which('tclsh')
pytestmark=pytest.mark.skipif(TCL is None,reason='Tcl unavailable')


def run(tmp_path,mutation=''):
    output=tmp_path/'raw'
    code='source {'+str(SOURCE)+'}\n'+r'''
proc h {kind name} {return "HANDLE:$kind:$name"}
proc typed {obj} {if {![regexp {^HANDLE:([^:]+):(.*)$} $obj -> kind name]} {error "Plain object rejected $obj"};return [list $kind $name]}
proc version {args} {return 2025.2}
proc get_designs {args} {return design}
proc get_drc_violations {args} {return [list [h drc RTSTAT-10#1] [h drc OTHER#1]]}
proc get_property {key obj} {
 lassign [typed $obj] kind name
 switch $key {
  NAME {return $name}
  CLASS {return $kind}
  CHECK {return [lindex [split $name #] 0]}
  REF_NAME {return LUT1}
  IS_PRIMITIVE {return 1}
  DIRECTION {return OUT}
  REF_PIN_NAME {return O}
  ROUTE_STATUS {return NOLOADS}
  TYPE {return SIGNAL}
  default {return 0}
 }
}
proc object {args} {set i [lsearch -exact $args -of_objects];if {$i<0} {error "Missing typed relation"};return [lindex $args [expr {$i+1}]]}
proc get_nets {args} {
 set objects [object {*}$args];foreach obj $objects {lassign [typed $obj] kind name}
 if {$kind eq "drc"} {return [list [h net WRAPPER/CL/new_unused] [h net WRAPPER/STATIC/old_unused]]}
 return [list [h net WRAPPER/CL/new_unused]]
}
proc get_pins {args} {foreach obj [object {*}$args] {typed $obj};return [list [h pin WRAPPER/CL/driver/O]]}
proc get_ports {args} {foreach obj [object {*}$args] {typed $obj};return {}}
proc get_cells {args} {foreach obj [object {*}$args] {typed $obj};return [list [h cell WRAPPER/CL/driver]]}
'''+mutation+'\n'
    code+='if {[catch {::c1_unused_capture::collect {'+str(output)+'} '+('a'*64)+'} result]} {puts stderr $result;exit 1}\nputs $result\n'
    r=subprocess.run([TCL],input=code,text=True,capture_output=True,timeout=30)
    return r,output.read_text() if output.exists() else ''


def test_discovers_actual_cl_net_and_preserves_full_finding_inventory(tmp_path):
    r,text=run(tmp_path)
    assert r.returncode==0,r.stderr
    assert 'kind c1_unused_output_nets' in text and 'related_count 2' in text
    assert 'WRAPPER/STATIC/old_unused' in text
    assert 'selected_nets names WRAPPER/CL/new_unused' in text
    assert 'complete nets 1 route_qualified 0' in text
    assert 'properties {NAME WRAPPER/CL/driver/O CLASS pin' in text


@pytest.mark.parametrize('mutation,expected',[
 ('proc get_drc_violations {args} {return {}}','Missing RTSTAT-10'),
 ('set ::c1_unused_capture::max_all 1','drc_findings cap'),
 ('set ::c1_unused_capture::max_current 0','current_CL_nets cap'),
 ('set ::c1_unused_capture::max_connections 0','connected_pins cap'),
 ('proc get_cells {args} {return {}}','Missing/multiple leaf owner'),
 ('proc get_pins {args} {return {}}','Missing net connection'),
 ('proc version {args} {return 2025.1}','Vivado 2025.2'),
])
def test_missing_and_capped_evidence_fails_without_complete(tmp_path,mutation,expected):
    r,text=run(tmp_path,mutation)
    assert r.returncode!=0 and expected in r.stderr
    assert 'complete nets' not in text


def test_real_ports_are_preserved_without_unused_acceptance(tmp_path):
    r,text=run(tmp_path,'proc get_ports {args} {foreach obj [object {*}$args] {typed $obj};return [list [h port external]]}')
    assert r.returncode==0,r.stderr
    assert 'port WRAPPER/CL/new_unused name external' in text
    assert 'route_qualified 1' not in text


def test_source_only_no_checkpoint_mutation_or_expected_count():
    text=SOURCE.read_text()
    for token in ('open_checkpoint','close_design','write_checkpoint','set_property','list_property','==35'):
        assert token not in text
    r=subprocess.run([TCL],input='source {'+str(SOURCE)+'}\nputs loaded\n',text=True,capture_output=True)
    assert r.returncode==0 and r.stdout.strip()=='loaded'
