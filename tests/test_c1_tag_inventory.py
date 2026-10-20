"""Current packed tag RAM scope equality and typed collection boundaries."""
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
MOCK=r'''
set ::macro {WRAPPER/CL/u_board/adapter/engines[0].tag_q_reg_0_31_0_13}
set ::leaves [list $::macro/RAMA $::macro/RAMH]
set ::mutation {}
proc tok {kind name} {return [list @$kind|$name]}
proc obj {objects kind} {
 if {[llength $objects]!=1 || ![string match @$kind|* [lindex $objects 0]]} {error "untyped $kind $objects"}
 return [string range [lindex $objects 0] [expr {[string length $kind]+2}] end]
}
proc inventory {} {
 set result [list $::macro {*}$::leaves]
 if {$::mutation eq "missing_leaf"} {set result [lrange $result 0 1]}
 if {$::mutation eq "empty_macro"} {lappend result $::macro/extra}
 return $result
}
proc get_cells {args} {
 set ix [lsearch -exact $args -of_objects]
 if {$ix>=0} {return [tok cell [file dirname [obj [lindex $args [expr {$ix+1}]] pin]]]}
 if {[lsearch -exact $args -hierarchical]>=0} {
  set filter_index [lsearch -exact $args -filter]
  if {$filter_index<0 || ![string match {NAME =~ "*"} [lindex $args [expr {$filter_index+1}]]]} {error "full NAME filter required"}
  set r {};foreach name [inventory] {lappend r {*}[tok cell $name]};return $r
 }
 return [tok cell [lindex [lindex $args end] 0]]
}
proc get_pins {args} {
 set ix [lsearch -exact $args -of_objects]
 if {$ix>=0} {
  set net [obj [lindex $args [expr {$ix+1}]] net]
  if {[lsearch -exact $args -leaf]<0 || [lindex $args end] ne {DIRECTION == OUT}} {error "global fanout"}
  return [tok pin [expr {[string match *CLK $net] ? "WRAPPER/CL/core_clock/O" : "WRAPPER/CL/receive_gate/O"}]]
 }
 return [tok pin [lindex [lindex $args end] 0]]
}
proc get_nets {args} {
 if {[lsearch -exact $args -segments]>=0} {error "global net expansion"}
 set ix [lsearch -exact $args -of_objects]
 if {$ix<0} {return [tok net [lindex [lindex $args end] 0]]}
 return [tok net [obj [lindex $args [expr {$ix+1}]] pin]]
}
proc get_ports {args} {obj [lindex $args end] net;return {}}
proc get_clocks {args} {
 set name [obj [lindex $args end] pin]
 return [tok clock [expr {$::mutation eq "wrong_clock" && [string match */CLK $name] ? "hbm_clk" : "core_clk"}]]
}
proc list_property {objects} {return {NAME REF_NAME IS_PRIMITIVE REF_PIN_NAME DIRECTION IS_INVERTED}}
proc get_property {key objects} {
 set values {}
 foreach object $objects {
  if {![regexp {^@(cell|pin|net|clock)\|(.+)$} $object -> kind name]} {error "untyped property $object"}
  set value {}
  if {$key eq "NAME"} {set value $name
  } elseif {$key eq "REF_NAME"} {
   if {$name eq $::macro || [string match */extra $name]} {set value RAM32M16
   } elseif {$name in $::leaves} {
    set value [expr {[file tail $name] eq "RAMA" ? "RAMD32" : "RAMS32"}]
    if {$::mutation eq "wrong_ref"} {set value RAMD64E}
   } else {set value LUT2}
  } elseif {$key eq "IS_PRIMITIVE"} {set value [expr {$::mutation eq "macro_not_primitive" && $name eq $::macro ? 0 : 1}]
  } elseif {$key eq "REF_PIN_NAME"} {set value [file tail $name]
  } elseif {$key eq "DIRECTION"} {set value [expr {[file tail $name] eq "O" ? "OUT" : "IN"}]
  } elseif {$key eq "IS_INVERTED"} {set value [expr {$::mutation eq "inverted_clock" && [file tail $name] eq "CLK" ? 1 : 0}]}
  lappend values $value
 }
 if {[llength $objects]==1} {return [lindex $values 0]};return $values
}
'''


def invoke(tmp_path,mutation='',capture=False):
    path=ROOT/'boards/c1_f2_alex'/('capture_native.tcl' if capture else 'audit_c1_alex.tcl')
    out=tmp_path/'raw';script=tmp_path/'run.tcl'
    text='source {'+str(path)+'}\n'+MOCK+'\nset ::mutation '+('{'+mutation+'}')+'\nset f [open {'+str(out)+'} w]\n'
    if capture:
        text+='set ::c1_native_capture::out $f\n::c1_native_capture::tag_memory 0\nif {$::c1_native_capture::gaps} {error "capture gaps"}\n'
    else:
        text+='set endpoints [list [lindex $::leaves 0]/WE [lindex $::leaves 1]/WE]\n'
        if mutation=='missing_fanout':text+='set endpoints [lrange $endpoints 0 0]\n'
        if mutation=='extra_fanout':text+='lappend endpoints $::macro/RAMB/WE\n'
        text+='::c1_alex::tag_ram_inventory WRAPPER/CL/ 0 $endpoints $f\n'
    script.write_text(text+'close $f\n')
    r=subprocess.run(['tclsh',str(script)],capture_output=True,text=True)
    return r,out.read_text()


def test_complete_tag_scope_audit(tmp_path):
    r,text=invoke(tmp_path);assert r.returncode==0,r.stderr
    assert 'tag_ram_inventory engine 0' in text and text.count('tag_ram_control engine 0')==4


@pytest.mark.parametrize('mutation',['missing_leaf','empty_macro','wrong_clock','wrong_ref','inverted_clock','missing_fanout','extra_fanout','macro_not_primitive'])
def test_tag_scope_audit_rejects_mutation(tmp_path,mutation):
    r,text=invoke(tmp_path,mutation);assert r.returncode!=0
    assert 'C1 Alex audit:' in r.stderr


def test_tag_capture_emits_complete_scoped_typed_controls(tmp_path):
    r,text=invoke(tmp_path,capture=True);assert r.returncode==0,r.stderr
    assert 'query {tag_memory_inventory 0}' in text
    assert text.count('cone_edge {tag_memory 0}')==4
    assert 'query tag_memory_reference_clock' in text and 'names core_clk' in text
