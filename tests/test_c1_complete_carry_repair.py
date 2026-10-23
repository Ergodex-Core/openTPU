"""Execute complete promoted-carry repair Tcl with an exact finite design model."""
from pathlib import Path
import importlib.util
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('c1_complete_implement',ROOT/'boards/c1_f2_alex/implement.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
STUB=r'''
# Vivado uses Tcl8; local Tcl9 removed bytelength. Match Tcl8 UTF-8 semantics.
if {[catch {string bytelength {}}]} {
 rename string native_string
 proc string {operation args} {
  if {$operation eq "bytelength"} {return [native_string length [encoding convertto utf-8 [lindex $args 0]]]}
  return [uplevel 1 [list native_string $operation {*}$args]]
 }
}
set cells {};set pins {};set nets {};set blocks {};set added {};set attack @ATTACK@
proc addpin {owner suffix direction} {
 set name $owner/$suffix
 dict set ::pins $name [dict create NAME $name OWNER $owner DIRECTION $direction REF_PIN_NAME $suffix]
 return $name
}
proc connect {name members} {
 dict set ::nets $name $members
 foreach p $members {dict set ::pins $p NET $name}
}
set model_chains @CHAINS@
set prior @PRIOR@
set promoted {};set frontier {};set all {}
foreach chain $model_chains {
 foreach name $chain {lappend all $name}
 foreach name [lrange $chain 0 end-1] {lappend promoted $name}
 lappend frontier [lindex $chain end]
}
foreach name $all {
 set parent [expr {$name in $promoted ? "WRAPPER/CL" : {WRAPPER/CL/u_board/core/cluster/engines[15].engine}}]
 set membership [expr {$name in $promoted && $name ni $prior ? "pblock_CL" : "pblock_CL_SLR0"}]
 dict set cells $name [dict create NAME $name PARENT $parent REF_NAME CARRY8 CARRY_TYPE SINGLE_CY8 DONT_TOUCH 0 KEEP 0 LOC {} BEL {} IS_LOC_FIXED 0 IS_BEL_FIXED 0 PBLOCK $membership]
 foreach {suffix direction} {CI IN CI_TOP IN {CO[7]} OUT {S[0]} IN {O[0]} OUT} {addpin $name $suffix $direction}
}
foreach name {WRAPPER/CL/GND WRAPPER/CL/GND_1} {dict set cells $name [dict create NAME $name REF_NAME GND];addpin $name G OUT}
set ci_gnd [list WRAPPER/CL/GND/G];set top_gnd [list WRAPPER/CL/GND_1/G]
foreach chain $model_chains {
 lappend ci_gnd [lindex $chain 0]/CI
 foreach name $chain {lappend top_gnd $name/CI_TOP}
 for {set i 0} {$i<[llength $chain]-1} {incr i} {
  set name [lindex $chain $i];set next [lindex $chain [expr {$i+1}]]
  connect edge/$i/$name [list $name/CO\[7\] $next/CI]
 }
}
connect ground $ci_gnd;connect ground_top $top_gnd
foreach name [dict keys $pins] {if {![dict exists $pins $name NET]} {connect own/$name [list $name]}}
foreach {name parent} {pblock_CL ROOT pblock_CL_SLR0 pblock_CL} {dict set blocks $name [dict create NAME $name PARENT $parent GRID_RANGES fixed-region-$name IS_SOFT 0 CONTAIN_ROUTING 0 SNAPPING_MODE NESTED EXCLUDE_PLACEMENT 0]}
set target [lindex [lindex $model_chains 0] 0];set endpoint [lindex [lindex $model_chains 0] end]
if {$attack eq "missing_promoted"} {dict unset cells $target}
if {$attack eq "wrong_membership"} {dict set cells $target PBLOCK pblock_CL_SLR0}
if {$attack eq "wrong_parent"} {dict set cells $target PARENT WRAPPER/CL/u_board}
if {$attack eq "wrong_primitive"} {dict set cells $target REF_NAME LUT6}
if {$attack eq "protected"} {dict set cells $target DONT_TOUCH 1}
if {$attack eq "wrong_ci"} {dict set pins $target/CI NET ground_top}
if {$attack eq "wrong_ci_top"} {dict set pins $target/CI_TOP NET ground}
if {$attack eq "extra_output_consumer"} {set n [dict get $pins $target/CO\[7\] NET];dict lappend nets $n $target/S\[0\]}
if {$attack eq "soft_region"} {dict set blocks pblock_CL_SLR0 IS_SOFT 1}
proc get_property {property objects} {
 if {[llength $objects]!=1} {error "Mock property cardinality"}
 set object [lindex $objects 0]
 foreach table {cells pins blocks} {if {[dict exists [set ::$table] $object $property]} {return [dict get [set ::$table] $object $property]}}
 if {$property eq "NAME" && [dict exists $::nets $object]} {return $object}
 error "Unavailable mock property $object $property"
}
proc list_property {object} {return [dict keys [dict get $::cells [lindex $object 0]]]}
proc get_cells {args} {
 if {[set at [lsearch -exact $args -of_objects]]>=0} {
  set result {};foreach p [lindex $args [expr {$at+1}]] {lappend result [dict get $::pins $p OWNER]};return [lsort -unique $result]
 }
 set at [lsearch -exact $args -filter];set filter [lindex $args [expr {$at+1}]]
 if {[string match {REF_NAME == CARRY8*} $filter]} {set result {};foreach n $::promoted {if {[dict exists $::cells $n] && [dict get $::cells $n REF_NAME] eq "CARRY8"} {lappend result $n}};return $result}
 if {![regexp {^NAME == "(.+)"$} $filter -> name]} {error "Unexpected cell query"}
 if {[dict exists $::cells $name]} {return [list $name]};return {}
}
proc get_pins {args} {
 if {[set at [lsearch -exact $args -of_objects]]>=0} {
  set result {}
  foreach object [lindex $args [expr {$at+1}]] {
   if {[dict exists $::cells $object]} {dict for {n v} $::pins {if {[dict get $v OWNER] eq $object} {lappend result $n}}} else {foreach p [dict get $::nets $object] {lappend result $p}}
  }
  set result [lsort -unique $result]
 } else {set result [lindex $args end];foreach p $result {if {![dict exists $::pins $p]} {return {}}}}
 if {[set at [lsearch -exact $args -filter]]>=0} {set filter [lindex $args [expr {$at+1}]];if {$filter ne "DIRECTION == OUT"} {error "Unexpected pin filter"};set selected {};foreach p $result {if {[dict get $::pins $p DIRECTION] eq "OUT"} {lappend selected $p}};set result $selected}
 return $result
}
proc get_nets {args} {set at [lsearch -exact $args -of_objects];set result {};foreach p [lindex $args [expr {$at+1}]] {lappend result [dict get $::pins $p NET]};return [lsort -unique $result]}
proc get_ports {args} {if {$::attack eq "external_port"} {return external};return {}}
proc get_pblocks {args} {
 if {[set at [lsearch -exact $args -of_objects]]>=0} {return [list [dict get $::cells [lindex [lindex $args [expr {$at+1}]] 0] PBLOCK]]}
 set name [lindex $args end];if {[dict exists $::blocks $name]} {return [list $name]};return {}
}
proc add_cells_to_pblock {block objects} {
 foreach c $objects {dict set ::cells $c PBLOCK $block;lappend ::added $c}
 if {[llength $::added]==188} {
  if {$::attack eq "change_region"} {dict set ::blocks pblock_CL_SLR0 GRID_RANGES changed}
  if {$::attack eq "change_property"} {dict set ::cells $::target KEEP 1}
  if {$::attack eq "change_net"} {dict set ::pins $::target/S\[0\] NET ground}
  if {$::attack eq "drop_pin"} {dict unset ::pins $::target/S\[0\]}
  if {$::attack eq "miss_add"} {dict set ::cells $::target PBLOCK pblock_CL}
 }
}
'''

def execute(tmp_path,attack):
    script=m.complete_carry_pblock_script()
    chains=script.split('set chains ',1)[1].split('\n',1)[0]
    prior=script.split('set prior ',1)[1].split('\n',1)[0]
    stub=STUB.replace('@CHAINS@',chains).replace('@PRIOR@',prior).replace('@ATTACK@',attack)
    command='set run {'+str(tmp_path)+'}\n'+stub+'\nset code [catch {\n'+script+'\n} message options]\nputs MOCK_ADDED_COUNT=[llength $::added]\nif {$code} {puts stderr [dict get $options -errorinfo];exit 1}\n'
    p=tmp_path/'repair.tcl';p.write_text(command)
    return subprocess.run([shutil.which('tclsh') or 'tclsh',str(p)],capture_output=True,text=True)

def test_complete_graph_adds_only_remaining188_preserving_every_cell(tmp_path):
    r=execute(tmp_path,'none');assert r.returncode==0,r.stderr
    raw=(tmp_path/'complete_carry_pblock_repair.tcldata').read_text()
    assert raw.count('\nbefore ')==349 and raw.count('\nafter ')==349
    assert raw.count('\nsuccessor ')==190 and 'added_cells 188 prior_repaired_cells 2' in raw
    assert 'MOCK_ADDED_COUNT=188' in r.stdout

@pytest.mark.parametrize('attack',['missing_promoted','wrong_membership','wrong_parent','wrong_primitive','protected','wrong_ci','wrong_ci_top','extra_output_consumer','soft_region','external_port','change_region','change_property','change_net','drop_pin','miss_add'])
def test_complete_graph_rejects_mutation(tmp_path,attack):
    r=execute(tmp_path,attack);assert r.returncode!=0
    expected={
        'missing_promoted':'Complete promoted carry inventory differs',
        'wrong_primitive':'Complete promoted carry inventory differs',
        'wrong_membership':'Carry pre-repair membership differs',
        'wrong_parent':'Carry primitive or hierarchy differs',
        'protected':'Carry primitive or hierarchy differs',
        'wrong_ci':'Carry input driver differs',
        'wrong_ci_top':'Carry input driver differs',
        'extra_output_consumer':'Carry output endpoint inventory differs',
        'soft_region':'Carry hard region hierarchy differs',
        'external_port':'Carry input net boundary differs',
        'change_region':'Carry region changed',
        'change_property':'Carry primitive, pin inventory or connectivity changed',
        'change_net':'Carry primitive, pin inventory or connectivity changed',
        'drop_pin':'Carry primitive, pin inventory or connectivity changed',
        'miss_add':'Carry membership repair incomplete',
    }
    assert expected[attack] in r.stderr
    count=188 if attack in {'change_region','change_property','change_net','drop_pin','miss_add'} else 0
    assert f'MOCK_ADDED_COUNT={count}' in r.stdout
    output=tmp_path/'complete_carry_pblock_repair.tcldata'
    assert not output.exists() or '\ncomplete ' not in output.read_text()

def test_new_strategy_replays_exact_historical_two_cell_repair_first():
    baseline='set_param general.maxThreads 4\nopt_design\nwrite_checkpoint -force $run/post_opt.dcp\n  phys_opt_design -directive AggressiveExplore\nroute_design\n'
    result=m.physical_script(baseline,'align-complete-promoted-carry-pblocks',8)
    assert result.count(m.CARRY_PBLOCK_SCRIPT)==1
    assert result.index(m.CARRY_PBLOCK_SCRIPT)<result.index(m.complete_carry_pblock_script())<result.index('write_checkpoint')
    assert 'set_param general.maxThreads 8' in result and '  phys_opt_design' not in result

def test_exact_component_inventory_has190_promoted_and159_frontiers():
    assert len(m.CARRY_COMPLETE_COMPONENTS)==159
    assert sum(len(c)-1 for c in m.CARRY_COMPLETE_COMPONENTS)==190
    assert len({x for c in m.CARRY_COMPLETE_COMPONENTS for x in c[:-1]})==190
    assert len({c[-1] for c in m.CARRY_COMPLETE_COMPONENTS})==159
