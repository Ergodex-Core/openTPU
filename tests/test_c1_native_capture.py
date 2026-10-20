"""Offline raw-collector boundary tests; no Vivado or hardware qualification."""
from pathlib import Path
import shutil
import subprocess

CAPTURE=Path(__file__).resolve().parents[1]/'boards/c1_f2_alex/capture_native.tcl'
MOCK=r'''
# Strings are not typed objects; only selectors mint typed tokens. Serializing
# a token via Tcl list deliberately destroys its type, modelling Vivado's trap.
rename list typed_original_list
proc list {args} {
    set result {}
    foreach x $args {lappend result [string map {@pin| plain-pin| @cell| plain-cell| @net| plain-net|} $x]}
    return $result
}
proc token {kind name} {return [typed_original_list @$kind|$name]}
proc typed {object kind} {
    if {[llength $object]!=1 || ![string match @$kind|* $object]} {error "untyped $kind: $object"}
    return [string range $object [expr {[string length $kind]+2}] end]
}
set pin_names {cl/gate/I0 cl/gate/I1 cl/gate/O cl/reset/Q cl/reset/C cl/const/G}
proc get_property {key object} {
    if {[llength $object]!=1 || ![regexp {^@(pin|cell|net|clock)\|(.+)$} $object -> kind name]} {error "raw name or bulk property query: $object"}
    if {$key eq "NAME"} {return $name}
    if {$kind eq "pin"} {
        switch $key {
            REF_PIN_NAME {return [file tail $name]}
            DIRECTION {return [expr {[file tail $name] in {O Q G} ? "OUT" : "IN"}]}
            IS_CLOCK {return [expr {[file tail $name] eq "C"}]}
            IS_INVERTED {return 0}
        }
    }
    if {$kind eq "cell"} {
        switch $key {
            REF_NAME - ORIG_REF_NAME {return [dict get {cl/gate LUT2 cl/reset FDRE cl/const GND} $name]}
            IS_SEQUENTIAL {return [expr {$name eq "cl/reset"}]}
            IS_PRIMITIVE {return 1}
            IS_BLACKBOX {return 0}
            INIT {return [expr {$name eq "cl/gate" ? "4'h2" : "1'b0"}]}
        }
    }
    return {}
}
proc list_property {object} {
    get_property NAME $object
    return {NAME REF_NAME ORIG_REF_NAME IS_PRIMITIVE IS_SEQUENTIAL IS_BLACKBOX INIT REF_PIN_NAME DIRECTION IS_CLOCK IS_INVERTED}
}
proc get_cells {args} {
    set i [lsearch -exact $args -of_objects]
    if {$i>=0} {set name [file dirname [typed [lindex $args [expr {$i+1}]] pin]]} else {set name [lindex $args end]}
    if {$name ni {cl/gate cl/reset cl/const}} {return {}}
    return [token cell $name]
}
proc get_pins {args} {
    set i [lsearch -exact $args -of_objects]
    if {$i>=0} {
        set obj [lindex $args [expr {$i+1}]]
        if {[string match @cell|* $obj]} {
            set owner [typed $obj cell];set names {}
            foreach p $::pin_names {if {[file dirname $p] eq $owner} {lappend names $p}}
        } else {
            set net [typed $obj net]
            set names [dict get {cl/data {cl/const/G cl/gate/I0} cl/rst {cl/reset/Q cl/gate/I1} cl/out {cl/gate/O}} $net]
        }
    } else {set p [lindex $args end];set names {};if {$p in $::pin_names} {set names [typed_original_list $p]}}
    set result {};set f [lsearch -exact $args -filter]
    foreach name $names {
        if {$f>=0 && [get_property DIRECTION [token pin $name]] ne [lindex [lindex $args [expr {$f+1}]] end]} {continue}
        lappend result {*}[token pin $name]
    }
    return $result
}
proc get_nets {args} {
    set i [lsearch -exact $args -of_objects]
    if {$i>=0} {
        set pin [typed [lindex $args [expr {$i+1}]] pin]
        set name [dict get {cl/gate/I0 cl/data cl/const/G cl/data cl/gate/I1 cl/rst cl/reset/Q cl/rst cl/gate/O cl/out cl/reset/C cl/rst} $pin]
    } else {set name [lindex $args end]}
    if {$name ni {cl/data cl/rst cl/out}} {return {}}
    return [token net $name]
}
proc get_ports {args} {set i [lsearch -exact $args -of_objects];typed [lindex $args [expr {$i+1}]] net;return {}}
proc get_clocks {args} {set i [lsearch -exact $args -of_objects];typed [lindex $args [expr {$i+1}]] pin;return {}}
'''

def run(tmp_path,body,mock=True):
    script='source {'+str(CAPTURE)+'}\n'+(MOCK if mock else '')
    script+='\nset ::c1_native_capture::out [open {'+str(tmp_path/'capture.tcldata')+'} w]\n'
    script+=body+'\nclose $::c1_native_capture::out\n'
    # tclsh reading stdin does not reliably exit nonzero on Tcl command errors.
    path=tmp_path/'test.tcl';path.write_text(script)
    proc=subprocess.run([shutil.which('tclsh'),str(path)],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    return (tmp_path/'capture.tcldata').read_text()

def test_typed_reselection_full_network_and_cone(tmp_path):
    text=run(tmp_path,r'''
set n [::c1_native_capture::network net cl/out]
if {[dict get $n drivers] ne {cl/gate/O}} {error "driver missing"}
::c1_native_capture::cone ready [dict get $n drivers]
if {$::c1_native_capture::gaps!=0} {error "typed handle gap"}
''')
    assert 'cone_terminal ready pin cl/reset/Q cell cl/reset reference FDRE' in text
    assert 'cone_terminal ready pin cl/const/G cell cl/const reference GND' in text
    assert 'value 4\'h2' in text and 'incomplete' not in text

def test_more_than_500_names_and_explicit_cap(tmp_path):
    text=run(tmp_path,r'''
proc many {} {set r {};for {set i 0} {$i<601} {incr i} {lappend r @pin|pin$i};return $r}
set got [::c1_native_capture::qnames full {many}]
if {[llength $got]!=601 || [lindex $got end] ne "pin600"} {error "truncated collection"}
set ::c1_native_capture::max_names 600
::c1_native_capture::qnames capped {many}
''')
    assert 'reason name_cap total 601 retained 600' in text
    assert '...' not in text

def test_errors_missing_objects_and_depth_cap_are_raw_markers(tmp_path):
    text=run(tmp_path,r'''
::c1_native_capture::qnames broken {error actual_api_error}
if {[::c1_native_capture::selection pin absent]} {error "invented pin"}
set ::c1_native_capture::max_depth 0
::c1_native_capture::cone ready {cl/gate/O}
''')
    assert 'reason api_error message actual_api_error' in text
    assert 'selection pin absent status missing_or_ambiguous observed {}' in text
    assert 'reason depth_cap' in text
    assert 'route_qualified 1' not in text

def test_checkpoint_mismatch_never_opens_design(tmp_path):
    dcp=tmp_path/'source.dcp';dcp.write_bytes(b'fixture')
    text=run(tmp_path,r'''
proc ::c1_native_capture::hash {p} {return [string repeat a 64]}
proc open_checkpoint {p} {error "MUST NOT OPEN"}
if {![catch {::c1_native_capture::capture {'''+str(tmp_path/'out')+'} {'+str(dcp)+r'''} [string repeat b 64]} msg] || $msg ne "Checkpoint binding mismatch"} {error "wrong binding guard"}
''',mock=False)
    assert not (tmp_path/'out').exists() and text==''


def test_disabled_input_captures_only_drivers_without_constant_segments(tmp_path):
    text=run(tmp_path,r'''
rename get_nets original_get_nets
proc get_nets {args} {
 if {"-segments" in $args} {error "global constant segment expansion forbidden"}
 return [original_get_nets {*}$args]
}
rename get_pins original_get_pins
proc get_pins {args} {
 if {"-leaf" in $args && "-filter" ni $args} {error "global constant sink enumeration forbidden"}
 return [original_get_pins {*}$args]
}
::c1_native_capture::disabled 0 aw cl/gate/I0
if {$::c1_native_capture::gaps!=0} {error "typed direct driver capture failed"}
''')
    assert 'scope selected_input_and_drivers' in text
    assert 'drivers cl/const/G' in text and 'value GND' in text
    assert 'route_qualified 1' not in text


def test_backward_cone_avoids_constant_sinks_and_caches_owner_names(tmp_path):
    text=run(tmp_path,r'''
rename get_nets original_get_nets
proc get_nets {args} {
 if {"-segments" in $args} {error "global segment expansion forbidden"}
 return [original_get_nets {*}$args]
}
rename get_pins original_get_pins
proc get_pins {args} {
 if {"-leaf" in $args && "-filter" ni $args} {error "unneeded sink enumeration forbidden"}
 return [original_get_pins {*}$args]
}
set first [::c1_native_capture::owners cl/gate/O]
rename get_cells original_get_cells
proc get_cells {args} {
 if {"-of_objects" in $args && [lindex $args end] eq "@pin|cl/gate/O"} {error "owner queried twice"}
 return [original_get_cells {*}$args]
}
if {[::c1_native_capture::owners cl/gate/O] ne $first} {error "cached owner changed"}
::c1_native_capture::cone ready {cl/gate/O}
if {$::c1_native_capture::gaps!=0} {error "typed direct cone failed"}
''')
    assert 'cone_terminal ready pin cl/const/G cell cl/const reference GND' in text
    assert 'scope selected_input_and_drivers' in text


def test_endpoint_batch_keeps_all_typed_owners_and_required_properties(tmp_path):
    text=run(tmp_path,r'''
rename get_pins original_get_pins
proc get_pins {args} {
 if {"-of_objects" ni $args && [llength [lindex $args end]]>1} {
  set result {}
  foreach name [lindex $args end] {lappend result {*}[original_get_pins -quiet $name]}
  return $result
 }
 return [original_get_pins {*}$args]
}
rename get_cells original_get_cells
proc get_cells {args} {
 if {"-of_objects" in $args && [llength [lindex $args end]]>1} {
  set result {}
  foreach object [lindex $args end] {foreach cell [original_get_cells -quiet -of_objects $object] {if {$cell ni $result} {lappend result $cell}}}
  return $result
 }
 return [original_get_cells {*}$args]
}
rename list_property original_list_property
proc list_property {object} {error "optional endpoint property enumeration forbidden"}
::c1_native_capture::endpoint_facts 0 cl/gate/O {cl/gate/I0 cl/gate/I1}
if {$::c1_native_capture::gaps!=0} {error "typed batch endpoint capture failed"}
''')
    assert 'property pin cl/gate/I0 DIRECTION status observed value IN' in text
    assert 'property pin cl/gate/I1 DIRECTION status observed value IN' in text
    assert 'property cell cl/gate REF_NAME status observed value LUT2' in text
    assert 'query {pin_owner cl/gate/I0} status observed total 1 names cl/gate' in text
    assert 'query {pin_owner cl/gate/I1} status observed total 1 names cl/gate' in text
