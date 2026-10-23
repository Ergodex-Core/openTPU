from pathlib import Path
import copy
import subprocess
import pytest
from boards.c2_f2_alex import exception_review as m

COLLECTOR=Path(__file__).parents[1]/'boards/c2_f2_alex/capture_handshake.tcl'

MOCK=r'''
set prefix WRAPPER/CL/u_ocl_cdc/inst/gen_clock_conv.gen_async_lite_conv.clock_conv_lite_
set all_names {}
foreach suffix {fwd_ar fwd_aw fwd_w resp_b resp_r} {
    set root ${prefix}${suffix}/handshake
    lappend all_names $root
    foreach leaf {src_hsdata_ff_reg dest_hsdata_ff_reg xpm_cdc_single_src2dest_inst/syncstages_ff_reg} {
        set width [expr {[string first syncstages $leaf]>=0 ? 3 : 2}]
        for {set i 0} {$i<$width} {incr i} {lappend all_names "${root}/${leaf}\[$i\]"}
    }
}
proc get_cells {args} {
    if {[lsearch -exact $args -of_objects]>=0} {
        set object [lindex [lindex $args end] 0]
        if {![string match pin|* $object]} {error "UNTYPED pin handle"}
        return [list cell|[string range $object 4 end-2]]
    }
    set filter [lindex $args end];set result {}
    if {[regexp {^NAME == "(.*)"$} $filter -> name]} {
        if {$name in $::all_names} {return [list cell|$name]};return {}
    }
    if {![regexp {^NAME =~ (.*)$} $filter -> pattern]} {error "Unsupported filter $filter"}
    foreach name $::all_names {if {[string match $pattern $name]} {lappend result cell|$name}}
    return $result
}
proc get_pins {args} {
    if {![regexp {^NAME == "(.*)"$} [lindex $args end] -> name]} {error "Bad pin filter"}
    if {[string range $name 0 end-2] ni $::all_names} {return {}}
    return [list pin|$name]
}
proc get_clocks {args} {
    set object [lindex [lindex $args end] 0]
    if {![string match pin|* $object]} {error "UNTYPED clock pin"}
    set forward [expr {[string first clock_conv_lite_fwd_ $object]>=0}]
    set source [expr {[string first /src_hsdata_ff_reg $object]>=0}]
    return [list clock|[expr {$forward==$source ? "ocl4" : "core5"}]]
}
proc get_property {key object} {
    if {[llength $object]!=1 || ![regexp {^(cell|pin|clock)\|(.*)$} [lindex $object 0] -> kind name]} {error "UNTYPED property handle $object"}
    if {$key eq "NAME"} {return $name}
    if {$kind eq "clock"} {
        if {$key eq "PERIOD"} {return [expr {$name eq "ocl4" ? "4.000" : "5.000"}]}
        if {$key eq "WAVEFORM"} {return {0.000 2.000}}
    }
    if {$kind eq "pin"} {return [dict get {REF_PIN_NAME C DIRECTION IN IS_CLOCK 1} $key]}
    if {[string match */handshake $name]} {return [dict get {REF_NAME xpm_cdc_handshake ORIG_REF_NAME xpm_cdc_handshake IS_BLACKBOX 0} $key]}
    return [dict get {REF_NAME FDRE ORIG_REF_NAME FDRE ASYNC_REG TRUE IS_SEQUENTIAL 1} $key]
}
'''


def capture(tmp_path):
    output=tmp_path/'handshake.tcldata';script=tmp_path/'test.tcl'
    script.write_text(MOCK+f'\nset ::c2_handshake_library_only 1\nsource {{{COLLECTOR}}}\nset out [open {{{output}}} w]\n::c2_handshake::capture $out '+('a'*64)+'\nclose $out\n')
    result=subprocess.run(['tclsh',str(script)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return output


def test_strict_typed_capture_and_replay(tmp_path):
    path=capture(tmp_path);result=m.parse_handshake_witness(path,'a'*64)
    assert set(result)==m.OCL_ROOTS
    for root,row in result.items():
        assert row['width']==2 and row['forward_sync_stages']==3
        assert m.Decimal(row['destination_period'])==(5 if '_fwd_' in root else 4)


@pytest.mark.parametrize('old,new,reason',[
    ('checkpoint_sha256 '+'a'*64,'checkpoint_sha256 '+'b'*64,'checkpoint'),
    ('ASYNC_REG TRUE','ASYNC_REG FALSE','ASYNC_REG'),
    ('complete roots 5 route_qualified 0','complete roots 4 route_qualified 0','Incomplete'),
    ('IS_BLACKBOX 0','IS_BLACKBOX 1','opacity'),
    ('IS_CLOCK 1','IS_CLOCK 0','clock pin'),
])
def test_reject_changed_handshake_evidence(tmp_path,old,new,reason):
    path=capture(tmp_path);path.write_text(path.read_text().replace(old,new))
    with pytest.raises(ValueError,match=reason):m.parse_handshake_witness(path,'a'*64)


def test_no_plain_string_object_query_regression():
    text=COLLECTOR.read_text()
    assert 'set cell [one cell $name]' in text
    assert 'set pin [one pin $name/C]' in text
    assert 'get_clocks -quiet -of_objects $pin' in text


def test_physical_skew_uses_actual_handshake_endpoints_and_bound(tmp_path):
    witnesses=m.parse_handshake_witness(capture(tmp_path),'a'*64)
    rows={}
    for index,(root,w) in enumerate(witnesses.items()):
        bound=m.Decimal(w['destination_period'])*w['forward_sync_stages']
        rows[str(index)]={'from':' '.join('{'+x+'}' for x in w['source_registers']),
            'to':' '.join('{'+x+'}' for x in w['destination_registers']),
            'requirement':str(bound),'actual':'1.000','slack':str(bound-1)}
    assert set(m.review_handshake_skew(rows,witnesses,{}))==m.OCL_ROOTS
    wrong=copy.deepcopy(rows);wrong['0']['requirement']='20.000'
    with pytest.raises(ValueError,match='formula'):m.review_handshake_skew(wrong,witnesses,{})
    wrong=copy.deepcopy(rows);wrong['0']['from']=wrong['0']['from'].replace('src_hsdata','unrelated')
    with pytest.raises(ValueError,match='endpoints'):m.review_handshake_skew(wrong,witnesses,{})
    wrong=copy.deepcopy(rows);wrong['0']['slack']='-0.001'
    with pytest.raises(ValueError,match='physical'):m.review_handshake_skew(wrong,witnesses,{})
    wrong=copy.deepcopy(rows);del wrong['0']
    with pytest.raises(ValueError,match='Incomplete'):m.review_handshake_skew(wrong,witnesses,{})


def test_signed_handshake_skew_preserves_physical_checks(tmp_path):
    witnesses=m.parse_handshake_witness(capture(tmp_path),'a'*64);rows={}
    for index,(root,w) in enumerate(witnesses.items()):
        bound=m.Decimal(w['destination_period'])*w['forward_sync_stages']
        rows[str(index)]={'from':' '.join('{'+x+'}' for x in w['source_registers']),
            'to':' '.join('{'+x+'}' for x in w['destination_registers']),
            'requirement':str(bound),'actual':'-0.008','slack':str(bound+m.Decimal('.008'))}
    assert set(m.review_handshake_skew(rows,witnesses,{}))==m.OCL_ROOTS
    wrong=copy.deepcopy(rows);wrong['0']['slack']='-0.001'
    with pytest.raises(ValueError,match='physical'):m.review_handshake_skew(wrong,witnesses,{})
    wrong=copy.deepcopy(rows);wrong['0']['slack']='0.001'
    with pytest.raises(ValueError,match='inconsistent'):m.review_handshake_skew(wrong,witnesses,{})
    for field in ('actual','slack'):
        for value in ('NaN','Infinity','-Infinity'):
            wrong=copy.deepcopy(rows);wrong['0'][field]=value
            with pytest.raises(ValueError,match='physical'):m.review_handshake_skew(wrong,witnesses,{})
