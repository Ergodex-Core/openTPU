"""Synthetic typed Tcl fixtures only, never routed qualification evidence."""
from pathlib import Path
import subprocess
import pytest

HELPER=Path(__file__).resolve().parents[1]/'boards/c2_f2_alex/routed_host_ready.tcl'
SCOPE='WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/s00_entry_pipeline/s00_mmu/inst/w_sreg'
ORIGINAL=SCOPE+'/state_reg[s_ready_i]'
REPLICA=ORIGINAL+'_replica'
PCIS='WRAPPER/CL/u_pcis_pipe/inst/w.w_pipe'


def tl(items):return ' '.join('{'+str(v)+'}' for v in items)
def td(value):return tl(v for pair in value.items() for v in pair)

MOCK=r'''
proc object_name {object} {
 if {[llength $object]!=1 || ![regexp {^(cell|pin|net|clock)\|(.*)$} [lindex $object 0] -> kind name]} {error "UNTYPED object $object"}
 return [list $kind $name]
}
proc get_property {key object} {
 lassign [object_name $object] kind name
 if {$key eq "NAME"} {return $name}
 return [dict get $::properties $kind|$name $key]
}
proc list_property {object} {
 lassign [object_name $object] kind name
 return [dict keys [dict get $::properties $kind|$name]]
}
proc get_cells {args} {
 if {[lsearch -exact $args -hierarchical]>=0} {
  set out {}
  dict for {key properties} $::properties {
   if {[string match "cell|*" $key] && [string first "state_reg\[s_ready_i\]" $key]>=0} {lappend out $key}
  }
  return $out
 }
 set oi [lsearch -exact $args -of_objects]
 if {$oi>=0} {
  lassign [object_name [lindex $args [expr {$oi+1}]]] kind name
  if {$kind ne "pin"} {error "Expected typed pin"}
  set owner [file dirname $name]
  if {[dict exists $::properties cell|$owner]} {return [list cell|$owner]};return {}
 }
 set name [lindex [lindex $args end] 0]
 if {[dict exists $::properties cell|$name]} {return [list cell|$name]};return {}
}
proc get_pins {args} {
 set oi [lsearch -exact $args -of_objects]
 if {$oi<0} {
  set name [lindex [lindex $args end] 0]
  if {[dict exists $::properties pin|$name]} {return [list pin|$name]};return {}
 }
 set out {}
 foreach object [lindex $args [expr {$oi+1}]] {
  lassign [object_name $object] kind name
  if {$kind eq "cell"} {set selected [dict get $::cell_pins $name]} elseif {$kind eq "net"} {set selected [dict get $::net_leaf $name]} else {error "Expected typed cell/net"}
  foreach pin $selected {
   if {[lsearch -exact $args -filter]>=0 && [dict get $::properties pin|$pin DIRECTION] ne "OUT"} {continue}
   lappend out pin|$pin
  }
 }
 return [lsort -unique $out]
}
proc get_nets {args} {
 set oi [lsearch -exact $args -of_objects]
 lassign [object_name [lindex $args [expr {$oi+1}]]] kind name
 if {$kind ne "pin"} {error "Expected typed pin for net"}
 set out {};foreach net [dict get $::pin_net $name] {lappend out net|$net};return $out
}
proc get_ports {args} {return $::ports}
proc get_clocks {args} {
 lassign [object_name [lindex $args end]] kind name
 if {$kind ne "pin"} {error "Expected typed clock pin"}
 if {[file tail $name] eq "C"} {return {clock|clk_main_a0}};return {}
}
'''


def fixture(direct=False):
 props={};cell_pins={};pin_net={};net_leaf={}
 def cell(name,ref='FDRE',origin=''):
  props['cell|'+name]={'NAME':name,'REF_NAME':ref,'ORIG_REF_NAME':ref,'INIT':"1'b0",'IS_SEQUENTIAL':'1','ORIG_CELL_NAME':origin,'PHYS_OPT_MODIFIED':'SLR_CROSSING_OPT'}
 def pin(name,direction,net):
  ref=name.rsplit('/',1)[-1]
  props['pin|'+name]={'NAME':name,'REF_PIN_NAME':ref,'DIRECTION':direction,'IS_CLOCK':'1' if ref=='C' else '0','IS_INVERTED':'0'}
  pin_net[name]=tl([net]);net_leaf.setdefault(net,[]).append(name)
 for name,origin in [(ORIGINAL,''),(REPLICA,'state_reg[s_ready_i]')]:
  cell(name,origin=origin);cell_pins[name]=tl(name+'/'+r for r in ('C','CE','D','Q','R'))
  for ref in ('C','CE','D','Q','R'):
   net='ready_replica' if ref=='Q' and name==REPLICA else 'ready_original' if ref=='Q' else 'control_'+ref
   pin(name+'/'+ref,'OUT' if ref=='Q' else 'IN',net)
 drivers={'C':'WRAPPER/IO_SHIM/RL_CLKS/BUFG_CLK_A_EXT/O','CE':SCOPE+'/VCC/P','D':SCOPE+'/state[s_ready_i]_i_1__3/O','R':SCOPE.rsplit('/',1)[0]+'/areset_reg/Q'}
 for ref,name in drivers.items():pin(name,'OUT','control_'+ref)
 for name,net in [(SCOPE+'/skid2vector_q_i_1__3/I2','ready_original'),(PCIS+'/state_i_1/I0','ready_replica')]:
  cell(name.rsplit('/',1)[0],'LUT6');pin(name,'IN',net)
 anchor='WRAPPER/CL/u_pcis_pipe/m_axi_wready';pin(anchor,'IN','ready_original');net_leaf['ready_original'].remove(anchor)
 if direct:
  props['cell|'+ORIGINAL]['PHYS_OPT_MODIFIED']=''
  props.pop('cell|'+REPLICA);cell_pins.pop(REPLICA)
  for ref in ('C','CE','D','Q','R'):
   name=REPLICA+'/'+ref;props.pop('pin|'+name);pin_net.pop(name)
   for leaves in net_leaf.values():
    if name in leaves:leaves.remove(name)
  for name in net_leaf.pop('ready_replica'):
   pin_net[name]=tl(['ready_original']);net_leaf['ready_original'].append(name)
 return props,cell_pins,pin_net,net_leaf


def run(tmp_path,mutation='',direct=False):
 props,cp,pn,nl=fixture(direct); path=tmp_path/'fixture.tcl';output=tmp_path/'witness.tcldata'
 declarations='\n'.join('set '+name+' {'+td(value)+'}' for name,value in [
  ('properties',{k:td(v) for k,v in props.items()}),('cell_pins',cp),('pin_net',pn),('net_leaf',{k:tl(v) for k,v in nl.items()})])
 path.write_text(declarations+'\nset ports {}\n'+MOCK+'\n'+mutation+f'\nsource {{{HELPER}}}\nset audit [open {{{output}}} w]\n::c2_alex_routed_ready::host_wready $audit\nclose $audit\n')
 result=subprocess.run(['tclsh',str(path)],capture_output=True,text=True)
 return result,output


def test_typed_routed_replica_witness_and_real_leaf_sink(tmp_path):
 result,path=run(tmp_path);assert result.returncode==0,result.stderr
 text=path.read_text();assert 'structural_pass 1 route_qualified 0' in text
 assert REPLICA+'/Q' in text and PCIS+'/state_i_1/I0' in text


@pytest.mark.parametrize('mutation,reason',[
 (f'dict set properties {{cell|{REPLICA}}} INIT {{1\'b1}}','primitive/initial state'),
 (f'dict set properties {{cell|{REPLICA}}} ORIG_CELL_NAME wrong','replica origin'),
 (f'dict set properties {{cell|{REPLICA}}} PHYS_OPT_MODIFIED OTHER','physical transformation'),
 (f'dict set properties {{pin|{REPLICA}/D}} IS_INVERTED 1','pin inversion'),
 (f'dict set pin_net {{{REPLICA}/D}} {{control_R}}','replica D drivers differs'),
 (f'dict set pin_net {{{REPLICA}/C}} {{control_D}}','replica C drivers differs'),
 (f'dict set cell_pins {{{REPLICA}}} {{{REPLICA}/Q}}','pin inventory'),
 ('set ports {port|external}','top port escape'),
 (f'dict set net_leaf ready_replica {{{REPLICA}/Q}}','missing/excess ready leaf'),
 (f'dict set net_leaf ready_replica {{{REPLICA}/Q {{{SCOPE}/skid2vector_q_i_1__3/I2}}}}','unexpected ready sink owner'),
 (f'dict lappend net_leaf ready_replica {{{ORIGINAL}/Q}}','multiple driver'),
 (f'dict set properties {{pin|{REPLICA}/Q}} IS_INVERTED {{}}','pin inversion'),
])
def test_rejects_non_equivalent_or_unproved_replica(tmp_path,mutation,reason):
 result,path=run(tmp_path,mutation);assert result.returncode!=0
 assert reason in result.stderr,result.stderr
 assert not path.exists() or 'structural_pass 1' not in path.read_text()


def test_direct_ready_has_same_control_anchor_and_both_consumer_scopes(tmp_path):
 result,path=run(tmp_path,direct=True);assert result.returncode==0,result.stderr
 text=path.read_text();assert 'kind routed_host_wready_direct' in text
 assert 'structural_pass 1 route_qualified 0' in text and REPLICA not in text
 assert PCIS+'/state_i_1/I0' in text and SCOPE+'/skid2vector_q_i_1__3/I2' in text


@pytest.mark.parametrize('mutation,reason',[
 (f'dict set properties {{cell|{ORIGINAL}}} PHYS_OPT_MODIFIED SLR_CROSSING_OPT','physical transformation'),
 (f'dict set properties {{cell|{ORIGINAL}}} ORIG_CELL_NAME other','original is itself'),
 (f'dict set properties {{cell|{ORIGINAL}}} INIT {{1\'b1}}','primitive/initial state'),
 (f'dict set properties {{pin|{ORIGINAL}/R}} IS_INVERTED 1','pin inversion'),
 (f'dict set pin_net {{{ORIGINAL}/D}} {{control_R}}','common D driver'),
 (f'dict set net_leaf ready_original {{{{{ORIGINAL}/Q}} {{{SCOPE}/skid2vector_q_i_1__3/I2}}}}','missing ready sink scope'),
 (f'dict set net_leaf ready_original {{{{{ORIGINAL}/Q}} {{{PCIS}/state_i_1/I0}}}}','missing ready sink scope'),
 (f'dict set properties {{cell|{ORIGINAL}_replica_2}} [dict get $properties {{cell|{ORIGINAL}}}]','register inventory'),
 ('set ports {port|external}','top port escape'),
])
def test_direct_ready_rejects_unknown_transform_or_incomplete_graph(tmp_path,mutation,reason):
 result,path=run(tmp_path,mutation,direct=True);assert result.returncode!=0
 assert reason in result.stderr,result.stderr
 assert not path.exists() or 'structural_pass 1' not in path.read_text()
