"""Typed synthetic register-equivalence tests; no hardware acceptance fixtures."""
import subprocess
import pytest
from test_c2_alex_routed_host_ready import MOCK, HELPER, tl, td

ORIGINAL='WRAPPER/CL/u_board/returned_reg[5][4]'
REPLICA=ORIGINAL+'_replica'


def run(tmp_path, mutation='', endpoint=None, flow_failure=False):
    props={};cp={};pn={};nl={}
    def pin(name,direction,net):
        ref=name.rsplit('/',1)[-1]
        props['pin|'+name]={'NAME':name,'REF_PIN_NAME':ref,'DIRECTION':direction,'IS_CLOCK':str(int(ref=='C')),'IS_INVERTED':'0'}
        pn[name]=tl([net]);nl.setdefault(net,[]).append(name)
    for name,origin in [(ORIGINAL,''),(REPLICA,'returned_reg[5][4]')]:
        props['cell|'+name]={'NAME':name,'REF_NAME':'FDRE','ORIG_REF_NAME':'FDRE','INIT':"1'b0",'IS_SEQUENTIAL':'1','ORIG_CELL_NAME':origin,'PHYS_OPT_MODIFIED':'CRITICAL_CELL_OPT'}
        cp[name]=tl(name+'/'+r for r in ('C','CE','D','Q','R'))
        for ref in ('C','CE','D','Q','R'):
            pin(name+'/'+ref,'OUT' if ref=='Q' else 'IN', name if ref=='Q' else ref)
        sink=name+'_sink/I0';owner=sink.rsplit('/',1)[0]
        props['cell|'+owner]={'REF_NAME':'LUT6'};pin(sink,'IN',name)
    for ref in ('C','CE','D','R'):
        driver='WRAPPER/CL/u_core_clk/inst/clkout1_buf/O' if ref=='C' else 'WRAPPER/CL/u_board/control_'+ref+'/O'
        pin(driver,'OUT',ref)
    declarations='\n'.join('set '+name+' {'+td(value)+'}' for name,value in [('properties',{k:td(v) for k,v in props.items()}),('cell_pins',cp),('pin_net',pn),('net_leaf',{k:tl(v) for k,v in nl.items()})])
    mock=MOCK.replace('state_reg\\[s_ready_i\\]','returned_reg\\[5\\]\\[4\\]')
    mock=mock.replace('&& [string first "returned_reg\\[5\\]\\[4\\]" $key]>=0','&& [string first "returned_reg\\[5\\]\\[4\\]" $key]>=0 && [dict exists $properties IS_SEQUENTIAL]')
    mock=mock.replace('clk_main_a0','clk_out1_f2_core_clk')
    endpoint=endpoint or REPLICA+'/CE'; out=tmp_path/'audit.tcldata'
    script=declarations+'\nset ports {}\n'+mock+'\n'+mutation+f'''\nsource {{{HELPER}}}
namespace eval ::c2_alex {{}}
proc ::c2_alex::native_receive_endpoint {{root pc endpoint}} {{
 if {{$pc==5 && $endpoint in {{{{{ORIGINAL}/R}} {{{ORIGINAL}/CE}}}}}} {{return returned}}
 if {{$endpoint eq "ordinary"}} {{return shared_state}}
 return {{}}
}}
set original_body [info body ::c2_alex::native_receive_endpoint]
proc ::c2_alex::flow {{prefix interface signal audit}} {{
 set family [::c2_alex::native_receive_endpoint WRAPPER/CL/ 5 {{{endpoint}}}]
 if {{$family eq {{}}}} {{error "unreviewed endpoint"}}
 {'error "downstream failure"' if flow_failure else ''}
 puts $audit [list family $family]
}}
set audit [open {{{out}}} w]
set code [catch {{::c2_alex_routed_ready::receive_flow WRAPPER/CL S_N5 rvalid $audit}} message options]
close $audit
if {{[info body ::c2_alex::native_receive_endpoint] ne $original_body || [llength [info commands ::c2_alex::native_receive_endpoint_frozen]]}} {{error "classifier restoration failed"}}
puts RESTORED
if {{$code}} {{return -options $options $message}}
'''
    path=tmp_path/'test.tcl';path.write_text(script)
    result=subprocess.run(['tclsh',str(path)],capture_output=True,text=True)
    return result,out


def test_equivalent_duplicate_and_frozen_delegate_restored(tmp_path):
    r,p=run(tmp_path);assert r.returncode==0,r.stderr
    assert 'RESTORED' in r.stdout and 'structural_pass 1 route_qualified 0' in p.read_text()


def test_ordinary_endpoint_uses_unchanged_classifier(tmp_path):
    r,p=run(tmp_path,endpoint='ordinary');assert r.returncode==0,r.stderr
    assert p.read_text()=='family shared_state\n'


def test_restore_after_downstream_failure(tmp_path):
    r,p=run(tmp_path,flow_failure=True);assert r.returncode!=0 and 'downstream failure' in r.stderr and 'RESTORED' in r.stdout


@pytest.mark.parametrize('mutation,reason',[
 (f'dict set properties {{cell|{REPLICA}}} INIT {{1\'b1}}','primitive/initial'),
 (f'dict set properties {{cell|{REPLICA}}} PHYS_OPT_MODIFIED OTHER','physical transformation'),
 (f'dict set properties {{cell|{REPLICA}}} ORIG_CELL_NAME wrong','replica origin'),
 (f'dict set properties {{cell|{ORIGINAL}}} ORIG_CELL_NAME other','replica origin'),
 (f'dict set properties {{pin|{REPLICA}/D}} IS_INVERTED 1','pin inversion'),
 (f'dict set pin_net {{{REPLICA}/D}} R','replica D drivers differs'),
 (f'dict set pin_net {{{REPLICA}/CE}} R','replica CE drivers differs'),
 (f'dict set pin_net {{{REPLICA}/R}} D','replica R drivers differs'),
 (f'dict set pin_net {{{REPLICA}/C}} D','core clock'),
 (f'dict set cell_pins {{{REPLICA}}} {{{{{REPLICA}/Q}}}}','pin inventory'),
 (f'dict set net_leaf {{{REPLICA}}} {{{{{REPLICA}/Q}}}}','ready leaf inventory'),
 (f'dict set properties {{cell|{ORIGINAL}_replica_2}} [dict get $properties {{cell|{ORIGINAL}}}]','register family'),
 ('set ports {external}','top port escape'),
])
def test_rejects_non_equivalent_or_unproven_pair_and_restores(tmp_path,mutation,reason):
    r,p=run(tmp_path,mutation);assert r.returncode!=0 and reason in r.stderr,r.stderr
    assert 'RESTORED' in r.stdout and 'structural_pass 1' not in p.read_text()


@pytest.mark.parametrize('endpoint',[REPLICA+'/D',REPLICA.replace('[5]','[4]')+'/CE',REPLICA+'_2/CE','WRAPPER/CL/u_board/unknown/R'])
def test_unknown_state_never_admitted_by_suffix(tmp_path,endpoint):
    r,p=run(tmp_path,endpoint=endpoint);assert r.returncode!=0 and 'unreviewed endpoint' in r.stderr
    assert 'RESTORED' in r.stdout
