"""Synthetic parser/semantic boundaries only; not evidence for real FPGA acceptance."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import shutil
import pytest

PATH=Path(__file__).resolve().parents[1]/'boards/c1_f2_alex/native_review.py'
spec=importlib.util.spec_from_file_location('c1_native_review_tested',PATH)
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
DCP='a'*64


def qword(value):
    value=str(value)
    assert not any(x in value for x in '{}\\\n')
    return '{'+value+'}' if not value or any(c.isspace() for c in value) else value

def tl(values):return ' '.join(qword(v) for v in values)
def nested(values):
    return ' '.join('{'+str(v)+'}' if any(c.isspace() for c in str(v)) or not str(v) else str(v) for v in values)
def dictionary(values):return nested([item for k,v in values.items() for item in (k,v)])


def cell(e,name,ref):
    e.queries[('select','cell',name)]=[name]
    e.objects['cell',name]={'NAME':name,'REF_NAME':ref,'IS_PRIMITIVE':'1','IS_BLACKBOX':'0','IS_SEQUENTIAL':'1' if ref=='FDRE' else '0'}

def pin(e,name,direction):
    e.queries[('select','pin',name)]=[name]
    e.queries[('pin_owner',name)]=[name.rsplit('/',1)[0]]
    e.objects['pin',name]={'NAME':name,'REF_PIN_NAME':name.rsplit('/',1)[1],'DIRECTION':direction,'IS_INVERTED':'0'}

def input_row(e,name,driver):
    pin(e,name,'IN');pin(e,driver,'OUT')
    net=name+'_net';e.queries[('input_nets',name)]=[net]
    e.queries[('input_drivers',name,net)]=[driver];e.queries[('input_ports',name,net)]=[]
    return {'scope':'selected_input_and_drivers','input':name,'properties':{'NAME':name,'DIRECTION':'IN'},'nets':[net],'drivers':[driver],'ports':[]}

def fixture():
    e=n.Evidence();constant=n.ROOT+'GND';cell(e,constant,'GND');reset=n.ROOT+'core_rst_reg';cell(e,reset,'FDRE')
    for pc in range(8):
        si=f's{pc+1:02d}'
        for ch in ('aw','w'):
            name=f'{n.SC}{si}_entry_pipeline/{si}_mmu/inst/{ch}_sreg/s_axi_{ch}valid'
            e.disabled[pc,ch]=input_row(e,name,constant+'/G')
        net=f'{n.ROOT}n{pc}_rvalid';gate=f'{n.ROOT}u_mem/synthetic_gate_{pc}'
        driver=f'{n.SC}{si}_entry_pipeline/{si}_mmu/inst/r_sreg/synthetic_rvalid_reg/Q'
        receiver=f'{n.ROOT}u_native{pc}/synthetic_receiver/I0'
        cell(e,driver.rsplit('/',1)[0],'FDRE');pin(e,driver,'OUT')
        cell(e,receiver.rsplit('/',1)[0],'LUT1');pin(e,receiver,'IN')
        cell(e,gate,'LUT2');e.objects['cell',gate]['INIT']="4'h2"
        for p,d in (('I0','IN'),('I1','IN'),('O','OUT')):pin(e,gate+'/'+p,d)
        e.cells[gate]=[gate+'/I0',gate+'/I1',gate+'/O']
        e.queries[('select','net',net)]=[net]
        leaves=[driver,receiver,gate+'/I0'];e.queries[('leaves',net)]=leaves;e.queries[('ports',net)]=[]
        e.networks['net',net]={'nets':[net],'leaves':leaves,'drivers':[driver],'ports':[]}
        label=('receive',str(pc),gate)
        for p,source in (('I0',driver),('I1',reset+'/Q')):
            e.edges[label,gate,gate+'/'+p]=input_row(e,gate+'/'+p,source)
        endpoints=n.required_endpoints(pc)
        for engine in (pc*2,pc*2+1):
            endpoints.update((f'{n.ROOT}u_board/adapter/engines[{engine}].queued_reg[{engine}][0]/CE',
                              f'{n.ROOT}u_board/adapter/engines[{engine}].wp_reg[{engine}][0]/CE',
                              f'{n.ROOT}u_board/adapter/engines[{engine}].tag_q_reg[0][slot][0]/CE'))
        endpoints=sorted(endpoints)
        for endpoint in endpoints:
            ref='RAMB18E2' if '.data_q_reg_7/' in endpoint else 'RAMB36E2' if '.data_q_reg_' in endpoint else 'FDRE'
            cell(e,endpoint.rsplit('/',1)[0],ref);pin(e,endpoint,'IN')
        e.receives[pc,gate,gate+'/O']=endpoints
        e.queries[('receive_endpoints',str(pc),gate+'/O')]=endpoints
    return e


def serialize(e, path, extra=(), incomplete=0, output_cap=0):
    rows=[nested(['header','schema','1','kind','c1_native_topology','checkpoint_sha256',DCP,'checkpoint','/bound/post_link.dcp','route_qualified','0','hardware_measured','0'])]
    for (kind,name),props in e.objects.items():
        for key,value in props.items():rows.append(nested(['property',kind,name,key,'status','observed','value',value]))
    for label,values in e.queries.items():rows.append(nested(['query',nested(label),'status','observed','total',len(values),'names',nested(values)]))
    for (kind,name),row in e.networks.items():rows.append(nested(['network',kind,name,*[v for k,values in row.items() for v in (k,nested(values))]]))
    for name,pins in e.cells.items():rows.append(nested(['cell_pins',name,'names',nested(pins)]))
    def input_text(row):
        return dictionary({k:dictionary(v) if isinstance(v,dict) else nested(v) if isinstance(v,list) else v for k,v in row.items()})
    for (pc,ch),row in e.disabled.items():rows.append(nested(['disabled_write',pc,'channel',ch,'network',input_text(row)]))
    for (label,owner,pinname),row in e.edges.items():rows.append(nested(['cone_edge',nested(label),'cell',owner,'input',pinname,'network',input_text(row)]))
    for (pc,owner,output),endpoints in e.receives.items():rows.append(nested(['receive_candidate',pc,'cell',owner,'output',output,'endpoints',nested(endpoints)]))
    rows.extend(extra)
    rows.append(nested(['summary','queries',len(e.queries),'incomplete_or_error',incomplete,'output_byte_cap',output_cap,'route_qualified',0,'hardware_measured',0]))
    path.write_text('\n'.join(rows)+'\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_raw_replay_never_qualifies(tmp_path):
    e=fixture();path=tmp_path/'raw.tcldata';sha=serialize(e,path)
    result=n.review_artifact(topology_path=path,topology_sha256=sha,checkpoint_sha256=DCP,checkpoint_path='/bound/post_link.dcp')
    assert result['issues']==[]
    assert len(result['canonical_facts']['disabled_write_inputs'])==16
    assert len(result['canonical_facts']['receive_gates'])==8
    assert result['route_qualified'] is False and result['hardware_measured'] is False
    assert 'current_routed_native_ready_equivalence' in result['pending_reviews']


@pytest.mark.parametrize('change',['missing','vcc','two_drivers','port','wrong_input','unbound_query'])
def test_disabled_mutations(change):
    e=fixture();row=e.disabled[0,'aw']
    if change=='missing':del e.disabled[0,'aw']
    elif change=='vcc':e.objects['cell',n.ROOT+'GND']['REF_NAME']='VCC'
    elif change=='two_drivers':row['drivers'].append('another/G')
    elif change=='port':row['ports']=['external']
    elif change=='wrong_input':row['input']='wrong/input'
    else:del e.queries['input_drivers',row['input'],row['nets'][0]]
    with pytest.raises(ValueError):n.disabled_review(e)


@pytest.mark.parametrize('change',['init','inversion','wrong_reset','wrong_pc','missing_ram','missing_tag','unknown_sink','wrong_reference','missing_query','extra_gate'])
def test_receive_mutations(change):
    e=fixture();gate=n.ROOT+'u_mem/synthetic_gate_0';key=(0,gate,gate+'/O');endpoint=e.receives[key][0]
    if change=='init':e.objects['cell',gate]['INIT']="4'h8"
    elif change=='inversion':e.objects['pin',gate+'/I1']['IS_INVERTED']='1'
    elif change=='wrong_reset':e.edges[('receive','0',gate),gate,gate+'/I1']['drivers']=[n.ROOT+'other/Q']
    elif change in ('wrong_pc','unknown_sink'):
        endpoint=n.ROOT+('u_native1/lr_reg[0]/CE' if change=='wrong_pc' else 'unreviewed/CE')
        pin(e,endpoint,'IN');cell(e,endpoint.rsplit('/',1)[0],'FDRE');e.receives[key].append(endpoint)
    elif change=='missing_ram':e.receives[key].remove(next(x for x in e.receives[key] if '.data_q_reg_0/' in x))
    elif change=='missing_tag':e.receives[key][:]=[x for x in e.receives[key] if 'engines[0].tag_q' not in x]
    elif change=='wrong_reference':e.objects['cell',endpoint.rsplit('/',1)[0]]['REF_NAME']='FDCE'
    elif change=='missing_query':del e.queries['receive_endpoints','0',gate+'/O']
    else:
        p=n.ROOT+'u_mem/extra/I0';pin(e,p,'IN');e.networks['net',n.ROOT+'n0_rvalid']['leaves'].append(p)
    with pytest.raises(ValueError):n.receive_review(e)


def test_unavailable_optional_property_is_retained(tmp_path):
    e=fixture();path=tmp_path/'raw';sha=serialize(e,path,['property cell WRAPPER/CL/GND ORIG_CELL_NAME status unavailable'])
    result=n.review_artifact(topology_path=path,topology_sha256=sha,checkpoint_sha256=DCP)
    assert not result['issues'] and ['cell','WRAPPER/CL/GND','ORIG_CELL_NAME'] in result['unavailable_properties']


@pytest.mark.parametrize('extra,count,cap',[('incomplete cap reason name_cap total 21 retained 20',1,0),('incomplete capture reason output_byte_cap limit 134217728',1,1),('property cell WRAPPER/CL/GND REF_NAME status observed value VCC',0,0),('unrecognized payload',0,0)])
def test_caps_conflicts_unknown_records_block_facts(tmp_path,extra,count,cap):
    path=tmp_path/'raw';sha=serialize(fixture(),path,[extra],count,cap)
    r=n.review_artifact(topology_path=path,topology_sha256=sha,checkpoint_sha256=DCP)
    assert r['issues'] and not r['canonical_facts'] and r['route_qualified'] is False


def test_hash_checkpoint_and_symlink_binding(tmp_path):
    path=tmp_path/'raw';sha=serialize(fixture(),path)
    with pytest.raises(ValueError):n.review_artifact(topology_path=path,topology_sha256='b'*64,checkpoint_sha256=DCP)
    r=n.review_artifact(topology_path=path,topology_sha256=sha,checkpoint_sha256='b'*64)
    assert r['issues'] and not r['canonical_facts']
    link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises(ValueError):n.review_artifact(topology_path=link,topology_sha256=sha,checkpoint_sha256=DCP)


def test_linear_multiline_framing_and_no_execution(tmp_path):
    path=tmp_path/'multiline';path.write_text('query safe status observed total 1 names {line1\n'+('x'*100+'\n')*10000+'line2}\n')
    rows=list(n.records(path));assert len(rows)==1 and 'line2' in rows[0][-1]
    assert n.tcl_list('{[exec touch /tmp/never]} {$env(HOME)}')==['[exec touch /tmp/never]','$env(HOME)']
    path.write_text('query {unclosed\n')
    with pytest.raises(ValueError):list(n.records(path))


def test_endpoint_family_matches_source_audit(tmp_path):
    audit=PATH.with_name('audit_c1_alex.tcl');lines=['source {'+str(audit)+'}']
    cases=[]
    for pc in range(8):
        endpoints=n.required_endpoints(pc)|{f'{n.ROOT}u_board/adapter/engines[{pc*2}].tag_q_reg[31][generation][31]/CE',f'{n.ROOT}u_native{pc}/lengths_reg_0_63_0_4/RAMA/WE'}
        for endpoint in sorted(endpoints):
            ref='RAMD64E' if 'lengths_reg' in endpoint else 'RAMB18E2' if '.data_q_reg_7/' in endpoint else 'RAMB36E2' if '.data_q_reg_' in endpoint else 'FDRE'
            cases.append(n.endpoint_family(pc,endpoint,ref));lines.append('puts [::c1_alex::receive_endpoint_class {'+n.ROOT+'} '+str(pc)+' {'+endpoint+'} '+ref+']')
    script=tmp_path/'compare.tcl';script.write_text('\n'.join(lines)+'\n')
    run=subprocess.run([shutil.which('tclsh'),str(script)],capture_output=True,text=True,check=True)
    assert run.stdout.splitlines()==cases
