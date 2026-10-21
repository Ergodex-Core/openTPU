"""Conservative state-pair mutations; fixtures do not qualify physical artifacts."""
from copy import deepcopy
import hashlib
import pytest
from boards.c1_f2_alex import admission_pair_review as P
from test_c1_admission_review import fixture as raw_fixture, serialize


def fixture():
    channels = {}
    for pc in map(str, range(8)):
        root = 'WRAPPER/CL/u_native'+pc+'/'
        nodes = {}
        def cell(name, reference, kind, inputs=None, init=None, props=None, role='O'):
            output = name+'/'+role
            properties = dict(NAME=name,REF_NAME=reference)
            if init is not None: properties['INIT']=init
            properties.update(props or {})
            nodes[output] = dict(cell=name,reference=reference,kind=kind,properties=properties,
                output_properties=dict(NAME=output,REF_PIN_NAME=role,IS_INVERTED='0'),inputs={},unavailable_cell_properties=[])
            for r,source in (inputs or {}).items():
                nodes[output]['inputs'][name+'/'+r]=dict(driver=source,properties=dict(REF_PIN_NAME=r,IS_INVERTED='0',IS_CLOCK=str(int(r in ('C','CLK')))))
            return output
        clk=cell('WRAPPER/CL/clock','BUFGCE','input');nodes[clk]['roles']=['clk']
        rst=cell('WRAPPER/CL/reset','FDRE','input',role='Q');nodes[rst]['roles']=['rst']
        data=cell('WRAPPER/CL/data'+pc,'FDRE','input',role='Q');nodes[data]['roles']=['data']
        zero=cell(root+'zero','GND','constant',role='G');one=cell(root+'one','VCC','constant',role='P')
        q=cell(root+'state','FDRE','state',dict(C=clk,CE=one,D=data,R=rst),"1'b0",role='Q')
        inputs=dict(CLK=clk,I=q,WE=data)
        inputs.update({f'RADR{i}':q if i==0 else zero for i in range(5)})
        inputs.update({f'WADR{i}':data if i==0 else zero for i in range(5)})
        ram=cell(root+'memory','RAMD32','state',inputs,"32'h00000000",dict(IS_CLK_INVERTED="1'b0"))
        out=cell(root+'logic','LUT2','logic',dict(I0=ram,I1=q),"4'h6")
        interfaces={r:dict(driver=p,reference=nodes[p]['reference'],inversion='0',clocks=['core'] if r=='clk' else []) for r,p in [('clk',clk),('rst',rst),('data',data)]}
        channels[pc]=dict(nodes=nodes,interfaces=interfaces,root=out,registers=[root+'state'],memories=[root+'memory'])
    return dict(channels=channels)


def test_exact_state_control_memory_and_root_correspondence():
    f=fixture(); result=P.compare(f,deepcopy(f))
    assert len(result)==8 and all(r['registers']==r['memories']==1 for r in result)


def rename_logic(f):
    ch=f['channels']['0'];old=ch['root'];new=old.replace('/logic/','/promoted_logic/')
    node=ch['nodes'].pop(old);oldcell=node['cell'];newcell=new.rsplit('/',1)[0]
    node['cell']=newcell;node['properties']['NAME']=newcell;node['output_properties']['NAME']=new
    node['inputs']={pin.replace(oldcell+'/',newcell+'/'):edge for pin,edge in node['inputs'].items()}
    ch['nodes'][new]=node;ch['root']=new


def test_pure_combinational_cell_renaming_preserves_expression():
    f=fixture();g=deepcopy(f);rename_logic(g)
    assert P.compare(f,g)


@pytest.mark.parametrize('change',['reg_init','reg_d','reg_ce','reg_r','reg_c','missing_pin','missing_pin_inversion','missing_cell_init','reg_polarity_conflict','reg_output','ram_init','ram_data','ram_we','ram_read_address','ram_write_address','ram_clock','ram_clock_inversion','ram_missing_clock_config','ram_bad_output','lut_init','lut_missing_init','lut_pin_polarity','renamed_state','interface_alias','interface_polarity','interface_clock','async_cycle','missing_clock_marker','extra_clock_marker'])
def test_state_and_memory_mutations_reject(change):
    f=fixture();g=deepcopy(f);ch=g['channels']['0'];nodes=ch['nodes'];root='WRAPPER/CL/u_native0/'
    q=root+'state/Q';ram=root+'memory/O';logic=root+'logic/O';zero=root+'zero/G';one=root+'one/P'
    reg=nodes[q];mem=nodes[ram]
    if change=='reg_init':reg['properties']['INIT']="1'b1"
    elif change.startswith('reg_') and change[4:] in ('d','ce','r','c'):reg['inputs'][root+'state/'+change[4:].upper()]['driver']=zero
    elif change=='missing_pin':del reg['inputs'][root+'state/D']
    elif change=='missing_pin_inversion':del reg['inputs'][root+'state/D']['properties']['IS_INVERTED']
    elif change=='missing_cell_init':del reg['properties']['INIT']
    elif change=='reg_polarity_conflict':reg['properties']['IS_C_INVERTED']="1'b1"
    elif change=='reg_output':reg['output_properties']['IS_INVERTED']='1'
    elif change=='ram_init':mem['properties']['INIT']="32'h00000001"
    elif change=='ram_data':mem['inputs'][root+'memory/I']['driver']=zero
    elif change=='ram_we':mem['inputs'][root+'memory/WE']['driver']=zero
    elif change=='ram_read_address':mem['inputs'][root+'memory/RADR0']['driver']=zero
    elif change=='ram_write_address':mem['inputs'][root+'memory/WADR0']['driver']=zero
    elif change=='ram_clock':mem['inputs'][root+'memory/CLK']['driver']=zero
    elif change=='ram_clock_inversion':mem['properties']['IS_CLK_INVERTED']="1'b1";mem['inputs'][root+'memory/CLK']['properties']['IS_INVERTED']='1'
    elif change=='ram_missing_clock_config':del mem['properties']['IS_CLK_INVERTED']
    elif change=='ram_bad_output':mem['output_properties']['REF_PIN_NAME']='UNKNOWN'
    elif change=='lut_init':nodes[logic]['properties']['INIT']="4'h8"
    elif change=='lut_missing_init':del nodes[logic]['properties']['INIT']
    elif change=='lut_pin_polarity':nodes[logic]['inputs'][root+'logic/I0']['properties']['IS_INVERTED']='1'
    elif change=='renamed_state':reg['cell']=root+'renamed_state'
    elif change=='interface_alias':ch['interfaces']['data']['driver']=ch['interfaces']['rst']['driver']
    elif change=='interface_polarity':ch['interfaces']['data']['inversion']='1'
    elif change=='interface_clock':ch['interfaces']['clk']['clocks']=['other']
    elif change=='async_cycle':mem['inputs'][root+'memory/RADR0']['driver']=logic
    elif change=='missing_clock_marker':reg['inputs'][root+'state/C']['properties']['IS_CLOCK']='0'
    elif change=='extra_clock_marker':mem['inputs'][root+'memory/I']['properties']['IS_CLOCK']='1'
    with pytest.raises((ValueError,TypeError,KeyError)):P.compare(f,g)


def test_read_symbol_depends_on_actual_address_not_only_memory_name():
    f=fixture();g=deepcopy(f);p='WRAPPER/CL/u_native0/memory/O';n=g['channels']['0']['nodes'][p]
    n['inputs'][n['cell']+'/RADR0']['driver']='WRAPPER/CL/u_native0/zero/G'
    assert P.Circuit(f['channels']['0']).value(p)!=P.Circuit(g['channels']['0']).value(p)


def test_empty_cell_fdre_polarity_requires_observed_effective_pin():
    f=fixture();g=deepcopy(f);reg=g['channels']['0']['nodes']['WRAPPER/CL/u_native0/state/Q']
    reg['properties'].update(IS_C_INVERTED='',IS_D_INVERTED='',IS_R_INVERTED='')
    assert P.compare(f,g)
    del reg['inputs'][reg['cell']+'/C']['properties']['IS_INVERTED']
    with pytest.raises(ValueError):P.compare(f,g)


def test_same_nontrivial_lut_rewrite_is_conservatively_rejected():
    f=fixture();g=deepcopy(f);n=g['channels']['0']['nodes']['WRAPPER/CL/u_native0/logic/O']
    # XOR inputs commute, but this checker deliberately has no Boolean fallback.
    a,b=n['inputs'].values();a['driver'],b['driver']=b['driver'],a['driver']
    with pytest.raises(ValueError,match='root expression'):P.compare(f,g)


def test_public_pair_replays_raw_graphs_and_leaves_source_and_route_pending(tmp_path):
    e=raw_fixture()
    for pc in range(8):e.objects['pin',f'WRAPPER/CL/u_native{pc}/state/C']['IS_CLOCK']='1'
    p=tmp_path/'capture';p.write_text(serialize(e))
    binding=dict(topology_path=p,topology_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),checkpoint_sha256='a'*64,helper_sha256='b'*64,native_helper_sha256='c'*64)
    same=P.review_pair(linked=binding,routed=binding)
    assert not same['paired_preservation_proven'] and 'distinct' in same['issues'][0]['reason']
    q=tmp_path/'routed';q.write_text(p.read_text().replace('a'*64,'d'*64))
    routed={**binding,'topology_path':q,'topology_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'checkpoint_sha256':'d'*64}
    r=P.review_pair(linked=binding,routed=routed)
    assert r['paired_preservation_proven'] and not r['issues']
    assert r['route_qualified'] is r['hardware_measured'] is r['source_equation_proven'] is False
    assert r['pending_reviews']==['take_read_admission_semantics','linked_final_accounting_preservation']
    changed={**routed,'topology_sha256':'e'*64}
    r=P.review_pair(linked=binding,routed=changed)
    assert not r['paired_preservation_proven'] and r['issues']
    assert len(r['pending_reviews'])==3
