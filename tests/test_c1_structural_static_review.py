"""Independent topology fixtures and mutations for the static replay."""
from copy import deepcopy
import pytest
from boards.c1_f2_alex import structural_static_review as R


def word(value):
    return '{' + str(value) + '}'


def seq(values):
    return ' '.join(map(word, values))


def data(values):
    return seq([x for pair in values.items() for x in pair])


def fixture():
    root = 'WRAPPER/CL/'
    hierarchy = {root[:-1]: 'cl_opentpu', root+'u_board': 'otpu_c1_hbm_kernel',
                 root+'u_board/core': 'otpu_c1_top'}
    hierarchy.update({root+f'u_native{i}': 'otpu_native_axi' for i in range(8)})
    hierarchy.update({root+k: v for k, v in zip(
        ['u_mem', 'u_hbm', 'u_ocl_cdc', 'u_pcis_pipe', 'u_core_clk', 'u_hbm_clk'],
        ['f2_mem_wrapper', 'cl_mem_hbm_wrapper', 'f2_ocl_cdc', 'f2_pcis_pipe', 'f2_core_clk', 'f2_hbm_clk'])})
    general = [['hierarchy_cell', name, 'expected_reference', ref, 'selected', seq([name]),
                'properties', data({'NAME':name,'REF_NAME':ref,'ORIG_REF_NAME':'','IS_BLACKBOX':'0'})]
               for name, ref in hierarchy.items()]
    for label, names in {'native':[root+f'u_native{i}' for i in range(8)], 'blackboxes':['WRAPPER/SHELL/ip'], 'tmem':[]}.items():
        general.append(['cell_inventory', label, 'count', str(len(names)), 'names', seq(names)])
    sc = root+'u_mem/f2_mem_i/memory_sc'
    general.extend([
        ['memory_interface_pins', seq([root+'u_mem/clk',root+'u_mem/resetn'])],
        ['smartconnect_properties','selected',seq([sc]),'properties',data({'NAME':sc,'REF_NAME':'generated_sc','IS_BLACKBOX':'0'})],
        ['smartconnect',sc,'reference','generated_sc','slave_interfaces','9','master_interfaces','8','address_map_validation','executor_generated_BD_required'],
        ['unresolved_shell_objects',seq(['WRAPPER/SHELL/ip'])]])
    members = {'host':[root+'u_pcis_pipe/register',root+'u_mem/f2_mem_i/memory_sc/inst/s00_entry_pipeline/register'],
               'core':[root+'u_board/reg']+[root+f'u_native{i}/reg' for i in range(8)]+[
                   root+f'u_mem/f2_mem_i/memory_sc/inst/s{i:02d}_entry_pipeline/reg' for i in range(1,9)],
               'hbm':[root+'u_hbm/reg']+[root+f'u_mem/f2_mem_i/memory_sc/inst/m{i:02d}_exit_pipeline/reg' for i in range(8)],
               'reference':[], 'unused':[]}
    pins = {'host':'clk_main_a0','core':'u_core_clk/clk_out1','hbm':'u_hbm_clk/clk_out1','reference':'clk_hbm_ref','unused':'u_core_clk/clk_out2'}
    periods = {'host':'4.000','core':'5.000','hbm':'2.222222222222','reference':'10.000','unused':'2.500'}
    clocks = []; unused = []
    for name in members:
        prop = {'NAME':name,'PERIOD':periods[name],'MASTER_CLOCK':'','SOURCE_PINS':seq([root+pins[name]]),
                'IS_GENERATED':'1','IS_PROPAGATED':'1','IS_VIRTUAL':'0'}
        if name != 'unused':
            clocks += [['clock_anchor',name,'pin',root+pins[name],'selected',seq([root+pins[name]]),'clocks',seq([name])],
                       ['clock',name,data(prop)]]
        if name in ('host','core','hbm'): clocks.append(['clock_registers',name,'count',str(len(members[name]))])
        clocks.append(['register_inventory',name,'properties',data({k:seq(['observed',v]) for k,v in prop.items()}),
                       'count',str(len(members[name])),'chunk_size','256','chunks',str(int(bool(members[name])))])
        if members[name]: clocks.append(['register_chunk',name,'offset','0','names',seq(members[name])])
        owners = {'host':{'u_pcis_pipe':1,'u_mem':1},'core':{'u_board':1,'u_mem':8,**{f'u_native{i}':1 for i in range(8)}},
                  'hbm':{'u_hbm':1,'u_mem':8},'reference':{},'unused':{}}[name]
        clocks.append(['fabric_clock',name,'label',name if name in ('host','core','hbm') else 'other',
                       'registers',str(len(members[name])),'owner_counts',data(owners)])
        unused.append(['clock_source_inventory',name,'sources',seq([root+pins[name]])])
    clocks.insert(0,['register_clock_inventory','count','5','names',seq(members)])
    ports = {**{f's{i:02d}_entry_pipeline':1 for i in range(9)},**{f'm{i:02d}_exit_pipeline':1 for i in range(8)}}
    clocks.append(['fabric_port_clock_consumers',data(ports)])
    unused.append(['unused_clock','unused','period','2.500','consumers','0','registers',''])
    return {'steps':[{'label':[name],'records':records,'result':{'state':'observed'}}
                     for name,records in [('general',general),('clocks',clocks),('unused_clocks',unused)]]}


def row(facts, step, tag, name=None):
    return next(r for s in facts['steps'] if s['label'] == [step] for r in s['records']
                if r[0] == tag and (name is None or r[1] == name))


def field(record, key, value):
    record[record.index(key)+1] = value


def test_exact_source_topology_and_consumers_prove_only_three_static_steps():
    result = R.review(fixture())
    assert result['proven_static_steps'] == ['general','clocks','unused_clocks'], result['issues']
    assert not result['pending_static_steps'] and not result['issues']
    assert not result['route_qualified'] and not result['hardware_measured']
    assert result['facts']['clocks']['port_consumers']['s08_entry_pipeline'] == 1


@pytest.mark.parametrize('mutation', ['reference','blackbox','missing_cell','native_extra','tmem','cl_blackbox','opaque_mismatch','memory_escape',
    'sc_blackbox','sc_wrong_name','sc_missing','sc_ports','clock_period','clock_anchor','clock_duplicate_role',
    'missing_register','duplicate_register','wrong_native_clock','wrong_si_clock','missing_si','owner_summary','port_summary',
    'missing_source','source_changed','unused_consumer','unused_missing','unused_duplicate','error_flag','empty_observed'])
def test_changed_topology_clock_or_missing_raw_evidence_remains_pending(mutation):
    f = fixture()
    g = row(f,'general','hierarchy_cell','WRAPPER/CL/u_native0')
    if mutation == 'reference': field(g,'properties',g[-1].replace('otpu_native_axi','wrong'))
    elif mutation == 'blackbox': field(g,'properties',g[-1].replace('{IS_BLACKBOX} {0}','{IS_BLACKBOX} {1}'))
    elif mutation == 'missing_cell': f['steps'][0]['records'].remove(g)
    elif mutation == 'native_extra': field(row(f,'general','cell_inventory','native'),'names',seq(['WRAPPER/CL/u_native9']))
    elif mutation == 'tmem':
        r=row(f,'general','cell_inventory','tmem');field(r,'count','1');field(r,'names','WRAPPER/CL/tmem')
    elif mutation == 'cl_blackbox': field(row(f,'general','cell_inventory','blackboxes'),'names','WRAPPER/CL/ip')
    elif mutation == 'opaque_mismatch': row(f,'general','unresolved_shell_objects')[1]='WRAPPER/SHELL/other'
    elif mutation == 'memory_escape': row(f,'general','memory_interface_pins')[1]='WRAPPER/CL/u_other/pin'
    elif mutation == 'sc_blackbox':
        r=row(f,'general','smartconnect_properties');field(r,'properties',r[-1].replace('{IS_BLACKBOX} {0}','{IS_BLACKBOX} {1}'))
    elif mutation == 'sc_wrong_name': field(row(f,'general','smartconnect_properties'),'selected','wrong')
    elif mutation == 'sc_missing': f['steps'][0]['records'].remove(row(f,'general','smartconnect_properties'))
    elif mutation == 'sc_ports': field(row(f,'general','smartconnect'),'slave_interfaces','8')
    elif mutation == 'clock_period': row(f,'clocks','clock','core')[2]=row(f,'clocks','clock','core')[2].replace('5.000','4.000')
    elif mutation == 'clock_anchor': field(row(f,'clocks','clock_anchor','core'),'clocks','host')
    elif mutation == 'clock_duplicate_role': row(f,'clocks','clock','core')[2]=row(f,'clocks','clock','core')[2].replace('{NAME} {core}','{NAME} {host}')
    elif mutation == 'missing_register': f['steps'][1]['records'].remove(row(f,'clocks','register_chunk','core'))
    elif mutation == 'duplicate_register':
        r=row(f,'clocks','register_chunk','core');field(r,'names',r[-1]+' {WRAPPER/CL/u_board/reg}')
    elif mutation in ('wrong_native_clock','wrong_si_clock','missing_si'):
        r=row(f,'clocks','register_chunk','host')
        replacement={'wrong_native_clock':'WRAPPER/CL/u_native0/reg','wrong_si_clock':'WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/s01_entry_pipeline/reg','missing_si':'WRAPPER/CL/u_mem/other/reg'}[mutation]
        field(r,'names',seq(['WRAPPER/CL/u_pcis_pipe/register',replacement]))
    elif mutation == 'owner_summary': field(row(f,'clocks','fabric_clock','core'),'owner_counts','u_mem 99')
    elif mutation == 'port_summary': row(f,'clocks','fabric_port_clock_consumers')[1]='s00_entry_pipeline 99'
    elif mutation == 'missing_source': f['steps'][2]['records'].remove(row(f,'unused_clocks','clock_source_inventory','unused'))
    elif mutation == 'source_changed': field(row(f,'unused_clocks','clock_source_inventory','unused'),'sources','somewhere/else')
    elif mutation == 'unused_consumer': field(row(f,'unused_clocks','unused_clock'),'consumers','1')
    elif mutation == 'unused_missing': f['steps'][2]['records'].remove(row(f,'unused_clocks','unused_clock'))
    elif mutation == 'unused_duplicate': f['steps'][2]['records'].append(deepcopy(row(f,'unused_clocks','unused_clock')))
    elif mutation == 'error_flag': f['steps'][0]['result']['state']='error'
    elif mutation == 'empty_observed': f['steps'][0]['records']=[]
    else: raise AssertionError(mutation)
    result=R.review(f)
    assert result['pending_static_steps'] and result['issues'], mutation
    assert not result['route_qualified']


def test_missing_full_clock_inventory_cannot_be_replaced_by_success_flags():
    f=fixture();f['steps'][1]['records']=[]
    result=R.review(f)
    assert result['proven_static_steps']==['general']
    assert result['pending_static_steps']==['clocks','unused_clocks']


def test_no_second_core_clock_is_valid_when_complete_source_inventory_says_so():
    f=fixture();r=row(f,'clocks','register_inventory','unused')
    field(r,'properties',r[r.index('properties')+1].replace('u_core_clk/clk_out2','u_hbm_clk/clk_out2'))
    field(row(f,'unused_clocks','clock_source_inventory','unused'),'sources','WRAPPER/CL/u_hbm_clk/clk_out2')
    f['steps'][2]['records'].remove(row(f,'unused_clocks','unused_clock'))
    result=R.review(f)
    assert not result['issues'], result['issues']
    assert result['facts']['unused_clocks']['unused_clock_count']==0
