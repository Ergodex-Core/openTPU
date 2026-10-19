import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('gray_review', Path(__file__).resolve().parents[1] / 'boards/c2_f2_alex/gray_review.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def example():
    e = m.Evidence(); root = 'WRAPPER/CL/u_mem/f2_mem_i/memory_sc/inst/gen_xpm_memory_fifo.inst_fifo/cdc'
    base = ('gray', root, 'bit', '0'); src = root+'/src_gray_ff_reg[0]'; dst = root+'/dest_graysync_ff_reg[0][0]'
    e.headers['ancestors'] = {'checkpoint_sha256': 'a'*64}
    e.coverage['ancestors'] = {'selected': '3', 'attempted': '3', 'skipped': '0'}
    e.groups[root] = {'source': '0 '+src, 'destination': '0 '+dst}
    e.relations[root] = dict(e.groups[root], source_cells=e.groups[root]['source'], destination_cells=e.groups[root]['destination'], fifo_candidate=root.rsplit('/cdc',1)[0], smartconnect_candidate='WRAPPER/CL/u_mem/f2_mem_i/memory_sc')
    for name, role, ref in ((root,'gray_cdc','xpm_cdc_gray'), (root.rsplit('/cdc',1)[0],'xpm_fifo','sc_util_v1_0_4_xpm_memory_fifo'), ('WRAPPER/CL/u_mem/f2_mem_i/memory_sc','smartconnect','f2_mem_memory_sc_0')):
        e.props[('gray_ancestor',name,'roles',role)] = {'NAME':name,'REF_NAME':ref,'ORIG_REF_NAME':ref,'IS_BLACKBOX':'0'}
    def pin(key, name, clock=None):
        e.props[key+('pin', name)] = {'NAME':name,'REF_PIN_NAME':name[-1],'DIRECTION':'OUT' if name.endswith('/Q') else 'IN','IS_CLOCK':'1' if clock else '0'}
        e.queries[key+('owning_cells',)] = '{'+name[:-2]+'}'
        if clock:
            e.queries[key+('associated_clocks',)] = clock
            e.props[key+('clock',)] = {'NAME':clock,'PERIOD':'5.000' if clock=='source' else '4.000'}
    pin(base+('source',),src+'/Q'); pin(base+('source_clock',),src+'/C','source'); pin(base+('first_destination',),dst+'/D')
    e.props[base+('source','cell',src)] = {'NAME':src,'IS_SEQUENTIAL':'1','ASYNC_REG':'0'}
    stages = {}
    for i in range(3):
        cell = root+f'/dest_graysync_ff_reg[{i}][0]'; stages[str(i)] = cell
        for kind, ending in (('data','D'),('output','Q'),('clock','C')):
            pin(base+('stage',str(i),kind),cell+'/'+ending,'dest' if kind=='clock' else None)
        e.props[base+('stage',str(i),'data','cell',cell)] = {'NAME':cell,'IS_SEQUENTIAL':'1','ASYNC_REG':'true'}
        prev = src if i==0 else stages[str(i-1)]
        e.drivers[base+('stage',str(i))] = {'drivers':'{'+prev+'/Q}', 'driver_count':'1'}
    e.drivers[base+('first_destination',)] = e.drivers[base+('stage','0')].copy()
    e.queries[('gray_destination_stage_inventory',)] = ' '.join('{'+v+'}' for v in stages.values())
    e.stages[base] = {'stages':' '.join(k+' {'+v+'}' for k,v in stages.items())}
    for ignored in ('0','1'):
        e.timing[base+(ignored,)] = {'returned':'4' if ignored=='0' else '0','capped':'0','max_paths':'16','nworst':'16','expected_source':src+'/Q','expected_source_clock_pin':src+'/C','expected_destination':dst+'/D'}
    for i in range(4):
        e.props[base+('path','user_ignored','0','index',str(i))] = {'STARTPOINT_PIN':src+'/C','ENDPOINT_PIN':dst+'/D','STARTPOINT_CLOCK':'source','ENDPOINT_CLOCK':'dest','REQUIREMENT':'5.000','EXCEPTION':'MaxDelay Path 5.000ns -datapath_only','SLACK':'4.5','DATAPATH_DELAY':'0.5'}
    ex = {root:{'set_max_delay':{'value':'5.000'},'set_bus_skew':{'value':'4.000'}}}
    return e,ex,base


def test_valid_current_clock_pin_relation():
    e,ex,_=example(); r=e.review(ex,'a'*64)
    assert r['gray_evidence_consistent'],r
    assert r['normal_path_count_histogram']=={4:1}
    assert not r['route_qualified'] and not r['hardware_measured']


@pytest.mark.parametrize('mutate,reason',[
    (lambda e,x,b:e.props[b+('path','user_ignored','0','index','0')].update(STARTPOINT_PIN='wrong/C'),'path_endpoint_relation'),
    (lambda e,x,b:e.props[b+('source_clock','clock')].update(PERIOD='6.000'),'vendor_export_bound'),
    (lambda e,x,b:e.drivers[b+('stage','1')].update(drivers='{wrong/Q}'),'data_driver_mismatch'),
    (lambda e,x,b:e.timing[b+('0',)].update(capped='1'),'path_count_or_cap'),
    (lambda e,x,b:e.queries.update({b+('source','owning_cells',):'wrong'}),'data_pin_owner'),
    (lambda e,x,b:e.props[b+('path','user_ignored','0','index','0')].update(EXCEPTION='False Path'),'effective_max_delay'),
    (lambda e,x,b:e.props.pop(b+('path','user_ignored','0','index','0')),'path_count_or_cap'),
    (lambda e,x,b:e.props[b+('stage','1','clock','clock')].update(NAME='other'),'stage_clock_mismatch'),
    (lambda e,x,b:e.stages[b].update(stages=e.stages[b]['stages'].split(' 2 ')[0]),'global_stage_cell_inventory'),
])
def test_corruption_stays_unresolved(mutate,reason):
    e,ex,b=example(); mutate(e,ex,b); r=e.review(ex,'a'*64)
    assert not r['gray_evidence_consistent'] and reason in r['unresolved_reasons']


def test_stream_multiline_once_and_braces(tmp_path,monkeypatch):
    path=tmp_path/'input'; path.write_text('query {pulse_report all} status observed value {line\n'+('x'*100+'\n')*10000+'last} review pending\nquery x status observed value y review pending\n')
    count=0; original=m.tcl_list
    def called(s):
        nonlocal count
        count+=1; return original(s)
    monkeypatch.setattr(m,'tcl_list',called)
    rows=list(m.records(path))
    assert len(rows)==2 and count==2 and rows[0][1]=='pulse_report all'


@pytest.mark.parametrize('raw',['query {unterminated\n','query "unterminated\n','query {x}oops\n'])
def test_malformed_records_fail(tmp_path,raw):
    path=tmp_path/'input'; path.write_text(raw)
    with pytest.raises(ValueError): list(m.records(path))


def test_duplicate_dict_rejected():
    with pytest.raises(ValueError): m.td('a 1 a 2')


def test_label_flatten():
    assert m.label('{{gray root bit 0} stage 1 clock} clock')==('gray','root','bit','0','stage','1','clock','clock')


def test_export_source_and_selectors(tmp_path):
    p=tmp_path/'export'; p.write_text("# Constraints from file : 'xpm_cdc_gray.tcl'\ncurrent_instance -quiet\ncurrent_instance {root}\nset_max_delay -datapath_only -from [get_cells src_gray_ff_reg*] -to [get_cells {dest_graysync_ff_reg[0]*}] 5.000\nset_bus_skew -from [get_cells src_gray_ff_reg*] -to [get_cells {dest_graysync_ff_reg[0]*}] 4.000\n")
    assert m.export_gray(p)['root']['set_max_delay']['value']=='5.000'
    p.write_text(p.read_text().replace('dest_graysync_ff_reg[0]*','dest_graysync_ff_reg*'))
    with pytest.raises(ValueError): m.export_gray(p)


def test_hash_mismatch_and_symlink(tmp_path):
    p=tmp_path/'x'; p.write_text('data'); link=tmp_path/'link'; link.symlink_to(p)
    with pytest.raises(ValueError): m.file_identity(p,'0'*64)
    with pytest.raises(ValueError): m.file_identity(link,'0'*64)


def test_rounding_is_display_bound():
    assert m.rounded_match('2.222',m.Decimal('2.2222'))
    assert not m.rounded_match('2.222',m.Decimal('2.223'))


def test_integrated_records_roundtrip(tmp_path):
    e,ex,base=example()
    def tl(words): return ' '.join('{'+str(x)+'}' for x in words)
    def dicttext(d): return tl([x for kv in d.items() for x in kv])
    supp=[]; ancestors=[]
    for key,value in e.props.items():
        row=tl(['object_properties',tl(key),dicttext({k:dicttext({'ok':'1','value':v}) for k,v in value.items()}),'review','pending'])
        (ancestors if key[0]=='gray_ancestor' else supp).append(row)
    for key,value in e.queries.items(): supp.append(tl(['query',tl(key),'status','observed','value',value,'review','pending']))
    for tag,items in [('data_driver',e.drivers),('gray_stage_inventory',e.stages),('gray_bit_timing',e.timing)]:
        for key,value in items.items():
            value=value.copy()
            if tag=='gray_bit_timing': value['user_ignored']=key[-1]; key=key[:-1]
            supp.append(tl([tag,tl(key),*[x for kv in value.items() for x in kv]]))
    for root,group in e.groups.items(): ancestors.append(tl(['vendor_gray_bus',root,dicttext(group),'review','pending']))
    for root,rel in e.relations.items(): ancestors.append(tl(['gray_ancestor_relation',root,*[x for kv in rel.items() for x in kv]]))
    ancestors.insert(0,tl(['schema','1','checkpoint_sha256','a'*64]))
    ancestors.append(tl(['gray_ancestor_coverage',*[x for kv in e.coverage['ancestors'].items() for x in kv]]))
    ancestors.append(tl(['diagnostic_summary',dicttext({'queries':'0','incomplete_or_error':'0'})]))
    supp.append(tl(['supplemental_summary',dicttext({'queries':str(len(e.queries)),'incomplete_or_error':'0'})]))
    s=tmp_path/'supp'; a=tmp_path/'ancestors'; s.write_text('\n'.join(supp)+'\n'); a.write_text('\n'.join(ancestors)+'\n')
    loaded=m.Evidence();loaded.read(s,'supplemental');loaded.read(a,'ancestors')
    result=loaded.review(ex,'a'*64)
    assert result['gray_evidence_consistent'],result
    assert result['canonical_gray_sha256']==e.review(ex,'a'*64)['canonical_gray_sha256']


def rows(root):
    return {'1':{'from':'{'+root+'/src_gray_ff_reg[0]}','to':'{'+root+'/dest_graysync_ff_reg[0][0]}','requirement':'4.000','actual':'0.125','slack':'3.875','corner':'Slow','position':'1'}}

def test_final_gray_positive_evidence():
    e,ex,b=example();result=m.final_gray_checks(e,rows(b[1]));assert not result['issues'] and result['matched_gray_bus_skew_rows']==1

@pytest.mark.parametrize('change,reason',[
    ({'slack':'-0.001'},'final_bus_skew_physical_failure'),
    ({'requirement':'5.000'},'final_bus_skew_vendor_bound'),
    ({'to':'{wrong/D}'},'final_bus_skew_exact_coverage')])
def test_final_skew_changes_fail(change,reason):
    e,ex,b=example();r=rows(b[1]);r['1'].update(change);result=m.final_gray_checks(e,r);assert reason in [v[0] for v in result['issues']]

def test_negative_path_not_hidden_by_global_positive_summary():
    e,ex,b=example();e.props[b+('path','user_ignored','0','index','2')]['SLACK']='-0.001'
    result=m.final_gray_checks(e,rows(b[1]));assert result['negative_path_count']==1

def test_objects_binding_and_embedded_ancestors(tmp_path):
    e,ex,b=example();e.coverage['supplemental']=e.coverage.pop('ancestors');e.groups={}
    p=tmp_path/'objects';p.write_text('checkpoint /route.dcp sha256 '+('a'*64)+'\nvendor_gray_bus root {source {0 source} destination {0 destination}} review pending\nbus_skew 1 {from source to destination requirement 4.000 actual 0.100 slack 3.900} review pending\n')
    binding,r=m.read_objects(e,p,'a'*64);assert e.groups and e.coverage['ancestors']['skipped']=='0'
    with pytest.raises(ValueError):m.read_objects(e,p,'b'*64)


def test_final_gray_signed_actual_skew_is_met():
    e,ex,b=example();r=rows(b[1]);r['1'].update(actual='-0.008',slack='4.008')
    assert not m.final_gray_checks(e,r)['issues']

@pytest.mark.parametrize('change,reason',[
    ({'actual':'-0.008','slack':'4.000'},'final_bus_skew_slack_inconsistent'),
    ({'actual':'4.008','slack':'0.008'},'final_bus_skew_physical_failure'),
    ({'actual':'-0.008','slack':'-0.001'},'final_bus_skew_physical_failure'),
])
def test_signed_skew_does_not_hide_failure_or_bad_arithmetic(change,reason):
    e,ex,b=example();r=rows(b[1]);r['1'].update(change)
    assert reason in [issue[0] for issue in m.final_gray_checks(e,r)['issues']]

@pytest.mark.parametrize('field',['actual','slack','requirement'])
@pytest.mark.parametrize('value',['NaN','Infinity','-Infinity'])
def test_final_gray_nonfinite_skew_rejected(field,value):
    e,ex,b=example();r=rows(b[1]);r['1'][field]=value
    with pytest.raises(ValueError,match='Nonfinite'):m.final_gray_checks(e,r)
