from pathlib import Path
import pytest
from boards.c2_f2_alex import reset_status_review as m


def L(values): return ' '.join('{'+str(value)+'}' for value in values)
def D(values): return L(x for item in values.items() for x in item)
def R(*values): return L(values)+'\n'


def register(e, root, name, clock, driver):
    cell=root+'/'+name; pins={key:cell+'/'+key for key in ('D','Q','C')}
    e.objects['cell',cell]={'NAME':cell,'REF_NAME':'FDRE','IS_SEQUENTIAL':'1','ASYNC_REG':'TRUE','INIT':"1'b0"}
    e.queries['cell',cell,'pins']=L(pins.values())
    for ref,pin in pins.items():
        e.objects['pin',pin]={'NAME':pin,'REF_PIN_NAME':ref,'DIRECTION':'OUT' if ref=='Q' else 'IN','IS_CLOCK':'1' if ref=='C' else '0'}
    e.queries['pin',pins['C'],'clocks']=L([clock]);e.objects['clock',clock]={'NAME':clock,'PERIOD':'5.000'}
    e.objects.setdefault(('pin',driver), {'NAME':driver,'DIRECTION':'OUT','REF_PIN_NAME':driver.rsplit('/',1)[-1]})
    net=cell+'/data_net';e.groups[net]=[net]
    e.queries['pin',pins['D'],'nets']=L([net]);e.queries['net',net,'leaf_pins']=L([driver,pins['D']]);e.queries['net',net,'typed_output_pins']=L([driver])
    return cell,pins['Q']


def graph(width=2,depth=4):
    e=m.Evidence();root='WRAPPER/CL/u_status_cdc';allcells=[]
    for bit in range(width):
        source,q=register(e,root,f'source_{bit}','source',f'logic_{bit}/O');allcells.append(source)
        for stage in range(depth):
            cell,q=register(e,root,f'dest_{bit}_{stage}','dest',q);allcells.append(cell)
    e.queries['hierarchy_discovery',root]=L(allcells)
    return e,root


def test_exact_status_chains_use_edges_not_counts():
    e,root=graph();facts=e.chains(root,'source','dest',2,4)
    assert e.issues==[] and len(facts['chains'])==2
    assert all(len(row['stages'])==4 for row in facts['chains'])


def test_equal_counts_cross_bit_wire_rejected():
    e,root=graph();net=root+'/dest_1_0/data_net'
    e.queries['net',net,'typed_output_pins']=L([root+'/source_0/Q'])
    e.chains(root,'source','dest',2,4)
    assert any('branching' in x['reason'] or 'cross_bit' in x['reason'] for x in e.issues)


def test_async_reg_and_clock_changes_are_material():
    e,root=graph();e.objects['cell',root+'/dest_0_2']['ASYNC_REG']='FALSE'
    e.queries['pin',root+'/dest_1_1/C','clocks']=L(['source'])
    e.chains(root,'source','dest',2,4)
    assert {'status_missing_async_reg','status_register_clock_inventory'} <= {x['reason'] for x in e.issues}


def test_ambiguous_net_group_cannot_supply_connectivity():
    e,root=graph();name=root+'/dest_0_0/data_net';e.groups['duplicate']=[name]
    e.chains(root,'source','dest',2,4)
    assert any(x['reason']=='missing_or_ambiguous_pin_net' for x in e.issues)


def test_raw_property_conflicts_are_typed_and_preserved():
    e=m.Evidence()
    def record(kind,value):
        return ['object_properties','label','kind',kind,'name','shared','values',D({'NAME':D({'ok':1,'value':'shared'}),'CLASS':D({'ok':1,'value':value})}),'review','pending']
    e.add(record('pin','pin'));e.add(record('net','net'))
    assert e.issues==[]
    e.add(record('pin','net'))
    assert any(x['reason']=='conflicting_property' for x in e.issues)


def test_error_and_cap_are_not_erased_by_zero_summary():
    e=m.Evidence();e.add(['query','failed','status','error','message','unsupported','review','pending'])
    e.add(['inventory_coverage','scope','selected','501','cap','500'])
    e.add(['incomplete','scope','detail','cap','review','pending'])
    e.add(['topology_summary','queries','1','incomplete_or_error','0','route_qualified','0'])
    assert {'query_error','capture_cap','incomplete_capture'} <= {x['reason'] for x in e.issues}


def test_abbreviated_literal_and_unexpanded_serializer_rejected():
    e=m.Evidence();e.queries['inventory',]='a b ...'
    assert e.observed('inventory')==['a','b','...']
    e.add(['collection_serialization','inventory','actual_count','501','serialized_count','501','representation','implicit_string'])
    assert {'abbreviated_collection','unreviewed_collection_serialization'} <= {x['reason'] for x in e.issues}


def test_missing_capture_returns_unresolved():
    result=m.review_artifact(reports={},auxiliary={},checkpoint_sha256='a'*64)
    assert result['issues'] and result['route_qualified'] is False


def test_wrong_checkpoint_and_claimed_success_do_not_accept(tmp_path):
    p=tmp_path/'topology.tcldata'
    p.write_text(R('schema',1,'kind','routed_reset_status_diagnostic','checkpoint_sha256','b'*64,'route_qualified',0)+R('topology_summary','queries',0,'incomplete_or_error',0,'route_qualified',1))
    result=m.review_artifact(reports={},auxiliary={'reset_status':p},checkpoint_sha256='a'*64)
    reasons={x['reason'] for x in result['issues']}
    assert {'checkpoint_binding','unexpected_collector_qualification'} <= reasons
    assert result['route_qualified'] is False


def test_malformed_tcl_remains_unresolved_and_never_executed(tmp_path):
    p=tmp_path/'topology.tcldata';p.write_text('{unclosed [exec rm sentinel]')
    result=m.review_artifact(reports={},auxiliary={'reset_status':p},checkpoint_sha256='a'*64)
    assert any(x['reason']=='malformed_evidence' for x in result['issues'])


def test_symlink_capture_rejected(tmp_path):
    p=tmp_path/'data';p.write_text('data');q=tmp_path/'linked';q.symlink_to(p)
    with pytest.raises(ValueError,match='Linked'):m.review_artifact(reports={},auxiliary={'reset_status':q},checkpoint_sha256='a'*64)


def test_only_exact_clock_detail_caps_can_be_disposed():
    e=m.Evidence();pin=m.ROOTS[0]+'/dest_clk';label='pin '+pin+' nets';nets=['net'+str(i) for i in range(65)]
    e.objects['pin',pin]={'REF_PIN_NAME':'dest_clk','DIRECTION':'IN'}
    e.objects['clock','clk_out1_f2_core_clk']={'NAME':'clk_out1_f2_core_clk','PERIOD':'5.000'}
    e.queries['pin',pin,'clocks']=L(['clk_out1_f2_core_clk']);e.queries['pin',pin,'nets']=L(nets)
    e.add(['inventory_coverage',label,'selected','65','cap','64'])
    e.add(['incomplete',label,'detail','detail_cap_reached 64','review','pending'])
    facts=e.clock_cap_dispositions()
    assert e.issues==[] and facts[0]['net_segments']==65 and facts[0]['original_marker']
    e.add(['incomplete','pin WRAPPER/CL/data nets','detail','detail_cap_reached 64','review','pending'])
    e.clock_cap_dispositions()
    assert any(x['reason']=='incomplete_capture' for x in e.issues)


def pulse_file(tmp_path,mutation=None):
    pin=m.ROOTS[0]+'/r/C';rows=[]
    for check,corner in [('Min Period','n/a'),('Low Pulse Width','Slow'),('Low Pulse Width','Fast'),('High Pulse Width','Slow'),('High Pulse Width','Fast')]:
        rows.append([check,corner,'FDCE/C','n/a','0.5','5.0','4.5','SLICE_X1Y1',pin])
    if mutation:mutation(rows)
    path=tmp_path/'pulse.rpt';path.write_text('| Design State      : Routed\n|get_cells -quiet '+m.ROOTS[0]+'\nClock Name: clock\n'+'\n'.join('  '.join(row) for row in rows))
    return path,{'cell':{'pins':{'C':pin},'clock':{'name':'clock'}}}


def test_exact_positive_pulse_rows(tmp_path):
    path,regs=pulse_file(tmp_path);issues,facts=m.pulse_review(path,m.ROOTS[0],regs)
    assert issues==[] and len(facts['rows'])==5 and facts['reset_event_pulse_width_asserted'] is False


@pytest.mark.parametrize('mutation',[lambda rows:rows.pop(),lambda rows:rows[1].__setitem__(6,'-0.1'),lambda rows:rows.append(rows[0]),lambda rows:rows[0].__setitem__(8,'wrong/C')])
def test_pulse_missing_negative_duplicate_or_wrong_endpoint(tmp_path,mutation):
    path,regs=pulse_file(tmp_path,mutation);assert m.pulse_review(path,m.ROOTS[0],regs)[0]


def test_polarity_is_actual_and_bound_not_primitive_default(tmp_path):
    root=m.ROOTS[0];cell=root+'/r';pins={ref:cell+'/'+ref for ref in ('D','Q','C','CE','CLR')}
    props={'NAME':cell,'REF_NAME':'FDCE','INIT':"1'b1",'ASYNC_REG':'1','IS_C_INVERTED':"1'b0",'IS_CLR_INVERTED':"1'b1"}
    regs={root:{cell:{'properties':props,'pins':pins}}}
    text=R('schema',1,'kind','narrow_routed_structure','checkpoint_sha256','a'*64,'route_qualified',0)+R('polarity_inventory','u_rst_core','names',L([cell]))+R('polarity_cell',cell,'properties',D(props))
    for ref,pin in pins.items():text+=R('polarity_pin',pin,'properties',D({'NAME':pin,'REF_PIN_NAME':ref,'IS_INVERTED':'1' if ref=='CLR' else '0'}))
    text+=R('complete','cells',1,'route_qualified',0)
    p=tmp_path/'polarity';p.write_text(text)
    assert m.polarity_review(p,'a'*64,regs)[0]==[]
    p.write_text(text+R('polarity_inventory','unreviewed_scope','names','')+R('polarity_pin','extra/pin','properties',D({'NAME':'extra/pin','REF_PIN_NAME':'D'})))
    assert {'polarity_exact_scope_inventory','polarity_exact_pin_inventory'} <= {x['reason'] for x in m.polarity_review(p,'a'*64,regs)[0]}
    p.write_text(text.replace("IS_CLR_INVERTED} {1'b1", "IS_CLR_INVERTED} {1'b0"))
    assert any(x['reason']=='conflicting_cell_pin_inversion' for x in m.polarity_review(p,'a'*64,regs)[0])
    assert any(x['reason']=='polarity_checkpoint_binding' for x in m.polarity_review(p,'b'*64,regs)[0])


def test_effective_pin_polarity_closes_empty_cell_literals(tmp_path):
    root=m.ROOTS[0];cell=root+'/r';pins={ref:cell+'/'+ref for ref in ('D','Q','C','CE','CLR')}
    props={'NAME':cell,'REF_NAME':'FDCE','INIT':"1'b1",'ASYNC_REG':'1','IS_C_INVERTED':'','IS_CLR_INVERTED':''}
    regs={root:{cell:{'properties':props,'pins':pins}}}
    text=R('schema',1,'kind','narrow_routed_structure','checkpoint_sha256','a'*64,'route_qualified',0)+R('polarity_inventory','u_rst_core','names',L([cell]))+R('polarity_cell',cell,'properties',D(props))
    for ref,pin in pins.items():text+=R('polarity_pin',pin,'properties',D({'NAME':pin,'REF_PIN_NAME':ref,'IS_INVERTED':'1' if ref=='CLR' else '0'}))
    text+=R('complete','cells',1,'route_qualified',0);p=tmp_path/'polarity';p.write_text(text)
    issues,facts=m.polarity_review(p,'a'*64,regs)
    assert issues==[] and facts['effective_polarity'][cell]['CLR']['effective_witness']=='pin'
    assert facts['effective_polarity'][cell]['C']['cell_literal']==''
    p.write_text(text.replace('{IS_INVERTED} {1}', '{IS_INVERTED} {}'))
    assert any(x['reason']=='missing_or_wrong_primitive_inversion' for x in m.polarity_review(p,'a'*64,regs)[0])
