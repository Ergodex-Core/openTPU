import copy
import importlib.util
from pathlib import Path
import pytest

from boards.c1_f2_alex import exception_review as m


def example():
    old=dict(id=100,source='xpm_memory_xdc.tcl',scope='fifo/gen_mem_rep[0].inst_xpm_memory',kind='set_max_delay',datapath_only=True,
             value='5.000',**{'from':'same scoped clka endpoints'},vendor_clka_leaf_source=True,vendor_lutram_through=True)
    new=dict(old,id=101,vendor_lutram_through=False,vendor_doutb_destination=True)
    a={'Position':100,'Type':'Max Delay Datapath Only','Setup':'max_dpo=5','Endpoints':132,'Through':'320 pins'}
    b=dict(a,Position=101,Through='')
    return old,new,a,b


def test_vendor_override_exact():
    old,new,a,b=example()
    assert m.validate_memory_override(old,new,a,b,'Totally overridden path by DPO 101','5.000')==[]


@pytest.mark.parametrize('mutation,reason',[
    (lambda o,n,a,b:n.update(scope='different'),'same_scope'),
    (lambda o,n,a,b:n.update(value='10.000'),'same_expected_clock_bound'),
    (lambda o,n,a,b:n.update(datapath_only=False),'datapath_only'),
    (lambda o,n,a,b:n.update(vendor_doutb_destination=False),'vendor_replacement_selectors'),
    (lambda o,n,a,b:n.update({'from':'wider set'}),'exact_clock_leaf_source'),
    (lambda o,n,a,b:b.update(Endpoints=131),'nonempty_same_covered_endpoint_count'),
    (lambda o,n,a,b:b.update(Through='1 pin'),'replacement_has_no_unreviewed_through'),
    (lambda o,n,a,b:n.update(source='user.xdc'),'vendor_source'),
])
def test_reject_weakened_override(mutation,reason):
    o,n,a,b=example();mutation(o,n,a,b)
    assert reason in m.validate_memory_override(o,n,a,b,'Totally overridden path by DPO 101','5')


def test_wrong_override_id_and_missing_clock():
    o,n,a,b=example()
    assert 'exact_effective_override' in m.validate_memory_override(o,n,a,b,'Totally overridden path by DPO 102','5')
    assert 'invalid_bound' in m.validate_memory_override(o,n,a,b,'Totally overridden path by DPO 101',None)


def test_missing_header_rejected():
    with pytest.raises(ValueError):m.parse_ignored('No table')


def table(headings,rows):
    widths=[max(len(str(row[i])) for row in [headings,*rows])+2 for i in range(len(headings))]
    def line(row):return ''.join(str(value).ljust(width) for value,width in zip(row,widths)).rstrip()
    return line(headings)+'\n'+line(['-'*(w-2) for w in widths])+'\n'+'\n'.join(line(row) for row in rows)+'\n'


def reports():
    ignored=table(['Position','From','Through','To','Setup','Hold','Status'],
                  [[100,'*','*','*','max_dpo=5','-','Totally overridden path by DPO 101']])
    coverage=table(['Position','Type','Setup','Hold','From','Through','To','Endpoints','From (%)','Through (%)','To (%)'],
                   [[100,'Max Delay Datapath Only','max_dpo=5','-','160 pins','320 pins','',132,'100.00','50.00',''],
                    [101,'Max Delay Datapath Only','max_dpo=5','-','160 pins','','132 cells',132,'82.50','','33.33']])
    return ignored,coverage


def export():
    fifo='WRAPPER/CL/u_mem/vendor/gen_xpm_memory_fifo.inst_fifo'
    text=f"# Constraints from file : 'xpm_memory_xdc.tcl'\ncurrent_instance {{{fifo}/gen_mem_rep[0].inst_xpm_memory}}\n# id:100\n"
    text+='set _xlnx_shared_i0 '+m.MEMORY_FROM+'\n'
    text+='set_max_delay -datapath_only -from $_xlnx_shared_i0 -through '+m.MEMORY_THROUGH+' 5.000\n# id:101\n'
    text+='set_max_delay -datapath_only -from $_xlnx_shared_i0 -to '+m.MEMORY_TO+' 5.000\n'
    return text,fifo


def test_actual_vendor_grammar_and_alias_replay():
    text,fifo=export();ignored,coverage=reports()
    result=m.review_exceptions(text,ignored,coverage,fifo_write_periods={fifo:'5.000'})
    assert result['issues']==[]
    assert result['canonical_facts']['memory_pairs']==1
    assert result['route_qualified'] is False


def test_missing_measured_clock_not_inferred_from_export():
    text,_=export();ignored,coverage=reports()
    result=m.review_exceptions(text,ignored,coverage,fifo_write_periods={})
    assert result['issues']


def test_memory_scope_or_alias_tampering():
    text,fifo=export();ignored,coverage=reports()
    with pytest.raises(ValueError,match='Unbound'):
        m.parse_export(text.replace('$_xlnx_shared_i0','$_xlnx_shared_i99'))
    bad=text.replace('# id:101','# id:101\ncurrent_instance other')
    with pytest.raises(ValueError,match='cross-scope'):
        m.parse_export(bad)
    result=m.review_exceptions(text.replace(' 5.000',' 6.000'),ignored,coverage,fifo_write_periods={fifo:'5.000'})
    assert any(x['reason']=='same_expected_clock_bound' for x in result['issues'])


def test_coverage_bound_cannot_change_independently():
    text,fifo=export();ignored,coverage=reports()
    result=m.review_exceptions(text,ignored,coverage.replace('max_dpo=5','max_dpo=7'),fifo_write_periods={fifo:'5.000'})
    assert any(x['reason']=='effective_coverage_numeric_bound' for x in result['issues'])


def test_unknown_clause_rejected_even_if_no_ignored_paths():
    text,fifo=export();ignored,coverage=reports()
    text+="# Constraints from file : 'user.xdc'\ncurrent_instance -quiet\n# id:102\nset_multicycle_path 4\n"
    result=m.review_exceptions(text,ignored,coverage,fifo_write_periods={fifo:'5.000'})
    assert any(x['reason']=='Unknown non-gray exception grammar/family' for x in result['issues'])


def test_new_custom_path_cannot_claim_inherited_shell_family():
    row=dict(id=0,source='sync_cdc.xdc',scope='',kind='set_false_path',command='set_false_path -setup -hold -to [get_pins [list {WRAPPER/CL/u_board/fake/D}]]')
    assert m.other_clause_family(row) is None


def test_repeated_zero_id_permitted_only_generated_records():
    rows=m.parse_export("# Constraints from file : 'sync_cdc.xdc'\n# id:0\nset_false_path -to a\n# id:0\nset_false_path -to b\n")
    assert len(rows)==2
    with pytest.raises(ValueError,match='Duplicate'):
        m.parse_export("# id:1\nset_false_path -to a\n# id:1\nset_false_path -to b\n")


def finding_report(check,severity,body):
    return f'| {check} | {severity} | Description | 1 |\n\n{check}#1 {severity}\n{body}\nRelated violations: <none>\n'


def test_finding_disposition_facts_pin_full_records():
    drc=finding_report('DPIP-2','Warning','Input pipelining\nDSP WRAPPER/CL/u_board/core/epilogue/u_m1/fp_mul_s1[p]_return input is not pipelined.')
    method=finding_report('SYNTH-6','Warning','Block RAM\nThe timing for WRAPPER/CL/u_board/mem might be sub-optimal as no output register was merged into the block.')
    first=m.review_methodology(drc,method);second=m.review_methodology(drc.replace('/u_m1/','/u_m2/'),method)
    assert first['issues']==[]
    assert first['canonical_facts']['groups'][0]['records_sha256']!=second['canonical_facts']['groups'][0]['records_sha256']
    assert first['route_qualified'] is False


def test_unreviewed_finding_and_critical_severity_rejected():
    bad=finding_report('NEW-1','Warning','Unknown new warning')
    known=finding_report('TIMING-9','Warning','See report_cdc for complete review.')
    assert m.review_methodology(bad,known)['issues']
    bad=finding_report('DPIP-2','Critical Warning','DSP WRAPPER/CL/u_board/core/epilogue/u_m1/fp_mul_s1[p]_return')
    assert m.review_methodology(bad,known)['issues']


def test_findings_summary_not_just_selected_details():
    text=finding_report('DPIP-2','Warning','DSP WRAPPER/CL/u_board/core/epilogue/u_m1/fp_mul_s1[p]_return')
    with pytest.raises(ValueError,match='summary/details'):
        m.parse_findings_report(text.replace('| 1 |','| 2 |'))


def test_required_methodology_prerequisite_not_silently_classified():
    drc=finding_report('RTSTAT-10','Warning','No routable loads\n2 net(s) have no routable loads.')
    method=finding_report('TIMING-9','Warning','See report_cdc for complete review.')
    result=m.review_methodology(drc,method)
    assert {row['family'] for row in result['issues']}=={'unused_net_topology_required','complete_current_cdc_review_required'}


def test_c1_small_multiplier_mapping_is_exact_shape_and_does_not_approve_records():
    body='Detected multiplier at WRAPPER/CL/engines[15].engine/sums_reg[0][10][11]_i_1 of size 8x8, it is implemented as LUTs due to its small size.'
    row=dict(check='SYNTH-9',severity='Warning',body=body)
    assert m.finding_family('methodology',row)=='c1_small_integer_multiplier_mapping_recommendations'
    for changed in (body.replace('engines[15]','engines[16]'),
                    body.replace('8x8','16x8'),body.replace('sums_reg','unreviewed'),
                    body.replace('sums_reg[0][10]','sums_reg[1][10]'),
                    body.replace('sums_reg[0][10]','sums_reg[0][32]'),
                    body.replace('[11]_i_1','[21]_i_1')):
        assert m.finding_family('methodology',{**row,'body':changed}) is None
    assert m.finding_family('methodology',{**row,'severity':'Critical Warning'}) is None
    for prefix in ('u_board/core/cluster/engines[1].engine','u_board/adapter'):
        relocated=body.replace('engines[15].engine',prefix).replace('_i_1 of','_i_18__13 of')
        assert m.finding_family('methodology',{**row,'body':relocated})=='c1_small_integer_multiplier_mapping_recommendations'


@pytest.mark.parametrize(('suffix','size'),[('','13x18'),('__0','18x18'),('__1','18x15')])
def test_c1_address_multiplier_mapping_retains_exact_current_dimensions(suffix,size):
    body=f'Detected multiplier at WRAPPER/CL/u_board/max_span0{suffix} of size {size}, it is decomposed from a wide multipler into 4 DSP blocks.'
    row=dict(check='SYNTH-10',severity='Warning',body=body)
    assert m.finding_family('methodology',row)=='c1_address_span_multiplier_mapping_recommendations'
    assert m.finding_family('methodology',{**row,'body':body.replace(size,'32x32')}) is None
    assert m.finding_family('methodology',{**row,'body':body.replace('max_span0','unreviewed')}) is None


def test_c1_dsp_warning_scope_does_not_inherit_c2_mxu_or_arbitrary_custom_dsp():
    for name in ('WRAPPER/CL/u_board/u_mxu/example','WRAPPER/CL/u_board/core/unreviewed'):
        assert m.finding_family('drc',dict(check='DPIP-2',severity='Warning',body=f'DSP {name} input is not pipelined.')) is None


def test_fifo_period_is_from_observed_addra_clock():
    fifo='WRAPPER/CL/u_mem/vendor/gen_xpm_memory_fifo.inst_fifo';root=fifo+'/gen_async_clocks.inst_cdc_addra_to_rd_clk'
    class Gray:
        groups={root:{'source':'0 {a} 1 {b}'}}
        relations={root:{'fifo_candidate':fifo}}
        issues=[]
        def clock(self,base,pin):
            assert base[:2]==('gray',root) and base[-1]=='source_clock'
            return ('writeclk',m.Decimal('2.222'))
    e=Gray();assert m.fifo_write_periods_from_gray(e)=={fifo:'2.222'}
    e.relations={root:{'fifo_candidate':'different'}}
    with pytest.raises(ValueError,match='relation'):m.fifo_write_periods_from_gray(e)


def test_cdc_report_inventory_must_match_raw_objects():
    text='CDC-15 Warning 1 Clock enable\nSource Clock: src\nDestination Clock: dst\nCDC Type: Asynchronous\n 1 CDC-15 Warning Clock Enable 2 Max Delay Datapath Only /src/Q /dest/D N\n'
    row=dict(source_clock='src',destination_clock='dst',cdc_type='Asynchronous',row='1',rule='CDC-15',severity='Warning',description='Clock Enable',depth='2',exception='Max Delay Datapath Only',**{'from':'/src/Q','to':'/dest/D'},waived='N',category='vendor_fifo_candidate',review='pending')
    assert m.current_cdc_inventory(text,[row])['counts']=={'CDC-15':1}
    with pytest.raises(ValueError,match='summary/detail'):m.current_cdc_inventory(text.replace('Warning 1 Clock','Warning 2 Clock'),[row])
    with pytest.raises(ValueError,match='object inventory'):m.current_cdc_inventory(text,[dict(row,waived='Y')])


def test_wrapper_missing_raw_prerequisite_never_accepts():
    result=m.review_artifact(reports={},auxiliary={},checkpoint_sha256='a'*64)
    assert result['issues'] and result['route_qualified'] is False
def test_reset_gate_warning_requires_exact_raw_gate_and_clear_pins():
    import copy
    root='WRAPPER/CL/u_rst_core';gate=root+'_i_1'
    pins=[root+f'/arststages_ff_reg[{i}]/CLR' for i in range(4)]
    row={'name':'LUTAR-1#1','check':'LUTAR-1','severity':'Warning',
         'body':'LUT drives async reset alert\nLUT cell '+gate+', with 2 or more inputs, drives asynchronous preset/clear pin(s) '+',\n'.join(pins)+'. The LUT may glitch.'}
    proof={'issues':[],'canonical_facts':{'resets':{root:{'gate':{'cell':gate},
        'registers':{pin.rsplit('/',1)[0]:{'pins':{'CLR':pin}} for pin in pins}}}}}
    result=m.reset_gate_findings([row],proof)
    assert result[root]['clear_pins']==pins
    assert result[root]['glitch_freedom_asserted'] is False
    assert result[root]['reset_event_minimum_pulse_asserted'] is False
    broken=copy.deepcopy(proof);broken['issues']=[{'reason':'polarity failure'}]
    with pytest.raises(ValueError,match='raw reset'):m.reset_gate_findings([row],broken)
    broken=copy.deepcopy(proof);broken['canonical_facts']['resets'][root]['gate']['cell']='other'
    with pytest.raises(ValueError,match='gate differs'):m.reset_gate_findings([row],broken)
    changed={**row,'body':row['body'].replace(pins[0],root+'/unexpected/CLR')}
    with pytest.raises(ValueError,match='endpoint inventory'):m.reset_gate_findings([changed],proof)
@pytest.mark.parametrize('cell',[
    'SL/UC_PCIE4_BRIDGE/inst/pcie4_ip_i/inst/bufg_gt_sysclk',
    'WRAPPER/RL/HOST_PCIE4C/inst/bufg_gt_sysclk'])
def test_exact_protected_pcie_clock_buffer_advisory(cell):
    row={'check':'CLKC-3','severity':'Advisory','body':'The BUFG_GT cell '+cell+' has a LOC constraint, but using the CLOCK_REGION property instead would give more flexibility.'}
    assert m.finding_family('methodology',row)=='exact_inherited_shell_clock_buffer_placement'
    assert m.finding_family('methodology',{**row,'body':row['body'].replace('bufg_gt_sysclk','unreviewed_buffer')}) is None
    assert m.finding_family('methodology',{**row,'severity':'Critical Warning'}) is None


def literal_memory_selector(n=132):
 cells=['xpm_memory_base_inst/gen_rd_b.gen_rd_b_synth_template.gen_rf_narrow_reg.doutb_reg_reg['+str(i)+']' for i in range(n)]
 return '[get_cells [list '+' '.join('{'+x+'}' for x in cells)+']]'


@pytest.mark.parametrize('union',[False,True])
def test_routed_literal_memory_selectors_keep_exact_endpoint_facts(union):
 text,fifo=export();ignored,coverage=reports()
 selector=literal_memory_selector(23 if union else 132)
 if union:selector='[list '+m.MEMORY_TO+' '+selector+']'
 result=m.review_exceptions(text.replace(m.MEMORY_TO,selector),ignored,coverage,fifo_write_periods={fifo:'5.000'})
 assert result['issues']==[]
 selected=next(row[3] for row in result['canonical_facts']['facts'] if row[0]=='memory_destination')
 assert selected['mode']==('vendor_filter_plus_explicit_cells' if union else 'explicit_cells')
 assert len(selected['cells'])==(23 if union else 132)


@pytest.mark.parametrize('mutation',[
 lambda s:s.replace('doutb_reg_reg[0]','doutb_reg_reg[*]'),
 lambda s:s.replace('xpm_memory_base_inst/','../xpm_memory_base_inst/'),
 lambda s:s.replace('xpm_memory_base_inst/','WRAPPER/CL/other/'),
 lambda s:s.replace('doutb_reg_reg','unreviewed_reg'),
 lambda s:s.replace('{xpm_memory_base_inst/','{[exec dangerous]/'),
 lambda s:s.replace('doutb_reg_reg[1]','doutb_reg_reg[0]'),
 lambda s:s.replace('[get_cells','[get_pins'),
 lambda s:s.replace('[list ','[list $unknown '),
])
def test_unknown_or_mutated_literal_memory_selectors_reject(mutation):
 with pytest.raises(ValueError):m.memory_destination(mutation(literal_memory_selector(2)))


def test_pure_literal_selector_cannot_use_larger_coverage_count():
 text,fifo=export();ignored,coverage=reports()
 result=m.review_exceptions(text.replace(m.MEMORY_TO,literal_memory_selector(23)),ignored,coverage,fifo_write_periods={fifo:'5.000'})
 assert any(x['reason']=='literal_destination_selected_coverage' for x in result['issues'])
