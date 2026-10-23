"""Mutation tests for the complete typed unused-output witness protocol."""
import copy
from pathlib import Path
import pytest
from boards.c2_f2_alex import exception_review as m

DCP='a'*64
HBM='WRAPPER/CL/u_hbm/HBM_CORE_I/inst/TWO_STACK.u_hbm_top'
DDR='WRAPPER/CL/SH_DDR/genblk1.IS_DDR_IO.DDR_CLK/O'
def tl(values):return ' '.join('{'+str(v)+'}' for v in values)
def td(values):return tl([x for item in values.items() for x in item])

def fixture(tmp_path):
    names=[DDR]+[f'{HBM}/sl_oport{s}[{bit}]' for s in range(2) for bit in range(17)]
    current={'RTSTAT-10#1':{'nets':['WRAPPER/RL/existing']+names,'ports':[]}}
    rows=[tl(['schema','1','kind','unused_output_nets','checkpoint_sha256',DCP,'route_qualified','0']),tl(['selected_nets','names',tl(names)])]
    for index,name in enumerate(names):
        owner=name.rsplit('/',1)[0]+'/IBUFCTRL_INST' if index==0 else HBM+'/i_'+str(index-1)
        driver=owner+'/O';ref='IBUFCTRL' if index==0 else 'LUT1'
        rows.append(tl(['net',name,'properties',td(dict(NAME=name,CLASS='net',ROUTE_STATUS='NOLOADS')),'segments',tl([name,name+'_alias']),'pins',tl([driver]),'leaf_pins',tl([driver]),'ports','']))
        rows.append(tl(['leaf',name,'name',driver,'properties',td(dict(NAME=driver,CLASS='pin',DIRECTION='OUT',REF_PIN_NAME='O',REF_NAME=ref)),'owner',tl([owner]),'owner_properties',td(dict(NAME=owner,CLASS='cell',REF_NAME=ref,IS_PRIMITIVE='1',IS_BLACKBOX='0'))]))
    rows.append(tl(['complete','nets',len(names),'route_qualified','0']))
    topology=tmp_path/'topology';topology.write_text('\n'.join(rows)+'\n')
    shell=tmp_path/'shell';shell.write_text('\n'.join(tl(['relation_inventory','drc RTSTAT-10#1',kind,'count',len(values),'names',tl(values),'detail_cap','128']) for kind,values in [('pins',[]),('cells',[]),('nets',['WRAPPER/RL/existing','WRAPPER/RL/removed']),('ports',[]),('pblocks',[])])+'\n')
    return current,shell,topology

def test_exact_inheritance_and_physical_unused_proof(tmp_path):
    args=fixture(tmp_path);result=m.review_unused_outputs(*args,DCP)
    assert result['issues']==[] and result['route_qualified'] is False
    assert result['canonical_facts']['inherited_shell_nets']==['WRAPPER/RL/existing']
    assert len(result['canonical_facts']['new_unused_outputs'])==35

@pytest.mark.parametrize('old,new',[
    ('{DIRECTION} {OUT}','{DIRECTION} {IN}'),
    ('{IS_BLACKBOX} {0}','{IS_BLACKBOX} {1}'),
    ('{IS_PRIMITIVE} {1}','{IS_PRIMITIVE} {0}'),
    ('{ROUTE_STATUS} {NOLOADS}','{ROUTE_STATUS} {UNROUTED}'),
    ('{ports} {}','{ports} {external}'),
    ('{complete} {nets} {35}','{complete} {nets} {34}'),
])
def test_actual_consumer_or_incomplete_proof_blocks(tmp_path,old,new):
    current,shell,topology=fixture(tmp_path);data=topology.read_text();assert old in data
    topology.write_text(data.replace(old,new,1))
    with pytest.raises(ValueError):m.review_unused_outputs(current,shell,topology,DCP)

def test_wrong_checkpoint_blocks(tmp_path):
    with pytest.raises(ValueError,match='checkpoint'):m.review_unused_outputs(*fixture(tmp_path),'b'*64)

def test_added_unreviewed_static_net_blocks(tmp_path):
    current,shell,topology=fixture(tmp_path);current['RTSTAT-10#1']['nets'].append('WRAPPER/RL/unreviewed')
    with pytest.raises(ValueError,match='source contract'):m.review_unused_outputs(current,shell,topology,DCP)

def test_current_port_blocks(tmp_path):
    current,shell,topology=fixture(tmp_path);current['RTSTAT-10#1']['ports']=['external']
    with pytest.raises(ValueError,match='port endpoint'):m.review_unused_outputs(current,shell,topology,DCP)

def test_baseline_truncation_blocks(tmp_path):
    current,shell,topology=fixture(tmp_path);shell.write_text(shell.read_text().replace('WRAPPER/RL/existing','...'))
    with pytest.raises(ValueError,match='truncated'):m.review_unused_outputs(current,shell,topology,DCP)

def test_extra_or_error_record_blocks(tmp_path):
    current,shell,topology=fixture(tmp_path);topology.write_text(topology.read_text()+'incomplete cap\n')
    with pytest.raises(ValueError,match='Unexpected'):m.review_unused_outputs(current,shell,topology,DCP)

def test_missing_raw_probe_blocks(tmp_path):
    current,shell,topology=fixture(tmp_path);topology.unlink()
    with pytest.raises(FileNotFoundError):m.review_unused_outputs(current,shell,topology,DCP)
