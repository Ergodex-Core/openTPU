"""Exact-source binding and bounded FIFO invariant tests; never hardware proof."""
from copy import deepcopy
from collections import deque
import hashlib
import json
from pathlib import Path
import random
import pytest
from boards.c1_f2_alex import admission_specialization_review as R

ROOT=Path(__file__).resolve().parents[1]

def record(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def top_fixture():
    rows=['otpu_c1_hbm_kernel #(.BUILD_ID(32\'h0d654453)) u_board (.n_cwe(n_cwe),.n_wvalid(n_wvalid),.n_caddr(n_caddr));']
    for pc in range(8):
        wires=[f'.{role}({role}[{pc}])' for role in ('n_cwe','n_wvalid','n_caddr','n_cvalid','n_cready')]
        rows.append(f"otpu_native_axi #(.BASE_ADDR(34'd0)) u_native{pc} (.clk(core_clk),.rst(core_rst),"+','.join(wires)+');')
    return '\n'.join(rows)


def sources():return {name:(ROOT/name).read_text() for name in R.SOURCE_SHA256}


def test_reviewed_source_bytes_are_exact_known_frozen_files():
    assert {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in R.SOURCE_SHA256}==R.SOURCE_SHA256
    facts=R.source_facts(sources(),top_fixture())
    assert facts['address']['maximum']==8389118 and facts['address']['bit_24']==0
    assert facts['address']['bit_23_is_not_assumed_zero']
    assert facts['fifo']['abstract_states']==272 and facts['fifo']['legal_transition_cases']==1024
    assert not facts['fifo']['unwritten_entries_assumed_zero']
    assert not facts['whole_native_state_equivalence_proven']


@pytest.mark.parametrize('case',['disabled_write','address_width','address_shift','qd','max_out','burst','idle','pop_guard','push_write','counter_case','pending_load','extra_pending_update','top_parameter','top_connection','top_clock','top_missing_instance'])
def test_source_or_wiring_mutation_requires_fresh_semantic_review(case):
    text=sources();top=top_fixture();kernel,native,adapter=list(R.SOURCE_SHA256)[:3]
    if case=='disabled_write':text[kernel]=text[kernel].replace("assign n_cwe='0", "assign n_cwe='1")
    elif case=='address_width':text[native]=text[native].replace('[15:0][22:0] row_base','[15:0][23:0] row_base')
    elif case=='address_shift':text[native]=text[native].replace(']>>1)',']>>0)')
    elif case in ('qd','max_out','burst','idle'):
        key={'qd':'QD','max_out':'MAX_OUT','burst':'READ_BURST','idle':'READ_IDLE'}[case]
        text[adapter]=text[adapter].replace(f'{key} = {R.CONFIGURATION[key]}',f'{key} = {R.CONFIGURATION[key]+1}')
    elif case=='pop_guard':text[adapter]=text[adapter].replace('!pending_valid && cn != 0','!pending_valid && cn == 0')
    elif case=='push_write':text[adapter]=text[adapter].replace('{n_cwe,n_caddr}',"{1'b1,n_caddr}")
    elif case=='counter_case':text[adapter]=text[adapter].replace("2'b10: cn <= cn+1'b1", "2'b10: cn <= cn+2'd2")
    elif case=='pending_load':text[adapter]=text[adapter].replace('pending_addr <= cq[cr][24:0]',"pending_addr <= 25'h1000000")
    elif case=='extra_pending_update':text[adapter]+="\nalways_ff @(posedge clk) pending_addr <= 25'h1000000;"
    elif case=='top_parameter':top=top.replace(".BASE_ADDR(34'd0)",".BASE_ADDR(34'd0),.QD(32)",1)
    elif case=='top_connection':top=top.replace('.n_cwe(n_cwe[0])',".n_cwe(1'b1)")
    elif case=='top_clock':top=top.replace('.clk(core_clk)', '.clk(other_clk)',1)
    elif case=='top_missing_instance':top=top.replace('u_native7','u_other7')
    with pytest.raises(ValueError):R.source_facts(text,top)


@pytest.mark.parametrize('cr,count,push,pop',[(0,0,False,True),(0,16,True,False),(0,16,True,True),(-1,0,False,False),(0,17,False,False)])
def test_illegal_source_fifo_operations_are_never_admitted(cr,count,push,pop):
    with pytest.raises(ValueError):R.fifo_step(cr,count,push,pop,16)


def test_induction_matches_independent_concrete_fifo_with_unknown_unused_slots():
    randomizer=random.Random(449);queue=deque();memory=[None]*16;cr=cp=0;pending_high=0
    for _ in range(10000):
        push=len(queue)<16 and bool(randomizer.getrandbits(1));pop=bool(queue) and bool(randomizer.getrandbits(1))
        before_count=len(queue)
        if pop:
            slot=queue.popleft();assert slot==cr and memory[slot]==0
            pending_high=memory[slot];cr=(cr+1)%16
        if push:
            memory[cp]=0;queue.append(cp);cp=(cp+1)%16
        step=R.fifo_step((cr-int(pop))%16,before_count,push,pop,16)
        assert (step['cr'],step['cp'],step['count'])==(cr,cp,len(queue))
        assert set(step['occupied'])==set(queue) and pending_high==0


@pytest.fixture
def bound(tmp_path,monkeypatch):
    # Synthetic manifest/top pins are substituted only in this unit fixture.
    # Production constants remain the exact actual frozen artifact identities.
    paths={}
    for index,name in enumerate(R.SOURCE_SHA256):
        p=tmp_path/f'source-{index}';p.write_bytes((ROOT/name).read_bytes());paths[name]=p
    top=tmp_path/'top';top.write_text(top_fixture());prepared={};source={}
    for name,path in paths.items():
        if name.startswith('boards/'):
            for prefix in ('source/rtl/boards/f2/','synth/rtl/','link/rtl/'):prepared[prefix+'otpu_native_axi.sv']=record(path)
        else:
            source[name]=record(path)
            if name.endswith('.sv'):prepared['source/'+name]=record(path)
    for prefix in ('source/rtl/boards/f2/','synth/rtl/','link/rtl/'):prepared[prefix+'cl_opentpu.sv']=record(top)
    inputs=dict(schema=1,kind='c1-alex-burst200-build-inputs',source_revision=R.HARDWARE_REVISION,build_id=R.HARDWARE_REVISION[:8],source_status='',prepared_files=prepared,source_inputs=source)
    p=tmp_path/'inputs.json';p.write_text(json.dumps(inputs))
    monkeypatch.setattr(R,'TOP_SHA256',record(top)['sha256']);monkeypatch.setattr(R,'SYNTH_INPUT_SHA256',record(p)['sha256'])
    return dict(source_paths=paths,synth_inputs_path=p,synth_inputs_sha256=R.SYNTH_INPUT_SHA256,generated_top_path=top)


def test_public_source_binding_proves_only_specialized_source_invariants(bound):
    r=R.review_sources(**bound)
    assert r['source_elided_bits_proven'] and not r['issues']
    assert r['source_hardware_equivalence_proven'] is r['route_qualified'] is r['hardware_measured'] is False
    assert r['canonical_facts']['configuration']==R.CONFIGURATION
    assert len(r['pending_reviews'])==3


@pytest.mark.parametrize('mutation',['source','top','input','input_hash','missing_source','extra_source','prepared_adapter','prepared_kernel','generator','revision'])
def test_public_byte_and_receipt_binding_mutations_reject(bound,monkeypatch,mutation):
    b=deepcopy(bound)
    if mutation=='source':next(iter(b['source_paths'].values())).write_text('changed')
    elif mutation=='top':b['generated_top_path'].write_text('changed')
    elif mutation=='input':b['synth_inputs_path'].write_text('changed')
    elif mutation=='input_hash':b['synth_inputs_sha256']='f'*64
    elif mutation=='missing_source':b['source_paths'].pop(next(iter(b['source_paths'])))
    elif mutation=='extra_source':b['source_paths']['unknown']=next(iter(b['source_paths'].values()))
    else:
        p=b['synth_inputs_path'];data=json.loads(p.read_text())
        if mutation=='prepared_adapter':data['prepared_files']['synth/rtl/otpu_native_axi.sv']['sha256']='f'*64
        elif mutation=='prepared_kernel':data['prepared_files']['source/rtl/c1/otpu_c1_hbm_kernel.sv']['sha256']='f'*64
        elif mutation=='generator':data['source_inputs']['rtl/c1_alex/generate.py']['sha256']='f'*64
        elif mutation=='revision':data['source_revision']='a'*40
        p.write_text(json.dumps(data));b['synth_inputs_sha256']=record(p)['sha256'];monkeypatch.setattr(R,'SYNTH_INPUT_SHA256',b['synth_inputs_sha256'])
    r=R.review_sources(**b)
    assert r['issues'] and not r['source_elided_bits_proven'] and not r['route_qualified']


def test_relocated_source_proof_is_byte_identical_and_has_no_host_paths(bound,tmp_path):
    import shutil
    before=R.review_sources(**bound)
    copied=tmp_path/'relocated';copied.mkdir()
    after_binding=deepcopy(bound)
    for name,path in bound['source_paths'].items():
        target=copied/path.name;shutil.copyfile(path,target);after_binding['source_paths'][name]=target
    for key in ('synth_inputs_path','generated_top_path'):
        target=copied/bound[key].name;shutil.copyfile(bound[key],target);after_binding[key]=target
    assert R.review_sources(**after_binding)==before
    assert str(tmp_path) not in json.dumps(before)
