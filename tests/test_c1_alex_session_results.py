"""Synthetic in-memory boundary tests only; never exported as measurements."""
import copy,importlib.util
from pathlib import Path
import pytest
from opentpu import c1_f2 as C
s=importlib.util.spec_from_file_location('fields',Path(__file__).resolve().parents[1]/'boards/c1_f2_alex/session_results.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
@pytest.fixture
def sample():
    f=C.make_fixture(3,1,123);bits,ints=f.reference();beats=f.expected_pc_beats()
    ident=dict(id=C.ID,abi=C.ABI,build_id=0x12345678,core_khz=200000,max_rows=512,depth=128,credits=32)
    counters={k:0 for k in C.COUNTERS};counters.update(accepted_32B_groups=12,advance=12,committed_rows=3,read_issued=6,read_returned=6,max_outstanding=6)
    pcs=[dict(issued=x,returned=x,max_outstanding=x,blocked_cycles=0,reserved_latency=0,credit_wait_cycles=0) for x in beats]
    r=dict(evidence_kind='f2-hardware',identity=ident,device='f2:0000:34:00.0',bit_exact=True,status=C.DONE,error=0,repeat=0,mode=0,rows=3,kb=1,base_beat=16,fixture_sha256=f.digest(),bits=bits.tolist(),integers=ints,cycles=100,counters=counters,pcs=pcs,weight_bytes=384,measured_at_utc='2026-10-20T00:00:00+00:00',measured_window='COMMAND acceptance through final fp32 result; excludes OCL preload, ACT snapshot copy and host readback')
    return f,ident,r

def check(sample):
    f,i,r=sample;return M.checked_result_facts(r,f,expected_identity=i,expected_device='f2:0000:34:00.0',repeat=0)

def test_no_session_or_profile_acceptance(sample):
    out=check(sample);assert out['accepted_32B_groups']==12 and out['native_read_bytes']==384
    assert not out['session_verified'] and not out['route_qualified'] and not out['lens_append_authorized']

@pytest.mark.parametrize('field,value',[('evidence_kind','rtl'),('status',C.BUSY),('error',1),('mode',1),('cycles',2**32),('fixture_sha256','0'*64),('repeat',1),('device','other'),('measured_at_utc','2026-10-20T00:00:00')])
def test_wrong_result(sample,field,value):
    sample[2][field]=value
    with pytest.raises(ValueError):check(sample)

@pytest.mark.parametrize('mutation',['bits','integer','groups','rows','reserved','pc_bytes','pc_credit','pc_reserved'])
def test_canonical_or_counter_corruption(sample,mutation):
    r=sample[2]
    if mutation=='bits':r['bits'][0]^=1
    elif mutation=='integer':r['integers'][0]+=1
    elif mutation=='groups':r['counters']['accepted_32B_groups']=3
    elif mutation=='rows':r['counters']['committed_rows']=12
    elif mutation=='reserved':r['counters']['output_wait']=1
    elif mutation=='pc_bytes':r['pcs'][0]['returned']+=1
    elif mutation=='pc_credit':r['pcs'][0]['max_outstanding']=65
    elif mutation=='pc_reserved':r['pcs'][0]['reserved_latency']=1
    with pytest.raises(ValueError):check(sample)

@pytest.mark.parametrize('field',['mode','repeat','status','error','rows','kb','base_beat','weight_bytes'])
@pytest.mark.parametrize('alias',[bool,float])
def test_integer_field_aliases(sample,field,alias):
    sample[2][field]=alias(sample[2][field])
    with pytest.raises(ValueError):check(sample)

@pytest.mark.parametrize('field',['abi','build_id','core_khz'])
def test_identity_float_alias(sample,field):
    sample[2]['identity']=dict(sample[2]['identity'])
    sample[2]['identity'][field]=float(sample[2]['identity'][field])
    with pytest.raises(ValueError):check(sample)

@pytest.mark.parametrize('field,value',[('bits',True),('bits',1.0),('bits',2**32),('bits',-1),('integers',False),('integers',0.0),('integers',2**47),('integers',-(2**47)-1)])
def test_result_word_types_and_ranges(sample,field,value):
    sample[2][field][0]=value
    with pytest.raises(ValueError):check(sample)

def test_empty_device(sample):
    sample[2]['device']=''
    with pytest.raises(ValueError):check(sample)
