"""Offline Alex adapter tests; injected evidence/devices are never FPGA results."""
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
A=load('alex_session_test',ROOT/'tools/c2_f2_alex_session.py')
B=load('alex_bench_test',ROOT/'tools/c2_f2_alex_bench.py')
D=A.D
F=load('canonical_session_fixtures',ROOT/'tests/test_c2_f2_session.py')
H=load('canonical_host_fixtures',ROOT/'tests/test_c2_f2_host.py')


def verified_fixture(tmp_path):
    root=tmp_path/'bundle';root.mkdir()
    inputs=dict(kind='c2-alex-burst200-build-inputs',source_revision='264d8413'+'a'*32,
        build_id='264d8413',core_khz=200000,hbm_mhz=450,hbm_ports=8,channel_bytes=2**29)
    path=root/'synth-inputs.json';path.write_text(json.dumps(inputs))
    v=dict(schema=1,kind='verified-c2-alex-portable-evidence',instance_id=A.S.INSTANCE,
        route_qualified=True,hardware_measured=False,source_revision=inputs['source_revision'],
        build_id=inputs['build_id'],manifest_sha256='a'*64,synth_inputs_sha256=A.S.sha(path),
        synth_inputs_relative_path='synth-inputs.json',final_dcp_relative_path='final.dcp',
        final_dcp_sha256='b'*64,ip_receipt_sha256='c'*64,simulation_receipt_sha256='d'*64,
        qualification_receipt_sha256='e'*64,files=1,bytes=path.stat().st_size)
    record=dict(schema=1,kind='c2-alex-fpga-image',account=D.ACCOUNT,region=D.REGION,bucket=D.BUCKET,
        state='available',source_revision=v['source_revision'],build_id=v['build_id'],
        evidence_manifest_sha256=v['manifest_sha256'],inputs_sha256=v['synth_inputs_sha256'],
        dcp_sha256=v['final_dcp_sha256'],ip_receipt_sha256=v['ip_receipt_sha256'],
        simulation_receipt_sha256=v['simulation_receipt_sha256'],qualification_receipt_sha256=v['qualification_receipt_sha256'],
        package_sha256='f'*64,package_receipt_sha256='1'*64,package_bytes=123,
        image_id=F.AFI,global_image_id=F.AGFI,prefix='c2-alex/'+v['build_id']+'/'+'2'*32)
    record['client_token']=D.token(record);record['publication_request']=D.make_request(record)
    image=tmp_path/'image.json';image.write_text(json.dumps(record))
    return root,v,record,image


def test_publisher_and_session_share_exact_request_contract(tmp_path):
    _,v,r,_=verified_fixture(tmp_path)
    assert len(D.image_tags(r))==5
    request=D.make_request(r)
    assert request['ClientToken']==D.token(r)
    assert request['InputStorageLocation']==dict(Bucket=D.BUCKET,Key=r['prefix']+'/design.tar')
    assert D.verify_deployment(r,v)['agfi_id']==F.AGFI
    assert request['TagSpecifications'][0]['Tags']==[dict(Key=k,Value=x) for k,x in D.image_tags(r).items()]


@pytest.mark.parametrize('state',['prepared','pending','failed','unavailable'])
def test_publisher_nonavailable_record_never_qualifies_session(tmp_path,state):
    _,v,r,_=verified_fixture(tmp_path);r['state']=state
    if state=='prepared':r.pop('image_id');r.pop('global_image_id')
    D.validate_record(r,v)
    with pytest.raises(ValueError,match='not available'):D.verify_deployment(r,v)


@pytest.mark.parametrize('key,value',[
    ('account','other'),('region','us-west-2'),('bucket','other'),('source_revision','0'*40),
    ('evidence_manifest_sha256','0'*64),('inputs_sha256','0'*64),('dcp_sha256','0'*64),
    ('ip_receipt_sha256','0'*64),('simulation_receipt_sha256','0'*64),('qualification_receipt_sha256','0'*64),
    ('package_receipt_sha256','invalid'),('client_token','0'*64),('global_image_id',D.ORIGINAL),
    ('image_id','not-an-afi'),('package_bytes',0)])
def test_wrong_deployment_binding_rejected(tmp_path,key,value):
    _,v,r,_=verified_fixture(tmp_path);r[key]=value
    with pytest.raises(ValueError):D.verify_deployment(r,v)


def test_rehashed_but_different_evidence_record_still_rejected(tmp_path):
    _,v,r,_=verified_fixture(tmp_path);r['evidence_manifest_sha256']='0'*64
    r['client_token']=D.token(r);r['publication_request']=D.make_request(r)
    with pytest.raises(ValueError,match='qualified evidence'):D.verify_deployment(r,v)


@pytest.mark.parametrize('key,value',[('route_qualified',False),('hardware_measured',True),('kind','archive-inventory')])
def test_pending_or_inventory_only_evidence_never_authorizes(tmp_path,key,value):
    _,v,r,_=verified_fixture(tmp_path);v[key]=value
    with pytest.raises(ValueError,match='not qualified'):D.verify_deployment(r,v)


def test_qualification_calls_portable_validator_and_pins_image_record(tmp_path,monkeypatch):
    root,v,r,path=verified_fixture(tmp_path);calls=[]
    def verify(directory,digest):calls.append((directory,digest));return v
    monkeypatch.setattr(A.E,'verify_bundle',verify)
    q=A.qualify(root,v['manifest_sha256'],path,A.S.sha(path))
    assert calls==[(root.resolve(),v['manifest_sha256'])]
    assert q['source_revision']==r['source_revision'] and q['inputs_sha256']==v['synth_inputs_sha256']
    assert q['kind']=='c2-alex-session-qualification'
    for name in ('opentpu/lens.py','opentpu/lens_app.html'):
        assert q['software_sources'][name]==A.S.sha(ROOT/name)
    with pytest.raises(ValueError,match='pinned'):A.qualify(root,v['manifest_sha256'],path,'0'*64)


@pytest.mark.parametrize('ancestor',[False,True])
def test_qualification_rejects_symlink_root_components_before_verification(tmp_path,monkeypatch,ancestor):
    root,v,_,path=verified_fixture(tmp_path)
    alias=tmp_path/'alias'
    alias.symlink_to(tmp_path if ancestor else root,target_is_directory=True)
    supplied=alias/'bundle' if ancestor else alias
    monkeypatch.setattr(A.E,'verify_bundle',lambda *args:pytest.fail('Symlink must be rejected before verification'))
    with pytest.raises(ValueError,match='Symlink path component'):
        A.qualify(supplied,v['manifest_sha256'],path,A.S.sha(path))


@pytest.mark.parametrize('name',['opentpu/lens.py','opentpu/lens_app.html'])
def test_benchmark_rejects_changed_lens_sources(tmp_path,monkeypatch,name):
    root,v,_,path=verified_fixture(tmp_path)
    monkeypatch.setattr(A.E,'verify_bundle',lambda *args:v)
    q=A.qualify(root,v['manifest_sha256'],path,A.S.sha(path))
    frozen=tmp_path/'qualification.json';frozen.write_text(json.dumps(q))
    digest=B.sha(frozen);original_sha=B.sha
    monkeypatch.setattr(B,'sha',lambda p:'0'*64 if Path(p)==ROOT/name else original_sha(p))
    with pytest.raises(ValueError,match='Frozen benchmark source changed: '+name):
        B.read_qualification(frozen,digest)


def test_real_portable_verifier_refuses_missing_evidence(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)):
        A.qualify(tmp_path,'0'*64,tmp_path/'absent.json','0'*64)


def test_fresh_aws_gate_preserves_original_checks_and_adds_evidence_tag(tmp_path):
    _,v,r,_=verified_fixture(tmp_path);q=D.verify_deployment(r,v)
    raw=F.aws_response(q)
    with pytest.raises(RuntimeError,match='portable evidence'):A.verify_aws_image(raw,q)
    raw['FpgaImages'][0]['Tags'].append(dict(Key='c2:evidence_manifest_sha256',Value=q['evidence_manifest_sha256']))
    assert A.verify_aws_image(raw,q)['agfi_id']==F.AGFI
    for key,value in [('OwnerId','wrong'),('ShellVersion','0xdeadbeef'),('FpgaImageGlobalId',D.ORIGINAL)]:
        changed=copy.deepcopy(raw);changed['FpgaImages'][0][key]=value
        with pytest.raises(RuntimeError):A.verify_aws_image(changed,q)


def test_canonical_session_control_flow_is_reused():
    assert A.session is A.S.session
    for name in ('acquire','load','rescan','discover','baseline','candidate','read_regs','command','no_foreign_handles'):
        assert getattr(A.AlexLocalOps,name) is getattr(A.S.LocalOps,name)


@pytest.mark.parametrize('failure',[None,'load','candidate','benchmark','cancel','restore'])
def test_new_qualification_preserves_restore_and_lock_order(tmp_path,failure):
    _,v,r,_=verified_fixture(tmp_path);q=D.verify_deployment(r,v)
    ops=F.FakeOps(failure,move=True)
    if failure:
        with pytest.raises(RuntimeError):A.session(ops,q,['first','second'],repeats=2)
    else:
        report=A.session(ops,q,['first','second'],repeats=2);assert report['restored']
    loads=[x for x in ops.events if x[0]=='load']
    assert loads==[('load',F.AGFI),('load',A.S.ORIGINAL)]
    assert not ops.held
    assert max(i for i,x in enumerate(ops.events) if x[0]=='load')<min(i for i,x in enumerate(ops.events) if x[0]=='release')
    if failure=='restore':assert 'restore_error' in ops.report and not ops.report['restored']


def test_plan_only_freezes_real_canonical_fixture_without_device_calls(tmp_path,monkeypatch):
    root,v,r,path=verified_fixture(tmp_path);monkeypatch.setattr(A.E,'verify_bundle',lambda *a:v)
    fixture=tmp_path/'fixture.npz';A.C.make_fixture(5,3).save(fixture)
    monkeypatch.setattr(A,'AlexLocalOps',lambda *a:pytest.fail('plan must not construct physical operations'))
    out=tmp_path/'session'
    A.main(['--evidence-dir',str(root),'--evidence-manifest-sha256',v['manifest_sha256'],
        '--image-record',str(path),'--image-record-sha256',A.S.sha(path),'--fixture',str(fixture),'--out',str(out)])
    plan=json.loads((out/'plan.json').read_text());q=plan['qualification']
    assert plan['execute'] is False and q['source_revision']==v['source_revision']
    frozen=A.C.Fixture.load(out/'fixtures/00.npz')
    assert frozen.digest()==A.C.Fixture.load(fixture).digest()
    assert B.read_qualification(Path(q['qualification_file']),q['qualification_sha256'])['build_id']==v['build_id']
    (out/'synthesis-inputs.json').write_text('{}')
    with pytest.raises(ValueError,match='inputs changed'):B.read_qualification(Path(q['qualification_file']),q['qualification_sha256'])


def test_inherited_lock_benchmark_command_uses_fresh_bdf(tmp_path):
    _,v,r,_=verified_fixture(tmp_path);q=D.verify_deployment(r,v)
    manifest=tmp_path/'inputs.json';manifest.write_text('source')
    frozen=tmp_path/'qualification.json';frozen.write_text('qualification')
    q.update(build_manifest=str(manifest),inputs_sha256=A.S.sha(manifest),qualification_file=str(frozen),
        qualification_sha256=A.S.sha(frozen),software_sources={})
    class Ops(A.AlexLocalOps):
        def require_owner(self,bdf):assert bdf=='0000:35:00.0'
        def command(self,role,args,**kwargs):self.captured=(role,args,kwargs)
    ops=Ops(tmp_path,'python')
    ops.benchmark('0000:35:00.0',SimpleNamespace(holder=4242),q,Path('fixture.npz'),3,0)
    _,cmd,kw=ops.captured
    assert cmd[1]==str(ROOT/'tools/c2_f2_alex_bench.py')
    assert cmd[cmd.index('--bdf')+1]=='0000:35:00.0'
    assert kw['env']['OTPU_LOCK_HELD']=='f2-0000:35:00.0' and kw['env']['OTPU_LOCK_PID']=='4242'
    assert kw['cancellable'] is True and kw['timeout']==900


def test_benchmark_adapter_uses_canonical_arithmetic_and_rejects_fake_profile():
    tr=H.Fake();fixture=A.C.make_fixture(5,3)
    results=list(B.checked_runs(tr,fixture,1234,2,1.))
    assert [r['mode'] for r in results]==[0,1,1,0]
    assert all(r['evidence_kind']=='unit-test' and r['bit_exact'] for r in results)
    for result in results:
        with pytest.raises(ValueError,match='physical F2'):A.C.lens_data(result,'never hardware')
    # Canonical preload uploaded and read back every PC before its first START.
    assert len(tr.pcis)==8


@pytest.mark.parametrize('bad',['bits','integer','pc','groups'])
def test_canonical_corruption_checks_survive_adapter(bad):
    tr=H.Fake();tr.corrupt=bad
    with pytest.raises(RuntimeError):list(B.checked_runs(tr,A.C.make_fixture(5,3),1234,1,1.))


def test_failed_prepublication_record_need_not_have_an_image(tmp_path):
    _,v,r,_=verified_fixture(tmp_path)
    r['state']='failed';r.pop('image_id');r.pop('global_image_id')
    D.validate_record(r,v)
    with pytest.raises(ValueError,match='not available'):D.verify_deployment(r,v)


def test_explicit_baseline_plan_freezes_exact_pinned_bytes(tmp_path,monkeypatch):
    root,v,r,path=verified_fixture(tmp_path);monkeypatch.setattr(A.E,'verify_bundle',lambda *a:v)
    fixture=tmp_path/'fixture.npz';A.C.make_fixture(5,3).save(fixture)
    baseline=tmp_path/'baseline.json';raw=(json.dumps(F.reviewed_baseline(),indent=3)+'\n').encode();baseline.write_bytes(raw)
    monkeypatch.setattr(A,'AlexLocalOps',lambda *a:pytest.fail('plan must not construct physical operations'))
    out=tmp_path/'session'
    A.main(['--evidence-dir',str(root),'--evidence-manifest-sha256',v['manifest_sha256'],
        '--image-record',str(path),'--image-record-sha256',A.S.sha(path),'--fixture',str(fixture),'--out',str(out),
        '--baseline-record',str(baseline),'--baseline-record-sha256',A.S.sha(baseline)])
    plan=json.loads((out/'plan.json').read_text());q=plan['qualification']
    assert (out/'baseline-record.json').read_bytes()==raw
    assert plan['baseline']==q['baseline']==F.reviewed_baseline()
    assert q['baseline_record_sha256']==A.S.sha(baseline)
    assert B.read_qualification(Path(q['qualification_file']),q['qualification_sha256'])['baseline']==F.reviewed_baseline()
    baseline.write_text('{}')
    assert (out/'baseline-record.json').read_bytes()==raw


@pytest.mark.parametrize('change',['hash','bytes','symlink','ancestor','missing_hash','missing_path','extra_field','source_build_id'])
def test_baseline_file_rejects_unpinned_or_unreviewed_data(tmp_path,change):
    p=tmp_path/'baseline.json';p.write_text(json.dumps(F.reviewed_baseline()));sha=A.S.sha(p)
    if change=='hash':sha='0'*64
    elif change=='bytes':p.write_text('{}')
    elif change in ('extra_field','source_build_id'):
        data=F.reviewed_baseline()
        if change=='extra_field':data['accepted']=True
        else:data['identity']['build_id']=int('f85ac03c',16)
        p.write_text(json.dumps(data));sha=A.S.sha(p)
    elif change=='symlink':
        alias=tmp_path/'alias';alias.symlink_to(p);p=alias
    elif change=='ancestor':
        alias=tmp_path/'alias';alias.symlink_to(tmp_path,target_is_directory=True);p=alias/'baseline.json'
    elif change=='missing_hash':sha=None
    else:p=None
    with pytest.raises(ValueError):A.load_baseline_record(p,sha)


def test_baseline_absence_preserves_legacy_mode():
    assert A.load_baseline_record(None,None)==(None,None)
