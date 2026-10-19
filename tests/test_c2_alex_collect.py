"""Local collection tests use temporary synthetic evidence, never physical proof."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

c=load('alex_collect_test',ROOT/'boards/c2_f2_alex/collect.py')
f=load('alex_collect_evidence_fixture',ROOT/'tests/test_c2_alex_evidence.py')


@pytest.fixture
def inputs(tmp_path,monkeypatch):
    origin,m,_=f.fixture_bundle(tmp_path,monkeypatch,original_base=str(tmp_path/'portable/runs'))
    # Keep real semantic evidence replay and copying. Only approved Git object
    # reads use these immutable parser-fixture snapshots. Real Git is separately
    # tested below, including dirty checkout and symlink object rejection.
    monkeypatch.setattr(c,'e',f.e)
    class SnapshotGit:
        def __init__(self,path,revision):self.path=Path(path)
        def read(self,name):return (self.path/name).read_bytes()
    monkeypatch.setattr(c,'GitSource',SnapshotGit)
    spec={'schema':1,'kind':'c2-alex-evidence-collection-inputs','instance_id':c.e.INSTANCE,
          'roots':{role:row['original_absolute_path'] for role,row in m['roots'].items()},
          'receipt_sha256':{role:row['sha256'] for role,row in m['receipts'].items()},
          'source_checkouts':{role:{'path':str(origin/row['bundle_relative_path']),'revision':row['revision']}
                              for role,row in m['source_snapshots'].items()}}
    return spec,origin,tmp_path/'collected'


def test_complete_collection_preserves_receipts_and_is_relocatable(inputs,tmp_path):
    spec,origin,out=inputs
    before=f.inventory(origin)
    receipt=c.collect(spec,out)
    assert receipt['state']=='complete' and receipt['hardware_actions'] is False
    assert receipt['verified']['route_qualified'] is True and receipt['verified']['hardware_measured'] is False
    assert f.inventory(origin)==before
    for role,name in c.e.RECEIPTS.items():
        assert (out/'bundle'/'runs'/role/name).read_bytes()==(Path(spec['roots'][role])/name).read_bytes()
    assert c.e.inspect_archive(out/'evidence.tar',receipt['manifest_sha256'])['route_qualified'] is False
    shutil.move(origin,tmp_path/'original-moved-away')
    assert c.e.verify_bundle(out/'bundle',receipt['manifest_sha256'])==receipt['verified']
    assert not (out/'failure.json').exists()


def test_archive_is_deterministic_across_fresh_outputs(inputs,tmp_path):
    spec,_,out=inputs
    a=c.collect(spec,out);b=c.collect(spec,tmp_path/'second')
    assert a['archive']==b['archive'] and a['manifest_sha256']==b['manifest_sha256']
    assert a['receipt_sha256']!=b['receipt_sha256']  # collection timestamps remain separate


@pytest.mark.parametrize('change',['wrong_hash','pending','missing_report','extra_stage_file','extra_include',
                                  'mutated_source','symlink','observer_failed','private_include_missing'])
def test_failed_collection_never_archives_or_overwrites(inputs,change):
    spec,origin,out=inputs
    if change=='wrong_hash':spec['receipt_sha256']['qualify']='0'*64
    elif change=='pending':
        path=Path(spec['roots']['qualify'])/'stage-result.json';r=f.read(path);r.update(route_qualified=False,status='pending-review')
        f.put(path,r);spec['receipt_sha256']['qualify']=c.e.sha(path)
    elif change=='missing_report':(Path(spec['roots']['qualify'])/'qualification/cdc.rpt').unlink()
    elif change=='extra_stage_file':f.put(Path(spec['roots']['refine'])/'unbound.log','unbound')
    elif change=='extra_include':f.put(Path(spec['roots']['ip'])/'ip/ip_project/f2_ip.srcs/new.vh','changed')
    elif change=='mutated_source':f.put(origin/'sources/hardware/rtl/mxu/otpu_c2_mxu.sv','changed')
    elif change=='observer_failed':f.put(Path(spec['roots']['dependencies'])/'attestation/failure.json',{'error':'failed'})
    elif change=='private_include_missing':(Path(spec['roots']['simulation_private'])/'ip/ip_project/f2_ip.srcs/include.vh').unlink()
    else:
        path=Path(spec['roots']['ip'])/'ip/ip_project/child.dcp';saved=path.read_bytes();path.unlink()
        target=origin/'same-bytes.dcp';target.write_bytes(saved);path.symlink_to(target)
    with pytest.raises((ValueError,FileNotFoundError)):c.collect(spec,out)
    assert (out/'failure.json').is_file() and not (out/'collection.json').exists()
    assert not (out/'evidence.tar').exists()
    failure=(out/'failure.json').read_bytes()
    with pytest.raises(ValueError,match='fresh'):c.collect(spec,out)
    assert (out/'failure.json').read_bytes()==failure


def test_source_change_during_copy_retains_failure(inputs,monkeypatch):
    spec,_,out=inputs;original=c.copy_file;changed=[]
    def copying(path,dest,expected):
        original(path,dest,expected)
        if not changed:
            changed.append(path);path.write_bytes(path.read_bytes()+b'changed after capture')
    monkeypatch.setattr(c,'copy_file',copying)
    with pytest.raises(ValueError,match='changed during collection'):c.collect(spec,out)
    assert not (out/'bundle'/c.e.MANIFEST).exists() and not (out/'evidence.tar').exists()
    assert (out/'failure.json').is_file()


def test_low_disk_does_not_copy_bundle(inputs,monkeypatch):
    spec,_,out=inputs;monkeypatch.setattr(c.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='free disk'):c.collect(spec,out)
    assert not (out/'bundle').exists() and (out/'failure.json').is_file()


def test_semantic_qualification_is_required_before_tar(inputs,monkeypatch):
    spec,_,out=inputs
    def pending(*args):raise ValueError('pending actual vendor review')
    monkeypatch.setattr(c.e,'verify_bundle',pending)
    with pytest.raises(ValueError,match='pending'):c.collect(spec,out)
    assert (out/'bundle'/c.e.MANIFEST).exists()
    assert not (out/'evidence.tar').exists() and not (out/'collection.json').exists()


def test_wrong_external_spec_hash_fails_before_output(inputs,tmp_path):
    spec,_,out=inputs;path=tmp_path/'spec.json';f.put(path,spec)
    with pytest.raises(ValueError,match='independently trusted'):
        c.main(['--spec',str(path),'--spec-sha256','0'*64,'--out',str(out)])
    assert not out.exists()


@pytest.mark.parametrize('change',['missing_role','missing_receipt','short_revision','wrong_instance','path_traversal'])
def test_spec_requires_explicit_complete_trusted_inputs(inputs,change):
    spec,_,out=inputs
    if change=='missing_role':spec['roots'].pop('ip')
    elif change=='missing_receipt':spec['receipt_sha256'].pop('simulation')
    elif change=='short_revision':spec['source_checkouts']['hardware']['revision']='264d8413'
    elif change=='wrong_instance':spec['instance_id']='other-host'
    else:spec['roots']['ip']='/path/../escape'
    with pytest.raises(ValueError):c.collect(spec,out)
    assert not out.exists()


def test_output_cannot_be_inside_source_or_input(inputs):
    spec,origin,_=inputs
    for out in (Path(spec['roots']['ip'])/'new-output',origin/'sources/hardware/new-output'):
        with pytest.raises(ValueError,match='outside'):c.collect(spec,out)
        assert not out.exists()


def test_installed_shell_alias_is_dereferenced_only_inside_hdk(tmp_path):
    root=tmp_path/'hdk';target=root/'hdk/common/shell-v1';target.mkdir(parents=True)
    (target/'input.v').write_text('source')
    alias=root/'hdk/common/shell_stable';alias.symlink_to('shell-v1',target_is_directory=True)
    assert c.original_file(root,'hdk/common/shell_stable/input.v',installed=True)==target/'input.v'
    with pytest.raises(ValueError,match='Symlink'):c.original_file(root,'hdk/common/shell_stable/input.v')
    alias.unlink();alias.symlink_to(tmp_path,target_is_directory=True)
    (tmp_path/'input.v').write_text('outside')
    with pytest.raises(ValueError,match='escapes'):c.original_file(root,'hdk/common/shell_stable/input.v',installed=True)


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def test_real_git_snapshot_uses_approved_commit_not_dirty_checkout(tmp_path):
    repo=tmp_path/'repo';repo.mkdir();git(repo,'init','-q');git(repo,'config','user.name','Fixture')
    git(repo,'config','user.email','fixture@example.invalid')
    (repo/'code.py').write_text('original bytes\n');(repo/'linked.py').symlink_to('code.py')
    git(repo,'add','.');git(repo,'commit','-qm','Parser test fixture')
    rev=git(repo,'rev-parse','HEAD');source=c.GitSource(str(repo),rev)
    (repo/'code.py').write_text('dirty unapproved bytes\n')
    assert source.read('code.py')==b'original bytes\n'
    with pytest.raises(ValueError,match='nonregular'):source.read('linked.py')
    with pytest.raises(ValueError,match='nonregular'):source.read('missing.py')
    with pytest.raises(ValueError,match='Full approved'):c.GitSource(str(repo),'HEAD')


@pytest.fixture
def synchronous_inputs(inputs,monkeypatch):
    spec,origin,out=inputs
    m=f.read(origin/c.e.MANIFEST);f.synchronous_fixture(origin,m,monkeypatch)
    old=origin/'runs/synth';new=Path(m['roots']['synth']['original_absolute_path'])
    shutil.move(old,new)
    spec['roots']['synth']=str(new)
    spec['source_checkouts']['hardware']['revision']='b'*40
    dep=Path(spec['roots']['dependencies']);source=dep/'src'
    shutil.copytree(origin/'sources/hardware',source)
    # A real temporary checkout, including .git/objects, index and config.
    # Its content remains a parser fixture, never qualified hardware evidence.
    git(source,'init','-q');git(source,'config','user.name','Fixture')
    git(source,'config','user.email','fixture@example.invalid')
    git(source,'add','.');git(source,'commit','-qm','Synthetic retry checkout')
    f.put(dep/'source.bundle',b'Synthetic preserved staging bundle')
    f.put(dep/'runner.log','Synthetic preserved wrapper log')
    identity=f.read(dep/'attestation/identity.json');identity['source']=str(source)
    f.put(dep/'attestation/identity.json',identity)
    before=f.read(dep/'attestation/before.json');before['identity_sha256']=c.e.sha(dep/'attestation/identity.json')
    f.put(dep/'attestation/before.json',before)
    command=f.read(dep/'attestation/command.json');command[1]=str(source/'boards/c2_f2_alex/build.py')
    f.put(dep/'attestation/command.json',command)
    complete=f.read(dep/'attestation/complete.json')
    complete.update(identity_sha256=c.e.sha(dep/'attestation/identity.json'),
                    before_sha256=c.e.sha(dep/'attestation/before.json'),
                    wrapper_command=f.rec(dep/'attestation/command.json'))
    f.put(dep/'attestation/complete.json',complete)
    for role,name in c.e.RECEIPTS.items():spec['receipt_sha256'][role]=c.e.sha(Path(spec['roots'][role])/name)
    return spec,origin,out


def test_synchronous_retry_collects_new_hardware_without_duplicating_synth(synchronous_inputs):
    spec,origin,out=synchronous_inputs
    receipt=c.collect(spec,out)
    manifest=f.read(out/'bundle'/c.e.MANIFEST)
    assert receipt['verified']['source_revision']=='b'*40
    assert not any(name.startswith('runs/dependencies/synthesis/') for name in manifest['files'])
    assert not any(name.startswith('runs/dependencies/src/') for name in manifest['files'])
    assert manifest['roots']['synth']['original_absolute_path']==spec['roots']['synth']
    assert (out/'bundle/runs/dependencies/run_synth_with_attestation.py').is_file()
    for name in ('source.bundle','runner.log','attestation/identity.json','attestation/before.json','attestation/complete.json'):
        assert (out/'bundle/runs/dependencies'/name).read_bytes()==(Path(spec['roots']['dependencies'])/name).read_bytes()
    assert (out/'bundle/sources/hardware/boards/c2_f2_alex/build.py').read_bytes()==(origin/'sources/hardware/boards/c2_f2_alex/build.py').read_bytes()


@pytest.mark.parametrize('change',['unbound_src','other_git','src_symlink','identity_hash'])
def test_retry_checkout_exclusion_is_exact_and_hash_bound(synchronous_inputs,change):
    spec,origin,out=synchronous_inputs;dep=Path(spec['roots']['dependencies'])
    if change=='unbound_src':
        identity=f.read(dep/'attestation/identity.json');identity['source']=str(dep/'other-src')
        f.put(dep/'attestation/identity.json',identity)
        complete=f.read(dep/'attestation/complete.json');complete['identity_sha256']=c.e.sha(dep/'attestation/identity.json')
        f.put(dep/'attestation/complete.json',complete)
        spec['receipt_sha256']['dependencies']=c.e.sha(dep/'attestation/complete.json')
    elif change=='other_git':f.put(dep/'unrelated/.git/config','unrelated Git metadata')
    elif change=='src_symlink':
        shutil.move(dep/'src',origin/'moved-src');(dep/'src').symlink_to(origin/'moved-src',target_is_directory=True)
    else:
        identity=f.read(dep/'attestation/identity.json');identity['source']=str(dep/'other-src')
        f.put(dep/'attestation/identity.json',identity)
    with pytest.raises(ValueError):c.collect(spec,out)
    assert (out/'failure.json').is_file() and not (out/'evidence.tar').exists()


def schema2_inputs(inputs):
    spec,origin,out=inputs;spec['schema']=2
    q=Path(spec['roots']['qualify']);receipt=f.read(q/'stage-result.json')
    receipt.update(status='pending-review',route_qualified=False)
    f.put(q/'stage-result.json',receipt)
    path=q/'audit-result.txt';f.put(path,path.read_text().replace('route_qualified=1','route_qualified=0'))
    path=q/'qualification/summary.tcldata';f.put(path,path.read_text().replace('route_qualified 1','route_qualified 0').replace('pending_reviews {}','pending_reviews {synthetic_pending}'))
    f.put(q/'qualification/NOT_QUALIFIED.txt','retain pending collection')
    helper='boards/c2_f2_alex/routed_host_ready.tcl'
    source=f.source_copy(origin,'qualify',[helper]);f.put(q/'routed_host_ready.tcl',(ROOT/helper).read_bytes())
    si=f.read(q/'stage-inputs.json');si['executor_sources'].update(source)
    si['prepared_files']['routed_host_ready.tcl']=f.rec(q/'routed_host_ready.tcl');f.put(q/'stage-inputs.json',si)
    manifest=f.read(origin/f.e.MANIFEST);f.update_last_receipt(origin,manifest)
    spec['receipt_sha256']['qualify']=f.e.sha(q/'stage-result.json')
    review=origin/'runs/review';f.put(review/'review-decisions.json',{'SYNTHETIC':True})
    source=f.source_copy(origin,'review',['boards/c2_f2_alex/evidence.py'])
    f.put(review/'review-result.json',{'kind':'c2-alex-final-semantic-review','status':'complete','route_qualified':True,'source_revision':'d'*40,'source_inputs':source,'outputs':{'review-decisions.json':f.rec(review/'review-decisions.json')}})
    spec['roots']['review']=str(review);spec['receipt_sha256']['review']=f.e.sha(review/'review-result.json')
    spec['source_checkouts']['review']={'path':str(origin/'sources/review'),'revision':'d'*40}
    return spec,origin,out


def test_schema2_collection_keeps_pending_evidence_and_replays_review(inputs,monkeypatch):
    spec,origin,out=schema2_inputs(inputs);old=c.e.local_module;calls=[]
    def verify(x):
        calls.append(x)
        assert x.text('qualify','qualification/NOT_QUALIFIED.txt')=='retain pending collection'
        return {'SYNTHETIC':'protocol test only'}
    monkeypatch.setattr(c.e,'local_module',lambda name:SimpleNamespace(verify_review=verify) if name=='final_review' else old(name))
    before=f.inventory(origin);receipt=c.collect(spec,out)
    assert f.inventory(origin)==before and calls
    assert receipt['verified']['qualification_receipt_sha256']==spec['receipt_sha256']['review']
    assert receipt['verified']['collection_receipt_sha256']==spec['receipt_sha256']['qualify']
    assert (out/'bundle/runs/qualify/qualification/NOT_QUALIFIED.txt').is_file()


def test_schema2_claimed_review_is_not_enough_to_publish(inputs,monkeypatch):
    spec,origin,out=schema2_inputs(inputs);old=c.e.local_module
    def fail(x):raise ValueError('Raw semantic predicates still unresolved')
    monkeypatch.setattr(c.e,'local_module',lambda name:SimpleNamespace(verify_review=fail) if name=='final_review' else old(name))
    with pytest.raises(ValueError,match='semantic predicates still unresolved'):c.collect(spec,out)
    assert not (out/'evidence.tar').exists() and not (out/'collection.json').exists()
