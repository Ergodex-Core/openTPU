"""Candidate-copy tests; synthetic receipt fixtures never become FPGA evidence."""
import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
c=load('c1_candidate_collect',ROOT/'boards/c1_f2_alex/collect_candidate.py')
f=load('c1_candidate_fixture',ROOT/'tests/test_c1_candidate_evidence.py')

@pytest.fixture
def inputs(tmp_path,monkeypatch):
    origin,m,_=f.fixture_bundle(tmp_path,monkeypatch,original_base=str(tmp_path/'portable/runs'))
    monkeypatch.setattr(c,'e',f.e)
    class FixtureGit:
        def __init__(self,path,revision):self.path=Path(path)
        def read(self,name):return (self.path/name).read_bytes()
    monkeypatch.setattr(c,'GitSource',FixtureGit)
    spec={'schema':1,'kind':'c1-alex-candidate-collection-inputs','instance_id':c.e.INSTANCE,
          'roots':{role:row['original_absolute_path'] for role,row in m['roots'].items()},
          'receipt_sha256':{role:row['sha256'] for role,row in m['receipts'].items()},
          'source_checkouts':{role:{'path':str(origin/row['bundle_relative_path']),'revision':row['revision']}
                              for role,row in m['source_snapshots'].items()}}
    return spec,origin,tmp_path/'collection'


@pytest.mark.parametrize('builder',(c.e.INSTANCE,'i-0ec8d33362b3100be'))
def test_pending_collection_preserves_raw_and_relocates(inputs,tmp_path,builder):
    spec,origin,out=inputs;spec['instance_id']=builder;before=f.inventory(origin);receipt=c.collect(spec,out)
    assert receipt['state']=='complete' and receipt['hardware_actions'] is False
    assert receipt['instance_id']==receipt['verified']['instance_id']==builder
    assert f.read(out/'bundle'/c.e.MANIFEST)['instance_id']==builder
    assert receipt['verified']['route_qualified'] is False and receipt['verified']['review_required'] is True
    assert f.inventory(origin)==before
    for role,name in c.e.RECEIPTS.items():
        assert (out/'bundle'/'runs'/role/name).read_bytes()==(Path(spec['roots'][role])/name).read_bytes()
    shutil.move(origin,tmp_path/'original-moved-away')
    assert c.e.verify_bundle(out/'bundle',receipt['manifest_sha256'])==receipt['verified']
    assert c.e.inspect_archive(out/'evidence.tar',receipt['manifest_sha256'])['route_qualified'] is False


@pytest.mark.parametrize('change',['wrong_hash','false_approval','missing_report','extra_stage_file','mutated_source','symlink'])
def test_failure_preserves_originals_and_never_overwrites(inputs,change):
    spec,origin,out=inputs
    if change=='wrong_hash':spec['receipt_sha256']['qualify']='0'*64
    elif change=='false_approval':
        p=Path(spec['roots']['qualify'])/'stage-result.json';r=f.read(p);r.update(route_qualified=True,status='complete');f.put(p,r);spec['receipt_sha256']['qualify']=c.e.sha(p)
    elif change=='missing_report':(Path(spec['roots']['qualify'])/'qualification/cdc.rpt').unlink()
    elif change=='extra_stage_file':f.put(Path(spec['roots']['refine'])/'unbound.log','unbound')
    elif change=='mutated_source':f.put(origin/'sources/hardware/rtl/c1/otpu_c1_native.sv','changed')
    else:
        p=Path(spec['roots']['ip'])/'ip/ip_project/child.dcp';saved=p.read_bytes();p.unlink();target=origin/'same-bytes.dcp';target.write_bytes(saved);p.symlink_to(target)
    with pytest.raises((ValueError,FileNotFoundError)):c.collect(spec,out)
    assert (out/'failure.json').is_file() and not (out/'collection.json').exists()
    assert not (out/'evidence.tar').exists()
    failure=(out/'failure.json').read_bytes()
    with pytest.raises(ValueError):c.collect(spec,out)
    assert (out/'failure.json').read_bytes()==failure


def test_changed_input_during_copy_fails_closed(inputs,monkeypatch):
    spec,_,out=inputs;original=c.copy_file;changed=[]
    def copying(path,dest,expected):
        original(path,dest,expected)
        if not changed:changed.append(path);path.write_bytes(path.read_bytes()+b'changed')
    monkeypatch.setattr(c,'copy_file',copying)
    with pytest.raises(ValueError,match='changed during collection'):c.collect(spec,out)
    assert not (out/'bundle'/c.e.MANIFEST).exists() and not (out/'evidence.tar').exists()


def test_external_spec_hash_required_before_output(inputs,tmp_path):
    spec,_,out=inputs;p=tmp_path/'spec.json';f.put(p,spec)
    with pytest.raises(ValueError,match='independently trusted'):c.main(['--spec',str(p),'--spec-sha256','0'*64,'--out',str(out)])
    assert not out.exists()


@pytest.mark.parametrize('builder',('unapproved-instance','i-0c78de2253f66519d',None,True,[]))
def test_unapproved_builder_rejected_before_output(inputs,builder):
    spec,_,out=inputs;spec['instance_id']=builder
    with pytest.raises(ValueError):c.collect(spec,out)
    assert not out.exists()


def test_builder_change_cannot_reuse_pinned_spec_hash(inputs,tmp_path):
    spec,_,out=inputs;p=tmp_path/'spec.json';f.put(p,spec);trusted=c.e.sha(p)
    spec['instance_id']='i-0ec8d33362b3100be';f.put(p,spec)
    with pytest.raises(ValueError,match='independently trusted'):
        c.main(['--spec',str(p),'--spec-sha256',trusted,'--out',str(out)])
    assert not out.exists()


def test_git_source_ignores_parent_repository_and_config_overrides(tmp_path,monkeypatch):
    for key,value in {'GIT_DIR':'/wrong/repository','GIT_WORK_TREE':'/wrong/tree',
                      'GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'core.hooksPath',
                      'GIT_CONFIG_VALUE_0':'/wrong/hooks','GIT_OBJECT_DIRECTORY':'/wrong/objects'}.items():
        monkeypatch.setenv(key,value)
    monkeypatch.setenv('C1_TEST_PRESERVED','yes')
    seen={}
    def run(argv,*,env,timeout):
        seen.update(argv=argv,env=env,timeout=timeout)
        return b'bound blob'
    monkeypatch.setattr(c.subprocess,'check_output',run)
    source=object.__new__(c.GitSource);source.path=tmp_path
    assert source.command('cat-file','blob','a'*40)==b'bound blob'
    assert seen['argv'][seen['argv'].index('-C')+1]==str(tmp_path)
    assert seen['env']['C1_TEST_PRESERVED']=='yes'
    assert seen['env']['GIT_NO_REPLACE_OBJECTS']=='1'
    assert not set(seen['env']) & {'GIT_DIR','GIT_WORK_TREE','GIT_CONFIG_COUNT','GIT_CONFIG_KEY_0',
                                    'GIT_CONFIG_VALUE_0','GIT_OBJECT_DIRECTORY'}
