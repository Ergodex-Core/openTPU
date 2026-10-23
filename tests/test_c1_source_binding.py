"""Source-role selection after Bundle authentication; no synthetic FPGA approval."""
from copy import deepcopy
import hashlib
import shutil
from types import SimpleNamespace
import pytest
from boards.c1_f2_alex import final_review as F


@pytest.fixture
def selected(tmp_path, monkeypatch):
    root=tmp_path/'bundle';root.mkdir()
    names=['rtl/c1/otpu_c1_hbm_kernel.sv','rtl/c1/otpu_c1_native.sv',
           'boards/c1_f2_alex/support/rtl/boards/f2/otpu_native_axi.sv','rtl/c1_alex/generate.py']
    files={}
    def add(name):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('synthetic bounded selection fixture: '+name)
        files[name]=F.record(path)
        return files[name]['sha256']
    source={name:add('sources/hardware/'+name) for name in names}
    inputs=add('runs/synth/inputs.json');top=add('runs/synth/source/rtl/boards/f2/cl_opentpu.sv')
    revision='1'*40
    module=SimpleNamespace(HARDWARE_REVISION=revision,SOURCE_SHA256=source,SYNTH_INPUT_SHA256=inputs,
                           REVIEWED_SYNTH_INPUT_SHA256=frozenset({inputs}),TOP_SHA256=top)
    original=F.module
    monkeypatch.setattr(F,'module',lambda name:module if name=='admission_specialization_review' else original(name))
    manifest={'hardware':{'source_revision':revision,'build_id':revision[:8]},
              'source_snapshots':{'hardware':{'revision':revision,'bundle_relative_path':'sources/hardware'}},
              'roots':{'synth':{'bundle_relative_path':'runs/synth'}}}
    receipt={'inputs_sha256':inputs}
    bundle=SimpleNamespace(root=root,manifest=manifest,files=files,receipt=lambda role:deepcopy(receipt))
    return bundle,receipt,module


def test_selects_only_authenticated_roles_and_exact_inputs(selected,tmp_path):
    bundle,_,module=selected
    binding=F.hardware_source_binding(bundle)
    assert set(binding['source_paths'])==set(module.SOURCE_SHA256)
    assert binding['synth_inputs_sha256']==module.SYNTH_INPUT_SHA256
    assert all(path.is_relative_to(bundle.root) for path in F.source_binding_files(binding))
    moved=tmp_path/'copied';shutil.copytree(bundle.root,moved)
    relocated=F.hardware_source_binding(bundle,relocated_root=moved)
    assert all(path.is_relative_to(moved) for path in F.source_binding_files(relocated))
    assert [F.record(p) for p in F.source_binding_files(binding)]==[F.record(p) for p in F.source_binding_files(relocated)]


@pytest.mark.parametrize('case',['revision','snapshot_revision','build_id','missing_source','source_sha','source_bytes',
    'source_content','inputs_receipt','inputs_inventory','top_inventory','top_content','snapshot_traversal',
    'synth_traversal','source_symlink','copy_changed','relative_copy','changed_current_source_irrelevant'])
def test_selected_evidence_mutations_fail_closed(selected,tmp_path,case):
    bundle,receipt,module=selected
    name='sources/hardware/'+next(iter(module.SOURCE_SHA256));target=bundle.root/name
    args={}
    if case=='revision':bundle.manifest['hardware']['source_revision']='2'*40
    elif case=='snapshot_revision':bundle.manifest['source_snapshots']['hardware']['revision']='2'*40
    elif case=='build_id':bundle.manifest['hardware']['build_id']='2'*8
    elif case=='missing_source':del bundle.files[name]
    elif case=='source_sha':bundle.files[name]['sha256']='0'*64
    elif case=='source_bytes':bundle.files[name]['bytes']+=1
    elif case=='source_content':target.write_text('changed')
    elif case=='inputs_receipt':receipt['inputs_sha256']='0'*64
    elif case=='inputs_inventory':bundle.files['runs/synth/inputs.json']['sha256']='0'*64
    elif case=='top_inventory':bundle.files['runs/synth/source/rtl/boards/f2/cl_opentpu.sv']['sha256']='0'*64
    elif case=='top_content':(bundle.root/'runs/synth/source/rtl/boards/f2/cl_opentpu.sv').write_text('changed')
    elif case=='snapshot_traversal':bundle.manifest['source_snapshots']['hardware']['bundle_relative_path']='../outside'
    elif case=='synth_traversal':bundle.manifest['roots']['synth']['bundle_relative_path']='../outside'
    elif case=='source_symlink':
        elsewhere=tmp_path/'elsewhere';shutil.copyfile(target,elsewhere);target.unlink();target.symlink_to(elsewhere)
    elif case=='copy_changed':
        moved=tmp_path/'copied';shutil.copytree(bundle.root,moved);(moved/name).write_text('changed');args['relocated_root']=moved
    elif case=='relative_copy':args['relocated_root']='relative'
    else:
        # Selected evidence never falls back to a current workspace source path.
        target.unlink();(tmp_path/target.name).write_text('replacement successful source')
    with pytest.raises((ValueError,OSError)):F.hardware_source_binding(bundle,**args)


@pytest.mark.parametrize('name',['structural_review','admission_specialization_review','admission_pair_review'])
def test_actual_fresh_local_validator_loader_supports_relative_imports(name):
    loaded=F.module(name)
    assert loaded.__package__=='boards.c1_f2_alex'
    assert loaded.__file__==str(F.HERE/(name+'.py'))
    assert loaded is not F.module(name)


@pytest.mark.parametrize('relocated',(False,True))
def test_selects_exact_authenticated_receipt_from_two_reviewed_inputs(selected,relocated):
    bundle,receipt,module=selected
    raw=b'second exact reviewed synthesis input fixture';checksum=hashlib.sha256(raw).hexdigest()
    module.REVIEWED_SYNTH_INPUT_SHA256=frozenset({module.SYNTH_INPUT_SHA256,checksum})
    if relocated:
        path=bundle.root/'runs/synth/inputs.json';path.write_bytes(raw)
        bundle.files['runs/synth/inputs.json']=F.record(path);receipt['inputs_sha256']=checksum
    binding=F.hardware_source_binding(bundle)
    assert binding['synth_inputs_sha256']==receipt['inputs_sha256']
    assert F.record(binding['synth_inputs_path'])['sha256']==receipt['inputs_sha256']


@pytest.mark.parametrize('attack',('allowlisted_receipt_only','unreviewed_coherent_input'))
def test_receipt_allowlist_does_not_replace_inventory_authentication(selected,attack):
    bundle,receipt,module=selected
    raw=b'another synthesis input';checksum=hashlib.sha256(raw).hexdigest()
    receipt['inputs_sha256']=checksum
    if attack=='allowlisted_receipt_only':
        module.REVIEWED_SYNTH_INPUT_SHA256=frozenset({module.SYNTH_INPUT_SHA256,checksum})
    else:
        path=bundle.root/'runs/synth/inputs.json';path.write_bytes(raw)
        bundle.files['runs/synth/inputs.json']=F.record(path)
    with pytest.raises(ValueError):F.hardware_source_binding(bundle)
