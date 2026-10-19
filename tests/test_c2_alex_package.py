"""Offline packaging tests: synthetic DCP/evidence, no vendor run or hardware.

The package subprocess uses a small fixture packager for normal tests. One
optional local test invokes the exact official source retained by the read-only
HDK inventory, still with a synthetic DCP and mock Vivado version executable.
"""
from __future__ import annotations
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('c2_alex_package', ROOT / 'boards/c2_f2_alex/package.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
TAG = '2026_10_19-183000'
FIXTURE = '''import hashlib,os,pathlib,tarfile
def generate_dcp_tarball(cl,tag,a,b,c,h):
    cp=pathlib.Path(os.environ['CL_DIR'])/'build/checkpoints'
    dcp=cp/(cl+'.'+tag+'.post_route.dcp')
    dest=cp/'to_aws';dest.mkdir()
    (dest/(tag+'.SH_CL_routed.dcp')).write_bytes(dcp.read_bytes())
    manifest=dict(manifest_format_version='2',date=tag,shell_version='0x10212415',hdk_version='2.3.0',
      tool_version='v2025.2',dcp_file_name=tag+'.SH_CL_routed.dcp',dcp_hash=hashlib.sha256(dcp.read_bytes()).hexdigest(),
      pci_vendor_id='0x1d0f',pci_device_id='0xf001',pci_subsystem_vendor_id='0xfedd',pci_subsystem_id='0x1d51',
      clock_recipe_a=a,clock_recipe_b=b,clock_recipe_c=c,clock_recipe_hbm=h)
    (dest/(tag+'.manifest.txt')).write_text(''.join(k+'='+v+'\\n\\n' for k,v in manifest.items()))
    with tarfile.open(cp/(tag+'.Developer_CL.tar'),'w') as t:t.add(dest,arcname='./to_aws')
    import shutil;shutil.rmtree(dest)
'''


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


@pytest.fixture
def context(tmp_path, monkeypatch):
    tmp_path = tmp_path.resolve()
    root = tmp_path / 'evidence'; root.mkdir()
    dcp = root / 'runs/refine/post_route.dcp'; write(dcp, 'SYNTHETIC DCP, NOT VALID FPGA CONTENT')
    top = root / 'runs/synth/source/rtl/boards/f2/cl_opentpu.sv'
    ids = dict(cl_sh_id0='f0011d0f', cl_sh_id1='1d51fedd', cl_sh_status0='43324842', cl_sh_status1='264d8413')
    write(top, '\n'.join(f"assign {k}=32'h{v};" for k,v in ids.items())+"\notpu_c2_hbm_kernel #(.BUILD_ID(32'h264d8413)) u_board();\n")
    manifest = {'roots': {'synth': {'bundle_relative_path': 'runs/synth'}},
                'files': {str(f.relative_to(root)): p.record(f) for f in (dcp, top)}}
    write(root / p.e.MANIFEST, manifest); manifest_sha = p.e.sha(root / p.e.MANIFEST)
    evidence = dict(schema=1, kind='verified-c2-alex-portable-evidence', instance_id=p.e.INSTANCE,
        manifest_sha256=manifest_sha, source_revision='264d84134da11630622002cd869cf88b07e49aec',
        build_id='264d8413', route_qualified=True, hardware_measured=False, final_dcp_sha256=p.e.sha(dcp),
        final_dcp_relative_path=str(dcp.relative_to(root)), synth_inputs_sha256='a'*64,
        synth_inputs_relative_path='runs/synth/inputs.json', ip_receipt_sha256='b'*64,
        simulation_receipt_sha256='c'*64, qualification_receipt_sha256='d'*64, files=2, bytes=123)
    # The separate verifier has full graph tests. This fixture isolates package
    # behavior at its explicitly typed, hash-bound verified-evidence boundary.
    def verify(root_arg, wanted):
        assert p.e.sha(root_arg / p.e.MANIFEST) == wanted
        return dict(evidence)
    def parse(raw, wanted):
        assert hashlib.sha256(raw).hexdigest() == wanted
        return json.loads(raw)
    monkeypatch.setattr(p.e, 'verify_bundle', verify)
    monkeypatch.setattr(p.e, 'manifest_data', parse)
    hdk = tmp_path / 'hdk'; hdk.mkdir()
    write(hdk / p.MODULE, FIXTURE)
    monkeypatch.setattr(p, 'OFFICIAL_SHA256', p.e.sha(hdk / p.MODULE))
    monkeypatch.setattr(p, 'OFFICIAL_BYTES', (hdk / p.MODULE).stat().st_size)
    write(hdk / p.SHELL_VERSION, 'small_shell=0x10212415\n')
    write(hdk / 'release_version.txt', 'RELEASE_VERSION=2.3.0\n')
    monkeypatch.setattr(p, 'git_revision', lambda path: p.IP_REVISION if path.name == 'ip' else p.HDK_REVISION)
    monkeypatch.setattr(p, 'git_file', lambda path, name: b'RELEASE_VERSION=2.3.0\n')
    vivado = tmp_path / 'tool/vivado'; write(vivado, '#!/bin/sh\nprintf "Vivado v2025.2 (64-bit)\\n"\n'); vivado.chmod(0o700)
    monkeypatch.setattr(p, 'VIVADO', vivado)
    return SimpleNamespace(root=root, hdk=hdk, out=tmp_path/'package', evidence=evidence, manifest=manifest,
                           manifest_sha=manifest_sha, dcp=dcp, top=top, tmp=tmp_path)


def package(c):
    return p.package(c.root, c.manifest_sha, c.hdk, c.out, tag=TAG)


def test_package_subprocess_and_portable_reverification(context):
    c=context; result=package(c)
    assert result['inputs_sha256']=='a'*64 and result['build_id']=='264d8413'
    assert result['package_sha256']==p.e.sha(Path(result['archive']))
    copied=c.out/f'cl_opentpu/build/checkpoints/cl_opentpu.{TAG}.post_route.dcp'
    assert copied.read_bytes()==c.dcp.read_bytes() and copied.stat().st_ino!=c.dcp.stat().st_ino
    command=json.loads((c.out/'command.json').read_text())
    assert command[1:3]==['-I','-c'] and str(c.hdk/p.MODULE) in command[3]
    assert str(c.root) not in command[3]
    assert len(tarfile.open(result['archive']).getmembers())==3
    # Verification is offline and does not need the original installed HDK.
    shutil.rmtree(c.hdk)
    relocated=c.tmp/'relocated';shutil.copytree(c.out,relocated)
    checked=p.verify_package(relocated,c.root,c.manifest_sha,result['package_receipt_sha256'])
    assert checked['package_sha256']==result['package_sha256']
    assert checked['archive'].startswith(str(relocated))


@pytest.mark.parametrize('change',['pending','hardware','wrong_pci','wrong_status','wrong_kernel_id','duplicate_id','dcp'])
def test_unqualified_or_wrong_compiled_identity_never_invokes_packager(context,monkeypatch,change):
    c=context
    if change=='pending':c.evidence['route_qualified']=False
    elif change=='hardware':c.evidence['hardware_measured']=True
    elif change=='dcp':c.dcp.write_text('changed')
    else:
        text=c.top.read_text()
        if change=='wrong_pci':text=text.replace('f0011d0f','f0021d0f')
        elif change=='wrong_status':text=text.replace("cl_sh_status1=32'h264d8413","cl_sh_status1=32'h11111111")
        elif change=='wrong_kernel_id':text=text.replace(".BUILD_ID(32'h264d8413)",".BUILD_ID(32'h11111111)")
        else:text+="\nassign cl_sh_id0=32'hf0011d0f;\n"
        c.top.write_text(text);c.manifest['files'][str(c.top.relative_to(c.root))]=p.record(c.top)
        write(c.root/p.e.MANIFEST,c.manifest);c.manifest_sha=p.e.sha(c.root/p.e.MANIFEST)
    monkeypatch.setattr(p,'official_inputs',lambda *a:pytest.fail('qualification must gate official invocation'))
    with pytest.raises(RuntimeError):package(c)
    assert not c.out.exists()


@pytest.mark.parametrize('change',['script','release','shell','revision','tool'])
def test_installed_official_inputs_fail_closed(context,monkeypatch,change):
    c=context
    if change=='script':write(c.hdk/p.MODULE,'not reviewed')
    elif change=='release':write(c.hdk/'release_version.txt','RELEASE_VERSION=9.0\n')
    elif change=='shell':write(c.hdk/p.SHELL_VERSION,'small_shell=0x1\n')
    elif change=='revision':monkeypatch.setattr(p,'git_revision',lambda path:'0'*40)
    else:write(p.VIVADO,'#!/bin/sh\necho "Vivado v2024.1"\n');p.VIVADO.chmod(0o700)
    with pytest.raises(RuntimeError):package(c)
    assert not c.out.exists()


@pytest.mark.parametrize('name',['package.json','package-inputs.json','official/aws_build_dcp_from_cl.py',
    'official/shell_version.txt','official/release_version.txt','official/vivado_version.txt',
    'console.log','exit.json','command.json',f'cl_opentpu/build/checkpoints/{TAG}.Developer_CL.tar'])
def test_package_corruption_refused(context,name):
    c=context;result=package(c)
    with (c.out/name).open('ab') as f:f.write(b'changed')
    with pytest.raises((RuntimeError,ValueError)):
        p.verify_package(c.out,c.root,c.manifest_sha,result['package_receipt_sha256'])


def test_official_mutation_during_process_retains_failure(context,monkeypatch):
    c=context;real=p.subprocess.run
    def run(*a,**kw):
        result=real(*a,**kw);write(c.hdk/p.MODULE,'mutated during packaging');return result
    monkeypatch.setattr(p.subprocess,'run',run)
    with pytest.raises(RuntimeError):package(c)
    assert (c.out/'failure.json').is_file() and not (c.out/'package.json').exists()


@pytest.mark.parametrize('change',['extra','duplicate','link','traversal','dcp','manifest','field','directory'])
def test_exact_tar_inventory_and_manifest(context,change):
    c=context;result=package(c);archive=Path(result['archive'])
    with tarfile.open(archive) as t:
        data=[(m,t.extractfile(m).read() if m.isfile() else b'') for m in t]
    if change=='extra':
        item=tarfile.TarInfo('to_aws/extra');item.size=1;data.append((item,b'x'))
    elif change=='duplicate':data.append(data[-1])
    elif change=='link':data[-1][0].type=tarfile.SYMTYPE;data[-1][0].linkname='/outside'
    elif change=='traversal':data[-1][0].name='../to_aws/'+Path(data[-1][0].name).name
    elif change=='directory':data[0][0].name='extra-dir'
    else:
        for i,(member,raw) in enumerate(data):
            if (change=='dcp' and member.name.endswith('.dcp')) or (change in ('manifest','field') and member.name.endswith('.manifest.txt')):
                raw=(raw+b'changed') if change=='dcp' else (raw.replace(b'clock_recipe_hbm=H2',b'clock_recipe_hbm=H0') if change=='manifest' else raw+b'date=duplicate\n')
                member.size=len(raw);data[i]=(member,raw)
    with tarfile.open(archive,'w') as t:
        for member,raw in data:t.addfile(member,io.BytesIO(raw) if member.isfile() else None)
    with pytest.raises(RuntimeError):p.verify_archive(archive,TAG,c.dcp,dict(cl_sh_id0='f0011d0f',cl_sh_id1='1d51fedd'))


@pytest.mark.parametrize('name',['bad path','bad;command','bad$substitution'])
def test_official_shell_unsafe_paths_rejected(context,name):
    with pytest.raises(RuntimeError,match='Unsafe'):
        p.package(context.root,context.manifest_sha,context.hdk,context.tmp/name)


def test_existing_and_evidence_nested_outputs_rejected(context):
    c=context;c.out.mkdir()
    with pytest.raises(RuntimeError,match='existing'):package(c)
    with pytest.raises(RuntimeError,match='must not mutate'):
        p.package(c.root,c.manifest_sha,c.hdk,c.root/'nested')


def test_concatenated_tar_cannot_hide_extra_member(context):
    result=package(context);archive=Path(result['archive'])
    extra=context.tmp/'extra.tar'
    with tarfile.open(extra,'w') as t:t.addfile(tarfile.TarInfo('hidden'))
    with archive.open('ab') as stream:stream.write(extra.read_bytes())
    with pytest.raises(RuntimeError,match='archive member'):
        p.verify_archive(archive,TAG,context.dcp,dict(cl_sh_id0='f0011d0f',cl_sh_id1='1d51fedd'))


def test_actual_official_function_from_readonly_inventory_when_available(context,monkeypatch):
    path=ROOT/'build/c2-alex-remote/read-official-packager-result.json'
    if not path.exists():pytest.skip('Optional exact installed-script inventory unavailable')
    row=json.loads(json.loads(path.read_text())['StandardOutputContent'])
    source=row['source'].encode()
    assert hashlib.sha256(source).hexdigest()=='bc86136e98de039029ecf202d6a676c8006ab0de3dd08adfa5825ec2e0327bdf'
    assert len(source)==13822
    (context.hdk/p.MODULE).write_bytes(source)
    monkeypatch.setattr(p,'OFFICIAL_SHA256',row['sha256']);monkeypatch.setattr(p,'OFFICIAL_BYTES',len(source))
    if not Path('/usr/bin/sha256sum').exists():
        # macOS fixture adapter for the official Linux utility dependency; the
        # reviewed HDK function itself remains byte-for-byte unchanged.
        shim=p.VIVADO.parent/'sha256sum'
        write(shim,'#!/bin/sh\nexec /usr/bin/shasum -a 256 "$@"\n');shim.chmod(0o700)
        monkeypatch.setenv('COPYFILE_DISABLE','1')  # no macOS-only AppleDouble tar entries
    result=package(context)
    assert result['dcp_sha256']==p.e.sha(context.dcp)
    assert result['package_bytes']>context.dcp.stat().st_size
