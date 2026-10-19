"""Offline publishing gates and interrupted-AWS request recovery; no cloud calls."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest
from tools import c2_f2_publish as P


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def tarball(path, manifest, dcp=b'qualified routed checkpoint', extra=None):
    with tarfile.open(path, 'w') as tar:
        payloads = {'to_aws/'+manifest['date']+'.manifest.txt':
                    ''.join(f'{k}={v}\n' for k,v in manifest.items()).encode(),
                    'to_aws/'+manifest['dcp_file_name']: dcp}
        if extra:
            payloads.update(extra)
        for name, data in payloads.items():
            member = tarfile.TarInfo(name);member.size=len(data)
            tar.addfile(member, io.BytesIO(data))


@pytest.fixture
def qualified(tmp_path):
    spec=importlib.util.spec_from_file_location('publisher_test_builder',P.ROOT/'boards/c2_f2/build.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    hdk=tmp_path/'hdk'
    wrapper=hdk/'hdk/cl/examples/cl_mem_perf/design/cl_mem_hbm_wrapper.sv'
    wrapper.parent.mkdir(parents=True);wrapper.write_text('// fake wrapper for offline fixture\n')
    shell=hdk/'hdk/common/shell_stable'
    script=shell/'build/scripts/build_all.tcl';script.parent.mkdir(parents=True);script.write_text('# fixture\n')
    run=tmp_path/'run';bid='1234abcd';cl=module.prepare(run,hdk,shell,bid)
    inputs=P.read_json(run/'inputs.json');inputs['git_commit']=bid+'0'*32;write_json(run/'inputs.json',inputs)
    write_json(run/'invocation.json',{'environment':{'BUILD_FLOW':'BuildAll','BUILD_TAG':bid,
                                                  'SHELL_MODE':'small_shell','VIVADO_TOOL_VERSION':'2025.2'}})
    (run/'exit_code').write_text('0\n');(run/'console.log').write_text(P.MARKER+'\n')
    checkpoints=cl/'build/checkpoints';checkpoints.mkdir(exist_ok=True)
    dcp=checkpoints/f'cl_opentpu.{bid}.post_route.dcp';dcp.write_bytes(b'qualified routed checkpoint')
    marker=checkpoints/f'cl_opentpu.{bid}.route_validation.passed';marker.write_text(P.MARKER+'\n')
    report=cl/f'build/reports/cl_opentpu.{bid}.route_validation.txt';report.parent.mkdir(exist_ok=True)
    report.write_text(''.join(f'{k}=0\n' for k in P.ZERO_FIELDS)+'gray_bus_constraints=39\ncdc_crossing_witnesses=6\n')
    tag='2026_10_19-120000';archive=checkpoints/f'{tag}.Developer_CL.tar'
    manifest=dict(manifest_format_version='2',date=tag,shell_version='0x10212415',hdk_version='2.3.0',
        tool_version='v2025.2',dcp_file_name=f'{tag}.SH_CL_routed.dcp',dcp_hash=P.sha(dcp),
        pci_vendor_id='0x1d0f',pci_device_id='0xf001',pci_subsystem_vendor_id='0xfedc',pci_subsystem_id='0x1d51',
        clock_recipe_a='A1',clock_recipe_b='B2',clock_recipe_c='C0',clock_recipe_hbm='H2')
    tarball(archive,manifest)
    packaging=checkpoints/f'{tag}.packaging.json'
    package=dict(build_id=bid,package_tag=tag,verified_manifest=manifest,inputs_sha256=P.sha(run/'inputs.json'),
        route_validation_marker_sha256=P.sha(marker),routed_checkpoint_sha256=P.sha(dcp),
        archive_sha256=P.sha(archive),archive_bytes=archive.stat().st_size,
        build_script_sha256=P.sha(P.ROOT/'boards/c2_f2/support/boards/f2/build.py'),
        official_packager_sha256='a'*64,archive=str(archive),routed_checkpoint=str(dcp))
    write_json(packaging,package)
    outputs=dict(flow='BuildAll',inputs_sha256=P.sha(run/'inputs.json'),artifacts={
        str(p.relative_to(run)):dict(sha256=P.sha(p),bytes=p.stat().st_size) for p in (dcp,archive)})
    write_json(run/'outputs.json',outputs)
    return dict(run=run,cl=cl,dcp=dcp,archive=archive,marker=marker,report=report,packaging=packaging,manifest=manifest)


def update_archive_hashes(q):
    package=P.read_json(q['packaging']);package['archive_sha256']=P.sha(q['archive']);package['archive_bytes']=q['archive'].stat().st_size
    write_json(q['packaging'],package)
    outputs=P.read_json(q['run']/'outputs.json')
    outputs['artifacts'][str(q['archive'].relative_to(q['run']))]=dict(sha256=P.sha(q['archive']),bytes=q['archive'].stat().st_size)
    write_json(q['run']/'outputs.json',outputs)


def test_qualified_fixture_bound_to_all_artifacts(qualified):
    result=P.verify(qualified['run'])
    assert result['schema']==1 and result['build_id']=='1234abcd'
    assert result['package_sha256']==P.sha(qualified['archive'])
    assert result['dcp_sha256']==P.sha(qualified['dcp'])
    assert all(len(result[k])==64 for k in P.HASH_FIELDS)


@pytest.mark.parametrize('change,match',[
    ('source','Frozen source changed'),('dcp','Output artifact changed'),('marker','marker invalid'),
    ('timing','blocking failures'),('gray','topology'),('crossing','topology'),
    ('synth_only','BuildAll'),('violated','Timing-violated'),('inputs','inputs hash mismatch')])
def test_unqualified_or_changed_run_never_reaches_aws(qualified,tmp_path,change,match):
    q=qualified
    if change=='source':(q['cl']/'design/rtl/c2/otpu_c2_hbm_kernel.sv').write_text('changed')
    elif change=='dcp':q['dcp'].write_bytes(b'changed')
    elif change=='marker':q['marker'].write_text('not passed')
    elif change=='timing':q['report'].write_text(q['report'].read_text().replace('hold_failed=0','hold_failed=1'))
    elif change=='gray':q['report'].write_text(q['report'].read_text().replace('constraints=39','constraints=38'))
    elif change=='crossing':q['report'].write_text(q['report'].read_text().replace('witnesses=6','witnesses=5'))
    elif change=='violated':q['dcp'].with_name('cl_opentpu.1234abcd.post_route.VIOLATED.dcp').write_bytes(b'bad')
    elif change=='inputs':(q['run']/'inputs.json').write_text((q['run']/'inputs.json').read_text()+'\n')
    else:
        outputs=P.read_json(q['run']/'outputs.json');outputs['flow']='SynthCL';write_json(q['run']/'outputs.json',outputs)
    calls=[]
    with pytest.raises(RuntimeError,match=match):P.publish(q['run'],tmp_path/'record.json',cli=lambda *a:calls.append(a))
    assert calls==[]
    assert not (tmp_path/'record.json').exists()


@pytest.mark.parametrize('bad', ['embedded_dcp','clock','traversal'])
def test_archive_content_is_checked_even_when_archive_hashes_agree(qualified,bad):
    q=qualified
    if bad=='embedded_dcp':tarball(q['archive'],q['manifest'],dcp=b'x'*q['dcp'].stat().st_size)
    elif bad=='traversal':tarball(q['archive'],q['manifest'],extra={'../escape':b'bad'})
    else:
        q['manifest']['clock_recipe_a']='A0';tarball(q['archive'],q['manifest'])
        package=P.read_json(q['packaging']);package['verified_manifest']=q['manifest'];write_json(q['packaging'],package)
    update_archive_hashes(q)
    with pytest.raises(RuntimeError,match='Embedded DCP|DCP missing/size mismatch|manifest identity|Unsafe provenance'):
        P.verify(q['run'])


class FakeCLI:
    """Offline scripted service, not a hardware or AWS evidence producer."""
    def __init__(self):self.calls=[];self.lose_create=False;self.record=None;self.state='pending';self.bad_tags=False
    def __call__(self,*args):
        self.calls.append(args)
        if args[:2]==('sts','get-caller-identity'):return dict(Account=P.ACCOUNT,Arn='offline-test')
        if args[:2]==('s3api','head-bucket'):return {}
        if args[:2]==('s3api','get-bucket-location'):return dict(LocationConstraint=None)
        if args[:2]==('s3','cp'):return {}
        if args[:2]==('ec2','create-fpga-image'):
            if self.lose_create:self.lose_create=False;raise TimeoutError('ambiguous create response')
            return dict(FpgaImageId='afi-0123456789abcdef0',FpgaImageGlobalId='agfi-0123456789abcdef0')
        if args[:2]==('ec2','describe-fpga-images'):
            r=self.record
            image=dict(FpgaImageId=r['image_id'],FpgaImageGlobalId=r['global_image_id'],OwnerId=P.ACCOUNT,
                ShellVersion='0x10212415',State=dict(Code=self.state),
                Tags=[dict(Key=k,Value=('bad' if self.bad_tags else v)) for k,v in P.tags(r).items()])
            return dict(FpgaImages=[image])
        raise AssertionError(args)


def test_interrupted_create_resumes_same_unique_destination_and_token(qualified,tmp_path):
    cli=FakeCLI();cli.lose_create=True;record_path=tmp_path/'deployment.json'
    with pytest.raises(TimeoutError):P.publish(qualified['run'],record_path,cli)
    prepared=P.read_json(record_path)
    assert prepared['state']=='prepared' and 'image_id' not in prepared
    assert not any('load' in str(a) for a in cli.calls)
    pending=P.publish(qualified['run'],record_path,cli)
    creates=[json.loads(a[-1]) for a in cli.calls if a[:2]==('ec2','create-fpga-image')]
    assert len(creates)==2 and creates[0]==creates[1]==prepared['publication_request']
    assert pending['prefix']==prepared['prefix'] and pending['client_token']==prepared['client_token']
    count=len(cli.calls)
    assert P.publish(qualified['run'],record_path,cli)==pending and len(cli.calls)==count
    cli.record=pending;cli.state='available'
    available=P.status(record_path,cli)
    assert available['state']=='available' and available['last_describe']['FpgaImages'][0]['State']['Code']=='available'
    assert all(a[0] in ('sts','s3api','s3','ec2') for a in cli.calls)


def test_status_rejects_provenance_tag_drift_without_claiming_available(qualified,tmp_path):
    cli=FakeCLI();record_path=tmp_path/'deployment.json'
    pending=P.publish(qualified['run'],record_path,cli);cli.record=pending;cli.state='available';cli.bad_tags=True
    with pytest.raises(RuntimeError,match='tags mismatch'):P.status(record_path,cli)
    assert P.read_json(record_path)['state']=='pending'


def test_upload_mutation_prevents_create(qualified,tmp_path):
    cli=FakeCLI()
    def mutate(*args):
        response=cli(*args)
        if args[:2]==('s3','cp') and args[2]==str(qualified['archive']):
            with qualified['archive'].open('ab') as f:f.write(b'mutated')
        return response
    with pytest.raises(RuntimeError,match='changed while uploading'):
        P.publish(qualified['run'],tmp_path/'deployment.json',mutate)
    assert not any(a[:2]==('ec2','create-fpga-image') for a in cli.calls)
