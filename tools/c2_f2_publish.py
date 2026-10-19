#!/usr/bin/env python3
"""Verify and publish a qualified C2 BuildAll archive; never load an FPGA.

  python tools/c2_f2_publish.py verify RUN
  python tools/c2_f2_publish.py publish RUN --record deployment.json
  python tools/c2_f2_publish.py status --record deployment.json --wait-seconds 60

Re-running publish with the same record resumes the same immutable request and
ClientToken after interruption. Status is a separate read-only AWS operation.
Only the existing account-owned us-east-1 bucket is supported; no gate override.
AWS API: https://docs.aws.amazon.com/cli/latest/reference/ec2/create-fpga-image.html
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
REGION = 'us-east-1'
ACCOUNT = '438374459156'
BUCKET = 'cg-rachit-afi-438374459156-us-east-1'
MARKER = 'OTPU_STAGE_ROUTE_VALIDATION_PASSED'
ZERO_FIELDS = ('setup_failed', 'hold_failed', 'timing_failed', 'bus_skew_failed',
               'unrouted_nets', 'drc_blockers', 'methodology_blockers')
HASH_FIELDS = ('inputs_sha256', 'outputs_sha256', 'qualified_marker_sha256',
               'validation_report_sha256', 'package_sha256', 'dcp_sha256',
               'packaging_sha256', 'invocation_sha256')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def relative(root, name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and p.parts, f'Unsafe provenance path: {name}')
    return root.joinpath(*p.parts)


def verify(run):
    """Recheck physical qualification and all frozen source/package bindings offline."""
    run = Path(run).resolve()
    inputs, outputs = read_json(run/'inputs.json'), read_json(run/'outputs.json')
    bid = inputs.get('build_id', '')
    require(bool(re.fullmatch('[0-9a-f]{8}', bid)), 'Invalid build ID')
    require(inputs.get('kind') == 'c2-hbm-kernel' and inputs.get('core_khz') == 200000 and
            inputs.get('hbm_ports') == 8 and inputs.get('part') == 'xcvu47p-fsvh2892-2-e', 'Wrong C2 build identity')
    require(inputs.get('git_commit', '').startswith(bid), 'Source revision/build ID mismatch')
    require(outputs.get('flow') == 'BuildAll', 'Only completed BuildAll outputs may be published')
    require(outputs.get('inputs_sha256') == sha(run/'inputs.json'), 'outputs/inputs hash mismatch')
    require((run/'exit_code').read_text().strip() == '0', 'Build process did not succeed')
    require(re.search('^'+MARKER+'$', (run/'console.log').read_text(), re.M), 'Console lacks route qualification')
    support = ROOT/'boards/c2_f2/support'
    require(inputs.get('builder_sha256') == sha(ROOT/'boards/c2_f2/build.py'), 'Build driver differs from reviewed driver')
    require(inputs.get('support_manifest_sha256') == sha(support/'manifest.json') and
            inputs.get('support_commit') == read_json(support/'manifest.json')['commit'], 'Frozen support provenance mismatch')
    cl = run/'cl_opentpu'
    required = {'design/rtl/c2/otpu_c2_hbm_kernel.sv', 'design/rtl/mxu/otpu_c2_mxu.sv',
                'design/rtl/boards/f2/otpu_f2_hbm_mem.sv', 'build/scripts/validate_route.tcl',
                'design/cl_id_defines.vh', 'design/opentpu_params.vh', 'design/otpu_f2_c2_sys.sv'}
    require(required <= inputs.get('sources', {}).keys(), 'Frozen source inventory incomplete')
    for name, digest in inputs['sources'].items():
        require(sha(relative(cl, name)) == digest, f'Frozen source changed: {name}')
    require(sha(cl/'build/scripts/validate_route.tcl') == sha(ROOT/'boards/c2_f2/validate_hbm_route.tcl'),
            'Route qualifier differs from reviewed qualifier')
    require(f".BUILD_ID(32'h{bid}), .CORE_KHZ(200000)" in (cl/'design/opentpu_params.vh').read_text(),
            'Compiled build identity mismatch')
    invocation = read_json(run/'invocation.json')
    env = invocation.get('environment', {})
    require(env.get('BUILD_FLOW') == 'BuildAll' and env.get('BUILD_TAG') == bid and
            env.get('SHELL_MODE') == 'small_shell' and env.get('VIVADO_TOOL_VERSION') == '2025.2',
            'Invocation is not the qualified F2 BuildAll flow')
    checkpoints = cl/'build/checkpoints'
    dcp = checkpoints/f'cl_opentpu.{bid}.post_route.dcp'
    require(not (checkpoints/f'cl_opentpu.{bid}.post_route.VIOLATED.dcp').exists(), 'Timing-violated checkpoint exists')
    marker = checkpoints/f'cl_opentpu.{bid}.route_validation.passed'
    require(marker.read_text().strip() == MARKER, 'Route validation marker invalid')
    validation = cl/f'build/reports/cl_opentpu.{bid}.route_validation.txt'
    checks = dict(line.split('=', 1) for line in validation.read_text().splitlines() if '=' in line)
    require(all(checks.get(k) == '0' for k in ZERO_FIELDS), 'Route validation reports blocking failures')
    # Post-opt retains all eight native write-counter groups, despite read-only
    # kernel traffic. Their skew bounds are required alongside the host counter.
    require(checks.get('gray_bus_constraints') == '47' and checks.get('cdc_crossing_witnesses') == '6',
            'CDC/gray constraint inventory differs from qualified topology')
    artifacts = outputs.get('artifacts', {})
    for name, meta in artifacts.items():
        file = relative(run, name)
        require(file.stat().st_size == meta.get('bytes') and sha(file) == meta.get('sha256'),
                f'Output artifact changed: {name}')
    archives = [relative(run, n) for n in artifacts if n.endswith('.Developer_CL.tar')]
    require(len(archives) == 1 and str(dcp.relative_to(run)) in artifacts and len(artifacts) == 2,
            'Expected exactly the routed DCP and one packaged archive')
    archive = archives[0]
    tag = archive.name.removesuffix('.Developer_CL.tar')
    require(bool(re.fullmatch(r'\d{4}_\d{2}_\d{2}-\d{6}', tag)), 'Invalid official package timestamp')
    packaging = checkpoints/f'{tag}.packaging.json'
    package = read_json(packaging)
    expected_package = dict(build_id=bid, package_tag=tag, inputs_sha256=sha(run/'inputs.json'),
        route_validation_marker_sha256=sha(marker), routed_checkpoint_sha256=sha(dcp),
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        build_script_sha256=sha(support/'boards/f2/build.py'))
    require(all(package.get(k) == v for k, v in expected_package.items()), 'Packaging provenance mismatch')
    require(Path(package.get('archive', '')).name == archive.name and
            Path(package.get('routed_checkpoint', '')).name == dcp.name, 'Packaged artifact names mismatch')
    require(re.fullmatch('[0-9a-f]{64}', package.get('official_packager_sha256', '')), 'Missing official packager hash')
    with tarfile.open(archive) as tar:
        members = {}
        for member in tar.getmembers():
            name = member.name.removeprefix('./')
            if name in ('', '.') and member.isdir():
                continue
            relative(Path('.'), name)
            require(member.isfile() or member.isdir(), 'Archive contains link or special file')
            require(name not in members, 'Duplicate archive member')
            members[name] = member
        manifest_name = f'to_aws/{tag}.manifest.txt'
        require(manifest_name in members and members[manifest_name].size < 65536, 'Missing/oversized AFI manifest')
        manifest = dict(line.split('=', 1) for line in tar.extractfile(members[manifest_name]).read().decode().splitlines() if '=' in line)
        require(manifest == package.get('verified_manifest'), 'Archive manifest differs from packaging record')
        ids = dict(re.findall(r"`define\s+(CL_SH_ID[01])\s+32'h([0-9a-fA-F_]+)", (cl/'design/cl_id_defines.vh').read_text()))
        id0, id1 = (int(ids[name].replace('_', ''), 16) for name in ('CL_SH_ID0', 'CL_SH_ID1'))
        expected = dict(manifest_format_version='2', date=tag, shell_version='0x10212415', hdk_version='2.3.0',
            dcp_file_name=f'{tag}.SH_CL_routed.dcp', dcp_hash=sha(dcp),
            pci_vendor_id=f'0x{id0&0xffff:04x}', pci_device_id=f'0x{id0>>16:04x}',
            pci_subsystem_vendor_id=f'0x{id1&0xffff:04x}', pci_subsystem_id=f'0x{id1>>16:04x}',
            clock_recipe_a='A1', clock_recipe_b='B2', clock_recipe_c='C0', clock_recipe_hbm='H2')
        require(all(manifest.get(k, '').lower() == v.lower() for k,v in expected.items()) and
                manifest.get('tool_version', '').lower().removeprefix('v') == '2025.2', 'AFI manifest identity/clock/hash mismatch')
        member = members.get('to_aws/'+manifest['dcp_file_name'])
        require(member is not None and member.isfile() and member.size == dcp.stat().st_size, 'Archive DCP missing/size mismatch')
        with tar.extractfile(member) as stream:
            require(hashlib.file_digest(stream, 'sha256').hexdigest() == sha(dcp), 'Embedded DCP differs from routed DCP')
    return dict(schema=1, build_id=bid, source_revision=inputs['git_commit'],
        inputs_sha256=sha(run/'inputs.json'), outputs_sha256=sha(run/'outputs.json'),
        qualified_marker_sha256=sha(marker), validation_report_sha256=sha(validation),
        package_sha256=sha(archive), dcp_sha256=sha(dcp), packaging_sha256=sha(packaging),
        invocation_sha256=sha(run/'invocation.json'), archive=str(archive),
        package_bytes=archive.stat().st_size, run=str(run),
        provenance_files=[str(p) for p in (run/'inputs.json', run/'outputs.json', packaging, marker, validation, run/'invocation.json')])


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(value, f, indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


@contextmanager
def locked(record):
    path = Path(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix+'.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def aws(*args):
    command = ['aws', '--region', REGION, '--output', 'json', '--no-cli-pager', '--no-cli-auto-prompt', *args]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def tags(record):
    return {'c2:build_id': record['build_id'], 'c2:inputs_sha256': record['inputs_sha256'],
            'c2:package_sha256': record['package_sha256'], 'c2:client_token': record['client_token']}


def check_record(record):
    require(record.get('schema') == 1 and record.get('region') == REGION and record.get('bucket') == BUCKET,
            'Wrong deployment schema/destination')
    require(all(re.fullmatch('[0-9a-f]{64}', record.get(k, '')) for k in HASH_FIELDS), 'Deployment hashes missing')
    require(re.fullmatch(r'c2/[0-9a-f]{8}/[0-9a-f]{32}', record.get('prefix', '')), 'Invalid deployment prefix')
    require(record['prefix'].startswith('c2/'+record.get('build_id', '')+'/'), 'Deployment prefix/build mismatch')
    token = hashlib.sha256((record['prefix']+record['package_sha256']+record['inputs_sha256']).encode()).hexdigest()
    require(record.get('client_token') == token, 'Invalid deployment token')
    expected = make_request(record)
    require(record.get('publication_request') == expected, 'Publication request binding changed')


def make_request(record):
    return dict(InputStorageLocation=dict(Bucket=BUCKET, Key=record['prefix']+'/design.tar'),
        LogsStorageLocation=dict(Bucket=BUCKET, Key=record['prefix']+'/logs'),
        Name='c2-hbm-'+record['build_id'], Description='C2 kernel archive SHA256 '+record['package_sha256'],
        ClientToken=record['client_token'], TagSpecifications=[dict(ResourceType='fpga-image',
            Tags=[dict(Key=k, Value=v) for k,v in tags(record).items()])])


def publish(run, record_path, cli=aws):
    with locked(record_path):
        qualified = verify(run)
        if Path(record_path).exists():
            record = read_json(record_path)
            check_record(record)
            require(all(record[k] == qualified[k] for k in HASH_FIELDS+('build_id', 'source_revision')),
                    'Deployment record belongs to another qualified artifact')
            if record.get('image_id'):
                return record  # Status is deliberately separate; never create a second AFI.
        else:
            record = qualified | dict(region=REGION, bucket=BUCKET, state='prepared',
                created_at=datetime.now(timezone.utc).isoformat(), prefix=f'c2/{qualified["build_id"]}/{uuid.uuid4().hex}')
            record['client_token'] = hashlib.sha256((record['prefix']+record['package_sha256']+record['inputs_sha256']).encode()).hexdigest()
            record['publication_request'] = make_request(record)
            atomic_json(record_path, record)  # Durable intent before any upload/create.
        identity = cli('sts', 'get-caller-identity')
        require(identity.get('Account') == ACCOUNT, 'Wrong AWS account for C2 publication')
        cli('s3api', 'head-bucket', '--bucket', BUCKET, '--expected-bucket-owner', ACCOUNT)
        location = cli('s3api', 'get-bucket-location', '--bucket', BUCKET, '--expected-bucket-owner', ACCOUNT)
        require(location.get('LocationConstraint') in (None, REGION), 'AFI bucket is not in us-east-1')
        files = [(Path(qualified['archive']), 'design.tar')]
        files += [(Path(p), 'provenance/'+Path(p).name) for p in qualified['provenance_files']]
        for path, key in files:
            digest = sha(path)
            cli('s3', 'cp', str(path), f's3://{BUCKET}/{record["prefix"]}/{key}',
                '--only-show-errors', '--checksum-algorithm', 'SHA256', '--metadata', 'sha256='+digest)
            require(sha(path) == digest, 'File changed while uploading')
        # Recheck gates immediately before the asynchronous create call.
        checked = verify(run)
        require(all(checked[k] == record[k] for k in HASH_FIELDS), 'Qualified build changed during publication')
        record['caller_identity'] = identity
        atomic_json(record_path, record)
        response = cli('ec2', 'create-fpga-image', '--cli-input-json', json.dumps(record['publication_request']))
        require(re.fullmatch(r'afi-[0-9a-f]+', response.get('FpgaImageId', '')) and
                re.fullmatch(r'agfi-[0-9a-f]+', response.get('FpgaImageGlobalId', '')), 'Create response lacks AFI/AGFI')
        record.update(image_id=response['FpgaImageId'], global_image_id=response['FpgaImageGlobalId'],
                      state='pending', create_response=response)
        atomic_json(record_path, record)
        return record


def status(record_path, cli=aws):
    with locked(record_path):
        record = read_json(record_path)
        check_record(record)
        require(re.fullmatch(r'afi-[0-9a-f]+', record.get('image_id', '')), 'Resume publish first: no saved AFI')
        response = cli('ec2', 'describe-fpga-images', '--fpga-image-ids', record['image_id'])
        images = response.get('FpgaImages', [])
        require(len(images) == 1, 'Expected exactly one AFI status result')
        image = images[0]
        require(image.get('FpgaImageId') == record['image_id'] and
                image.get('FpgaImageGlobalId') == record.get('global_image_id') and image.get('OwnerId') == ACCOUNT,
                'AFI/AGFI/owner binding mismatch')
        actual_tags = {v['Key']: v['Value'] for v in image.get('Tags', [])}
        require(all(actual_tags.get(k) == v for k,v in tags(record).items()), 'AFI provenance tags mismatch')
        state = image.get('State', {}).get('Code')
        require(state in ('pending','available','failed','unavailable'), 'Unknown AFI state')
        if state == 'available':
            require(image.get('ShellVersion', '').lower() == '0x10212415', 'Available AFI shell mismatch')
        record.update(state=state, last_describe=response, checked_at=datetime.now(timezone.utc).isoformat())
        atomic_json(record_path, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('verify').add_argument('run', type=Path)
    pub = sub.add_parser('publish');pub.add_argument('run', type=Path);pub.add_argument('--record', type=Path, required=True)
    stat = sub.add_parser('status');stat.add_argument('--record', type=Path, required=True)
    stat.add_argument('--wait-seconds', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'verify':
        print(json.dumps(verify(args.run), indent=2))
    elif args.action == 'publish':
        print(json.dumps(publish(args.run, args.record), indent=2))
    else:
        require(args.wait_seconds >= 0, 'Wait duration must be nonnegative')
        deadline = time.monotonic()+args.wait_seconds
        while True:
            record = status(args.record)
            print(json.dumps({k:record[k] for k in ('image_id','global_image_id','state','checked_at')}), flush=True)
            if record['state'] != 'pending' or time.monotonic() >= deadline:
                break
            time.sleep(min(15, max(0, deadline-time.monotonic())))
        if record['state'] in ('failed','unavailable'):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
