"""C1 publication/runtime trust boundaries; synthetic records, no AWS/device access."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('test_c1_' + name, ROOT / 'boards/c1_f2_alex' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


d, a = load('deployment'), load('physical_artifact')


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def records():
    source = '0123456789abcdef0123456789abcdef01234567'
    hashes = {key: hashlib.sha256(key.encode()).hexdigest() for key in d.HASH_FIELDS}
    image = dict(schema=1, kind='c1-alex-fpga-image', account=d.ACCOUNT, region=d.REGION, bucket=d.BUCKET,
        source_revision=source, build_id=source[:8], package_bytes=1234, **hashes,
        prefix='c1-alex/' + source[:8] + '/' + 'a' * 32, state='available',
        image_id='afi-' + '1' * 17, global_image_id='agfi-' + '2' * 17)
    image['client_token'] = d.token(image)
    image['publication_request'] = d.make_request(image)
    live = dict(FpgaImages=[dict(FpgaImageId=image['image_id'], FpgaImageGlobalId=image['global_image_id'],
        OwnerId=d.ACCOUNT, State={'Code': 'available'}, ShellVersion='0x10212415',
        Tags=[dict(Key=k, Value=v) for k, v in d.image_tags(image).items()])])
    image['last_describe'] = deepcopy(live)
    evidence = dict(schema=1, kind='verified-c1-alex-portable-evidence', instance_id='i-06cf8792fb8f15534',
        route_qualified=True, review_required=False, hardware_measured=False,
        source_revision=source, build_id=source[:8], manifest_sha256=hashes['evidence_manifest_sha256'],
        synth_inputs_sha256=hashes['inputs_sha256'], final_dcp_sha256=hashes['dcp_sha256'],
        **{k: hashes[k] for k in ('ip_receipt_sha256', 'simulation_receipt_sha256', 'qualification_receipt_sha256')})
    package = dict(hashes, source_revision=source, build_id=source[:8], package_bytes=1234)
    return SimpleNamespace(image=image, live=live, evidence=evidence, package=package)


def test_qualified_cpu_built_image_requires_matching_available_aws_record(records):
    r = records
    assert d.verify_deployment(r.image, r.evidence)['agfi_id'] == r.image['global_image_id']
    assert d.verify_aws_image(r.live, r.image)['owner'] == d.ACCOUNT


@pytest.mark.parametrize('field,value', [
    ('route_qualified', False), ('review_required', True), ('hardware_measured', True),
    ('schema', True), ('kind', 'verified-c1-alex-portable-candidate-evidence'),
    ('kind', 'verified-c2-alex-portable-evidence'), ('instance_id', 'unapproved-instance'),
    ('source_revision', 'f' * 40), ('final_dcp_sha256', 'f' * 64),
    ('qualification_receipt_sha256', 'f' * 64)])
def test_candidate_or_different_evidence_cannot_authorize_image(records, field, value):
    records.evidence[field] = value
    with pytest.raises(ValueError):
        d.verify_deployment(records.image, records.evidence)


@pytest.mark.parametrize('field,value', [
    ('state', 'pending'), ('state', 'failed'), ('account', '000000000000'),
    ('region', 'us-west-2'), ('bucket', 'different-bucket'), ('schema', True),
    ('kind', 'c2-alex-fpga-image'), ('package_bytes', True),
    ('global_image_id', 'agfi-00c2d61ec51bd89aa'), ('global_image_id', 'agfi-0d06cb48544a02106'),
    ('client_token', 'f' * 64), ('prefix', 'c1-alex/other/' + 'a' * 32)])
def test_wrong_publication_contract_rejected(records, field, value):
    records.image[field] = value
    with pytest.raises(ValueError):
        d.verify_deployment(records.image, records.evidence)


@pytest.mark.parametrize('case', ['owner', 'regional', 'global', 'pending', 'shell', 'missing_tags',
    'wrong_tag', 'duplicate_tag', 'malformed_tag', 'missing_image', 'multiple_images', 'null_state', 'null_shell'])
def test_live_and_saved_image_responses_fail_identically(records, case):
    r = records; row = r.live['FpgaImages'][0]
    if case == 'owner': row['OwnerId'] = '000000000000'
    elif case == 'regional': row['FpgaImageId'] = 'afi-' + '3' * 17
    elif case == 'global': row['FpgaImageGlobalId'] = 'agfi-' + '3' * 17
    elif case == 'pending': row['State']['Code'] = 'pending'
    elif case == 'shell': row['ShellVersion'] = '0xdeadbeef'
    elif case == 'missing_tags': row.pop('Tags')
    elif case == 'wrong_tag': row['Tags'][0]['Value'] = 'wrong'
    elif case == 'duplicate_tag': row['Tags'].append(dict(row['Tags'][0]))
    elif case == 'malformed_tag': row['Tags'][0] = None
    elif case == 'missing_image': r.live['FpgaImages'] = []
    elif case == 'multiple_images': r.live['FpgaImages'].append(deepcopy(row))
    elif case == 'null_state': row['State'] = None
    elif case == 'null_shell': row['ShellVersion'] = None
    with pytest.raises(ValueError): d.verify_aws_image(r.live, r.image)
    r.image['last_describe'] = r.live
    with pytest.raises(ValueError): d.verify_deployment(r.image, r.evidence)


@pytest.fixture
def runtime(tmp_path, monkeypatch, records):
    tmp_path = tmp_path.resolve()
    evidence = tmp_path / 'evidence'; evidence.mkdir()
    package = tmp_path / 'package'; package.mkdir()
    image_path = tmp_path / 'image.json'
    image_sha = write(image_path, records.image)
    request = dict(schema=1, kind='c1-f2-session-artifact-request', instance_id=a.INSTANCE,
        hardware_revision=records.evidence['source_revision'], evidence_directory=str(evidence),
        evidence_manifest_sha256=records.evidence['manifest_sha256'], package_directory=str(package),
        package_receipt_sha256=records.package['package_receipt_sha256'],
        image_record=str(image_path), image_record_sha256=image_sha)
    request_path = tmp_path / 'request.json'; request_sha = write(request_path, request)
    calls = []
    def verify_evidence(root, sha):
        assert (root, sha) == (evidence, records.evidence['manifest_sha256'])
        calls.append('evidence')
        return deepcopy(records.evidence)
    def verify_package(root, evroot, evsha, pkgsha):
        assert (root, evroot, evsha, pkgsha) == (package, evidence, records.evidence['manifest_sha256'], request['package_receipt_sha256'])
        calls.append('package')
        return deepcopy(records.package)
    modules = dict(evidence=SimpleNamespace(verify_bundle=verify_evidence),
                   package=SimpleNamespace(verify_package=verify_package), deployment=d)
    monkeypatch.setattr(a, 'module', lambda name: modules[name])
    return SimpleNamespace(**vars(records), request=request, request_path=request_path, request_sha=request_sha,
        image_path=image_path, modules=modules, calls=calls)


def test_runtime_verifies_all_offline_layers_then_checks_live_image(runtime):
    r = runtime
    result = a.verify_session_artifact(r.request_path, r.request_sha)
    assert r.calls == ['evidence', 'package']
    assert result['identity'] == dict(id=0x43314D42, abi=1, build_id=int(r.evidence['build_id'], 16),
                                    core_khz=200000, max_rows=512, depth=128, credits=32)
    assert a.verify_aws_image(r.live, result)['agfi_id'] == r.image['global_image_id']
    result['qualification_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Runtime publication binding changed'):
        a.verify_aws_image(r.live, result)


@pytest.mark.parametrize('field', list(d.HASH_FIELDS) + ['source_revision', 'build_id', 'package_bytes'])
def test_verified_package_must_match_published_image_in_every_binding(runtime, field):
    r = runtime; r.package[field] = 999 if field == 'package_bytes' else '0' * len(r.package[field])
    with pytest.raises(ValueError, match='Published image differs'):
        a.verify_session_artifact(r.request_path, r.request_sha)


@pytest.mark.parametrize('field,value', [('schema', True), ('instance_id', 'another-instance'),
    ('kind', 'c2-f2-session-artifact-request'), ('hardware_revision', '0' * 40),
    ('evidence_directory', 'relative'), ('image_record_sha256', 'bad')])
def test_invalid_runtime_request_never_reaches_package(runtime, field, value):
    r = runtime; r.request[field] = value; r.request_sha = write(r.request_path, r.request)
    with pytest.raises(ValueError): a.verify_session_artifact(r.request_path, r.request_sha)
    assert 'package' not in r.calls


@pytest.mark.parametrize('field,value', [('route_qualified', False), ('review_required', True),
    ('kind', 'verified-c1-alex-portable-candidate-evidence'), ('schema', True)])
def test_nonqualified_candidate_never_reaches_package(runtime, field, value):
    r = runtime; r.evidence[field] = value
    with pytest.raises(ValueError): a.verify_session_artifact(r.request_path, r.request_sha)
    assert r.calls == ['evidence']


def test_missing_final_verifier_does_not_fall_back_to_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(a, 'HERE', tmp_path.resolve())
    (tmp_path / 'candidate_evidence.py').write_text('raise RuntimeError("must not load candidate")')
    with pytest.raises(ValueError, match='verifier is unavailable'): a.module('evidence')


@pytest.mark.parametrize('case', ['hash', 'duplicate', 'nonfinite', 'symlink', 'hardlink'])
def test_pinned_records_reject_ambiguous_or_mutated_data(tmp_path, case):
    path = tmp_path.resolve() / 'record.json'
    raw = b'{"value":1}'
    if case == 'duplicate': raw = b'{"value":1,"value":2}'
    elif case == 'nonfinite': raw = b'{"value":NaN}'
    path.write_bytes(raw); sha = hashlib.sha256(raw).hexdigest()
    if case == 'hash': path.write_bytes(b'{"value":2}')
    elif case == 'symlink':
        link = path.parent / 'linked.json'; link.symlink_to(path); path = link
    elif case == 'hardlink': (path.parent / 'hardlink.json').hardlink_to(path)
    with pytest.raises(ValueError): a.pinned_record(path, sha)


def test_record_changed_during_deep_verification_is_rejected(runtime):
    r = runtime; original = r.modules['package'].verify_package
    def changed(*args):
        result = original(*args)
        r.request_path.write_text(r.request_path.read_text() + '\n')
        return result
    r.modules['package'].verify_package = changed
    with pytest.raises(ValueError, match='changed or hash differs'):
        a.verify_session_artifact(r.request_path, r.request_sha)
