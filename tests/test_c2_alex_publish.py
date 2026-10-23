"""Publication lifecycle tests with a fake CLI; no network or FPGA actions."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('alex_publish_tested', ROOT / 'tools/c2_f2_alex_publish.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class CLI:
    def __init__(self):
        self.calls = []
        self.objects = {}
        self.account = p.ACCOUNT
        self.interrupt = False
        self.image = None

    def __call__(self, *args):
        self.calls.append(args)
        kind = args[:2]
        def value(flag): return args[args.index(flag) + 1]
        if kind == ('sts', 'get-caller-identity'):
            return {'Account': self.account}
        if kind == ('s3api', 'head-bucket'):
            return {}
        if kind == ('s3api', 'get-bucket-location'):
            return {'LocationConstraint': None}
        if kind == ('s3api', 'put-object'):
            assert value('--if-none-match') == '*'
            assert value('--expected-bucket-owner') == p.ACCOUNT
            key = value('--key')
            if key in self.objects:
                raise subprocess.CalledProcessError(1, args, stderr='PreconditionFailed')
            raw = Path(value('--body')).read_bytes()
            checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode()
            assert checksum == value('--checksum-sha256')
            self.objects[key] = {'ContentLength': len(raw), 'ChecksumSHA256': checksum,
                                 'Metadata': json.loads(value('--metadata')), 'ServerSideEncryption': 'AES256'}
            return {'ETag': 'fake-etag'}
        if kind == ('s3api', 'head-object'):
            return self.objects[value('--key')]
        if kind == ('ec2', 'create-fpga-image'):
            if self.interrupt:
                self.interrupt = False
                raise TimeoutError('response lost after server accepted create')
            request = json.loads(value('--cli-input-json'))
            self.image = {'FpgaImageId': 'afi-' + '1' * 17, 'FpgaImageGlobalId': 'agfi-' + '2' * 17,
                          'OwnerId': p.ACCOUNT, 'Tags': request['TagSpecifications'][0]['Tags'],
                          'State': {'Code': 'available'}, 'ShellVersion': '0x10212415'}
            return {key: self.image[key] for key in ('FpgaImageId', 'FpgaImageGlobalId')}
        if kind == ('ec2', 'describe-fpga-images'):
            return {'FpgaImages': [self.image]}
        raise AssertionError(args)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    package = tmp_path / 'package'; package.mkdir()
    evidence = tmp_path / 'evidence'; evidence.mkdir()
    for path, content in [(package / 'design.tar', b'design'), (package / 'package.json', b'package'),
                          (evidence / 'bundle-manifest.json', b'manifest'), (tmp_path / 'evidence.tar', b'evidence')]:
        path.write_bytes(content)
    checked = {key: hashlib.sha256(key.encode()).hexdigest() for key in p.identity.HASH_FIELDS}
    checked.update(source_revision='a' * 40, build_id='a' * 8,
                   archive=str(package / 'design.tar'), package_sha256=p.common.sha(package / 'design.tar'),
                   package_receipt_sha256=p.common.sha(package / 'package.json'),
                   evidence_manifest_sha256=p.common.sha(evidence / 'bundle-manifest.json'),
                   package_bytes=6, evidence_archive_sha256=p.common.sha(tmp_path / 'evidence.tar'),
                   evidence_archive_bytes=8)
    monkeypatch.setattr(p, 'verify', lambda *args: dict(checked))
    args = (package, evidence, checked['evidence_manifest_sha256'], checked['package_receipt_sha256'],
            tmp_path / 'evidence.tar', tmp_path / 'image.json')
    return args, checked, CLI()


def test_rejected_qualification_precedes_every_cloud_call(prepared, monkeypatch):
    args, _, cli = prepared
    def reject(*args): raise ValueError('pending current-design review')
    monkeypatch.setattr(p, 'verify', reject)
    with pytest.raises(ValueError, match='pending'):
        p.publish(*args, cli=cli)
    assert not cli.calls and not args[-1].exists()


def test_wrong_account_never_uploads(prepared):
    args, _, cli = prepared; cli.account = '000000000000'
    with pytest.raises(ValueError, match='account'):
        p.publish(*args, cli=cli)
    assert [call[:2] for call in cli.calls] == [('sts', 'get-caller-identity')]


def test_resume_lost_response_reuses_request_and_immutable_objects(prepared):
    args, _, cli = prepared; cli.interrupt = True
    with pytest.raises(TimeoutError): p.publish(*args, cli=cli)
    intent = p.read_record(args[-1]); assert intent['state'] == 'prepared'
    result = p.publish(*args, cli=cli)
    creates = [call for call in cli.calls if call[:2] == ('ec2', 'create-fpga-image')]
    assert len(creates) == 2 and creates[0] == creates[1]
    assert result['client_token'] == intent['client_token']
    assert result['state'] == 'pending' and len(cli.objects) == 4
    assert any(key.endswith('/provenance/evidence.tar') for key in cli.objects)
    count = len(cli.calls)
    assert p.publish(*args, cli=cli)['image_id'] == result['image_id']
    assert len(cli.calls) == count


def test_existing_wrong_object_is_never_overwritten(prepared):
    args, _, cli = prepared; cli.interrupt = True
    with pytest.raises(TimeoutError): p.publish(*args, cli=cli)
    first = next(iter(cli.objects)); cli.objects[first]['ChecksumSHA256'] = 'wrong'
    count = sum(call[:2] == ('ec2', 'create-fpga-image') for call in cli.calls)
    with pytest.raises(ValueError, match='readback'):
        p.publish(*args, cli=cli)
    assert cli.objects[first]['ChecksumSHA256'] == 'wrong'
    assert sum(call[:2] == ('ec2', 'create-fpga-image') for call in cli.calls) == count


def test_reverification_blocks_changed_qualified_inputs(prepared, monkeypatch):
    args, checked, cli = prepared; calls = []
    def changing(*args):
        calls.append(1)
        return checked | ({'dcp_sha256': 'f' * 64} if len(calls) > 1 else {})
    monkeypatch.setattr(p, 'verify', changing)
    with pytest.raises(ValueError, match='changed during'):
        p.publish(*args, cli=cli)
    assert len(cli.objects) == 4 and not any(call[0] == 'ec2' for call in cli.calls)


def test_bound_record_rejects_another_package(prepared, monkeypatch):
    args, checked, cli = prepared
    p.publish(*args, cli=cli); before = len(cli.calls)
    monkeypatch.setattr(p, 'verify', lambda *args: checked | {'package_sha256': '0' * 64})
    with pytest.raises(ValueError, match='another qualified package'):
        p.publish(*args, cli=cli)
    assert len(cli.calls) == before


@pytest.mark.parametrize('field,value', [('OwnerId', '0' * 12), ('FpgaImageGlobalId', 'agfi-' + '3' * 17),
                                         ('ShellVersion', '0xwrong'), ('Tags', []),
                                         ('State', {'Code': 'unknown'})])
def test_live_image_mismatch_preserves_record(prepared, field, value):
    args, _, cli = prepared
    p.publish(*args, cli=cli); before = args[-1].read_bytes()
    cli.image[field] = value
    with pytest.raises(ValueError): p.status(args[-1], cli=cli)
    assert args[-1].read_bytes() == before


def test_available_status_checks_five_tags_and_owner(prepared):
    args, _, cli = prepared
    p.publish(*args, cli=cli)
    result = p.status(args[-1], cli=cli)
    assert result['state'] == 'available'
    assert len(p.identity.image_tags(result)) == 5
    assert result['last_describe']['FpgaImages'][0]['OwnerId'] == p.ACCOUNT


def test_duplicate_record_keys_rejected(prepared):
    args, _, cli = prepared
    args[-1].write_text('{"schema": 1, "schema": 1}')
    with pytest.raises(ValueError, match='Duplicate'):
        p.publish(*args, cli=cli)
    assert not cli.calls


def test_upload_size_limit_precedes_put(prepared, monkeypatch):
    args, _, cli = prepared
    monkeypatch.setattr(p, 'MAX_PUT_BYTES', 1)
    with pytest.raises(ValueError, match='single-PUT'):
        p.immutable_upload(args[0] / 'design.tar', 'unused', args[2], '0' * 64, 6, cli)
    assert not cli.calls


def test_bytes_changed_after_qualification_never_upload(prepared):
    args, _, cli = prepared
    (args[0] / 'design.tar').write_bytes(b'changed after qualified snapshot')
    with pytest.raises(ValueError, match='differs? from qualified'):
        p.publish(*args, cli=cli)
    assert not any(call[:2] == ('s3api', 'put-object') for call in cli.calls)
