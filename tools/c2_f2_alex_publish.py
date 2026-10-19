#!/usr/bin/env python3
"""Publish a qualified Alex/C2 package through AWS CLI; never load an FPGA.

External receipt hashes bind the portable evidence and package. The complete
portable evidence archive is preserved alongside the design. A durable request
and client token let an interrupted publication resume without creating another
image. AWS reference: https://docs.aws.amazon.com/cli/latest/reference/ec2/create-fpga-image.html
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


common = module('alex_publish_common', 'tools/c2_f2_publish.py')
identity = module('alex_publish_identity', 'boards/c2_f2_alex/deployment.py')
ACCOUNT, REGION, BUCKET = identity.ACCOUNT, identity.REGION, identity.BUCKET
BINDINGS = identity.HASH_FIELDS + ('source_revision', 'build_id', 'package_bytes')
MAX_PUT_BYTES = 5 * 1024**3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_record(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate record key: ' + key)
            result[key] = value
        return result
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Record is missing or linked')
    return json.loads(path.read_text(), object_pairs_hook=unique)


def verify(package_run, evidence_root, manifest_sha256, package_receipt_sha256, evidence_archive):
    package = module('alex_publish_package', 'boards/c2_f2_alex/package.py')
    evidence = module('alex_publish_evidence', 'boards/c2_f2_alex/evidence.py')
    result = package.verify_package(package_run, evidence_root, manifest_sha256, package_receipt_sha256)
    archive = evidence.inspect_archive(evidence_archive, manifest_sha256)
    result.update(evidence_archive_sha256=archive['archive_sha256'],
                  evidence_archive_bytes=Path(evidence_archive).stat().st_size)
    # The streaming archive check establishes exact equality to the same trusted
    # inventory that verify_package semantically qualified in the extracted tree.
    return result


def immutable_upload(path, key, manifest_sha256, expected_sha256, expected_bytes, cli):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Upload input is missing or linked')
    size = path.stat().st_size
    require(size == expected_bytes, 'Upload size differs from qualified artifact: ' + key)
    require(0 < size <= MAX_PUT_BYTES, 'Artifact exceeds supported single-PUT size')
    digest = common.sha(path)
    require(digest == expected_sha256, 'Upload bytes differ from qualified artifact: ' + key)
    checksum = base64.b64encode(bytes.fromhex(digest)).decode()
    metadata = {'sha256': digest, 'evidence-manifest-sha256': manifest_sha256}
    try:
        cli('s3api', 'put-object', '--bucket', BUCKET, '--key', key, '--body', str(path),
            '--expected-bucket-owner', ACCOUNT, '--server-side-encryption', 'AES256',
            '--checksum-algorithm', 'SHA256', '--checksum-sha256', checksum,
            '--metadata', json.dumps(metadata), '--if-none-match', '*')
    except subprocess.CalledProcessError as error:
        # A resumed request may already own this exact object. Never overwrite
        # it; compare the server's checksum, metadata and size before proceeding.
        require('PreconditionFailed' in (error.stderr or ''), 'Immutable upload failed: ' + str(error))
    head = cli('s3api', 'head-object', '--bucket', BUCKET, '--key', key,
               '--expected-bucket-owner', ACCOUNT, '--checksum-mode', 'ENABLED')
    require(head.get('ContentLength') == size and head.get('ChecksumSHA256') == checksum and
            head.get('Metadata') == metadata and head.get('ServerSideEncryption') == 'AES256',
            'S3 readback differs from immutable artifact: ' + key)
    require(path.stat().st_size == size and common.sha(path) == digest, 'Input changed during upload')
    return {'key': key, 'sha256': digest, 'bytes': size, 'head': head}


def publish(package_run, evidence_root, manifest_sha256, package_receipt_sha256,
            evidence_archive, record_path, *, cli=common.aws):
    package_run, evidence_root, evidence_archive = map(Path, (package_run, evidence_root, evidence_archive))
    with common.locked(record_path):
        checked = verify(package_run, evidence_root, manifest_sha256, package_receipt_sha256, evidence_archive)
        if Path(record_path).exists():
            record = read_record(record_path)
            identity.validate_identity(record)
            require(all(record.get(key) == checked[key] for key in BINDINGS),
                    'Publication record belongs to another qualified package')
            require(record.get('evidence_archive_sha256') == checked['evidence_archive_sha256'] and
                    record.get('evidence_archive_bytes') == checked['evidence_archive_bytes'],
                    'Publication evidence archive changed')
            if record.get('image_id'):
                return record
        else:
            record = {key: checked[key] for key in BINDINGS}
            record.update(schema=1, kind='c2-alex-fpga-image', account=ACCOUNT, region=REGION,
                          bucket=BUCKET, state='prepared', created_at=now(),
                          prefix=f'c2-alex/{checked["build_id"]}/{uuid.uuid4().hex}',
                          evidence_archive_sha256=checked['evidence_archive_sha256'],
                          evidence_archive_bytes=checked['evidence_archive_bytes'])
            record['client_token'] = identity.token(record)
            record['publication_request'] = identity.make_request(record)
            identity.validate_identity(record)
            common.atomic_json(record_path, record)
        caller = cli('sts', 'get-caller-identity')
        require(caller.get('Account') == ACCOUNT, 'Wrong publication account')
        cli('s3api', 'head-bucket', '--bucket', BUCKET, '--expected-bucket-owner', ACCOUNT)
        location = cli('s3api', 'get-bucket-location', '--bucket', BUCKET, '--expected-bucket-owner', ACCOUNT)
        require(location.get('LocationConstraint') in (None, REGION), 'Wrong AFI bucket region')
        files = [(Path(checked['archive']), 'design.tar', checked['package_sha256'], checked['package_bytes']),
                 (package_run / 'package.json', 'provenance/package.json', package_receipt_sha256,
                  (package_run / 'package.json').stat().st_size),
                 (evidence_root / 'bundle-manifest.json', 'provenance/bundle-manifest.json', manifest_sha256,
                  (evidence_root / 'bundle-manifest.json').stat().st_size),
                 (evidence_archive, 'provenance/evidence.tar', checked['evidence_archive_sha256'],
                  checked['evidence_archive_bytes'])]
        uploads = [immutable_upload(path, record['prefix'] + '/' + suffix, manifest_sha256, expected, size, cli)
                   for path, suffix, expected, size in files]
        require(all(upload['sha256'] == expected and upload['bytes'] == size
                    for upload, (_, _, expected, size) in zip(uploads, files)),
                'Uploaded artifacts differ from qualified records')
        fresh = verify(package_run, evidence_root, manifest_sha256, package_receipt_sha256, evidence_archive)
        require(all(fresh[key] == record[key] for key in BINDINGS) and
                fresh['evidence_archive_sha256'] == record['evidence_archive_sha256'],
                'Qualified inputs changed during publication')
        record.update(caller_identity=caller, uploads=uploads, request_started_at=now())
        common.atomic_json(record_path, record)
        response = cli('ec2', 'create-fpga-image', '--cli-input-json', json.dumps(record['publication_request']))
        record.update(image_id=response.get('FpgaImageId'), global_image_id=response.get('FpgaImageGlobalId'),
                      state='pending', create_response=response)
        identity.validate_identity(record)
        common.atomic_json(record_path, record)
        return record


def status(record_path, *, cli=common.aws):
    with common.locked(record_path):
        record = read_record(record_path)
        identity.validate_identity(record)
        require(record.get('image_id'), 'Publication has no saved AFI; resume its original request')
        require(cli('sts', 'get-caller-identity').get('Account') == ACCOUNT, 'Wrong status account')
        response = cli('ec2', 'describe-fpga-images', '--fpga-image-ids', record['image_id'])
        images = response.get('FpgaImages', [])
        require(len(images) == 1, 'Expected exactly one image status')
        image = images[0]
        require(image.get('FpgaImageId') == record['image_id'] and
                image.get('FpgaImageGlobalId') == record['global_image_id'] and
                image.get('OwnerId') == ACCOUNT, 'Image ID/owner binding mismatch')
        pairs = image.get('Tags', [])
        tags = {row['Key']: row['Value'] for row in pairs}
        require(len(tags) == len(pairs) and all(tags.get(k) == v for k, v in identity.image_tags(record).items()),
                'Image provenance tags mismatch')
        state = image.get('State', {}).get('Code')
        require(state in ('pending', 'available', 'failed', 'unavailable'), 'Unknown image state')
        if state == 'available':
            require(image.get('ShellVersion', '').lower() == '0x10212415', 'Available image shell differs')
        record.update(state=state, last_describe=response, checked_at=now())
        identity.validate_identity(record)
        common.atomic_json(record_path, record)
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for name in ('verify', 'publish'):
        action = actions.add_parser(name)
        action.add_argument('--package-run', type=Path, required=True)
        action.add_argument('--evidence-root', type=Path, required=True)
        action.add_argument('--evidence-manifest-sha256', required=True)
        action.add_argument('--package-receipt-sha256', required=True)
        action.add_argument('--evidence-archive', type=Path, required=True)
        if name == 'publish':
            action.add_argument('--record', type=Path, required=True)
    actions.add_parser('status').add_argument('--record', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == 'status':
        result = status(args.record)
    else:
        values = (args.package_run, args.evidence_root, args.evidence_manifest_sha256,
                  args.package_receipt_sha256, args.evidence_archive)
        result = verify(*values) if args.action == 'verify' else publish(*values, args.record)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get('state') in ('failed', 'unavailable'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
