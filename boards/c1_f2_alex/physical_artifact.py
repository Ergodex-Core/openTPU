"""Offline C1 runtime artifact verification. No AWS calls or device access.

The caller pins the request's SHA256 independently. A request only locates
evidence: the current C1 semantic verifier, official package verifier and image
reconciliation must all succeed before any runtime binding can be returned.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat

API_VERSION = 1
HERE = Path(__file__).resolve().parent
INSTANCE = 'i-0c78de2253f66519d'
MAX_RECORD = 1 << 20


def need(ok, message):
    if not ok:
        raise ValueError(message)


def module(name):
    path = HERE / (name + '.py')
    need(path.is_file() and not path.is_symlink(), 'Required C1 verifier is unavailable: ' + name)
    spec = importlib.util.spec_from_file_location('c1_runtime_' + name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def digest(value):
    need(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def safe_path(value):
    need(isinstance(value, (str, Path)), 'Expected artifact path')
    path = Path(value)
    need(path.is_absolute() and str(path) == os.path.normpath(path), 'Artifact path must be canonical and absolute')
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'Linked artifact path')
    return path


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate artifact record key: ' + key)
        result[key] = value
    return result


def pinned_record(path, expected):
    path = safe_path(path)
    digest(expected)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= MAX_RECORD,
             'Invalid artifact record type or size')
        raw = stream.read(MAX_RECORD + 1)
        after = os.fstat(stream.fileno())
    stamp = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    need(stamp(before) == stamp(after) == stamp(path.stat()) and len(raw) == before.st_size
         and hashlib.sha256(raw).hexdigest() == expected, 'Artifact record changed or hash differs')
    result = json.loads(raw, object_pairs_hook=unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite artifact value')))
    need(isinstance(result, dict), 'Artifact record must be an object')
    return result


def verify_session_artifact(request_path, trusted_request_sha):
    request = pinned_record(request_path, trusted_request_sha)
    keys = {'schema', 'kind', 'instance_id', 'hardware_revision', 'evidence_directory',
            'evidence_manifest_sha256', 'package_directory', 'package_receipt_sha256',
            'image_record', 'image_record_sha256'}
    need(set(request) == keys and type(request['schema']) is int and request['schema'] == 1 and
         request['kind'] == 'c1-f2-session-artifact-request' and request['instance_id'] == INSTANCE,
         'Wrong C1 runtime request schema or target')
    need(isinstance(request['hardware_revision'], str) and
         re.fullmatch('[0-9a-f]{40}', request['hardware_revision']), 'Missing C1 hardware revision')
    for key in keys:
        if key.endswith('_sha256'):
            digest(request[key])
    root = safe_path(request['evidence_directory'])
    package_root = safe_path(request['package_directory'])
    image = pinned_record(request['image_record'], request['image_record_sha256'])
    # Qualified C1 evidence is a distinct API from candidate_evidence.py.
    # Missing semantics, pending raw captures and C2 results cannot enter here.
    evidence = module('evidence').verify_bundle(root, request['evidence_manifest_sha256'])
    need(type(evidence.get('schema')) is int and evidence['schema'] == 1 and evidence.get('kind') == 'verified-c1-alex-portable-evidence' and
         evidence.get('route_qualified') is True and evidence.get('review_required') is False and
         evidence.get('hardware_measured') is False and
         evidence.get('source_revision') == request['hardware_revision'], 'C1 evidence is not independently qualified')
    package = module('package').verify_package(package_root, root, request['evidence_manifest_sha256'],
                                              request['package_receipt_sha256'])
    deployment = module('deployment')
    binding = deployment.verify_deployment(image, evidence)
    for key in deployment.HASH_FIELDS + ('source_revision', 'build_id', 'package_bytes'):
        need(package.get(key) == image.get(key), 'Published image differs from verified package: ' + key)
    need(package['package_receipt_sha256'] == request['package_receipt_sha256'] and
         binding['evidence_manifest_sha256'] == request['evidence_manifest_sha256'], 'Runtime receipt binding differs')
    build = evidence['build_id']
    need(build == request['hardware_revision'][:8], 'Runtime hardware/build identity differs')
    identity = dict(id=0x43314D42, abi=1, build_id=int(build, 16), core_khz=200000,
                    max_rows=512, depth=128, credits=32)
    # The package verifier reads these exact parameters from the compiled top;
    # this ABI is not inferred from runtime registers or a user-supplied record.
    result = dict(schema=1, kind='c1-f2-verified-artifact-binding',
        source_revision=evidence['source_revision'], build_id=build, identity=identity,
        afi_id=binding['afi_id'], agfi_id=binding['agfi_id'],
        evidence_manifest_sha256=request['evidence_manifest_sha256'], package_sha256=package['package_sha256'],
        image_record_sha256=request['image_record_sha256'],
        qualification_sha256=evidence['qualification_receipt_sha256'],
        package_receipt_sha256=request['package_receipt_sha256'], publication_record=image)
    need(pinned_record(request_path, trusted_request_sha) == request and
         pinned_record(request['image_record'], request['image_record_sha256']) == image,
         'Runtime request/image changed during verification')
    return result


def verify_aws_image(response, qualification):
    """Check the live AWS response before the session discovers or loads a slot."""
    need(isinstance(qualification, dict) and type(qualification.get('schema')) is int and
         qualification['schema'] == 1 and qualification.get('kind') == 'c1-f2-verified-artifact-binding',
         'Wrong C1 runtime qualification')
    record = qualification.get('publication_record')
    need(isinstance(record, dict), 'Missing verified publication record')
    for key, image_key in (('afi_id', 'image_id'), ('agfi_id', 'global_image_id'),
                           ('build_id', 'build_id'), ('source_revision', 'source_revision'),
                           ('evidence_manifest_sha256', 'evidence_manifest_sha256'),
                           ('package_sha256', 'package_sha256'),
                           ('package_receipt_sha256', 'package_receipt_sha256'),
                           ('qualification_sha256', 'qualification_receipt_sha256')):
        need(qualification.get(key) == record.get(image_key), 'Runtime publication binding changed: ' + key)
    return module('deployment').verify_aws_image(response, record)
