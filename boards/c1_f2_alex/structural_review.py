"""Authenticate additive C1 structural evidence; unresolved semantics stay closed.

Receipt identity and collector success are not structural equivalence. The final
adapter must separately replay complete structural facts, both arithmetic cones,
admission/RAM preservation and the actual ready-replica transformation.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import stat
from . import native_review as N

HERE = Path(__file__).resolve().parent
HARDWARE_REVISION = '0d654453e2c77ea9bc9d0e854c33d9166bb27375'
# Exact read-only capture executors; separate from the frozen hardware revision.
ANCESTRY_EXECUTORS = {'link': {'revision': '0278cb668faa078524f8d7dc97c6401adef0fe48', 'hashes': {'implement.py': 'c02a0164e57a154065fe3e39caf8df1ff5c79376cbd18641120404df65324941', 'build.py': 'de5d55aa379f6b127249af308a761d418f3e3466cdc3af2787e2389dc8b859eb'}}, 'refine': {'revision': '85addb6f352bf4b3a9fed1ecacfe4a6d3acbf1a6', 'hashes': {'implement.py': '370c994857c41109c48f9623b3f9b54fc320c67663d041f9e7c5499d9d9f718e', 'build.py': 'de5d55aa379f6b127249af308a761d418f3e3466cdc3af2787e2389dc8b859eb'}}}
SPECS = {
    'accounting': ('c1-accounting-raw-capture', 'raw/accounting.tcldata',
                   ('capture_accounting.tcl', 'capture_native.tcl'),
                   'c1_read_accounting_topology', 'accounting_sources'),
    'accounting_boundary': ('c1-accounting-boundary-raw-capture', 'raw/admission.tcldata',
                            ('capture_accounting_boundary.tcl', 'capture_native.tcl'),
                            'c1_accounting_boundary_topology', 'admission_sources'),
    'structural': ('c1-structural-raw-capture', 'raw/structural.tcldata',
                   ('capture_structure.tcl', 'qualify_c1_alex.tcl',
                    'audit_c1_alex.tcl', 'routed_host_ready.tcl'),
                   'c1_structural_scope_capture', None),
}
RECEIPT_FIELDS = frozenset({
    'schema', 'kind', 'stage', 'collector_revision', 'hardware_revision',
    'ancestry_executor_revision', 'ancestry_executor_hashes', 'checkpoint_sha256',
    'parent_receipt_sha256', 'started_at', 'completed_at', 'returncode',
    'inputs_unchanged', 'postcheck_error', 'helpers', 'files', 'summary',
    'route_qualified', 'hardware_measured', 'hardware_actions',
})


def digest(value):
    N.require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid capture digest')
    return value


def file_record(path):
    path = Path(path)
    N.require(not any(p.is_symlink() for p in (path, *path.parents)), 'linked additive capture')
    info = path.stat()
    N.require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and 0 < info.st_size <= N.MAX_FILE, 'unsafe additive capture')
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return N.identity(path, checksum)


def unique_json(pairs):
    result = {}
    for key, value in pairs:
        N.require(key not in result, 'duplicate capture receipt key')
        result[key] = value
    return result


def authenticate_capture(*, role, raw_path, receipt_path, checkpoint_sha256,
                         parent_receipt_sha256, stage):
    """Bind a raw supplement to the independently authenticated stage ancestry."""
    N.require(role in SPECS and stage in ('link', 'refine') and
              (role != 'structural' or stage == 'refine'), 'unsupported additive capture role/stage')
    kind, raw_name, helper_names, raw_kind, source_tag = SPECS[role]
    receipt_identity = file_record(receipt_path)
    N.require(receipt_identity['bytes'] <= 1 << 20, 'oversized additive receipt')
    receipt = json.loads(Path(receipt_path).read_bytes(), object_pairs_hook=unique_json)
    N.require(isinstance(receipt, dict) and set(receipt) == RECEIPT_FIELDS,
              'additive receipt field inventory differs')
    N.require(type(receipt['schema']) is int and receipt['schema'] == 1 and
              receipt['kind'] == kind and receipt['stage'] == stage,
              'additive receipt role differs')
    N.require(receipt['hardware_revision'] == HARDWARE_REVISION and
              receipt['checkpoint_sha256'] == digest(checkpoint_sha256) and
              receipt['parent_receipt_sha256'] == digest(parent_receipt_sha256),
              'additive capture checkpoint/ancestry differs')
    N.require(type(receipt['returncode']) is int and receipt['returncode'] == 0 and
              receipt['inputs_unchanged'] is True and receipt['postcheck_error'] is None,
              'additive collector failed or input changed')
    N.require(all(receipt[key] is False for key in ('route_qualified', 'hardware_measured', 'hardware_actions')),
              'raw supplement claimed qualification or hardware action')
    for key in ('collector_revision', 'ancestry_executor_revision'):
        N.require(isinstance(receipt[key], str) and re.fullmatch('[0-9a-f]{40}', receipt[key]),
                  'unidentified additive source revision')
    executor = receipt['ancestry_executor_hashes']
    N.require(isinstance(executor, dict) and set(executor) == {'implement.py', 'build.py'},
              'additive executor inventory differs')
    for value in executor.values(): digest(value)
    N.require(receipt['ancestry_executor_revision'] == ANCESTRY_EXECUTORS[stage]['revision'] and
              executor == ANCESTRY_EXECUTORS[stage]['hashes'], 'unreviewed additive ancestry executor')
    times = [datetime.fromisoformat(receipt[key]) for key in ('started_at', 'completed_at')]
    N.require(all(value.tzinfo is not None for value in times) and times[0] <= times[1],
              'additive capture time interval differs')
    helpers = {name: file_record(HERE / name)['sha256'] for name in helper_names}
    N.require(receipt['helpers'] == helpers, 'additive helper bytes differ from trusted source')
    files = receipt['files']
    N.require(isinstance(files, dict) and set(files) == set(helper_names) |
              {'run.tcl', 'driver.py', 'runner.sh', 'vivado.log', raw_name},
              'additive captured file inventory differs')
    for record in files.values():
        N.require(isinstance(record, dict) and set(record) == {'bytes', 'sha256'} and
                  type(record['bytes']) is int and 0 < record['bytes'] <= 536870912,
                  'invalid additive file record')
        digest(record['sha256'])
    for name in helper_names:
        N.require(files[name] == file_record(HERE / name), 'captured helper file binding differs')
    raw_identity = N.identity(raw_path, files[raw_name]['sha256'])
    N.require(raw_identity == files[raw_name], 'additive raw size binding differs')
    records = N.records(raw_path)
    try:
        first = next(records)
        N.require(first and first[0] == 'header', 'additive raw header missing')
        header = N.pairs(first[1:])
        N.require(header.get('schema') == '1' and header.get('kind') == raw_kind and
                  header.get('checkpoint_sha256') == checkpoint_sha256 and
                  header.get('route_qualified') == header.get('hardware_measured') == '0',
                  'additive raw header identity differs')
        if source_tag:
            second = next(records)
            N.require(second[0] == source_tag and N.pairs(second[1:]) == {
                'helper_sha256': helpers[helper_names[0]],
                'native_helper_sha256': helpers['capture_native.tcl']},
                'additive raw helper source binding differs')
        else:
            N.require(N.td(header.get('helpers', '')) == helpers, 'structural raw helper binding differs')
    finally:
        records.close()
    N.require(file_record(receipt_path) == receipt_identity and
              N.identity(raw_path, raw_identity['sha256']) == raw_identity,
              'additive evidence changed during authentication')
    return {'role': role, 'stage': stage, 'raw': raw_identity, 'receipt': receipt_identity,
            'checkpoint_sha256': checkpoint_sha256, 'parent_receipt_sha256': parent_receipt_sha256,
            'helpers': helpers, 'collector_revision': receipt['collector_revision'],
            'ancestry_executor_revision': receipt['ancestry_executor_revision'],
            'ancestry_executor_hashes': executor, 'route_qualified': False}


def review_artifact(*, reports, auxiliary, checkpoint_sha256, linked_checkpoint_sha256, prerequisites):
    result = {'schema': 1, 'kind': 'c1-structural-closure-review', 'issues': [],
              'canonical_facts': {}, 'resolved_pending_reviews': [],
              'route_qualified': False, 'hardware_measured': False}
    try:
        parents = prerequisites['receipt_sha256']
        N.require(set(parents) == {'link', 'refine'}, 'missing verified structural stage ancestry')
        captures = {}
        for role in SPECS:
            for stage in (('refine',) if role == 'structural' else ('link', 'refine')):
                suffix = '_linked' if stage == 'link' else ''
                raw_role = role + suffix
                receipt_role = role + suffix + '_receipt'
                captures[raw_role] = authenticate_capture(role=role,
                    raw_path=auxiliary[raw_role], receipt_path=auxiliary[receipt_role],
                    checkpoint_sha256=linked_checkpoint_sha256 if stage == 'link' else checkpoint_sha256,
                    parent_receipt_sha256=parents[stage], stage=stage)
        result['canonical_facts']['additive_capture_bindings'] = captures
        # Receipt success is deliberately insufficient. Actual final raw facts
        # must close every structural, admission and replica predicate below.
        result['issues'].append({'reason': 'structural_semantics_pending',
            'obligations': ['complete_structural_fact_replay', 'original_structural_failure_scope',
                            'paired_accounting_and_admission_preservation', 'ready_replica_scope_closure']})
    except (ValueError, OSError, KeyError, TypeError, StopIteration, UnicodeError) as exc:
        result['issues'].append({'reason': str(exc)})
    return result
