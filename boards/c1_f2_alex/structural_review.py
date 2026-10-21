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


def original_structural_observation(reports, checkpoint_sha256):
    """Preserve and cross-bind the exact original fail-fast result, never waive it."""
    summary_path = reports['summary.tcldata']; objects_path = reports['objects.tcldata']
    before = {str(path): file_record(path) for path in (summary_path, objects_path)}
    summary = N.td(Path(summary_path).read_text())
    N.require(summary.get('checkpoint_sha256') == checkpoint_sha256 and
              summary.get('kind') == 'c1_final_raw_capture' and summary.get('route_qualified') == '0',
              'original structural summary identity differs')
    structural = N.td(summary['structural_capture'])
    N.require(structural.get('state') in ('observed', 'unresolved') and structural.get('route_qualified') == '0',
              'unknown original structural capture state')
    expected = {'state', 'route_qualified'} | ({'topology'} if structural['state'] == 'observed' else {'error', 'error_info'})
    N.require(set(structural) == expected and all(structural[k] for k in expected),
              'incomplete original structural result')
    if structural['state'] == 'unresolved':
        N.require(structural['error'] in structural['error_info'], 'original structural error/stack disagree')
    checkpoints = []; observations = []
    for row in N.records(objects_path):
        if row[0] == 'checkpoint':
            N.require(len(row) == 4 and row[2] == 'sha256', 'original object checkpoint schema differs')
            checkpoints.append(row[3])
        elif row[0] == 'structural_capture':
            N.require(len(row) == 2, 'original structural object schema differs')
            observations.append(N.td(row[1]))
    N.require(checkpoints == [checkpoint_sha256] and observations == [structural],
              'original structural summary/object identity differs')
    N.require(all(file_record(path) == before[str(path)] for path in (summary_path, objects_path)),
              'original structural observation changed')
    return {'inputs': before, 'observation': structural,
            'failed_scope_resolved': False, 'route_qualified': False}


def replay_current_native(path, checkpoint_sha256, prerequisite):
    """Reread typed facts only when they match the already verified native replay."""
    bound = file_record(path)
    N.require(prerequisite.get('capture') == bound and prerequisite.get('checkpoint_sha256') == checkpoint_sha256 and
              prerequisite.get('issues') == [], 'native prerequisite binding differs')
    result = N.review_artifact(topology_path=path, topology_sha256=bound['sha256'],
                               checkpoint_sha256=checkpoint_sha256)
    N.require(result == prerequisite, 'current native replay differs from prerequisite')
    evidence = N.Evidence()
    for row in N.records(path): evidence.add(row)
    N.require(not evidence.issues and not evidence.incomplete and file_record(path) == bound,
              'native typed evidence changed or incomplete')
    return evidence, result['canonical_facts']['receive_gates']


def replay_source_specialization(prerequisites):
    """Reread the exact hardware sources; caller success flags are not evidence."""
    from . import admission_specialization_review as S
    value = S.review_sources(**prerequisites['source_binding'])
    N.require(value == prerequisites['source_specialization'],
              'current source specialization differs from prerequisite')
    N.require(value.get('source_elided_bits_proven') is True and value.get('issues') == [],
              'source specialization replay incomplete')
    return value


def admission_source_composition(source, transitions):
    """Internal composition of fresh public replays; final scope stays pending."""
    from . import admission_transition_review as T, admission_equation_review as E
    N.require(source.get('source_elided_bits_proven') is True and source.get('issues') == [],
              'source specialization unavailable for admission composition')
    configuration = source['canonical_facts']['configuration']
    N.require(configuration == T.CONFIGURATION, 'admission source configuration differs')
    N.require(set(transitions) == {'link', 'refine'}, 'paired transition replay missing')
    checked = {}
    for stage, value in transitions.items():
        N.require(value.get('one_step_equivalent_under_source_specialization') is True and
                  value.get('issues') == [], 'admission transition replay incomplete')
        channels = value['canonical_facts']['channels']
        N.require(set(channels) == {str(i) for i in range(8)}, 'admission composition channel inventory')
        for channel in channels.values():
            N.require(channel['configuration_required'] == configuration and
                      channel['rtl_sha256'] == E.RTL_SHA256 and
                      channel['one_step_equivalent_under_source_specialization'] is True,
                      'admission transition source/configuration binding differs')
        checked[stage] = {'transition_facts_sha256': digest(value['facts_sha256']),
                          'roots': {pc: row['root'] for pc, row in channels.items()}}
    return {'source_facts_sha256': digest(source['facts_sha256']),
            'configuration': configuration, 'checked': checked, 'configuration_bound': True,
            'scope': 'conditional one-step admission-slice equations with separately replayed source specialization',
            'whole_native_equivalence_proven': False, 'paired_preservation_proven': False,
            'route_qualified': False, 'hardware_measured': False,
            'pending_reviews': ['external_admission_input_producer_equivalence',
                                'linked_final_accounting_preservation', 'ready_replica_scope_closure']}


def review_artifact(*, reports, auxiliary, checkpoint_sha256, linked_checkpoint_sha256, prerequisites):
    from . import structural_capture_review as C, structural_static_review as T
    from . import structural_flow_review as F, accounting_review as A
    from . import admission_transition_review as D, admission_pair_review as P
    result = {'schema': 1, 'kind': 'c1-structural-closure-review', 'issues': [],
              'canonical_facts': {}, 'resolved_pending_reviews': [],
              'route_qualified': False, 'hardware_measured': False}
    # No combination of partial proofs closes admission/RAM preservation or a
    # routed ready-replica rewrite. Their actual paired witnesses are mandatory.
    result['issues'].append({'reason': 'structural_semantics_pending',
        'obligations': ['original_structural_failure_scope', 'paired_accounting_and_admission_preservation',
                        'ready_replica_scope_closure']})
    facts = result['canonical_facts']; captures = {}; facts['additive_capture_bindings'] = captures
    def attempt(label, operation):
        try:
            value = operation(); facts[label] = value
            if value.get('issues'):
                result['issues'].append({'reason': label + '_unresolved', 'issues': value['issues']})
            return value
        except (ValueError, OSError, KeyError, TypeError, StopIteration, UnicodeError, IndexError) as exc:
            result['issues'].append({'reason': label + '_unresolved', 'detail': str(exc)})
            return None
    parents = prerequisites.get('receipt_sha256', {})
    if set(parents) != {'link', 'refine'}:
        result['issues'].append({'reason': 'missing verified structural stage ancestry'})
        return result
    for role in SPECS:
        for stage in (('refine',) if role == 'structural' else ('link', 'refine')):
            suffix = '_linked' if stage == 'link' else ''; raw_role = role + suffix
            try:
                captures[raw_role] = authenticate_capture(role=role,
                    raw_path=auxiliary[raw_role], receipt_path=auxiliary[role + suffix + '_receipt'],
                    checkpoint_sha256=linked_checkpoint_sha256 if stage == 'link' else checkpoint_sha256,
                    parent_receipt_sha256=parents[stage], stage=stage)
            except (ValueError, OSError, KeyError, TypeError, StopIteration, UnicodeError) as exc:
                result['issues'].append({'reason': raw_role + '_authentication', 'detail': str(exc)})
    attempt('original_structural', lambda: original_structural_observation(reports, checkpoint_sha256))
    if 'structural' in captures:
        bound = captures['structural']
        parsed = attempt('structural_envelope', lambda: C.review_artifact(path=auxiliary['structural'],
            sha256=bound['raw']['sha256'], checkpoint_sha256=checkpoint_sha256, helpers=bound['helpers']))
        if parsed and parsed.get('capture_complete') is True and not parsed.get('issues'):
            canonical = parsed['canonical_facts']
            attempt('static', lambda: T.review(canonical))
            native = None; receive = None
            try:
                native, receive = replay_current_native(auxiliary['native'], checkpoint_sha256, prerequisites['native'])
            except (ValueError, OSError, KeyError, TypeError, UnicodeError) as exc:
                result['issues'].append({'reason': 'native_flow_prerequisite', 'detail': str(exc)})
            attempt('flows', lambda: F.review(canonical, native=native, receive_facts=receive))
    for raw_role in ('accounting_linked', 'accounting'):
        if raw_role not in captures: continue
        bound = captures[raw_role]
        attempt(raw_role + '_arithmetic', lambda: A.review_artifact(topology_path=auxiliary[raw_role],
            topology_sha256=bound['raw']['sha256'], checkpoint_sha256=bound['checkpoint_sha256'],
            helper_sha256=bound['helpers']['capture_accounting.tcl'],
            native_helper_sha256=bound['helpers']['capture_native.tcl']))
    source = attempt('source_specialization', lambda: replay_source_specialization(prerequisites))
    admission_bindings = {}; transitions = {}
    for stage, raw_role in (('link', 'accounting_boundary_linked'), ('refine', 'accounting_boundary')):
        if raw_role not in captures: continue
        bound = captures[raw_role]
        def replay_admission():
            binding = dict(topology_path=auxiliary[raw_role], topology_sha256=bound['raw']['sha256'],
                checkpoint_sha256=bound['checkpoint_sha256'],
                helper_sha256=bound['helpers']['capture_accounting_boundary.tcl'],
                native_helper_sha256=bound['helpers']['capture_native.tcl'])
            admission_bindings[stage] = binding
            return D.review_artifact(**binding)
        value = attempt(raw_role + '_transition', replay_admission)
        if value is not None: transitions[stage] = value
    if set(admission_bindings) == {'link', 'refine'}:
        attempt('admission_pair', lambda: P.review_pair(linked=admission_bindings['link'],
                                                       routed=admission_bindings['refine']))
    if source is not None and set(transitions) == {'link', 'refine'}:
        attempt('admission_source_composition', lambda: admission_source_composition(source, transitions))
    return result
