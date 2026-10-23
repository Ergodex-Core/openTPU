#!/usr/bin/env python3
"""C1 current raw semantic replay and additive portable qualification.

No C1 artifact is accepted until its exact disposition is separately reviewed and
pinned in reviewed_artifacts.json, all physical checks pass, and trusted C1 raw
predicates close every structural and semantic obligation. No bundled code runs.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from boards.c1_f2_alex import candidate_evidence as E, review_contract as R

HERE = Path(__file__).resolve().parent
POLICY = HERE / 'reviewed_artifacts.json'
require = E.require


def module(name):
    return E.local_module(name)


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def canonical(value):
    return hashlib.sha256(R.canonical(value)).hexdigest()


def record(path):
    path = Path(path); E.no_symlink(path)
    require(path.is_file() and path.stat().st_size <= E.MAX_FILE, 'Invalid review file')
    return {'sha256': E.sha(path), 'bytes': path.stat().st_size}


def read_json(path):
    require(record(path)['bytes'] <= E.MAX_MANIFEST, 'Oversized review JSON')
    return E.decode(Path(path).read_bytes())


def source_inventory():
    return {name: record(E.ROOT / name) for name in sorted(R.REVIEW_SOURCES)}


def hardware_source_binding(bundle, *, relocated_root=None):
    """Select source proof inputs solely from an authenticated candidate Bundle.

    The collector may supply its exact copied root before writing schema3. Each
    selected copy must still equal the original manifest inventory and receipt.
    No source code in that snapshot is imported or executed.
    """
    specialization = module('admission_specialization_review')
    hardware = bundle.manifest['hardware']
    snapshot = bundle.manifest['source_snapshots']['hardware']
    require(hardware['source_revision'] == snapshot['revision'] == specialization.HARDWARE_REVISION and
            hardware['build_id'] == specialization.HARDWARE_REVISION[:8],
            'Source specialization hardware snapshot differs')
    root = Path(relocated_root) if relocated_root is not None else bundle.root
    require(root.is_absolute(), 'Source binding root must be absolute')
    E.no_symlink(root)
    def selected(name, expected_sha256):
        name = E.relative(name)
        row = bundle.files.get(name)
        require(isinstance(row, dict), 'Missing source proof inventory: ' + name)
        E.metadata(row)
        require(row['sha256'] == expected_sha256 and record(root / name) == row,
                'Source proof input differs from authenticated inventory: ' + name)
        return root / name
    source_prefix = E.relative(snapshot['bundle_relative_path'])
    paths = {name: selected(source_prefix + '/' + name, digest)
             for name, digest in specialization.SOURCE_SHA256.items()}
    synth_prefix = E.relative(bundle.manifest['roots']['synth']['bundle_relative_path'])
    inputs_sha = E.digest(bundle.receipt('synth')['inputs_sha256'])
    require(inputs_sha in specialization.REVIEWED_SYNTH_INPUT_SHA256, 'Unreviewed synthesis input identity')
    inputs = selected(synth_prefix + '/inputs.json', inputs_sha)
    top = selected(synth_prefix + '/source/rtl/boards/f2/cl_opentpu.sv', specialization.TOP_SHA256)
    return dict(source_paths=paths, synth_inputs_path=inputs,
                synth_inputs_sha256=inputs_sha, generated_top_path=top)


def source_binding_files(binding):
    require(isinstance(binding, dict) and set(binding) == {
        'source_paths', 'synth_inputs_path', 'synth_inputs_sha256', 'generated_top_path'},
        'Missing/extra source specialization binding fields')
    require(isinstance(binding['source_paths'], dict) and
            set(binding['source_paths']) == set(module('admission_specialization_review').SOURCE_SHA256),
            'Missing/extra source specialization files')
    E.digest(binding['synth_inputs_sha256'])
    return [*binding['source_paths'].values(), binding['synth_inputs_path'], binding['generated_top_path']]


def artifact_records(g, objects, summary):
    records = []
    for row in g.records(objects):
        if not row: continue
        if row[0] == 'finding':
            fields = g.pairs(row[2:])
            records.append({'family': row[1], 'record': {'properties': g.td(fields['properties']), 'rule_properties': g.td(fields['rule_properties'])}})
        elif row[0] == 'cdc': records.append({'family': 'cdc', 'record': g.td(row[1])})
        elif row[0] == 'exception': records.append({'family': 'exceptions', 'record': g.pairs(row[1:])})
    checks = g.td(g.td(summary['timing'])['checks'])
    for name, value in checks.items():
        # No-clock is mechanically checked from exact typed shell evidence.
        if name == 'pulse_width_clock' and int(value):
            records.append({'family': 'check_timing', 'record': {'check': name, 'count': value}})
    require(records, 'Empty current finding/exception inventory')
    return records


def physical(g, summary):
    require(summary.get('schema') == '1' and summary.get('kind') == 'c1_final_raw_capture' and summary.get('stage') == 'final_raw' and summary.get('hardware_measured') == '0', 'Not a final route collection')
    require(summary.get('route_qualified') == '0', 'Expected original pending collector result')
    require(set(g.tcl_list(summary.get('blockers', ''))) <= {'structural_review_incomplete'}, 'Physical/DRC blocker cannot be overridden by review')
    timing = g.td(summary['timing']); totals = g.td(timing['totals']); checks = g.td(timing['checks']); route = g.td(summary['routing'])
    for key in ('wns', 'whs', 'wpws'): require(g.number(totals[key]) >= 0, 'Negative final timing: ' + key)
    for key in ('tns', 'ths', 'tpws', 'setup_failing', 'hold_failing', 'pulse_failing'): require(g.number(totals[key]) == 0, 'Failing final timing: ' + key)
    for key in ('setup_total', 'hold_total', 'pulse_total'): require(int(totals[key]) > 0, 'Missing final timing endpoints')
    expected = {'no_clock','constant_clock','pulse_width_clock','unconstrained_internal_endpoints','no_input_delay','no_output_delay','multiple_clock','generated_clocks','loops','partial_input_delay','partial_output_delay','latch_loops'}
    require(set(checks) == expected and not g.tcl_list(timing['blockers']), 'Incomplete/failing timing checks')
    require(all(int(value) >= 0 and (name in ('no_clock','pulse_width_clock') or int(value) == 0) for name,value in checks.items()), 'Unhandled unconstrained/timing check')
    require(route.get('pass') == '1' and int(route['routable']) > 0 and route['routable'] == route['routed'] and route['errors'] == '0', 'Routing not complete')
    supplemental = g.td(summary['supplemental_evidence'])
    require(supplemental.get('incomplete_or_error') == '0', 'Current collection has incomplete/error evidence')
    return {'timing': totals, 'checks': checks, 'routing': route}



def protected_pulse_context(*, reports, auxiliary, physical_result):
    """Compare inherited visible shell context, never infer a hidden pin identity."""
    from decimal import Decimal, InvalidOperation
    require('shell_timing' in auxiliary, 'Missing original shell timing witness')
    def parse(path):
        text = Path(path).read_text()
        sections = re.findall(r'(?m)^3\. checking pulse_width_clock \(\d+\)\n-+\n(.*?)(?=^4\. checking)', text, re.S)
        require(len(sections) == 1, 'Missing/ambiguous detailed pulse warning')
        warning = tuple(line.strip() for line in sections[0].splitlines() if line.strip())
        require(len(warning) == 2 and warning[0] == 'There is 1 register/latch pin which need pulse_width check. (LOW)' and
                re.fullmatch(r'<hidden>(?:/<hidden>)+', warning[1]), 'Unexpected protected pulse warning')
        clocks = {}; rows = []
        blocks = re.split(r'(?m)^Pulse Width Checks\s*\n-+\n', text)[1:]
        require(blocks, 'No pulse check tables')
        for block in blocks:
            head = re.match(r'Clock Name:\s*([^\n]+)\nWaveform\(ns\):\s*([^\n]+)\nPeriod\(ns\):\s*([^\n]+)\nSources:\s*([^\n]+)\n', block)
            require(head is not None, 'Malformed pulse clock metadata')
            name, waveform, period, sources = (value.strip() for value in head.groups())
            require(name not in clocks, 'Duplicate pulse clock table')
            try:
                p = Decimal(period); wave = [Decimal(v) for v in waveform.strip('{} ').split()]
                require(p.is_finite() and p > 0 and len(wave) == 2 and all(v.is_finite() for v in wave) and 0 <= wave[0] < wave[1] < p, 'Invalid pulse clock waveform')
            except InvalidOperation: raise ValueError('Invalid pulse clock number')
            clocks[name] = dict(waveform=waveform, period=period, sources=sources)
            tail = block[head.end():].lstrip().splitlines()
            require(tail and tail[0].startswith('Check Type'), 'Pulse table header absent')
            count = 0
            for line in tail[1:]:
                if not line.strip(): break
                match = re.fullmatch(r'(Min Period|Max Period|Low Pulse Width|High Pulse Width)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', line.strip())
                require(match is not None, 'Unparsed pulse row')
                kind, corner, lib, ref, req, act, slack, loc, pin = match.groups()
                try:
                    required, actual, margin = map(Decimal, (req, act, slack))
                    require(all(v.is_finite() and v >= 0 for v in (required, actual, margin)), 'Negative/nonfinite pulse result')
                    expected = required - actual if kind == 'Max Period' else actual - required
                    require(abs(expected - margin) <= Decimal('0.002'), 'Pulse arithmetic mismatch')
                except InvalidOperation: raise ValueError('Invalid pulse row number')
                rows.append([name, *match.groups()]); count += 1
            require(count > 0, 'Empty pulse table')
        return warning, clocks, rows
    old_warning, old_clocks, old_rows = parse(auxiliary['shell_timing'])
    warning, clocks, rows = parse(reports['timing.rpt'])
    require(warning == old_warning, 'Protected pulse context differs from original shell')
    require(all(name in clocks and clocks[name] == value for name, value in old_clocks.items()), 'Original shell pulse clock metadata changed')
    require(physical_result['checks']['pulse_width_clock'] == '1' and
            Decimal(physical_result['timing']['pulse_failing']) == 0 and
            Decimal(physical_result['timing']['tpws']) == 0 and
            Decimal(physical_result['timing']['wpws']) >= 0, 'Final complete pulse totals fail')
    return dict(warning=list(warning), inherited_clock_metadata=old_clocks,
                current_rows_sha256=canonical(rows), current_rows=len(rows), original_rows_sha256=canonical(old_rows),
                inputs={'current':record(reports['timing.rpt']), 'shell':record(auxiliary['shell_timing'])},
                hidden_endpoint_identity_asserted=False, scoped_report_exhaustiveness_asserted=False,
                basis='exact_original_shell_visible_warning_and_clock_context_plus_complete_final_pulse_totals')



SHELL_CDC_RECORD = {'bytes':3764,'sha256':'ae239477bad8f6bd9df021f5ee42102013c1691a824b127ba2419c9c74e6a349'}


def protected_cdc_context(g, records, auxiliary):
    """Exact original-shell report context and multiplicity, not hidden identity."""
    require({'shell_cdc','shell_complete'} <= set(auxiliary),'Missing original shell CDC witness')
    require(record(auxiliary['shell_cdc']) == SHELL_CDC_RECORD,'Original shell CDC report differs from reviewed capture')
    complete=g.td(Path(auxiliary['shell_complete']).read_text())
    require(complete.get('kind')=='original_shell_reference_captured' and complete.get('checkpoint_sha256')==module('build').SHELL_SHA256,'Original shell checkpoint provenance differs')
    original=module('exception_review').parse_cdc_report(Path(auxiliary['shell_cdc']).read_text())['rows']
    current=[row['record'] for row in records if row['family']=='cdc' and all(not row['record'].get(k,'').startswith('WRAPPER/CL/') for k in ('from','to'))]
    fields=('source_clock','destination_clock','cdc_type','rule','severity','description','depth','exception','from','to','waived')
    normalize=lambda row:{key:row[key] for key in fields}
    require(len(original)==5 and Counter(canonical(normalize(row)) for row in original)==Counter(canonical(normalize(row)) for row in current),'Protected shell CDC context/multiplicity changed')
    require(all(row['waived']=='N' for row in current),'Protected shell CDC record is waived')
    return dict(count=len(current),signatures=sorted(canonical(normalize(row)) for row in current),
                current_records=[normalize(row) for row in current],shell_report=SHELL_CDC_RECORD,
                shell_checkpoint_sha256=complete['checkpoint_sha256'],hidden_endpoint_identity_asserted=False)


def review_dispositions(value, *, checkpoint, capture_receipt, records, auxiliary):
    """Exact C1 artifact membership, never a rule-count or severity waiver."""
    require(isinstance(value, dict) and set(value) == {
        'schema', 'kind', 'checkpoint_sha256', 'collection_receipt_sha256',
        'record_inventory_sha256', 'auxiliary_sha256', 'dispositions'}, 'Malformed C1 dispositions')
    require(type(value['schema']) is int and value['schema'] == 1 and
            value['kind'] == 'c1-alex-artifact-dispositions', 'Wrong C1 disposition schema')
    require(value['checkpoint_sha256'] == checkpoint and
            value['collection_receipt_sha256'] == capture_receipt, 'Disposition ancestry differs')
    require(value['auxiliary_sha256'] == auxiliary, 'Disposition auxiliary evidence differs')
    inventory = sorted(canonical(row) for row in records)
    require(inventory and len(set(inventory)) == len(inventory), 'Empty/duplicate finding records')
    require(value['record_inventory_sha256'] == canonical(inventory), 'Finding inventory differs')
    by_id = {canonical(row): row for row in records}; covered = Counter()
    classes = {'exact-inherited-shell': {'cdc', 'drc', 'methodology', 'check_timing'},
               'current-vendor-cdc': {'cdc'}, 'current-reset-status': {'cdc', 'methodology'},
               'current-findings': {'drc', 'methodology'}, 'current-exceptions': {'exceptions'}}
    require(isinstance(value['dispositions'], list) and value['dispositions'], 'Missing C1 dispositions')
    for item in value['dispositions']:
        require(isinstance(item, dict) and set(item) == {
            'classification', 'family', 'record_sha256', 'evidence', 'rationale'}, 'Malformed disposition')
        require(item['classification'] in classes and item['family'] in classes[item['classification']],
                'Unknown disposition classification/family')
        require(isinstance(item['rationale'], str) and item['rationale'].strip() and
                isinstance(item['evidence'], list) and item['evidence'] and
                len(item['evidence']) == len(set(item['evidence'])) and
                set(item['evidence']) <= set(auxiliary), 'Missing disposition rationale/evidence')
        require(isinstance(item['record_sha256'], list) and item['record_sha256'], 'Empty record membership')
        for ident in item['record_sha256']:
            require(ident in by_id and by_id[ident]['family'] == item['family'], 'Unknown/wrong-family record')
            covered[ident] += 1
    require(covered == Counter(inventory), 'Missing/duplicate current record disposition')
    return {'record_inventory_sha256': canonical(inventory), 'records': len(inventory),
            'dispositions': len(value['dispositions'])}


def successful(value, label):
    require(isinstance(value, dict) and value.get('issues') == [] and
            isinstance(value.get('canonical_facts'), dict) and value['canonical_facts'],
            label + ' raw predicate unresolved')


def replay(*, reports, auxiliary, disposition_path, checkpoint, linked_checkpoint,
           capture_receipt_sha256, capture_receipts, source_binding=None):
    policy = read_json(POLICY)
    require(isinstance(policy, dict) and set(policy) == {'schema', 'artifacts'} and
            type(policy['schema']) is int and policy['schema'] == 1 and
            isinstance(policy['artifacts'], dict), 'Malformed C1 reviewed artifact policy')
    approved = policy['artifacts'].get(checkpoint)
    require(approved is not None and E.digest(approved) == record(disposition_path)['sha256'],
            'This C1 DCP/disposition has not been source-reviewed')
    require(set(reports) == set(E.FINAL_REPORTS) and
            set(auxiliary) == set(R.RAW_AUXILIARY) | R.REFERENCE_AUXILIARY,
            'Missing/extra current raw review inputs')
    before = {str(path): record(path) for path in [*reports.values(), *auxiliary.values(), disposition_path, POLICY,
                                                 *source_binding_files(source_binding)]}
    sources = source_inventory()
    g = module('gray_review'); summary = g.td(Path(reports['summary.tcldata']).read_text())
    require(summary.get('checkpoint_sha256') == checkpoint, 'Summary checkpoint mismatch')
    pending = g.tcl_list(summary['pending_reviews'])
    R.pending_categories(pending, {})
    result = {'physical': physical(g, summary)}
    result['source_specialization'] = module('admission_specialization_review').review_sources(**source_binding)
    successful(result['source_specialization'], 'Source specialization')
    require(result['source_specialization'].get('source_elided_bits_proven') is True and
            result['source_specialization'].get('source_hardware_equivalence_proven') is False and
            result['source_specialization'].get('route_qualified') is False,
            'Source specialization scope differs')
    result['gray'] = g.review_final(reports['objects.tcldata'], reports['supplemental.tcldata'],
                                    reports['timing_all_ids.xdc'], auxiliary['gray_vendor'], checkpoint)
    require(result['gray'].get('gray_evidence_consistent') is True, 'Gray predicate unresolved')
    result['no_clock'] = module('no_clock_review').review_raw(g, auxiliary['shell_objects'],
        reports['objects.tcldata'], reports['supplemental.tcldata'], auxiliary['shell_waivers'], reports['waivers.rpt'])
    require(result['no_clock'].get('predicate_satisfied') is True and
            result['no_clock']['pins'] == int(result['physical']['checks']['no_clock']), 'No-clock predicate unresolved')
    result['reset_status'] = module('reset_status_review').review_artifact(
        reports=reports, auxiliary=auxiliary, checkpoint_sha256=checkpoint)
    successful(result['reset_status'], 'Reset/status')
    result['exceptions'] = module('exception_review').review_artifact(reports=reports, auxiliary=auxiliary,
        checkpoint_sha256=checkpoint, prerequisites={name: result[name] for name in ('gray', 'no_clock', 'reset_status')})
    successful(result['exceptions'], 'Exception/methodology')
    for key, dcp in (('linked_native', linked_checkpoint), ('native', checkpoint)):
        path = auxiliary[key]
        result[key] = module('native_review').review_artifact(topology_path=path,
            topology_sha256=record(path)['sha256'], checkpoint_sha256=dcp)
        successful(result[key], key)
    bindings = {}
    for name, role, aux, dcp in (('linked', 'linked_ready', 'linked_ready', linked_checkpoint),
                               ('routed', 'capture', 'ready', checkpoint)):
        helpers = {name: capture_receipts[role]['files'][name]['sha256']
                   for name in ('capture_ready.tcl', 'capture_native.tcl')}
        bindings[name] = dict(topology_path=auxiliary[aux], topology_sha256=record(auxiliary[aux])['sha256'],
            checkpoint_sha256=dcp, helper_sha256=helpers['capture_ready.tcl'],
            native_helper_sha256=helpers['capture_native.tcl'])
    result['ready_pair'] = module('ready_pair_review').review_pair(**bindings)
    require(result['ready_pair'].get('observed_ready_equivalent') is True and
            result['ready_pair'].get('issues') == [], 'Observed LINK/routed ready equivalence unresolved')
    structural = module('structural_review').review_artifact(reports=reports, auxiliary=auxiliary,
        checkpoint_sha256=checkpoint, linked_checkpoint_sha256=linked_checkpoint,
        prerequisites={**{name: result[name] for name in ('linked_native', 'native', 'ready_pair', 'source_specialization')},
            'source_binding': source_binding,
            'receipt_sha256': {'link': E.digest(capture_receipts['linked_native']['linked_receipt_sha256']),
                               'refine': E.digest(capture_receipts['capture']['refine_receipt_sha256'])}})
    successful(structural, 'Structural/replica closure')
    nested = {name: result[name]['pending_reviews'] for name in ('linked_native', 'native', 'ready_pair')}
    R.pending_categories(pending, nested)
    remaining = set().union(*(set(items) for items in nested.values())) - {
        'current_reset_status', 'physical_timing_and_exceptions', 'C1_final_artifact_qualification'}
    require(isinstance(structural.get('resolved_pending_reviews'), list) and
            len(structural['resolved_pending_reviews']) == len(set(structural['resolved_pending_reviews'])) and
            set(structural['resolved_pending_reviews']) == remaining,
            'Structural closure does not resolve exact nested obligations')
    result['structural'] = structural
    result['protected_pulse'] = protected_pulse_context(reports=reports, auxiliary=auxiliary,
                                                      physical_result=result['physical'])
    rows = artifact_records(g, reports['objects.tcldata'], summary)
    result['protected_cdc'] = protected_cdc_context(g, rows, auxiliary)
    result['dispositions'] = review_dispositions(read_json(disposition_path), checkpoint=checkpoint,
        capture_receipt=capture_receipt_sha256, records=rows,
        auxiliary={key: record(path)['sha256'] for key, path in auxiliary.items()})
    require(source_inventory() == sources and all(record(path) == row for path, row in before.items()),
            'Review inputs/source changed during replay')
    result.update(resolved_pending_reviews=sorted(pending), original_nested_pending_reviews=nested,
                  route_qualified=True, hardware_measured=False)
    return json.loads(encode(result))


def verify_review(bundle, candidate):
    require(candidate.get('kind') == 'verified-c1-alex-portable-candidate-evidence' and
            candidate.get('schema') == 2 and candidate.get('route_qualified') is False and
            candidate.get('hardware_measured') is False, 'Full schema2 provenance replay required')
    receipt = bundle.receipt('review')
    R.validate_receipt(bundle.path('review', R.RECEIPT_NAME).read_bytes(),
        bundle.path('review', R.INPUTS_NAME).read_bytes(), bundle.manifest, source_inventory())
    require(receipt['status'] == 'complete' and receipt['route_qualified'] is True,
            'No completed C1 final review receipt')
    inputs = bundle.read('review', R.INPUTS_NAME, receipt['inputs_sha256'])
    reports = {name: bundle.path(row['role'], row['relative_path']) for name, row in inputs['reports'].items()}
    auxiliary = {name: bundle.path(row['role'], row['relative_path']) for name, row in inputs['auxiliary'].items()}
    result = replay(reports=reports, auxiliary=auxiliary, disposition_path=bundle.path('review', 'disposition.json'),
        checkpoint=inputs['checkpoints']['final'], linked_checkpoint=inputs['checkpoints']['linked'],
        capture_receipt_sha256=inputs['receipt_sha256']['capture'],
        capture_receipts={role: bundle.receipt(role) for role in E.RAW_ROLES},
        source_binding=hardware_source_binding(bundle))
    require(bundle.read('review', R.DECISIONS_NAME, receipt['decisions_sha256']) == result,
            'Recorded C1 decisions differ from current raw replay')
    require(set(candidate['pending_reviews']) == set(result['resolved_pending_reviews']),
            'Candidate pending review categories differ')
    return result

