#!/usr/bin/env python3
"""Replay immutable artifact review, then optionally write a fresh review receipt.
Never changes collected Vivado reports, checkpoints, timing exceptions or rules.
An artifact-specific disposition must be pinned in reviewed_artifacts.json.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POLICY = HERE / 'reviewed_artifacts.json'
MODULES = ('evidence.py', 'build.py', 'final_review.py', 'gray_review.py', 'no_clock_review.py', 'exception_review.py', 'reset_status_review.py', 'native_ready_review.py', 'reviewed_artifacts.json')
CLASSES = {
    'inherited-protected-shell': {'cdc', 'check_timing'},
    'reviewed-vendor-fifo-cdc': {'cdc'},
    'reviewed-vendor-handshake-cdc': {'cdc'},
    'reviewed-reset-status-cdc': {'cdc'},
    'reviewed-memory-output-register-suggestion': {'methodology'},
    'reviewed-clock-placement-suggestion': {'methodology'},
    'reviewed-dsp-pipeline-suggestion': {'drc'},
    'reviewed-disabled-ddr-support': {'drc', 'methodology'},
    'inherited-visible-shell': {'drc', 'methodology'},
    'reviewed-vendor-exception': {'exceptions'},
    'reviewed-current-cdc-summary': {'methodology'},
    'reviewed-unused-net-report': {'drc'},
}


def require(ok, message):
    if not ok: raise ValueError(message)


def module(name):
    path = HERE / (name + '.py')
    require(path.is_file(), 'Required reviewed predicate unavailable: ' + name)
    spec = importlib.util.spec_from_file_location('c2_review_' + name, path)
    value = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(value)
    finally: sys.dont_write_bytecode = old
    return value


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def record(path):
    e = module('evidence'); path = Path(path); e.no_symlink(path)
    require(path.is_file() and path.stat().st_size <= e.MAX_FILE, 'Invalid review input file')
    return {'sha256': e.sha(path), 'bytes': path.stat().st_size}


def source_inventory():
    result = {}
    for name in MODULES:
        path = HERE / name
        require(path.is_file(), 'Missing reviewed predicate source: ' + name)
        result['boards/c2_f2_alex/' + name] = record(path)
    return result


def read_json(path):
    e = module('evidence'); require(Path(path).stat().st_size <= e.MAX_MANIFEST, 'Oversized JSON')
    return e.decode(Path(path).read_bytes())


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
    require(summary.get('schema') == '1' and summary.get('stage') == 'final_route' and summary.get('hardware_measured') == '0', 'Not a final route collection')
    require(summary.get('route_qualified') == '0', 'Expected original pending collector result')
    require(g.tcl_list(summary.get('blockers', '')) == [], 'Physical/DRC blocker cannot be overridden by review')
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


def review_dispositions(disposition, *, checkpoint, receipt_sha, records, auxiliary):
    expected_fields = {'schema','kind','checkpoint_sha256','collection_receipt_sha256','record_inventory_sha256','auxiliary_sha256','dispositions'}
    require(set(disposition) == expected_fields and disposition['schema'] == 1 and disposition['kind'] == 'c2-alex-artifact-dispositions', 'Malformed reviewed dispositions')
    require(disposition['checkpoint_sha256'] == checkpoint and disposition['collection_receipt_sha256'] == receipt_sha, 'Dispositions bind another artifact/collection')
    require(disposition['auxiliary_sha256'] == auxiliary, 'Reviewed auxiliary evidence hashes differ')
    inventory = sorted(canonical(row) for row in records)
    require(disposition['record_inventory_sha256'] == canonical(inventory), 'Current findings/exceptions differ from reviewed inventory')
    by_id = {canonical(row): row for row in records}; covered = Counter()
    for item in disposition['dispositions']:
        require(set(item) == {'classification','family','record_sha256','evidence','rationale'}, 'Malformed disposition fields')
        require(item['classification'] in CLASSES and item['family'] in CLASSES[item['classification']], 'Unreviewed classification/family')
        require(isinstance(item['rationale'],str) and item['rationale'].strip() and isinstance(item['evidence'],list) and item['evidence'] and set(item['evidence']) <= set(auxiliary), 'Missing narrow disposition evidence/rationale')
        require(isinstance(item['record_sha256'],list) and item['record_sha256'], 'Empty disposition record set')
        for ident in item['record_sha256']:
            require(ident in by_id and by_id[ident]['family'] == item['family'], 'Disposition includes unknown/wrong-family record')
            covered[ident] += 1
    require(covered == Counter(inventory), 'Unreviewed or multiply disposed current records')
    return {'record_inventory_sha256': canonical(inventory), 'records':len(inventory),'dispositions':len(disposition['dispositions'])}


def linked_checkpoint(auxiliary):
    """Read a bound raw header; portable replay also checks the LINK artifact."""
    require('native_ready_linked' in auxiliary, 'Missing linked native-ready witness')
    g = module('gray_review')
    headers = [g.pairs(row) for row in g.records(auxiliary['native_ready_linked']) if row and row[0] == 'schema']
    require(len(headers) == 1, 'Missing/duplicate linked native-ready identity')
    header = headers[0]
    require(header.get('schema') == '1' and header.get('kind') == 'linked_native_ready' and header.get('route_qualified') == '0', 'Wrong linked native-ready witness')
    return module('evidence').digest(header.get('checkpoint_sha256'))


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


def replay(*, reports, auxiliary, disposition_path, checkpoint, collection_receipt_sha256, linked_checkpoint_sha256=None):
    g = module('gray_review'); n = module('no_clock_review')
    policy = read_json(POLICY)
    require(set(policy) == {'schema','artifacts'} and policy['schema'] == 1, 'Malformed reviewed artifact policy')
    approved = policy['artifacts'].get(checkpoint)
    require(approved and record(disposition_path)['sha256'] == approved, 'Artifact dispositions are not source-reviewed for this DCP')
    required_aux = {'gray_vendor','shell_objects','shell_waivers','shell_complete','reset_status','native_ready_linked','native_ready_refined'}
    require(required_aux <= set(auxiliary), 'Required raw semantic evidence absent')
    summary = g.td(Path(reports['summary.tcldata']).read_text())
    require(summary.get('checkpoint_sha256') == checkpoint, 'Collection summary binds another DCP')
    result = {'physical': physical(g, summary)}
    linked = linked_checkpoint(auxiliary)
    require(linked_checkpoint_sha256 is None or linked == linked_checkpoint_sha256, 'Native witness binds another LINK checkpoint')
    native = module('native_ready_review')
    result['native_ready'] = native.review_artifact(reports=reports, auxiliary=auxiliary, checkpoint_sha256=checkpoint, linked_checkpoint_sha256=linked)
    require(isinstance(result['native_ready'], dict) and result['native_ready'].get('issues') == [] and result['native_ready'].get('canonical_facts'), 'Linked-to-routed native-ready equivalence unresolved')
    shell = g.td(Path(auxiliary['shell_complete']).read_text())
    require(shell.get('kind') == 'original_shell_reference_captured' and shell.get('checkpoint_sha256') == module('build').SHELL_SHA256, 'Original shell evidence not bound')
    result['gray'] = g.review_final(reports['objects.tcldata'], reports['supplemental.tcldata'], reports['timing_all_ids.xdc'], auxiliary['gray_vendor'], checkpoint)
    require(result['gray']['gray_evidence_consistent'], 'Final gray predicate unresolved')
    result['no_clock'] = n.review_raw(g, auxiliary['shell_objects'], reports['objects.tcldata'], reports['supplemental.tcldata'], auxiliary['shell_waivers'], reports['waivers.rpt'])
    require(result['no_clock']['predicate_satisfied'], 'Inherited no-clock/rule predicate unresolved')
    require(result['no_clock']['pins'] == int(result['physical']['checks']['no_clock']), 'No-clock summary and typed finding inventory differ')
    # These reviewed adapters consume raw evidence. Their absence or unresolved
    # diagnostics cannot be replaced with a claimed-success JSON receipt.
    reset = module('reset_status_review')
    require(hasattr(reset,'review_artifact'), 'Reset/status artifact predicate is not implemented')
    result['reset_status'] = reset.review_artifact(reports=reports, auxiliary=auxiliary, checkpoint_sha256=checkpoint)
    require(isinstance(result['reset_status'],dict) and result['reset_status'].get('issues') == [] and 'canonical_facts' in result['reset_status'], 'Reset/status predicate unresolved/incomplete')
    exception = module('exception_review')
    require(hasattr(exception,'review_artifact'), 'Exception artifact predicate is not implemented')
    result['protected_pulse'] = protected_pulse_context(reports=reports, auxiliary=auxiliary, physical_result=result['physical'])
    result['exceptions'] = exception.review_artifact(reports=reports, auxiliary=auxiliary, checkpoint_sha256=checkpoint,
        prerequisites={name: result[name] for name in ('gray', 'no_clock', 'reset_status')})
    require(isinstance(result['exceptions'],dict) and result['exceptions'].get('issues') == [] and 'canonical_facts' in result['exceptions'], 'Exception/methodology predicate unresolved/incomplete')
    rows = artifact_records(g, reports['objects.tcldata'], summary)
    result['dispositions'] = review_dispositions(read_json(disposition_path), checkpoint=checkpoint, receipt_sha=collection_receipt_sha256, records=rows, auxiliary={name:record(path)['sha256'] for name,path in auxiliary.items()})
    pending = set(g.tcl_list(summary['pending_reviews']))
    expected = {'current_vendor_CDC_and_gray_max_delay_scope','current_exception_coverage_and_ignored_paths','current_DRC_methodology_and_severity_configuration','exact_shell_inherited_witnesses','current_9SI_8MI_vendor_simulation_and_generated_IP_binding','current_routed_native_ready_equivalence'}
    require(pending == expected, 'Unknown/missing original pending review category')
    result['resolved_pending_reviews'] = sorted(pending)
    # Receipt replay compares JSON values: normalize tuple records and integer
    # histogram keys before both writing and comparing after relocation.
    return json.loads(encode(result))


def verify_review(x):
    """Replay a schema2 portable bundle; executes only trusted local validators."""
    e = module('evidence'); receipt = x.receipt('review')
    require(receipt.get('schema') == 1 and receipt.get('kind') == 'c2-alex-final-semantic-review' and receipt.get('status') == 'complete' and receipt.get('route_qualified') is True and receipt.get('hardware_measured') is False and receipt.get('hardware_actions') is False, 'Missing successful immutable final review')
    inputs = x.read('review','review-inputs.json',receipt['inputs_sha256'])
    require(set(inputs) == {'schema','kind','checkpoint_sha256','receipt_sha256','reports','auxiliary','disposition_sha256'} and inputs['schema'] == 1 and inputs['kind'] == 'c2-alex-final-review-inputs', 'Malformed final review inputs')
    require(set(inputs['receipt_sha256']) == {'qualify','refine','simulation'}, 'Review receipt chain incomplete')
    checkpoint = x.record('refine','post_route.dcp')['sha256']
    require(receipt['checkpoint_sha256'] == inputs['checkpoint_sha256'] == checkpoint, 'Review checkpoint mismatch')
    for role in ('qualify','refine','simulation'):
        require(inputs['receipt_sha256'][role] == x.manifest['receipts'][role]['sha256'], 'Review receipt chain differs')
    sources = source_inventory()
    require(receipt['source_inputs'] == sources, 'Review verifier source differs')
    x.source('review',receipt['source_revision'],sources,current=sources)
    x.inventory('review',receipt['outputs'],exact=x.names('review')-{'review-result.json'})
    x.inventory('qualify',inputs['reports'])
    reports = {name:x.path('qualify','qualification/'+name) for name in e.FINAL_REPORTS}
    auxiliary = {}
    for name,row in inputs['auxiliary'].items():
        path = e.relative(row['relative_path']); require(path.startswith('auxiliary/'), 'Auxiliary relocation escapes review root')
        require(x.record('review',path) == {'sha256':row['sha256'],'bytes':row['bytes']}, 'Auxiliary binding differs')
        auxiliary[name] = x.path('review',path)
    disposition = x.path('review','disposition.json')
    require(x.record('review','disposition.json')['sha256'] == inputs['disposition_sha256'], 'Disposition binding differs')
    linked = x.record('link', 'post_link.dcp')['sha256']
    require(linked_checkpoint(auxiliary) == linked, 'Native witness does not bind the bundled successful LINK DCP')
    result = replay(reports=reports, auxiliary=auxiliary, disposition_path=disposition, checkpoint=checkpoint, collection_receipt_sha256=inputs['receipt_sha256']['qualify'], linked_checkpoint_sha256=linked)
    require(x.read('review','review-decisions.json',receipt['decisions_sha256']) == result, 'Recorded review differs from raw predicate replay')
    return result


def exclusive(path, raw):
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def finalize(request_path, request_sha256, output):
    e=module('evidence'); request_path=Path(os.path.abspath(request_path)); output=Path(os.path.abspath(output))
    require(record(request_path)['sha256']==e.digest(request_sha256),'Untrusted review request')
    request=read_json(request_path)
    require(set(request)=={'schema','kind','instance_id','roots','receipt_sha256','auxiliary','disposition','source_revision'} and request['schema']==1 and request['kind']=='c2-alex-final-review-request' and request['instance_id']==e.INSTANCE,'Malformed review request')
    require(set(request['roots'])==set(request['receipt_sha256'])=={'qualify','refine','simulation'},'Incomplete review receipt inputs')
    require(re.fullmatch('[0-9a-f]{40}',request['source_revision']),'Review source revision must be full commit')
    roots={k:Path(e.absolute(v)) for k,v in request['roots'].items()};original={};receipts={}
    def bind(path,wanted=None):
        actual=record(path)
        if wanted is not None: require(actual['sha256']==e.digest(wanted),'Trusted input hash mismatch: '+str(path))
        original[str(path)]=actual;return actual
    bind(request_path, request_sha256)
    for role,root in roots.items():
        name=e.RECEIPTS[role];bind(root/name,request['receipt_sha256'][role]);receipts[role]=read_json(root/name)
    collection=receipts['qualify'];refine=receipts['refine'];simulation=receipts['simulation']
    require(collection.get('stage')=='qualify' and collection.get('status')=='pending-review' and collection.get('returncode')==0 and collection.get('route_qualified') is False and collection.get('hardware_actions') is False,'Failed/incomplete collector cannot be finalized')
    require(refine.get('stage')=='refine' and refine.get('status')=='complete' and refine.get('returncode')==0,'Failed refinement')
    require(simulation.get('state')=='passed' and simulation.get('hardware_actions') is False,'Simulation not passed')
    checkpoint=bind(roots['refine']/'post_route.dcp',refine['checkpoint_sha256'])['sha256']
    require(collection.get('checkpoint_sha256')==checkpoint and collection.get('checkpoint')==str(roots['refine']/'post_route.dcp'),'Collector used another checkpoint')
    stage_path=roots['qualify']/'stage-inputs.json';bind(stage_path,collection['inputs_sha256']);stage=read_json(stage_path)
    require(stage.get('previous_receipt_sha256')==request['receipt_sha256']['refine'] and stage.get('simulation')=={'run':str(roots['simulation']),'receipt_sha256':request['receipt_sha256']['simulation'],'inputs_sha256':simulation['inputs_sha256']},'Review ancestry/simulation binding differs')
    for path in roots['qualify'].rglob('*'):
        e.no_symlink(path); require(path.is_file() or path.is_dir(), 'Special collector output')
    actual={p.relative_to(roots['qualify']).as_posix() for p in roots['qualify'].rglob('*') if p.is_file()}
    require(actual==set(collection['outputs'])|{'stage-result.json','stage-inputs.json'} and 'failure.json' not in actual,'Collector output inventory differs')
    for name,wanted in collection['outputs'].items():require(bind(roots['qualify']/e.relative(name))==wanted,'Collected output changed')
    reports={name:roots['qualify']/'qualification'/name for name in e.FINAL_REPORTS}
    require((roots['qualify']/'qualification/NOT_QUALIFIED.txt').is_file(),'Original pending marker must remain')
    auxiliary={}
    for name,item in request['auxiliary'].items():
        require(re.fullmatch('[a-z][a-z0-9_]*',name) and set(item)=={'path','sha256','bytes'},'Unsafe auxiliary key/metadata');path=Path(e.absolute(item['path']));require(bind(path)=={'sha256':e.digest(item['sha256']),'bytes':item['bytes']},'Auxiliary input changed');auxiliary[name]=path
    require(set(request['disposition']) == {'path','sha256','bytes'}, 'Malformed disposition metadata')
    disposition=Path(e.absolute(request['disposition']['path']));require(bind(disposition)=={'sha256':e.digest(request['disposition']['sha256']),'bytes':request['disposition']['bytes']},'Disposition input changed')
    protected = list(roots.values()) + [HERE, request_path, disposition] + list(auxiliary.values())
    require(all(not output.is_relative_to(path) and not path.is_relative_to(output) for path in protected), 'Review output must be outside immutable inputs and validator sources')
    sources=source_inventory()
    result=replay(reports=reports,auxiliary=auxiliary,disposition_path=disposition,checkpoint=checkpoint,collection_receipt_sha256=request['receipt_sha256']['qualify'])
    for path,wanted in original.items():require(record(path)==wanted,'Input changed during final review')
    e.no_symlink(output);output.mkdir(parents=False,exist_ok=False)
    exclusive(output/'review-request.json',request_path.read_bytes());exclusive(output/'disposition.json',disposition.read_bytes());(output/'auxiliary').mkdir()
    require(record(output/'review-request.json') == original[str(request_path)] and record(output/'disposition.json') == original[str(disposition)], 'Copied review controls changed')
    aux={}
    for name,path in auxiliary.items():
        target=output/'auxiliary'/name
        with path.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1<<20)
        wanted=record(path);require(record(target)==wanted,'Auxiliary copy changed');aux[name]={'relative_path':'auxiliary/'+name,**wanted}
    inputs={'schema':1,'kind':'c2-alex-final-review-inputs','checkpoint_sha256':checkpoint,'receipt_sha256':request['receipt_sha256'],'reports':{'qualification/'+name:record(path) for name,path in reports.items()},'auxiliary':aux,'disposition_sha256':record(disposition)['sha256']}
    exclusive(output/'review-inputs.json',encode(inputs));exclusive(output/'review-decisions.json',encode(result))
    for path,wanted in original.items():require(record(path)==wanted,'Input changed before final receipt')
    require(source_inventory()==sources,'Reviewer source changed during execution')
    outputs={p.relative_to(output).as_posix():record(p) for p in output.rglob('*') if p.is_file()}
    receipt={'schema':1,'kind':'c2-alex-final-semantic-review','status':'complete','route_qualified':True,'hardware_measured':False,'hardware_actions':False,'checkpoint_sha256':checkpoint,'inputs_sha256':record(output/'review-inputs.json')['sha256'],'decisions_sha256':record(output/'review-decisions.json')['sha256'],'source_revision':request['source_revision'],'source_inputs':sources,'outputs':outputs,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    exclusive(output/'review-result.json',encode(receipt));return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['finalize-review']);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(finalize(a.inputs,a.inputs_sha256,a.out),indent=2,sort_keys=True))


if __name__=='__main__':main()
