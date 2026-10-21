#!/usr/bin/env python3
"""Add independently replayed C1 review evidence to an immutable schema2 bundle.

No AWS, FPGA, synthesis, or bundled-code execution. The input specification is
externally hash-pinned; output is fresh-only and original evidence is preserved.
Only trusted final_review predicates can discharge pending candidate obligations.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from boards.c1_f2_alex import candidate_evidence as E, collect_candidate as C
from boards.c1_f2_alex import evidence as Q, final_review as F, review_contract as R


def bound_file(row):
    E.require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'}, 'Malformed external file binding')
    path = Path(E.absolute(row['path'])); E.no_symlink(path)
    expected = {key: row[key] for key in ('bytes', 'sha256')}; E.metadata(expected)
    stamp = C.regular(path)
    E.require(stamp[2] == expected['bytes'] and E.sha(path) == expected['sha256'] and
              C.regular(path) == stamp, 'External file changed: ' + str(path))
    return path, expected


def checked_spec(spec):
    E.require(isinstance(spec, dict) and set(spec) == {'schema', 'kind', 'candidate_directory',
        'candidate_manifest_sha256', 'review_source', 'reference_files', 'disposition_file'},
        'Invalid qualified collection fields')
    E.require(type(spec['schema']) is int and spec['schema'] == 1 and
              spec['kind'] == 'c1-alex-qualified-collection-inputs', 'Wrong qualified collection schema')
    root = Path(E.absolute(spec['candidate_directory'])); E.no_symlink(root)
    E.digest(spec['candidate_manifest_sha256'])
    source = spec['review_source']
    E.require(isinstance(source, dict) and set(source) == {'path', 'revision'} and
              Path(E.absolute(source['path'])) == ROOT, 'Reviewer must be the current trusted checkout')
    R.revision(source['revision']); E.no_symlink(ROOT)
    E.require(isinstance(spec['reference_files'], dict) and
              set(spec['reference_files']) == R.REFERENCE_AUXILIARY, 'Incomplete/extra reference evidence')
    references = {key: bound_file(row) for key, row in spec['reference_files'].items()}
    disposition = bound_file(spec['disposition_file'])
    return root, references, disposition


def source_bytes(revision):
    """Match trusted live reviewer files to immutable approved Git object bytes."""
    inventory = F.source_inventory()
    git = C.GitSource(ROOT, revision)
    values = {name: git.read(name) for name in inventory}
    E.require({name: C.record_bytes(raw) for name, raw in values.items()} == inventory,
              'Current reviewer differs from approved Git revision')
    return inventory, values


def proof_ref(manifest, role, name):
    return {'role': role, 'relative_path': name, **R.inventory_record(manifest, role, name)}


def inputs_for(manifest):
    return dict(schema=1, kind='c1-alex-final-review-inputs', hardware=manifest['hardware'],
        checkpoints={'linked': R.inventory_record(manifest, 'link', 'post_link.dcp')['sha256'],
                     'final': R.inventory_record(manifest, 'refine', 'post_route.dcp')['sha256']},
        receipt_sha256={role: manifest['receipts'][role]['sha256'] for role in E.schema_receipts(2)},
        reports={name: proof_ref(manifest, 'capture', 'final/' + name) for name in E.FINAL_REPORTS},
        auxiliary={**{key: proof_ref(manifest, role, name) for key, (role, name) in R.RAW_AUXILIARY.items()},
                   **{key: proof_ref(manifest, 'review', 'auxiliary/' + key) for key in R.REFERENCE_AUXILIARY}},
        disposition_sha256=R.inventory_record(manifest, 'review', 'disposition.json')['sha256'])


def archive_bundle(bundle, destination, manifest_sha):
    """Deterministic inventory-only tar; schema3-aware streaming verification."""
    manifest = Q.manifest_data((bundle / E.MANIFEST).read_bytes(), manifest_sha)
    with destination.open('xb') as stream, tarfile.open(fileobj=stream, mode='w|', format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(set(manifest['files']) | {E.MANIFEST}):
            path = bundle / E.relative(name); E.no_symlink(path)
            row = C.record_bytes(path.read_bytes()) if name == E.MANIFEST else manifest['files'][name]
            E.require(path.stat().st_size == row['bytes'] and E.sha(path) == row['sha256'], 'File changed before archival')
            info = tarfile.TarInfo(name); info.size = row['bytes']; info.mode = 0o644
            info.uid = info.gid = info.mtime = 0; info.uname = info.gname = ''
            with path.open('rb') as source: archive.addfile(info, source)
    return Q.inspect_archive(destination, manifest_sha)


def collect(spec, out, *, spec_raw=None):
    root, references, disposition = checked_spec(spec)
    out = Path(E.absolute(str(out))); E.no_symlink(out)
    protected = [root, ROOT, *[path for path, _ in references.values()], disposition[0]]
    E.require(not out.exists() and all(not out.is_relative_to(path) and not path.is_relative_to(out)
              for path in protected), 'Fresh output must be outside every original input/source')
    raw = F.encode(spec) if spec_raw is None else spec_raw
    E.require(E.decode(raw) == spec, 'Specification bytes differ')
    out.mkdir(parents=True, exist_ok=False)
    C.write(out / 'collection-inputs.json', raw)
    started = C.now()
    try:
        candidate = E.verify_bundle(root, spec['candidate_manifest_sha256'])
        E.require(candidate.get('schema') == 2 and candidate.get('kind') == 'verified-c1-alex-portable-candidate-evidence'
                  and candidate.get('route_qualified') is False and candidate.get('review_required') is True
                  and candidate.get('hardware_measured') is False, 'Full pending schema2 candidate required')
        base = E.Bundle(root, spec['candidate_manifest_sha256'])
        E.require(base.manifest['schema'] == 2, 'Only schema2 can receive an additive review role')
        sources, blobs = source_bytes(spec['review_source']['revision'])
        total = sum(row['bytes'] for row in base.files.values()) + sum(row['bytes'] for _, row in references.values()) + disposition[1]['bytes'] + sum(len(value) for value in blobs.values())
        E.require(total <= E.MAX_TOTAL and shutil.disk_usage(out).free >= 2 * total + (1 << 30), 'Insufficient bundle/archive capacity')
        bundle = out / 'bundle'; bundle.mkdir()
        manifest = copy.deepcopy(base.manifest); manifest['schema'] = 3
        review_prefix, source_prefix = 'runs/review', 'sources/review'
        for prefix in (review_prefix, source_prefix):
            E.require(all(not Q.overlaps(prefix, row['bundle_relative_path']) for row in
                [*manifest['roots'].values(), *manifest['source_snapshots'].values()]), 'New review root overlaps original role')
        review = bundle / review_prefix
        manifest['roots']['review'] = {'original_absolute_path': str(review), 'bundle_relative_path': review_prefix}
        manifest['source_snapshots']['review'] = {'revision': spec['review_source']['revision'], 'bundle_relative_path': source_prefix}
        for name, row in sorted(base.files.items()): C.copy_file(root / E.relative(name), bundle / name, row)
        def add(name, value):
            E.require(name not in manifest['files'], 'Review would overwrite original evidence')
            C.write(bundle / E.relative(name), value); manifest['files'][name] = C.record_bytes(value)
        for name, value in blobs.items(): add(source_prefix + '/' + name, value)
        for key, (path, row) in references.items():
            name = review_prefix + '/auxiliary/' + E.relative(key)
            C.copy_file(path, bundle / name, row); manifest['files'][name] = row
        C.copy_file(disposition[0], review / 'disposition.json', disposition[1])
        manifest['files'][review_prefix + '/disposition.json'] = disposition[1]
        inputs = inputs_for(manifest)
        R.validate_inputs(inputs, manifest)
        add(review_prefix + '/' + R.INPUTS_NAME, F.encode(inputs))
        def resolve(row): return bundle / manifest['roots'][row['role']]['bundle_relative_path'] / row['relative_path']
        decisions = F.replay(reports={key: resolve(row) for key, row in inputs['reports'].items()},
            auxiliary={key: resolve(row) for key, row in inputs['auxiliary'].items()}, disposition_path=review / 'disposition.json',
            checkpoint=inputs['checkpoints']['final'], linked_checkpoint=inputs['checkpoints']['linked'],
            capture_receipt_sha256=inputs['receipt_sha256']['capture'],
            capture_receipts={role: base.receipt(role) for role in E.RAW_ROLES})
        E.require(decisions.get('route_qualified') is True and decisions.get('hardware_measured') is False and
                  set(decisions['resolved_pending_reviews']) == set(candidate['pending_reviews']), 'Final reviewer did not close candidate obligations')
        add(review_prefix + '/' + R.DECISIONS_NAME, F.encode(decisions))
        outputs = {name[len(review_prefix) + 1:]: row for name, row in manifest['files'].items() if name.startswith(review_prefix + '/')}
        receipt = dict(schema=1, kind='c1-alex-final-semantic-review', status='complete', route_qualified=True,
            hardware_measured=False, hardware_actions=False, checkpoint_sha256=inputs['checkpoints']['final'],
            linked_checkpoint_sha256=inputs['checkpoints']['linked'],
            inputs_sha256=manifest['files'][review_prefix + '/' + R.INPUTS_NAME]['sha256'],
            decisions_sha256=manifest['files'][review_prefix + '/' + R.DECISIONS_NAME]['sha256'],
            source_revision=spec['review_source']['revision'], source_inputs=sources, outputs=outputs)
        add(review_prefix + '/' + R.RECEIPT_NAME, F.encode(receipt))
        manifest['receipts']['review'] = {'relative_path': review_prefix + '/' + R.RECEIPT_NAME,
            'sha256': manifest['files'][review_prefix + '/' + R.RECEIPT_NAME]['sha256']}
        R.validate_receipt((review / R.RECEIPT_NAME).read_bytes(), (review / R.INPUTS_NAME).read_bytes(), manifest, sources)
        manifest_raw = F.encode(manifest); manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        C.write(bundle / E.MANIFEST, manifest_raw)
        def recheck():
            E.require(E.verify_bundle(root, spec['candidate_manifest_sha256']) == candidate, 'Original candidate changed')
            E.require(source_bytes(spec['review_source']['revision']) == (sources, blobs), 'Reviewer source changed')
            checked_spec(spec)
        recheck()
        verified = Q.verify_bundle(bundle, manifest_sha)
        E.require(verified.get('kind') == 'verified-c1-alex-portable-evidence' and verified.get('route_qualified') is True
                  and verified.get('review_required') is False, 'Complete portable qualification failed')
        archived = archive_bundle(bundle, out / 'evidence.tar', manifest_sha)
        recheck()
        E.require(Q.verify_bundle(bundle, manifest_sha) == verified, 'Qualified bundle changed during archive creation')
        result = dict(schema=3, kind='c1-alex-qualified-evidence-collection', state='complete', instance_id=E.INSTANCE,
            started_utc=started, finished_utc=C.now(), inputs_sha256=hashlib.sha256(raw).hexdigest(),
            original_candidate_manifest_sha256=spec['candidate_manifest_sha256'], manifest_sha256=manifest_sha,
            review_receipt_sha256=manifest['receipts']['review']['sha256'],
            collector_sha256=E.sha(Path(__file__)), verifier_sha256=E.sha(ROOT / 'boards/c1_f2_alex/evidence.py'),
            archive={'relative_path': 'evidence.tar', 'sha256': archived['archive_sha256'], 'bytes': (out / 'evidence.tar').stat().st_size},
            bundle_relative_path='bundle', verified=verified, hardware_actions=False)
        C.write(out / 'collection.json', F.encode(result))
        return {**result, 'receipt_sha256': E.sha(out / 'collection.json')}
    except BaseException as error:
        C.write(out / 'failure.json', F.encode(dict(schema=1, kind='c1-alex-qualified-collection-failure', state='failed',
            started_utc=started, failed_utc=C.now(), error=f'{type(error).__name__}: {error}', hardware_actions=False,
            instruction='Preserve original and partial evidence; correct the cause and use a fresh output.')))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True); p.add_argument('--spec-sha256', required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args(argv)
    path = Path(E.absolute(str(a.spec))); E.no_symlink(path)
    E.require(path.stat().st_size <= E.MAX_MANIFEST, 'Oversized collection specification')
    raw = path.read_bytes(); spec = R.decode(raw, a.spec_sha256)
    result = collect(spec, a.out, spec_raw=raw)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__': main()
