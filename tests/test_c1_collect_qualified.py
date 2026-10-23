"""Additive collection boundaries with synthetic raw data, never deployable evidence.

The real receipt graph, schema3 binding and archive reader run here. Only the
semantic predicates and immutable reviewer Git bytes are test fixtures; their
independent negative suites test physical and source acceptance separately.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
from boards.c1_f2_alex import collect_qualified as p

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('c1_collection_raw_fixture', ROOT / 'tests/test_c1_raw_candidate_evidence.py')
raw = importlib.util.module_from_spec(spec); spec.loader.exec_module(raw)


@pytest.fixture(params=('i-06cf8792fb8f15534','i-0ec8d33362b3100be'))
def context(tmp_path, monkeypatch, request):
    tmp_path = tmp_path.resolve()
    root, manifest, digest = raw.fixture_raw(tmp_path, monkeypatch)
    manifest['instance_id']=request.param;digest=raw.refresh(root,manifest)
    e = raw.e
    original_modules = e.local_module
    monkeypatch.setattr(e, 'local_module', lambda name: p.F if name == 'final_review' else original_modules(name))
    monkeypatch.setattr(p, 'E', e)
    monkeypatch.setattr(p.Q, 'E', e)
    blobs = {name: ('# Synthetic reviewer identity: ' + name + '\n').encode() for name in p.R.REVIEW_SOURCES}
    sources = {name: p.C.record_bytes(value) for name, value in blobs.items()}
    monkeypatch.setattr(p.F, 'source_inventory', lambda: copy.deepcopy(sources))
    class Git:
        def __init__(self, path, revision):
            assert Path(path) == ROOT and revision == '1' * 40
        def read(self, name): return blobs[name]
    monkeypatch.setattr(p.C, 'GitSource', Git)
    decisions = dict(kind='synthetic-test-decisions-never-deploy', route_qualified=True, hardware_measured=False,
                     resolved_pending_reviews=['SYNTHETIC_pending_review'], original_nested_pending_reviews={})
    monkeypatch.setattr(p.F, 'replay', lambda **kwargs: copy.deepcopy(decisions))
    # This fixture replaces source semantics; dedicated binding tests exercise
    # actual manifest role selection, receipt hashes, relocation and tampering.
    monkeypatch.setattr(p.F, 'hardware_source_binding', lambda *args, **kwargs: {'synthetic_only': True})
    refs = tmp_path / 'references'; refs.mkdir()
    def external(name):
        path = refs / name; path.write_text('Synthetic test reference: ' + name)
        return dict(path=str(path), **p.C.record_bytes(path.read_bytes()))
    value = dict(schema=1, kind='c1-alex-qualified-collection-inputs', candidate_directory=str(root),
        candidate_manifest_sha256=digest, review_source=dict(path=str(ROOT), revision='1' * 40),
        reference_files={key: external(key) for key in p.R.REFERENCE_AUXILIARY},
        disposition_file=external('disposition.json'))
    return dict(spec=value, root=root, manifest=manifest, sha=digest, out=tmp_path / 'qualified',
                decisions=decisions, sources=sources, blobs=blobs, tmp=tmp_path)


def test_additive_collection_preserves_original_and_relocates(context):
    c = context; before = raw.f.inventory(c['root'])
    result = p.collect(c['spec'], c['out'])
    assert result['schema'] == 3 and result['state'] == 'complete' and not result['hardware_actions']
    assert result['instance_id']==result['verified']['instance_id']==c['manifest']['instance_id']
    assert json.loads((c['out']/'bundle'/p.E.MANIFEST).read_text())['instance_id']==c['manifest']['instance_id']
    assert result['original_candidate_manifest_sha256'] == c['sha']
    assert raw.f.inventory(c['root']) == before
    assert p.E.verify_bundle(c['root'], c['sha'])['route_qualified'] is False
    moved = c['tmp'] / 'relocated'; shutil.move(c['out'] / 'bundle', moved)
    verified = p.Q.verify_bundle(moved, result['manifest_sha256'])
    assert verified == result['verified'] and verified['review_required'] is False
    assert verified['original_pending_reviews'] == ['SYNTHETIC_pending_review']
    assert verified['qualification_receipt_sha256'] == result['review_receipt_sha256']
    assert p.Q.inspect_archive(c['out'] / 'evidence.tar', result['manifest_sha256'])['archive_sha256'] == result['archive']['sha256']


@pytest.mark.parametrize('case', ['missing_reference', 'extra_reference', 'changed_reference', 'linked_reference',
    'wrong_kind', 'boolean_schema', 'wrong_source', 'relative_output', 'existing_output'])
def test_invalid_inputs_create_no_qualified_receipt(context, case):
    c = context; s = c['spec']; out = c['out']
    key = next(iter(s['reference_files']))
    if case == 'missing_reference': del s['reference_files'][key]
    elif case == 'extra_reference': s['reference_files']['extra'] = dict(s['reference_files'][key])
    elif case == 'changed_reference': Path(s['reference_files'][key]['path']).write_text('changed')
    elif case == 'linked_reference':
        link = c['tmp'] / 'linked'; link.symlink_to(s['reference_files'][key]['path']); s['reference_files'][key]['path'] = str(link)
    elif case == 'wrong_kind': s['kind'] = 'c2-alex-qualified-collection-inputs'
    elif case == 'boolean_schema': s['schema'] = True
    elif case == 'wrong_source': s['review_source']['path'] = str(c['tmp'])
    elif case == 'relative_output': out = Path('relative-output-must-not-be-created')
    elif case == 'existing_output': out.mkdir()
    with pytest.raises(ValueError): p.collect(s, out)
    assert not (c['out'] / 'collection.json').exists()
    assert not (c['out'] / 'evidence.tar').exists()


def test_semantic_rejection_never_creates_archive_or_success(context, monkeypatch):
    c = context
    def reject(**kwargs): raise ValueError('Actual C1 obligation remains unresolved')
    monkeypatch.setattr(p.F, 'replay', reject)
    with pytest.raises(ValueError, match='unresolved'): p.collect(c['spec'], c['out'])
    assert json.loads((c['out'] / 'failure.json').read_text())['state'] == 'failed'
    assert not (c['out'] / 'evidence.tar').exists() and not (c['out'] / 'collection.json').exists()
    assert p.E.sha(c['root'] / p.E.MANIFEST) == c['sha']


@pytest.mark.parametrize('change', ['original_file', 'reference', 'review_source', 'missing_resolution', 'false_qualification'])
def test_changes_during_replay_do_not_yield_qualified_collection(context, monkeypatch, change):
    c = context; calls = []
    def mutate(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            if change == 'original_file': (c['root'] / 'runs/refine/post_route.dcp').write_bytes(b'changed original')
            elif change == 'reference': Path(next(iter(c['spec']['reference_files'].values()))['path']).write_bytes(b'changed external')
            elif change == 'review_source': c['blobs'][next(iter(c['blobs']))] = b'changed approved blob'
            elif change == 'missing_resolution': c['decisions']['resolved_pending_reviews'] = []
            else: c['decisions']['route_qualified'] = False
        return copy.deepcopy(c['decisions'])
    monkeypatch.setattr(p.F, 'replay', mutate)
    with pytest.raises(ValueError): p.collect(c['spec'], c['out'])
    assert (c['out'] / 'failure.json').exists() and not (c['out'] / 'collection.json').exists()
    assert not (c['out'] / 'evidence.tar').exists()


def test_approved_source_snapshot_mismatch_rejected_before_replay(context, monkeypatch):
    c = context; c['blobs'][next(iter(c['blobs']))] = b'different code'
    monkeypatch.setattr(p.F, 'replay', lambda **kwargs: pytest.fail('unattested source reached review'))
    with pytest.raises(ValueError, match='approved Git revision'): p.collect(c['spec'], c['out'])
    assert not (c['out'] / 'bundle').exists()


def test_schema3_archive_payload_mutation_rejected(context):
    c = context; result = p.collect(c['spec'], c['out'])
    archive = c['out'] / 'evidence.tar'
    # A tar remains readable after this payload mutation; inventory hashes must fail.
    data = bytearray(archive.read_bytes()); needle = b'Synthetic test reference:'
    index = data.index(needle); data[index] ^= 1; archive.write_bytes(data)
    with pytest.raises(ValueError): p.Q.inspect_archive(archive, result['manifest_sha256'])


def test_builder_change_cannot_reuse_schema3_manifest_hash(context):
    c=context;result=p.collect(c['spec'],c['out']);path=c['out']/'bundle'/p.E.MANIFEST
    manifest=json.loads(path.read_text());manifest['instance_id']=next(value for value in p.E.BUILDER_INSTANCES if value!=manifest['instance_id'])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):p.Q.verify_bundle(path.parent,result['manifest_sha256'])
