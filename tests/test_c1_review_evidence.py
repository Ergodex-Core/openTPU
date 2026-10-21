import copy
import io
import tarfile
import unittest
from types import SimpleNamespace

import pytest

from boards.c1_f2_alex import evidence as P
from boards.c1_f2_alex import review_contract as R
from test_c1_final_review_contract import fixture, meta, rebind_receipt


def manifest_fixture():
    inputs, receipt, m, sources = fixture()
    m.update(schema=3, kind="c1-alex-portable-candidate-evidence", instance_id=R.E.INSTANCE)
    for role, row in m["roots"].items():
        row["original_absolute_path"] = "/original/" + role
    m["roots"]["simulation_private"] = {
        key: value + "/private-ip" for key, value in m["roots"]["simulation"].items()}
    for role in R.E.schema_sources(2):
        m["source_snapshots"][role] = {"revision": "2" * 40, "bundle_relative_path": "sources/" + role}
        m["files"]["sources/" + role + "/source.py"] = meta(b"synthetic parser-only source")
    return inputs, receipt, m, sources


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.inputs, self.receipt, self.m, self.sources = manifest_fixture()

    def project(self):
        raw = R.canonical(self.m)
        return P._project_manifest(raw, meta(raw)["sha256"])

    def test_additive_projection_is_exact_and_unqualified(self):
        before = copy.deepcopy(self.m)
        _, projected, result = self.project()
        self.assertEqual(self.m, before)
        self.assertEqual(projected["schema"], 2)
        for field in ("roots", "source_snapshots", "receipts"):
            self.assertEqual(projected[field], {k: v for k, v in before[field].items() if k != "review"})
        self.assertEqual(projected["files"], {k: v for k, v in before["files"].items()
                         if not k.startswith(("runs/review/", "sources/review/"))})
        self.assertFalse(result["route_qualified"])
        self.assertTrue(result["base_provenance_replay_required"])
        self.assertTrue(result["review_semantic_replay_required"])

    def test_external_hash_required(self):
        with self.assertRaises(ValueError):
            P._project_manifest(R.canonical(self.m), "0" * 64)

    def test_every_base_role_preserved(self):
        for role in R.E.schema_roles(2):
            m = copy.deepcopy(self.m); del m["roots"][role]; raw = R.canonical(m)
            with self.subTest(role=role), self.assertRaises(ValueError):
                P._project_manifest(raw, meta(raw)["sha256"])

    def test_review_role_required(self):
        for field in ("roots", "source_snapshots", "receipts"):
            m = copy.deepcopy(self.m); del m[field]["review"]; raw = R.canonical(m)
            with self.subTest(field=field), self.assertRaises(ValueError):
                P._project_manifest(raw, meta(raw)["sha256"])

    def test_unknown_role_rejected(self):
        self.m["roots"]["unreviewed"] = self.m["roots"]["review"].copy()
        with self.assertRaises(ValueError): self.project()

    def test_schema_bool_or_legacy_rejected(self):
        for value in (True, 2, 4):
            self.m["schema"] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.project()

    def test_cross_role_overlaps_rejected(self):
        for prefix in ("runs/synth", "runs/synth/extra", "sources/hardware", "runs", "sources/review"):
            self.m["roots"]["review"]["bundle_relative_path"] = prefix
            with self.subTest(prefix=prefix), self.assertRaises(ValueError): self.project()

    def test_original_path_overlap_rejected(self):
        self.m["roots"]["review"]["original_absolute_path"] = "/original/refine/child"
        with self.assertRaises(ValueError): self.project()

    def test_path_traversal_rejected(self):
        self.m["roots"]["review"]["bundle_relative_path"] = "../review"
        with self.assertRaises(ValueError): self.project()

    def test_extra_file_cannot_be_laundered_through_projection(self):
        self.m["files"]["unmapped/extra"] = meta(b"x")
        with self.assertRaises(ValueError): self.project()

    def test_removed_review_control_rejected(self):
        for name in (R.RECEIPT_NAME, R.INPUTS_NAME, R.DECISIONS_NAME, "disposition.json"):
            m = copy.deepcopy(self.m); del m["files"]["runs/review/" + name]; raw = R.canonical(m)
            with self.subTest(name=name), self.assertRaises(ValueError):
                P._project_manifest(raw, meta(raw)["sha256"])

    def test_wrong_review_receipt_hash_rejected(self):
        self.m["receipts"]["review"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): self.project()

    def test_changed_base_receipt_detected_by_existing_parser(self):
        self.m["receipts"]["refine"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): self.project()

    def test_empty_review_source_rejected(self):
        self.m["files"] = {k: v for k, v in self.m["files"].items() if not k.startswith("sources/review/")}
        with self.assertRaises(ValueError): self.project()


def inventory_bundle(tmp_path):
    """Synthetic parser-only bundle, intentionally lacks valid base provenance."""
    _, _, manifest, _ = manifest_fixture()
    root = tmp_path / 'bundle'; root.mkdir()
    for name in list(manifest['files']):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('SYNTHETIC inventory only ' + name).encode())
        manifest['files'][name] = meta(path.read_bytes())
    for row in manifest['receipts'].values():
        row['sha256'] = manifest['files'][row['relative_path']]['sha256']
    raw = R.canonical(manifest); (root / P.MANIFEST).write_bytes(raw)
    return root, manifest, meta(raw)['sha256']


def test_schema3_bundle_authenticates_review_files_before_projection(tmp_path):
    root, _, digest = inventory_bundle(tmp_path)
    assert P.Bundle(root, digest).manifest['schema'] == 3
    (root / 'runs/review/review-decisions.json').write_text('changed success claim')
    with pytest.raises(ValueError, match='Changed bundle file'):
        P.verify_bundle(root, digest)


def test_schema3_full_base_replay_is_mandatory(tmp_path, monkeypatch):
    root, _, digest = inventory_bundle(tmp_path)
    calls = []
    original = P.E._verify_authenticated
    def replay(base, wanted):
        calls.append((base.manifest['schema'], wanted, set(base.manifest['roots'])))
        return original(base, wanted)
    monkeypatch.setattr(P.E, '_verify_authenticated', replay)
    with pytest.raises((ValueError, KeyError)):
        P.verify_bundle(root, digest)
    assert calls == [(2, digest, set(P.E.schema_roles(2)))]


def test_schema3_review_claim_cannot_bypass_current_raw_review(tmp_path, monkeypatch):
    root, _, digest = inventory_bundle(tmp_path)
    monkeypatch.setattr(P.E, '_verify_authenticated', lambda *_: {'pending_reviews': ['unclosed']})
    def fail(*_):
        raise ValueError('current raw predicate unresolved')
    monkeypatch.setattr(P.E, 'local_module', lambda name: SimpleNamespace(verify_review=fail))
    with pytest.raises(ValueError, match='current raw predicate unresolved'):
        P.verify_bundle(root, digest)


@pytest.mark.parametrize('mode', ['valid', 'extra', 'missing', 'changed', 'linked'])
def test_schema3_archive_exact_inventory(tmp_path, mode):
    root, manifest, digest = inventory_bundle(tmp_path)
    archive = tmp_path / 'bundle.tar'
    with tarfile.open(archive, 'w') as out:
        for path in sorted(p for p in root.rglob('*') if p.is_file()):
            name = path.relative_to(root).as_posix()
            if mode == 'missing' and name == 'runs/review/review-decisions.json': continue
            raw = path.read_bytes()
            if mode == 'changed' and name == 'runs/review/review-decisions.json': raw += b'changed'
            row = tarfile.TarInfo(name); row.size = len(raw)
            out.addfile(row, io.BytesIO(raw))
        if mode in ('extra', 'linked'):
            row = tarfile.TarInfo('extra')
            if mode == 'linked': row.type, row.linkname = tarfile.SYMTYPE, '/outside'
            out.addfile(row, io.BytesIO())
    if mode == 'valid':
        result = P.inspect_archive(archive, digest)
        assert result['archive_sha256'] == P.sha(archive)
        assert result['files'] == len(manifest['files']) and result['route_qualified'] is False
    else:
        with pytest.raises(ValueError): P.inspect_archive(archive, digest)


if __name__ == "__main__":
    unittest.main()
