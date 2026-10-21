"""C1 qualified portable evidence, with full schema3 raw semantic replay.

Schemas1/2 remain unqualified candidates. Schema3 authenticates every file before
projecting its unchanged candidate roles into the original provenance verifier.
Recorded review success never substitutes for current trusted raw replay.
"""
from __future__ import annotations
import copy
import hashlib
import re
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boards.c1_f2_alex import review_contract as R

E = R.E
MANIFEST = E.MANIFEST
INSTANCE = E.INSTANCE
MAX_FILES, MAX_TOTAL, MAX_FILE, MAX_MANIFEST = E.MAX_FILES, E.MAX_TOTAL, E.MAX_FILE, E.MAX_MANIFEST
# Pure shared transport helpers; none grants semantic qualification.
digest, decode, sha = E.digest, E.decode, E.sha
relative, absolute, metadata, no_symlink = E.relative, E.absolute, E.metadata, E.no_symlink
require = E.require


def overlaps(first, second):
    return first == second or first.startswith(second + "/") or second.startswith(first + "/")


def _project_manifest(raw, trusted_sha256):
    manifest = R.decode(raw, trusted_sha256)
    R.need(isinstance(manifest, dict) and set(manifest) == {
        "schema", "kind", "instance_id", "hardware", "roots",
        "source_snapshots", "receipts", "files"}, "Unknown schema3 manifest fields")
    R.need(type(manifest["schema"]) is int and manifest["schema"] == 3 and
           manifest["kind"] == "c1-alex-portable-candidate-evidence" and
           manifest["instance_id"] == E.INSTANCE, "Wrong schema3 identity")
    for field, expected in (
        ("roots", set(E.schema_roles(2)) | {"review"}),
        ("source_snapshots", set(E.schema_sources(2)) | {"review"}),
        ("receipts", set(E.schema_receipts(2)) | {"review"}),
    ):
        R.need(isinstance(manifest[field], dict) and set(manifest[field]) == expected,
               "Missing/extra schema3 " + field)
    root = manifest["roots"]["review"]
    source = manifest["source_snapshots"]["review"]
    R.need(isinstance(root, dict) and set(root) == {"original_absolute_path", "bundle_relative_path"},
           "Malformed review root")
    R.need(isinstance(source, dict) and set(source) == {"revision", "bundle_relative_path"} and
           isinstance(source["revision"], str) and re.fullmatch("[0-9a-f]{40}", source["revision"]),
           "Malformed review source snapshot")
    original = E.absolute(root["original_absolute_path"])
    run_prefix = E.relative(root["bundle_relative_path"])
    source_prefix = E.relative(source["bundle_relative_path"])
    R.need(not overlaps(run_prefix, source_prefix), "Overlapping review source/output roots")
    for role, row in manifest["roots"].items():
        if role == "review":
            continue
        R.need(not overlaps(original, row["original_absolute_path"]), "Review original root overlaps candidate")
        for prefix in (run_prefix, source_prefix):
            R.need(not overlaps(prefix, row["bundle_relative_path"]), "Review files overlap candidate run")
    for role, row in manifest["source_snapshots"].items():
        if role == "review":
            continue
        for prefix in (run_prefix, source_prefix):
            R.need(not overlaps(prefix, row["bundle_relative_path"]), "Review files overlap candidate source")
    files = manifest["files"]
    R.need(isinstance(files, dict) and 0 < len(files) <= E.MAX_FILES, "Manifest inventory bound")
    total = 0
    for name, row in files.items():
        E.relative(name); E.metadata(row); total += row["bytes"]
    R.need(total <= E.MAX_TOTAL and E.MANIFEST not in files, "Manifest file/size bound")
    review_run = {name: row for name, row in files.items() if name.startswith(run_prefix + "/")}
    review_source = {name: row for name, row in files.items() if name.startswith(source_prefix + "/")}
    R.need(review_run and review_source, "Empty review role")
    receipt_path = run_prefix + "/" + R.RECEIPT_NAME
    receipt = manifest["receipts"]["review"]
    R.need(isinstance(receipt, dict) and set(receipt) == {"relative_path", "sha256"} and
           receipt["relative_path"] == receipt_path and receipt_path in review_run and
           E.digest(receipt["sha256"]) == review_run[receipt_path]["sha256"],
           "Review receipt binding differs")
    for name in (R.INPUTS_NAME, R.DECISIONS_NAME, "disposition.json"):
        R.need(run_prefix + "/" + name in review_run, "Review control absent")
    projected = copy.deepcopy(manifest)
    projected["schema"] = 2
    for field in ("roots", "source_snapshots", "receipts"):
        del projected[field]["review"]
    projected["files"] = {name: row for name, row in files.items()
                          if name not in review_run and name not in review_source}
    encoded = R.canonical(projected)
    # This synthetic projection hash authenticates no new external artifact; it
    # only adapts the already externally bound original bytes to the old parser.
    checked = E.manifest_data(encoded, hashlib.sha256(encoded).hexdigest())
    R.need(checked == projected, "Schema2 projection changed")
    return manifest, projected, {
        "schema": 1, "kind": "c1-schema3-review-role-binding-preparation",
        "manifest_sha256": trusted_sha256, "base_manifest_shape_validated": True,
        "base_provenance_replay_required": True, "review_semantic_replay_required": True,
        "review_run_files": len(review_run), "review_source_files": len(review_source),
        "route_qualified": False, "hardware_measured": False,
    }


def manifest_data(raw, trusted_sha256):
    """Authenticate manifest shape only, including the complete schema3 role."""
    value = R.decode(raw, trusted_sha256)
    if isinstance(value, dict) and type(value.get('schema')) is int and value['schema'] in (1, 2):
        return E.manifest_data(raw, trusted_sha256)
    return _project_manifest(raw, trusted_sha256)[0]


class Bundle(E.Bundle):
    parse_manifest = staticmethod(manifest_data)

    def receipt(self, role):
        if role == 'review':
            return self.read(role, R.RECEIPT_NAME, self.manifest['receipts'][role]['sha256'])
        return super().receipt(role)


def _base_view(bundle, projected):
    """Exact subset of an already authenticated complete schema3 bundle."""
    view = object.__new__(E.Bundle)
    view.root, view.manifest, view.files = bundle.root, projected, projected['files']
    return view


def verify_bundle(root, trusted_sha256):
    bundle = Bundle(root, trusted_sha256)
    if bundle.manifest['schema'] in (1, 2):
        return E._verify_authenticated(bundle, trusted_sha256)
    _, projected, _ = _project_manifest((bundle.root / MANIFEST).read_bytes(), trusted_sha256)
    candidate = E._verify_authenticated(_base_view(bundle, projected), trusted_sha256)
    # The base body retains capture-time dependencies, pending reports and
    # original NOT_QUALIFIED markers. Only current trusted review can close them.
    review = E.local_module('final_review').verify_review(bundle, candidate)
    require(review.get('route_qualified') is True and review.get('hardware_measured') is False,
            'Final raw review did not qualify this artifact')
    for name, row in bundle.files.items():
        path = bundle.root / name
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Bundle changed during final review: ' + name)
    require(sha(bundle.root / MANIFEST) == trusted_sha256, 'Manifest changed during final review')
    return {**candidate, 'schema': 1, 'kind': 'verified-c1-alex-portable-evidence',
            'route_qualified': True, 'review_required': False, 'hardware_measured': False,
            'original_pending_reviews': candidate['pending_reviews'], 'pending_reviews': [],
            'qualification_receipt_sha256': bundle.manifest['receipts']['review']['sha256'],
            'final_review': review, 'files': len(bundle.files),
            'bytes': sum(row['bytes'] for row in bundle.files.values())}


def inspect_archive(archive_path, trusted_manifest_sha256):
    """Exact complete schema3 inventory, with the same bounded streaming checks."""
    return E._inspect_archive(archive_path, trusted_manifest_sha256, manifest_data)

