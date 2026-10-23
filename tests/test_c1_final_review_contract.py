"""Synthetic binding tests only; no C1 artifact is accepted by this skeleton."""
import copy
import hashlib
import unittest

from boards.c1_f2_alex import review_contract as R


def meta(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fixture():
    revision = "0d654453e2c77ea9bc9d0e854c33d9166bb27375"
    manifest = {
        "hardware": {"source_revision": revision, "build_id": revision[:8]},
        "roots": {role: {"bundle_relative_path": "runs/" + role}
                  for role in (*R.E.schema_roles(2), "review")},
        "source_snapshots": {"review": {"revision": "1" * 40, "bundle_relative_path": "sources/review"}},
        "receipts": {}, "files": {},
    }
    def add(role, name, raw):
        manifest["files"]["runs/" + role + "/" + name] = meta(raw)
        return {"role": role, "relative_path": name, **meta(raw)}
    for role, name in R.E.schema_receipts(2).items():
        row = add(role, name, role.encode())
        manifest["receipts"][role] = {"relative_path": "runs/" + role + "/" + name, "sha256": row["sha256"]}
    linked = add("link", "post_link.dcp", b"linked")
    final = add("refine", "post_route.dcp", b"routed")
    reports = {name: add("capture", "final/" + name, name.encode()) for name in R.E.FINAL_REPORTS}
    auxiliary = {key: add(role, name, key.encode()) for key, (role, name) in R.RAW_AUXILIARY.items()}
    auxiliary.update({key: add("review", "auxiliary/" + key, key.encode()) for key in R.REFERENCE_AUXILIARY})
    disposition = add("review", "disposition.json", b"unreviewed test disposition")
    inputs = {
        "schema": 1, "kind": "c1-alex-final-review-inputs",
        "hardware": manifest["hardware"].copy(),
        "checkpoints": {"linked": linked["sha256"], "final": final["sha256"]},
        "receipt_sha256": {role: row["sha256"] for role, row in manifest["receipts"].items()},
        "reports": reports, "auxiliary": auxiliary, "disposition_sha256": disposition["sha256"],
    }
    inputs_raw = R.canonical(inputs)
    input_record = add("review", R.INPUTS_NAME, inputs_raw)
    decisions = add("review", R.DECISIONS_NAME, b'{"claimed_pass":true}')
    sources = {name: meta(name.encode()) for name in R.REVIEW_SOURCES}
    manifest["files"].update({"sources/review/" + name: row for name, row in sources.items()})
    receipt = {
        "schema": 1, "kind": "c1-alex-final-semantic-review", "status": "pending-review",
        "route_qualified": False, "hardware_measured": False, "hardware_actions": False,
        "checkpoint_sha256": final["sha256"], "linked_checkpoint_sha256": linked["sha256"],
        "inputs_sha256": input_record["sha256"], "decisions_sha256": decisions["sha256"],
        "source_revision": "1" * 40, "source_inputs": sources,
        "outputs": {name[len("runs/review/"):]: row for name, row in manifest["files"].items()
                    if name.startswith("runs/review/")},
    }
    receipt_raw = R.canonical(receipt)
    row = add("review", R.RECEIPT_NAME, receipt_raw)
    manifest["receipts"]["review"] = {"relative_path": "runs/review/" + R.RECEIPT_NAME, "sha256": row["sha256"]}
    return inputs, receipt, manifest, sources


def rebind_receipt(receipt, manifest):
    raw = R.canonical(receipt)
    manifest["files"]["runs/review/" + R.RECEIPT_NAME] = meta(raw)
    manifest["receipts"]["review"]["sha256"] = meta(raw)["sha256"]
    return raw


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.inputs, self.receipt, self.manifest, self.sources = fixture()

    def check(self):
        return R.validate_receipt(R.canonical(self.receipt), R.canonical(self.inputs), self.manifest, self.sources)

    def test_valid_envelope_never_qualifies(self):
        self.assertFalse(self.check()["route_qualified"])
        self.assertTrue(self.check()["semantic_replay_required"])

    def test_complete_claim_is_still_not_semantic_proof(self):
        self.receipt.update(status="complete", route_qualified=True)
        rebind_receipt(self.receipt, self.manifest)
        self.assertFalse(self.check()["route_qualified"])

    def test_missing_every_receipt_role(self):
        for role in R.E.schema_receipts(2):
            value = copy.deepcopy(self.inputs)
            del value["receipt_sha256"][role]
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "receipt proof role"):
                R.validate_inputs(value, self.manifest)

    def test_missing_every_auxiliary(self):
        for key in self.inputs["auxiliary"]:
            value = copy.deepcopy(self.inputs)
            del value["auxiliary"][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "auxiliary"):
                R.validate_inputs(value, self.manifest)

    def test_unknown_receipt_role(self):
        self.inputs["receipt_sha256"]["old_c2"] = "1" * 64
        with self.assertRaises(ValueError):
            R.validate_inputs(self.inputs, self.manifest)

    def test_wrong_checkpoint(self):
        self.inputs["checkpoints"]["final"] = self.inputs["checkpoints"]["linked"]
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            R.validate_inputs(self.inputs, self.manifest)

    def test_wrong_raw_role(self):
        self.inputs["auxiliary"]["native"]["role"] = "linked_native"
        with self.assertRaisesRegex(ValueError, "role/path"):
            R.validate_inputs(self.inputs, self.manifest)

    def test_changed_proof_hash(self):
        self.inputs["auxiliary"]["ready"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            R.validate_inputs(self.inputs, self.manifest)

    def test_wrong_receipt_ancestry(self):
        self.inputs["receipt_sha256"]["linked_ready"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Receipt proof"):
            R.validate_inputs(self.inputs, self.manifest)

    def test_relative_escape(self):
        self.inputs["auxiliary"]["native"]["relative_path"] = "../other/topology"
        with self.assertRaises(ValueError):
            R.validate_inputs(self.inputs, self.manifest)

    def test_unbound_inputs_fail_before_parsing(self):
        self.inputs["hardware"]["build_id"] = "deadbeef"
        with self.assertRaisesRegex(ValueError, "control hash"):
            self.check()

    def test_claimed_hardware_action_rejected(self):
        self.receipt["hardware_actions"] = True
        rebind_receipt(self.receipt, self.manifest)
        with self.assertRaisesRegex(ValueError, "hardware actions"):
            self.check()

    def test_runtime_reviewer_source_is_separate(self):
        self.assertNotEqual(self.inputs["hardware"]["source_revision"], self.receipt["source_revision"])
        self.assertTrue(self.check()["binding_consistent"])

    def test_source_dependency_missing(self):
        self.sources = copy.deepcopy(self.sources)
        del self.sources["boards/c1_f2_alex/native_review.py"]
        with self.assertRaisesRegex(ValueError, "dependency inventory"):
            self.check()

    def test_changed_trusted_source_fails(self):
        self.sources = copy.deepcopy(self.sources)
        self.sources["boards/c1_f2_alex/native_review.py"] = meta(b"new source")
        with self.assertRaisesRegex(ValueError, "trusted local source"):
            self.check()

    def test_receipt_output_omission(self):
        del self.receipt["outputs"]["auxiliary/shell_objects"]
        rebind_receipt(self.receipt, self.manifest)
        with self.assertRaisesRegex(ValueError, "output inventory"):
            self.check()

    def test_duplicate_json_keys(self):
        raw = b'{"schema":1,"schema":1}'
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            R.decode(raw, meta(raw)["sha256"])

    def test_nonfinite_json(self):
        raw = b'{"value":NaN}'
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            R.decode(raw, meta(raw)["sha256"])

    def test_each_category_required(self):
        for category in R.REQUIRED_CATEGORIES:
            with self.subTest(category=category), self.assertRaises(ValueError):
                R.pending_categories(sorted(R.REQUIRED_CATEGORIES - {category}), {})

    def test_duplicate_or_unknown_categories(self):
        for categories in (sorted(R.REQUIRED_CATEGORIES) + ["unknown"],
                           sorted(R.REQUIRED_CATEGORIES) + [next(iter(R.REQUIRED_CATEGORIES))]):
            with self.assertRaises(ValueError):
                R.pending_categories(categories, {})

    def test_unknown_nested_obligation_is_retained_not_discharged(self):
        result = R.pending_categories(sorted(R.REQUIRED_CATEGORIES), {"ready": ["new_unknown_gap"]})
        self.assertEqual(result["nested_pending_reviews"]["ready"], ["new_unknown_gap"])
        self.assertFalse(result["route_qualified"])



if __name__ == "__main__":
    unittest.main()
