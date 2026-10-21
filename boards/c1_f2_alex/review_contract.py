"""C1 review-envelope binding; no semantic or image acceptance.

These functions validate additive bindings to a separately authenticated portable
candidate. They never execute bundled code or accept recorded predicate results
as proof. A future final reviewer must run trusted C1 predicates on the raw files.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boards.c1_f2_alex import candidate_evidence as E

RECEIPT_NAME = "review-result.json"
INPUTS_NAME = "review-inputs.json"
DECISIONS_NAME = "review-decisions.json"
REQUIRED_CATEGORIES = frozenset({
    "current_structural_and_ready_equivalence",
    "current_vendor_CDC_and_gray_max_delay_scope",
    "current_exception_coverage_and_ignored_paths",
    "current_DRC_methodology_and_severity_configuration",
    "exact_shell_inherited_witnesses",
    "current_9SI_8MI_vendor_simulation_and_generated_IP_binding",
    "current_reset_status",
})
RAW_AUXILIARY = {
    "linked_native": ("linked_native", "raw/topology.tcldata"),
    "native": ("capture", "native/topology.tcldata"),
    "linked_ready": ("linked_ready", "raw/ready.tcldata"),
    "ready": ("capture", "ready/ready.tcldata"),
    "reset_status": ("capture", "reset/topology.tcldata"),
    "reset_status_core_pulse": ("capture", "reset/u_rst_core_pulse.rpt"),
    "reset_status_hbm_pulse": ("capture", "reset/u_rst_hbm_pulse.rpt"),
    "reset_status_status_pulse": ("capture", "reset/u_status_cdc_pulse.rpt"),
    "handshake": ("capture", "handshake.tcldata"),
    "unused_outputs": ("capture", "final/unused_outputs.tcldata"),
}
REFERENCE_AUXILIARY = frozenset({
    "shell_objects", "shell_cdc", "shell_complete", "shell_timing",
    "shell_waivers", "gray_vendor", "handshake_vendor", "memory_vendor",
    "accounting", "accounting_linked", "accounting_receipt", "accounting_linked_receipt",
    "structural", "structural_receipt",
})
REVIEW_SOURCES = frozenset({
    "boards/c1_f2_alex/final_review.py",
    "boards/c1_f2_alex/evidence.py",
    "boards/c1_f2_alex/review_contract.py",
    "boards/c1_f2_alex/reviewed_artifacts.json",
    "boards/c1_f2_alex/ready_pair_review.py",
    "boards/c1_f2_alex/accounting_review.py",
    "boards/c1_f2_alex/structural_review.py",
    "boards/c1_f2_alex/structural_capture_review.py",
    "boards/c1_f2_alex/capture_accounting.tcl",
    "boards/c1_f2_alex/capture_native.tcl",
    "boards/c1_f2_alex/capture_structure.tcl",
    "boards/c1_f2_alex/routed_host_ready.tcl",
    "boards/c1_f2_alex/audit_c1_alex.tcl",
    "boards/c1_f2_alex/qualify_c1_alex.tcl",
    "boards/c1_f2_alex/candidate_evidence.py",
    "boards/c1_f2_alex/collect_candidate.py",
    "boards/c1_f2_alex/collect_qualified.py",
    "boards/c1_f2_alex/build.py",
    "boards/c1_f2_alex/implement.py",
    "boards/c1_f2_alex/simulate.py",
    "boards/c1_f2_alex/gray_review.py",
    "boards/c1_f2_alex/no_clock_review.py",
    "boards/c1_f2_alex/reset_status_review.py",
    "boards/c1_f2_alex/exception_review.py",
    "boards/c1_f2_alex/native_review.py",
    "boards/c1_f2_alex/ready_review.py",
})


def need(value, message):
    if not value:
        raise ValueError(message)


def revision(value):
    need(isinstance(value, str) and re.fullmatch("[0-9a-f]{40}", value), "Invalid revision")
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode(raw, wanted):
    need(isinstance(raw, bytes) and len(raw) <= E.MAX_MANIFEST, "Oversized review control")
    need(hashlib.sha256(raw).hexdigest() == E.digest(wanted), "Review control hash mismatch")
    return E.decode(raw)


def inventory_record(manifest, role, name):
    """Only look up a safe relative file within its already validated role."""
    need(role in manifest["roots"], "Unknown proof role")
    name = E.relative(name)
    prefix = manifest["roots"][role]["bundle_relative_path"]
    row = manifest["files"].get(prefix + "/" + name)
    need(row is not None, "Missing proof file: " + role + "/" + name)
    E.metadata(row)
    return row


def reference(manifest, row, role, name):
    need(isinstance(row, dict) and set(row) == {
        "role", "relative_path", "bytes", "sha256"}, "Malformed proof reference")
    need(row["role"] == role and row["relative_path"] == name, "Proof role/path mismatch")
    E.metadata({key: row[key] for key in ("bytes", "sha256")})
    actual = inventory_record(manifest, role, name)
    need(actual == {key: row[key] for key in ("bytes", "sha256")}, "Proof bytes differ")


def validate_inputs(value, manifest):
    """Bind every required raw proof; does not interpret its semantic content."""
    need(isinstance(value, dict) and set(value) == {
        "schema", "kind", "hardware", "checkpoints", "receipt_sha256",
        "reports", "auxiliary", "disposition_sha256"}, "Malformed C1 review inputs")
    need(type(value["schema"]) is int and value["schema"] == 1 and
         value["kind"] == "c1-alex-final-review-inputs", "Wrong C1 review input schema")
    need(value["hardware"] == manifest["hardware"], "Mixed hardware identity")
    hardware = value["hardware"]
    need(set(hardware) == {"source_revision", "build_id"} and
         hardware["build_id"] == revision(hardware["source_revision"])[:8], "Hardware revision/build mismatch")
    checkpoints = value["checkpoints"]
    need(isinstance(checkpoints, dict) and set(checkpoints) == {"linked", "final"},
         "Missing linked/final checkpoint")
    need(checkpoints["linked"] == inventory_record(manifest, "link", "post_link.dcp")["sha256"] and
         checkpoints["final"] == inventory_record(manifest, "refine", "post_route.dcp")["sha256"] and
         checkpoints["linked"] != checkpoints["final"], "Checkpoint ancestry mismatch")
    roles = set(E.schema_receipts(2))
    need(isinstance(value["receipt_sha256"], dict) and set(value["receipt_sha256"]) == roles,
         "Missing/extra receipt proof role")
    for role, digest in value["receipt_sha256"].items():
        need(E.digest(digest) == manifest["receipts"][role]["sha256"], "Receipt proof mismatch: " + role)
    need(isinstance(value["reports"], dict) and set(value["reports"]) == set(E.FINAL_REPORTS),
         "Incomplete/extra final report inventory")
    for name, row in value["reports"].items():
        reference(manifest, row, "capture", "final/" + name)
    need(isinstance(value["auxiliary"], dict) and
         set(value["auxiliary"]) == set(RAW_AUXILIARY) | REFERENCE_AUXILIARY,
         "Missing/extra raw auxiliary proof")
    for key, (role, name) in RAW_AUXILIARY.items():
        reference(manifest, value["auxiliary"][key], role, name)
    for key in REFERENCE_AUXILIARY:
        reference(manifest, value["auxiliary"][key], "review", "auxiliary/" + key)
    need(E.digest(value["disposition_sha256"]) ==
         inventory_record(manifest, "review", "disposition.json")["sha256"],
         "Disposition bytes mismatch")
    return {"binding_consistent": True, "route_qualified": False,
            "semantic_replay_required": True, "hardware_measured": False}


def validate_receipt(receipt_raw, inputs_raw, manifest, trusted_source_inventory):
    """Check receipt bytes/ancestry, even if it *claims* complete qualification.

    The caller must supply current trusted local source hashes, never copy these
    from the receipt. This validator cannot grant acceptance in either state.
    """
    receipt_record = inventory_record(manifest, "review", RECEIPT_NAME)
    need(manifest["receipts"]["review"] == {
        "relative_path": manifest["roots"]["review"]["bundle_relative_path"] + "/" + RECEIPT_NAME,
        "sha256": receipt_record["sha256"]}, "Review receipt relocation mismatch")
    receipt = decode(receipt_raw, receipt_record["sha256"])
    need(len(receipt_raw) == receipt_record["bytes"], "Review receipt length mismatch")
    need(isinstance(receipt, dict) and set(receipt) == {
        "schema", "kind", "status", "route_qualified", "hardware_measured", "hardware_actions",
        "checkpoint_sha256", "linked_checkpoint_sha256", "inputs_sha256", "decisions_sha256",
        "source_revision", "source_inputs", "outputs"}, "Malformed C1 review receipt")
    need(type(receipt["schema"]) is int and receipt["schema"] == 1 and
         receipt["kind"] == "c1-alex-final-semantic-review", "Wrong review receipt kind")
    need(receipt["hardware_measured"] is False and receipt["hardware_actions"] is False,
         "Semantic review cannot claim hardware actions/results")
    need((receipt["status"] == "pending-review" and receipt["route_qualified"] is False) or
         (receipt["status"] == "complete" and receipt["route_qualified"] is True),
         "Invalid claimed review state")
    inputs_record = inventory_record(manifest, "review", INPUTS_NAME)
    inputs = decode(inputs_raw, inputs_record["sha256"])
    need(len(inputs_raw) == inputs_record["bytes"], "Review inputs length mismatch")
    validate_inputs(inputs, manifest)
    need(receipt["checkpoint_sha256"] == inputs["checkpoints"]["final"] and
         receipt["linked_checkpoint_sha256"] == inputs["checkpoints"]["linked"],
         "Review receipt binds another checkpoint")
    for field, name in (("inputs_sha256", INPUTS_NAME), ("decisions_sha256", DECISIONS_NAME)):
        need(E.digest(receipt[field]) == inventory_record(manifest, "review", name)["sha256"],
             "Review control inventory mismatch")
    need(revision(receipt["source_revision"]) == manifest["source_snapshots"]["review"]["revision"],
         "Reviewer source revision differs")
    need(isinstance(trusted_source_inventory, dict) and
         REVIEW_SOURCES <= set(trusted_source_inventory), "Trusted reviewer dependency inventory incomplete")
    for name, row in trusted_source_inventory.items():
        E.relative(name); E.metadata(row)
    need(receipt["source_inputs"] == trusted_source_inventory, "Reviewer bytes differ from trusted local source")
    source_prefix = manifest["source_snapshots"]["review"]["bundle_relative_path"]
    source_names = {name[len(source_prefix) + 1:] for name in manifest["files"]
                    if name.startswith(source_prefix + "/")}
    need(source_names == set(trusted_source_inventory), "Reviewer snapshot inventory differs")
    for name, row in trusted_source_inventory.items():
        need(manifest["files"][source_prefix + "/" + name] == row, "Reviewer snapshot bytes differ")
    run_prefix = manifest["roots"]["review"]["bundle_relative_path"]
    outputs = {name[len(run_prefix) + 1:]: row for name, row in manifest["files"].items()
               if name.startswith(run_prefix + "/") and name != run_prefix + "/" + RECEIPT_NAME}
    need(receipt["outputs"] == outputs and
         {INPUTS_NAME, DECISIONS_NAME, "disposition.json"} <= set(outputs),
         "Review output inventory differs")
    return {"binding_consistent": True, "recorded_status": receipt["status"],
            "route_qualified": False, "semantic_replay_required": True, "hardware_measured": False}


def pending_categories(top_level, nested):
    """Retain all nested obligations; unknown categories cannot disappear."""
    need(isinstance(top_level, list) and all(isinstance(item, str) for item in top_level) and
         len(top_level) == len(set(top_level)) and set(top_level) == REQUIRED_CATEGORIES,
         "Missing/extra/duplicate final pending category")
    need(isinstance(nested, dict), "Malformed nested obligations")
    result = {}
    for role, items in nested.items():
        need(isinstance(role, str) and isinstance(items, list) and
             all(isinstance(item, str) and item for item in items) and len(items) == len(set(items)),
             "Malformed nested pending category")
        result[role] = sorted(items)
    return {"pending_reviews": sorted(top_level), "nested_pending_reviews": result,
            "route_qualified": False, "semantic_replay_required": True}

