"""Synthetic schema2 transport fixtures, never FPGA qualification evidence."""
import copy
import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c1_raw_fixture", ROOT / "tests/test_c1_candidate_evidence.py")
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
e = f.e


def refresh(root, m):
    m["files"] = {name: row for name, row in f.inventory(root).items() if name != e.MANIFEST}
    for role, name in e.schema_receipts(m["schema"]).items():
        relative = m["roots"][role]["bundle_relative_path"] + "/" + name
        m["receipts"][role] = {"relative_path": relative, "sha256": m["files"][relative]["sha256"]}
    f.put(root / e.MANIFEST, m)
    return e.sha(root / e.MANIFEST)


def fixture_raw(tmp_path, monkeypatch, *, original_base="/unavailable/original"):
    root, m, _ = f.fixture_bundle(tmp_path, monkeypatch, original_base=original_base)
    shutil.rmtree(root / "runs/qualify")
    shutil.rmtree(root / "sources/qualify")
    for key in ("roots", "source_snapshots", "receipts"):
        del m[key]["qualify"]
    m["schema"] = 2
    final_sha = e.sha(root / "runs/refine/post_route.dcp")
    for role in e.RAW_ROLES:
        final = role == "capture"
        destination = root / "runs" / role
        m["roots"][role] = {"original_absolute_path": original_base + "/" + role,
                            "bundle_relative_path": "runs/" + role}
        m["source_snapshots"][role] = {"revision": "e" * 40, "bundle_relative_path": "sources/" + role}
        f.source_copy(root, role, ["boards/c1_f2_alex/" + name for name in e.RAW_HELPERS[role]])
        for name in e.RAW_HELPERS[role]:
            f.put(destination / name, (e.HERE / name).read_bytes())
        receipt = {
            "kind": "c1-final-raw-capture" if final else "c1-linked-" + role.removeprefix("linked_") + "-raw-capture",
            "source_revision": "e" * 40, "inputs_unchanged": True,
            "route_qualified": False, "hardware_measured": False, "hardware_actions": False,
            "checkpoint_sha256": final_sha if final else e.sha(root / "runs/link/post_link.dcp"),
            ("refine" if final else "linked") + "_receipt_sha256":
                e.sha(root / "runs" / ("refine" if final else "link") / "stage-result.json"),
        }
        if final:
            receipt.update(schema=1, stages={stage: {"returncode": 0} for stage in
                                            ("final", "native", "reset", "handshake", "ready")},
                           excluded_mutating_output=["console.log"],
                           helpers={name: e.sha(destination / name) for name in e.RAW_HELPERS[role]})
            for name in e.FINAL_REPORTS:
                f.put(destination / "final" / name, "SYNTHETIC RAW REPORT FOR PARSER TEST ONLY\n")
            for name in ("NOT_QUALIFIED.txt", "unused_outputs.tcldata"):
                f.put(destination / "final" / name, "SYNTHETIC pending review\n")
            f.put(destination / "final/summary.tcldata",
                  f"schema 1 kind c1_final_raw_capture stage final_raw route_qualified 0 hardware_measured 0 "
                  f"checkpoint {original_base}/refine/post_route.dcp checkpoint_sha256 {final_sha} "
                  "blockers {structural_review_incomplete} pending_reviews {SYNTHETIC_pending_review}\n")
            for stage in receipt["stages"]:
                f.put(destination / (stage + ".tcl"), "# SYNTHETIC wrapper, never execute\n")
                f.put(destination / (stage + ".log"), "SYNTHETIC log\n")
            for name in ("native/topology.tcldata", "ready/ready.tcldata", "handshake.tcldata",
                         "reset/topology.tcldata", "reset/timing.rpt", "reset/u_rst_core_pulse.rpt",
                         "reset/u_rst_hbm_pulse.rpt", "reset/u_status_cdc_pulse.rpt"):
                f.put(destination / name, "SYNTHETIC RAW\n")
        else:
            receipt["returncode"] = 0
            for name in ("run.tcl", "vivado.log",
                         "raw/" + ("topology" if role == "linked_native" else "ready") + ".tcldata"):
                f.put(destination / name, "SYNTHETIC RAW\n")
        f.put(destination / "driver.py", "# SYNTHETIC driver, never execute\n")
        receipt["files"] = f.inventory(destination)
        f.put(destination / "capture-receipt.json", receipt)
    return root, m, refresh(root, m)


@pytest.fixture
def raw(tmp_path, monkeypatch):
    return fixture_raw(tmp_path, monkeypatch)


def repin(root, m, role):
    path = root / "runs" / role / "capture-receipt.json"
    receipt = f.read(path)
    receipt["files"] = {name: row for name, row in f.inventory(path.parent).items()
                        if name != "capture-receipt.json"}
    f.put(path, receipt)
    return refresh(root, m)


@pytest.mark.parametrize('builder',(e.INSTANCE,'i-0ec8d33362b3100be'))
def test_schema2_retains_pending_and_structural_error_after_relocation(raw, tmp_path, builder):
    root, m, _ = raw;m['instance_id']=builder;digest=refresh(root,m)
    destination = tmp_path / "relocated"
    shutil.move(root, destination)
    result = e.verify_bundle(destination, digest)
    assert result['instance_id']==builder
    assert result["schema"] == 2 and result["route_qualified"] is False
    assert result["qualification_receipt_sha256"] is None
    assert result["review_required"] is True and result["hardware_measured"] is False
    assert result["blockers"] == ["structural_review_incomplete"]
    assert result["pending_reviews"] == ["SYNTHETIC_pending_review"]
    assert set(result["raw_capture_receipt_sha256"]) == set(e.RAW_ROLES)


@pytest.mark.parametrize("change", ["approve", "ancestor", "dcp", "failed_stage", "missing_stage",
                                  "missing_reset", "helper", "helper_map", "extra_file", "summary"])
def test_schema2_rejects_bad_binding_or_incomplete_capture(raw, change):
    root, m, _ = raw
    destination = root / "runs/capture"
    path = destination / "capture-receipt.json"
    receipt = f.read(path)
    if change == "approve": receipt["route_qualified"] = True
    elif change == "ancestor": receipt["refine_receipt_sha256"] = "0" * 64
    elif change == "dcp": receipt["checkpoint_sha256"] = "0" * 64
    elif change == "failed_stage": receipt["stages"]["reset"]["returncode"] = 1
    elif change == "missing_stage": del receipt["stages"]["ready"]
    elif change == "missing_reset": (destination / "reset/topology.tcldata").unlink()
    elif change == "helper": f.put(destination / "capture_ready.tcl", "# changed source")
    elif change == "helper_map": receipt["helpers"]["capture_ready.tcl"] = "0" * 64
    elif change == "extra_file": f.put(destination / "unbound.txt", "not receipt bound")
    elif change == "summary":
        p = destination / "final/summary.tcldata"
        f.put(p, p.read_text().replace("route_qualified 0", "route_qualified 1"))
    f.put(path, receipt)
    digest = refresh(root, m) if change == "extra_file" else repin(root, m, "capture")
    with pytest.raises(ValueError):
        e.verify_bundle(root, digest)


@pytest.mark.parametrize("role", ["linked_native", "linked_ready"])
def test_schema2_rejects_linked_capture_from_another_checkpoint(raw, role):
    root, m, _ = raw
    path = root / "runs" / role / "capture-receipt.json"
    receipt = f.read(path)
    receipt["checkpoint_sha256"] = "0" * 64
    f.put(path, receipt)
    with pytest.raises(ValueError, match="checkpoint ancestry"):
        e.verify_bundle(root, refresh(root, m))


@pytest.mark.parametrize('builder',(e.INSTANCE,'i-0ec8d33362b3100be'))
def test_schema2_collects_frozen_capture_inventory_without_copying_live_logs(tmp_path, monkeypatch, builder):
    root, m, _ = fixture_raw(tmp_path, monkeypatch, original_base=str(tmp_path / "portable/runs"))
    module_spec = importlib.util.spec_from_file_location("c1_raw_collect", e.HERE / "collect_candidate.py")
    collector = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(collector)
    monkeypatch.setattr(collector, "e", e)
    class FixtureGit:
        def __init__(self, path, revision): self.path = Path(path)
        def read(self, name): return (self.path / name).read_bytes()
    monkeypatch.setattr(collector, "GitSource", FixtureGit)
    f.put(root / "runs/capture/console.log", "Live outer log excluded by receipt\n")
    plan = {
        "schema": 2, "kind": "c1-alex-candidate-collection-inputs", "instance_id": builder,
        "roots": {role: row["original_absolute_path"] for role, row in m["roots"].items()},
        "receipt_sha256": {role: row["sha256"] for role, row in m["receipts"].items()},
        "source_checkouts": {role: {"path": str(root / row["bundle_relative_path"]), "revision": row["revision"]}
                             for role, row in m["source_snapshots"].items()},
    }
    before = f.inventory(root)
    output = tmp_path / "collection"
    result = collector.collect(plan, output)
    assert result['instance_id']==result['verified']['instance_id']==builder
    assert f.read(output/'bundle'/e.MANIFEST)['instance_id']==builder
    assert result["schema"] == 2 and result["verified"]["route_qualified"] is False
    assert before == f.inventory(root)
    assert not (output / "bundle/runs/capture/console.log").exists()
    assert e.inspect_archive(output / "evidence.tar", result["manifest_sha256"])["route_qualified"] is False


def test_schema2_allowed_builder_swap_still_requires_manifest_hash(raw):
    root,m,digest=raw;m['instance_id']='i-0ec8d33362b3100be';refresh(root,m)
    with pytest.raises(ValueError):e.verify_bundle(root,digest)
