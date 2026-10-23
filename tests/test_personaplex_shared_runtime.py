"""Hash-pinned isolated runtime packaging; execute on EC2 with the other checks."""
import hashlib
import json

import pytest

from tools.package_personaplex_shared_runtime import build


def fixture(tmp_path):
    runtime, source, output = (tmp_path / n for n in ("runtime", "source", "output"))
    runtime.mkdir(); source.mkdir()
    (runtime / "REVISION").write_text("pinned-test-revision\n")
    files = {"opentpu/base.py": b"base source", "targets/f2-hbm.toml": b"fake target"}
    overlays = {"opentpu/llm/personaplex_fake.py": b"overlay source"}
    for root, entries in ((runtime, files), (source, overlays)):
        for name, data in entries.items():
            p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    digest = lambda v: hashlib.sha256(v).hexdigest()
    manifest = {"format_version": 1, "physical_qualified": False,
                "runtime_revision": "pinned-test-revision",
                "target": "f2-hbm:p4rl8a64c150", "image": "agfi-0001de1dd3df8e21e",
                "runtime_files": {k: digest(v) for k, v in files.items()},
                "overlay_files": {k: digest(v) for k, v in overlays.items()}}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    return runtime, source, output, path, manifest


def test_isolated_builder_copies_only_verified_allowlisted_sources(tmp_path):
    runtime, source, output, path, manifest = fixture(tmp_path)
    receipt = build(runtime, source, output, path)
    assert receipt["status"] == "completed" and receipt["physical_qualified"] is False
    assert receipt["runtime_file_count"] == 2 and receipt["overlay_file_count"] == 1
    assert receipt["files"] == {**manifest["runtime_files"], **manifest["overlay_files"]}
    assert (output / "PERSONAPLEX_INTEGRATION.json").is_file()
    assert (output / "tools/data/personaplex/shared_runtime_dependencies.json").read_bytes() == path.read_bytes()
    assert (output / "opentpu/llm/personaplex_fake.py").read_bytes() == b"overlay source"
    with pytest.raises(FileExistsError):
        build(runtime, source, output, path)


@pytest.mark.parametrize("which", ["runtime", "source"])
def test_builder_rejects_changed_source_before_creating_output(tmp_path, which):
    runtime, source, output, path, _ = fixture(tmp_path)
    target = runtime / "opentpu/base.py" if which == "runtime" else source / "opentpu/llm/personaplex_fake.py"
    target.write_bytes(b"unverified changed source")
    with pytest.raises(ValueError, match="hash differs"):
        build(runtime, source, output, path)
    assert not output.exists()


def test_builder_rejects_linked_dependency_and_path_escape(tmp_path):
    runtime, source, output, path, manifest = fixture(tmp_path)
    linked = runtime / "opentpu/base.py"
    linked.unlink(); linked.symlink_to(runtime / "REVISION")
    with pytest.raises(ValueError, match="linked"):
        build(runtime, source, output, path)
    assert not output.exists()
    linked.unlink(); linked.write_bytes(b"base source")
    manifest["runtime_files"] = {"opentpu/../../outside": "0" * 64}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="relative"):
        build(runtime, source, output, path)
    assert not output.exists()


def test_builder_requires_exact_contract_revision_and_isolated_output(tmp_path):
    runtime, source, output, path, manifest = fixture(tmp_path)
    with pytest.raises(ValueError, match="isolated"):
        build(runtime, source, source / "nested-output", path)
    (runtime / "REVISION").write_text("another revision")
    with pytest.raises(ValueError, match="revision differs"):
        build(runtime, source, output, path)
    (runtime / "REVISION").write_text("pinned-test-revision")
    manifest["physical_qualified"] = True; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe"):
        build(runtime, source, output, path)
    assert not output.exists()
