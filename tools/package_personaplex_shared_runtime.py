#!/usr/bin/env python3
"""Build an isolated, hash-pinned PersonaPlex software qualification tree.

Standard library only. No production imports, installs, model conversion, board
access, services or downloads. The published old-runtime branch alone is not a
qualified common-image runtime; this explicit dependency manifest is required.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "tools/data/personaplex/shared_runtime_dependencies.json"


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def checked_file(root, name, expected):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Dependency path must be relative and stay inside its root")
    path = root / relative
    if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
        raise ValueError(f"Dependency is missing, linked or outside its source root: {name}")
    data = path.read_bytes()
    if sha256(data) != expected:
        raise ValueError(f"Dependency hash differs: {name}")
    return data


def build(runtime_root, source_root, output_dir, manifest_path=DEFAULT_MANIFEST):
    runtime_root, source_root, output_dir, manifest_path = (
        Path(p).resolve() for p in (runtime_root, source_root, output_dir, manifest_path))
    if output_dir.exists():
        raise FileExistsError("Integration output must be new; inspect prior build before retry")
    for protected in (runtime_root, source_root):
        if output_dir == protected or output_dir in protected.parents or protected in output_dir.parents:
            raise ValueError("Integration output must be isolated from both source roots")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("format_version") != 1 or manifest.get("physical_qualified") is not False
            or manifest.get("target") != "f2-hbm:p4rl8a64c150"
            or manifest.get("image") != "agfi-0001de1dd3df8e21e"):
        raise ValueError("Unexpected or unsafe shared-runtime integration contract")
    if (runtime_root / "REVISION").read_text().strip() != manifest["runtime_revision"]:
        raise ValueError("Pinned runtime revision differs")
    # Validate every input before creating the output. The allowlist is explicit;
    # no model/cache/credential/Git/session data can be copied by this builder.
    inputs = []
    for name, digest in manifest["runtime_files"].items():
        if not name.startswith(("opentpu/", "targets/")):
            raise ValueError("Runtime dependency is outside the code/target allowlist")
        checked_file(runtime_root, name, digest)
        inputs.append((runtime_root, name, digest))
    for name, digest in manifest["overlay_files"].items():
        if not name.startswith(("opentpu/llm/personaplex", "tools/personaplex_",
                                "tools/package_personaplex_shared_runtime.py",
                                "tests/test_personaplex_", "tools/data/personaplex/")):
            raise ValueError("PersonaPlex dependency is outside the source allowlist")
        checked_file(source_root, name, digest)
        inputs.append((source_root, name, digest))
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "building", "runtime_revision": manifest["runtime_revision"],
               "target": manifest["target"], "image_context": manifest["image"],
               "dependency_manifest_sha256": sha256(manifest_bytes),
               "physical_qualified": False, "full_speech_enabled": False,
               "files": {}}
    receipt_path = output_dir / "PERSONAPLEX_INTEGRATION.json"
    try:
        for root, name, digest in inputs:
            data = checked_file(root, name, digest)
            destination = output_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            receipt["files"][name] = digest
        (output_dir / "REVISION").write_text(manifest["runtime_revision"] + "\n")
        destination = output_dir / "tools/data/personaplex/shared_runtime_dependencies.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(manifest_bytes)
        receipt.update(status="completed", runtime_file_count=len(manifest["runtime_files"]),
                       overlay_file_count=len(manifest["overlay_files"]))
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    receipt = build(args.runtime_root, args.source_root, args.output_dir, args.manifest)
    print(json.dumps({k: receipt[k] for k in
                      ("status", "runtime_revision", "runtime_file_count", "overlay_file_count",
                       "target", "dependency_manifest_sha256", "physical_qualified")}, indent=2))


if __name__ == "__main__":
    main()
