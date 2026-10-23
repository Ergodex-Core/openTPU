#!/usr/bin/env python3
"""Build a deterministic openTPU PersonaPlex runtime and RTL test source archive.

Includes the Python runtime, RTL/testbenches, profiles, port tests and deployment guide.
Model weights, credentials, caches, git metadata and FPGA bitstreams are excluded.
No network, AWS or git writes are performed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "personaplex-fpga-source"


def source_paths(root: Path) -> list[Path]:
    # The explicit roots are part of the packaging contract. Do not expand to
    # an arbitrary checkout walk: a deployment tree can contain model files.
    paths = [root / name for name in ("LICENSE", "pyproject.toml", "README.md",
             "docs/personaplex_fpga.md", "tools/personaplex_fpga.py",
             "tools/personaplex_tunnel.sh", "tools/package_personaplex.py",
             "tools/personaplex_host_check.py", "tools/personaplex_reference.py",
             "tools/personaplex_isa_check.py",
             "tools/personaplex_verilator.py", "tests/conftest.py", "tests/test_f2_rtl.py")]
    for directory in ("opentpu", "targets", "tools/data/personaplex", "rtl", "sim/verilator"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            if path.is_file() and path.suffix in {".py", ".toml", ".html", ".sh", ".c", ".h", ".json",
                                                  ".csv", ".conf", ".rules", ".patch",
                                                  ".sv", ".svh", ".v", ".tcl", ".txt"}:
                paths.append(path)
    paths.extend((root / "tests").glob("test_personaplex_*.py"))
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular source file: {path.relative_to(root)}")
        if root.resolve() not in path.resolve().parents:
            raise ValueError(f"Source escapes checkout: {path}")
    return sorted(set(paths), key=lambda p: p.relative_to(root).as_posix())


def build_archive(root: Path, output: Path) -> dict:
    files = source_paths(root)
    if (root / ".git").exists():
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                check=True, capture_output=True, text=True).stdout.strip()
    else:
        # Extracted runtime bundles deliberately omit Git metadata. Preserve
        # their recorded base while hashing the current source again.
        commit = json.loads((root / "SOURCE_MANIFEST.json").read_text())["base_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", commit):
        raise ValueError("Invalid base commit in source manifest")
    contents = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
    manifest = {
        "format": "opentpu-personaplex-source-v1",
        "base_commit": commit,
        "model_repo": "aufklarer/PersonaPlex-7B-MLX-4bit",
        "model_revision": "8c686347007de51d9017706e5543d6f1757ecf5d",
        "weights_included": False,
        "rtl_simulation_sources_included": True,
        "hardware_validated": False,
        "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in contents.items()},
    }
    contents["SOURCE_MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create keeps an earlier reviewable package intact.
    with output.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name, data in sorted(contents.items()):
                    entry = tarfile.TarInfo(f"{PREFIX}/{name}")
                    entry.size = len(data)
                    entry.mode = 0o755 if name.endswith(".sh") else 0o644
                    entry.mtime = entry.uid = entry.gid = 0
                    entry.uname = entry.gname = ""
                    archive.addfile(entry, io.BytesIO(data))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"archive": str(output.resolve()), "sha256": digest,
            "source_files": len(files), "bytes": output.stat().st_size,
            "base_commit": commit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_archive(ROOT, args.output), indent=2))


if __name__ == "__main__":
    main()
