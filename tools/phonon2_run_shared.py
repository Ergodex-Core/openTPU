"""Run Phonon encoder qualification against an existing shared-profile runtime.

Useful before the isolated ASR commit is merged into the common F2 branch.
Imports only explicitly selected local user-repository source; never downloads
code, opens the FPGA, changes images or enables serving.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parents[1]))
    args, remaining = parser.parse_known_args()
    runtime, source = Path(args.runtime_root).resolve(), Path(args.source_root).resolve()
    if not (runtime / "opentpu/isasim.py").is_file() or not (runtime / "targets/f2-hbm.toml").is_file():
        raise ValueError("runtime root must contain the common F2-HBM runtime and target declaration")
    expected = json.loads((source / "docs/phonon2/shared_profile.json").read_text())
    manifest = json.loads((runtime / "source_manifest.json").read_text())
    if (manifest["revision"] != expected["runtime_source_revision"] or
            manifest["files"] != expected["runtime_source_sha256"]):
        raise RuntimeError("runtime manifest differs from qualified shared-profile source")
    for name, sha in manifest["files"].items():
        if hashlib.sha256((runtime / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError("runtime source differs from manifest: " + name)
    sys.path.insert(0, str(runtime))
    import opentpu
    if str(source / "opentpu") not in opentpu.__path__:
        opentpu.__path__.append(str(source / "opentpu"))
    sys.path.insert(0, str(source))
    os.environ["OTPU_TARGET"] = "f2-hbm:p4rl8a64c150"
    from opentpu.asr.phonon2_encoder import shared_profile_config, SHARED_ARCH_HASH
    from opentpu import target
    if f"{target.arch_hash(shared_profile_config()):08x}" != SHARED_ARCH_HASH:
        raise RuntimeError("shared-profile configuration mismatch")
    sys.argv = ["tools.phonon2_qualify_encoder", *remaining]
    runpy.run_module("tools.phonon2_qualify_encoder", run_name="__main__")


if __name__ == "__main__":
    main()
