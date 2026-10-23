"""Snapshot integrity tests; these do not run Vivado or claim FPGA results."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c2_ooc", ROOT / "tools/c2_ooc.py")
ooc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ooc)


def test_frozen_snapshot_is_relocatable_and_uses_real_dsp(tmp_path):
    source = ooc.prepare(tmp_path / "source")
    moved = tmp_path / "moved"
    shutil.move(str(source), moved)
    info = ooc.verify(moved)
    assert info["part"] == "xcvu47p-fsvh2892-2-e"
    assert info["threads"] == 4
    assert "rtl/tech/xilinxus/otpu_tech_dsp.sv" in info["sources"]
    assert len(info["files"]) == 9
    assert "DSP48E2" in (moved / "src/rtl/tech/xilinxus/otpu_tech_dsp.sv").read_text()


def test_snapshot_rejects_changed_rtl_and_fresh_directory_reuse(tmp_path):
    out = ooc.prepare(tmp_path / "run")
    with pytest.raises(FileExistsError):
        ooc.prepare(out)
    with (out / "src/rtl/mxu/otpu_c2_mxu.sv").open("a") as f:
        f.write("\n// changed after freeze\n")
    with pytest.raises(RuntimeError, match="frozen build input modified"):
        ooc.verify(out)


def test_snapshot_rejects_inventory_or_parameter_change(tmp_path):
    out = ooc.prepare(tmp_path / "run")
    info = ooc.verify(out)
    info["parameters"]["DEPTH"] = 128
    ooc.write_json(out / "inputs.json", info)
    with pytest.raises(RuntimeError, match="parameters"):
        ooc.verify(out)


def test_resume_requires_matching_input_and_checkpoint_hashes(tmp_path):
    out = ooc.prepare(tmp_path / "run")
    dcp = out / "checkpoints/post_synth.dcp"
    dcp.parent.mkdir()
    dcp.write_bytes(b"test fixture, not a Vivado checkpoint")
    summary = {"tool_exit_code": 0, "completed": True,
               "inputs_sha256": ooc.sha(out / "inputs.json"),
               "artifacts": {"checkpoints/post_synth.dcp": {"sha256": ooc.sha(dcp)}}}
    ooc.write_json(out / "outputs-synth.json", summary)
    assert ooc.resume_checkpoint(out) == dcp
    dcp.write_bytes(b"changed fixture")
    with pytest.raises(RuntimeError, match="unverified or changed"):
        ooc.resume_checkpoint(out)


@pytest.mark.parametrize("banner", ["Vivado v2025.2 (64-bit)", "Vivado 2025.2\nBuild 123"])
def test_exact_tool_version(banner):
    ooc.check_version(banner)


@pytest.mark.parametrize("banner", ["Vivado v2024.2", "Vivado v2025.1", "2025.2", "Vivado v2025.20"])
def test_other_tool_version_rejected(banner):
    with pytest.raises(RuntimeError, match="2025.2"):
        ooc.check_version(banner)


def test_tcl_scripts_are_complete():
    if shutil.which("tclsh") is None:
        pytest.skip("Tcl interpreter unavailable")
    # Parse only. Vivado command existence/behavior is validated on the build host.
    for name in ("ooc.tcl", "constraints.xdc"):
        script = (ROOT / "tools/c2_ooc" / name).read_text()
        result = subprocess.run(["tclsh"], input="puts [info complete {" + script + "}]\n",
                                capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "1"
