"""C2 image identity and frozen-input checks, without requiring the AWS HDK."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location("c2_f2_build", ROOT / "boards/c2_f2/build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(tmp_path):
    module = builder()
    hdk = tmp_path / "hdk"
    wrapper = hdk / "hdk/cl/examples/cl_mem_perf/design/cl_mem_hbm_wrapper.sv"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("// test HDK wrapper\n")
    shell = hdk / "hdk/common/shell_stable"
    script = shell / "build/scripts/build_all.tcl"
    script.parent.mkdir(parents=True)
    script.write_text("# test HDK flow\n")
    out = tmp_path / "run"
    cl = module.prepare(out, hdk, shell, "1234abcd")
    return module, out, cl, hdk, shell


def test_kernel_snapshot_contains_one_c2_and_no_production_core(tmp_path):
    _, out, cl, _, _ = snapshot(tmp_path)
    sources = (cl / "design/sources.tcl").read_text()
    assert "rtl/mxu/otpu_c2_mxu.sv" in sources
    assert "rtl/c2/otpu_c2_hbm_kernel.sv" in sources
    assert "rtl/boards/f2/otpu_f2_hbm_mem.sv" in sources
    assert "rtl/top/otpu_slice.sv" not in sources
    assert "rtl/shell/otpu_board.sv" not in sources
    system = (cl / "design/otpu_f2_c2_sys.sv").read_text()
    assert system.count("otpu_c2_hbm_kernel #(") == 1
    assert "otpu_board #(" not in system
    assert '(* ASYNC_REG = "TRUE" *) logic [1:0] ready_sync;' in system
    assert ".hbm_ready(hbm_rdy)" in system
    assert ".BUILD_ID(32'h1234abcd)" in (cl / "design/opentpu_params.vh").read_text()
    info = json.loads((out / "inputs.json").read_text())
    assert info["kind"] == "c2-hbm-kernel"
    assert info["hbm_ports"] == 8
    for name, digest in info["sources"].items():
        assert hashlib.sha256((cl / name).read_bytes()).hexdigest() == digest


def test_snapshot_never_replaces_existing_run(tmp_path):
    module, out, _, hdk, shell = snapshot(tmp_path)
    before = (out / "inputs.json").read_bytes()
    with pytest.raises(RuntimeError, match="Refusing to reuse"):
        module.prepare(out, hdk, shell, "ffffffff")
    assert (out / "inputs.json").read_bytes() == before


def test_hbm_qualification_has_no_ddr_or_pumped_tmem_assumptions(tmp_path):
    _, _, cl, _, _ = snapshot(tmp_path)
    scripts, constraints = cl / "build/scripts", cl / "build/constraints"
    synth = (scripts / "synth_cl_opentpu.tcl").read_text()
    impl = (scripts / "build_level_1_cl.tcl").read_text()
    assert "otpu_clk2x.tcl" not in synth
    assert "read_xdc -unmanaged ${::constraints_dir}/otpu_clk2x.tcl" not in impl
    assert "read_xdc -unmanaged ${constraints_dir}/cl_timing_user.tcl" in synth
    assert "source ${scripts_dir}/validate_route.tcl" in impl
    assert (scripts / "validate_route.tcl").read_bytes() == (
        ROOT / "boards/c2_f2/validate_hbm_route.tcl").read_bytes()
    pnr = (constraints / "cl_pnr_user.tcl").read_text()
    assert "create_pblock pb_otpu_tmem" not in pnr
    assert "create_pblock pb_otpu_slr0" in pnr
    assert "core_reset_n hbm_reset_n" in pnr


def test_support_mutation_is_rejected_before_build(tmp_path, monkeypatch):
    module = builder()
    support = tmp_path / "support"
    support.mkdir()
    (support / "test.sv").write_text("original")
    (support / "manifest.json").write_text(json.dumps({"sources": {
        "test.sv": hashlib.sha256(b"original").hexdigest()}}))
    monkeypatch.setattr(module, "SUPPORT", support)
    module.verify_support()
    (support / "test.sv").write_text("changed")
    with pytest.raises(RuntimeError, match="Frozen F2 support changed"):
        module.verify_support()


def test_generated_top_uses_separate_kernel_identity():
    module = builder()
    module.verify_support()
    system = module.make_sys()
    assert "module otpu_f2_c2_sys" in system
    assert "NPORT != 8" in system
    assert "CH_LOG2" not in system
    assert "TMEM_WORDS" not in system
