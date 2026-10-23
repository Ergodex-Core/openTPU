"""C2 image identity and frozen-input checks, without requiring the AWS HDK."""
import hashlib
import importlib.util
import json
from pathlib import Path
from fractions import Fraction
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location("c2_f2_build", ROOT / "boards/c2_f2/build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smartconnect_fixture(module, hdk):
    helper = module.support_builder()
    bd, wrapper = hdk / helper.HBM_SC_BD, hdk / helper.HBM_SC_WRAPPER
    bd.parent.mkdir(parents=True, exist_ok=True)
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    design = {"ports": {"aclk_250": {"type": "clk", "parameters": {"FREQ_HZ": {"value": "250000000"}}},
                        "aclk_450": {"type": "clk"}, "aresetn_250": {"type": "rst"}},
              "interface_ports": {}, "components": {"smartconnect_0": {
                  "vlnv": "xilinx.com:ip:smartconnect:1.0", "parameters": {
                      k: {"value": v} for k, v in {"NUM_CLKS": "2", "NUM_MI": "1", "NUM_SI": "1"}.items()}}}}
    for name, mode, width, ids in [("AXI4", "Slave", 512, 16), ("AXI3", "Master", 256, 0)]:
        design["interface_ports"][name] = {"mode": mode, "parameters": {
            k: {"value": str(v)} for k, v in {"PROTOCOL": name, "DATA_WIDTH": width,
            "ID_WIDTH": ids, "ADDR_WIDTH": 64, "NUM_READ_OUTSTANDING": 32}.items()}}
    bd.write_text(json.dumps({"design": design}))
    # Actual Vivado wrapper style: non-ANSI port declarations. Interface fixture
    # is independent of the builder's declaration parser.
    pins = {"aclk_250": ("input", 1), "aclk_450": ("input", 1), "aresetn_250": ("input", 1)}
    for bus, send, receive, data, length, lock in [
            ("AXI4", "input", "output", 512, 8, 1), ("AXI3", "output", "input", 256, 4, 2)]:
        for ch in ("aw", "ar"):
            for suffix, width in [("addr", 64), ("len", length), ("size", 3), ("burst", 2),
                                  ("lock", lock), ("cache", 4), ("prot", 3), ("qos", 4), ("valid", 1)]:
                pins[bus+"_"+ch+suffix] = (send, width)
            pins[bus+"_"+ch+"ready"] = (receive, 1)
        for name, width in [("wdata", data), ("wstrb", data//8), ("wlast", 1), ("wvalid", 1),
                            ("bready", 1), ("rready", 1)]:
            pins[bus+"_"+name] = (send, width)
        for name, width in [("wready", 1), ("bresp", 2), ("bvalid", 1), ("rdata", data),
                            ("rresp", 2), ("rlast", 1), ("rvalid", 1)]:
            pins[bus+"_"+name] = (receive, width)
    pins.update({"AXI4_awid": ("input", 16), "AXI4_arid": ("input", 16),
                 "AXI4_bid": ("output", 16), "AXI4_rid": ("output", 16)})
    wrapper.write_text("module cl_axi_sc_1x1_wrapper("+",".join(pins)+");\n" + "\n".join(
        f"{direction} " + (f"[{width-1}:0] " if width > 1 else "") + name + ";"
        for name, (direction, width) in pins.items()) + "\nendmodule\n")
    # The offline fixture has its own explicit pins. Production accepts only the
    # independently recorded dedicated-host bytes, with no CLI override.
    module.SMARTCONNECT_SHA256 = {p.name: module.sha(p) for p in (bd, wrapper)}
    return bd, wrapper


def snapshot(tmp_path):
    module = builder()
    hdk = tmp_path / "hdk"
    smartconnect_fixture(module, hdk)
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
    assert "rtl/boards/f2/otpu_native_axi.sv" in sources
    assert "otpu_f2_hbm_nch.sv" not in sources
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
    assert info["support_commit"] == "fbf21635d7ac7ab4633eaa0953dbc39274dcb690"
    assert info["implementation_tuning"] == {"place": "ExtraNetDelay_high", "phys": "AggressiveExplore",
                                              "route": "Explore", "retime": True}
    assert info["smartconnect"]["driven_clock_mhz"] == {"aclk_250": 200, "aclk_450": 450}
    for name, record in info["smartconnect"]["files"].items():
        assert hashlib.sha256((out / "vendor-inputs" / name).read_bytes()).hexdigest() == record["sha256"]
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
    assert "general.maxThreads 4" in synth and "general.maxThreads 4" in impl
    assert "add_files ${HDK_BD_SRC_DIR}/cl_axi_sc_1x1/cl_axi_sc_1x1.bd" in synth
    assert "${HDK_BD_GEN_DIR}/cl_axi_sc_1x1/hdl/cl_axi_sc_1x1_wrapper.v" in synth
    assert "otpu_hdk_phys_opt_design -retime" in impl
    assert "source ${scripts_dir}/validate_route.tcl" in impl
    assert (scripts / "validate_route.tcl").read_bytes() == (
        ROOT / "boards/c2_f2/validate_hbm_route.tcl").read_bytes()
    for name in ("classify_methodology.tcl", "check_scoped_clocks.tcl"):
        assert (scripts / name).read_bytes() == (ROOT / "boards/c2_f2" / name).read_bytes()
    for name in ("check_shell_baseline.tcl", "shell_baseline_2025_2.tcl"):
        assert (scripts / name).read_bytes() == (ROOT / "boards/c2_f2/support/boards/f2/build/scripts" / name).read_bytes()
    pnr = (constraints / "cl_pnr_user.tcl").read_text()
    assert "create_pblock pb_otpu_tmem" not in pnr
    assert "create_pblock pb_otpu_slr0" in pnr
    assert re.findall(r"^create_pblock (\w+)$", pnr, re.M) == [
        "pb_otpu_slr0", "pb_otpu_fab", "pb_otpu_pcis", "pb_otpu_xs"]
    assert "CLOCKREGION_X0Y0:CLOCKREGION_X7Y0" in pnr
    assert "WRAPPER/CL/g_sc[*].u_sc" in pnr
    assert "u_slice" not in pnr[pnr.index("create_pblock pb_otpu_slr0"):]
    assert "OTPU_FLOORPLAN_APPLIED pblocks=4" in pnr
    assert "core_reset_n hbm_reset_n" in pnr


@pytest.mark.parametrize("change,reason", [
    ("missing_bd", "Missing HDK SmartConnect"),
    ("address", "address width must be 64"),
    ("protocol", "expected"),
    ("data", "AXI4_rdata"),
    ("direction", "AXI4_rid"),
    ("clocks", "two clocks"),
    ("bytes", "Unreviewed HDK SmartConnect bytes"),
])
def test_smartconnect_preflight_rejects_incompatible_or_unreviewed_inputs(tmp_path, change, reason):
    module = builder()
    bd, wrapper = smartconnect_fixture(module, tmp_path)
    data = json.loads(bd.read_text())
    if change == "missing_bd":
        bd.unlink()
    elif change == "address":
        data["design"]["interface_ports"]["AXI3"]["parameters"]["ADDR_WIDTH"]["value"] = "34"
        bd.write_text(json.dumps(data))
    elif change == "protocol":
        data["design"]["interface_ports"]["AXI4"]["parameters"]["ID_WIDTH"]["value"] = "0"
        bd.write_text(json.dumps(data))
    elif change == "data":
        wrapper.write_text(wrapper.read_text().replace("output [511:0] AXI4_rdata;", "output [255:0] AXI4_rdata;"))
    elif change == "direction":
        wrapper.write_text(wrapper.read_text().replace("output [15:0] AXI4_rid;", "input [15:0] AXI4_rid;"))
    elif change == "clocks":
        data["design"]["components"]["smartconnect_0"]["parameters"]["NUM_CLKS"]["value"] = "1"
        bd.write_text(json.dumps(data))
    else:
        wrapper.write_text(wrapper.read_text() + "// different generated version\n")
    with pytest.raises(RuntimeError, match=reason):
        module.check_smartconnect(tmp_path)


def test_vendor_input_change_after_snapshot_is_rejected(tmp_path):
    module, out, _, hdk, _ = snapshot(tmp_path)
    record = json.loads((out / "inputs.json").read_text())["smartconnect"]
    module.check_vendor_unchanged(hdk, record)
    wrapper = hdk / module.support_builder().HBM_SC_WRAPPER
    wrapper.write_text(wrapper.read_text().replace("input aclk_450;", "input aclk_451;"))
    with pytest.raises(RuntimeError, match="input changed during build"):
        module.check_vendor_unchanged(hdk, record)


@pytest.mark.parametrize("change", ["edit", "add", "remove"])
def test_child_ip_synthesis_inventory_detects_changes_and_ignores_logs(tmp_path, change):
    module = builder()
    _, wrapper = smartconnect_fixture(module, tmp_path)
    child = wrapper.parent.parent / "ip/child"
    child.mkdir(parents=True)
    constraint, checkpoint = child / "child.xdc", child / "child.dcp"
    constraint.write_text("# generated child constraint\n")
    checkpoint.write_bytes(b"test-only checkpoint envelope")
    record = module.check_smartconnect(tmp_path)
    assert record["files"][str(constraint.relative_to(tmp_path))]["preserved"]
    assert not record["files"][str(checkpoint.relative_to(tmp_path))]["preserved"]
    (child / "vivado.log").write_text("new log content\n")
    module.check_vendor_unchanged(tmp_path, record)
    if change == "edit":
        checkpoint.write_bytes(b"different checkpoint envelope")
    elif change == "add":
        (child / "new.mem").write_text("00000000\n")
    else:
        constraint.unlink()
    with pytest.raises(RuntimeError, match="input changed during build"):
        module.check_vendor_unchanged(tmp_path, record)


def test_child_ip_directory_link_is_not_silently_omitted(tmp_path):
    module = builder()
    _, wrapper = smartconnect_fixture(module, tmp_path)
    outside = tmp_path / "shared-generated"
    outside.mkdir()
    (outside / "child.sv").write_text("// generated child\n")
    (wrapper.parent / "child").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="directory link"):
        module.check_smartconnect(tmp_path)


def run_hclk_gate(tmp_path, timing, cells, fifos=3, counters=1):
    tclsh = shutil.which("tclsh")
    if not tclsh:
        pytest.skip("tclsh required")
    script = tmp_path / "gate.tcl"
    driver = r'''
set bounds {}
set skews {}
proc get_pins {args} { return [lindex $args end] }
proc get_ports {args} { return [lindex $args end] }
proc get_clocks {args} { return [lindex $args end] }
proc get_property {key object} {
  if {$key eq "NAME"} {return $object}
  if {$object eq "u_clk/O"} {return 5.0}
  if {$object eq "u_hclk/O"} {return 2.222222222222}
  return 4.0
}
proc get_cells {args} {
  set selector [lindex $args end]
  if {[string first "otpu_afifo" $selector] >= 0} {
    set result {}; for {set i 0} {$i<$::fifo_count} {incr i} {lappend result host/fifo$i}; return $result
  }
  if {[string first "otpu_f2_hbm_host" $selector] >= 0} {
    set result {}; for {set i 0} {$i<$::counter_count} {incr i} {lappend result host$i}; return $result
  }
  return [lindex $args end]
}
proc all_registers {args} {return $::registers}
proc set_max_delay {args} {lappend ::bounds $args}
proc set_bus_skew {args} {lappend ::skews $args}
proc set_false_path {args} {}
'''
    script.write_text(f"set fifo_count {fifos}\nset counter_count {counters}\n" +
        "set registers [list " + " ".join("{"+c+"}" for c in cells) + "]\n" + driver + timing +
        '\nif {[llength $bounds] != 6 || [llength $skews] != 7} {error "wrong bounds inventory"}\n')
    return subprocess.run([tclsh, str(script)], text=True, capture_output=True)


def vendor_cells():
    tail = ("cl_axi_sc_1x1_i/smartconnect_0/inst/s00_nodes/s00_w_node/inst/"
            "inst_mi_handler/gen_normal_area.inst_fifo_node_payld/gen_xpm_memory_fifo.inst_fifo/"
            "gen_async_clocks.inst_cdc_addrb_to_wr_clk/src_gray_ff_reg[5]")
    return [f"g_sc[{p}].u_sc/{tail}" for p in range(8)] + [
        "u_hbm/HBM_CORE_I/inst/reset_ff", *[f"hbm_reset_n_reg[{p}]" for p in range(3)]]


def test_hclk_gate_accepts_actual_vivado_vendor_names_and_exact_host_cdc(tmp_path):
    _, _, cl, _, _ = snapshot(tmp_path)
    timing = (cl / "build/constraints/cl_timing_user.tcl").read_text()
    result = run_hclk_gate(tmp_path, timing, vendor_cells())
    assert result.returncode == 0, result.stderr
    assert "OTPU_HBM_CDC fifos=3 counters=1" in result.stdout
    assert "OTPU_HBM_NO_HCLK_RTL vendor=12" in result.stdout


@pytest.mark.parametrize("bad", [
    "g_sc[8].u_sc/x", "g_sc[10].u_sc/x", "g_sc[00].u_sc/x", "g_sc[7]Xu_sc/x",
    "g_sc[7].u_sc_evil/x", "prefix/g_sc[7].u_sc/x", "g_sc[7].u_sc/", "u_hbm_evil/x",
    "prefix/u_hbm/x", "hbm_reset_n_reg[3]", "hbm_reset_n_reg[0]_replica", "hbm_reset_n_reg_bad",
    "u_sys/u_mem/u_host/hr_v_reg", "u_sys/u_board/state_reg[0]",
])
def test_hclk_gate_rejects_custom_logic_and_near_spoof_names(tmp_path, bad):
    _, _, cl, _, _ = snapshot(tmp_path)
    result = run_hclk_gate(tmp_path, (cl / "build/constraints/cl_timing_user.tcl").read_text(),
                          vendor_cells() + [bad])
    assert result.returncode != 0
    assert "1 openTPU cells on the HBM clock" in result.stderr


@pytest.mark.parametrize("fifos,counters", [(2,1), (4,1), (3,0), (3,2)])
def test_custom_cdc_inventory_is_exact(tmp_path, fifos, counters):
    _, _, cl, _, _ = snapshot(tmp_path)
    result = run_hclk_gate(tmp_path, (cl / "build/constraints/cl_timing_user.tcl").read_text(),
                          vendor_cells(), fifos, counters)
    assert result.returncode != 0
    assert "expected exactly" in result.stderr



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
    assert ".CH_LOG2(29)" in system
    assert ".CH_LOG2(CH_LOG2)" not in system
    assert "TMEM_WORDS" not in system


def test_snapshot_mmcm_is_legal_and_preserves_all_clock_frequencies(tmp_path):
    module, _, cl, _, _ = snapshot(tmp_path)
    frozen = (module.SUPPORT / "boards/f2/design/cl_opentpu_hbm.sv").read_text()
    generated = (cl / "design/cl_opentpu_hbm.sv").read_text()
    module.verify_support()  # Fix applies to generated inputs, never frozen bytes.
    assert ".CLKFBOUT_MULT_F(4.8)" in frozen
    assert ".CLKFBOUT_MULT_F(4.8)" not in generated
    # Restoring the intentional clock and benchmark identity edits must recover
    # every frozen byte, including the complete 450 MHz HBM MMCM and its wiring.
    restored = generated.replace(".DIVCLK_DIVIDE(5), .CLKFBOUT_MULT_F(24.0)",
                                 ".DIVCLK_DIVIDE(1), .CLKFBOUT_MULT_F(4.8)")
    restored = restored.replace("otpu_f2_c2_sys #(", "otpu_f2_hbm_sys #(")
    restored = restored.replace("32'h43324842; // C2HB", "32'h4f545055; // OTPU")
    assert restored == frozen
    params = generated.split("MMCME4_ADV #(", 1)[1].split(") u_mmcm", 1)[0]
    values = {key: Fraction(value) for key, value in
              re.findall(r"\.(\w+)\(([0-9.]+)\)", params)}
    assert values["DIVCLK_DIVIDE"] == 5
    assert values["CLKFBOUT_MULT_F"] == 24
    assert (values["CLKFBOUT_MULT_F"] * 8).denominator == 1
    pfd = 1000 / values["CLKIN1_PERIOD"] / values["DIVCLK_DIVIDE"]
    vco = pfd * values["CLKFBOUT_MULT_F"]
    assert pfd == 50 and vco == 1200
    assert vco / values["CLKOUT0_DIVIDE_F"] == 200
    assert vco / values["CLKOUT1_DIVIDE"] == 400


@pytest.mark.parametrize("old,new,reason", [
    (".DIVCLK_DIVIDE(5), .CLKFBOUT_MULT_F(24.0)",
     ".DIVCLK_DIVIDE(1), .CLKFBOUT_MULT_F(4.8)", "granularity"),
    (".CLKFBOUT_MULT_F(24.0)", ".CLKFBOUT_MULT_F(24.1)", "granularity"),
    (".DIVCLK_DIVIDE(5)", ".DIVCLK_DIVIDE(4)", "must produce 200"),
    (".CLKOUT0_DIVIDE_F(6.0)", ".CLKOUT0_DIVIDE_F(6.1)", "granularity"),
    (".CLKFBOUT_MULT_F(24.0)", ".CLKFBOUT_MULT_F(40.0)", "PFD/VCO"),
    (".CLKFBOUT_MULT_F(9.0)", ".CLKFBOUT_MULT_F(8.0)", "must produce 450"),
])
def test_snapshot_rejects_invalid_clock_recipes(old, new, reason):
    module = builder()
    with pytest.raises(RuntimeError, match=reason):
        module.check_clocks(module.make_top().replace(old, new))
