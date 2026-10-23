"""The F2 build of P2 (targets/f2-p2.toml, docs/v4/p2-f2.md "M5"): the HBM port map of 1 GiB
channel windows, and what boards/f2/build.py's prepare() writes for it (the CL's geometry header,
the floorplan's port variables, the system top's parameters and sources), without Vivado."""
import importlib.util
import re
from pathlib import Path

import pytest

from opentpu import target as T

ROOT = Path(__file__).resolve().parent.parent


def _build():
    spec = importlib.util.spec_from_file_location("f2_build", ROOT / "boards/f2/build.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_targets_lanes_and_lane_memory():
    t = T.load("f2-p2")
    assert t.variant == "c150" and t.core_khz == 150000
    p = T.rtl_params(t)
    assert (p["MXU_C"], p["NCH"], p["CH_LOG2"], p["DRAM_LOG2"], p["PLATFORM"]) == (4, 8, 30, 31, 3)
    src = T.sources(t)
    assert "mem/otpu_lane_mem.sv" in src and "boards/f2/otpu_f2_hbm_sys.sv" in src
    assert src.index("shell/otpu_xr.sv") < src.index("shell/otpu_board.sv")
    for f in src:
        assert (ROOT / "rtl" / f).exists(), f
    assert T.load("f2-p2:c200").core_khz == 200000 and T.config("f2-p2:c200") == T.config(t)
    assert T.load("f2-p2:c162").core_khz == 162500 and T.config("f2-p2:c162") == T.config(t)


def test_core_mmcm_recipes_are_legal():
    """Every core clock of the HBM CL (build.py HBM_CORE_MMCM) from clk_main_a0's 250 MHz: the
    feedback multiplier in the MMCME4's 0.125 steps (4.8 was rounded to 4.75: DRC AVAL-168), the
    PFD and VCO in range (-2: 10-500 MHz, 800-1600 MHz), the core exactly its kHz, the pump 2x."""
    b = _build()
    for khz, (div, mult, d0, d1) in b.HBM_CORE_MMCM.items():
        assert (mult * 8) == int(mult * 8) and (d0 * 8) == int(d0 * 8), khz
        pfd, vco = 250000.0 / div, 250000.0 * mult / div
        assert 10000 <= pfd <= 500000 and 800000 <= vco <= 1600000, khz
        assert vco / d0 == khz and d0 == 2 * d1, khz
    for v in ("c150", "c162", "c175", "c200"):
        assert T.load(f"f2-p2:{v}").core_khz in b.HBM_CORE_MMCM, v


def test_hbm_ports_of_1_gib_windows():
    b = _build()
    t = T.load("f2-p2")
    assert b.hbm_ports(t) == (2, 15)       # channel c on port 2c, the host on 15
    raw = dict(t.platform)
    for bad in ({"hbm_port_stride": 1}, {"hbm_host_port": 4}, {"hbm_host_port": 32}):
        t2 = T.Target(**{**t.__dict__, "platform": {**raw, **bad}})
        with pytest.raises(RuntimeError, match="F2 HBM ports"):
            b.hbm_ports(t2)


def test_prepare_writes_the_geometry_and_the_floorplans_ports(tmp_path):
    b = _build()
    t = T.load("f2-p2")
    sh = tmp_path / "hdk/common/shell_stable"
    (sh / "build/scripts").mkdir(parents=True)
    (sh / "build/scripts/build_all.tcl").write_text("# stub\n")
    wrapper = tmp_path / "cl_mem_hbm_wrapper.sv"
    wrapper.write_text("module cl_mem_hbm_wrapper; endmodule\n")
    cl = b.prepare(tmp_path / "run", sh, t, "0badc0de", wrapper)
    geo = (cl / "design/opentpu_geometry.vh").read_text()
    for want in ("S = 1;", "NCH = 8;", "PSTRIDE = 2;", "HPORT = 15;", "CORE_KHZ_CL = 150000;"):
        assert want in geo, want
    pnr = (cl / "build/constraints/cl_pnr_user.tcl").read_text()
    assert "set otpu_pstride 2\nset otpu_hport 15\n" in pnr
    assert "otpu_hbm_port_x [expr {$otpu_p < $otpu_nport ? $otpu_pstride * $otpu_p : $otpu_hport}]" in pnr
    params = (cl / "design/opentpu_params.vh").read_text()
    for k, v in (("MXU_C", 4), ("NCH", 8), ("CH_LOG2", 30), ("DRAM_LOG2", 31)):
        assert re.search(rf"\.{k}\(32'h{v:x}\)", params), k
    srcs = (cl / "design/sources.tcl").read_text()
    assert "rtl/mem/otpu_lane_mem.sv" in srcs and "rtl/boards/f2/otpu_f2_hbm_sys.sv" in srcs
    cmd = b.hbm_sc_command(cl, t, tmp_path / "sc.log", tmp_path / "hdk")
    assert cmd[-1] == "30" and cmd[cmd.index("-tclargs") + 2] == "8"   # 1 GiB windows, 8 channels
