"""P1 (the wide port, CPC 2: docs/v4/p1-status.md) on the AWS F2 burst200 fabric (docs/v4/p1-status.md section 6):
the target's geometry (four native channels of 1 GiB, each in its own 2 GiB HBM window on HBM AXI
ports 0 / 4 / 8 / 12, two f2_mem groups), the generated CL top and its floorplan, the build plan,
the control block's CPC field (CAPS2 [3:1]) and channels (CFG_DRAM), the host's channel map
against the adapter's, the F2 transport's windows, and on Verilator the F2 memory path in the
loop (per channel otpu_native_axi and an AXI model of the SmartConnect / HBM side): kernels at
CPC 1 (Alex's two channels) and the tiny Gemma 3 on the board model at CPC 2 through the host's
map, bit-exact against the ISA simulator."""
import importlib.util
import os
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opentpu import rtlsim, target as T
from opentpu.host import regs as R
from opentpu.host.board import BEAT, beat_at, join, split

ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "boards/f2/burst200"
TOP = ROOT / "rtl/boards/f2/burst200-p1/cl_opentpu.sv"
FLOORPLAN = FLOW / "constraints/floorplan_p1.tcl"
HDK = Path(os.environ.get("OTPU_F2_HDK", "/home/ubuntu/aws-fpga/hdk"))


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the target and the flow
def test_target_geometry():
    """targets/f2.toml: CPC 2, four native channels of 1 GiB (the slice's 4 GiB), HBM ports
    0 / 4 / 8 / 12 (four distinct HBM channels), the burst200 flow's top and floorplan; the
    control block's parameters follow (otpu_board derives NCH and CH_LOG2 from CPC)."""
    t = T.load("f2")
    p = T.rtl_params(t)
    assert (p["CPC"], p["NCH"], p["CH_LOG2"], p["DRAM_LOG2"]) == (2, 4, 30, 32)
    assert t.platform["hbm_ports"] == [0, 4, 8, 12]
    assert len({q // 2 for q in t.platform["hbm_ports"]}) == 4
    assert t.platform["rtl_board"] == "f2-burst200-p1"
    assert T.sources(t)[-len(T.SHELL):] == T.SHELL       # the core and the shell, no board top
    assert t.core_khz == 200000


def test_build_plan_takes_the_p1_target():
    """build.py's plan for targets/f2.toml: the generated top and Alex's adapters, the P1
    floorplan, the geometry the IP / PCIS / link stages read (one slice, two groups, four
    ports) and the 200 MHz core clock."""
    build = _module(FLOW / "build.py", "otpu_b200_build")
    pl = build.plan(ROOT / "targets/f2.toml")
    assert [p.name for p in pl["rtl"]] == ["cl_opentpu.sv", "otpu_native_axi.sv",
                                           "otpu_hbm_geometry.sv"]
    assert pl["rtl"][0] == TOP and pl["floorplan"] == FLOORPLAN
    assert "set otpu_S 1\nset otpu_G 2\nset otpu_ports {0 4 8 12}\n" in pl["geometry"]
    assert pl["core_mhz"] == 200.0 and "set otpu_core_mhz 200.0" in pl["core_clock"]


def test_ip_and_link_scripts_iterate_the_groups():
    """create_f2_ip.tcl makes a f2_mem_s<g> block design per group and the f2_host crossbar
    over the groups (otpu_G, default otpu_S: the three-slice build's per-slice designs); the
    link stage reads them, the native register slice only at more than one slice."""
    ip = (FLOW / "scripts/create_f2_ip.tcl").read_text()
    assert "if {![info exists otpu_G]} {set otpu_G $otpu_S}" in ip
    assert "for {set s 0} {$s < $otpu_G} {incr s}" in ip and "otpu_host_bd $otpu_G" in ip
    link = (FLOW / "scripts/build_f2.tcl").read_text()
    assert "if {$otpu_G > 1} {" in link and "if {$otpu_S > 1} {" in link


def _p1_top() -> str:
    return TOP.read_text()


def _instances(text: str) -> dict:
    return dict((m.group(2), m.group(1)) for m in re.finditer(
        r"^(\w+)(?: #\((?:.|\n)*?\n\))? (u_\w+) \($", text, re.M))


def test_p1_top_wiring():
    """The checked-in P1 top: one otpu_board at CPC 2 with the target's parameters (sized
    literals from 2^31 up), four otpu_native_axi at windows 0..3 (BASE_ADDR j x 2 GiB), two
    f2_mem groups on ports 0 / 4 and 8 / 12, the host crossbar and a register slice per group,
    SLR0's resets for the board and the memory side, the wrapper on ports 0..12."""
    text = _p1_top()
    inst = _instances(text)
    assert inst["u_board"] == "otpu_board"
    assert [inst[f"u_native{j}"] for j in range(4)] == ["otpu_native_axi"] * 4
    assert inst["u_mem0"] == "f2_mem_s0_wrapper" and inst["u_mem1"] == "f2_mem_s1_wrapper"
    assert inst["u_host"] == "f2_host_wrapper"
    assert inst["u_host_pipe0"] == inst["u_host_pipe1"] == "f2_pcis_pipe"
    assert "u_npipe" not in text                     # one slice: no SLR hops on the native side
    for j in range(4):
        assert f".BASE_ADDR(34'h{j * 0x80000000:x})" in text
    p = T.rtl_params(T.load("f2"))
    assert ".CPC(2)" in text and f".FIFO_DEPTH({p['FIFO_DEPTH']})" in text
    assert f".ARCH_HASH(32'd{p['ARCH_HASH']})" in text or f".ARCH_HASH({p['ARCH_HASH']})" in text
    assert ".BUILD_ID(32'd1345388546)" not in text and ".BUILD_ID(1345388546)" in text
    for big in re.findall(r"\.\w+\((\d+)\)", text):
        assert int(big) < 1 << 31, big                # (Vivado truncates unsized decimals)
    assert "cl_mem_hbm_wrapper #(\n  .NUM_OF_AXI_PORTS(13)" in text
    for port in (0, 4, 8, 12):
        assert f"assign h_arid[{port}] = 6'd0;" in text
    assert "u_rst_core_slr0" in text and "u_rst_main_slr0" in text and "u_rst_core_slr2" not in text
    assert re.search(r"otpu_board #\((?:.|\n)*?\n\) u_board \((?:.|\n)*?\.rst\(core_rst_slr0\)", text)
    for g in (0, 1):
        assert re.search(rf"f2_mem_s{g}_wrapper u_mem{g} \(\n  \.core_clk\(core_clk\),\n  "
                         rf"\.host_clk\(clk_main_a0\),\n  \.hbm_clk\(hbm_clk\),\n  "
                         rf"\.aresetn\(core_rst_n_slr0\)", text)
    assert "calib({4{hbm_ready_core}})" in text and ".ded(4'b0)" in text


@pytest.mark.skipif(not (HDK / "common/shell_stable/design/interfaces/cl_ports.vh").is_file(),
                    reason="the AWS HDK (OTPU_F2_HDK) is not on this host")
def test_p1_top_is_the_generators():
    """rtl/boards/f2/burst200-p1/cl_opentpu.sv is generate_f2_top.py's output for
    targets/f2.toml byte for byte (regenerate it after a change to the board's parameters)."""
    gen = _module(FLOW / "scripts/generate_f2_top.py", "otpu_b200_gen")
    assert gen.generate(T.load("f2"), HDK) == _p1_top()


# Alex's burst200 target (tpubench's targets/f2-burst200.toml, its T2 key left out: no such Config
# field here), for the generator's one-slice, two-channel path
ALEX = """[target]
name = "f2a"
kind = "fpga"
platform_id = 3
description = "AWS F2 VU47P: Small Shell, HBM, preserved openTPU m4 compute"
[arch]
S = 1
D = 128
MCOLS = 4
ACT_BLOCKS = 128
ACT_ROWS = 16
TMEM_WORDS = 65536
IMEM_WORDS = 32768
DRAM_BYTES = 4294967296
LANES = 8
PAIR = true
DSTEP = true
STREAM = true
SEQ_WC = true
CPC = 1
[uarch]
MXU_IMPL = 2
MXU_CL = 16
MXU_E = 4
VPU_CL = 2
ULANES = 8
TMEM_IMPL = 3
TMEM_NSB = 2
TMEM_RPL = true
TMEM_TDP = true
TMEM_GST = true
TMEM_PMP = 1
RPB = 0
WPB = 1
WIN = 16
FIFO_DEPTH = 1024
TRACE_DEPTH = 16384
PQ_WIN = 1024
CHASH = true
[tech]
family = "xilinxus"
part = "xcvu47p-fsvh2892-2-e"
vendor = "xilinx"
[platform]
board = "f2"
rtl_board = "f2-burst200"
core_mhz = 200.0
mem = "hbm"
mem_ctrl = "amd_hbm"
hostcal = false
channels_per_slice = 2
channel_bytes = 2147483648
beat_bytes = 64
host = "pcie"
pcie_gen = 4
pcie_lanes = 8
"""


@pytest.mark.skipif(not (HDK / "common/shell_stable/design/interfaces/cl_ports.vh").is_file(),
                    reason="the AWS HDK (OTPU_F2_HDK) is not on this host")
def test_one_slice_two_channel_wrapper_regenerates_byte_identical(tmp_path):
    """The generator at one slice of two channels (CPC 1: Alex's burst200 image, build 6) still
    writes his checked-in rtl/boards/f2/burst200/cl_opentpu.sv byte for byte: the grouped mode,
    the CPC parameter and the build_id key leave his path alone. (His ARCH_HASH line is the
    card's 4ffd1e17, from a base whose hash left SEQ_WC out; this tree's hash of the same Config
    has it, so that one line is set aside.)"""
    (tmp_path / "f2a.toml").write_text(ALEX)
    gen = _module(FLOW / "scripts/generate_f2_top.py", "otpu_b200_gen1")
    got = gen.generate(T.load(str(tmp_path / "f2a.toml")), HDK)
    want = (ROOT / "rtl/boards/f2/burst200/cl_opentpu.sv").read_text()
    strip = lambda s: re.sub(r"\.ARCH_HASH\([^)]*\)", ".ARCH_HASH()", s)
    assert strip(got) == strip(want)


def test_floorplan_names_every_cell_of_the_top():
    """floorplan_p1.tcl names only instances the P1 top has, and places every SLR0 piece (the
    board, the adapters, the host's SLR0 register slices, the wrapper, the SLR0 resets), each
    SmartConnect in its grandchild by its ports, and the host crossbar in SLR1."""
    fp = FLOORPLAN.read_text()
    inst = set(_instances(_p1_top()))
    named = set(re.findall(r"\$cl/(u_\w+)(?![\w$])", fp))     # (u_mem$g: below)
    assert named <= inst, named - inst
    for name in ("u_board", "u_native0", "u_native1", "u_native2", "u_native3", "u_host_pipe0",
                 "u_host_pipe1", "u_hbm", "u_rst_core_slr0", "u_rstpipe_core_slr0",
                 "u_rst_main_slr0", "u_rstpipe_main_slr0", "u_pcis_pipe", "u_ocl_cdc",
                 "u_status_cdc", "u_host"):
        assert f"$cl/{name}" in fp, name
    assert "0 {CLOCKREGION_X0Y0:CLOCKREGION_X1Y1}" in fp and "1 {CLOCKREGION_X2Y0:CLOCKREGION_X3Y1}" in fp
    assert "add_cells_to_pblock pblock_CL_SLR0_MEM$g [otpu_p1_cells [list $cl/u_mem$g]]" in fp
    assert "\\[" not in fp                            # (get_cells takes [] literally)


# ------------------------------------------------------------------ the control block, the host
def test_caps2_reports_cpc():
    """CAPS2 [3:1] is CPC (0 at one chunk a cycle, the same field as the wide-port line's
    PORTB); regs.caps2 reads it, a map before 5 is one chunk."""
    ctrl = (ROOT / "rtl/shell/otpu_ctrl.sv").read_text()
    assert "CAPS2 = {28'd0, 3'((CPC > 1) ? CPC : 0)," in ctrl
    assert R.caps2(None) == {"gather": True, "cpc": 1}
    assert R.caps2(1) == {"gather": True, "cpc": 1}
    assert R.caps2(1 | 2 << 1) == {"gather": True, "cpc": 2}
    board = (ROOT / "rtl/shell/otpu_board.sv").read_text()
    assert ".DRAM_LOG2(32), .NCH(NCH), .CH_LOG2(CHL)" in board and ".CPC(CPC)" in board


def _rtl_channel(addr: int, nch: int) -> tuple[int, int]:
    """otpu_native_dram's chan_of / cbeat_of (rtl/mem/otpu_native_dram.sv) on a byte address:
    the channel {chunk index bit 0 at CPC 2, half ^ parity(chunk)} and the channel offset."""
    w = addr // 4
    h = (w >> 4 & 1) ^ (bin(w >> 5).count("1") & 1)
    if nch == 4:
        return (w >> 5 & 1) << 1 | h, (w >> 6 & ((1 << 24) - 1)) * BEAT + addr % BEAT
    return h, (w >> 5) * BEAT + addr % BEAT


@pytest.mark.parametrize("nch", [2, 4])
def test_host_channel_map_is_the_adapters(nch):
    """board.beat_at is otpu_native_dram's map at 2 CPC channels (chunk m on pair m % CPC, no
    pair hash; the halves by the chunk's parity), split / join invert each other over groups of
    CPC chunks, and at CPC 2 every pair of consecutive chunks (a wide read at any alignment)
    is one beat on each of the four channels."""
    rng = np.random.default_rng(nch)
    for a in [0, 64, 128, 192, 256, 4096 + 64, (1 << 30) - 64, (1 << 32) - 64,
              *map(int, rng.integers(0, 1 << 32, 2000))]:
        assert beat_at(a, True, nch) == _rtl_channel(a, nch), hex(a)
    for _ in range(20):
        a0 = int(rng.integers(0, 1 << 20)) * nch * BEAT
        data = rng.integers(0, 256, nch * BEAT * int(rng.integers(1, 50)), dtype=np.uint8)
        parts = split(a0, data, True, nch)
        assert [c for c, _, _ in parts] == list(range(nch))
        assert all(off == a0 // nch for _, off, _ in parts)
        for c, off, part in parts:              # each piece where beat_at puts its bytes
            for k in range(0, len(part), BEAT * 7):
                x = a0 + k // BEAT * nch * BEAT  # (a sample: group k / BEAT, the channel's beat)
                hit = [y for y in range(x, x + nch * BEAT, BEAT) if beat_at(y, True, nch)[0] == c]
                assert len(hit) == 1 and beat_at(hit[0], True, nch)[1] == off + k
                assert np.array_equal(part[k:k + BEAT], data[hit[0] - a0:hit[0] - a0 + BEAT])
        assert np.array_equal(join([p for _, _, p in parts], a0, True), data)
    if nch == 4:
        for m in range(0, 4096):
            chans = {beat_at((m + j) * 2 * BEAT + b * BEAT, True, 4)[0] for j in (0, 1) for b in (0, 1)}
            assert chans == {0, 1, 2, 3}, m


def test_f2_windows():
    """The F2 transport's geometry (CFG_DRAM, CFG_SLICES): P1's four channels of 1 GiB at BAR4
    0 / 2 / 4 / 6 GiB, Alex's two of 2 GiB at 0 / 2 GiB, three slices of two at 2 GiB apart;
    others refused."""
    from opentpu.host.f2 import windows
    cfg = lambda dram, nch, ch: dram | nch << 8 | ch << 16
    assert windows(cfg(32, 4, 30), 1) == (1 << 30, (0, 2 << 30, 4 << 30, 6 << 30))
    assert windows(cfg(32, 2, 31), 1) == (1 << 31, (0, 2 << 30))
    assert windows(cfg(32, 2, 31), 3)[1] == tuple(j << 31 for j in range(6))
    for bad in [(cfg(32, 4, 31), 1), (cfg(32, 8, 29), 1), (cfg(33, 4, 30), 1), (cfg(32, 4, 30), 3)]:
        with pytest.raises(RuntimeError):
            windows(*bad)


def test_device_config_takes_cpc_from_the_bitstream(monkeypatch):
    """device_config reads CPC from CAPS2 [3:1] (OTPU_CPC, when set, must agree) and checks
    CFG_HASH against the reported Config, CPC in it: a CPC 2 bitstream gives a CPC 2 Config,
    and a host's CPC 1 hash of the same registers is refused."""
    from opentpu.host import board as B
    t = T.load("f2")
    cfg = T.config(t)
    info = {"D": cfg.D, "MCOLS": cfg.MCOLS, "LANES": cfg.LANES, "act_rows": cfg.act_rows,
            "caps": {"pair": True, "dstep": True, "stream": True, "mx": True},
            "caps2": {"gather": True, "cpc": 2},
            "cfg": {"S": 1, "TMEM_WORDS": cfg.TMEM_WORDS, "IMEM_WORDS": cfg.IMEM_WORDS,
                    "ACT_BLOCKS": cfg.ACT_BLOCKS, "DRAM_BYTES": cfg.DRAM_BYTES, "channels": 4,
                    "channel_bytes": 1 << 30, "SEQ_WC": int(cfg.SEQ_WC), "platform": 3,
                    "hash": T.arch_hash(cfg)}}
    monkeypatch.delenv("OTPU_CPC", raising=False)
    monkeypatch.setattr(B, "check_numerics", lambda i: None)
    got = B.device_config(info)
    assert got.CPC == 2 and T.arch_hash(got) == T.arch_hash(cfg)
    monkeypatch.setenv("OTPU_CPC", "1")
    with pytest.raises(B.ConfigMismatch):
        B.device_config(info)
    monkeypatch.delenv("OTPU_CPC")
    one = dict(info, caps2={"gather": True, "cpc": 1})
    with pytest.raises(B.ConfigMismatch):
        B.device_config(one)


# ------------------------------------------------------------------ Verilator: the F2 path
def test_mlp_on_the_f2_path_at_cpc1(have_verilator):
    """Alex's geometry (one chunk a cycle, two channels) with the F2 path in the loop (per
    channel otpu_native_axi and the AXI side's model, random stalls and bubbles, the boot
    loader): the MLP decode kernel equals the ISA simulator, DRAM and TMEM."""
    from conftest import assert_same_state, rel
    from opentpu.isasim import board_config
    from opentpu.kernels import mlp
    from opentpu.runtime import collect, compile_kernel, run
    from test_kernels import mlp_args
    cfg = board_config(CPC=1, DRAM_BYTES=1 << 24)
    args, want = mlp_args(np.random.default_rng(4), M=1, H=1152, Fd=2304)
    comp, imgs = compile_kernel(mlp, cfg, **args)
    ri = run(comp, [i.copy() for i in imgs], "isa")
    drams, tmems, st = rtlsim.run(cfg, comp.programs, [i.copy() for i in imgs],
                                  uarch=dict(rtlsim.BOARD_UARCH), axi=3, boot=True, stall=20)
    rr = collect(comp, drams, tmems, st)
    assert_same_state(ri, rr)
    assert rel(rr.outputs["out"], want) < 0.03


def _board_cfg(**kw):
    from conftest import sim_board_config
    return sim_board_config(CPC=2, DRAM_BYTES=1 << 24, **kw)


def test_board_model_reports_cpc2(have_verilator, monkeypatch):
    """otpu_board at CPC 2 under tb_board with the F2 path (MEM_NATIVE 3), as the target builds
    it: CAPS2 says two chunks a cycle, CFG_DRAM four channels of 1 GiB, CFG_HASH the f2
    target's Config (CPC in it), and device_config gives that Config."""
    from opentpu.host.board import Board, SimTransport, device_config
    monkeypatch.setenv("OTPU_CPC", "2")
    tr = SimTransport(ch_bytes=1 << 22, stall=10, seed=3, target="f2", native="f2")
    assert tr.nch == 4 and len(tr.ch) == 4
    b = Board(tr, lock=False)
    i = b.info()
    assert i["caps2"]["cpc"] == 2
    assert (i["cfg"]["channels"], i["cfg"]["channel_bytes"], i["cfg"]["DRAM_BYTES"]) == \
        (4, 1 << 30, 1 << 32)
    assert i["cfg"]["platform"] == 3
    cfg = device_config(i)
    assert cfg.CPC == 2 and b.nch == 4


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_tiny_gemma3_on_the_board_model_f2_path(have_verilator, monkeypatch, tmp_path):
    """The board as the F2 image builds it (otpu_board at CPC 2, its four channels each through
    otpu_native_axi and the AXI side's model with random stalls, tb_board MEM_NATIVE 3), the
    host writing and reading the DRAM through its CPC 2 channel map (Board.write / read, the
    BoardBackend as on the card): the odd-block tiny Gemma 3 (KB 3 / 5 rows: half pairs) decodes
    three tokens with logits equal to the ISA simulator's bit for bit, and the DRAM image after
    them is the ISA simulator's."""
    from opentpu.host.board import BoardBackend, SimTransport
    from opentpu.llm.qwen3 import Engine
    from test_gemma3 import _model
    monkeypatch.setenv("OTPU_CPC", "2")
    _, W, spec = _model(tmp_path, layers=2, hidden=384, inter=640)
    cfg = _board_cfg()
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 4, stall=10, seed=7, target="f2", native="f2")
    brd = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd.backend.info["caps2"]["cpc"] == 2 and brd.backend.info["cfg"]["channels"] == 4
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
        st = brd.stats[-1]
        assert st["cycles"] > 0 and st["instructions"] == isa.stats[-1]["instructions"]
        print(f"token {tok}: {st['cycles']} cycles, {st['instructions']} instructions")
    n = isa.image.nbytes
    assert np.array_equal(brd.backend.read(0, 0, n), isa.backend.machine.slices[0].dram[:n])
