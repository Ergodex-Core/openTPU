"""P4 on F2: four slices on Alex's burst200 fabric (targets/f2-burst200-s4*.toml,
docs/v4/p4-status.md "On the F2"): the generator's four-slice tops (two slices in SLR2), the
checked-in tops are its output, the floorplan names every cell of the four-slice top, the build
driver takes the targets, the top lints under Verilator against stubs of the vendor IP, and
Gemma 3 runs bit-exactly at S = 4: the tiny model on the four-slice board model (otpu_board
S = 4 under tb_board) and Gemma 3 1B's token on the four-slice RTL (otpu_top S = 4)."""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

from opentpu import target as T

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / 'boards/f2/burst200'
S3 = ROOT / 'targets/f2-burst200-s3.toml'
TOP3 = ROOT / 'rtl/boards/f2/burst200-s3/cl_opentpu.sv'
S4 = ROOT / 'targets/f2-burst200-s4.toml'
TOP4 = ROOT / 'rtl/boards/f2/burst200-s4/cl_opentpu.sv'
# the build variants of the four-slice target: toml, top, CORE_KHZ, core MHz
S4_SLOWER = [(ROOT / 'targets/f2-burst200-s4-175.toml',
              ROOT / 'rtl/boards/f2/burst200-s4-175/cl_opentpu.sv', 175000, 175.0),
             (ROOT / 'targets/f2-burst200-s4-162.toml',
              ROOT / 'rtl/boards/f2/burst200-s4-162/cl_opentpu.sv', 162500, 162.5)]
PORTS4 = [0, 4, 8, 12, 16, 20, 26, 28]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gen = _module('generate_f2_top', BOARD / 'scripts/generate_f2_top.py')
flow = _module('burst200_build', BOARD / 'build.py')


def hdk() -> Path:
    from test_f2_burst200_s3 import hdk as h
    return h()


def _instances(text: str) -> dict:
    """{instance: module} of a generated top (every instantiation ends in `) name (`)."""
    return {n: m for m, n in re.findall(r'^(\w+)(?: #\([^;]*?\))? (u_\w+) \(', text, re.M | re.S)}


def _connections(text: str, inst: str) -> dict:
    """{port: net} of one instance of a generated top."""
    body = re.search(rf'\) {inst} \(\n(.*?)\n\);', text, re.S) or \
        re.search(rf'^\w+ {inst} \(\n(.*?)\n\);', text, re.S | re.M)
    return dict(re.findall(r'\.(\w+)\(([^)]*)\)', body.group(1)))


def test_target_geometry():
    """The four-slice targets: f2-burst200-s3's arch and uarch at S = 4, two 2 GiB channels a
    slice on eight distinct HBM channels' ports (slice 3's in X6Y0), slices 2 and 3 in SLR2, the
    s4 floorplan; the build variants differ from the 200 MHz target in the core clock alone."""
    t3, t4 = T.load(str(S3)), T.load(str(S4))
    assert t4.arch == dict(t3.arch, S=4) and t4.uarch == t3.uarch and t4.tech == t3.tech
    assert t4.platform_id == t3.platform_id == 3
    assert T.sources(t4) == T.sources(t3)
    p = T.rtl_params(t4)
    assert (p['S'], p['NCH'], p['CH_LOG2'], p['DRAM_LOG2']) == (4, 2, 31, 32)
    assert gen.geometry(t4) == (4, 2, PORTS4, 29)
    assert gen.slice_slrs(t4) == [0, 1, 2, 2] and gen.slice_slrs(t3) == [0, 1, 2]
    assert gen.host_xbar_slr(t4) == 0 and gen.host_xbar_slr(t3) == 1
    # every window on its own memory channel (ports 2k / 2k + 1 share one), all 16 GiB used
    assert len({q // 2 for q in PORTS4}) == 8 and t4.arch['DRAM_BYTES'] * 4 == 1 << 34
    pl = flow.plan(S4)
    assert pl['S'] == 4 and pl['floorplan'].name == 'floorplan_s4.tcl'
    assert [q.name for q in pl['rtl']] == ['cl_opentpu.sv', 'otpu_native_axi.sv',
                                           'otpu_hbm_geometry.sv']
    assert pl['geometry'].splitlines()[1:] == ['set otpu_S 4',
                                               'set otpu_ports {0 4 8 12 16 20 26 28}']
    for toml, top, khz, mhz in S4_SLOWER:
        t = T.load(str(toml))
        assert t.core_khz == khz and t4.core_khz == 200000
        assert T.arch_hash(T.config(t)) == T.arch_hash(T.config(t4))
        assert t.platform == dict(t4.platform, rtl_board=toml.stem, core_mhz=mhz)
        pl = flow.plan(toml)
        assert pl['core_mhz'] == mhz and pl['S'] == 4 and any(q == top for q in pl['rtl'])


def test_generator_keeps_the_one_and_three_slice_tops():
    """The slice-to-SLR map changes nothing at S = 1 and S = 3 (slice s in SLR s)."""
    assert gen.generate(T.load(str(S3)), hdk()) == TOP3.read_text()
    assert gen.generate(T.load(str(ROOT / 'targets/f2-burst200.toml')), hdk()) == \
        (ROOT / 'rtl/boards/f2/burst200/cl_opentpu.sv').read_text()


@pytest.mark.parametrize('toml,top', [(S4, TOP4)] + [(a, b) for a, b, _, _ in S4_SLOWER])
def test_four_slice_wrappers_are_the_generators(toml, top):
    """The checked-in four-slice tops are the generator's output for their targets."""
    assert gen.generate(T.load(str(toml)), hdk()) == top.read_text()


def test_build_variants_differ_in_the_core_clock_alone():
    """Each variant's top is the 200 MHz s4 top with its CORE_KHZ and nothing else."""
    a = TOP4.read_text().splitlines()
    for _, top, khz, _ in S4_SLOWER:
        b = top.read_text().splitlines()
        assert len(a) == len(b)
        assert [(x, y) for x, y in zip(a, b) if x != y] == \
            [('  .CORE_KHZ(200000),', f'  .CORE_KHZ({khz}),')]


def test_four_slice_top_wiring():
    """What the four-slice top instantiates: eight adapters at the windows' bases, four memory
    block designs and the host crossbar, a register slice per SLR hop (slices 2 and 3 two hops
    each from SLR2), the wrapper covering port 28, per-SLR resets (SLR0 and SLR2; SLR1 the
    shell's), each adapter on its SLR's reset, an 11-bit status (eight channel errors, three
    LEDs), and the host crossbar in SLR0 behind one register slice at the crossing."""
    text = TOP4.read_text()
    inst = _instances(text)
    assert [inst[f'u_native{j}'] for j in range(8)] == ['otpu_native_axi'] * 8
    for j in range(8):
        assert f"34'h{j * 0x80000000:x}" in text
    assert {inst[f'u_mem{s}'] for s in range(4)} == {f'f2_mem_s{s}_wrapper' for s in range(4)}
    assert inst['u_host'] == 'f2_host_wrapper' and inst['u_pcis_pipe'] == 'f2_pcis_pipe'
    assert {k for k, m in inst.items() if m == 'f2_pcis_pipe'} == {
        'u_pcis_pipe', 'u_pcis_hop', 'u_host_pipe0', 'u_host_pipe1', 'u_host_pipe2',
        'u_host_pipe3'}
    # the PCIS pipe (SLR1) -> the hop (SLR0) -> the crossbar (SLR0) -> a pipe per slice
    hop, xbar = _connections(text, 'u_pcis_hop'), _connections(text, 'u_host')
    assert hop['s_axi_awaddr'] == 'pcis_awaddr' and hop['s_axi_awid'] == 'pcis_awid'
    assert hop['aresetn'] == 'rst_main_n_slr0' and hop['aclk'] == 'clk_main_a0'
    assert xbar['S_PCIS_awaddr'] == 'ph_awaddr' and xbar['aresetn'] == 'rst_main_n_slr0'
    assert {k for k, m in inst.items() if m == 'f2_native_pipe'} == {
        'u_npipe2_slr0', 'u_npipe3_slr0',
        'u_npipe4_slr1', 'u_npipe4_slr0', 'u_npipe5_slr1', 'u_npipe5_slr0',
        'u_npipe6_slr1', 'u_npipe6_slr0', 'u_npipe7_slr1', 'u_npipe7_slr0'}
    assert '.NUM_OF_AXI_PORTS(29)' in text and '.S(4)' in text
    assert '.calib({8{hbm_ready_core}})' in text and ".ded(8'b0)" in text
    assert {k for k, m in inst.items() if m == 'lib_pipe'} == {
        'u_rstpipe_core_slr0', 'u_rstpipe_core_slr2', 'u_rstpipe_main_slr0'}
    for j, rst in enumerate(['core_rst_slr0'] * 2 + ['core_rst'] * 2 + ['core_rst_slr2'] * 4):
        assert _connections(text, f'u_native{j}')['rst'] == rst, j
    assert '.WIDTH(11)' in text and "{24'd0,shell_status[10:3]}" in text
    for port in PORTS4:
        assert f"assign h_arid[{port}] = 6'd0;" in text
        assert f"assign h_arvalid[{port}] = '0;" not in text
    for port in (1, 2, 24, 25, 27):
        assert f"assign h_arvalid[{port}] = '0;" in text
    # slice 3's SmartConnect on ports 26 / 28, its masters two hops from SLR2
    c = _connections(text, 'u_mem3')
    assert c['S_N0_awaddr'] == 'np6_slr0_awaddr' and c['S_N1_awaddr'] == 'np7_slr0_awaddr'
    assert c['M_H26_araddr'] == 'h_araddr[26]' and c['M_H28_araddr'] == 'h_araddr[28]'
    assert c['S_HOST_awaddr'] == 'hp3_awaddr' and c['aresetn'] == 'core_rst_n_slr0'
    assert _connections(text, 'u_npipe6_slr1')['s_axi_awaddr'] == 'n6_awaddr'
    assert _connections(text, 'u_npipe6_slr1')['aresetn'] == 'core_rst_n'
    assert _connections(text, 'u_npipe6_slr0')['s_axi_awaddr'] == 'np6_slr1_awaddr'
    assert _connections(text, 'u_npipe6_slr0')['aresetn'] == 'core_rst_n_slr0'
    # the host crossbar's master ports carry no IDs (SmartConnect's downstream ports)
    for s in range(4):
        for k in ('arid', 'awid', 'rid', 'bid'):
            assert f'.M_S{s}_{k}(' not in text and f'hx{s}_{k}' not in text
    assert text.count(".s_axi_awid(16'd0)") == 4 and text.count(".s_axi_arid(16'd0)") == 4
    assert text.count('.s_axi_rid()') == 4 + 6 and text.count('.s_axi_bid()') == 4 + 6


@pytest.mark.parametrize('top', [TOP4] + [b for _, b, _, _ in S4_SLOWER])
def test_no_unsized_literal_from_2_31_up(top):
    """Vivado reads an unsized decimal literal as a signed 32-bit integer (Synth 8-11057): no
    parameter value of 2^31 or more is unsized (the S = 4 ARCH_HASH, 0x2f01ab9b, is below it,
    so the top has no sized literal; the S = 3 hash was above)."""
    text = top.read_text()
    big = [m for m in re.findall(r'\((\d+)\)', text) if int(m) >= 1 << 31]
    assert not big, big
    h = T.rtl_params(T.load(str(S4)))['ARCH_HASH']
    assert re.findall(r"32'd(\d+)", text) == ([str(h)] if h >= 1 << 31 else [])
    assert f'.ARCH_HASH({h})' in text or f".ARCH_HASH(32'd{h})" in text


def test_floorplan_names_every_cell_of_the_top():
    """floorplan_s4.tcl places every top-level instance that is not a clock or CDC primitive
    and names no instance the top lacks; slices 2 and 3 in the halves of SLR2; every
    SmartConnect beside its ports; the crossing registers of slices 0, 2 and 3 are marked."""
    fp = (BOARD / 'constraints/floorplan_s4.tcl').read_text()
    inst = _instances(TOP4.read_text())
    placed = set(re.findall(r'\$cl/(u_\w+)(?=[\s\]])', fp))
    placed |= {'u_board'} if '$cl/u_board/' in fp else set()   # placed by its pieces
    assert '[list $cl/u_mem$s]' in fp                            # the SmartConnect loop
    placed |= {f'u_mem{s}' for s in range(4)}
    loop = re.search(r'foreach \{s n0 n1\} \{([\d ]+)\}', fp).group(1).split()
    assert loop == ['2', '4', '5', '3', '6', '7'] and '$cl/u_native$n0 $cl/u_native$n1' in fp
    placed |= {f'u_native{j}' for j in range(4, 8)}             # the SLR2 halves' loop
    free = {'u_core_clk', 'u_hbm_clk', 'u_rst_main', 'u_rst_core', 'u_rst_hbm', 'u_ready_cdc'} | {
        f'u_geometry{j}' for j in range(29)}
    assert placed - set(inst) == set(), placed - set(inst)
    assert set(inst) - placed - free == set(), set(inst) - placed - free
    assert 'g_slice\\[1\\].u_slice' in fp and '$cl/u_board/u_slice' in fp
    assert 'g_slice\\[$s\\].u_slice' in fp and 'g_slice\\[$s\\].u_mem' in fp
    for s in range(4):
        for st in ('u_dn_rx', 'u_up_tx', 'u_cport') if s != 1 else ():
            assert f'g_x\\[{s}\\].{st}' in fp or (s > 1 and f'g_x\\[$s\\].{st}' in fp)
        assert f'g_x\\[{s}\\].u_dn_tx' in fp and f'g_x\\[{s}\\].u_up_rx' in fp
    halves = re.search(r'foreach \{s ranges\} \{\n  2 \{(\S+)\}\n  3 \{(\S+)\}\n\}', fp)
    assert halves.groups() == ('CLOCKREGION_X0Y8:CLOCKREGION_X3Y11',
                               'CLOCKREGION_X4Y8:CLOCKREGION_X7Y11')
    assert 'set_property parent pblock_CL_SLR2 [get_pblocks pblock_CL_SLR2_S$s]' in fp
    assert 'IS_SOFT false [get_pblocks pblock_CL_SLR2_S$s]' in fp
    mem = re.search(r'foreach \{s ranges\} \{\n  0 (.*?)\n\} \{', fp, re.S).group(0)
    assert '0 {CLOCKREGION_X0Y0:CLOCKREGION_X1Y1}' in mem
    assert '1 {CLOCKREGION_X2Y0:CLOCKREGION_X3Y1}' in mem
    assert '2 {CLOCKREGION_X4Y0:CLOCKREGION_X5Y0 CLOCKREGION_X5Y1:CLOCKREGION_X5Y1}' in mem
    assert '3 {CLOCKREGION_X6Y0:CLOCKREGION_X7Y0 CLOCKREGION_X6Y1:CLOCKREGION_X6Y1}' in mem
    assert 'IS_SOFT false [get_pblocks pblock_CL_SLR0_MEM$s]' in fp
    assert 'pblock_CL_SLR1_XBAR' not in fp                     # the crossbar is SLR0's
    slr0 = re.search(r'add_cells_to_pblock pblock_CL_SLR0 \[otpu_s4_cells \[list (.*?)\]\]',
                     fp, re.S).group(1)
    assert '$cl/u_pcis_hop $cl/u_host ' in slr0
    assert 'USER_SLL_REG TRUE' in fp and 'foreach s {0 2 3}' in fp
    # the SLR0 parent's ranges are floorplan_s3.tcl's (the legal CL ranges)
    s3 = (BOARD / 'constraints/floorplan_s3.tcl').read_text()
    rng = lambda t, p: re.search(rf'resize_pblock {p} -add \{{(.*?)\}}', t, re.S).group(1)
    for p in ('pblock_CL_SLR0', 'pblock_CL_SLR1', 'pblock_CL_SLR2'):
        assert rng(fp, p) == rng(s3, p), p


def test_build_plan_takes_the_targets(tmp_path):
    for toml, mhz in [(S4, 200.0)] + [(a, m) for a, _, _, m in S4_SLOWER]:
        out = subprocess.run([sys.executable, str(BOARD / 'build.py'), '--hdk', '/nonexistent',
                              '--target', str(toml), '--plan'], text=True, capture_output=True,
                             cwd=tmp_path)
        assert out.returncode == 0, out.stderr
        assert '4 slice(s), HBM ports [0, 4, 8, 12, 16, 20, 26, 28]' in out.stdout
        assert f'{toml.stem[3:]}/cl_opentpu.sv' in out.stdout and 'floorplan_s4.tcl' in out.stdout
        assert f'Core clock: {mhz} MHz (TMEM {2 * mhz} MHz)' in out.stdout
        assert out.stdout.strip().endswith('7. audit')


def test_generated_top_lints_under_verilator(tmp_path):
    """The four-slice top elaborates under Verilator with the openTPU RTL, the HDK's interface
    headers and lib_pipe, and stubs of the vendor IP (the S = 3 test's): no error, no width
    truncation in the wiring."""
    if shutil.which('verilator') is None:
        pytest.skip('verilator not installed')
    from test_f2_burst200_s3 import _sh_ddr_stub, _stubs
    h = hdk()
    t = T.load(str(S4))
    S, _, ports, _ = gen.geometry(t)
    (tmp_path / 'stubs.sv').write_text(_stubs(S, ports) + _sh_ddr_stub(h))
    iface = h / 'common/shell_stable/design/interfaces'
    srcs = [h / 'common/lib/interfaces.sv', h / 'common/lib/lib_pipe.sv'] + \
        [ROOT / 'rtl' / f for f in T.sources(t, 'generic')] + \
        [ROOT / 'rtl/boards/f2/burst200/otpu_native_axi.sv',
         ROOT / 'rtl/boards/f2/burst200/otpu_hbm_geometry.sv', tmp_path / 'stubs.sv', TOP4]
    cmd = ['verilator', '--lint-only', '-Wno-fatal', '-Wno-DECLFILENAME', '-Wno-UNUSEDSIGNAL',
           '-Wno-UNUSEDPARAM', '-Wno-PINCONNECTEMPTY', '-Wno-UNOPTFLAT', '-Wno-MULTIDRIVEN',
           '-Wno-WIDTHEXPAND', '-Wno-ASCRANGE', '-Wno-CASEINCOMPLETE', '-Wno-TIMESCALEMOD',
           '-Wno-SYMRSVDWORD', '-Wno-IMPLICITSTATIC', '-Wno-CASEOVERLAP', '-Wno-INITIALDLY',
           '-Wno-LATCH', '-Wno-UNDRIVEN', '-Wno-BLKSEQ', '-Wno-COMBDLY', '-Wno-SYNCASYNCNET',
           f'+incdir+{iface}', '--top-module', 'cl_opentpu', '-DSYNTHESIS'] + \
        [str(s) for s in srcs]
    out = subprocess.run(cmd, text=True, capture_output=True, cwd=tmp_path)
    msgs = [l for l in out.stderr.splitlines() if l.startswith('%')]
    errors = [l for l in msgs if l.startswith('%Error')]
    assert not errors, '\n'.join(msgs[:40])
    own = [l for l in msgs if 'cl_opentpu.sv' in l and 'WIDTHTRUNC' in l]
    assert not own, '\n'.join(own)


# ---- Gemma 3 at S = 4 on the RTL
torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')


def test_tiny_gemma3_four_slices_on_board_model(have_verilator, tmp_path):
    """otpu_board at S = 4 (four slices, each on its two native channels behind the native
    memory model, one control block, the collective hub and a port per slice: tb_board with the
    f2-burst200-s4 target, platform 3) runs the tiny Gemma 3 split over four slices evenly:
    logits bit-identical to the ISA simulator at S = 4, each slice's DRAM the ISA simulator's."""
    from conftest import sim_board_config
    from opentpu.host.board import BoardBackend, SimTransport
    from opentpu.llm.qwen3 import Engine
    from test_gemma3 import _model
    _, W, spec = _model(tmp_path, layers=2, hidden=256, ffn=512)
    cfg = sim_board_config(DRAM_BYTES=1 << 24, S=4)
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=0, seed=5, target='f2-burst200-s4',
                      native=True)
    brd = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd.backend.info['cfg']['S'] == 4 and brd.backend.info['cfg']['platform'] == 3
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
        st = brd.stats[-1]
        assert st['cycles'] > 0 and st['instructions'] == isa.stats[-1]['instructions']
        print(f"token {tok}: {st['cycles']} cycles, {st['instructions']} instructions a slice")
    n = isa.image.nbytes
    for s in range(4):
        assert np.array_equal(brd.backend.read(s, 0, n), isa.backend.machine.slices[s].dram[:n])


REAL = ROOT / 'models/gemma-3-1b-it'


@pytest.mark.skipif(not REAL.exists(), reason='models/gemma-3-1b-it not downloaded')
def test_gemma3_1b_token_on_four_slice_rtl_is_bit_exact(have_verilator):
    """The one-token gate of the real model at S = 4 (the S = 3 gate's pattern): the chat
    prompt's tokens but the last on the ISA simulator, then the last step on the Verilator RTL
    (otpu_top S = 4, the native memory model never withholding a handshake, 30 cycles of
    latency: the HBM-like reference) and on the ISA simulator from the same DRAM state: logits
    and every slice's DRAM equal bit for bit. Slow (about a quarter of an hour)."""
    from opentpu import rtlsim
    from opentpu.host.board import sim_config
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import Engine, load_weights
    from opentpu.llm.rtl_backend import RtlBackend
    spec = load_spec(REAL)
    cfg = sim_config(spec, 256, board_config(PAIR=True, DSTEP=True, STREAM=True, S=4),
                     lookup=True, wformat='int8')
    assert cfg.S == 4
    eng = Engine(spec, load_weights(REAL), cap=256, cfg=cfg, wformat='int8', resident=True)
    tok = transformers.AutoTokenizer.from_pretrained(REAL)
    ids = tok.apply_chat_template([{'role': 'user', 'content': 'Why is the sky blue?'}],
                                  add_generation_prompt=True, tokenize=True)
    ids = list(ids['input_ids'] if hasattr(ids, 'keys') else ids)
    for t in ids[:-1]:
        eng.step(t)
    n = eng.image.nbytes
    rtlsim.MEMORY['LAT'] = 30
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in eng.backend.machine.slices],
                     uarch=dict(rtlsim.BOARD_UARCH), axi=True, stall=0)
    isa = eng.backend
    want = eng.step(ids[-1])
    eng.backend, eng.pos = rtl, eng.pos - 1
    got = eng.step(ids[-1])
    st = eng.stats[-1]
    print(f"S = 4 RTL token at pos {len(ids) - 1}: {st['cycles']} cycles, next token "
          f"{int(np.argmax(got))} ({tok.decode([int(np.argmax(got))])!r})")
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))
    assert int(np.argmax(want)) == int(np.argmax(got))
    for s in range(4):
        assert np.array_equal(isa.machine.slices[s].dram[:n], rtl.drams[s][:n])


def test_tmem_floorplan_is_s4s_with_a_box_per_slice():
    """floorplan_s4_tmem.tcl (the 175 MHz variant, build.py --floorplan): floorplan_s4.tcl as it
    is, then one hard box per slice for its pumped TMEM, DMA and collective port inside the slice's
    own pblock (SLR0 above the SmartConnect rows, SLR1, SLR2's halves), each with the block RAM
    for the TMEM's 64 RAMB36 and the DMA's 14 at under 60%."""
    base = (BOARD / 'constraints/floorplan_s4.tcl').read_text()
    tm = (BOARD / 'constraints/floorplan_s4_tmem.tcl').read_text()
    first, rest = tm.split('\n', 1)
    assert first.startswith('# floorplan_s4_tmem.tcl') and rest.startswith(base)
    boxes = re.findall(r'^  (\d) (pblock_CL_\w+) +\{CLOCKREGION_X(\d)Y(\d+):CLOCKREGION_X(\d)Y(\d+)\}',
                       rest[len(base):], re.M)
    assert [b[:2] for b in boxes] == [('0', 'pblock_CL_SLR0'), ('1', 'pblock_CL_SLR1'),
                                      ('2', 'pblock_CL_SLR2_S2'), ('3', 'pblock_CL_SLR2_S3')]
    inside = {'pblock_CL_SLR0': (0, 3, 2, 3),         # X0Y0:X3Y3 less the SmartConnect rows
              'pblock_CL_SLR1': (0, 3, 4, 7), 'pblock_CL_SLR2_S2': (0, 3, 8, 11),
              'pblock_CL_SLR2_S3': (4, 7, 8, 11)}
    ramb36 = [24, 24, 36, 12, 24, 12, 12, 24]            # RAMB36 sites a clock region, X0-X7
    for s, parent, x0, y0, x1, y1 in boxes:
        X0, X1, Y0, Y1 = inside[parent]
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        assert X0 <= x0 <= x1 <= X1 and Y0 <= y0 <= y1 <= Y1, (s, parent)
        sites = sum(ramb36[x0:x1 + 1]) * (y1 - y0 + 1)
        assert 78 / sites < 0.6, (s, sites)              # the TMEM's 64 and the DMA's 14
    assert 'otpu_s4_cells [list $tm/u_tmem $tm/u_dma $cp]' in tm
    assert "IS_SOFT false [get_pblocks pblock_CL_TMEM$s]" in tm


def test_place_net_delay_weight_reaches_place_design(tmp_path):
    """build.py --place-net-delay-weight writes place_net_delay_weight into implement/strategy.tcl
    and implement.tcl hands it to place_design (Vivado 2025.2's -net_delay_weight, compatible with
    -directive); without it the strategy and the command are as before."""
    assert flow.strategy_tcl(flow.STRATEGY, 'RuntimeOptimized') == flow.strategy_tcl(
        flow.STRATEGY, 'RuntimeOptimized', '')
    assert 'place_net_delay_weight' not in flow.strategy_tcl(flow.STRATEGY, 'RuntimeOptimized')
    assert flow.strategy_tcl(flow.STRATEGY, 'RuntimeOptimized', 'high').endswith(
        'set place_net_delay_weight high\n')
    impl = (BOARD / 'scripts/implement.tcl').read_text()
    assert 'set place_net_delay_weight {}' in impl
    assert 'place_design -directive $place_directive -no_bufg_opt -net_delay_weight $place_net_delay_weight' in impl
    out = subprocess.run([sys.executable, str(BOARD / 'build.py'), '--hdk', '/nonexistent',
                          '--target', str(S4_SLOWER[0][0]), '--plan', '--place-net-delay-weight',
                          'high'], text=True, capture_output=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert 'place_design Explore -net_delay_weight high, phys_opt_design AggressiveExplore' in out.stdout
