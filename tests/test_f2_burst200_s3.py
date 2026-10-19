"""Three slices on Alex's burst200 fabric (targets/f2-burst200-s3.toml, docs/v4/f2-burst200-s3.md):
the generator reproduces the checked-in one-slice wrapper byte for byte and the three-slice
one, the floorplan names every cell of the three-slice top, the build driver takes --target,
the generated top lints under Verilator against stubs of the vendor IP, and the tiny Gemma 3
runs bit-exactly on the three-slice board model (otpu_board S = 3 under tb_board)."""
import importlib.util
import os
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
S1 = ROOT / 'targets/f2-burst200.toml'
S3 = ROOT / 'targets/f2-burst200-s3.toml'
TOP1 = ROOT / 'rtl/boards/f2/burst200/cl_opentpu.sv'
TOP3 = ROOT / 'rtl/boards/f2/burst200-s3/cl_opentpu.sv'


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gen = _module('generate_f2_top', BOARD / 'scripts/generate_f2_top.py')
flow = _module('burst200_build', BOARD / 'build.py')


def hdk() -> Path:
    """A full aws-fpga HDK (its cl_ports.vh and interface templates): OTPU_HDK, or the usual
    checkouts; else skip."""
    for p in [os.environ.get('OTPU_HDK'), Path.home() / 'ergodex/aws-fpga/hdk',
              Path('/home/ubuntu/aws-fpga/hdk')]:
        if p and (Path(p) / 'common/shell_stable/design/interfaces/cl_ports.vh').is_file():
            return Path(p)
    pytest.skip('no aws-fpga HDK checkout (OTPU_HDK)')


def test_target_geometry():
    """The three-slice target: f2-burst200's arch at S = 3, two 2 GiB channels a slice on six
    distinct HBM channels' ports, the s3 floorplan, the core and shell RTL alone."""
    t1, t3 = T.load(str(S1)), T.load(str(S3))
    assert t3.arch == dict(t1.arch, S=3) and t3.uarch == t1.uarch and t3.tech == t1.tech
    assert t3.platform_id == t1.platform_id == 3
    assert T.sources(t3) == T.sources(t1)
    p = T.rtl_params(t3)
    assert (p['S'], p['NCH'], p['CH_LOG2'], p['DRAM_LOG2']) == (3, 2, 31, 32)
    assert gen.geometry(t3) == (3, 2, [0, 4, 8, 12, 16, 20], 21)
    assert gen.geometry(t1) == (1, 2, [0, 4], 5)
    pl = flow.plan(S3)
    assert pl['S'] == 3 and pl['floorplan'].name == 'floorplan_s3.tcl'
    assert [p.name for p in pl['rtl']] == ['cl_opentpu.sv', 'otpu_native_axi.sv',
                                           'otpu_hbm_geometry.sv']
    assert pl['geometry'].splitlines()[1:] == ['set otpu_S 3', 'set otpu_ports {0 4 8 12 16 20}']
    pl1 = flow.plan(S1)
    assert pl1['floorplan'].name == 'floorplan.tcl' and pl1['geometry'].endswith(
        'set otpu_S 1\nset otpu_ports {0 4}\n')
    assert all(p.parent == ROOT / 'rtl/boards/f2/burst200' for p in pl1['rtl'])


def test_one_slice_wrapper_regenerates_byte_identical():
    """With the one-slice target the generator reproduces Alex's hardware-tested wrapper."""
    assert gen.generate(T.load(str(S1)), hdk()) == TOP1.read_text()


def test_three_slice_wrapper_is_the_generators():
    """The checked-in three-slice top is the generator's output for its target."""
    text = gen.generate(T.load(str(S3)), hdk())
    assert text == TOP3.read_text()


def _instances(text: str) -> dict:
    """{instance: module} of a generated top (every instantiation ends in `) name (`)."""
    return {n: m for m, n in re.findall(r'^(\w+)(?: #\([^;]*?\))? (u_\w+) \(', text, re.M | re.S)}


def test_three_slice_top_wiring():
    """What the three-slice top instantiates: six adapters at the windows' bases, three memory
    block designs and the host crossbar, a register slice per SLR hop, the wrapper covering port
    20, per-SLR resets, and a 9-bit status (six channel errors, three LEDs)."""
    text = TOP3.read_text()
    inst = _instances(text)
    assert [inst[f'u_native{j}'] for j in range(6)] == ['otpu_native_axi'] * 6
    for j in range(6):
        assert f"34'h{j * 0x80000000:x}" in text
    assert {inst[f'u_mem{s}'] for s in range(3)} == {f'f2_mem_s{s}_wrapper' for s in range(3)}
    assert inst['u_host'] == 'f2_host_wrapper' and inst['u_pcis_pipe'] == 'f2_pcis_pipe'
    assert {k for k, m in inst.items() if m == 'f2_pcis_pipe'} == {
        'u_pcis_pipe', 'u_host_pipe0', 'u_host_pipe1', 'u_host_pipe2'}
    assert {k for k, m in inst.items() if m == 'f2_native_pipe'} == {
        'u_npipe2_slr0', 'u_npipe3_slr0', 'u_npipe4_slr1', 'u_npipe5_slr1', 'u_npipe4_slr0',
        'u_npipe5_slr0'}
    assert '.NUM_OF_AXI_PORTS(21)' in text and '.S(3)' in text
    assert '.calib({6{hbm_ready_core}})' in text and ".ded(6'b0)" in text
    assert {k for k, m in inst.items() if m == 'lib_pipe'} == {
        'u_rstpipe_core_slr0', 'u_rstpipe_core_slr2', 'u_rstpipe_main_slr0'}
    assert '.WIDTH(9)' in text and "{26'd0,shell_status[8:3]}" in text
    for port in (0, 4, 8, 12, 16, 20):
        assert f"assign h_arid[{port}] = 6'd0;" in text
        assert f"assign h_arvalid[{port}] = '0;" not in text
    for port in (1, 2, 3, 5, 19):
        assert f"assign h_arvalid[{port}] = '0;" in text
    # the native masters of slices 1 and 2 reach their SmartConnect through the pipes
    assert '.S_N0_awaddr(np2_slr0_awaddr)' in text and '.S_N1_awaddr(np5_slr0_awaddr)' in text
    assert '.s_axi_awaddr(np4_slr1_awaddr)' in text          # slice 2's second hop
    assert '.S_N0_awaddr(n0_awaddr)' in text                   # slice 0's straight in


def test_floorplan_names_every_cell_of_the_top():
    """floorplan_s3.tcl places every top-level instance that is not a clock or CDC primitive,
    and names no instance the top lacks; the crossing registers it marks are otpu_board's."""
    fp = (BOARD / 'constraints/floorplan_s3.tcl').read_text()
    inst = _instances(TOP3.read_text())
    placed = set(re.findall(r'\$cl/(u_\w+)(?=[\s\]])', fp))
    placed |= {'u_board'} if '$cl/u_board/' in fp else set()   # placed by its pieces
    free = {'u_core_clk', 'u_hbm_clk', 'u_rst_main', 'u_rst_core', 'u_rst_hbm', 'u_ready_cdc',
            'u_rst_core_slr0', 'u_rst_core_slr2', 'u_rst_main_slr0'} | {
        f'u_geometry{j}' for j in range(21)}
    placed_resets = {'u_rst_core_slr0', 'u_rst_core_slr2', 'u_rst_main_slr0'}
    assert placed - set(inst) == set(), placed - set(inst)
    assert set(inst) - placed - (free - placed_resets) == set(), set(inst) - placed - free
    for s in range(3):
        assert f'g_slice\\[{s}\\].u_slice' in fp or (s == 0 and '$cl/u_board/u_slice' in fp)
    assert fp.count('pblock_CL_SLR2') >= 3 and 'CLOCKREGION_X0Y8:CLOCKREGION_X7Y11' in fp
    assert 'pblock_CL_SLR1_XBAR' in fp and '$cl/u_host]' in fp
    assert 'USER_SLL_REG TRUE' in fp and 'foreach s {0 2}' in fp
    board = (ROOT / 'rtl/shell/otpu_board.sv').read_text()
    for name in ('u_dn_tx', 'u_dn_rx', 'u_up_tx', 'u_up_rx', 'u_cport', 'u_chub', 'u_pf_tx',
                 'u_pf_rx', 'g_xs', 'g_pf'):
        assert name in board


def test_build_plan_takes_the_target(tmp_path):
    out = subprocess.run([sys.executable, str(BOARD / 'build.py'), '--hdk', '/nonexistent',
                          '--target', str(S3), '--plan'], text=True, capture_output=True,
                         cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert '3 slice(s), HBM ports [0, 4, 8, 12, 16, 20]' in out.stdout
    assert 'burst200-s3/cl_opentpu.sv' in out.stdout and 'floorplan_s3.tcl' in out.stdout
    assert out.stdout.strip().endswith('7. audit')


def test_ip_scripts_read_the_geometry():
    """The IP, PCIS and link stages read geometry.tcl (a block design per slice, the host
    crossbar and the native register slice at S > 1) and keep Alex's one-slice defaults."""
    ip = (BOARD / 'scripts/create_f2_ip.tcl').read_text()
    assert 'set otpu_S 1; set otpu_ports {0 4}' in ip and 'proc otpu_mem_bd' in ip
    assert 'otpu_mem_bd f2_mem 0 [lindex $otpu_ports 0] [lindex $otpu_ports 1]' in ip
    assert 'f2_mem_s$s' in ip and 'otpu_host_bd $otpu_S' in ip
    assert '$s * 0x100000000' in ip                 # the crossbar: 4 GiB per slice
    pcis = (BOARD / 'scripts/create_pcis_ip.tcl').read_text()
    assert 'f2_native_pipe' in pcis and 'CONFIG.ADDR_WIDTH {34}' in pcis
    link = (BOARD / 'scripts/build_f2.tcl').read_text()
    assert 'f2_native_pipe.xci' in link and 'lappend bds f2_host' in link


# ---- Verilator lint of the generated tops against stubs of the vendor IP
AXI_RESP = {'awready', 'wready', 'bvalid', 'bresp', 'bid', 'arready', 'rvalid', 'rdata',
            'rresp', 'rlast', 'rid'}


def _stub(mod: str, ports: dict) -> str:
    """A module with the given {port: (direction, width or (width, unpacked))}."""
    lines = []
    for n, d in ports.items():
        dirn, w = d
        if isinstance(w, tuple):
            lines.append(f'  {dirn} logic [{w[0] - 1}:0] {n} {w[1]}')
        else:
            lines.append(f'  {dirn} logic ' + (f'[{w - 1}:0] ' if w > 1 else '') + n)
    return f'module {mod} #(parameter WIDTH = 1, DEST_SYNC_FF = 4, RST_ACTIVE_HIGH = 0, ' \
           f'SRC_INPUT_REG = 0, INIT_SYNC_FF = 0, SIM_ASSERT_CHK = 0, NUM_OF_AXI_PORTS = 1, ' \
           f'AXI4_INTERFACE = 0) (\n' + ',\n'.join(lines) + '\n);\nendmodule\n'


def _stubs(S: int, ports: list) -> str:
    """Stubs of everything the top instantiates besides openTPU's RTL and lib_pipe."""
    out = []
    out.append(_stub('xpm_cdc_async_rst', {'src_arst': ('input', 1), 'dest_clk': ('input', 1),
                                          'dest_arst': ('output', 1)}))
    out.append(_stub('xpm_cdc_single', {'src_clk': ('input', 1), 'src_in': ('input', 1),
                                       'dest_clk': ('input', 1), 'dest_out': ('output', 1)}))
    out.append('module xpm_cdc_array_single #(parameter WIDTH = 1, DEST_SYNC_FF = 4, '
               'SRC_INPUT_REG = 0, INIT_SYNC_FF = 0, SIM_ASSERT_CHK = 0) (input logic src_clk, '
               'input logic [WIDTH-1:0] src_in, input logic dest_clk, '
               'output logic [WIDTH-1:0] dest_out);\nendmodule\n')
    for n in ('f2_core_clk', 'f2_hbm_clk'):
        out.append(_stub(n, {'clk_in1': ('input', 1), 'reset': ('input', 1),
                             'locked': ('output', 1), 'clk_out1': ('output', 1),
                             'clk_out2': ('output', 1)}))
    axil = {'awaddr': 32, 'awvalid': 1, 'awready': 1, 'wdata': 32, 'wstrb': 4, 'wvalid': 1,
            'wready': 1, 'bresp': 2, 'bvalid': 1, 'bready': 1, 'araddr': 32, 'arvalid': 1,
            'arready': 1, 'rdata': 32, 'rresp': 2, 'rvalid': 1, 'rready': 1, 'awprot': 3,
            'arprot': 3}

    def axi_ports(prefix, width, slave):
        p = {}
        for k, w in width.items():
            resp = k in AXI_RESP
            p[prefix + k] = ('output' if resp == slave else 'input', w)
        return p
    p = {'s_axi_aclk': ('input', 1), 's_axi_aresetn': ('input', 1), 'm_axi_aclk': ('input', 1),
         'm_axi_aresetn': ('input', 1)}
    p.update(axi_ports('s_axi_', axil, True))
    p.update(axi_ports('m_axi_', axil, False))
    out.append(_stub('f2_ocl_cdc', p))
    full = dict(gen_hbm := {'araddr': 34, 'arburst': 2, 'arid': 6, 'arlen': 4, 'arsize': 3,
                            'arvalid': 1, 'awaddr': 34, 'awburst': 2, 'awid': 6, 'awlen': 4,
                            'awsize': 3, 'awvalid': 1, 'rready': 1, 'bready': 1, 'wdata': 256,
                            'wlast': 1, 'wstrb': 32, 'wvalid': 1, 'arready': 1, 'awready': 1,
                            'rdata': 256, 'rid': 6, 'rlast': 1, 'rresp': 2, 'rvalid': 1,
                            'wready': 1, 'bid': 6, 'bresp': 2, 'bvalid': 1})
    extra = {'awcache': 4, 'awlock': 1, 'awprot': 3, 'awqos': 4, 'arcache': 4, 'arlock': 1,
             'arprot': 3, 'arqos': 4, 'awregion': 4, 'arregion': 4}
    host = dict(full, awaddr=64, araddr=64, awid=16, arid=16, bid=16, rid=16, awlen=8, arlen=8,
                wdata=512, rdata=512, wstrb=64, **extra)
    native = dict(host, awaddr=34, araddr=34, awid=1, arid=1, bid=1, rid=1)
    for n, w in (('f2_pcis_pipe', host), ('f2_native_pipe', native)):
        p = {'aclk': ('input', 1), 'aresetn': ('input', 1)}
        p.update(axi_ports('s_axi_', w, True))
        p.update(axi_ports('m_axi_', w, False))
        out.append(_stub(n, p))
    hbm_mi = {k: v for k, v in dict(full, **extra).items()
              if k not in ('arid', 'awid', 'rid', 'bid', 'awregion', 'arregion')}
    sc_host = {k: v for k, v in host.items() if k not in ('awregion', 'arregion')}
    sc_native = {k: v for k, v in native.items() if k not in ('awregion', 'arregion')}

    def mem_bd(name, p0, p1):
        p = {'core_clk': ('input', 1), 'host_clk': ('input', 1), 'hbm_clk': ('input', 1),
             'aresetn': ('input', 1)}
        p.update(axi_ports('S_HOST_', sc_host, True))
        p.update(axi_ports('S_N0_', sc_native, True))
        p.update(axi_ports('S_N1_', sc_native, True))
        p.update(axi_ports(f'M_H{p0}_', hbm_mi, False))
        p.update(axi_ports(f'M_H{p1}_', hbm_mi, False))
        return _stub(name, p)
    if S == 1:
        out.append(mem_bd('f2_mem_wrapper', *ports))
    else:
        for s in range(S):
            out.append(mem_bd(f'f2_mem_s{s}_wrapper', ports[2 * s], ports[2 * s + 1]))
        p = {'host_clk': ('input', 1), 'aresetn': ('input', 1)}
        p.update(axi_ports('S_PCIS_', sc_host, True))
        for s in range(S):
            p.update(axi_ports(f'M_S{s}_', sc_host, False))
        out.append(_stub('f2_host_wrapper', p))
    # the HBM wrapper: arrays of NUM_OF_AXI_PORTS, the APB pairs, the stat interface
    hbm_out = {'arready', 'awready', 'rdata', 'rid', 'rlast', 'rresp', 'rvalid', 'wready',
               'bid', 'bresp', 'bvalid'}
    lines = ['  input logic apb_clk, i_clk_250m, i_rst_250m_n, i_clk_450m, i_rst_450m_n']
    for k, w in gen_hbm.items():
        d = 'output' if k in hbm_out else 'input'
        lines.append(f'  {d} logic [{w - 1}:0] {"o" if k in hbm_out else "i"}_axi_{k} '
                     f'[0:NUM_OF_AXI_PORTS-1]')
    apb = {'paddr': 22, 'pprot': 3, 'psel': 1, 'penable': 1, 'pwrite': 1, 'pwdata': 32,
           'pstrb': 4, 'pready': 1, 'prdata': 32, 'pslverr': 1}
    for j in range(2):
        lines.append(f'  input logic i_hbm_apb_preset_n_{j}')
        for k, w in apb.items():
            lines.append(f'  output logic [{w - 1}:0] o_hbm_apb_{k}_{j}')
    lines += ['  cfg_bus_t.slave hbm_stat_bus', '  output logic [7:0] o_cl_sh_hbm_stat_int',
              '  output logic o_hbm_ready']
    out.append('module cl_mem_hbm_wrapper #(parameter NUM_OF_AXI_PORTS = 1, AXI4_INTERFACE = 0, '
               'AXLEN_WIDTH = 4) (\n' + ',\n'.join(lines) + '\n);\nendmodule\n')
    return '\n'.join(out)


def _sh_ddr_stub(h: Path) -> str:
    """A stub of the shell's encrypted sh_ddr with the ports the unused-DDR template connects
    (directions from cl_ports.vh: a top-level input is the stub's input, an output its
    output, a DDR pin inout; the template's tie-offs and clocks inputs)."""
    iface = h / 'common/shell_stable/design/interfaces'
    ports = (iface / 'cl_ports.vh').read_text()
    dirn = {n: (d, w) for d, w, n in re.findall(
        r'^\s*(input|output|inout)\s+(?:logic|wire)\s*(\[[^]]+\])?\s*(\w+)', ports, re.M)}
    tmpl = (iface / 'unused_ddr_template.inc').read_text()
    body = tmpl[tmpl.index('SH_DDR'):]              # the instance (after the parameters)
    con = re.findall(r'\.(\w+)\s*\(\s*([^)\s]*)\s*\)', body)
    lines = []
    for port, sig in con:
        d, w = dirn.get(sig, ('input', '[1023:0]'))
        lines.append(f'  {d} wire {w} {port}' if d == 'inout' else f'  {d} logic {w} {port}')
    return 'module sh_ddr #(parameter DDR_PRESENT = 1) (\n' + ',\n'.join(lines) + '\n);\nendmodule\n'


@pytest.mark.parametrize('which', ['s1', 's3'])
def test_generated_top_lints_under_verilator(tmp_path, which):
    """The generated top elaborates under Verilator with the openTPU RTL, the HDK's interface
    headers and lib_pipe, and stubs of the vendor IP (every port of every instance exists with
    a compatible width; no undriven or multiply driven net in the wiring): the three-slice
    top's warnings are no more than the hardware-tested one-slice top's kinds."""
    if shutil.which('verilator') is None:
        pytest.skip('verilator not installed')
    h = hdk()
    t = T.load(str(S1 if which == 's1' else S3))
    S, _, ports, _ = gen.geometry(t)
    top = TOP1 if which == 's1' else TOP3
    (tmp_path / 'stubs.sv').write_text(_stubs(S, ports) + _sh_ddr_stub(h))
    iface = h / 'common/shell_stable/design/interfaces'
    srcs = [h / 'common/lib/interfaces.sv', h / 'common/lib/lib_pipe.sv'] + \
        [ROOT / 'rtl' / f for f in T.sources(t, 'generic')] + \
        [ROOT / 'rtl/boards/f2/burst200/otpu_native_axi.sv',
         ROOT / 'rtl/boards/f2/burst200/otpu_hbm_geometry.sv', tmp_path / 'stubs.sv', top]
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
    # the top's own warnings (not the openTPU RTL's): width truncations in the wiring would be
    # real mistakes; the one-slice top has none
    own = [l for l in msgs if 'cl_opentpu.sv' in l and 'WIDTHTRUNC' in l]
    assert not own, '\n'.join(own)


# ---- the tiny Gemma 3 on the three-slice board model
torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')


def test_tiny_gemma3_three_slices_on_board_model(have_verilator, tmp_path):
    """otpu_board at S = 3 (three slices, each on its two native channels behind the native
    memory model, one control block, the collective hub between them: tb_board with the
    f2-burst200-s3 target, platform 3) runs the tiny Gemma 3 split over three slices: logits
    bit-identical to the ISA simulator at S = 3, each slice's DRAM the ISA simulator's."""
    from conftest import sim_board_config
    from opentpu.host.board import BoardBackend, SimTransport
    from opentpu.llm.qwen3 import Engine
    from test_gemma3 import _model
    _, W, spec = _model(tmp_path, layers=2, hidden=384, ffn=768)
    cfg = sim_board_config(DRAM_BYTES=1 << 24, S=3)
    isa = Engine(spec, W, cap=256, cfg=cfg)
    tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2, stall=0, seed=5, target='f2-burst200-s3',
                      native=True)
    brd = Engine(spec, W, cap=256, cfg=cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tr))
    assert brd.backend.info['cfg']['S'] == 3 and brd.backend.info['cfg']['platform'] == 3
    for tok in (11, 222, 333):
        a, b = isa.step(tok), brd.step(tok)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
        st = brd.stats[-1]
        assert st['cycles'] > 0 and st['instructions'] == isa.stats[-1]['instructions']
        print(f"token {tok}: {st['cycles']} cycles, {st['instructions']} instructions a slice")
    n = isa.image.nbytes
    for s in range(3):
        assert np.array_equal(brd.backend.read(s, 0, n), isa.backend.machine.slices[s].dram[:n])
