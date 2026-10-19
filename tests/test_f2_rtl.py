"""The AWS F2 memory arbiter and OCL crossing, independent of the proprietary shell."""
import subprocess

import pytest

from opentpu import rtlsim


@pytest.mark.parametrize("core_half,axi_half", [(5, 2), (3, 5)])
def test_f2_native_protocol(have_verilator, core_half, axi_half):
    """A pending head resets cleanly; separate AXI stalls preserve ordering and payloads."""
    exe = rtlsim.build("tb_f2nch_protocol", [
        rtlsim.RTL / "shell/otpu_afifo.sv",
        rtlsim.RTL / "boards/f2/otpu_f2_nch.sv",
        rtlsim.TB / "tb_f2nch_protocol.sv",
    ], {"CORE_HALF": core_half, "AXI_HALF": axi_half})
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "PASS F2 native protocol" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("core_half,axi_half", [(5, 2), (3, 5), (17, 2)])
def test_f2_memory_ordering(have_verilator, core_half, axi_half):
    """Reordered AXI completions must preserve each native command stream and PCIS fences."""
    exe = rtlsim.build("tb_f2mem", [
        rtlsim.RTL / "shell/otpu_afifo.sv",
        rtlsim.RTL / "boards/f2/otpu_f2_nch.sv",
        rtlsim.RTL / "boards/f2/otpu_f2_mem.sv",
        rtlsim.TB / "tb_f2mem.sv",
    ], {"CORE_HALF": core_half, "AXI_HALF": axi_half})
    for seed, beats in [(1, 1), (7, 6), (31, 4096)]:
        r = subprocess.run([str(exe), f"+seed={seed}", f"+nb={beats}", "+cycles=20000"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and "PASS" in r.stdout and "FAIL" not in r.stdout, r.stdout + r.stderr


def test_f2_memory_detects_missing_hazards(have_verilator):
    """The same workload must catch stale data if native same-beat ordering is removed."""
    exe = rtlsim.build("tb_f2mem", [
        rtlsim.RTL / "shell/otpu_afifo.sv",
        rtlsim.RTL / "boards/f2/otpu_f2_nch.sv",
        rtlsim.RTL / "boards/f2/otpu_f2_mem.sv",
        rtlsim.TB / "tb_f2mem.sv",
    ], {"HAZ": 0})
    r = subprocess.run([str(exe), "+seed=1", "+nb=1", "+cycles=3000"],
                       capture_output=True, text=True, timeout=120)
    assert "FAIL" in r.stdout and "returned stale or wrong data" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("core_half,axi_half,nxr", [(5, 2, 0), (3, 5, 0), (5, 2, 3), (3, 5, 3)])
def test_f2_ocl_crossing(have_verilator, core_half, axi_half, nxr):
    """Control accesses cross once, preserve data/errors, obey hold and ignore other windows;
    nxr 3: through the HBM build's SLR crossing stages (otpu_axil_xr) as well."""
    exe = rtlsim.build("tb_f2axil", [rtlsim.RTL / "shell/otpu_skidbuf.sv",
                                    rtlsim.RTL / "boards/f2/otpu_axil_cdc.sv",
                                    rtlsim.RTL / "shell/otpu_xr.sv",
                                    rtlsim.RTL / "boards/f2/otpu_axil_xr.sv",
                                    rtlsim.TB / "tb_f2axil.sv"],
                       {"CORE_HALF": core_half, "AXI_HALF": axi_half, "NXR": nxr})
    for seed in (1, 7):
        r = subprocess.run([str(exe), f"+seed={seed}", "+count=300"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr


# ---------------------------------------------------------------- the HBM build (f2-hbm)
HBM_FABRIC = ["shell/otpu_afifo.sv", "shell/otpu_xr.sv", "shell/otpu_skidbuf.sv", "shell/otpu_link.sv",
              "boards/f2/otpu_f2_hbm_nch.sv",
              "boards/f2/otpu_f2_hbm_port.sv", "boards/f2/otpu_f2_hbm_host.sv",
              "boards/f2/otpu_f2_hbm_mem.sv"]


def _hbm_exe(**params):
    return rtlsim.build("tb_f2hbm", [rtlsim.RTL / f for f in HBM_FABRIC] +
                        [rtlsim.TB / "tb_f2hbm.sv"], params)


@pytest.mark.parametrize("nport,core_half,axi_half,hbm_half,nxr", [
    (8, 5, 4, 2, 3),    # S = 1, NCH = 8 at 100 / 125 / 250 (the build's 200 / 250 / 450 ratios),
                        #   the native ports through the build's three crossing stages each way
    (8, 5, 4, 2, 0),    # the same wired (the fabric alone)
    (16, 5, 4, 2, 3),   # S = 2 x 8, or S = 1 x 16 (the 27B's 8 GiB slice)
    (4, 3, 5, 2, 3),    # a fast core against a slow shell clock
    (32, 5, 4, 2, 0)])  # every port (S = 4 x 8): no port left for reads of zeros
def test_f2_hbm_fabric(have_verilator, nport, core_half, axi_half, hbm_half, nxr):
    """Every channel's native contract over a reordering, interleaving HBM model; bridges stay in
    their pseudo-channels and form bursts; the host reaches every port in its own order. nxr 3:
    the adapter's side sees the readies three cycles late and its beats arrive three later (the
    bridges' FIFOs keep the room: otpu_f2_hbm_nch XL)."""
    exe = _hbm_exe(NPORT=nport, CORE_HALF=core_half, AXI_HALF=axi_half, HBM_HALF=hbm_half, NXR=nxr)
    for seed, nb in [(1, 1), (7, 6), (31, 64)]:
        r = subprocess.run([str(exe), f"+seed={seed}", f"+nb={nb}", "+cycles=8000"],
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and "PASS" in r.stdout and "FAIL" not in r.stdout, \
            r.stdout[-3000:] + r.stderr[-3000:]


def test_f2_hbm_fabric_detects_missing_hazards(have_verilator):
    """Without the bridges' same-beat waits the HBM's reordering returns stale data."""
    exe = _hbm_exe(NPORT=4, HAZ=0)
    r = subprocess.run([str(exe), "+seed=1", "+nb=1", "+cycles=4000"],
                       capture_output=True, text=True, timeout=600)
    assert "FAIL" in r.stdout and "stale or wrong data" in r.stdout, r.stdout[-3000:]


def test_pcis_registers_only_drive_the_link_stage():
    """Structural (the RTL source): every signal driven by an output of a u_pcis_* instance of
    otpu_f2_hbm_host is read only by u_pcis_* / u_xs_* instances or assigned to the block's PCIS
    output ports, so the registers beside the shell (pb_otpu_pcis, SLR1) reach u_host's logic
    (SLR0) through the link stages alone (round 7, docs/wide_port.md "Round 7 RTL")."""
    import re
    src = (rtlsim.RTL / "boards/f2/otpu_f2_hbm_host.sv").read_text()
    body = src[:src.index("endmodule")]
    outs = {"otpu_skidbuf": {"i_r", "o_v", "o_d", "any"}, "otpu_link_tx": {"i_r", "o_v", "o_d", "busy"},
            "otpu_link_rx": {"o_pop", "o_v", "o_d"}, "otpu_f2_hbm_oreg": {"i_r", "o_v", "o_d"},
            "otpu_xr": {"q"}}
    insts = re.findall(r"\n\s+(\w+)\s*(?:#\((?:[^()]|\([^()]*\))*\))?\s+(u_\w+)\s*\((.*?)\);", body, re.S)
    assert any(n.startswith("u_pcis_") for _, n, _ in insts) and any(n.startswith("u_xs_") for _, n, _ in insts)
    sig = lambda e: re.sub(r"\[.*?\]|\{|\}|\s", "", e.split("||")[0]) if "||" not in e else e
    driven, readers = {}, {}
    for mod, name, pins in insts:
        for port, expr in re.findall(r"\.(\w+)\(([^()]*(?:\([^()]*\))?[^()]*)\)", pins):
            names = set(re.findall(r"\b[a-z]\w*\b", expr)) - {"clk", "rst"}
            for n in names:
                (driven if port in outs.get(mod, set()) else readers).setdefault(n, set()).add(name)
    pcis_nets = {n for n, d in driven.items() if any(i.startswith("u_pcis_") for i in d)}
    assert pcis_nets, "no u_pcis_* outputs found"
    ports = set(re.findall(r"output\s+logic\s*(?:\[[^\]]*\])?\s*(h_\w+)", src))
    bad = {}
    for n in pcis_nets:
        if n in ports:
            continue
        for i in readers.get(n, set()):
            if not (i.startswith("u_pcis_") or i.startswith("u_xs_")):
                bad.setdefault(n, set()).add(i)
        # a continuous assignment or an always block of the host reading it
        for m in re.finditer(r"^(?!\s*//)(.*)$", body, re.M):
            line = m.group(1)
            if re.search(rf"\b{n}\b", line) and "u_pcis_" not in line and "u_xs_" not in line \
                    and not re.search(rf"\blogic\b.*\b{n}\b", line) and not line.strip().startswith("."):
                bad.setdefault(n, set()).add(line.strip()[:60])
    assert not bad, f"u_pcis_* outputs read outside the link stages: {bad}"


# ---------------------------------------------------------------- the HBM build since the alex-shell
# rework (docs/f2_hbm.md "The memory path"): Alex's burst200 memory path (ergodex-openTPU
# 06c169cb) at NCH channels, otpu_native_axi per channel into SmartConnect (modelled)
NAXI = ["boards/f2/otpu_native_axi.sv"]
HBM_AXI = ["shell/otpu_xr.sv", "shell/otpu_skidbuf.sv", "boards/f2/otpu_native_axi.sv",
           "boards/f2/otpu_f2_pcis_gate.sv", "boards/f2/otpu_f2_hbm_axi.sv"]


@pytest.mark.parametrize("burst,out,xl,ch_log2,idle", [
    (1, 8, 0, 31, 4), (16, 8, 0, 31, 4), (16, 32, 0, 31, 4), (16, 128, 0, 31, 4),   # Alex's four
    (16, 128, 6, 29, 4), (4, 128, 6, 29, 1), (4, 64, 6, 29, 1)])                    # the build's
def test_native_axi(have_verilator, burst, out, xl, ch_log2, idle):
    """otpu_native_axi alone (Alex's tb_native_axi, ported): random AXI stalls, data before /
    after / with commands, saturated credits, 4 KiB and window boundaries, idle flushes, injected
    RRESP / RLAST errors, reset; XL 6: behind three register stages each way, nothing lost; a
    stream takes a command a cycle; CH_LOG2 29: a 512 MiB window, a command outside it flagged."""
    exe = rtlsim.build("tb_native_axi", [rtlsim.RTL / f for f in NAXI] + [rtlsim.TB / "tb_native_axi.sv"],
                       {"BURST": burst, "OUT": out, "XL": xl, "CH_LOG2": ch_log2, "IDLE": idle})
    for seed in (1, 42, 2026, 4700):
        r = subprocess.run([str(exe), f"+seed={seed}"], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and "PASS native AXI" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]


def _hbm_axi_exe(**params):
    return rtlsim.build("tb_f2hbm_axi", [rtlsim.RTL / f for f in HBM_AXI] +
                        [rtlsim.TB / "tb_f2hbm_axi.sv"], params)


@pytest.mark.parametrize("nport,core_half,axi_half,hbm_half,nxr,burst", [
    (8, 5, 4, 2, 3, 4),     # the build: 8 channels at 200 / 250 / 450 (ratios), three SLR stages
    (8, 5, 4, 2, 0, 4),     # wired
    (8, 5, 4, 2, 3, 16),    # Alex's 16-beat bursts
    (16, 5, 4, 2, 3, 4),    # f2-hbm16 / S = 2 x 8
    (4, 3, 5, 2, 3, 4),     # a fast core against a slow shell clock
    (32, 5, 4, 2, 0, 4)])   # every port (S = 4 x 8)
def test_f2_hbm_axi(have_verilator, nport, core_half, axi_half, hbm_half, nxr, burst):
    """Every channel's native contract through otpu_native_axi and a SmartConnect + HBM model
    (crossing latencies, in order per slave, reads overtaking writes, two HBM beats a 512-bit
    beat); channels inside their pseudo-channels, INCR within 4 KiB, streams coalesced; the
    host's writes and reads (held behind its unanswered writes) in its order with its IDs;
    pcis_wbusy while a host write is unanswered."""
    exe = _hbm_axi_exe(NPORT=nport, CORE_HALF=core_half, AXI_HALF=axi_half, HBM_HALF=hbm_half,
                       NXR=nxr, READ_BURST=burst)
    for seed, nb in [(1, 1), (7, 6), (31, 64)]:
        r = subprocess.run([str(exe), f"+seed={seed}", f"+nb={nb}", "+cycles=8000"],
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and "PASS" in r.stdout and "FAIL" not in r.stdout, \
            r.stdout[-3000:] + r.stderr[-3000:]


def test_f2_hbm_axi_detects_missing_order(have_verilator):
    """Without otpu_native_axi's direction barriers and the gate's read hold, the model's reads
    overtaking writes return stale data, on the channels and on the host."""
    exe = _hbm_axi_exe(NPORT=8, NXR=3, ORDER=0)
    r = subprocess.run([str(exe), "+seed=1", "+nb=1", "+cycles=8000"],
                       capture_output=True, text=True, timeout=600)
    assert "FAIL" in r.stdout and "stale or wrong data" in r.stdout and \
        "is not its own writes'" in r.stdout, r.stdout[-3000:]
