"""The F2 HBM memory path's RTL (wide-port-alexshell's tests/test_f2_rtl.py, its otpu_native_axi and
fabric cases, ported; docs/v4/p2-f2.md "M5"): otpu_native_axi alone, and the eight channels through
otpu_f2_hbm_axi into a SmartConnect + HBM model, at P2's geometry too (1 GiB channel windows, the
150 MHz core's clock ratio)."""
import re
import subprocess

import pytest

from opentpu import rtlsim

NAXI = ["boards/f2/otpu_native_axi.sv"]
HBM_AXI = ["shell/otpu_xr.sv", "shell/otpu_skidbuf.sv", "boards/f2/otpu_native_axi.sv",
           "boards/f2/otpu_f2_pcis_gate.sv", "boards/f2/otpu_f2_hbm_axi.sv"]


@pytest.mark.parametrize("burst,out,xl,ch_log2,idle,wait,porder,zb", [
    (1, 8, 0, 31, 4, 1, 0, 0), (16, 8, 0, 31, 4, 1, 0, 0), (16, 32, 0, 31, 4, 1, 0, 0),   # Alex's
    (16, 128, 0, 31, 4, 1, 0, 0),
    (16, 128, 6, 29, 4, 1, 0, 0), (4, 128, 6, 29, 1, 1, 0, 0),  # behind the SLR chain, waiting bursts
    (4, 128, 6, 29, 1, 0, 0, 0), (16, 128, 6, 29, 1, 0, 0, 0), (4, 64, 6, 29, 1, 0, 0, 0),   # no wait
    (8, 128, 6, 30, 1, 1, 0, 0),                               # P2: 1 GiB windows, the build's bursts
    (8, 128, 6, 30, 1, 1, 1, 0), (1, 8, 0, 31, 4, 1, 1, 0), (16, 32, 0, 31, 4, 1, 1, 0),   # page buckets
    (8, 128, 6, 30, 1, 1, 0, 1), (8, 128, 6, 30, 1, 1, 1, 1), (4, 64, 6, 29, 1, 0, 1, 1),   # zero-bubble bursts
    (16, 32, 0, 31, 4, 1, 0, 1)])
def test_native_axi(have_verilator, burst, out, xl, ch_log2, idle, wait, porder, zb):
    """otpu_native_axi alone (Alex's tb_native_axi): random AXI stalls, data before / after /
    with commands, saturated credits, 4 KiB and window boundaries, idle flushes, injected RRESP /
    RLAST errors, reset; XL 6: behind three register stages each way, nothing lost; a command
    outside the window flagged."""
    exe = rtlsim.build("tb_native_axi", [rtlsim.RTL / f for f in NAXI] + [rtlsim.TB / "tb_native_axi.sv"],
                       {"BURST": burst, "OUT": out, "XL": xl, "CH_LOG2": ch_log2, "IDLE": idle,
                        "WAIT": wait, "PORDER": porder, "ZB": zb})
    for seed in (1, 42, 2026, 4700):
        r = subprocess.run([str(exe), f"+seed={seed}"], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and "PASS native AXI" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]


def _hbm_axi_exe(**params):
    return rtlsim.build("tb_f2hbm_axi", [rtlsim.RTL / f for f in HBM_AXI] +
                        [rtlsim.TB / "tb_f2hbm_axi.sv"], params)


@pytest.mark.parametrize("nport,core_half,axi_half,hbm_half,nxr,burst,wait,ch_log2,porder", [
    (8, 5, 4, 2, 3, 8, 1, 29, 0),     # alexshell's build: 8 channels, three SLR stages, waiting bursts
    (8, 5, 4, 2, 0, 8, 1, 29, 0),  # wired
    (8, 5, 4, 2, 3, 8, 0, 29, 0),  # bursts only from what is queued
    (8, 7, 4, 2, 3, 8, 1, 29, 0),  # a 150 MHz core against 250 / 450 (the ratios)
    (8, 7, 4, 2, 3, 8, 1, 30, 0),  # P2 (f2-p2:c150): 4 copies x 2 channels of 1 GiB windows
    (8, 5, 4, 2, 3, 8, 1, 30, 0),  # ... at 200 MHz
    (8, 7, 4, 2, 3, 8, 1, 30, 1),  # ... with the barriers per page bucket (PORDER)
    (8, 5, 4, 2, 0, 8, 1, 29, 1),
    (8, 7, 4, 2, 3, 8, 1, 30, 3),  # ... and the zero-bubble burst switch (ZB: porder bit 1)
    (8, 7, 4, 2, 3, 8, 0, 30, 2)])
def test_f2_hbm_axi(have_verilator, nport, core_half, axi_half, hbm_half, nxr, burst, wait, ch_log2,
                    porder):
    """Every channel's native contract through otpu_native_axi and a SmartConnect + HBM model (the
    host on its own HBM port through the switch): crossing latencies, in order per slave, reads
    overtaking writes, two HBM beats a 512-bit beat; channels inside their windows, INCR within
    4 KiB, streams coalesced; the host's writes and reads in its order with its IDs."""
    exe = _hbm_axi_exe(NPORT=nport, CORE_HALF=core_half, AXI_HALF=axi_half, HBM_HALF=hbm_half,
                       NXR=nxr, READ_BURST=burst, READ_WAIT=wait, READ_IDLE=1, CH_LOG2=ch_log2,
                       PORDER=porder & 1, ZB=porder >> 1)
    for seed, nb in [(1, 1), (7, 6), (31, 64)]:
        r = subprocess.run([str(exe), f"+seed={seed}", f"+nb={nb}", "+cycles=8000"],
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and "PASS" in r.stdout and "FAIL" not in r.stdout, \
            r.stdout[-3000:] + r.stderr[-3000:]


def test_f2_hbm_axi_detects_missing_order(have_verilator):
    """Without otpu_native_axi's direction barriers and the gate's read hold, the model's reads
    overtaking writes return stale data, on the channels and on the host."""
    exe = _hbm_axi_exe(NPORT=8, NXR=3, ORDER=0, CH_LOG2=30)
    r = subprocess.run([str(exe), "+seed=1", "+nb=1", "+cycles=8000"],
                       capture_output=True, text=True, timeout=600)
    m = re.search(r"FAIL \(\d+ errors: (\d+) stale channel reads, (\d+) stale host reads\)", r.stdout)
    assert m and int(m[1]) > 0 and int(m[2]) > 0, r.stdout[-3000:]
