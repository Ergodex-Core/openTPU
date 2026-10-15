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


@pytest.mark.parametrize("core_half,axi_half", [(5, 2), (3, 5)])
def test_f2_ocl_crossing(have_verilator, core_half, axi_half):
    """Control accesses cross once, preserve data/errors, obey hold and ignore other windows."""
    exe = rtlsim.build("tb_f2axil", [rtlsim.RTL / "boards/f2/otpu_axil_cdc.sv",
                                    rtlsim.TB / "tb_f2axil.sv"],
                       {"CORE_HALF": core_half, "AXI_HALF": axi_half})
    for seed in (1, 7):
        r = subprocess.run([str(exe), f"+seed={seed}", "+count=300"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and "PASS" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("rod,rd,lat,core_half,axi_half,want", [
    (32, 64, 20, 5, 2, 2.0),      # 100 MHz core: both channels, a beat a core cycle each
    (32, 64, 20, 5, 4, 1.25),     # 200 MHz core: the DIMM port's beat an aclk cycle
    (32, 64, 150, 5, 2, 1.04),    # a long latency: 2 x ROD reads per round trip (Little's law)
    (128, 128, 150, 5, 2, 2.0),
])
def test_f2_read_throughput(have_verilator, rod, rd, lat, core_half, axi_half, want):
    """Streamed reads go out one an aclk cycle (the AR queues of otpu_f2_nch and otpu_f2_mem;
    one AR register each took a read every second cycle: 32 B an aclk cycle), else as many as
    the reads in flight allow."""
    exe = rtlsim.build("tb_f2bw", [rtlsim.RTL / "shell/otpu_afifo.sv",
                                  rtlsim.RTL / "boards/f2/otpu_f2_nch.sv",
                                  rtlsim.RTL / "boards/f2/otpu_f2_mem.sv",
                                  rtlsim.TB / "tb_f2bw.sv"],
                       {"ROD": rod, "RD": rd, "CORE_HALF": core_half, "AXI_HALF": axi_half})
    r = subprocess.run([str(exe), f"+lat={lat}", "+cycles=8000"], capture_output=True,
                       text=True, timeout=120)
    import re
    m = re.search(r"F2BW .*: ([0-9.]+) beats/core cycle", r.stdout)
    assert m, r.stdout + r.stderr
    assert abs(float(m.group(1)) - want) < 0.03 * want, r.stdout
