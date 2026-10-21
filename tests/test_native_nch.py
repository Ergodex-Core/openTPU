"""The native DRAM adapter (rtl/mem/otpu_native_dram.sv) at NCH channels: the address map, the
ports' ordering, and NCH = 2 bit- and cycle-identical to the two-channel adapter it generalized
(sim/verilator/tb_native_nch.sv; docs/f2_hbm.md)."""
import subprocess

import pytest

from opentpu import rtlsim

# the tree's two-channel adapter before NCH (the reference of the equivalence test)
TWO_CHANNEL = "4e9616a4"


def _ref(tmp_path):
    r = subprocess.run(["git", "-C", str(rtlsim.ROOT), "show",
                        f"{TWO_CHANNEL}:rtl/mem/otpu_native_dram.sv"],
                       capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"the two-channel adapter is not in this checkout's history ({TWO_CHANNEL})")
    p = tmp_path / "otpu_native_dram_ref.sv"
    p.write_text(r.stdout.replace("module otpu_native_dram #(", "module otpu_native_dram_ref #(", 1))
    return p


@pytest.mark.parametrize("chash", [1, 0])
def test_native_nch2_matches_two_channel(have_verilator, tmp_path, chash):
    """NCH = 2: every output equals the two-channel adapter's in every cycle, on random traffic
    of all three ports with random native readies, latencies and a_flush; wr_idle (since round
    8 a register's compare) never high where the reference's is low and at most a cycle late on
    a rise."""
    exe = rtlsim.build("tb_native_nch", [rtlsim.RTL / "mem/otpu_native_dram.sv", _ref(tmp_path),
                                         rtlsim.TB / "tb_native_nch.sv"],
                       {"NCH": 2, "REF": 1, "CHASH": chash})
    for seed in (1, 5):
        r = subprocess.run([str(exe), f"+seed={seed}", "+cycles=20000"],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0 and "PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]


@pytest.mark.parametrize("nch,chash,high", [(2, 1, 0), (4, 1, 0), (8, 1, 0), (8, 0, 0), (16, 1, 0),
                                             (16, 1, 1), (8, 1, 1)])
def test_native_nch_map_and_order(have_verilator, nch, chash, high):
    """NCH channels: reads of unwritten beats find init(logical beat) where the host's map puts
    it, port B reads see port B's writes, port A's runs and reuse return the right beats, and
    every written byte lands where the map says. high: port B's region above 4 GiB (a slice of
    16 x 512 MiB holds 8 GiB: the word address's bits 31:30 in the chunk index)."""
    exe = rtlsim.build("tb_native_nch", [rtlsim.RTL / "mem/otpu_native_dram.sv",
                                         rtlsim.TB / "tb_native_nch.sv"],
                       {"NCH": nch, "CHASH": chash, "HIGH": high})
    for seed in (1, 9):
        r = subprocess.run([str(exe), f"+seed={seed}", "+cycles=20000"],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0 and "PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
        # every channel carried traffic
        for c in range(nch):
            assert f"ch{c}: 0 reads" not in r.stdout, r.stdout


def test_native_nch_detects_a_wrong_map(have_verilator):
    """The testbench's map without the pair's fold: reads find other beats' data (FAIL)."""
    exe = rtlsim.build("tb_native_nch", [rtlsim.RTL / "mem/otpu_native_dram.sv",
                                         rtlsim.TB / "tb_native_nch.sv"],
                       {"NCH": 8, "CHASH": 1, "MAPBUG": 1})
    r = subprocess.run([str(exe), "+seed=1", "+cycles=3000"], capture_output=True, text=True,
                       timeout=300)
    assert "FAIL" in r.stdout and "wrong data" in r.stdout, r.stdout[-3000:]


@pytest.mark.parametrize("rd", [64, 128])
def test_native_nch8_wide_read_credit_saturation(have_verilator, rd):
    """All eight channels hit RD B reservations while A runs and SW fills share read tags.

    The directed native model holds responses until B credits saturate, then forces B port
    backpressure before releasing them. This tests the adapter, not the F2 AXI fabric.
    """
    exe = rtlsim.build("tb_native_rd_stress", [rtlsim.RTL / "mem/otpu_native_dram.sv",
                                                 rtlsim.TB / "tb_native_rd_stress.sv"],
                       {"RD": rd})
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and f"PASS native RD{rd} NCH8 PORTB4 ALINE64" in r.stdout, \
        r.stdout[-3000:] + r.stderr[-3000:]
