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
    of all three ports with random native readies, latencies and a_flush."""
    exe = rtlsim.build("tb_native_nch", [rtlsim.RTL / "mem/otpu_native_dram.sv", _ref(tmp_path),
                                         rtlsim.TB / "tb_native_nch.sv"],
                       {"NCH": 2, "REF": 1, "CHASH": chash})
    for seed in (1, 5):
        r = subprocess.run([str(exe), f"+seed={seed}", "+cycles=20000"],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0 and "PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]


@pytest.mark.parametrize("nch,chash,high,wdq", [(2, 1, 0, 0), (4, 1, 0, 0), (8, 1, 0, 0), (8, 0, 0, 0),
                                                 (16, 1, 0, 0), (16, 1, 1, 0), (8, 1, 1, 0),
                                                 (8, 1, 0, 1), (2, 1, 0, 1)])
def test_native_nch_map_and_order(have_verilator, nch, chash, high, wdq):
    """NCH channels: reads of unwritten beats find init(logical beat) where the host's map puts
    it, port B reads see port B's writes, port A's runs and reuse return the right beats, and
    every written byte lands where the map says. high: port B's region above 4 GiB (a slice of
    16 x 512 MiB holds 8 GiB: the word address's bits 31:30 in the chunk index). wdq: the
    write-done compare a register (the F2 HBM board's, and the burst200 fabric's two channels a
    slice at f2-burst200-s4-175p), wr_idle a cycle later, never earlier."""
    exe = rtlsim.build("tb_native_nch", [rtlsim.RTL / "mem/otpu_native_dram.sv",
                                         rtlsim.TB / "tb_native_nch.sv"],
                       {"NCH": nch, "CHASH": chash, "HIGH": high, "WDQ": wdq})
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
