"""Cycle-exact timing cuts: independent traffic scoreboards and RTL assertions."""
import pytest

from opentpu import rtlsim


@pytest.mark.parametrize("depth,lanes", [(2, 4), (8, 8), (16, 8)])
def test_collective_registered_head(have_verilator, depth, lanes):
    exe = rtlsim.build("tb_coll_port", [rtlsim.RTL / "top/otpu_pkg.sv",
                                      rtlsim.RTL / "top/otpu_coll.sv",
                                      rtlsim.TB / "tb_coll_port.sv"],
                      {"F": depth, "LANES": lanes})
    out = rtlsim.run_sim([exe], timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "HEAD PASS checks=11970" in out.stdout, out.stdout
