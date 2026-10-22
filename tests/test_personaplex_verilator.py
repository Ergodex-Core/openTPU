"""Focused PersonaPlex temporal checks on the real Verilator RTL."""
import numpy as np
import pytest

from tools.personaplex_verilator import run_case, simulation_config
from opentpu.llm.personaplex_temporal import Spec


def test_burst200_simulation_preserves_compute_geometry():
    profile, cfg, uarch = simulation_config(Spec(hidden=256, layers=2, n_heads=2, ffn=512), 256)
    assert cfg.D == 128 and cfg.MCOLS == 4 and cfg.LANES == 8
    assert cfg.TMEM_WORDS == 65536 and cfg.IMEM_WORDS == 32768
    assert cfg.DRAM_BYTES < profile.arch["DRAM_BYTES"]
    assert cfg.SEQ_WC and cfg.PAIR and uarch["SEQ_WC"] == 1
    assert uarch["TMEM_PMP"] == 1 and uarch["MXU_E"] == 4


def test_tiny_temporal_rtl_matches_isa_reference_and_reset(have_verilator):
    result = run_case()
    assert result["spec"]["layers"] == 2
    assert result["frames"] == 4 and len(result["runs"]) == 5
    assert result["reset_bit_exact"]
    assert np.isfinite(result["float64_relative_error"])
    assert result["float64_relative_error"] < result["float64_relative_error_limit"]
    assert all(run["rtl_isa_bit_exact"] and run["cycles"] > 0 for run in result["runs"])
    assert not result["full_model_measured"] and not result["fpga_used"]


@pytest.mark.parametrize("kwargs", [{"frames": 4, "cap": 3}, {"bw": 0}, {"stall": 100}])
def test_invalid_run_parameters_fail_before_simulation(kwargs):
    with pytest.raises(ValueError):
        run_case(**kwargs)


def test_timing_override_is_rejected_before_simulation(monkeypatch):
    monkeypatch.setenv("OTPU_SIM_PLUSARGS", "+axi_bw=50")
    with pytest.raises(ValueError, match="OTPU_SIM_PLUSARGS"):
        run_case()
