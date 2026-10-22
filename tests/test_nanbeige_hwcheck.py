"""The FPGA correctness gate rejects bit differences and never falls back."""
import json

import numpy as np

from tools import nanbeige_hwcheck as H


def test_bit_comparison_rejects_signed_zero_nonfinite_and_wrong_shape():
    assert H.compare_words(np.ones((2, 4)), np.ones((2, 4)))["passed"]
    result = H.compare_words(np.array([-0.], np.float32), np.array([0.], np.float32))
    assert not result["passed"] and result["unequal_words"] == 1
    assert result["max_abs_diff"] == 0
    assert not H.compare_words([np.nan], [np.nan])["passed"]
    assert not H.compare_words([np.inf], [np.inf])["passed"]
    assert not H.compare_words(np.zeros((2, 4)), np.zeros(8))["passed"]


def test_board_open_failure_is_recorded_without_isa_fallback(monkeypatch, tmp_path):
    def fail_open(*args):
        raise RuntimeError("test board unavailable")

    def forbidden_fallback(*args):
        raise AssertionError("an engine must not be created after FPGA open fails")

    monkeypatch.setattr(H, "BoardSession", fail_open)
    monkeypatch.setattr(H, "make_engine", forbidden_fallback)
    path = tmp_path / "failure.json"
    assert H.main(["--dev", "f2:0", "--json", str(path)]) == 1
    report = json.loads(path.read_text())
    assert not report["passed"] and report["phase"] == "open"
    assert report["error"] == "RuntimeError: test board unavailable"
    assert "stages" not in report
