"""Known timestamp vectors and fail-closed receipt validation, not performance results."""
import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).parents[1] / "tools/context-bench/metrics.py"
spec = importlib.util.spec_from_file_location("context_bench_metrics", path)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_decode_uses_n_minus_one_intervals():
    stream = {"submit_ns": 0, "tokens": [{"index": i, "observed_ns": 1_000_000_000 + i * 10_000_000}
                                        for i in range(64)]}
    got = M.stream_metrics(stream)
    assert got["ttft_s"] == 1
    assert got["decode_intervals"] == 63
    assert got["decode_seconds_first_to_last"] == 0.63
    assert got["decode_tps"] == 100
    assert got["last_32_observed_tokens_tps"] == 100
    assert got["observed_interval_p95_s"] == 0.01


def test_arrival_coalescing_is_disclosed():
    stream = {"submit_ns": 0, "tokens": [{"index": i, "observed_ns": 1000} for i in range(3)]}
    got = M.stream_metrics(stream)
    assert got["coalesced_observations"] and got["decode_tps"] is None


def test_timestamp_clock_must_be_consistent():
    with pytest.raises(AssertionError):
        M.stream_metrics({"submit_ns": 3, "tokens": [{"index": 0, "observed_ns": 2}]})


@pytest.mark.parametrize("kind", ["cpu_inference", "isa_simulation", "estimate"])
def test_refuses_non_fpga_evidence(kind):
    with pytest.raises(AssertionError):
        M.validate({"identity": {}, "evidence_kind": kind}, {})


def test_p95_nearest_rank():
    assert M.p95([1, 2, 3, 4, 5]) == 5


def test_immediate_eos_has_no_visible_ttft_or_decode_rate():
    got = M.stream_metrics({"submit_ns": 1, "tokens": []})
    assert got["generated_tokens"] == 0
    assert got["ttft_s"] is None and got["decode_tps"] is None


def fixture_receipt():
    # Synthetic validation fixture only; never written as measurement evidence.
    case = {"model": "gemma-3-1b-it", "case_id": "fixture", "prompt_tokens": 1024,
            "input_ids_sha256": "input", "checkpoint_sha256": "checkpoint",
            "config_sha256": "config", "runtime_source_digest": "source"}
    receipt = {"evidence_kind": "physical_fpga_measurement", "identity": {
        "image": M.IMAGE, "profile": M.PROFILE, "build_id": 0x417243F1, "clock_mhz": 150,
        "weights_format": "int8", "head_format": "int8", "batch": 1, "greedy": True,
        "model": case["model"], "runtime_source_digest": "source", "checkpoint_sha256": "checkpoint",
        "config_sha256": "config", "normal_locks": ["f2:0", "f2-0000:34:00.0"]},
        "case_id": "fixture", "prompt_tokens": 1024, "input_ids_sha256": "input",
        "requested_output_tokens": 64, "capacity": 4096, "truncated": False,
        "prefix_kv_reused": 0, "residency": "warm_resident", "ownership_window_id": "fixture",
        "raw_phase_trace": ["synthetic"], "worker": {"tokens": []},
        "contract": {"resident": True, "device_inputs": True, "can_generate": True,
                     "chains": True, "stateless_kv": True, "host_service": False,
                     "host_embeddings": False, "host_ple": False}}
    return receipt, case


def test_matched_receipt_contract():
    receipt, case = fixture_receipt()
    assert M.validate(receipt, case)


@pytest.mark.parametrize("field,value", [("model", "gemma-3-270m-it"),
    ("weights_format", "int4"), ("clock_mhz", 200), ("runtime_source_digest", "other"),
    ("checkpoint_sha256", "other"), ("normal_locks", [])])
def test_unmatched_identity_is_rejected(field, value):
    receipt, case = fixture_receipt()
    receipt["identity"][field] = value
    with pytest.raises(AssertionError):
        M.validate(receipt, case)


def test_owner_observer_cannot_turn_simulation_into_physical_receipt():
    from types import SimpleNamespace
    owner_path = path.with_name("observe_owned.py")
    s = importlib.util.spec_from_file_location("owner_observer", owner_path)
    owner = importlib.util.module_from_spec(s)
    s.loader.exec_module(owner)
    with pytest.raises(ValueError, match="Physical BoardBackend"):
        owner.capture_owned_turn(SimpleNamespace(backend=object()), None, {}, {}, "fixture", lambda: False)
