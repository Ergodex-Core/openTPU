"""Held-out write/reuse oracles: mutations must fail independent predictions."""
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.wideport_memory_feedback import read_trace
from tools.wideport_native_adapter import NativeAdapter
from tools.wideport_write_coupled import validate_write_coupled


@pytest.fixture
def evidence():
    supplied = os.environ.get("WP_WRITE_ORACLES")
    if not supplied:
        pytest.skip("requires preserved bounded three-MM write fixtures")
    matches = list(Path(supplied).glob("page-reuse-*/two_credit/page-reuse-summary.json"))
    assert len(matches) == 1, matches
    return matches[0].parent


def test_generated_write_and_new_epoch_closure(evidence):
    result = validate_write_coupled(evidence)
    assert result["stores"] == dict(starts=5, writes=5, completions=5)
    assert result["generated_page_read_requests"] == result["generated_page_read_responses"]
    assert len(result["generated_write_events"]) == 5
    assert result["a_cache_payloads_checked"] == 108
    assert result["native_payloads_checked"] == 1252
    assert result["new_epoch_read_ownership_closed"]
    assert not result["calibrated"] and result["tok_s"] is None


@pytest.mark.parametrize("mutation", ["write_payload", "early_ack", "request", "early_done",
                                     "early_idle", "stale_a_data", "cache_valid", "joined_b_data"])
def test_corrupt_write_reuse_oracles_rejected(evidence, mutation):
    _, native = read_trace(evidence / "native-feedback.jsonl")
    _, adapter = read_trace(evidence / "adapter-feedback.jsonl")
    _, slices = read_trace(evidence / "tmfeedback.jsonl")
    native, adapter, slices = deepcopy((native, adapter, slices))
    starts = [e["cycle"] for e in slices if e["start"] & 1]
    if mutation == "write_payload":
        row = next(r for e in native for r in e["channels"] if r["wv"])
        bit = (int(row["wmask"], 16) & -int(row["wmask"], 16)).bit_length() - 1
        row["wdata"] = f"{int(row['wdata'], 16) ^ (1 << (8 * bit)):0128x}"
        signature = "native.write_data"
    elif mutation == "early_ack":
        row = next(r for e in native[starts[5]:] for r in e["channels"] if r["wv"])
        row["wdone"] += 1
        signature = "native.ch.*wdone"
    elif mutation == "request":
        e = next(e for e in slices[starts[5]:] if e["memory_sources"]["dma_req"])
        e["memory_sources"]["dma_req"] = 0
        signature = "dma.generated_write_request"
    elif mutation == "early_done":
        slices[starts[5] + 2]["done"] |= 1
        signature = "dma.generated_done"
    elif mutation == "early_idle":
        e = next(e for e in adapter[starts[5]:] if not e["wr_idle"])
        e["wr_idle"] = 1
        signature = "adapter.wr_idle"
    elif mutation == "cache_valid":
        e = next(e for e in adapter[starts[5]:] if e["a_cache_valid"])
        e["a_cache_valid"] ^= e["a_cache_valid"] & -e["a_cache_valid"]
        signature = "adapter.a_cache_valid"
    elif mutation == "joined_b_data":
        e = next(e for e in adapter[starts[6]:] if e["b_rvalid"] and not e["b_rtag"])
        chunk = (e["b_rcm"] & -e["b_rcm"]).bit_length() - 1
        e["b_rdata"] = f"{int(e['b_rdata'],16) ^ (1 << (1024*chunk)):01024x}"
        signature = "adapter.b_data"
    else:
        patch = json.loads((evidence / "feedback-inputs.json").read_text())["epoch2"]
        initial = np.fromfile(evidence / "dram_0.bin", dtype=">u4").astype("<u4").view(np.uint8)
        fresh = initial[patch["scale_stage"]:patch["scale_stage"] + 64].tobytes()
        old = initial[patch["scale_page"]:patch["scale_page"] + 64].tobytes()
        third = [e["cycle"] for e in slices if e["start"] & 2][2]
        e = next(e for e in adapter[third:] if e.get("a_rdata") and
                 int(e["a_rdata"], 16).to_bytes(64, "little") == fresh)
        e["a_rdata"] = f"{int.from_bytes(old, 'little'):0128x}"
        signature = "adapter.a_data"
    with pytest.raises(ValueError, match=signature):
        validate_write_coupled(evidence, native_oracles=native,
                               adapter_oracles=adapter, slice_oracles=slices)


def test_disabled_write_invalidation_is_rejected(evidence, monkeypatch):
    original = NativeAdapter.step

    def skip_invalidation(self, *args, **kwargs):
        self.write_invalidate = [None] * 8
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NativeAdapter, "step", skip_invalidation)
    with pytest.raises(ValueError):
        validate_write_coupled(evidence)
