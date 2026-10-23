"""Corrupt saved small RTL oracles; never rerun RTL or accept corrupt signals."""
from copy import deepcopy
import json
import os
from pathlib import Path
import numpy as np
import pytest
from tools.wideport_memory_feedback import read_trace
from tools.wideport_pipeline_ownership import validate_pipeline_ownership
from tools.wideport_page_reuse import validate_page_epochs


@pytest.fixture
def evidence():
    supplied = os.environ.get("WP_OWNERSHIP_ORACLES")
    if not supplied:
        pytest.skip("requires preserved bounded ownership fixtures")
    root = Path(supplied)
    matches = list((root / "reuse").glob("page-reuse-*/two_credit/page-reuse-summary.json"))
    assert len(matches) == 1, matches
    return matches[0].parent


def test_saved_positive_protocol_and_payload_oracles(evidence):
    assert validate_pipeline_ownership(evidence)["rows_drained"] == 12
    assert validate_page_epochs(evidence)["changed_epoch_lines_checked"] >= 2


@pytest.mark.parametrize("mutation", ["accumulator_first", "backpressure", "premature_commit", "slot_epoch"])
def test_corrupt_accumulator_or_output_oracles_rejected(evidence, mutation):
    _, original = read_trace(evidence / "tmfeedback.jsonl")
    events = deepcopy(original)
    if mutation == "accumulator_first":
        e = next(x for x in events if x["ownership"]["acc_first"])
        e["ownership"]["acc_first"] ^= 1
        signature = "ownership.acc_first"
    elif mutation == "backpressure":
        e = next(x for x in events if x["mxu"]["d_pop"])
        e["mxu"]["t_gnt"] = 0
        signature = "backpressure"
    elif mutation == "premature_commit":
        e = next(x for x in events if x["mxu"]["pop"])
        e["mxu"]["c_done"] = 1
        signature = "committed before"
    else:
        starts = [x for x in events if x["start"] & 2]
        starts[1]["seq"]["mxu_start_slot"] = starts[0]["seq"]["mxu_start_slot"]
        signature = "expected PC|matching dispatched slot"
    with pytest.raises(ValueError, match=signature):
        validate_pipeline_ownership(evidence, slice_events=events)


def test_old_epoch_payload_is_rejected_after_reuse(evidence):
    manifest = json.loads((evidence / "feedback-inputs.json").read_text())
    patch = manifest["epoch2"]
    image = np.fromfile(evidence / "dram_0.bin", dtype=">u4").astype("<u4").view(np.uint8)
    fresh = image[patch["scale_stage"]:patch["scale_stage"] + 64].tobytes()
    old = image[patch["scale_page"]:patch["scale_page"] + 64].tobytes()
    assert fresh != old
    _, slices = read_trace(evidence / "tmfeedback.jsonl")
    third = [e["cycle"] for e in slices if e["start"] & 2][2]
    _, original = read_trace(evidence / "native-feedback.jsonl")
    events = deepcopy(original)
    corrupted = False
    for event in events:
        if event["cycle"] <= third:
            continue
        for row in event["channels"]:
            if row["rv"] and "rdata" in row and int(row["rdata"], 16).to_bytes(64, "little") == fresh:
                row["rdata"] = f"{int.from_bytes(old, 'little'):0128x}"
                corrupted = True
                break
        if corrupted:
            break
    assert corrupted, "no fresh scale canary response was available to corrupt"
    with pytest.raises(ValueError, match="stale/data epoch"):
        validate_page_epochs(evidence, native_events=events)
