"""Replay diagnostics preserve overlap and never turn missing timing into qualified TPS."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from tools import wideport_lpn as model
from tools import wideport_tmem_accounting as A


COUNTERS = ("TMEMSTAT sid=0 cyc=891870 lose=21441,4459,1211,0 "
            "n= 784092,70264,32831,4683,0,0,0 d= 784092,70296,32799,4683,0,0,0,0,0 "
            "dw= 690292,143948,32560,23479,1591,0,0,0,0,0 "
            "rd= 10244,32768,7568,41699,21072,1296,1808,0 "
            "rpl= 257,0,1575,292,13521,0,3,0")


def test_real_gemma_counters_are_port_events_not_cycle_correction():
    counters = A.parse_tmemstat(COUNTERS)
    assert counters["cycles"] == 891870
    assert sum(counters["rpl"]) == 15648
    assert counters["rpl"][4] == 13521


@pytest.mark.parametrize("text", [
    COUNTERS.replace("cyc=891870", "cyc=891871"),
    COUNTERS.replace("sid=0", "sid=1"),
    COUNTERS.replace("13521", "-1"),
    COUNTERS.replace("rpl= 257,0,1575,292,13521,0,3,0", "rpl= 0"),
    COUNTERS + " rpl= 0,0,0,0,0,0,0,0",
    COUNTERS.replace(" rd=", " bogus=1 rd="),
])
def test_incomplete_or_inconsistent_counters_fail_closed(text):
    with pytest.raises(ValueError):
        A.parse_tmemstat(text)


@pytest.mark.parametrize("pmp", [0, 1])
@pytest.mark.parametrize("error", [None, .001, .20])
def test_missing_replay_timing_never_publishes_tps_even_with_good_phase_error(pmp, error):
    scope = A.replay_scope({"TMEM_IMPL": 3, "TMEM_RPL": True, "TMEM_PMP": pmp})
    result = A.throughput_result(891870, 200, error, scope)
    assert scope["cycle_correction"] == 0
    assert not scope["replay_timing_modeled"]
    assert result["tok_s"] is None
    assert not result["calibrated"]


def test_complete_timing_still_requires_error_validation():
    scope = A.replay_scope({"TMEM_IMPL": 1})
    assert A.throughput_result(100, 1, None, scope)["tok_s"] is None
    assert A.throughput_result(100, 1, .11, scope)["tok_s"] is None
    assert A.throughput_result(100, 1, .02, scope)["tok_s"] == 10000


def test_variant_resolution_retains_tmem_mode_and_board_credit_default():
    assert model.target_uarch("f2-hbm:p4rl8a64")["TMEM_PMP"] == 1
    assert model.target_uarch("f2-hbm:p4rl8a64tdp")["TMEM_PMP"] == 0
    assert model.target_uarch("f2-hbm:p4rl8a64tdp")["FIFO_DEPTH"] == 1024
    assert [model.channel_depth(n) for n in (1, 2, 4, 8)] == [128, 128, 64, 128]


@pytest.fixture
def reference(tmp_path):
    log = b"checked correctness True\n"
    lens = {"cycles": 891870, "config": {"TMEM_PMP": 1, "FIFO_DEPTH": 128},
            "instrs": [[0, 0, 1, 3, "VOP.mul", "", 0, 0, -1, 10, 100, 0, {"frz": 65}]]}
    run = {"pmp": 0, "returncode": 0, "bit_exact_isa": True,
           "log_sha256": hashlib.sha256(log).hexdigest(), "cycles": 891870,
           "parameters": {"S": 1, "TMEM_PMP": 0, "FIFO_DEPTH": 1024},
           "tmem_counters": [COUNTERS]}
    (tmp_path / "status.json").write_text(json.dumps({"state": "passed", "revision": "test", "runs": [run]}))
    (tmp_path / "pmp0.log").write_bytes(log)
    (tmp_path / "pmp0-lens.json").write_text(json.dumps(lens))
    (tmp_path / "pmp0-tmemstat.log").write_text(COUNTERS + "\n")
    (tmp_path / "pmp0.jsonl").write_text(json.dumps({"cycles": 891870, "phases": {"other": {"cycles": 891870}}}) + "\n")
    return tmp_path


def test_reference_uses_frozen_parameters_not_stale_lens_uarch(reference):
    report, _, _ = A.load_reference(reference, 0, {"FIFO_DEPTH": 1024})
    assert report["parameters"]["TMEM_PMP"] == 0
    assert report["instruction_counters"]["VPU"]["frz"] == 65
    assert report["per_port"]["VPU_B"]["replay_loss_cycles"] == 13521
    assert report["cycle_correction"] == 0


@pytest.mark.parametrize("change, message", [
    ("log", "hash mismatch"), ("pmp", "PMP mismatch"), ("cycles", "cycle counts disagree"),
    ("correctness", "ISA correctness"), ("counters", "counters disagree"),
    ("parameter", "FIFO_DEPTH"), ("freeze", "invalid Lens instruction counter"),
])
def test_wrong_or_modified_reference_is_rejected(reference, change, message):
    status_file = reference / "status.json"
    status = json.loads(status_file.read_text())
    run = status["runs"][0]
    if change == "log":
        (reference / "pmp0.log").write_text("modified")
    elif change == "pmp":
        run["parameters"]["TMEM_PMP"] = 1
    elif change == "cycles":
        run["cycles"] += 1
    elif change == "correctness":
        run["bit_exact_isa"] = False
    elif change == "counters":
        run["tmem_counters"] = []
    elif change == "parameter":
        run["parameters"]["FIFO_DEPTH"] = 128
    elif change == "freeze":
        path = reference / "pmp0-lens.json"
        lens = json.loads(path.read_text())
        lens["instrs"][0][12]["frz"] = -1
        path.write_text(json.dumps(lens))
    status_file.write_text(json.dumps(status))
    with pytest.raises(ValueError, match=message):
        A.load_reference(reference, 0, {"FIFO_DEPTH": 1024})


def test_program_and_dynamic_dispatch_must_match():
    ins = SimpleNamespace(ra=0, rb=0, rc=0, rd=0, flags=1, op=32, w=[1, 2, 3])
    saved = [0, "MM", "", "", -1, [1, 2, 3], [0, 0, 0, 0], 1, 32]
    lens = {"programs": [[saved]], "instrs": [[0, 0, 0], [0, 1, 0]]}
    command = SimpleNamespace(pc=0, instruction=ins)
    A.verify_program(lens, [[ins]], [command, command], {32: 1})
    with pytest.raises(ValueError, match="dispatch order"):
        A.verify_program(lens, [[ins]], [command], {32: 1})
    saved[5] = [1, 2, 4]
    with pytest.raises(ValueError, match="PC 0"):
        A.verify_program(lens, [[ins]], [command, command], {32: 1})
