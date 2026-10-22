"""Directed native boundary timing cases; RTL trace comparison is separate."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import pytest

from opentpu import rtlsim
from tools.wideport_native_memory import (
    NativeCommand, NativeInput, NativeMemory, NativeMemoryConfig,
    READ_JITTER, WRITE_JITTER, deterministic_draw, logical_beat,
)


def _idle(n=2):
    return (NativeInput(),) * n


def _start(model):
    model.step(_idle(model.config.channels), reset=True)


def test_nch8_pair_and_half_mapping_is_bijective_per_interleave_group():
    for channel_beat in range(8):
        mapped = {logical_beat(channel_beat, c, 8) for c in range(8)}
        assert len(mapped) == 8
        assert mapped == set(range(channel_beat * 8, channel_beat * 8 + 8))
    assert logical_beat(0, 0, 2) == 0
    assert logical_beat(1, 0, 2) == 3  # parity hash swaps the half
    assert logical_beat(1, 0, 2, chash=False) == 2


def test_read_accepted_then_performed_then_returned_without_observed_response():
    cfg = NativeMemoryConfig(channels=2, seed=7, stall=0, lat=2)
    model = NativeMemory(cfg)
    _start(model)                         # edge 0, also clocks ready registers
    cmd = NativeCommand(False, 5, "r0")
    edge = model.step((NativeInput(cmd), NativeInput()))  # edge 1
    assert edge.accepted_commands == (cmd, None)
    assert edge.performed == (None, None)  # cannot perform on acceptance edge
    edge = model.step(_idle())            # edge 2
    assert edge.performed == (cmd, None)
    due = 2 + cfg.lat + deterministic_draw(cfg.seed, 2, 0, READ_JITTER) % 8
    arrivals = []
    for _ in range(due + 4):
        snap = model.snapshot()
        if snap.read_valid[0]:
            arrivals.append((snap.cycle, snap.read_tag[0]))
        model.step(_idle())
    assert arrivals == [(due + 1, "r0")]
    assert model.snapshot().write_done == (0, 0)


def test_old_write_blocks_later_read_until_same_edge_write_data_arrives():
    model = NativeMemory(NativeMemoryConfig(channels=2, stall=0, lat=0, write_done_lag=0))
    _start(model)
    wr, rd = NativeCommand(True, 3, "w"), NativeCommand(False, 4, "r")
    model.step((NativeInput(wr), NativeInput()))  # edge 1: command only
    assert model.step((NativeInput(rd), NativeInput())).performed[0] is None  # edge 2
    assert model.step(_idle()).performed[0] is None  # edge 3: no write data
    edge = model.step((NativeInput(write_data=True), NativeInput()))  # edge 4
    assert edge.accepted_write_data == (True, False)
    assert edge.performed[0] == wr  # newly accepted data is visible to perform
    assert model.step(_idle()).performed[0] == rd  # edge 5: in-order read
    write_due = 4 + deterministic_draw(model.config.seed, 4, 0, WRITE_JITTER) % 8
    while model.cycle <= write_due:
        model.step(_idle())
    assert model.snapshot().write_done[0] == 1


def test_command_depth_and_credit_are_separate_backpressure_sources():
    model = NativeMemory(NativeMemoryConfig(channels=2, stall=0, bandwidth=33,
                                            command_depth=1))
    _start(model)
    cmd = NativeCommand(False, 1, "slow")
    assert model.step((NativeInput(cmd), NativeInput())).accepted_commands[0] == cmd  # c1
    assert not model.snapshot().command_ready[0]
    assert model.step(_idle()).performed[0] is None  # c2, credit 66
    assert model.step(_idle()).performed[0] is None  # c3, credit 99
    assert model.step(_idle()).performed[0] == cmd   # c4, credit 132
    assert model.snapshot().command_ready[0]


def test_native_reset_clears_pending_responses_not_free_running_cycle():
    model = NativeMemory(NativeMemoryConfig(channels=2, stall=0, lat=20))
    _start(model)
    cmd = NativeCommand(False, 2, "discard")
    model.step((NativeInput(cmd), NativeInput()))
    model.step(_idle())  # performed, long response still pending
    assert model.read_due[0]
    reset_edge = model.step(_idle(), reset=True)
    assert reset_edge.before.cycle == 3
    assert model.snapshot().cycle == 4
    assert not model.read_due[0] and not model.snapshot().read_valid[0]
    assert model.snapshot().command_ready[0]


def test_legacy_random_and_ddr3_modes_fail_closed():
    with pytest.raises(ValueError, match="opt-in"):
        NativeMemoryConfig(deterministic=False)
    with pytest.raises(ValueError, match="axi_dram=0"):
        NativeMemoryConfig(dram=1)
    with pytest.raises(ValueError, match="beyond logical memory"):
        model = NativeMemory(NativeMemoryConfig(channels=2, stall=0, words=32))
        _start(model)
        model.step((NativeInput(NativeCommand(False, 9)), NativeInput()))


@pytest.mark.parametrize("stall,lat", [(0, 8), (50, 120)])
def test_native_memory_feedback_rtl(have_verilator, monkeypatch, tmp_path, stall, lat):
    """A saved pre-edge RTL trace verifies generated responses from requests alone.

    One binary serves both timing cases. The predictor reads only cv/we/addr,
    wv and reset as inputs; cr/wr/rv/wdone are comparison targets.
    """
    from tools.wideport_memory_feedback import validate_native_trace

    evidence = Path(os.environ.get("WP_NATIVE_EVIDENCE", tmp_path)).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    keep = Path(tempfile.mkdtemp(prefix=f"native-s{stall}-l{lat}-", dir=evidence))
    trace = keep / "native-feedback.jsonl"
    sources = [rtlsim.TB / "otpu_native_mem.sv", rtlsim.TB / "tb_native_memory_feedback.sv"]
    flags = rtlsim.VFLAGS.copy()
    flags[flags.index("-j") + 1] = "4"
    monkeypatch.setattr(rtlsim, "VFLAGS", flags)
    exe = rtlsim.build("tb_native_memory_feedback", sources)
    plusargs = [f"+dir={keep}", "+nat_deterministic=1", "+axi_seed=7",
                f"+axi_stall={stall}", f"+axi_lat={lat}", "+axi_bw=100",
                f"+nat_feedback={trace}", "+nat_feedback_limit=10000"]
    receipt = {
        "top": "tb_native_memory_feedback", "channels": 8, "words": 16384,
        "command_count_per_channel": 40, "write_count_per_channel": 8,
        "plusargs": plusargs,
        "source_sha256": {str(p.relative_to(rtlsim.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
    }
    (keep / "inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result = rtlsim.run_sim([str(exe), *plusargs], timeout=300)
    (keep / "sim.stdout.log").write_text(result.stdout)
    (keep / "sim.stderr.log").write_text(result.stderr)
    assert result.returncode == 0 and "TEST_PASS native memory feedback" in result.stdout, \
        result.stdout[-3000:] + result.stderr[-3000:]
    assert trace.is_file() and trace.stat().st_size, "missing native feedback trace"
    with trace.open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    assert records[0]["kind"] == "config" and records[0]["schema"] == 1
    assert records[0]["channels"] == 8 and records[0]["deterministic"] == 1
    assert records[0]["stall"] == stall and records[0]["lat"] == lat
    assert records[-1]["kind"] == "end", "native feedback trace incomplete"
    assert not any(row["kind"] == "truncated" for row in records)
    summary = validate_native_trace(trace)
    assert summary["cycles_checked"] == records[-1]["cycles"] == len(records) - 2
    assert summary["commands"] == 8 * 40
    assert summary["read_responses"] == 8 * 32
    assert summary["writes_done"] == 8 * 8
    assert summary["mismatches"] == 0
    assert summary["calibrated"] is False and summary["tok_s"] is None
    # Deliberately corrupt an observed result: a response must be an oracle,
    # never an input that could make the predictor reproduce the corruption.
    damaged = json.loads(json.dumps(records))
    result_row = next(row for row in damaged[1:-1] if row["channels"][0]["rv"])
    result_row["channels"][0]["rv"] = 0
    corrupted = keep / "negative-response-oracle.jsonl"
    corrupted.write_text("".join(json.dumps(row) + "\n" for row in damaged))
    with pytest.raises(ValueError, match="native.channel0.rv"):
        validate_native_trace(corrupted)
    (keep / "feedback-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
