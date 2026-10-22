"""CLI safety, measurement accounting and reset contracts without live hardware."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import pytest


_path = Path(__file__).parents[1] / "tools/nanbeige_fpga.py"
_spec = importlib.util.spec_from_file_location("nanbeige_cli", _path)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _profile_info():
    from opentpu.target import config, load
    profile = load(str(cli.PROFILE))
    info = {"cfg": {"platform": profile.platform_id, "channels": 2,
                    "channel_bytes": 1 << 31}, "core_khz": 200000,
            "calibrated": True, "running": False}
    return profile, config(profile), info


def test_board_must_match_burst200_before_programming():
    profile, cfg, info = _profile_info()
    cli.validate_board(info, cfg, profile)
    with pytest.raises(ValueError, match="architecture"):
        cli.validate_board(info, replace(cfg, MCOLS=8), profile)
    with pytest.raises(ValueError, match="kHz"):
        cli.validate_board(dict(info, core_khz=100000), cfg, profile)
    with pytest.raises(RuntimeError, match="already running"):
        cli.validate_board(dict(info, running=True), cfg, profile)
    with pytest.raises(ValueError, match="calibrated"):
        cli.validate_board(dict(info, calibrated=False), cfg, profile)


def _stub_board_session(monkeypatch, states):
    from opentpu.host import board
    profile, cfg, healthy = _profile_info()
    events = []
    states = iter(states)
    transport = SimpleNamespace(close=lambda: events.append("transport-close"))

    class Probe:
        def __init__(self, supplied, *, lock=True, calibrate=True):
            assert supplied is transport
            events.append(("probe", lock, calibrate))
            if calibrate:
                events.append("calibration-write")
            if lock:
                events.append("quiesce")

        def info(self):
            return dict(healthy, **next(states))

    class Backend:
        def __init__(self, actual, images, **kwargs):
            assert actual == cfg and kwargs["transport"] is transport
            events.append("backend-upload")

        def close(self):
            events.append("backend-close")

    monkeypatch.setattr(board, "open_transport", lambda dev: transport)
    monkeypatch.setattr(board, "Board", Probe)
    monkeypatch.setattr(board, "device_config", lambda info: cfg)
    monkeypatch.setattr(board, "BoardBackend", Backend)
    return profile, cfg, events


@pytest.mark.parametrize("state, error, message", [
    ({"running": True}, RuntimeError, "already running"),
    ({"calibrated": False}, ValueError, "calibrated"),
    ({"core_khz": 100000}, ValueError, "kHz"),
])
def test_initial_board_refusal_does_not_calibrate_or_upload(monkeypatch, state, error, message):
    profile, cfg, events = _stub_board_session(monkeypatch, [state])
    with pytest.raises(error, match=message):
        cli.BoardSession("f2:0", profile)
    assert events == [("probe", False, False), "transport-close"]


@pytest.mark.parametrize("state, error, message", [
    ({"running": True}, RuntimeError, "already running"),
    ({"calibrated": False}, ValueError, "calibrated"),
    ({"core_khz": 100000}, ValueError, "kHz"),
])
def test_board_state_rechecked_before_backend_upload(monkeypatch, state, error, message):
    profile, cfg, events = _stub_board_session(monkeypatch, [{}, state])
    session = cli.BoardSession("f2:0", profile)
    try:
        with pytest.raises(error, match=message):
            session.factory(cfg, [])
        assert events == [("probe", False, False), ("probe", False, False)]
    finally:
        session.close()


def test_healthy_board_rechecked_without_calibration_before_upload(monkeypatch):
    profile, cfg, events = _stub_board_session(monkeypatch, [{}, {}])
    session = cli.BoardSession("f2:0", profile)
    try:
        session.factory(cfg, [])
        assert events == [("probe", False, False), ("probe", False, False), "backend-upload"]
    finally:
        session.close()
    assert events[-2:] == ["backend-close", "transport-close"]


def test_factory_rejects_changed_config_before_upload(monkeypatch):
    profile, cfg, events = _stub_board_session(monkeypatch, [{}])
    session = cli.BoardSession("f2:0", profile)
    try:
        with pytest.raises(ValueError, match="configuration differs"):
            session.factory(replace(cfg, MCOLS=8), [])
        assert events == [("probe", False, False)]
    finally:
        session.close()


class Target:
    def __init__(self):
        self.spec = SimpleNamespace(eos=(9,))
        self.image = SimpleNamespace(rows=4)
        self.cap, self.pos, self.stats = 20, 0, []
        self.resets = 0

    def reset(self):
        self.pos = 0
        self.resets += 1

    def _logits(self, pos):
        result = np.zeros(10, np.float32)
        result[min(pos + 1, 9)] = 10
        return result

    def step(self, token):
        self.pos += 1
        self.stats.append({"cycles": 100})
        return self._logits(self.pos)

    def run_rows(self, rows, tokens, logit_rows):
        self.stats.append({"cycles": 150})
        return np.array([self._logits(rows[r][1] + 1) for r in logit_rows])


def test_target_measurement_excludes_first_token_from_decode_and_resets():
    target = Target()
    first = cli.run_target(target, [1, 2, 3], 4)
    second = cli.run_target(target, [1, 2, 3], 4)
    assert first["tokens"] == second["tokens"] == [4, 5, 6, 7]
    assert target.resets == 2
    assert first["generated_tokens"] == 4 and first["decode_tokens"] == 3
    assert first["decode_tps"] == 3 / first["decode_seconds"]
    assert first["total_seconds"] >= first["prefill_seconds"]
    assert first["stop_reason"] == "length"
    raw = target.stats[:first["target_prefill_runs"] + first["target_decode_runs"]]
    measured = cli.backend_evidence(raw, "isa", first["target_prefill_runs"])
    assert measured["target_prefill_cycles"] == 150
    assert measured["target_decode_cycles"] == 300
    assert measured["target_cycles"] == 450


def test_target_respects_eos_capacity_and_zero_generation():
    target = Target()
    result = cli.run_target(target, [1] * 7, 10)
    assert result["tokens"] == [8, 9] and result["stop_reason"] == "eos"
    target.cap = 3
    result = cli.run_target(target, [1, 2, 3], 4)
    assert result["tokens"] == [4] and result["stop_reason"] == "capacity"
    assert cli.run_target(target, [1], 0)["tokens"] == []


def test_cycles_never_presented_as_hardware_for_isa():
    result = cli.backend_evidence([{"cycles": 12}, {"cycles": np.int64(5)}], "isa")
    assert result["target_cycles"] == 17
    assert result["cycles_are_hardware_measured"] is False
    assert cli.backend_evidence([{"cycles": 12}, {}], "board")["target_cycles"] is None
    result = cli.backend_evidence([{"cycles": 12}, None], "board", 1)
    assert result["target_prefill_cycles"] == 12
    assert result["target_decode_cycles"] is None
    assert "excludes draft" in result["target_cycles_scope"]


def test_atomic_json_and_numpy_values(tmp_path, capsys):
    path = tmp_path / "run.json"
    cli.write_json(path, {"tokens": np.array([1, 2]), "cycles": np.int64(42)})
    assert json.loads(path.read_text()) == {"tokens": [1, 2], "cycles": 42}
    assert len(list(tmp_path.iterdir())) == 1
    assert json.loads(capsys.readouterr().out)["cycles"] == 42


def test_cli_defaults_and_missing_draft_rejected():
    args = cli.parser().parse_args(["benchmark", "--model-dir", "/tmp/model"])
    assert args.backend == "board" and args.device == "f2:0"
    assert args.cap == 128 and args.max_new_tokens == 16 and args.draft_tokens == 3
    assert args.warmup_runs == 0 and args.repeat == 1 and args.draft_backend == "cpu"
    with pytest.raises(SystemExit):
        cli.main(["benchmark", "--model-dir", "/tmp/model"])
    for options in (["--repeat", "0"], ["--warmup-runs", "-1"]):
        with pytest.raises(SystemExit):
            cli.main(["benchmark", "--mode", "target", "--model-dir", "/tmp/model", *options])


def test_warmups_are_excluded_and_medians_keep_last_run_compatibility():
    calls, count = [], {"target": 0, "hybrid": 0}
    values = [999., 1., 3., 2.]
    def run(mode):
        calls.append(mode)
        value = values[count[mode]]
        count[mode] += 1
        return {"tokens": [1, 2], "decode_tps": value,
                "target_cycles": int(value * 100), "target_prefill_cycles": 10,
                "target_decode_cycles": int(value * 100) - 10}
    result = cli.collect_runs(run, ["target", "hybrid"], warmup_runs=1, repeat=3)
    assert calls == ["target", "hybrid"] * 4
    assert result["runs"] == result["measured_runs"][-1]["runs"]
    assert result["runs"]["target"]["decode_tps"] == 2
    assert result["aggregates"]["target"]["decode_tps_median"] == 2
    assert result["aggregates"]["target"]["target_cycles_median"] == 200
    assert result["aggregates"]["target"]["samples"] == 3
    assert result["warmup_seconds"] >= 0
    assert len(result["warmups"]) == 1


@pytest.mark.parametrize("draft_backend", ["cpu", "board"])
def test_repeated_benchmark_resets_and_keeps_per_run_cycle_scope(monkeypatch, draft_backend):
    import opentpu.llm.nanbeige as nanbeige
    import opentpu.llm.nanbeige_weights as weight_module
    import opentpu.llm.dspark as dspark
    import opentpu.llm.nanbeige_specdec as specdec
    target = Target()
    target.cfg = _profile_info()[1]
    monkeypatch.setattr(nanbeige, "Engine", lambda *a, **k: target)
    monkeypatch.setattr(nanbeige, "load_weights", lambda path: {})
    monkeypatch.setattr(weight_module, "FloatSafetensorWeights", lambda path: {})
    tokenizer = SimpleNamespace(decode=lambda tokens, **kw: str(list(tokens)))
    monkeypatch.setattr(cli, "_tokenize", lambda *a: (tokenizer, [1, 2, 3]))

    class Draft:
        backend = "test-draft"
        precision = {"backbone": "W4A8", "host": "fusion and Markov"}
        def __init__(self, *a, **kw):
            self.stats = []
    monkeypatch.setattr(dspark, "Reference", Draft)
    module = types.ModuleType("opentpu.llm.dspark_fpga")
    module.Engine = Draft
    monkeypatch.setitem(sys.modules, "opentpu.llm.dspark_fpga", module)

    class FakeDecoder:
        def __init__(self, engine, draft, **kw):
            self.engine, self.draft = engine, draft
        def generate(self, prompt, max_new_tokens):
            start = len(self.engine.stats)
            result = cli.run_target(self.engine, prompt, max_new_tokens)
            if draft_backend == "board":
                self.draft.stats.extend([{"cycles": 40}, {"cycles": 50}])
            stats = specdec.Stats("fake", "test-draft", generated_tokens=result["generated_tokens"],
                                 decode_tokens=result["decode_tokens"],
                                 prefill_seconds=result["prefill_seconds"],
                                 decode_seconds=result["decode_seconds"],
                                 total_seconds=result["total_seconds"],
                                 target_prefill_runs=result["target_prefill_runs"],
                                 target_decode_runs=result["target_decode_runs"],
                                 target_run_stats=self.engine.stats[start:])
            return specdec.Result(tuple(result["tokens"]), stats, result["stop_reason"])
    monkeypatch.setattr(specdec, "Decoder", FakeDecoder)
    args = cli.parser().parse_args(["benchmark", "--model-dir", "/tmp/model",
                                   "--draft-dir", "/tmp/draft", "--backend", "isa",
                                   "--draft-backend", draft_backend, "--max-new-tokens", "4",
                                   "--cap", "20", "--warmup-runs", "1", "--repeat", "3"])
    args.rows = 4
    spec = SimpleNamespace(vocab=10, capture_layers=(1,))
    report = cli.benchmark(args, spec, _profile_info()[0], SimpleNamespace(block_size=7))
    assert target.resets == 8  # Both modes, one warmup plus three measured repetitions.
    assert report["comparison"]["exact_target_tokens_match_by_repeat"] == [True] * 3
    assert report["comparison"]["exact_target_tokens_match"]
    assert report["fpga_draft_implemented"] == (draft_backend == "board")
    assert not report["fpga_draft_hardware_executed"]  # Even board draft uses ISA here.
    assert not report["draft_fully_on_fpga"]
    for repetition in report["measured_runs"]:
        for mode in ("target", "hybrid"):
            result = repetition["runs"][mode]
            assert result["target_cycles"] == 450
            assert result["target_prefill_cycles"] == 150
            assert result["target_decode_cycles"] == 300
            assert not result["cycles_are_hardware_measured"]
        assert repetition["runs"]["hybrid"]["draft_cycles"] == (90 if draft_backend == "board" else 0)


def test_inspection_compiles_without_loading_weights():
    from opentpu.llm.nanbeige import Spec
    _, cfg, _ = _profile_info()
    spec = Spec(hidden=128, layers=1, n_q=2, n_kv=1, head_dim=128,
                ffn=256, vocab=256, loops=2)
    layout = cli.inspect_layout(spec, cfg, 128, 2)
    assert layout["fits_dram"]
    assert layout["weights_copied_per_physical_layer"] == 1
    assert layout["resident"] is False
    assert layout["embedding_precision"] == "source BF16 values expanded to FP32"
    assert len(layout["programs"]) == 4
    assert all(p["fits_imem"] for p in layout["programs"])
    assert layout["physical_layers"] == 1 and layout["logical_layers"] == 2
