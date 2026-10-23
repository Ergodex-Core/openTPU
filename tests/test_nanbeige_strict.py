"""Strict placement, first-token selection, precision, replay and fail-closed gates."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu.isasim import Config
from opentpu.kernels.gather import gathered
from opentpu.llm.nanbeige import Engine, Spec
from opentpu.llm.nanbeige_strict import StrictEngine, plan, request_tokens
from test_nanbeige import tiny_weights


@pytest.fixture(scope="module")
def fixture():
    spec = Spec(hidden=128, layers=2, n_q=6, n_kv=1, head_dim=128,
                ffn=256, vocab=128, eos=(), eps=1e-5, theta=7e7)
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=8 << 20)
    return spec, cfg, tiny_weights(spec)


def oracle(spec, cfg, weights, prompt, budget, embedding="int8"):
    # Independent host-fed baseline, with exactly the candidate's embedding values.
    w = dict(weights)
    if embedding == "int8":
        w["model.embed_tokens.weight"] = gathered(w["model.embed_tokens.weight"], "int8", cfg.D)
    engine = Engine(spec, w, cap=128, cfg=cfg, rows=1, runtime_host=False, prog_cache=False)
    for token in prompt:
        logits = engine.step(token)
    out = []
    for i in range(budget):
        token = int(np.argmax(logits))
        out.append(token)
        if token in spec.eos:
            break
        if i + 1 < budget:
            logits = engine.step(token)
    return out


@pytest.mark.parametrize("embedding", ["int8", "f32"])
def test_full_loop_prefill_and_first_token_match_oracle_without_model_reads(fixture, embedding):
    spec, cfg, weights = fixture
    prompt = [7, 13, 2]
    expected = oracle(spec, cfg, weights, prompt, 4, embedding)
    eng = StrictEngine(spec, weights, cfg=cfg, backend="isa", embedding_format=embedding)
    reads, writes = [], []
    read, write = eng.backend.read, eng.backend.write
    out = eng.image.lookup["gen"]["out"]
    def guarded_read(sid, at, n):
        # Logits, residuals, KV, embedding and RoPE tensors never leave the device.
        assert out <= at and at + n <= out + 4 * (eng.cap + 1)
        reads.append((at, n))
        return read(sid, at, n)
    def tracked_write(sid, at, data):
        writes.append(at)
        return write(sid, at, data)
    eng.backend.read = guarded_read
    eng.backend.write = tracked_write
    streamed = []
    result = eng.generate_strict(prompt, 4, on_token=streamed.append)
    assert result["tokens"] == expected == streamed
    assert result["prefill_runs"] == len(prompt) - 1
    assert result["generation_runs"] >= 1
    assert eng.pos == len(prompt) + 4 - 1
    assert reads and eng.embed is None and eng._rope is None
    assert eng.image.io["x"] not in writes
    assert eng.image.io["cos"] not in writes and eng.image.io["sin"] not in writes
    assert not eng.image.embed_host and not eng.runtime_host
    # Reset replay must replace the entire logical KV history, including loop 2.
    assert eng.generate_strict([4, 19], 2)["tokens"] == oracle(spec, cfg, weights, [4, 19], 2, embedding)
    assert eng.generate_strict(prompt, 4)["tokens"] == expected


def test_fpga_eos_first_output_and_lowest_id_ties(fixture):
    spec, cfg, weights = fixture
    w = dict(weights)
    w["lm_head.weight"] = np.zeros_like(w["lm_head.weight"])
    eng = StrictEngine(replace(spec, eos=(0,)), w, cfg=cfg, backend="isa")
    result = eng.generate_strict([9], 5)
    assert result["tokens"] == [0] and result["stop_reason"] == "eos"
    assert result["prefill_runs"] == 0 and eng.pos == 1


def test_split_generate_is_equivalent_and_never_reads_logits(fixture):
    spec, cfg, weights = fixture
    eng = StrictEngine(spec, weights, cfg=cfg, backend="isa")
    eng.gen_split = True  # Production Nanbeige requires this existing device-only fallback.
    assert eng.generate_strict([2, 8], 3)["tokens"] == oracle(spec, cfg, weights, [2, 8], 3)
    assert all(isinstance(p, tuple) for p in eng._gens.values())


@pytest.mark.parametrize("prompt,budget", [([], 1), ([True], 1), ([1.25], 1),
    ([-1], 1), ([128], 1), ([0], True), ([0], 1.2), ([0], -1), ([0] * 129, 0), ([0, 1], 128)])
def test_invalid_requests_have_no_state_or_dma_effect(fixture, prompt, budget):
    spec, _, _ = fixture
    with pytest.raises(ValueError):
        request_tokens(prompt, budget, spec, 128)


def test_zero_output_does_not_access_device(fixture):
    spec, cfg, weights = fixture
    eng = StrictEngine(spec, weights, cfg=cfg, backend="isa")
    def deny(*args, **kwargs):
        pytest.fail("zero output touched device")
    eng.backend.read = eng.backend.write = eng.backend.run = deny
    eng.pos = 7
    result = eng.generate_strict([1], 0)
    assert result["tokens"] == [] and result["raw_runs"] == [] and eng.pos == 0


def test_mutated_host_paths_are_rejected_before_reset(fixture):
    spec, cfg, weights = fixture
    eng = StrictEngine(spec, weights, cfg=cfg, backend="isa")
    for name, value in (("gen_debug", True), ("runtime_host", True), ("embed", object()),
                        ("_rope", object()), ("_conv_lo", 1), ("row_server", object())):
        before = getattr(eng, name)
        setattr(eng, name, value)
        eng.pos = 9
        with pytest.raises(RuntimeError):
            eng.generate_strict([1], 1)
        assert eng.pos == 9
        setattr(eng, name, before)
    for method in (eng.step, eng.prefill, eng.run_rows, eng.prefill_chunks, eng.step_batch, eng.generate_batch):
        with pytest.raises(RuntimeError, match="host logits"):
            method([1])
    with pytest.raises(ValueError, match="greedy"):
        eng.generate([1], sampler=lambda _: 0)
    for kwargs in ({"sampling": object()}, {"rng": object()}, {"context": [1]}, {"stop_ids": []}):
        with pytest.raises(ValueError, match="greedy"):
            eng.generate_card(1, 1, **kwargs)


def test_production_fit_uses_device_embedding_and_split_programs():
    from pathlib import Path
    from opentpu.target import load, config
    cfg = config(load(str(Path(__file__).parents[1] / "tools/data/nanbeige/burst200.toml")))
    spec = Spec(3072, 22, 48, 8, 128, 10752, 166144, eps=1e-5, theta=7e7)
    report = plan(spec, cfg)
    assert report["required_bytes"] == 4239351808 < cfg.DRAM_BYTES
    assert report["programs"] == [{"bucket": 1, "prefill_words": [9568],
        "generate_words": [[9696], [840]], "split_generate": True}]
    with pytest.raises(MemoryError, match="host offload is forbidden"):
        plan(spec, cfg, embedding_format="f32")


def test_env_precision_override_rejected_before_backend(fixture, monkeypatch):
    spec, cfg, weights = fixture
    monkeypatch.setenv("OTPU_FORMATS", "mlp=int4")
    with pytest.raises(ValueError, match="uniform INT8"):
        StrictEngine(spec, weights, cfg=cfg, backend=lambda *a: pytest.fail("opened backend"))


def test_memory_failure_precedes_backend(fixture):
    spec, cfg, weights = fixture
    with pytest.raises(MemoryError):
        StrictEngine(spec, weights, cfg=replace(cfg, DRAM_BYTES=1024),
                     backend=lambda *a: pytest.fail("opened backend"))


def test_board_dual_lock_release_and_capability_guard(monkeypatch):
    from tools import nanbeige_strict as cli
    monkeypatch.delenv("OTPU_RUN_DIR", raising=False)
    events = []
    class Lock:
        def __init__(self, name, wait):
            assert name == "f2:0" and wait == 0
            events.append("legacy-take")
        def release(self):
            events.append("legacy-release")
    monkeypatch.setattr(cli, "DeviceLock", Lock)
    def fail_init(*args):
        events.append("canonical-busy")
        raise cli.DeviceBusy("f2-test", 17)
    monkeypatch.setattr(cli.BoardSession, "__init__", fail_init)
    with pytest.raises(cli.DeviceBusy):
        cli.StrictBoardSession("f2:0", None)
    assert events == ["legacy-take", "canonical-busy", "legacy-release"]


def test_late_capability_change_rejected_before_upload(monkeypatch):
    from tools import nanbeige_strict as cli
    from test_nanbeige_cli import _stub_board_session
    monkeypatch.delenv("OTPU_RUN_DIR", raising=False)
    healthy = {"caps": {"args": True, "gen": True, "mx": True}, "status": 0}
    profile, cfg, events = _stub_board_session(monkeypatch, [healthy, {"caps": {}}])
    lock = SimpleNamespace(release=lambda: events.append("legacy-release"))
    monkeypatch.setattr(cli, "DeviceLock", lambda name, wait: lock)
    session = cli.StrictBoardSession("f2:0", profile)
    try:
        with pytest.raises(ValueError, match="capability"):
            session.factory(cfg, [])
        assert "backend-upload" not in events and "quiesce" not in events
    finally:
        session.close()
    assert events[-2:] == ["transport-close", "legacy-release"]


@pytest.mark.parametrize("env,value", [("OTPU_STOP_RUN", "1"),
                                      ("OTPU_RUN_DIR", "/tmp/wrong"), ("OTPU_RUN_DIR", "")])
def test_force_and_alternative_locks_rejected_before_locking(monkeypatch, env, value):
    from tools import nanbeige_strict as cli
    monkeypatch.setenv(env, value)
    monkeypatch.setattr(cli, "DeviceLock", lambda *a, **k: pytest.fail("lock attempted"))
    with pytest.raises(ValueError):
        cli.StrictBoardSession("f2:0", None)


@pytest.mark.parametrize("patch", [{"caps": {"args": True, "mx": True}}, {"status": 2},
                                   {"status": 16}, {"status": 512}, {"status": 4}])
def test_board_missing_generate_or_error_never_passes_probe(monkeypatch, patch):
    from tools import nanbeige_strict as cli
    info = {"caps": {"args": True, "gen": True, "mx": True}, "status": 0, **patch}
    monkeypatch.setattr(cli.BoardSession, "_probe", lambda self: (info, None))
    session = object.__new__(cli.StrictBoardSession)
    with pytest.raises((ValueError, RuntimeError)):
        session._probe()


@pytest.mark.parametrize("payload", [{"prompt": "a", "temperature": 0.1},
    {"prompt": "a", "prompt_tokens": [1]}, {"prompt": ""}, {"prompt_tokens": [1], "max_new_tokens": True}])
def test_worker_request_contract_rejects_hybrid_or_lossy_inputs(fixture, payload):
    from tools.nanbeige_strict import encode_request
    with pytest.raises(ValueError):
        encode_request(payload, fixture[0], 128, "unused")


def stub_cli(monkeypatch, spec, cfg, fail=False):
    from tools import nanbeige_strict as cli
    from opentpu import progcache, imgcache
    events = []
    monkeypatch.setattr(cli.Spec, "from_hf", lambda _: spec)
    monkeypatch.setattr(cli, "plan", lambda *a: {"fits_dram": True, "fits_imem": True})
    monkeypatch.setattr(cli, "config", lambda _: cfg)
    monkeypatch.setattr(cli, "load_weights", lambda _: {})
    monkeypatch.setattr(progcache, "source", lambda: "test-source")
    monkeypatch.setattr(imgcache, "checkpoint", lambda _: [])
    class Session:
        def __init__(self, *args):
            events.append("open-board")
            self.cfg, self.info = cfg, {}
        def factory(self, *args):
            pytest.fail("the test engine must not allocate an image")
        def close(self):
            events.append("close-board")
    class Engine:
        def __init__(self, *args, backend, **kwargs):
            events.append("isa" if backend == "isa" else "board-factory")
            if fail:
                raise RuntimeError("injected device/compiler failure")
            self.spec, self.cap = spec, 128
        def generate_strict(self, tokens, budget, on_token=None):
            events.append("run")
            if on_token:
                on_token(2)
            return {"tokens": [2], "stop_reason": "length", "raw_runs": [{"cycles": 17}]}
    monkeypatch.setattr(cli, "StrictBoardSession", Session)
    monkeypatch.setattr(cli, "StrictEngine", Engine)
    return cli, events


@pytest.mark.parametrize("command", ["infer", "validate-isa"])
def test_cli_backend_is_explicit_and_receipt_labels_simulation(fixture, monkeypatch, capsys, command):
    import json
    cli, events = stub_cli(monkeypatch, *fixture[:2])
    assert cli.main([command, "--model-dir", "unused", "--prompt-tokens", "[1]", "--max-new-tokens", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    hardware = command == "infer"
    assert result["fpga_only"] is hardware
    assert result["hardware_executed"] is hardware
    assert result["cycles_are_hardware_measured"] is hardware
    assert result["source_hash_algorithm"] == "blake2b-128"
    assert events == (["open-board", "board-factory", "run", "close-board"] if hardware else ["isa", "run"])


def test_cli_device_failure_closes_session_without_fallback(fixture, monkeypatch):
    cli, events = stub_cli(monkeypatch, *fixture[:2], fail=True)
    with pytest.raises(RuntimeError, match="injected"):
        cli.main(["infer", "--model-dir", "unused", "--prompt-tokens", "[1]"])
    assert events == ["open-board", "board-factory", "close-board"]


def test_worker_rejects_sampling_then_streams_one_valid_request(fixture, monkeypatch, capsys):
    import io
    import json
    cli, events = stub_cli(monkeypatch, *fixture[:2])
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(
        '{"prompt_tokens":[1],"temperature":0.7}\n'
        '{"id":"r1","prompt_tokens":[1],"max_new_tokens":1}\n'))
    assert cli.main(["worker", "--model-dir", "unused"]) == 0
    output = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [x["event"] for x in output] == ["ready", "error", "token", "result"]
    assert output[-1]["id"] == "r1" and output[-1]["tokens"] == [2]
    assert events.count("run") == 1 and events[-1] == "close-board"
