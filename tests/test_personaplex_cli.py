"""Exercise the upstream server launch boundary without network or audio devices."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import numpy as np

pytest.importorskip("torch")


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "tools/personaplex_fpga.py"
    spec = importlib.util.spec_from_file_location("personaplex_fpga_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args():
    return SimpleNamespace(backend="board", threads=1, port=8998,
                           model_dir=Path("/model/PersonaPlex-7B-MLX-4bit"),
                           mimi_weight=Path("/codec/mimi.safetensors"),
                           voice_prompt_dir=Path("/voices"), static=Path("/static"))


@pytest.fixture
def server_environment(monkeypatch):
    from opentpu.llm import personaplex_bridge
    called = {}
    host = object()

    def load_host(weights, engine):
        called["host_load"] = (weights, engine)
        return host

    monkeypatch.setattr(personaplex_bridge, "load_host_model", load_host)
    loaders = ModuleType("moshi.models.loaders")
    def original_loader(*args, **kwargs):
        raise AssertionError("Original full CPU/GPU LM loader must not execute")
    loaders.get_moshi_lm = original_loader
    models = ModuleType("moshi.models")
    models.loaders = loaders
    moshi = ModuleType("moshi")
    moshi.models = models
    monkeypatch.setitem(sys.modules, "moshi", moshi)
    monkeypatch.setitem(sys.modules, "moshi.models", models)
    monkeypatch.setitem(sys.modules, "moshi.models.loaders", loaders)

    class Router:
        def __init__(self):
            self.routes = []
        def add_get(self, path, handler):
            self.routes.append(("GET", path, handler))
        def add_static(self, prefix, **kwargs):
            self.routes.append(("STATIC", prefix, None))

    class Application:
        def __init__(self, *args, **kwargs):
            self.router = Router()

    def original_run_app(app, **kwargs):
        called["run"] = (app, kwargs)

    web = ModuleType("aiohttp.web")
    web.Application = Application
    web.run_app = original_run_app
    web.json_response = lambda data: data
    aiohttp = ModuleType("aiohttp")
    aiohttp.web = web
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp)
    monkeypatch.setitem(sys.modules, "aiohttp.web", web)
    return SimpleNamespace(called=called, host=host, loaders=loaders, web=web,
                           original_loader=original_loader,
                           original_run_app=original_run_app, Application=Application)


def test_launch_uses_mlx_host_loader_and_loopback_before_static(cli, server_environment, monkeypatch):
    env = server_environment
    original_argv = sys.argv
    config = args()
    weights = object()
    engine = SimpleNamespace(report=lambda: {"backend": "board", "frames_measured": 3})

    def run_upstream(name, *, run_name):
        assert (name, run_name) == ("moshi.server", "__main__")
        argv = sys.argv
        # Mirrors the upstream default checkpoint-download branch: giving a
        # local moshi-weight prevents downloading/constructing the original LM.
        assert "--moshi-weight" in argv
        filename = argv[argv.index("--moshi-weight") + 1]
        assert filename == str(config.model_dir)
        assert argv[argv.index("--device") + 1] == "cpu"
        assert argv[argv.index("--tokenizer") + 1].endswith("tokenizer_spm_32k_3.model")
        assert env.loaders.get_moshi_lm(filename, device="cpu") is env.host
        for options in ({"device": "cuda"}, {"cpu_offload": True}, {"delays": [0]}):
            with pytest.raises(ValueError, match="CPU host"):
                env.loaders.get_moshi_lm(filename, **options)
        app = env.web.Application()
        app.router.add_get("/api/chat", lambda request: None)
        app.router.add_get("/", lambda request: None)
        app.router.add_static("/", path="/static")
        env.web.run_app(app, port=config.port)

    monkeypatch.setattr(cli.runpy, "run_module", run_upstream)
    cli.serve(config, weights, engine)
    assert env.called["host_load"] == (weights, engine)
    app, kwargs = env.called["run"]
    assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 8998
    status_indices = [i for i, (_, path, _) in enumerate(app.router.routes)
                      if path == "/api/fpga-status"]
    assert len(status_indices) == 1
    assert status_indices[0] < next(i for i, (kind, _, _) in enumerate(app.router.routes)
                                    if kind == "STATIC")
    handler = app.router.routes[status_indices[0]][2]
    assert asyncio.run(handler(None)) == engine.report()
    assert env.loaders.get_moshi_lm is env.original_loader
    assert env.web.run_app is env.original_run_app
    assert env.web.Application is env.Application
    assert sys.argv is original_argv


def test_upstream_exception_restores_process_globals(cli, server_environment, monkeypatch):
    env = server_environment
    original_argv = sys.argv
    def broken_upstream(*args, **kwargs):
        raise RuntimeError("codec load failure")
    monkeypatch.setattr(cli.runpy, "run_module", broken_upstream)
    with pytest.raises(RuntimeError, match="codec load failure"):
        cli.serve(args(), object(), object())
    assert env.loaders.get_moshi_lm is env.original_loader
    assert env.web.run_app is env.original_run_app
    assert env.web.Application is env.Application
    assert sys.argv is original_argv


def test_serving_refuses_simulation(cli):
    config = args()
    config.backend = "isa"
    with pytest.raises(ValueError, match="no CPU/ISA fallback"):
        cli.serve(config, object(), object())


def test_offline_launch_resolves_only_local_assets(cli, server_environment, monkeypatch, tmp_path):
    from opentpu.llm.personaplex_assets import MIMI_FILENAME, TOKENIZER_FILENAME, OfflineDownloadError
    config = args()
    config.assets_dir = tmp_path / "assets"
    config.model_dir = tmp_path / "model"
    for folder in (config.assets_dir, config.assets_dir / "voices",
                   config.assets_dir / "dist", config.model_dir):
        folder.mkdir(parents=True, exist_ok=True)
    for path, content in ((config.assets_dir / "config.json", "{}"),
                          (config.assets_dir / MIMI_FILENAME, "codec fixture"),
                          (config.assets_dir / "voices/NATF2.pt", "voice fixture"),
                          (config.assets_dir / "dist/index.html", "browser fixture"),
                          (config.model_dir / TOKENIZER_FILENAME, "tokenizer fixture")):
        path.write_text(content)
    hub = ModuleType("huggingface_hub")
    def no_network(*args, **kwargs):
        raise AssertionError("Original Hub resolver must never execute")
    hub.hf_hub_download = no_network
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    def upstream(name, *, run_name):
        assert Path(hub.hf_hub_download("nvidia/personaplex-7b-v1", "config.json")) == config.assets_dir / "config.json"
        with pytest.raises(OfflineDownloadError):
            hub.hf_hub_download("nvidia/personaplex-7b-v1", "model.safetensors")
        for flag, path in (("--mimi-weight", config.assets_dir / MIMI_FILENAME),
                           ("--voice-prompt-dir", config.assets_dir / "voices"),
                           ("--static", config.assets_dir / "dist"),
                           ("--tokenizer", config.model_dir / TOKENIZER_FILENAME)):
            assert sys.argv[sys.argv.index(flag) + 1] == str(path)
    monkeypatch.setattr(cli.runpy, "run_module", upstream)
    cli.serve(config, object(), object())
    assert hub.hf_hub_download is no_network


def test_preflight_reports_missing_inputs_without_board_access(cli, monkeypatch, capsys, tmp_path):
    import json
    from opentpu.isasim import Config
    from opentpu.llm import personaplex_temporal
    from opentpu import target
    spec = personaplex_temporal.Spec(hidden=64, layers=1, n_heads=2, ffn=128)
    cfg = Config(S=1, D=32, MCOLS=4, ACT_ROWS=8, ACT_BLOCKS=32,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, PAIR=True, DRAM_BYTES=1 << 22)
    monkeypatch.setattr(personaplex_temporal, "Spec", lambda: spec)
    monkeypatch.setattr(target, "config", lambda profile: cfg)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    def no_board(*args, **kwargs):
        raise AssertionError("Preflight must not open the FPGA")
    monkeypatch.setattr(cli, "BoardSession", no_board)
    monkeypatch.setattr(sys, "argv", ["personaplex_fpga.py", "preflight", "--cap", "4",
                                     "--assets-dir", str(tmp_path)])
    assert cli.main() == 2
    report = json.loads(capsys.readouterr().out)
    assert not report["source_preflight_passed"] and not report["hardware_validated"]
    assert report["layout"]["fits"] and report["program_words"]
    assert any("Missing runtime modules" in e for e in report["errors"])
    assert any("--assets-dir requires --model-dir" in e for e in report["errors"])


@pytest.mark.parametrize("bad", [np.zeros((0, 64), np.float32),
                                np.full((1, 64), np.nan, np.float32),
                                np.full((1, 64), 1e300, np.float64),
                                np.zeros((5, 64), np.float32),
                                np.zeros((1, 63), np.float32)])
def test_bad_benchmark_inputs_rejected_before_board(cli, monkeypatch, tmp_path, bad):
    from opentpu.llm import personaplex_weights, personaplex_temporal
    spec = personaplex_temporal.Spec(hidden=64, layers=1, n_heads=2, ffn=128)
    monkeypatch.setattr(personaplex_weights, "MLXWeights", lambda path: SimpleNamespace(spec=spec))
    def no_board(*args, **kwargs):
        raise AssertionError("Bad embeddings must fail before device access")
    monkeypatch.setattr(cli, "BoardSession", no_board)
    path = tmp_path / "inputs.npy"
    np.save(path, bad)
    monkeypatch.setattr(sys, "argv", ["personaplex_fpga.py", "benchmark", "--model-dir", str(tmp_path),
                                     "--embeddings", str(path), "--cap", "4"])
    with pytest.raises(ValueError, match="expected a .npy|finite floating-point"):
        cli.main()


def test_context_overflow_rejected_before_board(cli, monkeypatch, tmp_path):
    from opentpu.llm import personaplex_weights, personaplex_temporal
    spec = personaplex_temporal.Spec(hidden=64, layers=1, n_heads=2, ffn=128, context=4)
    monkeypatch.setattr(personaplex_weights, "MLXWeights", lambda path: SimpleNamespace(spec=spec))
    def no_board(*args, **kwargs):
        raise AssertionError("Context overflow must fail before device access")
    monkeypatch.setattr(cli, "BoardSession", no_board)
    monkeypatch.setattr(sys, "argv", ["personaplex_fpga.py", "benchmark", "--model-dir", str(tmp_path),
                                     "--embeddings", str(tmp_path / "unused.npy"), "--cap", "5"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.fixture
def burst200_probe(cli, monkeypatch):
    """Use the real Board register decoder, but never open or write a device."""
    from opentpu.host import board, regs as R
    from opentpu.host.fake import FakeTransport
    from opentpu.target import arch_hash, config, load
    profile = load(str(cli.PROFILE))
    cfg = config(profile)
    reported = {**profile.arch, **profile.uarch,
                "channels": profile.platform["channels_per_slice"],
                "channel_bytes": profile.platform["channel_bytes"],
                "platform": profile.platform_id, "hash": arch_hash(cfg)}
    transport = FakeTransport(ch_bytes=64, devname=None, regmap=5, cfg=reported,
                              D=cfg.D, MCOLS=cfg.MCOLS, LANES=cfg.LANES,
                              act_rows=cfg.ACT_ROWS, pair=cfg.PAIR, dstep=cfg.DSTEP,
                              stream=cfg.STREAM, seq_wc=cfg.SEQ_WC, args=True,
                              chash=True, w4=True, core_khz=profile.core_khz)
    overrides, events = {}, {"opened": [], "writes": [], "uploads": [], "closed": 0}
    original_read = transport.reg_read
    transport.reg_read = lambda off: overrides[off] if off in overrides else original_read(off)

    def forbidden_write(*args, **kwargs):
        events["writes"].append((args, kwargs))
        raise AssertionError("The qualification probe must be read-only")

    def close():
        events["closed"] += 1

    def open_transport(dev):
        events["opened"].append(dev)
        return transport

    def no_calibration(*args, **kwargs):
        raise AssertionError("The qualification probe must not invoke host calibration")

    def backend(cfg, images, *, transport, model, own_transport):
        events["uploads"].append((cfg, images, transport, model, own_transport))
        return SimpleNamespace(close=lambda: None)

    transport.reg_write = transport.mem_write = forbidden_write
    transport.close = close
    monkeypatch.setattr(board, "open_transport", open_transport)
    monkeypatch.setattr(board.memcal, "ensure", no_calibration)
    monkeypatch.setattr(board, "BoardBackend", backend)
    return SimpleNamespace(transport=transport, overrides=overrides, events=events, cfg=cfg)


@pytest.mark.parametrize("condition", ["unready0", "unready1", "error", "axi_error",
                                       "wait_timeout", "ecc_error", "loading",
                                       "running", "pending_writes"])
@pytest.mark.parametrize("after_probe", [False, True])
def test_unhealthy_board_cannot_reach_upload(cli, burst200_probe, condition, after_probe):
    from opentpu.host import regs as R
    fixture = burst200_probe
    ready = R.ST_CALIB0 | R.ST_CALIB1 | R.ST_WR_IDLE
    statuses = {
        "unready0": ready & ~R.ST_CALIB0, "unready1": ready & ~R.ST_CALIB1,
        "error": ready | R.ST_ERROR, "axi_error": ready | R.ST_AXI_ERR,
        "wait_timeout": ready | R.ST_WAIT_TO, "ecc_error": ready | R.ST_ECC_DED,
        "loading": ready | R.ST_LOADING, "running": ready | R.ST_RUN,
        "pending_writes": ready & ~R.ST_WR_IDLE,
    }
    session = cli.BoardSession("f2:0") if after_probe else None
    fixture.overrides[R.R_STATUS] = statuses[condition]
    try:
        with pytest.raises(RuntimeError, match="not calibrated|reports an error|not idle"):
            if session is None:
                session = cli.BoardSession("f2:0")
            session.factory(fixture.cfg, [b"must not upload"])
    finally:
        if session is not None:
            session.close()
    assert fixture.events["opened"] == ["f2:0"]
    assert fixture.events["closed"] == 1
    assert not fixture.events["writes"] and not fixture.events["uploads"]


@pytest.mark.parametrize("condition", ["id", "regmap", "platform", "hash", "missing_hash",
                                       "clock", "channels", "uarch", "config", "chash",
                                       "hostcal", "args", "w4"])
def test_incompatible_board_cannot_reach_upload(cli, burst200_probe, condition):
    from opentpu.host import regs as R
    fixture = burst200_probe
    read = fixture.transport.reg_read
    changes = {
        "id": (R.R_ID, 0), "regmap": (R.R_REGMAP, 3),
        "platform": (R.R_CFG_PLATFORM, 2), "hash": (R.R_CFG_HASH, 0x12345678),
        "missing_hash": (R.R_CFG_HASH, 0), "clock": (R.R_CORE_KHZ, 100000),
        "channels": (R.R_CFG_DRAM, read(R.R_CFG_DRAM) ^ (3 << 8)),
        "uarch": (R.R_CFG_UARCH, read(R.R_CFG_UARCH) ^ 1),
        "config": (R.R_VERSION, read(R.R_VERSION) ^ (1 << 8)),
        "chash": (R.R_CAPS, read(R.R_CAPS) & ~R.CAP_CHASH),
        "hostcal": (R.R_CAPS, read(R.R_CAPS) | R.CAP_HOSTCAL),
        "args": (R.R_CAPS, read(R.R_CAPS) & ~R.CAP_ARGS),
        "w4": (R.R_CAPS, read(R.R_CAPS) & ~R.CAP_W4),
    }
    reg, value = changes[condition]
    fixture.overrides[reg] = value
    with pytest.raises(RuntimeError):
        session = cli.BoardSession("f2:0")
        session.factory(fixture.cfg, [b"must not upload"])
    assert fixture.events["closed"] == 1
    assert not fixture.events["writes"] and not fixture.events["uploads"]


def test_non_burst200_profile_rejected_before_transport(cli, burst200_probe, monkeypatch, tmp_path):
    from opentpu.llm import personaplex_weights, personaplex_temporal
    # The stock F2 profile has matching compute dimensions but a 100 MHz DDR
    # platform. A matching Config alone must not qualify it as burst200.
    class HeadersOnly:
        spec = personaplex_temporal.Spec(hidden=64, layers=1, n_heads=2, ffn=128)

        @property
        def layers(self):
            raise AssertionError("Invalid board profile must fail before unpacking model tensors")

    monkeypatch.setattr(personaplex_weights, "MLXWeights", lambda path: HeadersOnly())
    inputs = tmp_path / "inputs.npy"
    np.save(inputs, np.zeros((1, 64), np.float32))
    monkeypatch.setattr(sys, "argv", ["personaplex_fpga.py", "benchmark", "--profile", "f2",
                                     "--model-dir", str(tmp_path), "--embeddings", str(inputs),
                                     "--cap", "4"])
    with pytest.raises(ValueError, match="recorded burst200 profile"):
        cli.main()
    assert not burst200_probe.events["opened"]


@pytest.mark.parametrize("wformat,halted", [("int4", False), ("int4", True), ("int8", False)])
def test_qualified_board_reaches_backend_only_after_read_only_probe(cli, burst200_probe,
                                                                  wformat, halted):
    from opentpu.host import regs as R
    fixture = burst200_probe
    if halted:
        fixture.overrides[R.R_STATUS] = (R.ST_RUN | R.ST_HALTED | R.ST_WR_IDLE
                                         | R.ST_CALIB0 | R.ST_CALIB1)
    if wformat == "int8":
        fixture.transport.w4 = False
    session = cli.BoardSession("f2:0", wformat=wformat)
    assert not fixture.events["uploads"] and not fixture.events["writes"]
    images = [b"qualified image"]
    session.factory(fixture.cfg, images)
    assert fixture.events["uploads"] == [(fixture.cfg, images, fixture.transport,
                                           "PersonaPlex temporal", True)]
    session.close()
    assert not fixture.events["writes"]
    assert fixture.events["closed"] == 1


def test_backend_configuration_cannot_change_after_qualification(cli, burst200_probe):
    from dataclasses import replace
    fixture = burst200_probe
    session = cli.BoardSession("f2:0")
    try:
        with pytest.raises(ValueError, match="backend configuration differs"):
            session.factory(replace(fixture.cfg, DRAM_BYTES=1 << 30), [b"must not upload"])
    finally:
        session.close()
    assert not fixture.events["writes"] and not fixture.events["uploads"]


def test_timed_engine_preserves_ordered_counters_and_reset(cli, monkeypatch):
    class Engine:
        backend = SimpleNamespace(info={"core_khz": 200000})
        last_stats = {}
        pos = 0

        def step(self, value):
            self.last_stats = {"frame": self.pos, "cycles": (self.pos + 1) * 200000,
                               "instructions": [self.pos + 7], "wall_seconds": 0.25}
            self.pos += 1
            return value

        def reset(self):
            self.pos = 0
            self.last_stats = {}

    clock = iter([10., 12., 20., 23.])
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(clock))
    raw = Engine()
    timed = cli.TimedEngine(raw, "board")
    assert timed.step("first") == "first"
    assert timed.step("second") == "second"
    raw.last_stats["instructions"][0] = 999
    report = timed.report()
    assert [run["engine_stats"]["frame"] for run in report["runs"]] == [0, 1]
    assert [run["engine_stats"]["cycles"] for run in report["runs"]] == [200000, 400000]
    assert [run["engine_stats"]["instructions"] for run in report["runs"]] == [[7], [8]]
    assert [run["host_round_trip_seconds"] for run in report["runs"]] == [2., 3.]
    assert [run["device_ms"] for run in report["runs"]] == [1., 2.]
    assert report["temporal_mean_ms"] == 2500. and report["device_time_measured"]
    assert "host round-trip" in report["timing_scope"]
    assert "excludes host transfers" in report["device_timing_scope"]
    timed.reset()
    assert raw.pos == 0 and timed.report()["runs"] == []
    assert timed.report()["frames_measured"] == 0
    assert timed.report()["temporal_mean_ms"] is None
    assert not timed.report()["device_time_measured"]


@pytest.mark.parametrize("backend,core_khz", [("isa", 200000), ("board", None), ("board", 0),
                                              ("board", -1), ("board", True), ("board", float("nan"))])
def test_device_timing_requires_board_and_positive_probed_clock(cli, backend, core_khz):
    engine = SimpleNamespace(backend=SimpleNamespace(info={"core_khz": core_khz}),
                             last_stats={"cycles": 200000, "core_khz": 200000},
                             step=lambda value: value)
    timed = cli.TimedEngine(engine, backend)
    timed.step(np.zeros(4))
    report = timed.report()
    assert not report["device_time_measured"]
    assert "device_ms" not in report["runs"][0]
    assert report["runs"][0]["engine_stats"]["cycles"] == 200000


@pytest.mark.parametrize("condition", ["existing", "existing_without_suffix", "missing_parent"])
@pytest.mark.parametrize("backend", ["board", "isa"])
def test_bad_output_is_rejected_before_backend_and_preserved(cli, monkeypatch, tmp_path,
                                                           condition, backend):
    from opentpu.llm import personaplex_temporal, personaplex_weights

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid output must fail before model/backend construction")

    monkeypatch.setattr(personaplex_weights, "MLXWeights", forbidden)
    monkeypatch.setattr(personaplex_temporal, "Engine", forbidden)
    monkeypatch.setattr(cli, "BoardSession", forbidden)
    requested = tmp_path / ("hidden" if condition == "existing_without_suffix" else "hidden.npy")
    actual = tmp_path / "hidden.npy"
    if condition == "missing_parent":
        requested = tmp_path / "missing" / "hidden.npy"
        error = FileNotFoundError
    else:
        actual.write_bytes(b"preserved evidence")
        error = FileExistsError
    monkeypatch.setattr(sys, "argv", ["personaplex_fpga.py", "benchmark", "--backend", backend,
                                     "--model-dir", str(tmp_path), "--embeddings", str(tmp_path / "unused.npy"),
                                     "--output", str(requested)])
    with pytest.raises(error, match="Benchmark output"):
        cli.main()
    if condition == "missing_parent":
        assert not requested.parent.exists()
    else:
        assert actual.read_bytes() == b"preserved evidence"


@pytest.mark.parametrize("formats", ["", "attn@0-3=int8,mlp@0-3=int8,attn@31=int8,mlp@31=int8"])
def test_benchmark_final_output_creation_is_exclusive(cli, monkeypatch, tmp_path, formats):
    from opentpu import target
    from opentpu.llm import personaplex_temporal, personaplex_weights

    output = tmp_path / "hidden.npy"
    spec = SimpleNamespace(hidden=4, context=4)
    monkeypatch.setattr(personaplex_weights, "MLXWeights",
                        lambda path: SimpleNamespace(spec=spec, layers=[], final_norm=[]))
    monkeypatch.setattr(target, "config", lambda profile: object())
    monkeypatch.setattr(cli, "inspect_layout", lambda *args, **kwargs: {"fits": True})
    closed = []

    class Engine:
        last_stats = {"frame": 0, "instructions": [1]}

        def __init__(self, *args, **kwargs):
            assert kwargs["formats"] == formats

        def step(self, value):
            output.write_bytes(b"another writer's evidence")
            return value

        def close(self):
            closed.append(True)

    monkeypatch.setattr(personaplex_temporal, "Engine", Engine)
    inputs = tmp_path / "inputs.npy"
    np.save(inputs, np.ones((1, 4), np.float32))
    monkeypatch.setattr(sys, "argv", ["personaplex_fpga.py", "benchmark", "--backend", "isa",
                                     "--model-dir", str(tmp_path), "--embeddings", str(inputs),
                                     "--output", str(output), "--cap", "4", "--formats", formats])
    with pytest.raises(FileExistsError):
        cli.main()
    assert output.read_bytes() == b"another writer's evidence" and closed == [True]
