#!/usr/bin/env python3
"""PersonaPlex temporal transformer on openTPU; CPU audio/depth runtime.

inspect needs no model download or board. benchmark consumes saved summed input
embeddings, and serve requires the NVIDIA/personaplex version of moshi.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import replace
import importlib.util
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROFILE = Path(__file__).parent / "data/personaplex/burst200.toml"


def _board_profile(profile):
    """Reject unsupported profiles without opening a transport or building an image."""
    from opentpu.target import config, load
    baseline, selected = load(str(PROFILE)), load(str(profile))
    if (config(selected) != config(baseline) or selected.kind != baseline.kind
            or selected.platform_id != baseline.platform_id
            or selected.core_khz != baseline.core_khz
            or selected.uarch != baseline.uarch
            or any(selected.platform.get(key) != baseline.platform.get(key)
                   for key in ("mem", "mem_ctrl", "hostcal", "channels_per_slice",
                               "channel_bytes", "beat_bytes"))):
        raise ValueError("board execution requires the recorded burst200 profile")
    return selected


class BoardSession:
    """Hold the transport lock and qualify burst200 before any device writes."""

    def __init__(self, dev, *, profile=PROFILE, wformat="int4", weight_formats=None):
        from opentpu.host.board import open_transport
        from opentpu.target import config
        selected = _board_profile(profile)
        if wformat not in ("int4", "int8"):
            raise ValueError("board weights must use int4 or int8")
        # The base format can be INT8 while explicit layer overrides use INT4.
        # Callers that omit resolved formats retain the original uniform API.
        try:
            resolved = frozenset((wformat,) if weight_formats is None else weight_formats)
        except TypeError as exc:
            raise ValueError("weight_formats must contain resolved int4/int8 formats") from exc
        if not resolved or not resolved <= {"int4", "int8"}:
            raise ValueError("weight_formats must contain resolved int4/int8 formats")
        self.weight_formats = resolved
        self.profile, self.wformat = selected, wformat
        self.expected_cfg = config(selected)
        self.transport = open_transport(dev)
        self.backend = None
        try:
            self.info, self.cfg = self._probe()
        except BaseException:
            self.close()
            raise

    def _probe(self):
        from opentpu.host import regs as R
        from opentpu.host.board import Board, ConfigMismatch, device_config
        from opentpu.target import arch_hash
        # In particular, do not invoke host DDR calibration on a different image.
        # The transport already owns the device lock; Board's lock/quiesce path
        # can stop a previous run, so this probe must not take that path either.
        info = Board(self.transport, lock=False, calibrate=False).info()
        reported, caps = info.get("cfg") or {}, info.get("caps") or {}
        expected = self.profile
        if (info["regmap"] < R.CFG_SINCE
                or reported.get("platform") != expected.platform_id
                or reported.get("hash") != arch_hash(self.expected_cfg)
                or info.get("core_khz") != expected.core_khz
                or reported.get("channels") != expected.platform["channels_per_slice"]
                or reported.get("channel_bytes") != expected.platform["channel_bytes"]
                or any(reported.get(key) != expected.uarch[key]
                       for key in ("MXU_IMPL", "TMEM_IMPL", "VPU_CL", "ULANES"))
                or bool(caps.get("chash")) != expected.uarch["CHASH"]
                or bool(caps.get("hostcal")) != expected.platform["hostcal"]):
            raise ConfigMismatch("FPGA identity does not match the recorded burst200 profile (200 MHz HBM)")
        cfg = device_config(info)
        if cfg != self.expected_cfg:
            raise ConfigMismatch("FPGA compute configuration does not match the burst200 profile")
        if not caps.get("args"):
            raise ConfigMismatch("PersonaPlex requires runtime argument registers (CAP_ARGS)")
        if "int4" in self.weight_formats and not caps.get("w4"):
            raise ConfigMismatch("INT4 PersonaPlex weights require CAP_W4")
        status = info["status"]
        calibration = R.ST_CALIB0 | R.ST_CALIB1
        if status & calibration != calibration:
            raise RuntimeError(f"burst200 memory is not calibrated (STATUS={status:#x})")
        if status & (R.ST_ERROR | R.ST_AXI_ERR | R.ST_WAIT_TO | R.ST_ECC_DED):
            raise RuntimeError(f"burst200 reports an error; diagnostics must be preserved (STATUS={status:#x})")
        if (status & R.ST_LOADING or not status & R.ST_WR_IDLE
                or status & R.ST_RUN and not status & R.ST_HALTED):
            raise RuntimeError(f"burst200 is not idle with drained writes (STATUS={status:#x})")
        return info, cfg

    def factory(self, cfg, images):
        from opentpu.host.board import BoardBackend
        if cfg != self.expected_cfg:
            raise ValueError("backend configuration differs from the qualified burst200 profile")
        # Quantizing the host image can take time. Recheck before BoardBackend's
        # constructor, which scrubs and uploads device memory immediately.
        self.info, self.cfg = self._probe()
        self.backend = BoardBackend(cfg, images, transport=self.transport,
                                    model="PersonaPlex temporal", own_transport=True)
        return self.backend

    def close(self):
        try:
            if self.backend is not None:
                self.backend.close()
        finally:
            self.transport.close()


class TimedEngine:
    def __init__(self, engine, backend):
        self.engine = engine
        self.backend_name = backend
        self.seconds = []
        self.runs = []

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def step(self, embedding):
        started = time.perf_counter()
        result = self.engine.step(embedding)
        elapsed = time.perf_counter() - started
        stats = copy.deepcopy(self.engine.last_stats)
        run = {"engine_stats": stats, "host_round_trip_seconds": elapsed}
        info = getattr(getattr(self.engine, "backend", None), "info", {})
        core_khz = info.get("core_khz") if isinstance(info, dict) else None
        cycles = stats.get("cycles")
        if (self.backend_name == "board" and type(core_khz) in (int, float)
                and np.isfinite(core_khz) and core_khz > 0
                and type(cycles) is int and cycles >= 0):
            run.update(device_ms=cycles / core_khz, device_core_khz=core_khz)
        self.seconds.append(elapsed)
        self.runs.append(run)
        return result

    def reset(self):
        self.engine.reset()
        self.seconds.clear()
        self.runs.clear()

    def report(self):
        durations = np.asarray(self.seconds, dtype=np.float64)
        return {
            "backend": self.backend_name,
            "fpga_component": "temporal transformer" if self.backend_name == "board" else None,
            "host_components": ["Mimi codec", "embeddings", "text head", "depth transformer"],
            "frames_measured": len(self.seconds),
            "temporal_mean_ms": float(durations.mean() * 1000) if len(durations) else None,
            "temporal_p95_ms": float(np.percentile(durations, 95) * 1000) if len(durations) else None,
            "timing_scope": "temporal_mean_ms and temporal_p95_ms measure host round-trip wall time "
                            "around Engine.step, including input/output transfers and submission/wait.",
            "device_timing_scope": "Per-frame device_ms is raw core cycles / probed backend core_khz; "
                                   "it excludes host transfers and is absent for ISA execution.",
            "device_time_measured": any("device_ms" in run for run in self.runs),
            "runs": copy.deepcopy(self.runs),
            "audio_frame_budget_ms": 80,
            "end_to_end_real_time_validated": False,
        }


def inspect_layout(spec, cfg, cap, wformat, *, formats=""):
    from opentpu import waits
    from opentpu.llm.personaplex_temporal import Image
    # Layout only: permit oversized layouts to report their deficit. Engine
    # construction always receives the real device capacity and enforces fit.
    image = Image(spec, replace(cfg, DRAM_BYTES=max(cfg.DRAM_BYTES, 1 << 40)),
                  cap=cap, wformat=wformat, formats=formats)
    traffic = image.weight_bytes
    code_bytes = image.weight_code_bytes
    size = image.nbytes
    program = waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
    allocated = image.required_bytes
    # burst200's 128-byte port at 200 MHz. This is a lower bound, not a prediction.
    return {
        "weight_formats": {"base": wformat, "rules": formats,
                           "layers": [list(f) for f in image.lf]},
        "context_frames": cap, "context_seconds_before_reset": cap / 12.5,
        "image_bytes": size, "program_reservation_bytes": program,
        "required_bytes": allocated, "dram_bytes": cfg.DRAM_BYTES,
        "fits": allocated <= cfg.DRAM_BYTES,
        "temporal_weight_bytes_per_frame": traffic,
        "weight_code_bytes_per_frame": code_bytes,
        "burst200_peak_weight_floor_ms": code_bytes / 25.6e9 * 1000,
        "audio_frame_budget_ms": 80,
        "caveat": "Packed-weight-only floor excludes scales/norms, KV, activations, CPU depth/codec and host overhead.",
    }


def validate_embeddings(path, spec, cap):
    """Reject a bad benchmark before allocating or writing a device image."""
    data = np.load(path, allow_pickle=False, mmap_mode="r")
    if (not isinstance(data, np.ndarray) or data.ndim != 2
            or data.shape[1] != spec.hidden or not 0 < len(data) <= cap):
        raise ValueError(f"expected a .npy array with 1..{cap} embedding frames of width {spec.hidden}")
    if data.dtype.kind != "f" or not np.isfinite(data).all():
        raise ValueError("embedding frames must be finite floating-point values")
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.isfinite(np.asarray(data, dtype=np.float32)).all():
            raise ValueError("embedding frames must remain finite floating-point values in float32")
    return data


def preflight(args, weights, spec):
    """Check deployment inputs without opening the FPGA or unpacking big tensors."""
    from opentpu.target import config
    from opentpu.llm.personaplex_temporal import Image
    errors = []
    dependencies = {}
    for name in ("torch", "safetensors", "huggingface_hub", "aiohttp", "sentencepiece", "sphn", "moshi"):
        try:
            dependencies[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            dependencies[name] = False
    missing = [name for name, present in dependencies.items() if not present]
    if missing:
        errors.append("Missing runtime modules: " + ", ".join(missing))
    cfg = config(args.profile)
    layout = inspect_layout(spec, cfg, args.cap, args.wformat,
                            formats=getattr(args, "formats", ""))
    programs = []
    if layout["fits"]:
        image = Image(spec, cfg, cap=args.cap, wformat=args.wformat,
                      formats=getattr(args, "formats", ""))
        for bucket in range(1, (args.cap + image.block - 1) // image.block + 1):
            compiled, _ = image.compile_bucket(bucket)
            programs.append(len(compiled[0]) * 8)
    else:
        errors.append("Temporal image does not fit the selected profile")
    assets = None
    if getattr(args, "assets_dir", None):
        from opentpu.llm.personaplex_assets import ServiceAssets
        try:
            if args.model_dir is None:
                raise ValueError("--assets-dir requires --model-dir for the local tokenizer")
            assets = ServiceAssets.resolve(args.assets_dir, args.model_dir)
        except (OSError, ValueError) as error:
            errors.append(str(error))
    else:
        # Existing explicit paths are validated too; omitted assets are left for
        # upstream's online mode. --assets-dir is the fully offline contract.
        for name, attr, directory in (("Mimi", "mimi_weight", False),
                                      ("voice prompts", "voice_prompt_dir", True),
                                      ("web UI", "static", True)):
            path = getattr(args, attr, None)
            if path is not None:
                path = Path(path)
                valid = path.is_dir() if directory else path.is_file() and path.stat().st_size > 0
                if valid and attr == "voice_prompt_dir":
                    valid = any(p.is_file() and p.stat().st_size > 0 for p in path.glob("*.pt"))
                if valid and attr == "static":
                    index = path / "index.html"
                    valid = index.is_file() and index.stat().st_size > 0
                if not valid:
                    errors.append(f"Missing or invalid {name}: {path}")
        if args.model_dir:
            tokenizer = Path(args.model_dir) / "tokenizer_spm_32k_3.model"
            if not tokenizer.is_file() or tokenizer.stat().st_size == 0:
                errors.append("Missing or empty tokenizer_spm_32k_3.model in model directory")
    host = None
    if weights is None:
        errors.append("--model-dir is required to validate the production checkpoint")
    elif not missing:
        from opentpu.llm.personaplex_bridge import validate_host_checkpoint
        try:
            host = validate_host_checkpoint(weights, spec)
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            errors.append(str(error))
    return {"source_preflight_passed": not errors, "hardware_validated": False,
            "offline_assets": assets is not None, "dependencies": dependencies,
            "layout": layout, "program_words": programs,
            "host_checkpoint": host, "errors": errors}


def serve(args, weights, engine):
    if args.backend != "board":
        raise ValueError("conversation serving requires --backend board; no CPU/ISA fallback")
    import torch
    from aiohttp import web
    from moshi.models import loaders
    from opentpu.llm.personaplex_bridge import load_host_model

    assets = None
    hub_context = nullcontext()
    if getattr(args, "assets_dir", None):
        from opentpu.llm.personaplex_assets import ServiceAssets, offline_hub
        assets = ServiceAssets.resolve(args.assets_dir, args.model_dir)
        hub_context = offline_hub(assets)

    torch.set_num_threads(args.threads)
    host_lm = load_host_model(weights, engine)
    previous_loader, previous_run_app, previous_argv = loaders.get_moshi_lm, web.run_app, sys.argv
    previous_application = web.Application

    def get_model(filename, copy_missing_weights=True, device="cpu", dtype=None,
                  delays=None, cpu_offload=False):
        if torch.device(device).type != "cpu" or cpu_offload or delays is not None:
            raise ValueError("FPGA bridge requires CPU host components and the checkpoint delays")
        return host_lm

    async def status(_request):
        return web.json_response(engine.report())

    def application(*app_args, **app_kwargs):
        app = previous_application(*app_args, **app_kwargs)
        app.router.add_get("/api/fpga-status", status)
        return app

    def run_app(app, **kwargs):
        kwargs["host"] = "127.0.0.1"
        return previous_run_app(app, **kwargs)

    argv = ["moshi.server", "--device", "cpu", "--host", "127.0.0.1",
            "--port", str(args.port), "--moshi-weight", str(args.model_dir),
            "--tokenizer", str(assets.tokenizer if assets else Path(args.model_dir) / "tokenizer_spm_32k_3.model")]
    for flag, value in (("--mimi-weight", assets.mimi_weight if assets else args.mimi_weight),
                        ("--voice-prompt-dir", assets.voice_prompt_dir if assets else args.voice_prompt_dir),
                        ("--static", assets.static if assets else args.static)):
        if value is not None:
            argv.extend([flag, str(value)])
    loaders.get_moshi_lm, web.run_app, sys.argv = get_model, run_app, argv
    web.Application = application
    try:
        # NVIDIA's module starts main at import time, so do not import it first.
        with hub_context:
            runpy.run_module("moshi.server", run_name="__main__")
    finally:
        loaders.get_moshi_lm, web.run_app, sys.argv = previous_loader, previous_run_app, previous_argv
        web.Application = previous_application


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["inspect", "preflight", "benchmark", "serve"])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--backend", choices=["board", "isa"], default="board")
    parser.add_argument("--dev", default="f2:0")
    parser.add_argument("--profile", default=str(PROFILE))
    parser.add_argument("--cap", type=int, default=1024,
                        help="bounded temporal context including voice/text prompt frames")
    parser.add_argument("--wformat", choices=["int4", "int8"], default="int4")
    parser.add_argument("--formats", default="",
                        help="explicit INT4/INT8 per-kind/layer rules; empty keeps uniform precision")
    parser.add_argument("--embeddings", type=Path, help="benchmark input: float32 .npy [frames,hidden]")
    parser.add_argument("--output", type=Path, help="benchmark normalized hidden outputs: .npy")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--mimi-weight", type=Path)
    parser.add_argument("--voice-prompt-dir", type=Path)
    parser.add_argument("--static", type=Path)
    parser.add_argument("--assets-dir", type=Path,
                        help="validated local upstream assets; disables implicit HF downloads")
    args = parser.parse_args()
    if args.cap < 1 or args.threads < 1 or not 1 <= args.port <= 65535:
        parser.error("cap/threads must be positive and port must be 1..65535")
    if args.mode == "benchmark" and args.output is not None:
        # Preserve np.save's filename suffix convention while checking before
        # backend construction; exclusive creation below also closes the race.
        if not str(args.output).endswith(".npy"):
            args.output = Path(str(args.output) + ".npy")
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Benchmark output must be new: {args.output}")
        if not args.output.parent.is_dir():
            raise FileNotFoundError(f"Benchmark output parent must exist: {args.output.parent}")
    from opentpu.target import config
    from opentpu.llm.personaplex_temporal import Engine, Spec
    weights = None
    spec = Spec()
    if args.model_dir:
        from opentpu.llm.personaplex_weights import MLXWeights
        weights = MLXWeights(args.model_dir)
        spec = weights.spec
    if args.cap > spec.context:
        parser.error(f"cap exceeds checkpoint context of {spec.context} frames")
    if args.mode == "inspect":
        report = inspect_layout(spec, config(args.profile), args.cap, args.wformat,
                                formats=args.formats)
        print(json.dumps(report, indent=2))
        return 0 if report["fits"] else 2
    if args.mode in ("preflight", "serve"):
        report = preflight(args, weights, spec)
        print(json.dumps(report, indent=2), flush=True)
        if not report["source_preflight_passed"]:
            return 2
        if args.mode == "preflight":
            return 0
    if weights is None:
        parser.error("--model-dir is required for benchmark/serve")
    if args.mode == "benchmark" and args.embeddings is None:
        parser.error("benchmark requires --embeddings")
    if args.mode == "serve" and args.backend != "board":
        parser.error("serve requires --backend board")
    inputs = (validate_embeddings(args.embeddings, spec, args.cap)
              if args.mode == "benchmark" else None)
    # Resolve every layer and reject invalid rules/layouts before opening a device.
    # BoardSession independently requires the probed config to equal this profile.
    if args.backend == "board":
        _board_profile(args.profile)
    cfg = config(args.profile)
    report = inspect_layout(spec, cfg, args.cap, args.wformat, formats=args.formats)
    if not report["fits"]:
        raise MemoryError("PersonaPlex temporal image exceeds the profile device memory")
    session = None
    engine = None
    try:
        if args.backend == "board":
            resolved = {fmt for layer in report["weight_formats"]["layers"] for fmt in layer}
            session = BoardSession(args.dev, profile=args.profile, wformat=args.wformat,
                                   weight_formats=resolved)
            cfg, backend = session.cfg, session.factory
            print(json.dumps({"board": session.info}, default=str), flush=True)
        else:
            backend = "isa"
        print(json.dumps(report, indent=2), flush=True)
        engine = TimedEngine(Engine(spec, weights.layers, weights.final_norm, cfg=cfg,
                                   backend=backend, cap=args.cap, wformat=args.wformat,
                                   formats=args.formats), args.backend)
        if args.mode == "serve":
            serve(args, weights, engine)
        else:
            outputs = np.stack([engine.step(row) for row in inputs])
            if args.output:
                with args.output.open("xb") as handle:
                    np.save(handle, outputs, allow_pickle=False)
            print(json.dumps(engine.report(), indent=2))
    finally:
        if session is not None:
            session.close()
        elif engine is not None:
            engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
