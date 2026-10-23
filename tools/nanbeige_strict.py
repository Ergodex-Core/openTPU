#!/usr/bin/env python3
"""Strict Nanbeige FPGA adapter, with explicit ISA validation and JSONL worker.

No downloads, CPU/GPU inference fallback, image loading or service control.
The coordinator must provide a card window before infer/worker is launched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.nanbeige_board import BoardSession, PROFILE, _json_default, _tokenize, validate_profile
from opentpu.host.runstate import DeviceBusy, DeviceLock
from opentpu.llm.nanbeige import Spec, load_weights
from opentpu.llm.nanbeige_strict import PRECISION, StrictEngine, plan, request_tokens
from opentpu.target import config, load


HOST_OPERATIONS = ["chat templating/tokenization and text decoding", "request validation and queueing",
                   "checkpoint import/quantization and immutable RoPE/mask tables during setup",
                   "program compilation, DMA transport, counters, token streaming and cancellation"]
DEVICE_OPERATIONS = ["embedding lookup and dequantization", "RoPE table lookup and rotation",
                     "RMSNorm, Q/K/V projections, attention/softmax, KV updates, output projection",
                     "SiLU gate/up/down MLP and residuals", "22 physical layers reused over two logical loops",
                     "inter-loop and final norm", "INT8 vocabulary head", "greedy ARGMAX, feedback and EOS"]


class StrictBoardSession(BoardSession):
    """Both normal locks, no force takeover, and capability/error checks before DMA."""
    def __init__(self, dev, profile):
        if dev != "f2:0":
            raise ValueError("strict adapter currently supports the coordinated f2:0 slot only")
        if os.environ.get("OTPU_STOP_RUN") == "1":
            raise ValueError("strict adapter forbids OTPU_STOP_RUN force takeover")
        if os.environ.get("OTPU_RUN_DIR") not in (None, "/tmp/otpu"):
            raise ValueError("strict adapter requires the canonical /tmp/otpu locks")
        self.legacy = DeviceLock("f2:0", wait=0)
        self.transport = self.backend = None
        try:
            super().__init__(dev, profile)  # F2Transport acquires f2-<BDF> normally.
        except BaseException:
            self.close()
            raise

    def _probe(self):
        from opentpu.host.regs import ST_ERROR, ST_AXI_ERR, ST_ECC_DED, ST_LOADING
        info, cfg = super()._probe()
        for capability in ("args", "gen", "mx"):
            if not (info.get("caps") or {}).get(capability):
                raise ValueError(f"strict Nanbeige requires FPGA capability {capability}")
        if info.get("status", 0) & (ST_ERROR | ST_AXI_ERR | ST_ECC_DED | ST_LOADING):
            raise RuntimeError("FPGA reports an error or program loading; refusing upload")
        return info, cfg

    def close(self):
        try:
            if self.transport is not None:
                super().close()
                self.transport = self.backend = None
        finally:
            if self.legacy is not None:
                self.legacy.release()
                self.legacy = None


def encode_request(request, spec, cap, model_dir):
    if not isinstance(request, dict) or set(request) - {"id", "prompt", "prompt_tokens", "max_new_tokens"}:
        raise ValueError("request accepts id, prompt or prompt_tokens, and max_new_tokens only; greedy only")
    if ("prompt" in request) == ("prompt_tokens" in request):
        raise ValueError("supply exactly one of prompt and prompt_tokens")
    tokenizer = None
    if "prompt" in request:
        if not isinstance(request["prompt"], str) or not request["prompt"].strip():
            raise ValueError("prompt must be a nonempty string")
        tokenizer, tokens = _tokenize(model_dir, request["prompt"])
    else:
        tokens = request["prompt_tokens"]
    tokens, budget = request_tokens(tokens, request.get("max_new_tokens", 16), spec, cap)
    return tokenizer, tokens, budget


def emit(value, path=None):
    data = json.dumps(value, default=_json_default, allow_nan=False)
    if path is not None:
        with Path(path).open("x") as stream:
            stream.write(data + "\n")
    print(data, flush=True)


def run_request(engine, request, model_dir, *, hardware, stream=False):
    tokenizer, tokens, budget = encode_request(request, engine.spec, engine.cap, model_dir)
    callback = (lambda t: emit({"event": "token", "id": request.get("id"), "token": t})) if stream else None
    result = engine.generate_strict(tokens, budget, on_token=callback)
    cycles = [r.get("cycles") for r in result["raw_runs"]]
    result.update(ok=True, event="result", id=request.get("id"), prompt_tokens=tokens,
                  text=tokenizer.decode(result["tokens"], skip_special_tokens=False) if tokenizer else None,
                  execution_backend="board" if hardware else "isa_validation",
                  hardware_executed=hardware and bool(result["raw_runs"]),
                  fpga_only=hardware, strict_model_path=True, greedy_selection="device",
                  cycles_are_hardware_measured=hardware and bool(cycles) and all(type(c) is int for c in cycles),
                  device_cycles=sum(cycles) if cycles and all(type(c) is int for c in cycles) else None)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("inspect", "infer", "worker", "validate-isa"))
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--profile", default=PROFILE)
    p.add_argument("--device", default="f2:0")
    p.add_argument("--cap", type=int, default=128)
    p.add_argument("--embedding-format", choices=("int8", "f32"), default="int8")
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument("--prompt", default=None)
    inputs.add_argument("--prompt-tokens", help="JSON array of integer token IDs for reproducible validation")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--json", type=Path, help="exclusive receipt file; existing output is never overwritten")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.json and args.json.exists():
        raise FileExistsError(args.json)
    profile = load(str(args.profile))
    contract = validate_profile(profile)
    spec, cfg = Spec.from_hf(args.model_dir), config(profile)
    layout = plan(spec, cfg, args.cap, args.embedding_format)
    precision = dict(PRECISION)
    if args.embedding_format == "f32":
        precision.update(target_execution="openTPU MX W8A8, INT8 head; source BF16 embedding values stored as FP32 on device",
                         embedding_change="none; source BF16 values expanded exactly to FP32")
    from opentpu import progcache, imgcache
    metadata = {"adapter": "nanbeige-strict-wideport-v1", "precision": precision, "layout": layout,
                "profile_contract": contract,
                "host_operations": HOST_OPERATIONS, "device_operations": DEVICE_OPERATIONS,
                "source_hash": progcache.source(), "source_hash_algorithm": "blake2b-128",
                "compile_environment": progcache.keyed_env(),
                "checkpoint_identity": imgcache.checkpoint(args.model_dir),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "cap": args.cap}
    if args.command == "inspect":
        emit({"ok": True, "command": "inspect", "hardware_executed": False, **metadata}, args.json)
        return 0
    request = {"max_new_tokens": args.max_new_tokens}
    request.update(prompt_tokens=json.loads(args.prompt_tokens)) if args.prompt_tokens is not None else \
        request.update(prompt=args.prompt if args.prompt is not None else "Hi")
    if args.command != "worker":
        encode_request(request, spec, args.cap, args.model_dir)  # before locks or upload
    elif args.json:
        raise ValueError("worker emits JSONL on stdout; --json is for one-shot receipts")
    hardware = args.command != "validate-isa"
    session = None
    try:
        weights = load_weights(args.model_dir)
        session = StrictBoardSession(args.device, profile) if hardware else None
        started = time.perf_counter()
        engine = StrictEngine(spec, weights, cfg=session.cfg if session else cfg,
                              backend=session.factory if session else "isa", cap=args.cap,
                              embedding_format=args.embedding_format)
        metadata.update(setup_seconds=time.perf_counter() - started,
                        board_info=session.info if session else None)
        if args.command == "worker":
            emit({"event": "ready", "ok": True, "fpga_only": True, **metadata})
            for line in sys.stdin:
                # Requests are serialized in this one worker. EOF releases its locks.
                try:
                    req = json.loads(line)
                    encode_request(req, engine.spec, engine.cap, args.model_dir)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    emit({"event": "error", "ok": False, "error": {"type": "invalid_request", "message": str(exc)}})
                    continue
                # Device/compiler failures escape and close the session; never retry on CPU.
                result = run_request(engine, req, args.model_dir, hardware=True, stream=True)
                emit({**metadata, **result})
        else:
            result = run_request(engine, request, args.model_dir, hardware=hardware)
            emit({**metadata, **result}, args.json)
        return 0
    except DeviceBusy as exc:
        emit({"ok": False, "error": {"type": "device_busy", "message": str(exc)}})
        return 3
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    raise SystemExit(main())
