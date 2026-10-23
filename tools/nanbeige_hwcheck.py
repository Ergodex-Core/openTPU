#!/usr/bin/env python3
"""Gate burst200 Nanbeige execution against ISA with deterministic tiny weights.

Overwrites the selected FPGA's model image under the usual exclusive device lock.
Checks single-token and four-row tiled logits plus all four logical-layer taps.
This is a hardware arithmetic check, not validation of the published checkpoint.
There is no software fallback for the FPGA side of a comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.nanbeige_fpga import BoardSession, PROFILE, write_json


def tiny_fixture(seed=4):
    from opentpu.llm.nanbeige import Spec

    spec = Spec(hidden=128, layers=2, n_q=6, n_kv=1, head_dim=128,
                ffn=256, vocab=128, eps=1e-5, theta=7e7,
                capture_layers=(0, 1, 2, 3))
    rng = np.random.default_rng(seed)

    def mat(n, k):
        return (rng.normal(size=(n, k)) * (0.24 / np.sqrt(k))).astype(np.float32)

    def norm(n):
        return rng.uniform(0.7, 1.3, size=n).astype(np.float32)

    h, d = spec.hidden, spec.head_dim
    weights = {"model.embed_tokens.weight": mat(spec.vocab, h) * 20,
               "lm_head.weight": mat(spec.vocab, h), "model.norm.weight": norm(h)}
    for i in range(spec.layers):
        p = f"model.layers.{i}."
        weights.update({p + "input_layernorm.weight": norm(h),
                        p + "post_attention_layernorm.weight": norm(h),
                        p + "self_attn.q_proj.weight": mat(spec.n_q * d, h),
                        p + "self_attn.k_proj.weight": mat(spec.n_kv * d, h),
                        p + "self_attn.v_proj.weight": mat(spec.n_kv * d, h),
                        p + "self_attn.o_proj.weight": mat(h, spec.n_q * d),
                        p + "mlp.gate_proj.weight": mat(spec.ffn, h),
                        p + "mlp.up_proj.weight": mat(spec.ffn, h),
                        p + "mlp.down_proj.weight": mat(h, spec.ffn)})
    return spec, weights


def compare_words(actual, expected):
    """Require finite floats and identical bits, including the sign of zero."""
    a = np.ascontiguousarray(actual, dtype=np.float32)
    b = np.ascontiguousarray(expected, dtype=np.float32)
    if a.shape != b.shape:
        return {"passed": False, "actual_shape": list(a.shape),
                "expected_shape": list(b.shape), "error": "shape mismatch"}
    unequal = a.view(np.uint32) != b.view(np.uint32)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    mismatches = np.argwhere(unequal)
    return {"passed": finite and not bool(unequal.any()), "shape": list(a.shape),
            "finite": finite, "unequal_words": int(unequal.sum()),
            "max_abs_diff": float(np.abs(a.astype(np.float64) - b).max())
            if finite and a.size else (0.0 if finite else None),
            "first_mismatch": mismatches[0].tolist() if mismatches.size else None}


def program_identity(programs):
    from opentpu.isa import assemble
    words = [assemble(p) for p in programs]
    return {"words_per_slice": [int(w.size) for w in words],
            "sha256_per_slice": [hashlib.sha256(w.tobytes()).hexdigest() for w in words]}


def make_engine(spec, weights, cfg, backend):
    from opentpu.llm.nanbeige import Engine
    engine = Engine(spec, weights, cap=128, cfg=cfg, backend=backend, rows=4,
                    resident=False, pipeline=False, prompt_runs=False,
                    prog_cache=False, wformat="int8", head_format="int8")
    # Use the model's own forward kernel; generic Qwen stream-fill is not Nanbeige.
    engine.stream_logits = False
    engine.image.stream_fill = False
    return engine


def check_engines(board, golden, report):
    """Both engines receive identical inputs; only board may populate FPGA results."""
    taps = (0, 1, 2, 3)
    report["image_bytes"] = board.image.nbytes
    report["image_matches_isa"] = board.image.nbytes == golden.image.nbytes
    report["tokens"] = [4, 9, 23, 11, 31]
    report["stages"] = stages = []
    report["phase"] = "single_step"
    started = time.perf_counter()
    actual = board.step(4)
    hardware_seconds = time.perf_counter() - started
    captured = board.read_hidden(taps)
    expected = golden.step(4)
    reference = golden.read_hidden(taps)
    decode = board._decode(0)
    programs = (board._host_program(0)[0] if getattr(board, "runtime_host", False)
                else decode[0] if decode is not None else board.image.compile_step(0))
    stages.append({"name": "single_step", "positions": [0],
                   "logits": compare_words(actual, expected),
                   "capture": compare_words(captured, reference),
                   "hardware_seconds": hardware_seconds,
                   "hardware_stats": board.stats[-1],
                   "program": program_identity(programs)})

    report["phase"] = "four_row_tiled"
    rows, tokens, output_rows = [(0, p) for p in range(1, 5)], [9, 23, 11, 31], list(range(4))
    # Exercise the runtime-position mask/argument path used by the benchmark.
    # The explicit static tiled program below is an independent schedule oracle.
    started = time.perf_counter()
    runtime_actual = board.run_rows(rows, tokens, output_rows)
    runtime_seconds = time.perf_counter() - started
    runtime_capture = board.read_hidden(taps, rows=4)
    runtime_expected = golden.run_rows(rows, tokens, output_rows)
    runtime_reference = golden.read_hidden(taps, rows=4)
    stages.append({"name": "runtime_four_rows", "positions": [1, 2, 3, 4],
                   "logits": compare_words(runtime_actual, runtime_expected),
                   "capture": compare_words(runtime_capture, runtime_reference),
                   "hardware_seconds": runtime_seconds,
                   "hardware_stats": board.stats[-1]})
    programs = board.image.compile_tiled_rows(rows, output_rows)
    golden_programs = golden.image.compile_tiled_rows(rows, output_rows)
    identity = program_identity(programs)
    if identity != program_identity(golden_programs):
        raise RuntimeError("FPGA and ISA compiled different tiled programs")
    started = time.perf_counter()
    actual = board._run_rows(rows, tokens, output_rows, programs)
    hardware_seconds = time.perf_counter() - started
    captured = board.read_hidden(taps, rows=4)
    expected = golden._run_rows(rows, tokens, output_rows, golden_programs)
    reference = golden.read_hidden(taps, rows=4)
    stages.append({"name": "four_row_tiled", "positions": [1, 2, 3, 4],
                   "logits": compare_words(actual, expected),
                   "capture": compare_words(captured, reference),
                   "hardware_seconds": hardware_seconds,
                   "hardware_stats": board.stats[-1], "program": identity})
    stages[-1]["runtime_logits"] = compare_words(runtime_actual, expected)
    stages[-1]["runtime_capture"] = compare_words(runtime_capture, reference)

    # An independent schedule also checks that multi-row causal attention and the
    # loop-boundary norm preserve single-token semantics, not just FPGA/ISA parity.
    report["phase"] = "sequential_equivalence"
    golden.reset()
    golden.step(4)
    sequential, sequential_capture = [], []
    for token in tokens:
        sequential.append(golden.step(token))
        sequential_capture.append(golden.read_hidden(taps)[0])
    stages[-1]["sequential_logits"] = compare_words(actual, np.stack(sequential))
    stages[-1]["sequential_capture"] = compare_words(captured, np.stack(sequential_capture))
    report["passed"] = report["image_matches_isa"] and all(
        value["passed"] for stage in stages for value in stage.values()
        if isinstance(value, dict) and "passed" in value)
    report["phase"] = "complete"


def main(argv=None):
    from opentpu.target import load

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default=os.environ.get("OTPU_DEVICE", "f2:0"))
    parser.add_argument("--json", type=Path, help="write evidence on success or failure")
    args = parser.parse_args(argv)
    spec, weights = tiny_fixture()
    report = {"passed": False, "phase": "open", "device": args.dev,
              "target_backend": "FPGA", "oracle_backend": "ISA",
              "utc": datetime.now(timezone.utc).isoformat(), "seed": 4,
              "profile": str(PROFILE.resolve()),
              "profile_sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
              "spec": asdict(spec), "precision": "openTPU W8A8, INT8 output head",
              "checkpoint_quality_validated": False}
    session = None
    try:
        session = BoardSession(args.dev, load(PROFILE))
        report["board_info"] = session.info
        report["architecture"] = asdict(session.cfg)
        report["bdf"] = getattr(session.transport, "bdf", None)
        report["phase"] = "initialize_engines"
        board = make_engine(spec, weights, session.cfg, session.factory)
        report["device_lock_held"] = session.backend.board.lock is not None
        if not report["device_lock_held"]:
            raise RuntimeError("FPGA backend did not acquire the device lock")
        golden = make_engine(spec, weights, session.cfg, "isa")
        check_engines(board, golden, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                report["passed"] = False
                report["close_error"] = f"{type(exc).__name__}: {exc}"
        write_json(args.json, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
