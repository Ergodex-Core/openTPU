#!/usr/bin/env python3
"""Compare co-resident DSpark FPGA and ISA execution with deterministic weights.

This validates arithmetic and memory isolation, not full-checkpoint quality.
The board path has no software fallback; CPU context/Markov work remains explicit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.nanbeige_fixture import tiny_fixture
from tools.nanbeige_fpga import BoardSession, PROFILE, write_json
from tools.nanbeige_hwcheck import compare_words, make_engine


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _proposal(actual, expected):
    return {"hidden": compare_words(actual.hidden, expected.hidden),
            "logits": compare_words(actual.logits, expected.logits),
            "tokens": {"passed": bool(np.array_equal(actual.tokens, expected.tokens)),
                       "actual": actual.tokens.tolist(), "expected": expected.tokens.tolist()}}


def check_engines(target, golden, draft, reference, report):
    taps = draft.spec.capture_layers
    actual_aux, expected_aux = [], []
    stages = report["stages"] = []
    report["phase"] = "target_prefill"
    for token in [2, 7]:
        actual, expected = target.step(token), golden.step(token)
        ah, eh = target.read_hidden(taps), golden.read_hidden(taps)
        stages.append({"name": "target_prefill", "logits": compare_words(actual, expected),
                       "capture": compare_words(ah, eh)})
        actual_aux.append(ah[0])
        expected_aux.append(eh[0])
    ctx = draft.project_context(np.stack(actual_aux), np.arange(2))
    ref_ctx = reference.project_context(np.stack(expected_aux), np.arange(2))

    report["phase"] = "seven_query_draft"
    before = _digest(target.backend.read(0, 0, target.image.nbytes))
    started = time.perf_counter()
    actual = draft.draft(9, 2, ctx, 3, query_tokens=7)
    seconds = time.perf_counter() - started
    expected = reference.draft(9, 2, ref_ctx, 3, query_tokens=7)
    initial = actual
    after = _digest(target.backend.read(0, 0, target.image.nbytes))
    stage = {"name": "seven_query_draft", "hardware_wall_seconds": seconds,
             "target_memory": {"passed": before == after, "before_sha256": before,
                               "after_sha256": after}, **_proposal(actual, expected)}
    stages.append(stage)

    report["phase"] = "target_verification_after_draft"
    ids = [9, *actual.tokens.tolist()]
    rows = [(0, p) for p in range(2, 6)]
    actual = target.run_rows(rows, ids, list(range(4)))
    expected = golden.run_rows(rows, ids, list(range(4)))
    ah, eh = target.read_hidden(taps, rows=4), golden.read_hidden(taps, rows=4)
    stages.append({"name": "target_verification_after_draft",
                   "logits": compare_words(actual, expected),
                   "capture": compare_words(ah, eh)})
    # Simulate a partial acceptance: only two input rows become draft context.
    ctx = ctx.append(draft.project_context(ah[:2], np.arange(2, 4)))
    ref_ctx = ref_ctx.append(reference.project_context(eh[:2], np.arange(2, 4)))
    report["phase"] = "incremental_context"
    actual = draft.draft(5, 4, ctx, 3, query_tokens=7)
    expected = reference.draft(5, 4, ref_ctx, 3, query_tokens=7)
    stages.append({"name": "incremental_context", **_proposal(actual, expected)})
    report["phase"] = "context_rollback"
    actual = draft.draft(9, 2, ctx.truncate(2), 3, query_tokens=7)
    expected = reference.draft(9, 2, ref_ctx.truncate(2), 3, query_tokens=7)
    stages.append({"name": "context_rollback", **_proposal(actual, expected),
                   "replay_hidden": compare_words(actual.hidden, initial.hidden)})
    report["draft_hardware_stats"] = getattr(draft, "stats", [])
    report["target_hardware_stats"] = target.stats
    report["target_image_bytes"] = target.image.nbytes
    report["draft_image_base"] = draft.image.base
    report["draft_image_end"] = draft.image.nbytes
    report["passed"] = all(value["passed"] for stage in stages
                           for value in stage.values()
                           if isinstance(value, dict) and "passed" in value)
    report["phase"] = "complete"


def main(argv=None):
    from opentpu.target import load
    from opentpu.llm.dspark_fpga import Engine as DraftEngine
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default="f2:0")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--device-fusion-markov", action="store_true")
    args = parser.parse_args(argv)
    ts, tw, ds, dw = tiny_fixture()
    report = {"passed": False, "phase": "open", "device": args.dev,
              "utc": datetime.now(timezone.utc).isoformat(), "seed": 92,
              "target_backend": "FPGA", "draft_backbone_backend": "FPGA",
              "draft_context_projection_backend": "FPGA" if args.device_fusion_markov else "CPU",
              "draft_markov_backend": "FPGA projection, CPU row lookup/selection" if args.device_fusion_markov else "CPU",
              "oracle_backbone_backend": "ISA", "checkpoint_quality_validated": False,
              "precision": "target W8A8; draft W4A8; shared INT8 target head",
              "target_spec": asdict(ts), "draft_spec": asdict(ds)}
    session = None
    try:
        session = BoardSession(args.dev, load(PROFILE))
        report["board_info"] = session.info
        report["architecture"] = asdict(session.cfg)
        target = make_engine(ts, tw, session.cfg, session.factory)
        if session.backend.board.lock is None:
            raise RuntimeError("FPGA backend did not acquire the device lock")
        report["device_lock_held"] = True
        golden = make_engine(ts, tw, session.cfg, "isa")
        report["phase"] = "initialize_draft"
        draft = DraftEngine(target, ds, dw, target_weights=tw, rows=7, block=256,
                            device_fusion_markov=args.device_fusion_markov)
        reference = DraftEngine(golden, ds, dw, target_weights=tw, rows=7, block=256,
                                device_fusion_markov=args.device_fusion_markov)
        report["draft_precision"] = draft.precision
        check_engines(target, golden, draft, reference, report)
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
