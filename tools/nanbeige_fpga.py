#!/usr/bin/env python3
"""Inspect/run Nanbeige on burst200 with CPU or accelerator-backed DSpark drafting.

No checkpoint downloads, bitstream changes or implicit CPU target fallback.
The target is the openTPU W8A8 repack, not the published GPTQ W8A16 arithmetic.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
import statistics
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROFILE = Path(__file__).parent / "data/nanbeige/burst200.toml"
PRECISION = {
    "checkpoint": "compressed-tensors channelwise INT8 weights, BF16 activations/head",
    "target_execution": "openTPU MXU W8A8 repack, INT8 head, original BF16 embedding values expanded to host FP32, FP32 residual/norm/RoPE/softmax",
    "source_equivalent": False,
    "draft_execution": "CPU NumPy FP32 DSpark reference with original BF16 shared weights",
}


def validate_board(info, actual, profile):
    """Require the requested burst200 architecture/clock before any model DMA."""
    from opentpu.target import config, arch_hash
    expected = config(profile)
    if profile.name != "burst200" or profile.platform.get("board") != "f2":
        raise ValueError("the benchmark requires the burst200 F2 profile")
    differences = {k: {"expected": v, "actual": asdict(actual)[k]}
                   for k, v in asdict(expected).items() if asdict(actual)[k] != v}
    if differences or arch_hash(actual) != arch_hash(expected):
        raise ValueError(f"loaded FPGA architecture does not match burst200: {differences}")
    cfg = info.get("cfg") or {}
    if cfg.get("platform") != profile.platform_id:
        raise ValueError("loaded FPGA platform does not match burst200 F2")
    if info.get("core_khz") != profile.core_khz:
        raise ValueError(f"expected {profile.core_khz} kHz burst200 core, got {info.get('core_khz')}")
    if cfg.get("channels") != profile.platform["channels_per_slice"]:
        raise ValueError("loaded FPGA memory channel count does not match burst200")
    if cfg.get("channel_bytes") != profile.platform["channel_bytes"]:
        raise ValueError("loaded FPGA channel capacity does not match burst200")
    if not info.get("calibrated"):
        raise ValueError("FPGA memory is not calibrated")
    if info.get("running"):
        raise RuntimeError("FPGA is already running; refusing to overwrite its model")


class BoardSession:
    def __init__(self, dev, profile):
        from opentpu.host.board import open_transport
        if not dev.startswith("f2:"):
            raise ValueError("burst200 requires an explicit f2:<slot> device")
        self.profile = profile
        self.transport = open_transport(dev)
        self.backend = None
        try:
            self.info, self.cfg = self._probe()
        except BaseException:
            self.close()
            raise

    def _probe(self):
        from opentpu.host.board import Board, device_config
        # A status check must not calibrate or quiesce another workload.
        info = Board(self.transport, lock=False, calibrate=False).info()
        cfg = device_config(info)
        validate_board(info, cfg, self.profile)
        return info, cfg

    def factory(self, cfg, images):
        from opentpu.host.board import BoardBackend
        if cfg != self.cfg:
            raise ValueError("backend configuration differs from qualified burst200 profile")
        # Preparing a model can take time; recheck before backend construction
        # scrubs and uploads device memory.
        self.info, self.cfg = self._probe()
        self.backend = BoardBackend(cfg, images, transport=self.transport,
                                    model="Nanbeige4.2 burst200 W8A8", own_transport=False)
        return self.backend

    def close(self):
        try:
            if self.backend is not None:
                self.backend.close()
        finally:
            self.transport.close()


def inspect_layout(spec, cfg, cap, rows):
    from opentpu import waits
    from opentpu.compiler import CompileError
    # Metadata only: allow an oversized address space to report fit errors;
    # actual Engine construction always uses the real board capacity.
    image = spec.image(replace(cfg, DRAM_BYTES=max(cfg.DRAM_BYTES, 1 << 40)), cap,
                       rows=rows, wformat="int8", head_format="int8", lookup=False)
    reservation = waits.area(cfg.IMEM_WORDS, cfg.SEQ_WC)
    programs = []
    for kind, pos in (("step", 0), ("step", cap - 1),
                      ("verify", 0), ("verify", max(0, cap - rows))):
        try:
            if kind == "step":
                progs = image.compile_step(pos)
            else:
                n = min(rows, cap - pos)
                progs = image.compile_rows([(0, pos + j) for j in range(n)],
                                            list(range(n)))
            words = [len(p) * 8 for p in progs]
            programs.append({"kind": kind, "position": pos, "words_per_slice": words,
                             "fits_imem": all(n <= cfg.IMEM_WORDS for n in words)})
        except (CompileError, ValueError, MemoryError) as exc:
            programs.append({"kind": kind, "position": pos, "error": str(exc),
                             "fits_imem": False})
    allocated = ((image.nbytes + 4095) // 4096) * 4096 + reservation
    return {"image_bytes": image.nbytes, "program_reservation_bytes": reservation,
            "required_bytes": allocated, "dram_bytes": cfg.DRAM_BYTES,
            "fits_dram": allocated <= cfg.DRAM_BYTES, "programs": programs,
            "physical_layers": spec.layers, "logical_layers": spec.logical_layers,
            "kv_bytes": image.kv_bytes,
            "capture_layers": list(spec.capture_layers), "rows": rows,
            "embedding_location": "host", "embedding_precision": "source BF16 values expanded to FP32",
            "resident": False, "weights_copied_per_physical_layer": 1}


def _next(logits):
    if not np.all(np.isfinite(logits)):
        raise ValueError("target produced nonfinite logits")
    return int(np.argmax(logits))


def run_target(engine, prompt, max_new_tokens, prefill_chunk=4):
    """Greedy target-only baseline with the same prompt-row policy as hybrid."""
    from opentpu.compiler import CompileError
    if not prompt or len(prompt) > engine.cap or max_new_tokens < 0:
        raise ValueError("invalid prompt length or max_new_tokens")
    started = time.perf_counter()
    stats_start = len(engine.stats)
    engine.reset()
    if max_new_tokens == 0:
        return {"tokens": [], "stop_reason": "length", "generated_tokens": 0,
                "decode_tokens": 0, "prefill_seconds": 0., "decode_seconds": 0.,
                "total_seconds": time.perf_counter() - started, "decode_tps": 0.,
                "target_prefill_runs": 0, "target_decode_runs": 0}
    at, fallbacks = 0, 0
    max_prefill_rows = min(prefill_chunk, engine.image.rows)
    while at < len(prompt):
        n = min(max_prefill_rows, len(prompt) - at)
        while True:
            try:
                if n == 1:
                    logits = engine.step(int(prompt[at]))
                else:
                    rows = [(0, at + i) for i in range(n)]
                    logits = engine.run_rows(rows, list(prompt[at:at + n]), [n - 1])[-1]
                    engine.pos = at + n
                break
            except CompileError:
                if n == 1:
                    raise
                n -= 1
                max_prefill_rows = n
                fallbacks += 1
        at += n
    tokens = [_next(logits)]
    prefill_seconds = time.perf_counter() - started
    prefill_end = len(engine.stats)
    decode_started = time.perf_counter()
    reason = "length"
    while len(tokens) < max_new_tokens:
        if tokens[-1] in engine.spec.eos:
            reason = "eos"
            break
        if engine.pos >= engine.cap:
            reason = "capacity"
            break
        tokens.append(_next(engine.step(tokens[-1])))
    if tokens[-1] in engine.spec.eos:
        reason = "eos"
    decode_seconds = time.perf_counter() - decode_started
    return {"tokens": tokens, "stop_reason": reason, "generated_tokens": len(tokens),
            "decode_tokens": len(tokens) - 1, "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds, "total_seconds": time.perf_counter() - started,
            "decode_tps": (len(tokens) - 1) / decode_seconds if decode_seconds else 0.,
            "compile_fallbacks": fallbacks,
            "target_prefill_runs": prefill_end - stats_start,
            "target_decode_runs": len(engine.stats) - prefill_end}


def _sum_cycles(raw):
    cycles = [s.get("cycles") if s is not None else None for s in raw]
    return sum(int(c) for c in cycles) if all(isinstance(c, (int, np.integer))
                                             for c in cycles) else None


def backend_evidence(stats, backend, prefill_run_count=None):
    raw = list(stats)
    cycles = [s.get("cycles") if s is not None else None for s in raw]
    complete = bool(cycles) and all(isinstance(c, (int, np.integer)) for c in cycles)
    return {"target_backend": backend, "raw_runs": raw,
            "target_cycles": _sum_cycles(raw),
            "target_prefill_cycles": _sum_cycles(raw[:prefill_run_count])
                if prefill_run_count is not None else None,
            "target_decode_cycles": _sum_cycles(raw[prefill_run_count:])
                if prefill_run_count is not None else None,
            "target_cycles_scope": "all target prefill and decode runs; excludes draft execution",
            "cycles_are_hardware_measured": backend == "board" and complete,
            "timing_scope": "host wall clock including compilation, target execution and transfers"}


def collect_runs(run_mode, modes, *, warmup_runs=0, repeat=1):
    """Warm each requested mode, then retain every measured run and its medians."""
    if warmup_runs < 0 or repeat < 1:
        raise ValueError("warmup_runs must be nonnegative and repeat positive")
    warmups = []
    warm_started = time.perf_counter()
    for iteration in range(warmup_runs):
        times = {}
        for mode in modes:
            started = time.perf_counter()
            run_mode(mode)  # Results/counters intentionally excluded from measured aggregates.
            times[mode] = time.perf_counter() - started
        warmups.append({"iteration": iteration, "seconds_by_mode": times})
    warmup_seconds = time.perf_counter() - warm_started if warmup_runs else 0.0
    measured = [{"iteration": i, "runs": {mode: run_mode(mode) for mode in modes}}
                for i in range(repeat)]
    aggregates = {}
    for mode in modes:
        samples = [r["runs"][mode] for r in measured]
        summary = {"samples": repeat,
                   "tokens_consistent": all(r["tokens"] == samples[0]["tokens"] for r in samples)}
        for name in ("decode_tps", "prefill_seconds", "decode_seconds", "total_seconds",
                     "target_seconds", "draft_seconds", "target_cycles",
                     "target_prefill_cycles", "target_decode_cycles", "draft_cycles",
                     "acceptance_rate"):
            values = [r.get(name) for r in samples]
            valid = all(isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)
                        for v in values)
            summary[name + "_median"] = float(statistics.median(values)) if valid else None
        aggregates[mode] = summary
    return {"runs": measured[-1]["runs"], "measured_runs": measured,
            "aggregates": aggregates, "warmup_runs": warmup_runs, "repeat": repeat,
            "warmup_seconds": warmup_seconds, "warmups": warmups}


def _tokenize(model_dir, prompt):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True,
                                               local_files_only=True)
    tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, preserve_thinking=False,
                                           return_dict=False)
    # Transformers versions differ in whether templates return a BatchEncoding.
    if hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    return tokenizer, [int(t) for t in tokens]


def benchmark(args, spec, profile, draft_spec=None):
    from opentpu.target import config
    from opentpu.llm.nanbeige import Engine, load_weights
    from opentpu.llm.nanbeige_weights import FloatSafetensorWeights
    tokenizer, prompt = _tokenize(args.model_dir, args.prompt)
    if not prompt or len(prompt) + max(args.max_new_tokens - 1, 0) > args.cap:
        raise ValueError("prompt plus requested generation exceeds --cap; increase capacity")
    if any(t < 0 or t >= spec.vocab for t in prompt):
        raise ValueError("tokenizer emitted a token outside target vocabulary")
    weights = load_weights(args.model_dir)
    session = BoardSession(args.device, profile) if args.backend == "board" else None
    try:
        cfg = session.cfg if session else config(profile)
        started = time.perf_counter()
        engine = Engine(spec, weights, cap=args.cap, cfg=cfg,
                        backend=session.factory if session else "isa", rows=args.rows,
                        resident=False, pipeline=False, prompt_runs=False,
                        wformat="int8", head_format="int8")
        setup_seconds = time.perf_counter() - started
        # Preserve model-specific forward compilation; do not use generic stream-fill kernels.
        engine.stream_logits = False
        engine.image.stream_fill = False
        draft, decoder, draft_setup_seconds = None, None, 0.0
        if args.mode in ("hybrid", "compare"):
            from opentpu.llm.dspark import Reference
            from opentpu.llm.nanbeige_specdec import Decoder
            started = time.perf_counter()
            draft_weights = FloatSafetensorWeights(args.draft_dir)
            if args.draft_backend == "board":
                from opentpu.llm.dspark_fpga import Engine as FPGADraft
                query_rows = args.draft_query_tokens or draft_spec.block_size
                draft = FPGADraft(engine, draft_spec, draft_weights,
                                  target_weights=weights, rows=query_rows,
                                  device_fusion_markov=(args.draft_fusion_markov == "board"))
            else:
                draft = Reference(draft_spec, draft_weights, target_weights=weights,
                                  target_vocab_size=spec.vocab)
            draft_setup_seconds = time.perf_counter() - started
            decoder = Decoder(engine, draft, draft_tokens=args.draft_tokens,
                              prefill_chunk=args.prefill_chunk,
                              draft_query_tokens=args.draft_query_tokens)

        def run_mode(mode):
            target_start = len(engine.stats)
            draft_start = len(getattr(draft, "stats", []))
            if mode == "target":
                result = run_target(engine, prompt, args.max_new_tokens, args.prefill_chunk)
                result["draft_backend"] = None
                target_stats = engine.stats[target_start:]
            else:
                generated = decoder.generate(prompt, max_new_tokens=args.max_new_tokens)
                result = asdict(generated.stats)
                result.update(tokens=list(generated.tokens), stop_reason=generated.stop_reason,
                              decode_tps=generated.stats.decode_tps,
                              acceptance_rate=generated.stats.acceptance_rate)
                # Target-only samples collected at each target call exclude draft runs,
                # including accelerator drafts sharing the same transport/backend.
                target_stats = result.pop("target_run_stats")
                draft_stats = list(getattr(draft, "stats", []))[draft_start:]
                result.update(draft_raw_runs=draft_stats, draft_cycles=_sum_cycles(draft_stats),
                              draft_cycles_scope="all draft accelerator runs, including enabled device fusion/Markov; excludes host operations",
                              draft_cycles_are_hardware_measured=(args.backend == "board"
                                  and args.draft_backend == "board" and bool(draft_stats)
                                  and _sum_cycles(draft_stats) is not None))
            result.update(backend_evidence(target_stats, args.backend,
                                          result["target_prefill_runs"]))
            result["text"] = tokenizer.decode(result["tokens"], skip_special_tokens=False)
            return result

        modes = ["target", "hybrid"] if args.mode == "compare" else [args.mode]
        collected = collect_runs(run_mode, modes, warmup_runs=args.warmup_runs, repeat=args.repeat)
        runs = collected["runs"]
        precision = dict(PRECISION)
        accelerator_draft = draft is not None and args.draft_backend == "board"
        if accelerator_draft:
            precision["draft_execution"] = (
                "openTPU INT4/W4A8 noncausal backbone and context K/V projections; shared target "
                "INT8 head; " + (
                    "FPGA INT8 context fusion and Markov projection; host embedding/Markov row lookup, "
                    "greedy selection and confidence"
                    if args.draft_fusion_markov == "board" else
                    "CPU FP32 context fusion/hidden norm, embedding lookup, "
                    "original BF16 Markov/confidence weights expanded to FP32"))
            precision["draft_implementation_details"] = getattr(draft, "precision", None)
        report = {"mode": args.mode, "backend": args.backend, **collected,
                  "board_info": session.info if session else None,
                  "architecture": asdict(cfg), "setup_seconds": setup_seconds,
                  "draft_setup_seconds": draft_setup_seconds,
                  "prompt": args.prompt, "prompt_tokens": prompt, "thinking": False,
                  "max_new_tokens": args.max_new_tokens, "cap": args.cap, "resident": False,
                  "capture_layers": list(spec.capture_layers), "precision": precision,
                  "draft_backend_selected": args.draft_backend,
                  "draft_fusion_markov_selected": args.draft_fusion_markov,
                  "timing_note": "setup and explicit warmups excluded; measured repetitions share compiled programs; wall time includes execution, transfers, host work and any remaining compilation",
                  "runs_compatibility": "runs contains the last measured repetition; measured_runs contains every repetition",
                  "fpga_draft_implemented": accelerator_draft,
                  "fpga_draft_hardware_executed": accelerator_draft and args.backend == "board",
                  "draft_fully_on_fpga": False}
        if args.mode == "compare":
            matches = [entry["runs"]["target"]["tokens"] == entry["runs"]["hybrid"]["tokens"]
                       for entry in collected["measured_runs"]]
            same = all(matches)
            target_tps = collected["aggregates"]["target"]["decode_tps_median"]
            hybrid_tps = collected["aggregates"]["hybrid"]["decode_tps_median"]
            report["comparison"] = {
                "exact_target_tokens_match": same,
                "exact_target_tokens_match_by_repeat": matches,
                "same_target_image_and_precision": True,
                "baseline_includes_capture_writes": bool(spec.capture_layers),
                "order": ["target", "hybrid"],
                "cache_note": "shared Engine; hybrid may reuse target compilation caches",
                "observed_decode_tps_ratio": (runs["hybrid"]["decode_tps"] / runs["target"]["decode_tps"])
                    if same and runs["target"]["decode_tps"] else None,
                "median_decode_tps_ratio": hybrid_tps / target_tps
                    if same and target_tps and hybrid_tps is not None else None,
            }
        return report
    finally:
        if session:
            session.close()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, report):
    encoded = json.dumps(report, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n"
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                             suffix=".tmp", delete=False) as stream:
                name = stream.name
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)
    print(encoded, end="")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("inspect", "benchmark"))
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--draft-dir", type=Path)
    p.add_argument("--profile", type=Path, default=PROFILE)
    p.add_argument("--backend", choices=("board", "isa"), default="board")
    p.add_argument("--draft-backend", choices=("cpu", "board"), default="cpu",
                   help="board uses accelerator draft kernels on the selected target backend")
    p.add_argument("--device", default="f2:0")
    p.add_argument("--mode", choices=("target", "hybrid", "compare"), default="compare")
    p.add_argument("--prompt", default="What is the capital of India? Reply briefly.")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--cap", type=int, default=128)
    p.add_argument("--draft-tokens", type=int, default=3)
    p.add_argument("--draft-query-tokens", type=int, default=None,
                   help="DSpark noncausal query block (default: checkpoint block_size)")
    p.add_argument("--draft-fusion-markov", choices=("host", "board"), default="host",
                   help="placement of context fusion and Markov projection for the FPGA draft")
    p.add_argument("--prefill-chunk", type=int, default=4)
    p.add_argument("--warmup-runs", type=int, default=0,
                   help="discard this many runs of every requested mode before measurement")
    p.add_argument("--repeat", type=int, default=1,
                   help="number of measured repetitions per requested mode")
    p.add_argument("--json", type=Path)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.cap < 1 or args.draft_tokens < 0 or args.max_new_tokens < 0 or args.prefill_chunk < 1:
        p.error("cap/prefill-chunk must be positive; token counts must be nonnegative")
    if args.draft_query_tokens is not None and args.draft_query_tokens < 1:
        p.error("draft-query-tokens must be positive")
    if args.warmup_runs < 0 or args.repeat < 1:
        p.error("warmup-runs must be nonnegative and repeat must be positive")
    if args.draft_fusion_markov == "board" and args.draft_backend != "board":
        p.error("board fusion/Markov requires --draft-backend board")
    if args.command == "benchmark" and args.mode != "target" and args.draft_dir is None:
        p.error("hybrid/compare requires --draft-dir")
    args.rows = max(args.draft_tokens + 1, args.prefill_chunk)
    if args.rows > args.cap:
        p.error("rows required by draft/prefill exceed --cap")
    from opentpu.target import load, config
    from opentpu.llm.nanbeige import Spec
    from opentpu.llm.dspark import Spec as DraftSpec
    profile = load(str(args.profile))
    spec = Spec.from_hf(args.model_dir)
    draft_spec = DraftSpec.from_hf(args.draft_dir, target_hidden_size=spec.hidden) \
        if args.draft_dir else None
    if draft_spec and args.mode != "target":
        spec = replace(spec, capture_layers=draft_spec.capture_layers)
    if args.command == "inspect":
        report = {"command": "inspect", "hardware_validated": False,
                  "profile": profile.to_dict(), "model": asdict(spec), "precision": PRECISION,
                  "layout": inspect_layout(spec, config(profile), args.cap, args.rows)}
    else:
        report = benchmark(args, spec, profile, draft_spec)
        report["command"] = "benchmark"
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.json, report)
    if args.command == "benchmark" and args.mode == "compare" and not report["comparison"]["exact_target_tokens_match"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
