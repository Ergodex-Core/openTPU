"""Physical F2 A/B measurement on the frozen 11-prompt Gemma 3 held-out set.

Baseline is ordinary Engine.generate_card; candidate is the EOS-vetoed,
on-device GemmaPromptLookupDecoder. Both modes have the same resident image
layout, independent Engines in one process, and no host drafter or sampler.
"cold" means only the first request of a fresh Engine, never process cold.
Results are in-memory callback observations, not terminal or service TPS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.gemma_specdec_card_measure import (  # noqa: E402
    CAP, CONFIG_SHA256, CORE_KHZ, BUILD_ID, _check_engine, _log, _persist, _require,
)

PLAN_SHA256 = "db6bfd21659b0b60fbd1f17a62641dae263f604f2496fcb3ec697921653d7dee"
DECODER_SHA256 = "9c0a0bb5d45e0d2e13ff969892653ffbe97f9cc8e4b2c7b2d333b9818b209a8e"
QUALIFIED_DECODER_COMMIT = "23c2c0666372ceceb70d979554aa2f5c4c5a2e03"
MAX_NEW = 64
PASSES = ("first_pass", "warm_pass")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str:
    revision = ROOT / "REVISION"
    if revision.is_file():
        commit = revision.read_text().strip()
        _require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
                 "invalid exported source revision")
        return commit
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _tokenizer_hashes(model: Path) -> dict[str, str]:
    names = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json")
    hashes = {name: _sha256(model / name) for name in names if (model / name).is_file()}
    _require("tokenizer.json" in hashes or "tokenizer.model" in hashes,
             "no local tokenizer data file found")
    return hashes


def _prompts(plan: dict, tokenizer) -> list[dict]:
    frozen = plan.get("frozen_prompts")
    _require(isinstance(frozen, list) and len(frozen) == 11,
             "the frozen plan must contain exactly 11 prompts")
    _require(plan.get("per_prompt_output_limit") == MAX_NEW,
             "frozen per-prompt output limit differs from 64")
    seen, prompts = set(), []
    for item in frozen:
        _require(isinstance(item, dict) and isinstance(item.get("id"), str)
                 and isinstance(item.get("category"), str), "invalid prompt metadata")
        _require(item["id"] not in seen, f"duplicate prompt id {item['id']}")
        seen.add(item["id"])
        if "messages" in item:
            _require("user" not in item and isinstance(item["messages"], list)
                     and item["messages"], f"invalid message list for {item['id']}")
            messages = item["messages"]
        else:
            _require(isinstance(item.get("user"), str),
                     f"missing user message for {item['id']}")
            messages = [{"role": "user", "content": item["user"]}]
        _require(all(isinstance(m, dict) and m.get("role") in ("user", "assistant")
                     and isinstance(m.get("content"), str) for m in messages)
                 and messages[-1]["role"] == "user",
                 f"invalid chat messages for {item['id']}")
        encoded = tokenizer.apply_chat_template(messages, tokenize=True,
                                                add_generation_prompt=True,
                                                enable_thinking=False)
        ids = [int(x) for x in (encoded["input_ids"] if hasattr(encoded, "keys") else encoded)]
        _require(ids and len(ids) + MAX_NEW <= CAP,
                 f"{item['id']}: prompt length plus output budget exceeds {CAP}")
        prompts.append({"id": item["id"], "category": item["category"],
                        "messages": messages, "prompt_ids": ids,
                        "prompt_tokens": len(ids)})
    return prompts


def _program_state(eng, decoder) -> dict:
    return {"disk_prog_cache_enabled": bool(eng.prog_cache),
            "resident_decode_buckets": len(eng._decodes),
            "ordinary_generate_programs": len(eng._gens),
            "prompt_lookup_buckets": len(decoder._done) if decoder is not None else 0}


def _measure(eng, decoder, mode: str, pass_name: str, index: int,
             prompt: dict, tokenizer, partial: dict) -> dict:
    eng.reset()
    _require(eng.pos == 0, "Engine reset failed")
    ids = prompt["prompt_ids"]
    stop_ids = [int(x) for x in eng.spec.eos]
    _require(ids and len(ids) + MAX_NEW <= eng.cap, "prompt exceeds pinned output budget")
    raw: list[int] = []
    arrivals: list[float] = []
    label = ("engine_cold" if pass_name == "first_pass" and index == 0 else
             "cold_request" if pass_name == "first_pass" else "warm_request")
    partial.update(mode=mode, pass_name=pass_name, request_label=label,
                   prompt_id=prompt["id"], prompt_ids=ids, raw_token_ids=raw,
                   callback_arrivals_s=arrivals)
    programs_before = _program_state(eng, decoder)

    def on_token(token) -> None:
        raw.append(int(token))
        arrivals.append(time.perf_counter())

    request_begin = time.perf_counter()
    partial["request_begin_s"] = request_begin
    prefill_at = len(eng.stats)
    for _ in eng.prefill_chunks(ids[:-1], logits=False):
        pass
    prefill_end = time.perf_counter()
    partial["prefill_end_s"] = prefill_end
    _require(eng.pos == len(ids) - 1, "prefill did not leave the final token unfed")
    prefill_stats = eng.stats[prefill_at:]
    _require(prefill_stats and all(isinstance(s, dict) and isinstance(s.get("cycles"), int)
                 and s["cycles"] > 0 for s in prefill_stats),
             "missing physical prefill counters")
    decode_at = len(eng.stats)
    if mode == "baseline":
        got = eng.generate_card(ids[-1], MAX_NEW, stop_ids=stop_ids, on_token=on_token)
        sd = None
    else:
        got = decoder.generate_card(ids[-1], MAX_NEW, stop_ids=stop_ids,
                                    on_token=on_token, context=ids)
        sd = decoder.last
    request_end = time.perf_counter()
    partial["request_end_s"] = request_end
    partial["decode_run_stats"] = eng.stats[-1] if len(eng.stats) > decode_at else None
    _require([int(x) for x in got] == raw and len(arrivals) == len(raw),
             "callback arrivals differ from generated device IDs")
    _require(raw and len(raw) <= MAX_NEW, "empty or over-budget device output")
    _require(len(eng.stats) == decode_at + 1,
             "decode did not complete in one physical BoardBackend run")
    run = eng.stats[-1]
    _require(isinstance(run, dict) and isinstance(run.get("cycles"), int)
             and run["cycles"] > 0, "missing physical decode cycles")
    eos_positions = [i for i, token in enumerate(raw) if token in stop_ids]
    _require(not eos_positions or eos_positions == [len(raw) - 1],
             "EOS appeared before the final emitted token")
    _require(eos_positions or len(raw) == MAX_NEW,
             "device stopped before EOS or output budget")
    visible = raw[:-1] if eos_positions else list(raw)
    visible_times = arrivals[:len(visible)]
    intervals = max(len(visible) - 1, 0)
    interval_s = (visible_times[-1] - visible_times[0]) if intervals else 0.0
    _require(not intervals or interval_s > 0, "nonincreasing visible callback times")
    clock_s = run["cycles"] / (CORE_KHZ * 1000)
    row = {
        "mode": mode, "pass_name": pass_name, "request_label": label,
        "engine_cold": label == "engine_cold", "prompt_id": prompt["id"],
        "prompt_ids": ids, "prompt_tokens": len(ids), "output_budget": MAX_NEW,
        "effective_output_budget": min(MAX_NEW, eng.cap - (len(ids) - 1)),
        "stop_ids": stop_ids, "raw_token_ids": raw, "visible_token_ids": visible,
        "visible_text": tokenizer.decode(visible, skip_special_tokens=True),
        "eos_id": raw[-1] if eos_positions else None,
        "termination": "eos" if eos_positions else "output_limit",
        "callback_arrivals_s": arrivals, "request_begin_s": request_begin,
        "prefill_end_s": prefill_end, "request_end_s": request_end,
        "request_wall_s": request_end - request_begin,
        "prefill_wall_s": prefill_end - request_begin,
        "ttft_s": visible_times[0] - request_begin if visible_times else None,
        "visible_decode_intervals": intervals,
        "visible_decode_wall_s": interval_s,
        "visible_decode_tps": intervals / interval_s if intervals else None,
        "prefill_cycles": sum(s["cycles"] for s in prefill_stats),
        "decode_cycles": run["cycles"], "core_khz": CORE_KHZ,
        "raw_steps": len(raw), "raw_device_s": clock_s,
        "raw_device_steps_per_s": len(raw) / clock_s,
        "device_run_stats": run, "program_state_before": programs_before,
        "program_state_after": _program_state(eng, decoder),
        "committed_input_rows": len(raw), "last_emitted_token_unfed": True,
        "next_input_position": eng.pos,
    }
    _require(eng.pos == len(ids) - 1 + len(raw), "logical output position differs")
    if sd is None:
        row["target_rows"] = len(raw)
    else:
        from opentpu.llm import generate as G

        _require(sd.run is run and sd.iterations > 0
                 and sd.verifies >= sd.accepted >= 0,
                 "invalid physical speculative counters")
        _require(sd.iterations + sd.accepted == len(raw),
                 "speculative output accounting differs from emitted raw IDs")
        state_at = eng.image.lookup["gen"]["state"]
        state = eng.backend.read(0, state_at, 4 * G.STATE_WORDS).view("<f4")
        committed = {"unfed_token": int(state[G.S_TOK]),
                     "next_input_position_in_bucket": int(state[G.S_TPOS]),
                     "remaining_output_budget": int(state[G.S_LEFT]),
                     "engine_position": int(eng.pos)}
        _require(committed["unfed_token"] == raw[-1]
                 and committed["next_input_position_in_bucket"] == eng.pos % eng.block
                 and committed["remaining_output_budget"] == MAX_NEW - len(raw)
                 and committed["engine_position"] == len(ids) - 1 + len(raw),
                 "device state does not match logically committed output positions")
        row.update(iterations=sd.iterations, verifies=sd.verifies,
                   accepted=sd.accepted, rejected=sd.verifies - sd.accepted,
                   target_rows=sd.iterations + sd.verifies,
                   speculative_compile_s=sd.compile_s,
                   committed_device_state=committed,
                   physical_verifier_row_note="extra speculative row is not a committed token")
    return row


def _aggregate(rows: list[dict]) -> dict:
    intervals = sum(r["visible_decode_intervals"] for r in rows)
    interval_s = sum(r["visible_decode_wall_s"] for r in rows)
    total_wall = sum(r["request_wall_s"] for r in rows)
    visible = sum(len(r["visible_token_ids"]) for r in rows)
    raw = sum(r["raw_steps"] for r in rows)
    cycles = sum(r["decode_cycles"] for r in rows)
    def counter(name):
        values = [r["device_run_stats"].get(name) for r in rows]
        return sum(values) if all(isinstance(x, int) for x in values) else None
    return {
        "scope": "in-memory callback; repeated measurements of 11 unique contexts; request wall includes prefill/compile",
        "unique_prompt_count": len({r["prompt_id"] for r in rows}) if all("prompt_id" in r for r in rows) else None,
        "requests": len(rows), "visible_tokens": visible, "raw_steps": raw,
        "interval_eligible_requests": sum(r["visible_decode_intervals"] > 0 for r in rows),
        "interval_ineligible_requests": sum(r["visible_decode_intervals"] == 0 for r in rows),
        "weighted_visible_intervals": intervals,
        "summed_first_to_last_visible_s": interval_s,
        "weighted_interval_wall_tps": intervals / interval_s if interval_s else None,
        "summed_full_request_wall_s": total_wall,
        "visible_tokens_per_total_request_wall_s": visible / total_wall if total_wall else None,
        "raw_steps_per_total_request_wall_s": raw / total_wall if total_wall else None,
        "summed_decode_cycles": cycles,
        "raw_steps_per_device_s": raw / (cycles / (CORE_KHZ * 1000)) if cycles else None,
        "target_rows": sum(r["target_rows"] for r in rows),
        "committed_input_rows": sum(r["committed_input_rows"] for r in rows),
        "iterations": sum(r["iterations"] for r in rows) if all("iterations" in r for r in rows) else None,
        "verifies": sum(r["verifies"] for r in rows) if all("verifies" in r for r in rows) else None,
        "accepted": sum(r["accepted"] for r in rows) if all("accepted" in r for r in rows) else None,
        "rejected": sum(r["rejected"] for r in rows) if all("rejected" in r for r in rows) else None,
        "b_reads": counter("b_reads"), "a_reads": counter("a_reads"),
        "b_writes": counter("b_writes"), "a_writes": counter("a_writes"),
        "b_stall": counter("b_stall"),
        "native_counter_units": "raw device counters; no conversion to payload bytes",
        "draft_lookup_cost": "included in device cycles; not separately instrumented",
    }


def main() -> int:
    from transformers import AutoTokenizer
    from opentpu.host.chat import make_backend
    from opentpu.llm.gemma3 import Spec
    from opentpu.llm.gemma_specdec import GemmaPromptLookupDecoder
    from opentpu.llm.qwen3 import Engine, load_weights

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path,
                    default=Path("/home/ubuntu/opentpu/models/gemma-3-1b-it"))
    ap.add_argument("--dev", default="f2:0")
    args = ap.parse_args()
    log = args.out.with_suffix(".jsonl")
    _require(not args.out.exists() and not log.exists(),
             "output or log exists; choose a new --out path")
    _require(args.plan.is_file() and _sha256(args.plan) == PLAN_SHA256,
             "frozen held-out plan is missing or its SHA-256 changed")
    plan = json.loads(args.plan.read_text())
    _require(plan.get("status", "").startswith("planned only"),
             "unexpected held-out plan status")
    config = args.model / "config.json"
    _require(config.is_file() and _sha256(config) == CONFIG_SHA256,
             "measured Gemma-3-1B config missing or changed")
    decoder_path = ROOT / "opentpu/llm/gemma_specdec.py"
    _require(_sha256(decoder_path) == DECODER_SHA256,
             "qualified EOS-veto decoder source changed")
    commit = _source_commit()
    spec = Spec.from_hf(args.model)
    _require(spec.layers == 26 and spec.hidden == 1152 and spec.vocab == 262144,
             "unexpected Gemma architecture")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = _prompts(plan, tokenizer)  # validate all 11 before opening a device
    hashes = _tokenizer_hashes(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "scope": "physical F2 in-memory callback; baseline vs EOS-vetoed prompt lookup",
        "source_commit": commit, "decoder_sha256": DECODER_SHA256,
        "qualified_decoder_commit": QUALIFIED_DECODER_COMMIT,
        "source_sha256": {name: _sha256(ROOT / name) for name in (
            "tools/gemma_specdec_heldout_measure.py", "opentpu/llm/gemma_specdec.py",
            "opentpu/llm/gemma4.py", "opentpu/llm/qwen3.py", "opentpu/llm/generate.py")},
        "plan": str(args.plan), "plan_sha256": PLAN_SHA256,
        "model": str(args.model), "config_sha256": CONFIG_SHA256,
        "tokenizer_sha256": hashes, "prompts": prompts,
        "cap": CAP, "output_budget": MAX_NEW, "sampling": "greedy",
        "weights_format": "int8", "head_format": "int8",
        "expected_build_id": f"0x{BUILD_ID:08x}", "expected_core_khz": CORE_KHZ,
        "device": args.dev, "passes": list(PASSES),
        "mode_order": ["baseline", "candidate"],
        "order_limitation": "fixed mode order; small wall-only differences may include host/thermal drift",
        "engine_cold_scope": "first prompt of first pass only, separately for each fresh Engine",
        "draft_lookup_cost": "included in candidate cycles; not separately measured",
        "runs": [], "complete": False,
    }
    _persist(args.out, doc)
    try:
        for mode in ("baseline", "candidate"):
            _log(log, {"event": "engine_begin", "mode": mode,
                       "host_s": time.perf_counter()})
            backend_factory, cfg = make_backend("board", spec, CAP, args.dev,
                                                args.model.name, lookup=True,
                                                wformat="int8", head_format="int8")
            eng = None
            try:
                eng = Engine(spec, load_weights(args.model), cap=CAP, cfg=cfg,
                             backend=backend_factory, resident=True,
                             prog_cache=False, snapshots=0, pipeline=False,
                             wformat="int8", head_format="int8",
                             gemma_prompt_lookup=True)
                _check_engine(eng, mode)
                decoder = GemmaPromptLookupDecoder(eng) if mode == "candidate" else None
                layout = {"image_bytes": eng.backend.image_bytes, "rows": eng.image.rows,
                          "cap": eng.cap, "wformat": eng.image.wformat,
                          "head_format": eng.image.head_format}
                if "image_layout" in doc:
                    _require(doc["image_layout"] == layout,
                             "baseline and candidate image layouts differ")
                else:
                    doc["image_layout"] = layout
                    doc["board_info"] = eng.backend.info
                    doc["actual_stop_ids"] = [int(x) for x in eng.spec.eos]
                for pass_name in PASSES:
                    for i, prompt in enumerate(prompts):
                        key = (pass_name, prompt["id"])
                        _log(log, {"event": "request_begin", "mode": mode,
                                   "pass_name": pass_name, "prompt_id": prompt["id"],
                                   "host_s": time.perf_counter()})
                        partial: dict = {}
                        try:
                            row = _measure(eng, decoder, mode, pass_name, i,
                                           prompt, tokenizer, partial)
                        except BaseException:
                            doc["failed_run"] = partial
                            _persist(args.out, doc)
                            raise
                        doc["runs"].append(row)
                        _persist(args.out, doc)
                        if mode == "candidate":
                            baseline = next((r for r in doc["runs"] if r["mode"] == "baseline"
                                             and (r["pass_name"], r["prompt_id"]) == key), None)
                            _require(baseline is not None and
                                     row["raw_token_ids"] == baseline["raw_token_ids"],
                                     f"candidate differs from baseline at {key}")
                        previous = next((r for r in doc["runs"] if r["mode"] == mode
                                         and r["pass_name"] == "first_pass"
                                         and r["prompt_id"] == prompt["id"]), None)
                        if pass_name == "warm_pass":
                            _require(previous is not None and
                                     row["raw_token_ids"] == previous["raw_token_ids"],
                                     f"cross-pass IDs changed for {mode} {prompt['id']}")
                        concise = {k: row[k] for k in
                                   ("mode", "pass_name", "prompt_id", "request_label",
                                    "raw_steps", "ttft_s", "visible_decode_tps",
                                    "request_wall_s", "decode_cycles", "target_rows")}
                        _log(log, {"event": "request_complete", **concise})
                        print(json.dumps(concise), flush=True)
            finally:
                if eng is not None:
                    eng.backend.close()
        groups = {mode: {pass_name: [r for r in doc["runs"]
                                    if r["mode"] == mode and r["pass_name"] == pass_name]
                         for pass_name in PASSES}
                  for mode in ("baseline", "candidate")}
        _require(all(len(groups[m][p]) == len(prompts)
                     for m in groups for p in PASSES), "incomplete held-out A/B matrix")
        doc["aggregate"] = {m: {p: _aggregate(groups[m][p]) for p in PASSES}
                            for m in groups}
        doc["aggregate_all_passes"] = {
            m: _aggregate([r for p in PASSES for r in groups[m][p]]) for m in groups}
        doc["complete"] = True
        _persist(args.out, doc)
        _log(log, {"event": "complete", "host_s": time.perf_counter()})
        return 0
    except BaseException as exc:
        doc["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _persist(args.out, doc)
        _log(log, {"event": "failed", **doc["error"]})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
