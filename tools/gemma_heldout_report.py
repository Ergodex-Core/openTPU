"""Audit a completed physical Gemma prompt-lookup held-out A/B receipt.

This is a JSON-only, causal reconstruction, not a model run or an estimate of
lookup-only cycles. Greedy target IDs score each proposal after that proposal
has been selected from the already committed prefix. The hardware counters
are checked against, never supplied to, the replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PLAN_SHA256 = "db6bfd21659b0b60fbd1f17a62641dae263f604f2496fcb3ec697921653d7dee"
MODES = ("baseline", "candidate")
PASSES = ("first_pass", "warm_pass")
COUNTERS = ("a_reads", "b_reads", "a_writes", "b_writes", "b_stall")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def integer(value, where: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{where}: expected integer >= {minimum}")
    return value


def number(value, where: str, minimum: float = 0.0) -> float:
    require(type(value) in (int, float) and value >= minimum,
            f"{where}: expected finite number >= {minimum}")
    import math
    require(math.isfinite(value), f"{where}: nonfinite number")
    return float(value)


def ids(value, where: str) -> list[int]:
    require(isinstance(value, list) and all(type(x) is int and x >= 0 for x in value),
            f"{where}: expected nonnegative integer IDs")
    return value


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    return candidate / baseline if candidate is not None and baseline is not None and baseline > 0 else None


def replay(prompt: list[int], raw: list[int], stop_ids: set[int], budget: int,
           cap: int, block: int) -> dict:
    """Reconstruct E/V dispatch using only prompt and committed target IDs.

    The first program is E. Each completed iteration updates the committed
    prefix, then searches the latest earlier matching token for its following
    token. A stop-ID proposal is vetoed even if it would match. No target ID
    beyond the current iteration is consulted to choose the next program.
    """
    require(prompt and raw and len(raw) <= budget, "replay: invalid prompt/output length")
    require(len(prompt) - 1 + budget <= cap, "replay: budget exceeds image capacity")
    require(block >= 2 and cap >= 2, "replay: invalid block/capacity")
    prefix = list(prompt)
    kind, proposal = "E", None
    emitted = iterations = verifies = accepted = stop_vetoes = 0
    decisions = []
    while emitted < len(raw):
        at = emitted
        first = raw[at]
        remaining = budget - emitted
        verify = kind == "V"
        iterations += 1
        if verify:
            verifies += 1
            require(proposal is not None and proposal not in stop_ids and remaining >= 2,
                    f"replay: ineligible V at output {at}")
        take_extra = verify and proposal == first and first not in stop_ids and remaining >= 2
        if take_extra:
            require(at + 1 < len(raw), f"replay: accepted V lacks second ID at output {at}")
            accepted += 1
            produced = raw[at:at + 2]
        else:
            produced = [first]
        require(all(t not in stop_ids for t in produced[:-1]),
                f"replay: stop before end of iteration at output {at}")
        prefix.extend(produced)
        emitted += len(produced)
        decisions.append({"output_index": at, "input_position": len(prefix) - len(produced) - 1,
                          "program": kind, "proposal": proposal if verify else None,
                          "first_target_id": first, "accepted_extra": bool(take_extra),
                          "emitted_ids": produced})
        terminal = produced[-1] in stop_ids or emitted == budget
        if terminal:
            require(emitted == len(raw), "replay: hardware emitted IDs past EOS/budget")
            break
        require(emitted < len(raw), "replay: raw IDs ended before EOS/budget")
        pn = len(prefix) - 1
        require(pn < cap, "replay: next input position exceeds capacity")
        latest = next((j for j in range(pn - 1, -1, -1) if prefix[j] == prefix[pn]), None)
        proposal = prefix[latest + 1] if latest is not None else None
        bucket_end = min(cap, (pn // block + 1) * block)
        room_for_two = pn <= bucket_end - 2
        budget_for_two = budget - emitted >= 2
        otherwise_eligible = proposal is not None and room_for_two and budget_for_two
        if otherwise_eligible and proposal in stop_ids:
            stop_vetoes += 1
        kind = "V" if otherwise_eligible and proposal not in stop_ids else "E"
        decisions[-1]["next_lookup_proposal"] = proposal
        decisions[-1]["next_program"] = kind
        decisions[-1]["next_veto_reason"] = (
            "no_match" if proposal is None else
            "bucket_or_cap" if not room_for_two else
            "budget" if not budget_for_two else
            "stop_proposal" if proposal in stop_ids else None)
    require(emitted == len(raw), "replay: did not consume every hardware output ID")
    return {"iterations": iterations, "accepted": accepted, "verifies": verifies,
            "rejected": verifies - accepted, "target_rows": iterations + verifies,
            "raw_emitted": emitted, "committed_input_rows": emitted,
            "otherwise_eligible_stop_proposal_vetoes": stop_vetoes,
            "lookup_calls_including_terminal": iterations,
            "lookup_call_note": "The E/V program computes a next lookup even on its terminal iteration; isolated lookup cycles were not measured.",
            "decisions": decisions}


def measured(row: dict) -> dict:
    raw = ids(row.get("raw_token_ids"), "run.raw_token_ids")
    visible = ids(row.get("visible_token_ids"), "run.visible_token_ids")
    require(len(raw) == integer(row.get("raw_steps"), "run.raw_steps"), "raw step count differs")
    require(len(raw) == integer(row.get("committed_input_rows"), "run.committed_input_rows"),
            "committed input row count differs")
    stop = set(ids(row.get("stop_ids"), "run.stop_ids"))
    termination = row.get("termination")
    require(termination in ("eos", "output_limit"), "missing termination kind")
    require(bool(raw and raw[-1] in stop) == (termination == "eos"), "EOS termination differs")
    require(visible == (raw[:-1] if termination == "eos" else raw), "visible IDs differ")
    budget = integer(row.get("effective_output_budget"), "run.effective_output_budget", 1)
    require(len(raw) <= budget, "output exceeds effective budget")
    require(termination != "output_limit" or len(raw) == budget, "early output-limit termination")
    intervals = integer(row.get("visible_decode_intervals"), "run.visible_decode_intervals")
    require(intervals == max(len(visible) - 1, 0), "visible interval count differs")
    interval_s = number(row.get("visible_decode_wall_s"), "run.visible_decode_wall_s")
    wall_s = number(row.get("request_wall_s"), "run.request_wall_s")
    cycles = integer(row.get("decode_cycles"), "run.decode_cycles", 1)
    khz = integer(row.get("core_khz"), "run.core_khz", 1)
    ttft = row.get("ttft_s")
    if visible:
        ttft = number(ttft, "run.ttft_s")
    else:
        require(ttft is None, "TTFT exists for zero visible outputs")
    require((interval_s > 0) == (intervals > 0), "visible interval duration differs")
    stats = row.get("device_run_stats")
    require(isinstance(stats, dict) and stats.get("cycles") == cycles,
            "device run cycles differ")
    native = {key: integer(stats[key], f"run.device_run_stats.{key}")
              if key in stats and stats[key] is not None else None for key in COUNTERS}
    return {"raw_token_ids": raw, "visible_token_ids": visible,
            "stop_ids": sorted(stop), "request_label": row.get("request_label"),
            "engine_cold": row.get("engine_cold"),
            "raw_emitted": len(raw), "visible_emitted": len(visible),
            "termination": termination, "eos_id": row.get("eos_id"),
            "effective_output_budget": budget, "ttft_s": ttft,
            "request_wall_s": wall_s, "prefill_wall_s": number(row.get("prefill_wall_s"), "run.prefill_wall_s"),
            "visible_decode_intervals": intervals,
            "visible_decode_wall_s": interval_s,
            "weighted_interval_wall_tps": intervals / interval_s if intervals else None,
            "visible_tokens_per_request_wall_s": len(visible) / wall_s if wall_s else None,
            "raw_steps_per_request_wall_s": len(raw) / wall_s if wall_s else None,
            "decode_cycles": cycles, "core_khz": khz,
            "prefill_cycles": integer(row.get("prefill_cycles"), "run.prefill_cycles"),
            "raw_device_steps_per_s": len(raw) * khz * 1000 / cycles,
            "native_counters": native,
            "target_rows": integer(row.get("target_rows"), "run.target_rows"),
            "committed_input_rows": integer(row.get("committed_input_rows"), "run.committed_input_rows"),
            "iterations": row.get("iterations"), "accepted": row.get("accepted"),
            "verifies": row.get("verifies"), "rejected": row.get("rejected")}


def aggregate(rows: list[dict]) -> dict:
    require(rows, "cannot aggregate empty group")
    raw = sum(r["raw_emitted"] for r in rows)
    visible = sum(r["visible_emitted"] for r in rows)
    intervals = sum(r["visible_decode_intervals"] for r in rows)
    interval_s = sum(r["visible_decode_wall_s"] for r in rows)
    wall_s = sum(r["request_wall_s"] for r in rows)
    cycles = sum(r["decode_cycles"] for r in rows)
    khz = rows[0]["core_khz"]
    require(all(r["core_khz"] == khz for r in rows), "mixed hardware clocks")
    native = {key: sum(r["native_counters"][key] for r in rows)
              if all(r["native_counters"][key] is not None for r in rows) else None
              for key in COUNTERS}
    return {"requests": len(rows), "raw_emitted": raw, "visible_emitted": visible,
            "eos_requests": sum(r["termination"] == "eos" for r in rows),
            "output_limit_requests": sum(r["termination"] == "output_limit" for r in rows),
            "interval_eligible_requests": sum(r["visible_decode_intervals"] > 0 for r in rows),
            "interval_ineligible_requests": sum(r["visible_decode_intervals"] == 0 for r in rows),
            "weighted_visible_intervals": intervals,
            "summed_first_to_last_visible_s": interval_s,
            "weighted_interval_wall_tps": intervals / interval_s if interval_s else None,
            "summed_full_request_wall_s": wall_s,
            "visible_tokens_per_total_request_wall_s": visible / wall_s if wall_s else None,
            "raw_steps_per_total_request_wall_s": raw / wall_s if wall_s else None,
            "summed_decode_cycles": cycles,
            "raw_steps_per_device_s": raw * khz * 1000 / cycles,
            "target_rows": sum(r["target_rows"] for r in rows),
            "committed_input_rows": sum(r["committed_input_rows"] for r in rows),
            "iterations": sum(r["iterations"] for r in rows) if all(r["iterations"] is not None for r in rows) else None,
            "accepted": sum(r["accepted"] for r in rows) if all(r["accepted"] is not None for r in rows) else None,
            "verifies": sum(r["verifies"] for r in rows) if all(r["verifies"] is not None for r in rows) else None,
            "rejected": sum(r["rejected"] for r in rows) if all(r["rejected"] is not None for r in rows) else None,
            "native_counters": native}


def compare(candidate: dict, baseline: dict) -> dict:
    return {"weighted_interval_wall_tps_ratio": ratio(candidate["weighted_interval_wall_tps"], baseline["weighted_interval_wall_tps"]),
            "visible_tokens_per_total_request_wall_s_ratio": ratio(candidate["visible_tokens_per_total_request_wall_s"], baseline["visible_tokens_per_total_request_wall_s"]),
            "raw_steps_per_device_s_ratio": ratio(candidate["raw_steps_per_device_s"], baseline["raw_steps_per_device_s"]),
            "decode_cycles_ratio": ratio(candidate["summed_decode_cycles"], baseline["summed_decode_cycles"]),
            "request_wall_s_ratio": ratio(candidate["summed_full_request_wall_s"], baseline["summed_full_request_wall_s"]),
            "target_rows_ratio": ratio(candidate["target_rows"], baseline["target_rows"])}


def build(doc: dict, input_sha256: str) -> dict:
    require(doc.get("complete") is True, "input receipt is incomplete")
    require(doc.get("plan_sha256") == PLAN_SHA256, "unrecognized frozen held-out plan")
    cap = integer(doc.get("cap"), "receipt.cap", 2)
    budget = integer(doc.get("output_budget"), "receipt.output_budget", 1)
    require(cap == 512 and budget == 64, "unexpected qualified image capacity or budget")
    prompts = doc.get("prompts")
    require(isinstance(prompts, list) and len(prompts) == 11, "expected all 11 frozen prompts")
    prompt_by_id = {}
    for prompt in prompts:
        require(isinstance(prompt, dict) and isinstance(prompt.get("id"), str)
                and prompt["id"] not in prompt_by_id, "duplicate or invalid prompt")
        ids(prompt.get("prompt_ids"), f"prompt {prompt['id']} IDs")
        prompt_by_id[prompt["id"]] = prompt
    runs = doc.get("runs")
    require(isinstance(runs, list) and len(runs) == 44, "expected all 44 physical requests")
    by_key = {}
    for row in runs:
        require(isinstance(row, dict), "invalid run")
        key = (row.get("pass_name"), row.get("prompt_id"), row.get("mode"))
        require(key[0] in PASSES and key[1] in prompt_by_id and key[2] in MODES
                and key not in by_key, f"missing/duplicate A/B key {key}")
        require(row.get("prompt_ids") == prompt_by_id[key[1]]["prompt_ids"],
                f"{key}: prompt IDs differ from frozen receipt")
        require(row.get("effective_output_budget") == budget,
                f"{key}: effective output budget differs from frozen 64")
        require(row.get("stop_ids") == doc.get("actual_stop_ids"),
                f"{key}: stop IDs differ from receipt's actual model stop IDs")
        by_key[key] = row
    block = integer(doc.get("attention_block", 256), "receipt.attention_block", 2)
    report_prompts = []
    for prompt in prompts:
        pid = prompt["id"]
        passes = {}
        for pass_name in PASSES:
            base, cand = (by_key[(pass_name, pid, mode)] for mode in MODES)
            require(base["raw_token_ids"] == cand["raw_token_ids"],
                    f"{pass_name}/{pid}: candidate raw IDs differ from ordinary generation")
            require(base.get("stop_ids") == cand.get("stop_ids"),
                    f"{pass_name}/{pid}: stop IDs differ")
            b, c = measured(base), measured(cand)
            require(b["target_rows"] == b["raw_emitted"], f"{pass_name}/{pid}: baseline rows differ")
            observed = {name: integer(cand.get(name), f"{pass_name}/{pid}.{name}")
                        for name in ("iterations", "accepted", "verifies", "rejected", "target_rows")}
            replayed = replay(prompt["prompt_ids"], cand["raw_token_ids"],
                              set(ids(cand["stop_ids"], "stop IDs")), budget, cap, block)
            for name, value in observed.items():
                require(replayed[name] == value,
                        f"{pass_name}/{pid}: causal replay {name}={replayed[name]}, hardware={value}")
            require(c["committed_input_rows"] == replayed["committed_input_rows"],
                    f"{pass_name}/{pid}: committed rows differ")
            passes[pass_name] = {"baseline": b, "candidate": c,
                                 "paired": {
                                     "weighted_interval_wall_tps_ratio": ratio(c["weighted_interval_wall_tps"], b["weighted_interval_wall_tps"]),
                                     "visible_tokens_per_request_wall_s_ratio": ratio(c["visible_tokens_per_request_wall_s"], b["visible_tokens_per_request_wall_s"]),
                                     "raw_device_steps_per_s_ratio": ratio(c["raw_device_steps_per_s"], b["raw_device_steps_per_s"]),
                                     "decode_cycles_ratio": ratio(c["decode_cycles"], b["decode_cycles"]),
                                     "request_wall_s_ratio": ratio(c["request_wall_s"], b["request_wall_s"]),
                                     "target_rows_ratio": ratio(c["target_rows"], b["target_rows"])},
                                 "causal_replay": replayed}
        require(by_key[("first_pass", pid, "baseline")]["raw_token_ids"] ==
                by_key[("warm_pass", pid, "baseline")]["raw_token_ids"],
                f"{pid}: raw IDs changed between passes")
        report_prompts.append({"id": pid, "category": prompt.get("category"),
                               "prompt_tokens": len(prompt["prompt_ids"]), "passes": passes})
    aggregates = {}
    for pass_name in PASSES:
        pair = {mode: aggregate([measured(by_key[(pass_name, p["id"], mode)]) for p in prompts])
                for mode in MODES}
        aggregates[pass_name] = {**pair, "paired": compare(pair["candidate"], pair["baseline"]),
                                 "causal_replay": {
                                     name: sum(p["passes"][pass_name]["causal_replay"][name]
                                               for p in report_prompts)
                                     for name in ("iterations", "accepted", "verifies", "rejected",
                                                  "target_rows", "raw_emitted", "committed_input_rows",
                                                  "otherwise_eligible_stop_proposal_vetoes",
                                                  "lookup_calls_including_terminal")}}
    all_pair = {mode: aggregate([measured(by_key[(pass_name, p["id"], mode)])
                                 for pass_name in PASSES for p in prompts]) for mode in MODES}
    aggregates["both_passes_descriptive_only"] = {
        **all_pair, "paired": compare(all_pair["candidate"], all_pair["baseline"]),
        "note": "22 requests repeat the same 11 contexts; warm pass is the primary comparison."}
    return {"scope": "Physical F2 in-memory callback A/B; all 11 frozen contexts and both passes",
            "input_sha256": input_sha256, "source_commit": doc.get("source_commit"),
            "source_sha256": doc.get("source_sha256"),
            "decoder_sha256": doc.get("decoder_sha256"),
            "plan_sha256": doc["plan_sha256"], "config_sha256": doc.get("config_sha256"),
            "tokenizer_sha256": doc.get("tokenizer_sha256"),
            "board_info": doc.get("board_info"), "core_khz": doc.get("expected_core_khz"),
            "attention_block": block, "cap": cap, "output_budget": budget,
            "mode_order": doc.get("mode_order"),
            "first_pass_note": "Only the first prompt of each fresh Engine is engine-cold; the other first-pass requests share that Engine.",
            "order_limitation": "Baseline mode ran before candidate; small wall-only differences may reflect host or thermal drift.",
            "metric_note": "Weighted interval wall TPS is summed visible intervals / summed first-to-last visible time; no average of per-prompt TPS. Raw device steps/s uses total decode cycles. These are callback, not terminal/SSE rates.",
            "lookup_cost_note": "All lookup/chain/drain cost is inside candidate cycles; isolated lookup cycles were not measured. Replay is derived from saved target IDs, not an FPGA proposal trace.",
            "hardware_equivalence": "Every paired raw ID sequence and every candidate replay counter matched or report generation failed.",
            "prompts": report_prompts, "aggregates": aggregates,
            "generalization": "This held-out set is 11 fixed contexts, each measured twice; no wider workload or throughput generalization is inferred."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="completed gemma-heldout.json")
    parser.add_argument("--out", type=Path, required=True, help="new report.json")
    args = parser.parse_args()
    require(not args.out.exists(), "refusing to overwrite an existing report")
    source = args.input.read_bytes()
    report = build(json.loads(source), hashlib.sha256(source).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
