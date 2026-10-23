"""Frozen correctness and timing evaluator. Nothing here is candidate-controlled."""
from __future__ import annotations

import math
import random
import statistics as st

from .common import digest, finite, require

METRICS = ("wall_decode_tps", "ttft_s", "prefill_tps")


def workload_key(w):
    return digest(w)


def validate_trace(t, w, seed, heldout, *, fixture=False):
    require(t.get("schema") == 1, "unsupported trace schema")
    require(t.get("kind") == ("TEST_FIXTURE" if fixture else "PHYSICAL_FPGA"), "wrong evidence stage")
    require(t.get("workload_sha256") == workload_key(w), "task changed")
    require(t.get("identity") == seed["identity"], "checkpoint/image/precision/host fingerprint changed")
    require(t.get("backend") == "board" and t.get("cpu_fallback") is False, "CPU fallback or non-board backend")
    require(t.get("complete") is True and t.get("finish") == "fixed_length", "incomplete or early EOS")
    require(t.get("state") == w["state"], "warm/cold state changed")
    require(t.get("prompt_tokens") == len(w["prompt_ids"]), "prompt token count changed")
    require(t.get("initial_context_tokens") == w["initial_context_tokens"], "context changed")
    require(t.get("loop_count") == w["loop_count"], "loop count changed")
    ids = t.get("accepted_token_ids")
    require(type(ids) is list and all(type(x) is int for x in ids), "accepted token IDs missing")
    require(len(ids) == w["output_tokens"], "fewer/more accepted output tokens")
    require(ids == w["expected_token_ids"], "frozen greedy reference mismatch")
    require(t.get("correctness_suite_sha256") == seed["correctness_suite_sha256"], "wrong held-out suite")
    require(t.get("heldout_pass") is True and t.get("fpga_stage_pass") is True, "correctness or FPGA gate failed")
    got = t.get("heldout_results", [])
    require([x.get("id") for x in got] == [x["id"] for x in heldout["cases"]], "missing held-out cases")
    for actual, expected in zip(got, heldout["cases"]):
        require(actual.get("accepted_token_ids") == expected["expected_token_ids"], "held-out token mismatch")
        require(actual.get("stage") == t["kind"] and actual.get("cpu_fallback") is False,
                "held-out case lacks FPGA-stage provenance")
    ev = t.get("events", {})
    request = finite(ev.get("request_start_s"), "request_start")
    end = finite(ev.get("request_done_s"), "request_done")
    accepted = ev.get("accepted_s", [])
    require(len(accepted) == len(ids) and len(ids) >= 2, "per-token accept timestamps missing")
    for x in accepted:
        finite(x, "accept timestamp")
    require(request <= accepted[0] and accepted[-1] <= end, "invalid request timing order")
    require(all(b >= a for a, b in zip(accepted, accepted[1:])), "non-monotone accept events")
    decode = finite(accepted[-1] - accepted[0], "decode interval", True)
    p0 = finite(ev.get("prefill_start_s"), "prefill_start")
    p1 = finite(ev.get("prefill_done_s"), "prefill_done")
    require(request <= p0 < p1 <= accepted[0], "invalid prefill boundary")
    require(t.get("timed_prefill_tokens") == w["timed_prefill_tokens"], "prefill numerator changed")
    require(0 < w["timed_prefill_tokens"] <= len(w["prompt_ids"]), "invalid prefill count")
    require(t.get("timing_boundary") == w["timing_boundary"], "wall timing boundary changed")
    require(t.get("cache_policy") == w["cache_policy"], "cache policy changed")
    device = t.get("device", {})
    cycles = finite(device.get("decode_cycles"), "device cycles", True)
    require(device.get("decoded_tokens") == len(ids)-1, "device counter covers different tokens")
    require(device.get("clock_mhz") == seed["identity"]["clock_mhz"], "clock mismatch")
    device_tps = (len(ids)-1) * device["clock_mhz"] * 1e6 / cycles
    if w.get("speculation", False):
        s = t.get("speculation", {})
        require(seed.get("speculation_qualified") is True, "combined speculation has no independent qualification")
        require(s.get("verification") == "exact_greedy", "unqualified speculative verification")
        require(s.get("rollback_pass") is True, "speculation rollback not qualified")
        expected_rollbacks=heldout.get("rollback_cases",[])
        actual_rollbacks=s.get("rollback_results",[])
        require(expected_rollbacks and len(actual_rollbacks)==len(expected_rollbacks), "frozen rollback cases missing")
        for a,b in zip(actual_rollbacks,expected_rollbacks):
            require(a.get("id")==b["id"] and a.get("committed_state_sha256")==b["committed_state_sha256"]
                    and a.get("continuation_token_ids")==b["continuation_token_ids"], "rollback state or continuation mismatch")
        require(s.get("reference_token_ids") == ids, "speculation changed target output")
        require(s.get("accepted_output_tokens") == len(ids), "draft proposals counted as output")
        for key in ("draft_s", "verify_s", "accept_s", "rollback_s"):
            finite(s.get(key), "speculation " + key)
        require(sum(s[k] for k in ("draft_s", "verify_s", "accept_s", "rollback_s")) <= end-request,
                "speculation overhead outside request wall interval")
        require(type(s.get("proposed_draft_tokens")) is int and type(s.get("accepted_draft_tokens")) is int
                and 0 <= s["accepted_draft_tokens"] <= s["proposed_draft_tokens"], "bad acceptance accounting")
    else:
        require(not t.get("speculation"), "unexpected speculation")
    return {"wall_decode_tps": (len(ids)-1)/decode,
            "ttft_s": finite(accepted[0]-request, "TTFT", True),
            "prefill_tps": w["timed_prefill_tokens"]/(p1-p0),
            "device_decode_tps": device_tps, "request_wall_s": end-request,
            "request_output_tps": len(ids)/(end-request), "accepted_output_tokens": len(ids)}


def interval(values, *, alpha, rng, samples=6000):
    """Paired log-ratio bootstrap, with dispersion floor to avoid false precision."""
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n))/n for _ in range(samples))
    # At least a normal-theory SE interval as protection against bootstrap collapse.
    z = st.NormalDist().inv_cdf(1-alpha/2)
    mean = st.mean(values)
    se = st.stdev(values)/math.sqrt(n)
    lo = min(means[int(samples*alpha/2)], mean-z*se)
    hi = max(means[min(samples-1, int(samples*(1-alpha/2)))], mean+z*se)
    return {"ratio": math.exp(mean), "low": math.exp(lo), "high": math.exp(hi)}


def decision(pairs, policy, look):
    """No mixed score. Decode must improve; BOTH auxiliary non-regression gates must pass."""
    require(1 <= look <= policy["max_candidates"], "look exceeds frozen experiment budget")
    require(len(pairs) >= policy["min_pairs"], "too few matched pairs")
    require(policy["min_pairs"] >= 6, "insufficient minimum repeats")
    by_case = {}
    for pair in pairs:
        by_case.setdefault(pair["case_id"], []).append(pair)
    rng = random.Random(policy["random_seed"] + look)
    # Bonferroni across predeclared search budget, cases and the three metrics; no optional stopping.
    alpha = policy["family_alpha"] / (policy["max_candidates"] * len(by_case) * len(METRICS))
    summary, failures, frontier_ok = {}, [], True
    for case, rows in sorted(by_case.items()):
        require(len(rows) == policy["min_pairs"], "exact fixed repeat budget required per case")
        stats = {}
        for metric in METRICS:
            av = [finite(p["a"][metric], metric, True) for p in rows]
            bv = [finite(p["b"][metric], metric, True) for p in rows]
            ratio = interval([math.log(b/a) for a,b in zip(av,bv)], alpha=alpha, rng=rng)
            ratio.update(a_median=st.median(av), b_median=st.median(bv),
                         a_range=[min(av),max(av)], b_range=[min(bv),max(bv)])
            stats[metric] = ratio
        # Detect sequential A drift, as well as excessive baseline variation.
        base = [p["a"]["wall_decode_tps"] for p in rows]
        cv = st.stdev(base)/st.mean(base)
        drift = abs(st.median(base[:len(base)//2])/st.median(base[len(base)//2:])-1)
        if max(cv,drift) > policy["max_baseline_noise"]:
            failures.append(case + ": baseline noise/drift; reject and rebaseline in a NEW experiment")
            frontier_ok = False
        decode, ttft, prefill = (stats[m] for m in METRICS)
        if decode["low"] <= 1+policy["min_decode_gain"]:
            failures.append(case + ": decode gain not established")
        if ttft["high"] > 1+policy["max_ttft_regression"]:
            failures.append(case + ": TTFT non-regression not established")
        if prefill["low"] < 1-policy["max_prefill_regression"]:
            failures.append(case + ": prefill non-regression not established")
        summary[case] = {"metrics": stats, "baseline_cv": cv, "baseline_drift": drift,
                         "pairs": len(rows), "per_comparison_alpha": alpha}
    return {"promote": not failures, "reasons": failures, "cases": summary,
            "archive_eligible": frontier_ok, "statistic": "paired geometric ratio with simultaneous conservative intervals"}


def dominates(a, b):
    """Conservative Pareto dominance over EVERY workload/state, using interval endpoints."""
    if set(a["cases"]) != set(b["cases"]):
        return False
    strictly = False
    for c in a["cases"]:
        for metric in METRICS:
            x, y = a["cases"][c]["metrics"][metric], b["cases"][c]["metrics"][metric]
            if metric == "ttft_s":
                if x["high"] > y["low"]: return False
                strictly |= x["high"] < y["low"]
            else:
                if x["low"] < y["high"]: return False
                strictly |= x["low"] > y["high"]
    return strictly


def pareto(entries):
    return [x for x in entries if not any(y["id"] != x["id"] and dominates(y["result"], x["result"])
                                         for y in entries)]
