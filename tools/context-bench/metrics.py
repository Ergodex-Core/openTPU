#!/usr/bin/env python3
"""Analyze raw, owner-recorded FPGA receipts. No hardware, network or model execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

IMAGE = "agfi-0001de1dd3df8e21e"
PROFILE = "f2-hbm:p4rl8a64c150"


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None


def stream_metrics(stream):
    """Times must share THIS observer's monotonic clock; never mix client/worker clocks."""
    submit = stream["submit_ns"]
    tokens = stream["tokens"]
    assert [x["index"] for x in tokens] == list(range(len(tokens)))
    if not tokens:
        return {"ttft_s": None, "generated_tokens": 0, "decode_intervals": 0,
                "decode_seconds_first_to_last": 0, "decode_tps": None,
                "last_32_observed_tokens_tps": None, "observed_interval_median_s": None,
                "observed_interval_p95_s": None, "coalesced_observations": False,
                "interval_scope": "No visible output token; EOS timing remains in raw trace"}
    times = [x["observed_ns"] for x in tokens]
    assert submit <= times[0] and times == sorted(times)
    n = len(times)
    intervals = [(b - a) / 1e9 for a, b in zip(times, times[1:])]
    seconds = (times[-1] - times[0]) / 1e9
    tail = times[-32:]
    tail_s = (tail[-1] - tail[0]) / 1e9
    return {"ttft_s": (times[0] - submit) / 1e9,
            "generated_tokens": n, "decode_intervals": max(0, n - 1),
            "decode_seconds_first_to_last": seconds,
            "decode_tps": (n - 1) / seconds if n > 1 and seconds > 0 else None,
            "last_32_observed_tokens_tps": (len(tail) - 1) / tail_s if tail_s > 0 else None,
            "observed_interval_median_s": statistics.median(intervals) if intervals else None,
            "observed_interval_p95_s": p95(intervals),
            "coalesced_observations": any(x == 0 for x in intervals),
            "interval_scope": "host readback/arrival; NOT device per-token latency"}


def validate(receipt, case):
    ident = receipt["identity"]
    assert receipt["evidence_kind"] == "physical_fpga_measurement"
    assert ident["image"] == IMAGE and ident["profile"] == PROFILE
    assert ident["build_id"] == 0x417243F1 and ident["clock_mhz"] == 150
    assert ident["weights_format"] == ident["head_format"] == "int8"
    assert ident["greedy"] is True and ident["batch"] == 1
    assert ident["model"] == case["model"]
    assert ident["runtime_source_digest"]
    assert ident["normal_locks"] == ["f2:0", "f2-0000:34:00.0"]
    if "checkpoint_sha256" in case:
        assert ident["checkpoint_sha256"] == case["checkpoint_sha256"]
        assert ident["config_sha256"] == case["config_sha256"]
        assert ident["runtime_source_digest"] == case["runtime_source_digest"]
    assert receipt["case_id"] == case["case_id"]
    assert receipt["prompt_tokens"] == case["prompt_tokens"]
    assert receipt["input_ids_sha256"] == case["input_ids_sha256"]
    assert receipt["requested_output_tokens"] == 64
    assert receipt["capacity"] >= receipt["prompt_tokens"] + 64
    assert receipt["truncated"] is False and receipt["prefix_kv_reused"] == 0
    assert receipt["residency"] in ("warm_resident", "cold_cached", "cold_uncached")
    for key in ("resident", "device_inputs", "can_generate", "chains", "stateless_kv"):
        assert receipt["contract"][key] is True
    for key in ("host_service", "host_embeddings", "host_ple"):
        assert receipt["contract"][key] is False
    assert receipt["ownership_window_id"] and receipt["raw_phase_trace"]
    assert all(t["selection"] == "device" for t in receipt["worker"]["tokens"])
    assert len(receipt["worker"]["tokens"]) <= 64
    return True


def analyze(receipt, case):
    validate(receipt, case)
    worker = stream_metrics(receipt["worker"])
    ids = [x["token_id"] for x in receipt["worker"]["tokens"]]
    hz = receipt["identity"]["clock_mhz"] * 1e6
    out = {"case_id": case["case_id"], "residency": receipt["residency"],
           "worker": worker, "input_tokens": case["prompt_tokens"],
           "prefill_tokens_excluding_held_final": receipt["prefill_tokens"],
           "prefill_device_s": receipt["prefill_cycles"] / hz,
           "prefill_host_s": receipt["prefill_host_s"],
           "generate_device_s_including_first_pick": receipt["generate_cycles"] / hz,
           "generate_steps_including_first_pick": receipt["generate_steps"],
           "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
           "quality": receipt["quality"], "end": receipt["end"],
           "complete_64_window": len(ids) == 64,
           "timing_note": "Device generate cycles include final prompt token + first selection; "
                          "they are not a decode-only denominator."}
    if receipt.get("client"):
        out["client"] = stream_metrics(receipt["client"])
        out["client"]["scope"] = receipt["client"]["scope"]
    return out


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["case_id"], row["residency"]), []).append(row)
    result = []
    for (case, residency), items in groups.items():
        obj = {"case_id": case, "residency": residency, "repetitions": len(items),
               "exact_greedy_repeat": len({x["token_ids_sha256"] for x in items}) == 1,
               "all_complete_64": all(x["complete_64_window"] for x in items),
               "p95_method": "nearest rank; small sample p95 is descriptive"}
        for observer in ("worker", "client"):
            for metric in ("ttft_s", "decode_tps", "last_32_observed_tokens_tps"):
                vals = [x[observer][metric] for x in items
                        if observer in x and x[observer].get(metric) is not None]
                if vals:
                    obj[f"{observer}_{metric}"] = {"median": statistics.median(vals), "p95": p95(vals)}
        result.append(obj)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--receipts", type=Path, required=True, help="Owner-produced JSONL")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    preparation = json.loads(a.preparation.read_text())
    source_digest = hashlib.sha256(json.dumps(preparation["sources"], sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    cases = {case["case_id"]: {**case,
                "checkpoint_sha256": m["files"]["model.safetensors"]["sha256"],
                "config_sha256": m["files"]["config.json"]["sha256"],
                "runtime_source_digest": source_digest}
             for m in preparation["models"].values() for case in m["cases"]}
    receipts = [json.loads(line) for line in a.receipts.read_text().splitlines() if line.strip()]
    rows = [analyze(r, cases[r["case_id"]]) for r in receipts]
    a.output.write_text(json.dumps({"runs": rows, "summary": summarize(rows)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
