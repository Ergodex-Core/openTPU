"""Synthetic integration fixture ONLY: no OpenTPU, model, network, or device imports.

Numbers here exercise decisions and must never be presented as hardware performance.
"""
import json
import sys


def execute(request):
    w, s = request["workload"], request["seed"]
    cfg = request["settings"]
    noise = [1.001, .999, 1.002, .998, 1., 1.001, .999, 1.][request["pair"] % 8]
    # Trade-off candidates deliberately exercise rejection and the Pareto archive.
    rate, ttft, prefill = 40*noise, .5*noise, 80/noise
    if cfg["quantum"] == 64:
        rate *= 1.15
        ttft *= .95
        prefill *= 1.05
    if cfg["quantum"] == 16:
        rate *= .97
        ttft *= .90
        prefill *= 1.12
    if not cfg["program_cache"]:
        rate *= .88
        ttft *= 1.2
    if not cfg["prompt_runs"]:
        rate *= 1.3
        ttft *= 1.4
        prefill *= .6
    if w["state"] == "cold_process":
        ttft += 2
    t0 = 10.0
    first = t0+ttft
    stamps = [first + i/rate for i in range(w["output_tokens"])]
    duration = w["timed_prefill_tokens"]/prefill
    return {"schema": 1, "kind": "TEST_FIXTURE", "identity": s["identity"],
            "backend": "board", "cpu_fallback": False, "complete": True, "finish": "fixed_length",
            "workload_sha256": request["workload_sha256"], "state": w["state"],
            "prompt_tokens": len(w["prompt_ids"]), "initial_context_tokens": w["initial_context_tokens"],
            "loop_count": w["loop_count"], "accepted_token_ids": w["expected_token_ids"],
            "correctness_suite_sha256": s["correctness_suite_sha256"], "heldout_pass": True,
            "heldout_results": [{"id": c["id"], "accepted_token_ids": c["expected_token_ids"],
                                 "stage": "TEST_FIXTURE", "cpu_fallback": False}
                                for c in request["heldout"]["cases"]],
            "fpga_stage_pass": True, "timed_prefill_tokens": w["timed_prefill_tokens"],
            "timing_boundary": w["timing_boundary"], "cache_policy": w["cache_policy"],
            "events": {"request_start_s": t0, "request_done_s": stamps[-1]+.003,
                       "accepted_s": stamps, "prefill_start_s": first-duration-.001,
                       "prefill_done_s": first-.001},
            "device": {"decode_cycles": (len(stamps)-1)*s["identity"]["clock_mhz"]*1e6/60,
                       "decoded_tokens": len(stamps)-1, "clock_mhz": s["identity"]["clock_mhz"]}}


if __name__ == "__main__":
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    request = json.loads(sys.stdin.read(1 << 20))
    print(json.dumps(execute(request), allow_nan=False))
