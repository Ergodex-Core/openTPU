from __future__ import annotations

import datetime as dt
import json
import platform
import re
import sys
import time
from pathlib import Path

from .adapters import FixtureAdapter, LiveAdapter
from .common import GateError, Journal, Queued, RunLock, Uncertain, digest, file_hash, immutable_json, read_json, require
from .evaluate import decision, pareto, validate_trace, workload_key
from .propose import BASELINE, candidate_id, propose, validate

ROOT = Path(__file__).resolve().parents[1]


def validate_config(c):
    require(c.get("schema") == 1, "unsupported experiment schema")
    require(c.get("mode") in ("fixture", "live"), "unknown mode")
    p = c["policy"]
    require(type(p["max_candidates"]) is int and 1 <= p["max_candidates"] <= 12, "candidate bound must be 1..12")
    require(type(p["min_pairs"]) is int and 6 <= p["min_pairs"] <= 20, "pair bound must be 6..20")
    require(0 < p["family_alpha"] <= .05, "invalid error budget")
    require(0 <= p["min_decode_gain"] <= .5, "invalid decode gain")
    require(0 <= p["max_ttft_regression"] <= .05 and 0 <= p["max_prefill_regression"] <= .05,
            "non-regression tolerance must be explicit and <=5%")
    require(0 < p["max_baseline_noise"] <= .1, "baseline noise budget invalid")
    require(0 < c["trial_timeout_s"] <= 900 and 0 < c["run_budget_s"] <= 7200, "timeout budget invalid")
    require(1 <= len(c["workloads"]) <= 16, "workload count invalid")
    require(len({w["id"] for w in c["workloads"]}) == len(c["workloads"]), "duplicate workloads")
    require(c["seed"]["correctness_suite_sha256"] == digest(c["heldout"]), "held-out suite hash mismatch")
    require(c["heldout"]["cases"], "empty held-out suite")
    for w in c["workloads"]:
        require(re.fullmatch(r"[a-z0-9_\-]{1,64}",w["id"]) is not None, "unsafe workload ID")
        require(type(w["prompt_ids"]) is list and len(w["prompt_ids"]) > 0 and
                all(type(x) is int and x >= 0 for x in w["prompt_ids"]), "invalid prompt token IDs")
        require(2 <= w["output_tokens"] <= 1024, "output length invalid")
        require(w["state"] in ("warm_resident", "cold_process"), "state must be explicit")
        require(w["output_tokens"] == len(w["expected_token_ids"]), "missing fixed-output reference")
        require(len(w["prompt_ids"])+w["initial_context_tokens"]+w["output_tokens"] <= w["context_capacity"],
                "workload exceeds fixed context capacity")
        require(w["loop_count"] == c["seed"]["identity"]["loop_count"], "loop objective changed")
        require(w["sampling"] == "greedy", "sampled speculation requires separate qualified verifier; disabled")
        require(w["warmup_requests"] >= 1, "explicit warmup policy required")
    validate(c["baseline_settings"])


def source_pins():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT/"evolve").glob("*.py"))}


def initialize(config_path, run):
    config = read_json(config_path)
    validate_config(config)
    run = Path(run)
    require(not (run/"manifest.json").exists(), "run already initialized; resume it")
    with RunLock(run):
        pins = source_pins()
        manifest = {"schema": 1, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "config": config, "config_sha256": digest(config), "evaluator_files": pins,
                    "python": sys.version, "platform": platform.platform(),
                    "architecture_reference": "https://github.com/algorithmicsuperintelligence/openevolve",
                    "kind": "TEST_FIXTURE" if config["mode"] == "fixture" else "LIVE_REQUEST_NOT_EXECUTED"}
        immutable_json(run/"manifest.json", manifest)
        # Self-contained replay: freeze actual evaluator/generator sources, not just their names.
        for rel in pins:
            dst = run/"frozen"/rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("xb") as out:
                out.write((ROOT/rel).read_bytes())
            dst.chmod(0o444)
        Journal(run/"journal").append("initialized", {"manifest_sha256": digest(manifest)})
    return {"status": "initialized", "run": str(run), "kind": manifest["kind"]}


def load_run(run):
    run = Path(run)
    m = read_json(run/"manifest.json")
    validate_config(m["config"])
    require(digest(m["config"]) == m["config_sha256"], "frozen config changed")
    require(source_pins() == m["evaluator_files"], "evaluator changed; create a NEW experiment")
    for rel, h in m["evaluator_files"].items():
        require(file_hash(run/"frozen"/rel) == h, "frozen evaluator changed")
    journal = Journal(run/"journal")
    require(journal.events and journal.events[0]["data"]["manifest_sha256"] == digest(m), "manifest changed")
    for event in journal.events:
        if event["kind"] == "trial_done":
            d = event["data"]
            require(file_hash(run/d["path"]) == d["file_sha256"], "immutable trial evidence changed")
        if event["kind"] in ("trial_done","trial_failed","trial_queued"):
            d=event["data"]
            if d.get("transport"):
                require(file_hash(run/d["transport"])==d["transport_sha256"], "transport evidence changed")
    return m, journal


def recover(j):
    inflight, receipts, candidates, outcomes, rejected = {}, {}, [], [], set()
    for e in j.events:
        d = e["data"]
        if e["kind"] == "candidate": candidates.append(d)
        if e["kind"] == "trial_started": inflight[d["key"]] = d
        if e["kind"] in ("trial_done", "trial_failed", "trial_queued", "trial_resolved"):
            inflight.pop(d["key"], None)
        if e["kind"] == "trial_done": receipts[d["key"]] = d
        if e["kind"] in ("candidate_done", "candidate_rejected"):
            outcomes.append(d)
        if e["kind"] == "trial_resolved": rejected.add(d["candidate_id"])
    return inflight, receipts, candidates, outcomes, rejected


def status(run):
    m,j = load_run(run)
    inflight, receipts, candidates, outcomes, _ = recover(j)
    archive = pareto([x for x in outcomes if x.get("result",{}).get("archive_eligible")])
    champion = candidate_id(m["config"]["baseline_settings"])
    for e in j.events:
        if e["kind"] == "promoted": champion = e["data"]["id"]
        if e["kind"] == "rollback": champion = e["data"]["to"]
    last = j.events[-1]
    return {"kind": m["kind"], "completed_candidates": len(outcomes), "trials": len(receipts),
            "champion": champion, "pareto": [x["id"] for x in archive],
            "uncertain": list(inflight), "last_event": last["kind"], "last_data": last["data"]}


def trial(run, j, config, adapter, candidate, w, pair, arm, timeout):
    key = f"{candidate['id']}-{w['id']}-{pair:02d}-{arm}"
    _, receipts, _, _, _ = recover(j)
    if key in receipts:
        return receipts[key]["metrics"]
    settings = config["baseline_settings"] if arm == "a" else candidate["settings"]
    req = {"schema": 1, "settings": settings, "workload": w, "seed": config["seed"],
           "heldout": config["heldout"], "pair": pair, "arm": arm,
           "workload_sha256": workload_key(w), "trial_id": key,
           "warmup_requests": w["warmup_requests"], "deadline_s": timeout}
    attempts = run/"scratch"/key
    attempts.mkdir(parents=True,exist_ok=True)
    scratch = attempts/f"attempt-{len(list(attempts.iterdir())):04d}"
    # Unique trial workspace; no copied checkout or candidate executable.
    scratch.mkdir(exist_ok=False)
    def transport_evidence():
        path=scratch/"transport.json"
        return {"transport":str(path.relative_to(run)),"transport_sha256":file_hash(path)} if path.exists() else {}
    j.append("trial_started", {"key": key, "candidate_id": candidate["id"], "request_sha256": digest(req),
                               "reserved_timeout_s": timeout})
    try:
        trace = adapter.evaluate(req, scratch, timeout)
        rel = "evidence/" + key + ".json"
        immutable_json(run/rel, trace)
        metrics = validate_trace(trace, w, config["seed"], config["heldout"], fixture=config["mode"]=="fixture")
    except Queued as e:
        j.append("trial_queued", {"key": key, "reason": str(e), **transport_evidence()})
        raise
    except Uncertain:
        # Started but nonterminal: resume is blocked until explicit owner reconciliation.
        raise
    except (GateError, OSError, KeyError, TypeError, ValueError) as e:
        j.append("trial_failed", {"key": key, "reason": str(e), **transport_evidence()})
        raise GateError(str(e)) from e
    load_run(run)  # detect any mutation of the evaluator/manifest during candidate evaluation
    j.append("trial_done", {"key": key, "metrics": metrics, "path": rel,
                            "file_sha256": file_hash(run/rel), **transport_evidence()})
    return metrics


def run_loop(run, stop_after=None):
    run = Path(run)
    with RunLock(run):
        m, j = load_run(run)
        c, p = m["config"], m["config"]["policy"]
        inflight, _, candidates, outcomes, rejected = recover(j)
        if inflight:
            raise Uncertain("started trial has no terminal receipt: " + ", ".join(inflight))
        if c["mode"] == "live":
            adapter = LiveAdapter(c, c["seed"])
            try:
                adapter.preflight(c.get("ownership_window", {}))
            except (Queued, GateError) as e:
                j.append("queued", {"target": "opentpu-f2", "action": "bounded matched baseline/evaluation",
                                    "reason": str(e), "hardware_attempted": False})
                return status(run)
        else:
            adapter = FixtureAdapter()
        limit = min(p["max_candidates"], stop_after if stop_after is not None else p["max_candidates"])
        require(0 <= limit <= p["max_candidates"], "invalid stop-after")
        started = time.monotonic()
        # Budget is cumulative across resumes, so restarting does not buy extra trials/time.
        used = sum(e["data"].get("elapsed_s",0) for e in j.events if e["kind"] == "budget_used")
        used += sum(e["data"].get("reserved_timeout_s",0) for e in j.events if e["kind"] == "trial_resolved")
        try:
            for generation in range(len(outcomes), limit):
                load_run(run)
                if used+time.monotonic()-started >= c["run_budget_s"]:
                    j.append("budget_exhausted", {"used_s": used+time.monotonic()-started})
                    break
                if generation < len(candidates):
                    cand = candidates[generation]
                else:
                    archive = pareto([o for o in outcomes if o.get("result",{}).get("archive_eligible")])
                    parent = archive[generation % len(archive)]["settings"] if archive else c["baseline_settings"]
                    cand = propose(parent, [candidate_id(c["baseline_settings"])] + [x["id"] for x in candidates], generation)
                    if cand is None: break
                    j.append("candidate", cand)
                    candidates.append(cand)
                pairs = []
                try:
                    require(cand["id"] not in rejected, "owner reconciled an uncertain trial; reject candidate, do not retry")
                    for w in c["workloads"]:
                        for pair in range(p["min_pairs"]):
                            # AB, BA, AB, BA: matched same-state requests; no concurrent card trials.
                            results = {}
                            for arm in (("a","b") if pair%2 == 0 else ("b","a")):
                                remaining = c["run_budget_s"]-used-(time.monotonic()-started)
                                if remaining <= 0: raise GateError("cumulative time budget exhausted")
                                results[arm] = trial(run,j,c,adapter,cand,w,pair,arm,min(c["trial_timeout_s"], remaining))
                            pairs.append({"case_id": w["id"], **results})
                    result = decision(pairs, p, generation+1)
                    outcome = {**cand, "result": result}
                    j.append("candidate_done", outcome)
                    outcomes.append(outcome)
                    if result["promote"]:
                        # Do not replace a faster incumbent with a later merely-above-baseline point.
                        champion = status(run)["champion"]
                        previous = next((o for o in outcomes if o["id"] == champion), None)
                        from .evaluate import dominates
                        if previous is None or dominates(result, previous["result"]):
                            j.append("promoted", {"id": cand["id"], "previous": champion,
                                                  "scope": "harness champion pointer only; no deployed runtime mutation"})
                    frontier = pareto([o for o in outcomes if o["result"].get("archive_eligible")])
                    j.append("checkpoint", {"generation": generation+1, "pareto": [o["id"] for o in frontier],
                                             "champion": status(run)["champion"]})
                except Queued as e:
                    j.append("queued", {"reason": str(e), "hardware_attempted": c["mode"]=="live"})
                    break
                except GateError as e:
                    if isinstance(e, Uncertain): raise
                    outcome = {**cand, "result": {"promote": False, "archive_eligible": False, "reasons": [str(e)]}}
                    j.append("candidate_rejected", outcome)
                    outcomes.append(outcome)
        finally:
            j.append("budget_used", {"elapsed_s": time.monotonic()-started})
        return status(run)


def rollback(run, reason):
    with RunLock(run):
        m,j = load_run(run)
        require(not recover(j)[0], "uncertain hardware action: reconcile first")
        current = status(run)["champion"]
        previous = candidate_id(m["config"]["baseline_settings"])
        for e in reversed(j.events):
            if e["kind"] == "promoted" and e["data"]["id"] == current:
                previous = e["data"]["previous"]
                break
        j.append("rollback", {"from": current, "to": previous, "reason": reason,
                              "scope": "harness pointer; never changes card image or another owner's runtime"})
    return status(run)


def reconcile(run, key, receipt):
    """An operator supplies immutable evidence; a failed candidate is never retried."""
    with RunLock(run):
        m,j = load_run(run)
        pending = recover(j)[0]
        require(key in pending, "no such uncertain trial")
        r = read_json(receipt)
        require(r.get("trial_id") == key and r.get("outcome") == "aborted_idle_runtime_restored",
                "reconciliation must certify this operation aborted, idle and runtime restored")
        require(r.get("owner") and r.get("source_receipt_sha256"), "reconciliation needs owner and evidence")
        rel = "reconciliation/"+key+".json"
        immutable_json(Path(run)/rel, r)
        j.append("trial_resolved", {**pending[key], "receipt": rel, "sha256": file_hash(Path(run)/rel)})
    return status(run)
