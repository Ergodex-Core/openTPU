# Measurement contract

This harness uses a fixed experiment for each image/model/precision/checkpoint/loop-count/host combination. It never ranks different models, image clocks, precision, context sizes, batch sizes, warm states or hybrid execution scopes as the same task. Each workload/state is a separate constraint; a fast short prompt cannot hide a slow long prompt.

The initial objective is single-request greedy inference through a fixed caller boundary. The eventual physical timing bridge must identify that boundary precisely (for example, caller of the existing in-process scheduler or streamed HTTP client); results from different boundaries cannot mix. There is no bridge deployed by this task and therefore no new physical metric.

| Metric | Numerator | Interval |
|---|---|---|
| **Primary: wall decode TPS** | Accepted output tokens after the first, `N - 1` | First accepted token delivered to caller → final accepted token delivered to caller. Includes scheduling, transfers, synchronization, sampling/verification, detokenization and delivery overhead within that boundary. |
| TTFT, minimize | None | Request submitted by caller → first accepted output token delivered to caller. Includes queueing, preparation/compilation attributable to the declared state, prefill and first-token selection. |
| Prefill TPS, maximize | Exact **timed prefill** token count | Observed prefill start → completion of those prompt tokens, with host dispatch overhead. Never infer this from TTFT. |
| Device decode TPS, report separately | Same `N - 1` accepted output tokens | Sum of device cycles for those tokens divided by measured core clock. Do not sum overlapping slices as elapsed time. |
| Request throughput, report separately | All `N` accepted output tokens | Request submission → full response completion, including TTFT and final delivery. |

All timestamps use one monotonic clock within each trace. Non-monotonic, missing, negative, zero-duration or non-finite data fails evaluation. Token IDs and final output count must exactly equal the frozen expected output. Empty text deltas do not change token counts; multiple tokens in one delivery may have equal timestamps. Finish reason must be the fixed length, with EOS disabled by the evaluator. An early EOS cannot earn a higher rate.

`opentpu/serve/scheduler.py::Worker._prefill` on the inspected wide-port tree can feed **P−1** prompt tokens, leaving the last prompt token to the card decode loop. A bridge must either instrument the completion of that last prompt token or record P−1 as `timed_prefill_tokens`. Dividing P by the P−1 interval is refused. Historical `tools/f2_bench.py` calls P/TTFT “prefill_tok_s”; this is a **proxy** and cannot enter this objective. Its historical EOS-limited output lengths also fail the new fixed-length comparison.

Freeze before search: prompt and expected output IDs; tokenizer/model weight/config hashes; initial context contents/hash and capacity; generated length; loop count; precision of weights, head, activations and any repacked tensors; sampling; batch/concurrency; image AFI/AGFI + package/DCP/manifest hash; clock; runtime tree/commit and dependency versions; hardware host identity; caller boundary; cache policy; warmup/repetitions; allowed settings; trial/search budgets; statistical policy; correctness suite and evaluator hashes. Unknown fingerprints block physical admission.

Warm state means weights resident, declared program caches initialized identically for each arm, empty prompt KV under this protocol, and one or more explicit unscored warmup requests with different token IDs. Cold process means a fresh **owned** runtime process for each measured request, with the card image unchanged; cache files and the exact inclusion of weight loading/initialization in TTFT must be declared. Do not silently call a freshly loaded process “warm” because filesystem pages were cached. Pair arms use the same declared reset policy and warmup count; raw warmup receipts must be retained by the bridge. The synthetic worker only models these states for software tests.

## Decision rule

Each candidate is compared with the **fixed original baseline**, eight matched pairs per workload/state by default, serially ordered AB/BA. Every arm runs the frozen correctness suite and fixed task. No early success stopping. Baseline decode coefficient of variation or first-half/second-half median drift above 5% rejects the candidate; a new experiment must explicitly rebaseline.

The evaluator computes paired log ratios, bootstrap intervals widened by a normal-theory standard-error interval, with Bonferroni allocation across the predeclared candidate budget, workloads and three metrics. These are conservative empirical screening intervals, not a claim of universal statistical coverage under arbitrary autocorrelation. Repeated search remains vulnerable to unmodelled drift: fresh held-out confirmation in another exclusive window is required before operational deployment.

Promotion requires the lower decode ratio bound to exceed 1.02 **for every workload**, upper TTFT ratio bound ≤1.0, and lower prefill ratio bound ≥1.0. Default auxiliary regression tolerances are zero; any explicit tolerance change creates a new frozen experiment. A later candidate cannot replace the incumbent unless it conservatively dominates the incumbent. No weighted sum hides a latency regression.

The Pareto archive retains correctness-passing, noise-qualified points across all three metrics and all workloads, using interval dominance. A tradeoff may stay in the archive but **cannot become the champion** if it fails a promotion gate. Rejected/failed evidence remains in the journal. All points are baseline-normalized within one frozen experiment, never across seeds.

## Speculation

Initial live Gemma speculation is disabled. The schema supports future **exact greedy** traces only after combined target/drafter physical qualification. It requires full accepted output IDs equal to the frozen target reference, accepted/proposed draft counts, target-output counts, rollback verification, and draft/verify/accept/rollback time within the measured request interval. Proposed tokens, rejected branches and extra computed rows never enter the TPS numerator. Record both first-token and steady-decode overhead.

Distribution-preserving sampled verification is **not implemented**, so sampled proposals fail closed. Nanbeige's existing loop2 hybrid measurements establish equivalence to its repacked target; they do not establish original-checkpoint fidelity. Their CPU fusion/Markov operations are part of a separately named hybrid task, never a fallback quietly accepted into the Gemma board-only track.
