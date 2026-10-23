# OpenTPU bounded evolution harness

Local implementation of the [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) propose/evaluate/archive pattern. It selects a parent from a Pareto archive, proposes one bounded runtime-settings change, runs serial matched baseline/candidate trials through a trusted evaluator, gates correctness and provenance, and checkpoints the outcome. It uses Python's standard library and installs nothing. No LLM/provider integration or downloaded OpenEvolve code is executed.

**Current status:** orchestration code, configurations, source snapshots, tests and runbook are prepared in this isolated repository/branch. **Tests and dry run have not executed. No live loop or hardware baseline is running.** The user explicitly kept tests on `opentpu-m8a-build`; current authorized connectivity and exclusive card ownership have not been established, and prior denied SSH/process/localhost checks were not retried. Live measurement still needs a reviewed runtime timing bridge and complete fixed-task references/fingerprints.

No existing checkout, nightly queue, optimization owner's files, portal, image or infrastructure was changed. No scheduler/daemon was enabled. There were no package installs, external agent messages, pushes or publication.

## Files

| Path | Purpose |
|---|---|
| `evolve/propose.py` | Pure allowlisted settings generator; 12 possible settings including baseline, at most 11 novel candidates. No source-code mutation or arbitrary commands. |
| `evolve/evaluate.py` | Frozen correctness, physical-stage and timing gates; independent metrics, matched confidence intervals, promotion and Pareto archive. |
| `evolve/loop.py` | Resumable serial execution, immutable evidence, hash-chained journal, explicit uncertain-operation reconciliation and champion rollback. |
| `evolve/adapters.py` | Bounded fixture subprocess and gated trusted live bridge protocol. No remote-access implementation. |
| `evolve/record.py` | Monotonic request/prefill/accepted-token recorder for trusted runtime instrumentation. |
| `configs/fixture.json` | Clearly labelled synthetic experiment: eight matched pairs for each warm/cold workload. All values are invented test data. |
| `configs/eligibility.json` | Fourteen seed/scope entries with 21 locally verified source snapshots. All live seeds disabled. |
| `configs/live-preparation.json` | Pending hardware integration checklist; deliberately not a runnable experiment. |
| `fixtures/wide-moon-historical.json` | Actual eight-token wide-port reference, with explicit scope limitations. |
| `tests/test_harness.py` | Contract and subprocess integration tests, currently unexecuted. |
| `docs/OBJECTIVES.md` | Timing boundaries, fixed task, warm/cold definitions, noise/promotion and speculation policy. |
| `docs/LIVE_BRIDGE.md` | Required owner integration and shared-card coordination contract. |
| `docs/ELIGIBILITY.md` | Human-readable verified eligibility matrix and limitations. |
| `docs/INSPECTION.md` | Inspected infrastructure, source versions and ownership constraints. |
| `docs/STATUS.md` | Execution ledger and exact remaining blockers. |

## Run the fixture suite on the builder

First establish current authorized access to **opentpu-m8a-build** and verify instance `i-0ec8d33362b3100be` and role `opentpu-f2-ssm`. Use the existing authorized transfer mechanism to place this repository in an isolated directory there. Do not launch instances, install packages, modify credentials/security, or bypass earlier denials. The commands below are prepared commands, not a record of execution.

```sh
python3 -m unittest discover -s tests -v
python3 -m evolve eligibility
python3 -m evolve init --config configs/fixture.json --run runs/fixture-demo
python3 -m evolve run --run runs/fixture-demo --stop-after 2
python3 -m evolve status --run runs/fixture-demo
python3 -m evolve run --run runs/fixture-demo
python3 -m evolve rollback --run runs/fixture-demo --reason 'Demonstrate pointer rollback'
```

The two `run` commands demonstrate pause/resume. `--stop-after` is the **total** completed-candidate limit, not an extra budget on every invocation. Identical completed trials are reused only after all immutable hashes pass. Outcomes print `TEST_FIXTURE`; they are never valid FPGA measurements. Source changes require a **new** run; to resume an old run after editing the evaluator, use the exact saved `frozen/evolve/` version with its original matching configuration.

For one bounded builder invocation with logs, run `python3 scripts/run_builder_checks.py --identity /path/to/current-owner-identity.json` after authorized transfer. The helper refuses a Mac or any local instance ID other than the builder. Its required current identity receipt contains `instance_name`, `instance_id`, `iam_profile`, `access: "authorized"`, and a timezone-qualified `verified_at` within 120 seconds. The existing authorized owner must supply that receipt from current metadata; the helper never connects remotely or obtains credentials.

`runs/<name>/manifest.json` freezes the whole experiment and evaluator hashes. `frozen/evolve/` contains the actual source. `evidence/` contains raw traces. `journal/` contains ordered hash-chained events, decisions, checkpoints, promotion and rollback. Read-only file modes deter accidental edits; the real candidate protection is a pure declarative generator with no file/code/action capability plus integrity verification before/after evaluations. This is not an OS security boundary against an unrelated privileged user.

The generator starts from the fixed baseline's program cache and prompt runs enabled, quantum 32. It explores cache toggles and quantum 16/32/64 only. It cannot change model, checkpoint, precision, context, output count, evaluator, held-out fixture, image, clocks or hardware commands. The initial harness deliberately supports runtime tuning only; software/kernel patches need a separate reviewed sandbox/allowlist and cannot be smuggled through settings.

## Failure, queueing and resuming

Busy/unknown ownership queues without preemption. Any trial with `trial_started` and no terminal receipt blocks resume as **uncertain**. Check the actual owner/runtime evidence before any retry. The reconciliation command accepts an owner receipt asserting this exact trial was aborted, is idle and had its owned runtime settings restored; it rejects that candidate without rerunning it:

```sh
python3 -m evolve reconcile --run runs/EXPERIMENT --trial TRIAL_ID --receipt owner-reconciliation.json
python3 -m evolve run --run runs/EXPERIMENT
```

Required reconciliation fields: `trial_id`, `outcome: "aborted_idle_runtime_restored"`, `owner`, and `source_receipt_sha256`. Keep the actual supporting receipt with the owner. Never invent reconciliation merely to unblock a run. Successful-but-incompletely-recorded hardware work must first be reconciled by its owner; do not infer it failed and blindly repeat it.

Promotion only updates this harness's champion pointer. It does not deploy a runtime or edit any other checkout. `rollback` returns that pointer to its preceding champion and preserves all evidence. The live evaluator must independently restore its own runtime state after **each** arm, including failures, while retaining the canonical card lock until verified idle. No card image replacement/restoration is implemented or authorized by this setup.

## Enable a physical experiment later

Follow [LIVE_BRIDGE.md](docs/LIVE_BRIDGE.md), freeze the complete protocol, and make a new config/experiment with a reviewed bridge hash, verified source registry hash, and current owner window. The registry and configuration both must enable the same physical seed and protocol. Existing historical rates are context only; they cannot seed fresh matched ratios. The pending portal has priority. Only already loaded verified images may be used, and only the existing `opentpu-f2` may execute hardware trials.

This setup is intentionally distinct from `openTPU-hwloop/tools/hwloop/nightly/queue.json`, which tracks new RTL/image build campaigns. It neither adds to nor schedules that queue.
