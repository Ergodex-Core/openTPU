# Execution status

| Phase | Actual outcome |
|---|---|
| Inspect existing loops, runtime, memories, receipts | Completed locally; references in INSPECTION.md and eligibility registry. |
| Isolated orchestration code/configuration/tests | Prepared in `codex/opentpu-evolve-harness`; no existing optimization owner modified. |
| Source eligibility review | 14 entries, 21 frozen source snapshots. Five historical Gemma INT8 physical-model candidates; all live protocol entries remain disabled. |
| Unit/integration tests | **20 test methods prepared; not run**. User explicitly retained the builder-only test rule. |
| Safe synthetic dry run | **Not run**. Prepared commands/configuration; no synthetic metrics reported as results. |
| Builder deployment | **Not attempted**. No fresh authorized connectivity established. |
| Hardware baseline | **Not attempted**. Current access, image/health and exclusive ownership unverified; portal pending. |
| Live optimization loop | **Not running**. No daemon, schedule, model provider or remote worker started. |
| Static editing check | `git diff --cached --check -- . ':!evidence/sources/**'` completed without findings. Raw source snapshots retain original whitespace. This is not a test-suite result. |

## Blockers

1. **Action:** transfer this small isolated harness and run `python3 -m unittest discover -s tests -v`, then the fixture resume/rollback demonstration. **Target:** existing `opentpu-m8a-build`, historical ID `i-0ec8d33362b3100be`. **State:** pending authorized access and current ID/role verification; not attempted and no uncertain outcome. **Reason:** inherited denied SSH checks cannot be retried or routed around; the user declined the local-fixture exception and reaffirmed builder-only execution. No new automatic-approval rejection occurred in this task. A materially new authorization/current permitted access receipt is needed before those checks can proceed.

2. **Action:** bounded matched baseline on the already loaded verified image. **Target:** existing `opentpu-f2`, slot 0 / BDF `0000:34:00.0`, historical ID `i-0c78de2253f66519d`. **State:** queued, not attempted; no uncertain hardware outcome. **Reason:** current authorized connectivity and exclusive canonical ownership are not established, and the interactive portal/existing owners have priority. Historical restore receipts do not satisfy this condition. No SSH, SSM, process inspection, localhost GET, image load, reset or HBM write was performed.

3. **Action:** enable a live seed under the new objective. **State:** pending implementation/qualification in the authorized owner session. **Reason:** the existing runtime does not expose all required caller/prefill/token/cleanup evidence in a compatible trace. A reviewed and pinned trusted instrumentation bridge is still needed, along with full checkpoint/tokenizer/runtime/image fingerprints and independently frozen fixed-length performance/held-out references. The historic eight-token smoke is available but not broad qualification. No live configuration was invented to bypass these gaps.

Prepared recovery path: establish permitted builder access without retrying a denied action absent new evidence; run the supplied suite; fix any findings in this isolated branch and repeat affected tests; coordinate a later exclusive card window with the portal/owners; implement and qualify the timing bridge under that owner; enable a new frozen experiment only for the currently loaded image. Do not start a new instance or RTL campaign to unblock setup.
