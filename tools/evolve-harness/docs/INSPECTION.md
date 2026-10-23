# Inspection and isolation

Work is isolated in `Documents/Codex/2026-10-21/task-8/opentpu-evolve`, branch `codex/opentpu-evolve-harness`. It is a new standalone orchestration repository, not an optimization fork of an existing owner's runtime/RTL tree. Existing source files were read only; small evidence excerpts were copied here with hashes.

Inspected:

- `ergodex-03/openTPU-hwloop/tools/hwloop/{cli.py,nightly.py,nightly/queue.json}`: the existing queue prepares new RTL/FPGA builds. The current entries are queued awaiting approval. New runtime tuning must not launch its BuildAll/package/AFI flow.
- `ergodex-03/openTPU-wp-alexshell`, clean branch `wide-port-alexshell`, observed HEAD `0a868822a71422beb11771c9fa046af762579e81`: serving API, scheduler, bench client, canonical runstate lock, lock tests, Gemma physical documentation.
- `ergodex-03/openTPU-model-infer-f2`: existing serving/runtime layout and model work; no new PersonaPlex eligibility assumed.
- `ergodex-04/openTPU`: main checkout has untracked `scalesim_v3/`; left untouched. `tpubench`, `openTPU-p2-f2`, `openTPU-p4-f2` documentation and local physical logs; Nanbeige source receipts in its isolated workspace; C2 physical artifacts.
- Local `~/.codex/memories/{MEMORY.md,memory_summary.md}` had no relevant OpenTPU entries. Relevant Claude memory files were read as historical project context: `reference-hwloop-visualsim.md`, `opentpu-shared-f2-etiquette.md`, `feedback-runs-on-ec2-only.md`, `feedback-verify-instance-before-run.md`, e04 `build-hosts.md`, `f2-card-sharing.md`, `tpubench-gemma3-worktree.md`.
- Relevant root/checkout `AGENTS.md`, `CLAUDE.md`, `.agents/skills`, `.claude/skills` were searched. No applicable checkout skill/instruction file was found at these roots; ASIC skills are unrelated and were not applied. The installed **fpga-engineer** SKILL.md was applied for physical provenance and release-fingerprint discipline. No synthesis/timing closure skill was needed.
- OpenEvolve GitHub README was read through web access for architecture only: bounded proposal/evaluation, population/archive, seeded reproducibility, artifacts and resume. No repository or package was downloaded/executed, and no provider credentials configured.
- Existing portal thread `01a15f28-ba77-7599-a1f7-c1684cf710f6` and wide/spec thread `01a15edd-6c33-7532-b189-b38c934aa0d6` were read through task metadata. The available returned turns showed active requests and no verified execution/ownership. No messages were sent. Their local task directories were empty at inspection. Wide200's isolated RTL owner remains out of scope.

Owner rule: canonical card lock is `/tmp/otpu/f2-0000:34:00.0.lock`, not a new harness-local lock. A historical restore to Alex burst200 is explicitly insufficient to prove availability. Memory describes a prior incident where leaving a foreign image loaded disrupted C2; this harness implements no image management. Portal/other owners win; the hardware request is queued until coordinated current evidence exists.

The user re-confirmed during this task: **keep tests on opentpu-m8a-build**. Therefore no unittest, fixture evaluator, dry-run loop, simulation, model inference or build was run on the Mac. Local work consists of inspection, editing, source extraction, packaging and static diff review. Prior denied SSH checks, process inspection and localhost GET were not retried through any channel.
