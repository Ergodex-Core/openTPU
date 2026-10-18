# Codex alternatives for openTPU v4 on F2

Status: **user explicitly approved parallel implementation of C1-C3**, with openTPU Lens
required for performance measurement and visualization. C1's first reference-validation
milestone is complete in this branch; see [C1 evidence](C1-VALIDATION.md).
Date: 2026-10-18. Base: `016e953ad70e9dd78f1b10d8a9cc07318bee66dd` on `tpubench`.
These C1-C3 designs are new alternatives to Claude's P1-P4 in [docs/v4](../v4/README.md).
This approval is the user's subsequent instruction for these new alternatives, independent
of the earlier approval recorded in Claude's documents.

| Candidate | Architectural change | Branch | Worktree directory |
|---|---|---|---|
| C1 | Independent narrow GEMV engines beside HBM pseudo-channels, sharing one vector/control backend | `codex/v4-c1-hbm-row-engines` | `opentpu-v4-c1-row-engines` |
| C2 | Reuse the same four MXU columns as four output-row engines during decode and activation columns during prefill | `codex/v4-c2-dual-mode-mxu` | `opentpu-v4-c2-dual-mode-mxu` |
| C3 | Dedicated vocabulary projection and exact greedy reduction alongside the existing layer datapath | `codex/v4-c3-vocab-engine` | `opentpu-v4-c3-vocab-engine` |

This branch's design is in [PROPOSAL.md](PROPOSAL.md). All worktrees are siblings under
`/Users/rachittibrewal/ergodex-04`. Each starts from the same committed baseline and contains
its own proposal, this comparison and its independently implemented milestone work. Existing
baseline profiles and the untracked `formal/` directory remain in the original worktree;
C1's new Lens artifacts are generated under this worktree's `build/lens/c1/`.

## Recommendation and evidence

Evaluate **C2 first**: at M=1 it aims to use all 512 existing product PEs, compared with
Claude P2's 512 useful product PEs out of 2,048. This is a proposed utilization improvement,
not a fourfold end-to-end performance claim. The new operand feeds, metadata, result
addressing and mode transitions still need validation.

C1 is the alternative if independent memory progress is more valuable than preserving the
MXU structure. C3 limits the scope of the accelerated operation, but accelerating a phase
that is roughly 30% of a token caps total speedup at about 1.43x if the rest is unchanged.
It is a bounded comparator, not the leading maximum-throughput design.

Claude's baseline provides Gemma 3 support and real RTL simulation traces. His P1-P4 profiles
are analytic re-timings: they do not execute candidate RTL or model HBM scheduling. Relevant
evidence is [Gemma 3 measurements](../gemma3.md), [candidate methodology](../v4/README.md),
[ISA arithmetic](../isa.md), and [current MXU](../../rtl/mxu/otpu_mxu.sv).
The 10,279,178-cycle trace corresponds to 51.40 ms **if** run at 200 MHz; it is not an
F2 card measurement. Existing row-blocked counters combine several causes, so no new
proposal assumes all that time disappears with HBM.

## Common F2 assumptions

Use 200 MHz as a comparison target, not achieved timing for these designs.
AWS's [memory performance example](https://awsdocs-fpga-f2.readthedocs-hosted.com/latest/hdk/cl/examples/cl-mem-perf/README.html)
exercises 32 HBM channels at 450 MHz. A 32-byte interface at that frequency has a
14.4 GB/s arithmetic peak; useful bandwidth depends on requests and contention.
The [F2 HDK](https://awsdocs-fpga-f2.readthedocs-hosted.com/latest/hdk/README.html)
uses Small Shell and requires a CL-side DMA solution. Budget its resources and the selected
shell's placement constraints before claiming FPGA fit. None of these proposals assigns an
entire SLR to an engine or treats all device resources as available custom logic.

All performance estimates must include scale traffic, activation distribution, dependent
vector work, tails, command transitions, response latency, drain credits and fallback work.
Keep int8 weights and existing quantization as the common comparison; mixed precision,
speculation, extra users and software row folding are separate experiments.

## Approved scope and current boundary

The user approved fan-out across all three branches and required Lens for performance work.
This branch implements C1's descriptor/packing and arithmetic-reference milestone, with
focused validation and explicitly analytic Lens profiles. Sibling branch status is tracked
in those worktrees. C1 does not yet change compiler dispatch or the existing ISA, and contains
no C1 RTL, synthesis result or cloud deployment.

Continue comparing each implementation against the existing ISA and a same-clock baseline
before full F2 integration. Reference CPU duration, analytic policy cycles, RTL simulation
cycles and actual device measurements must remain explicitly distinguished in Lens.
