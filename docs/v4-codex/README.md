# Codex alternatives for openTPU v4 on F2

Status: **user approved parallel implementation of C1-C3; C2 reference milestone complete**.
The user requires openTPU Lens for performance measurement and visualization.
Date: 2026-10-18. Base: `016e953ad70e9dd78f1b10d8a9cc07318bee66dd` on `tpubench`.
These C1-C3 designs are new alternatives to Claude's P1-P4 in [docs/v4](../v4/README.md).
This is a new user approval for these alternatives, separate from Claude's recorded approval.

| Candidate | Architectural change | Branch | Worktree directory |
|---|---|---|---|
| C1 | Independent narrow GEMV engines beside HBM pseudo-channels, sharing one vector/control backend | `codex/v4-c1-hbm-row-engines` | `opentpu-v4-c1-row-engines` |
| C2 | Reuse the same four MXU columns as four output-row engines during decode and activation columns during prefill | `codex/v4-c2-dual-mode-mxu` | `opentpu-v4-c2-dual-mode-mxu` |
| C3 | Dedicated vocabulary projection and exact greedy reduction alongside the existing layer datapath | `codex/v4-c3-vocab-engine` | `opentpu-v4-c3-vocab-engine` |

This branch's design is in [PROPOSAL.md](PROPOSAL.md). All worktrees are siblings under
`/Users/rachittibrewal/ergodex-04`. Each starts from the same committed baseline. This branch
contains only C2 work; its generated Lens profiles live in `build/lens/c2/`.
The original worktree's generated profiles and untracked `formal/` directory remain there.
See [C2 validation evidence and reproduction](VALIDATION.md).

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

## Approval and current boundary

The user approved fan-out implementation of all three proposals. This branch completes
C2's first validation milestone: an executable packed reference with exact arithmetic,
canonical fallback, and Lens event profiles. It does not report the other worktrees' status.

Next are RTL operand routing and metadata, followed by measured cycle traces in Lens.
No C2 RTL, synthesis, deployment or cloud execution has been performed. The current Lens
microbenchmarks measure executed functional work and traffic; their arithmetic-slot axis
is a model, not a measured FPGA clock timeline or a Gemma throughput forecast.
