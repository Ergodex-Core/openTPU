# C1: independent HBM row engines

**Approved for implementation; reference and bounded RTL milestones implemented.** New Codex
proposal, 2026-10-18. The user explicitly approved parallel work on C1-C3 and required openTPU
Lens for performance measurement/visualization. See [reference evidence](C1-VALIDATION.md)
and [RTL validation scope](C1-RTL.md).
Branch: `codex/v4-c1-hbm-row-engines`. Base: `016e953a`.
See [comparison and approval boundary](README.md).

## Architectural decision

Give each selected HBM pseudo-channel a narrow GEMV engine that owns complete output rows.
Engines progress independently, using private activation snapshots and local memory queues.
One sequencer, VPU, quantizer and TMEM backend serves the whole accelerator.

Claude P2/P3 synchronize wide groups of complete MXU arrays; P4 replicates entire slices.
C1 instead distributes small row accumulators and keeps weight traffic within each engine's
assigned PC. This changes the memory layout and execution contract, rather than simply
setting a wider port parameter.

Initially accelerate M=1 int8 static linear matrices. Retain the existing MXU for prefill,
attention, embedding gathers, four-bit weights and unsupported operations. In particular,
the existing embedding gather's N=9, KB=1, M=4 case remains valid without padded lane reads.

## Initial datapath

| Item | Initial evaluation point | Later experiment |
|---|---:|---:|
| Engines / assigned PCs | 16 | 32 |
| Int8 products per engine per core cycle | 32 | 32 |
| Core frequency target | 200 MHz | 200 MHz |
| Weight consumption per engine | 32 B/cycle | 32 B/cycle |
| Aggregate weight-path ceiling | 102.4 GB/s | 204.8 GB/s |
| Useful M=1 products per cycle | 512 | 1,024 |

A 128-byte block takes four cycles in one engine. At 200 MHz its 6.4 GB/s demand is below
the 14.4 GB/s arithmetic peak of a 32-byte, 450 MHz PC interface. Requests, metadata and
refresh consume the margin; no utilization percentage is assumed proven.

Each engine has a bounded read queue, row/header buffers, a 48-bit accumulator, a result
FIFO and local activation SRAM. Two accumulator contexts may hide row transitions if the
epilogue cannot accept the next completed row; their benefit must be measured. Begin with
the existing eight-lane vector/quantizer backend; a 16-lane variant is a separate comparison.

The quantizer's completed activation snapshot is multicast into engine SRAMs. Start commands
only after the corresponding snapshot is ready, and prevent overwriting it while engines
read it. For a maximum 16 KiB activation vector, 16 copies need 256 KiB of byte storage,
plus scales and RAM fragmentation. Double buffering is optional, not free.

## Explicit storage and command contract

Introduce a versioned matrix descriptor and `GEMV_ROWS` operation (symbolic name; no opcode
assigned in this draft). Retain canonical weights for fallback and create an immutable
row-local copy with identical quantized bytes and scale values.

For E engines, full matrix row n, and full_KB blocks:

```text
pc        = n mod E
local_row = floor(n / E)
header    = align_up(4 + full_KB, 32)
pitch     = header + 128 * full_KB
address   = base[pc] + local_row * pitch

record: original fp32 Sw; original full_KB block-offset bytes; alignment padding;
        original weight blocks 0 .. full_KB-1
```

The descriptor includes shape, per-PC bases, pitch, format and allocation bounds. Command
fields retain n0, k0, N and KB so existing chunked down-projection calls select the original
columns and scale offsets. Matrices occupy nonoverlapping PC-local ranges; the allocator
checks capacity explicitly. A host-visible descriptor uses PC identity plus offset rather
than pretending the legacy 32-bit flat address covers all HBM.

Legal AXI bursts may prefetch consecutive records on one PC, split at controller limits and
4 KiB boundaries. Never issue padding as a logical weight. Each engine stops at the exact
row tail. Keeping the canonical image costs approximately another model-sized copy for
accelerated weights; this is intentional initial validation overhead.

## Arithmetic, retirement and dependencies

A 32-element int8 beat is one ISA quantization group. Compute its exact integer dot product,
apply the existing shifted-group rule, and accumulate in k/group order into 48 bits.
No fp32 partial-row accumulation is introduced. Preserve the existing final conversion,
multiplication order, rounding and flush behavior; see [ISA MM](../isa.md) and
[the current MXU](../../rtl/mxu/otpu_mxu.sv).

Initial fast-path flags cover plain output and ACC; other combinations use the original MM.
ACC must read the correct previous result and complete before a dependent command uses it.
Results carry command, activation-generation and row identifiers. A bounded reorder/retire
buffer writes canonical TMEM addresses and completes a command only after all effects retire.
A delayed PC must backpressure dispatch before exhausting that buffer. RMAX/ASCALE remain
on the compatibility path initially; a future extension must match their exact semantics.

This explicitly adds compiler dispatch, an image descriptor, an ISA-simulator operation,
capability reporting and a new timing model. It does not reuse Lens's C multiplier as a
simulation of independent engines.

## Performance and resource limits

For approximately 1.005 GB of baseline streamed bytes, ideal weight-only time is about
9.8 ms at 16 engines or 4.9 ms at 32. These are optimistic lower bounds that assume all those
bytes use the accelerated path; actual execution has fallback work and additional traffic.

Model each command using per-engine rows, header bytes, activation-copy time, PC latency,
credit stalls and result retirement. Then add the dependency-critical fallback/vector work.
Do not divide the complete 10.28M-cycle baseline by 4 or 8. Prefill initially has the existing
MXU's compute width and no promised speedup.

Initial product PEs alone require 512 multipliers, plus the retained MXU, group scaling,
accumulators, epilogues and shared backend. This is not a total DSP estimate. Activation SRAM,
per-PC queues and result buffering require a shell-constrained resource report. Many small
engines may trade an easier local weight path for expensive activation fanout and retirement.

Principal failure modes: activation barriers dominate short MMs; slow-PC head-of-line blocking;
scale/header request overhead; central TMEM drain saturation; and insufficient usable CL area.

## Approved milestones

1. Implement descriptor packing and an executable arithmetic model. Verify exact bytes/scales
   and outputs for tails, k0 subranges, KB=1/6/9/54, ACC and unsupported-case fallback.
2. Build a per-PC timing model with bounded queues and a deliberately delayed PC. Require
   predicted improvement over the same-clock baseline after all exposed costs; reject the
   design if activation distribution and retirement erase it.
3. Verify one engine, then 16 engines, under randomized AXI latency, backpressure, reset and
   response ordering. Require exact ISA logits/tokens and unchanged fallback behavior.
4. Integrate the selected F2 shell/DMA, measure actual PC traffic, and synthesize/place before
   deciding whether 32 engines or a wider VPU is justified.

Milestone 1 now has a versioned byte-preserving sidecar, an independent executable arithmetic
reference and tests against canonical ISA execution. An initial Lens event-policy experiment
helps inspect independent progress, bounded row credits and delayed PCs; it is not the full
the later bounded RTL milestone below. Compiler dispatch and F2 integration remain future work.


## Approved bounded RTL milestone

The user approved continuing all three candidates. C1 now has a synthesizable-style
per-PC row engine with bounded multi-outstanding slots, an ordered cluster, and the
repository-exact shared fp32 epilogue. See [C1-RTL.md](C1-RTL.md) for the executable
interface contract, tests, Lens artifacts and exclusions. Plain-output kernel execution
is separate from the earlier reference ACC support; AXI/header/TMEM integration and
F2 timing remain future gates. The original proposal estimates are not measured RTL.
