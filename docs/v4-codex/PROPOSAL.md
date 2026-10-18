# C3: dedicated vocabulary-head engine

**Draft; awaiting user approval before implementation.** New Codex proposal, 2026-10-18.
Branch: `codex/v4-c3-vocab-engine`. Base: `016e953a`.
See [comparison and approval boundary](README.md).

## Architectural decision

Keep transformer layers on the existing slice and add a separate engine for the final
Gemma 3 vocabulary projection. Four autonomous engines each own 65,536 complete vocabulary
rows, compute exact logits, and return a best (value, global token ID) pair.

This is physical specialization of the head. Claude P1-P3 widen general MM execution;
P4 splits the whole model over slices. Claude's software row folding stays on the generic
MXU and changes row quantization. C3 instead preserves every original row's quantization
and adds dedicated memory queues, row arithmetic and a direct reduction output.

The initial accelerated case is **greedy decoding without repetition penalties**. Sampling,
penalties, full-logit debug output, unsupported formats and other models keep the existing
head path.

## Datapath and memory layout

Broadcast the final normalized/QACT activation vector once: 1,152 quantized bytes plus its
scale metadata. Each engine stores a local snapshot and reads its owned rows independently.
A starting point is 128 int8 products per cycle per engine at a 200 MHz target, consuming
25.6 GB/s per engine before metadata. Four PCs per engine provide local bandwidth headroom;
only small within-engine queues/multiplexers join those PC streams.

A concrete sidecar layout for engine e and its local row r, with KB=9:

```text
row_global = e * 65536 + r
pc_local   = r mod 4
pc         = 4 * e + pc_local
local_row  = floor(r / 4)
header     = 32 bytes: original Sw + 9 original block-offset bytes + padding
pitch      = 32 + 9 * 128 = 1184 bytes
address    = base[pc] + local_row * pitch
```

Whole rows stay on one PC, while successive rows use the four local PCs. Multiple outstanding
rows hide per-PC latency; ordered tags reconstruct the engine's stream. Burst splitting must
respect the chosen HBM controller's limits and 4 KiB boundaries.

Initially retain the canonical tied embedding/head image and add this byte-exact head copy
(about 310 MB including headers). Existing embedding gathers and fallback head execution
therefore remain available. Both images must derive from the same quantized tensor, with
matching checksums and no independent requantization. Updates require a new consistent image.
Removing the duplicate later would require a separately validated embedding lookup mapping.

The fp32 epilogue feeds a running local argmax without writing every logit to shared TMEM.
A small final reducer combines four pairs. A diagnostic mode must expose complete logits
for numerical verification; the fast greedy contract returns only the winner.

## Exactness and integration

Use the current int8x/a8x32 arithmetic: exact shifted integer groups accumulated into 48 bits,
then the existing i2f48, Sw and Sa scaling, with the same rounding and flush behavior.
Do not replace it with a conventional fp32 dot product. Preserve the canonical logits.

Local and global selection use the ISA's flushed-bit total ordering, including signed zero.
Equal keys select the lowest global token ID regardless of response arrival order. Exceptional
values must follow the same ISA comparator, not a host-language floating comparison.
See [ARGMAX](../isa.md) and [Greedy](../../opentpu/llm/generate.py).

Add an explicit versioned head-command capability at
[the final-head handoff](../../opentpu/llm/qwen3.py). Its completion feeds the existing card
generate loop, preserving token feedback, EOS, host-stop polling, bucket transitions and
output posting. A new command requires simulator, compiler, dependency and capability support.

One winner per engine cannot reproduce top-k/top-p sampling or repetition penalties.
Those modes fall back before issuing the specialized command. Later support would need
per-token penalties, enough retained candidates, matching ordering and matching RNG behavior.

## Realistic payoff

The recorded baseline is about 10.28M cycles, of which about 3.1M is the head. With all other
cycles unchanged, total time is 7.18M + 3.1M / head_speedup. At an assumed 200 MHz:

| Conditional head speedup | Total modeled cycles | Conditional token rate | Whole-token speedup |
|---|---:|---:|---:|
| 4x | 7.955M | 25.1 tok/s | 1.29x |
| 8x | 7.568M | 26.4 tok/s | 1.36x |
| Infinite | 7.180M | 27.9 tok/s | 1.43x ceiling |

These are Amdahl sensitivity calculations, not measured or predicted engine performance.
Four 128-byte engines give a nominal 102.4 GB/s weight-path ceiling; the head's 302 MB of
weight bytes alone take about 2.95 ms at full duty. Scales, row turnarounds, broadcasts and
reduction add time. Layers remain serially before the head for one sequence.

The current greedy card loop already avoids full-logit host transfers and overlaps chunk
ARGMAX with streaming. C3 must not count their removal as an additional large speedup.
The appeal is a bounded specialization with an unchanged layer compute path; it does not
offer the multi-fold whole-model gains sought by the broader proposals.

## Resources and failure modes

The new product PEs alone number 512, in addition to the existing slice. Budget scale logic,
48-bit accumulators, epilogues, row queues, activation storage, shell/DMA and local PC routing.
Do not assume they fit based on device-wide DSP totals. The added engines are idle throughout
most of the layer phase, so area efficiency is a significant disadvantage.

Other risks are output-selection corner cases, PC-scale traffic, new command completion
ordering, stale sidecar weights and shared HBM arbitration slowing the existing layers.
C3 should be rejected as a performance project if measured layer slowdown cancels head gains.

## First milestones, only after approval

1. Model sidecar addressing and exact head logits against the existing ISA. Check every row's
   weight/scale identity, extreme offsets, dropped terms and rounding boundaries.
2. Prove selection equivalence for cross-engine ties, signed zero, exceptional bit patterns,
   first/last vocabulary IDs and shuffled completion order.
3. Build and verify one engine plus diagnostic logit output, then four engines and reduction.
   Measure phase-separated cycles and traffic under randomized stalls.
4. Integrate the command with greedy generation and explicitly exercise all fallback modes,
   EOS, cancellation, reset and bucket chaining before F2 shell synthesis.

No implementation, simulation or synthesis of C3 has been performed.
