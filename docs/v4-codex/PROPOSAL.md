# C3: dedicated vocabulary-head engine

**Implementation approved; reference/layout and reduction milestones completed.**
New Codex proposal, 2026-10-18. Hardware integration is still pending.
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

The original DDR-like recorded baseline is about 10.28M cycles, of which about 3.1M is the head. With all other
cycles unchanged, total time is 7.18M + 3.1M / head_speedup. At an assumed 200 MHz:

| Conditional head speedup | Total modeled cycles | Conditional token rate | Whole-token speedup |
|---|---:|---:|---:|
| 4x | 7.955M | 25.1 tok/s | 1.29x |
| 8x | 7.568M | 26.4 tok/s | 1.36x |
| Infinite | 7.180M | 27.9 tok/s | 1.43x ceiling |

These are Amdahl sensitivity calculations, not measured or predicted engine performance.
They are historical comparisons, not the preferred F2 baseline. Concurrent Claude milestone-0
work on `tpubench` (`112ef960`, `docs/v4/m0-gating-split.md`) measured the existing RTL at
8,221,777 cycles with stall=0, LAT=30. Reading that run's actual Lens profile, the first head
MM starts at cycle 5,806,145 and the remaining head/tail is 2,415,632 cycles. The corresponding
infinite-head speedup cap is **1.416x**, with unchanged layers. This is still a DDR memory
simulation with HBM-like settings, not an F2 card measurement. Its head has no blocked cycles
from result credits or missing scales. C3 does not count recovery of the original 18.2% as
an architectural gain: milestone 0 attributes almost all that item to the memory's delivery
rate. These concurrent RTL changes have not been cherry-picked into this isolated branch.

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

## Implementation milestones (approved)

1. Model sidecar addressing and exact head logits against the existing ISA. Check every row's
   weight/scale identity, extreme offsets, dropped terms and rounding boundaries.
2. Prove selection equivalence for cross-engine ties, signed zero, exceptional bit patterns,
   first/last vocabulary IDs and shuffled completion order.
3. Build and verify one engine plus diagnostic logit output, then four engines and reduction.
   Measure phase-separated cycles and traffic under randomized stalls.
4. Integrate the command with greedy generation and explicitly exercise all fallback modes,
   EOS, cancellation, reset and bucket chaining before F2 shell synthesis.

## Completed reference milestone and Lens evidence

Implemented [vocab_engine.py](../../opentpu/llm/vocab_engine.py), an isolated executable
reference, with [tests](../../tests/test_vocab_engine.py) and a reproducible
[validation/export command](../../tools/c3_head_check.py):

- Versioned per-PC layout, address/range validation, uneven vocabulary tails, and legal
  32-byte read bursts that respect 4 KiB boundaries and a configurable beat limit.
- Immutable sidecar images made from existing quantized bytes, preserving every canonical
  weight and scale byte (including scale padding). Source and per-PC SHA-256 checks verify
  descriptor reloads, payload integrity and consistency with the retained tied image.
- Independent scalar/Python-integer row accumulation, repository fp32 epilogue semantics,
  and local/global reduction with lowest-global-ID ties under the ISA bit ordering.
  Duplicate, missing and foreign completions are rejected. The completion order is varied.
- Explicit fallback reasons for unsupported models/dimensions/formats, sampling, penalties,
  full-logit output and absent/wrong capability versions. No caller dispatches C3 hardware;
  the current chat and generate loop remain on the established path.

The tests compare full logit bits against **executed canonical ISA LD/QACT/MM/ST programs**,
not a second call to the same C3 implementation. They include extreme offsets, dropped
groups, 48-bit integer-to-float rounding ties, negative accumulations, exceptional values,
signed zero, shuffled shard completions and multi-MM vocabulary chunks. Reduced dimensions
are supported for validation; production eligibility remains Gemma 3 1B only.

```sh
python -m pytest -q tests/test_vocab_engine.py
PYTHONPATH=. python tools/c3_head_check.py --output build/lens/c3
python -m opentpu.lens open build/lens/c3/c3-head-reference.otpuprof
```

The default validation executes a **synthetic 4,096 x 1,152 head** with seed 3. All 4,096
logits match the canonical ISA bit for bit, with winner token 1987. The sidecar is 4,849,664
bytes. The input is synthetic, not a claim of full-checkpoint model validation.

Every performance characterization must be exported to **openTPU Lens**, as requested.
The command writes a two-profile `.otpuprof`, a standalone Lens HTML page, `validation.json`,
the sidecar descriptor and `summary.txt` under `build/lens/c3/`:

1. Canonical ISA head execution with stock Lens nominal instruction timings (37,808 cycles
   for this workload, including QACT and full-logit stores).
2. C3's executed row counts, bytes, MACs, dropped groups and transaction counts, with a
   clearly labeled **serial service accounting model** (27,918 schedule cycles here).
   It assumes 128 read bytes/cycle/engine, 128 integer products/cycle/engine, 8 epilogue
   cycles/row, and no overlap among read, GEMV, epilogue and local reduction phases.
   It starts after QACT and includes activation distribution and final reduction.

These profiles have different timing abstractions and boundaries; their cycle ratio is
**not a speedup result**. C3 has no RTL timing, HBM latency/queue/contention simulation or
predicted full-model token rate yet. The reference's CPU wall time is recorded separately.
The Lens generic floorplan's `S=4` represents head engines, not four full TPU slices or a
validated physical floorplan. Milestones 3 and 4 must replace the accounting model with
measured RTL events and preserve baseline/fallback behavior before any F2 synthesis.
