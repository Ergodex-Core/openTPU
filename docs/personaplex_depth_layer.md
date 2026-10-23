# Reusable PersonaPlex depth layer

This component implements pre-RMSNorm attention, persistent device KV, residuals
and a SiLU-gated feedforward block with16 device-selected weight banks. It is a
bounded depth pass, not complete PersonaPlex speech inference. Source checkpoint
validation now covers all six layers over16 steps plus reset replay. This is
CPU/ISA execution on saved diagnostic inputs, not generated speech or physical
FPGA validation. Resident integration remains separate.

## Exact native contract

The checkpoint JSON is not sufficient to infer the native streaming behavior.
Verified installed source hashes are pinned by the checker:

- `models/loaders.py` sets native depth width1024,16 heads, six layers,
  `rms_norm_f32`, SiLU gating, no position embedding and no layer scale.
- Native constructor feedforward width4224 produces actual gated width2816.
  Checkpoint `dim_feedforward=2816` records that effective width; gate input has
 5632 rows and down projection has2816 columns.
- `models/lm.py:321` replaces the merged depth context with`None`. JSON/loaders
  context8 is therefore inactive. Native cache capacity falls back to16 steps.
- Each layer has two shared FP32 norm vectors, contiguous step-selected packed
  Q/K/V and output projections, and16 separate SiLU gate/down pairs.
- Norm order is`x * (alpha * rsqrt(eps + mean(x*x)))`, epsilon1e-8. There is no
  RoPE, q/k norm, bias, layer scale or final depth normalization.
- Each temporal frame enters a fresh depformer streaming context. The six
  layers have independent caches during its16 codebooks and reset before the
  next frame; no cross-frame depth history is retained.

There is an observed native boundary defect in `RingKVCache.complete`: on the
16th write it assigns positions`[16,1,2,...,15]` to physical slots0..15. Query15
therefore excludes slot0, although it has not been overwritten. The default
`cache_policy="native"` reproduces this exact behavior: prefixes0..step for the
first15 steps, then1..15 at the final step. An explicit`"causal"` diagnostic
policy retains0..15 at the final step. It is a deliberate source deviation,
not silently described as native parity. Selecting corrected production behavior
requires a separate source/quality decision, not a hardware image change.

## Device implementation

`DepthLayerEngine.step(hidden)` uploads only the diagnostic input vector and
returns nothing. A persistent device counter selects every weight bank, KV
position and mask row. RLD computes all scaled addresses on device; the program
requires no host step/address arguments. Explicit `read_output()`, `read_counter()`
and `read_cache()` calls support validation. A future unified allocator must
connect resident conditioning/output slots directly instead of diagnostic uploads.

KV and shared norms are allocated outside the repeated weight banks. For a
1024-wide layer,16 native64-wide heads use padded128-wide K/query storage on the
D128 profile. V products emit only64 dimensions; the attention scale remains
`1/sqrt(64)`. The existing padded KV primitives support this without inflating
Q/K/V/output checkpoint matrices. Finite16-step passes use physical capacity128
and resident validity masks; they need no KV ring or context8 eviction.

INT8 QACT/QST and matrix operations execute projections/attention/feedforward;
residuals, norms, SiLU and online softmax use ISA FP32/FTZ. Source affine INT4
weights are dequantized and converted offline to MX INT8. Reset writes only the
counter; masked stale KV is overwritten before becoming visible. The engine
rejects another step after16 until reset. Failures latch until reset; direct
`backend="board"` is rejected before payload preparation. A physical factory
must qualify its exact measured profile and CAP_GEN before upload.

A single program can apply multiple layers sequentially, each with its own
norms, step banks and KV, incrementing the shared step counter once after the
last layer. Synthetic two-layer output is bit-exact against separate one-layer
ISA engines, including cache and reset behavior. That proves the composition
mechanism for the tested fixture. The later real six-layer check below measures
source-reference error separately; complete speech quality remains unaccepted.

## Capacity and fresh EC2 tests

Measured compilation/layout on the recorded burst200 ISA profile:

| Configuration | Required bytes | Weight bytes | KV bytes | Instruction words |
|---|---:|---:|---:|---:|
| One layer,16 banks | 209,563,648 | 208,838,656 | 540,672 | 4,128 |
| Six layers,16 banks each | 1,256,460,288 | 1,253,031,936 | 3,244,032 | 24,088 |

Both fit4GiB DRAM and32,768 instruction words. Neither needs host program
arguments. One step bank is13,051,904 bytes. The first compile failed on a
broadcast-only RMS expression; using a full tile preserved native multiply
order and resolved it. This was a compiler expression issue, not an RTL gap.

EC2 aggregate `c2ed51eb-23d9-4bd9-b1ca-75579dc8ec66` passed451 tests in24.39
seconds, physical opens0. It includes35 new layer cases and10 new reader cases.
Eight native-checker helper tests then passed in1.67 seconds under
`ef122f48-fc14-48e3-a0c4-3ba9fb2258d9`, physical opens0. Tests cover independent
FP64 references, both mask policies, source-row order, 64-wide padding onD128,
device counter/bank selection, poisoned host sources and transfer limits,
exact reset/cache replay, composition, capacity/lazy-read/failure guards and
actual BoardBackend over IsaTransport. That protocol test is CPU simulation.
All execution used existing EC2 dependencies, one thread, nice19 and affinity.

The bounded checker is:

```sh
python tools/personaplex_depth_layer_check.py \
  --model-dir /existing/pinned/model \
  --conditioning-dir /existing/depth-input-isa-002 \
  --output-dir /new/isolated/depth-layer-check
```

The default one-layer check reorders the first16 saved conditioning cases by
codebook ID. Those inputs
come from four captured temporal frames, so they are a bounded diagnostic
sequence, not an end-to-end generated audio pass. It builds the original native
FP32 streaming layer from dequantized source matrices, checks actual native
cache positions and native reset, then runs16 ISA steps plus16 reset replays.
It saves source-reference errors, device counters, exact output/KV replay,
source/baseline hashes and arrays. Five native source hashes are pinned because
the context override and boundary behavior are part of the verified contract.
Checkpoint identity uses revision plus file size/mtime, not exhaustive payload
hashing. No temporal resimulation or model download is required.

## Cached real-checkpoint result

EC2 command `762e7ce6-c1bc-4b4d-9546-841d135ee020` completed successfully.
The checker took 12.34 seconds of CPU wall time (14.23 seconds including its
process wrapper); image construction took 4.61 seconds. The physical-open guard
recorded zero. These times are CPU diagnostics, not FPGA frame latency.

All 16 device-selected steps completed. Output and raw KV replay after reset
were bit-exact; the native reference independently repeated exactly. Aggregate
relative L2 against the original dequantized-source FP32 native layer was
0.009831740967520484, cosine similarity 0.9999516731649192, and maximum absolute
error 0.023688793182373047. Per-step relative L2 ranged from
0.005037733479337598 to 0.028161990397831745. The recorded native positions
confirmed the final-step slot-zero exclusion. This is one layer on identical
saved diagnostic inputs, not accumulated six-layer error or accepted speech
quality. No tokens/audio were generated by this check.

The pinned checkpoint revision remained
`8c686347007de51d9017706e5543d6f1757ecf5d`; baseline hashes, source hashes and
checkpoint metadata were unchanged. Small JSON/log receipts are preserved in the
task handoff; raw arrays remain in the isolated EC2 `depth-layer-isa-001` folder,
with their SHA-256 values in `depth_layer_check.json`.

## Six-layer real-checkpoint execution

EC2 command `78e484be-4414-4b2c-8f78-6dae64e344ab` completed successfully with
all six checkpoint layers, their16 selected banks and independent KV caches.
The updated checker uses the installed original `StreamingTransformer` container,
strictly loads every source layer, records every native cache's positions and
captures layerwise native outputs. It compares one combined ISA program with
that FP32 source reference on the same saved16-input diagnostic sequence.

```sh
python tools/personaplex_depth_layer_check.py \
  --model-dir /existing/pinned/model \
  --conditioning-dir /existing/depth-input-isa-002 \
  --output-dir /new/isolated/depth-stack-check \
  --layers 6 --single-layer-dir /existing/depth-layer-isa-001
```

A completed, hash-verified layer0 baseline is required. Native layer0 outputs and
integrated ISA layer0 KV match that prior baseline bit-for-bit. All16 full-stack
outputs and all six raw KV caches repeat exactly after reset. The native output
and every layerwise native intermediate independently repeat exactly. All six
native caches record final positions `[16,1,2,...,15]`, preserving the pinned
boundary defect rather than changing semantics silently.

Aggregate relative L2 against original dequantized-source native FP32 is
0.022068660575960623, cosine0.9997565657586498, maximum absolute error
0.6853427886962891. Per-step relative L2 ranges from0.010948993364013228 to
0.03788415210113754. These measure additional INT8 stack error on identical
saved conditioning inputs, not original unquantized checkpoint quality or full
speech error. No audio or greedy codebook sequence is generated by this check.

The run took73.33 seconds process wall time /70.97 seconds inside the checker,
including5.70 seconds source materialization and27.75 seconds ISA image build.
These are CPU diagnostics, not FPGA latency. The guard recorded0 physical opens.
All execution used one CPU thread, nice19 and affinity23, with245GiB headroom.
Fifteen updated checker tests passed in1.71 seconds beforehand, command
`ae5c10e9-4df7-4f73-9e2e-b3ec655da059`, physical opens0. They include actual
native container ordering versus separate layers, independent cache/reset,
forward-hook cleanup, layer-count guards and required-baseline guards. Existing
core arithmetic and the prior451-test aggregate remain unchanged.

The first helper submission failed at wrapper parse time because of newline
escaping; no source copy, import, test or model job executed. The corrected
syntax-checked wrapper passed. Checkpoint revision/file metadata, native source
hashes, conditioning and one-layer baseline hashes remained unchanged. Raw arrays
remain in isolated EC2 `depth-stack-isa-001`; small receipts are exported through
the task handoff. The legacy receipt key `six_layer_capacity_only` now explicitly
records `checkpoint_execution_tested: true` for this six-layer check.

## Composition within the residency plan

The [unified plan](personaplex_residency_plan.md) reserves independent persistent
KV/activation/token state and two weight overlays. Six-layer weight footprint
1,253,031,936 bytes plus the already validated input projections67,698,688 bytes
is1,320,730,624 bytes, leaving21,446,656 bytes inside the proposed1.25GiB depth
bank. Shared norms can be placed by the allocator independently of matrices;
all cache and interstage activations stay in the persistent region. Program
storage belongs to the reserved program region. These relocations and unified
allocation checks are not implemented by constructing independent engines.

The measured six-layer layout/program and checkpoint execution remove the prior
standalone depth capacity/execution gates for this bounded INT8 design. Relocation
and integrated conditioning→depth→head token flow still need tests. The
[concrete next implementation plan](personaplex_depth_chain_plan.md) starts with
one sparse resident depth chain before temporal paging.
Keep head selection and next-token IDs in device slots; host submission and
transport may orchestrate predetermined stages but must not compute activations,
embeddings, logits or token selection. Mimi remains unbudgeted and unsupported.

## Supported precision and remaining gates

| Item | Status |
|---|---|
| Seven selected matrix kinds per layer/step: Q,K,V,O,gate,up,down | MX INT8 implemented;16 banks/layer |
| Shared norms, SiLU, softmax, residuals | ISA FP32/FTZ implemented |
| KV and intermediate matrix activations | INT8 with scales implemented |
| Native affine INT4 checkpoint packing executed directly | Unsupported; offline conversion required |
| MX INT4 depth weights | Not implemented/validated by this component |
| FP16/BF16/FP32 depth matrix/activation execution | Not implemented by this INT8 path; native FP32 is diagnostic reference only |
| Six-layer checkpoint execution | CPU/ISA16steps + reset validated; source error reported above |
| Complete model quality | Pending; standalone depth correctness is insufficient |
| Mimi convolution/transposed convolution, codec attention, RVQ and streaming state | Unimplemented/unbudgeted |
| Unified temporal paging, resident stage chaining, prompt/cache/delay scheduling | Unimplemented |

No required new RTL/image capability has been demonstrated. The current kernels
and capacity pass on ISA. A genuine unsupported operation, non-partitionable
capacity limit or required addressing/performance change must be demonstrated
and reviewed before new image/synthesis/resources. Live-card qualification is
unsubmitted while the shared Nanbeige window is arranged. No reload, lock action,
service restart or new paid resource occurred. Transfer bandwidth and complete
frame latency remain unmeasured; no realtime or full speech claim is made.
