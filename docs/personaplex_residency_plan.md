# Candidate unified PersonaPlex residency and device chaining

This is a compiler/runtime design, not a demonstrated full-model fit or a
hardware result. Its P/B depth-only subset now has a [shared ISA implementation](personaplex_depth_chain.md)
and cached generated-sequence checks. Optional resident RVQ, causal upsample and
decoder transformer are now software-validated; temporal paging and complete
codec/PCM integration remain unimplemented. No image change, synthesis, card access or paid resource is
required to investigate it. Existing standalone images total5,270,028,288 bytes
versus4,294,967,296 mapped bytes. Shared workspaces alone cannot solve residency:
input embedding tables exceed the temporal/token free space by940,683,264 bytes.

Use one backend with persistent state and tables separated from paged weights.
Host transport may copy immutable prepacked weights and submit predetermined
stages. During inference it must not convert weights, compute embeddings,
normalize activations, calculate logits, select codebook heads or choose tokens.
All such model operations remain device programs. Offline conversion is separate
from inference and produces versioned, hashed packets.

## Candidate address map

All intervals below are end-exclusive. The final2^32 endpoint is a size boundary,
never a stored pointer. Estimates need actual allocator and compiler validation.

| Region | Interval | Capacity |
|---|---|---:|
| Persistent P | `0x00000000–0x60000000` | 1.5GiB |
| Temporal overlay A | `0x60000000–0xB0000000` | 1.25GiB |
| Depth/codec overlay B | `0xB0000000–0x100000000` | 1.25GiB |

| Persistent content | Interval | Bytes |
|---|---|---:|
| State reservation | `0x00000000–0x04000000` | 67,108,864 |
| Program slots including WAITC data | `0x04000000–0x05000000` | 16,777,216 |
| FP32 temporal input tables including zero rows | `0x05000000–0x44488000` | 1,061,715,968 |
| Four captured voice frames | `0x44488000–0x44498000` | 65,536 |
| FP32 depth step embeddings including zero rows | `0x44498000–0x539B8000` | 257,032,192 |
| INT8 text head | `0x539B8000–0x5B7D1400` | 132,224,000 |
| All INT8 audio heads | `0x5B7D1400–0x5D831400` | 33,947,648 |
| Resident temporal RoPE | `0x5D831400–0x5D841400` | 65,536 |
| Final temporal norm | `0x5D841400–0x5D845400` | 16,384 |
| Guard, uncommitted | `0x5D845400–0x60000000` | 41,659,392 |

State subdivisions are control/token/delay/failure queues at`0–0x10000`, temporal
KV at`0x10000–0x02110000` (34,603,008 bytes), a4MiB depth-KV reservation at
`0x02110000–0x02510000`,16MiB activation/scratch at`0x02510000–0x03510000`, and
remaining guard. Temporal KV derives from32 layers×32 heads×128 positions×
(128 K bytes+4 K-scale bytes+128 V bytes+4 V-scale bytes).
The compiled depth layout with six layers, 16 heads, 128 physical positions
and 64-wide heads padded to 128 uses 3,244,032 bytes. The native runtime overrides
JSON context8 to None and uses a 16-step cache; its final-step boundary defect
excludes slot0. The [reusable depth layer](personaplex_depth_layer.md) implements
that pinned policy, independent per-layer KV and reset. All six checkpoint
layers now execute over16 saved diagnostic inputs with exact output/cache reset
on CPU/ISA; source-reference error remains separate from implementation parity.
Codec state, prompt-cache capacity and complete activation liveness remain
unverified.

The current Qwen-derived temporal layout interleaves KV within each layer's
weight block (`opentpu/llm/qwen3.py`). It cannot be overwritten as a page. Refactor
its descriptors so immutable weights/norms address A and every layer's KV points
to fixed P addresses. Persist residual activations between page submissions.

## Temporal packet sequence

Metadata arithmetic using existing mixed rules and current chunked weight/scale
packing gives these candidate packets:

| Packet | Layers | Packed layer bytes |
|---|---|---:|
| A0 | 0–3 INT8 | 829,915,136 |
| A1 | 4–15 INT4 | 1,318,649,856 |
| A2 | 16–27 INT4 | 1,318,649,856 |
| A3 | 28–30 INT4;31 INT8 | 537,141,248 |

Each is below A's1,342,177,280-byte capacity before actual aligned placement.
Layer payload sums to4,004,356,096 bytes. Uploading fixed final norm once adds
16,384 bytes at session initialization; explicitly refreshing it every frame
would make per-frame weight traffic4,004,372,480 bytes. Report which policy is
measured. These byte counts are packing estimates, not measured DMA transfers.

The straightforward bounded schedule halts and drains the previous program,
refills A, marks the complete packet ready, and runs that layer range. Repeat
for all four packets, final norm and text head. Keep all KV and residuals in P.
No host activation readback is required. Do not assume asynchronous large DMA
can overlap active compute: `opentpu/host/board.py` documents corruption with
large H2C writes during active runs and restricts such calls to4KiB. Establish
correct halted paging before considering a separately qualified overlap scheme.

## Depth bank and codec limits

Measured six-layer layout and compilation give 1,253,031,936 bytes for all
16 INT8 banks per layer, including shared norms and repeated chunk scales.
The program uses 24,088 of 32,768 instruction words with no host arguments.
All six checkpoint layers have now been materialized and executed on CPU/ISA,
including exact reset replay. This is standalone depth, not full speech. Sixteen 4096→1024 input projections add 67,698,688
bytes, already materialized and CPU/ISA-validated. B would hold 1,320,730,624
bytes with 21,446,656 bytes of slack. Programs, KV and interstage activations
belong in P; the standalone layout must be relocated accordingly. Unified
allocation/alignment checks and complete activation liveness remain pending.
Step embeddings already reside in P. Individual capacity/compile evidence does
not establish a working combined runtime or complete speech fit.

The optional [Mimi RVQ latent slice](personaplex_mimi.md) now occupies unused P
guard space beginning0x5D850000, with eight FP32 centroid tables and two INT8
projections. Its [causal upsample and decoder transformer](personaplex_mimi_decoder.md)
now keep overlap/KV/offset state in P and place25,616,384 decoder weight bytes at
the start of A. Depth B is unchanged. The remaining Mimi encoder and SEANet
convolution/transposed convolution/PCM path still need lowering and capacity
checks. Temporal/codec A packet switching remains unimplemented. Prefer testing codec packets in A
before temporal execution and after depth execution, when temporal weights are
no longer live. If codec packets/scratch fit that overlay, B can retain the depth
bank and the added per-frame traffic is encoder/decoder packet transfer rather
than a depth reload. If codec must occupy B, depth weights must be reloaded when
B changes back; add up to the full depth bank plus codec traffic. Codec state
must persist independently of either overlay. This map is a plausible paged LM
allocation only. It is not proof that the complete speech model fits4GiB.

## Device chaining contract

1. Create one qualified backend and allocate P/A/B once. Upload fixed tables,
   initial states and program packets. Existing independent Engine wrappers
   must not each upload/scrub their own overlapping images.
2. Device input lookup/sum writes the persistent temporal input. Temporal page
   submissions exchange only persistent addresses, never host activation math.
3. Device final norm/text head stores its greedy text ID in a token slot.
4. Depth step0 reads that ID, selects its projection/text embedding and runs
   the six-layer depth program after shared-residency/integration validation. Its audio head stores the
   greedy audio ID directly for the next step. Steps1–15 use device-selected
   projection/previous-audio embeddings and appropriate per-step weights/state.
5. Update delayed stream queues on device. Add Mimi stages only after their
   compute/state/residency contract exists; final model output must be PCM.

Initially the host may submit each program and predetermined packet transfer;
activation/KV/token exchange stays in shared DRAM. Later use preloaded programs
and `HALT CHAIN` where no packet refill is needed. CHAIN drains stores but resets
temporary registers; crossing state must have explicit persistent locations
(`docs/isa.md`). Preserve native audio0…15/text-last addition,`-1` positive-zero
semantics, direct audio-head input and greedy lowest-ID tie behavior.

Required runtime work: unified region/overlap/alignment/unsigned32 validation;
relocatable descriptors and independent KV; bounded layer-range compilation for
every context bucket; explicit IMEM/TMEM/WAITC budgets; packed-packet hash and
generation identity; an empty/loading/ready/in-use/retired packet state machine;
DMA-complete publication and drained retirement; failure-latched reset/replay;
and transfer accounting split into weights/programs/control/PCM/diagnostics.
Native depth ordering and reset are implemented in the standalone layer; shared
state relocation, complete prompt/delay scheduling and integration remain. Greedy is the explicit final mode; stochastic sampling/RNG is
optional additional scope.

## Correctness, timing and hardware gate

Require paged versus equivalent unpaged arithmetic to match bits for outputs,
KV, logits and IDs across every packet boundary, all16 depth steps, resets and
context exhaustion. Require device greedy to match its own saved logits without
host replacement. Separately report source-reference relative L2/cosine/max error
and token disagreements; implementation equality does not establish quality.
Stop on nonfinite values, stale packet generations or corrupted state.

Do not assume bandwidth or claim realtime. For the synchronous schedule:

```text
T_frame >= temporal_weight_DMA_bytes / measured_H2C_bytes_per_second
           + measured_total_FPGA_compute_seconds
           + measured_program/control/wait_seconds
           + any depth/codec reload_seconds
```

Existing181.392ms/frame temporal compute already misses80ms, with32.9% reference
relative L2 still unaccepted. New CPU image-build time is not FPGA latency.
Combined DMA bandwidth, full depth/codec compute and complete-frame latency have
not been measured. Full speech acceptance requires native-greedy sequence,
prompt/delay and PCM end-to-end comparisons plus explicit quality thresholds.

No new RTL or image requirement is established. First attempt relocation,
partitioning and these supported ISA operations in CPU/ISA. Propose a hardware
change only for demonstrated unsupported semantics, non-partitionable capacity,
addressing outside the qualified mapping or an explicit performance requirement.
That proposal must identify image/build scope and cost and receive authorization
before synthesis or resource changes.

Any eventual physical test requires a coordinated idle window with the UI owner,
normal canonical/legacy locks and fresh exact image/profile/capability/calibration/
idle/drained/error checks. Preserve warm Gemma and wide-port priority. Do not
restore burst200 or infer compatibility from frequency. This planning step used
no hardware and measured no new FPGA performance.

## Bounded implementation sequence

The [resident depth-chain plan](personaplex_depth_chain_plan.md) is the next
concrete software deliverable. Reuse final P/B placements with A empty; combined
listed depth weights/projections/step tables/audio heads/KV total1,614,954,496B
before I/O/masks/programs/alignment. Add a sparse shared allocator and one runner,
alias interstage vectors, and keep counter/token feedback on device. The current
depth program increments the counter; audio selection must use counter−1.
Validate this16-step resident chain before refactoring four temporal packets.
The later depth-only shared runtime now implements P/B relocation and device
feedback, with measured B slack21,495,808B after moving norms to P. Temporal
paging and full-model scheduling are not supplied by that depth-only slice.
