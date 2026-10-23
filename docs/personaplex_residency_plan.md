# Candidate unified PersonaPlex residency and device chaining

This is a compiler/runtime design, not a demonstrated full-model fit or a
hardware result. No image change, synthesis, card access or paid resource is
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
A possible padded depth layout with six layers,16 heads,128 physical positions
and64-wide heads padded to128 uses3,244,032 bytes. This is a reservation, not a
verified implementation of native depth context8/window/reset semantics. Codec
state, prompt-cache capacity and complete activation liveness remain unverified.

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

Integer estimates from actual checkpoint tensor dimensions, `quant.row_bytes`,
`quant.scale_bytes` and `kernels/mlp._chunk` give1,253,031,936 bytes for all16
steps of the six-layer depth transformer at INT8, including norms and repeated
chunk scales. Sixteen4096→1024 input projections add67,698,688 bytes; that payload
has now been materialized and CPU/ISA-validated in the standalone depth-input
stage. The transformer-bank estimate remains unmaterialized. B would
hold1,320,730,624 bytes with21,446,656 bytes of slack before verified alignment,
materialized quantization, compiler program area and final activation accounting.
The step embeddings already reside in P. The standalone depth-input component
now provides a way to test the projection and ragged embedding semantics; it
does not establish that the complete depth bank or program fits.

Mimi remains unimplemented and unbudgeted: streaming convolution/transposed
convolution, encoder/decoder attention, RVQ/codebook operations and streaming
state all need lowering and capacity checks. Prefer testing codec packets in A
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
   the six-layer depth program once implemented. Its audio head stores the
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
Native depth ordering, cache restoration and prompt/delay scheduling still need
implementation. Greedy is the explicit final mode; stochastic sampling/RNG is
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
