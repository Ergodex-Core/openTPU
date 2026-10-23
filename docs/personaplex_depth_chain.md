# Shared resident PersonaPlex depth chain

`DepthChainEngine` implements one diagnostic depth pass on the ISA simulator:
selected temporal-to-depth projection and previous-token embedding, all six
depth layers with independent KV, then selected audio head and greedy ID. It
produces sixteen codebook IDs. It does not implement temporal inference, prompt
and delay scheduling, Mimi/RVQ or PCM speech, and is not a serving fallback.
Physical backends are refused before source payload access.

## Resident state and execution

One backend owns the final sparse P/B layout. A remains unused. Immutable source
weights are converted offline through the established component packers, hashed
by named packed span and uploaded once. Temporary component packing buffers are
local preparation data; they do not become separate execution backends. Sparse
spans exclude unused map holes. The ISA simulator maps4GiB virtual DRAM; actual
named occupied spans are reported separately. No physical transfer or bandwidth
is measured.

`begin_frame(hidden, text_id)` uploads a diagnostic4096-wide hidden vector and
an8-byte control vector once. Each `step()` submits three fixed programs and
performs no implicit host read or write:

1. Conditioning reads device counter s and previous ID. It selects projection s
   and the text table at s0, otherwise the previous-audio embedding table.
2. The depth program reads that conditioned vector, uses each layer's bank/KV/
   mask s and increments the counter once after all six layers.
3. The head reads counter−1, selects that audio projection, computes greedy ID,
   and stores it into both the feedback slot and persistent `audio_ids[s]`.

Aliased descriptors connect interstage vectors; the host never uploads an
intermediate activation, selected weight bank, per-step index or feedback ID.
`run_frame()` submits all16 steps. `read_ids()` is an explicit diagnostic read
available only after the entire pass. Other diagnostic methods expose projected,
conditioned and depth vectors, logits, counter, feedback ID and per-layer cache.
None is needed to control subsequent execution.

Reset invalidates output readiness and writes only the control vector. All16 ID
slots are overwritten before they become readable. Old KV remains masked and is
overwritten before visibility. A failure latches the runner until explicit reset;
no following stage continues from partial state. The host counter mirror guards
the public bounded API; device state selects all model banks and cache addresses.

Native context8 is overridden to None, so the actual pass has16 steps. Preserve
its pinned final-step cache bug: positions `[16,1,2,...,15]` exclude slot0 on the
last query. An explicit causal diagnostic policy is available in the layer spec;
it is not described as native parity. No silent cache correction was made.

## Measured production-profile compilation

All addresses are unsigned32, with end-exclusive region checks and explicit
alignment/non-overlap validation before lazy source materialization.

| Item | Measured value |
|---|---:|
| Mapped DRAM | 4,294,967,296 bytes |
| Highest required aligned extent | 4,273,471,488 bytes |
| Sum of named occupied spans, including state/program reservations | 1,615,491,144 bytes |
| Immutable uploaded payload spans, including masks/norms | 1,611,718,656 bytes |
| Six independent KV caches | 3,244,032 bytes |
| B used by depth matrix banks and input projections | 1,320,681,472 bytes |
| B remaining | 21,495,808 bytes |
| Conditioning / depth / feedback head instructions | 208 /24,088 /200 words |
| Per-program IMEM capacity | 32,768 words |
| Reserved P program/WAITC slots | 3×163,840 bytes |

Shared norms are in P, counted once; this increases B slack relative to the
plan's conservative norm-in-B total. Step embeddings stay at the final P
embedding interval, audio heads at the final P head interval, and each KV cache
inside the reserved P depth-KV region. The head's extra instructions calculate
counter−1 and store the full ID sequence. Program slots reserve future physical
placement; the ISA backend currently loads each fixed program into simulated
IMEM when submitted. Board program upload/chaining is not qualified here.

No instruction, DRAM or unsupported-kernel gap has been demonstrated for this
depth-only INT8 path. Source affine INT4 is converted offline to MX INT8. Matrix
activations and KV are INT8 with scales; elementwise norm/SiLU/softmax/residual
operations use ISA FP32/FTZ. Direct affine INT4, MX INT4 depth and FP16/BF16/FP32
matrix execution remain outside this component.

## Software validation

EC2 command `cd830465-ef11-4f8a-aae1-988fc93dee5c` passed38 tests in2.85 seconds,
physical opens0. It includes the new shared-chain integration and existing audio
head regressions after the reusable head kernel gained an optional device index
and token return. Every projected/conditioned/depth/logit vector, generated ID
and raw per-layer KV matches equivalent independent standalone ISA engines.
A constructed feedback case generates2,0,3,1 using distinct token rows and head
banks. Poisoned host payloads and transfer spies verify source independence,
exactly two frame-start writes and zero host transfers during all steps.
Layout/program preflight, lazy preparation and failure/reset guards also pass.

Native generated-sequence/teacher-forcing helper tests passed3 in1.66 seconds,
`fafba018-64b5-4eaa-ab01-e24c2a3a6b34`, physical opens0. They execute the installed
native layer/container and source input-preparation path on a tiny fixture,
including final-step cache positions. All model arithmetic/tests run on existing
EC2, one thread, nice19; none run on the development Mac.

The bounded checkpoint checker selects one of the four saved temporal hidden/text
inputs using `--frame-index` (default0). It compares the ISA's own-feedback ID sequence with original-source
native FP32 greedy generation. A second native pass is forced with the ISA's
previous IDs to distinguish arithmetic error from autoregressive drift. ISA IDs
must match independent argmax of their own saved logits; no native replacement
or host feedback is allowed. Reset must reproduce every intermediate vector,
logit, all16 IDs and all six raw caches exactly. Source/baseline/revision/file
metadata guards and raw array hashes remain in the isolated receipts.

## Real-checkpoint generated sequences

Two cached temporal inputs completed with the shared resident chain:

| Saved frame | EC2 command | ISA/native greedy IDs | Output, IDs and six KV reset |
|---|---|---|---|
| 0 | `a9d70cfc-9059-409b-9053-7ac22fbe32a0` | 16/16 exact | All bit-exact |
| 1 | `2cfd097a-744b-4d0f-bcd8-9af745f2fd83` | 16/16 exact | All bit-exact |

The distinct native own-feedback sequences and native teacher-forced sequences
both match the ISA's16 device-generated IDs for each input. The ISA IDs also
match independent lowest-ID argmax of its own saved logits. No native IDs or
host argmax were substituted into the execution feedback path. All projected,
conditioned, depth and logit vectors replay exactly after reset, along with the
required ID array and raw bytes of all six KV caches. Native cache histories
retain the pinned final-step slot0 exclusion.

| Input | Depth relative L2 | Logit relative L2 | CPU process/checker wall time |
|---|---:|---:|---|
| Frame0 | 0.02337168964330375 | 0.020083488708821217 | 84.94s /82.33s |
| Frame1 | 0.020603500158555263 | 0.01866185786737739 | 82.73s /80.12s |

Errors compare identical diagnostic hidden/text inputs with original
dequantized-source FP32 native computations. The reference is not an original
unquantized NVIDIA checkpoint. CPU times include source preparation, conversion
and simulation; they are not FPGA latency or audio frames/s. Both runs report0
physical opens. These two examples do not establish speech quality or acceptance.

Frame0's first receipt retained the preflight layout's empty packed-hash field,
although materialized source payload hashes were recorded. The checker now
refreshes the budget from the executed image. Frame1 records24 packed immutable
span SHA-256 values. The saved-frame selector and corrected checker passed7
helper cases in1.59s (`38d7f46d-34ce-4cff-9541-99e6d2910844`); the core38-test
integration result remains unchanged. Checkpoint revision, file metadata,
native source hashes and all ancestor baseline hashes remained unchanged.
Raw arrays stay in isolated EC2 `depth-chain-isa-001` /`depth-chain-isa-002`;
small JSON/log receipts and artifact hashes are in the task handoff.

## Required common image qualification

The deployment target is now the SAME shared Gemma image
`agfi-0001de1dd3df8e21e`, profile `f2-hbm:p4rl8a64c150`, for every model.
The compilation and measurements above target the recorded burst200 profile;
they do not qualify the shared image. Do not infer opcode/capability or numeric
compatibility from a clock or variant name. No per-model reload is permitted.

Before physical acceptance, obtain that exact target definition and saved
architecture/capability/numerics evidence, compile and run the software checks
against it, and verify required RLD/ARGMAX, INT8 MX, STREAM and FP32 operations,
unsigned32 mapping, IMEM/TMEM/ACT limits and padded KV behavior. A later qualified
backend must preserve the shared image and recheck actual identity/ownership.
No live probing, serving pause or reload is authorized by these software checks.
The user's30-second serving-interruption maximum is preserved with zero
interruption here. Missing evidence is a qualification gate, not proof of a
missing hardware capability. Full speech remains disabled.

## Remaining completion gates

The [implementation plan](personaplex_depth_chain_plan.md) now has a first
software-only chain implementation; its temporal paging and full-model steps
remain pending. Next validate further diagnostic frames/inputs only where they
resolve a concrete correctness concern, then separate temporal fixed KV from
paged A weights and connect the resident text/hidden outputs directly.
Prompt/cache/delay scheduling and Mimi encode/decode, RVQ, codec state and PCM
remain unimplemented/unbudgeted. Prior temporal32.9% source error and181.39ms/frame
are unaccepted; this component does not establish speech quality or realtime.

Full speech stays disabled. Any physical qualification requires separately
coordinated ownership and exact current image/profile/capability checks. This
software task opens no card, changes no lock/image/service, starts no paid
resource or synthesis, and enables no CPU/GPU model-compute fallback in serving.
