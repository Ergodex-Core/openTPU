# PersonaPlex input embeddings and audio heads

These are separate device component programs, not an end-to-end speech engine.
They use the existing indexed-load, FP32 vector, matrix and greedy ISA operations;
no new RTL, synthesis, model download or physical run is part of this change.
The complete speech path remains unavailable in a strict FPGA-only UI.

## Implemented boundary

`EmbeddingEngine.step(codes)` accepts text ID followed by the 16 audio IDs.
Resident FP32 tables preserve the source values. The device adds audio0 through
15 in order, then text last, matching native `LMModel.embed_codes`. Every table
has a prepended positive-zero vector so the native `-1` zero ID is selected on
device. Original IDs, including the extra initial/padding row, map to ID+1.
The runtime uploads only the 17 IDs and does not return activations implicitly.
`step_voice(index)` selects a resident stored voice embedding on device; this is
replay of a previously captured embedding, not generation of a voice prompt.
Explicit diagnostic reads are separate. Prompt-token cache restoration, delayed
stream scheduling and transfer into temporal-engine memory are not implemented.

`AudioHeadEngine.step(hidden, codebook)` accepts the native depth-transformer
output and a codebook index. A single compiled program selects that resident
INT8 head with device RLD, projects directly without another normalization, and
returns its four-byte greedy ID. All 16 heads remain resident in the standalone
image. The host does not select a head, compute logits, or replace the device
ID with a host argmax. Diagnostic logits are available explicitly. This stage
requires the caller to supply a depth hidden vector; the depth transformer is
not implemented here. Native stochastic sampling is unsupported; greedy mode is
explicit and does not require device RNG.

Both wrappers reject direct `backend="board"` before checkpoint reads. A physical
adapter must provide the measured config and a factory that qualifies the exact
image/profile, CAP_GEN, state and ownership before upload. Runtime exceptions
latch a failure until reset. Tests exercise actual BoardBackend protocol over
IsaTransport; that is CPU simulation, not physical hardware evidence.

## Production capacity on the recorded burst200 profile

| Standalone component | Required bytes including program area | Instruction words |
|---|---:|---:|
| 17 input tables plus four stored voice frames | 1,061,965,824 | codes704; voice56 |
| 16 INT8 audio heads | 34,127,872 | 168 |
| Existing temporal/token slice, cap128 mixed precision | 4,173,934,592 | temporal25,240; text head184 |
| Later depth input projection plus step embeddings | 324,923,392 | 208 |

Each program fits the recorded32,768-word IMEM. Input-table payload alone is
1,061,715,968 bytes; four voice frames add65,536 bytes. Audio-head payload is
33,947,648 bytes. These are measured compiler/layout results, not timing estimates.
The prior three independent images total5,270,028,288 bytes versus4,294,967,296
mapped bytes; adding standalone depth conditioning gives5,594,951,680 bytes.
These sums double-count workspaces and are not lower bounds, but the
input tables alone exceed the temporal/token image's121,032,704-byte free space.
No unified allocator, phased residency scheme or device stage chaining is
implemented. Capacity must be resolved before calling this a combined model.

## Reproducible validation

All execution is EC2-only with existing dependencies, one CPU thread, nice19,
CPU affinity, offline model access, headroom checks and a physical-open guard.
The initial aggregate exposed one real streamed-weight alignment defect for a
small vocabulary at D128:337 passed,1 failed. The corrected aggregate passed343
in20.20 seconds, physical opens0. A final targeted receipt covers the checker
artifact ordering and native-input wording without repeating unaffected suites.
It passed36 tests in1.62 seconds, physical opens0. Aggregate receipt:
`4031dc2d-53da-4e21-b139-213a34f70d00`; final targeted receipt:
`f766ef42-4860-41fd-88f4-9141bfca4044`.

The bounded actual-checkpoint checker is:

```sh
python tools/personaplex_components_check.py \
  --model-dir /existing/pinned/model \
  --voice-embeddings /existing/four-frame/input_embeddings.npy \
  --output-dir /new/isolated/components-check
```

It requires a new output directory and the pinned existing revision
`8c686347007de51d9017706e5543d6f1757ecf5d`. It checks five input-ID cases against
native `ScaledEmbedding`/`LMModel.embed_codes`, including initial IDs, mixed IDs,
all-zero and partial-zero sentinels; four stored voice rows; all16 audio heads
in nonsequential order on explicitly synthetic1024-wide inputs; and reset replay.
Embedding/voice correctness requires exact FP32 bits. Audio device greedy must
match saved device logits. Original source-head FP32 results are separate
numerical diagnostics; disagreement is retained, not treated as a failed device
selection or silently repaired. Audio input/output rows share an explicit
codebook order. The checker saves arrays, source hashes, capacity, precision and
reset receipts, with hardware/full-model/speech/quality flags false. Checkpoint
identity is revision metadata plus size/mtime, not exhaustive payload hashing.

Actual cached-checkpoint job `975204c3-89f5-42e9-9f23-1dd0b12ffafa` completed
exit0 in9.60 seconds of process wall time (checker7.71 seconds). All five input
cases and four voice rows matched native FP32 bits exactly. Reset replay was
bit-exact for embeddings, voice and audio logits/IDs. All16 source-head top-1
IDs matched on identical synthetic hidden inputs; aggregate logit relativeL2
was0.010807183703853139, cosine0.999941627274997. This is additional head-only
quantization error, not full-model output error. The guard recorded0 physical
transport opens. Image builds took4.70 seconds for embeddings and2.51 seconds
for the audio bank; these are CPU/ISA construction times, not FPGA timing.

## Next implementation gate

The later [depth conditioning slice](personaplex_depth_input.md) now implements
device-selected `depformer_in[codebook]` (or the shared projection when configured),
4096→1024 projection and text/previous-audio embedding addition. Its source
references isolate projection quantization from exact embedding addition.
The next compute gate is a bounded depth-transformer layer/state path, followed
by all six layers with verified per-step banks and native context/window/reset
semantics. The [unified residency plan](personaplex_residency_plan.md) supplies a
candidate fixed-state/paged-weight map and explicit host transport budget.

The header-only checkpoint inventory found16 input projections
`[1024,4096]` (268,435,456 expanded FP32 bytes), one text and15 audio step
embedding tables (256,966,656 FP32 bytes), and216 depth-transformer tensors
(4,932,550,656 expanded FP32 bytes). These are validated metadata counts, not
device-layout estimates or a quantized lower bound. Even the next standalone
conditioning slice needs an explicit packed-bank and residency budget. No
depth program has been compiled, and no fit or numerical correctness is implied.

Mimi streaming convolution/transposed convolution, encoder/decoder attention,
RVQ/codebooks and codec state still need a bounded lowering/capacity study; no
claim that these fit existing ISA or require new RTL has been established.
Complete prompt/delay scheduling, unified residency, component chaining and PCM
end-to-end validation remain. Prior temporal precision error32.9% and181.39ms/frame
are still unaccepted; new component success cannot establish speech quality.

The UI owns the current hardware window and is qualifying a different150MHz
wide-port image. Do not restore burst200 or assume compatibility from clock.
The next physical component check requires a coordinated idle window, normal
canonical and legacy locks, exact image/profile/capability qualification and
matching capacity tests. No physical execution or hardware performance is claimed
for these new components.
