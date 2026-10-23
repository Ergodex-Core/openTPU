# Next implementation: resident PersonaPlex depth chain

This plan follows the six-layer cached-checkpoint validation in
[depth-layer evidence](personaplex_depth_layer.md). It describes software changes
on the existing ISA, not a completed shared runtime, FPGA measurement or speech
endpoint. Keep full speech disabled until complete physical acceptance. No new
image, synthesis or resource is established as necessary.

## First executable scope

Build one resident depth-only chain before refactoring temporal paging:

```text
diagnostic temporal hidden 4096 + text ID (one upload per frame)
  → device-selected input projection + text/previous-audio embedding
  → six depth layers with independent persistent KV
  → device-selected audio head + greedy ID
  → next codebook, using resident counter and previous ID
```

Run sixteen codebooks, then reset the depth state for the next frame. The first
checker may upload captured temporal hidden and a supplied text ID as diagnostic
inputs. Subsequent projection banks, embeddings, head banks and feedback tokens
must be selected on device; the host must not read or calculate them for control.
This is diagnostic input injection, not full model inference. Temporal generation,
Mimi, prompt/cache/delay scheduling and PCM are outside this first chain.

Use the final [P/A/B address map](personaplex_residency_plan.md), leaving A empty
and reserving unused P intervals as holes. This avoids another address migration
when temporal paging is added. Use sparse spans rather than allocating/uploading
all holes as a giant zero-filled image.

| Payload | Bytes | Placement |
|---|---:|---|
| Six depth layers, including shared norms | 1,253,031,936 | B matrices; shared norms may move to P, counted once |
| Sixteen input projections | 67,698,688 | B |
| Zero-prefixed step embeddings | 257,032,192 | P `0x44498000–0x539B8000` |
| Sixteen INT8 audio heads | 33,947,648 | P `0x5B7D1400–0x5D831400` |
| Six independent KV caches | 3,244,032 | Inside P `0x02110000–0x02510000` |
| Sum above | 1,614,954,496 | Before I/O, masks, programs and alignment |

The listed payloads leave 2,680,012,800 bytes of the 4,294,967,296-byte mapping.
That arithmetic does not permit overlapping final reserved intervals. B reserves
1,320,730,624 bytes for depth weights plus projections against its 1,342,177,280
capacity: 21,446,656 bytes remain. Moving shared norms to P reduces B usage;
do not count them twice. Actual allocation must check each aligned span.

Keep control and token IDs inside P `0–0x10000`, including a required 64-byte
FP32 `audio_ids[16]` array written by the device for sequence validation and later
codec input. The feedback slot alone retains only the last ID. Keep programs/WAITC data inside
`0x04000000–0x05000000`, and interstage vectors/scratch in the reserved state
area. Program slots are independent of current IMEM contents. A maximum program
slot uses 131,072 bytes without WAITC or 163,840 bytes with WAITC on this profile;
three such reserved slots use at most 491,520 bytes before slot alignment. Each
compiled program must still fit IMEM/TMEM and its actual WAITC table. Current
standalone counts are conditioning 208, six-layer depth 24,088 and audio-head 168
instruction words versus 32,768; relocation/controller changes require fresh
compilation and are not assumed to keep those counts.

## Required code and ownership changes

1. Add a shared allocation/span manifest. Record base, size, alignment, lifetime,
   immutable payload hash, checkpoint revision and packet generation. Validate
   non-overlap, exact region bounds and unsigned32 addressing; `0x100000000` is
   an end-exclusive boundary, never a stored pointer. Matrix bases retain D-byte
   alignment; packets and program slots use4KiB alignment. Permit explicit aliases
   only for matching interstage vectors/control slots.
2. Refactor descriptor construction in `personaplex_depth_input.py`,
   `personaplex_depth_layer.py` and `personaplex_audio_heads.py` to accept an
   externally allocated layout. Separate payload packing from standalone
   address-zero image creation. Reuse the existing arithmetic kernels, INT8
   packing, padded KV descriptors and Greedy implementation. Do not instantiate
   three independent Engine wrappers on the card or concatenate their images.
3. Alias conditioning output to depth input and depth output to audio-head input.
   Place the depth counter and previous ID contiguously so conditioning can load
   its two-value control vector. Relocate KV/norms/masks to P and all depth matrix
   banks/projections to B. Keep the shared layer counter authoritative.
4. Add a resident head index expression: load `depth_step`, subtract one on device
   and use that value for its RLD bank selection. Preserve the existing depth
   program's end-of-pass increment. Store the greedy token into `previous_id` and
   the required persistent `audio_ids[s]`; do not bounce it through host feedback.
5. Add one shared backend/runner accepting sparse spans and an explicit P program
   base. Current BoardBackend derives its program area after the supplied image;
   independent constructors own uploads/scrubbing and cannot safely share this
   plan. First qualify this runner over IsaTransport, not the physical card.
   A later physical factory must recheck its exact image/profile/capabilities.

## Counter and token contract

At frame start write diagnostic hidden once, `depth_step=0`, and
`previous_id=text_id`. For each fixed host-submitted stage sequence:

| Boundary | Counter | Previous ID | Device operation |
|---|---:|---|---|
| Before conditioning | s | text at s0, audio from s−1 otherwise | Select projection s and ragged embedding row |
| After six depth layers | s+1 | Unchanged | Each layer uses bank/KV/mask s, then one shared increment |
| After audio head | s+1 | audio_s | Select head using counter−1, greedy store for next step |

After final head the counter is16. Reject further work until reset. Reset before
the next frame; masked stale KV must be overwritten before becoming visible.
Failure or interruption latches the runner; no subsequent stage may continue
from partial state. Stop/drain the owned software backend before explicit reset
and replay. Preserve native contextNone/16-step cache semantics, including the
pinned final-step slot0 exclusion. A corrected causal mask is a separate explicit
source/quality decision, not a silent implementation change.

## Correctness and acceptance for this chain

Use small fixtures first, with transfer spying that permits only one diagnostic
hidden/text upload per frame and explicit final diagnostic reads. Require zero
host program arguments, no activation/logit/token feedback for control, and
poison host source weights/tables after preparation. Check bounds and controller
state before execution and fail on nonfinite diagnostic outputs.

Compare the shared chain with equivalent standalone quantized stages: all16
conditioned vectors, depth outputs, per-layer KV, logits and greedy IDs must match
bits, including lowest-ID ties, reset replay, failure/reset, sentinel/padding IDs,
step0/text versus later/audio selection and capacity exhaustion. Use the
standalone arithmetic as the implementation oracle only; separately compare
native checkpoint outputs and greedy sequence, and report numerical/token errors
without replacing device IDs. Full six-layer source-reference agreement does
not imply end-to-end quality acceptance.

After the sparse final layout compiles and passes those checks, run a bounded
cached-checkpoint depth chain on EC2 ISA. Keep raw receipts/arrays and payload
hashes. Do not repeat the full temporal simulation or access hardware for this
software gate. The host orchestration may submit predetermined stages; CHAIN is
an optional later optimization once state persists correctly across program
boundaries and drained stores are established.

## Subsequent integration and remaining gates

Next split temporal `qwen3.Image` / `personaplex_temporal.Image` KV from each
weight block, point all caches/residuals to P, and compile logical layer ranges
for the four A packets and every supported context bucket. The temporal `_step`
currently loops all layers and final-normalizes; apply final norm/text head only
after the last packet. Avoid TokenImage's duplicate/unused inherited head in the
shared allocator. Temporal packet payloads remain829,915,136 /1,318,649,856 /
1,318,649,856 /537,141,248 bytes; estimated temporal weight transfer is
4,004,356,096 bytes per frame, with final norm16,384 bytes once per session.

Initially use drained, synchronous refills. Packet states are empty→loading→
ready→in-use→retired; publish only after DMA completion and retire only after
stores drain. Current driver restricts active-run H2C writes to4KiB and warns of
large concurrent DMA corruption. No overlap or bandwidth claim is justified.

Mimi convolutions/transposed convolutions, codec attention, RVQ and codec state
remain unimplemented/unbudgeted. Prefer codec packets in A while temporal is
idle if capacity and state liveness permit; otherwise account for B/depth
reloads explicitly. Prior temporal32.9% error and181.39ms/frame remain unaccepted.
The depth path uses MX INT8 matrices/activation/KV and ISA FP32 elementwise work;
direct source affine INT4, MX INT4 depth and FP16/BF16/FP32 matrix paths are not
implemented by it. Demonstrate any unsupported semantic/capacity/addressing gap
before proposing new RTL, synthesis or resources.

Physical tests require a separately coordinated idle window, normal ownership
locks and exact current image/profile/capability/idle/drained/error checks. This
plan authorizes none of those actions. Complete speech stays disabled until the
entire codec-to-PCM model path executes on FPGA with accepted correctness and
measured frame latency; host compute fallback is not an acceptable completion.
