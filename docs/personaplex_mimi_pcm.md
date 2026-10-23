# Resident SEANet decoder to raw PCM

The optional SEANet stage consumes the resident two-token Mimi decoder output
and generates 1920 mono FP32 samples per code frame. All convolution, ELU,
residual and streaming-state updates execute through fixed ISA programs. This
is software qualification on existing EC2, not physical FPGA acceptance or a
complete PersonaPlex serving path. Full speech remains disabled.

The exact shared target is `f2-hbm:p4rl8a64c150`, architecture hash `dd6b92a6`,
runtime `18fb978033799f0f21258f0e6e085c0360322d96`, image context
`agfi-0001de1dd3df8e21e`. No clock-only or opcode-equivalence substitution is
used. The pinned builder and dependency manifest reconstruct this source tree.

## Source layout and streaming state

The installed native configuration uses one causal mono SEANet decoder, input
dimension 512, 64 base filters, one residual layer per stage, compression 2,
constant zero padding, true identity skips and ELU with alpha 1. The cached
checkpoint contains 28 plain FP32 weight/bias tensors for 14 convolutions.
The inference weights already incorporate training weight normalization; no
runtime weight-normalization parameters or computation are added.

The stages are initial Conv1d 512→1024/kernel 7; four ELU/ConvTranspose/residual
stages 1024→512→256→128→64 with strides 8/6/5/4 and kernels 16/12/10/8; and final
ELU plus Conv1d 64→1/kernel 3. Each residual branch applies ELU, kernel-3
convolution into half as many channels, ELU and kernel-1 convolution, then adds
the original stage input. All 15 top-level source boundaries have explicit
diagnostic output descriptors.

Conv1d retains the last kernel−1 activated input rows. Initial history is zero,
and a frame shorter than its history carries the appropriate old prefix and
new rows forward. Updates copy the complete retained window before writing
its overlapping prefix. Reset restores zero input history.

ConvTranspose packs each kernel phase into matrix rows. Its first stride rows
add the previous raw biased tail minus the channel bias; the final stride rows
become the next tail. A resident `seen` word controls a zero-or-one-iteration
device loop, so the initial `partial=None` case skips overlap entirely. Reset
clears validity and partial state. No host inspection selects this operation.
Within-chunk convolution accumulation order and INT8 quantization differ from
native FP32; measured stage and final PCM errors must therefore remain visible.

Input/weight columns are padded to the actual ISA block size with zeros. Native
Conv1d `[out,in,kernel]` weights are packed as `[out,kernel*in]`; transposed
`[in,out,kernel]` weights become `[kernel*out,in]`. Weights and activations use
INT8; bias, history, overlap, ELU and residuals remain FP32. ELU uses the stable
existing EXP2 primitive with nonpositive exponent inputs. No new opcode or RTL
capability gap has been demonstrated for this slice.

## PCM semantics and lifecycle

The original decoder has no final activation. It returns raw float PCM with
no clipping, peak/RMS normalization, scaling or int16 conversion. The ISA path
preserves that interface, including values outside ±1. Counts outside the unit
interval are diagnostics rather than evidence of an applied clipping step.
The source rates are 24 kHz audio and 12.5 code frames/s. They explain the 1920
samples per frame; they are not measured FPGA throughput or a real-time claim.

`DepthChainEngine(..., pcm_weights=..., pcm_spec=...)` requires resident RVQ and
the Mimi decoder. `run_pcm()` consumes the completed decoder output once, using
14 fixed argument-free programs and no intermediate host reads or writes.
The next frame is rejected until PCM is consumed. `read_pcm`,
`read_pcm_layer(index)` and `read_pcm_state` are explicit diagnostics only.
Failures latch until reset. Decoder, KV and PCM state replay are checked after
the same complete written footprint. The existing 128-code-frame decoder limit
still bounds this composed path.

## Reproduction and remaining gates

Use `tools/package_personaplex_shared_runtime.py` with the updated
`tools/data/personaplex/shared_runtime_dependencies.json`. The builder verifies
103 runtime files and 35 PersonaPlex files before creating an isolated tree.
The previously published old-runtime branch alone is insufficient for the
shared-profile qualification; undocumented overlays are not required here.

The checkpoint checker requires the cached model/codec and completed token,
conditioning, six-layer depth, RVQ and two-frame decoder receipts. It verifies
all baseline hashes, preserves prior ISA decoder outputs, and compares native
RVQ→decoder→SEANet accumulated error separately from native SEANet on identical
ISA decoder tokens. It records per-frame and per-stage numerical errors, raw
PCM statistics, transfer counts, source/payload hashes and complete reset replay.
Native execution is an eager offline CPU reference, never a serving fallback.
Whole checkpoint payload integrity is not claimed by metadata/materialized hashes.

## Measured software result

Checkpoint job `36622204-dc22-4a6a-80eb-b4f0703cce31` generated two consecutive
frames, 3840 raw samples total, and replayed them after reset. All 15 SEANet
boundaries, PCM outputs, final persistent PCM state and six depth/eight decoder
KV regions replayed bit-exact. Native fresh-stream replay also matched exactly.
The previous two-frame ISA IDs, RVQ latents and all decoder outputs remained
byte-identical. Each complete frame had zero implicit reads and only two initial
hidden/control writes totaling 16,392 bytes; diagnostic reads followed execution.

Accumulated native RVQ-through-PCM relative L2 is 0.1602663172, cosine 0.9874448005,
RMSE 0.0005219106 and maximum absolute error 0.0033991961. Per-frame relative L2
is 0.1475103384 and 0.4062426380. Native SEANet on identical ISA decoder tokens
gives relative L2 0.1547284114, with per-frame 0.1421092744 and 0.3937969461.
The second frame is quieter: native accumulated RMS is 0.0007623602 compared
with 0.0045418784 in the first. These numerical differences remain a quality
blocker; no intelligibility, perceptual quality or full-model distribution
acceptance follows from state correctness or this short sample.

ISA raw range is −0.0156734418 to 0.0203784574, with RMS 0.0033037728. All three
ISA/native comparison streams have zero samples outside ±1 in this fixture.
No clipping or scaling was applied. Separate synthetic ±2.5 output tests prove
the implementation preserves out-of-range raw samples rather than clipping them.
On identical decoder inputs, relative L2 is 0.0036202170 after the first Conv1d,
0.0742099065 before the final convolution and 0.1547284114 at PCM. This identifies
the remaining precision bottleneck within SEANet; it does not by itself separate
INT8 quantization from FP32 accumulation-order effects.

The 85-case aggregate suite passed in 8.37s before the checkpoint run. The
checkpoint checker took 120.44s CPU wall time, not FPGA latency. Reproduction
job `7a58b0a1-bde1-400e-9b3d-d2fe86f99ca4` verified the 103+35-file builder output,
passed the same 85 cases in 8.32s and compiled the production layout. These are
overlapping aggregate suites, not additive independent sample counts. All used
one numerical/preparation thread, CPU affinity and nice 19; physical open count
was zero. Initial failures exposed a missing eager diagnostic setting, a device
loop count that incorrectly added one to overlap validity, and one guard-test
exception expectation. All are resolved in the tested source/environment.

SEANet occupies 22,934,944 bytes, including 14,921,736 matrix bytes and 46,612
bytes of persistent reset state. Its 14 programs use 280/344/272/240/344/272/240/
344/272/240/344/272/240/296 words. Combined named occupancy is 1,689,738,732 bytes;
highest aligned extent remains 4,273,471,488 within 4 GiB. All 27 programs fit
32,768-word IMEM individually, and their program/WAITC slots reserve 4,423,680
bytes. B slack remains 21,495,808 bytes. The layout fits this slice without a new
image; it does not establish simultaneous temporal-weight residency.

Remaining full-speech gates are the input codec, unified temporal paging and
resident temporal-to-depth handoff, prompt/voice/delay scheduling, indefinite
streaming, qualified common-image transport, numerical/audio-quality acceptance
and end-to-end physical validation. Overlay A holds decoder and SEANet weights;
temporal weight packets need drained retirement and switching. Concurrent
residency and the transport floor remain unmeasured. This increment uses no
physical card, image reload, service mutation, synthesis or new paid resources.
