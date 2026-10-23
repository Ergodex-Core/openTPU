# Resident Mimi learned input downsample

The input codec now appends one argument-free FP32 convolution program to the
resident raw-PCM -> SEANet -> eight-layer input-transformer path. It consumes
two 512-wide rows and emits one 512-wide latent; source kernel [512,512,4],
stride two, groups one, bias false. Initial left history repeats the first
incoming row twice. Subsequent frames retain the previous pair; reset clears
history and restores the resident first-frame flag. No clipping, gain,
normalization, resampling substitute or host activation feedback occurs.

RLD reads a TMEM-loaded FP32 flag and converts it to an integer count; it does
not address the DRAM flag directly. Actual checkpoint/source metadata and the
installed resampler/streaming convolution implementation are pinned. Native
full-input orchestration and RVQ ID acceptance remain separate later gates.

All 24 input programs and 51 joint input/output instruction slots fit the exact
shared f2-hbm:p4rl8a64c150 profile, arch dd6b92a6, image context
agfi-0001de1dd3df8e21e, pinned runtime18fb978033799f0f21258f0e6e085c0360322d96.
Output and encoder-transformer addresses remain unchanged. The new component
occupies 4,370,440 named bytes, weights4,194,304,payload4,196,352, persistent
reset4,104; its program uses312words. Joint named occupancy1,989,298,060B,
highest extent4,273,471,488B within4GiB. Input-state slack17,723,392B,
weight gap993,554,432B. These are software budgets, not measured FPGA speed.

Actual cached checkpoint comparisons on the same two raw saved PCM frames pass
both declared aggregate AND per-frame gates (relative L2<=1e-4, RMS<=1e-5,
peak<=1e-4). Accumulated relative L2 is4.559318761197226e-6,
RMS2.6878344856036493e-7,peak1.3262033462524414e-6. On identical ISA
transformer pairs, downsample-only relative L2 is3.989768683999437e-7,
RMS2.3520704258433836e-8,peak1.1920928955078125e-7. Final raw persistent
state and all output boundaries replay bit-exact after reset; native fresh
streams agree exactly. No implicit interstage backend transfers.

Builder126 tests pass (one missing-Torch native comparison skipped; one older
native-only case excluded) in11.22s. Existing F2 CPU196 tests pass in50.59s;
checkpoint checker457.31306751293596s. Both one-thread/nice19 with zero physical
opens. All165 pinned runtime/overlay hashes match builder/F2/local frozen
source; manifestd04aeaf1d3f4300ce0d85f785c3dc751a2764ca91cf1d2e7137393fd969266df.
The earlier test-message mismatch is preserved in its failed builder receipt;
no numerical acceptance threshold changed. Consumed native arrays and packed
payloads are hashed; no independent whole-checkpoint payload digest claim.

The source append-before-attention/delta<=0 ring defect and128-frame table
limit are inherited without a silent fix. Tiny ISA wrap and24-frame cache tests
were separately qualified; this checker covers two saved raw generated PCM
frames and reset. No microphone/perceptual claim. Remaining input RVQ,
bidirectional temporal paging, prompt/voice/delay scheduling, transport and
physical/quality acceptance keep full speech and serving readiness disabled.
CPU/ISA validation only; no FPGA/image/service/production changes.
