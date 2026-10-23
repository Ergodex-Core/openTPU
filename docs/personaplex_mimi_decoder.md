# Resident Mimi causal upsample and decoder transformer

This increment consumes the resident RVQ latent, applies the learned causal2×
upsampler and all eight decoder-transformer layers. It stops before SEANet and
PCM. It remains CPU/ISA software qualification, with no physical backend or
full-speech enablement. The unchanged deployment context is shared image
`agfi-0001de1dd3df8e21e`, target `f2-hbm:p4rl8a64c150` on pinned18fb9780 runtime.

## Native streaming semantics

The cached checkpoint has a bias-free channelwise ConvTranspose1d with512 groups,
kernel4 and stride2. One512-wide latent produces two512-wide tokens. Save two
overlap samples per channel: add the previous tail to the first two products,
emit them, and retain the last two products. Native `partial=None` on the first
frame skips that addition. Initial/reset overlap uses negative zero as an FP32
additive identity, including signed-zero products. Every operation stays in the
device program; no intermediate latent or overlap is computed on the host.

The transformer receives both tokens in one native call. Each layer appends
both K and V positions before either query attends. With context250 it uses a
250-slot ring, mapped into a256-slot padded physical layout. The remaining six
physical slots are always masked. Do not replace this with two sequential
one-token calls: the second new key overwrites a slot before the first query
attends. Native `delta<=0` cache-position computation also labels the next slot
as the future `end_offset` once full; both queries exclude it. That source defect
is preserved explicitly, including at the first fill and wrap boundaries.

Frozen tables contain both slot indices, source-compatible two-query masks and
RoPE coefficients per frame. The device frame counter selects them. Each frame
uses positions2f and2f+1; only the last decoder layer increments that counter.
Q/K projection rows are permuted offline within each head from adjacent real/
imaginary pairs to real-half/imaginary-half layout, making the existing device
rotate-half primitive implement native adjacent-pair RoPE. V and output weights
retain native order. Actual-checkpoint coefficients are extracted through the
original native RoPE source during offline preparation.

Each layer applies LayerNorm with learned weight+bias and epsilon1e-5, bias-free
QKV/out matrices, per-channel layer scale and residual; then LayerNorm2,
bias-free512→2048 projection, non-gated erf-form GELU,2048→512 projection, second
layer scale and residual. No input/output projection or final norm exists in
this native ProjectedTransformer configuration.

Matrices, activations and KV use INT8; normalization, RoPE, upsample, residuals
and layer scales use ISA FP32. Native GELU uses erf; the ISA lowers the erf form
using an explicit polynomial and EXP2 approximation. It does not substitute the
different tanh-form GELU. A synthetic ISA/native sweep over[-10,10] requires
maximum absolute GELU error below1e-5; actual layer and accumulated errors remain
separate. No exact-FP32 or quality equivalence is claimed from that sweep.

## Residency and lifecycle

The optional decoder requires resident RVQ. `run_decoder()` submits one fixed
upsample program and eight fixed layer programs, with no host arguments, reads
or writes. The final layer output, per-layer diagnostics, all KV, frame counter
and overlap remain resident. `read_decoder`, `read_upsampled`,
`read_decoder_layer` and `read_decoder_cache` are explicit diagnostics.
The caller must finish the decoder frame before beginning another depth frame;
duplicate consumption, unconfigured stages and frame exhaustion are rejected.
Failures latch until reset.

Reset clears offsets and overlap while retaining masked old KV, matching native
reset semantics. All visible outputs must replay exactly. Final raw KV replay
is compared after the same complete written footprint; stale future slots at
an intermediate replay frame need not match their initial allocation contents.
The bounded default is128 code frames/256 decoder tokens. It explicitly rejects
exhaustion rather than silently resetting a conversation or reusing RoPE rows.

Decoder persistent state resides after the RVQ P allocation, below0x60000000.
Its immutable matrices occupy the beginning of overlay A, leaving depth B
unchanged. Production compilation gives4,325,376 decoder KV bytes and25,616,384
weight bytes. Decoder occupied spans total30,316,548 bytes, including25,946,112
immutable bytes. Combined named occupancy is1,664,510,028 bytes; highest aligned
extent remains4,273,471,488 bytes inside4GiB. Programs use208/24,088/200/864 for
depth/RVQ,128 for upsample, seven3352-word layers and one3368-word final layer,
each below32,768. Thirteen WAITC program slots reserve2,129,920 bytes in P.

A is now used by decoder weights. Integrating paged temporal execution later
requires drained retirement and immutable packet switching in A; simultaneous
temporal-and-codec residency is not established. That transport floor remains
unmeasured. No image change or new RTL is demonstrated as necessary here.

## Validation and remaining gates

The synthetic native/ISA suite covers two-token wrap/reset, asymmetric adjacent
RoPE, affine LayerNorm/layer scale, GELU error, bounds and resident transfer
silence. A separate native tokenwise diagnostic must differ at the ring boundary,
so a mistakenly sequential implementation cannot pass unnoticed. Production
capacity compilation uses the exact selected shared profile, not a clock-only
substitution from burst200.

`tools/personaplex_mimi_decoder_check.py` loads the actual cached upsample tensor
and80 decoder tensors strictly, runs captured temporal inputs through resident
depth/RVQ/upsample/decoder, and compares to original native streaming modules.
It measures accumulated error from native RVQ through decoder, and separately
decoder error on identical ISA RVQ latents. The native codec receives the same
generated IDs; this does not establish native LM token-distribution quality.
Reset checks include every output/layer and final depth/decoder raw KV state.
Whole checkpoint payload integrity is not claimed by header/metadata and
materialized-weight hashes. CPU wall time is not FPGA frame latency.

Actual-checkpoint job`bf759534-4c10-4a53-8fb4-dabbee49f503` completed two frames
and their full reset replay on the exact shared profile. Upsample outputs on
identical ISA RVQ latents are bit-exact with native streaming convolution.
Accumulated relative L2 is0.0153765324 at RVQ,0.0159331986 after upsample and
0.0270071256 after all eight decoder layers. Final decoder cosine is0.9996354077
and maximum absolute error0.0565579683. On identical RVQ latents, decoder-only
relative L2 is0.0139889176. Every frame/output/layer and final raw KV of all eight
decoder layers and six depth layers replay bit-exact; native replay also does.
The first frame's IDs/RVQ latent retain exact agreement with the prior receipt.
Quality remains unaccepted. The checker took105.08s CPU wall time; no physical
latency, speech quality or waveform throughput was measured.

Each full resident frame made zero implicit reads and only two initial hidden/
control writes totaling16,392 bytes. Native diagnostics run afterward and never
drive device feedback. All checks used existing EC2, one numerical/preparation
thread and CPU affinity, nice19, offline dependencies and zero physical opens.
The initial decoder suite passed41 cases in5.29s; eleven new checker guards
passed1.26s before the checkpoint run. Reproducible-build job
`3cdb813e-328b-484f-b56d-729a411b29d9` verified103 pinned runtime and29 PersonaPlex
files, passed57 aggregate tests in5.30s and recompiled the production layout.
These aggregate cases include the preceding suites rather than adding new
independent sample counts. No repeated full checkpoint run was needed for the
hash-identical reconstructed tree.

Use the updated pinned [dependency manifest](../tools/data/personaplex/shared_runtime_dependencies.json)
and `tools/package_personaplex_shared_runtime.py` to build a new isolated tree;
the published legacy-runtime branch alone remains insufficient. This step uses
no card/locks/image/service/production actions or new paid resources. Serving
interruption is zero and full speech stays disabled.

The remaining output codec is causal SEANet:512→1024 Conv1d/kernel7, ELU,
transposed-convolution/residual stages with strides8,6,5,4 and kernels16,12,10,8,
then ELU/mono64→1 Conv1d/kernel3. Exact convolution/overlap/trim/reset state and
PCM quality/latency remain unimplemented. Input codec, temporal paging/fixed KV,
prompt/voice/delay scheduler and qualified common-image transport also remain.
Physical acceptance needs coordinated ownership and exact live identity/capability
checks; none were submitted by this software-only increment.
