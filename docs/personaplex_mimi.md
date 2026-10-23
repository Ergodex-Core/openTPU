# Resident Mimi RVQ latent integration

This slice extends the shared resident depth chain to a continuous 512-channel
Mimi latent. It does not implement waveform decode, microphone encode, PCM,
complete PersonaPlex inference or a physical backend. Full speech stays disabled.
The common target is `f2-hbm:p4rl8a64c150` on the unchanged shared image
`agfi-0001de1dd3df8e21e`; no per-model image reload is permitted.

## Executable boundary

The native server passes `tokens[:,1:9]` after the text token, so this stage
consumes resident `audio_ids[0:8]`. IDs8–15 are not codec inputs for the pinned
eight-codebook server configuration. One semantic codebook and seven acoustic
codebooks have2048 rows of256 FP32 values. Native code computes each centroid
as `embedding_sum / clamp(cluster_usage, min=1e-5)`; frozen normalization is
offline checkpoint preparation. Inference performs lookup, source-ordered
zero-initialized sums in each branch, separate bias-free256→512 projections,
and semantic-plus-acoustic addition in device programs.

Centroids and sums remain FP32. The two projections use offline INT8 weights
and block-scaled INT8 activations, followed by ISA FP32 accumulation/addition.
This differs from the original FP32 projection, so numerical error must be
reported independently of exact lookup/sum agreement and reset parity.

`DepthChainEngine(..., mimi_tables=..., mimi_projections=..., mimi_spec=...)`
preflights the combined allocation and all four programs before lazy payload
reads. The three existing programs still execute each depth step. After all16
steps, `run_mimi()` submits one fixed fourth program, reads no host IDs and
uploads no activation, selected row or weights. `read_mimi(name)` is an explicit
diagnostic boundary; names are `semantic`, `acoustic`, `semantic_projected`,
`acoustic_projected`, and `output`. Reset/begin-frame invalidate Mimi output;
execution failures latch until reset. Legacy no-Mimi behavior is preserved.

The RVQ payload/state occupies the unused persistent guard beginning
`0x5D850000`, below`0x60000000`; it must not consume temporal overlay A or depth
overlay B. The table itself is16,777,216 bytes. Actual alignment, identity guard
table, projection scale/code sizes, output buffers, fourth WAITC program slot
and instruction words are reported by the executed image budget.

Lookup addresses carry RLD bounds. An additional identity-table comparison
rejects fractional/non-finite resident IDs in the ISA simulator before model
outputs are stored. **Current physical RTL ignores RLD F_BOUND.** These are
software validation guards, not hardware range protection. Physical serving
must establish the qualified greedy-producer/ID integrity contract and fault
policy; it cannot infer physical protection from an ISA exception test.

## Checkpoint validation and reproducibility

`tools/personaplex_mimi_check.py` uses the exact common target and active pinned
compiler/ISA hashes. It checks the cached codec header identity and installed
native source hashes, constructs the original32-codebook quantizer, strictly
loads all `quantizer.*` state, then activates8 codebooks. It compares the offline
tables with original native centroids and captures original native branch
projection inputs/outputs during `quantizer.decode`.

The checker runs a captured temporal frame through all16 resident depth steps
and Mimi without interstage reads, compares IDs/final depth vectors/all six KV
caches with the completed common-profile baseline, then checks every RVQ
boundary and full reset replay. Native CPU computations are diagnostic only;
no serving CPU/GPU inference fallback is implemented. Codec identity uses the
audited size/header and hashes of materialized payloads; whole checkpoint tensor
payload integrity is not claimed. CPU wall time is not FPGA latency.

Fresh existing-EC2 job`42e351b8-195c-4aa5-b238-3bac8dd46f31` completed the real
checkpoint comparison: semantic and acoustic lookup/sums are bit-exact against
the original native quantizer. Semantic projection relative L2 is0.0079981741;
acoustic projection0.0178978314; final latent0.0142262403, cosine0.9998988511,
max absolute error0.171115309. All16 generated IDs, final depth vectors and six
KV caches match the preceding same-target baseline. RVQ vectors, IDs and KV
replay bit-exact after reset. Quality remains unaccepted. The executed checker
took96.30s CPU wall time; no physical latency or speech throughput was measured.
From frame start through RVQ completion, the only runtime transfers were two
initial hidden/control writes totaling16,392 bytes and zero reads. Packing and
initial immutable uploads precede that measured transfer scope.

The integrated production layout uses1,632,718,920 named bytes, of which
1,628,774,400 are immutable payload; the highest required aligned extent stays
4,273,471,488 within the4GiB mapping. RVQ occupies17,063,936 bytes plus its
163,840-byte program/WAITC reservation; its immutable payload is17,055,744 bytes.
All four programs use208/24,088/200/864 words, each below32,768. A remains unused
and B slack remains21,495,808 bytes. Six packed RVQ spans are hashed.

Reproducible-build job`36e5eb6a-c71b-4c0d-9f54-cde4bfd61e59` verified the pinned
dependency tree, passed41 aggregate tests in9.61s, and compiled this exact-target
production layout. This includes the earlier26 core/depth tests and10 checker
guards; it is not51 distinct tests. Existing dependencies were NumPy2.1.3,
PyTorch2.4.1+cpu and safetensors0.4.5; Moshi has source hashes but no installed
distribution-version metadata. All execution used one EC2 CPU affinity, one
numerical/preparation thread, nice19 and a physical-open guard reporting0.

The published legacy-runtime branch alone is not the software-qualified shared
runtime. Use the explicit [dependency manifest](../tools/data/personaplex/shared_runtime_dependencies.json)
and standard-library builder, which verify103 pinned runtime source/target
files plus the listed PersonaPlex sources before creating a new isolated tree:

```sh
python tools/package_personaplex_shared_runtime.py \
  --runtime-root /existing/pinned/18fb9780-runtime \
  --source-root /reviewed/personaplex-checkout \
  --output-dir /new/isolated/personaplex-common-runtime
```

This builder installs/downloads nothing, touches no model or hardware, and emits
`PERSONAPLEX_INTEGRATION.json` with complete resulting file hashes. Existing
NumPy, PyTorch, safetensors and the pinned native Moshi source remain separate
dependencies. The checkpoint CLI requires explicit existing model/codec and
completed baseline locations. No readiness is enabled by building this tree.

## Decoder and PCM progress

The optional [causal upsample and decoder transformer](personaplex_mimi_decoder.md)
now consumes the resident RVQ latent without host intermediate transfers. It
implements the learned channelwise 2× upsample and all eight decoder layers,
including source-compatible two-token ring updates, RoPE and reset state. The
actual-checkpoint two-frame comparison and reset replay passed in CPU/ISA on
the exact shared profile. Accumulated RVQ-through-decoder relative L2 error is
0.0270071; decoder-only error on identical RVQ latents is 0.0139889. These are
numerical diagnostics, with no accepted speech quality or physical latency.
The bounded implementation supports 128 code frames before explicit exhaustion.

The optional [SEANet decoder](personaplex_mimi_pcm.md) now lowers all 14
convolutions, ELU and residuals to raw float PCM with persistent causal history
and bias-corrected overlap. Actual-checkpoint two-frame/reset execution passed
in CPU/ISA. Accumulated PCM relative L2 is 0.1602663; the quieter second frame is
0.4062426. State correctness does not establish accepted waveform quality.

Native configuration specifies24kHz and12.5 code frames/s, or1920 PCM samples
per code frame after latent2× upsample and the8×6×5×4 stages. These are source
format rates, not measured FPGA throughput. PCM numerical/audio-quality and
physical acceptance remain open. Temporal paged weights/fixed KV, text/hidden wiring,
input codec, prompt/voice/delay scheduler and complete quality/latency acceptance
also remain. No missing RTL capability has yet been demonstrated by these slices;
unsupported lowering or capacity must be shown before proposing synthesis.

## Physical transport gate

The common pinned runtime lacks legacy `opentpu.host.isaboard`. Its replacement
test must exercise actual `BoardBackend(cfg, images, transport=...)` against an
isolated ISA-backed implementation of the pinned register/memory protocol, or
an already available compatible simulator transport. Required operations are
`mem_write`, `mem_read`, `reg_write`, `reg_read`, `reg_read_many`, `poll` and
ownership-aware `close`. Test upload/program/WAITC placement, prepare/start/wait,
stateful memory retention, indexed reads, failures/timeouts and shutdown; do not
merely mock `BoardBackend.run` return values. This protocol replacement remains
unimplemented and does not follow from direct `IsaBackend` agreement.

`open_transport('f2:...')` opens PCI/locks, and `BoardBackend` initialization
scrubs memory. Neither may be instantiated during these software-only checks.
Physical acceptance additionally requires coordinated ownership, unchanged exact
image/build/clock/CFG/geometry/capabilities, idle/drained/error-free/calibrated
state and a bounded owner-approved window. No service pause is authorized here;
the user’s maximum interruption is30 seconds, and this slice uses zero.
