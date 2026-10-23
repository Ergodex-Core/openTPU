# Resident Mimi input codec to eight audio IDs

This increment chains raw mono24kHz FP32 PCM through fourteen SEANet encoder,
nine encoder-transformer, one learned downsample and nine input-RVQ programs.
Every80ms frame contains1920samples and produces one eight-codebook audio ID
vector. Programs are argument-free; no model-compute host fallback or implicit
interstage backend transfer is allowed. Output-codec reservations remain but
its programs do not execute in this input qualification.

The semantic and acoustic branches each project the same512-wide latent to256
values. Semantic codebook0 and acoustic codebooks1–7 select source-order nearest
normalized centroids, then subtract the selected centroid from their independent
branch residual. Tables are the exact existing normalized shared `mimi_tables`
span; no duplicate16MiB table is added. Device selection uses ARGMAX of negative
FP32 squared distances, with first minimum ID across ordered32-row tiles.
The64-row draft exhausted TMEM (49,537 live plus16,384 requested >65,472 usable);
32-row tiling compiles without changing the image. Source torch.cdist p=2 has
different rounding. Exact native IDs, source winner margins and distance-error
metrics are separate evidence; ISA arithmetic agreement alone cannot prove
native nearest-token correctness.

Each search writes resident [8,256] before-residual, selected-centroid and
after-residual snapshots. All distances and snapshots are observed only after
the complete frame; no host selection or activation feedback is used. Semantic
and acoustic projection, residual and centroid order, exact IDs, source-byte
provenance, streaming/reset and range checks are explicit gates. Both accumulated
and same-ISA-downsample-latent references use declared float limits relativeL2
1e-4,RMS1e-5,peak1e-4, aggregate AND each frame, plus exact eight IDs per frame.

The native full-input helper checks original streamed MimiModel.encode with
its CPU-disabled CUDAGraphed wrapper, strict-loaded constructed downsample,
original KV positions and logical/fresh reset against the native component
chain. Its Identity output decoder and unused constructor upsampler are
explicitly outside output-codec qualification. The helper does not change ring
positions or interleave encode_to_latent calls that would double-advance state.

Exact shared profile f2-hbm:p4rl8a64c150, archdd6b92a6, image context
agfi-0001de1dd3df8e21e, runtime18fb978033799f0f21258f0e6e085c0360322d96.
Pinned dependency manifest/builder required; old runtime branch alone is not
qualified. RVQ adds2,617,380named bytes including9instruction slots; matrix
payload1,048,576bytes,persistent reset94,244bytes. Programwords240 then8x3272,
each below32768IMEM. Jointnamed1,991,915,440B,highest4,273,471,488B within4GiB;
60slots disjoint,inputstate slack17,625,088B,weightgap992,505,856B. No physical
performance or real-time claim follows from these software budgets.

Actual-checkpoint two-frame qualification passes. All16device IDs match both
the full native raw-PCM path and native quantization on identical ISA downsample
latents, independently for each frame. Original streamed MimiModel.encode
matches component boundaries/IDs/positions bit-exact, including logical/fresh
reset. All resident outputs, snapshots and raw history/KV/counters replay exactly
after reset; no interstage backend transfers. Both accumulated and same-input
projection/residual float gates pass aggregate AND each frame.

Accumulated branch-projection relative L2=3.1889152738303467e-06,
RMS=1.7379391140370483e-06,peak=9.655952453613281e-06.
On identical downsample latents, projection relative L2=1.6418540148069266e-07,
RMS=8.948003793999618e-08,peak=3.5762786865234375e-07.
Minimum native Euclidean winner margin=0.002774745225906372;
maximum sqrt(device FP32 squared distance)-native distance error on identical
latents=1.0907649993896484e-05.
This evidence is limited to the saved two-frame fixture.

Preferred builder149 tests pass in11.34s (one missing-native dependency skip,
one existing native-only case excluded). F2 CPU219 tests pass in52.57s;
checkpoint checker465.92758972290903s. All one-thread/nice19,physicalopens0.
All172runtime/overlay hashes match builder,F2 and frozen source, manifest
`aa3e8ab4978be4d85ae8113b8528c5d0f63e1737b8a0256f4f01b92fd30e7951`.
The two stopped native preflights (source-hash typo and spec-field guard error)
are preserved; neither started ISA inference. Both fixes have regression guards.
Consumed payload/header/source hashes are recorded; whole-checkpoint payload
independent authentication flag staysfalse. 60 diagnostic arrays remain EC2.


The source append-before-attention/delta<=0 ring defect remains explicit and
unmodified. Separate tiny ISA wrap/context40 padded64 24-frame and native-only
128-frame context250 probes were previously qualified; this checkpoint chain
uses two saved generated PCM frames and reset, not microphone/perceptual speech.
Bidirectional codec/LM chaining, temporal weight paging, prompt/voice/delay
scheduling, supported transport and coordinated physical/quality acceptance
remain gates. Full speech and serving readiness stay disabled. No FPGA,
image, service, production or paid-resource changes in this increment.

The older four-packet temporal A-region refill is incompatible with this codec
layout and would overwrite live state/weights. A source-derived five-packet
candidate fits only nominally in the992,505,856-byte post-codec gap; its largest
packet leaves3,518,464bytes before allocator/alignment verification. Cap128
temporal KV needs34,603,008bytes and must use the separate reserved P region,
not the17,625,088-byte codec slack. Exact incoming-eight-ID duplex destinations
and native delay/supplied-token scheduling remain unverified. These are next
integration gates, not authorization to run or reload hardware.
