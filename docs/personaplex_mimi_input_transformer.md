# Resident input encoder and transformer

This increment follows the independently validated input SEANet encoder at
`2e62bb38`. Raw FP32 PCM runs through fourteen encoder programs and nine
input-transformer programs with no implicit host activation reads or writes
between them. It implements neither learned downsampling nor input RVQ encoding.
Full speech and serving readiness remain disabled.

The first transformer program copies the encoder's two resident 512-dimensional
rows into its input snapshot and transformer state. The next eight programs
reuse the validated FP32 Mimi layer kernels with the encoder checkpoint weights.
The decoder upsampler does not execute. The reused allocator retains 8,192 bytes
of explicitly zero, unused upsample compatibility storage. That storage is
included in budgets and is not claimed as an encoder checkpoint payload.

The strict native contract pins exactly 80 encoder-transformer tensors,
100,761,600 bytes, their names/shapes/dtypes/offsets, constructor and installed
source. Dimensions are 512, eight heads/layers, FFN 2,048, context 250, LayerNorm,
RoPE period 10,000 and causal two-token streaming. Coefficients are packed offline;
matrices, activations, K/VT caches, masks, frame counters and
transformer computation execute in resident programs. No encoder pair is scaled,
normalized, resampled or computed by a host fallback.

The inherited native two-token append-before-attention and `delta<=0` ring
position defect are retained explicitly. Logical reset resets the counter and
encoder convolution history while retaining masked transformer KV bytes. Tiny
ISA tests exercise context 4 wrap and 24 frames with context 40 padded to 64,
two attention tiles, source-equation layers/KV and bit-exact reset replay.
The checkpoint checker separately captures native positions for 128 repeated
input pairs across the real context-250 boundary. That probe is native-only;
it does not establish 128-frame checkpoint ISA acceptance. The existing
128-frame coefficient-table limit and the unqualified context-2 all-masked
boundary remain explicit limitations.

The exact shared profile is `f2-hbm:p4rl8a64c150`, architecture hash `dd6b92a6`,
image context `agfi-0001de1dd3df8e21e`, runtime
`18fb978033799f0f21258f0e6e085c0360322d96`. Use the pinned reproducible builder.
All output and encoder addresses remain unchanged. Transformer state appends
inside the existing 32 MiB input reservation and matrices follow encoder weights.
All 50 resident instruction slots are disjoint within the existing program region.

Transformer named occupancy including instruction slots is 111,335,428 bytes,
matrices/constant vectors 100,769,792 bytes, immutable payload 101,099,520 bytes,
raw KV 8,388,608 bytes. Its program sizes are 32 words for the pair copy,
seven times 21,728 and a final 21,744, each below 32,768-word IMEM. Joint named
occupancy is 1,984,927,620 bytes; highest mapped extent remains 4,273,471,488
within 4 GiB. Input-state slack is 17,739,776 bytes and weight gap before depth
matrices is 997,748,736 bytes. These budgets do not measure FPGA performance.

The checkpoint checker consumes the same two saved raw PCM frames and cached
codec as the encoder increment, executes the resident 14+9 chain, persists
outputs/layers/raw state before reset, and checks identical input snapshots,
reset replay, native fresh streams and all layer errors. Accumulated raw-PCM
to transformer-output agreement and transformer-only agreement on identical
saved device encoder pairs have separate gates. Both require relative L2 <=1e-4,
RMS error <=1e-5 and peak error <=1e-4 in aggregate AND each frame. No threshold
or waveform adjustment is applied to hide errors.

Actual-checkpoint results on the saved two-frame fixture pass both declared gates.
The raw-PCM-to-transformer aggregate relative L2 is 3.759307080857217e-6,
RMS error 2.182059189570653e-8, and peak error 4.470348358154297e-7.
On identical saved ISA encoder pairs, transformer-only relative L2 is
5.987775967242368e-7, RMS 3.4755571489183533e-9, and peak
1.1920928955078125e-7. Both frames independently pass all three limits.
All output snapshots and final raw history/KV/counter bytes replay bit-exact
after reset, and native fresh streams are bit-exact. The native-only
128-frame context-250 position probe passes; it remains separate from
the two-frame real-checkpoint ISA comparison.

The preferred builder passed 108 regressions (one native-only case excluded)
in 11.04 seconds. Existing F2 CPU passed 177 regressions in 50.48 seconds;
the checkpoint checker took 459.19779580796603 seconds. Both used one CPU,
nice 19 and recorded zero physical-device open attempts. All 159 runtime
and overlay files match the frozen builder, EC2 and repository source map;
manifest SHA256 is `4c8b9a3022e2d5f6b5d0f453102ced4838e033492633273dd9840437b05e5fc3`.
Consumed arrays and packed payloads are hashed; the whole-checkpoint
payload is not independently authenticated (the corresponding receipt flag
remains false). Thirty-four diagnostic arrays remain on EC2.


All numerical execution is on existing EC2 CPU, one thread and nice 19. The
preferred builder runs existing NumPy/ISA tests and capacity compilation; native
checkpoint comparison uses the F2 Torch/Moshi cache. Mac is a Git/source/SSH/AWS
and stdlib AST/hash bridge. No installs, new resources, synthesis, FPGA/image/
service/production changes or full-speech enablement occur here. Remaining input
components are source-replicate learned downsampling and semantic/acoustic RVQ
encoding, followed by bidirectional temporal scheduling, transport and coordinated
physical and quality acceptance.
