# SEANet precision isolation on the shared profile

This increment replays saved two-frame decoder output through SEANet. It does
not rerun the language model, use a physical FPGA, enable speech serving, or
establish perceptual quality. The original INT8 path remains the default.
The preserved baseline from `18567fef` remains unchanged.

## Isolated comparison and precision options

`MimiPCMSpec(final_precision="fp32", residual_precision="fp32")` enables FP32
weights and activations for the final mono convolution and all eight
convolutions in the four residual blocks. Other convolution matrices remain
INT8. Bias, ELU, residual addition, causal history and ConvTranspose overlap
remain FP32 with the original source semantics. No PCM clipping, gain,
normalization, resampling or int16 conversion is applied.

The FP32 path uses the existing VPU multiply/reduction primitive. Each output
channel loads one padded coefficient row, reduces the tap-major input window,
adds bias and, where applicable, the true identity skip. A device loop selects
coefficient rows. The original INT8 packing and execution path are unchanged.
No new opcode, RTL or image is needed for this software/ISA slice. Its physical
performance remains unmeasured; replacing matrix operations with channel-wise
VPU reductions can be substantially slower.

`tools/personaplex_pcm_precision_check.py` requires the cached codec and the
preserved completed PCM receipt. It verifies the receipt, all saved arrays,
ancestry, runtime/profile sources and materialized source weights. Each native
SEANet module receives the identical saved input from the preceding ISA stage,
once per frame in a fresh native stream. This separates local operator error
from accumulated upstream error while retaining causal state across frames.
Native fresh-stream repetition must match exactly.

The checker compares the unchanged INT8 path, an FP32 final head, and FP32
residuals plus final head. All unchanged prefixes must match historical arrays.
Each candidate must replay all stage outputs, PCM and persistent state after
reset, with zero implicit interstage host reads/writes. Changed residual
activations are expected to change downstream state relative to INT8; they must
still replay exactly within that candidate. Diagnostic reads occur afterward.

## Waveform metrics

`tools/personaplex_pcm_metrics.py` accepts finite FP32 `[frames,samples]` arrays
and reports aggregate and per-frame RMS error, peak absolute error, mean/DC
error, SNR and signal/error RMS dBFS. Full scale is the raw value 1. Relative L2
is RMS error divided by reference RMS. SNR is 20 log10(reference RMS/error RMS);
error dBFS is 20 log10(error RMS). Undefined zero-reference/error values become
JSON null with explicit flags. A raw RMS gain ratio is diagnostic only.

Quiet frames need absolute waveform metrics alongside relative L2: the
historical second frame has reference RMS 0.00076236. Its 40.62% relative error
corresponds to RMS error 0.0003097032, SNR 7.8243 dB and error RMS -70.1811 dBFS.
These measurements do not establish intelligibility or listening quality.

## Measured software evidence

The first bounded remedy isolated the final convolution: its local relative L2
fell from 0.0158234963 to 0.0000003579264 and RMS error from 0.0000523000 to
0.000000001183. Accumulated native RVQ-through-PCM relative L2 improved only
0.1602663172 to 0.1595941602. The final head is therefore not the main remaining
source of waveform error. Forced-input residual-block relative L2 was
0.0339981, 0.0521152, 0.0408372 and 0.0233801. ELU local error was negligible
by comparison. This evidence motivated one bounded residual-FP32 candidate.

Checkpoint job `0848de9a-7e72-4b96-906e-a3d0178f8a8f` completed the three
candidates on two consecutive frames (3840 raw PCM samples) and reset replay.
The final-plus-residual FP32 candidate reduced accumulated relative L2 to
0.0944946988, RMS error to 0.0003077239 and peak absolute error to 0.0018251020.
SNR rose from 15.9032 to 20.4919 dB; error RMS fell from -65.6481 to
-70.2368 dBFS. On identical ISA decoder inputs, relative L2 fell from
0.1547284114 to 0.0779104617.

On the quieter second frame, accumulated relative L2 fell from 0.4062426380 to
0.1285265673. RMS error fell from 0.0003097032 to 0.0000979835, SNR rose from
7.8243 to 17.8201 dB and error RMS fell from -70.1811 to -80.1769 dBFS.
Residual-block local relative L2 became 2.97e-7, 1.02e-6, 9.62e-7 and 5.04e-7.
The final head local relative L2 was 3.55e-7. These show the selected arithmetic
remedy works locally; remaining accumulated error is still material. Forced-input
transposed-convolution relative L2 remains 0.0235944, 0.0241047, 0.0181380 and
0.00428386; the initial convolution remains 0.00362022. Those unchanged INT8
operators and the upstream RVQ/decoder mismatch remain numerical bottlenecks.
A next bounded diagnostic should isolate a transposed phase on identical inputs
before expanding precision scope or making performance claims.

The candidate raw range was -0.0152960150 to 0.0202990007 and RMS 0.0032855432;
zero of 3840 samples exceeded ±1. No waveform gain or clipping was applied.
All candidate stages, PCM and persistent state replayed bit-exact after reset.
The unchanged INT8 path matched all historical stage/PCM/state arrays, and the
residual candidate's initial three stages matched historical arrays. Native
full and forced-stage reset replay also matched exactly. Between fixed device
programs, the checker recorded zero host reads or writes; each component frame
had one 4096-byte fixture-input write before execution.

The F2 aggregate suite passed 115 tests in 16.80 seconds; component checking
took 241.39 seconds CPU wall time (260.24 seconds including the suite), never
FPGA frame latency. Preferred-builder checks passed 47 compatible tests in
0.36 seconds and compiled all three full production layouts. Suites overlap
and are not additive independent validation counts. Physical-device open
attempts were zero. Reproducible integration verified all 103+39 file hashes;
dependency-manifest SHA256 is
`4cd15957f4215ea10a9b26a1c03a4db0a2c2707e27968966333cb05b02f6d670`.

The original INT8 result and first head-only remedy receipts are retained.
No physical timing, listening, intelligibility or end-to-end speech acceptance
was performed. Precision options remain explicit and default to INT8.

## Capacity and reproduction

Target is exactly `f2-hbm:p4rl8a64c150`, architecture hash `dd6b92a6`, image
context `agfi-0001de1dd3df8e21e`; image identity is context rather than a
physical qualification. Use `tools/package_personaplex_shared_runtime.py` with
the committed dependency manifest and runtime
`18fb978033799f0f21258f0e6e085c0360322d96`. The builder verifies 103 runtime and
39 PersonaPlex files. The old-runtime published branch alone is insufficient.

The residual-FP32 candidate occupies 25,059,736 SEANet bytes, including
17,046,528 weight bytes and unchanged 46,612 persistent reset bytes. Its
14 programs use 280/344/408/432/344/408/432/344/408/432/344/408/432/304 words.
Combined named occupancy is 1,691,863,524 bytes, 2,124,792 bytes above INT8.
Highest aligned extent remains 4,273,471,488 within 4 GiB. All 27 programs fit
32,768-word IMEM individually; B slack remains 21,495,808 bytes. This does not
establish concurrent temporal-weight residency or a measured transport floor.

Preferred `opentpu-m8a-build` ran the compatible suite and production capacity
compilation using its existing NumPy environment. Its existing environments
lack Torch/Moshi, so real-checkpoint native comparison used the already
provisioned F2 CPU environment and cached codec. No dependencies were installed
and no model data was downloaded to the Mac. All numerical jobs used one
thread, affinity and nice 19, with physical-device opens forbidden. The Mac
was only a source/Git/SSH/AWS control bridge.

Remaining full-speech gates include numerical/audio-quality acceptance, the
input Mimi codec, unified temporal paging and resident temporal-to-depth
handoff, prompt/voice/delay scheduling, streaming beyond 128 code frames,
qualified common-image transport and coordinated end-to-end physical
acceptance. No FPGA/card/lock/image/service changes, synthesis, new resources
or CPU/GPU serving fallback occurred. Full speech stays disabled.
