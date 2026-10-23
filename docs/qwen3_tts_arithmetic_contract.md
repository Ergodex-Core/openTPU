# Qwen3-TTS arithmetic and acceptance contract

The implemented main-model path targets the pinned source weights evaluated
with FP32 algebra. BF16 checkpoint weights are extended exactly to FP32 before
offline packing. The decoder checkpoint contains F32 weights. Device affine
products may use the documented two-term INT8 correction; norms, residuals,
attention and nonlinear arithmetic use the existing FPGA FP32 primitives.
Measured source-reference error is required at each boundary. This is not native
FP16/BF16 MM, bit-exact BF16 execution, or a claim of every upstream CPU/CUDA
kernel's bitwise behavior.

The preserved main-model receipt at commit `883edbdd` matches all 144 FP32 codec
IDs in its three-prompt/nine-frame corpus and fails the unchanged overall gate
because BF16 agrees on only 127/144 IDs and exceeds the 3.5% numerical bound.
That receipt, all BF16 comparisons, the older failures and accepted=false remain
unchanged. A decoder component passing an FP32 reference does not turn the main
model's multi-oracle gate into a pass.

Neither FP32 source-algebra parity nor BF16 parity alone establishes model
quality. Prefix codec frames are not complete utterances. Complete waveform
decoding, raw output scaling/clamping, causal/streaming/reset behavior, full
utterances, quality assessment and coordinated physical/serving acceptance are
separate required evidence. Full TTS stays disabled until those gates are met.

All learned request-time computation must remain in FPGA programs: embedding
lookup/normalization, affine layers, attention, convolution/upsampling,
nonlinearities, residual mixing and waveform output. Host BPE, offline verified
weight packing/constants, deterministic control, transport and WAV serialization
are orchestration. CPU references and the ISA simulator are explicit diagnostic
tools and are never production fallback backends. No CPU/GPU/API fallback or
upstream custom executable model code is connected to serving.

The hardware contract is still the shared image `agfi-0001de1dd3df8e21e`, profile
`f2-hbm:p4rl8a64c150`, build `417243f1`, configuration hash `dd6b92a6`, with the
qualified runtime `30100afb075374381601a6e01d9f914a58dbec91` or an independently
verified compatible runtime. Only weights/programs may switch. No image load,
synthesis, physical-card access or service change is authorized by these
software decoder experiments; the serving interruption budget remains 30 seconds.

Subsequent software compute and checkpoint data use the existing
`opentpu-m8a-build` through its verified SSH route. The Mac is the available
control/Git bridge. Existing jobs and immutable receipts are not duplicated or
migrated; credentials, host-key policy, remotes, system PATH and other workloads
are not changed.
