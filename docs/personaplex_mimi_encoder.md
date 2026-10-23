# Resident raw-PCM SEANet input encoder

This increment follows the [output-codec correction](personaplex_output_codec_precision.md)
at `aeceba36` and preserves its implementation
and receipts. It adds a software-qualified resident input SEANet component,
not the complete input Mimi codec or bidirectional PersonaPlex inference.

## Source contract

The cached codec contains 28 FP32 encoder tensors, 50,513,024 bytes including
biases. `mimi_encoder_contract.json` pins every name, shape, dtype and payload
offset, the existing safetensors header, installed Moshi sources, constructor
and fifteen top-level output shapes. Only those encoder tensors are loaded.
No independent whole-checkpoint payload digest is claimed.

The native constructor is mono, dimension 512, 64 base filters, one residual
block per level, true skips, ELU alpha 1, constant causal padding and no norm.
Encoder strides reverse decoder ratios to 4/5/6/8. Its fourteen Conv1d kernels
retain native `[out, in, tap]` orientation. Model stages emit:

| Stage | Time rows | Channels |
|---|---:|---:|
| Initial, residual, ELU | 1,920 | 64 |
| Stride 4, residual, ELU | 480 | 128 |
| Stride 5, residual, ELU | 96 | 256 |
| Stride 6, residual, ELU | 16 | 512 |
| Stride 8, ELU | 2 | 1,024 |
| Final convolution | 2 | 512 |

`StreamingConv1d` adds initial left padding of `kernel - stride` once.
`RawStreamingConv1d` retains input after `num_frames * stride`; for a complete
stride-aligned frame this is exactly `kernel - stride` history rows. Device
programs prepend that history, gather windows with native stride, multiply
FP32 coefficient rows, reduce them, add bias and apply the original residual
order. History, ELU and interstage activations remain in device DRAM/programs.
Logical reset zeroes the frame counter and persistent convolution history.
The one-row final-convolution test also exercises a chunk shorter than history.

PCM remains raw mono FP32. There is no clipping, input gain, normalization,
resampling or integer conversion. The host writes one complete PCM input frame
before execution and reads diagnostics afterward. Fourteen argument-free
programs run without implicit host activation reads or writes between stages.
Offline packing and native CPU references are diagnostics, not a serving path.

## Shared profile and residency

Exact profile `f2-hbm:p4rl8a64c150`, architecture hash `dd6b92a6`, image context
`agfi-0001de1dd3df8e21e`, runtime
`18fb978033799f0f21258f0e6e085c0360322d96`. Use the committed reproducible
builder and dependency manifest; the old-runtime branch alone is insufficient.

The first placement under the existing output-codec state ceiling failed at
`conv12_prepared`. That failed preflight is preserved. `after_output_codec`
instead uses the unallocated interval after immutable output-codec weights
and before depth matrices at `0xB0000000`. It reserves 32 MiB for input state,
places encoder matrices afterward, and appends fourteen instruction slots
within the existing `0x04000000..0x05000000` region. Every combined named span
is checked for overlap; all output-codec addresses remain unchanged. This is
a software layout change with no image reload, new opcode or RTL change.

Encoder named occupancy is 59,501,980 bytes including the raw input and full
reserved instruction slots. Packed matrices use 50,593,792 bytes; immutable
payload including bias uses 50,609,536 bytes. Padding adds 96,512 bytes over
the native payload. Persistent reset bytes are 42,012. State ends at
1,776,844,800, matrices at 1,854,377,984, leaving 26,939,392 bytes in the input
state reservation and 1,098,412,032 bytes before the depth matrix boundary.

Combined output-plus-encoder named occupancy is 1,873,592,192 bytes. Highest
mapped extent remains 4,273,471,488 within 4 GiB. All 41 instruction slots fit
the existing region, and each program fits 32,768-word IMEM. Encoder program
sizes in words are 416/408/440/936/408/440/1080/408/440/1224/408/440/696/384.
These are static budgets, not physical throughput or latency measurements.

## Validation and numerical gate

Independent global-time NumPy equations exercise consecutive frames, all four
strided phases, reversed nonuniform ratios, short history, true residuals,
raw scale, reset and host-transfer guards. Synthetic tensors also load strictly
into the native class. Production residency and instruction checks use the
exact shared profile. The preferred builder uses existing NumPy dependencies;
actual checkpoint/native comparison uses the existing F2 Torch/Moshi cache.
All numerical execution is on EC2, one thread, nice 19 and guarded against
device access. Mac is only the source/Git/SSH/AWS and stdlib AST/hash bridge.

The checker consumes two saved generated raw PCM frames from the preserved
native output-codec fixture, 3,840 samples, including a quiet frame. It does
not rerun the LM, depth transformer or output decoder. It captures all fifteen
boundaries and raw persistent state, persists first-pass arrays before reset
or evaluation, and compares reset replay, native fresh streams and direct native
forward versus captured traversal. It also compares every stage on identical
saved device inputs to distinguish local arithmetic error from accumulation.

The declared final encoder latent gate applies to aggregate AND each frame:
relative L2 <= 1e-4, RMS error <= 1e-5, maximum absolute error <= 1e-4.
The same limits apply separately to the final stage on identical device input.
All intermediate accumulated and identical-input errors are reported unchanged.

The actual-checkpoint run completed both frames and reset replay. Both declared
latent gates passed in aggregate and for each frame:

| Comparison | Relative L2 | RMS error | Maximum absolute error |
|---|---:|---:|---:|
| Accumulated raw PCM → encoder latent | 1.20078493e-6 | 6.19596663e-8 | 1.78813934e-6 |
| Final convolution on identical device input | 9.62732813e-7 | 4.96763370e-8 | 1.90734863e-6 |
| Quiet frame, accumulated | 1.27546656e-6 | 6.01310843e-8 | 1.25169754e-6 |

All fifteen stage outputs and final persistent history/counter replayed
bit-exactly after reset. Native fresh-stream replay and direct source forward
versus captured module traversal were also bit-exact. Implicit interstage host
reads and writes were zero. All 138 baseline/ancestry hashes remained unchanged;
59 diagnostic arrays remain on EC2. Every stage's accumulated and identical-input
errors are retained in the receipt; none are clipped, normalized or suppressed.


Preferred builder: 100 tests passed in 5.35 seconds (one native-only test
excluded because Torch/Moshi is not installed there). The existing F2 CPU
environment passed 169 aggregate regressions in 33.74 seconds. The real-checkpoint
encoder checker completed in 347.25697344506625 seconds CPU wall time; wrapper
382.2704115980305 seconds. Both hosts used one thread and nice 19; physical-open
guards recorded zero attempts. All 153 runtime/overlay files match; manifest
SHA256 `be899a9e31e55d4dd412517ffc16c538d70f24b3dd1f46a8bfde5b09a429cfed`.
These times are simulator wall time, not FPGA frame latency or throughput.

This component has no attention ring. The inherited source transformer ring
position defect is not changed or masked; the following encoder-transformer
increment must retain and explicitly qualify its streaming/reset boundaries.
The next source audit confirms that learned downsampling uses a bias-free
512-to-512 kernel-4/stride-2 convolution with initial replicate padding, not
zero history. The native RVQ search uses `torch.cdist(p=2).argmin` on normalized
centroids; distance rounding, ties and exact IDs require separate validation.
Neither operation is implemented by this encoder increment.

Input transformer, learned downsample, semantic/acoustic RVQ encoding, temporal
paging, prompt/voice/delay scheduling, same-image transport and coordinated
physical/quality acceptance remain separate gates. Full speech remains disabled.
No FPGA, image, service or production changes, synthesis, dependency installation,
new resources or CPU/GPU serving fallback are part of this increment.
