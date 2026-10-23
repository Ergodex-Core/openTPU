# PersonaPlex precision and FPGA-only audit, 2026-10-23

Full-FPGA model inference is the completion requirement. The measured runs here
execute only the 32-layer temporal transformer, attention/KV updates and final
RMSNorm. They produce hidden activations from captured voice embeddings, not
audio or generated tokens. Neither numerical quality nor full speech is accepted.

Later development adds CPU/ISA-validated standalone device slices for resident
RoPE/text-head/greedy selection and input embeddings/stored voice/audio heads.
See [token slice](personaplex_token_slice.md) and
[device components](personaplex_device_components.md). These were not part of the
physical measurements below and do not make the hybrid server FPGA-only.

## Diagnosis and targeted source correction

Both runs use the existing pinned `aufklarer/PersonaPlex-7B-MLX-4bit` checkpoint,
revision `8c686347007de51d9017706e5543d6f1757ecf5d`, and the first four NATF2
stored-voice embeddings, starting at position zero with empty KV. The FP32 input
NPY SHA-256 is `fdb7ba233a26c5b47c3fcbbbb340831203619d1277dd02eb3016248b2281a39d`.
The reference dequantizes the source MLX affine INT4/group64 weights and performs
independent FP64 transformer arithmetic. Reported error is relative L2 against
that reference, not against the original unquantized NVIDIA checkpoint.

The first layer's weight-only requantization error was 7.299%; full ISA error was
7.381%. An arithmetic emulator including activation and KV quantization matched
that layer's ISA output to relative L2 `1.3953e-7`. A full 32-layer FP64 trace
exactly reproduced the saved reference; with MX weights alone its final error
was 76.713%. The largest same-input local residual errors were in early layers
and layer31. These observations implicate additional weight requantization loss;
they do not establish that activation quantization is harmless in every layer.

The explicit candidate uses INT8 attention and MLP weights in layers0–3 and31,
with INT4 in the remaining layers. `Image`, `Engine`, the ISA checker and FPGA
CLI now propagate the same explicit `--formats` rules through layout, packing,
per-layer down-projection chunking and weight-byte accounting. Empty rules retain
uniform precision and ignore inherited `OTPU_FORMATS` overrides.

```text
--cap 128 --wformat int4 \
--formats 'attn@0-3=int8,mlp@0-3=int8,attn@31=int8,mlp@31=int8'
```

The cap1024 candidate fits DRAM but fails instruction capacity: 33,688 words
versus 32,768. Cap128 fits: 4,039,421,952 required bytes in 4,294,967,296 bytes,
with 25,240 instruction words. It provides 10.24 seconds of total context,
including prompt frames, and requires an explicit reset at exhaustion. Four
initial frames do not test longer attention buckets or conversational context.

## Measured physical results

The same existing burst200 image was qualified at 200MHz. Both runs acquired the
normal canonical and legacy locks after checking holders and physical handles.
Each ran four initial frames and four bit-exact reset replays. Outputs matched
their corresponding ISA results bit-for-bit; the loaded image remained unchanged.

| Metric | Uniform INT4 baseline, cap1024 | Targeted mixed precision, cap128 |
|---|---:|---:|
| Aggregate relative L2 | 0.7631925061 | 0.3286395682 |
| Cosine similarity | 0.7118711952 | 0.9458499652 |
| Mean device milliseconds/frame, 8 runs | 162.9994394 | 181.3921169 |
| Temporal frames/second from device counters | 6.1349904 | 5.5129187 |

Mixed per-frame relative L2 is `0.2467017893, 0.2565805661, 0.4481266869,
0.2706557370`. Initial device times are `181.359040, 181.414310, 181.396165,
181.386215` milliseconds; reset times are `181.365275, 181.391555, 181.402010,
181.422365`. Device times use raw cycles divided by the probed 200MHz clock.
Host first-frame wall time was 1042.15ms; warm initial frames averaged182.57ms.
Cached image build/upload took102.12 seconds and is separate from frame timing.

The correction reduces aggregate error by56.94%, while increasing device latency.
32.9% residual error is still substantial; no quality threshold is inferred.
Both mappings miss the80ms budget for12.5 audio frames/second. The differing
context caps and short prefix must accompany comparisons. No speech was generated.

Fresh EC2 CPU verification:194 tests passed in15.61s with zero physical transport
open attempts. The combined test/cap1024 job exited1 solely at the instruction
capacity gate; cap128 then passed separately. The four-frame mixed ISA check and
four reset replays completed successfully in386.81 seconds of CPU job wall time.

Follow-up review added capability guards for INT4 overrides under an INT8 base:
the CLI resolves actual layer formats before opening a transport, checks CAP_W4
at both probes, and preserves early rejection of an unsupported board profile.
The direct `Engine(backend="board")` reverse-base path requires a qualified
backend factory. After fixing an error-ordering regression caught by the first
guard run (106 passed,1 failed), both changed suites passed:107 tests in14.27s,
zero physical transport opens, SSM `017409d5-f798-4f3f-981e-84a6699c633e`.
These guards do not alter the measured base-INT4 candidate's arithmetic, so the
checkpoint simulation and physical execution were not repeated.

Evidence IDs: layerwise attribution `ec1b905f-b600-4640-a0d3-6569ccdba87d`;
CPU aggregate `0cff5212-37ac-41b9-bb08-42a4e373a4d8`; cap128
`6679458e-4f0b-4b64-94d0-ea5d71da9c5d`; mixed ISA
`562fac9b-49aa-4ba0-b8bc-9f9347e2b230`; mixed physical
`743004de-37b3-489b-8956-893fd4eb65f4`. Immutable baseline artifacts are `isa/`
and `board-001/`; corrected artifacts are `isa-mixed-001/` and `board-mixed-001/`.
Output NPY SHA-256 for both mixed backends is
`232d09089cda8293506f1ea33a967167cfb8fb29b73f8dd4cddc2406e3dc58ad`.

## Remaining host model computation

| Component | Current execution boundary | Required for full-FPGA completion |
|---|---|---|
| Mimi audio encoder/decoder, codebook operations and codec state | Native upstream CPU codec | Device codec kernels, weights and streaming state; PCM-in/PCM-out validation |
| Text/audio embeddings and their sum; voice conditioning tensors | Standalone device programs validated on CPU/ISA; hybrid server still uses native CPU | Combined residency, device chaining and native prompt/cache/delay scheduling |
| Temporal transformer, KV and final norm | FPGA, measured above | Precision acceptance, context/capacity and latency improvements |
| RoPE coefficients | Resident device table in the CPU/ISA-validated token slice; original temporal bridge computes on host | Qualified physical combined-path validation |
| Text vocabulary projection | INT8 device head/greedy in the CPU/ISA-validated token slice; original server still uses CPU | Combined physical path and quality acceptance |
| Depth projections, step embeddings, six-layer depth transformer and audio heads | Standalone device audio heads validated on CPU/ISA; depth path remains CPU | Device conditioning/transformer, per-codebook state, residency and chaining |
| Temperature, top-k and token sampling/RNG | Native CPU LMGen; device greedy is implemented in standalone heads | Explicit greedy-only final mode, or separately implement native stochastic mode |

`personaplex_bridge.install_bridge` replaces only `lm.transformer` and
`lm.out_norm`; `load_host_model` deliberately loads remaining tensors on CPU.
`tools/personaplex_fpga.py serve` constructs that CPU model and passes
`--device cpu` to the upstream server. `Spec._core` has no vocabulary and the
temporal image uses no embedding lookup; it contains no codec, depth or head
weights/programs. The existing header audit reports7,177,895,936 bytes of FP32
host parameters, while the current mixed temporal image nearly fills its4GiB
address space. Those FP32 bytes are not a lower bound for a quantized FPGA port,
but full residency cannot be assumed. A unified precision/memory plan and new
device programs are required. Whether RTL/image changes are necessary must be
established by compilation and capacity analysis before proposing synthesis.

CPU input checks, transport, UI and scheduling may remain on the host. Model
arithmetic above cannot silently fall back to CPU/GPU for the final path. No new
synthesis, installation, paid resource or full-model speech service was started.

## Local UI integration boundary

Existing CLI modes are `inspect`, `preflight`, `benchmark` and `serve`.
`benchmark` accepts captured `[frames,4096]` floating-point NPY embeddings and
requires a new output file. The only HTTP route defined here is
`GET /api/fpga-status`; there is no implemented OpenAI-compatible text endpoint.
The upstream speech server has not been exercised end-to-end and is a hybrid
experimental path. A strict FPGA-only UI must keep it unavailable until the
complete model path above is implemented and verified.

Queue all models per physical card, acquire the normal canonical and legacy
locks nonblocking, and queue/return busy on `DeviceBusy`. Monitors read status
and lock metadata without attempting lock acquisition or quiescing the card.
Never release another owner's locks, restart services or program an image as an
availability probe. Saved receipts do not establish current card availability.
`BoardSession` owns the canonical transport lock; the validation driver acquires
the legacy lock separately. A UI adapter must provide that same wrapper and a
shared queue; neither the queue nor the legacy wrapper is an HTTP API here.
A long-lived `serve` process holds the card for its lifetime, so it must not be
launched as an automatic availability or UI startup check.
