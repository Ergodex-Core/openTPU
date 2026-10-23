# Qwen3-TTS: resident embeddings and one complete talker block

This is the next bounded compiler milestone after `cb98d4f3`, for
**Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice**. It implements device text/codec
embedding lookup, reuses the validated text projection unchanged, and executes
one complete talker layer with resident KV state. It does not produce codec
frames or audio, and the existing `synthesize` gate remains closed. No serving,
portal, shared image, RTL, synthesis, or device-access code changes.

The original [architecture and full-model audit](qwen3_tts_fpga.md) describes the
first milestone. This document supersedes its embedding/layer readiness only;
its full text-to-waveform blockers still apply.

## Pinned source contract

- [Exact model config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/blob/85e237c12c027371202489a0ec509ded67b5e4b5/config.json),
  revision `85e237c12c027371202489a0ec509ded67b5e4b5`.
- [Qwen implementation](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py),
  source revision `022e286b98fbec7e1e916cb940cdf532cd9f488e`, package `qwen-tts 0.1.1`.
- Model and source are Apache-2.0; [upstream license](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/LICENSE).

`tests/fixtures/qwen3_tts_talker_contract.json` records the source file/function
hashes and the shapes/dtypes of all 17 tensors from the real safetensors header.
They contain 336,336,128 stored elements, including the large text table;
15,730,944 belong to the transformer layer. No weight payload or upstream
executable source is vendored, downloaded, installed, or executed by this slice.

The source-derived reference is independently written NumPy FP64 mathematics.
It imports no compiler, quantizer, ISA helpers, or Qwen Python. This is a pinned
source/layout comparison, not a run of upstream PyTorch or a BF16 parity claim.

## Implemented boundary

`TalkerBlockImage(TalkerSpec.from_config(config), cfg, capacity=256)` lays out
resident device memory without allocating the weights on the host.
`build(weights)` separately converts supplied arrays, with an explicit allocation
budget (256 MiB by default). Full-vocabulary packing deliberately exceeds that
default; it never causes host embedding offload. `compile(position)` returns
one ordinary ISA program plus an instruction/memory budget, rejecting capacity
and IMEM overflow. `token_bytes(text_id, codec_id)` validates IDs before DMA.
`ISASession` is explicitly a CPU diagnostic ISA simulator, never a serving
backend or fallback.

For each sequential position, the device performs:

1. Two resident FP32 embedding reads. RLD converts each exact integer ID and
   multiplies by the row byte stride on device; no host row gather.
2. The existing biased 2048→2048→1024 SiLU projection, then addition of the
   codec embedding. Intermediate scratch transfers stay in device DRAM/TMEM.
3. Input RMSNorm; Q/K/V projections; per-head Q/K RMSNorm; rotate-half RoPE;
   K/V quantization and append into persistent device cache.
4. Causal grouped-query attention (16 query / 8 KV heads, dimension 128),
   output projection, and residual addition.
5. Post-attention RMSNorm, 3072-wide SwiGLU MLP, down projection, and residual.

All model arithmetic in this boundary is emitted as ISA instructions. The
normalization epsilon is 1e-6 and RoPE theta is 1e6. The three position axes are
equal, matching the source's text-only path; selecting interleaved axes then
reduces to ordinary rotate-half RoPE. Distinct multimodal axes are unsupported.
The kernel uses shared arithmetic/attention primitives, not a Gemma or
PersonaPlex model adapter.

This boundary accepts **one aligned text ID plus one codec ID**, batch size one,
on one slice. It is not the full CustomVoice prompt: pure-text/control-prefix
positions, language/speaker sequencing, trailing text scheduling, and summing
the 15 predictor-code embeddings are not yet implemented. It is layer 0 only;
it does not repeatedly run all 28 layers or include final norm/codec head.

## Exact implemented precision

| Data/operation | Representation |
|---|---|
| Text/codec tables, biases, RMS weights, hidden residuals | FP32; offline conversion flushes subnormals |
| Nine affine matrices (two projection, seven layer matrices) | OpenTPU MX INT8: signed byte elements, FP32 scale per output row, one power-of-two offset byte per D-element block; padded scale rows |
| MM activations | QACT INT8 with FP32 row scale and per-D/4 power-of-two offsets |
| Native K/V cache | INT8, one FP32 scale per head/token for each of K and V; K token-major, V in transposed 256-token tiles |
| RMSNorm, residuals, SiLU, online softmax | Existing ISA FP32 round-to-nearest-even/flush-to-zero arithmetic and its exp2/reciprocal/rsqrt implementations |
| RoPE | Fixed FP32 cos/sin tables prepared offline once from theta and capacity; row selection and rotations on device |

The MM accumulator/drop rules are those of `docs/isa.md` and
`opentpu/fp32.py`, with the INT8 shift window of 16. The MLP down projection uses
static 384-column chunks at native width; each chunk's activation is separately
quantized. Static chunks avoid an unsupported loop-dependent `MM k0` immediate
for a monolithic scaled matrix. The program fits IMEM without repacking weights
or changing the ISA. Attention itself uses existing hardware loops at long
contexts. Dense context rows permit the attention-output reshape without
copying through the host.

This is a changed precision recipe from upstream BF16. Synthetic-reference
errors are not evidence of checkpoint speech quality, and no full-model
quantization or accuracy claim is made.

## Resident memory and instructions

The descriptor uses the existing `targets/f2.toml`: D=128, MCOLS=4, S=1,
4 GiB address window, 65,536 TMEM words and 32,768 IMEM words. These are compile
limits, not a query of the shared 150 MHz image or a claim of free board memory.

| Resident component, native shapes | Bytes |
|---|---:|
| Text table, 151,936 × 2,048 FP32 | 1,244,659,712 |
| Codec table, 3,072 × 1,024 FP32 | 12,582,912 |
| INT8 matrix elements | 22,020,096 |
| Matrix scale rows | 233,472 |
| Biases and RMS weights | 21,504 |
| IDs and visible scratch/output tensors | 24,584 |
| Alignment padding | 120 |
| KV, per capacity position, this one layer | 2,112 |
| RoPE, per capacity position | 512 |

| Cache capacity | Total resident bytes | KV bytes | RoPE bytes |
|---:|---:|---:|---:|
| 256 | 1,280,214,144 | 540,672 | 131,072 |
| 2,048 | 1,284,916,352 | 4,325,376 | 1,048,576 |
| 32,768 | 1,365,525,632 | 69,206,016 | 16,777,216 |

All tables, parameters, and cache fit this **single-layer** address window.
This does not establish full-model residency. Packing FP32 embedding tables is
a conservative first implementation and doubles their native BF16 storage.

| Position, capacity 32,768 | Instructions | IMEM words | Peak live TMEM words |
|---:|---:|---:|---:|
| 0 | 341 | 2,728 | 15,162 |
| 255 | 325 | 2,600 | 15,162 |
| 256 | 429 | 3,432 | 15,162 |
| 32,767 | 718 | 5,744 | 15,162 |

These are measured compilation points, not an exhaustive bound over every
position. Every invocation independently enforces its IMEM limit. At position
32,767 the static program has LD 53, ST 5, RLD 2, QACT 61, QST 16, MM 110,
VOP 426, LI 4, ADDI 32, LOOP 8, HALT 1. Loop execution counts and runtime grow
with context; instruction count is not a latency or throughput estimate.
There is no WAITW, DSTEP, STREAM, external host-weight request, or new opcode.

## Tests and host audit

The tests run at low priority with one numerical thread on the existing EC2
instance, in an isolated directory: **32 passed in 3.84s**. The original 18 projection/contract tests
are retained. New tests cover independent FP64 stage comparisons at D32 and
D128, native layer widths with small synthetic vocabularies, causality with
poisoned future KV, cache persistence across an attention-block boundary,
long-cache hardware loops, input validation before writes, removal of host
weights after packing, residual identity with zero layer matrices, and native
full-vocabulary/max-context descriptor compilation without allocation.

Native-width synthetic maximum relative L2 errors over two positions were
0.012206 for projection, 0.003460 for assembled input, 0.010796 after attention,
and 0.011775 at layer output. Acceptance is <0.035 for these independent
full-precision comparisons; the 35-step causal test uses <0.04. Bitwise checks
cover future-cache poisoning and the direct token-ID-only wire path. These
thresholds are diagnostic, not a speech quality acceptance criterion.

| Host dependency | Status in this slice |
|---|---|
| Offline weight loading, INT8 packing and constant RoPE generation | Explicit build stage; no hidden downloads or request-time neural work |
| Text BPE and CustomVoice prompt schedule | Not wired; future deterministic frontend/control only |
| Per-step input | Exactly two validated token IDs (8 bytes), sequential position/program selection |
| Embedding, projection, block math and KV updates | Entirely in the emitted device program |
| Output handling | Test harness reads hidden vector; future transport may do DMA/orchestration |
| NumPy reference / ISA simulator | CPU diagnostic tests only, not production inference |
| Token validation | Host rejects bad IDs before DMA; RLD BOUND is not enforced in RTL and is not claimed as hardware memory protection |
| CPU/GPU/external-API fallback | None; complete synthesis still fails closed |

## Remaining blockers and minimum next decisions

No new primitive or synthesis is needed for this bounded layer. Production
transport and hardware correctness remain **pending**, not denied: parent
coordinates a validation window after Gemma releases the shared card. This
work does not request the card, reload its image, or measure hardware latency.

Before a full text-to-waveform route can open, implement the actual prompt
schedule and 16-code embedding accumulation, all talker layers/final head,
frame-local five-layer predictor and 15 heads, on-device sampling and generation
control, RVQ reconstruction, and the waveform decoder/state machine. Greedy
ARGMAX exists, but replacing the configured stochastic sampler silently is
not accepted; the device RNG/sampling design still needs a decision.

The audio decoder additionally needs causal/depthwise/transposed convolution
lowering and streaming state, 64-wide-head handling, LayerNorm and layer scales.
Its exact GELU requires an erf implementation and SnakeBeta requires dynamic
sine squared. The current ISA has neither erf nor sine. The minimum decision
is a separately validated ISA approximation/table strategy with explicit
range/error and audio-quality criteria, or new VOP RTL plus simulation and
later coordinated synthesis. No host sine/GELU, host im2col, or hidden decoder
fallback is acceptable. No RTL or costly build was started.

See `qwen3_tts_talker_validation.json` for source hashes, EC2 receipts, measured
budgets, and numerical errors. The original `cb98d4f3` milestone is preserved;
the parent publisher owns publication of this isolated follow-up.
