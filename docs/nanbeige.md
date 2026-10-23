# Nanbeige 4.2 on burst200

The looped Nanbeige target and a co-resident DSpark accelerator draft run on
burst200. Tiny hardware gates pass against ISA for both, including the cached
target execution path. On the short prompt measured below, warmed FPGA-draft
decoding reached **4.225 tokens/s versus 2.955 tokens/s** for the same target
image, a **1.430×** median improvement across three repetitions with exact token
matches. The CPU-draft variant reached a similar 4.258 tokens/s. These are short
prompt results; a longer 60-decode-token response reached **3.650 tokens/s** with
the FPGA draft, a **1.235×** improvement with exact matches in all repetitions.
These two prompts are not a general throughput or model-quality benchmark. Some
draft operations remain on the CPU; this is not a fully FPGA-resident decoding
loop. An optional mode also executes context fusion and the Markov projection
on the FPGA; it passed hardware checks but was about 5% slower on both prompts,
so host fusion/Markov remains the default.

## Checkpoints and arithmetic

| Role | Hugging Face repository | Pinned revision |
|---|---|---|
| Target | `Nanbeige/Nanbeige4.2-3B-GPTQ-Int8` | `e8c6ccdb655bbcf410952686026d3fcbf4c07392` |
| Draft | `Nanbeige/Nanbeige4.2-3B-DSpark` | `5f12c792dabbfcdc4a0cf504e75f7216706d9590` |

The target checkpoint is compressed-tensors channelwise symmetric INT8
**weight-only** quantization, with BF16 activations, embeddings and an untied
BF16 LM head. `nanbeige_weights.py` lazily decodes the packed weights: each byte
stores `q + 128`, so unpacking subtracts 128 rather than interpreting the byte as
two's-complement INT8. Decompression applies the per-channel scale and rounds to
the scale dtype. Tensor shapes, metadata and unsupported options are validated.

The openTPU image builder then requantizes these values into its **W8A8 MXU
format and INT8 LM head**. The CLI retains the original BF16 embedding values,
expanded to FP32 on the host, and uses nonresident host-fed embeddings. Residual,
normalization, RoPE and softmax operations use the existing openTPU FP32 kernels.
This is not numerically identical to the published W8A16/BF16 model. An exact
FPGA/ISA match validates the port's arithmetic, not the model-quality impact of
this conversion.

The CPU draft uses FP32 computation over its BF16 source weights and explicitly
shares the original target embedding and LM-head tensors. It is a correctness
reference and initial hybrid baseline, not a GPU-runtime performance equivalent.

The accelerator draft in `dspark_fpga.py` instead uses **INT4 weights and INT8
activations/KV** for the noncausal backbone and shares the target's INT8 output
head read-only. It introduces additional draft quantization; target verification
still determines every emitted token. `--draft-fusion-markov board` also
quantizes `fc` and `Markov_w2` to INT8 and executes them on the FPGA. The default
is `--draft-fusion-markov host`; neither accelerator mode is source-arithmetic
equivalent. Their execution split is explicit:

| Draft operation | Default host fusion/Markov | Optional board fusion/Markov |
|---|---|---|
| Context feature fusion (`fc`) | CPU FP32 | FPGA INT8 projection |
| Context hidden normalization | CPU FP32 | FPGA FP32 |
| Embedding lookup | Host, original target values | Host, original target values |
| Per-layer context K/V projections, QK norm and RoPE | FPGA | FPGA |
| Five-layer noncausal query backbone | FPGA W4A8 | FPGA W4A8 |
| LM head | Shared target FPGA INT8 head | Shared target FPGA INT8 head |
| Markov `w2` projection and logit addition | CPU FP32 | FPGA INT8 projection, FP32 addition |
| Markov `w1` row lookup, sequential argmax and confidence | CPU | CPU |

## Target and speculative execution

`nanbeige.Spec.layers` denotes 22 physical layers. Two loops produce 44 logical
layers: weights are stored once, while each logical layer has distinct KV state.
The target applies the loop-boundary final norm because
`skip_loop_final_norm=false`. Its 48 query heads and eight KV heads use head
dimension 128, giving Q width 6144 and K/V width 1024. The six query heads per
KV head are split across burst200's four MXU columns.

Single-token and multirow kernels are in `nanbeige.py`. A tiled multirow fallback
reduces TMEM requirements. The selected hidden-state capture table preserves
compact layer loops and stores only requested taps; `capture_layers=()` removes
capture writes from a target-only image.

Host-input runtime programs now take position arguments and cache compiled
programs by attention bucket, row count and requested output rows. This avoids
recompiling the same forward structure at each token while retaining the source
embedding values. On burst200, the cached production target uses 9,696 instruction
words for decode and 31,824 for four-row verification, within 32,768-word IMEM.

DSpark is a noncausal parallel query backbone followed by sequential low-rank
Markov logits, rather than an autoregressive small language model. Its checkpoint
layer-output IDs `[1, 10, 21, 32, 40]` map to the target's **pre-layer** capture
indices `[2, 11, 22, 33, 41]`, matching the official runtime's `+1` conversion.
The loop-boundary capture is taken after the intermediate final norm.

`nanbeige_specdec.Decoder` maintains a pending anchor token. It uses the selected
CPU or accelerator draft, verifies `[anchor, draft_1, ...]` together on the target,
commits only the anchor
and accepted draft inputs, rolls back rejected KV positions, and emits the
target's correction or bonus token as the next anchor. Only committed target
states enter the draft context. Greedy EOS and token-budget rules apply to both
paths; stochastic speculative sampling is not implemented.

The target verification setting is **three proposals plus one anchor**, with
context capacity 128. The draft now evaluates its full **seven-query noncausal
block** and verifies only the first three predictions. Shortening that query
block changes even its prefix predictions; the initial measurements below used
a three-query experiment and do not measure the current seven-query default.
The verifier reduces its block size on a compilation-capacity failure; device
execution failures propagate. Eight-row verification remains
too large for the current single-program IMEM path even when its TMEM allocation
fits. The generic Qwen `SpecDecoder` is deliberately rejected for Nanbeige.

## Recorded hardware evidence

The [tiny hardware gate](data/nanbeige/hwcheck.json), recorded on
2026-10-20 at 16:45 UTC, passed on the 200 MHz burst200 image. Single-token and
tiled four-row FPGA logits and captured states had **zero unequal words**
against ISA. The tiled outputs also matched the sequential ISA schedule exactly.
This validates the port's tested arithmetic, not quantization quality on a
benchmark suite.

Three subsequent hardware gates also passed with zero unequal words in the
checked tensors: [cached target](data/nanbeige/hwcheck-runtime.json) and
[default accelerator draft](data/nanbeige/hwcheck-draft.json) at 17:09 UTC, then
[board fusion/Markov draft](data/nanbeige/hwcheck-draft-device-heads.json) at
17:21 UTC. Draft gates exercise context processing, the noncausal backbone,
shared output head, target verification after draft execution and context
rollback. The optional-mode gate also covers device fusion and Markov
projections. Real-checkpoint acceptance and throughput are measured separately
from these tiny arithmetic gates.

The [initial real-checkpoint trial](data/nanbeige/compare-initial.json), recorded
at 16:52 UTC, used the prompt “What is the capital of India? Reply briefly.”
Both paths emitted the same ten token IDs, including EOS, and the text
“The capital of India is **New Delhi**.” The prompt occupied 45 tokens; nine
generated tokens belong to decode after the first token from prefill.

| Initial trial, context 128 | Target only | FPGA target + CPU draft |
|---|---:|---:|
| Decode tokens | 9 | 9 |
| Decode wall time | 12.835 s | 18.121 s |
| End-to-end decode throughput | 0.701 tokens/s | 0.497 tokens/s |
| Target FPGA decode time, from cycle counters | 2.639 s | 0.949 s |
| Drafts accepted / verified | — | 7 / 9 (77.8%) |
| Target decode invocations | 9 | 3 |

The FPGA-time row sums only the final nine target-only runs or three speculative
verify runs at 200 MHz; it excludes compilation, transfer/readback and CPU draft
time. Despite fewer target cycles, the hybrid's wall-clock throughput was
**0.708×** the baseline. This is one short, cold/order-dependent trial using a
three-query draft block, not a steady-state speedup result.

The cached runtime was then measured with the full seven-query draft block,
one warmup per path and three measured repetitions at context capacity 128.
The short prompt above generates ten tokens including EOS, nine during decode.
The longer prompt, “Explain why the sky is blue in three short sentences.”,
has 46 prompt tokens and produces the same coherent three-sentence answer on
both paths: 61 generated tokens including EOS, 60 during decode, within its
64-token output budget.

Each TPS cell below is **speculative / paired target-only** median end-to-end
decode tokens/s. Speedups use each comparison's own measured target baseline.

| Draft mode | Short TPS | Short speedup | Longer TPS | Longer speedup |
|---|---:|---:|---:|---:|
| CPU FP32 | 4.258 / 2.955 | 1.441× | Not measured | — |
| FPGA backbone, host fusion/Markov (default) | 4.225 / 2.955 | 1.430× | 3.650 / 2.955 | 1.235× |
| FPGA backbone and fusion/Markov projections | 4.026 / 2.944 | 1.368× | 3.483 / 2.946 | 1.183× |

All five comparisons match their target-only token IDs in **all three
repetitions**. Short-prompt acceptance is **7/9 (77.8%)** for every draft mode;
long-prompt acceptance is **37/72 (51.4%)** for both measured FPGA modes. The
FPGA draft executed on hardware in every measured accelerator run. Evidence:
[CPU short](data/nanbeige/compare-runtime-cpu.json),
[default FPGA short](data/nanbeige/compare-runtime-fpga.json),
[default FPGA longer](data/nanbeige/compare-runtime-fpga-long.json),
[board fusion/Markov short](data/nanbeige/compare-device-heads-short.json), and
[board fusion/Markov longer](data/nanbeige/compare-device-heads-long.json).

The default FPGA draft is close to, and slightly below, the CPU-draft result on
the short prompt. Moving fusion and Markov projections to the FPGA lowers
throughput by approximately 5% on both measured prompts, so the faster host
placement remains the default. Lower acceptance on the longer prompt coincides
with a smaller speedup. The default FPGA mode's median prefill-plus-decode wall
time is 6.646 s versus 7.368 s for its paired short baseline, and 20.911 s versus
24.516 s on the longer prompt.

All comparisons retain target capture writes and execute target before hybrid
on a shared engine. Warmups remove initial compilation cost from measured
repetitions. These are two prompt samples, with nine or 60 decode tokens per
repetition; the initial cold three-query experiment and these seven-query
results are not directly interchangeable. The [provenance record](data/nanbeige/provenance.json)
contains checkpoint revisions, F2 image/profile, dependency versions, thread
settings and runner source hashes. Earlier reports used earlier source overlays,
as identified in that record.

## Capacity and performance interpretation

The current burst200 profile exposes **4 GiB**, two logical 2 GiB memory channels,
at a 200 MHz core clock, with 65,536 TMEM words and 32,768 IMEM words. The target
image at context 128 and four rows is approximately **3.71 GB**; run `inspect`
for the exact allocation and program reservation of a given configuration. The
original BF16 draft checkpoint adds approximately **1.70 GB**, so both do not fit
resident together unchanged in this address space. CPU-draft mode keeps that
draft on the host. Accelerator-draft mode fits by quantizing draft matrices to
INT4 and avoiding duplicate embedding and output-head tables.

The current target image is **3,711,529,472 bytes** at context 128 with four target
rows. The seven-query draft image adds **391,215,616 bytes**, using a padded KV
capacity of 256 for committed context plus temporary query slots. The allocator
also reserves target program/wait storage and rejects overlap or capacity
overflow. Its runtime query program, with three output-head predictions, uses
27,648 instruction words; the context-projection program uses 1,200 words.
Draft and target share one locked device and execute synchronously.

Optional board fusion/Markov adds approximately **92.4 MB** for INT8 projections
and work buffers. Its co-resident image ends at **4,195,039,744 bytes**, leaving
**99,927,552 bytes** within the exposed 4 GiB address space at this configuration.

Benchmark JSON distinguishes target and draft backends, reports token IDs and
decoded text, acceptance and per-iteration verification widths, and compares
the generated tokens exactly. Decode TPS excludes the first token selected
after prefill. Wall-clock timing includes compilation, transfers, readback and
host draft work; setup is reported separately. `target_cycles` sums **prefill and
decode** target runs; the updated reports also separate `target_prefill_cycles`,
`target_decode_cycles`, and `draft_cycles`. Draft cycle counts include enabled
device fusion/Markov projections and exclude all host operations.

The `compare` trials share one capture-enabled target image and run target-only
first, then hybrid. The baseline therefore includes capture writes, and the
hybrid can reuse compilation caches. The updated CLI supports explicit warmups
and repeated measurements, retaining every measured result and its median.
Setup and warmups are excluded from the measured aggregates; compiled programs
are shared across repetitions. The warmed results remain separate from the
initial cold trial.

## Run

Run from the repository root with the existing openTPU dependencies and locally
downloaded checkpoint directories. These commands do not download checkpoints
or reconfigure the FPGA. The board tools acquire the normal exclusive device
lock and require the burst200 architecture, clock and calibrated memory.

```sh
TARGET=/home/ubuntu/rachit/nanbeige-burst200/models/Nanbeige4.2-3B-GPTQ-Int8
DRAFT=/home/ubuntu/rachit/nanbeige-burst200/models/Nanbeige4.2-3B-DSpark
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OTPU_BUILD_JOBS=4

# Metadata/layout compilation only; no checkpoint weight load or hardware run.
python tools/nanbeige_fpga.py inspect \
  --model-dir "$TARGET" --draft-dir "$DRAFT" \
  --cap 128 --draft-tokens 3 --json results/inspect.json

# Deterministic tiny model: FPGA versus ISA logits and captures, including tiled R4.
python tools/nanbeige_hwcheck.py --dev f2:0 --json results/hwcheck.json

# Co-resident draft hardware gate.
python tools/nanbeige_draft_hwcheck.py --dev f2:0 --json results/hwcheck-draft.json

# Optional device fusion/Markov hardware gate.
python tools/nanbeige_draft_hwcheck.py --dev f2:0 --device-fusion-markov \
  --json results/hwcheck-draft-device-heads.json

# Default accelerator draft, with the faster measured host fusion/Markov placement.
python tools/nanbeige_fpga.py benchmark \
  --model-dir "$TARGET" --draft-dir "$DRAFT" \
  --backend board --device f2:0 --draft-backend board --mode compare \
  --draft-fusion-markov host \
  --cap 128 --draft-tokens 3 --draft-query-tokens 7 \
  --prefill-chunk 4 --max-new-tokens 16 \
  --warmup-runs 1 --repeat 3 \
  --prompt 'What is the capital of India? Reply briefly.' \
  --json results/compare.json

# Optional FPGA fusion/Markov projections, demonstrated on the longer prompt.
python tools/nanbeige_fpga.py benchmark \
  --model-dir "$TARGET" --draft-dir "$DRAFT" \
  --backend board --device f2:0 --draft-backend board --mode compare \
  --draft-fusion-markov board \
  --cap 128 --draft-tokens 3 --draft-query-tokens 7 \
  --prefill-chunk 4 --max-new-tokens 64 \
  --warmup-runs 1 --repeat 3 \
  --prompt 'Explain why the sky is blue in three short sentences.' \
  --json results/compare-device-heads-long.json
```

`--mode target` runs the target-only baseline without enabling DSpark captures;
`--mode hybrid` runs only the speculative path. Set `--draft-backend cpu` to
compare the FP32 CPU draft instead. `--backend isa` is explicitly a
simulator run and must not be reported as FPGA TPS. In `compare` mode a token
mismatch returns exit status 2. The hardware check writes evidence on success
or failure and has no software fallback for the FPGA side.

Local correctness suite: **83 tests passed**.

```sh
python -m pytest -q tests/test_nanbeige.py tests/test_nanbeige_weights.py \
  tests/test_dspark.py tests/test_dspark_fpga.py tests/test_nanbeige_specdec.py \
  tests/test_nanbeige_cli.py tests/test_nanbeige_hwcheck.py
```

These cover packed weight conversion, shared physical storage and independent
KV, loop normalization, sequential/tiled verification agreement, selected taps,
greedy acceptance/rejection rollback, EOS, and explicit backend reporting.

## Source references

- [Target checkpoint](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-GPTQ-Int8)
- [DSpark checkpoint](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-DSpark)
- Official [Nanbeige vLLM implementation](https://github.com/Nanbeige/vllm/tree/nanbeige42),
  inspected at `62f6de733d7ae63b759329993bc209e67afdf431`, including the target,
  Qwen3 DSpark/DFlash models, speculative worker and auxiliary-layer mapping.
