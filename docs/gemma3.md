# Gemma 3 and FunctionGemma

`opentpu/llm/gemma3.py` runs Gemma 3 1B and FunctionGemma 270M through the existing
`qwen3.Engine`, ISA simulator and `BoardBackend`. It reuses Gemma 4's image layout,
layer loops, sliding KV rings, device embedding gather, decode and tiled prompt kernels.

| Checkpoint | Layers | Hidden | Q / KV heads | Head width | MLP width |
|---|---:|---:|---:|---:|---:|
| `google/gemma-3-1b-it` | 26 | 1152 | 4 / 1 | 256 | 6912 |
| `mlx-community/functiongemma-270m-it-bf16` | 18 | 640 | 4 / 1 | 256 | 2048 |

Both have a tied 262144-token embedding / output table, four RMSNorms per layer,
GeGLU, and five sliding layers followed by one global layer. Sliding attention includes
the current token in its 512-token window; local RoPE theta is 10000, global theta
1000000, with every pair rotated. RMSNorm uses `1 + weight`; Q and K are normed,
V is not. Scores are divided by 16. There is no PLE, KV sharing or layer scalar.
The generation config's stop IDs override the model config's, as in Hugging Face
(FunctionGemma's two files differ).

The reference is fp32 math, including the embedding's `sqrt(hidden)` multiplier.
The device gathers its embedding from the quantized tied head and uses int8 activations
and KV caches. These numerics do not reproduce Hugging Face's bf16 rounding at every
operation. Matching a quantized model's greedy output to an unquantized checkpoint is
a measurement, not an invariant.

## Checkpoints and execution

The short names resolve to directories under `models/`:

| Name | Directory | Source |
|---|---|---|
| `gemma3` | `gemma-3-1b-it` | HF safetensors |
| `gemma3-int4` | `gemma-3-1b-it-int4` | one GGUF plus the HF config/tokenizer files |
| `functiongemma` | `functiongemma-270m-it` | HF or MLX safetensors |

The lazy loader accepts `model.*`, unprefixed MLX names, and language-model prefixes.
GGUF accepts Q4_0, F32 and F16 tensors. Its norm tensors contain the effective gain;
the loader removes one so the same model code handles both formats. Q4_0 is decoded
exactly (signed fp16 scales, split nibble halves, codes -8 through 7), then quantized
into the selected openTPU format. This is not a lossless GGUF-to-MXU repack: their
scale encodings differ. Unsupported GGUF tensor types and scaled RoPE are refused.
The GGUF used in the measurements is `gaunernst/gemma-3-1b-it-int4-gguf`, whose model
card identifies it as a QAT checkpoint different from Google's other QAT releases.

The Python model tools need `torch`, `transformers`, `safetensors` and `tokenizers`;
GGUF loading also needs `gguf`. With checkpoints in those directories:

```sh
otpu-chat --model gemma3 --backend isa --wformat int4 --greedy --max-new 16
otpu-chat --model functiongemma --backend isa --wformat int4 --greedy --max-new 16
```

On an F2 host with the matching accelerator image loaded and the host's transport
permissions configured:

```sh
otpu-chat --model gemma3-int4 --backend board --dev f2:0 --wformat int4 --greedy --max-new 16
otpu-chat --model functiongemma --backend board --dev f2:0 --wformat int4 --greedy --max-new 16
```

`otpu-serve --model gemma3` and `tools/perf_qwen.py --model gemma3` use the same generic
model registration. `--layers 0` in `perf_qwen.py` compiles all layers. No FPGA result
is claimed here; these model changes have been checked on the ISA simulator.

## Measurements

On 2026-10-14, the card configuration (one slice, D=128), int4 layers and head,
capacity 1024, one decode row, resident decode bucket 1:

| Model | Image bytes | Program words | IMEM capacity |
|---|---:|---:|---:|
| Gemma 3 1B | 664568448 | 6304 | 32768 |
| FunctionGemma 270M | 228125312 | 4400 | 32768 |

Program sizes count uint32 words (eight per instruction).

`tools/gemma3_check.py` reports per-step argmax agreement and logit cosine against
the same checkpoint's fp32 reference. It teacher-forces the device's chosen history
into the reference; agreement after a divergence is not independent sequence agreement.
`--chat` applies the checkpoint's chat template; `--tools FILE` supplies its tool
schema; `--emulated` additionally compares the quantization-aware emulation.
`tools/validate.py` refuses Gemma 3 until its separate HF golden hooks are ported.

```sh
python tools/gemma3_check.py --model gemma3-int4 --wformat int4 --steps 3
python tools/gemma3_check.py --model functiongemma --wformat int4 --steps 3
```

The default plain prompt `Hello` (after BOS), three generated tokens:

| Source | Argmax agreement with fp32 | Logit cosine by step |
|---|---:|---|
| Gemma 3 QAT Q4_0 GGUF, then MXU int4 | 2 / 3 | 0.98271, 0.99691, 0.99821 |
| FunctionGemma bf16 weights, then MXU int4 | 0 / 3 | 0.98508, 0.96649, 0.93647 |

These measurements do not establish token-exact reproduction of the original
checkpoints. The tiny-model tests separately check the fp32 reference against HF,
including the sliding-window boundary; int8 and int4 ISA execution against emulation;
tiled prompt runs against decode; MLX names, GGUF loading and real model compilation.

The full FunctionGemma checkpoint's fp32 reference was also compared with
transformers 5.19.0 / torch 2.14.1 on tokens `[2, 9259]`: maximum absolute logit
error 0.000124, minimum cosine 0.99999988, both argmax positions equal.

A function-call probe used the checkpoint's chat template, a no-argument
`get_current_time` function, and `What is the current time?` (48 prompt tokens).
The fp32 reference selected `<start_function_call>` first. The all-int4 ISA run
instead began `I cannot assist`, with 1 / 3 teacher-forced argmax matches. Using
an int8 head still missed the initial call token. The quantization-aware emulation
also diverged from ISA argmax at two of these three positions (cosines 0.99877,
0.98337, 0.98503). Function-call correctness and token-exact real-model agreement
are therefore not established for these int4 configurations; the architecture /
transport integration is usable, but int4 needs further numerical work.

## Experimental device prompt lookup

`Engine(..., resident=True, rows=2, gemma_prompt_lookup=True)` reserves a
separate history and two chain programs per attention bucket. Wrap that engine
with `opentpu.llm.gemma_specdec.GemmaPromptLookupDecoder` and call
`generate_from_unfed_prompt(prompt_ids, n, stop_ids=...)` on a fresh engine.
The final prompt token is fed by the device loop, which also selects the first
output. The host loads the prompt and receives accepted token IDs; it does not
draft tokens or select logits.

The greedy-only proposal is the continuation of the latest earlier occurrence
of the current token. One two-row run verifies that proposal with the target
model. It publishes its second output only if the draft matches the first
target pick, the first pick is not a stop ID, and two outputs remain. Drafts
stay in private history rather than the streaming output buffer. A single-row
program handles misses, the final bucket position, and a one-token budget.
It also handles a proposed stop ID: verification could not produce a second
output whether that proposal matches or fails. This rule does not use measured
acceptance rates or prompt-specific thresholds.
Host stop requests take effect after the current one- or two-output iteration.

This path has separate ISA numerical/control tests. Instruction-memory fit is
checked before execution; `tools/gemma_specdec_compile_check.py` can check the
exact configuration without reading weights. ISA equivalence and compiler fit
do not establish RTL timing, on-card correctness, acceptance rate, or throughput.
The existing Gemma1B hardware measurements did not use this experimental path.

To reproduce that function-call comparison, save this as `time-tools.json`:

```json
[{"type":"function","function":{"name":"get_current_time","description":"Get the current time.","parameters":{"type":"object","properties":{},"required":[]}}}]
```

```sh
OTPU_BUILD_JOBS=1 OTPU_IMAGE_CACHE=0 OTPU_IMG_CACHE=0 python tools/gemma3_check.py \
  --model functiongemma --wformat int4 --chat --tools time-tools.json \
  --prompt 'What is the current time?' --steps 3 --emulated --json functiongemma-check.json
```

A bounded intermediate-tensor diagnostic localized the first material ISA/emulation
drift in this prompt to token position 7, layer 8 (both zero-based). The scaled
embedding agreed within 3.6e-8 relative L2 error; residuals through all 18 layers
of positions 0–6 agreed to approximately 1e-7. Unrolling the layer program preserved
the three original ISA token choices, so layer-loop addressing did not explain them.

At position 7 / layer 8, just one of 1024 query quantization codes differed: head 0,
channel 249 was 15 on the ISA and 14 in the float64 emulation. The emulation's ideal
quantization ratio was 14.4999997800, immediately below a rounding boundary. The
pre-quantization query vectors differed by only 6.60e-8 relative L2 error. Quantizing
the device query without its attention scale produced the same codes, ruling out
the scale placement as the cause of this particular discrepancy. The K group
offsets and V integer codes matched; the probability-times-V-scale quantization
codes also matched. Recomputing attention with the device's captured query and KV
values reproduced its output within 1.30e-7 relative L2 error, versus 0.00104 for
the original emulation.

This identifies float32-versus-float64 rounding across a quantization boundary as
the source of that first material drift. Subsequent cached attention and MLP steps
amplify the difference. The emulation is quantization-aware, not a bit-exact oracle
for every floating-point operation. This diagnostic does not establish full-model
correctness or show that every later difference has the same cause; the failed
int4 function-call comparison remains unresolved. No model arithmetic or test tolerances
were changed in response to it.

A follow-up using **int8 for every layer and the head** selected the correct first
token, `<start_function_call>` (ID 48), on the same 48-token prompt. The ISA and
fp32 reference argmax matched, with logit cosine 0.99872047. The image occupied
323134080 bytes. Extending that probe through `<end_function_call>` (ID 49) produced
the complete no-argument call:

```text
<start_function_call>call:get_current_time{}<end_function_call>
```

All 10 generated tokens matched the teacher-forced fp32 reference, with minimum
logit cosine 0.99787611. Since there was no divergence, this is also full greedy
sequence agreement for this probe. This verifies the call syntax, name, empty
arguments and terminator for this particular prompt and checkpoint; it does not
establish general tool-use accuracy or resolve the int4 failure. The full comparison
build took 3.3 seconds and the run took 44.4 seconds; its wall time includes ISA
simulation and repeated fp32 reference evaluation and is not a hardware throughput
measurement. Reproduce it with:

```sh
OTPU_BUILD_JOBS=1 OTPU_IMAGE_CACHE=0 OTPU_IMG_CACHE=0 python tools/gemma3_check.py \
  --model functiongemma --wformat int8 --head-format int8 \
  --chat --tools time-tools.json --prompt 'What is the current time?' \
  --steps 16 --stop-token 49 --json functiongemma-int8-check.json
```

The existing Gemma 4 `test_resident_gathers_are_bit_exact[fp4]` fails before model
computation with a 33280-word gather tile and 33537 of 65536 TMEM words already in use.
Compiling the same fixture at unmodified HEAD reproduces exactly that failure.
The Gemma 3 suite passed 11 tests. A bounded regression run of Gemma 4, quantization
and Gemma 4 prefill completed 88 tests with 3 missing-checkpoint skips and no new
failure before interruption (7 cases excluded: resident gathers, long ring wrap,
and board RTL). The remaining long prefill cases were not completed.
