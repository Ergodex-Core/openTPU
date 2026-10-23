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

## Prefill on the f2 target (2026-10-15)

The card measured Gemma-3-1B int8 / int8 head at 9.4 prompt tokens/s (TTFT 28.4 s for the
268-token bench prompt, 4.69 s for the 16-token chat prompt; docs/fpga_inference.md), about a
third of Qwen3-0.6B's. This section is the device side: RTL co-simulation of the tiled prompt
runs (`tools/perf_gemma_prefill.py`) on the f2 target's configuration (one slice, D = 128,
MCOLS 4, ACT_ROWS 16, 4K-instruction IMEM) with the memory a port of 32 B a core cycle
(`--bw 25`, the f2 image's DDR bridge as decode measures it) and 60 cycles' latency.

### Where a prompt run's cycles went

Before (12 rows a run, the first 6 layers: 5 sliding and 1 global, bucket 1, cycles of the
run):

| phase | cycles | share |
|---|---:|---:|
| MLP (gate / up / down, GeGLU, chunk QACTs) | 4,610,825 | 64% |
| attention loop (W_q, q norm and RoPE, q.K^T, softmax, P.V) | 1,990,953 | 28% |
| W_o | 227,333 | 3% |
| K / V projections, norms, RoPE, appends | 212,538 | 3% |
| post-norms (`_add_norm`) | 109,834 | 2% |
| embedding gather (one-hot MMs from the int8 head) | 48,042 | 1% |
| total | 7,202,748 | |

The 6 layers' weights are 161 MB, 5.03M cycles at 32 B/cycle, against 3.77M of MXU (12 rows: 3
passes of MCOLS): the run is bound by streaming the weights, once a run. Two things made it
slow:

- **12 rows a run, not 16.** At 16 rows the MLP's [16, 768] chunk activation found no TMEM gap:
  x and the MLP's output ([16, 1152] each), the run's [16, 512] RoPE rows and the gate / up
  pieces held it. A weight pass then served 12 rows where ACT_ROWS allows 16 (at 32 B/cycle
  and 16 rows the MXU's 4 passes and the stream meet).
- **One row a K / V stream, each K row a scale-line request.** G = 4 heads a KV head is MCOLS,
  so a packed tile was one row (QPART: 2 rows of 2 heads). Gemma's K rows are 2048 B apart
  (`gemma4._KV`: the RoPE row's stride, the row's int8 scale row after its data), so q.K^T
  over a 256-token block makes 256 port-A scale-line requests, ~73 cycles each, one at a
  time: 18.8K cycles for 64 KB, where P.V over V^T's same bytes (no scales) takes 1.9K. Every
  row of the run paid that again for every block of every layer.

What it did not cost: no fallback paths (every bucket's runs are tiled, the attention in
hardware loops at level 2, the outputs whole through the I/O scratch); the LM head runs only
on an L run's last row, and the server's prompts end in P runs (the decode loop feeds the last
token); the device embedding gather is 1%; at 268 tokens the sliding window still reaches
position 0 (no ring wrap, the same spans as the global layers').

### What changed

1. 16-row tiles (`9c0304db`): x out of TMEM while the MLP runs (spilled to the I/O rows,
   back for the post-norm's add), a layer kind's RoPE rows loaded by its attention, the
   MLP's gate / up in pieces of 384 or 256 columns where TMEM holds them (GEMMA_SUBS, fewer
   instructions), `_add_norm` in groups of rows, and `tiled._sync_frames`: on Python 3.12,
   `compiler.unnamed`'s `f_locals` reads leave snapshot dicts on the frames they walk, which
   kept dead tiles (gate / up pieces, K / V rows) alive.
2. Wide tiles (`3620594b`): the looped attention's packed tiles take up to ACT_ROWS query
   rows (16: 4 rows of all 4 heads), the MXU replaying K and V^T for each MCOLS of them, the
   row maxima (RMAX) and the accumulator's rescale (ASCALE) on the VPU where M > MCOLS (the
   same roundings: the same bits). 4 K / V streams a layer and block where there were 12.

Every row's arithmetic is unchanged: the tiled runs equal today's prompt runs bit for bit
(test_gemma3 test_tiled_prompt_runs_are_todays; on the real checkpoint, the chat prompt and
the 268-token prompt: the logits, the layers' DRAM and 8 greedy tokens).

### Measured

The first 6 layers, cycles a row (the whole model's estimate: the layers' cycles scaled to
26):

| bucket | before (12 rows) | 16 rows | + wide tiles | whole model a row | tok/s at 200 MHz |
|---|---:|---:|---:|---|---|
| 1 | 600,229 | 507,594 | 409,461 | 2.588M -> 1.765M | 77.3 -> 113.3 |
| 2 | 741,437 | 644,206 | 446,917 | 3.200M -> 1.927M | 62.5 -> 103.8 |

The whole model's runs (26 layers, co-simulated; the server's split: up to R_max rows a run,
cut at a bucket's end):

| run | before | after |
|---|---:|---:|
| bucket 1, 16 rows | | 28,234,823 |
| bucket 1, 15 rows | | 28,030,568 |
| bucket 1, 12 rows | 31,048,199 | |
| bucket 1, 4 rows | 25,105,702 | |
| bucket 1, 3 rows | 24,475,609 | |
| bucket 2, 11 rows | 38,277,077 | 27,713,134 |

| prompt | prefill runs before -> after | prefill cycles | at 200 MHz | TTFT (+ one decode step, 32.5M) |
|---|---|---:|---:|---:|
| chat, 16 tokens (15 prefill) | 12 + 3 -> 15 | 55.5M -> 28.0M | 0.278 -> 0.140 s | 0.44 -> 0.30 s |
| bench, 268 tokens (267) | 21 x 12 + 4, 11 -> 16 x 16, 11 | 715.4M -> 479.5M | 3.58 -> 2.40 s | 3.74 -> 2.56 s |

Device prefill of the long prompt: 74.6 -> 111.4 tokens/s. Qwen3-0.6B's whole-model 16-row run
in bucket 1 on the same setup is 22,688,390 cycles (1.418M a row): Gemma 3's rows cost 1.82x
Qwen's before and 1.24x now, for 1.58x the weight bytes a row (697 MB against 440 MB). The card's 28.4 s and 4.69 s TTFT
are far above the device's 3.7 s and 0.44 s: most of Gemma 3's measured TTFT is outside the
device's prefill (the host: the prompt programs' compile and fit probes, transfers), the
host / TTFT track's.

A run is now the MLP (4.9M of 6.55M cycles in bucket 1's 6 layers, 91% of its 32 B/cycle
roofline) and the attention loop (0.94M, 1.55M in bucket 2). What is left there is K's
scale-line latency, now once a tile of 4 rows: a K cache with its scale rows apart (as
qwen3's KVDesc), 8 tokens' scales a 32-byte line, would cut it about 4x more (and decode's
attention with it); that is the image's KV layout, shared with decode, not changed here.
The V^T column writes of the appends (one QST of 256 bytes a token, ~1.2K cycles each) are
next: about 20K cycles a layer.

To reproduce (on a host with Verilator 5.048):

```sh
python tools/perf_gemma_prefill.py --model gemma3 --layers 6 --rows 16 --blocks 1
python tools/perf_gemma_prefill.py --model gemma3 --layers 0 --rows 16 --blocks 1   # 26 layers
```

On the card (not run here): `python tools/f2_bench.py --dev f2:0 --models gemma3-1b --json
bench.json --max-tokens 128`.
