# Gemma 3 on openTPU (gemma-3-1b-it)

openTPU runs Google's Gemma 3 1B, checked with
[gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it), on Gemma 4's code
(`opentpu/llm/gemma4.py`, [gemma4.md](gemma4.md)): its DRAM image, kernels, resident decode,
chunked prefill and Engine. `opentpu/llm/gemma3.py` only reads the config and folds the
parameters Gemma 3 states differently. No new hardware op, no new ISA; the same bitstream and
W8A8 / W4A8 numerics as the other models.

```sh
hf download google/gemma-3-1b-it --local-dir models/gemma-3-1b-it      # MODELS' gemma3
otpu-chat --model gemma3 --backend isa                                  # the simulator
otpu-chat --model gemma3 --backend board                                # the card
python3 -m pytest -q tests/test_gemma3.py                               # tiny random model
python3 tools/compare_hf.py --model gemma3 --chat --cfg board --resident --tokens 12
python3 -m opentpu.lens record model --model gemma3 --isa -o g3.otpuprof   # one decode token
python3 -m opentpu.lens record model --model gemma3 --axi -o g3-rtl.otpuprof   # on the RTL
```

## The model

Gemma 3 1B has 26 layers, hidden size 1152, a 262,144-token vocabulary and a tied LM head (the
embedding table, scaled by sqrt(1152) on input).

- **Attention.** 4 query heads on 1 KV head, 256 wide, RMSNorm on each q and k head (not on
  v), scores scaled by `query_pre_attn_scalar^-1/2` = 1 / 16. Five layers of every six are
  sliding (the last 512 tokens, the current one included; RoPE theta 1e4), the sixth is global
  (all tokens, RoPE theta 1e6 over the whole head). Every layer has its own K / V.
- **MLP.** GeGLU, `gelu_tanh(W_gate x) * (W_up x)`, 6912 wide.
- **Norms.** `x * (1 + w)` (Gemma 4's are `x * w`) before and after every block.
- No per-layer embeddings, no KV sharing, no layer scalars, no logit soft cap (the 1B's
  `final_logit_softcapping` is null; a checkpoint with one gets Gemma 4's cap).

## How it maps

Gemma 3 is a Gemma 4 `Spec` with `ple_dim` 0, `kv_src` the identity, `global_rot` =
`head_dim / 2` (the whole head rotated, as the sliding layers), and two fields added for it:

| Spec field | Gemma 3 | What it does |
|:--|:--|:--|
| `v_norm` | False | V is the projection as it is (Gemma 4 norms it without a weight): `_attention`, the tiled attention, `reference_logits`, `emulated_logits` |
| `global_rope_div` | 1 (the 4B+ checkpoints: 8) | the global layers' RoPE frequencies divided (Hugging Face's linear scaling) |

The rest is folded into the weights as `gemma4.Weights` reads them when the checkpoint's
`model_type` is `gemma3_text` / `gemma3` (`gemma3.folder`; `gemma3.fold_weights` for a state
dict in memory, as the tests'):

- every norm weight is read as `1 + w`, in fp32, exactly as Gemma3RMSNorm computes it;
- q_norm's gain also takes the attention scale 1 / 16: the scale commutes with RoPE, and the
  query's int8 rounding is per 128-block with its own scale, so the device's attention is
  Hugging Face's up to fp32 rounding;
- `layer_scalar` (which Gemma 3 has not) is 1.

The layers run as two hardware loops, (5 sliding + 1 global) x 4 with the five sliding layers
a loop inside, then 2 sliding layers; the K / V of the sliding layers in rings of 768 slots,
the global layers' over the capacity. The int8 image with the resident decode's tables at a
2048-token capacity is 1.03 GiB (the KV caches 55 MiB); the embedding rows are gathered on the device from the int8
LM head, as Gemma 4's.

The numpy reference (`gemma4.reference_logits` on the folded weights) matches Hugging Face's
fp32 Gemma3ForCausalLM (transformers 5.20) to 7.5e-6 in the logits over 600 random tokens of
a tiny model, past its 512-token window (`tests/test_gemma3.py`).

## On the ISA simulator

`otpu-chat --model gemma3 --backend isa --greedy --prompt "Why is the sky blue? Answer in one
sentence."` (2026-10-18, int8 weights and head, the card's configuration):

> The sky appears blue because of a phenomenon called Rayleigh scattering, where sunlight is
> scattered by the tiny air molecules in the atmosphere

The image builds in 8 s; the simulator runs the 20-token prompt at 1.0 tok/s and decodes at
0.5 tok/s on a laptop.

## Accuracy

Greedy decoding of the real model on the ISA simulator (the card's configuration, resident
decode, int8 weights and head) against Hugging Face's greedy decoding in fp32, each prompt a
user turn of the chat template, 12 tokens (`tools/compare_hf.py --model gemma3 --chat --cfg
board --resident --tokens 12`, 2026-10-18). The logit error is over the steps with the same
context.

| Prompt | Same tokens | First difference: HF's rank of the device's token, logit gap | Max logit error | Min cosine |
|:--|:--|:--|--:|--:|
| Why is the sky blue? | 0 | #2, 0.609 (`The` / `Okay`) | 2.27 | 0.9987 |
| Write a haiku about the ocean. | all 12 | | 3.46 | 0.9987 |
| What is 17 times 3? | all 12 | | 4.70 | 0.9939 |
| def fibonacci(n): | all 12 | | 3.84 | 0.9942 |

The one difference is a near tie at the first token (HF's second choice, 0.6 logits below the
top on a span of 73), and the device's continuation is fluent ("Okay, let's break down why the
sky is blue"). The errors are the W8A8 quantization's, in the range of the other models'
([llama.md](llama.md), [gemma4.md](gemma4.md)).

## Lens

`lens record model` (added with this model) records one decode token of any checkpoint after a
prompt: the prompt as a user turn of the chat template, prefilled and decoded greedily on the
ISA simulator up to `--model-pos` (default the prompt's last token), the step there traced on
the RTL (the Verilator board model, the card's configuration for a one-slice model) or, with
`--isa`, on the ISA simulator with analytic timing. The resident decode program, as otpu-chat
runs it (`--per-position` for the per-position programs).

Gemma 3 1B's first decode token (position 14 after the chat template's 15-token prompt, int8
weights and head, the resident decode program, 5,350 instructions; 2026-10-18):

| Run | Cycles | At 133.33 MHz | Of the DRAM roofline (7.85 M cycles) |
|:--|--:|--:|--:|
| ISA simulator, analytic timing (`--isa`) | 8.28 M | 62.1 ms | 94.8% |
| RTL, the board model with the AXI DDR3 path (`--axi`) | 10.28 M | 77.1 ms, 13.0 tok/s | 76.4% |

On the RTL, DRAM port B streams 76.4% of the cycles. Of the rest, 18.2% is the MXU not
consuming (drain / row credits), 3.9% the MXU computing between streams, 0.6% the chunk FIFO
starved; the MXU drain lost 2.1% to TMEM bank arbitration. The MXU is in flight 100% of the
run, the VPU 4.5% (gelu_tanh's and the norms' VOPs: the hottest source lines), the quantizer
1.2%. The profiles are `build/lens/gemma3-decode-{rtl,isa}.otpuprof` on the build host.
Split by cause ([v4/m0-gating-split.md](v4/m0-gating-split.md), the Q line's `mbc` / `mbr` /
`mbs` / `mbo`): 97% of the "not consuming" cycles are rows whose chunks had not all arrived
with the FIFO under one row deep, the memory model delivering 0.76 chunks a cycle below the
port's one; 2.6% scale entries behind their chunks; 0.1% result-FIFO credits. With the memory
at the port's rate (`--stall 0`, `OTPU_LAT=30`) the token is 8.22 M cycles, 95.5% of the
roofline, and the item is 2.1%.
