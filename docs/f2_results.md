# openTPU on AWS F2: results (2026-10-15)

Every table of the F2 work in one place. The methodology and reproduction commands are in
[`docs/fpga_inference.md`](fpga_inference.md) and [`docs/f2.md`](f2.md).

## Images

| Image | AGFI | Core / TMEM | Memory | Status |
|---|---|---|---|---|
| Alex's (this tree's f2 target, BUILD_ID `7e42a626`) | `agfi-0fb51b57637d7fac0` | 200 / 400 MHz | HBM, both stacks, behind a SmartConnect (450 MHz AXI) | smoke pass, benchmarked |
| This branch's (build `f9e0e5bf`) | `agfi-0014449e89d7fceee` | 100 / 200 MHz | DDR4 DIMM via `otpu_f2_nch` / `otpu_f2_mem` / `sh_ddr` | smoke pass, benchmarked |

Build `f9e0e5bf`'s routed timing: setup +0.045 ns, hold +0.009 ns, all 14 bus-skew constraints
met, no DRC or route failures.

## Setup

**Host.** `opentpu-f2` (f2.6xlarge, us-east-1), slot 0. Inference runs on the card only: the
resident programs and the card's sampler.

**Decoding.** Greedy (temperature 0), no thinking, top-k ≤ 64, each model's chat template.

**Prompts.**
- *Short:* `Write one sentence about the Moon.` (16–19 tokens).
- *Long:* `Summarize the following passage in two sentences.` followed by about 250 words on
  FPGAs (268–274 tokens; `tools/f2_bench.py` `LONG`).

**Checkpoints.**

| Model | HF repository | Revision |
|---|---|---|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | `c1899de2` |
| Gemma-3-1B | `google/gemma-3-1b-it` | `dcc83ea8` |
| Ternary-Bonsai-1.7B | `prism-ml/Ternary-Bonsai-1.7B-unpacked` | `3aca8400` |
| Ternary-Bonsai-4B | `prism-ml/Ternary-Bonsai-4B-unpacked` | `4485fae7` |

## 1. Correctness (`tools/fpga_inference_check.py`, Alex's image)

The run feeds the chat prompt token by token and generates 8 tokens. It compares the board's
logits with Hugging Face fp32 for the same history, and bit for bit with the ISA simulator.

| Model | Weights / head | Image | FPGA output (8 tokens) | HF fp32 argmax | ISA bit-exact |
|---|---|---:|---|---:|---|
| Qwen3-0.6B | int8 / int8 | 1.29 GB | The Moon is a natural satellite of Earth | 8 / 8 | yes |
| Gemma-3-1B | int8 / int8 | 1.08 GB | The Moon is a celestial body that orbits | 8 / 8 (cos 0.9993–0.9997) | yes |
| Ternary-Bonsai-1.7B | int8 / int8 | 3.05 GB | The Moon is the only natural satellite of | 8 / 8 | yes |
| Ternary-Bonsai-1.7B | int4 / int4 | 2.23 GB | the same tokens | 8 / 8 | yes |
| Ternary-Bonsai-4B | int4 / int8 | 3.98 GB | The Moon is Earth's only natural satellite | 8 / 8 | yes |

## 2. Serving baseline (`tools/f2_bench.py`, Alex's image, before optimization)

`otpu-serve --backend board`, context 2048, batched prefill, one warm-up request, at most 128
tokens; Bonsai-4B with `--slots 1`.

| Model | Load | TTFT short | TTFT long | Prefill long | Decode |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B int8 | 13.3 s | 2.16 s | 9.86 s | 27.8 tok/s | 10.0 tok/s |
| Gemma-3-1B int8 | 11.2 s | 4.69 s | 28.4 s | 9.4 tok/s | 6.1 tok/s |
| Ternary-Bonsai-1.7B int4 | 30.4 s | 2.32 s | 14.4 s | 19.0 tok/s | 6.7 tok/s |
| Ternary-Bonsai-4B int4 | 39.7 s | 2.25 s | 55.4 s | 4.9 tok/s | 2.7 tok/s |

## 3. Host TTFT optimization (Alex's image, before → after)

These commits change host code only (`18da545b`..`107bd2c1`):

- contexts move between KV slots on the card instead of being read back over BAR4;
- unavoidable saves are read in the background;
- prompt programs are compiled ahead in worker processes, with a disk program cache.

Outputs are identical, and decode is unchanged.

| Model | TTFT short | TTFT long | Prefill long |
|---|---:|---:|---:|
| Qwen3-0.6B int8 | 2.16 → **0.57 s** (3.8×) | 9.86 → **6.36 s** (1.55×) | 27.8 → 43.1 tok/s |
| Gemma-3-1B int8 | 4.69 → **0.55 s** (8.5×) | 28.4 → **9.60 s** (3.0×) | 9.4 → 27.9 tok/s |
| Ternary-Bonsai-1.7B int4 | 2.32 → **0.88 s** (2.7×) | 14.4 → **9.81 s** (1.5×) | 19.0 → 27.9 tok/s |
| Ternary-Bonsai-4B int4 | 2.25 → **1.34 s** (1.7×) | 55.4 → **15.2 s** (3.6×) | 4.9 → 18.0 tok/s |

## 4. This branch's own image (100 MHz, DDR), with the host TTFT changes

| Model | TTFT short | TTFT long | Prefill long | Decode |
|---|---:|---:|---:|---:|
| Qwen3-0.6B int8 | 0.48 s | 5.51 s | 49.7 tok/s | 11.5 tok/s |
| Gemma-3-1B int8 | 0.58 s | 5.16 s | 51.9 tok/s | 7.0 tok/s |
| Ternary-Bonsai-1.7B int4 | 0.80 s | 10.06 s | 27.2 tok/s | 7.6 tok/s |
| Ternary-Bonsai-4B int4 | 1.66 s | 22.5 s | 12.2 tok/s | 3.1 tok/s |

## 5. Decode tokens/s on the FPGA, all runs

| Model | Bytes per token | Alex's image, baseline | Alex's image + TTFT | Our image + TTFT |
|---|---:|---:|---:|---:|
| Qwen3-0.6B int8 | 613 MB | 10.0 | 10.0 | **11.5** (+14%) |
| Gemma-3-1B int8 | 1015 MB | 6.1 | 6.1 | **7.0** (+14%) |
| Ternary-Bonsai-1.7B int4 | 932 MB | 6.7 | 6.6 | **7.6** (+14%) |
| Ternary-Bonsai-4B int4 / int8 head | 2357 MB | 2.7 | 2.7 | **3.1** (+14%) |

Decode is memory-bound: tokens/s follows the bytes each token streams. Our image's weight
stream runs at about 7.0 GB/s, Alex's at about 6.3 GB/s.

## 6. Decode profile (Alex's image)

Cycles per token come from the card's counters; bytes from `tools/dataflow.py`. The fit is
cycles = 0.67 M + bytes / 31.93 B, within 0.3% for all five images.

| Model | MB / token (layers, head, scales, KV) | Cycles / token | GB/s | Not streaming | At port B (25.6 GB/s) | Gap |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B int8 / int8 | 612.8 (440.4, 155.6, 7.4, 8.4) | 19.90 M | 6.16 | 0.74 M (3.7%) | 41.8 tok/s | 4.16× |
| Gemma-3-1B int8 / int8 | 1015.0 (697.8, 302.0, 11.7, 1.9) | 32.46 M | 6.25 | 0.73 M (2.2%) | 25.2 tok/s | 4.09× |
| Bonsai-1.7B int4 / int4 | 932.3 (704.6, 155.3, 62.8, 8.4) | 29.85 M | 6.25 | 0.70 M (2.4%) | 27.5 tok/s | 4.10× |
| Bonsai-1.7B int8 / int8 | 1747.4 (1409.3, 310.6, 17.9, 8.4) | 55.34 M | 6.32 | 0.70 M (1.3%) | 14.7 tok/s | 4.05× |
| Bonsai-4B int4 / int8 | 2356.5 (1816.7, 388.3, 139.3, 10.8) | 74.52 M | 6.32 | 0.84 M (1.1%) | 10.9 tok/s | 4.05× |

Everything other than the weight stream is 1.1–3.7% of a token: that is the upper bound for
decode-side kernel fusion on this image. The weight stream's 32 B per cycle sets everything
else, and its cause in the HBM / SmartConnect path is still open.

## 7. Prefill optimizations, simulated (RTL co-simulation, not yet on the FPGA)

Both use the f2 config and a 32 B / cycle memory model at 200 MHz. Bits are unchanged.

| Model | Change | Long-prompt device prefill | Per row |
|---|---|---:|---:|
| Qwen3-0.6B int8 | tiled feed runs for batched slots (`f2-opt-prefill-qwen`) | 5.8 → **2.0 s** | 299.3 K → 99.5 K cycles |
| Ternary-Bonsai-1.7B int4 | the same | 9.1 → **4.6 s** | 471.8 K → 235.6 K cycles |
| Ternary-Bonsai-4B int4 | compile only what runs; one KV append per head | host compile 26.7 → 15.1 s (bucket 1), 36.2 → 7.6 s (bucket 2) | −0.4% device |
| Gemma-3-1B int8 | 16-row tiles and wide attention tiles (`f2-opt-prefill-gemma`) | 3.58 → **2.40 s** | 2.588 M → 1.765 M cycles |

## 8. In progress

- **Fused decode megakernel with compute / memory overlap:** keep more weight reads in
  flight, fuse the epilogues.
- **Speculative decoding:** several drafted tokens verified per weight pass.
- **Fewer bytes per token:** int4 heads and layers where accuracy holds.
- **Rebuild of this branch's image at a 200 MHz core,** with more reads in flight in the
  bridge.
