# Qwen3-0.6B and Gemma 3 1B on the FPGA

`tools/fpga_inference_check.py` runs the full checkpoint through `Engine` and
`BoardBackend`. The default backend is the physical board; inability to open it is
an error. `--backend isa` must be specified explicitly for a simulator run.

The requested checkpoints are `Qwen/Qwen3-0.6B` in `models/Qwen3-0.6B` (alias
`qwen3`) and `google/gemma-3-1b-it` in `models/gemma-3-1b-it` (alias `gemma3`).
Gemma's QAT GGUF and FunctionGemma are different checkpoints and are not substitutes
for this comparison. The initial configuration uses int8 weights and an int8 head.

## Runtime and checkpoint preparation

The working Python environment uses numpy 2.5.4, torch 2.14.1, transformers 5.19.0,
safetensors 0.8.0, tokenizers 0.23.3, huggingface-hub 2.2.0 and textual 8.2.8.
Use CPU PyTorch on the FPGA host. The harness loads local files only; checkpoint
download or transfer is a separate preparation step. `pyproject.toml` alone does
not install the optional model libraries above. No GGUF package is required here.

The available official Gemma snapshot is revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`. Its `model.safetensors` is 1999811208
bytes with SHA256
`3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6`.
The required bundle also contains `config.json`, `generation_config.json`,
`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` and
`added_tokens.json`; retaining `tokenizer.model` makes the snapshot portable to
other tokenizer implementations. These eight files total 2039043660 bytes. Transfer
the resolved files, since cache snapshot entries can be symlinks. Keep the snapshot
revision and verify the transfer against a SHA256 manifest.

No Qwen3-0.6B weights were available in the checked local workspace or Hugging Face
cache on 2026-10-14. The FPGA host's subsequent official download is pinned to
revision `c1899de289a04d12100db370d81485cdf75e47ca`; its checkpoint directory contains
`download-manifest.json` with the transferred file hashes.

Before touching a board, inspect each checkpoint's image and compile its first
resident decode program:

```sh
python tools/fpga_inference_check.py --model qwen3 --inspect --json qwen3-layout.json
python tools/fpga_inference_check.py --model gemma3 --inspect --json gemma3-layout.json
```

For the checked Gemma revision and the pinned Qwen configuration, F2 target,
capacity 1024, int8 layers and head:

| Model | Image bytes | First resident program, uint32 words |
|---|---:|---:|
| Qwen3-0.6B | 1293673600 | 3712 |
| Gemma 3 1B | 1083579008 | 6048 |

Program sizes count 32-bit words, eight per instruction: 464 instructions for
Qwen and 756 for Gemma. Both full models fit the 4 GiB model memory and 32768-word
instruction memory. The remote official Qwen checkpoint's `--inspect` confirmed
its layout above. These inspections use the target file; an actual board run
obtains its architecture and available memory from the loaded image's registers
and checks that the model fits.

## Completed software baseline; hardware inference pending

Run `model-baseline-001` evaluated the full official checkpoints above using the
ISA simulator configured for F2, int8 layers and head, context capacity 1024 and
greedy decoding. The exact user prompt was `Write one sentence about the Moon.`
with each model's chat template and Qwen thinking disabled. Qwen's rendered prompt
contained 19 tokens; Gemma's contained 16. The probe stopped after eight generated
tokens. Hugging Face's CPU fp32 eager model evaluated that same token history.

| Model and revision | Generated eight-token prefix | HF argmax matches | Minimum logit cosine | Maximum absolute logit difference |
|---|---|---:|---:|---:|
| Qwen3-0.6B, `c1899de289a04d12100db370d81485cdf75e47ca` | `The Moon is a natural satellite of Earth` | 8 / 8 | 0.9987950159 | 1.38134861 |
| Gemma-3-1B-it, `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | `The Moon is a celestial body that orbits` | 8 / 8 | 0.9992680399 | 2.57377052 |

The evidence is saved in
`build/f2-transfer/model-baseline-001/qwen3-isa.json` and
`build/f2-transfer/model-baseline-001/gemma3-isa.json`. Both reports explicitly say
`backend: isa` and `board_info: null`. The runs took 91.99 and 67.32 seconds,
respectively, including image construction, simulation and HF evaluation on the
CPU; these are not FPGA latency or throughput measurements. The runtime used
torch 2.14.1+cpu, transformers 5.19.0, numpy 2.5.4 and two CPU threads.

There was no token divergence in either measured eight-token prefix. The logits
are not bit-exact to the unquantized HF model, and these short samples do not
establish broad model accuracy, long-context behavior or complete answers. These
software-only baselines do not test the physical FPGA, DDR or PCIe execution.
Actual board inference and the board-versus-ISA comparison remain pending.

For reproduction, use the same environment controls below and replace the board
command's `--backend board --dev f2:0 --compare-isa` with `--backend isa`. The
original baseline used the initial harness; the subsequent failure-report changes
do not alter model arithmetic. The updated remote source's host/model suite
completed 38 tests with two skips in FunctionGemma/GGUF coverage.

## Inference and two separate comparisons

Run after the F2 accelerator image has loaded and the transport self-tests pass.
From the repository root on that host:

```sh
export OTPU_IMAGE_CACHE=0 OTPU_IMG_CACHE=0 OTPU_PROG_CACHE=0
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false

OTPU_BUILD_JOBS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python tools/fpga_inference_check.py --model qwen3 --backend board --dev f2:0 \
  --chat --prompt 'Write one sentence about the Moon.' --steps 8 \
  --compare-isa --json qwen3-fpga.json --save-logits qwen3-fpga.npz

OTPU_BUILD_JOBS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python tools/fpga_inference_check.py --model gemma3 --backend board --dev f2:0 \
  --chat --prompt 'Write one sentence about the Moon.' --steps 8 \
  --compare-isa --json gemma3-fpga.json --save-logits gemma3-fpga.npz
```

`--chat` uses the checkpoint's local chat template and disables Qwen's optional
thinking mode for this short probe. Omit it to use raw text and the tokenizer's
normal special-token handling. `--stop-token ID` adds an explicit stop token;
checkpoint EOS tokens always stop generation. The prompt plus fed generated history
must fit `--cap` (default 1024).

The report makes two different comparisons against the exact same token history:

* `isa_comparison`: the board's fp32 logit words against the ISA simulator using
  the same quantized image and architecture. `isa_bit_exact` requires every word
  to match. The process returns status 2 if this check fails.
* `golden_comparison`: the board against Hugging Face's unquantized CPU fp32 eager
  model. Argmax agreement, logit cosine and maximum absolute error measure model
  quality. Quantization can change the selected token; disagreement is reported
  separately from device correctness. `--golden numpy` selects the repository's
  architectural reference when debugging instead.

The board identity is saved before building/uploading the image. Generated tokens
and raw device logits are checkpointed after each selected token, before any CPU
comparison; the ISA results are saved before the HF comparison. A later failure
leaves `comparison_complete: false`, `phase: failed`, the failed phase and error,
plus all completed token outputs. JSON and NPZ files are replaced atomically, so
a partial write does not destroy the previous checkpoint. The owned transport is
closed before CPU comparisons, including when Engine construction fails. A forced
process kill can only preserve the last completed checkpoint.

Both references then consume the board's chosen
history; a match after an earlier divergence is teacher-forced agreement, not an
independent greedy sequence match. NPZ output retains full logits for investigation.

Timing separates image build/upload, prompt engine calls, subsequent decode engine
calls and board cycle counters. CPU reference and ISA simulation time are included
only in total wall time. Token-by-token prompt feeding and host-selected argmax make
this a correctness probe, not a throughput benchmark. Report/NPZ checkpoint writes
are outside the engine-call timings. An eight-token reply may be unfinished; it is
a bounded inference sample. The two-thread settings above keep CPU references from
using every core while FPGA implementation is running. The checkpoint loader's
cached fp32 tensors are released before HF loads its own model copy.

The harness has fourteen focused passing tests covering tiny Qwen3 and Gemma 3 models,
HF oracle comparisons, exact ISA comparisons, stopping and the board factory/close
path with a simulated test transport. Injected image-build, device-execution, ISA
and HF failures verify that device identity and completed logits survive and the
transport closes; chat identity/configuration probe failures also close the newly
opened transport. An incomplete oracle logit history is rejected instead of
reporting a partial comparison as complete. These tests do not establish physical
FPGA inference. Actual board result files are required for that claim.

## FPGA results and methodology (2026-10-15)

### Setup

* **Instance and image.** EC2 `i-0c78de2253f66519d` (`opentpu-f2`, f2.6xlarge, us-east-1),
  slot 0, AppPF `0000:34:00.0` (`--dev f2:0000:34:00.0`). The image was Alex's
  `agfi-0fb51b57637d7fac0` (`opentpu-200mhz-20261014_42_afi`). It is this tree's f2 target
  (BAR0 reports `ID` OTPU, `BUILD_ID` `7e42a626`, `CFG_PLATFORM` 3, `CFG_HASH` `4ffd1e17`)
  built with a 200 MHz core, 400 MHz TMEM and HBM. This branch's own image,
  `afi-0ee1306df4f9f9ed4` (build `f9e0e5bf`, 100 MHz, DDR), has not been run on hardware yet.
* **Load.** After `fpga-load-local-image`, `fpga-describe-local-image -S 0 -R -H` (a PCI
  rescan) is needed: before it the AppPF shows device `0x9048`, after it the AFI's `0xf001`.
* **Smoke test first.** The full `tools/f2_smoke.py` passed in about 4 s: calibration, BAR4
  round trips on both channels, 200 partial writes, interleave, kernels and masked stores
  against the ISA simulator, and host ordering. It ran with a one-off local copy that also
  accepts `CORE_KHZ` 200000; the tool itself expects the target's 100 MHz.
* **Software.** Python 3.12 venv, numpy 2.5.4, torch 2.14.1+cpu, transformers 5.19.0,
  `OMP_NUM_THREADS=8`. Checkpoints:
  - `Qwen/Qwen3-0.6B` @ `c1899de2`;
  - `google/gemma-3-1b-it` @ `dcc83ea8`;
  - `prism-ml/Ternary-Bonsai-1.7B-unpacked` @ `3aca8400`;
  - `prism-ml/Ternary-Bonsai-4B-unpacked` @ `4485fae7`.
  The Bonsai checkpoints are Qwen3 with YaRN RoPE (factor 4 of 8192): `qwen3.Spec` takes it
  as LongRoPE's divisors and attention factor (`llm.yarn_rope`, tests/test_yarn.py).
* **Device-only inference.** The model is computed and sampled on the card. The host
  compiles token ids into the resident programs, moves bytes and polls registers; the
  sampler is the card's (top-k at most 64), with greedy decoding (temperature 0) and no
  thinking mode. The HF and ISA comparisons below run on the host afterwards, from the
  board's own token history.

### 1. Correctness: `tools/fpga_inference_check.py`

**Prompt.** `Write one sentence about the Moon.` in each checkpoint's chat template
(`--chat`): 19 tokens for Qwen3 and Bonsai, 16 for Gemma 3.

**Run.** 8 greedy tokens, capacity 1024, `--compare-isa`; the HF fp32 golden is the default.
The prompt is fed token by token through the decode program, so this run's prefill time is
not a prefill measurement.

```sh
python tools/fpga_inference_check.py --model qwen3  --dev f2:0000:34:00.0 --chat --steps 8 --compare-isa --json qwen3.json
python tools/fpga_inference_check.py --model gemma3 --dev f2:0000:34:00.0 --chat --steps 8 --compare-isa --json gemma3.json
python tools/fpga_inference_check.py --model models/Ternary-Bonsai-1.7B --dev f2:0000:34:00.0 --chat --steps 8 --compare-isa --json bonsai17.json
python tools/fpga_inference_check.py --model models/Ternary-Bonsai-1.7B --dev f2:0000:34:00.0 --chat --steps 8 --wformat int4 --head-format int4 --compare-isa --json bonsai17_int4.json
python tools/fpga_inference_check.py --model models/Ternary-Bonsai-4B --dev f2:0000:34:00.0 --chat --steps 8 --wformat int4 --head-format int8 --compare-isa --json bonsai4.json
```

| Model | Weights / head | Image | FPGA output (8 tokens) | HF fp32 argmax | ISA bit-exact |
|---|---|---:|---|---:|---|
| Qwen3-0.6B | int8 / int8 | 1.29 GB | `The Moon is a natural satellite of Earth` | 8 / 8 | yes |
| Gemma-3-1B-it | int8 / int8 | 1.08 GB | `The Moon is a celestial body that orbits` | 8 / 8 (cosine 0.9993–0.9997) | yes |
| Ternary-Bonsai-1.7B | int8 / int8 | 3.05 GB | `The Moon is the only natural satellite of` | 8 / 8 | yes |
| Ternary-Bonsai-1.7B | int4 / int4 | 2.23 GB | the same tokens | 8 / 8 | yes |
| Ternary-Bonsai-4B | int4 / int8 | 3.98 GB | `The Moon is Earth's only natural satellite` | 8 / 8 | yes |

Device decode cycles per token at 200 MHz:

| Model | Cycles / token | Time | Wall per token |
|---|---:|---:|---:|
| Qwen3-0.6B | 19.9 M | 99.5 ms | 113–120 ms |
| Gemma-3-1B | 32.5 M | 162 ms | 167–170 ms |
| Bonsai-1.7B int8 | 55.3 M | 277 ms | 279 ms |
| Bonsai-1.7B int4 | 29.9 M | 149 ms | 157 ms |
| Bonsai-4B | 74.5 M | 373 ms | 375 ms |

Decode is memory-bound: every token streams all the layers' weights from HBM.

### 2. Serving: `tools/f2_bench.py`

* **Server.** Each model through `otpu-serve --backend board`: context 2048, batched prefill
  (prompt runs, chunked prefill 64), the card's sampler, the arguments of
  `tools/f2_web/gateway.py` `MODELS`. Bonsai-4B runs with `--slots 1`: its 3.9 GB int4 image
  leaves room for one KV slot, while the default 8 need 4924 MiB against the slice's 4096.
* **Per model.** Start the server (load time: weights and compile), one warm-up request
  (`Hi.`, 4 tokens), then two streamed greedy chat completions of at most 128 tokens:
  - *short*: the prompt above;
  - *long*: `Summarize the following passage in two sentences.`, then the passage in
    `tools/f2_bench.py` (`LONG`, about 250 words on FPGAs). It is 268 tokens for Gemma 3
    and 274 for the others.
* **Measured at the client.**
  - TTFT: request sent to first streamed token, the prompt's prefill plus the first decode
    step.
  - Prefill tok/s: prompt tokens / TTFT.
  - Decode tok/s: (tokens − 1) / (last token − first token).

```sh
python tools/f2_bench.py --dev f2:0000:34:00.0 --json bench.json --max-tokens 128
```

| Model | Weights / head | Load | TTFT short | TTFT long | Prefill long | Decode |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | int8 / int8 | 13.3 s | 2.16 s | 9.86 s | 27.8 tok/s | 10.0 tok/s |
| Gemma-3-1B-it | int8 / int8 | 11.2 s | 4.69 s | 28.4 s | 9.4 tok/s | 6.1 tok/s |
| Ternary-Bonsai-1.7B | int4 / int4 | 30.4 s | 2.32 s | 14.4 s | 19.0 tok/s | 6.7 tok/s |
| Ternary-Bonsai-4B | int4 / int8 | 39.7 s | 2.25 s | 55.4 s | 4.9 tok/s | 2.7 tok/s |

Open problems these numbers show:

* The short-prompt TTFT is about 2 s whatever the model: a fixed cost before the first
  token, likely on the host.
* Gemma 3's and Bonsai-4B's long-prompt prefill is 3–6 times slower per token than
  Qwen3-0.6B's.
* Decode tracks the bytes of weights streamed per token: int4 Bonsai-1.7B decodes 1.8×
  faster than its int8 image.

The result files are on the instance in `/home/ubuntu/opentpu/build/alex-inference/`:
`*.json` and `*.log` for the correctness runs, `bench_alex200.json` and `bench*.out` for
serving.

### 3. Decode profile and roofline (2026-10-15)

**Method.** Bytes per token come from the ISA ledger (`tools/dataflow.py MODEL decode 1 32
--wformat W --head-format H`: every DRAM byte the decode program reads and writes, zero
image, f2 target); cycles per token are the card's counters above (`timing.decode_cycles /
decode_runs` of the section 1 JSON files, the same images). No RTL co-simulation went into
this table (no per-phase cycle split).

**One law fits every model.** The two Bonsai-1.7B images run the same program with other
weight bytes; their difference gives the memory rate alone: 25.49 M cycles for 815.2 MB, 31.99
bytes a core cycle. A least-squares fit over all five images gives cycles = 0.67 M + bytes /
31.93 B, every residual under 64 K cycles (0.3%):

| Model | MB / token (layers, head, scales, KV) | Cycles / token | GB/s | Not streaming | VPU issue | At 25.6 GB/s | Gap |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B int8 / int8 | 612.8 (440.4, 155.6, 7.4, 8.4) | 19.90 M | 6.16 | 0.74 M (3.7%) | 0.22 M | 41.8 tok/s | 4.16× |
| Gemma-3-1B int8 / int8 | 1015.0 (697.8, 302.0, 11.7, 1.9) | 32.46 M | 6.25 | 0.73 M (2.2%) | 0.35 M | 25.2 tok/s | 4.09× |
| Bonsai-1.7B int4 / int4 | 932.3 (704.6, 155.3, 62.8, 8.4) | 29.85 M | 6.25 | 0.70 M (2.4%) | 0.33 M | 27.5 tok/s | 4.10× |
| Bonsai-1.7B int8 / int8 | 1747.4 (1409.3, 310.6, 17.9, 8.4) | 55.34 M | 6.32 | 0.70 M (1.3%) | 0.33 M | 14.7 tok/s | 4.05× |
| Bonsai-4B int4 / int8 | 2356.5 (1816.7, 388.3, 139.3, 10.8) | 74.52 M | 6.32 | 0.84 M (1.1%) | 0.68 M | 10.9 tok/s | 4.05× |

* *Not streaming*: the cycles minus bytes / 31.99 B, everything the token spends beyond the
  weight stream at the memory's rate (VPU chains, barriers, the head's tail, the program's
  start). *VPU issue*: the ledger's VPU issue cycles (`vpu_hack`'s model) if none overlapped:
  GLU 44-78% of them, RoPE and q/k norms 10-16% each (Qwen3 family).
* *At 25.6 GB/s*: tokens a second if the token's bytes moved at port B's full rate, 128 bytes a
  core cycle at 200 MHz: the slice's own limit. *Gap*: measured cycles over bytes / 128 B.
* The image's memory is HBM, not the DDR4 DIMM: its AFI's own timing report (AWS's ingestion
  output, `s3://.../opentpu-alex/afi-logs/`) has the HBM controller
  (`WRAPPER/CL/u_hbm/HBM_CORE_I/inst/TWO_STACK.u_hbm_top`, both stacks) on two HBM AXI ports
  at 450 MHz behind a SmartConnect (`WRAPPER/CL/u_mem/f2_mem_i/memory_sc`), and no `sh_ddr`
  cell on any path. It is Alex's wrapper, not this branch's `boards/f2` (whose
  `otpu_f2_nch` / `otpu_f2_mem` DDR bridge is not in that image).

**What this says.**

* **Decode moves 6.2-6.3 GB/s whatever the model, 32 bytes a core cycle at 200 MHz,** a
  quarter of port B (128 B a cycle). Every model sits about 4× above port B's roofline for
  this one reason.
* **Everything that is not the weight stream is 1.1-3.7% of a token** (0.7-0.84 M cycles).
  That is the upper bound for decode megakernels (fused norm / RoPE / quant epilogues, fewer
  barriers, VPU overlapped with the stream) on this image: Qwen3-0.6B 10.05 -> 10.44 tok/s,
  Gemma 3 6.16 -> 6.30, Bonsai-1.7B int4 6.70 -> 6.86, Bonsai-4B 2.68 -> 2.71 if all of it
  went away. The VPU's whole issue time is under its share, so the VPU already overlaps the
  stream almost entirely.
* **The rate is the memory path's, and its cause is open.** 32 bytes a core cycle whatever
  the model is a quarter of port B and far below two HBM stacks. Candidates, none measured:
  the SmartConnect's configuration (outstanding transactions, burst handling, width
  conversion), single 64-byte transactions through it, the slice adapter's reads in flight,
  and the crossings into the HBM's 450 MHz domain. Raising it is an RTL / block-design change
  and an FPGA build; it is worth up to 4× on every model's decode.
* **Bytes are the other lever.** The int8 LM head is 25% of Qwen3-0.6B's token, 30% of
  Gemma 3's and 16% of Bonsai-4B's; an int4 head (as Bonsai-1.7B already runs, 8/8 HF argmax)
  halves it, block scales included: about -12% (Qwen3), -14% (Gemma 3), -8% (Bonsai-4B) of
  the cycles, a numerics
  decision. The int4 images' fp32 block scales are 6-7% of their bytes.

### 4. Host TTFT optimization: before and after on the FPGA (2026-10-15)

**What changed.** The fixed ~2.2 s short-prompt TTFT was not compilation, so the fix is host
only, with no device program changed. These are the commits `18da545b`..`107bd2c1`:

* **Contexts stay on the card.** The baseline read the previous request's KV context back to
  the host over BAR4 before each request's first token. BAR4 reads are uncached MMIO with no
  DMA engine. Now a context about to be overwritten moves to a free KV slot on the card.
* **Unavoidable saves run in the background.** They are staged on the card and read in 64 KB
  pieces while the next request runs.
* **Prompt programs compile ahead.** Every prompt bucket's programs are compiled in worker
  processes at nice 10, no longer by a server thread that held the GIL during requests, and
  are kept in a disk program cache.

`tools/ttft_probe.py` and the `board-isa` backend (`opentpu/host/isaboard.py`: the card's
driver over the ISA simulator) measured the host side; `docs/serve.md` has the details.

**Measured.** `tools/f2_bench.py` on the same card, image, prompts and settings as section 2,
fresh server per model. Outputs are identical to the baseline; decode is unchanged.

| Model | TTFT short, before → after | TTFT long, before → after | Prefill long, before → after | Decode |
|---|---:|---:|---:|---:|
| Qwen3-0.6B int8 | 2.16 → **0.57 s** (3.8×) | 9.86 → **6.36 s** (1.55×) | 27.8 → 43.1 tok/s | 10.0 tok/s |
| Gemma-3-1B int8 | 4.69 → **0.55 s** (8.5×) | 28.4 → **9.60 s** (3.0×) | 9.4 → 27.9 tok/s | 6.1 tok/s |
| Ternary-Bonsai-1.7B int4 | 2.32 → **0.88 s** (2.7×) | 14.4 → **9.81 s** (1.5×) | 19.0 → 27.9 tok/s | 6.7 tok/s |
| Ternary-Bonsai-4B int4 | 2.25 → **1.34 s** (1.7×) | 55.4 → **15.2 s** (3.6×) | 4.9 → 18.0 tok/s | 2.7 tok/s |

The run's files are `/home/ubuntu/opt/logs/bench_ttft.json` and `.log` on the instance. The
prefill and decode device programs are unchanged by this step; fused-kernel prefill work is in
progress on separate branches.

### 5. Bytes per token: weight formats against accuracy (2026-10-16)

Decode streams every weight once a token, so with section 3's law (cycles = 0.67 M + bytes /
31.93 B at 200 MHz) a smaller weight format is a faster token. Each lower-byte choice was held
to the HF fp32 model before any of it goes on the card.

**Method.** `tools/format_agree.py` (ISA simulator, f2 target, no card):

* `ref`: 10 prompts in each checkpoint's chat template (thinking off): f2_bench's SHORT and
  LONG, arithmetic, Python code, translation, history, explanation, a haiku, a word problem,
  JSON. HF fp32 (eager) decodes each greedily for 32 tokens or to EOS: 276 positions (Qwen3),
  292 (Gemma 3), 285 / 286 (Bonsai 1.7B / 4B).
* `run`: the image in the given formats takes each prompt as a prefill and is then
  teacher-forced along HF's tokens. Per position it records the argmax against HF's token
  (*TF*), the logit cosine and KL(HF ‖ device) in nats. *Seq*: prompts whose free-running
  greedy output is HF's in full (the first TF miss ends it). *Confident misses*: TF misses where
  HF's top-1 leads its top-2 by at least 0.5 logits (the others are near ties).
* Bytes per token: `tools/dataflow.py MODEL decode 1 32 --wformat W --head-format H`, with
  `OTPU_FORMATS` for the mixes (DRAM read + written, QST's useful bytes not counted twice).
  Projected tok/s = 200 MHz / (0.67 M + bytes / 31.93).

**Criterion.** A config replaces the current one if all three hold:

* its TF agreement is at least the current config's minus 1% of the positions;
* every prompt's mean cosine is at least 0.99;
* its mean KL is not clearly above the current config's (not a multiple of it).

Free-running *seq* is reported but not gated: Gemma 3's own int8 image matches HF on only 2 of
10 prompts, through near ties.

| Model | Layers / head (formats) | MB / token | Projected tok/s | TF | Seq | Confident misses | KL (nats) | Cosine: min / worst prompt mean |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-0.6B | **int8 / int8 (current)** | 612.7 | 10.07 | 275/276 | 9/10 | 0 | 0.0051 | 0.993 / 0.999 |
| | int8 / int4 | 539.2 | 11.39 | 268/276 | 6/10 | 1 | 0.0168 | 0.988 / 0.996 |
| | int8 / fp4 | 539.2 | 11.39 | 264/276 | 3/10 | 4 | 0.0196 | 0.986 / 0.994 |
| | int8 / nvfp4 | 545.3 | 11.27 | 263/276 | 1/10 | 3 | 0.0196 | 0.988 / 0.995 |
| | int8 + gateup=int4 / int4 | 455.9 | 13.38 | 252/276 | 1/10 | 9 | 0.067 | 0.956 / 0.986 |
| | int8 + mlp=nvfp4 / int4 | 428.2 | 14.20 | 253/276 | 0/10 | 9 | 0.071 | 0.921 / 0.985 |
| | int8 + mlp=int4 / int4 | 415.8 | 14.61 | 253/276 | 1/10 | 9 | 0.100 | 0.939 / 0.979 |
| | int4 / int8 | 406.0 | 14.94 | 247/276 | 0/10 | 17 | 0.178 | 0.850 / 0.964 |
| | fp4 / int4 | 332.4 | 18.05 | 240/276 | 0/10 | 16 | 0.139 | 0.867 / 0.966 |
| | int4 / int4 | 332.4 | 18.05 | 243/276 | 0/10 | 22 | - | 0.843 / 0.960 |
| Gemma-3-1B-it | **int8 / int8 (current)** | 1014.9 | 6.16 | 281/292 | 2/10 | 2 | 0.0068 | 0.987 / 0.998 |
| | int8 / int4 | 888.9 | 7.02 | 282/292 | 5/10 | 7 | 0.0305 | 0.968 / 0.994 |
| | int8 / fp4 | 888.9 | 7.02 | 281/292 | 5/10 | 5 | 0.0313 | 0.966 / 0.994 |
| | int8 / nvfp4 | 900.4 | 6.93 | 280/292 | 3/10 | 3 | 0.0210 | 0.978 / 0.994 |
| | int8 + gateup=int4 / int4 | 716.0 | 8.66 | 268/292 | 0/10 | 18 | 0.152 | 0.820 / 0.970 |
| | int8 + mlp=nvfp4 / int4 | 642.8 | 9.61 | 261/292 | 0/10 | 19 | 0.179 | 0.871 / 0.969 |
| | int8 + mlp=int4 / int4 | 618.4 | 9.98 | 259/292 | 0/10 | 23 | 0.251 | 0.781 / 0.956 |
| | int4 / int8 | 710.8 | 8.72 | 249/292 | 0/10 | 33 | 0.428 | 0.730 / 0.944 |
| | fp4 / int4 | 584.7 | 10.54 | 257/292 | 0/10 | 26 | 0.224 | 0.788 / 0.958 |
| | int4 / int4 | 584.7 | 10.54 | 250/292 | 0/10 | 30 | - | 0.712 / 0.941 |
| Ternary-Bonsai-1.7B | int4 / int8 | 1079.7 | 5.80 | 282/285 | 7/10 | 0 | 0.00062 | 0.9992 / 0.9999 |
| | **int4 / int4 (current)** | 932.3 | 6.70 | 282/285 | 7/10 | 0 | 0.00065 | 0.9991 / 0.9999 |
| | int4 / fp4 | 932.3 | 6.70 | 282/285 | 7/10 | 0 | 0.00065 | 0.9991 / 0.9999 |
| Ternary-Bonsai-4B | int4 / int8 (before) | 2356.5 | 2.69 | 284/286 | 8/10 | 0 | 0.00054 | 0.9996 / 0.9999 |
| | **int4 / int4 (now)** | 2172.0 | 2.91 | 284/286 | 8/10 | 0 | 0.00057 | 0.9996 / 0.9999 |

(KL "-": runs before the harness recorded it. Bonsai-4B int4 / fp4 has the int4 head's bytes
and was not finished. The same TF misses, all near ties, occur with every Bonsai head.)

**Choices** (`tools/f2_web/gateway.py` `MODELS`, what `tools/f2_bench.py` runs):

* **Ternary-Bonsai-4B: an int4 head, 3 KV slots.** It gives the same greedy tokens as the int8
  head on every prompt (the same two near-tie misses, KL 0.00057 against 0.00054). The token
  is 184.5 MB (7.8%) smaller: projected 2.69 → 2.91 tok/s. Its image (3.80 GB at context
  1024) leaves room for 3 KV slots at context 2048: 4011 of 4096 MiB with the batch and
  program areas (`tools/format_agree.py fit`). With the int8 head 2 slots fit (4037 MiB) and
  3 do not (4187).
* **Ternary-Bonsai-1.7B: int4 / int4, as before.** It is already the smallest; an int8 head
  measures the same.
* **Qwen3-0.6B, Gemma 3 1B: int8 / int8, as before.**
  - Every 4-bit layer kind, alone or mixed, fails clearly. Gate / up alone is the best mix,
    and it costs Qwen3 23 TF positions and Gemma 3 13, at 13-22 times the KL. The weights are
    not ternary: at 0.6-1 B parameters 4 bits per weight is too coarse for these checkpoints.
  - A 4-bit head is the near miss, about 12% faster.
    - On Qwen3 it costs 7-12 TF positions (int4 268, fp4 264, nvfp4 263 against 275).
    - On Gemma 3 TF holds (int4 282, fp4 281, nvfp4 280 against 281) and every prompt's mean
      cosine stays at or above 0.994. But KL triples to quintuples (nvfp4 0.021, int4 0.031,
      against 0.0068) and confident misses rise (nvfp4 3, int4 7, against 2).
  - Gemma 3 with an nvfp4 head (900.4 MB, 6.93 tok/s, +12.5%) is the one config close enough
    to be a product decision rather than a measured tie. It is not the default.

The run's files are on the instance in `/home/ubuntu/opt/bytes-out/`: `ref-*.npz` (HF tokens
and logits), one `.json` per config (per prompt: TF, divergence, cosine, KL, misses with HF's
margins), `bytes.jsonl` (the ledger runs).
