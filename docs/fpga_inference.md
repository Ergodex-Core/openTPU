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
