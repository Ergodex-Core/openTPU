# Nanbeige strict FPGA adapter

This adapter runs the Nanbeige target model with greedy decoding. It uses the
existing burst200 ISA and image; it adds no RTL and needs no synthesis. It does
not call the speculative DSpark adapter. Production commands have no CPU/GPU
model fallback. `validate-isa` is an explicitly named software validation command
and never counts as FPGA execution.

The code is experimental until its actual-checkpoint FPGA receipt is recorded.
The existing target already changes published W8A16 arithmetic to openTPU W8A8.
The fitting strict configuration additionally quantizes the original BF16 input
embeddings to MX INT8. Receipts expose both changes and retain
`source_equivalent=false` and `checkpoint_quality_validated=false`.

## Complete model-computation inventory

| Dependency | Existing target path | Strict adapter |
|---|---|---|
| Input embedding | Host indexes expanded BF16 rows each token | Full INT8 table and gather scales in HBM; FPGA gather/dequantization |
| Position inputs | Host writes cos/sin rows each token | Immutable cos/sin tables in HBM; FPGA indexes each position |
| RMSNorm, Q/K/V, RoPE rotation | FPGA | FPGA |
| GQA, causal attention, softmax | FPGA | FPGA |
| KV cache | FPGA; separate caches per logical layer | FPGA; same 44 logical caches |
| Attention output and residual | FPGA | FPGA |
| Gate/up/down MLP, SiLU, residual | FPGA | FPGA |
| Repeated transformer loops | FPGA, 22 weights reused twice | Same kernels, including the inter-loop norm |
| Final norm and vocabulary head | FPGA INT8 head | Same FPGA INT8 head |
| First output token | CPU argmax of downloaded logits | FPGA ARGMAX after the last prompt token |
| Later output tokens | CPU argmax and host feed | FPGA ARGMAX, token feedback and EOS table |
| Draft/confidence/accept/reject | Optional hybrid DSpark | Not used by target-only inference |

The original BF16 embedding values cannot be retained as FP32 in the current
4 GiB layout. Even an ideal packed BF16 table would leave the complete model
above 4.73 GB; the current ISA also has no BF16 gather/unpack path. Retaining
that precision therefore needs additional addressable device memory and a BF16
gather kernel/compiler/RTL path, or a separately accepted reduction elsewhere.
The INT8 candidate is an explicit approximation, not a source-fidelity claim.

Host work is chat templating/tokenization, text decoding, request validation,
queueing, cancellation, program compilation, transport, counters, and token
streaming. Setup imports/decompresses/requantizes checkpoint tensors and builds
immutable RoPE/mask constants before upload. There is no per-token host embedding,
RoPE coefficient generation, forward operator, logits readback, sampling or RNG.
The ISA simulator executes model arithmetic on CPU only in explicit validation.

## Capacity and compiler contract

Measured metadata/compiler check, pinned production shape, cap128, one-row
prefill, one slice, INT8 weights/head:

| Embedding representation | HBM bytes including program reservation | Result |
|---|---:|---|
| Source BF16 values stored as FP32 | 5,754,519,552 | Exceeds 4 GiB; rejected, no host offload |
| MX INT8 plus device gather scales | 4,239,351,808 | Fits, 55,615,488 bytes spare |

The single-program greedy form exhausts TMEM. The existing device split form
works: 9,696 instruction words for embedding/44 layers and 840 words for the
final norm/head/greedy selector. They chain entirely on FPGA. The prefill program
uses 9,568 words. Every program fits the loaded profile's 32,768-word IMEM.
`plan()` checks every bucket, memory reservation and program capacity before
checkpoint allocation or backend construction. No host or precision fallback is
attempted when a configuration cannot fit.

Required hardware profile is `tools/data/nanbeige/burst200.toml`: F2 platform3,
one slice, D128, MCOLS4, LANES8, 65,536-word TMEM, 32,768-word IMEM,
two 2 GiB channels, 200 MHz, and the exact full architecture hash. It additionally
requires `caps.args`, `caps.gen` (RLD/ARGMAX/HALT CHAIN), and `caps.mx`;
calibration, idle state and absence of hardware/loading errors are checked before
upload and rechecked immediately before constructing the backend. Clock alone
never establishes compatibility. In particular, the UI's 150 MHz wide-port image
must not be assumed compatible and must not be reloaded by this adapter.

## Entry points

Library: `opentpu.llm.nanbeige_strict.StrictEngine`, constructed with a qualified
board factory, then `generate_strict(prompt_ids, max_new_tokens, on_token=None,
stop=None)`. `plan(spec, cfg, cap, embedding_format)` is CPU metadata/compile only.
The engine fixes resident device inputs, no host embedding slots, no host servers,
greedy selection and one sequence. It rejects host-logit `step`, `prefill`,
`run_rows` and host samplers. Requests are validated before context reset/DMA.

CLI:

```sh
python tools/nanbeige_strict.py inspect --model-dir "$MODEL" --cap 128
python tools/nanbeige_strict.py validate-isa --model-dir "$MODEL" --prompt Hi --max-new-tokens 2 --json isa.json
# Launch only in a coordinator-provided compatible card window:
python tools/nanbeige_strict.py infer --model-dir "$MODEL" --prompt Hi --max-new-tokens 2 --json hardware.json
python tools/nanbeige_strict.py worker --model-dir "$MODEL" --cap 128
```

`--prompt-tokens '[...]'` selects reproducible raw input IDs. `--embedding-format
int8` is the default fitting candidate; `f32` is available for small correctness
fixtures and explicitly fails production capacity. Output files use exclusive
creation. `infer` and `worker` always mean board; an exception never changes the
backend. Receipts include the source hash, checkpoint file/header identity,
precision, layout, raw runs, token IDs, decoded text, and host timings. Device
cycles are marked hardware-measured only for actual board runs with counters.
`generation_seconds` includes the last prompt token and the first generated token.

Worker input is one JSON object per line:

```json
{"id":"request-1","prompt":"Hi","max_new_tokens":8}
```

Exactly one of `prompt` and `prompt_tokens` is required. No temperature, top-k,
top-p, draft or sampling parameters are accepted. Startup emits `event=ready`;
each selected token emits `event=token` with its ID; completion emits
`event=result`. Invalid requests emit an error without running. Device/compiler
failures terminate the worker and close its session. EOF releases its locks.
The parent UI must serialize all model workers by physical card and retain this
adapter's experimental precision label until quality acceptance.

## Ownership and validation

`StrictBoardSession` acquires normal legacy `f2:0` and canonical `f2-<BDF>` locks.
A busy card returns exit3 and a `device_busy` JSON error. It never deletes lock
files, changes the image, calibrates during probing, stops another worker,
starts paid resources or overrides ownership. It rejects force-takeover settings
and noncanonical lock-directory overrides. The coordinator must arrange model
handoff before launching a worker because an idle resident worker still owns HBM.

`tests/test_nanbeige_strict.py` compares complete synthetic Nanbeige inference
against an independent host-fed ISA baseline using the same embedding precision,
including GQA6, both logical loops, inter-loop norm, first-token selection,
multi-token prefill, greedy ties, EOS, split generation and reset replay. A read
guard allows only output-token transport, never model tensors. Invalid input,
capacity, format override, host-path, capability/error and dual-lock failure
guards are covered. The broader existing target/CLI/importer/speculative suites
are retained. These checks establish execution-path correctness, not equivalence
to the published BF16-activation model or general checkpoint quality.

The hardware acceptance gate is a bounded actual-checkpoint run against the same
token input as explicit ISA validation, under both locks, recording loaded image
identity and raw counters, exact output-token equality, reset replay, and lock
release. It remains pending until a compatible card window is coordinated; no
physical receipt should be inferred from compiler/ISA success.

## Recorded CPU validation, 2026-10-23

Final synthetic/CLI/importer/speculative aggregate: **124 passed in 22.10 s**,
no skips and zero physical transport opening attempts. It ran at nice19 on one
existing EC2 CPU with offline mode and physical-transport constructors blocked.
The exact source SHA-256 digests are in `nanbeige_strict_validation.json`.
The first collection run exposed a reserved pytest parameter name; the next
run exposed the test harness's temporary lock directory. Both test fixtures were
corrected. Later worker/no-fallback and inherited-API guard checks passed.

Actual-checkpoint ISA validation is separate from this aggregate and from the
pending physical gate. A first attempt stopped before model construction because
the PersonaPlex environment lacks Transformers. An existing Nanbeige environment
was located without installing or downloading dependencies.
