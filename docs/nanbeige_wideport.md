# Strict Nanbeige4.2 on the existing F2 wide-port image

This branch ports the strict Nanbeige adapter to runtime
`18fb978033799f0f21258f0e6e085c0360322d96` and image
`agfi-0001de1dd3df8e21e`. It adds no RTL, synthesis, image reload, or service
management. `infer` and `worker` always use the FPGA; device failures never
select a CPU/GPU model fallback. `validate-isa` is an explicit software test.

The cached checkpoint is **Nanbeige/Nanbeige4.2-3B-GPTQ-Int8**, revision
`e8c6ccdb655bbcf410952686026d3fcbf4c07392`. Its configuration has hidden size
3072, FFN size 10752, vocabulary 166144, 48 query heads, 8 KV heads, head size
128, and 22 physical layers used twice. Both logical loops have independent KV
state, and the inter-loop norm is retained. No DSpark model is loaded.

## Layout fix and capacity

The wide-port Qwen image API no longer accepts the old trailing positional
constructor arguments. Nanbeige now passes explicit compatible keywords and
rejects a nonzero image base. The original wide-port allocation also stored two
copies of the vocabulary head. Nanbeige's untied embedding table does not use
the standard head copy, and its head instructions use the row-lane copy.
`Spec.compact_head` therefore opts out of the unused standard allocation and
upload. Tied models and other models retain their existing layouts.

The change removes 515,047,424 bytes including allocation alignment. It changes
neither model weights nor numerical operations relative to the strict baseline
at `16205f3fd1a94043b9076039cd8a1139a699eef6`.

Measured metadata and compiler results, one sequence, context capacity 128:

| Quantity | Value |
|---|---:|
| Image bytes | 4,247,624,448 |
| Image plus reserved program area | 4,247,789,568 |
| Addressable device memory | 4,294,967,296 |
| Spare memory | 47,177,728 |
| Prefill program words | 7,136 |
| Generation program words | 7,576 |
| Instruction capacity | 32,768 |

Generation fits one program on this profile. `plan()` checks complete memory
and program requirements before checkpoint construction or transport. Larger
capacities are not implied by this result; requests that exceed capacity fail.

The pinned default profile is `f2-hbm:p4rl8a64c150`: architecture hash
3714814630, build ID `0x417243f1`, platform 5, MCOLS8, PORTB4, ROWLANES,
ALINE64, 150 MHz, and eight 512 MiB memory channels. Full configuration, build
ID, clock, geometry, calibration, idle state, error bits, and `args/gen/mx`
capabilities are checked before upload and rechecked at backend construction.
The legacy burst200 profile remains separately pinned for explicit validation;
the adapter does not load either image.

## Model operations and precision

| Operation | Execution |
|---|---|
| Embedding lookup and dequantization | FPGA |
| Position lookup and RoPE rotation | FPGA |
| RMSNorm, Q/K/V, GQA attention, softmax, KV updates | FPGA |
| Output projection, residuals, SiLU gate/up/down MLP | FPGA |
| Both transformer loops, inter-loop and final norm | FPGA |
| INT8 vocabulary head | FPGA |
| First and subsequent greedy ARGMAX, token feedback, EOS | FPGA |

Host work is tokenization/chat templating, text decoding, request validation,
queueing, cancellation, compilation, transport, counters, and token streaming.
Setup imports and packs checkpoint tensors and builds immutable RoPE/mask
tables. There is no per-token host embedding lookup, logits readback, sampling,
RNG, or forward computation in the strict board path.

This remains an experimental precision configuration: the published checkpoint
uses BF16 activations, while the target uses MX W8A8 and an INT8 head; the fitting
strict configuration also quantizes input embeddings to INT8. Receipts retain
`source_equivalent=false` and `checkpoint_quality_validated=false`. The layout
fix does not add another precision change. Preserving source embedding values
as FP32 cannot fit in 4 GiB. A packed BF16 representation would also require
more model memory and a BF16 gather/unpack kernel/compiler/RTL path; no host
offload is substituted.

## Adapter contract

Library entry point:
`opentpu.llm.nanbeige_strict.StrictEngine(...).generate_strict(prompt_ids,
max_new_tokens, on_token=None, stop=None)`. Production callers supply the
qualified board factory from `tools.nanbeige_strict.StrictBoardSession`.
Host-logit APIs, host servers, nonresident input, format overrides, and sampling
are rejected. Requests are validated before reset or DMA.

```sh
python tools/nanbeige_strict.py inspect --model-dir "$MODEL" --cap 128
python tools/nanbeige_strict.py validate-isa --model-dir "$MODEL" --prompt Hi --max-new-tokens 2 --json isa.json
# Only after the coordinator grants the existing card's idle worker window:
timeout --signal=INT --kill-after=10s 300s python tools/nanbeige_strict.py infer --model-dir "$MODEL" --prompt Hi --max-new-tokens 2 --json hardware.json
python tools/nanbeige_strict.py worker --model-dir "$MODEL" --cap 128
```

`--prompt-tokens '[166100,15050]'` accepts reproducible raw input.
`--embedding-format int8` is the fitting default; `f32` supports small fixtures
and fails production capacity rather than offloading. Receipt files are created
exclusively. They include source/checkpoint identity, profile, layout, precision,
tokens, text, timings, raw runs, and whether cycles were measured on hardware.
Software validation cycles are never reported as FPGA measurements.

The worker consumes one JSON object per line:

```json
{"id":"request-1","prompt":"Hi","max_new_tokens":2}
```

Exactly one of `prompt` and `prompt_tokens` is required. Startup emits `ready`,
each generated token emits `token`, and completion emits `result`. Invalid
requests emit an error; device/compiler failures terminate the worker. EOF
closes its session. The UI must serialize model workers by physical card and
preserve the precision label.

Normal legacy `f2:0` and canonical `f2-<BDF>` locks are both acquired. A busy
card returns exit 3 with `device_busy`; lock deletion, force takeover,
noncanonical lock directories, image changes, and interruption of another
worker are not supported. An idle warm worker still owns its model memory.

## Validation and remaining acceptance gate

On the existing EC2 host, **105 focused regressions passed in 41.66 seconds**,
with zero physical transport opens. Coverage includes target/importer behavior,
strict first-token selection and feedback, reset replay, input/error/ownership
guards, the production layout, tied/untied embedding and head layouts, and
existing Gemma/Qwen row-lane decode/prefill comparisons. Complete tiny-model
logits are bit-identical with and without the grouped head layout.

The saved strict baseline and independent source-weight FP64 oracle agree on
raw input `[166100,15050]` producing `[90956,13]`, and the 36-token chat template
for `Hi` producing `[17782,152518]` (`Hello!`). The source oracle uses original
BF16 tensor values with FP64 arithmetic; it does not reproduce official BF16
activation rounding. These are bounded comparisons, not general quality
acceptance.

The current wide-port full-checkpoint comparison passed all three cases: raw
input produced `[90956,13]`, chat produced `[17782,152518]` (`Hello!`), and raw
reset replay again produced `[90956,13]`. This used 40 backend calls and only
three output-token reads, with zero physical transport opens. Packing took
103.75 seconds; raw, chat, and reset ISA execution took 36.97, 425.80, and
35.60 seconds respectively. These are software validation times. The complete
job finished in 607.60 seconds within its 900-second process bound; no job
remains running.

The initial checkpoint driver exited before packing in 0.139 seconds because
it compared Python tuples with JSON lists. Normalizing the identity to its JSON
representation fixed the validation driver without weakening checkpoint
equality or changing the tested runtime. The corrected run used the exact same
checkpoint metadata and headers as both saved baseline receipts.

See [validation metadata](nanbeige_wideport_validation.json),
[test output](data/nanbeige_wideport/regression-003.log), and
[full-checkpoint receipt](data/nanbeige_wideport/checkpoint-isa.json).

Physical acceptance remains pending a coordinated idle window on the existing
image. It must use the same checkpoint and inputs, record loaded configuration
and raw hardware counters, match the ISA token decisions, exercise reset, and
release both locks. Compiler and ISA success alone are not physical acceptance.
The UI-owned runtime and warm Gemma worker are untouched by this branch.

The prepared [physical gate](data/nanbeige_wideport/physical-gate.py) is not
executed evidence. It requires a coordinator window label, verifies source and
checkpoint identity, checks the actual loaded image, uses both normal locks,
runs the same three cases, requires hardware counters, and closes the session.
Its process bound is 300 seconds. Run it on the existing EC2 host only after
the coordinator has arranged the worker handoff; it contains no service or
image-control commands.
