# PersonaPlex resident temporal-to-token diagnostic

This is an intermediate path toward FPGA-only inference: captured summed
embeddings enter the device; resident RoPE coefficients, all temporal layers,
the final RMSNorm, an INT8 text head and greedy selection produce a token ID.
It is not a complete PersonaPlex model, speech service or quality-accepted result.
Input embeddings/conditioning, Mimi, the depth transformer/audio heads and native
stochastic sampling/RNG remain outside this slice. The strict FPGA-only UI must
keep full PersonaPlex speech unavailable.

## Implementation boundary

`opentpu.llm.personaplex_token.TokenImage` extends the temporal layout with
resident FP32 RoPE tables, an untied INT8 vocabulary head, logits and token slots.
RoPE tables cover the full aligned physical capacity; the runtime still enforces
the requested logical context. Constants and checkpoint quantization are prepared
offline when constructing the image. Inference does not calculate RoPE on the host.

`TokenEngine(spec, layers, final_norm, text_head, vocab=32000, ...)` compiles both
programs and every temporal bucket before evaluating a lazy `text_head` callable.
The inherited temporal Engine uses a protected image-construction hook; its
ordinary Image and behavior remain unchanged.

`step(embedding)` uploads the input, runs temporal and head programs on the same
backend, and reads only the four-byte device token. It returns an integer greedy
token ID. It does not read hidden activations or logits, run a host projection,
or select a token on the host. Separate programs avoid combining the instruction
budgets; normalized hidden state remains in device DRAM between them. The head
does not apply final RMSNorm again. Device argmax resolves ties to the lowest ID.
`read_hidden()` and `read_logits()` are explicit diagnostic transfers only.

The new `MLXWeights.text_head_metadata()` validates headers without payload reads.
The lazy `text_head` property materializes only `text_linear.weight` from the
embeddings checkpoint; it does not load the CPU depth model or other embeddings.
There is no fallback from a failed/unavailable physical backend to ISA or CPU
inference. The explicitly selected ISA backend is for development verification.

## CPU validation and capacity

Isolated EC2 aggregate SSM `688e8e3e-ad24-463d-ad48-9079ba671c11` passed
**280 tests in19.55 seconds**, with **zero physical transport opens**. Tests cover
34-frame/bucket-boundary comparisons against the existing temporal Engine and
independent MX head arithmetic, input-only writes/token-only reads, poisoned
per-frame host RoPE, chunk/tail ties, reset/failure/context behavior, lazy-reader
and early capacity checks, CAP_GEN guards, and actual BoardBackend protocol over
IsaTransport. No FPGA or production checkpoint is used by these tests.

The first aggregate completed with278 passed and2 failed: a test omitted its
initial CAP_GEN flag, and its dense FP64 tie construction was not exactly tied.
Both fixtures were corrected. The subsequent run above passed all cases.

Production-shape CPU compilation with the existing mixed rules and cap128 passed:

| Resource | Verified value |
|---|---:|
| Required DRAM, including program reservation | 4,173,934,592 bytes |
| Available DRAM | 4,294,967,296 bytes |
| Spare | 121,032,704 bytes |
| INT8 head codes and scales | 132,224,000 bytes |
| Resident RoPE tables | 65,536 bytes |
| Additional conservative workspace | 2,097,152 bytes |
| Temporal program | 25,240 instruction words |
| Head/greedy program | 184 instruction words |
| Per-program IMEM limit | 32,768 words |

Rules: `attn@0-3=int8,mlp@0-3=int8,attn@31=int8,mlp@31=int8`.
Cap128 is10.24 seconds total, including prompt frames. Existing cap1024 mixed
precision still exceeds instruction capacity. Compilation is not physical timing
or full-model residency validation. Earlier temporal-only precision error and
latency remain unaccepted; adding a head does not resolve either.

## Reproducible real-checkpoint ISA check

`tools/personaplex_token_check.py` accepts the pinned existing model,1–4 captured
initial FP32 embeddings and a completed matching temporal ISA directory:

```sh
python tools/personaplex_token_check.py \
  --model-dir /path/to/existing/model \
  --embeddings /path/to/capture/input_embeddings.npy \
  --temporal-dir /path/to/completed/isa-mixed-001 \
  --output-dir /path/to/new/token-isa \
  --cache-dir /path/to/isolated/quant-cache \
  --cap 128 \
  --formats 'attn@0-3=int8,mlp@0-3=int8,attn@31=int8,mlp@31=int8'
```

Run only on the existing EC2 environment after checking40GiB memory/disk headroom,
with one numerical thread, nice19 and an available CPU affinity. The tool reuses
the existing checkpoint/cache, rejects an existing output, binds input/baseline
arrays by hashes, and checks the pinned revision and checkpoint file metadata.
It requires hidden states to match the saved temporal ISA bits and replays all
frames after reset. Diagnostic source-head FP64 comparisons use identical hidden
states to isolate additional head error; their host calculations are verification,
not the production token-selection path. Numerical/top-1 results do not imply
complete model quality. The tool never opens a physical device or downloads data.

The real-checkpoint check completed under SSM
`72561d5b-b13a-402e-8999-827515eafbc0`, exit0, in422.41 seconds of CPU job wall time.
It used the pinned existing checkpoint and the same four NATF2 inputs as the
previous mixed-precision run. All four temporal outputs matched that saved ISA
baseline bit-for-bit. Device greedy IDs were `[3, 3, 0, 3]`; all four matched the
source head's FP64 top-1 IDs on those identical hidden states. Aggregate INT8
head logit relativeL2 was0.0050865172, cosine0.9999871689. Four reset replays matched
tokens, hidden states and logits bit-for-bit. This measures additional head error;
it does not remove the earlier32.9% temporal error or validate speech quality.

Remote artifacts are `token-isa-001/token_check.json`, `tokens.npy`, `hidden.npy`
and `logits.npy`, beside `token-pipeline-001.json`. The source staging manifest
binds all nine changed files to the successful aggregate; existing dependencies
come from the preserved isolated source baseline. The checker receipt itself
lists only four direct source hashes. Checkpoint revision and file metadata are
verified; multi-gigabyte checkpoint payload hashes are not verified.

## Eventual physical execution gate

No hardware run or image reload is part of this implementation validation. The
shared card is being used for the UI's Gemma work, which may load a different
150MHz wide-port image. Do not restore burst200 or infer compatibility from clock
alone. A future run needs a coordinated window and fresh ownership/profile checks.

The current qualified adapter is the recorded burst200 profile:200MHz, platform3,
architecture hash1341988375, S1/D128/MCOLS4/LANES8,4GiB logical DRAM,64K TMEM words,
32K IMEM words, the recorded microarchitecture/memory geometry, CAP_ARGS, CAP_W4
and CAP_GEN, calibrated idle/drained memory and no error status. For tokens use
`BoardSession(..., require_gen=True, weight_formats=resolved_formats)`; its factory
rechecks before upload. The driver must also hold the normal legacy lock beside
the canonical transport lock. Direct `TokenEngine(backend="board")` is rejected
before image construction; a qualified backend factory is required.

A different image needs its exact profile/capabilities and CPU compilation
validated before extending the qualification gate. No new synthesis, resources,
live service, UI inference endpoint or full speech capability is provided here.
