# Speculative decoding on the card (draft and verify)

Status, 2026-10-15: implemented on the ISA simulator (`opentpu/llm/specdec.py`), with otpu-serve
and `tools/f2_bench.py` options. No FPGA run yet. The FPGA numbers below are **projected**: the
measured memory law of [fpga_inference.md](fpga_inference.md) section 3 applied to the ISA
ledger's DRAM bytes of each program, and the acceptance of the Hugging Face models (or of the ISA
simulator's tokens where noted).

## 1. Why

Decode on the 200 MHz F2 image is memory-bound: a token streams every weight once at 32 bytes a
core cycle (6.3 GB/s), and everything else is 1-4% of a token (cycles = 0.67 M + bytes / 31.93).
A run of R token rows streams each weight once for all R rows: every MM takes the R rows as its
stationary operand, and up to R = MCOLS = 4 the MXU (128 B a cycle int8, 64 B a cycle int4
without PAIR) is still faster than the memory. So verifying K drafted tokens in one run of
K + 1 rows costs little more than one decode step, and every accepted draft is a token for free.

## 2. The loop on the card

One device run generates up to n tokens (`SpecDecoder.generate_card`, the semantics of
`Engine.generate_card`: feed the token at `Engine.pos`, return the tokens, the last one not fed).
Programs per attention bucket, chained with HALT CHAIN through a table in the image's `sd` area
(`Engine(spec_rows=K + 1)`, `Image(spec_rows=...)`):

| program | rows | what it does | chains to |
|---|---|---|---|
| V | K + 1: t at p, the drafts d_1 .. d_K at p + 1 .. (their tokens from `hist`) | the target's rows run, an ARGMAX per row (`generate.Greedy`), accept, emit, commit; prompt lookup: the next drafts | V or E of the bucket of p' (prompt lookup), D or D2 (draft model); HALT at a stop |
| E | 1: t at p | the same without drafts (no draft found; K + 1 rows would leave the bucket or the cache) | as V |
| D | the draft model's one-row decode steps, a hardware loop | feeds `hist[q]` from the first position it lacks, then its own drafts, to p' + K - 1; the drafts to `hist[p' + i]` | V (K drafts and K + 1 rows fit the bucket), else E |
| D2 | the draft model, 2 rows: p' - 1, p' | after a verify that accepted all K drafts: the last draft's K / V and d_1 in one weight stream | D (K - 1 more steps) |

- **Accept (greedy).** `c_j = c_(j-1) * [d_j == a_(j-1)] * [a_(j-1) not a stop id] * [j < tokens
  left]`, `n = sum c_j`; a_0 goes to out[p + 1] and a_j to out[p + 1 + j] under a LOOP of count c_j;
  `tok = a_n`, `tpos += n + 1`, `left -= n + 1`. Every row's token goes to `hist[p + 1 + j]` (the
  history, one fp32 id a position; the ones past p' are overwritten later).
- **Tokens identical to plain greedy, bit for bit.** Each emitted token is the argmax of a logits
  row that the decode step would compute on the same context: the rows kernels are bit-identical
  to decode steps (MM outputs are exact integers, replayed or paired alike; attention is per row,
  the tiled kernel's additive masks give each row's online softmax). A rejected row's K / V is
  overwritten by the next run before anything attends to it (the cache is positional). The tests
  compare tokens with `Engine.generate_card`.
- **V's rows** are `tiled.qwen3_tile`'s (a whole form, today's bits: `TILED`) where a form fits,
  else `qwen3_rows`. The tiled kernel keeps TMEM independent of the widths (Bonsai-4B's 4 rows do
  not fit `qwen3_rows`: TMEM exhausted) and streams each KV head's K / V once for MCOLS / G query
  rows (G = 2 on Qwen3-0.6B and Bonsai-1.7B: half the extra rows' KV bytes).
- **The host** writes `hist[0 .. p]` (the context), the state block and out[]'s marks, starts the
  run (`BoardBackend.run_generate`: tokens as they land, 1 .. K + 1 an iteration) and reads the
  iteration counters back (words 40-44 of the state block: iterations, accepted, verifies, the
  draft's K / V extent, tpos - 1).

### 2.1 Prompt lookup (no second model)

V and E end with the lookup over `hist[0 .. p']` (`p'` the new position, its token a_n): three
equality vectors over the bucket's positions (g = 1, 2, 3 tokens ending at j equal those ending at
p'), a ramp mask (j < p'), one ARGMAX of `e1 (j + 1) + (e2 + e3) (cap + 1)`: the latest match of
the longest g. The drafts are `hist[j + 1 + (i mod L)]`, L = p' - j: the continuation after the
match, repeated with its period where it runs into the present (what the lookup would draft again
after each draft; a loop of period 1 gets K drafts, not 1). `specdec.ngram_drafts` is the same
rule on the host (a run's first drafts, the acceptance study).

### 2.2 A draft model on the same device

`draft_engine(target, "models/Qwen3-0.6B")` builds the draft model's Engine with its image at the
top of the slice's DRAM (`Image(base=...)`), above the target's image, the program area and
kvio's 256 MiB staging area (else MemoryError), written there through the target's backend
(`SubBackend`). Its vocabulary is cut to the target's (Qwen3-0.6B's 151936 rows are the
tokenizer's 151669 ids and padding; the three checkpoints' tokenizer.json files are identical), so
its drafts are target ids. The draft's K / V follows the target's context: the host rolls it back
to the longest common prefix and prefills the rest before a run (its prompt runs: this lands in
TTFT, not in decode), and reads back what the device left (state word 43).

## 3. Which model pairs fit (4 GiB a slice)

| target | draft | images (target + draft) | fits |
|---|---|---|---|
| Bonsai-4B int4 / int8 head, fp32 embedding | any | 3.98 GB + ... | no: 0.3 GB free, less the program area and staging |
| Bonsai-4B int4 / int8 head, **int8 embedding** (the tied int8 head's rows) | Qwen3-0.6B int4 / int4 | 2422 MiB + 1034 MiB (draft at 3062 MiB, measured) | yes |
| Bonsai-4B int8 embedding | Bonsai-1.7B int4 (int8 embedding table) | 2422 MiB + about 1.2 GiB | yes, but a slower draft (30 M cycles a step) |
| Bonsai-1.7B int4 / int4 | Qwen3-0.6B int4 / int4 | 2206 MiB + 1034 MiB (measured) | yes |
| Qwen3-0.6B | (prompt lookup) | 1.29 GB | yes |

The fp32 embedding table is the 4B's largest single item (151669 x 2560 x 4 = 1.55 GB) and is never
streamed by decode; `--embed int8` gathers the rows from the tied int8 LM head instead (a numerics
change of the target: plain greedy of that image is the reference, and speculative decoding gives
its tokens).

## 4. Measured inputs

### 4.1 Cost of the programs (ISA ledger, F2 configuration, law cycles at 200 MHz)

`tools/specdec_decode.py ledger --model M --k 3 --pos P` (zero image, K = 3):

| model | pos | plain step | E | V (4 rows, tiled) | V / plain |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B int8 / int8 | 100 | 20.18 M | 20.18 M | 21.38 M | 1.060 |
| Qwen3-0.6B int8 / int8 | 300 | 20.65 M | 20.65 M | 22.57 M | 1.093 |
| Bonsai-1.7B int4 / int4 | 100 | 30.18 M | 30.18 M | 31.39 M | 1.040 |
| Bonsai-1.7B int4 / int4 | 300 | 30.66 M | 30.66 M | 32.58 M | 1.063 |
| Bonsai-4B int4 / int8 | 100 | 74.89 M | 74.89 M | 77.80 M | 1.039 |
| Bonsai-4B int4 / int8 | 300 | 75.49 M | 75.49 M | 80.55 M | 1.067 |
| Qwen3-0.6B int4 / int4 (the draft's step) | 100 / 300 | 11.40 / 11.87 M | | | |

The plain steps agree with the card's measured cycles (19.90 M, 29.85 M, 74.52 M near position 30).
V's extra bytes are the extra rows' K / V (the bucket's blocks, a row each MCOLS / G rows) and
activations; with `qwen3_rows` (no tiling) Qwen3-0.6B's V was 1.089 / 1.156 (each row streams
the bucket's K / V: 29.4 MB a row at position 300).

### 4.2 Acceptance (Hugging Face fp32 greedy, 11 prompts, 128 tokens)

`tools/specdec_accept.py`: f2_bench's SHORT and LONG and mtp_accept's nine prompts (chat, code,
summaries), each target's greedy tokens; the drafters' proposals walked through the loop exactly.
Tokens per iteration (accepted drafts + 1), all prompts:

| target | drafter | k = 1 | 2 | 3 | 4 | 5 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | prompt lookup | 1.21 | 1.29 | 1.32 | 1.34 | 1.36 |
| Bonsai-1.7B | prompt lookup | 1.17 | 1.23 | 1.25 | 1.26 | 1.28 |
| Bonsai-1.7B | Qwen3-0.6B | 1.74 | 2.28 | 2.70 | 3.03 | 3.23 |
| Bonsai-4B | prompt lookup | 1.14 | 1.17 | 1.19 | 1.20 | 1.20 |
| Bonsai-4B | Qwen3-0.6B | 1.65 | 2.14 | 2.43 | 2.62 | 2.79 |
| Bonsai-4B | Bonsai-1.7B | 1.74 | 2.28 | 2.66 | 2.95 | 3.19 |

Prompt lookup drafts on about half the iterations (no match: E, a plain step); summaries gain the
most (Qwen3-0.6B k = 3: 1.59 tokens an iteration), short chat the least.

### 4.3 The loop on the ISA simulator (real checkpoints, F2 configuration)

`tools/specdec_decode.py run` (cap 2048; plain `Engine.generate_card`, then the loop from the same
prefill; f2_bench's SHORT and LONG prompts):

| target | drafter, K | prompt | tokens | iterations | accepted | tokens / iteration | identical |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3-0.6B int8 / int8 | prompt lookup, 3 | short | 17 | 17 | 0 | 1.00 | yes |
| Qwen3-0.6B int8 / int8 | prompt lookup, 3 | long | 103 | 89 | 13 (46 verifies) | 1.15 | yes |
| Bonsai-1.7B int4 / int4 | Qwen3-0.6B int4 / int4, 2 | short | 30 | 17 | 13 | 1.76 | yes |
| Bonsai-1.7B int4 / int4 | Qwen3-0.6B int4 / int4, 2 | long | 64 | 31 | 33 | 2.06 | yes |
| Bonsai-4B int4 / int8, int8 embedding | Qwen3-0.6B int4 / int4, 3 | short | 34 | 17 | 17 | 2.00 | yes |

The card's prompt-lookup counters equal the host's walk of the same tokens (89 / 13). The int4
draft and the quantized target agree a little less than the fp32 models of 4.2 (Bonsai-4B short:
2.00 against 2.27; Bonsai-1.7B long: 2.06 against 2.17). otpu-serve over the card's driver on the
ISA simulator (`--backend board-isa`, `--spec-k 3 --slots 1`, Qwen3-0.6B) served f2_bench's two
prompts through the loop (its 24 programs compiled and placed with their wait entries at start,
23.4 s).

## 5. Projected decode tok/s on the F2 image

Per iteration: V (or E without a draft) plus, with a draft model, k draft steps (D2's 2-row step
shares one weight stream). Costs of section 4.1 at position 300 (100 in brackets); "today" is
f2_bench's measured decode.

| model | today | drafter, k | speedup | tok/s |
|---|---:|---|---:|---:|
| Qwen3-0.6B int8 / int8 | 10.0 | prompt lookup, 3 | 1.26x | 12.6 |
| Bonsai-1.7B int4 / int4 | 6.7 | prompt lookup, 3 | 1.21x (1.22x) | 8.1 |
| Bonsai-1.7B int4 / int4 | 6.7 | Qwen3-0.6B int4, 2 | 1.26x (1.28x) | 8.4 |
| Bonsai-4B int4 / int8, int8 embedding | 2.7 | prompt lookup, 3 | 1.15x | 3.1 |
| Bonsai-4B int4 / int8, int8 embedding | 2.7 | Qwen3-0.6B int4, 3 | 1.58x (1.62x) | 4.2 |

With the simulator's acceptance (4.3) instead: Bonsai-4B short (2.00 tokens an iteration) 1.34x,
3.6 tok/s; Bonsai-1.7B with the draft (1.96 over both prompts) only 1.08x, so prompt lookup is
the 1.7B's drafter. The FPGA run decides; `/v1/stats` reports the iterations and acceptance.

## 6. Use

```sh
# prompt lookup, K = 3 (any Qwen3-family model)
python -m opentpu.serve.api --model qwen3 --wformat int8 --head-format int8 --backend board \
    --dev f2:0 --spec-k 3
# a draft model beside the target
python -m opentpu.serve.api --model models/Ternary-Bonsai-4B --wformat int4 --head-format int8 \
    --slots 1 --embed int8 --spec-k 3 --spec-draft models/Qwen3-0.6B --backend board --dev f2:0
# the serving benchmark with them
python tools/f2_bench.py --dev f2:0 --json spec.json --models bonsai-4b --tag spec \
    --serve-args "--embed int8 --spec-k 3 --spec-draft models/Qwen3-0.6B"
```

Greedy requests decode on the speculative loop (scheduler.Worker and BatchWorker's card loop);
sampled requests take the plain loop. `/v1/stats` has spec_iters, spec_accepted, spec_verifies.

## 7. Limits and next steps

- Greedy only. Sampled verification is the MTP loop's "sample, then compare" ([mtp.md](mtp.md) 11):
  the same structure with `generate.Sampler` sinks per row.
- A V verifies K + 1 rows whatever the drafts; the last K positions of each bucket take E steps.
- Prompt lookup and the draft model are alternatives; a lookup hit could replace the draft model's
  steps when its match is long (summaries, code edits).
- The draft model's prefill adds to TTFT (its prompt runs for the prompt's tokens).
