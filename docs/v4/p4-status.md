# P4 (four tensor-parallel slices): status

Milestone 1 of [proposal-4-four-slices.md](proposal-4-four-slices.md) section 7: Gemma 3 /
Gemma 4 split over S slices on the ISA simulator, with its correctness gate. Branch
`v4-p4-four-slices` off `tpubench` (016e953a), 2026-10-18. Nothing of the hardware side
(the 4-slice RTL top, the collective hub, the HBM bridge, the F2 shell) is built; see "What
remains".

## What was done

**The model split** (`opentpu/llm/gemma4.py`; [../gemma4.md](../gemma4.md) "Slices" is the
reference): `Spec.check` takes S in (1, 2, 4) (n_q, vocab, hidden, the MLP widths, the K / V
rows and ple_dim divisible by S; a MoE model stays on one slice) instead of one slice. Every
matrix is split by output rows, never by K, with qwen3.py's S = 2 scheme and dataflow's split
(a): slice s holds rows s / S .. (s + 1) / S of W_q (its query heads), W_k / W_v, W_o, gate /
up, W_down (over all of F, in the one-slice image's column parts) and the PLE projections; the
norms, the tables, the generate loop's area, the whole int8 LM head table (the embedding
gathers) and the KV cache (the single KV head appended by every slice to its own copy) are
replicated. Per layer: the slice's K / V rows gathered (2 GATHERs), the heads gathered for
W_o and its rows gathered (2), the gelu x up product gathered to [R, F] in F order and the
down rows gathered (2): 6 GATHERs, 10,752 words a layer of Gemma 3 1B at S = 4, as the
proposal's table; the LM head's (max, id) combine is generate.py's existing one. `Image`
builds S images (the same offsets, the slice's rows: `_job_matrix`'s `rowrange` / `rowpad` /
`rowcols` selections; the head quantized once for all), `descriptors(sid)` points the head MM
at the slice's vocab / S rows, `compile_decode` / `compile_rows` / `compile_step` /
`compile_prompt_run` return one program per slice; the Engine, generate.py, prefill.py and
the ISA simulator's `Machine` already handled S slices (Qwen3 at S = 2), so otpu-chat's and
compare_hf's loops run unchanged. The resident decode, the generate loop (HALT CHAIN
buckets) and the chunked prefill (8-row runs, prompt runs) all compile and run at S = 4.

**Bit-exact to one slice.** A matrix's output rows are independent dots over the same K
blocks; the gathered tiles hold the one-slice tiles' values in the same column order; the
norms, residual and RoPE run on the same tiles; the down projection quantizes the gathered
product over the same 768-column QACT ranges before the same 9 accumulating MMs; one head's
attention at M = 1 is the 4-head entry's row. No sum is reassociated, so S = 2 and S = 4 give
S = 1's logits bit for bit (measured, below). At S = 1 nothing changes: the tiny model's image
and every program kind (rows 1 / 4 / 8, decode buckets, the generate loop, the prompt run) are
byte-identical to tpubench's (checked with `build/p4_dump.py` against an extract of 016e953a).

**Host side.** `tools/compare_hf.py --cfg board --slices S`; `python -m opentpu.lens record
model --slices S` (implies `--board`; the profile's name says "4 slices"); otpu-chat and Lens
keep Gemma 3 / Gemma 4 in the card's one-slice configuration by default
(`Spec.one_slice_default`: lifting the one-slice check would otherwise have moved them to the
design's two slices); Lens's ISA profile now counts the GATHERs a slice reaches (the
simulator resolves collectives outside `execute()`, so they were missing).

**Coordinator's note (2026-10-18).** The same Gemma 3 decode token on the RTL board model
with `--axi --stall 0` and `OTPU_LAT=30` (an HBM-like memory that never withholds handshakes;
the proposal's trace used the default `+axi_stall=20`) takes 8,221,777 cycles, 95.5% of the
roofline, 41.1 ms at 200 MHz = 24.3 tok/s on one slice: "MXU not consuming" falls from 18.2%
to 1.4%. A slice's intrinsic efficiency is therefore ~95%, and P4's per-slice estimate should
start from 8.22 M / 4 cycles plus the collective, not from 10.28 M (proposal section 5's
"2.8 cycles a row" loss was the DDR3 model's stalls). Any RTL check of this branch should use
`--stall 0` with `OTPU_LAT=30` as the HBM-like reference. This milestone is on the ISA
simulator, so nothing in it changes.

## Measurements

All on the ISA simulator, the card's configuration (D 128, MCOLS 4, ACT_ROWS 16, 8 lanes,
64K-word TMEM) with S slices, int8 weights and head. The tiny random Gemma 3 is
`tests/test_gemma3.py`'s (8 layers, hidden 256, 4 heads on 1 KV head, vocab 1000).

| Check | Result |
|:--|:--|
| `tests/test_gemma3.py`, `tests/test_gemma4.py` at S = 1 | as before (one pre-existing failure, below) |
| tiny Gemma 3, S = 2 and 4, per-position and resident, 20 tokens vs HF and S = 1 | cosine vs HF 0.998 (S = 1's); **bit-exact to S = 1** |
| tiny Gemma 3 (256-token window), S = 4, 530 tokens: resident decode and chunked prefill vs the per-position programs across the ring's wrap at 512 | **bit-exact** (`test_four_slices_resident_and_prefill_are_bit_exact`) |
| tiny Gemma 4 (PLE, 8 heads, double-wide MLP), S = 2, per-position and resident vs S = 1 | **bit-exact** (`test_two_slices_match_one`) |
| Gemma 3 1B, `compare_hf.py --model gemma3 --chat --cfg board --resident --tokens 12 --slices 4` | the S = 1 tokens on both prompts: `Okay, let's break down why the sky is blue` (HF `The`, its #2 at 0.609 below) and the haiku identical; logits vs HF max 2.269 / 3.455, min cosine 0.99869 / 0.99872, the S = 1 values |
| Gemma 3 1B image a slice (2048-token capacity) | 1,053 MiB at S = 1, 715 at S = 2, 547 at S = 4 (layers 731 / 393 / 225 MiB; KV 55 MiB and the head 292 MiB in every slice) |

**Lens, one decode token** (`lens record model --model gemma3 --isa --board --slices 4 -o
build/lens/p4-gemma3-isa.otpuprof`; position 14 after the chat template's 15-token prompt,
the resident decode program; `--slices` omitted for the S = 1 profile,
`build/lens/p4-gemma3-isa-s1.otpuprof`). The ISA simulator's analytic timing: every
instruction back to back for its nominal work, no overlap between units; a GATHER is
S x rows x ceil(cols / 8) cycles (profile.py), no hub latency or skew.

| | S = 1 | S = 4, each slice |
|:--|--:|--:|
| instructions executed | 5,350 | 3,820 (156 of them GATHERs) |
| MM instructions / chunks streamed (port B) | 922 / 7,837,472 | 586 / 1,979,552 (25.26%: the quarter plus the replicated KV reads and gathers' one-hot MMs) |
| port B transfers (the DRAM bound, cycles) | 7,850,855 | 1,986,791 |
| GATHERs / words received a token | 0 | 156 / 279,552 (6 a layer: 2 x 256 for k and v, 1,024 the heads, 1,152 W_o's rows, 6,912 the product, 1,152 the down rows; the generate loop's head combine adds 2 GATHERs of 4 words, not in the step program Lens traces) |
| unit busy: MXU / VPU / QUANT / DMA / COLL | 7,837,472 / 291,399 / 99,232 / 53,451 / 0 | 1,979,552 / 115,431 / 89,248 / 28,875 / 34,944 |
| analytic cycles a token | 8,281,554 (94.8% of the bound) | 2,248,050 (88.4% of the bound; 34,944 of them the GATHERs' nominal cycles) |
| at 200 MHz | 41.4 ms, 24.1 tok/s | 11.24 ms, 89.0 tok/s (the bound 9.93 ms, 100.7 tok/s) |

The VPU's share rises from 3.5% to about 5% (the norms stay 1152 wide and replicated, the
gelu and the attention's softmax split), the quantizer's from 1.2% to 4% (the one-slice QACT
ranges of the gathered product are quantized in every slice: 9 x 768 words a layer, as at
S = 1, over a quarter of the MM work). The analytic per-slice token is 3.7x shorter than one
slice's; the proposal's Lens estimate (README: "P4: 4 slices (C=4, 8 lanes) 2.40 M cycles")
was a re-timing of the one-slice instruction stream, this is the split's own.

## What remains (proposal section 7, milestones 2-5)

1. **The collective hub.** The ISA simulator resolves a GATHER instantly once every slice
   reaches it; the RTL's `otpu_coll_hub` moves one beat per ~7-cycle round trip (the BCU1525's:
   ~245 K cycles a token here, 12%), and the proposal's pipelined hub (8 reads in flight, 8
   words a cycle) is new RTL. Lens's "waiting on collectives" line (CAUSES `w4`) is where it
   will show.
2. **The 4-slice RTL top** (`otpu_board` at S = 4, `otpu_native_dram` NCH, `tb_board.sv` at
   S = 4 with the behavioural DRAM): the tiny Gemma 3 at S = 4 on the board model bit-exact
   against the ISA simulator (the `test_tiny_qwen3_two_slices_on_board_model` pattern), with
   `--stall 0` and `OTPU_LAT=30` as the HBM-like reference per the coordinator's note.
3. **The MXU row credits** (optional; the HBM-like RTL reference above says the loss is mostly
   the DDR3 model's stalls, so its worth is smaller than section 5 assumed).
4. **HBM bridge, target and board** (`targets/f2.toml`, `rtl/boards/f2/`, `otpu_hbm_ch`, the
   SLR floorplan), then **the AFI and the EC2 F2 bring-up**.

Also open: the Gemma 4 E2B / E4B real checkpoints at S > 1 are not run (the tiny Gemma 4 at
S = 2 is); the MoE models, the layer-major prefill and the expert runs stay one-slice; the
board model's `BoardBackend` at S > 1 for these models is not exercised. Pre-existing on
tpubench (016e953a), unrelated to this branch:
`tests/test_gemma4.py::test_resident_gathers_are_bit_exact[fp4]` fails with "TMEM exhausted"
in `_gathered_rows` (the fp4 one-hot operand of 33,280 words); it fails identically on an
extract of 016e953a.

## Commands to reproduce

From the worktree root with its interpreter (`PYTHONPATH=.` so that the checkout's package is
imported, `models/gemma-3-1b-it` present):

```sh
PYTHONPATH=. python -m pytest -q tests/test_gemma3.py tests/test_gemma4.py        # S = 1 and the slice tests
PYTHONPATH=. python tools/compare_hf.py --model gemma3 --chat --cfg board --resident --tokens 12 \
    --slices 4 "Why is the sky blue?" "Write a haiku about the ocean."         # and without --slices
PYTHONPATH=. python -m opentpu.lens record model --model gemma3 --isa --board --slices 4 \
    -o build/lens/p4-gemma3-isa.otpuprof                                       # and without --slices
PYTHONPATH=. python -m opentpu.lens summary build/lens/p4-gemma3-isa.otpuprof --slice 2
PYTHONPATH=. python -m opentpu.lens info build/lens/p4-gemma3-isa.otpuprof
```

Per-slice counts from a profile (what the table above reads): the `instrs` rows of the
profile dict are `[slice, ...]`; `MM` rows' port-B transfers are their chunks, `GATHER` rows'
detail is `rows x cols x S`.
