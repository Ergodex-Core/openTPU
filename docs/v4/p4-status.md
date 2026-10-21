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
| `tests/test_gemma3.py` (11, the new ones included), `tests/test_gemma4.py` (43), `tests/test_lens.py` on the ISA simulator | all pass but the pre-existing fp4 failure (below); `test_one_sequence_one_slice` now asserts the new rule (the design's two slices build, three are refused); the S = 1 ring-wrap test 294 s as at the baseline |
| tiny Gemma 3, S = 2 and 4, per-position and resident, 20 tokens vs HF and S = 1 | cosine vs HF 0.998 (S = 1's); **bit-exact to S = 1** |
| tiny Gemma 3 (256-token window), S = 4, 530 tokens: resident decode and chunked prefill vs the per-position programs across the ring's wrap at 512 | **bit-exact** (`test_four_slices_resident_and_prefill_are_bit_exact`; 544 s on a laptop: the ISA simulator runs the four slices in turn) |
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

Status 2026-10-20 (below, "On the F2"): item 2 is done (S = 4 bit-exact on the board model
and on the RTL with the real model); item 4 is the burst200 fabric instead of a new bridge,
built at S = 4; item 1 is deferred to the S = 4 card trace (0.6% of the RTL token); item 3 is
not taken (4.5% "row's scales not all in" is its share now).

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

## On the F2: four slices on the burst200 fabric (branch p4-f2, 2026-10-20)

**The base.** The ergodex-04 session's build 11 (e04/tpubench 23f3f77d, taken here by
fast-forward: P4's milestone 1 is its ancestor) put three of today's slices on Alex's burst200
fabric at 175 MHz and decoded Gemma 3 1B on the card at **48.3 int8 device tok/s** (3.62 M
cycles a token), the one-slice image's text word for word ([f2-burst200-s3.md](f2-burst200-s3.md),
[README.md](README.md)'s build 11 notes). P4 on the F2 is that design at S = 4: target
`f2-burst200-s4` (200 MHz, the design point) and its build variants `f2-burst200-s4-175` and
`f2-burst200-s4-162` (the core clock alone differs), commit af7dddf5.

**Decisions, and why.**

1. *S = 4 on burst200* rather than the v5 engines (full_hbm, placed at -1.803 ns at 150 MHz) or
   the proposal's own HBM bridge (section 2's `otpu_hbm_ch` / NCH 8): burst200 is the only F2
   fabric that has run Gemma on the card (one slice at 200 MHz, three at 175) and its two HBM
   ports a slice feed the slice's 128 B a cycle at 86-89% ([hbm-integration.md](hbm-integration.md)
   rule 3). The ISA split (every matrix by output rows) is P4's milestone 1 unchanged; the S = 3
   build's uneven split is not used at S = 4.
2. *Two slices share SLR2* (`slice_slr = [0, 1, 2, 2]`). At S = 3 SLR0 holds slice 0, three
   SmartConnects, the HBM wrapper and the register slices' SLR0 ends (~64% of its CL LUTs,
   f2-burst200-s3.md) and gains a fourth SmartConnect here; SLR1 holds slice 1, the control
   block, the hub, the trace, the host crossbar and the shell-facing paths in the 23 CL clock
   regions beside the shell; SLR2 holds one slice in 32 regions (~23%). Two slices (~99K LUT
   each) are ~46% of SLR2. The SLL budget (23,040 a boundary, the v5 checkpoint's report):
   SLR1 <-> SLR2 ~7.9K (four native channel hops at ~1,226 signals, two board crossings at
   ~1.5K), SLR0 <-> SLR1 ~14.2K (six native hops, four host channels at ~1.3K, slice 0's
   crossing): 34% and 62%.
3. *SLR2 in halves*, slice 2 in X0-X3, slice 3 in X4-X7 (`pblock_CL_SLR2_S2/3`): v5 placed two
   slices loose in one SLR2 pblock and its MXU's DSP-to-DSP hops took 6.6 ns of route
   ([../full_hbm.md](../full_hbm.md) on branch full-hbm).
4. *Slice 3 on HBM ports 26 / 28* (windows 6 / 7 = pseudo-channels 24-27 / 28-31), both in
   clock region X6Y0. The HDK's `cl_hbm.xci` has the AXI switch on (`USER_SWITCH_ENABLE_00/01
   TRUE`), so a port reaches every pseudo-channel of its stack and 26 / 28 sit inside windows 6
   and 7's mini-switches. Build 11's lesson is a SmartConnect beside its ports
   (`pblock_CL_SLR0_MEM<s>`): u_mem2 X4Y0:X5Y0 + X5Y1 (ports 16 / 20), u_mem3 X6Y0:X7Y0 + X6Y1;
   ports 24 / 28 would have put slices 2 and 3's SmartConnects in the same region X5Y0.
5. *No pipelined collective hub in this image.* The RTL token waits 0.6% on collectives (below;
   0.4% at S = 3): the hub's one beat per ~7 cycles hides behind the MXU's prefetch when the
   slices run in step. On the card the S = 3 trace shows 8.8% waiting on collectives, which a
   memory model that never stalls cannot show (the slices' skew), so the RTL profile does not
   confirm a gain and the rule (Lens before FPGA) keeps the new hub out of this build; the S = 4
   card trace decides it.
6. *Clocks.* At S = 3, 200 MHz routed at -0.436 ns on intra-slice paths (build 8) and 175 MHz
   closed at +0.010 ns (build 11), so the image is built at 175 MHz with 162.5 MHz as the hedge,
   both from one tree, side by side. The 200 MHz work on those paths is the ergodex-04
   session's (build 12) and is not touched here.

**What changed** (af7dddf5): `generate_f2_top.py` takes each slice's SLR from the target
(`slice_slrs`: the native register-slice hops and the reset per SLR; S = 1 and S = 3
regenerate byte-identical); `targets/f2-burst200-s4{,-175,-162}.toml` and their generated tops
`rtl/boards/f2/burst200-s4*/cl_opentpu.sv` (eight adapters, four `f2_mem` SmartConnects, the
1 x 4 host crossbar, ten native register slices, `cl_mem_hbm_wrapper` on 29 ports, an 11-bit
status; ARCH_HASH 0x2f01ab9b); `boards/f2/burst200/constraints/floorplan_s4.tcl`;
`tests/test_f2_burst200_s4.py`. The core RTL (`rtl/`) is unchanged: `otpu_board` and
`otpu_top` were already S-parameterized.

### Milestone 1 at S = 4: gates, Lens, LPN

On the m8a (54.174.195.152), worktree `/home/ubuntu/work/p4`, receipts (each with its command
on top) in `/home/ubuntu/lens-runs/p4/`.

| Check | Result | Receipt |
|:--|:--|:--|
| `test_f2_burst200_s4.py` / `_s3.py` / `test_target.py` without the two RTL gates (the generator, tops, floorplan names, build plans, the Verilator lint of the four-slice top) | 50 passed | |
| tiny Gemma 3 (2 layers, hidden 256, MLP 512) at S = 4 on the four-slice board model (`tb_board`, `otpu_board` S = 4, the f2-burst200-s4 target, two native channels a slice, stall 0) | logits bit-identical to the ISA simulator on 3 tokens, every slice's DRAM equal; 20,588 / 20,589 / 20,618 cycles, 263 instructions a slice; 286 s | `p4-tiny-s4-board.txt` |
| Gemma 3 1B int8 (int8 head), the decode step at pos 14 after the chat prompt, on the Verilator RTL (`otpu_top` S = 4, the hub, the AXI adapter, stall 0, `OTPU_LAT=30`) and on the ISA simulator from the same DRAM | **logits equal bit for bit, all four slices' DRAM equal**; **2,301,972 cycles**, next token 19058 'Okay'; 18 min | `p4-gemma3-1b-s4-rtl-gate.txt` |
| the card's own generate loop at S = 4 (HALT CHAIN, the (max, id) combine over the slices; `test_four_slices_generate_on_the_card`, the path `chat --slices 4` runs on the image), with the S = 3 and slice-split tests | the S = 1 tokens, greedy and sampled; 6 passed | `p4-gemma3-s4-generate-card.txt` |

**Lens, the same RTL token** (`OTPU_LAT=30 lens record model --model gemma3 --slices 4 --axi
--stall 0 --mhz 175`, `p4-gemma3-s4-rtl.otpuprof`; the same 2,301,972 cycles; every slice
alike under a memory that never stalls):

| | S = 3, RTL (f2-burst200-s3.md) | **S = 4, RTL** |
|:--|--:|--:|
| cycles a token (the slowest slice) | 2,993,389 | **2,301,972** |
| its DRAM bound (port B chunks a slice) | 2,668,691 | 1,986,791 |
| port B streaming | 89.2% | 86.3% |
| MXU not consuming: a row's scales not all in | 3.0% | 4.5% |
| MXU computing, not streaming | 2.4% | 2.6% |
| waiting on VPU / quantizer work | 2.0% / 2.0% | 2.7% / 2.4% |
| waiting on collectives | 0.4% | 0.6% |
| unit busy: MXU / VPU / quantizer / COLL | | 99.6% / 8.4% / 5.1% / 1.6% |
| tok/s at 175 / 200 MHz | 58.5 / 66.8 | **76.0 / 86.9** |

Labels: RTL, whole model (26 layers and the head), one decode token at pos 14, int8 weights
and head, the HBM-like memory model (stall 0, 30-cycle reads; not the SmartConnect fabric),
bit-exact against the ISA simulator. The same token with every channel withholding 5% of its
handshakes at random (`--stall 5`, `p4-gemma3-s4-rtl-stall5.otpuprof`, the probe for the hub
decision): 2,386,716 cycles (+3.7%), port B 83.3%, **waiting on collectives 0.2%** in every
slice. Independent per-cycle stalls average out over the ~15K cycles between gathers, so the
card's 8.8% at S = 3 has to come from something the memory model does not have (a slice's
sustained bandwidth below the others', long correlated stalls): only the S = 4 card trace
can price the pipelined hub. The ISA model's analytic S = 4 token is 2,248,050 (the
RTL +2.4%). **Prediction for the card** (not a measurement): build 11's ratio of card to RTL
(3.62 M / 2.993 M = 1.21) on 2.302 M gives ~2.78 M cycles a token: **~63 tok/s at 175 MHz**,
~58 at 162.5, ~72 at 200 (build 11: 48.3).

**LPN.** The VLA track (pi0.5, the vla tree e896ab4f's `tools/pi05_lpn.py --backend lpnlang
--full`, f2-vla MCOLS 32 / LANES 64 / PACK): 295,472,463 cycles = **1.477 s, 0.68 Hz at 200
MHz**, docs/pi05.md 5.5's number exactly (P4 changes no slice RTL; `p4-vla-pi05-lpn.txt`). The
Gemma track, per slice (`tools/p4_lpn.py`: that lpnlang net on P4's slice programs with the m4
slice's constants): 2,750,209 cycles against the RTL's 2,301,972, **+19.5%: above the rule's
~10%, so no tok/s is quoted from it** (`p4-gemma3-s4-lpn.txt`). No constant moves it by more
than 0.6% (L_MM, the write arbitration, the quantizer latency, VPU_CL, the load latency, the
window, the DMA's lookahead): the net, built and calibrated for the packed 64-lane f2-vla
slice, charges 0.75 M cycles of dependent vector / quantizer / collective work between the
matmuls where the RTL has 0.32 M. The same net on the one-slice token (`--slices 1`,
`p4-gemma3-s1-lpn.txt`) reads 8,755,262 against the coordinator's S = 1 RTL reference
8,221,777 (stall 0, `OTPU_LAT=30`): +6.5%, inside the rule. The excess is about the same in
cycles at both points (0.53 M at S = 1, 0.45 M at S = 4): the net over-charges the norms',
activations' and quantizations' chains, which are replicated and so do not shrink with S,
and the error grows with their share. The model validated on this slice (LPNDialect's
CompactOpenTPU, +0.9% on Gemma decode at PORTB 1) models collectives for one slice only
(its workload frontend runs a single program), so it cannot take the S = 4 programs either.

### Milestones 2-4: the F2 image

Builds on the m8a from a `git archive` of af7dddf5 (`/home/ubuntu/opt/hwloop-nightly/p4-af7dddf5/`,
units `opt-p4-viv-s4-175` / `opt-p4-viv-s4-162`, logs `/home/ubuntu/opt/logs/p4-viv-s4-*.log`):
`boards/f2/burst200/build.py --hdk /home/ubuntu/aws-fpga/hdk --target targets/f2-burst200-s4-175.toml
--output .../f2-burst200-s4-175` (ip, pcis, clocks, synth_link: the synthesis and shell link;
implement and refine: place, phys_opt, route and the post-route rounds; audit), started 15:43 UTC.

Synthesis and link (both clocks, 0 errors; the critical warnings are the shell's `cl_pins.xdc`
ones every F2 build has, and the IP stage's read-only block-design parameters; Vivado 2025.2's
parallel-synthesis workers segfaulted five times in the 175 MHz build's IP stage and it
recovered, as build 11's did): the link stage verified the core / TMEM periods 5.714 / 2.857 ns
and 6.154 / 3.077 ns. The core MMCM is clk_wiz's own legal choice (`clocks/console.log`:
CORE_CLOCK_ACTUAL M 21.000 / D 5 / O 6.000 and 3, VCO 1050 MHz at 175; M 26.000 / D 5 / O 8.000
and 4, VCO 1300 MHz at 162.5), and no report of either run has DRC AVAL-168 (the fractional
multiplier the coordinator found in boards/f2/build.py's hand-set HBM_CORE_MMCM, which this
flow does not use). `synth_link/synth_utilization.rpt`: the CL **548,455 LUTs (42% of the
VU47P), 539,417 FFs, 932 RAMB36 (46%), 33 RAMB18, 0 URAM, 2,908 DSPs (32%)**; a slice 99.6K
LUTs (slice 0 100.6K), a memory SmartConnect 16.6K, the host crossbar 12.6K, a native
register slice 0.63K, the HBM wrapper 1.0K. By SLR: SLR0 ~181K LUTs (slice 0, the four
SmartConnects, six native and four host register slices: ~2/3 of its CL LUTs), SLR1 ~125K,
SLR2 ~205K (each half ~103K of ~216K, ~232 RAMB36 of ~336, 727 DSPs). After `opt_design`
the floorplan's Laguna registers are 4,566 on slice 0's crossing and 1,964 on each of slices 2
and 3's, `otpu_clk2x.tcl` constrains 64 pumped TMEM RAMs, and `pblock_CL_SLR0` holds 179.8K
LUTs of its 384K (47%), 214K FFs (28%), 233 block-RAM tiles of 600 (39%), 727 DSPs (29%)
(`implement/slr0_utilization.rpt`).

The m8a's root disk filled at 17:02 UTC for about two minutes (another job's files); the 175
MHz run's implement process lost its log streams at that moment (`implement/vivado.log` stops
at 17:02:10 at exactly 64 KiB, the console mid-table) while it kept placing, so the same
netlist is implemented again from 17:25 (`build.py --from-netlist .../f2-burst200-s4-175
--output .../f2-burst200-s4-175-r2`, unit `opt-p4-viv-s4-175r2`; the netlist check passed,
`post_link.dcp` SHA-256 0849213f...). The 162.5 MHz run's logs continued.

**Placed** (`implement/place_timing.rpt`, after `place_design -directive Explore`, before
phys_opt; setup WNS a clock, 0 failing endpoints unless stated; line numbers of each run's
report):

| clock | 175 MHz (`f2-burst200-s4-175`) | 162.5 MHz (`f2-burst200-s4-162`) | S = 3 at 175 (build 11 run2, the same point) |
|:--|--:|--:|--:|
| core `clk_out1_f2_core_clk` (2.0 M endpoints) | **+0.032** (line 185) | +0.183 (line 185) | -0.315 |
| TMEM `clk_out2_f2_core_clk` | +0.555 (186) | +0.942 (186) | +0.639 |
| HBM AXI 450 MHz `clk_out1_f2_hbm_clk` | +0.027 (183) | +0.177 (183) | +0.059 |
| shell `clk_main_a0` 250 MHz | +0.467 (184) | +0.177 (184) | |
| HBM AXI reset recovery (`**async_default**`, 32 endpoints) | -0.261, 3 failing (213; path at 2847) | -0.261, 3 failing (213; 2812) | |

The one failure is AWS's own HBM wrapper: `PIPE_HBM_AXI_RST/pipe_reg[3][0]` (a `lib_pipe` in
`cl_mem_hbm_wrapper.sv` line 193) drives all 32 ports' `ARESET_N` across the die's bottom
edge, and the far ports' recovery checks at 450 MHz miss with 2.3 ns of route (the learnings'
"single PIPE_HBM_AXI_RST to 32 ports" lesson). Post-place phys_opt replicates it (162.5 MHz
run: `Processed net WRAPPER/CL/u_hbm/PIPE_HBM_AXI_RST/AXI_00_ARESET_N. Replicated 6 times`).
Both runs place every clock with positive slack, the 175 MHz core with 0.35 ns more than the
three-slice build had at this point. After post-place phys_opt the 162.5 MHz run estimated
setup +0.177 ns, TNS 0 (`implement/console.log` line 1557).

**Routing, and the SLR0 / SLR1 boundary.** The route did not keep that: the 162.5 MHz router
reached -0.631 ns (TNS -185 ns) after its first global iteration (console line ~1800 onward;
build 11 was at -0.038 at that point) and its second iteration ripped back up to 130,693
overlapping nodes. Placement had left level 5-6 long North / South congestion in SLR0's columns
X0-X3, rows Y1-Y3 (`implement/place_congestion.rpt`), and the placed checkpoint's SLR report
(`f2-burst200-s4-162/place_rpt/slr_util.rpt`, `report_utilization -slr` on post_phys_opt.dcp)
says why: **SLR1 <-> SLR0 14,370 SLLs** (SLR0 -> SLR1 6,885, SLR1 -> SLR0 7,485) of 23,040,
SLR2 <-> SLR1 6,618. Only the Laguna columns X0-X5 serve the CL at that boundary (SLR0's
X6-X7 top rows are not the CL's and SLR1's are the shell's), so it is ~14.4K of ~17.3K usable,
83%, all of it crossing over slice 0's region to reach the SmartConnects on row Y0. Per SLR the
LUTs are SLR0 39.6%, SLR1 30.6%, SLR2 49.6%, block RAM SLR2 68%: not a logic-capacity problem.
~5.1K of those SLLs are the host's four 512-bit channels (the 1 x 4 crossbar in SLR1, then a
register slice per slice in SLR0). Commit 866c920e moves the crossbar into SLR0 behind one
register slice at the crossing (`host_xbar_slr = 0`, `u_pcis_hop`), so one host channel
crosses: ~10.5K SLLs at the boundary, ~61% of the usable columns. The address map and the core
RTL are unchanged (the RTL gates above stand; the 50 fast tests pass on the new tops). Its
175 MHz build runs from 20:16 UTC (`/home/ubuntu/opt/hwloop-nightly/p4-866c920e/f2-burst200-s4-175`,
unit `opt-p4-viv-s4x-175`); the duplicate 175 MHz implementation (r2) was stopped for it, the
original 175 MHz run (logs silent since 17:02, still routing) and the 162.5 MHz run continue.

The original af7dddf5 175 MHz run then failed in its implement stage at 22:21 UTC without
writing `post_route.dcp` (`/home/ubuntu/opt/logs/p4-viv-s4-175.log`: "implement failed"); its
console and Vivado log had been dead since the 17:02 disk-full, so the cause is not on record.
The SLR0-crossbar build (866c920e, 175 MHz) placed (`implement/place_timing.rpt`) with core
+0.067 ns (line 185), TMEM +0.143 (186), HBM AXI 450 MHz +0.066 (183), shell +0.063 (184) and
the vendor HBM reset recovery at -0.372 over 6 endpoints (213), which phys_opt again
replicated: after phys_opt +0.063 ns / TNS 0 (`implement/console.log` line 1613), router
initialization +0.058 / TNS 0 (line 1735).

**The SLR1 crossbar does not route; the SLR0 one does.** The af7dddf5 162.5 MHz run failed in
`route_design` at 23:12 UTC: "3826 signals failed to route due to routing congestion" and
"Design is not legally routed. There are 2055 node overlaps" (`implement/console.log` lines
1933 and 2052), so the af7dddf5 floorplan is unroutable at either clock (its 175 MHz run died
in the same stage, silently). The 866c920e run at 175 MHz routed legally: its first global
iteration ended at 0 overlaps and -0.338 ns (af7dddf5 at 162.5: -0.631 with 5,455 overlaps),
`route_design` at -0.144 ns, its own post-route phys_opt at **-0.077 ns setup / +0.004 hold**
(`implement/implementation_result.txt`), and the refine stage's Explore and AggressiveExplore
rounds held -0.076 / 0.000 (`refine/progress.txt`), so the flow's gate failed (01:15 UTC).
Per clock in `refine/route_timing.rpt` (final of that run): core **+0.002** (line 224), HBM AXI
450 MHz **+0.001** (222), shell **+0.000** (223); TMEM 350 MHz **-0.066** over 2 endpoints (225)
and core -> TMEM **-0.076** over 39 of 9,088 (241): 41 failing endpoints of 2.32 M, TNS -1.5 ns.
Every failing path is inside a pumped TMEM bank (`refine/worst_paths.rpt`): the bank's
write-staging registers (`g_pmp.pk_v_reg`, `pk_a_reg`, `evr_reg`) into its BRAMs' ENARDEN /
ADDRARDADDR / WEA, one logic level with ~2.0 ns of route against the 2.857 ns fast period
(91% route), in all four slices (slice 3 most), plus DMA `t_re_q` -> the banks' port-B address
(11 levels, ~4 ns of route on the 5.714 ns multicycle). The core, HBM and shell domains close.

Next (08:51 UTC, the box idle after the night's jobs): (1) a post-route recovery pass on that
checkpoint (`/home/ubuntu/opt/p4_recover.tcl`: forced replication of the failing launch
registers' nets so a copy sits by each BRAM, then AggressiveFanoutOpt, AlternateReplication and
AggressiveExplore, each with a reroute; the first pass with setup and hold >= 0 is kept; unit
`opt-p4-recover-s4x-175`, output `.../p4-866c920e/f2-burst200-s4-175/recover/`), and (2) the
same tree at 162.5 MHz (the 2x clock's period +0.22 ns; unit `opt-p4-viv-s4x-162`, output
`.../p4-866c920e/f2-burst200-s4-162`).

The recovery's forced replication is refused after routing ("-force_replication_on_nets ... not
supported yet for post-route physical synthesis"), and its AggressiveFanoutOpt pass left
-0.076 / 0.000 (`recover/recover_progress.txt`). Forcing the replication on the placed
checkpoint instead made the placed estimate +0.063 -> -0.620 ns, so that reroute was stopped.
Where the cells sit explains the paths (`f2-burst200-s4-175/where/tmem_where.txt`, from the
routed checkpoint): each slice's 64 TMEM BRAMs are spread over 6-8 clock regions (slice 0
X2-X4 Y1-Y3; slice 1 X1-X3 Y6-Y7 with half its TMEM logic in X4-X5; slice 2 X1-X2 Y9-Y11; slice
3 X4-X7 Y10-Y11), and the 75K-cell VPU of each slice spreads over 10-16. `floorplan_s4_tmem.tcl`
(dc28a9b7: floorplan_s4.tcl plus one hard six-region box per slice for `u_tmem`, `u_dma` and the
slice's collective port: X1Y2:X3Y3, X1Y6:X3Y7, X1Y9:X2Y11, X5Y9:X6Y11) is implemented on the same
175 MHz netlist from 09:35 UTC (`build.py --from-netlist .../f2-burst200-s4-175 --floorplan ...`,
output `.../p4-866c920e/f2-burst200-s4-175-tmem`, unit `opt-p4-viv-s4x-175tm`).

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
