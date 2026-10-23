# openTPU v4 on AWS F2, candidate P1 "Wide port": two chunks a cycle

Status: a proposal (2026-10-18), nothing built. Every number is arithmetic on the repo's measured
figures (the 133.33 MHz Lens trace of docs/gemma3.md "Lens", the resource tables of
docs/mx_mxu.md, the HBM figures the shared brief states) with its derivation or a file:line.

## 1. Summary

- One slice (S = 1), D = 128, MCOLS = 4, LANES = 8 as today; DRAM port B widened from one to two
  128-byte chunks a core cycle (256 B/cycle), fed from the VU47P's HBM2e through a new adapter.
- The MXU takes the two chunks in one advance: a second group of 4 systolic columns (128 deep) in
  lock step with the first, the two groups' row sums added before the result FIFO, which is the
  int8 generalization of PAIR (rtl/mxu/otpu_mxu.sv:13-18) and docs/wide_dram.md "2. MXU".
- The row gating and result-FIFO credits are reworked so that the per-row bubble (2.6 cycles a
  streamed row today, section 5) does not double its share when rows pop twice as fast.
- Expected Gemma 3 1B int8 decode at 200 MHz: roofline 51 tok/s (19.6 ms), realistic **38-41 tok/s**
  (3x today's 13.0); prefill about 2x today's per-MHz rate (the array doubles with the port).
- Risk: low. +552 DSP (14% of the part used in all), no new numerics, no ISA change, the HBM
  adapter is the one new block; the row-gating fix is the one piece that needs measurement first.

## 2. Datapath changes

**HBM / port B.** The F2's HBM2e has 32 pseudo-channels (PC), each a 256-bit AXI3 port at
450 MHz: 32 x 32 B x 450 MHz = 460.8 GB/s. The port needs 256 B x 200 MHz = 51.2 GB/s, 11.1% of
that. The slice's 4 GiB window (32-bit addresses, docs/targets.md "Limits") is interleaved over
all 32 PCs at a 256-byte granule: chunk pair p lives on PC `(p ^ hash(p >> 5)) mod 32` at PC
offset `(p >> 5) x 256` (the CHASH idea of rtl/mem/otpu_native_dram.sv:9-16 carried to 32 ways),
so a pair is one 8-beat AXI burst on one PC, successive pairs go to successive PCs, and each PC
carries 1.6 GB/s (11% of its 14.4 GB/s): refresh, page misses and the 450 -> 200 MHz crossing
have 9x headroom, so the port, not the memory, is the limit. 128 MiB a PC are in the window; the
other 12 GB are reachable only through the per-slice window registers reserved at 0xA0 + 4s
(docs/targets.md "Limits"), not needed for a 1.03 GiB image. A new `rtl/mem/otpu_hbm_dram.sv`
replaces otpu_native_dram for this target: port B requests are 256 B aligned pairs, port A 32 B
lines (one beat), port SW 32 B masked writes, each PC behind its own command / write / read
async FIFOs (rtl/shell/otpu_afifo.sv) between core_clk and the HBM IP's 450 MHz clock. Read
latency (HBM controller + AXI + two crossings) is taken as 300-500 ns = 60-100 core cycles =
120-200 chunks in flight; PCs return out of order against each other, so the MXU's chunk FIFO
becomes a slot-reserved reorder buffer (a slot is reserved at the request and filled at the
return, as otpu_dma's LD buffer does, rtl/dma/otpu_dma.sv:7-9) and the read tag is the slot.

**MXU** (rtl/mxu/otpu_mxu.sv). New parameter `CPC` = 2 (chunks per cycle; 1 today):
- `b_rdata` is CPC x D x 8 bits; the chunk FIFO holds CPC-chunk entries, `FIFO_DEPTH` = 512
  entries (1024 chunks, 128 KiB, the same bytes as today), which covers 2.5-5x the 120-200
  chunks in flight and the longest row (down's KB = 54, replay rows KB <= DEPTH:1313).
- Two column groups of MCOLS columns each, 128 deep, 16-long DSP48E2 cascades as today (CLS =
  16, mx_mxu.md 9); at every advance group 0 takes chunk 2c with ACT block 2c, group 1 chunk
  2c + 1 with ACT block 2c + 1 (PAIR's `act_blk` / `act_blk2` read, otpu_mxu.sv:77-79, used for
  int8 too). The groups' column ends accumulate separately; their integers are added before the
  result FIFO, where PAIR's adder already sits (otpu_mxu.sv:16-18, :778).
- Odd KB (Gemma 3's hidden 1152 = 9 blocks, the LM head too): milestone 1 streams the last chunk
  as a half pair with group 1's multipliers at 0 (PAIR's "+0 for the missing block of an odd
  KB", :17-18): 5 port cycles for 9 chunks, 11% lost on those matrices. Milestone 4 removes it:
  the issuer streams the matrix as a flat sequence of 256 B pairs across row boundaries, each
  group follows its own row boundary (group 0 gets chunks 0, 2, 4, 6, 8 of row n then 1, 3, 5,
  7 of row n + 1), and a two-entry per-group row-total FIFO pairs the totals by row before the
  adder. Weight matrices become 256 B aligned in the image (they are 128 B aligned today).
- 4-bit formats: each group takes a PAIR (two blocks a chunk), four blocks an advance, the same
  2M <= MCOLS rule; nothing new for the compiler.
- Row gating and credits (the "MXU not consuming" loss, otpu_mxu.sv:331-338): (a) RF 32 -> 128
  rows (LUT RAM, :829) and the credit counted per row group so the drain's latency (LROW 25 +
  the lane 9, :114, :122) never gates a 4.5-cycle row; (b) the consumer starts a command only
  once two rows of chunks and scales are in the FIFOs (a one-row lead that steady state keeps;
  today a row waits for its own last chunk when the lead is gone, :335-336); (c) the row's Sw
  line is prefetched one row ahead by the scale reader so it never costs the issuer a chunk
  cycle (:30-32: "a request of its own (a cycle without a chunk)", which at 2 chunks a cycle
  costs two). Which of the three carries the 18.2% is milestone 0's measurement.
- Epilogue lanes E = 4 (MXU_E) unchanged: decode rows yield 1 result each, prefill rows at most
  MCOLS = 4, and rows now complete at most every ceil(9 / 2) = 5 cycles.

**VPU** (rtl/vpu/otpu_vpu.sv): unchanged, LANES 8, VPU_CL 2. The trace has the VPU in flight
4.5% of the cycles but exposed 0.2% (docs/gemma3.md "Lens"): its exposure is the dependency
chain norm -> QACT -> MM, not its throughput; at half the stream cycles the exposed part is
~0.4%. docs/seq_vpu.md:715-719's composite-width study is re-run at the new rate as a
verification step (section 8), not as a change.

**Quantizer** (rtl/vpu/otpu_quant.sv): unchanged (8 elements a cycle, two passes, QUANT_XD
1024, rtl/top/otpu_slice.sv:52). In flight 1.2%, exposed 0.5% today; ~1% after.

**TMEM** (rtl/mem/otpu_tmem.sv): 64K words, LANES 8 banks, TMEM_IMPL 3 pumped 2x (core_clk2x =
400 MHz), RPB 0 / WPB 1, TAW 16 (rtl/seq/otpu_seq.sv:129): all unchanged. The MXU drain writes
one result a cycle at most, as today. In block RAM (64 RAMB36) or UltraRAM (48-96 URAM288,
docs/tmem_pump.md:676-680, P3 at 200 / 400 MHz meets: :788-790); the RAMB36 budget below keeps
block RAM.

**Sequencer** (rtl/seq/otpu_seq.sv): unchanged, WIN 16, SEQ_WC as the card. The MM issue rate
is one per ~5K cycles.

Config / target keys: `[arch]` gains `CPC = 2` (program visible: the 256 B alignment and the
analytic timing); `[uarch]` FIFO_DEPTH = 512 (entries of CPC chunks), MXU_E = 4, the rest as
targets/bcu1525.toml; `[tech]` family xilinxus, part xcvu47p; `[platform]` mem = "hbm",
core_mhz 200, channels_per_slice 32, beat_bytes 32, a new `hbm_pcs` / `interleave_bytes` 256.

## 3. ISA / compiler / host changes

- ISA: none. MM, QACT, PAIR, the flags and the image's block scales are as they are.
- `opentpu/isasim.py`: `Config.CPC` (default 1); the analytic MM timing becomes
  `N x ceil(KB / CPC)` cycles a row group (KB / CPC after milestone 4); SimError when a weight
  matrix is not CPC x D aligned (as the DSTEP / STREAM alignment checks, isasim.py:265, :333).
- `opentpu/compiler.py` / the image builders (`_Bump.alloc` in opentpu/llm/qwen3.py, Gemma 4's
  Image, gemma4.py:954-957): weight and head allocations rounded to CPC x D = 256 B. No change
  to row bytes (`Q.row_bytes`), so the 1.03 GiB image grows by at most 256 B an allocation.
- `opentpu/target.py`: CPC in the arch hash; `device_config` reads it from a new CFG_* field;
  `check_fits` refuses a CPC-1 program on a CPC-2 device only where alignment fails.
- Host: `opentpu/host/board.py`'s channel map becomes the 32-PC hash above for image upload
  and readback; otpu-chat / Engine unchanged (`OTPU_TARGET=f2`); the on-card decode loop
  (docs/autodecode.md) still moves 64 B a token.

## 4. Resource budget on the VU47P (9,024 DSP48E2 / 2,016 RAMB36 / 960 URAM288)

| resource | today's slice (YPCB production, docs/mx_mxu.md:1264-1268) | P1 | derivation |
|---|---:|---:|---|
| DSP | 716 (u_mxu 554-574) | ~1,268 (14%) | +4 columns x 138 (128 PEs + 10 column-end DSPs a 128-deep column, the brief's figure; mx_mxu.md:1408-1410 counts the same 1,024 + 80 PEs and ends for a doubling) = +552; VPU, quantizer and the E = 4 drain lanes unchanged |
| RAMB36 | ~602 tiles whole image (docs/infra_hunt.md:78); u_act 68, TMEM 64, chunk FIFO 32, IMEM 32 | ~640 (32%) | ACT RAM 68 -> 132 (the read doubles: 2 x 64 data RAMs + records, mx_mxu.md:1420-1424); chunk FIFO to URAM (-32); 32 PC read FIFOs +16; the rest as today |
| URAM288 | 0 | 29 (3%) | chunk FIFO 512 x 2048 bits = 29 URAM288 at 72 bits wide (4096 deep, 1/8 used); +48-96 if the TMEM moves to UltraRAM later |
| LUT | ~177K (xc7k480t, incl. shell) | ~200K of 1,304K | +12-25K for the second group, its ACT RAM read and the adder (wide_dram.md 2: +6K at MCOLS 2) |
| HBM | - | 51.2 of 460.8 GB/s (11.1%) at the port's peak; ~41 GB/s at the realistic 80% port duty | 256 B x 200 MHz |

The 716 -> 1,268 DSP figure is per slice; the F2 has room for S = 4 such slices by DSPs (5,072)
and RAMB36 (2,560: too many, which is why TMEM in URAM matters for S > 1, not for this candidate).

## 5. Expected performance, Gemma 3 1B int8, one sequence

**What the 133.33 MHz trace says per row.** Streaming 76.4% of 10.28 M = 7.85 M cycles = the
model's 7.81 M chunks (26 x (9 x 15,360 + 8 x 1,152 + 54 x 1,152) + 262,144 x 9). The losses
split the same way in the layers (7.2 M cycles for 5.45 M chunks, 24.3% lost) and the LM head
(3.1 M for 2.36 M, 23.9% lost), whose rows are all KB 9 or 8: the loss is per streamed row, not
per command. Rows a token: 459,264 (layers) + 262,144 (head) = 721,408. So:

| loss (docs/gemma3.md "Lens") | cycles at 133 MHz | scales with | per row | at 2 chunks / cycle |
|---|---:|---|---:|---:|
| streaming | 7.85 M | the stream: halves | 10.9 | 3.93 M |
| MXU not consuming (drain / row credits) 18.2% | 1.87 M | per row: unchanged unless fixed | 2.6 | 0.36 M at 0.5 a row (fixed), 0.72 M at 1.0 |
| MXU computing, not streaming 3.9% | 0.40 M | per row (turnarounds, epilogue) | 0.56 | 0.40 M |
| chunk FIFO starved 0.6% | 0.06 M | DRAM latency: the reorder FIFO covers it | | ~0 |
| VPU 0.2%, quantizer 0.5%, other ~0.3% | 0.10 M | fixed per instruction | | 0.10 M, plus ~0.1 M for the HBM's longer latency at command boundaries |
| TMEM arbitration 2.1% | inside the drain gap | per row | | inside the row figure |

**Decode at 200 MHz.** Roofline 7.85 M / 2 = 3.93 M cycles = 19.6 ms = **51 tok/s**. With the
row bubble at 0.5 cycle a row: 3.93 + 0.36 + 0.40 + 0.20 = 4.89 M = 24.4 ms = **41 tok/s**; at
1.0 a row 5.25 M = 26.2 ms = **38 tok/s**. If the bubble were left at 2.6 a row, 6.40 M = 32 ms =
31 tok/s: the gating fix is worth 25-30%. Milestone 1's half pairs cost 0.33 M more (0.5 cycle x
661,504 KB-9 rows): 5.2 M = 26 ms = 38 tok/s before milestone 4 at the 0.5 figure. Host 0.3 ms
a token (docs/autodecode.md) is 1%.

**Per-token cycle budget** (0.5 a row): layers 2.73 M stream + 0.23 M rows + 0.26 M turnaround
+ 0.2 M fixed = 3.4 M (17 ms); LM head 1.18 M + 0.13 M + 0.15 M = 1.46 M (7.3 ms). Today 7.2 M
and 3.1 M at 133 MHz (54 ms and 23 ms).

**Prefill, qualitative.** The array doubles with the port (1,024 MACs a cycle at M = 4), so
prefill at MCOLS 4, which is MXU-bound from 4 rows (docs/scoreboard.md:534-540), gains close to
2x per MHz over today, and 3x at 200 MHz against the card's 133.33: of the order of 120 tok/s
on 1024-token prompts, scaled from the card's 41.9 (scoreboard.md "4. Baselines", Qwen3-0.6B).
The KV reads at 2048 tokens are 55 MiB = 0.22 M cycles at 256 B a cycle, and the attention's
composites run at today's VPU rate, which the seq_vpu.md study re-checks.

## 6. Timing and risks at 200 MHz

The user's F2 synthesis closes the current design at 200 MHz (the card's closed 133.33 MHz on
Kintex-7 at +0.032 ns; the UltraScale+ stand-in met 250 MHz on the sequencer's worst paths,
docs/seq_vpu.md:711-714). What P1 adds:
- The ACT RAM read, 2 x 4 x 1024 bits a cycle: the same registered read as today, twice the RAMs
  and the column multiplexers (mx_mxu.md:1411-1413: 272 RAMB36 at +0.203 ns for 4x the width on
  Kintex-7, so 2x on UltraScale+ is easy).
- The 2048-bit reorder-buffer FIFO in URAM: URAM288's registered read, one more stage.
- The group adder before the result FIFO: PAIR's adder, one stage; the slow clock's worst path
  stays the result FIFO write (mx_mxu.md:1404-1406), which stays as it is.
- The HBM adapter: 32 async crossings 450 -> 200 MHz, the HBM IP's own closure, the AXI ports in
  SLR0 next to the stacks; the slice floorplanned in SLR0 (the bcu1525 floorplan tcl's pblocks
  and Laguna handling, docs/bcu1525.md:195-199, reused).
- core_clk2x = 400 MHz for the pumped TMEM: measured +0.180 ns in URAM, +0.195 in block RAM at
  200 / 400 on the UltraScale+ stand-in (docs/tmem_pump.md:722-724).
- Risk to the number: the "not consuming" mechanism is inferred from otpu_mxu.sv's gating, not
  measured by cause; milestone 0 measures it before the fix is designed in detail.

## 7. Implementation plan

| # | milestone | effort | verification |
|---|---|---|---|
| 0 | Lens splits `mb` (MXU not consuming) by cause: credits / row not in FIFO / scales / drain, a 2-bit reason in the P trace line (opentpu/profile.py, docs/lens.md:130-135) | 1 week | re-record the 133 MHz trace; the 18.2% attributed |
| 1 | MXU CPC = 2 with half pairs, ACT RAM 2-block read, 2-chunk FIFO entries, otpu_dram at CPC chunks a cycle (LAT 80 for HBM), otpu_slice / otpu_board plumbing | 3-4 weeks | token-exact tests (tests/test_gemma3.py and the matrix, tests/test_config_matrix.py) at CPC 1 and 2; tests/test_tech.py on the DSP48E2 cells |
| 2 | Row gating and credits (section 2), by milestone 0's finding | 1-2 weeks | Lens on the Verilator board model at CPC 2: `mb` under 5% |
| 3 | ISA simulator CPC and analytic timing, image alignment, target `targets/f2.toml`, CFG_* field, arch hash | 1 week | `python3 -m opentpu.target show f2`, tests/test_target.py, `lens record --isa` matches the RTL within 5% |
| 4 | Flat pair streaming across odd rows (section 2) | 2-3 weeks | token-exact tests; Lens: KB-9 rows at 4.5 cycles |
| 5 | `otpu_hbm_dram`, `rtl/boards/f2`, `boards/f2` flow on the AWS F2 developer AMI (HBM IP, the shell's XDMA / PCIe, clocks), host channel map; reuse from BCU1525: rtl/tech/xilinxus, run_vivado.sh, the floorplan and Laguna tcl, TMEM_URAM, the target machinery, `make synthcheck` | 4-6 weeks | the HBM path in xsim against otpu_dram's behaviour (the native adapter's test); synthcheck |
| 6 | Bitstream and AFI, bring-up on an F2 instance: `tools/qual/qual.sh fast`, `tools/compare_hf.py --model gemma3`, `otpu-chat --backend board` | 2-3 weeks | 42 qual PASS as the card's dev build (docs/autodecode.md:292-300); HF greedy tokens |
| 7 | Lens on the F2 (section 8), the composite-width re-run, prefill numbers | 1 week | the numbers of section 5 |

About 4-5 months of one engineer, the HBM adapter and the AWS flow the long poles.

## 8. What to verify in Lens once built

- Port B streaming: from 76.4% to >= 80% of the cycles at 2 chunks a cycle (roofline 3.93 M of
  <= 4.9 M cycles); DRAM roofline fraction >= 80%.
- "MXU not consuming (drain / row credits)": from 18.2% to <= 7% (0.36 M of 4.9 M); the cause
  split of milestone 0 shows which remains.
- "MXU computing, not streaming": stays ~0.4 M cycles (8%); if it doubles, the turnaround is per
  chunk and not per row, and section 5's model is wrong there.
- Chunk FIFO starved <= 1% with the reorder FIFO (HBM latency hidden); FIFO level never 0 inside a
  command.
- VPU in flight ~9%, exposed <= 1%; quantizer exposed <= 1%; TMEM arbitration of the drain <= 2%.
- MACs a cycle: 2 x 99 = ~200 of 1,024 at decode; the LM head at 1.46 M cycles, the layers 3.4 M.
- Prefill 1024 tokens: MXU busy >= 90% (docs/scoreboard.md "The tile kernel").

## 9. Every other unit, resized

| unit | today (card, targets/ypcb-00338.toml) | this candidate | why |
|---|---|---|---|
| ACT RAM (rtl/mem/otpu_actram.sv) | 16 rows x 128 blocks x 128 B + records; read 4 x 128 B a cycle (one copy, column-banked); 68 RAMB36 | same store; read 2 x 4 x 128 B a cycle: block 2c to group 0, 2c + 1 to group 1 (PAIR's second read port, shared, not replicated); 132 RAMB36 | the array takes CPC x MCOLS x D bytes a cycle and a RAMB36 port gives 8 (mx_mxu.md:1420-1424) |
| MXU epilogue lanes E | 4 (MXU_E); up to 4 results a cycle | 4 | results a row are M <= 4; rows complete every >= 4.5 cycles |
| result FIFO / row gating | RF 32 rows (otpu_mxu.sv:117); a row pops once all its chunks and scales are in and rows_live + rows_p < RF (:335-338) | RF 128; credits per row group; a two-row lead at command start; Sw lines prefetched a row ahead | section 2; the bubble's share doubles at 2 chunks a cycle otherwise |
| chunk FIFO | 1024 x 128 B, 128 KiB; DDR3 latency ~40 cycles | 512 x 256 B, 128 KiB, slot-reserved reorder buffer in 29 URAM288; HBM latency 60-100 cycles x 2 chunks = 120-200 chunks in flight | covers 2.5-5x the latency and the longest row (KB 54) |
| port A (scale lines) | one 32 B line a cycle, one entry a chunk, beats reused in the adapter | one line request a pair, two scale entries filled from it (a line holds 32 int8 block bytes: 2.4 KB-9 rows) | 2 entries a cycle without a second port |
| port SW (QST / KV writes) | 32 B masked writes, gathered a beat | same, one PC beat each; the KV append is 2 x 26 x 256 B = 13 KiB a token | negligible bandwidth |
| quantizer row buffer / rate | QUANT_XD 1024 chunks, 8 elements a cycle, two passes | unchanged | exposed 0.5% today, ~1% after (section 5) |
| TMEM words / banks / ports | 64K words, 8 banks, RPB 0 / WPB 1, TMEM_IMPL 3 pumped 2x, 64 RAMB36 | unchanged (TAW stays 16) | the VPU width is unchanged, so the arbitration is |
| DMA LD / ST | 8 words a cycle TMEM side (otpu_dma.sv:3-5), LD buffer of D-byte chunks | LD buffer entries of 2 chunks; ST unchanged. Logits: argmax'd on the device in the decode loop (docs/autodecode.md:5-7); a host-read 1 MiB logits store would be 32K cycles (0.7%) | the DMA's TMEM side is the limit, not port B |
| IMEM / program / sequencer | 32K words (4K instructions, the resident decode's 5,350 run as two loops, docs/gemma3.md); WIN 16, SEQ_WC, TAW 16 | unchanged | one MM every ~5K cycles |
| collective | otpu_coll at S = 1: BAR only | none | S = 1 |
| PCIe / host | XDMA Gen2 x8; 64 B a token in the decode loop, 0.3 ms host cost | the F2 shell's PCIe (Gen4 x8 assumed, not checked) through XDMA; image upload 1.03 GiB in ~0.1-0.5 s | the host is off the token path |
| clocks | core 133.33, core_clk2x 266.67, DDR3 sys, axi_aclk 250 (PCIe) | core 200 (closed by the user's synthesis), core_clk2x 400 (TMEM pump: docs/tmem_pump.md:722-724), HBM AXI 450 (32 async crossings), the shell's PCIe clock | one new domain, the HBM's |
