# P3 "eight lock-step chunk lanes, packed DSP48E2": status (2026-10-21)

Branch `p3-eight-lanes` (P2's `v4-p2-four-lanes` 72eb4b4c, then P2's `p2-f2` b4b3eb03 merged at
74cf3dc0). Proposal: [proposal-3-eight-lanes-packed.md](proposal-3-eight-lanes-packed.md). This
note says what landed, how it was checked and measured, and what is next. Every run below ran on
the m8a (54.174.195.152); receipts are files under `/home/ubuntu/lens-runs/p3/` and
`/home/ubuntu/opt/hwloop-nightly/p3-<sha>/` with the exact command at the top. Nothing here ran on
an F2 card: the token numbers are Verilator RTL, the timing numbers Vivado's (out of context, or
the whole CL synthesized and placed; nothing has been routed).

## What landed

Against the proposal's section 7 plan, the parts that give a bit-exact P3 RTL now:

- **Packed PEs (milestone 0 / 2's packed column).** `otpu_pe_2r` (rtl/tech/generic: the
  behavioural twin the simulators run; rtl/tech/xilinxus: one DSP48E2, A = x1 2^19 and D = x0 in
  the 27-bit pre-adder, B = the weight picked in fabric; rtl/tech/xilinx7: refuses, the DSP48E1's
  pre-adder is 25 bits). `otpu_mxu` parameter `SPACK`: MCOLS / 2 physical columns a lane, slots
  2p / 2p + 1 sharing a weight, P registered at the chain end and split in fabric (low slot
  sext(P[18:0]), high slot P[47:19] + P[18]), the column ends unchanged. Exact because QACT
  saturates activations to [-127, 127] (`fp32.q8`, `otpu_fp q8_s2`): |sum of 16 x0 w| <= 260,096
  < 2^18, and the pre-adder cannot overflow; `otpu_mxu_lanes` checks every consumed ACT byte is
  not -128 in simulation. PAIR at an odd M would put columns M - 1 (low block) and M (high block)
  on one DSP and one weight, so the command moves the high columns up a slot (`qent_t` sh / hs,
  the ACT block and record picked per slot, the PAIR partner at j + M + 1). That needs a slot
  above 2M, so packing takes MCOLS a multiple of 4 (P3's 4, the card's 8): at MCOLS 2, PAIR at M
  1 puts both columns on the one DSP (the whole RTL suite under OTPU_MXU_PACK=1 found it: 5 tests
  at MCOLS 2 differed; those configurations now build unpacked). Names chosen apart
  from the vla branch's `otpu_pe_pk` / `PACK`, which pack two *weight* rows a DSP (CAPS2 bit6, an
  arch key): a different feature.
- **The uarch key `MXU_PACK`** (opentpu/target.py MXU_OPT_KEYS, sim_uarch, rtl_params;
  rtlsim `OTPU_MXU_PACK`; tb_top / otpu_top / otpu_slice / otpu_board / tb_board /
  otpu_f2_hbm_sys; the board model). Timing only: the ISA, images, programs and CFG_HASH do not
  change. Card variants `ypcb-00338:c8` (MXU_C 8) and `:c8p` (MXU_C 8, MXU_PACK 1) for the
  simulators; F2 variants `f2-p2:p3c150 / p3c175 / p3c200` (MXU_C 8, MXU_PACK 1, 16 channels).
- **Eight lanes** are P2's `otpu_mxu_lanes` at C = 8 (E 4 -> one epilogue lane a chunk lane,
  the 8 lanes' results 8 consecutive TMEM words = the 8 banks).
- **The adapter's A runs at a stride (`otpu_native_dram` APS).** At C = 8 lane l's scale rows are
  8 rows apart; int8's 16-byte scale rows then sit every other 64-byte beat, and the channel hash
  spreads them over both channels' runs, so every such read missed and started a 32-beat run.
  APS lets a read ahead in its channel's run take its beat and drop the skipped ones (the order
  FIFO's existing drop count). Set by the lane memory and the lane-port path at MXU_C > 1;
  default 0 everywhere else (cycle-exact).
- **P2's memory side merged** (otpu_lane_mem: an image copy and a two-channel adapter a lane;
  the F2 HBM path from wide-port-alexshell; boards/f2 flow). P3's geometry is P2's at 16
  channels: copy l on channels 2 l, 2 l + 1 (HBM ports 0, 2, .. 30), the host on port 15, eight
  2 GiB copies = the 16 GiB. `boards/f2/build.py --target f2-p2:p3c150 --prepare-only` prepares
  (NCH 16, PSTRIDE 2, HPORT 15, MXU_C 8, MXU_PACK 1).
- Tools: `tools/p3_gemma3_token.py` (P2's real-token tool with MXU_PACK and a tokenizer that
  needs no transformers), `tools/p3_cycle_probe.py` (cycles + state hashes of a fixed run set, two
  trees), `tools/p3_lpn.py` (the vla track's lpnlang net with the lanes in its MM cost: Gemma and
  pi0.5), `rtl/exp/p3_ooc/` (OOC synthesis / place / route of `otpu_mxu_lanes` on the F2's part),
  `sim/tech/tb_tech_pk.sv` + `run_xsim_pk.sh` (the DSP48E2 cell against the generic one in xsim),
  `tools/perf_qwen.py` with the lanes (parse C, the core-port roofline / C, the Lens dump's uarch).

## Correctness

- **Old settings cycle-exact** (`tools/p3_cycle_probe.py`, base 72eb4b4c vs 04b4848d at MXU_PACK
  0, C 1): 10 of 10 records equal in cycles and DRAM / TMEM hashes (mlp 7,508, attn 2,275, fuzz
  2,102 / 1,744, replay 3,468, mcols4 3,117, board 5,973 / 5,116, card 5,778, card at MCOLS 4
  5,665). Receipts: `p3-04b4848d/probe/{base,up}.json`; after the merge with P2's memory side and
  APS, 10 of 10 equal again (`p3-8a9d75d3/probe-up.json`).
- **Packed, C 1**: the same 10 runs bit-exact; at MCOLS 4 the same cycles (a packed lane's row
  latency LROW is 25 either way: two physical columns, each PKL = 2 later), at MCOLS 8 1-10 cycles
  fewer (LROW 27 against 29). `p3-04b4848d/probe/pk.json`.
- **DSP48E2 cell** = the generic model cycle for cycle (xsim on the unisim DSP48E2, 3 seeds x
  3,984 cycles of random operands over the whole byte range and random enables: identical traces;
  `tests/test_tech.py::test_xilinxus_packed_pe_equals_the_generic_model`).
- **RTL tests** (6fa8aed2): `tests/test_rtl.py -k "packed or chunk_lanes"` 31 passed (random
  programs packed at C 1 / 3 / 4 / 8 and MCOLS 4 / 8 with PAIR at odd and even M, every format;
  PAIR at every M on packed lanes at C 1 and 8; packed = unpacked cycles at MCOLS 4 at C 1 and 8;
  8 packed lanes on the board memory path at stalls 0 / 40 / 60 and latency 20-120; the card's
  uarch at c8p; the decode stream at 8 chunks a cycle); `tests/test_tech.py tests/test_target.py`
  24 passed, 1 skipped (no yosys DSP48E1 model).
- **The whole RTL suite packed** (`OTPU_MXU_PACK=1 pytest tests/test_rtl.py`, e44c9104: every
  configuration with MCOLS a multiple of 4 packed, the rest unpacked): 206 passed, 1 skipped (the
  suite's own PAIR skip); log `/home/ubuntu/opt/logs/p3-t13.log`. After the merges (8a9d75d3 and
  92fcae38): `-k "packed or chunk_lanes"` 30 passed each, the old settings' probe 10 / 10 equal
  again (`p3-8a9d75d3/probe-up.json`), test_tech / test_target / test_f2_copies / test_f2_rtl 55
  passed, test_f2_p2_build 3 passed (in a git worktree: it calls git).
- **Tiny Gemma 3** (`tests/test_gemma3.py -k token_on_board --runslow`, the board memory path,
  position 600): logits and the whole DRAM image bit-exact at (MXU_C, MXU_PACK) = (1, 0), (4, 0),
  (8, 0), (8, 1): 4 passed.
- **Gemma 3 1B int8, the real token** (`tools/p3_gemma3_token.py`, "Why is the sky blue?", the
  step at position 14, '\n' -> 'Okay', argmax 19058 on the ISA and the RTL): all 262,144 logits
  and the DRAM image (the KV cache the token wrote) bit-exact at C 8 packed and unpacked.

## Measurements (Gemma 3 1B int8, int8 head, one decode token at position 14)

RTL whole-model token on the board memory path, latency 30, stall 0 (the HBM-like reference),
the card's m4 arch, quoted at 200 MHz. The DRAM bound at 8 chunks a cycle is 981,357 cycles.

| run | memory model | cycles | of the bound | tok/s at 200 MHz | check |
|:--|:--|--:|--:|--:|:--|
| P2 C 4 (P2's receipt) | lane ports, lane 0 through the adapter | 2,292,283 | 85.6% of 1.96 M | 87.2 | bit-exact |
| C 8 packed, before APS (6fa8aed2) | lane ports, lane 0 through the adapter | 2,271,800 | 43.2% | 88.0 | bit-exact |
| C 8 unpacked, before APS (6fa8aed2) | the same | 2,271,800 | 43.2% | 88.0 | bit-exact |
| **C 8 packed, APS (8a9d75d3)** | the lane memory (8 copies, 8 adapters) | **1,299,884** | **75.5%** | **153.9** | bit-exact |
| C 8 unpacked, APS (8a9d75d3) | the same | 1,299,884 | 75.5% | 153.9 | bit-exact |
| C 8 packed, stall 5 (92fcae38) | the lane memory, 5% of the handshakes withheld | 1,356,709 | 72.3% | 147.4 | bit-exact |
| C 8 packed, the F2 channel path (92fcae38) | + per channel the SLR chain (XR 3) and otpu_native_axi (bursts of 8, 32 in flight), latency 49 at 150 MHz | 1,524,984 | 64.4% | **98.4 at 150 MHz** | bit-exact |
| P2 C 4, the F2 channel path (P2's receipts) | the same model, P2's settings | 2,571,208-2,667,016 | 73.6-76.3% of 1.96 M | 56.2-58.3 at 150 MHz | bit-exact |

Receipts: `/home/ubuntu/lens-runs/p3/p3-gemma3-c8p-stall0-8a9d75d3.{txt,json,otpuprof}` (and the
`-prefix-6fa8aed2` pair before APS), `p3-gemma3-c8p-stall5-92fcae38.*`,
`p3-gemma3-c8p-naxi-xr3-lat49-150mhz-92fcae38.*` (OTPU_NAXI=1 OTPU_NAXI_XR=3 OTPU_NAXI_BURST=8
OTPU_NAXI_IDLE=1 OTPU_NAXI_WAIT=0 OTPU_NAXI_OT=32 OTPU_LAT=49, P2's F2-path settings; P2's C 4
receipts there: `/home/ubuntu/opt/logs/p2-tok-naxi150*.log`). On the F2 path the wait on the
quantizer grows to 10.6% of the cycles (4.3% on the lane memory alone): its QST stores go to every
copy, through the path's direction barriers. With APS, `lens summary`: port B streaming 80.5%, MXU computing
not streaming 6.4% (each lane's command is an eighth of the MM: the start and drain per command
weigh 8x), waiting on the VPU 4.8% and on the quantizer 4.3% (the card's 8-lane vector units
against an 8x faster stream: the proposal's M4), TMEM arbitration 1.4% (the drain lost 15.5% of
the cycles, mostly absorbed), not consuming 1.4%, starved 0.9%; 780 MACs a cycle of 4,096. The LM
head's 32 chunks run at 1.16x their stream, the MLP's 768-row chunks at 1.68x, the attention's
256- / 1,024-row projections at 2.6-3.4x (`/home/ubuntu/opt/p3/mmshape.py`).

Before APS, Lens: port B "streaming" 68.9% but the lanes moved only 0.98 M chunk-cycles of work in
those 1.56 M cycles; every long stream ran at 2.3-2.5x its ideal time (`/home/ubuntu/opt/p3/
mmshape.py`). A 2,048-row int8 MM on the board path (`/home/ubuntu/opt/p3/p3_lane0.py`): C 1
19,128 cycles (ideal stream 18,432), C 4 5,299 (4,608), C 8 5,387 (2,304): lane 0's channels read
2,720 scale beats each at C 8 against 256 at C 1 / 4 (85 runs instead of 8), with 2,370 chunk
beats: saturated, and the lock step held every lane to lane 0. With APS: C 8 2,997 cycles (2,304
+ the same 693 fixed cycles as C 1 and C 4), 256 scale beats a channel, the 8 copies' hashes equal.

**LPN** (`tools/p3_lpn.py`, lpn-estimate; the vla tree e896ab4f's lpnlang net, lpnlang c55ef8d;
receipts `p3-gemma3-c{1,4,8}-lpn.{txt,json}`, `p3-vla-pi05-lpn.{txt,json}`; mean |error| 2.7%
over the three graded points, under the 10% rule):

| | lanes | lpn-estimate | RTL (measured) | error |
|:--|--:|--:|--:|--:|
| Gemma 3 1B token, pos 14, lat 30 | 1 | 8,755,262 | 8,221,777 (P2's C 1 stall 0) | +6.5% |
| | 4 | 2,307,854 | 2,292,283 (P2's C 4 stall 0) | +0.7% |
| | 8 | 1,312,435 = 6.56 ms = **152.4 tok/s** at 200 MHz | 1,299,884 (C 8 packed, APS) | +1.0% |
| the same, lat 49 at 150 MHz | 8 | 1,316,254 | 1,524,984 (the F2 channel path) | **-13.7%**: over the 10% rule; the net has no model of the path's bursts, its 32 transactions in flight, the SLR chain or the direction barriers: quote the RTL there |
| pi0.5 chunk, f2-vla as it is | - | 295,472,463 = 1.477 s = 0.68 Hz | | |
| pi0.5 chunk, f2-vla with P3's MXU (8 lanes x MCOLS 4, weights unpacked) | 8 | 582,586,902 = 2.913 s = 0.34 Hz | | |

The pi0.5 chunk is prefill (M up to 255): P3's 8 x 4 x 128 = 4,096 MACs a cycle are half f2-vla's
8,192 (MCOLS 32 with two weight rows a DSP), so P3's MXU halves its rate; P3 is a decode design.
(That run's "MXU %" column is the single-lane tally's and means nothing at 8 lanes.)

## Synthesis and timing (out of context, xcvu47p-fsvh2892-2-e, Vivado 2025.2, 5.0 ns)

`rtl/exp/p3_ooc/run_ooc.sh`: one lane (`otpu_mxu_lanes` C 1, E 1, D 128, MCOLS 4, ACT_ROWS 16,
FIFO 1024, ULANES 8) synthesized, placed and routed with 2.0 ns of input / output delay on every
port. Receipts: `/home/ubuntu/opt/hwloop-nightly/p3-89c220c6/ooc/{lane-pk,lane-up}/` and
`progress.txt`.

| lane | LUT | FF | RAMB36 | DSP | reg-to-reg WNS | all endpoints | packed split / weight pick into B |
|:--|--:|--:|--:|--:|--:|--:|:--|
| packed (SPACK 1) | 18,024 | 19,588 | 33.5 | 300 | +0.773 ns | +0.276 / WHS +0.019 | +1.849 / +1.526 ns |
| unpacked | 15,733 | 21,602 | 33.5 | 556 | +1.093 ns | +0.360 / WHS +0.019 | - |

Both close 200 MHz out of context on the F2's part (the proposal's 167-220 MHz were on the
xcku5p stand-in). Packing a lane: -256 DSPs (-46%), -2.0 K flip-flops, +2.3 K LUTs (the fabric
weight picks, 2 x 128 x 8 bits, and the splits). Eight lanes: ~2,400 MXU DSPs packed against
~4,450 unpacked; P2's floorplan holds the board in SLR2 (3,008 DSPs), which the unpacked eight
lanes would not fit.

**The whole CL, synthesized** (`boards/f2/build.py --target f2-p2:p3c150 --flow SynthCL` at
92fcae38, 30 minutes, `OTPU_STAGE_SYNTHESIS_PASSED`; post_synth.dcp sha256 1aaca920...; the report
`/home/ubuntu/opt/hwloop-nightly/p3-92fcae38/synth-util/util{,_hier}.txt`). The resource budget
the proposal's section 4 estimated, now measured (at the card's arch: the VPU / quantizer / TMEM
not yet widened, the chunk FIFOs in block RAM):

| block | LUT | LUT-RAM | SRL | FF | RAMB36 | RAMB18 | DSP |
|:--|--:|--:|--:|--:|--:|--:|--:|
| the CL (of the VU47P) | 381,851 (29.3%) | 75,318 | 53,364 | 424,685 (16.3%) | 488 | 16 | 2,546 (28.2%) |
| u_sc: 16 SmartConnects | 37,088 | 12,192 | 784 | 46,624 | 0 | 0 | 0 |
| u_sys/u_mem: 16 otpu_native_axi + SLR chains | 9,418 | 5,824 | 0 | 60,722 | 0 | 0 | 0 |
| u_board (SLR2's pblock) | 315,717 | 56,612 | 52,531 | 248,973 | 484 | 16 | 2,546 |
| .. u_mem: the lane memory (8 adapters) | 77,600 | 45,224 | 0 | 23,103 | 0 | 0 | 0 |
| .. u_slice | 233,683 | 10,638 | 52,396 | 218,557 | 456 | 15 | 2,546 |
| .. .. u_mxu: 8 packed lanes | 149,759 | 5,472 | 49,184 | 151,125 | 264 | 8 | 2,393 |
| .. .. .. one lane | 20,493 | 684 | 6,148 | 19,581 | 33 | 1 | 300 |
| .. .. u_vpu | 36,016 | 1,554 | 2,760 | 29,551 | 4 | 0 | 86 |
| .. .. u_tmem | 11,828 | 0 | 0 | 5,123 | 64 | 0 | 0 |
| .. .. u_quant | 9,447 | 100 | 444 | 8,921 | 7 | 5 | 64 |

The proposal's estimate for the full design (VPU 32, URAM FIFOs and TMEM) was ~2,910 DSPs,
~280 RAMB36, 192 URAM, ~400 K LUTs; the MXU's 2,393 DSPs match its 2,416. In SLR2 (P2's floorplan:
~435 K LUTs, 672 RAMB36, 3,008 DSPs a side) the board is ~73% of the LUTs, ~73% of the block RAM
and 85% of the DSPs; the eight adapters' LUT-RAM response FIFOs alone are 45 K LUTs (the notes'
"LUT-RAM-dense adapters congest").

**Placement** (`--flow ImplCL --from-synth synth150`, SSI_SpreadLogic_high, P2's floorplan;
`/home/ubuntu/opt/hwloop-nightly/p3-92fcae38/impl150-1`, stopped after place_design as the
coordinator asked, before phys_opt and routing; reports `impl150-1/placed-report/`): placed
completely (global, detail, post-placement optimization) at 18:33 UTC.

| | |
|:--|:--|
| SLLs, SLR2 <-> SLR1 | 18,295 of 23,040 (79.4%) |
| SLLs, SLR1 <-> SLR0 | 19,707 of 23,040 (85.5%) |
| SLR2 (the board) | CLB 88.4%, LUT 67.7% (LUT-RAM / SRL 50.0%), BRAM 73.2%, DSP 82.9% |
| SLR0 / SLR1 | LUT 12.8% / 1.5% |
| placer congestion | global level 7 north and east in SLR2: the lane memory's adapters (g_cp[*].u_adapt), the MXU lanes, the VPU, the DMA |
| placed timing at 150 MHz | WNS -2.752 ns, TNS -12,604 ns, 18,278 failing endpoints; WHS -0.558 ns |

The worst placed paths are the eight lanes' own (each ~-2.7 ns at 6.667 ns): lane 0's RMAX fold of
the other lanes' keys into the TMEM write (29-33 levels: a combinational max over 8 lanes), the
lock-step pop (each lane's c_left enable depends on every lane's want, 12 levels, 93% route) and
an enable of fanout 7,968 across the lanes. They need RTL: the fold pipelined as a registered
tree, the pop decision a cycle early and replicated per lane (the notes' "compute enables a cycle
early"). Its MMCM is the old M 4.8 (DRC AVAL-168; the analysis is at 6.667 ns).

**The pumped chain** (the coordinator's pick, 2026-10-20): `otpu_xr2` (rtl/shell/otpu_xr.sv),
`otpu_f2_hbm_axi` NXF, `f2-p2:p3c*` hbm_chain_nxf 6: each channel's 604 / 532-bit native word
crosses as two halves on clk_f (2x, phase aligned) through 6 clk_f stages (two per SLR), latency
NXF / 2 + 2 = 5 core cycles each way (3 before), 9.1 K SLLs a boundary for the 16 channels instead
of 18.2 K. Every crossing between the clocks is local and takes half a core period (no multicycle
path); each end has its own self-aligning phase. otpu_native_axi keeps XL = 2 x 5 = 10 of its queue
slots for the chain's round trip, so its queues go to the next power of two keeping NXR's 10
usable slots (QD 32: 22 usable). The NAXI model and the fabric test run the same RTL.

| F2 channel path, C 8 packed, 150 MHz, latency 49 | cycles | tok/s | vs NXR 3 |
|:--|--:|--:|--:|
| NXR 3, QD 16 (10 usable) (92fcae38) | 1,524,984 | 98.4 | |
| NXF 6, QD 16 (6 usable) (3530bf77) | 1,874,316 | 80.0 | +22.9% |
| **NXF 6, QD 32 (22 usable)** (2d90f987) | **1,523,024** | **98.5** | **-0.13%** |
| NXR 3, QD 32 (26 usable) (2d90f987) | 1,515,733 | 99.0 | -0.6% |

All four bit-exact (logits and DRAM; receipts `p3-gemma3-c8p-naxi-*-lat49-150mhz-*.txt:7-9`). The
23% was u_naxi's usable queue slots (6 of 16), not the latency: at the same queue the pumped
chain's 4 more cycles of round trip cost 0.5%. Lens on the pumped run: port B streaming 80.5%,
waiting on the quantizer 10.6%, on the VPU 3.9%, MXU start / drain 2.8%. The LPN at the round
trip's latency (lat 53) reads 1,316,280, -13.6% (outside the 10% rule on this path: quote the RTL).
Tests at 7f191bb6: tests/test_f2_rtl.py 26 passed (the pumped chain at 16 / 8 channels, NXF 6 /
4 / 2, PORDER, two core clocks); the 8-lane board tests and P2's on the NAXI path with the pumped
chain 8 passed; test_rtl -k "packed or chunk_lanes" 30 passed; test_f2_p2_build (the p3c150
variant prepares NCH 16, NXF 6, the exact MMCM, the floorplan's pumped branch), test_target,
test_tech, test_f2_copies 35 passed.

**The pumped design, placed** (7f191bb6, `f2-p2:p3c150`: clk 150 / clk_f 300 MHz from the exact
MMCM, DIVCLK 5, M 24; SynthCL then ImplCL, SSI_SpreadLogic_high, P2's floorplan with its pumped
branch, "OTPU_FLOORPLAN pumped chains: NXF 6, 16 channels";
`/home/ubuntu/opt/hwloop-nightly/p3-7f191bb6/impl150-1`: synthesis passed 20:32 UTC, post_synth
sha256 bc32575d; placed by 22:00 UTC and stopped before phys_opt and routing; reports
`impl150-1/placed-report/`: `util_slr.txt:28-47`, `congestion.txt`, `timing_summary.txt:144` and
its clock tables at :174-210):

| placed, 150 MHz | NXR 3 (92fcae38) | NXF 6, pumped (7f191bb6) |
|:--|--:|--:|
| SLLs, SLR2 <-> SLR1 | 18,295 (79.4%) | **9,206 (40.0%)**: 4,892 down, 4,314 up |
| SLLs, SLR1 <-> SLR0 | 19,707 (85.5%) | **10,620 (46.1%)**: 5,673 down, 4,947 up |
| SLLs in all | 38,002 | 19,826 (18,196 through both LAGUNA registers) |
| SLR2: CLB / LUT / LUT-RAM / BRAM / DSP | 88.4 / 67.7 / 50.0 / 73.2 / 82.9% | 93.0 / 68.8 / 50.0 / 73.2 / 82.9% |
| SLR0 / SLR1 LUTs | 12.8 / 1.5% | 13.3 / 1.5% |
| placer congestion windows at level 7 (at level 6) | 10 (12) | 6 (10) |
| WNS / TNS / failing endpoints, all clocks | -2.752 / -12,604 / 18,278 | -1.722 / -4,978 / 9,616 |
| core clock clk_raw (150 MHz) | -2.752 / -12,577 / 17,544 | -1.722 / -3,674 / 4,967 |
| clk_raw -> clk_f_raw (half a core period) | -0.119 / -0.5 / 9 | -1.648 / -870 / 1,411 |
| clk_f_raw (300 MHz); clk_f_raw -> clk_raw | +0.978; +1.269 | -0.004 (1 endpoint); -0.098 (1) |
| HBM AXI hclk_raw (450 MHz) | -0.132 / -26 / 723 | -0.534 / -434 / 3,236 |
| WHS before routing | -0.558 | -0.753 |

The SLLs halved as designed: each boundary lost 9,089 / 9,087, the 16 channels' 568 signals (half
of 604 + 532) = 9,088. A boundary now carries the chains' 4,832 down and 4,256 up plus ~60 each way
of control; the ~1.4 K more that SLR1 <-> SLR0 carries is the same in both designs, so it is not
the chains. Level-7 congestion remains in two places: north over MXU lanes 1-3 in SLR2, and around
the SLR2 / SLR1 boundary's LAGUNA rows (otpu_f2_hbm_axi's stages 24-27% of the window, lane 2,
adapter g_cp[2]). SLR2's CLBs went to 93%: each chain end holds ~2.5 K flops a channel (serializer
and two clk_f stages down; two stages and the deserializer's 2 x 266 + 532 up) instead of 1,136.

The worst core-clock paths (the top 20 all -1.68 to -1.72 ns) are the first design's RTL classes:
the RMAX fold of the lanes' keys into the TMEM write (`g_lane[0]` / `g_lane[1].u_m/mk_reg` ->
`u_tmem/g_iq.wi_data_reg`, 29-33 levels, 10-12 CARRY8); the lock-step pop and the cross-lane enables (`g_lane[2].u_m/pop_q_reg[wf]` ->
`g_lane[5].u_m/occ_reg`, 12-13 levels; `g_lane[7].u_m/c_left_reg` -> `g_lane[2].u_m/s_rd_reg` CE,
9; lane 7's `u_rw` -> lane 6's FP multiplier DSP reset, 9); the lane memory's write readiness
(`g_cp[2].u_adapt/qb_n_reg` -> `u_slice/b_wdata_reg` CE, 8). New with the pumped chain:

- **The down word into the serializer at half a core period** (clk_raw -> clk_f_raw, -1.648 ns,
  1,411 endpoints): `g_cp[5].u_adapt/ra_res_reg` -> `g_xf.g_ch[11].u_dn/ser_reg`, 5 LUT levels,
  4.34 ns (93% route) against 3.33 ns. The native port's down word comes out of the adapter's
  logic, not a register; at NXR 3 it had a whole period to the first stage. The fix is local: a
  core register in front of each serializer (otpu_xr2; +1 core cycle each way, LX 6, QDX stays 32
  with 20 usable slots), then the F2-path token again.
- **The 450 MHz SmartConnects** (hclk_raw -0.534 ns, 3,236 endpoints, from -0.132 / 723):
  `u_sc/otpu_hbm_sc_i/sc_c7`'s AXI3 write splitter into its W node FIFO, 4 levels, 84% route. The
  IP is the same (187,818 endpoints in both); its placement in SLR0 is what changed.
- The up chain's last stage into the deserializer's core capture (clk_f_raw -> clk_raw): one
  endpoint, -0.098 ns (2.8 ns of route).

Hold before routing is WHS -0.753 ns (clk_f_raw -> clk_raw, 13,659 endpoints; clk_raw -> clk_f_raw
-0.574, 7,149): the router fixes hold, with detours through the congested windows. Nothing here
is routed, and there is no DCP or tar for an AFI.

**Its failing paths by class** (a Vivado pass over its post_place checkpoint, every failing
endpoint folded by start and end hierarchy, lanes / copies / channels as `[*]`:
`impl150-1/placed-report/classes.txt`): core clock 4,967 endpoints, the RMAX fold (256, -1.722),
the lock-step pop's FIFO compares and `c_left` into the lanes' `occ`, `s_rd`, `s_head`, `f_head`,
`f_rdl`, `f_count`, `c_left` (~600, to -1.714), the copies' B readiness into the slice's port B
registers and the DMA (`qb_n` -> `b_wdata` 1,024, `u_dma/gh` 223, ... to -1.709), the sequencer's
MXU command into the eight lanes' issuers and their `q_n` back into it (~1,000, to -1.641), the
MXU drain's TMEM grant into every lane's epilogue (`u_rw` -> `p1_x` 208, the FP DSPs, ..., to
-1.696), the copies' write-done counts (the up chain's `o`) through `wr_idle` into the slice and
the control block (~200, to -1.458), the collective's RLD read into the VPU's write-buffer enables
(361, -0.638); clk -> clk_f 1,411: the adapters' native word into the serializers (1,003 + 119 +
76 + the up direction's 9, to -1.648) and the same RLD read into a pumped TMEM bank (50, -0.536);
the 450 MHz SmartConnects 3,236 (2-level W-FIFO enables of 450-580 fanout in sc_c0 .. sc_c15).

**Timing round 2** (9a863228; every change at C > 1 only, C 1 cycle for cycle: the probe's 10
programs identical in cycles, DRAM and TMEM to 8a9d75d3's):

- `otpu_xr2`: a core register (`dr`) in front of each serializer, so the half-period crossing
  into it is register to register; 6 core cycles each way (NXF / 2 + 3), QDX still 32 (20 usable).
- RMAX: the C lanes' keys folded in a registered tree (`g_fold`, log2 C = 3 levels), the writes
  waiting three drained cycles instead of one (the same maximum: a total order).
- Lock step: each lane's want is a register (`wq`), the next cycle's `pop_w` computed a cycle early
  (no pop: this cycle's arrivals; a pop: within a row 1, at a row's end the next row's first
  advance with the counts after the pop), `c_left != 0` a register (`cl_nz`), the ordinals
  compared by their low bit (a lane is lane 0's command or one ahead: checked), a gather copy per
  lane (kept), `occ`'s decrement a cycle late (P2's ed7b59f0). `wq` implies `pop_w` (checked).
- The MXU's commands through registers: start, go and each lane's command a cycle later
  (`lcmd_q`), `rdy` a register (low a cycle after a start).
- The lane memory: every ready a register of the copies' room for two (`otpu_native_dram`
  `sw_rdy2` / `b_rdy2` / `a_rdy2`: P2's SW fix extended to ports A and B and the lane reads), a
  request reaching a copy only with the ready the slice or the lane sees; `wr_idle` / `rd_idle`
  registers (idle now and no request of the kind: exact or a cycle late). Checked: every request
  a copy takes has room; the idles are never early.
- The MXU's TMEM grant does not wait for its own requests: it is admitted unless a group before it
  (DMA, collective, VPU) writes TMEM at all that cycle (a superset of the bank conflicts).

Gates (all on the m8a): tests/test_rtl.py -k "packed or chunk_lanes" 30 passed; the 8-lane board
tests on the pumped NAXI path 8 passed; tests/test_f2_rtl.py 26 passed; test_gemma3, test_target,
test_tech, test_f2_copies 39 passed (6 skipped); test_f2_p2_build 4 passed; the C 1 probe 10 / 10.

| Gemma-3-1B int8, one decode token at pos 14, Verilator RTL, C 8 packed | cycles | tok/s | vs before |
|:--|--:|--:|--:|
| F2 channel path (NXF 6, latency 49, 150 MHz): 2d90f987 | 1,523,024 | 98.5 at 150 MHz | |
| **9a863228** | **1,524,729** | **98.4 at 150 MHz** | **+0.11%** |
| 9a863228 without the grant change | 1,523,062 | 98.5 | +0.00% |
| 9a863228 without the serializer register | 1,519,934 | 98.7 | -0.20% |
| 9a863228 without both | 1,518,522 | 98.8 | -0.30% |
| lane memory, stall 0 (200 MHz label): 8a9d75d3 / 9a863228 | 1,299,884 / 1,300,176 | 153.9 / 153.8 | +0.02% |

The F2 build writes the quantizer's partial SW beats with byte masks (otpu_f2_hbm_sys SWFILL 0,
P2's 9cb57f61), which every F2-path receipt above modeled as fill reads (rtlsim's default,
OTPU_SWFILL=1): **9a863228 with OTPU_SWFILL=0 is 1,451,711 cycles, 103.3 tok/s at 150 MHz**
(Verilator RTL, the same F2 channel path, bit-exact;
`p3-gemma3-c8p-naxi-xf6-swfill0-lat49-150mhz-9a863228.txt:7-9`), 4.8% under the SWFILL 1 run: the
F2 build's own setting, the number to quote for it.

All bit-exact (logits and the KV-cache DRAM image; receipts
`/home/ubuntu/lens-runs/p3/p3-gemma3-c8p-naxi-xf6-lat49-150mhz-9a863228{,-nogrant,-nodr,-nogrant-nodr}.txt:7-9`,
`p3-gemma3-c8p-stall0-9a863228.txt:7-9`; the variants are the snapshot with the one change
undone). The delta: the serializer register's two more cycles of round trip +4,540 to +4,795
(+0.3%), the conservative grant +1,412 to +1,667 (+0.1%), the other registers -4,502 (-0.3%) on
the F2 path and +292 at stall 0. Lens (9a863228 against 2d90f987): port B streaming 80.3% (80.5%),
waiting on the quantizer 10.0% (10.6%: attention.py:117's QACT wait 150,730 cycles, from
158,597), MXU computing not streaming 3.4% (2.8%: the command and first-pop registers), the MXU
drain's TMEM arbitration losses 6.8% (13.3%). LPN at the new round trip (lat 55): 1,316,289,
-13.7% against the SWFILL 1 run and -9.3% against the build's SWFILL 0 run (inside the 10% rule
there; `p3-gemma3-c8-lat55-150mhz-lpn-9a863228.txt`); the VLA net unchanged, f2-vla 0.68 Hz and with
P3's MXU 0.34 Hz at 200 MHz (`p3-vla-pi05-lpn-9a863228.txt`: its knobs are the vla tree's).

**9a863228 synthesized and placed** (`f2-p2:p3c150`, the exact MMCM, SynthCL then ImplCL
`--package` as unit opt-p3-impl-9a863228; `/home/ubuntu/opt/hwloop-nightly/p3-9a863228/`, job
script /home/ubuntu/opt/p3/synth-place-po.sh, log /home/ubuntu/opt/logs/p3-sp3.log). Synthesis
passed 10:34 UTC (post_synth sha256 7e3116cd; its estimate: core +3.143 ns where 7f191bb6's was
-2.258 over the RMAX fold's 256 endpoints; registers 493 K, from 471 K). Placed (SSI_SpreadLogic_high,
the same floorplan) 11:52, post-place phys_opt (AggressiveExplore) 12:03:

| post-place phys_opt, 150 MHz (`impl150-1/po-report/`) | 7f191bb6 (placed) | **9a863228** |
|:--|--:|--:|
| WNS / TNS, all clocks | -1.722 / -4,978 | **+0.075 / 0** |
| core clk_raw (150 MHz) | -1.722 | **+0.087** |
| clk_raw -> clk_f_raw; clk_f_raw -> clk_raw | -1.648; -0.098 | +0.108; +0.281 |
| clk_f_raw (300 MHz) | -0.004 | +0.109 |
| HBM AXI hclk_raw (450 MHz) | -0.534 | **+0.075** |
| shell clk_main_a0 (250 MHz) | | +0.470 |
| WHS before routing | -0.753 | -0.533 |
| SLLs SLR2 <-> SLR1 / SLR1 <-> SLR0 | 9,206 / 10,620 | 9,206 / 10,620 |
| SLR2 CLB / LUT / LUT-RAM / registers | 93.0 / 68.8 / 50.0 / 32.0% | 94.2 / 71.2 / 51.0 / 33.6% |
| congestion windows at level 7 (6) | 6 (10) | 5 (10) |

(`timing_summary.txt:144` and its clock tables, `util_slr.txt:28-47`, `congestion.txt`.) The
worst placed paths: hclk, SmartConnect sc_c2's reset into its W-node FIFO (+0.075, 4 levels);
core, the slice's port B register b_we into dw_r (+0.087, 9 levels, 89% route). By the
coordinator's rule (route only at WNS >= 0 after the post-place phys_opt) the route went on at
12:03 UTC; the router's first estimates WNS +0.080, WHS -0.585 (console.log `Route 35-416`). The
learned rule says a placed margin under ~0.2 ns tends to lose 0.1-0.3 ns in routing on this part.

## Risks found on the real part

- **SLR crossings at 16 channels.** P2's floorplan (wide-port-alexshell's) keeps the board in SLR2
  and the HBM fabric in SLR0, every channel's native port across two boundaries in 3-stage
  register chains (`otpu_f2_hbm_axi` DW 604 + UW 532 = 1,136 signals a channel). The VU47P has
  3,840 LAGUNA sites a side of each boundary (Vivado on the part: SLR0 3,840, SLR1 7,680, SLR2
  3,840; ~23 K SLLs a boundary). Sixteen channels need ~18.2 K a boundary plus the host's and the
  control chains. **The placer confirmed it** (p3-92fcae38/impl150-1, global placement): "[Place
  30-356] This design requires 19708 Super Long Lines (SLLs) out of 23040 for the crossing of SLR#
  0 to SLR# 1" (85.5%). **The pumped chains took the placed design to 9,206 (40.0%) and 10,620
  (46.1%)** (p3-7f191bb6/impl150-1, above).
  Further ways down, if needed: **the write data crossing once** (every write reaches every copy in the same order,
  so the 16 channels' 576-bit write beats carry two distinct halves, ~10.1 K SLLs instead of
  ~19.7 K; but each copy's adapter sends its beats at its own cycles, as its lane's reads
  interleave, so SLR0 needs a write-data FIFO a channel, a join of the token and the data at each
  otpu_native_axi, and backpressure across the copies); the lane memories' adapters in SLR0 (the
  lane ports cross instead, ~12 K, with credits on the ports); the chains at clk_f with half the
  width (done, 7f191bb6); the proposal's floorplan (the lanes, their chunk FIFOs and the ACT RAM in SLR0 beside the HBM, the
  rest in SLR2: only the ACT writes, the commands and the results cross; the vla branch's
  `otpu_mxu_arr` registered cut is the starting point, but the drain's same-cycle TMEM grant
  needs a credit scheme across the SLRs).
- **DSPs in SLR2**: ~2,400 packed MXU DSPs + ~170 VPU / quantizer of SLR2's 3,008 (85%): the
  packing is what lets eight lanes stay in one SLR (unpacked: ~4,450).
- **The core clock at 150 MHz**: the pumped design placed at -1.722 ns (4,967 failing endpoints,
  the lanes' cross-lane logic); after round 2 (9a863228) every clock meets after the post-place
  phys_opt, but by +0.075 ns (hclk) and +0.087 ns (core): the route may take that margin.
- **Crossings at half a core period**: the pumped chains and the TMEM's pumped RAMs give 12 K
  core -> clk_f and 22.6 K clk_f -> core endpoints with a 3.33 ns budget, plus their hold to fix;
  after routing, the skew between the two clocks' trees could make these the limit.
- **SLR2 at 94% of its CLBs** (9a863228) with five level-7 windows: a routing risk even with
  timing met.

## What is next

1. **The route of 9a863228** (running from 12:03 UTC as unit opt-p3-impl-9a863228, ImplCL with
   `--package`: route, the HDK's checks, validate_route's qualification and the Developer_CL tar
   if it qualifies). If the route loses the +0.075 ns: the next build adds the F2 system's
   collective unit as the simulations build it (COLL_GATHER 0 at S = 1: BAR alone; every gate ran
   that unit, the F2 build had the GATHER unit, which no one-slice program uses and which carried
   the RLD -> VPU write-buffer class), then the remaining paths of the routed design by class.
2. **The cycles Lens names** (C 8 packed, 1.30 M against the 0.98 M bound): the MXU's start and
   drain per command (6.4%: each lane's command is an eighth of the MM; a third command held, or
   the MLP's 768-row chunks merged), the vector units (9.1%: the proposal's M4, VPU 32 lanes / CL 8
   and the quantizer at 32, or the MLP's software pipelining), the TMEM drain's arbitration.
3. **Memory capacity**: eight 2 GiB copies are the whole 16 GiB, so the copies hold Gemma 3 1B
   (~1.05 GB) and nothing larger; the proposal's laned layout (one image, lane l's rows on its own
   channels, group scale rows) is the way past that, an ISA change (Config.CLANES, MM LANE / bs).
4. Receipt still to take: `tools/perf_qwen.py --layers 0 --pos 14 --check` at c8p (the stall-5
   and F2-channel-path tokens are in the measurements above).
