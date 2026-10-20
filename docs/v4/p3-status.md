# P3 "eight lock-step chunk lanes, packed DSP48E2": status (2026-10-20)

Branch `p3-eight-lanes` (P2's `v4-p2-four-lanes` 72eb4b4c, then P2's `p2-f2` b4b3eb03 merged at
74cf3dc0). Proposal: [proposal-3-eight-lanes-packed.md](proposal-3-eight-lanes-packed.md). This
note says what landed, how it was checked and measured, and what is next. Every run below ran on
the m8a (54.174.195.152); receipts are files under `/home/ubuntu/lens-runs/p3/` and
`/home/ubuntu/opt/hwloop-nightly/p3-<sha>/` with the exact command at the top. Nothing here ran on
an F2 card: the token numbers are Verilator RTL, the timing numbers out-of-context Vivado.

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
  the ACT block and record picked per slot, the PAIR partner at j + M + 1). Names chosen apart
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
  5,665). Receipts: `p3-04b4848d/probe/{base,up}.json`. After the merge: (pending,
  `p3-8a9d75d3/probe-up.json`).
- **Packed, C 1**: the same 10 runs bit-exact; at MCOLS 4 the same cycles (a packed lane's row
  latency LROW is 25 either way: two physical columns, each PKL = 2 later), at MCOLS 8 1-10 cycles
  fewer (LROW 27 against 29). `p3-04b4848d/probe/pk.json`.
- **DSP48E2 cell** = the generic model cycle for cycle (xsim on the unisim DSP48E2, 3 seeds x
  3,984 cycles of random operands over the whole byte range and random enables: identical traces;
  `tests/test_tech.py::test_xilinxus_packed_pe_equals_the_generic_model`).
- **RTL tests** (6fa8aed2): `tests/test_rtl.py -k "packed or chunk_lanes"` 31 passed (random
  programs packed at C 1 / 3 / 8 and MCOLS 2 / 4 / 8 with PAIR at odd and even M, every format;
  PAIR at every M on packed lanes at C 1 and 8; packed = unpacked cycles at MCOLS 4 at C 1 and 8;
  8 packed lanes on the board memory path at stalls 0 / 40 / 60 and latency 20-120; the card's
  uarch at c8p; the decode stream at 8 chunks a cycle); `tests/test_tech.py tests/test_target.py`
  24 passed, 1 skipped (no yosys DSP48E1 model). After the merge (8a9d75d3): (pending).
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
receipts there: `/home/ubuntu/opt/logs/p2-tok-naxi150*.log`). On the F2 path the VPU wait grows
to 10.0% of the cycles. With APS, `lens summary`: port B streaming 80.5%, MXU computing
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
"LUT-RAM-dense adapters congest"). Placement and routing: (running, below).

## Risks found on the real part

- **SLR crossings at 16 channels.** P2's floorplan (wide-port-alexshell's) keeps the board in SLR2
  and the HBM fabric in SLR0, every channel's native port across two boundaries in 3-stage
  register chains (`otpu_f2_hbm_axi` DW 604 + UW 532 = 1,136 signals a channel). The VU47P has
  3,840 LAGUNA sites a side of each boundary (Vivado on the part: SLR0 3,840, SLR1 7,680, SLR2
  3,840; ~23 K SLLs a boundary). Sixteen channels need ~18.2 K a boundary plus the host's and the
  control chains, ~80% of the boundary before the shell takes its share: unlikely to route.
  Ways down: the lane memories' adapters in SLR0 (the lane ports cross instead, ~12 K), the
  write data crossing once and fanned out in SLR0 (every write goes to every copy: the 8 copies'
  576-bit write ports carry the same data), the chains at clk_f with half the width, or the
  proposal's floorplan (the lanes, their chunk FIFOs and the ACT RAM in SLR0 beside the HBM, the
  rest in SLR2: only the ACT writes, the commands and the results cross; the vla branch's
  `otpu_mxu_arr` registered cut is the starting point, but the drain's same-cycle TMEM grant
  needs a credit scheme across the SLRs).
- **DSPs in SLR2**: ~2,400 packed MXU DSPs + ~170 VPU / quantizer of SLR2's 3,008 (85%): the
  packing is what lets eight lanes stay in one SLR (unpacked: ~4,450).

## What is next

1. **Place and route the F2 CL** (`boards/f2/build.py --target f2-p2:p3c150`, ImplCL from the
   synthesis above) on P2's floorplan, and read the SLR crossings first; then the floorplan or
   RTL change the crossings need (the list under "Risks"). A packaged Developer_CL tar for the
   coordinator only after a routed, qualified build.
2. **The cycles Lens names** (C 8 packed, 1.30 M against the 0.98 M bound): the MXU's start and
   drain per command (6.4%: each lane's command is an eighth of the MM; a third command held, or
   the MLP's 768-row chunks merged), the vector units (9.1%: the proposal's M4, VPU 32 lanes / CL 8
   and the quantizer at 32, or the MLP's software pipelining), the TMEM drain's arbitration.
3. **Memory capacity**: eight 2 GiB copies are the whole 16 GiB, so the copies hold Gemma 3 1B
   (~1.05 GB) and nothing larger; the proposal's laned layout (one image, lane l's rows on its own
   channels, group scale rows) is the way past that, an ISA change (Config.CLANES, MM LANE / bs).
4. Receipts still to take: the tokens at stall 5 and on the F2 channel path (OTPU_NAXI, the SLR
   chain and otpu_native_axi at 150 MHz: P2's latency 49), `tools/perf_qwen.py --layers 0 --pos 14
   --check` at c8p.
