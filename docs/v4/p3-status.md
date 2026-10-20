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
| C 8 packed, APS (8a9d75d3) | the lane memory (8 copies, 8 adapters) | (pending) | | | |
| C 8 unpacked, APS (8a9d75d3) | the same | (pending) | | | |

Before APS, Lens: port B "streaming" 68.9% but the lanes moved only 0.98 M chunk-cycles of work in
those 1.56 M cycles; every long stream ran at 2.3-2.5x its ideal time (`/home/ubuntu/opt/p3/
mmshape.py`). A 2,048-row int8 MM on the board path (`/home/ubuntu/opt/p3/p3_lane0.py`): C 1
19,128 cycles (ideal stream 18,432), C 4 5,299 (4,608), C 8 5,387 (2,304): lane 0's channels read
2,720 scale beats each at C 8 against 256 at C 1 / 4 (85 runs instead of 8), with 2,370 chunk
beats: saturated, and the lock step held every lane to lane 0. With APS: C 8 2,997 cycles (2,304
+ the same 693 fixed cycles as C 1 and C 4), 256 scale beats a channel, the 8 copies' hashes equal.

**LPN** (`tools/p3_lpn.py`, lpn-estimate; the vla tree e896ab4f's lpnlang net, lpnlang c55ef8d):

| | lanes | lpn-estimate | RTL (measured) | error |
|:--|--:|--:|--:|--:|
| Gemma 3 1B token, pos 14, lat 30 | 1 | 8,755,262 | 8,221,777 (P2's C 1 stall 0) | +6.5% |
| | 4 | 2,307,854 | 2,292,283 (P2's C 4 stall 0) | +0.7% |
| | 8 | 1,312,435 = 6.56 ms = **152.4 tok/s** at 200 MHz | (pending) | |
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
lanes would not fit. Whole-CL synthesis / placement on the F2: (pending).

## What is next

(written at the end of the session)
