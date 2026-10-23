# openTPU v4, candidate P3: eight chunk lanes with packed DSP48E2 (1 KiB/cycle, MXU x8)

Status: a proposal (2026-10-18), not built. Every number is derived here or cited as `file:line`;
nothing below was measured on an F2. The target is one Gemma 3 1B int8 sequence (W8A8, batch 1)
on an AWS EC2 F2 (VU47P + 16 GB HBM2e, 32 pseudo-channels (PCs) of 256 bits at 450 MHz = 460.8 GB/s),
core at 200 MHz (the user's synthesis of the current design on the F2).

## 1. Summary

- One slice, one sequencer, **C = 8 lock-step chunk lanes**: DRAM port B delivers one 1 KiB line
  (8 x 128 B chunks, 8 weight rows' block k) per core cycle = 204.8 GB/s, 44% of the HBM.
- The MXU becomes 8 lanes x (MCOLS 4 stationary rows x 128 deep) on **packed DSP48E2** PEs
  (two int8 MACs a DSP, `docs/mx_mxu.md` section 5): 4,096 MACs/cycle peak, 1,024 useful at M = 1,
  2,368 DSPs instead of 8 x 552. VPU 32 lanes, quantizer 32/cycle, TMEM 256K words in URAM, chunk
  FIFO 4 MiB in URAM, WIN 32, a laned weight layout in the image (one MM addressing change).
- Decode roofline: 7.81 M chunks a token / 8 = 0.976 M cycles = 4.88 ms = **205 tok/s**; realistic
  after bring-up **~165-175 tok/s** (short context 175, 2048-token context 164), floor ~130 if the
  per-group gating keeps 2 cycles. Prefill MXU-bound at ~4,096 MACs/cycle: ~800 tok/s projected.
- Risk: **high**. New DRAM adapter (32 AXI PCs), a reworked MXU issuer / row gating / epilogue, four
  units widened at once, a 200 MHz packed-DSP timing closure that measured 167-220 MHz out of
  context on a stand-in part, and an AWS build/bring-up flow nobody here has run yet.

## 2. Datapath changes

**HBM / port B** (`rtl/mem/otpu_native_dram.sv` replaced by a new `rtl/mem/otpu_hbm_dram.sv`;
`rtl/shell/otpu_mem_ch.sv`'s CDC/credit structure reused per PC). Lane c (0..7) owns PCs 4c..4c+3.
The slice's 32-bit byte address a maps as: line L = a / 1024, lane c = (a % 1024) / 128, offset
o = a % 128; lane c's chunks of the 8 lines 8m..8m+7 are one 1 KiB, 32-beat AXI burst on PC
4c + (m % 4) at PC address (m / 4) x 1024 + (L % 8) x 128 + o. So each PC serves 1 KiB every 32
core cycles (6.4 GB/s of its 14.4, 44%), bursts are long, and a lane's four PCs return in order
each; the lane merges them 4-way in burst order (the per-channel tag FIFO of `otpu_native_dram`,
widened). HBM read latency ~150 ns + the 450/200 MHz crossing + merge: budget 300-400 ns = 60-80
core cycles; at 25.6 GB/s a lane needs 10 KiB in flight = 10 bursts, 2-3 outstanding per PC;
the credits are chunk-FIFO slots reserved at the request, as today (`otpu_native_dram.sv` header).
Port A (scale lines, 32 B) and port SW (QST/KV byte writes, 32 B strobed beats: HBM AXI has
WSTRB, so no fill read-modify-write) get their own small requesters on PC 0 and PC 1 with APF
prefetch runs as today. The host applies the same map when it fills the memory
(`opentpu/host/board.py` channel_map, as CHASH today). 4 GiB a slice = 128 MiB a PC of its 512 MiB.

**MXU** (`rtl/mxu/otpu_mxu.sv`, new cells in `rtl/tech/xilinxus/`). Lanes are lock-step: one
issuer, one command pair, one ACT RAM read a cycle broadcast to the 8 lanes (registered per lane);
lane c computes row 8i + c of the stream against the same MCOLS = 4 stationary rows. A pipeline
"row" becomes a **group** of 8 rows: the first pop of a group needs its KB lines (not 8 x KB
separate rows) and one result-FIFO credit; RF stays 32 (groups), result rows are 32 integers
(8 lanes x 4 columns) with their Sw[8] and Sa[4]. Packing: `rtl/exp/mx_mxu/otpu_mx_us.sv`'s
column (two stationary rows per DSP48E2: a0 and a1 in the 30-bit A port 18 bits apart, times w on
B; CL = 16 keeps each 16-product field under 2^18 = 262,144 > 16 x 127 x 127 = 258,064; the sign
borrow is fixed at the chain end, bit-exact in xsim per `docs/mx_mxu.md:870`). Accumulation stays
the column end's exact 48-bit integer; the epilogue (i2f48, x Sw, x Sa 2^-W) is unchanged per
result, in E = 8 shared lanes (`otpu_mxu.sv:117` RF, `:338` the pop gate). Scales: group scale
rows (section 3) so one 32 B port-A line serves 4 blocks of all 8 lanes; the scale FIFO holds one
entry per lane-chunk. The issuer never spends a port-B cycle on an A request (today's "a request
of its own (a cycle without a chunk)", `otpu_mxu.sv` header) because A has its own PC.

**VPU** (`rtl/vpu/otpu_vpu.sv`): LANES 32, VPU_CL 8 (composite lanes), so EXP2 2.67 -> 10.7
cols/cycle, RECIP 4 -> 16, RSQRT 2 -> 8 (`docs/stream.md:476-482` x4). Why: gelu_tanh on 6,912 is
6912 x (1/2.67 + 1/4 + 6/8) = 9.5 K cycles a layer at 8 lanes against the gate+up stream of
2 x 6912 x 9 / 8 = 15.6 K cycles at C = 8; add the norms, the QACT and the row chain and the
8-lane VPU is on the critical path of every layer. At 32 lanes gelu is 2.4 K. The stream engine
(DSTEP/STREAM, unused by Gemma) stays 8 wide behind a lane split (ULANES-style), to keep
`otpu_se_comp`/`otpu_se_tail` untouched. `docs/seq_vpu.md:715-719` asks for the composite-width
rerun at HBM rates: the inventory above is that rerun's first-order answer for Gemma 3.

**Quantizer** (`rtl/vpu/otpu_quant.sv`): 32 elements a cycle both passes, XD 512 chunks of 32
words (a 6,912 row is 216 chunks), 4 group-max RAMs stay. Per layer: (1152 x 2 + 6912 + 1024) x 2 /
32 + 4 x ~60 latency = 0.9 K cycles (2.8 K at 8 lanes).

**TMEM** (`rtl/mem/otpu_tmem.sv`): 256K words, 32 banks, TMEM_IMPL 3 as the card's pick (one copy,
TDP, lost reads replayed, pumped 2x on core_clk2x = 400 MHz; URAM288 min period 2.0 ns at -2,
`docs/tmem_pump.md:680`) in URAM: 32 banks x NSB 2 = 64 URAM288. `rtl/seq/otpu_seq.sv:129`
already derives TAW from TMEM_WORDS (18 bits); the STREAM descriptors' 16-bit addresses
(`docs/scoreboard.md:618`) only constrain the STREAM area, which Gemma does not use.

**Sequencer** (`rtl/seq/otpu_seq.sv`): WIN 32 (the RTL default in `otpu_slice.sv:46`), SEQ_WC off
(its 4-bit fields need WIN <= 16, `opentpu/target.py:198`), the LD/ST segment W = min(D/4, LANES)
= 32 words = 128 B a cycle (`rtl/dma/otpu_dma.sv` header). One new register form, RLD ADD
(R[rc] = R[ra] + R[rb]), for laned KV addresses (section 3).

**Config / uarch keys** (`opentpu/target.py`, `targets/f2.toml`): arch: S 1, D 128 (the
quantization block does not change), **CLANES 8 (new: programs depend on it through the layout,
so it is in CFG_HASH)**, MCOLS 4, ACT_ROWS 16, ACT_BLOCKS 128, LANES 32, TMEM_WORDS 262144,
IMEM_WORDS 32768, PAIR/DSTEP/STREAM true, SEQ_WC false. uarch: MXU_IMPL 2, MXU_CL 16, MXU_E 8,
**MXU_PACK 1 (new)**, VPU_CL 8, ULANES 32, SE_LANES 8 (new), TMEM_IMPL 3 (+ the card's TMEM_* knobs,
TMEM_PMP 1, URAM), RPB 0, WPB 1, WIN 32, FIFO_DEPTH 4096 (per lane), QUANT_XD 512, PQ_WIN,
TRACE_DEPTH 16384, CHASH n/a (the lane map replaces it). platform: core_mhz 200, mem "hbm",
mem_ctrl "xilinx_hbm", channels_per_slice 32, channel_bytes 134217728, beat_bytes 32,
hbm_mhz 450 (new), host "pcie" through the F2 shell.

## 3. ISA / compiler / host changes

**MM, laned layout.** Today block k of row n is at `sa + n*rs + k*D` (`docs/isa.md:179-200`). A
new flag bit8 `LANE` reads block k of row n at `sa + (n / C) * rs + k * bs + (n % C) * D`, with rs
the group stride and bs a new block stride (w7's high half is free: srs needs 16 bits). Weights:
bs = C x D = 1 KiB (a group is KB contiguous lines). N % C != 0: the image pads the last group with
zero rows, the drain discards results beyond N. Non-LANE MMs still run, on lane 0 at 128 B/cycle
(compatibility, odd tensors). PAIR/4-bit, RSC, ACC, ASCALE, RMAX are unchanged per result.
**Scale rows** of a group: 8 x Sw (32 B), then 8 bytes (one dw a lane) per block, so KB = 9 is 104 B
= 3.25 lines, srs the group stride; k0 indexes blocks as before. The ISA simulator
(`opentpu/isasim.py`) implements the same address function and the group scale row; `opentpu/isa.py
mm()` takes `lane=True, bs=`; `Config.CLANES`.
**KV caches.** K: laned with bs = the plane stride (slot s's block k at kv + k x PLANE + s x 128,
linear in s), so `kv_append`'s QST and the resident decode's `RLD MUL` arguments (`docs/autodecode.md`
"Run-time arguments") work unchanged. V^T (rows = 256 dims, columns = tokens): milestone 1 keeps it
legacy (lane 0, 128 B/cycle: +34 K cycles a token at a 2048 context, 0 at the trace's 14);
milestone 2 lays it laned with the token's byte at (t / 128) x 1024 + t % 128, formed per token
from two state words (tblk, tin) with RLD MUL + RLD ADD.
**Compiler / image** (`opentpu/compiler.py`, `opentpu/llm/gemma4.py` Image/Weights): every weight
tensor written as `w.reshape(N/8, 8, KB, 128).transpose(0, 2, 1, 3)` with its group scale rows;
`ol.dot` emits LANE MMs; the LM head's chunks stay `min(HEAD_CHUNK, tmem_words/8)` = 32,768 rows
(`qwen3.py:1689`); the embedding gather reads one laned row (8 x 128 B at stride 128 within a
line: the gather kernel's address changes). `swiglu_down`'s down-projection chunks and
`_attend_heads` are unchanged in structure. The compiler should software-pipeline the MLP: gate/up
in two column halves so gelu/QACT of half 1 overlaps the MM of half 2 (section 5).
**Host / Engine / otpu-chat.** `device_config` reads a new CFG register (CLANES, PCs) and
`check_fits` refuses non-laned programs on a laned device via the arch hash; `channel_map` applies
the PC map; the transport is the F2 shell's XDMA (`/dev/xdma*`, the F2 developer kit), so
`opentpu/host/board.py` keeps its read/write calls and only the BAR/OCL register window moves. Use
the on-card generate loop (`docs/autodecode.md`): the host reads 64 B a token; its 0.3 ms a token
(60 K cycles at 200 MHz, 6% of a 5 ms token) leaves the critical path.

## 4. Resource budget on VU47P (9,024 DSP48E2 / 2,016 RAMB36 / 960 URAM288)

| Resource | Item | Count | Derivation |
|---|---|---:|---|
| DSP | MXU PEs | 2,048 | 8 lanes x 4 rows x 128 / 2 packed |
| DSP | column ends | 320 | 8 lanes x 4 columns x 10 (552 - 512 = 40 per 4 columns today, `docs/mx_mxu.md:885`) |
| DSP | epilogue lanes E 8 | 48 | 6 a lane (`docs/mx_mxu.md:890`) |
| DSP | VPU 32 lanes | ~180 | 86 today (`docs/vpu_pump.md:63-67`) + 3 x 32 more slot-0 units; SE stays 8 wide |
| DSP | quantizer 32/cycle | ~310 | 716 - 552 - 86 = 78 a slice today, x4 |
| DSP | **total** | **~2,910 (32%)** | against 9,024; today's slice 716 |
| RAMB36 | ACT RAM 16 x 128 x 128 B | 68 | 64 data + 4 record at MCOLS 4, full at ACT_ROWS 16 (`docs/mx_mxu.md:1420-1431`) |
| RAMB36 | scale FIFO | ~40 | 8 lanes x 4096 x ~40 b |
| RAMB36 | IMEM 32K words | 32 | 1 Mbit |
| RAMB36 | quantizer XD 512 x 128 B | 16 | 64 KiB |
| RAMB36 | DMA LD buffer 128 x 1 KiB lines | 32 | 128 KiB (one line a cycle) |
| RAMB36 | trace 16K, result FIFO, misc, shell | ~90 | |
| RAMB36 | **total** | **~280 (14%)** | against 2,016 |
| URAM | chunk FIFO 8 x 4096 x 128 B | 128 | 16 URAM288 (4096 x 72) side by side a lane = 1152 >= 1024 bits; 4 MiB |
| URAM | TMEM 256K x 32 b, one copy pumped | 64 | 32 banks x NSB 2, a URAM288 each (two words a 72-bit row, `docs/bcu1525.md:233`) |
| URAM | **total** | **192 (20%)** | against 960; 128 + 64 must sit in SLR0/SLR1 (320 each) |
| LUT | slice | ~400 K (31%) | today 152 K yosys (`docs/bcu1525.md:222`); 8 packed lane datapaths at ~11 K (`mx_mxu.md:886`), VPU 4x of 31 K, 32-bank TMEM rotators, the adapter |
| HBM | port B | 204.8 GB/s (44.4%) | 1 KiB x 200 MHz; 6.4 of 14.4 GB/s a PC |
| HBM | ports A + SW | ~1.7 GB/s | scales 1/128 of the stream + 4 B a row; KV ~0.6 KB a layer a token |

## 5. Expected performance, Gemma 3 1B int8, 200 MHz

**Roofline.** Per layer 209,664 chunks (15,360 rows x 9 + 1152 x 8 + 1152 x 54 = 26.8 MB), 26 layers
5.45 M; LM head 262,144 x 9 = 2.36 M; 7.81 M chunks = 1.00 GB (the trace's 7.85 M includes KV and
tables). At 8 chunks a cycle: layers 681 K + head 295 K = **976 K cycles = 4.88 ms = 205 tok/s**.

**Losses, re-estimated from the 133 MHz trace** (10.28 M cycles):
- *Scale with the stream* (/8 then at 200 MHz): the 7.85 M streaming cycles -> 0.98 M.
- *Per row, today 2.27 M cycles* ("MXU not consuming" 18.2% + "computing, not streaming" 3.9%) over
  781 K rows (519 K layer rows + 262 K head rows) = **2.9 cycles a row** (the head alone: 0.74 M /
  262 K = 2.8). Lock-step lanes turn rows into 97.7 K groups; the gating redesign (a group's lines
  present, A on its own PC, RF 32 groups, E 8 draining a group's 8 results in one cycle at M = 1)
  targets <= 0.5-1 cycle a group: **50-100 K cycles (5-10%)**. This is the make-or-break number.
- *Fixed per instruction* (do not shrink with C): 5,350 instructions, 922 MM, 1,825 VOP, 366 QACT.
  The VPU's 4.5% in flight (463 K cycles) and the quantizer's 1.2% (123 K) shrink 4x with the
  widening to ~115 K + 31 K, but they now run against a 15.6 K-cycle gate/up stream a layer, not
  124 K: what is exposed is the dependent chain per layer (norm -> QACT -> MM -> ... -> gelu -> QACT
  -> down MM) minus what the 4 MiB chunk FIFO prefetches of the next MM (the MXU holds two
  commands, `otpu_mxu.sv` header). Budget **1.0-1.5 K exposed cycles a layer = 26-40 K**; the
  attention chain at a 2048 context (q.K^T, RMAX, EXP2 of 4 x 2048 at 10.7/cycle = 0.8 K, RSUM, P
  QACT 0.5 K, P.V, RECIP) adds ~1.5 K a layer = **+40 K**; the legacy V^T stream **+34 K** at 2048.
- *HBM jitter / chunk FIFO starved* (0.6% today): 32 PCs at 44% with 2-3 bursts outstanding each:
  budget **3% = 30 K**. *TMEM arbitration* (2.1% today, the MXU drain): 8 writes a cycle into 32
  banks: **<= 1% = 10 K**. *Dispatch* (WIN 32, 5,350 instructions): **~20 K**. LM-head tail (last
  chunk's ARGMAX at 32 lanes 1 K + drain): **5 K**. Host: 0 with the on-card loop (else 60 K).

| | cycles | ms | tok/s |
|---|---:|---:|---:|
| roofline | 976 K | 4.88 | 205 |
| short context (the trace's position 14) | ~1.14 M | 5.7 | **~175** |
| 2048-token context, V^T legacy | ~1.22 M | 6.1 | ~164 |
| if group gating keeps 2 cycles a group | ~1.35 M | 6.8 | ~148 |
| C = 8 but VPU / quantizer left at 8 lanes | +9.5 K x 26 exposed | ~7.3 | ~137 |

Per-token budget: layers ~800 K (681 K stream + the chains), LM head ~330 K (295 K + ~10% gating
+ ARGMAX tail). Today's split is 7.2 M / 3.1 M at 133 MHz (`docs/gemma3.md` Lens).

**Prefill** (qualitative, a projection): M = 4 rows a pass shared by the lanes and ACT_ROWS 16
replay (4 groups) give 4,096 MACs/cycle at a quarter of the stream rate; a 1024-token prompt is
64 passes x 4 x 976 K = 250 M cycles = 1.25 s, ~800 tok/s before attention, QACT and the row-by-row
K/V appends (today's card: 41.9 tok/s on the scoreboard's prompt, `docs/scoreboard.md:307`).
The tile kernel and MCOLS 8 (another +1,184 DSPs, 4,096 more MACs) are the follow-ups if prefill
matters; `docs/scoreboard.md:315` puts the F2 "port 512 B + MXU x4" at 663 prefill tok/s.

## 6. Timing and risks at 200 MHz

- **Packed DSP48E2 column.** Out of context on the xcku5p-2 at 7.5 ns: 2 packed columns E 2
  WNS +2.997 ns (~220 MHz), 4 packed columns E 4 +1.517 ns (~167 MHz) (`docs/mx_mxu.md:917-921`);
  the worst path is "the chain ends into the column end (the packed split)" (`:946`). Use the
  2-column (4-row) unit per lane and add a register at the packed split (+1 cycle latency, free in
  LROW). Rerun OOC on the VU47P -2 part in week 1; if it misses, pump the column ends on core_clk2x
  or drop CL to 8 (two chains a group, CPG 2, `otpu_mxu.sv:110-111`).
- **URAM at 200 / 400 MHz.** P3/pumped URAM met 400 MHz with +0.180 ns on the stand-in
  (`docs/tmem_pump.md:725`); the chunk FIFO is unpumped. Cascaded URAM columns in one SLR only.
- **The 1 KiB datapath.** 8 x 1024 data bits from the adapter plus the 8 x 4,096-bit ACT broadcast
  are 40 K wires: register per lane, lanes in SLR0 beside the HBM's 32 AXI ports (the HBM stacks
  are on SLR0's edge), ACT RAM and the quantizer's write path in SLR0, TMEM/VPU/sequencer in SLR1
  (the drain's 8 x 32 b and the quantizer's 32 B a cycle cross with `USER_SLL_REG` registers as
  the BCU1525's `otpu_xr`, `docs/bcu1525.md:316-330`). SLR0 holds 3,008 DSPs: the MXU's 2,416 is 80%
  of it, so the VPU and quantizer DSPs go to SLR1.
- **Clock domains.** core 200, core_clk2x 400 (TMEM), HBM AXI 450 (32 async FIFO pairs, the
  `otpu_mem_ch` pattern), the shell's PCIe 250. The 450/200 crossing at 1 KiB a cycle is 32 FIFOs
  of 256 bits: fine; the credits make read data never back-pressure (as today).
- **HBM efficiency.** Reads only, 1 KiB bursts, 44% load: the IP's documented ~90% at long bursts
  is ample; the risk is page conflicts between the A/SW streams and B on PCs 0-1 (give A and SW
  their own PCs if it shows: the budget has 30 spare).
- **Gating redesign correctness**: the "row never stalls in its middle" invariant must hold for
  groups; the simulation `$fatal` checks (`otpu_mxu.sv:342-343`) carry over per lane.
- **Verilator time.** 8 lanes x 512 MACs x a 1.03 GiB image: the board model will run one token in
  hours; use the ISA simulator for programs and the RTL on tiny models plus one recorded token.

## 7. Implementation plan

| # | Milestone | Effort | Verification | Reused from BCU1525 |
|---|---|---|---|---|
| 0 | OOC timing of one packed lane (2 columns, E 2) + column end on xcvu47p-2 at 5.0 ns; URAM FIFO slice | 1 wk | WNS >= +0.3 ns at 200 MHz, hold met | `rtl/exp/mx_mxu/run_ooc.sh`, `rtl/tech/xilinxus` cells |
| 1 | ISA: `Config.CLANES`, MM LANE/bs, group scale rows, RLD ADD in `isasim.py`, `isa.py`; image layout + `ol.dot` in the compiler; `targets/f2.toml`; `test_config_matrix` arm | 2 wk | `tests/test_gemma3.py` tiny model bit-exact vs numpy; `compare_hf.py --cfg f2`; Lens `--isa` budget at C 8 | target loader, config matrix, test harness |
| 2 | RTL MXU: lanes, group gating, RF/E 8, packed PE + generic behavioural twin, group scale reader; `otpu_actram` broadcast | 4 wk | token-exact tiny Gemma 3 on the board model vs ISA sim; `tests/test_tech.py` packed vs generic cycle for cycle | `otpu_mx_us.sv`, `otpu_pe`/`otpu_mxce` structure |
| 3 | `otpu_hbm_dram.sv`: 32 AXI masters, lane merge, credits, A/SW requesters; behavioural 8-lane `otpu_dram` with latency for Verilator; xsim against the HBM IP model | 3 wk | the board model's DRAM after a token bit-identical to the ISA sim; latency sweep 50-200 cycles with no starvation | `otpu_mem_ch` CDC/credits, `otpu_afifo` |
| 4 | VPU 32 / CL 8 with the SE lane split, quantizer 32 / XD 512, TMEM 32 banks 256K URAM (card's IMPL 3 knobs), WIN 32, DMA 32-word segments | 3 wk | every token-exact test at `OTPU_TARGET=f2`; `tmem_pump`'s replay-cost check | `TMEM_URAM` xdc, `vpu_pump` testbenches |
| 5 | `rtl/boards/f2/`: the AWS CL top (shell interfaces, HBM IP, OCL registers, XDMA), floorplan pblocks (SLR0 lanes / SLR1 the rest), `boards/f2/` flow (`aws_build_dcp_from_cl`, AFI) | 3 wk | `make synthcheck`-style elaboration; a first 100 MHz AFI | `boards/bcu1525` flow scripts, `otpu_board`, `otpu_ctrl` CFG registers |
| 6 | Bring-up on an f2.6xlarge: CFG registers, HBM write/readback, tiny model tokens, Gemma 3 greedy 12 tokens vs HF, then the 200 MHz AFI and perf | 3-4 wk | `compare_hf.py --backend board`; `otpu-chat --model gemma3 --backend board`; Lens `--axi` record | `otpu-ddr4`-style bring-up tool, `tools/qual/perf.py` |

About 4-5 months of one engineer, with milestones 1-4 overlappable by two. Milestone 2 (the V^T
laned layout) and MCOLS 8 come after first light.

## 8. What to verify in Lens once built

- Port B streaming (of 1 KiB lines): 76.4% today -> **>= 85%** of cycles; roofline fraction >= 85%.
- "MXU not consuming (drain / row credits)": 18.2% -> **< 5%** (the group gating); "MXU computing,
  not streaming": 3.9% -> < 2%; attribute both per group with a new counter (gating reason: lines
  missing, scales missing, RF credit).
- Chunk FIFO starved: 0.6% -> < 2% at 60-80-cycle HBM latency; new strips: per-lane FIFO level,
  outstanding bursts per PC, lane-merge stalls.
- TMEM arbitration on the MXU drain: 2.1% -> < 1% at 32 banks. VPU / quantizer exposed: 0.2% /
  0.5% -> each < 2-3% (they grow in share by design; if > 5%, the MLP software pipelining first).
- MACs a cycle: 99 -> **>= 850** at M = 1 (of 1,024 useful, 4,096 peak). Timeline split: layers
  ~0.8 M, LM head ~0.33 M cycles; the head's gaps gone.
- Prefill: MXU in flight >= 90% on a 1024-token prompt with replay; ARGMAX and the K/V append not
  on the critical path (the floorplan view's roofline-gap bar).

## 9. Every other unit, resized

| Unit | Today (card, `targets/ypcb-00338.toml`) | This candidate | Why |
|---|---|---|---|
| ACT RAM (`otpu_actram`) | ACT_ROWS 16 x 128 blocks x 128 B; read MCOLS x D = 512 B/cycle; 64 + 4 RAMB36 | same store and read width (4 rows x 128 B), **one read broadcast to 8 lanes** (a register stage a lane); 68 RAMB36; `r_use` check per lane | lock-step lanes read the same block each cycle; a per-lane ACT read would need 8 x 64 RAMB36 |
| MXU epilogue lanes E | E 4 (MCOLS), up to 4 results a cycle | **E 8**: 48 DSP; at M 1 a group's 8 results in one cycle, at M 4 32 results in 4 (KB 9 rows produce 3.6 a cycle); attention q.K^T (KB 2, M 4: 16 a cycle) runs at half rate, ~1% of the token | the drain must empty a group faster than groups arrive |
| Result FIFO / row gating | RF 32 rows, `rows_live + rows_p < RF`; a row's first pop needs its KB chunks + scales | RF 32 **groups** (32 x (32 x 48 b + scales) in distributed RAM); the pop gate per group; `KB <= DEPTH` per lane | the per-row 2.9 cycles become per-group; RF never binds (32 groups x 9 cycles >> LROW 25) |
| Chunk FIFO | 1024 x 128 B = 128 KiB in RAMB36 (1024 cycles of cover) | **4096 lines x 1 KiB = 4 MiB in 128 URAM288** (4096 cycles): HBM latency 60-80 cycles needs 10 KiB a lane; the rest hides the inter-MM chains (section 5) | at 1 KiB a cycle 128 KiB is 128 cycles: less than one HBM round trip |
| Port A (scales) | one 32 B line a cycle, shares the DDR3 channels with B | one line per 4 blocks of a group (8 dw bytes a block) + one Sw line a group; own PC, APF runs; ~1.7 GB/s | 8 independent scale readers would need 8 A lines a cycle |
| Port SW (QST / KV) | 64 B beats gathered, partial beats fill-read then written | 32 B strobed AXI writes, no fill reads; own PC; 256 writes a layer a token for V^T | HBM AXI WSTRB; the fill reads held the channel today |
| Quantizer | 8 elements/cycle, XD 1024 chunks of 32 B (8 RAMB36) | 32 elements/cycle, XD 512 chunks of 128 B (16 RAMB36), ~310 DSP; writes 32 B a cycle into ACT RAM (`otpu_actram` w_data LANES bytes) | 2.8 K -> 0.9 K cycles a layer against a 15.6 K-cycle MLP stream |
| TMEM words / banks / ports | 64K words, 8 banks, IMPL 3 one copy pumped 2x, RPB 0 / WPB 1, 64 RAMB36 | 256K words, 32 banks, the same IMPL 3 pick in 64 URAM288 pumped at 400 MHz; RPB 0 / WPB 1; rotators 32 x 5 | VPU 32 + quantizer 32 + drain 8 + DMA 32 lane requests a cycle; the head's 32,768-word logits chunk and 16 prefill rows |
| DMA LD/ST | 8 words = 32 B a cycle a segment (`min(D/4, LANES)`), LD buffer 128 chunks | 32 words = 128 B a cycle, requests whole 1 KiB lines (8 chunks) on port B, LD buffer 128 lines (32 RAMB36); the logits ST (1 MiB a token in host-driven decode) 32 K -> 8 K cycles; the on-card loop stores no logits | the 1 MiB store would be 3% of a 5 ms token at 32 B a cycle |
| IMEM / program / sequencer | 32K words (4K instructions), WIN 16, TAW 16, SEQ_WC on | 32K words (the loops keep the static program small; 5,350 is the dynamic count), **WIN 32**, TAW 18 (derived), SEQ_WC off, RLD ADD | more units in flight a layer; 256K words |
| Collective (`otpu_coll`) | present at S 1 (GATHER path) | kept, LANES 32, unused by Gemma (S 1) | ISA completeness; drop only if LUTs bind |
| PCIe / host | XDMA Gen2 x8, 64 B a token in the card loop; 0.3 ms host a token | the F2 shell's XDMA (~20 GB/s assumed, `tools/scoreboard.py:340`), same 64 B a token; image upload 1.03 GiB < 1 s | the host leaves the per-token path |
| Clock domains | core 133.33, core_clk2x 266.67, LiteDRAM sys, PCIe 250 | core 200, core_clk2x 400 (TMEM), **HBM AXI 450** (32 CDC FIFO pairs), shell PCIe 250 | the even divider rule (`docs/board.md:699`) holds at 200/400 |
| VPU | 8 lanes, CL 2, SE 8 | 32 lanes, CL 8, SE 8 behind a split; ~180 DSP | section 2 |
| MXU array | 4 columns x 128, 552 DSP, 512 MACs (128 useful at M 1) | 8 lanes x 2 packed columns x 128, 2,368 DSP, 4,096 MACs (1,024 useful at M 1) | the port's 8 chunks a cycle |

Section 4's totals (DSP ~2,910, RAMB36 ~280, URAM 192) are this table's sums.
