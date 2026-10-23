# openTPU v4 candidate P2: four lock-step chunk lanes (512 B/cycle, MXU x4) on the AWS F2

Status: a proposal (2026-10-18). Every number is derived here or cited as `file:line`; nothing
below is measured on the F2. "Today" is the YPCB card config (docs/gemma3.md "Lens": 133.33 MHz,
10.28 M cycles a decode token, 13.0 tok/s), run at 200 MHz as the F2 synthesis already does.

## 1. Summary

- One slice, one sequencer, DRAM port B widened to C = 4 chunks of 128 B a cycle (512 B/cycle =
  102.4 GB/s at 200 MHz, 22% of the HBM2e's 460 GB/s), fed by a new HBM adapter over all 32
  pseudo-channels; the MXU becomes 4 lock-step lanes (each today's MCOLS 4 x 128 array), lane l
  streaming row n + l of every MM, so M = 1 decode does 512 useful MACs a cycle (4x today's 128).
- VPU 16 lanes (composites 4 cols/cycle), quantizer 16/cycle, TMEM 128K words in URAM (16 banks),
  ACT_ROWS 16, chunk FIFO 1024 chunks a lane. The programs do not change: C is timing-only.
- Decode roofline 1.96 M cycles = 9.8 ms = 102 tok/s; realistic 2.5-2.8 M cycles = **69-79 tok/s**
  (host path included), 5-6x today's 13.0 tok/s; prefill MXU-bound at 2,048 MAC/cycle, ~300 tok/s.
- Resources: ~2,540 DSP (28%), ~340 RAMB36 (17%), 32 URAM288 (3%) of the VU47P; no DSP packing.
- Risk: medium-high. New memory adapter (32 AXI ports at 450 MHz, a 32-to-4 chunk network), a
  4-lane MXU, a 16-lane VPU and a new shell; each piece has a precedent in the tree.

## 2. Datapath changes

**HBM / port B** (new `rtl/mem/otpu_hbm_dram.sv` in place of `otpu_native_dram.sv`; `otpu_mem_ch`
is not used: the HBM IP's 32 AXI3 256-bit ports at 450 MHz replace LiteDRAM's native ports).
- 32 pseudo-channels of 256 bits: 32 B x 450 MHz = 14.4 GB/s each (460 GB/s in all). One AXI
  master per PC; each master only addresses its own PC, so the HBM IP's lateral switch carries
  no traffic. A 128 B chunk is one 4-beat burst on one PC; the adapter issues 512 B bursts (4
  chunks of consecutive row bytes) to keep HBM page locality.
- Address map: chunk m of the slice's flat 4 GiB goes to PC `hash(m)` over 32 PCs (CHASH's rule,
  `otpu_native_dram.sv:10-15`, widened from 2 channels to 32; `opentpu/host/board.py
  channel_map` applies the same). The four lanes' streams are therefore spread over all PCs
  whatever the row stride rs, and a lane's 128 B/cycle (25.6 GB/s) draws 0.8 GB/s from each PC.
- Lane feed: per PC an async read-data FIFO (450 -> 200 MHz, 256 b in, 1024 b out: one chunk a
  core cycle a PC); per group of 8 PCs an 8:2 chunk mux (2 chunks a cycle a group, 2x the
  group's share of the demand); a 4-group x 4-lane crossbar into the lanes' chunk FIFOs, each
  with 2 write ports (even / odd slots). Read data lands in the slot reserved at the request
  (the DMA's rule, `rtl/dma/otpu_dma.sv:8-9`), so out-of-order PCs need no reorder buffer.
- Why rows across lanes, not KB chunks of one row: lane l streams row n + l, block k, in the same
  cycle as the other lanes stream their rows' block k. (a) ACT RAM: all lanes read the same
  block of the same stationary rows, one read broadcast to 4 lanes, the ACT RAM's 68 RAMB36
  unchanged (KB-interleave needs 4 different blocks a cycle: 4x the read RAMs, 272 as MCOLS 16's,
  docs/mx_mxu.md 1412-1416). (b) Numerics: each row's `acc` is summed in one lane in today's
  order (docs/isa.md 199-212); KB-interleave would add four partial integers (exact, but a new
  column end) and change every drain. (c) Output order: a row-group's results y[j][n..n+3] are 4
  consecutive TMEM words: one contiguous write. (d) ISA: no field changes; N padded to a multiple
  of 4 when it is not (Gemma 3's N are 1024 / 256 / 1152 / 6912 / 262144, all multiples of 4).
- Read latency: HBM IP read latency is a few hundred ns on the AXI side; with the async FIFO,
  mux and crossbar take 60-100 core cycles at 200 MHz (to measure on the F2, milestone M7). At 1
  chunk/cycle/lane that is 100 chunks in flight a lane against a 1024-chunk FIFO: a 10x margin;
  the adapter keeps up to 128 chunk requests in flight a lane (one AXI ID per 4-chunk burst).

**MXU** (`rtl/mxu/otpu_mxu.sv`, parameter `C` = 4 chunk lanes; today's module is C = 1).
- 4 arrays of MCOLS 4 x 128 (today's `otpu_pe` / `otpu_mxce` cells, `rtl/tech/xilinxus`), 4
  issuers in lock-step (one command, 4 address generators: `sa + (n+l)*rs + k*D`), one ACT read
  registered once per lane, 4 scale readers on port A, 4 chunk FIFOs of DEPTH 1024.
- Decode (M <= MCOLS): a row-group (rows n..n+3) is one pipeline row; its first advance pops when
  all 4 lanes hold their KB chunks and scales (today's `row_in`, `otpu_mxu.sv:332-338`, per
  lane) and fewer than RF = 32 row-groups are live (today's RF, `:117`; the entry is 4x wider).
  The `row_in` of row-group r + 1 is evaluated while r pops, so a row transition costs no bubble
  (today a cycle without a pop is one, `:330`).
- Prefill (M > MCOLS, up to ACT_ROWS 16): replay from the lane FIFOs as today (`:38-43`), each
  lane replaying its own row G = ceil(M / 4) times. The chunk-cycles equal the MACs / 2,048, so
  the stream is 4x faster than the compute: prefill is MXU-bound, no "broadcast mode" needed.
- Drain: E = 4 epilogue lanes (MXU_E 4, as the card); at M = 1 one row-group (4 results) a
  cycle, 4 consecutive output words (a rotator port, not the GEN crossbar); at M = 4 the 16
  results of a row-group drain over 4 cycles, which needs KB >= 4 (Gemma 3: 8, 9, 54). RMAX folds
  the 4 lanes' y in n order inside the cycle (4 chained fp max), so its result stays bit-exact.
- MACs: 4 x 4 x 128 = 2,048 a cycle; M = 1 uses 512 (25%, against 128 of 512 today: docs/
  mxu_systolic.md, "at M=1 only 128 of 512").

**VPU** (`rtl/vpu/otpu_vpu.sv`): LANES 16, VPU_CL 4 (composites 4 cols/cycle). Justification: the
VPU is in flight 4.5% x 10.28 M = 0.46 M cycles a token today and the quantizer 1.2% = 0.12 M;
against a 1.96 M-cycle stream that is 30%, and the FIFO hides at most 1024 cycles of it per MM
boundary, so about half would be exposed (+0.29 M, +15%). At 16 lanes / CL 4 the VPU time halves:
gelu_tanh's 2 x 6912 composites a layer (EXP2 + RECIP for tanh) x 26 = 359K columns: 180K
cycles at CL 2, 90K at CL 4; the 1,825 VOP.mul (GeGLU's 6912-wide product, the norms' x*r*gamma
over 1152) halve at 16 lanes; rmsnorm's 182 one-element RSQRTs a token (2 x 26 + 5 x 26 heads)
are latency-bound (~100 cycles each, 18K cycles, 0.7%) and stay. Rerun docs/seq_vpu.md 715-719's
composite-width study at this rate before freezing CL (section 7, M1).

**Quantizer** (`rtl/vpu/otpu_quant.sv`): LANES 16 elements a pass, the XD row buffer at 512
chunks (a 6912-element row is 432 chunks of 16), the ACT RAM write 16 B a cycle.

**TMEM** (`rtl/mem/otpu_tmem.sv`): 128K words, 16 banks (= LANES), TMEM_IMPL 3 pumped 2x on
core_clk2x (400 MHz) as the card (`targets/ypcb-00338.toml` TMEM_PMP 1), in URAM: P3 in UltraRAM
meets 400 MHz (docs/tmem_pump.md 726, 787). 128K because ACT_ROWS 16 prefill tiles hold 16 x
6912 = 110K words of GeGLU intermediate (> 64K), as the card's m8 variant already needs 128K for
8 rows (`ypcb-00338.toml` TMEM_WORDS comment).

**Sequencer** (`rtl/seq/otpu_seq.sv`): TAW follows TMEM_WORDS already (`:129`: 17 bits at 128K);
STREAM descriptors stay 16-bit, so the compiler keeps stream buffers below word 65536
(docs/scoreboard.md ~620). WIN 16, SEQ_WC (WAITC v2) on, as decided for the F2 in docs/seq_vpu.md.

**Keys** (`targets/f2.toml`, `opentpu/target.py` UARCH_KEYS + new ones): `[arch]` S 1, D 128,
MCOLS 4, ACT_BLOCKS 128, ACT_ROWS 16, TMEM_WORDS 131072, IMEM_WORDS 32768, LANES 16, PAIR / DSTEP /
STREAM / SEQ_WC true. `[uarch]` **MXU_C 4** (new: chunk lanes), MXU_E 4, VPU_CL 4, ULANES 16,
TMEM_IMPL 3 + TMEM_PMP 1 + TMEM_NSB 2 + TMEM_URAM, RPB 0, WPB 1, WIN 16, FIFO_DEPTH 1024 (a lane),
**HBM_PCS 32, HBM_GROUPS 4, HBM_INFLIGHT 128, PCHASH** (new). `[tech]` xilinxus, xcvu47p-fsvh2892-2.
`[platform]` mem "hbm", mem_ctrl "xilinx_hbm", channels_per_slice 32, channel_bytes 512 MiB
(16 GB / 32; the slice addresses 4 GiB of it: 32-bit addresses), beat_bytes 32, core_mhz 200.

## 3. ISA / compiler / host changes

- **ISA: none.** MM's fields and semantics (docs/isa.md 179-212) are unchanged; the lane
  assignment n mod 4 is the RTL's. Results are bit-identical to C = 1 (one lane per row, RMAX
  folded in n order), so `MXU_C` is a uarch key and `target.config` is the same arch as the
  card's but for LANES / TMEM / ACT_ROWS: the YPCB's programs and images run unchanged.
- **Image layout:** unchanged. The row interleave is in the address generators, not the image;
  the only layout rule is the PC hash in `opentpu/host/board.py channel_map` (host DMA writes
  and reads apply it, as CHASH today).
- **ISA simulator** (`opentpu/isasim.py`): the analytic timing takes C: an MM costs
  `ceil(N / C) x KB` chunk-cycles (+ the row-group bubble model of section 5); Config grows
  nothing (C comes from `sim_uarch`). `tools/scoreboard.py` BOARDS["f2"] becomes S 1, 200 MHz,
  C 4 (today's preset assumes S 4 at 250 MHz, `tools/scoreboard.py:345-348`).
- **Compiler** (`opentpu/compiler.py`, `opentpu/llm/gemma4.py`): pad N to a multiple of C where
  it is not (no Gemma 3 matrix needs it; attention's N = ctx is padded by the KV ring already);
  prefer M = 16 tiles (ACT_ROWS 16) for prefill; stream buffers below word 65536.
- **Host / otpu-chat / Engine:** the F2 shell's PCIe (AWS shell DMA) replaces XDMA in
  `opentpu/host/board.py`; the on-card decode loop still reads 64 B a token (docs/autodecode.md);
  `device_config` reads the new CFG registers (channels 32, MXU_C in CFG_UARCH).

## 4. Resource budget on the VU47P

| resource | derivation | total | of the VU47P |
|---|---|---:|---:|
| DSP48E2 | MXU: 4 lanes x 4 columns x 138 (`docs/mxu_systolic.md:69-71`) = 2,208 (the card's MXU is 552 of its 716, `docs/bcu1525.md` 7); VPU 16 lanes + CL 4 + quantizer 16: 2 x (716 - 552) = 328 | ~2,540 | 28% of 9,024 |
| RAMB36 | chunk FIFOs 4 x 1024 x 128 B = 512 KiB = 128; scale FIFOs 8; ACT RAM 68 (ACT_ROWS 16, MCOLS 4: `docs/mx_mxu.md` 1412); IMEM 32; DMA buffer 8; 32 PC async FIFOs 16; group muxes 32; quantizer XD + VPU 8; trace 16; misc 20 | ~340 | 17% of 2,016 |
| URAM288 | TMEM 128K words x 32 b = 4 Mib, one copy pumped 2x, 16 banks x NSB 2 = 32 RAMs of 4K x 32 b, one URAM288 each (half its width; 16 with yosys' two-words-a-row packing, `docs/tmem_pump.md` 676-677) | 32 | 3% of 960 |
| LUT | 4 x 22.5K MXU (`docs/mxu_systolic.md:127`) + 60-80K chunk network + 2 x ~45K VPU/quantizer + 60K rest = ~300K | ~300K | ~23% of 1.3 M |
| HBM | stream 512 B x 200 MHz = 102.4 GB/s; scales 1 B a block + 4 B a row = 10.7 MB a token (1.1%); KV + logits 1.1 MB a token | ~105 GB/s | 23% of 460 |

No DSP packing: 28% leaves room; packed MX columns (288 DSP per 4-row group, `docs/mx_mxu.md` 5)
would halve the MXU to ~1,150 for a later 8-lane variant.

## 5. Expected performance, Gemma 3 1B int8, one sequence

**Roofline.** 7.85 M chunk-cycles today (docs/gemma3.md: 1.005 GB at 128 B) / 4 = 1.96 M cycles =
9.8 ms at 200 MHz = **102 tok/s**. Check: layers 26 x 209,664 chunks (q 1024x9 + k 256x9 + v 256x9
+ o 1152x8 + gate, up 6912x9 each + down 1152x54) = 5.45 M, LM head 262,144 x 9 = 2.36 M.

**Losses re-estimated from the 133 MHz trace** (10.28 M cycles; the buckets overlap by ~2%):

| loss today | cycles | kind | candidate | why |
|---|---:|---|---:|---|
| stream (port B busy) | 7.85 M | per byte | 1.96 M | 4 lanes at 1 chunk/cycle each |
| MXU not consuming, 18.2% | 1.87 M | per row + memory rate | 0.10-0.57 M | `pf_block` (`otpu_mxu.sv:398`) counts chunks present but `row_in` false: on the YPCB mostly the DDR3 below the port (port A and QST share its 2 channels) with the consumer waiting a whole row; on HBM each group has 4.5x the lane's demand, so the FIFOs stay full: that part -> 0. The per-row part (721K rows: 2.6 cycles a row if all of it) becomes per row-group with no transition bubble: 0.1 M; the pessimistic case keeps 1.87 M / 4 = 0.47 M |
| MXU computing, not streaming, 3.9% | 0.40 M | per MM (435 cycles x 922) | 0.15 M | the FIFO backlog at a command's end drains 4x faster; the start / drain latency (LROW 25 + drain) stays |
| chunk FIFO starved, 0.6% | 0.06 M | per MM start | 0.03 M | HBM first-chunk latency 60-100 cycles where the 2-command lookahead does not cover |
| TMEM arbitration (drain), 2.1% | 0.22 M | per result | 0.08 M | 4 results a cycle in one contiguous write, 16 banks |
| VPU + quantizer on the chain | ~0.29 M exposed of 0.58 M | per instruction | 0.15 M | 16 lanes / CL 4 halve the chain; the 1024-chunk FIFO hides ~1K cycles an MM boundary |
| host path | 0.3 ms | per token | 0.3 ms | docs/autodecode.md 38-45 (0.303 ms steady) |

Totals: **2.47 M cycles = 12.4 ms + 0.3 ms = 79 tok/s** (expected); 2.84 M = 14.2 + 0.3 ms =
**69 tok/s** if the "not consuming" loss only divides by 4. Headline: 69-79 tok/s, 5-6x today.

**Per-token budget** (expected case): layers 5.45 M / 4 = 1.36 M + 0.40 M of losses = 1.77 M
cycles (8.8 ms, 72%); LM head 2.36 M / 4 = 0.59 M + 0.11 M = 0.70 M (3.5 ms, 28%). Today 7.2 M /
3.1 M. The LM head's share is its byte share (30%); its losses shrink most since its 262,144 rows
of KB 9 are the short-row case.

**Prefill** (qualitative): M = 16 tiles replayed from the lane FIFOs, MXU-bound at 2,048 MAC/cycle:
1.0 G MACs a token / 2,048 = 488K cycles = 2.4 ms; attention (q.K^T, P.V on the MXU, softmax on 16
lanes with EXP2 at 4 cols/cycle: ~15K cycles a token at a 1024 prompt) and the per-tile chain add
~30%: **~300 tok/s** for a 1024-token prompt, 7-8x the card class (docs/scoreboard.md 4: 41.9 on
Qwen3-0.6B). Sane: the stream at 512 B/cycle is 4x the compute's need, so prefill never waits on
HBM.

## 6. Timing and risks at 200 MHz

- The core at 200 MHz is the user's F2 synthesis of today's design; each lane is that MXU. New
  200 MHz paths: the 4-lane deskew into one result-FIFO entry (register per lane), the ACT read
  broadcast (1024 b x 4 fan-out: one register stage a lane, the issuer requests blocks a cycle
  earlier), the 4-word drain write.
- TMEM: 16 banks, 128K words, URAM pumped at 400 MHz: +0.180 ns on the stand-in (docs/tmem_pump.md
  726); the arbiter's grant is the slice's deepest path (`otpu_tmem.sv:15`) and 16 lanes add a
  rotator level: budget a pipeline stage (TMEM_RQ) and the 2-cycle read if it misses.
- VPU 16 lanes: did not route on the xc7k480t at 212K LUT (docs/wide_dram.md 2); the VU47P has 6x
  the LUTs and the VPU is in its own SLR (below). Composite lanes at CL 4: 2x the long lanes.
- HBM adapter: the 450 MHz domain is the IP's; 32 async FIFOs; the 200 MHz chunk network (8:2 a
  group, then 4 x 4 of 1024 b) is 2-3 pipeline stages; the FIFO slack absorbs conflicts. Risk:
  QST's byte writes (V^T: one byte a beat) need WSTRB writes; with the HBM IP's ECC on they are
  read-modify-writes inside the IP (slow): run ECC off, or keep the native adapter's fill-read
  gather (`otpu_native_dram.sv:40-50`).
- SLRs (3): HBM attaches to SLR0, so the 4 lanes (2,208 of SLR0's ~3,008 DSPs, ~150K LUT) and the
  adapter sit in SLR0; TMEM, VPU, quantizer, sequencer in SLR1; the shell in SLR2. Two crossings:
  the drain's TMEM writes and the ACT RAM's quantizer writes, both registered. The ACT RAM (68
  RAMB36) stays with the lanes.
- Clocks: core 200, core_clk2x 400 (even divider, URAM <= 500 MHz at -2), HBM AXI 450, shell PCIe
  250. 450 / 200 is not integral: async FIFOs, not a pumped crossing.
- Schedule risk: the F2 shell flow (AWS developer AMI, AFI build and registration, the shell's
  DMA in place of XDMA) is new to the tree; budget it as its own milestone.

## 7. Implementation plan

| # | milestone | effort | verification | reused |
|---|---|---|---|---|
| M0 | Lens: split today's `mb` bucket by cause (memory rate vs `row_in` vs credits vs transition bubble) on the existing `build/lens/gemma3-decode-rtl.otpuprof`; count row-transition bubbles | 1 wk | the sum reproduces 18.2%; decides section 5's 0.10 vs 0.47 M | docs/lens.md |
| M1 | ISA sim + scoreboard: C in `sim_uarch`, `ceil(N/C) x KB` timing, row-group bubble model; F2 preset S 1 / 200 MHz / C 4; rerun seq_vpu.md 5's composite-width study at HBM rate (fixes VPU_CL) | 1 wk | `lens record --isa` on Gemma 3 gives ~2.1 M cycles; tests/test_config_matrix.py at C 4 | tools/scoreboard.py, docs/seq_vpu.md |
| M2 | RTL MXU lanes: `otpu_mxu` parameter C, 4 issuers, broadcast ACT read, 4 chunk / scale FIFOs, 4-wide result FIFO, per-row-group credits, zero-bubble `row_in`, RMAX lane fold | 4-6 wk | token-exact tests at C 1 and C 4 bit-identical to the ISA sim (`OTPU_TARGET=f2`); the MXU cell testbench in xsim on unisim | `rtl/tech/xilinxus` DSP48E2 cells (BCU1525, tested) |
| M3 | VPU 16 / CL 4, quantizer 16 + XD 512, TMEM 128K x 16 banks in URAM, ACT_ROWS 16 | 2-3 wk | the same tests; OOC Vivado at 200 / 400 MHz | `bit LANES=16` knob, `vivado/tmem_uram.xdc` (BCU1525 TMEM_URAM) |
| M4 | `otpu_hbm_dram`: 32 AXI masters, PC hash, async FIFOs, group muxes, lane crossbar, ports A / SW / DMA | 4-6 wk | a behavioural 32-PC model (`otpu_dram` with LAT 80, per-PC queues) in Verilator; `test_rtl` token-exact; the FIFO slot-reservation assertions | `otpu_native_dram` queues and CHASH, `otpu_afifo` |
| M5 | `targets/f2.toml`, `rtl/boards/f2/` top, `boards/f2/` flow (HBM IP, clocks, the AWS shell CL wrapper) | 2-3 wk | `python3 -m opentpu.target show f2`; `tests/test_target.py` file lists; synthesis + place at 200 MHz | `boards/bcu1525` flow, `rtl/shell` (ctrl, trace, afifo), SLR floorplan notes (docs/bcu1525.md 10) |
| M6 | Bitstream / AFI: Vivado on the F2 developer AMI, timing closure, AFI registration | 2-4 wk | WNS >= 0 on all 4 domains; CFG_* registers read back | docs/bcu1525.md 6 |
| M7 | Bring-up on an f2 instance: host `board.py` for the shell DMA and the PC hash, HBM memory test (as `otpu-ddr4`), latency measurement, first tokens, `compare_hf --model gemma3`, `lens record --axi` | 3-4 wk | the 4 prompts of docs/gemma3.md match the ISA sim token for token; Lens per section 8 | `opentpu/host/ddr4.py` test pattern, docs/bcu1525.md 11-12 |

About 5-6 months of one engineer; M2 / M3 and M4 / M5 can run in parallel pairs.

## 8. What to verify in Lens once built

- Port B streaming (sum over 4 lanes / 4) >= 80% of cycles (today 76.4%); a token 2.5-2.8 M
  cycles at 200 MHz; MXU MACs/cycle at M = 1 from 99 to >= 400 of 2,048.
- "MXU not consuming" from 18.2% to < 5% (expected case) with `bs` (port B waiting on memory) ~ 0:
  if `mb` stays near 18%, the loss is per row-group and M0's split was wrong about the memory.
- Chunk FIFO starved < 1% and the FIFO level near DEPTH in steady state (the 4.5x headroom).
- "MXU computing, not streaming" from 3.9% to ~5% of a 4x shorter token (0.15 M cycles).
- TMEM arbitration (drain) < 1.5%; VPU in flight ~12% with < 6% exposed; quantizer ~1.5%.
- LM head ~0.70 M cycles (28% of the token), layers ~1.77 M.

## 9. Every other unit, resized

| unit | today (card config) | this candidate | why |
|---|---|---|---|
| ACT RAM (`otpu_actram`) | 16 rows x 128 blocks x 128 B; read MCOLS x D = 512 B/cycle; write 8 B/cycle; 68 RAMB36 | same store; one 512 B read a cycle broadcast to C = 4 lanes (+1 register a lane); write 16 B/cycle; 68 RAMB36 | lock-step row interleave: every lane reads the same block of the same rows (section 2) |
| MXU epilogue lanes E | MXU_E 4: <= 4 results a cycle, 1 used at M = 1 | E 4 shared by the lanes: 4 results a cycle at M = 1 (one row-group); 16 over 4 cycles at M = 4 (needs KB >= 4) | 4 results a cycle is exactly today's E |
| result FIFO / credits | RF 32 rows (`otpu_mxu.sv:117`), entry MCOLS x 48 b + scales; `rows_live < RF` | RF 32 row-groups, entry 4x wider (distributed RAM); credits per row-group; `row_in` of the next group precomputed | credits cover LROW 25 + drain, far under 32 x KB cycles; no transition bubble |
| chunk FIFO | 1024 x 128 B, 32 RAMB36; DDR3 latency 100-200 cycles | 4 x 1024 x 128 B, 128 RAMB36, 2 write ports a lane (even / odd slots), slots reserved at request | holds 60-100 cycles x 1 chunk/cycle/lane 10x over; KB 54 rows and M = 16 replay fit |
| port A (scales) | 1 x 32 B line/cycle, TD 128 tags, line reuse | 4 lane readers, each 32 B lines with reuse, TD 128 a lane; ~0.25 requests/cycle/lane (int8x scale rows 16 B at KB 9: 2 rows a line) | 1.1% of the bytes; through the PC network's narrow path |
| port SW (QST / KV) | byte writes gathered a beat, fill reads for partial beats | the same gather; WSTRB on AXI, ECC off (else the IP's RMW); KV append 2 x 256 B + V^T 256 scattered bytes a layer | tiny bandwidth, latency-sensitive only through `wr_idle` |
| quantizer | 2 passes x 8 elements; XD row buffer | 2 passes x 16; XD 512 chunks; +~24 DSP | QACT ~0.07 M cycles a token at 8 lanes -> 0.035 M; keeps pace with the 16-lane VPU |
| TMEM | 64K words, 8 banks, IMPL 3 pumped 2x, NSB 2, 64 RAMB36; RPB 0, WPB 1 | 128K words, 16 banks, IMPL 3 pumped 2x (400 MHz) in 32 URAM288; RPB 0, WPB 1; the drain's M = 1 write on a rotator port; +1 arbiter stage if needed | prefill tiles of 16 rows; banks = LANES; URAM frees 128 RAMB36 |
| DMA LD / ST | 1 chunk/cycle on port B, DEPTH-chunk buffer, SRUN write runs | the same on lane 0's path (128 B/cycle); logits ST 1 MiB/token = 8,192 cycles (0.3%); GATHER of the 1152 B embedding row | not worth widening: 0.3% of the token |
| IMEM / sequencer | 32K words = 4,096 instructions (the 5,350 are dynamic: hardware loops); WIN 16; TAW 16 | IMEM unchanged; WIN 16; TAW 17 (`otpu_seq.sv:129` follows TMEM_WORDS); STREAM buffers below word 65536; SEQ_WC on | the program is the card's |
| collective | BAR-only (`COLL_GATHER 0`, S = 1) | the same | one slice |
| PCIe / host | XDMA Gen2 x8; 64 B/token in the decode loop; 0.30 ms/token host path | the F2 shell's PCIe DMA; 64 B/token; 0.3 ms = 2.4% of 12.4 ms (move sampling on-device later) | docs/autodecode.md |
| clock domains | core 133.33, core_clk2x 266.67, LiteDRAM, XDMA 250 | core 200, core_clk2x 400 (URAM), HBM AXI 450 (32 async FIFOs), shell 250 | 450 / 200 non-integral: async, not pumped |
| chunk network (new) | 2 channels x 64 B beats, CHASH | 32 PCs, PC hash, 8:2 a group, 4 x 4 lane crossbar of 1024 b, 2-3 stages | keeps the ISA and image layout unchanged |

Fallback if the chunk network does not close at 200 MHz: a lane-striped image (row n in PC group
n mod 4 at `sa + (n mod 4) x 2^30 + (n div 4) x rs`), one group per lane and no crossbar; it
costs an MM / QST / GATHER address rule, the image builder and the ISA simulator (an arch change).
