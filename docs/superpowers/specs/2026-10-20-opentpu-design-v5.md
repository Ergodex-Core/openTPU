# openTPU — architecture design v5 (the full-HBM F2 variant)

Date: 2026-10-20. Status: design + RTL on branch `full-hbm` (worktree openTPU-fullhbm, from
wide-port-alexshell 71919ae2), verified in Verilator only; no synthesis, no board. Where it
conflicts with `2026-10-18-opentpu-design-v4.md`, v5 wins for the `f2-hbm:v5*` targets; every
other target is v4 as built. Every number is labelled: **measured** = a Verilator run of this
tree's RTL, bit-exact with the ISA simulator; **lpn-estimate** = `tools/wideport_lpn.py`;
**estimated** = arithmetic from the v4 synthesis figures (`docs/wide_port.md` "Synthesis",
"Placement, round 2") or from the HBM facts in `docs/f2_hbm.md`. The measured results are in
`docs/full_hbm.md` and summarized in section 9.

## 0. The arithmetic that sizes everything

| quantity | value | source |
|---|---|---|
| HBM | 2 stacks, 32 pseudo-channels x 512 MiB, each an AXI3 256-bit port at 450 MHz = 14.4 GB/s; 460.8 GB/s raw | docs/f2_hbm.md |
| measured by AWS (cl_mem_perf) | 433.7 GB/s write / 426.3 GB/s read, 512 B bursts (AXLEN 0xF), 64 outstanding per channel, 172 / 280 ns latency; 13.3 GB/s read per channel | the brief |
| a channel's native beat | 64 B per core cycle (two 256-bit AXI beats): 12.8 GB/s at 200 MHz = 96% of the channel's measured 13.3, 89% of its raw 14.4; 9.6 GB/s at 150 MHz (67%) | docs/f2_hbm.md "Bandwidth" |
| the whole HBM at 200 MHz | 32 channels x 64 B = **2,048 B a cycle = 409.6 GB/s = 16 chunks of 128 B a cycle**, 96% of the measured read rate | arithmetic |
| the v4 slice's port B | PORTB 4: 512 B a cycle = 4 chunks, 102.4 GB/s on NCH 8 at a beat a cycle a channel; measured 3.34 beats a cycle a channel of 4 on the native model (64 reads in flight over a 74-cycle round trip: 0.865 by Little's law) | docs/wide_port.md "Lens profile", "LPN estimate" |
| int8 decode at M = 1 | one MAC per weight byte: 2,048 B a cycle needs **2,048 int8 MACs a cycle**; v4 has 512 (4 row lanes x D 128 at MCOLS 4; the MCOLS 8 build leaves 4 columns idle in int8 RL 4) | docs/row_lanes.md "The MXU utilization bound" |
| Gemma-3-1B int8 token | 1,033,510,176 bytes (int8 head), 7,830,202 chunks, 890 MMs (838 RL) | docs/wide_port.md "The whole-model tally" |
| the port-bound token at 16 chunks a cycle | 7,830,202 / 16 = **489,388 cycles = 2.45 ms = 409 tok/s** at 200 MHz (the HBM's 426 GB/s: 2.42 ms, 413 tok/s) | arithmetic |
| the v4 point to beat | 2,394,257 cycles = 83.5 tok/s int8 whole model (the 4-layer + head receipt 949,550); 58.5 tok/s at the 150 MHz build in flight | docs/wide_port.md, the brief |
| a realistic v5 target | >= 200 tok/s int8 on the RTL: a token under 1.0 M cycles | the brief |

The two constraints the arithmetic does not show, and that decide the shape (section 2):

- **An SLR holds one v4-class slice.** CL LUTs per SLR: ~272K in SLR0 (the HBM fabric lives there
  too), ~230K in SLR1 (the shell takes part of it), ~435K in SLR2; the wide-port slice (PORTB 4,
  MCOLS 8, RL 8, NCH 8) measured 226K LUT / 416 RAMB36 / 1,275 DSP after synthesis
  (docs/wide_port.md "Synthesis", "Placement, round 2": u_slice 174,618, the adapter 46,717).
- **An SLR boundary carries ~23K SLL wires** (the brief's figure; the VU47P's exact count is not
  in any document in this tree and must be read from `report_utilization -slr` of a placed
  checkpoint: section 6 states every crossing against this assumption). A channel's native port
  is 604 bits down (take, we, beat, the 512-bit write beat and its mask) and 532 up (readies,
  the 512-bit read beat, the write count, err): 1,136 bits; 32 channels into SLR2 would be
  36K per boundary before the control path. So the read data of a channel can cross at most the
  boundaries between SLR0 and the slice that owns it, the write data must be shared, and the
  channels of a slice must be as few as its bandwidth needs.

## 1. v5 in one paragraph

v5 keeps the v4 slice (D 128, MCOLS 4, PORTB 4, four int8 row lanes, ALINE 64, LANES 8, the
pumped TMEM, NCH 8) and multiplies it: **S = 4 slices, one per SLR with two in SLR2, each on 8
HBM pseudo-channels, 32 channels in all, tensor-parallel over the model's rows with the
collective unit's GATHER**; every HBM port gets **its own 450 MHz engine** (`otpu_hbm_ch`, in
cl_mem_perf's style: a command FIFO popped into a registered AR/AW generator that coalesces
consecutive beats into bursts of up to 512 B, the read data into a BRAM asynchronous FIFO to
the core clock, write data and the B count through small asynchronous FIFOs, nothing else at
450 MHz) in place of the per-channel SmartConnect chain, with 64 bursts outstanding and the
adapter's reads in flight raised from 64 to 128 beats a channel. The host keeps AWS's
cl_axi_sc_1x1 chain, on its own port at S = 3 (24 channels, the no-arbiter fallback) or
through a registered 2 x 1 AXI3 mux in front of port 31 at S = 4. The compiler shards Gemma 3
over the slices as it shards Qwen3: query heads, K / V projections, wo, gate / up, down and the
LM head by rows, the K / V cache replicated, six all-gathers a layer. Bandwidth 409.6 GB/s (16
chunks a cycle), 2,048 int8 MACs a cycle, 4 x 8 = 32 fp32 VPU lanes, 4 x 64K words of TMEM.

## 2. Alternatives, with numbers

| alternative | bytes a cycle | MACs a cycle | fit | why not |
|---|---|---|---|---|
| **A. one super-wide slice**: PORTB 16, 16 int8 row lanes, MCOLS 16, NCH 32, in SLR2 | 2,048 | 2,048 | LUT estimated ~460K (MXU ~200K at 16 columns and 16 streams: the hops carry NS x 8 = 128 bits x D positions x 15 columns = 245K FF, the PE stream select 8:1 ~65K LUT; the rest of the slice ~110K; an NCH 32 / PORTB 16 adapter ~150K with a 16-way merge) against SLR2's 435K; SLL 32 channels x 532 up = 17K + writes on both boundaries | does not fit one SLR, and the ISA has no L = 16 (rs bits 3:2 hold log2 L <= 3; QACT DUP16 and ALINE 128 for the 64-byte Sw header would be new); the MXU's hop structure (every stream through every column) scales as NS x MCOLS |
| **B. S = 2 slices**, each PORTB 8, 8 int8 lanes, MCOLS 8, NCH 16 | 2,048 | 2,048 | a slice estimated ~278K (the v4 MXU 97K + ~3K for 8-chunk entries, the rest 78K, an NCH 16 adapter 79K + ~20K for the 8-way merge): SLR2 61%, but SLR1 (230K) cannot hold the second; its adapter in SLR0 with the core in SLR1 needs the adapter-to-slice port B (a same-cycle 4,096-bit handshake) redesigned to cross an SLR, and puts 19.1K bits on the SLR0-SLR1 boundary (83%) | placement; the bound is the same as D's (398 tok/s) |
| **C. S = 3 x NCH 8**, PORTB 4 / RL 4 / MCOLS 4 (this RTL at S = 3) | 1,536 | 1,536 | fits (section 6); the host on port 24, no 2 x 1 | 72% of the HBM; the attention's 4 query heads do not split 3 ways (wq is replicated: +31 MB a token a slice, 9%); **kept as the no-arbiter fallback variant `f2-hbm:v5s3`** |
| **D. S = 4 x NCH 8**, PORTB 4 / RL 4 / MCOLS 4 (chosen, `f2-hbm:v5`) | 2,048 | 2,048 | fits (section 6): SLR0 ~79%, SLR1 ~72%, SLR2 ~76% LUT; SLL 72% / 53% | the one new 450 MHz arbiter is the host's 2 x 1 on port 31 (section 7) |
| PORTB 8 with 8 lanes a slice (two slices) | = B | | | = B |
| PORTB 16 with 16 lanes | = A | | | = A |
| MCOLS 8 slices (int4 L 4 at 4 chunks a cycle) | | | the p4rl8 slice is 226K: two of them exceed SLR2 (452K), one exceeds SLR1 | int4 is secondary (the brief); int4 runs L 2 at MCOLS 4: 2 chunks a cycle, the same token time as int8 |
| VPU ULANES / LANES 16 a slice | | | +~46K LUT a slice (the VPU 36K -> ~70K, the TMEM's 8 -> 16 banks): SLR0 and SLR1 over 90% | the per-slice VPU work at S = 4 is ~1/3 of v4's token (gelu and the attention sharded, the norms replicated): taken only if Lens shows it exposed (section 9) |
| TMEM port count | | | the pumped TMEM gives 16 words a cycle a slice; 4 slices = 64 | unchanged |
| a 150 MHz core | 32 x 64 B x 150 MHz = 307 GB/s | | the native beat caps a channel at 9.6 GB/s (67% of its port); a 96 B beat would be needed | = C's bandwidth at more timing margin; 200 MHz is the design clock, the 150 MHz v4 build is its fallback |
| int8 DSP packing (two weight rows a DSP48E2, the VLA branch's PACK) | | 2x per DSP | DSPs are 32% of the device at S = 4 (section 6): not the binding resource | changes the PE, the column end and the row sum; not needed |
| reads in flight: more credits vs bursts | | | RD 128 a channel is +4K LUT RAM a slice; the engine's 64 bursts x 8 beats = 512 beats are never the limit | both taken: RD 128 (Little's law: ~80 cycles round trip at a beat a cycle), bursts for the HBM's efficiency |

## 3. Data path (v5)

```
 SLR0 (272K LUT)                      SLR1 (230K)                     SLR2 (435K)
 32 x otpu_hbm_ch (450 MHz engines)   slice 1 + adapter (NCH 8)       slice 2 + adapter (NCH 8)
 host: pcis gate -> cl_axi_sc_1x1     collective hub, ctrl, trace     slice 3 + adapter (NCH 8)
 slice 0 + adapter (NCH 8)            pcis / ocl crossings (the shell)
 HBM ports 0..31 below SLR0
   |<-- chains of 3 otpu_xr stages, one per SLR, on every net between SLRs -->|
```

Per slice (unchanged from v4 except where marked):

```
 SEQ -> WIN 16 -> DMA | MXU (MM plain / PAIR / QUAD / RL 4 / walk; port B 4 chunks a cycle,
                       FIFO_DEPTH 256 entries of 4 chunks [v5: 1024 -> 256], 4 columns x D 128
                       DSP48E2, E 4 epilogue lanes) | QUANT (QACT DUP8 / QST) | VPU 8 lanes |
                       COLL (GATHER over S slices through the hub's registered links)
 TMEM 64K words pumped 2x (64 RAMB36); ACT RAM ACT_ROWS 8 [v5: 16 -> 8]; IMEM 32K; TRACE 4096 [v5: 16384 -> 4096]
 otpu_native_dram NCH 8, PORTB 4, ALINE 64, CHASH, RD 128 [v5: 64 -> 128], APR 16
   port B / A / SW queues per channel -> the native ports -> [write bus tx (one beat a cycle a slice)]
   -> otpu_xr chains -> SLR0: otpu_hbm_ch x 8
```

The per-channel engine `otpu_hbm_ch` (rtl/boards/f2/otpu_hbm_ch.sv), core clock side in SLR0
behind the chain, 450 MHz side next to the port:

- **Native port in** (the adapter's contract, docs/f2_hbm.md "The native port"): a command
  stream {we, beat[24:0]} with ready; write beats {data 512, mask 64} with ready, in command
  order, before or after their command; read beats back in command order with no backpressure;
  `n_wdone` counting completed writes; `n_err` sticky.
- **Core-clock front**: the `cont` bit of every read command (the beat follows the previous
  command's beat and is not the first of an aligned 8-beat group: `beat == last + 1 && beat[2:0]
  != 0 && !we && !last_we`), computed here from a 25-bit register and an incrementer at 200 MHz,
  so the 450 MHz generator never compares addresses; the command FIFO's ready keeps XL = 6 slots
  below full for the chain's late take (as u_naxi's queues).
- **Asynchronous FIFOs** (`otpu_hbm_afifo`: gray-coded pointers, two-flop synchronizers, a
  simple dual-port RAM read in the read clock; BRAM for the read data, distributed RAM for the
  rest; `xpm_fifo_async` is the production drop-in, selected by `OTPU_XPM`): commands 27 bits x
  32 (core -> hclk), write beats 576 bits x 16 (core -> hclk), read beats 512 bits x 256 (hclk ->
  core; the adapter's reservation RD + AD + WQD = 128 + 64 + 32 = 224 <= 256 keeps it from ever
  filling, and the engine checks a beat credit anyway), B tokens 1 bit x 32 (hclk -> core, each
  one `n_wdone` increment).
- **450 MHz generator**, registered state only: an open read burst {addr, len <= 8}; the
  command FIFO's head extends it when `cont` and `len < 8`, else closes it into the AR register
  (ARADDR = port base + beat x 64, ARLEN = 2 len - 1, ARSIZE 32 B, INCR, ID 0) and starts a new
  one; an idle timer (2 hclk cycles) closes a burst the stream stops behind; bursts never cross
  a 512 B group, so never 4 KiB. Writes: AW (one native beat = AWLEN 1) with W's two 256-bit
  beats and strobes from the mask, in the order of the commands. Direction barriers as
  u_naxi's: a read burst issues only when no write lacks its B, a write only when no read beat
  is outstanding (registered zero tests). Outstanding: at most 64 read bursts (AWS's measured
  point) and 256 beats. R: two 256-bit beats assembled into one 512-bit beat pushed into the
  read FIFO; RRESP / BRESP checked into `n_err`. Every AXI output is a flip-flop; every AXI
  input is taken into a flip-flop before use (the HDK's cl_axi3_256b_reg_slice sits between the
  engine and the port as in AWS's examples).
- **The host** (unchanged: `otpu_f2_pcis_gate` -> axi_register_slice -> cl_axi_sc_1x1 at 250 /
  450 MHz -> port NPORT) at S = 3; at S = 4 `otpu_hbm_mux2` (rtl/boards/f2/otpu_hbm_mux2.sv),
  a registered 450 MHz AXI3 2 x 1 in front of port 31: AW / AR arbitrated (the channel first, the
  host after a bounded wait), W in AW order through a 4-entry order FIFO, B / R routed by ID bit
  5 (the engine's 0, the host's 1); the global address switch lets the host reach all 16 GiB
  through that port as it does through its own.
- **The chains** (`otpu_f2_hbm_eng.sv`, replacing otpu_f2_hbm_axi's u_naxi + SmartConnect):
  per channel down {take, we, beat} 27 bits, up {cready, wcredit, rvalid, rdata 512, wdone-pulse,
  err} 517 bits; per slice one **write bus** down {take, channel 3, data 512, mask 64} 580 bits:
  the slice-side arbiter takes one channel's write beat a cycle (round robin among the channels
  holding a credit), SLR0 demuxes it into the channel's write FIFO; the credit returns on the
  channel's up word. Writes are rare in decode (the KV append's rows and QST's gathered beats), so
  one beat a cycle a slice costs nothing measurable and saves 8 x 576 - 580 = 4,028 SLL bits a
  slice on every boundary the slice's traffic crosses.

The slice's memory ports are v4's; what changed for the new width is the count of slices on the
HBM, not the port: at PORTB 4 the port's 512 B a cycle equals 8 channels at a beat a cycle, so
the per-slice path is balanced as it was, and the engine's job is to deliver that beat a cycle
(v4's native model delivered 0.865: the channel's 64 beats in flight over a 74-cycle round
trip; v5's 128 over ~80).

## 4. The compiler: Gemma 3 on S slices (opentpu/llm/gemma4.py)

v4's `Spec.check` refused `cfg.S != 1` ("one slice (a single KV head)"). v5 takes qwen3's
sharding rules (opentpu/llm/qwen3.py, kernels/mlp.py `swiglu_down`, `_lm_head`: all already
written for `ol.num_programs()` slices) into the Gemma 3 kernels:

- **Attention**: slice s owns query heads s n_q / S .. (s + 1) n_q / S - 1 (S must divide n_q;
  else, S = 3, every slice owns all of them and wq is replicated); K and V projections are
  sharded by rows (n_kv d / S each) and **all-gathered** (two GATHERs of n_kv d words), so every
  slice norms, RoPEs and appends the same full K / V rows into its own copy of the cache (the
  cache is replicated: 4 x the v4 KV bytes over the card, 15 MB a slice at cap 256); the heads'
  outputs are all-gathered ([1, n_q d], slice-major head order), wo is sharded by output rows
  (h_loc = H / S) over the full K, its result all-gathered to [1, H] before the post-attention
  norm, which every slice computes (replicated VPU work: the norms are 1152 wide).
- **MLP**: gate / up by rows (f_loc = F / S = 1,728), the activation `gelu(g) * u` of each chunk
  all-gathered ([1, S C]) and the down projection accumulated over the S slices' chunks of its
  full K with the slice's h_loc rows: exactly `swiglu_down`'s existing S > 1 path; the output
  all-gathered before the post-FFN norm.
- **LM head**: v_loc = 262,144 / S rows a slice (`_lm_head` already loops over `m.v_loc` at
  column `sid v_loc`): each slice stores its logits shard; the embedding gather keeps the full
  standard-layout head copy on every slice (302 MB int8 a slice; the row-lane copy is the shard).
- **Programs**: the slices run the **same program** (no `program_id()` term outside `_lm_head`'s
  store column); each slice's DRAM holds its rows at the same addresses. The image build makes S
  images with per-slice row selections of every matrix.
- **Numerics**: every dot product is computed whole on one slice in the same K order as at S = 1,
  the norms over the gathered full vectors: the S = 4 logits are **bit-identical to S = 1's**
  on the ISA simulator (tests/test_gemma3_slices.py), which is the correctness gate.
- **GATHER cost** (otpu_coll: LANES = 8 words a cycle, every slice's segment in turn, then all
  released together): per layer k 256 + v 256 + o 1,024 + y 1,152 + act 6,912 + y 1,152 =
  10,752 words = 1,344 cycles + ~6 x 30 of barrier and pipeline = ~1,500; 26 layers ~39K cycles a
  token (6% of a 650K token). The act gathers are per MLP chunk and overlap the next chunk's
  gate / up MMs; the others sit on the critical path.

## 5. Sizing: the chosen point

| item | S = 4 (`f2-hbm:v5`) | S = 3 (`f2-hbm:v5s3`) |
|---|---|---|
| channels | 32 (host via the 2 x 1 on port 31) | 24 (host on port 24) |
| bytes a cycle at 200 MHz | 2,048 (409.6 GB/s, 96% of AWS's measured read) | 1,536 (307 GB/s) |
| int8 MACs a cycle | 2,048 (4 slices x 4 lanes x 128) | 1,536 |
| the port-bound int8 token | 489,388 cycles (409 tok/s) | 652,517 (306 tok/s) |
| weight bytes a slice a token | 258 MB + ~1 MB replicated | 344 MB + 31 MB (wq) |
| DSP48E2 | ~2,900 of 9,024 (727 a slice: the f2-hbm base) | ~2,200 |
| VPU | 4 x 8 fp32 lanes | 3 x 8 |
| TMEM | 4 x 64K words, pumped | 3 x 64K |
| gathers a token | ~39K cycles | ~39K |

## 6. Utilization and placement plan (estimated from the measured v4 figures; the final table is in docs/full_hbm.md)

Per slice (PORTB 4 / RL 4 / MCOLS 4 / NCH 8 / ALINE 64), scaled from the synthesized f2-hbm base
(the MCOLS 4 / PORTB 1 slice 99K LUT, its NCH 8 adapter 39K; docs/f2_hbm.md "Synthesis") plus
the wide-port additions measured on p4rl8 (docs/wide_port.md "Resources and timing": the QUAD
datapath ~7K, the S1 wide select ~3K, the adapter's merge 4.5K, the row lanes ~2.5K, ALINE 64
~0.5K) and RD 128 (+4K LUT RAM):

| module | LUT | FF | RAMB36 | DSP | note |
|---|---|---|---|---|---|
| u_mxu (MCOLS 4, RL 4, PORTB 4) | ~62K | ~45K | ~45 | ~640 | the chunk FIFO 256 x 4,096 b = 29 RAMB36, the scale FIFOs 5, result FIFO in LUT RAM |
| u_vpu (8 lanes) | 36K | 20K | 2 | ~50 | measured 36,165 on p4rl8 |
| u_tmem (64K words, pumped) | 12K | 5K | 64 | 0 | measured 11,913 / 5,129 / 64 |
| u_quant, u_seq, u_dma, ACT RAM, IMEM, misc | 30K | 25K | 29 + 29 + 8 | 0 | measured 9.4K / 9.0K / 9.3K; ACT RAM at ACT_ROWS 8: 29; IMEM 29; trace 4096: 8 |
| otpu_native_dram (NCH 8, PORTB 4, RD 128) | 48K | 12K | 0 | 0 | 46,717 measured at RD 64, +~1.5K LUT RAM at 128 |
| **a slice** | **~190K** | **~107K** | **~181** | **~690** | |
| otpu_hbm_ch x 32 (SLR0) | ~40K | ~45K | 128 (the read FIFOs, 4 each) | 0 | ~1.2K LUT each: the generator ~400, the FIFOs' pointers and the write FIFO's distributed RAM ~600, the front ~200 |
| host chain (pcis gate, axi_register_slice, cl_axi_sc_1x1, the 450 MHz slice) + mux2 | ~7K | ~14K | 0 | 0 | docs/wide_port.md "Alex's shell at 150 MHz" IP sizes; mux2 ~0.5K |
| collective hub, ctrl, trace, OCL / PCIS crossings | ~6K | ~8K | 8 | 0 | |
| chain stages (otpu_xr) | 0 | ~2 x (16.5K + 12.3K) = 58K | 0 | 0 | one stage per SLR on each crossing bit |

Per SLR, S = 4:

| SLR | contents | LUT (of budget) | RAMB36 (of 672) | DSP (of 3,008) | flag |
|---|---|---|---|---|---|
| SLR0 | slice 0, 32 engines, the host chain, mux2 | ~237K of 272K (**87%**) | ~309 (46%) | ~690 (23%) | **over 80%**: the lever is the engines' write FIFOs in BRAM (-10K LUT) or RD 96 on slice 0; S = 3 drops 8 engines (-10K: 83%) |
| SLR1 | slice 1, the hub, ctrl, trace, the shell-side crossings | ~196K of 230K (**85%**) | ~189 (28%) | ~690 (23%) | **over 80%**: the SLR1 budget is the brief's 230K (the v4 brief said 313K); at 313K it is 63% |
| SLR2 | slices 2 and 3 | ~380K of 435K (87%) | ~362 (54%) | ~1,380 (46%) | **over 80%** |

(These are the pessimistic line: every slice at the p4rl8-derived 190K. The f2-hbm base slice +
adapter measured 138K at PORTB 1; the honest range for a v5 slice is 165-190K, i.e. SLR0 at
79-87%, SLR1 72-85%, SLR2 76-87%. The brief's rule flags anything over 80%: v5 is at the line
in every SLR, which is why the slice's BRAM and FIFO depths were cut and why MCOLS 8 and LANES
16 were rejected. Section 9 carries the number the first synthesis gives.)

SLR crossings (bits; 3-stage chains, one stage per SLR; the collective hub in SLR1 so that only
slice 0's and slices 2-3's links cross one boundary each):

| boundary | memory (per channel 27 down + 517 up = 544; per slice 8 x 544 + 580 = 4,932) | collective links (~1.1K a slice) | control, reset, ready | total | of 23K |
|---|---|---|---|---|---|
| SLR0-SLR1 | slices 1, 2, 3: 14.8K | slice 0: 1.1K | 0.3K | **16.2K** | **70%** |
| SLR1-SLR2 | slices 2, 3: 9.9K | slices 2, 3: 2.2K | 0.1K | **12.2K** | **53%** |

Without the shared write bus the SLR0-SLR1 count is 27.3K (119%): the bus is not optional.
S = 3: 11.2K (49%) and 6.0K (26%).

Floorplan (boards/f2/build/constraints/cl_pnr_user_hbm.tcl, extended for an engine target):
pb_otpu_hbm (SLR0: AWS's outline) holds the 32 engines, the host chain, the mux and slice 0
(`u_board/g_slice[0]`, its adapter); pb_otpu_mid (SLR1) holds slice 1, the hub (`u_board/u_coll`,
the links' stages), ctrl and trace; pb_otpu_board (SLR2) holds slices 2 and 3; pb_otpu_pcis as
before. Every chain stage k in SLR k. The pumped TMEM boxes (pb_otpu_tmem) become one per slice.

## 7. Risks

1. **Timing at 450 MHz.** The engine's generator is 2-3 LUT levels (the `cont` bit is computed at
   200 MHz; the open burst's length compare is 4 bits; the outstanding counters' zero tests are
   registered), the asynchronous FIFOs are gray pointers, and every AXI signal is a flip-flop:
   cl_mem_perf's structure, which closed in AWS's example. The one arbiter at 450 MHz is the
   host's 2 x 1 on port 31 (S = 4 only): its W path is a 256-bit 2:1 mux into a register, the
   kind of cone that failed in rounds 1-8 (`m_wdata` through the two-master select at -2.2 ns,
   docs/wide_port.md "Placement"). Mitigation: S = 3 needs none; at S = 4 the mux's W register
   is fed from the two masters' own output registers one hop away, and the mux lives in port 31's
   clock-region column with the HDK's slice before the port.
2. **SLR crossings.** 70% / 53% of an assumed 23K; the shared write bus and the hub in SLR1 are
   what keep it there. Verify the SLL count on a placed checkpoint before trusting the margin.
3. **TMEM arbitration.** The GATHER's broadcast writes into all four TMEMs contend with each
   slice's MXU drain (v4 lost ~8% of the token there: "MXU drain lost 197,741 cycles to TMEM
   bank arbitration"); the receipt's Lens profile carries the per-unit loss.
4. **The LM head.** 65,536 rows a slice = 147K of the token's ~600K cycles (still ~25%); the
   argmax over the S logit shards is the host's (perf_qwen's step stores logits) or a VOP ARGMAX
   per slice plus a 4-word gather in the resident loop (not built here).
5. **HBM efficiency at 64 B granularity.** AWS's 426 GB/s was with 512 B bursts. The engine
   forms bursts only from what is queued; in v4's receipt the channels saw a beat at a time
   because the adapter's 64 credits paced them. With RD 128 the queue backs up only when the HBM
   is slower than the demand: the receipt's `HBMENG` burst histogram says what the HBM would see;
   the Verilator model charges a beat a core cycle a channel whatever the burst length (the
   port's 2 hclk a native beat, 1.125 a core cycle, minus nothing for row misses).
6. **The GATHER barrier.** Every gather waits for the slowest slice; the per-slice channel
   jitter adds up over 156 gathers a token. The receipt measures it (the collective's wait
   cycles per slice in the Lens profile).
7. **The compiler's new ground.** Gemma 3 at S > 1 is new code; the gate is bit-identity with
   S = 1 on the ISA simulator (tiny model and the real 1B), then RTL = ISA simulator bit for bit.
8. **Resources at the line.** Section 6: every SLR is at 76-87% LUT on the pessimistic estimate.
9. **Verilator cost.** Four slices on 32 channels: ~4x the logic of a v4 run for ~1/4 the cycles;
   the whole-model receipt is ~15-25 minutes with `--threads 4` (measured in docs/full_hbm.md).
10. **The ISA's 4 GiB a slice** (docs/f2_hbm.md "Past 4 GiB a slice") is untouched: a slice's
    image at S = 4 is ~600 MB.

## 8. Bring-up order and the gates

1. Spec (this document). 2. The compiler at S > 1 with the bit-identity test (tiny Gemma 3 at
S = 2 / 4; the real 1B's logits at S = 4 against S = 1). 3. The engine RTL with its two-clock
testbench (tb_hbm_eng: the native contract through the chains, bursts, barriers, the mux, 32
ports) and the core-clock receipt model (otpu_hbmeng_mem). 4. The existing RTL gates on the
new tree (test_native_nch, test_f2_rtl, test_wide_port_rtl, test_rtl selections, test_row_lanes).
5. The S = 4 and S = 3 RTL runs: the 4-layer + head receipt, then the whole model, with
`--check` (RTL = ISA simulator). 6. Lens and LPN on every quoted configuration. 7. The
utilization table, docs/full_hbm.md, the target variants. Synthesis / P&R are a later,
separately approved step (the brief).

## 9. Results (filled from docs/full_hbm.md at the end of the work)

See docs/full_hbm.md "The numbers": the configurations table (RTL cycles / tok/s / LPN / Lens
roofline %), the utilization table and what blocks the FPGA build.
