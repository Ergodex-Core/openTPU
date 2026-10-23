# openTPU v4 candidate P4: four tensor-parallel slices on the F2 (VU47P + HBM2e)

Status: a design proposal (2026-10-18), not implemented. Every number is derived in place or
cites its source; nothing here was simulated or built. Target: AWS EC2 F2, xcvu47p, 16 GB HBM2e
(32 pseudo-channels x 256 bit, ~460 GB/s), 9,024 DSP48E2, 2,016 RAMB36, 960 URAM288, core 200 MHz.

## 1. Summary

- Keep today's slice exactly (D 128, MCOLS 4, E 4, LANES 8, one chunk a cycle, 716 DSP48E2:
  `docs/bcu1525.md` 7) and instantiate S = 4 of them, each on its own 8 HBM pseudo-channels
  (4 GiB = the ISA's 32-bit address space), joined by the existing collective (`rtl/top/otpu_coll.sv`,
  GATHER / BAR) through the BCU1525's cross-SLR hub (`otpu_coll_hub` / `otpu_coll_port`, `docs/bcu1525.md` 10).
- Gemma 3 1B is split over the 4 slices with the scheme Qwen3 uses at S = 2 (`opentpu/llm/qwen3.py`
  1322-1385, dataflow's split (a): every MM by output rows, inputs all-gathered), plus one Gemma-only
  gather of the single KV head's k / v and a replicated KV cache.
- Decode bound at 200 MHz: 1.005 GB / 4 / (128 B x 200 MHz) = 9.81 ms = **102 tok/s**; realistic
  **72 tok/s as the RTL is, 78 with a one-beat-a-cycle collective hub, ~89 with the MXU row-credit
  fix** (section 5), against 13.0 tok/s measured today (`docs/gemma3.md` Lens, 133.33 MHz).
- Risk: low on the datapath (each slice is the 200 MHz design the user synthesized), medium on the
  shell (HBM bridge, 4 slices across SLRs) and the model code (gemma4.py has `cfg.S == 1` baked in).

## 2. Datapath changes

**What stays identical:** `otpu_slice`, `otpu_seq`, `otpu_dma`, `otpu_mxu` + `otpu_actram`,
`otpu_quant`, `otpu_vpu`, `otpu_tmem`, `otpu_coll` (the unit), the ISA, the compiler. The arch is
the card's `[arch]` with S = 4 (`targets/ypcb-00338.toml`: D 128, MCOLS 4, ACT_BLOCKS 128, ACT_ROWS 16,
TMEM 64K, IMEM 32K, LANES 8, PAIR / DSTEP / STREAM / SEQ_WC), the uarch the card's m4 (MXU_IMPL 2,
MXU_E 4, TMEM_IMPL 3 pumped on core_clk2x, WIN 16, FIFO_DEPTH 1024, QUANT_XD 1024).

**HBM per slice.** Slice s owns pseudo-channels 8s..8s+7: 8 x 512 MiB = 4 GiB, exactly `DRAM_BYTES`
(32-bit addresses, `docs/targets.md` Limits), so no window registers. Available: 8 x 14.4 GB/s
(256 bit x 450 MHz) = 115 GB/s; used by the slice's port: 128 B x 200 MHz = **25.6 GB/s (22%)**.
The adapter is `rtl/mem/otpu_native_dram.sv` (64-byte beats interleaved over channels, CHASH) with
its channel count raised from 2 to a parameter NCH = 8 (beat bits 6-8 pick the channel): each
channel is one HBM AXI port behind a new bridge `otpu_hbm_ch` (the role of `rtl/shell/otpu_mem_ch.sv`:
`otpu_afifo` command / write / read-data FIFOs between core_clk and the HBM's 450 MHz AXI clock,
read credits, the host DMA as the second master). With NCH 8 every port carries 3.2 GB/s (22% of
its PC) and addresses only its own PC, so the HBM IP's switch can be off (lowest latency); the
fallback is NCH 4 with the switch on (each port 6.4 GB/s, 2 PCs). **Not widened to 256 B**: a
256-byte port needs an MXU consuming two chunks a cycle or D 256 (`docs/wide_dram.md`,
`docs/scoreboard.md` "A wider port alone gives at most ~8%"), i.e. new datapath RTL, which this
candidate avoids; the 8 PCs leave 4.5x headroom for a later P-candidate that does.

**Collective.** `otpu_coll` streams one GATHER beat of LANES = 8 words a cycle at S = 1 (`otpu_coll.sv`
header). At S > 1 the BCU1525 design puts it behind `otpu_coll_hub` with an `otpu_coll_port` per
slice (reads one in flight, writes a FIFO of 8 with credits; `docs/bcu1525.md` 10): **one beat per
~7-cycle round trip**, 1.1 words a cycle. This candidate takes the follow-up that doc names, a hub
that pipelines its reads (up to 8 in flight per port): 8 words a cycle, ~10-cycle latency per
GATHER plus the slices' arrival skew. Section 5 costs both. Config / uarch keys: `[arch] S = 4`,
`[platform] mem = "hbm"`, `channels_per_slice = 8`, `channel_bytes = 536870912`, `core_mhz = 200`,
`[uarch] COLL_PIPE = 8` (new: the hub's reads in flight), everything else the card's.

## 3. ISA, compiler, host and model-code changes

No new instruction: GATHER (`docs/isa.md` 577-585) and BAR suffice. Compiler: none (`ol.all_gather`
emits GATHER at S > 1, `opentpu.language`). The work is in `opentpu/llm/gemma4.py`, derived from
qwen3.py's S = 2 scheme, one program per slice with `ol.program_id()` picking its rows:

| Block | Today (S = 1, gemma4.py) | At S = 4 (per slice s) | Collective per layer |
|---|---|---|---|
| Input norm | `rmsnorm(x, g_in)`, x [1, 1152] fp32 in TMEM | replicated: every slice holds the full x and norms it (1152 / 8 lanes = 144 cycles) | none |
| Wq (1024 x 1152) | one MM, 1024 rows, KB 9 | rows 256s..256s+255 = q head s; 256 rows, KB 9 | none |
| Wk, Wv (256 x 1152, ONE KV head) | 256 rows each | **by output rows**: rows 64s..64s+63 of each (KB 9), then `all_gather` k and v (2 x 256 fp32 = 512 words); k_norm + RoPE replicated after the gather; each slice appends to its **own full KV copy** (`kv_append`) | 1 gather, 512 words |
| Attention | 4 q heads stationary (M = 4) over the KV head, one pass | head s alone (M = 1) over the slice's KV copy: the same KV bytes a slice as today, 1/4 of the MACs; `_attend_heads` unchanged | none |
| Wo (1152 x 1024) | 1152 rows, KB 8 | `all_gather` o (4 x 256 = 1024 words) then rows 288s..288s+287 (KB 8), `all_gather` y (1152) (qwen3.py 1374-1375) | 2 gathers, 2176 words |
| Post-attn norm, residual | `_add_norm(x, y, g_attn)` | replicated on the gathered y | none |
| Gate / Up (6912 x 1152) | 6912 rows each, KB 9, F chunks of 768 (`MLP_CHUNK`, gemma4.py:669) | rows 1728s..1728s+1727 of each (KB 9); gelu_tanh x up locally (1728 wide) | none |
| Down (1152 x 6912) | 1152 rows, KB 54 | `all_gather` the product (6912 words, per 768-chunk pieces under the stream), rows 288s..+287 (KB 54), `all_gather` y (1152) (qwen3.py `_mlp`) | 2 gathers, 8064 words |
| LM head (262,144 x 1152, tied) | 262,144 rows, 8192-row chunks, ARGMAX per chunk | rows 65,536s..+65,535, chunked ARGMAX as today; then `all_gather` (max, id) and ARGMAX over the 4 (generate.py:254-257, already written for `b.S > 1`) | 2 gathers, 8 words a token |
| Embedding of the next token | gathered from the int8 head on the device | every slice's image holds the **whole** int8 head table (302 MB; the head MM streams only its 65,536 rows of it): no gather, 75 MB cheaper a slice than qwen3's separate table (`shared = ... S == 1`, qwen3.py:997) | none |

Why by output rows everywhere: splitting the hidden-input MMs by K would give rows of 1152 / 4 =
288 elements = 2.25 blocks (KB 2-3), and the MXU pays ~3 cycles a row today (section 5: the "MXU not
consuming" row gating), i.e. 2.3x on those MMs; splitting Wo by K (each head's 256 columns) gives KB 2.
Dataflow's rule (c) (down by K, `docs/dataflow.md` 4.12) needs PUTADD, which the ISA does not have;
with GATHER alone it would gather 4 x 1152 partials and add them (576 + 432 cycles) and stream
1152 rows of KB 13-14 (an uneven 14/14/13/13-block split of F's 54 blocks) instead of 288 rows of
KB 54: more cycles than (a)'s 6912-word gather (864 cycles). So (a), qwen3's scheme, as adopted here.
Total collective traffic: 10,752 words a layer x 26 = 279,552 words = **1.1 MB a token** (fp32).

Code: `Spec.check` drops `cfg.S == 1` for `n_q % S == 0 and vocab % S == 0 and hidden % S == 0`
(gemma4.py:240); `Image` builds S images (qwen3.py:1017-1046's row slicing) with per-slice row
ranges of every weight, a replicated KV region (`_KV` with every slice owning head 0) and the same
offsets in every slice; `_attention` / `_dense` / the head take the slice's ranges and the gathers
above; `compile_*` return one program per slice (qwen3.py:1106-1135); the resident decode's `out[]`
is written identically by every slice (the argmax after the gather is the same everywhere) and the
host keeps reading 64 B a token from slice 0 (`docs/autodecode.md`:49, qwen3.py:3306). Engine /
otpu-chat: `backend.write(s, ...)` of x, cos / sin, the generate state to all S slices
(qwen3.py:3296-3300 does this); the `self.cfg.S != 1` guard at gemma4.py:1256 (offload) stays off
for this model; the prefill chunk path (`gemma4.compile_rows`, `compile_prompt_run`) splits the same
way, 4 rows a pass a slice as today. Tests: `tests/test_gemma3.py` tiny random model at S = 4 on the
ISA simulator against `reference_logits`, then the board model (`tb_board.sv` with S, as
`test_tiny_qwen3_two_slices_on_board_model`).

## 4. Resource budget on the VU47P

| Resource | Per slice | 4 slices | Shell | Total | Of the VU47P |
|---|---:|---:|---:|---:|---:|
| DSP48E2 | 716 (`docs/bcu1525.md` 7, `make synthcheck`) | 2,864 | 0 | 2,864 | 31.7% of 9,024 |
| RAMB36 (TMEM in block RAM) | 177.5 (IMEM, ACT RAM at 4 rows, FIFOs; bcu1525 7) + 51 (ACT_ROWS 16: 12 more rows x 128 blocks x 136 B = 204 KB / 4 KB) + 64 (TMEM_IMPL 3 at 64K words, `ypcb-00338.toml` TMEM_NSB) = 292.5 | 1,170 | ~60 (trace 16K deep, 8 x 4 HBM bridge FIFOs, host) | ~1,230 | 61% of 2,016 |
| RAMB36 (TMEM in UltraRAM, recommended) | 228.5 | 914 | ~60 | ~975 | 48% |
| URAM288 | 16 (16 RAMs of 4K x 32: 2 a bank x 8 banks, one URAM each; pumps at 400 MHz, under URAM's 500 MHz limit, `docs/tmem_pump.md` 683-686) | 64 | 0 | 64 | 6.7% of 960 |
| LUT | ~152K (yosys logic, bcu1525 7) | ~608K | ~80K (HBM IP, bridges, F2 shell interface, control) | ~690K | ~53% of 1,304K |
| HBM bandwidth | 25.6 GB/s used of 115 available | 102.4 GB/s | host DMA at load time | 102.4 GB/s | 22% of 460 |
| HBM capacity | ~535 MB image (174 MB of layer slices + 302 MB head + 55 MiB KV + tables) of 4 GiB | 2.1 GB | | | 13% of 16 GB |

SLRs: the VU47P's three SLRs hold 672 RAMB36, 320 URAM, 3,008 DSPs each; the HBM's 32 AXI ports
are on SLR0. Floorplan: slices 0 and 1 in SLR0 beside their HBM ports; slices 2 and 3 in SLR1,
their 8 + 8 HBM bridges in SLR0 with the slow (core_clk, 64-byte native) side crossing the
SLR0/SLR1 boundary through `otpu_native_reg` stages in Laguna sites, as the BCU1525 does for its
SLR1 channels (`docs/bcu1525.md` 10: ~1,150 crossing signals a channel pair; here 4 channels'
worth a slice, ~2,300); the control block, the collective hub and the F2 shell interface in SLR1;
SLR2 free (a fifth and sixth slice later, or MCOLS 8). Two slices an SLR is 457 RAMB36 (68%) with
the TMEM in URAM, 585 (87%) in block RAM, 1,432 DSPs (48%), ~304K LUTs (~70%): the URAM TMEM is
what makes the pair fit comfortably.

## 5. Expected performance, Gemma 3 1B int8, one sequence

**Per-token byte budget** (from the model: 26 x 26.84 MB of layers + 302 MB head = 999.8 MB of
int8 weights; the Lens roofline's 7.85 M cycles x 128 B = 1.005 GB adds scales, Sw and the KV at
position 14): a slice streams 251.2 MB a token = 1.9625 M cycles = **9.81 ms at 200 MHz: 102 tok/s
at 100%**, of which the 26 layers 1.37 M cycles (6.9 ms) and the LM head 0.59 M (3.0 ms).

**Losses, re-estimated from the trace** (`docs/gemma3.md` Lens: 10.28 M cycles, port B streaming
76.4%, "MXU not consuming" 18.2%, "computing, not streaming" 3.9%, TMEM arbitration 2.1%):
- The LM head today: 3.1 M cycles for 262,144 rows x 9 chunks = 2.36 M chunk cycles, an excess of
  **2.8 cycles a row**; the layers 7.2 M for 5.45 M chunk cycles over 459 K rows, 3.8 a row (the
  attention, norms and VPU in it). The loss is per row, not per byte, so what matters is the row
  count a slice: 4,416 rows a layer (256 + 64 + 64 + 288 + 1728 + 1728 + 288) x 26 + 65,536 =
  **180 K rows a slice** against 721 K today (4x fewer chunks, 4x fewer rows: the split by output
  keeps every row at today's length, KB 8-9 and 54).
- Collective: 279,552 words a token. The BCU1525 hub (1 beat / 7 cycles): 245 K cycles; the
  pipelined hub: 35 K; plus ~133 GATHERs x ~60 cycles of latency and skew: 8 K. The MXU's 1024-chunk
  FIFO (128 KB, ~1,000 cycles) prefetches the next MM's weights across a gather whose consumer is
  the next MM (`otpu_seq` ustart / urel), so the product gather (864 cycles) + its QACT (864) exposes
  ~700 cycles a layer; counted in the 60-cycle average plus the 35 K above (conservative).
- KV read at 2048 tokens of context: 9.96 MB a slice (21.7 sliding x 512 x 512 B + 4.3 global x
  2048 x 512 B), 0.39 ms: today's absolute cost, replicated, 4% here instead of 1%.
- VPU / quantizer / TMEM arbitration exposed: ~50 K cycles a slice (today's 3.4% of 10.28 M scaled
  by the slice's share of the wide ops; the norms stay 1152 wide and replicated).

| Scenario | Cycles a slice | ms | tok/s |
|---|---:|---:|---:|
| Bound (port streaming 100%) | 1.963 M | 9.81 | 102 |
| RTL as it is: 2.8 cycles a row, BCU1525 hub | 1.963 + 0.505 + 0.253 + 0.05 = 2.77 M | 13.9 | **72** |
| Pipelined collective hub | 2.56 M | 12.8 | **78** |
| + MXU row-credit fix (1 cycle a row) | 2.24 M | 11.2 | **89** |
| + 2048-token context | 2.32 M | 11.6 | 86 |

Against today's 13.0 tok/s on the card and 19.5 at 200 MHz on one slice: **5.5-6.9x**. Per-token
cycle budget at 89 tok/s: layers ~1.60 M (6.9 M stream + row and gather overheads: 72%), head
~0.64 M (28%: 65,536 rows x 1 cycle over 0.59 M stream).

**The row-credit fix** (uarch, in `otpu_mxu.sv`): the result FIFO is RF = 32 rows at MCOLS 4 (64 at
8; `docs/mx_mxu.md`:1187) and "a row pops only while fewer than RF rows are between the pop and the
drain"; the pipeline from pop to drain is ~128 + 16 cycles deep, 16 rows of KB 9, so the gating is
the credit return and the row's Sw scale line ("a request of its own: a cycle without a chunk",
`otpu_mxu.sv`:30). Raise RF to 64 at MCOLS 4 (LUT RAM, +3 copies x 32 rows) and let the issuer
fetch the next row's Sw line ahead of its first chunk. Lens (section 8) decides whether it holds.

**Prefill:** per slice unchanged (4 rows a pass; ACT_ROWS 16 replay), weights 4x fewer a slice: a
16-row pass streams 251 MB at 4 cycles a chunk (G = 4 groups) = 7.85 M cycles = 39 ms, 2.5 ms a token,
**~400 tok/s MXU-bound before attention**; realistic 250-350 tok/s for a 256-token prompt (the
per-slice attention at M = 1 head and the per-pass gathers, 10,752 words x rows). The product gather
grows with the rows (16 x 6912 words a pass): 13.8 K cycles of 7.85 M, 0.2%.

## 6. Timing and risks at 200 MHz

- The slice: the user's 200 MHz F2 synthesis of today's design is the basis; P&R with two slices
  an SLR and the TMEM in URAM (P3 at 200 / 400 MHz meets by +0.180 ns on UltraScale+ in URAM,
  `docs/tmem_pump.md`:726) is the open item. core_clk2x = 400 MHz needs the MMCM's even divider.
- Cross-SLR: every slice-to-shell signal is already registered twice each way (`otpu_board` g_xs,
  `otpu_xr`, USER_SLL_REG) and the collective's grant is local per slice (`otpu_coll_port`): no
  combinational SLR crossing; the HBM bridges cross on the core_clk native side only.
- HBM: the IP's AXI at 450 MHz is the IP's own closure; the bridge's async FIFOs decouple it. Read
  latency ~300 ns = 60 core cycles plus FIFO; the 1024-chunk FIFO covers it 16x. Risk: the IP's
  address map / switch choice (NCH 8 switch-off vs NCH 4), decided at bring-up.
- The four sequencers run free, not in lock step (the spec's "independent slices with a gather
  network", `docs/superpowers/specs/2026-09-23-opentpu-design.md`:171-175, 253): GATHER / BAR are
  the only rendezvous, 133 a token. Identical programs over identical row counts keep the skew to
  HBM refresh jitter (tens of cycles). Risk: an uneven MM split (none planned: 256 / 64 / 288 / 1728
  / 288 / 65,536 rows all divide by 4).
- The model code: `cfg.S == 1` assumptions in gemma4.py (one image, `v_loc = vocab`, KV descriptors,
  the generate program's out[]) are the bulk of the work; qwen3.py is the template for each.
- Host: the F2 shell's PCIe (PCIS AXI4, ~20 GB/s, `tools/scoreboard.py`:341) replaces XDMA; the
  control block's AXI4-Lite moves to the shell's OCL port. New board directory, no slice change.

## 7. Implementation plan

1. **gemma4.py at S = 4 on the ISA simulator** (2-3 weeks). `Spec.check`, per-slice `Image`
   (row ranges, replicated KV, whole head table), the gathers of section 3, per-slice `compile_*`,
   Engine writes to all slices. Verify: `tests/test_gemma3.py` tiny model at S = 1, 2, 4 bit-exact
   against `reference_logits`; `tools/compare_hf.py --model gemma3` at S = 4 matches the S = 1 tokens;
   `lens record model --isa` at S = 4 gives the per-slice analytic profile (~2.0 M cycles a slice).
2. **RTL top with 4 slices** (1-2 weeks): `otpu_board` S = 4 (parameterized already), the
   pipelined `otpu_coll_hub`, `otpu_native_dram` NCH parameter; Verilator `tb_board.sv` at S = 4
   with the behavioural DRAM. Verify: the tiny Gemma 3 at S = 4 on the board model bit-exact
   against the ISA simulator (the `test_tiny_qwen3_two_slices_on_board_model` pattern); `lens
   record model --axi` equivalent on the 4-slice model for the per-slice losses.
3. **MXU row credits** (1 week, optional but worth ~14%): RF 64 at MCOLS 4 and the Sw prefetch;
   verify with the MXU cell testbench and the token-exact tests, measure in Lens on the LM head.
4. **Target and board** (2-3 weeks): `targets/f2.toml` (platform_id 3, section 2's keys),
   `rtl/boards/f2/` (the F2 shell's CL top, the HBM IP, `otpu_hbm_ch`, clocks), `boards/f2/`
   (constraints, the SLR floorplan of section 4, `create_project.tcl`, the AFI build flow), the
   host's `channel_map` for 8 channels a slice and the shell's DMA. Verify: `make lint`,
   `tests/test_target.py`, `elab_check` of the floorplan names, P&R timing report at 200 / 400 / 450.
5. **Bitstream and bring-up on EC2 F2** (2-4 weeks with the AFI turnaround): HBM calibration and
   the memory test (`tools/rw_bench.py`, `dma_bench.py`), one slice first, then 4 with the tiny
   model, then gemma-3-1b-it: `compare_hf.py --backend board`, `bench_llm.py`, Lens on the card.
Reused from the BCU1525 work (`docs/bcu1525.md` 8, 10): `otpu_board` at S > 1, `otpu_coll_hub` /
`otpu_coll_port`, the `otpu_xr` crossings and `USER_SLL_REG` constraints, `otpu_native_reg`, the
per-slice host driver (`Board.write / read(..., s)`, `load_program` per slice, ICOUNT_s), the
S-parameterized board model and its tests, `targets/bcu1525.toml` as the template.

## 8. What to verify in Lens once built

- Per-slice profiles (`lens record model --axi`, one `.otpuprof` a slice): port B streaming should
  rise from 76.4% to ~85-88%; "MXU not consuming (drain / row credits)" should fall from 18.2% to
  ~9% at 2.8 cycles a row (4x fewer rows over 4x fewer chunks keeps the ratio) and to ~3-4% with
  the credit fix; "MXU computing, not streaming" stays ~4%.
- A new line, "waiting on collective": the gathers' exposed cycles, expected 1.5-2% with the
  pipelined hub (12% with the BCU1525 hub: the signal to build it); and the slices' skew at each
  GATHER (arrival-time spread), expected under 100 cycles.
- The LM head's share: ~28% of the token (0.6 M cycles a slice); the argmax combine's two
  gathers invisible.
- Chunk FIFO starved (HBM latency) stays under 1%; TMEM arbitration of the MXU drain ~2% (the
  collective's broadcast writes add a port above the drain in priority).
- At 2048 tokens of context the attention's KV stream at ~4%: if it shows, the sequence-split
  attention (each slice 1/4 of the positions for all 4 heads, (m, l, acc) combined over the link,
  `docs/dataflow.md` 4.12) is the follow-up, not a hardware change.

## 9. Every other unit, resized

Per slice unless stated; "total" is over the 4 slices.

| Unit | Today (card, S = 1, 133.33 MHz) | This candidate (S = 4, 200 MHz) | Why |
|---|---|---|---|
| ACT RAM | 16 rows x 128 blocks x 128 B + 8 B records = 272 KB, ~68 RAMB36 (`otpu_actram`, ACT_ROWS 16) | identical a slice; total 4x (1.09 MB, ~272 RAMB36) | the stationary operand (the activation row, 1152 or 6912 wide) is full width in every slice under split (a) |
| MXU epilogue lanes E | 4 (= MCOLS, full rate, `MXU_E`) | 4 | at M = 1 a row drains one result; E binds only in prefill, unchanged a slice |
| Result FIFO RF, row gating | RF 32 rows at MCOLS 4; ~2.8 cycles a row lost; rows KB 8-9 and 54 | RF 64 (the fix), Sw line prefetched; rows keep KB 8-9 / 54 because every MM splits by output rows; 180 K rows a token a slice (721 K today) | split by K would shorten rows to KB 2-3 and triple the per-row loss |
| Chunk FIFO vs read latency | 1024 chunks = 128 KB, DDR3 latency ~30 cycles | 1024 chunks; HBM ~300 ns = 60 core cycles + bridge FIFOs: covered 16x; doubles as the prefetch across a gather | FIFO_DEPTH unchanged; 2048 is the knob if "starved" or the gather exposure shows |
| Port A (scale lines), port SW (QST) | 32 B a line, ~0.14 lines a chunk; QST writes the K row and V^T bytes | identical a slice; QST (KV append) total 4x: 4 x ~1 KB a layer (the KV replicated) | scale traffic scales with the slice's chunks; the replicated appends are ~1% of the DMA's time |
| Quantizer | LANES 8 elements a cycle; row buffer XD 1024 chunks (32 KB, rows up to 8192 elements) | identical: the longest QACT row is the gathered product, 6912 <= 8192 | the down MM's input is gathered to full F in every slice |
| TMEM | 64K words, TMEM_IMPL 3 (64 RAMB36, pumped 2x at 266.67 MHz) | 64K words a slice in 16 URAM288, pumped at 400 MHz; total 256K words | per-slice footprints shrink (products 1728 wide) while x, norm(x) and the gathered rows stay full width: 64K suffices; URAM frees block RAM for two slices an SLR |
| Banks / ports | LANES 8 banks, RPB 0 (replicated read lanes), WPB 1 | identical | the collective's write port exists already (priority DMA, COLL, VPU, MXU drain, QUANT) |
| DMA LD / ST | 128 B a cycle; logits stay in TMEM (chunked ARGMAX) | identical; the generate loop's `out[]` written by every slice; ST of x / state a slice | the per-slice logits are 65,536 fp32 = 256 KB only in debug (`lm_keep`) |
| IMEM / program / WIN | 32K words = 4,096 instructions; 5,350 executed a token in loops; WIN 16 | identical plus 5 GATHERs a layer inside the loops (~+30 static) | the programs differ by `program_id` only; a GATHER holds one window slot until all slices arrive, the MXU's ustart prefetches past it |
| Collective | `otpu_coll` 8 words a cycle at S = 1 (BAR only in the card's bitstream, COLL_GATHER 0) | hub in SLR1, a port per slice, reads pipelined 8 deep: 8 words a cycle, ~10 cycles latency a GATHER; 1.1 MB a token; GATHER is the barrier of 4 free-running sequencers | the BCU1525 hub's 1 beat / 7 cycles would cost 12% here |
| PCIe / host | XDMA Gen2 x8; 64 B of `out[]` a token from slice 0 | the F2 shell's PCIS (~20 GB/s) with a 32-port HBM address decode; image load 2.1 GB in ~0.15 s; `out[]` still 64 B a token from slice 0 (every slice writes the same ids after the (max, id) gather) | the argmax combine is on the device (generate.py:254-257) |
| Clock domains | core 133.33; core_clk2x 266.67 (TMEM); MIG / LiteDRAM user clock; XDMA 250 MHz | core 200; core_clk2x 400 (URAM TMEM, <= 500); HBM AXI 450 (the IP's, async FIFOs in `otpu_hbm_ch`); F2 shell 250 MHz (PCIS / OCL); the collective hub and all SLR crossings in core_clk | every crossing is a FIFO or a registered pair; nothing in the slice sees a new clock |
