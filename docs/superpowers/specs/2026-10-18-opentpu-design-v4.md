# openTPU — architecture design v4 (the wide-port F2 HBM variant, as built)

Date: 2026-10-18. Status: v4 describes the wide-port branch's RTL at commit 2b08ae69 (branch
`wide-port`), the `f2-hbm` target family of `targets/f2-hbm.toml` / `f2-hbm16.toml`, and what
`tests/` and the Verilator runs recorded in `docs/wide_port.md`, `docs/row_lanes.md`,
`docs/ternary.md` and `docs/f2_hbm.md` measured of it. Where it conflicts with
`2026-09-23-opentpu-design.md` (v3 / v3.1), v4 wins for the F2 HBM targets; the K7 / YPCB-00338
targets (`targets/ypcb-00338.toml`) stay v3.1 unless a row below says otherwise. The VLA
variant (branch `vla`: PACK, MXROT, 64 lanes, 255 ACT rows, a 1.5M-word TMEM) is a sibling,
v4-VLA, referenced here and not described.

Every number in this document is one a doc or a commit in the tree records, cited inline.
"Measured" means a Verilator run of the RTL (bit-exact with the ISA simulator) or a run on the
card; "LPN estimate" means the Petri-net tier's number, always labelled; "extrapolated" means
docs/wide_port.md's own label for a whole-model figure scaled from a 4-layer run. Nothing
projected is presented as a result; where a number does not exist it says "not measured".

## v4: what was built, and how it differs from v3

| Topic | v3.1 as built (K7) | v4 (F2 HBM, wide port) |
|---|---|---|
| Board and memory path | xc7k480t on the YPCB-00338, 2 x DDR3-1066 through LiteDRAM, core 133.33 MHz, XDMA Gen2 x8 (`targets/ypcb-00338.toml`); the slice's two DRAM ports on two channels through `otpu_native_dram` (commit 8b2ab28f). | xcvu47p on the AWS F2 (small shell, HDK 2.3.0, Vivado 2025.2): `NCH = 8` channels per slice, each its own HBM2 pseudo-channel of 512 MiB on its own AXI3 port, 256 bits at 450 MHz, 14.4 GB/s a port (`docs/f2_hbm.md`; `f2-hbm16`: NCH 16, 8 GiB in the slice's channels, CFG_DRAM 33). The adapter `rtl/mem/otpu_native_dram.sv` takes `NCH` (a power of two, 2 to 32; at NCH = 2 it is the two-channel adapter every cycle, `tests/test_native_nch.py`); a 128 B chunk is two 64 B beats on one channel pair, successive chunks rotate over the NCH / 2 pairs with the CHASH fold (`docs/f2_hbm.md` "Address map"). Per channel a bridge `otpu_f2_hbm_nch` (order check at 200 MHz, 16 writes and 64 reads in flight, an asynchronous FIFO to hclk, bursts of up to BMAX 8 native beats, GATHER 4 hclk cycles), a two-master port `otpu_f2_hbm_port` (ID 0 the bridge, ID 1 the host). Commits e5ecbf86, b354ad62. |
| The RTL's memory model for the measurements | The behavioural `otpu_dram` at a fixed latency (v3 "Known simplifications"); LiteDRAM co-simulated for the card's numbers (`docs/mtp.md`). | `otpu_native_dram` on 8 channels + `sim/verilator/otpu_native_mem.sv`, read latency `--lat 60` core cycles (tools/perf_qwen.py's default at 200 MHz, `round(0.3 * mhz)`), no stalls: every decode figure below is "OTPU_TARGET=f2-hbm OTPU_NCH=8, --lat 60, 200 MHz" (`docs/wide_port.md` "Measured on the RTL"). The three SLR crossing stages add 6 core cycles a read on the card's fabric that tb_top does not build; the equivalent `--lat 66` costs +5.9% (int4) / +7.0% (int8) on the 4-layer token (`docs/wide_port.md` "The SLR crossing registers"). |
| Port B width | One 128 B chunk a cycle: 25.6 GB/s at 200 MHz (v3 "Data path"; the v2 draft's arithmetic). | `Config.PORTB` in {1, 2, 4}: a request carries `b_n` (chunks - 1), the response `PORTB x 1024` bits with a chunk mask `b_rcm` and `b_rlast`; the slice's port-B registers are PORTB chunks wide, the MXU's chunk FIFO entry PORTB chunks with per-chunk write enables, the adapter sends a request's chunks to PORTB distinct pairs in one cycle when they lie in one aligned group of NP = NCH / 2 chunks, else in two parts a cycle apart (`docs/wide_port.md` "Design"; commits a9d539a6 the ISA, 585ebe85 / c68be81c the RTL). PORTB 1 builds none of it (every term a constant: the PORTB 1 cycles and cache keys are unchanged). CAPS2 [3:1] reports 2 or 4 (0 on a one-chunk build). |
| The column-reuse cap | PAIR for 4-bit weights only (`2 M <= MCOLS`), v3.1's `docs/isa.md`. | PAIR (flag bit6) for every format, QUAD (`rs` bit 1 with PAIR) for every format: reuse r = 1 / 2 / 4 blocks a cycle, an MM's rate r / (blocks a chunk) at most PORTB (`docs/wide_port.md` "ISA"). int8 takes 1 / 2 / 4 chunks a cycle, 4-bit 1 / 1 / 2, t2 1 (QUAD is its whole chunk). So **PORTB 8 buys nothing and int4 at PORTB 4 is int4 at PORTB 2 with the port half idle**: no instruction asks for more than 4 blocks a cycle, and a 9-block row (Gemma-3-1B, K = 1152) takes ceil(9 / 4) = 3 advances (4, 4, 1) where the port could deliver 2.25 (`docs/wide_port.md` "The whole-model tally", `docs/row_lanes.md` "Why"). The RTL's port B, slice and adapter are built to PORTB 4 (`b_n` two bits). |
| The row-lane MM mode | None. | `MM RL` (`rs` bits 3:2 = log2 L, L in {2, 4, 8}; `Config.ROWLANES`, CAPS2 bit4): at M = 1 each of L lanes streams **its own weight row**, a chunk of each a cycle, column group l computing output row n + l against the one stationary row (QACT DUP8, flag bit5, fills ACT rows 0 .. L bpc - 1 with it); the MXU's cycles are ceil(N / L) x ceil(KB / bpc) with no rounding of KB to the reuse. This is the SGMV engine's structure (Ergodex-Core/coral-gemma-f2 rtl/engine: four lanes, each its own rows) inside the MXU with openTPU's numerics unchanged: every output the same integer and the same three fp32 multiplies (`docs/row_lanes.md`; commits 04a0e45f the ISA, c9d8f221 the RTL). Legal: `L <= PORTB`, `L bpc <= MCOLS`, `L bpc bs <= 32` (int8 L 2 / 4 / 8, int4 / fp4 L 2 / 4, nvfp4 and t2 L 2). The RTL's `RL` parameter is 4 at MCOLS 4 and 8 at MCOLS 8 (`target.rl_streams`). The group walk (`srs` bit 0, commit 60b402ca) runs a plain MM of any M over the same image, so one image serves decode and prefill. |
| Port A's line | 32 bytes a cycle (the MXU's scale line). | `Config.ALINE` 32 or 64 (CAPS2 bit5; commit feb0b200). At 64 every row-lane layout reads one line an advance: int4 L 4's group Sw (H = A = 32) rode a line of its own every group at 32 (1.2 lines an advance on KB 9), at 64 it shares the first advance's line (`docs/row_lanes.md` "Port A at 64 bytes"). The adapter's beat is 64 bytes already, so the line costs no traffic; the MXU's line register and tag RAM double (512 bits). Every ALINE 32 build's netlist is unchanged. |
| Ternary T2 weights | None. | `T2` builds (CAPS bit28; commit 4e9616a4, branch ternary-bonsai merged at abf1002b): the t2 format (2-bit codes, four blocks a 128 B chunk, `MM WF 1` with `rs` bit 0), `MM QUAD` (a whole t2 chunk a cycle into four columns) and `QACT DUP4`; the QUAD datapath (four weight streams through the column hops, the PE input select by `hq`, two more scale decoders, the four-column row sum: ~7K LUT, ~10K FF estimated) is what the wide port's `QD = T2 || PORTB > 1` reuses (`docs/ternary.md` 6). Bonsai-2 decodes in t2 QUAD at MCOLS 4 and as t2 RL 2 at MCOLS 8 / PORTB 2 (`f2-hbm:p2rl8`, `docs/ternary.md` 8.1). |
| The MXU's columns on F2 | MCOLS 4 (the card's m4 arch, the `f2-hbm` base). | MCOLS 8 on the row-lane variants `p4rl8`, `p4rl8a64`, `p2rl8`, `p2m8` (`targets/f2-hbm.toml`): int4 L 4 needs eight columns (each lane's chunk is two blocks), int8 L 4 four (identical cycles at MCOLS 4 and 8: 949,485 against 949,474, `docs/row_lanes.md` "What MCOLS the mode needs"); t2 at M = 1 needs the eight columns for RL 2. The RL RMAX word `mkr` is a running max in two register stages, not a reduce over the eight columns (commit c5af9505: the first MCOLS 8 synthesis had it at -2.427 ns). |
| Arch and uarch parameters | Design config S = 2, D = 128, MCOLS = 8, LANES = 16, WIN = 32, a 128-deep MXU FIFO (v3 "Measured", "Concurrency"). | `f2-hbm`: S = 1 (variants s2 / s4), D = 128, MCOLS 4 (8 on the `*8` variants), LANES 8, ACT_BLOCKS 128, ACT_ROWS 16, TMEM 64K words, IMEM 32K, DRAM 4 GiB a slice, PAIR / DSTEP / STREAM / SEQ_WC / T2 on; uarch MXU_IMPL 2 (systolic, DSP48E2 `rtl/tech/xilinxus`), MXU_E 4, VPU_CL 2, ULANES 8, TMEM_IMPL 3 (one copy, NSB 2, RPL, TDP, GST, PMP 1: pumped at 2x), RPB 0, WPB 1, WIN 16, FIFO_DEPTH 1024, TRACE_DEPTH 16384, CHASH (`targets/f2-hbm.toml`). The arch is the YPCB card's m4 (CFG_HASH 4ffd1e17 at S = 1; `docs/f2_hbm.md`): the same programs run on both until PORTB / ROWLANES / ALINE enter the hash (`f2-hbm:p2` a9d1d2b0, `p4rl` d6ed551d, `p4rl8` 172e5c73, `p4rl8a64` dd6b92a6, `p2rl` e476f0b7, `p2rl8` 4317000c, `p2m8` 8a2b063d; `docs/wide_port.md` "What the nightly needs"). |
| Register map | Map 4: CFG_* at 0x80.. (`2026-10-10-multi-target-design.md` D3). | Map 5: `CAPS2` at 0x5C (`rtl/shell/otpu_ctrl.sv`): bit0 GATHER (the collective unit runs GATHER; a one-slice build with `COLL_GATHER = 0` runs BAR alone), [3:1] PORTB (2 or 4; 0 = one chunk, so a PORTB 1 bitstream's CAPS2 reads as before), bit4 RL (row lanes, QACT DUP8), bit5 ALINE 64. Bits 6 and 7 are the VLA branch's PACK (the MXU packs two int8 weight rows a DSP) and MXROT (branch `vla`'s `rtl/shell/otpu_ctrl.sv`, which keeps [5:1] at 0; not in this branch's). CFG_UARCH gains bit 24 SEQ_WC. The host takes PORTB / ROWLANES / ALINE from CAPS2 (`device_config`); `OTPU_PORTB` / `OTPU_ROWLANES` / `OTPU_ALINE` must agree with the bitstream. |
| ISA and flags since v3.1 (`docs/isa.md`) | NOP, HALT, LI, ADDI, LOOP, BAR, LD, ST, MM, QACT, QST, VOP, GATHER; MM RMAX / ASCALE, QACT CSCALE / RSCALE, VOP RSSQ. | Opcodes: `RLD` 0x06 (a TMEM word into a register; RAW / MUL / BOUND), `WAITW` 0x07 (the DMA waits for a word the host writes; CAPS bit31), `DSTEP` 0x12 (a Gated DeltaNet head step on the DMA; CAPS bit6), `STREAM` 0x13 (the stream engine's descriptor pass; CAPS bit26), `HALT` flag bit0 CHAIN (the board loads the next program from DRAM; CAPS bit30 with RLD and VOP ARGMAX, func 20), the sequencer's wait entries (SEQ_WC, WAITC builds). MM: `WF` bits 5:4 (int8 / int4 / fp4 / nvfp4), bit7 RSC, `rs` bit 0 t2, bit 1 QUAD, bits 3:2 RL, `srs` bit 0 the group walk; PAIR (bit6) for every format. QACT: DUP bit3, DUP4 bit4, DUP8 bit5. MX numerics (exact accumulation with power-of-two block scales, CAPS bit29). The run's arguments ARG0..7 (CAPS bit25). This branch's own additions are PAIR / QUAD for every format (a9d539a6), RL and DUP8 (04a0e45f, c9d8f221), the walk (60b402ca) and ALINE (feb0b200); the rest is main's since v3.1. |
| Host path | XDMA (Gen2 x8), the control block on its AXI-Lite. | PCIS DMA (AppPF BAR4 offset = HBM address, 250 MHz aclk): `otpu_f2_hbm_host` takes every 64 B beat through two-entry skid buffers (`otpu_skid`), crosses it to hclk and writes it as one two-beat AXI3 burst on the port that owns the address (~7 hclk cycles a beat, 4 GB/s), reads one port at a time with RQ 16 in flight, returns data in PCIS order; `pcis_wbusy` (registers only since round 5) holds the host's reads and the OCL's writes so a RUN after data finds the data in the HBM. OCL AXI-Lite: `otpu_axil_cdc` (skid buffers on AW / W / AR, one access at a time) -> `otpu_axil_xr` (pulses across the SLRs) -> `otpu_ctrl`. Every output to the shell is a flip-flop (`docs/f2_hbm.md` "The host path"; commits 7d3da502, 1197a113). |
| SLR crossing chains | None (one die). | The VU47P's three SLRs: the HBM fabric (`u_mem`: bridges, host registers, arbiters, the host path, `u_ocl`) in SLR0, the board (the slice, its adapter, the control block, the trace) in SLR2, the PCIS registers in SLR1 beside the shell's column (`pb_otpu_hbm`, `pb_otpu_mid`, `pb_otpu_board`, `pb_otpu_tmem`, `pb_otpu_pcis`; `boards/f2/build/constraints/cl_pnr_user_hbm.tcl`, read unmanaged before opt_design). Every net between SLR0 and SLR2 is a chain of three `otpu_xr` register stages (`rtl/shell/otpu_xr.sv`), one per SLR: the native ports (`otpu_f2_hbm_mem` NXR = 3: down the adapter's takes with command and 512-bit beat, 604 bits x 8; up the readies, read data, write count, error, 532 x 8; the bridges' native FIFOs 4 -> 16 entries with the ready's threshold at 8, `XL` 6), the control path (`otpu_axil_xr`), the board's reset and HBM-ready. +3 cycles each way: a read's data 6 core cycles later (`docs/f2_hbm.md` "Floorplan and SLR crossings"; commits 24422567, 953716f7, 76a8f878, d37b6b53). |
| The compiler's row-lane weight layout | Weights in rows, scale rows per row. | With `Config.ROWLANES` the projections, the down parts and the LM head are stored in **groups of L rows** (`quant.rl_rows`): group g at `sa + g rs`, chunk q of row gL + l at `sa + g rs + (q L + l) D`, so an advance is one aligned run of L chunks inside one NP-chunk group (never a two-part read: 0 of 1,952,640 wide reads on the int8 token, `docs/wide_port.md` "Row lanes"); the scale rows in groups too (`quant.rl_scales`: the L Sw words, then block k's data of the L rows at `H + k L bs + l bs`), `ssa` / `srs` multiples of ALINE. The head keeps a standard-layout copy for the embedding gather (+302 MB int8, +151 MB int4; `docs/row_lanes.md` "Why the groups"). `dot()` emits MM RL at M = 1 and the walk otherwise (`opentpu/compiler.py`); the images: `opentpu/llm/qwen3.py` `rl_alloc`, `gemma4.py`, `qwen35.py` (Bonsai-2). |
| Verification layers added | The three of v3 (fp units on 245K vectors; the ISA simulator against numpy; RTL = ISA simulator bit for bit on kernels and fuzz). | (4) `tests/test_wide_port_rtl.py`, 168 cases (the 53 wide-port, 43 row-lane, the ALINE 64 and walk cases): every format and reuse on the behavioural DRAM and on the native path (NCH 4 / 8 / 16, 20% random stalls), DRAM and TMEM bit for bit the ISA simulator's, plus the throughput and two-part-read counters (`docs/wide_port.md` "Tests", `docs/row_lanes.md` "RTL"). (5) `tests/test_native_nch.py`: NCH 2 every-cycle equal to the two-channel adapter (git 4e9616a4), NCH 2 / 4 / 8 / 16 under the host's map. (6) `tests/test_f2_rtl.py`, 16: `tb_f2hbm` (4 / 8 / 16 / 32 ports, three clock ratios, NXR 3 and 0, an HBM model that reorders across IDs, bursts formed, host order) and `tb_f2axil` (the OCL crossing). (7) `tests/test_f2_build.py`, 87: the HBM snapshots, the floorplan and the pump multicycles under a Tcl mock, failing closed. (8) `tests/test_row_lanes.py` (38), `test_wide_port.py`, `test_t2_rtl.py` (24), `test_observability.py` (18, CAPS2). (9) The LPN tier (`tools/wideport_lpn.py` on this branch; `tools/hwloop/lpn/` on branch `hwloop`): a Latency Petri Net of the slice graded against Verilator receipts, every number labelled `lpn-estimate` (below). (10) The hwloop receipts (`tools/hwloop/receipts/2026-10-16/`, `2026-10-17/` on branch `hwloop`): the frozen export's cycles with the Verilator flags and binary hashes; `docs/hwloop.md` (branch `hwloop`). |
| Clocks | core 133.33 MHz on the card (v3's 200 MHz target was the DDR3 user clock's limit). | `clk_main_a0` 250 MHz (the shell: OCL, PCIS, the HBM wrapper's 250 MHz side); `clk` 200 MHz and `clk_f` 400 MHz from `u_mmcm` (VCO 1200, /6 and /3: the slices, the bridges' core side; the TMEM pumped on clk_f); `hclk` 450 MHz from `clk_hbm_ref` 100 MHz (VCO 900, /2: the HBM AXI ports, the bridges' and host's HBM side); `clk_hbm_ref` the HBM's reference and APB clock (`docs/f2_hbm.md` "Clocks and CDC"). The CDC constraints bound each crossing direction by the destination's period (`cl_timing_user_hbm.tcl`); the pump's multicycles (`otpu_clk2x.tcl`) are read on the linked design since round 5 and fail closed (commit 5f8a940f). |
| Performance model tier | The ISA profile's nominal costs (`opentpu/profile.py`), the workloads scoreboard. | The LPN tier: MJChku/LPNDialect's compiled nets, upstream's model pinned to f237dab2 (+0.902% on the Gemma bundle against the pinned RTL), the wide model (`tools/hwloop/lpn/wide.py`, branch `hwloop`: port B as a credit of PORTB chunks with the two-part reads, the row lanes, the NCH channels at RD / round trip = 64 / 74 = 0.865 beats a cycle a channel) graded on export 96a90c2f at PORTB 2 +1.23%, PORTB 4 +0.96%, int8 RL 4 +4.11%, int4 RL 4 on MCOLS 8 +4.16%; `tools/wideport_lpn.py` (this branch, commit d3cfa8da) extends it with ALINE, t2, DSTEP / STREAM and the phases, calibrated per phase on the receipts (`docs/wide_port.md` "LPN estimate", `docs/ternary.md` 8.2). The rule: no tok/s from a point whose mean phase error is above 10%; nothing fitted. |

### Data path per slice (v4)

```
 SEQ (16 regs, 4-deep LOOP, ARG0..7, RLD, WAITC wait entries) -- one instruction a cycle into WIN = 16
  |-- DMA    LD/ST/DSTEP/STREAM/WAITW   DRAM burst port (one chunk) <-> TMEM lanes (LANES = 8 words)
  |-- MXU    MM (plain / PAIR / QUAD / RL / walk)
  |            port B: PORTB chunks a cycle (one request of b_n + 1 chunks) -> chunk FIFO
  |            (FIFO_DEPTH 1024 entries of PORTB chunks) -> S1 decode (int8 / 4-bit / t2 ->
  |            RL = 4 or 8 weight streams, block-major) -> D x MCOLS DSP48E2 columns, the PE
  |            input select by {hq, hi} -> column ends x decoded block scales (8 decoders at
  |            RL 8) -> the row sum over the reuse's columns -> result FIFO (Sw per column)
  |            -> E = MXU_E = 4 epilogue lanes -> bank-aware TMEM drain (+ L a row in RL)
  |            port A: one ALINE-byte scale line a cycle (the entry's 32 bytes an advance)
  |-- QUANT  QACT (DUP / DUP4 / DUP8) / QST   TMEM -> ACT RAM (LANES wide) | DRAM bytes (the SW queue)
  |-- VPU    VOP (ARGMAX, ...)                TMEM x (TMEM | row | col | imm) -> TMEM, 8 fp32 lanes
  '-- COLL   GATHER / BAR (GATHER optional at S = 1: COLL_GATHER)
      TMEM   64K words, one copy pumped at clk_f (TMEM_IMPL 3, NSB 2, 64 RAMB36), lost reads replayed
      otpu_native_dram (NCH 8): port B / port A / SW queues per channel, the CHASH fold, BREG,
      64 reads in flight a channel, APR 16 kept port-A beats on RL builds
```

What changed against v3's paragraph, unit by unit:

- **MXU.** The chunk FIFO entry is PORTB chunks (per-chunk write enables, counted on `b_rlast`);
  a wide MM asks for an advance's chunks in one request (`i_nl` for a row's last advance) and
  steps its chunk address by them. The scale entry holds a reuse group's data in three RAMs
  (a fourth and fifth at RL 8: slots 4 .. 7 and the other lanes' Sw), its A-byte window taken
  from one line (ALINE 64) or two (the 4-bit / NVFP4 QUAD crossings at 56 / 48). S1 feeds the
  QUAD datapath's four (RL 8: eight) streams from the entry's blocks in block-major order; the
  column's stream is `{s_hq, s_hi}[J] = J` in RL, else the command's `hq`, `hi`; the row sum adds
  columns j + L, j + 2L, j + 3L into j. The drain writes a pipeline row's L results at
  consecutive words, min(L, E) lanes a cycle, RMAX as `mkr`. Since round 4 (commit 2b763946) the
  entry is a register later (`S0p` / `S0`), so a pop's chunk reaches S1 two cycles after the pop
  (+23 cycles on the int8 receipt, -61 on int4; `docs/wide_port.md` "Round 4 RTL"), the S0
  decode selects are replicated per 64 loads and `wd` is a flip-flop (31c8d5bb). The MXU on the
  F2 is `rtl/tech/xilinxus` DSP48E2 cells (1,275 DSP for the CL; `docs/wide_port.md` "Synthesis").
- **Cycle model, MXU** (`opentpu/profile.py`, `docs/row_lanes.md` "The cycle tally",
  `docs/wide_port.md` "ISA"): a plain / PAIR / QUAD MM row is ceil(KB / r) advances at
  r / bpc chunks a cycle (int8 PAIR 1.8 chunks a cycle on a 9-chunk row: advances of 2, 2, 2,
  2, 1; QUAD 4, 4, 1); an RL MM is ceil(N / L) x ceil(KB / bpc) advances, L chunks each; t2
  QUAD one chunk an advance, t2 RL 2 two. Port A: one line an advance, plus the group's Sw line
  where it is not the first advance's (none at ALINE 64). The command's roofline is the max of
  port B (chunks / PORTB), the MXU's advances and port A's lines; the tally's "bound" adds the
  other units' nominal cycles (417K a Gemma token, ~130K net on the RTL, which overlaps them).
- **DMA.** `LD` / `ST` are one chunk a request as before; `DSTEP` (a DeltaNet head step at 8
  state words a cycle, `docs/isa.md`), `STREAM` and `WAITW` run on it. Its chunks go on port B
  one at a time (the DMA's and the loader's requests stay one chunk; `docs/wide_port.md`
  "Design").
- **QUANT.** `QACT DUP` / `DUP4` / `DUP8` write the row into 2 / 4 / 8 ACT rows in the same
  cycles (the reuse's and the lanes' operand); the MX offsets (`da`) per sub-block; `QST` on the
  adapter's scalar write queue (`SW_WR`).
- **VPU.** 8 fp32 lanes (`ULANES` 8, `VPU_CL` 2), `ARGMAX` for the decode loop; the VPU's
  `ta_addr` / `tb_addr` are the registers at the pump boundary (`docs/wide_port.md`
  "Placement", "The pumped TMEM address").
- **COLL.** GATHER is optional at S = 1 (CAPS2 bit0); BAR alone otherwise.
- **The ports.** Port B wide (above); port A at ALINE bytes with the adapter keeping 16 beats
  for reuse on RL builds (`APR`, the walk's rows after a group's first find every beat kept;
  `docs/row_lanes.md` "The adapter's port-A beat reuse"); the SW queue for QST and the host's
  partial beats. At NCH > 2 port B's request passes `BREG` (one cycle) and the per-chunk
  address, pair and halves are computed there (`docs/f2_hbm.md`).
- **The memory round trip** on the RTL runs: `--lat 60` plus the adapter's path (the LPN
  section derives 14 cycles from its structure: a 74-cycle round trip; `docs/wide_port.md`
  "LPN estimate"), 64 reads in flight a channel (`otpu_board` NRD 64, the bridge's RD 64).

### Roofline arithmetic (v4)

| Rate | Arithmetic | Source |
|---|---|---|
| v3's port B | 128 B x 200 MHz = 25.6 GB/s (on the K7 at 133.33 MHz less) | v3 spec; `docs/f2_hbm.md` "Bandwidth" |
| Port B at PORTB 4 | 512 B x 200 MHz = 102.4 GB/s; the native port's limit is 64 B a core cycle a channel = 12.8 GB/s, 102 GB/s at NCH 8, 205 at NCH 16 | `docs/f2_hbm.md` "Bandwidth"; `docs/row_lanes.md` "the 102 GB/s ceiling" |
| The HBM raw | 8 x 256 bit x 450 MHz = 115 GB/s at NCH 8 (14.4 GB/s a port; 230 at NCH 16; 460 over 32 ports) | `docs/f2_hbm.md` "Bandwidth" |
| The bridges' reads in flight | 64 x 64 B / ~400 ns = ~10 GB/s a channel, ~80 GB/s at NCH 8: 3x a PORTB 1 stream's share (3.2 GB/s a channel), under PORTB 4's 12.8 | `docs/f2_hbm.md` "The channel bridge" |
| Gemma-3-1B int8 token | 1,033,510,176 bytes (int8 head), 7,830,202 port-B chunks; PORTB 1: 7,830,202 port-B cycles (24.3 tok/s with the other units serial); PORTB 4 with row lanes: 1,957,551 port-B cycles = 102 tok/s at 200 MHz, the int8 ceiling of a 4-chunk port | `docs/wide_port.md` "The whole-model tally"; `docs/row_lanes.md` "The cycle tally" |
| 100 tok/s | 10 ms a token = 2.0 M cycles at 200 MHz; the int8 token's chunks alone take 1.96 M of them at four a cycle (98%), so int8 at PORTB 4 reaches 100 only with the port at 98% and every other unit hidden | arithmetic on the two rows above |
| Gemma-3-1B int4 token | 575,970,592 bytes, 4,255,674 chunks; 1,063,919 port-B cycles at PORTB 4 (= the MXU's 1,065,924 advances at L 4: the two rooflines coincide) | `docs/wide_port.md` "The whole-model tally", "Port A at 64 bytes" |
| Where the measured int8 token sits | 2,394,111 cycles = 83.5 tok/s, 82.1% of its roofline 1,966,576 (port A's lines; port B = MXU 1,959,556): the port's channels deliver ~3.43 of the 4 read beats a cycle the lanes ask for (the doc's "DRAM read beats a cycle a channel" column), i.e. the token is **channel-bound**: the LPN's channels at 0.865 beats a cycle a channel (64 reads in flight over a 74-cycle round trip) bound it at 2,294,625 cycles, and the RTL's head runs at 0.859 of its requests a cycle with 12.9% of the MXU's cycles blocked | `docs/wide_port.md` "Port A at 64 bytes", "LPN estimate" |
| Where the measured int4 token sits | 1,440,727 cycles = 138.8 tok/s, 74.5% of the coinciding rooflines 1,072,684 (port B = port A; the MXU 1,065,924): the same channel bound (LPN 1,321,067) plus the other units' ~130K cycles (9% of an int4 token) | `docs/wide_port.md` "Port A at 64 bytes", "LPN estimate" |

### Measured (RTL, f2-hbm NCH 8, --lat 60, 200 MHz)

Verilator 5.048, bit-exact with the ISA simulator in every run; `tools/perf_qwen.py --model
gemma3 --layers 4 | 0 --lat 60 --mhz 200 --portb N [--rowlanes] [--mcols 8] [--aline 64]
[--wformat int4 --head-format int4]`; "4L + head" is 4 layers and the whole 262K-row head, "whole"
the 26 layers and the head. All from `docs/wide_port.md` ("Measured on the RTL", "Row lanes",
"Port A at 64 bytes", "LPN estimate", "Round 4 RTL") unless cited otherwise.

| Gemma-3-1B decode | PORTB / MCOLS / mode | 4L + head, cycles | efficiency to its roofline | whole model, cycles | tok/s | LPN estimate, whole (error) |
|---|---|---|---|---|---|---|
| int8 | 1 / 4 / plain | 3,228,204 | 99.4% (port B) | not measured (the 26-layer run hit rtlsim's 3600 s timeout; 7.96 M extrapolated) | (25.1 extrapolated) | - |
| int8 | 2 / 4 / PAIR | 1,894,313 | 93.0% (MXU) | **4,652,571 measured** (91.2% of the MXU roofline 4,243,208; 300,202 of 3,574,528 wide reads two-part) | **43.0** | - |
| int8 | 4 / 4 / QUAD | 1,514,346 | 70.3% (MXU; 56.8% of the wide reads two-part) | not measured (3.78 M extrapolated) | (52.9 extrapolated) | - |
| int8 | 4 / 4 / RL 4 | 949,485 | 84.5% (port B 801,995; port A 801,632; MXU 800,592); 0 two-part reads | **2,393,990 measured** (699 s; 82.1% of 1,966,576; MAC 81.9%, blocked 11.6%) | **83.5** | 2,444,351 (+2.10%; branch `hwloop` README) |
| int8 | 4 / 8 / RL 4, ALINE 32 | 949,474 | 84.5% | " (11 cycles from MCOLS 4) | 83.5 | - |
| int8 | 4 / 8 / RL 4, ALINE 64 | 949,474 (949,497 after round 4's fix 3) | 84.5% | **2,394,111 measured** (681 s; unchanged to the cycle: int8 L 4's Sw shared its line at 32 already) | **83.5** | 2,441,652 (+2.0%), 81.9 tok/s |
| int8 | 2 / 4 / RL 2 | 1,637,409 | 98.0% (port B) | 4,063,309 (the doc gives it without a "measured" label) | 49.2 | - |
| int4 | 4 / 8 / QUAD | 1,610,911 | 66.0% (MXU) | 3,933,136 (as above, unlabelled) | 50.9 | - |
| int4 | 4 / 8 / RL 4, ALINE 32 | 579,306 | 92.3% (port A 534,712; MXU 441,256; port B 442,699); 0 two-part | **1,474,025 measured** (288 s; 89.1% of 1,312,940 port-A lines) | **135.7** | 1,502,766 (+1.95%; branch `hwloop` README) |
| int4 | 4 / 8 / RL 4, ALINE 64 | 565,739 (565,678 after round 4's fix 3) | 78.3% (port B 442,699 = port A 442,296 = MXU 441,256); 440,232 lines, 0 Sw alone | **1,440,727 measured** (352 s; 74.5% of 1,072,684; MAC 74.0%, blocked 12.6%, starved 2.6%) | **138.8** | 1,476,251 (+2.5%), 135.5 tok/s |
| int4 | 2 / 4 / RL 2 | 922,723 | 96.0% (port B) | 2,307,621 (unlabelled) | 86.7 | - |

The 2-layer receipts the LPN tier is graded on (`tools/perf_qwen.py --layers 2`, int8): PORTB
1 / 2 / 4 = 2,798,478 / 1,643,595 / 1,308,691 cycles (`docs/wide_port.md` "Gemma-3-1B int8, two
layers"); LPN +0.83% / +1.23% / +0.96% (branch `hwloop`, `tools/hwloop/lpn/README.md` 2). The
per-phase calibration of the int8 RL 4 / ALINE 64 4-layer token: head 686,825 against 723,866
(+5.4%), MLP 222,470 against 224,931 (+1.1%), attention 40,177 against 39,344 (-2.1%), the
token +4.1% (`docs/wide_port.md` "LPN estimate"); the int4 4-layer token +5.9%.

Prefill over the row-lane image (`--rows 4`, 4L + head, PORTB 4 / MCOLS 8, ALINE 64): int4
1,334,346 cycles on the row-lane image against 1,761,842 on the standard one (24% faster, 0
two-part reads), int8 1,614,575 against 1,675,893 (3.7%) (`docs/wide_port.md` "Port A at 64
bytes", `docs/row_lanes.md` "Prefill over the group layout").

| Bonsai-2 (t2 QUAD with the rotation unless said; `tools/perf_bonsai2.py MODEL f2-hbm --layers 1`) | MCOLS / PORTB / mode | 1 layer + head, cycles (speedup) | bound, efficiency | whole model, cycles | tok/s | LPN estimate (error) |
|---|---|---|---|---|---|---|
| 2B | 4 / 1 | 1,171,020 | port B 1,144,993, 97.8% | **4,766,291 measured** (670 s; 89.1% of the port) | **42.0** | - |
| 2B | 8 / 2 / QUAD | 1,170,384 (1.00x) | MXU 1,119,488, 95.7%; port B 48.9% | not measured (still running at the doc's hand-back) | - | - |
| 2B | 8 / 2 / RL 2 | 786,339 (1.49x) | port A 705,024 lines, 89.7% | **3,698,534 measured** (856 s; 1.29x; the head 1.54x, a layer 1.23x) | **54.1** | 3,537,068 (-4.4%), 56.5 |
| 4B | 4 / 1 | 1,554,988 | port B 1,517,376, 97.6% | **10,443,310 measured** (1579 s; 90.1% of the port; `docs/ternary.md` 8) | **19.2** | - |
| 4B | 8 / 2 / QUAD | 1,549,005 (1.00x) | MXU 1,477,248, 95.4% | not measured | - | - |
| 4B | 8 / 2 / RL 2 | 1,005,571 (1.55x) | port A 892,800 lines, 88.8% | **7,490,637 measured** (`docs/ternary.md` 8.2) | **26.7** | 7,367,742 (-1.6%), 27.1 |
| 2B, unfolded int4 + int8 head | 4 / 1 -> 4 / 2 | 4,234,618 -> 2,203,107 (1.92x; 0.05% of the wide reads two-part) | port B, 99.9% -> 96.0% | not measured | - | - |

Sources: `docs/wide_port.md` "Bonsai-2 2B", "Bonsai-2 at MCOLS 8"; `docs/ternary.md` 8, 8.1,
8.2. The 9B is not run on the RTL. t2 at MCOLS 4 is the same at every PORTB (a QUAD takes its
whole chunk a cycle); MCOLS 8 alone leaves it where it was; RL 2 is bound by port A's one line a
cycle (an RL 2 t2 advance reads 1.25 lines for two chunks at ALINE 32) and, on the whole model,
by the layers' VPU work no longer hidden behind the halved MMs (`docs/ternary.md` 8.1).

**The board (the only board numbers; all on Alex's PORTB 1 burst200 image, AFI
agfi-0d06cb48544a02106: 200 MHz, HBM over 2 channels, no source in the tree).** Gemma-3-1B int8
served by otpu-serve, `tools/f2_bench.py`, 2026-10-16: **22.3 tok/s** decode (short prompt) /
21.7 (long), ~89% of the RTL's port-bound ~25; prefill 3.7-4.5 tok/s (TTFT 4.3 s for 16
tokens, 59.6 s for 268) (`docs/wide_port.md` "Baseline on the board"). Bonsai-2 on the same image
(`docs/ternary.md` 5): 2B int4 + int8 head 16.8 / 16.7 tok/s served (9.06 by the logits probe;
12.36 M cycles a token, 78% of the port), rotated int4 19.3; 4B 8.0 (6.28 probe), rotated 8.6,
with MTP 10.9 / 6.2. Nothing of this branch's RTL (NCH 8, PORTB 2 / 4, row lanes, ALINE 64,
MCOLS 8) has run on a card.

#### FPGA timing (F2, Vivado 2025.2, `f2-hbm:p4rl8`: PORTB 4 / MCOLS 8 / RL 8 / NCH 8)

Synthesis first (`docs/wide_port.md` "Synthesis", 2026-10-17): the whole CL 251,520 LUT (19%),
196,289 FF (7.5%), 426 BRAM (21%), 1,275 DSP (14%); post-synthesis every path family positive
but the MXU's `mkr` reduce at -2.427 ns (fixed in c5af9505). Module sizes at fb42c14e (LUT / FF /
RAMB36 / DSP): `u_slice` 174,618 / 131,573 / 388 / 1,275 (`u_mxu` 97,272 / 70,237 / 128 / 1,122,
`u_vpu` 36,165, `u_tmem` 11,913 / 5,129 / 64, `u_quant` 9,383, `u_seq` 9,044, `u_dma` 9,251), the
adapter 46,717, the HBM fabric `u_sys/u_mem` 29,514 / 39,848 (`docs/wide_port.md` "Placement,
round 2"). The base `f2-hbm` (PORTB 1, MCOLS 4) at 4bef3758 synthesized at 174,919 LUT, 727 DSP,
261.5 RAMB36 (`docs/f2_hbm.md` "Synthesis"); placed at 96a90c2f with SpreadLogic_high at
-4.102 ns (the shell and the host path fail on their own at PORTB 1).

| Round | Commit | What changed | Placed WNS | Per clock (ns) |
|---|---|---|---|---|
| 1 (2026-10-17) | not recorded in `docs/wide_port.md` (the first placed p4rl8 build; directive SSI_SpreadLogic_high) | the first synthesis and placement; free placement over three SLRs | **-4.888** (TNS -123,690; 102,854 failing endpoints of 1,019,569) | hclk -4.888 (the host path's W half select into 8 ports, 6.9 ns route); clk -4.052 (the reset's 4,700 pins, the bridge's `cf_h`); clk -> clk_f -2.356; clk_main_a0 -0.744; main -> hclk -2.861 |
| 2 (2026-10-17) | fb42c14e (= cf4ec4e5: the reset trees, the host's master registered per port `otpu_f2_hbm_hreg`, the bridge's head register and `WT`, the CDC bounds by destination period, `mkr`; directive SSI_SpreadLogic_low) | the round-1 fixes; SLR0 161,809 LUTs, SLR1 91,118; 7,693 unregistered SLLs | **-4.608** | clk -> clk_f -2.995; main -> hclk -1.279; hclk worst path 6.338 ns (6.127 route: the host block far from the port registers); clk worst 8.455 ns (8.147 route: the adapter, the arbiter and the units across the split); hclk / clk WNS not recorded separately |
| 3 (2026-10-18) | 76a8f878 (the floorplan read unmanaged before opt_design, `--place-directive Default`; d9cdda12 before it failed opt_design's DRC HDPR-23 on empty pblocks) | the four pblocks applied; the board 168K LUTs at 50% of SLR2, the fabric 30K at 11% of SLR0 | **-5.567** after place_design | after phys_opt: clk_raw -1.42 (MXU-internal: `mx_gq` -> `p1_sw`, the weight FIFO's BRAM -> `wd`); hclk_raw -1.98 (the bridge's `wf_n` -> 604 `k_d` enables); clk_raw -> clk_f_raw -1.34 (`pk_a` -> the bank's RAMs); clk_main_a0 -5.01 (the shell's PCIS register through `u_host` and back across two SLR hops) |
| 4 (2026-10-18) | d37b6b53 (= RTL 1ea47dc1: c4c6e23a the bridge's `k_ld` replicated, 1f8dccca the pumped RAM in 1K-word pieces, 2b763946 the MXU's entry a register later, 7d3da502 the PCIS skid buffers; + `pb_otpu_pcis`) | the round-3 fixes; five pblocks | **-4.546** after place_design | clk_raw -1.62 (`lg0` into the weight-chunk SRLs, 5.4 of 6.0 ns route); hclk_raw -2.04 (the bridge's W count, 6 LUT levels); clk_raw -> clk_f_raw -1.97 (the pump's multicycle not applied on the linked design: a 4.6 ns path timed at one fast period); clk_main_a0 -4.55 (the OCL's `awready` through `pcis_wbusy` on the shell's live `wvalid`) |
| 5 | RTL 9c5eb07b (the bridge's W count in 3 levels), 31c8d5bb (the MXU's decode selects per 64 loads, `wd` a flip-flop), 1197a113 (`pcis_wbusy` registers only, the OCL's skid buffers), 5f8a940f (the pump multicycles read on the linked design, failing closed); docs 2b08ae69 | the round-4 fixes, each bit-exact (receipts int8 949,497, int4 565,678: round 4's exactly) | **running** (not in the docs at 2b08ae69) | - |

Each round's RTL fixes were bit-exact on the regression suites and the decode receipt (the
m7a box: `tests/test_f2_rtl.py` 16, `test_wide_port_rtl.py` 168, `test_rtl.py -k "replay or
act_reads or attention or lane_count"` 27 + 1 skipped, `test_observability.py` 18,
`test_f2_build.py` 87; `docs/wide_port.md` "Round 5 RTL").

### Concurrency (v4)

What changed against v3.1's scoreboard, two-phase MM, TMEM arbiter and port policies:

- **Scoreboard.** WIN = 16 on the F2 targets (v3's design config was 32). The sequencer's wait
  entries (`SEQ_WC`, WAITC builds, `docs/isa.md` "Wait counts"): the program carries each
  instruction's dependencies; `perf_bonsai2`'s runs use that sequencer, `perf_qwen`'s the
  range scoreboard, with the same programs (`docs/wide_port.md` "LPN estimate").
- **Port B.** The MXU's request is `b_n + 1` consecutive chunks (DMA and loader requests stay
  one chunk; writes too). The adapter's `BREG` register takes it a cycle after the grant;
  `b_rdy` reserves QD - 1 / OD - 1 of the queues (QD - 2 / OD - 2 on a wide build, for the
  two-part case) and gains `b_hold`, a register, for a split request's first part. A request
  inside one aligned group of NP chunks goes out in one cycle on PORTB distinct pairs; across a
  group boundary in two parts, each a port-B cycle (7% of the wide reads at PORTB 2 on Gemma's
  9-chunk rows, 57% at PORTB 4 QUAD, 0 with row lanes). Each part is a tag entry and a response
  once every one of its pairs' two channels holds a beat; the merge pops a channel at most once
  a cycle; the MXU writes the response's chunks into its entry and counts the entry on
  `b_rlast` (`docs/wide_port.md` "Design"). No commit on this branch since f237dab2 changes
  port B's DMA-before-MXU order (v3.1).
- **The adapter's reads in flight.** 64 reads a channel (`otpu_board` NRD 64; the bridge's
  return FIFO RD 64), so R is never stalled; the bridge's order check (a read waits on a write
  to its beat in flight and the reverse) runs at 200 MHz over 16 writes and 64 reads. At NCH 8
  a PORTB 1 stream needs 3.2 GB/s a channel of the bridge's ~10 GB/s in flight; PORTB 4 with
  row lanes asks every channel for a beat every cycle, and the RTL delivers ~3.4 of the 4
  (`docs/f2_hbm.md` "The channel bridge", `docs/wide_port.md` "Port A at 64 bytes").
- **The chunk FIFO.** FIFO_DEPTH 1024 entries of PORTB chunks (2048 x 1024 b at PORTB 2, 4096 x
  1024 b at PORTB 4, estimated 58 / 116 RAMB36; `docs/wide_port.md` "Resources and timing"); a
  replayed row (M > MCOLS) still needs KB entries. The two-phase MM (`ustart` / `urel`, the
  2-entry command queue) is v3.1's; the pop head moves to the next command only if the
  pipeline treats both alike (`rl` and L added to the condition, `docs/row_lanes.md`).
- **Port A.** One ALINE-byte line a cycle, one request an advance; the issuer stages a lone Sw
  line (`r_stl`) where the group's Sw does not fit the first advance's line (ALINE 32, int4 L 4);
  the adapter keeps APR = 16 port-A beats for reuse on RL builds (1 elsewhere) so the walk's
  rows after a group's first do not restart the channel's run.
- **The TMEM pump.** TMEM_IMPL 3, PMP 1: one copy of 8 banks x NSB 2 RAMs at clk_f = 2 x clk
  (64 RAMB36 at 64K words against v3's replicated copies: RPB = 4 read ports per bank implied
  replicated BRAM, v3 "Known simplifications"), each RAM since round 4 in NPG = 4 pieces of 1,024
  words (`otpu_pump_bank`, one RAMB36 each, with its own copy of the registered write stage; the
  word selected by a register loaded with the read: one LUT level on the clk_f -> clk path). The
  contract (`docs/tmem_pump.md`, `otpu_clk2x.tcl`): a bank's read address and enable pins are
  sampled on even fast edges only, a multicycle of two fast periods `-from clk -through` every
  bank's `ra / rb / rena / renb` `-to clk_f`, so every core register that reaches a pumped
  bank's address (the VPU's `ta_addr`, the MXU's `t_raddr`, the slice's `rl_pq` through the
  collective's port) gets one core period; a further register would be a cycle of TMEM read
  latency on every port and was not added (`docs/wide_port.md` "Round 4 RTL" 2, "The pumped
  TMEM address"). The TMEM's lost reads are replayed under the board's arbiter (RPL,
  `docs/tmem_nocopy.md`); the arbiter's grant order of v3.1 is not revisited here.
- **The drain at MCOLS 8.** E = MXU_E = 4 epilogue lanes (`targets/f2-hbm.toml`; the MXU_E
  comment in `targets/ypcb-00338.toml`: E 4 halves prefill q.K^T at d 128); an RL row's L results
  go to consecutive words in distinct banks, min(L, E) lanes a cycle (two cycles a row at MCOLS
  4 and L 4, covered by a row's KBc >= 2 advances; the result FIFO absorbs KB 1 rows), then + L
  for the next row (`docs/row_lanes.md` "The drain"). The slice's grants (`t_gnt`, the MXU's
  enable for ~5,000 flip-flops) are replicated per 256 loads (commit 721095b2).
- **The host and the control path** are not on a run's cycle count: the OCL-after-PCIS order
  holds a cycle later than before round 5 (`docs/f2_hbm.md` "The host path").

### Known simplifications and open items

- **Timing closure at 200 / 450 MHz is not reached.** Round 4 placed at WNS -4.546 ns; the
  bridge domain (hclk 450 MHz, 2.2 ns) has cones of 3-5 levels left after fix 7 (the head's
  25-bit compares at load, the open burst's updates, the queue push, `m_wdata` through the
  port's two-master select), each to be brought to 3 levels by hand; the OCL / PCIS handshakes
  on clk_main_a0 were -4.55 before fix 5; the pump's multicycles were in effect in no placed
  round before round 5 (the Tcl's exceptions did not survive the link; most banks were timed
  single-cycle in rounds 1-4). Round 5's placement is running. The honest fallback if hclk stays
  negative: hclk at 300 MHz on f2-hbm16's 16 pseudo-channels (16 x 256 bit x 300 MHz = 153 GB/s
  raw, above port B's 102 GB/s at PORTB 4) (`docs/wide_port.md` "Round 5 RTL" 7).
- **The int8 100 tok/s gap.** Measured 83.5 tok/s (2,394,111 cycles) against the port's 102
  ceiling, channel-bound on 8 channels at ~0.86 beats a cycle a channel. Levers, none
  measured on the RTL yet: a quantized (int4) LM head with int8 layers (the int4 token is 138.8
  with the whole model in int4; the head is 686,825 of the int8 4-layer token's 949,474 cycles),
  NCH 16 (`f2-hbm16:p4`: half a beat a channel a cycle, where the LPN says the channels stop
  binding; no Verilator receipt at NCH 16), and MTP (`docs/mtp.md`: the k = 1 loop runs on the
  ISA simulator for Qwen3.5, the loop on the card is phase 3 and not built on this branch; on t2
  at MCOLS 8 it is `f2-hbm:p2m8`, 1.21x over plain t2 decode on the ISA simulator but 0.79x
  against RL 2, `docs/ternary.md` 9.1, 10). The other units' ~130K net cycles (5% of an int8
  token, 9% of an int4 one) want the VPU / quantizer / DMA overlapped with the next MM's
  streaming (`docs/vpu_pump.md`, `docs/seq_vpu.md`; the Bonsai-2 whole model shows it first).
- **Nothing board-measured beyond the baseline.** The 22.3 tok/s (Gemma int8) and the Bonsai-2
  rows are Alex's PORTB 1, 2-channel image; the LPN's and the RTL's memory path (NCH 8, 64 reads
  in flight, `--lat 60`) has no card number against it (`docs/hwloop.md` "Missing calibration
  inputs", branch `hwloop`). The card's 78% port efficiency on that image, applied to the int4
  point, is the doc's ~105-120 tok/s expectation, not a measurement.
- **PORTB 8 needs NCH 16 and an 8-lane mode.** The RTL's port is built to PORTB 4; int8 L 8 on
  f2-hbm16 (0.98 M advances, 1.10 M port-A lines at ALINE 32, 132 tok/s serial on the tally) is
  ISA and simulator only, and at the native port's limit on 16 channels the bridge's ~10 GB/s
  of reads in flight a channel caps it near 78% (`docs/row_lanes.md` "The cycle tally",
  `docs/wide_port.md` "The whole-model tally").
- **The slice addresses 4 GiB** (32-bit byte addresses shifted to words in six places); f2-hbm16
  runs programs of up to 4 GiB interleaved over 16 channels until the word-address ISA change
  (`docs/f2_hbm.md` "Past 4 GiB a slice"). Bonsai-2 27B's t2 image (~7.3 GiB) waits on it or
  on `f2-hbm:s2`.
- **The embedding gather's head copy** (+29% of the int8 head image) stays until the gather
  computes the group address on the VPU (`docs/row_lanes.md` "The gather over the group layout").
- **The t2 scale entry** for RL 2 is 1.25 lines an advance at ALINE 32 (port-A bound; the LPN
  agrees); the 64-byte line for t2 L 2 or two lines a cycle is the next step there
  (`docs/ternary.md` 8.2, 10). Bonsai-2's RL image is a decode image; its prefill is the walk's.
- **The ArcSim backend** (`OTPU_SIM=arcsim`, branch `arcsim`) is experimental: bit-exact on the
  behavioural-DRAM tests, the f2-hbm board model's adapter check open, 20-70 minute builds
  (`docs/hwloop.md` inventory, branch `hwloop`).
- **PACK and MXROT** (CAPS2 bits 6 and 7) are the sibling branch's (`vla`); this branch's
  `otpu_ctrl.sv` leaves bits 6 .. 31 zero.
- **The LPN tier is an estimate**: +2.0% / +2.5% on the Gemma whole-model tokens, -1.6% /
  -4.4% on Bonsai-2's (the DeltaNet phase reads low), the adapter's 14-cycle path derived, not
  measured (10 / 12 / 14 / 16 cycles move the int8 token -1.1% / +1.5% / +4.1% / +6.7%). It does
  not model the DMA's and QST's beats on the channels, QACT DUP8's copies, the adapter's A runs,
  SEQ_WC's wait entries, HALT CHAIN (`docs/wide_port.md` "LPN estimate").
- v3's simplifications that stand on this branch: QST's byte-a-cycle writes; the fp functions'
  pipelining for 200 MHz is the F2 synthesis's to confirm (post-synthesis every core path met
  setup at 4bef3758, `docs/f2_hbm.md` "Synthesis").
