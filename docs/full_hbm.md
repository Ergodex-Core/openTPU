# openTPU v5: the full HBM on the F2 (docs/full_hbm.md)

Status: branch `full-hbm` (worktree openTPU-fullhbm from wide-port-alexshell 71919ae2), design
`docs/superpowers/specs/2026-10-20-opentpu-design-v5.md`, Verilator only: no synthesis, no
board. Labels: **measured** = a Verilator run of this tree's RTL, bit-exact with the ISA
simulator (`--check`); **lpn-estimate** = `tools/wideport_lpn.py`; **estimated** = arithmetic
from the v4 synthesis figures. int8 first, int4 secondary.

## What changed

v4 (`docs/wide_port.md`, `docs/f2_hbm.md`) streams one slice's port B at 512 B a cycle over 8 of
the 32 HBM pseudo-channels through per-channel SmartConnect chains, and measured 83.5 tok/s int8
on the RTL (2,394,257 cycles a token) at 0.865 beats a cycle a channel. v5 uses all 32 channels:

- **S = 4 slices** (`f2-hbm:v5`; `f2-hbm:v5s3` three), each the v4 slice at PORTB 4 / four int8
  row lanes / MCOLS 4 / ALINE 64 / LANES 8 on its own 8 pseudo-channels, one slice per SLR and
  two in SLR2; 2,048 B a cycle = 409.6 GB/s at 200 MHz (96% of AWS's measured 426 GB/s read),
  2,048 int8 MACs a cycle. The slice's BRAM is trimmed so two fit an SLR: the MXU's chunk FIFO
  256 entries of 4 chunks (was 1024), ACT RAM 8 rows (16), the trace 4,096 records (16,384).
- **Gemma 3 sharded over the slices** (opentpu/llm/gemma4.py; the spec's section 4): the query
  heads, the K / V projection rows (all-gathered before the norm and RoPE, the cache
  replicated), wo, gate / up, the down projection and the LM head by rows; six GATHERs a layer
  (k, v, the heads' outputs, wo's result, the MLP's activation chunks, the down's result).
  The MLP's width is padded to a multiple of S x D (6912 -> 7168 on four slices, +3.7% of the
  MLP) and chunked in the largest chunk that divides a slice's share (896 on four, 1152 on two
  or three: `_mlp_chunk`), since every chunk is six MMs and a gather on every slice. The
  programs are the same on every slice but for the heads a slice attends with and its LM-head
  columns; the images hold each slice's rows at the same addresses.
- **A 450 MHz engine per HBM port** (rtl/boards/f2/otpu_hbm_ch.sv, otpu_f2_hbm_eng.sv; the
  spec's section 3): the native port's commands into an asynchronous FIFO, a registered AR / AW
  generator coalescing consecutive beats into bursts of up to 8 (512 B, never across a 512 B
  group; the `cont` bit computed at the core clock), 64 bursts outstanding, direction barriers,
  the read data through the read path below to the core clock; the adapter's reads
  in flight raised to 128 a channel (`NRD`). The host keeps AWS's cl_axi_sc_1x1 chain: on its own
  port at S = 3, through a registered 2 x 1 AXI3 mux in front of port 31 at S = 4
  (otpu_hbm_mux2: the channel first, the host after 8 denied cycles; W in AW order through a
  4-entry order FIFO; B / R routed by ID bit 5 into skid buffers). Per slice the channels'
  write beats share one 580-bit bus across the SLRs (the SLL budget, section "Utilization");
  n_wdone crosses as a one-bit pulse and is counted on the slice side. The hclk logic depth as
  written: the burst's extend / start one LUT level, its close two, the AR load three, the
  command FIFO's pop into its fetch four (the deepest: a registered pop stage would cut it),
  every counter flag from its next value into a register, every AXI output a flip-flop. One
  bug the two-clock testbench caught: an extend coinciding with the registered idle timeout
  dropped a beat (the burst closed with its old length); the extend now excludes the timeout.
  `otpu_f2_hbmeng_sys` lints at S = 1 / 3 / 4 (the generic DSP; the DSP48E2 primitive is not on
  this Mac).
- **The engines' memories** (rtl/boards/f2/otpu_hbm_rpath.sv, the engine's read path): the first
  v5 synthesis (full-hbm 6a52448c on the m7a box) put each engine's 512-bit x 256 read FIFO in
  7 RAMB36 (224 in all, next to slice 0's 249 in SLR0's ~525 of CL area) and ImplCL stopped
  three times in placement on them: the port boxes' bottom-row clock regions hold few RAM
  columns (DRC UTLZ-1 at 4-5 engines x 7 RAMB36 a region, then the macro placer at 102-126%
  of the left half's boxes too). The read beats now go through a synchronous FIFO in
  UltraRAM at hclk (256 bits wide, the AXI beat, 4K deep: 4 x URAM288 an engine, 128 of
  SLR0's 320; a URAM has one clock, so it cannot cross by itself) read out at a beat a cycle
  (1.125 native beats a core cycle: the AXI side's own rate, no new limit) into pairs and a
  32-entry x 512-bit crossing FIFO in distributed RAM (512 LUT6) to the core clock. The
  command / write / B FIFOs were distributed RAM already. An engine is therefore 0 RAMB36 /
  4 URAM / ~2.0K LUT / ~2.9K FF (was 7 RAMB36 / 0 / 1.4K / 1.4K measured); the path's input
  register and the URAM's two output registers let the URAMs sit a clock region from the
  port, and the floorplan lets the whole path float in SLR0 with them (the engine's logic
  stays by its port). The engine's reservation is unchanged (RDEPTH 256 native beats held or
  in flight, the margin 8 for the longer count path): the in-flight budget stays what the
  receipts measured; the path adds 5 hclk cycles from the R register to the crossing (the
  receipt model's LAT_UP 7, was 5: the receipt delta is in "RTL (measured)"). The not-yet-
  proven assumptions, for the next SynthCL + ImplCL: Vivado infers the four URAM288 from the
  `ram_style = "ultra"` array with its two output registers (the CDC script now stops a
  synthesis with any block RAM in an engine or fewer than four URAMs an engine, so a fallback
  fails at synthesis, not two hours later in placement), and SLR0's CL area holds 128 URAM288
  sites the placer may use (the VU47P's URAM columns run through every clock-region row;
  verify with `get_sites -of_objects [get_pblocks pb_otpu_hbm] -filter {SITE_TYPE == URAM288}`
  on the box).
- **The write bus's arbiter, two registers deep** (otpu_f2_hbm_eng g_wb; the receipt model
  the same): the second v5 synthesis (01f65cb4: the engines right, hclk +0.140 over 448K
  endpoints) had the core clock at -1.363 ns over 8,445 endpoints, worst
  `g_slice[1].u_mem/wpart_reg[6][27] -> iss_all_q_reg`, 26 logic levels: the adapter's
  write-ready cone (its SW queue's head slot, a 32:1 select, the source select, n_wvalid) ran
  through the bus's combinational round robin (n_wready) back into the adapter's take (dtk),
  wiss, the issued-count add and compare into iss_all_q, and every register wiss drives (the
  bus word, the queues' pointers, c_done / d_done, iss_w: the 8K endpoints over four slices).
  Now the requests (n_wvalid against the engines' readies) are registered, the round robin
  runs over them, and its grant is registered as the ready the adapter sees (a grant two
  cycles after the request; the adapter holds a write until taken, a grant meeting no write
  takes nothing; the engine's FIFOs keep XL = 2 NXR + 2 entries below their readies), and the
  adapter's iss_all_q compares registers only (iss_w against n_wdone), the write of the last
  cycle holding wr_idle down through its wk_v register instead of an add: wr_idle is the same
  every cycle (tests/test_native_nch.py's equivalence). Each piece is now under ten levels:
  the write-ready cone into the request register (~9), the round robin into the grant
  register (~8), the grant into wiss and its registers (~9), the compare into iss_all_q (~5).
  The receipt: +0.23% on the 1-layer S = 4 point ("RTL (measured)").
- **The floorplan: a pblock per slice** (cl_pnr_user_hbm5.tcl; the first placed v5 build,
  340c4ed6 = the 6a52448c RTL with the block-RAM engines floating: WNS -4.430 ns, 161,901 of
  3.12 M endpoints failing, WHS -0.557). What it placed badly: the core clock at -4.43 ns on
  `g_slice[2].u_mem/w_blk_reg[7]_rep__2 -> w_blk_reg[6]_rep__2`, 17 levels with 8.0 of 9.1 ns
  in route (slices 2 and 3 shared one SLR2 pblock: the adapters' replicated registers across
  the die); the pump at -3.57 ns on 4,120 endpoints, `g_slice[3].u_slice/u_dma/t_re_q_reg ->
  u_tmem/...g_pc` with 7.1 ns of route (slice 3's DMA far from its TMEM box); hclk at -2.46
  ns on 40,734 endpoints, `g_port[31].u_eng/r_v_reg -> u_rq/memb_reg_2/WEBWE` (the floating
  block-RAM read FIFO far from port 31, whose region X7Y0 has no RAM column: the URAM read
  path's answer); clk_main_a0 at -1.10 ns, `u_hsc/.../s00_si_converter -> u_hrs512/.../
  m_payload_i_reg/CE` with 4.06 ns of route (the host chain's SLR crossing straight into the
  SmartConnect: the 150 MHz Alex-shell build's deb7ec12 / ba86b5df fix). The pblocks now
  (AWS's CL outline: SLR0 X0Y0:X3Y3, X4Y0:X7Y0, X4Y3, X5Y1:X5Y3, X6Y1; SLR1 X0Y4:X3Y7, X4Y4,
  X4Y6:X4Y7, X5Y4:X5Y7; SLR2 X0Y8:X7Y11; a clock region ~13.6K LUT / 24 RAMB36 / 96 DSP):

  | pblock (parent) | clock regions | cells | estimated fill (LUT / RAMB36 / DSP of the regions') |
  |---|---|---|---|
  | pb_otpu_hbm (pblock_CL) | SLR0's 25 CL regions (~290K LUT with the bottom row short) | slice 0 + its adapter (~185K LUT, 266 RAMB36, 727 DSP), the 32 engines' memories (128 URAM) and chains' SLR0 stages, the host pieces below | ~265K / ~290K = 91%; RAMB36 266 / 600 = 44%; URAM 128 / 320 = 40%; DSP 727 / 2,400 = 30% |
  | pb_otpu_hbmx0..7 (hbm) | X<x>Y0:X<x>Y1 (the right half: Y0 only) | 4 engines' logic (~5.6K LUT) + 4 register slices a column; X7 also u_hrs256 and the mux | ~7K / ~20K = 35% (the bottom row short) |
  | pb_otpu_hsc (hbm) | X7Y0:X7Y3 + X5Y1:X5Y3 + X6Y0:X6Y1 (cut to the CL: X7Y0, X5Y1:X5Y3, X6Y0:X6Y1) | u_hsc (cl_axi_sc_1x1, ~5K LUT) | 5K / 80K = 6% |
  | pb_otpu_hxs (hbm) | X4Y3 | u_hrs512x | 2K / 13.6K = 15% |
  | pb_otpu_tmem0 (hbm) | X2Y1:X3Y3 | slice 0's u_tmem, u_vpu, u_dma (~57K LUT, 68 RAMB36) | 57K / 82K = 70%; RAMB36 68 / 144 = 47% |
  | pb_otpu_mid (pblock_CL) | X0Y4:X3Y7 + X4Y4 + X4Y6:X4Y7 + X5Y4 + X5Y7 (21 regions, ~285K LUT) | slice 1 + its adapter (~185K), the hub, ctrl, trace (~7K), the chains' middle stages (FFs) | ~192K / 285K = 67%; RAMB36 274 / 504 = 54%; DSP 727 / 2,016 = 36% |
  | pb_otpu_mid3 (mid) | X4Y4 + X4Y6:X4Y7 + X5Y4 + X5Y7 | slice 3's chains' middle stages (~5K FF) | FFs only |
  | pb_otpu_tmem1 (mid) | X1Y5:X3Y6 | slice 1's u_tmem, u_vpu, u_dma | 57K / 82K = 70% |
  | pb_otpu_pcis (pblock_CL) | X5Y5:X5Y6 | u_pcis, u_hrs512, u_ocl | ~4K / 27K = 15% |
  | pb_otpu_board (pblock_CL) | X0Y8:X7Y11 | the control chain's SLR2 cells only | - |
  | pb_otpu_s2 (board) | X0Y8:X3Y11 (the whole SLR at S = 3) | slice 2 + its adapter, its hub link ends, its chains' last stages | ~185K / 218K = 85%; RAMB36 266 / 384 = 69%; DSP 727 / 1,536 = 47% |
  | pb_otpu_s3 (board) | X4Y8:X7Y11 | slice 3 likewise | 85% / 69% / 47% |
  | pb_otpu_tmem2 / 3 (s2 / s3) | X0Y9:X3Y10 / X4Y9:X7Y10 | the slice's u_tmem, u_vpu, u_dma | 57K / 109K = 52% |

  The second placed build (20f64ee3 at 200 MHz on the old floorplan, stopped after placement:
  WNS -2.991, 197,974 of 3.32 M endpoints) added two path families: the collective's port
  into slice 2's pumped TMEM (`g_xs.g_x[2].u_cport/hv_reg -> u_tmem/...g_pm`, 6.0 of 6.8 ns in
  route on the pump's 5.0: u_cport now sits in the TMEM box with the DMA and the VPU) and the
  hclk reset replica far from engine 7's floating read path (`u_mem/hrst_q_reg_rep -> u_rp/
  aq_wd_reg/CE`, 4.7 ns of route on 2.2: each engine now registers its hclk reset itself,
  otpu_hbm_ch hrst_i, and its read path once more, otpu_hbm_rpath hrst_r, both max_fanout 32
  replicas placed with their loads; the read path's data registers lost their reset enables).
  Its core-clock worst paths were slice 2's MXU DSP-to-DSP hops with 6.6 ns of route (the
  SLR2 pair scattered, which the halves address); every clk <-> hclk and main <-> hclk group
  was positive (clk_raw -> hclk +0.514 the tightest).
  What moved: slices 2 and 3 into halves of SLR2 with their adapters (both were loose in
  pb_otpu_board); slice 1's pblock takes the CL's columns beside the shell (16 -> 21 regions;
  slice 1 with its adapter would be 88% of the left half alone); slice 3's chain middle
  stages into those right columns, under its half; the DMA into every TMEM box; the host
  chain two hops (u_hrs512 in the PCIS box, a new u_hrs512x in SLR0's top row beside the
  crossing, the SmartConnect boxed toward port 31's column, u_hrs256 and the mux at the port).
  The two SLR2 halves at 85% LUT and SLR0 at ~91% are the floorplan's risk (the slice is
  132K of it, the adapter ~53K); the levers if the placer refuses are the adapter's queues
  (RD 96: -4K LUT a slice) and the engines' write FIFOs in URAM (-18K LUT in SLR0).
- **A 150 MHz core target, `f2-hbm:v5c150`** (targets/f2-hbm.toml; core_mhz 150, the TMEM
  pumped at 300: build.py's HBM_CORE_MMCM (1, 4.8, 8.0, 4)): the same slices, engines and
  sharding, the first v5 target to close timing. At 150 MHz the HBM's 280 ns is 42 core
  cycles, its 172 ns write latency 26, and each engine's AXI side delivers 1.5 native beats a
  core cycle against the port's one (the receipt model's RATE 1, run.sh MHZ=150): the memory
  path is further from the bound than at 200, so the token's cycle count can only fall; 3/4
  of the 200 MHz rate is the ceiling. LPN (lpn-estimate, --lat 42 --adapter-latency 26):
  1 layer + head 177,650 cycles (177,760 at 200), the whole model 672,448 (674,558): 223.1
  tok/s at 150 against the LPN's 296.5 at 200. The RTL receipts are in "RTL (measured)".
- **The engine path's CDC constraints** (boards/f2/build/constraints/cl_timing_user_hbm5.tcl,
  put in cl_timing_user.tcl's place for mem_path "engine"; the SmartConnect path's script
  requires the fabric u_sc, which stopped the first SynthCL of `f2-hbm:v5` right after
  synth_design): the core and clk_main_a0 bounded each way at the destination's period,
  datapath only, as v4 (the OCL's held words, the LED and HBM-ready levels); the core and hclk
  the same way (2.222 ns into hclk, the core period into the core), which meet only in each
  engine's four asynchronous FIFOs (the command and write-beat FIFOs core -> hclk, the
  read-beat and B-token FIFOs hclk -> core: a two-flip-flop synchronizer per gray pointer per
  direction, a RAM written in one clock and read in the other) and in the engine's error level
  through two core flip-flops; a bus-skew constraint per gray pointer at the shorter period
  (eight an engine, 256 at 32 ports); the LOCKED and rst_main_n exemptions. The mux, the
  geometry and the 450 MHz register slices are hclk only; cl_axi_sc_1x1 and the HDK's HBM
  wrapper carry their own. The script fails closed on what it counts (at S = 4 it must print
  `OTPU_HBM_CDC core=5.000 main=4.000 hbm=2.222 engines=32 fifos=128 bits=cq=6,wq=5,rq=6,bq=6
  sync_ffs=1472 bus_skews=256 err_levels=32 mux=1 urams=128 brams=0`; the first SynthCL with
  it, at 6a52448c, printed the same with the block-RAM read FIFOs: rq=9, sync_ffs=1664), on a
  u_sc, on a block RAM in an engine and on fewer than four URAMs an engine; validate_route.tcl
  then requires the four bounds on the routed design and the eight gray buses an engine. Both
  run under Tcl mocks of the synthesized hierarchy in tests/test_f2_build.py.
- **The receipt's memory model** (sim/verilator/otpu_hbmeng_mem.sv, `OTPU_HBMENG=1`): the
  engine's behaviour at the core clock behind the three-stage SLR chains, the HBM at 280 ns
  (`--lat 56` at 200 MHz) with 64 bursts outstanding a channel and the port's rate.
- **Verilator on this Mac**: `OTPU_VL_THREADS=4` (`--threads 4`, the m7a box's 1.8x) crashes the
  four-slice model at its first evaluation after the images load (three runs, the same
  signature: ~50 s of CPU, then an exiting process stuck in the kernel; the worker threads get
  macOS's 512 KB default stacks, the brief's RLIMIT_STACK pitfall in thread form), while the
  same model single-threaded runs through. The v5 receipts here are single-threaded (the
  1-layer + head S = 4 run: 501 s; the whole model ~30 min); on Linux the threads are the
  known 1.8x.

## The numbers

### The gates (Verilator 5.048 on this Mac, tree 1a315aa4 and after; every count a full pass)

| suite | result |
|---|---|
| tests/test_f2_rtl.py -k hbm_eng (the engine: 1x8 at NXR 3 and 0, 2x8 with and without the host mux, 4x8 with the mux = `f2-hbm:v5`, a 150 MHz core; ORDER 0 detects stale reads) | 7 passed |
| tests/test_native_nch.py | 10 passed |
| tests/test_full_hbm.py --runslow (the sharding on the ISA simulator, the real 1B at S = 4) | 6 passed |
| tests/test_f2_build.py (the snapshots incl. v5 / v5s3, both floorplans under the Tcl mock) | 104 passed |
| tests/test_target.py | 25 passed |
| OTPU_TARGET=f2-hbm OTPU_NCH=8 tests/test_rtl.py -k "replay or act_reads or attention or lane_count" | 27 passed, 1 skipped (759 s) |
| tests/test_wide_port_rtl.py (168 cases, every format, reuse and lane mode on the behavioural DRAM and the native path) | 168 passed (1:50 h, paused for the receipts) |
| ISA suites touched by the compiler change (test_gemma4, test_prefill_gemma4, test_gemma3, test_offload, test_compiler, test_row_lanes) | 16 passed, 10 skipped (no transformers in the venv) |

### ISA simulator: the token-exact gate

`tests/test_full_hbm.py`: a tiny random Gemma 3 on 2 and 4 slices against one (bit-identical
logits where the MLP's chunking is the same, int8-level agreement and the same token where it
is not), and the real Gemma-3-1B int8 decode step at position 9 on 4 slices against one (the
same token; the logits within the activation quantization's noise). The S = 1 programs of this
tree are word-identical to the baseline tree's (the receipt's step: 678 instructions, the same
sha256) so every v4 number is reproducible here.

### LPN (lpn-estimate; tools/wideport_lpn.py with --slices, the collective timed from otpu_coll's structure)

Gemma-3-1B int8 / int8 head, 26 layers + head, PORTB 4 / MCOLS 4 / RL 4 / ALINE 64, NCH 8 a
slice, --lat 60, 200 MHz, the native path's 14 cycles of adapter latency, RD 128 beats in
flight a channel (the v4 point at RD 64 for reference):

| configuration | cycles (lpn-estimate) | tok/s | binds | MLP / head / attention |
|---|---|---|---|---|
| v4: S 1, RD 64 (docs/wide_port.md; measured 2,394,257) | 2,441,652 | 81.9 | channel at 0.865 | 1,462,050 / 723,866 / 255,736 |
| S 1, RD 128 | 2,156,871 | 92.7 | channel at 1.0 | 1,271,202 / 630,947 / 254,722 |
| S 2, RD 128 | 1,159,284 | 172.5 | channel | 657,137 / 315,571 / 186,576 |
| **S 3, RD 128 (`v5s3`)** | **883,739** | **226.3** | channel | 451,502 / 210,456 / 221,780 |
| **S 4, RD 128 (`v5`)** | **673,252** | **297.1** | channel (508,631) | 366,337 / 157,883 / 149,032 |

The same on the engine path's round trip (`--lat 56`, the HBM's 280 ns, `--adapter-latency 28`:
the native path's 14 plus the chains' 6 and the engine's 8 structural cycles from
otpu_hbmeng_mem's header; RD 128 over 84: the channel at 1.0): S 4 **674,558 (296.5 tok/s)**, S 3
885,045 (226.0), S 2 1,160,590 (172.3), S 1 2,158,177 (92.7); 4 layers + head at S 4 237,377, 1
layer + head 177,760 (the whole-token numbers move 0.2% between the two round trips: the token
is not latency-bound at RD 128). One caveat of the model: upstream's matrix engine prefetches
in batches of 8 rows that must fit its FIFO, where the RTL issues advances continuously up to
FIFO_DEPTH entries; the v5 slice's FIFO_DEPTH 256 against the down projection's 8 x 54
advances a batch makes the model read 1,134,656 a token at 256 where 1024 reads 673,252, and
the S = 4 receipt (the RTL at 256) calibrates the latter (+3.8%): every engine-path point is
quoted at `--fifo-chunks 1024`.

What the model says about S = 4: the port-bound token is 506K cycles (the slice's 1,957,551 chunks
at 4 a cycle); the MLP phase sits 15% over its roofline (the two gathers a layer of 3,584 words
and the twelve MMs' tails), the head 6% over, and the **attention phase is 149K cycles against
a 40K roofline**: per layer 5.7K cycles of which 1.5K stream weights; the rest is the VPU's
softmax and norms (replicated on every slice: they do not shrink with S), the small MMs' tails
and the four gathers (k, v, o, y: ~460 cycles). It is 22% of the token where it was 12% of
v4's: the next lever after the memory path is the attention's VPU work (ULANES 16, or the
composite functions pipelined), not the port. The 4-layer + head point at S = 4: 264,302.

### RTL (measured)

The first S = 4 run of the RTL (`OTPU_TARGET=f2-hbm:v5 OTPU_NCH=8 tools/perf_qwen.py --model
gemma3 --layers 1 --wformat int8 --head-format int8 --lat 60 --mhz 200 --portb 4 --rowlanes
--mcols 4 --aline 64 --slices 4`; four slices on 32 channels of the native memory model, the
adapter still at RD 64, 501 s of simulation): **1 layer + head 195,339 cycles**, 82.7% of the
core-port roofline 161,635 (port B 646,541 chunks at 4 a cycle; the slice's MXU 161,258
advances), the channels at 0.84 beats a cycle each (the RD 64 bound, as v4), MAC 82.6%, 0
two-part reads, the 7 GATHERs of the layer 1,434 cycles in all (205 each). LPN (lpn-estimate)
202,690 (+3.8%; the head +4.5%, the MLP -1.9%, the attention -1.9%: calibrated at a mean
2.8%, the v4 model's quality).

**The engine-path receipts** (`~/ergodex-03/lens-runs/gemma3-int8-v5/run.sh`: `OTPU_HBMENG=1`,
`--lat 56` (the HBM's 280 ns), RD 128 from the target, `--check`, single-threaded; the `.log`,
`.jsonl`, `.otpuprof`, `.summary.txt` and `.html` of each run are in that directory; every run
**bit-exact with the ISA simulator on every slice**):

| configuration | RTL cycles (measured) | int8 tok/s at 200 MHz | LPN (lpn-estimate) | LPN gap | core-port roofline, efficiency | Lens: port B streaming |
|---|---|---|---|---|---|---|
| v4: S 1, PORTB 4 / RL 4 / MCOLS 8, native model RD 64 (docs/wide_port.md) | 2,394,257 | 83.5 | 2,441,652 | +2.0% | 1,966,316, 82.1% | 82.4% |
| v5 `f2-hbm:v5`, S 4, 1 layer + head (native model RD 64, lat 60) | 195,339 | - | 202,690 | +3.8% | 161,635, 82.7% | - |
| v5 `f2-hbm:v5`, S 4, 1 layer + head (engine model) | 203,697 (396 s) | - | 177,760 | -12.7% | 161,635, 79.4% | 80.9% |
| v5 `f2-hbm:v5`, S 4, 1 layer + head (engine model, the read path in URAM: LAT_UP 7) | 203,636 (385 s) | - | 177,760 | -12.7% | 161,635, 79.4% | 80.9% |
| v5 `f2-hbm:v5`, S 4, 1 layer + head (engine model, LAT_UP 7, the write bus's arbiter two registers deep) | 204,109 (396 s) | - | 177,760 | -12.9% | 161,635, 79.2% | 80.8% |
| **v5 `f2-hbm:v5`, S 4, whole model (engine model)** | **796,958** (740 s) | **251.0** | 674,558 | -15.4% | 512,938, 64.4% | 64.5% |
| v5 `f2-hbm:v5c150`, S 4, 1 layer + head (engine model at 150 MHz: lat 42, RATE 1, LAT_UP 7, the registered grant) | 202,110 (392 s) | - | 177,650 | -12.1% | 161,635, 80.0% | 81.5% |
| **v5 `f2-hbm:v5c150`, S 4, whole model (engine model at 150 MHz)** | **792,174** (745 s) | **189.4** | 672,448 | -15.1% | 512,938, 64.8% | 64.9% |
| **v5 `f2-hbm:v5s3`, S 3, whole model (engine model)** | **1,045,874** (604 s) | **191.2** | 885,045 | -15.4% | 724,726, 69.3% | 69.5% |

The read path in URAM (otpu_hbm_rpath, "The engines' memories") costs the receipt nothing:
with its five more hclk cycles from the port's R to the crossing (the model's LAT_UP 7, the
defaults of otpu_top / tb_top / otpu_hbmeng_mem since) the 1-layer + head point is 203,636
cycles, -61 (-0.03%) against 203,697 at LAT_UP 5 (`eng_s4_l1_uram.*`; the same run at LAT_UP 5
by mistake first, `eng_s4_l1_uram_lat5_stale.log`, gave 203,697 again): the engines' in-flight
budget (RDEPTH 256, RD 128) is unchanged and the port is demand-bound, so a longer return path
moves nothing. The write bus's registered arbiter ("The write bus's arbiter, two registers
deep": a write's grant two cycles after its request, in the model as in the RTL) costs +473
cycles, +0.23% (`eng_s4_l1_regw.*`: 204,109, bit-exact): the layer's writes (the K / V rows,
the QST stores) wait two cycles more each. The whole-model points above stand (their path is
the same model at LAT_UP 5 with the combinational grant: within 0.3% by these two).

At 150 MHz (`f2-hbm:v5c150`, MHZ=150 run.sh: the HBM's 280 ns = 42 cycles, its 172 ns write
latency 26, the engine model at RATE 1 since the AXI side's 1.5 native beats a core cycle
exceed the port's one) the whole token is 792,174 cycles, 0.6% under the 200 MHz count
(the latencies in cycles shrink, nothing else moves: the port is demand-bound at both), so
**189.4 tok/s at 150 MHz** against the 3/4 ceiling of 188.3; the 1-layer + head point
202,110 (-1.0% on 204,109). Lens at 150 reads as at 200 (port B 81.5% / 64.9% streaming,
the quantizer and VPU waits the whole-model losses). The engine path at 150: each engine
has 50% more AXI beats a core cycle than the port takes and the read path's 32-entry
crossing FIFO sees 1.5 pushes a pop at most, the room counted the same way; nothing in the
engines changes with the core clock (they run at 450 either way).

So the S = 4 token is 3.0x v4's on the RTL (251 against 83.5 tok/s) and the S = 3 fallback 2.3x
(191); the port-bound token at S = 4 would be 409 tok/s, and the RTL sits at 64% of it. Where
the 36% went (Lens, the S = 4 token, slice 0; `eng_s4_whole.summary.txt`):

| port B cycle | share | note |
|---|---|---|
| streaming | 64.5% | 506,308 MXU chunks + 7,596 DMA at 4 a cycle |
| waiting on quantizer work | 10.6% | the MXU waited 72.6K cycles for QACT at attention.py:117 (the queries), 44.8K at gemma4.py `_dense`, 43.3K at `_attention` |
| waiting on VPU work | 9.7% | the norms (replicated on every slice) and gelu between the MMs; VPU busy 22.4% of the token |
| MXU not consuming (drain / row credits) | 7.1% | the drain lost 83,705 cycles (10.5%) to TMEM bank arbitration |
| MXU computing, not streaming | 5.3% | the 476 MMs' prologues / epilogues (a quarter of v4's rows a slice, the same per-MM cost) |
| waiting on collectives | 1.4% | 182 GATHERs, 43,139 cycles busy in all (237 each): the collective is not the bound |
| MXU starved (latency) | 0.5% | |

Per phase (cycles; the roofline at 4 chunks a cycle): MLP 417,499 (its port roofline 314,496:
+33%), attention 199,870 (44,356: **4.5x**, 149K of it with no MM streaming), the LM head
179,587 (147,456: +22%). The channels delivered 0.80 beats a cycle each on the 1-layer run and
0.65 over the whole token -- exactly port B's busy fraction (80.9% / 64.5%): **the engine path
delivers what the MXU asks**, and the idle is on the slice's side. Two experiments on the
1-layer receipt confirm it: 128 bursts outstanding a channel instead of 64 (`OTPU_HBMENG_OT=128`)
gives the same 203,697 cycles, an idle timer of 6 hclk cycles so bursts form behind the
one-beat-a-cycle stream (`OTPU_HBMENG_IDLE=6`) 203,471 (-0.1%). The LPN's -15% gap is therefore
not its channel model (RD 128 over 84: 1.0, as measured) but what it under-charges on the MXU
side at S = 4 -- the drain's row credits and TMEM arbitration (11% of the 1-layer run) and the
per-MM prologues / epilogues, four times more frequent per byte than at S = 1 -- and the
quantizer / VPU waits of the whole token.

What this says, in order: (1) at S = 4 the token is no longer memory-bound in the layers: the
**quantizer and the VPU** (20% of the token together, every slice doing the norms and the
queries' QACT in full) and the **attention phase** (25% of the token for 2% of its bytes) are
the levers -- ULANES 16 / a wider QACT, or overlapping the attention's VPU work with the MLP's
streaming; (2) the MXU drain's TMEM arbitration (10.5%) is v4's 8% again; (3) the memory path
is not the bound: the channels deliver the port's demand at RD 128 (the OT and IDLE experiments
above move nothing); (4) the collectives cost 1.4%: the sharding itself is cheap.

## Utilization (estimated per SLR; tools/v5_utilization.py)

Block RAMs and DSPs counted from the RTL's parameters, LUTs and flip-flops scaled from the
modules Vivado measured on the v4 builds (the script reproduces the v4 board: 222K LUT against
the measured 226K, 1,275 DSP exactly, 359 RAMB36 against 416); since the first v5 synthesis
(full-hbm 6a52448c: 798K LUT, 1,290 RAMB36, 2,908 DSP in all) the script takes the v5 slice as
measured there, 132K LUT / 249 RAMB36 / 727 DSP (the estimate had 170K / 236), and the engine
as measured plus its read path in URAM (`otpu_hbm_rpath`: the 32 engines were 44K LUT / 224
RAMB36 with the block-RAM read FIFOs). Budgets: SLR0 272K LUT (the HBM fabric's SLR), SLR1
230K, SLR2 435K; 672 RAMB36, 3,008 DSP, 320 URAM each; 23K SLLs a boundary assumed (the
brief's figure: verify on a placed checkpoint). Nothing over 80%.

| | LUT | FF | RAMB36 | DSP | URAM |
|---|---|---|---|---|---|
| a v5 slice (MCOLS 4, PORTB 4, RL 4, FIFO_DEPTH 256, ACT_ROWS 8, TRACE 4096, NRD 128), measured | 132K (estimated 170K: u_mxu 41K, u_vpu 36K, u_tmem 12K, quant / seq / dma 28K, the adapter 51K) | 79K (estimated) | 249 (estimated 236: chunk FIFO 57, scale FIFOs 7, TMEM 64, ACT RAM 29, IMEM 29, trace 8, the rest 42) | 727 (512 PEs, 36 column ends, 26 epilogue, 153 VPU / quantizer) | 0 |
| an engine (otpu_hbm_ch with otpu_hbm_rpath) x 32 | 64K (2.0K each: 1.4K measured + the 512 x 32 crossing FIFO's 512 LUT6 + its control) | 93K | 0 (was 224: 7 each, the 512 b x 256 read FIFOs in 512 x 72 mode) | 0 | 128 (4 each: the 256 b x 4K read FIFOs) |
| the host chain + mux2 | 7K | 15K | 0 | 0 | 0 |
| the hub, ctrl, trace; the chain stages | 7K | 8K + 16K a boundary | 8 | 0 | 0 |

| SLR (`f2-hbm:v5`) | contents | LUT | FF | RAMB36 | DSP | URAM |
|---|---|---|---|---|---|---|
| SLR0 | slice 0, 32 engines, the host chain, mux2 | **203K (75%)** | 202K | 249 (37%; was 473 with the block-RAM read FIFOs) | 727 (24%) | 128 (40%) |
| SLR1 | slice 1, the hub, ctrl, trace, the OCL / PCIS crossings | **139K (60%)** | 103K | 257 (38%) | 727 (24%) | 0 |
| SLR2 | slices 2 and 3 | **264K (61%)** | 157K | 498 (74%) | 1,454 (48%) | 0 |

`f2-hbm:v5s3`: SLR0 187K (69%), SLR1 139K (60%), SLR2 132K (30%); RAMB36 249 / 257 / 249;
URAM 96 in SLR0.

SLR crossings (bits, both directions, every one a 3-stage otpu_xr chain): per slice's memory
8 channels x (27 down + 517 up) + the 580-bit shared write bus = 4,932 (8,960 without the bus);
the collective's links ~1.3K a slice; the control path 0.15K. With the hub in SLR1: **SLR0-SLR1
16.2K (71%), SLR1-SLR2 12.6K (55%)** at S = 4; 11.3K (49%) / 6.3K (28%) at S = 3.

The levers if a synthesis lands over the line: ACT_ROWS 4 (-12 RAMB36 a slice; prefill
streams 4 rows a pass), TRACE_DEPTH 1024 (-6), the TMEM's 64 RAMB36 into 8 URAM288 a slice
(the pumped bank in URAM: a design change to otpu_tmem, docs/tmem_pump.md's contract), RD 96 on
slice 0 (-2K LUT in SLR0).

## What still blocks an FPGA build

Nothing in this tree has been synthesized (the brief: Verilator first; the F2 host's Vivado
seat is taken, the build is a later, separately approved step). In order of risk:

1. **The 450 MHz engine's timing** is a structure, not a report: the generator's 2-3 LUT
   levels, the asynchronous FIFOs' gray pointers and the registered AXI outputs follow
   cl_mem_perf's recipe, which closed in AWS's example, but no placed WNS exists for
   otpu_hbm_ch; and the host's 2 x 1 mux (S = 4) is the kind of 256-bit two-master select that
   failed at -2.2 ns in rounds 1-8. `f2-hbm:v5s3` has no mux.
2. **The SLL budget is an assumption** (23K a boundary; 71% / 55% used at S = 4): read the
   count from `report_utilization -slr` on the v4 placed checkpoint before trusting it.
3. **Placement in SLR0**: the first synthesis landed the slice under the estimate (132K LUT
   against 170K) but its RAMB36 over it (249 against 236), and the engines' 224 RAMB36 on top
   did not place (three ImplCL rounds); the read path in URAM takes SLR0 to 249 RAMB36 (37%),
   128 URAM (40%) and ~203K LUT (75%) on the estimate, every SLR under the line. The levers
   if the next placement still fails are in "Utilization" (ACT_ROWS 4, the TMEM in URAM).
4. **The build flow**: `boards/f2/build.py` accepts `f2-hbm:v5` / `v5s3` (mem_path "engine":
   the CL top cl_opentpu_hbm5.sv, the floorplan cl_pnr_user_hbm5.tcl, the host's
   cl_axi_sc_1x1 alone from the HDK's prebuilt block design), checked under the Tcl mocks of
   tests/test_f2_build.py and never against the HDK on this Mac. The first SynthCL (full-hbm
   6214b93b on the m7a box) completed synth_design, applied otpu_clk2x.tcl (256 pumped RAMs)
   and stopped in the SmartConnect CDC script's u_sc check; with cl_timing_user_hbm5.tcl
   (6a52448c) SynthCL passed (`OTPU_HBM_CDC ... engines=32 fifos=128 sync_ffs=1664
   bus_skews=256 mux=1`; post-synthesis WNS -0.012 ns on the core clock, hclk +0.328, every
   inter-clock group positive; 798K LUT, 1,290 RAMB36, 2,908 DSP in all) and ImplCL stopped
   three times in placement on the engines' block RAMs (the floorplan commits dfb41dbe,
   bd5af5b4, 340c4ed6 and "The engines' memories" above); the fourth placed (340c4ed6, the
   block-RAM engines floating) at -4.43 ns with the four path families "The floorplan: a
   pblock per slice" answers. The next ImplCL is the first with the read path in URAM, the
   per-slice pblocks and the two-hop host chain; `f2-hbm:v5c150` is the target to close
   first. What the mocks cannot prove: the URAM inference and placement, the halves' 85%
   fill, and that report_bus_skew lists the 256 constraints (validate_route.tcl counts them).
5. **The multi-slice control path**: otpu_board at S = 4 (the hub, the four slices' resets and
   starts through the chains) has run only in Verilator; the v4 card ran S = 1.
6. **The host at S = 4** reaches the HBM through port 31's mux; the ordering rule (a run's
   reads after the host's writes, pcis_wbusy) holds as in v4 since every host transaction
   completes through the global switch before RUN; the card is where this is proven.
