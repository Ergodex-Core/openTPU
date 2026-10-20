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
  the read data through a 256-beat BRAM asynchronous FIFO to the core clock; the adapter's reads
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
| **v5 `f2-hbm:v5`, S 4, whole model (engine model)** | **796,958** (740 s) | **251.0** | 674,558 | -15.4% | 512,938, 64.4% | 64.5% |
| **v5 `f2-hbm:v5s3`, S 3, whole model (engine model)** | **1,045,874** (604 s) | **191.2** | 885,045 | -15.4% | 724,726, 69.3% | 69.5% |

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
the measured 226K, 1,275 DSP exactly, 359 RAMB36 against 416). Budgets: SLR0 272K LUT (the HBM
fabric's SLR), SLR1 230K, SLR2 435K; 672 RAMB36, 3,008 DSP, 320 URAM each; 23K SLLs a
boundary assumed (the brief's figure: verify on a placed checkpoint). Nothing over 80%, nothing
far under it.

| | LUT | FF | RAMB36 | DSP | URAM |
|---|---|---|---|---|---|
| a v5 slice (MCOLS 4, PORTB 4, RL 4, FIFO_DEPTH 256, ACT_ROWS 8, TRACE 4096, NRD 128) | 170K (u_mxu 41K, u_vpu 36K, u_tmem 12K, quant / seq / dma 28K, the adapter 51K) | 79K | 236 (chunk FIFO 57, scale FIFOs 7, TMEM 64, ACT RAM 29, IMEM 29, trace 8, the rest 42) | 727 (512 PEs, 36 column ends, 26 epilogue, 153 VPU / quantizer) | 0 |
| an engine (otpu_hbm_ch) x 32 | 38K | 45K | 256 (the 512 b x 256 read FIFOs, 512 x 72 mode) | 0 | 0 |
| the host chain + mux2 | 7K | 15K | 0 | 0 | 0 |
| the hub, ctrl, trace; the chain stages | 7K | 8K + 16K a boundary | 8 | 0 | 0 |

| SLR (`f2-hbm:v5`) | contents | LUT | FF | RAMB36 | DSP |
|---|---|---|---|---|---|
| SLR0 | slice 0, 32 engines, the host chain, mux2 | **215K (79%)** | 154K | 492 (73%) | 727 (24%) |
| SLR1 | slice 1, the hub, ctrl, trace, the OCL / PCIS crossings | **177K (77%)** | 103K | 244 (36%) | 727 (24%) |
| SLR2 | slices 2 and 3 | **340K (78%)** | 157K | 472 (70%) | 1,454 (48%) |

`f2-hbm:v5s3`: SLR0 205K (75%), SLR1 177K (77%), SLR2 170K (39%); RAMB36 428 / 244 / 236.

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
3. **Every SLR is at 75-79% LUT on the estimate**, SLR0 with 73% of its RAMB36: a synthesis
   that lands 10% higher needs the levers in "Utilization" (ACT_ROWS 4, the TMEM in URAM).
4. **The build flow**: `boards/f2/build.py` accepts `f2-hbm:v5` / `v5s3` (mem_path "engine":
   the CL top cl_opentpu_hbm5.sv, the floorplan cl_pnr_user_hbm5.tcl, the host's
   cl_axi_sc_1x1 alone from the HDK's prebuilt block design), checked under the Tcl mocks of
   tests/test_f2_build.py and never against the HDK: the first SynthCL run will find what the
   mocks do not (the HBM wrapper's port instance for 32 ports, the CDC constraints of the
   engines' crossings, which cl_timing_user_hbm.tcl must bound by the destination period as
   the OCL's are).
5. **The multi-slice control path**: otpu_board at S = 4 (the hub, the four slices' resets and
   starts through the chains) has run only in Verilator; the v4 card ran S = 1.
6. **The host at S = 4** reaches the HBM through port 31's mux; the ordering rule (a run's
   reads after the host's writes, pcis_wbusy) holds as in v4 since every host transaction
   completes through the global switch before RUN; the card is where this is proven.
