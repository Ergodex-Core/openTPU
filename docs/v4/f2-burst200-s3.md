# f2-burst200-s3: three openTPU slices on Alex's burst200 fabric

Target `f2-burst200-s3` (`targets/f2-burst200-s3.toml`): three of today's slices (D 128,
MCOLS 4, 8 lanes, the f2-burst200 arch and uarch) on the VU47P, one per SLR, each on its own
pair of 2 GiB HBM windows behind its own copy of Alex's `f2_mem` SmartConnect, six HBM AXI
ports in all; Gemma 3 1B int8 split over the three slices (tensor parallel, one stream) so
that a decode token streams a third of the weights per slice through three times the port
bandwidth of the one-slice image (build 6: 22.2 tok/s, port B streaming 89% of the token,
[README.md](README.md)). The flow is Alex's (`boards/f2/burst200/build.py`), extended to take
the target; the one-slice path is unchanged. Rules followed: [hbm-integration.md](hbm-integration.md)
(the HDK example `cl_dram_hbm_dma`'s structure and floorplan; the AWS Shell Interface
Specification's multi-SLR tips).

## The split (gemma4.py at S = 3)

P4's split ([proposal-4-four-slices.md](proposal-4-four-slices.md) section 3,
[../gemma4.md](../gemma4.md) "Slices": every matrix by output rows, the inputs all-gathered,
never a split by K) needs every split dimension divisible by S. Gemma 3 1B at S = 3: hidden
1152 = 3 x 384 and FFN 6912 = 3 x 2304 divide; the 4 query heads, the single KV head's 256
K / V rows and the 262,144-row vocabulary do not. Pipeline parallelism (layers on different
slices) would not raise a single stream's tokens/s (one slice streams at a time), so the
tensor split is kept and the three odd dimensions are split unevenly:

| Block (per layer) | S = 1 | Per slice at S = 3 | Gather |
|:--|:--|:--|:--|
| input norm | x [1, 1152] | replicated | |
| W_q (1024 x 1152) | one MM | `split_rows(1024, 3, 128)` = 384 rows: slices 0 and 1 rows 0..767, slice 2 rows 768..1023 and 128 zero rows (`rowpad`); q gathered to [1, 1152] and cut to [1, 1024]: every slice holds the whole q | 1152 words |
| W_k, W_v (256 x 1152) | | `split_rows(256, 3, 128)` = 128 rows: slice 0 rows 0..127, slice 1 128..255, slice 2 zero rows; gathered to [1, 384] and cut to [1, 256]; k_norm, RoPE and the append to the slice's own KV copy replicated | 2 x 384 words |
| attention | 4 heads, M = 4 | the same 4-head entries in every slice (the one-slice kernel: `nq_loc = n_q`, `h0 = 0`) over its own KV copy; the output o [1, 1024] local, no gather | |
| W_o (1152 x 1024) | | rows 384 s .. + 383 on the local o; y gathered | 1152 words |
| gate / up (6912 x 1152) | | rows 2304 s .. + 2303 of each, pieces of the F chunk (768); gelu_tanh x up per piece | |
| down (1152 x 6912) | | the product gathered to [1, 6912] in F order; rows 384 s .. + 383 over all 54 blocks in the one-slice image's 9 column parts; y gathered | 6912 + 1152 words |
| LM head (262,144 x 1152) | 32 chunks of 8192 | `split_ranges(262144, 3, 128)`: rows [0, 87296), [87296, 174720), [174720, 262144) (682 / 683 / 683 blocks); `descriptors(sid)` exposes `v_off`, which `_lm_head`, `fill_logits`, the sampler and the Engine's logits reads use; the (max, id) combine is generate.Greedy's | 2 x 3 words a token |

Six gathers a layer as at S = 4 (the q gather replaces the o gather). The attention
projections are padded rather than replicated: a slice streams 640 rows of W_q / W_k / W_v a
layer (0.74 MB) against the even third's 512 (0.59 MB) and a replicated 1536 (1.77 MB). Per
slice per layer 9.08 MB against the even 8.95 MB (+1.5%); the head 683 blocks of 2048 (+0.15%).

**Bit-exact.** Every fp32 sum stays the one-slice program's: a padded row is a zero row whose
dot is never read (the gathered tile is cut before use), the 4-head attention is the one-slice
kernel itself, W_o / gate / up / down follow P4's proof, the head's rows are the same dots at
other offsets. Measured: the tiny Gemma 3 at hidden 384 / MLP 768 at S = 3 equals S = 1 bit
for bit per position and resident (`tests/test_gemma3.py::test_three_slices_match_hf_and_one_slice`);
Gemma 3 1B at S = 3 on the ISA simulator gives S = 1's tokens on both compare_hf prompts with
the same logits-vs-HF statistics (below). S = 1, 2 and 4 take the old code paths (the even
splits) and are unchanged: the slice tests of test_gemma3.py pass as before.

`Spec.check` now requires `S in (1, 2, 3, 4)`, `hidden % S == 0`, `ffn % S == 0` and
`ple_dim % S == 0`; the head and vocabulary conditions are gone (`SLICES`, `split_rows`,
`split_ranges` in `opentpu/llm/gemma4.py`). The image per slice at a 2048-token capacity:
layers ~236 MiB + the head 292 MiB + KV 55 MiB (the head table replicated for the embedding
gathers); 4 GiB a slice available.

## Predicted tokens/s

Lens on the ISA simulator, the resident decode program, one Gemma 3 1B int8 token at position
14 after the chat template's 15-token prompt ("Why is the sky blue?"), `--isa --mhz 200`
(`build/lens/s{1,2,3,4}-gemma3-isa.otpuprof`; the analytic timing: every instruction back to
back for its nominal work per slice, GATHER = S x rows x ceil(cols / 8) cycles, no hub
latency or skew):

| | S = 1 | S = 2 | **S = 3** | S = 4 |
|:--|--:|--:|--:|--:|
| instructions a slice | 5,350 | 4,460 | 3,826 (156 GATHERs) | 3,820 |
| port B chunks a slice (MB) | 7,850,855 (1,005) | 3,941,479 (504.5) | 2,668,691 (341.6; slice 0 2,667,535) | 1,986,791 (254.3) |
| analytic cycles a token (the slowest slice) | 8,281,554 | 4,278,354 | **2,991,906** | 2,248,050 |
| of its DRAM bound | 94.8% | 92.1% | 89.2% | 88.4% |
| ms at 200 MHz, tok/s (ISA model) | 41.4, 24.2 | 21.4, 46.7 | **15.0, 66.8** | 11.2, 89.0 |
| RTL, measured (Verilator, the native memory model never withholding a handshake, `--stall 0`, `OTPU_LAT=30`) | 8,029,592 (97.8% of the bound; 24.9 tok/s) | | **2,993,389** (89.2%; 15.0 ms, 66.8 tok/s; 12 min of Verilator) | |
| hardware, measured (build 6, two ports at 89% utilization) | 9.00 M (22.2 tok/s) | | | |
| **hardware prediction** (the stream at the fabric's 86-89% port utilization, the rest as the RTL has it; below) | 22.2 | 42-43 | **58-60** | 77-80 |

Reading it. The ISA model's S = 3 token is 2.77x shorter than S = 1's (not 3x: the uneven
split's padded rows, the replicated norms / attention / quantizer work, 156 gathers). **The
same token on the Verilator RTL** (`OTPU_LAT=30 lens record model --model gemma3 --slices 3
--axi --stall 0 --mhz 200`, `build/lens/s3-gemma3-rtl.otpuprof`: otpu_top with three slices,
the collective hub and a port per slice as the board has them, two native channels a slice
through the AXI adapter and a memory that never withholds a handshake) takes **2,993,389
cycles**, 0.05% over the ISA model and 89.2% of its bound, and picks the same next token
('Okay'); its Lens trace: port B streaming 89.2%, "MXU not consuming: row's scales not all in"
3.0%, MXU computing not streaming 2.4%, waiting on VPU work 2.0%, on the quantizer 2.0%, TMEM
arbitration 0.5%, **waiting on collectives 0.4%** (12 K cycles: the hub's one beat per ~7-cycle
round trip, docs/bcu1525.md 10, mostly hidden behind the MXU's 1024-chunk prefetch; the
pipelined hub of P4 section 2 is not needed for this split). So the RTL's overlap gains (3%
at S = 1) are spent on the collective and the replicated vector work at S = 3, and the ISA
model is the RTL's number here.

On hardware the one-slice token takes 9.00 M cycles against the RTL's 8.03 M: the two-port
fabric streams 89% of the port rate (build 6's Lens counters), so the streaming part of the
token (the bound, 7.85 M) takes 7.85 / 0.89 = 8.82 M and the rest (0.18 M) stays. The same
on S = 3: 2.669 M / 0.89 = 3.00 M + (2.993 - 2.669 = 0.324 M) = **3.32 M cycles = 16.6 ms =
60 tok/s**; at the fabric's measured low of 86% port utilization 3.43 M = **58 tok/s**:
**2.6-2.7x the one-slice image's 22.2 tok/s**, with three slices' skew (the sequencers run free
between the 156 rendezvous; identical row counts keep it to the HBM's jitter) and the native
register slices' latency (two hops for slice 2, covered by the chunk FIFO) as the unmodelled
rest. S = 4 by the same method: 77-80 tok/s (its hardware would be P4's four port groups, two
slices in one SLR).

Also recorded: `tools/compare_hf.py --model gemma3 --chat --cfg board --resident --tokens 12
--slices 3` gives the S = 1 run's tokens on both prompts ("Okay, let's break down why the sky
is blue"; the haiku identical to HF for 12 tokens) and the same statistics, logits vs HF max
|diff| 2.269 / 3.455, min cosine 0.99869 / 0.99872 (the S = 1 and S = 4 values of p4-status.md).

## The hardware

### Port map

Window j = 2 s + c (slice s's channel c) is HBM byte range [j x 2 GiB, (j + 1) x 2 GiB) on
HBM AXI port `hbm_ports[j]`; the slice's `otpu_native_axi` adapter emits absolute addresses
(`BASE_ADDR` = j x 2 GiB) and its SmartConnect's address map routes each native master to
its own window and the host to both of the slice's. Ports chosen as Alex's 0 and 4 were:
port 4k heads the 2 GiB window k x 2 GiB it serves (the HBM's 32 pseudo-channels of 512 MiB
are consecutive in the address space; ports 2k and 2k + 1 share memory channel k, so the six
ports below are six distinct channels; ports 0-15 are rack 0, 16-31 rack 1 in
`cl_mem_hbm_wrapper`, which maps its port gg to the cl_hbm IP's AXI port gg). The port-to-
pseudo-channel addressing is the cl_hbm IP's configuration (`cl_hbm.xci` lives in the HDK on
the instance, not here): check `report_property [get_ips cl_hbm]` in the ip stage's log
(create_f2_ip.tcl prints it) before relying on the pairing; any six distinct ports work
functionally through the IP's switch, the pairing only shortens the switch path.

| slice | SLR | windows (HBM address) | ports | rack |
|:--|:--|:--|:--|:--|
| 0 | 0 | 0: 0-2 GiB, 1: 2-4 GiB | 0, 4 | 0 |
| 1 | 1 | 2: 4-6 GiB, 3: 6-8 GiB | 8, 12 | 0 |
| 2 | 2 | 4: 8-10 GiB, 5: 10-12 GiB | 16, 20 | 1 |

`cl_mem_hbm_wrapper #(NUM_OF_AXI_PORTS 21, AXI4_INTERFACE 0)`: ports 0..20, the fifteen
unused ones tied low (as Alex's 1-3), `otpu_hbm_geometry` on every port's addresses, both
racks' APB monitors wired as in his wrapper (rack 1 is used, so its monitor must be:
hbm-integration.md). 12 GiB of the 16 GiB HBM reachable; the host sees window j at BAR4
offset j x 2 GiB (`opentpu/host/f2.py` now accepts S slices of two 2 GiB windows, S <= 4;
`CFG_DRAM` / `CFG_SLICES` tell it the geometry). The `calib` bit of every channel is the
wrapper's `o_hbm_ready` (both racks initialized), through the core-clock synchronizer.

### The CL top, generated

`boards/f2/burst200/scripts/generate_f2_top.py --target targets/f2-burst200-s3.toml --hdk
<hdk> --output rtl/boards/f2/burst200-s3/cl_opentpu.sv` writes the checked-in top (3,175
lines). The generator takes S and `hbm_ports` from the toml; with the one-slice target its
output is Alex's `rtl/boards/f2/burst200/cl_opentpu.sv` byte for byte (the three `otpu_board`
parameters added since his base, NCH / CH_LOG2 / DRAM_LOG2, are left out at their defaults
as his wrapper has them; `tests/test_f2_burst200_s3.py::test_one_slice_wrapper_regenerates_byte_identical`).

```
 SLR2  slice 2 (u_board/g_slice[2]) -- u_native4/5 --> u_npipe4/5_slr1 (SLR1) --> u_npipe4/5_slr0 (SLR0) --> u_mem2 (SLR0)
 SLR1  slice 1 (u_board/g_slice[1]) -- u_native2/3 --> u_npipe2/3_slr0 (SLR0) -----------------------------> u_mem1 (SLR0)
       u_board/u_ctrl, u_iso, g_trace, g_coll.u_coll, g_xs.u_chub (the collective hub), u_ocl_cdc (OCL)
       PCIS --> u_pcis_pipe (f2_pcis_pipe) --> u_host (f2_host: 1 x 3 SmartConnect, 4 GiB a slice)
 SLR0  slice 0 (u_board/u_slice, u_mem) -- u_native0/1 -----------------------------------------------------> u_mem0 (SLR0)
       u_host_pipe0/1/2 (f2_pcis_pipe, the host's hop into SLR0) --> u_mem0/1/2's S_HOST
       u_mem0/1/2 (f2_mem_s0/1/2: 3 SI x 2 MI, Alex's, at the slice's windows) --> u_hbm (cl_mem_hbm_wrapper, 21 ports)
```

- **One `otpu_board`** with `S = 3`: it already multiplexes the slices behind one control port
  (RUN / LOAD / the arguments to every slice, HALTED when all halted, ICOUNT_s per slice) and
  at S > 1 puts two `otpu_xr` register stages each way between a slice and the control block /
  the collective hub (`g_xs`, as the BCU1525's two-SLR build; `otpu_coll_hub` and an
  `otpu_coll_port` per slice). Nothing in the slices changes.
- **Native path across SLRs**: a slice's two `otpu_native_axi` adapters sit in its SLR (the
  native handshake with `otpu_native_dram` is combinational); their AXI4 masters cross to
  SLR0 through `f2_native_pipe` (`axi_register_slice`, 512-bit, 34-bit addresses, 1-bit ID,
  every channel registered), one per hop, placed in the hop's destination SLR
  (`u_npipe<j>_slr<k>`): slice 1 one hop, slice 2 two. Slice 0's go straight into its
  SmartConnect as Alex's do.
- **Host path**: the PCIS enters in SLR1 (the small shell's PCIS is there) through Alex's
  `f2_pcis_pipe`, fans out in the `f2_host` SmartConnect crossbar (1 SI x 3 MI on the host
  clock alone, master s at HBM address [4 s GiB, 4 (s + 1) GiB), the PCIS's 16-bit IDs and
  64-bit addresses on its slave side, the grandchild pblock away from the shell) and reaches
  each slice's SmartConnect in SLR0 through another `f2_pcis_pipe` (`u_host_pipe<s>`). The
  crossbar is the HDK example's pattern (`cl_axi_sc_2x2_wrapper` in `cl_dma_pcis_slv.sv`),
  not a 7 x 6 SmartConnect. Its generated master ports carry no ID signals (SmartConnect's
  downstream ports are one ordered thread, as Alex noted for `f2_mem`'s M_H*; build 7 failed
  elaboration on `M_S0_arid`): the host pipes take `awid` / `arid` 16'd0 and their response
  IDs go nowhere, so each slice's `S_HOST` (16-bit IDs, Alex's `f2_mem` unchanged) sees one
  thread. Transactions to one slice stay ordered through the crossbar; the host's accesses to
  different slices are independent windows.
- **Parameter literals**: a parameter value of 2^31 or more is emitted as a sized 32-bit
  literal (`.ARCH_HASH(32'd2167823793)`): Vivado reads an unsized decimal as a signed 32-bit
  integer and truncates it (Synth 8-11057 in build 7), which would have left the image with a
  wrong CFG_HASH; the one-slice top's values are all below 2^31, so it is unchanged
  (`test_no_unsized_literal_from_2_31_up`).
- **Resets per SLR**: SLR1 keeps Alex's `core_rst_n` / `rst_main_n_sync` (the control block,
  the OCL converter, the PCIS pipe, the crossbar); SLR0 and SLR2 get their own
  `xpm_cdc_async_rst` and `lib_pipe #(.WIDTH(1), .STAGES(4))` in each clock they use
  (`u_rst_core_slr0/2`, `u_rstpipe_core_slr0/2`, `u_rst_main_slr0`, `u_rstpipe_main_slr0`;
  the HDK example's `CDC_ASYNC_RST_N_SLR*` / `SLR*_PIPE_RST_N`); the slices' own resets come
  through the board's `otpu_xr` stages. No reset net crosses an SLR.
- **Clocks**: Alex's two clk_wiz (200 / 400 from clk_main_a0, 450 from clk_hbm_ref); every
  openTPU block, the register slices and the SmartConnects' native side at 200 MHz, the 450
  MHz side inside the SmartConnects as before.
- **Status**: `cl_sh_status2` carries the six channel errors (bits 5:0), the vLEDs as before.

### Block designs and IP (`create_f2_ip.tcl`, `create_pcis_ip.tcl`, `build_f2.tcl`)

`build.py` writes `geometry.tcl` (`set otpu_S 3; set otpu_ports {0 4 8 12 16 20}`) into the
ip, pcis and synth_link directories. The ip stage makes one `f2_mem_s<s>` block design per
slice (`otpu_mem_bd`: Alex's 3 SI x 2 MI SmartConnect with the windows' absolute addresses
and the exclusions) plus `f2_host`; the pcis stage adds the `f2_native_pipe` register slice;
the link stage reads them all. Without geometry.tcl the scripts make Alex's `f2_mem` on
ports 0 and 4, so the one-slice build is the same build.

### Floorplan (`constraints/floorplan_s3.tcl`)

One pblock per SLR, children of the shell's `pblock_CL`, the HDK example's legal ranges
(`hdk/cl/examples/cl_dram_hbm_dma/build/constraints/small_shell_cl_pnr_user.xdc`; its SLR0
and SLR1 ranges are the ones Alex copied into floorplan.tcl, SLR2 is
`CLOCKREGION_X0Y8:CLOCKREGION_X7Y11`), `IS_SOFT false`, `SNAPPING_MODE ON` on SLR1, the
grandchild `pblock_CL_SLR1_XBAR` (`X0Y4:X4Y7`) for the host crossbar:

| pblock | cells |
|:--|:--|
| `pblock_CL_SLR0` | `u_board/u_slice`, `u_board/u_mem`, slice 0's `g_xs.g_x[0]` rx / tx / port stages and the activity bundle's tx, `u_native0/1`, `u_npipe2/3/4/5_slr0`, `u_host_pipe0/1/2`, `u_mem0/1/2`, `u_hbm`, the SLR0 resets |
| `pblock_CL_SLR1` | `g_slice[1].u_slice / u_mem`, `g_x[1]`'s four stages and port, the hub-side stages of slices 0 and 2 (`u_dn_tx`, `u_up_rx`, `u_pf_rx`), `g_xs.u_chub`, `u_ctrl`, `u_iso`, `g_trace.u_trace`, `g_coll.u_coll`, `u_native2/3`, `u_npipe4/5_slr1`, `u_pcis_pipe`, `u_ocl_cdc`, `u_status_cdc` |
| `pblock_CL_SLR1_XBAR` | `u_host` |
| `pblock_CL_SLR2` | `g_slice[2].u_slice / u_mem`, `g_x[2]`'s rx / tx / port, `u_native4/5`, the SLR2 resets |

`USER_SLL_REG` on both `otpu_xr` stages of slices 0 and 2 (the crossings), as
`boards/bcu1525/constraints/bcu1525_floorplan.tcl` does. Every cell lookup fails closed (a
renamed instance stops the build); `tests/test_f2_burst200_s3.py::test_floorplan_names_every_cell_of_the_top`
checks the floorplan against the generated top. The clock wizards, the SLR1 reset
synchronizers and the per-port `otpu_hbm_geometry` wires are left to the placer.

### Build

On the instance, from the repository root with the venv's Python and Vivado 2025.2 on PATH:

```bash
python boards/f2/burst200/build.py --hdk /home/ubuntu/aws-fpga/hdk --target targets/f2-burst200-s3.toml --plan
python boards/f2/burst200/build.py --hdk /home/ubuntu/aws-fpga/hdk --target targets/f2-burst200-s3.toml \
    --output /home/ubuntu/rachit/f2burst200-s3-run1
```

The stages, the gates and the packaging are the one-slice build's ([f2-burst200.md](f2-burst200.md)).
The one-slice build is `build.py --hdk ...` as before (the default target).

Then: `OTPU_DEVICE=f2:0 python -m opentpu.host.chat --backend board --dev f2:0 --model gemma3
--slices 3 --prompt ...` (the host transport reads the geometry from CFG_DRAM / CFG_SLICES).

## Verification (this branch, on the Mac)

- `generate_f2_top.py` with the one-slice target reproduces `rtl/boards/f2/burst200/cl_opentpu.sv`
  byte for byte; with the three-slice target the checked-in `rtl/boards/f2/burst200-s3/cl_opentpu.sv`.
- `pytest tests/test_target.py tests/test_f2_build.py tests/test_f2_scoped_clocks.py -q`: 183 passed.
- Both generated tops elaborate under Verilator 5.048 against the openTPU RTL, the HDK's
  interface headers and `lib_pipe`, and stubs of the vendor IP with the right port directions
  (`test_generated_top_lints_under_verilator[s1,s3]`: no error, no width truncation in the
  wiring).
- Alex's `tb_native_axi` (the adapter is his, unchanged) under Verilator 5.048 (`--binary
  --timing`, BURST 16, OUT 32, seeds 1 / 42 / 2026 / 4700): PASS, 297 reads and 267 masked
  writes each; his `simulate.py` under Icarus: PASS, 16 trials.
- The tiny Gemma 3 at S = 3 on `tb_board` (otpu_board S = 3, the native memory model, target
  f2-burst200-s3, platform 3): logits bit-identical to the ISA simulator over three tokens
  (29,469 / 29,475 / 29,442 cycles, 268 instructions a slice), every slice's DRAM the ISA
  simulator's (`test_tiny_gemma3_three_slices_on_board_model`; `pytest
  tests/test_f2_burst200_s3.py -q`: 10 passed).
- Gemma 3 1B's token at S = 3 on the Verilator RTL, the bit-exact gate
  (`test_gemma3_1b_token_on_three_slice_rtl_is_bit_exact`: the chat prompt's 14 tokens on
  the ISA simulator, the step at position 14 on the RTL and on the ISA simulator from the
  same DRAM state): **logits equal bit for bit, every slice's DRAM equal, 2,993,389 cycles,
  next token 19058 'Okay'** (the Lens run above's cycles exactly; 10 min of Verilator).
- Gemma 3 1B at S = 3 on the ISA simulator: S = 1's tokens and HF statistics (above).

## Risks

- **SLR crossings.** Two slices' native masters cross into SLR0 (slice 2 through two hops) and
  the collective / control bundles cross SLR1 <-> SLR0 and SLR1 <-> SLR2: all registered,
  but the SLL count (Laguna) grows with each 512-bit AXI channel pair: ~1.2K signals per
  native hop x 6 hop-channels plus ~2 x (DW + UW) ~ 2 x 3.3K bits for the board's bundles
  per crossing slice. The VU47P has ~17K SLLs per boundary: within budget, but the placer
  may fight for them near the shell's columns.
- **The PCIS fan-out** adds a crossbar and a register slice to every host access (image
  load, `out[]` reads): latency only; the decode path is untouched. Build 6's worst path
  (+0.035 ns, the SmartConnect's host entry into the PCIS pipe) now has a crossbar beside it
  in SLR1: the grandchild pblock is there to keep it off the shell's columns.
- **The HBM wrapper's 21 ports**: its reset pipe fans into 21 ports (round 2's -0.9 ns was
  32 ports with our own fabric); the vendor pipe is placed in SLR0 by the pblock.
- **Three SmartConnects + the wrapper in SLR0** with slice 0: Alex's one SmartConnect was
  15K LUTs, the slice 99K, the wrapper 1K; three SmartConnects, two host pipes' worth of
  register slices and four native pipes add ~40K: ~175K LUTs of SLR0's 272K in `pblock_CL`
  (~64%), against build 6's 34%. Tight but placeable; if it fails, slice 0's TMEM box of the
  round-3 floorplan (pb_otpu_tmem) is the next lever.
- **The collective links**: the hub's one beat per ~7 cycles would be 8% of the token if
  every gather were exposed; the RTL trace says 0.4% (above), the rest hidden behind the
  MXU's prefetch. On hardware the two `otpu_xr` stages per crossing are the same cycles as in
  the RTL; Lens's "waiting on collectives" line on the hardware trace will say; the pipelined
  hub (P4 section 2) is the fix if it shows.
- **The port-to-channel pairing** is the IP's address map; verified only by the ip stage's
  `report_property` on the instance (functionally any distinct ports work through the switch).
- **Slice skew**: the three sequencers run free; the uneven head (682 / 683 / 683 blocks) and
  the padded attention rows are the same work on every slice to within 0.15%.
