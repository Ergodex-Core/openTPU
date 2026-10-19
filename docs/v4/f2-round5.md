# f2-hbm round 5: the HBM fabric at the core clock, the 450 MHz side vendor IP

Rounds 1-4 ([f2-timing.md](f2-timing.md)) fought the `f2-hbm` CL's 450 MHz domain (the HBM's
AXI ports' clock, hclk) from WNS -4.0 (build 1) to -2.2 (build 2) to -1.4 ns (build 3; build 4
pending on opentpu-f2 as unit `otpu-f2hbm-build4`), every time in openTPU RTL that ran on hclk:
the channel bridges' burst builders, the port arbiters, the host path's registers and links.
Alex's burst200 image (`opentpu_burst200_20261015`, the only openTPU F2 image that closed 200
MHz; its sources in tpubench's `build/alex-burst200/`) has **no openTPU RTL on hclk at all**:
his two channel adapters are AXI4 masters at the core clock, and one SmartConnect block
design, vendor IP, does the clock, width and protocol conversion into the HBM. Round 5 is the
same structure on our 8 channels, on branch `v4-f2-round5` (tpubench f449957a).

## The two fabrics

| | Alex's burst200 (runs 50 / 55 / 63 / 64 / 68) | ours, rounds 1-4 (builds 61d07199, a2af0904, 8b2fd3a0, build 4) |
|---|---|---|
| channels | 2 per slice, 2 GiB each (`channels_per_slice 2`, `channel_bytes 2 GiB`) on HBM ports 0 and 4 | 8 per slice, 512 MiB each, on ports 0-7 (`f2-hbm`); 16 and 32 ports for f2-hbm16 / :s4 |
| the native port's bridge | `otpu_native_axi` (160 lines, core clock): a 16-entry command queue and write queue in LUT RAM; adjacent reads coalesced into one AXI4 burst of up to 16 x 64 B (1 KiB; flushed when full, at a page / direction / discontinuity boundary, or after 4 idle cycles); writes single-beat; a **direction barrier** (a read is collected only while no write is unanswered, a write goes only while no read is outstanding: the native read-after-write contract); 128 beats of read credit; all reads ID 0, a lengths FIFO checking RLAST | `otpu_f2_hbm_nch` (400 lines): core-side command / data FIFOs, per-beat hazard tables (64 reads, 16 writes in flight, compared against the head), a 602-bit afifo into hclk, there a burst builder (joins consecutive beats into AXI3 bursts of up to 8 native beats, GATHER 4 hclk cycles), AR / AW queues, a 32-entry W FIFO, R pairing, a gray write count back |
| the host (PCIS, 250 MHz) | straight into the SmartConnect as its third slave (512-bit AXI4, 16-bit IDs, 64-bit addresses), behind an `axi_register_slice` IP (`f2_pcis_pipe`, placed in SLR1); the SmartConnect routes by address to the two ports and returns DECERR outside the map | `otpu_f2_hbm_host`: skid buffers and links at aclk, afifos into hclk, a one-beat-at-a-time master at every port through link stages and a register per port (`hlink`, `hreg`), the write count gray-coded back |
| the ports' sharing | the SmartConnect's crossbar (3 SI x 2 MI, `memory_sc`) | `otpu_f2_hbm_port` at hclk: a 2:1 AXI3 arbiter per port (bridge ID 0, host ID 1) |
| the 450 MHz side | **inside the SmartConnect**: per master port an exit pipeline (`m00_exit`: AW / AR / W / R `axi_reg_stall` register slices, the `sc_exit` with its `gen_axi3.splitter_inst` AXI3 converter: 740 LUTs each) and the per-channel nodes (`m00_aw/ar/w/r/b_node`: `xpm_memory_fifo` crossing FIFOs with the width conversion in the W and R nodes: ~1.5K LUTs per MI); and the HDK wrapper's reset pipe. The whole bd: 15,037 LUTs, 25,599 registers, 203 LUT RAM (3 SI x 2 MI, the host's `si_converter` 1,061 of them) | our bridges' hclk halves (3.3K LUTs each), the arbiters, the host path's hclk half and link stages, the HDK wrapper's reset pipe |
| clocks | `f2_core_clk` (clk_wiz from clk_main_a0: 200 and 400 MHz), `f2_hbm_clk` (clk_wiz from clk_hbm_ref: 450), clk_main_a0 250; resets `xpm_cdc_async_rst` per domain; the core and TMEM clocks' roots in one `CLOCK_DELAY_GROUP` | the same four clocks from two MMCME4_ADV (u_mmcm, u_hmmcm), hand-written synchronizers |
| CDC constraints | none of his own: the clock-interaction report shows the SmartConnect's (max delay, datapath only, between clk_main_a0 / core / hbm at the source clock's period: core -> hbm 5.00 ns with 3.74 of slack, hbm -> core 2.22 with 1.27) and the wrapper's xpm false paths; his `core_timing.tcl` is otpu_clk2x.tcl plus the delay group, `platform_constraints.tcl` the DDR clock pair's IOSTANDARD and `CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN` on the HBM reference | `cl_timing_user_hbm.tcl`: max delay by the destination period on every pair, bus skew on every afifo's gray pointers and every write count |
| floorplan | two pblocks: `pblock_CL_SLR0` (explicit ranges "copied from installed AWS cl_mem_perf constraints": whole clock regions X0Y0:X3Y3, X4Y0:X7Y0, X4Y3, X6Y1, X5Y1:X5Y3 and site ranges of the partial regions; IS_SOFT false) holding the board, both adapters, the SmartConnect and the HBM wrapper; `pblock_CL_SLR1` (SNAPPING_MODE ON) holding the PCIS register slice and the OCL converter | five pblocks (SLR0, the TMEM box, the fabric row, the PCIS registers in SLR1, the links' SLR0 ends), contained in the HDK's pblock_CL by `otpu_contain` |
| tuning | run 55: place Explore, phys AggressiveExplore, route Explore -> **-0.022 ns**; run 63 (`tuning.tcl`): place **ExtraNetDelay_high**, phys AggressiveExplore, `phys_opt_design -retime`, route Explore -> -0.033; run 64: `phys_opt_design -directive Explore` + `route_design -directive Explore -tns_cleanup` on 63's routed checkpoint -> **+0.032 / hold +0.010**, the packaged image. His worst paths at closure are all inside the slice: the TMEM's pumped RAM into the quantizer's DSP B input (4.33 of 5 ns), the pump's write stage (1.77 of 2.5), the MXU's `qr_reg` into `drow` enables (4.48 of 5); nothing in the fabric | SSI_SpreadLogic_high (build 1), Default (2-4), phys AggressiveExplore, route AggressiveExplore |
| area (SLR0) | 132,062 LUTs (34%), 116,214 registers, 261.5 RAMB36, 727 DSPs: the slice 99K, the adapter 11K (2 channels), the SmartConnect 15K, the adapters 2 x 487, the wrapper 1K | 175K LUTs for the slice, the 8-channel adapter (39K) and 8 bridges (26K) |
| bandwidth, the slice's demand | 2 x 512 bits x 200 MHz = **25.6 GB/s** at the native ports (the slice streams a 128-byte chunk a cycle: 25.6 GB/s); the HBM side 2 x 14.4 = 28.8 | 8 x 12.8 = 102 GB/s at the native ports, 8 x 14.4 = 115 at the HBM; the demand 25.6: **3.2 GB/s a channel** |
| bandwidth, measured | the direct MXU weight stream (run 71, 8 MiB per trial): **23.8 GB/s** (`counter_GBps 23.83`, useful 23.80) of the 25.6; the int8 Gemma decode: **22.1 GB/s** of DRAM per running second, 33.7 tok/s on the device (27.0 wall), against the single-beat image's 6.3 GB/s and 9.6 tok/s (run 74's comparison; both images' tokens identical over 65) | not built |

What the comparison says: Alex's image streams 86-93% of its native ports' 25.6 GB/s through a
vendor fabric that never appears among the worst paths, and closed with Vivado's own tuning on
a design whose timing-critical logic is the slice's. Our 450 MHz RTL was the only part of the
CL that failed, and it was there only to build bursts and share ports, which the SmartConnect
does.

## The plan, and what the branch does

Every openTPU block of the HBM fabric at the core clock (200 MHz); per port one vendor
SmartConnect (the HDK's own `cl_axi_sc_1x1` block design) for the crossing into 450 MHz, the
512 -> 256 width conversion and the AXI4 -> AXI3 conversion into the HBM wrapper's port:

```
   core clock (200 MHz)                                            | 450 MHz (vendor)         | HBM
 slice -- otpu_native_dram (NCH 8) --+-- ch 0: otpu_native_axi --+-- otpu_f2_hbm_port --> cl_axi_sc_1x1 --> wrapper port 0 (PC 0)
                                     +-- ch 1: otpu_native_axi --+-- otpu_f2_hbm_port --> cl_axi_sc_1x1 --> port 1
                                     ...                   host's |                                   
 PCIS (250 MHz) -- otpu_f2_hbm_host: skids, links, queues ==afifo==> per-port single-beat master -+ (master 1 of every port)
 OCL  (250 MHz) -- otpu_axil_cdc -- otpu_ctrl
```

- **The channel bridge is Alex's `otpu_native_axi`** (`rtl/boards/f2/otpu_native_axi.sv`),
  taken as it closed and ran on his card, with the channel's size a parameter (`CH_LOG2` 29: his
  BASE alignment was 2 GiB), the read return registered (`n_rvalid / n_rdata`: the
  SmartConnect's R exit register sits at the HBM port, the adapter's queues by the slice; the
  response is never stalled, so the stage costs one cycle of latency and no logic) and a
  `BARRIER` knob for the negative test. It fits the native protocol as is: his module was
  written against the same otpu_board port (`n_cvalid / n_cready / n_cwe / n_caddr`, `n_wvalid /
  n_wdata / n_wmask`, `n_rvalid / n_rdata`, `n_wdone`, `n_err`) and ran the same programs. Its
  direction barrier is coarser than our hazard tables (a read waits for every earlier write's
  B, a write for every earlier read's data), which costs a drain of a few hundred ns at each
  direction change: Alex's 22.1 GB/s on the Gemma decode (a token's ~3.7K writes per channel
  against ~2M reads, in batches) includes that cost. Our `otpu_f2_hbm_nch` is retired (git
  history keeps it).
- **The port arbiter at the core clock** (`otpu_f2_hbm_port`, rewritten): a 2:1 AXI4 mux of
  512 bits (the channel's master ID 0, the host's ID 1 in bit 0 of the SmartConnect's 16-bit
  ID; AW round-robin with W locked to the granted master until WLAST, AR round-robin, B and R
  routed back by ID), its AW / W / AR outputs registers (the SmartConnect's entry registers
  beside them).
- **The host path's back half at the core clock** (`otpu_f2_hbm_host`): the PCIS side (skid
  buffers, links, the queues and the order rule) is round 3-4's unchanged; the three afifos
  now land in the core clock, where the head beat goes to its port as one single-beat AXI4
  write of the whole 512 bits with the PCIS beat's strobes (a narrow PCIS write is a beat with
  the lanes it covers, the shell spec's unaligned examples; the address goes out 64-byte
  aligned, so the SmartConnect's downsizer sends two aligned 32-byte beats and no address
  geometry fix is needed: Alex's `otpu_hbm_geometry` existed because he passed the PCIS's
  unaligned narrow addresses straight into the SmartConnect, which keeps a single-beat narrow
  transaction's address), reads one beat at a time on one port, the write count gray-coded
  back. The link stages and the per-port registers of the 450 MHz design are gone: at 5 ns a
  flip-flop here into the arbiter's register beside its port is a core period of wire.
- **`otpu_f2_hbm_mem`** holds the masters, the arbiters and the host and leaves NPORT AXI4
  masters at the core clock; **`otpu_f2_hbm_sys`** no longer takes hclk.
- **The CL top** (`boards/f2/design/cl_opentpu_hbm.sv`) instantiates `cl_axi_sc_1x1_wrapper`
  per port (`g_sc[p].u_sc`: `aclk_250` driven by the 200 MHz core clock, `aresetn_250` the core
  reset, `aclk_450` hclk; the AXI4 side the port's master, the AXI3 side the HDK wrapper's port
  p with IDs 0) and `cl_mem_hbm_wrapper` as before (`NUM_OF_AXI_PORTS` NPORT, `AXI4_INTERFACE`
  0). The only openTPU flip-flops on hclk are the wrapper's reset synchronizer (`hbm_reset_n`).
  This is the block design cl_mem_perf's and cl_dram_hbm_dma's HBM paths use
  (`cl_mem_hbm_axi4.sv`: `cl_axi_sc_1x1_wrapper AXI_CONVERTER_AXI4_AXI3`, the PCIS format in at
  250 MHz, AXI3 256 bits out at 450), read in synthesis the way their scripts read it
  (`add_files ${HDK_BD_SRC_DIR}/cl_axi_sc_1x1/cl_axi_sc_1x1.bd`, `read_verilog
  ${HDK_BD_GEN_DIR}/cl_axi_sc_1x1/hdl/cl_axi_sc_1x1_wrapper.v`), so nothing is generated by
  build.py. Alex generated his own 3 x 2 bd (`f2_mem`) in a separate IP project; a 1 x 1 per
  port scales to 16 and 32 ports (a SmartConnect takes at most 16 slaves) and keeps each
  channel's path independent.
- **Constraints**: `cl_timing_user_hbm.tcl` bounds the remaining openTPU crossings (aclk <->
  core: the host's three afifos, its write count, the OCL crossing) and keeps the max-delay
  bounds on the clock pairs with hclk (the SmartConnect's own, more specific constraints take
  precedence; Alex's image timed the same crossings under such bounds with margin), and gates
  that **no openTPU cell is clocked by hclk** (every sequential cell on it is under `g_sc[*].u_sc`
  or `u_hbm`, or is `hbm_reset_n_reg*`; the synthesis run prints `OTPU_HBM_NO_HCLK_RTL`). The
  bridges' exceptions of rounds 1-4 (16 afifos, 8 gray counts) are gone with the bridges.
- **Floorplans, selectable** (`build.py --floorplan`): `five` (ours, updated: the fabric row
  `pb_otpu_fab` is now the whole bottom row X0Y0:X7Y0 holding the masters, the arbiters and the
  SmartConnects beside their ports; the TMEM box, the PCIS and link boxes as in round 4),
  `slr0` (Alex's two pblocks with his explicit ranges, `cl_pnr_user_hbm_slr0.tcl`: the board,
  the fabric, the SmartConnects and the wrapper in his SLR0 pblock, the PCIS registers and the
  OCL crossing in his SLR1 one), `none`.
- **Tuning** (`build.py`): `--place-directive`, `--phys-directive`, `--route-directive`,
  `--retime` (a `phys_opt_design -retime` after the HDK's post-place phys_opt, through
  build_level_1_cl.tcl's hook), and `--recommended` = Alex's set (ExtraNetDelay_high /
  AggressiveExplore / Explore / retime). The defaults stay the flow's earlier ones. The HDK's
  HBM `DONT_TOUCH` workaround (its build_level_1_cl.tcl after opt_design, on its instance;
  Alex's on his) is in our opt_design wrapper under our instance name. His other workarounds
  (`route.optimizeLeafClkProgDlys false` for a Vivado 2025.2 router crash with this shell, the
  bank-74 `INTERNAL_VREF 0.84` after placement) were not needed by our builds 1-4 and are not
  applied; if route_design crashes in the leaf-clock optimizer, that param is the first thing
  to set.

### Bandwidth per channel

A SmartConnect's slave side at 200 MHz x 512 bits is 12.8 GB/s per channel (64 bytes a core
cycle, the native port's own limit); the HBM side 14.4. Alex's 2-slave SmartConnect carried
23.8 of 25.6 GB/s on the weight stream (93%), so ~11.9 GB/s per channel is what to expect
through one. The 128 beats of read credit in `otpu_native_axi` sustain 128 x 64 B over ~400 ns
of fabric-plus-HBM latency = 20 GB/s, above the port's width. Against the demand:

| | per channel needed | of a channel's 12.8 GB/s | the 8 (16) SmartConnects' area, estimated |
|---|---|---|---|
| f2-hbm today (128 B/cycle, NCH 8) | 3.2 GB/s | 25% | 8 x ~5K LUTs (a 1 x 1 is Alex's 15K bd less the crossbar and the host's converter: read it from the first synthesis) |
| P1, the wide port (256 B/cycle, NCH 8) | 6.4 GB/s | 50% | the same |
| P2, four lanes (512 B/cycle, NCH 16) | 6.4 GB/s | 50% | 16 x ~5K: with the NCH 16 adapter's 79K the SLR0 box (272K) overflows; P2's floorplan must spill the slice into SLR1 (the SmartConnects stay by the ports) |

So the per-port vendor path leaves P1 half a channel of headroom and serves P2 at the same
per-channel rate; neither needs anything on hclk.

### What could not be seen here

- **The `cl_axi_sc_1x1` block design itself** is not on this machine (the HDK checkout here,
  `/Users/rachittibrewal/ergodex/aws-fpga`, has `hdk/common/ip` unpopulated; a copy of the HDK's
  `cl_axi_sc_2x2.bd` under a Codex download gave the schema). Its interface comes from its use
  in `cl_mem_hbm_axi4.sv` (the PCIS-format AXI4 slave: 512 bits, 64-bit addresses, 16-bit IDs,
  at `aclk_250`; an AXI3 master of 256 bits, 64-bit addresses, no IDs, at `aclk_450`; one reset
  `aresetn_250`) and from `axi_bus_t`'s defaults. `build.py` reads the bd's JSON at prepare
  time and fails closed if the interfaces, widths, ID widths or clock pins differ
  (`check_hbm_sc`); the generated wrapper's pins are checked the same way. Its outstanding
  counts (`NUM_READ_OUTSTANDING`: the 2x2's slaves allow 32-64) are not checked: a lower count
  would throttle the master's 128-beat credit to N bursts of up to 1 KiB in flight.
- **Its slave clock pin is named `aclk_250` and the bd records 250 MHz**; the CL drives it at
  200. The crossing to 450 MHz is asynchronous either way (250 and 450 are unrelated, so the
  generated converter is the async one), and a bd port's `FREQ_HZ` is IP-integrator metadata,
  but a `CRITICAL WARNING` about the frequency at synthesis would be new; Alex's `f2_mem` was
  generated with its native slaves at 200 MHz, so he never had this question.
- **Alex's `f2_mem.bd` definition** (his IP project, `$iprun`) is not in the snapshot; its
  structure above is read from the routed utilization hierarchy and his testbench
  (`tb_f2_mem.sv`, which simulates the bd in xsim with his `axi_hbm_model`).
- **The SmartConnect's behaviour on two IDs per slave** (the channel's 0, the host's 1) with an
  ID-less master: it keeps one slave's responses in issue order and reflects the IDs (Alex's
  testbench checked the host's 16-bit IDs through his); it may hold a transaction of a new ID
  until the previous ID's are complete, which serializes a host access against the channel's
  outstanding reads (harmless: the host is idle during a run).
- The vendor IP has **no Verilator model**: `tb_f2hbm` drives the new boundary through
  `sim/verilator/tb_f2hbm_sc.sv`, a behavioural stand-in (the crossing, the 512 -> 256 split
  with the low half first, the AXI3 split at 16 beats with WLAST per piece and one B / one
  RLAST per AXI4 transaction, in-order responses with the request's ID, random readies), and
  the HBM model's responses were registered (a continuous assignment from a SystemVerilog
  queue raced the blocks popping it: the master's `always_ff` and a probe block in the same
  cycle saw different beats). What the stand-in does not reproduce: the IP's latency,
  buffering and outstanding limits.

## Verification

Verilator 5.048, one run at a time (the tree's `build/verilator` cleared between suites):

- `tests/test_f2_rtl.py`: **14 passed** (tb_f2hbm at 4 / 8 / 16 / 32 ports over the
  SmartConnect stand-ins and the reordering HBM model with the three clock ratios: every
  channel's native contract, every port's traffic inside its pseudo-channel and in 4 KiB,
  runs of reads as bursts (the `longs * 20 < bursts` check), the host's writes, reads, B's and
  R's in order with its IDs, zeros past the last port; `BARRIER 0` returns stale data; the DDR
  protocol and ordering, the OCL crossing, the PCIS structural test).
- `tests/test_native_nch.py`: see the hand-back (the adapter is untouched by this round).
- `tests/test_f2_build.py`, `tests/test_f2_scoped_clocks.py`, `tests/test_target.py`: **183
  passed** (the HBM snapshot with the SmartConnect block design in the synthesis script and the
  CL top; the three floorplans, the five-pblock one on the new cells and its containment, Alex's
  under the same Tcl mocks; `check_hbm_sc` on a bd of the HDK's schema, failing closed; the
  link wrapper's retime hook and the DONT_TOUCH workaround).
- **Gate 4** ([f2-port-status.md](f2-port-status.md)): see the hand-back (`tb_top`'s board
  model does not include the fabric, so the token's cycles and logits are this branch's
  core, which is untouched).

## The build (opentpu-f2, i-0c78de2253f66519d)

The tree is `/home/ubuntu/rachit/src/tpubench` (editable in `/home/ubuntu/opentpu-venv`), the
HDK `/home/ubuntu/aws-fpga` (its `hdk/common/ip` with `cl_axi_sc_1x1` generated by
`hdk_setup.sh`), Vivado by `source /opt/Xilinx/2025.2/Vivado/settings64.sh`. With
`v4-f2-round5` merged:

```
cd /home/ubuntu/rachit/src/tpubench
source /opt/Xilinx/2025.2/Vivado/settings64.sh
# synthesis first: the SmartConnects' size, OTPU_HBM_NO_HCLK_RTL and OTPU_HBM_CDC in console.log
/home/ubuntu/opentpu-venv/bin/python boards/f2/build.py --hdk /home/ubuntu/aws-fpga \
    --target f2-hbm --out /home/ubuntu/rachit/runs/f2hbm-r5-synth --flow SynthCL
# the full build with Alex's tuning (= --place-directive ExtraNetDelay_high
# --phys-directive AggressiveExplore --route-directive Explore --retime) and our floorplan
/home/ubuntu/opentpu-venv/bin/python boards/f2/build.py --hdk /home/ubuntu/aws-fpga \
    --target f2-hbm --out /home/ubuntu/rachit/runs/f2hbm-r5 --flow BuildAll --package \
    --recommended --floorplan five
# the alternative: Alex's one-SLR0 floorplan
/home/ubuntu/opentpu-venv/bin/python boards/f2/build.py --hdk /home/ubuntu/aws-fpga \
    --target f2-hbm --out /home/ubuntu/rachit/runs/f2hbm-r5-slr0 --flow BuildAll --package \
    --recommended --floorplan slr0
```

If the routed WNS is within ~-0.05 ns, Alex's run 64 is the recipe: `boards/f2/route_opt.py`
(phys_opt Explore on the routed checkpoint, then route_design Explore -tns_cleanup) took his
-0.033 to +0.032. What to read first in the post-route report: any path with a `g_sc[*].u_sc`
cell at either end (the vendor fabric: Alex's never showed), then the slice's own, which are
Alex's worst paths too (the TMEM pump into the quantizer's DSPs, the MXU's `qr_reg` into
`drow` enables) and which the TMEM box of the five-pblock floorplan is for.
