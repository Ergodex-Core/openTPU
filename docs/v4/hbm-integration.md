# HBM integration on the F2: what works, from three sources

The three designs that reached a routed VU47P with openTPU on HBM, and the HDK's own example,
agree on one structure. This note fixes it as the rule for every F2 HBM target from here on
(the one-slice `f2-burst200` built today, the three-slice `f2-burst200-s3`, and whatever
carries P1 / P2 / P4 to silicon).

Sources: Alex's burst200 (`boards/f2/burst200/`, closed at +0.032 ns on his slice and +0.035
ns on ours, build 6), the HDK example `hdk/cl/examples/cl_dram_hbm_dma` (github.com/aws/aws-fpga
tree `f2`; its small-shell floorplan is copied verbatim to
[ref/hdk_cl_dram_hbm_dma_small_shell_cl_pnr_user.xdc](ref/hdk_cl_dram_hbm_dma_small_shell_cl_pnr_user.xdc)),
and our rounds 1-5 ([f2-timing.md](f2-timing.md), [f2-round5.md](f2-round5.md)) as the
negative result.

## The rules

1. **No openTPU logic on the HBM clock.** Every openTPU-side block runs at the core clock
   (200 MHz). The 450 MHz side (clock crossing, 512 -> 256 width conversion, AXI4 -> AXI3,
   the HBM ports' handshakes) lives inside vendor IP only: one SmartConnect per port group
   (Alex's `f2_mem`: 3 SI x 2 MI; the HDK's `cl_axi_sc_1x1` per port) into
   `cl_mem_hbm_wrapper #(NUM_OF_AXI_PORTS = highest port + 1, AXI4_INTERFACE = 0)`, unused
   ports tied off, `apb_clk = clk_hbm_ref`. Rounds 1-4 put burst builders, arbiters and host
   registers on hclk and never got past WNS -1.4 ns; build 4 did not even route.
2. **A read coalescer per native channel at the core clock** (`otpu_native_axi`: 16-entry
   queues, reads joined into AXI4 bursts of up to 16 x 64 B, single-beat writes, a direction
   barrier for the native read-after-write contract, 128 beats of read credit). This is what
   turns the slice's one-beat-per-command stream into bursts the SmartConnect and the HBM
   controller run at 86-93% of the port rate (23.8 of 25.6 GB/s measured).
3. **Ports, not shells, set the bandwidth.** A slice consumes 128 B a cycle = 25.6 GB/s; two
   HBM ports (2 x 14.4 GB/s) feed it at 89% utilization. More bandwidth means more native
   width or more slices, each on its own port pair (S = 3: six ports, 76.8 GB/s raw); eight
   ports behind one 128 B/cycle slice (round 5) buy headroom only.
4. **One pblock per SLR, with the HDK's legal ranges** (SLR0: the example's and Alex's ranges;
   SLR1: `CLOCKREGION_X0Y4:X3Y7`, `X5Y4:X5Y7`, `X4Y6:X4Y7`, `X4Y4` plus the partial ranges,
   `SNAPPING_MODE ON`; SLR2: `CLOCKREGION_X0Y8:CLOCKREGION_X7Y11`), all children of the
   shell's `pblock_CL`. The HBM wrapper and its SmartConnects go in SLR0 (the ports are at the
   bottom of the die); the PCIS pipe and OCL converter in SLR1 next to the shell interface;
   a crossbar, if any, in a grandchild pblock (`pblock_CL_SLR1_XBAR`, `X0Y4:X4Y7`) away from
   the shell to avoid congestion. `IS_SOFT false` on the slice pblocks.
5. **Every SLR hop is an `axi_register_slice` placed in the destination SLR**
   (`AXI4_REG_SLC_PCIS_SLR1/2`, `AXI4_REG_SLC_DDRA_SLR1/2` ... in the example; Alex's
   `f2_pcis_pipe`). Fan-out of the PCIS to several memory windows is a SmartConnect crossbar
   (`cl_axi_sc_2x2_wrapper` in the example) in SLR1, then a register slice per destination
   SLR. Never a combinational path across an SLR.
6. **Resets per SLR**: an `xpm_cdc_async_rst` per clock domain and per SLR
   (`CDC_ASYNC_RST_N_SLR0/1/2`), each followed by a `lib_pipe #(.WIDTH(1), .STAGES(4))`
   placed in its SLR; the HBM wrapper's own reset pipe stays the vendor's. Our round-2
   lesson (the reset pipe fanning into 32 ports at -0.9 ns) is the same point.
7. **Clocks**: `clk_wiz` from clk_main_a0 for the related 200 / 400 MHz core and TMEM
   clocks (zero phase, `CLOCK_DELAY_GROUP`, otpu_clk2x.tcl's multicycles), a second
   `clk_wiz` from the 100 MHz clk_hbm_ref for the 450 MHz HBM AXI clock; no CDC constraints
   of our own beyond the SmartConnect's (datapath-only max delays at the source period) and
   bus-skew on our own afifos.
8. **The host path** goes straight into a SmartConnect slave (512-bit AXI4, full IDs and
   64-bit addresses) behind one register slice; 64-byte-aligned beats with strobes, so no
   address-geometry fix is needed (Alex's `otpu_hbm_geometry` existed only because his PCIS
   went in unaligned).
9. **Tuning that closed**: place `Explore` (or `ExtraNetDelay_high`), phys_opt
   `AggressiveExplore` (+ `-retime`), route `Explore -tns_cleanup`, the leaf-clock optimizer
   off (`route.optimizeLeafClkProgDlys false`, a Vivado 2025.2 crash workaround), bank 74 Vref
   restored to 0.84 V after placement, then a post-route phys_opt + reroute pass
   (`refine_route.tcl`) that lifted -0.02 to +0.03 ns. Keep `-keep_equivalent_registers`
   and `-flatten_hierarchy rebuilt` at synthesis.
10. **Gates before packaging**: setup / hold / pulse width >= 0 with 0 failing endpoints,
    0 nets with routing errors, DRC 0 errors / 0 critical, every CDC critical under an
    exception on shell or xpm-reset cells only, bus skew 0 violated, 0 unconstrained
    internal endpoints, the no_clock pins all in the unchanged shell. A review JSON binding
    the checkpoint's SHA-256 to those conclusions is what `package_f2.py` takes.

## The shell's own placement (AWS Shell Interface Specification, "Multi-SLR FPGA")

The F2 docs (awsdocs-fpga-f2.readthedocs-hosted.com, HDK > AWS Shell Interface Specification,
Implementation Tips > Multi-SLR FPGA) say where the small shell's interfaces land on the
VU47P's three dies, and the rule that follows: "place logic that talks to the Shell in the
same SLR as the Shell logic for that interface; if that is not possible, put the first flop or
register slice in that SLR", flops on both sides of every SLR crossing or AXI register slices,
one pblock per SLR with the logic assigned (the `cl_dram_hbm_dma` floorplan above is the
reference they name), and dataflow organized so pipeline stages stay inside one SLR.

| SLR | shell interfaces there |
|---|---|
| TOP (SLR2) | PCIS (the DMA data path), CL_SH_DDR, DDR stats |
| MID (SLR1) | OCL, PCIM, SDA |
| MID / BOTTOM (SLR1 / SLR0) | HBM monitor APB |

The Shell Floorplan page's figure ([ref/aws_small_shell_fp.png](ref/aws_small_shell_fp.png),
AWS's `small_shell_fp.png`) shows the die: the SMALL_SHELL block occupies the right two
columns of clock regions in SLR1 (X6-X7, Y4-Y7; DDR_STAT at its top, then PCIS, PCIM, and
OCL / SDA at its bottom edge near Y4), which is exactly what `pblock_CL_SLR1`'s ranges
(X0Y4:X3Y7, X5Y4:X5Y7, X4Y6:X4Y7, X4Y4) leave out; SH_DDR sits at the top of SLR2 (X0-X3,
Y10-Y11); the two **High Bandwidth Memory** stacks are the bottom edge of SLR0 (the left
stack under X0-X3, rack 0 = AXI ports 0-15; the right stack under X4-X7, rack 1 = ports
16-31), so every HBM port is reached from SLR0's bottom row, and `pblock_CL_SLR0` spans the
whole die width at Y0 for that reason (X4Y0:X7Y0 in its ranges). The CL's `pblock_CL` is the
union; the shell's `pblock_WRAPPER` wraps it.

Consequences for our designs:

- **The PCIS enters in SLR1 on the SMALL_SHELL.** The spec's table is the generic one
  (PCIS in the top SLR on the XDMA shell), but the Shell Floorplan page states for the small
  shell: "The PCIS bus is physically located in SLR1 in SMALL_SHELL. Any customer logic
  interfacing with this bus should reside in SLR1." The HDK example's small-shell floorplan
  agrees (`AXI4_REG_SLC_PCIS_SLR1` and the crossbar in SLR1; its SLR2 pblock holds only the
  DDR side). Alex's `f2_pcis_pipe` in SLR1 is therefore right; build 6's worst setup path
  (+0.035 ns, the SmartConnect's host entry into that pipe) is a same-SLR path to tune, not
  a placement error. In a three-slice design the host path is the PCIS register slice and
  the fan-out crossbar in SLR1 (the `pblock_CL_SLR1_XBAR` grandchild), then one register
  slice per destination SLR (SLR0 and SLR2 one hop each).
- **OCL (our control registers) stays in SLR1**: Alex's `f2_ocl_cdc` is already there; the
  one `otpu_board` control port is reached from SLR1, so the board's control fan-out to
  slices in SLR0 and SLR2 needs a registered stage per hop (the board's control path is at
  the core clock; check `otpu_board`'s `s_ctl_*` into each slice).
- **HBM monitor APB**: `hbm_apb_preset_n_0/1` are asynchronous resets for the HBM IP and
  must connect straight through to `MON_APB_0/1_PRESET_N` (no pipelining); the other APB
  signals may take APB register slices; both racks' monitor interfaces must be connected
  when their HBM is used (an unconnected monitor on a used rack fails AFI creation); an
  unused rack may be left unconnected. Alex's wrapper instantiation connects both; keep it.
- **Clocks**: `clk_hbm_ref` is 100 MHz and the HBM monitor is synchronous to it; our 450 MHz
  HBM AXI clock is our own `clk_wiz` from it (rule 7).

## Where the HBM AXI ports are on the die (VU47P, measured 2026-10-20)

`link_design -part xcvu47p-fsvh2892-2-e` and `get_sites -filter {SITE_TYPE =~ *HBM*}` (no
design needed): the 32 AXI port interfaces are the sites `BLI_HBM_AXI_INTF_X<k>Y0`, port k at
X = k, all in clock-region row Y0 (SLR0's bottom edge), in these clock regions:

| ports | clock region | | ports | clock region |
|---|---|---|---|---|
| 0-3 | X0Y0 | | 16-19 | X4Y0 |
| 4-7 | X1Y0 | | 20-24 | X5Y0 |
| 8-12 | X2Y0 | | 25-29 | X6Y0 |
| 13-15 | X3Y0 | | 30-31 | X7Y0 |

(The APB interfaces `BLI_HBM_APB_INTF_X<k>Y0` follow the same map; `HBM_REF_CLK_X0Y0/1` sit in
X4Y0; the Laguna SLR-crossing columns are in rows Y3/Y4 and Y7/Y8, every column X0-X7.) So
rack 0 (ports 0-15) spans X0-X3 and rack 1 (16-31) X4-X7, and a port's AXI handshake at 450
MHz (2.222 ns) has to come from logic placed in or next to its clock region. Our targets:
f2-burst200's ports 0 / 4 are X0Y0 / X1Y0 (Alex's SmartConnect sits on the left, where
pblock_CL_SLR0 has its whole clock regions X0Y0:X3Y3); f2-burst200-s3's slice 2 uses ports 16
/ 20 = X4Y0 / X5Y0 on the right, where the SLR0 pblock has only the bottom row X4Y0:X7Y0 plus
X5Y1:X5Y3, X6Y1 and X4Y3 — build 10's post-placement worst path at 450 MHz is exactly port
16's interface clock into slice 2's SmartConnect exit pipeline (-0.338 ns before physical
optimization). Build 8 (200 MHz) routed with exactly this class failing: the 450 MHz domain at
-0.087 ns over 1095 endpoints, every one a port 16 / 20 interface clock into u_mem2's exit
pipelines, with one logic level and ~2.1 ns of wire, because the placer had put u_mem2 in
SLR0's roomy left half. Build 10 (175 MHz) showed it is not a slice-2 quirk: all three
SmartConnects landed two or three rows above their ports (by the Laguna columns where their
slices' register slices arrive: u_mem0's mass in X0Y0-X0Y2, u_mem1's in row Y3, u_mem2's in
X3-X5 rows Y2-Y3) and all three failed, -0.200 ns over 1824 endpoints (u_mem2 658, u_mem1 633,
u_mem0 524; sources ports 20 / 12 / 8 / 16 / 4 / 0 and the SmartConnects' own 450 MHz logic;
the worst path port 16's ACLK into u_mem2's `m00_exit` register enable, 2.02 ns with one
logic level). The remedy, in floorplan_s3.tcl from build 11 on, is a hard child pblock per
SmartConnect holding it in the clock regions of its two ports plus the row above:
`pblock_CL_SLR0_MEM0` = X0Y0:X1Y1, `MEM1` = X2Y0:X3Y1, `MEM2` = X4Y0:X6Y0 + X5Y1:X6Y1 (15K
LUTs / 25K FFs each in four or five clock regions; the slices' SLR0 register-slice ends and
host pipes stay free and reach them with two or three regions of wire at the core clock).
The alternative is ports on the left for all three
slices (e.g. 0 / 4, 8 / 12, 1 / 5 — distinct memory channels are ports 2k / 2k+1 pairs, so 1 /
5 share channels with 0 / 4 and that pairing costs the switch path, rule 6).

## What this means for the candidates

| target | slices x native width | ports | raw GB/s | fabric |
|---|---|---|---|---|
| f2-burst200 (build 6, on silicon) | 1 x 128 B | 2 | 25.6 | Alex's, as is |
| f2-burst200-s3 (in design) | 3 x 128 B | 6 | 76.8 | 3 x Alex's port group, one slice per SLR |
| P1 wide port on this fabric | 1 x 256 B | 4 | 51.2 | 2 port groups per slice, a 2-chunk adapter |
| P4 four slices | 4 x 128 B | 8 | 102.4 | 4 port groups; SLR budget: 2 slices in one SLR |
| P2 four lanes | 1 x 512 B | 8 | 102.4 | 4 port groups per slice, 4 adapters |

The round-5 fabric (`f2-hbm`, eight `cl_axi_sc_1x1` behind one slice) is retired as a build
target unless build 5 closes; its adapter and its tests stay as the reference for a
per-port SmartConnect.
