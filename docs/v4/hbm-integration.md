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

Consequences for our designs:

- **The PCIS's first register slice belongs in SLR2**, not SLR1. Alex's `f2_pcis_pipe` sits
  in SLR1 and still closed, but build 6's worst setup path (+0.035 ns) is exactly the
  SmartConnect's host entry into that pipe. The HDK example does it right: `AXI4_REG_SLC_PCIS_SLR2`
  in SLR2, then `AXI4_REG_SLC_PCIS_SLR1` and the crossbar in SLR1, then one slice per
  destination SLR. In a three-slice design the host path is a chain SLR2 -> SLR1 -> SLR0
  with a register slice at each hop, and slice 2 (in SLR2) is the host's nearest window.
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
