# floorplan_s4_tmem.tcl (P4): floorplan_s4.tcl plus the TMEM boxes at its end.
# Four slices (targets/f2-burst200-s4*.toml, docs/v4/p4-status.md "On the F2"): floorplan_s3.tcl
# (build 11, closed at 175 MHz and run on the card) with the fourth slice. The VU47P has three
# SLRs, so two slices share one: slices 0 | 1 | 2 and 3 in SLR0 | SLR1 | SLR2 (the target's
# slice_slr). SLR2 is the one with room: at S = 3 SLR0 holds slice 0, every SmartConnect, the
# HBM wrapper and the SLR0 ends of the host and native register slices, and SLR1 slice 1, the
# control block, the collective hub and the shell-facing paths beside the shell's columns
# X6-X7. Legal child regions as floorplan_s3.tcl's (the HDK's cl_dram_hbm_dma small-shell
# ranges); AWS pblock_CL must already exist.
#
#   SLR0: slice 0 and its adapters, every SmartConnect (u_mem0..3, each in a grandchild
#         pblock_CL_SLR0_MEM<s> beside its two HBM ports), the HBM wrapper, the host crossbar
#         (u_host) behind its register slice at the crossing (u_pcis_hop; the targets'
#         host_xbar_slr 0: one host channel crosses SLR1 -> SLR0 instead of four), the SLR0 ends
#         of the host and native register slices, the SLR0 resets
#   SLR1: slice 1 and its adapters, the control block, the trace, the collective unit and its
#         hub, the OCL converter, the PCIS pipe, the SLR1 ends of slices 2 and 3's native
#         register slices
#   SLR2: slices 2 and 3, each with its adapters in a half of the SLR (pblock_CL_SLR2_S2 the left
#         columns X0-X3, pblock_CL_SLR2_S3 the right X4-X7: the full_hbm (v5) lesson, where two
#         slices loose in one SLR2 pblock scattered their MXUs' DSP chains and adapters across the
#         die, 6.6 ns of route on a core path), the SLR2 resets
# Every signal between SLRs leaves a flip-flop and lands in one: otpu_board's otpu_xr pairs
# (control, status, the collective; USER_SLL_REG on both stages of the SLR0 and SLR2
# crossings), the axi_register_slice IPs (one per hop, placed in the hop's destination SLR).
# Cell lookups fail closed: a missing cell stops the build (a renamed instance would otherwise
# float free of its SLR).

proc otpu_s4_cells {patterns} {
  set found {}
  foreach p $patterns {
    set c [get_cells -quiet $p]
    if {![llength $c]} {error "floorplan_s4.tcl: no cell matches $p"}
    set found [concat $found $c]
  }
  return $found
}

set cl WRAPPER/CL
set xs $cl/u_board/g_xs

# ---- SLR0 (Alex's ranges: complete and partial CRs; floorplan_s3.tcl's)
create_pblock pblock_CL_SLR0
resize_pblock pblock_CL_SLR0 -add {\
    CLOCKREGION_X0Y0:CLOCKREGION_X3Y3 \
    CLOCKREGION_X4Y0:CLOCKREGION_X7Y0 \
    CLOCKREGION_X4Y3:CLOCKREGION_X4Y3 \
    CLOCKREGION_X6Y1:CLOCKREGION_X6Y1 \
    CLOCKREGION_X5Y1:CLOCKREGION_X5Y3 \
    SLICE_X120Y120:SLICE_X145Y179 \
    DSP48E2_X16Y42:DSP48E2_X19Y65 \
    RAMB18_X8Y48:RAMB18_X9Y71 \
    RAMB36_X8Y24:RAMB36_X9Y35 \
    URAM288_X2Y32:URAM288_X2Y47 \
    SLICE_X118Y60:SLICE_X145Y119 \
    DSP48E2_X16Y18:DSP48E2_X19Y41 \
    RAMB18_X8Y24:RAMB18_X9Y47 \
    RAMB36_X8Y12:RAMB36_X9Y23 \
    URAM288_X2Y16:URAM288_X2Y31 \
    SLICE_X206Y60:SLICE_X219Y119 \
    DSP48E2_X30Y18:DSP48E2_X30Y41 \
    RAMB18_X12Y24:RAMB18_X12Y47 \
    RAMB36_X12Y12:RAMB36_X12Y23 \
    SLICE_X221Y60:SLICE_X232Y119 \
    DSP48E2_X31Y18:DSP48E2_X31Y41 \
    RAMB18_X13Y24:RAMB18_X13Y47 \
    RAMB36_X13Y12:RAMB36_X13Y23 \
    BUFG_GT_X1Y24:BUFG_GT_X1Y47 \
    BUFG_GT_SYNC_X1Y15:BUFG_GT_SYNC_X1Y29 \
    GTYE4_COMMON_X1Y1:GTYE4_COMMON_X1Y1 \
    GTYE4_CHANNEL_X1Y4:GTYE4_CHANNEL_X1Y7}
set_property parent pblock_CL [get_pblocks pblock_CL_SLR0]

# ---- SLR1 (complete CRs and the partial ranges, as the HDK example)
create_pblock pblock_CL_SLR1
resize_pblock pblock_CL_SLR1 -add {\
    CLOCKREGION_X0Y4:CLOCKREGION_X3Y7 \
    CLOCKREGION_X5Y4:CLOCKREGION_X5Y7 \
    CLOCKREGION_X4Y6:CLOCKREGION_X4Y7 \
    CLOCKREGION_X4Y4:CLOCKREGION_X4Y4 \
    SLICE_X122Y300:SLICE_X145Y359 \
    DSP48E2_X16Y114:DSP48E2_X19Y137 \
    RAMB18_X9Y120:RAMB18_X9Y143 \
    RAMB36_X9Y60:RAMB36_X9Y71 \
    URAM288_X2Y80:URAM288_X2Y95 \
    RAMB18_X8Y120:RAMB18_X8Y143 \
    RAMB36_X8Y60:RAMB36_X8Y71}
set_property SNAPPING_MODE ON [get_pblocks pblock_CL_SLR1]
set_property parent pblock_CL [get_pblocks pblock_CL_SLR1]

# ---- SLR2 (complete CRs) and its halves, one slice each
create_pblock pblock_CL_SLR2
resize_pblock pblock_CL_SLR2 -add {CLOCKREGION_X0Y8:CLOCKREGION_X7Y11}
set_property parent pblock_CL [get_pblocks pblock_CL_SLR2]
foreach {s ranges} {
  2 {CLOCKREGION_X0Y8:CLOCKREGION_X3Y11}
  3 {CLOCKREGION_X4Y8:CLOCKREGION_X7Y11}
} {
  create_pblock pblock_CL_SLR2_S$s
  resize_pblock pblock_CL_SLR2_S$s -add $ranges
  set_property parent pblock_CL_SLR2 [get_pblocks pblock_CL_SLR2_S$s]
}

# ---- SLR0: slice 0, every SmartConnect by the HBM ports, the HBM wrapper
add_cells_to_pblock pblock_CL_SLR0 [otpu_s4_cells [list \
    $cl/u_board/u_slice $cl/u_board/u_mem \
    "$xs.g_x\[0\].u_dn_rx" "$xs.g_x\[0\].u_up_tx" "$xs.g_x\[0\].u_cport" "$xs.g_x\[0\].g_pf.u_pf_tx" \
    $cl/u_native0 $cl/u_native1 \
    $cl/u_npipe2_slr0 $cl/u_npipe3_slr0 $cl/u_npipe4_slr0 $cl/u_npipe5_slr0 \
    $cl/u_npipe6_slr0 $cl/u_npipe7_slr0 \
    $cl/u_pcis_hop $cl/u_host $cl/u_host_pipe0 $cl/u_host_pipe1 $cl/u_host_pipe2 $cl/u_host_pipe3 \
    $cl/u_hbm \
    $cl/u_rst_core_slr0 $cl/u_rstpipe_core_slr0 $cl/u_rst_main_slr0 $cl/u_rstpipe_main_slr0]]
set_property IS_SOFT false [get_pblocks pblock_CL_SLR0]

# ---- SLR0, by the HBM ports: one grandchild pblock per SmartConnect (floorplan_s3.tcl's build 11
# fix: left to itself the placer puts the SmartConnects two or three rows up, by the Laguna
# columns where their slices' register slices land, and the port-to-exit-pipeline paths at
# 450 MHz fail with one logic level and ~2 ns of wire). The HBM AXI port interfaces are the
# sites BLI_HBM_AXI_INTF_X<k>Y0 in clock-region row Y0: ports 0-3 X0Y0, 4-7 X1Y0, 8-12 X2Y0,
# 13-15 X3Y0, 16-19 X4Y0, 20-24 X5Y0, 25-29 X6Y0, 30-31 X7Y0 (docs/v4/hbm-integration.md).
# Each SmartConnect (15K LUTs, 25K FFs) is held in the clock regions of its ports plus the row
# above where the parent has it (the right half of SLR0 is short: X4Y1:X4Y2, X6Y2:X7Y3 and X7Y1
# are not the CL's). At S = 3 u_mem2 had X4Y0:X6Y0 + X5Y1:X6Y1; the fourth SmartConnect takes
# X6-X7, which is why slice 3's ports are 26 / 28 (both X6Y0) rather than 24 / 28.
#   u_mem0: ports 0 (X0Y0) and 4 (X1Y0)   -> X0Y0:X1Y1
#   u_mem1: ports 8 and 12 (both X2Y0)    -> X2Y0:X3Y1
#   u_mem2: ports 16 (X4Y0) and 20 (X5Y0) -> X4Y0:X5Y0 + X5Y1
#   u_mem3: ports 26 and 28 (both X6Y0)   -> X6Y0:X7Y0 + X6Y1
foreach {s ranges} {
  0 {CLOCKREGION_X0Y0:CLOCKREGION_X1Y1}
  1 {CLOCKREGION_X2Y0:CLOCKREGION_X3Y1}
  2 {CLOCKREGION_X4Y0:CLOCKREGION_X5Y0 CLOCKREGION_X5Y1:CLOCKREGION_X5Y1}
  3 {CLOCKREGION_X6Y0:CLOCKREGION_X7Y0 CLOCKREGION_X6Y1:CLOCKREGION_X6Y1}
} {
  create_pblock pblock_CL_SLR0_MEM$s
  resize_pblock pblock_CL_SLR0_MEM$s -add $ranges
  set_property parent pblock_CL_SLR0 [get_pblocks pblock_CL_SLR0_MEM$s]
  add_cells_to_pblock pblock_CL_SLR0_MEM$s [otpu_s4_cells [list $cl/u_mem$s]]
  set_property IS_SOFT false [get_pblocks pblock_CL_SLR0_MEM$s]
}

# ---- SLR1: slice 1, the control block, the collective, the shell-facing host and OCL paths, the
# hub-side stages of every crossing slice (0, 2, 3), slices 2 and 3's native register slices' first hop
add_cells_to_pblock pblock_CL_SLR1 [otpu_s4_cells [list \
    "$cl/u_board/g_slice\[1\].u_slice" "$cl/u_board/g_slice\[1\].u_mem" \
    "$xs.g_x\[1\].u_dn_rx" "$xs.g_x\[1\].u_up_tx" "$xs.g_x\[1\].u_cport" \
    "$xs.g_x\[0\].u_dn_tx" "$xs.g_x\[0\].u_up_rx" "$xs.g_x\[0\].g_pf.u_pf_rx" \
    "$xs.g_x\[1\].u_dn_tx" "$xs.g_x\[1\].u_up_rx" \
    "$xs.g_x\[2\].u_dn_tx" "$xs.g_x\[2\].u_up_rx" \
    "$xs.g_x\[3\].u_dn_tx" "$xs.g_x\[3\].u_up_rx" \
    "$xs.u_chub" $cl/u_board/u_ctrl $cl/u_board/u_iso $cl/u_board/g_trace.u_trace \
    $cl/u_board/g_coll.u_coll \
    $cl/u_native2 $cl/u_native3 \
    $cl/u_npipe4_slr1 $cl/u_npipe5_slr1 $cl/u_npipe6_slr1 $cl/u_npipe7_slr1 \
    $cl/u_pcis_pipe $cl/u_ocl_cdc $cl/u_status_cdc]]
set_property IS_SOFT false [get_pblocks pblock_CL_SLR1]

# ---- SLR2: the resets in the parent; slice 2 (left half) and slice 3 (right half), each with its
# adapters and its side of the board's crossing (the receiving and sending stages, the port)
add_cells_to_pblock pblock_CL_SLR2 [otpu_s4_cells [list $cl/u_rst_core_slr2 $cl/u_rstpipe_core_slr2]]
set_property IS_SOFT false [get_pblocks pblock_CL_SLR2]
foreach {s n0 n1} {2 4 5 3 6 7} {
  add_cells_to_pblock pblock_CL_SLR2_S$s [otpu_s4_cells [list \
      "$cl/u_board/g_slice\[$s\].u_slice" "$cl/u_board/g_slice\[$s\].u_mem" \
      "$xs.g_x\[$s\].u_dn_rx" "$xs.g_x\[$s\].u_up_tx" "$xs.g_x\[$s\].u_cport" \
      $cl/u_native$n0 $cl/u_native$n1]]
  set_property IS_SOFT false [get_pblocks pblock_CL_SLR2_S$s]
}

# ---- Laguna registers on the board's own SLR crossings (otpu_board g_xs, otpu_xr r_reg):
# slice 0's (SLR1 <-> SLR0) and slices 2 and 3's (SLR1 <-> SLR2) stages, both sides; slice 1's
# pairs stay inside SLR1
foreach s {0 2 3} {
  set sll [get_cells -quiet [list "$xs.g_x\[$s\].u_*_tx/r_reg*" "$xs.g_x\[$s\].u_*_rx/r_reg*" \
                                  "$xs.g_x\[$s\].g_pf.u_pf_*/r_reg*"]]
  if {![llength $sll]} {error "floorplan_s4.tcl: slice $s: no otpu_xr crossing registers found"}
  set_property USER_SLL_REG TRUE $sll
  puts "floorplan_s4.tcl: slice $s's control / status / collective crossing, [llength $sll] Laguna registers"
}

# ---- floorplan_s4_tmem.tcl: floorplan_s4.tcl and a box per slice for the pumped TMEM with its DMA
# and its collective port (the learnings' "box the pumped TMEM (2x clock) with its quantizer / VPU /
# collective / DMA"). 866c920e at 175 MHz routed with every failing path inside a pumped bank: the
# write-staging registers (g_pmp.pk_v / pk_a / evr) into their BRAMs' ENARDEN / ADDR / WEA, ~2.0 ns of
# route at the 2.857 ns fast period, and the DMA's t_re_q into the banks' port B. The routed
# checkpoint had each slice's 64 TMEM BRAMs over 6-8 clock regions (slice 0 X2-X4 Y1-Y3, slice 1
# X1-X3 Y6-Y7 with its TMEM logic in X4-X5, slice 2 X1-X2 Y9-Y11, slice 3 X4-X7 Y10-Y11). Each box is
# 6 clock regions (~126 RAMB36 for the TMEM's 64 and the DMA's 15), a hard child of the slice's own
# pblock, clear of the SmartConnect boxes on SLR0's rows Y0-Y1.
foreach {s parent ranges tm cp} [list \
  0 pblock_CL_SLR0    {CLOCKREGION_X1Y2:CLOCKREGION_X3Y3}  $cl/u_board/u_slice           "$xs.g_x\[0\].u_cport" \
  1 pblock_CL_SLR1    {CLOCKREGION_X1Y6:CLOCKREGION_X3Y7}  "$cl/u_board/g_slice\[1\].u_slice" "$xs.g_x\[1\].u_cport" \
  2 pblock_CL_SLR2_S2 {CLOCKREGION_X1Y9:CLOCKREGION_X2Y11} "$cl/u_board/g_slice\[2\].u_slice" "$xs.g_x\[2\].u_cport" \
  3 pblock_CL_SLR2_S3 {CLOCKREGION_X5Y9:CLOCKREGION_X6Y11} "$cl/u_board/g_slice\[3\].u_slice" "$xs.g_x\[3\].u_cport"] {
  create_pblock pblock_CL_TMEM$s
  resize_pblock pblock_CL_TMEM$s -add $ranges
  set_property parent $parent [get_pblocks pblock_CL_TMEM$s]
  add_cells_to_pblock pblock_CL_TMEM$s [otpu_s4_cells [list $tm/u_tmem $tm/u_dma $cp]]
  set_property IS_SOFT false [get_pblocks pblock_CL_TMEM$s]
  puts "floorplan_s4_tmem.tcl: slice $s's TMEM, DMA and collective port in $ranges"
}
