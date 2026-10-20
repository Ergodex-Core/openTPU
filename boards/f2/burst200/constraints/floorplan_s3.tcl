# Three slices, one per SLR (targets/f2-burst200-s3.toml, docs/v4/f2-burst200-s3.md), in the
# style of floorplan.tcl: legal child regions copied from the installed AWS cl_mem_perf /
# cl_dram_hbm_dma small-shell constraints (hdk/cl/examples/cl_dram_hbm_dma/build/constraints/
# small_shell_cl_pnr_user.xdc: SLR0 and SLR1 as Alex's floorplan.tcl has them, SLR2 the whole
# clock regions X0Y8:X7Y11, the PCIS crossbar in the grandchild pblock_CL_SLR1_XBAR). AWS
# pblock_CL must already exist.
#
#   SLR0 (the HBM's AXI ports at its bottom edge): slice 0 and its adapters, every SmartConnect
#        (u_mem0/1/2, each in a grandchild pblock_CL_SLR0_MEM<s> beside its two ports), the HBM
#        wrapper, the SLR0 ends of the host and native register slices, the SLR0 resets
#   SLR1 (the shell's PCIS / OCL columns X6-X7): slice 1 and its adapters, the control block,
#        the trace, the collective unit and its hub, the OCL converter, the PCIS pipe, the
#        host crossbar (pblock_CL_SLR1_XBAR), the SLR1 ends of slice 2's native register slices
#   SLR2: slice 2 and its adapters, the SLR2 resets
# Every signal between SLRs leaves a flip-flop and lands in one: otpu_board's otpu_xr pairs
# (control, status, the collective; USER_SLL_REG on both stages of the SLR0 and SLR2
# crossings), the axi_register_slice IPs (one per hop, placed in the hop's destination SLR).
# Cell lookups fail closed: a missing cell stops the build (a renamed instance would otherwise
# float free of its SLR).

proc otpu_s3_cells {patterns} {
  set found {}
  foreach p $patterns {
    set c [get_cells -quiet $p]
    if {![llength $c]} {error "floorplan_s3.tcl: no cell matches $p"}
    set found [concat $found $c]
  }
  return $found
}

set cl WRAPPER/CL
set xs $cl/u_board/g_xs

# ---- SLR0 (Alex's ranges: complete and partial CRs)
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

# the host crossbar away from the shell interface (the example's grandchild for its PCIS crossbar)
create_pblock pblock_CL_SLR1_XBAR
resize_pblock pblock_CL_SLR1_XBAR -add {CLOCKREGION_X0Y4:CLOCKREGION_X4Y7}
set_property parent pblock_CL_SLR1 [get_pblocks pblock_CL_SLR1_XBAR]

# ---- SLR2 (complete CRs)
create_pblock pblock_CL_SLR2
resize_pblock pblock_CL_SLR2 -add {CLOCKREGION_X0Y8:CLOCKREGION_X7Y11}
set_property parent pblock_CL [get_pblocks pblock_CL_SLR2]

# ---- SLR0: slice 0, every SmartConnect by the HBM ports, the HBM wrapper
add_cells_to_pblock pblock_CL_SLR0 [otpu_s3_cells [list \
    $cl/u_board/u_slice $cl/u_board/u_mem \
    "$xs.g_x\[0\].u_dn_rx" "$xs.g_x\[0\].u_up_tx" "$xs.g_x\[0\].u_cport" "$xs.g_x\[0\].g_pf.u_pf_tx" \
    $cl/u_native0 $cl/u_native1 \
    $cl/u_npipe2_slr0 $cl/u_npipe3_slr0 $cl/u_npipe4_slr0 $cl/u_npipe5_slr0 \
    $cl/u_host_pipe0 $cl/u_host_pipe1 $cl/u_host_pipe2 \
    $cl/u_hbm \
    $cl/u_rst_core_slr0 $cl/u_rstpipe_core_slr0 $cl/u_rst_main_slr0 $cl/u_rstpipe_main_slr0]]
set_property IS_SOFT false [get_pblocks pblock_CL_SLR0]

# ---- SLR0, by the HBM ports: one grandchild pblock per SmartConnect. The HBM AXI port
# interfaces are the sites BLI_HBM_AXI_INTF_X<k>Y0 in clock-region row Y0: ports 0-3 X0Y0, 4-7
# X1Y0, 8-12 X2Y0, 13-15 X3Y0, 16-19 X4Y0, 20-24 X5Y0, 25-29 X6Y0, 30-31 X7Y0
# (docs/v4/hbm-integration.md). Left to itself the placer puts the SmartConnects two or three
# rows up, by the Laguna columns where their slices' register slices land (build 10: u_mem0's
# mass in X0Y0-X0Y2, u_mem1's in row Y3, u_mem2's in X3-X5 rows Y2-Y3), and the port-to-exit-
# pipeline paths at 450 MHz (2.222 ns) fail with one logic level and ~2 ns of wire: build 8
# -0.087 ns over 1095 endpoints, build 10 -0.200 over 1824 spread over all three (u_mem2 658,
# u_mem1 633, u_mem0 524 destinations; sources ports 20 / 12 / 8 / 16 / 4 / 0 and the
# SmartConnects' own 450 MHz logic). Each SmartConnect (15K LUTs, 25K FFs) is held in the
# clock regions of its two ports plus the row above (all within the parent's ranges); the
# core-clock paths from its slice's SLR0 register-slice ends get two or three regions of wire
# at 5.7 ns, which is fine.
#   u_mem0: ports 0 (X0Y0) and 4 (X1Y0)   -> X0Y0:X1Y1
#   u_mem1: ports 8 and 12 (both X2Y0)    -> X2Y0:X3Y1
#   u_mem2: ports 16 (X4Y0) and 20 (X5Y0) -> X4Y0:X6Y0 + X5Y1:X6Y1 (X4Y1 is the shell's)
foreach {s ranges} {
  0 {CLOCKREGION_X0Y0:CLOCKREGION_X1Y1}
  1 {CLOCKREGION_X2Y0:CLOCKREGION_X3Y1}
  2 {CLOCKREGION_X4Y0:CLOCKREGION_X6Y0 CLOCKREGION_X5Y1:CLOCKREGION_X6Y1}
} {
  create_pblock pblock_CL_SLR0_MEM$s
  resize_pblock pblock_CL_SLR0_MEM$s -add $ranges
  set_property parent pblock_CL_SLR0 [get_pblocks pblock_CL_SLR0_MEM$s]
  add_cells_to_pblock pblock_CL_SLR0_MEM$s [otpu_s3_cells [list $cl/u_mem$s]]
  set_property IS_SOFT false [get_pblocks pblock_CL_SLR0_MEM$s]
}

# ---- SLR1: slice 1, the control block, the collective, the shell-facing host and OCL paths
add_cells_to_pblock pblock_CL_SLR1 [otpu_s3_cells [list \
    "$cl/u_board/g_slice\[1\].u_slice" "$cl/u_board/g_slice\[1\].u_mem" \
    "$xs.g_x\[1\].u_dn_rx" "$xs.g_x\[1\].u_up_tx" "$xs.g_x\[1\].u_cport" \
    "$xs.g_x\[0\].u_dn_tx" "$xs.g_x\[0\].u_up_rx" "$xs.g_x\[0\].g_pf.u_pf_rx" \
    "$xs.g_x\[1\].u_dn_tx" "$xs.g_x\[1\].u_up_rx" \
    "$xs.g_x\[2\].u_dn_tx" "$xs.g_x\[2\].u_up_rx" \
    "$xs.u_chub" $cl/u_board/u_ctrl $cl/u_board/u_iso $cl/u_board/g_trace.u_trace \
    $cl/u_board/g_coll.u_coll \
    $cl/u_native2 $cl/u_native3 $cl/u_npipe4_slr1 $cl/u_npipe5_slr1 \
    $cl/u_pcis_pipe $cl/u_ocl_cdc $cl/u_status_cdc]]
add_cells_to_pblock pblock_CL_SLR1_XBAR [otpu_s3_cells [list $cl/u_host]]
set_property IS_SOFT false [get_pblocks pblock_CL_SLR1]

# ---- SLR2: slice 2
add_cells_to_pblock pblock_CL_SLR2 [otpu_s3_cells [list \
    "$cl/u_board/g_slice\[2\].u_slice" "$cl/u_board/g_slice\[2\].u_mem" \
    "$xs.g_x\[2\].u_dn_rx" "$xs.g_x\[2\].u_up_tx" "$xs.g_x\[2\].u_cport" \
    $cl/u_native4 $cl/u_native5 \
    $cl/u_rst_core_slr2 $cl/u_rstpipe_core_slr2]]
set_property IS_SOFT false [get_pblocks pblock_CL_SLR2]

# ---- Laguna registers on the board's own SLR crossings (otpu_board g_xs, otpu_xr r_reg):
# slice 0's (SLR1 <-> SLR0) and slice 2's (SLR1 <-> SLR2) stages, both sides; slice 1's pairs
# stay inside SLR1
foreach s {0 2} {
  set sll [get_cells -quiet [list "$xs.g_x\[$s\].u_*_tx/r_reg*" "$xs.g_x\[$s\].u_*_rx/r_reg*" \
                                  "$xs.g_x\[$s\].g_pf.u_pf_*/r_reg*"]]
  if {![llength $sll]} {error "floorplan_s3.tcl: slice $s: no otpu_xr crossing registers found"}
  set_property USER_SLL_REG TRUE $sll
  puts "floorplan_s3.tcl: slice $s's control / status / collective crossing, [llength $sll] Laguna registers"
}
