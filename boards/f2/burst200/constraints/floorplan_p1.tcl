# P1's one slice at CPC 2 (targets/f2.toml, docs/v4/p1-status.md section 6) on the burst200 fabric, in the style
# of floorplan.tcl (Alex's one slice) and floorplan_s3.tcl (their SLR0 / SLR1 ranges, the
# SmartConnect grandchildren by their HBM ports, the host crossbar's grandchild in SLR1; legal
# ranges from the installed AWS cl_mem_perf / cl_dram_hbm_dma small-shell constraints). AWS
# pblock_CL must already exist.
#
#   SLR0 (the HBM's AXI ports at its bottom edge): the whole board (the slice, its adapter, the
#        control block, the trace, the collective), the four native AXI adapters, both
#        SmartConnects (u_mem0 / u_mem1, each in a grandchild pblock beside its two ports), the
#        SLR0 ends of the host's register slices, the HBM wrapper, the SLR0 resets
#   SLR1 (the shell's PCIS / OCL): the OCL converter, the PCIS pipe, the status synchronizer, the
#        host crossbar (pblock_CL_SLR1_XBAR)
# The SLR crossings are the host path's (register slices on both sides: f2_pcis_pipe in SLR1, the
# crossbar, u_host_pipe<g> in SLR0) and the control path's (the OCL converter's registered master
# into the board's otpu_axil_iso, as Alex's one slice). Cell lookups fail closed: a missing cell
# stops the build (a renamed instance would otherwise float free of its SLR).

proc otpu_p1_cells {patterns} {
  set found {}
  foreach p $patterns {
    set c [get_cells -quiet $p]
    if {![llength $c]} {error "floorplan_p1.tcl: no cell matches $p"}
    set found [concat $found $c]
  }
  return $found
}

set cl WRAPPER/CL

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

# ---- SLR0: the board, the adapters, the host's SLR0 register slices, the HBM wrapper, the resets
add_cells_to_pblock pblock_CL_SLR0 [otpu_p1_cells [list \
    $cl/u_board \
    $cl/u_native0 $cl/u_native1 $cl/u_native2 $cl/u_native3 \
    $cl/u_host_pipe0 $cl/u_host_pipe1 \
    $cl/u_hbm \
    $cl/u_rst_core_slr0 $cl/u_rstpipe_core_slr0 $cl/u_rst_main_slr0 $cl/u_rstpipe_main_slr0]]
set_property IS_SOFT false [get_pblocks pblock_CL_SLR0]

# ---- SLR0, by the HBM ports: one grandchild pblock per SmartConnect (floorplan_s3.tcl's: left to
# itself the placer puts them rows up and the 450 MHz port-to-exit-pipeline paths fail). HBM AXI
# port sites BLI_HBM_AXI_INTF_X<k>Y0: ports 0-3 X0Y0, 4-7 X1Y0, 8-12 X2Y0, 13-15 X3Y0.
#   u_mem0: ports 0 (X0Y0) and 4 (X1Y0)   -> X0Y0:X1Y1
#   u_mem1: ports 8 and 12 (both X2Y0)    -> X2Y0:X3Y1
foreach {g ranges} {
  0 {CLOCKREGION_X0Y0:CLOCKREGION_X1Y1}
  1 {CLOCKREGION_X2Y0:CLOCKREGION_X3Y1}
} {
  create_pblock pblock_CL_SLR0_MEM$g
  resize_pblock pblock_CL_SLR0_MEM$g -add $ranges
  set_property parent pblock_CL_SLR0 [get_pblocks pblock_CL_SLR0_MEM$g]
  add_cells_to_pblock pblock_CL_SLR0_MEM$g [otpu_p1_cells [list $cl/u_mem$g]]
  set_property IS_SOFT false [get_pblocks pblock_CL_SLR0_MEM$g]
}

# ---- SLR1: the shell-facing host and OCL paths, the status synchronizer, the crossbar
add_cells_to_pblock pblock_CL_SLR1 [otpu_p1_cells [list \
    $cl/u_pcis_pipe $cl/u_ocl_cdc $cl/u_status_cdc]]
add_cells_to_pblock pblock_CL_SLR1_XBAR [otpu_p1_cells [list $cl/u_host]]
set_property IS_SOFT false [get_pblocks pblock_CL_SLR1]
puts "floorplan_p1.tcl: the board, four adapters and two SmartConnects in SLR0, the host crossbar in SLR1"
