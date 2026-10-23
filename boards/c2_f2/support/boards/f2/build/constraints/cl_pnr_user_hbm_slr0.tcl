# The HBM build's floorplan, Alex's (docs/v4/f2-round5.md): boards/f2/build.py --floorplan slr0
# appends this file to cl_pnr_user.tcl for an HBM target in place of the five-pblock
# cl_pnr_user_hbm.tcl. It is the floorplan.tcl of his burst200 runs (20261014_50_burst_link,
# 20261015_63_burst_retime: the only openTPU F2 image that closed 200 MHz), with his cell names
# replaced by this CL's: one pblock in SLR0 for everything that computes or moves data (the
# board, the fabric, the SmartConnects, the HBM wrapper), one in SLR1 for the shell-facing
# registers (the PCIS skid buffers, output registers and link ends; the OCL crossing). His
# ranges are "legal child regions copied from installed AWS cl_mem_perf constraints": complete
# clock regions where the shell leaves them to the CL and explicit site ranges where it does not
# (the partial regions beside the shell's columns), so no containment step is needed and
# IS_SOFT is off on both; the SLR1 pblock snaps (as his). CONTAIN_ROUTING stays off. The cell
# queries fail closed, as in the five-pblock file.

proc otpu_cells {pattern} {
  set cells [get_cells -quiet $pattern]
  if {![llength $cells]} { error "openTPU floorplan: no cells match $pattern" }
  return $cells
}
set otpu_nport [llength [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_axi}]]
puts "OTPU_FLOORPLAN ports=$otpu_nport"

# Legal child regions copied from installed AWS cl_mem_perf constraints.
# AWS pblock_CL must already exist.
create_pblock pblock_CL_SLR0

# Complete and partial CRs; one batched add, same legal ranges.
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

create_pblock pblock_CL_SLR1

# Complete CRs in SLR1
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

# Keep accelerator state, cascades and wide datapaths in one die.
add_cells_to_pblock pblock_CL_SLR0 [otpu_cells {WRAPPER/CL/u_sys/u_board}]
add_cells_to_pblock pblock_CL_SLR0 [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_axi}]
add_cells_to_pblock pblock_CL_SLR0 [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_port}]
add_cells_to_pblock pblock_CL_SLR0 [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host}]
add_cells_to_pblock pblock_CL_SLR0 [otpu_cells {WRAPPER/CL/g_sc[*].u_sc}]
add_cells_to_pblock pblock_CL_SLR0 [otpu_cells {WRAPPER/CL/u_hbm}]
set_property IS_SOFT false [get_pblocks pblock_CL_SLR0]
# Shell-facing registered PCIS endpoint in the shell interface's die.
foreach k {aw w ar b r aw_tx w_tx ar_tx b_rx r_rx wb} {
  add_cells_to_pblock pblock_CL_SLR1 [otpu_cells WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_$k]
}
add_cells_to_pblock pblock_CL_SLR1 [otpu_cells {WRAPPER/CL/u_sys/u_ocl}]
set_property IS_SOFT false [get_pblocks pblock_CL_SLR1]
foreach pb {pblock_CL_SLR0 pblock_CL_SLR1} { set_property CONTAIN_ROUTING 0 [get_pblocks $pb] }
puts "OTPU_FLOORPLAN_APPLIED pblocks=2"
