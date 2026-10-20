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
add_cells_to_pblock pblock_CL_SLR0 [get_cells {WRAPPER/CL/u_board WRAPPER/CL/u_native0 WRAPPER/CL/u_native1 WRAPPER/CL/u_hbm WRAPPER/CL/u_mem}]
set_property IS_SOFT false [get_pblocks pblock_CL_SLR0]
# Shell-facing registered PCIS endpoint in the shell interface's die.
add_cells_to_pblock pblock_CL_SLR1 [get_cells {WRAPPER/CL/u_pcis_pipe WRAPPER/CL/u_ocl_cdc}]
set_property IS_SOFT false [get_pblocks pblock_CL_SLR1]
