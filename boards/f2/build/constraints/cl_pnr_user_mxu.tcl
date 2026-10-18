# The MXU's array side in SLR1 (docs/vla_frequency_plan.md 6i; f2-vla32, otpu_slice MXU_ARR 1).
# boards/f2/build.py appends this file after cl_pnr_user_hbm.tcl for an HBM target whose uarch
# states MXU_ARR, so pb_otpu_board (SLR2) holds the rest of the slice: the issuer and the drain
# of the MXU, the TMEM, the VPU, the quantizer, the DMA and the sequencer, which share same-cycle
# grants and must stay in one SLR, at ~395K LUTs of SLR2's ~435K (91%) by 6h's count; SLR1's
# box, pb_otpu_mid's 16 clock regions (~217K LUTs), takes ~150K (69%). The cells here (fail
# closed, otpu_cells: a renamed hierarchy stops the build):
#   u_slice/u_mxu/u_arr    the array side (otpu_mxu_arr: the chunk decode, the shifts, the PEs,
#                          the column ends, the row assembly), behind a registered boundary at
#                          both ends (ARR): ~105K LUTs, 4,672 DSPs.
#   u_slice/u_mxu/g_fd[*]  the chunk FIFO's block RAMs (u_fd, PACK's u_fdb): their read registers
#                          feed u_arr's input stages; the write pins and the read addresses come
#                          from registers in otpu_mxu (SLR2: the slice's port B registers, the
#                          queue's slot copies, f_rdl).
#   u_slice/u_mxu/u_fs0, u_fs1, g_fsb.u_fsb   the scale FIFO's block RAMs, likewise.
#   u_slice/u_act          the ACT RAM (1,088 RAMB36 beside the columns): its write side takes
#                          the quantizer's bytes through two registers (otpu_actram we_i / we_h,
#                          the long hop by design), its read request through RREG (an input
#                          register stage; otpu_mxu's act_* outputs are registers with ARR).
#   u_mem                  the slice's DRAM adapter (otpu_native_dram, ~38K LUTs): the only other
#                          block SLR2 can spare. The slice registers its side of the port
#                          interfaces (port A / B / SW requests out, the readies, valids and data
#                          in); the adapter's b_rdy and b_rdata are a LUT level or two in front
#                          of the slice's registers, its a_* and b_rvalid registers: one hop with
#                          at most two LUT levels, not the chains' flip-flop to flip-flop (6i).
# Every other net across the SLR1 / SLR2 boundary of this split is a flip-flop (or a block RAM
# pin) at both ends: the S0 bundle into u_arr's input stages, u_arr's row and push registers into
# the result FIFO, the registered ACT read request into otpu_actram's RREG stage.
create_pblock pb_otpu_mxu
set_property PARENT pblock_CL [get_pblocks pb_otpu_mxu]
resize_pblock pb_otpu_mxu -add {CLOCKREGION_X0Y4:CLOCKREGION_X3Y7}
add_cells_to_pblock pb_otpu_mxu [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_mxu/u_arr}]
add_cells_to_pblock pb_otpu_mxu [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_mxu/g_fd[*].u_fd}]
add_cells_to_pblock pb_otpu_mxu [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_mxu/g_fd[*].g_b.u_fdb}]
add_cells_to_pblock pb_otpu_mxu [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_mxu/u_fs0}]
add_cells_to_pblock pb_otpu_mxu [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_mxu/u_fs1}]
add_cells_to_pblock pb_otpu_mxu [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_mxu/g_fsb.u_fsb}]
add_cells_to_pblock pb_otpu_mxu [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_act}]
add_cells_to_pblock pb_otpu_mxu [otpu_cells {WRAPPER/CL/u_sys/u_board/u_mem}]
foreach pb {pb_otpu_mxu} {
  set_property CONTAIN_ROUTING 0 [get_pblocks $pb]
  set_property SNAPPING_MODE OFF [get_pblocks $pb]
  set_property IS_SOFT 0 [get_pblocks $pb]
  otpu_contain $pb
}
puts "OTPU_FLOORPLAN_MXU_APPLIED pblocks=1"
