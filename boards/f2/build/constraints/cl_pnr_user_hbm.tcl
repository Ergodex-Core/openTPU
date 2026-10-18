# The HBM build's floorplan (cl_opentpu_hbm, rtl/boards/f2/otpu_f2_hbm_sys.sv; docs/wide_port.md
# "Placement"). boards/f2/build.py appends this file to cl_pnr_user.tcl for an HBM target, which
# build_level_1_cl.tcl reads unmanaged (read_xdc -unmanaged: Tcl, not a managed XDC, which drops
# proc / if / foreach) right before the HDK's opt_design, after the HDK's own
# small_shell_level_1_fp_cl.xdc, whose pblock_CL (the CL's region of the VU47P: all of SLR2,
# parts of SLR1 and SLR0) is the parent of every pblock here. Ranges by complete clock regions,
# added as the HDK adds its own, then cut down to the parent (otpu_contain below); the VU47P's
# SLRs are rows Y0-Y3 (SLR0, the HBM stacks and their AXI ports below it), Y4-Y7 (SLR1) and
# Y8-Y11 (SLR2); a clock region holds ~13.6K LUTs, ~27K flip-flops.
#
# The second placed f2-hbm:p4rl8 build (fb42c14e, SSI_SpreadLogic_low) put 161,809 LUTs in SLR0,
# 91,118 in SLR1 and 21 in SLR2, with 7,693 SLR0 <-> SLR1 crossings, 2 of them registered, and
# failed at -4.6 ns: the host path 6.1 ns of route from the eight ports' registers, the slice's
# grant and address paths 8 ns across the split. Here:
#   pb_otpu_hbm   (SLR0, the CL's 20 clock regions there, ~272K LUTs): the HBM fabric u_sys/u_mem
#                 (8 bridges, host registers and arbiters, the host path: ~30K LUTs, the 450 MHz
#                 domain) by the HBM's AXI ports, the OCL's crossing and the SLR0 side of every
#                 crossing register chain.
#   pb_otpu_mid   (SLR1, X0Y4:X3Y7, 16 clock regions): the middle stage of every chain (~18K
#                 flip-flops, no logic).
#   pb_otpu_pcis  (SLR1, X5Y5:X5Y6, beside the shell's PCIS column): u_host's PCIS skid and output
#                 registers (round 4; ~1.2K flip-flops).
#   pb_otpu_board (SLR2, all 32 clock regions, ~435K LUTs): u_sys/u_board, the slice with its
#                 adapter, the control block and the trace (226K LUTs, 52%; 416 of the SLR's 672
#                 RAMB36, 1,275 of 3,008 DSPs), and the SLR2 side of the chains. One SLR, so the
#                 adapter's readies into the units, the TMEM's grants into the MXU and the pumped
#                 banks' addresses never cross an SLR. (SLR1's 23 clock regions, ~313K LUTs,
#                 would hold it at 72%: too full for 200 MHz routing.)
#   pb_otpu_tmem  (a child of pb_otpu_board, X2Y9:X5Y10, 8 clock regions, ~109K LUTs): the TMEM
#                 (f2-hbm's 12K LUTs and 64 RAMB36; f2-vla32's ~63K and 384 URAM, the URAM
#                 columns of those regions): the pumped banks' address paths (clk -> clk_f, one
#                 core period: otpu_clk2x.tcl) come from the VPU's ta_addr and the MXU's
#                 t_raddr, which place around the box. The wide-port branch's file held the VPU
#                 in the box too (48K for the pair on f2-hbm); on the vla branch the VPU is out
#                 of it: at 32 lanes with its 16 composite lanes it is ~160K LUTs by the
#                 estimate (docs/vla_frequency_plan.md 6g), which with the TMEM is twice the
#                 box, and an over-full child pblock stops opt_design (a fail-closed outcome
#                 the count should not have to find).
#
# The vla branch's f2-vla32 (docs/vla_frequency_plan.md 6g): the slice is ~465-515K LUTs by
# the estimate from the third f2-vla synthesis (the MXU's 233K and the VPU's 16 composite lanes
# do not shrink with the lanes), so pb_otpu_board at SLR2's ~435K of CL area is over-full
# before the adapter and the control block; this file is the mechanism, read by build.py for
# every HBM target, and the one-SLR placement needs the measured count under ~330K first.
# Every net between SLR0 and SLR2 is a chain of three otpu_xr stages, one per SLR (the native
# ports in otpu_f2_hbm_mem g_xr.g_x[k], the control path in otpu_axil_xr u_cx, the reset and HBM
# ready in g_rs[k]), so each SLR crossing runs from a flip-flop to a flip-flop (TX_REG / RX_REG
# eligible). CONTAIN_ROUTING is off on every pblock (routes may leave the box). The cell queries
# fail closed: a renamed hierarchy stops the build rather than placing freely.
#
# Containment: opt_design's DRC (HDPR-23) requires a child's ranges within its parent's, and
# pblock_CL's are not whole clock regions: the HDK's file takes the shell's columns out, and
# SNAPPING_MODE on a reconfigurable partition's pblock derives site ranges that leave out the
# static shell's sites and the SLR boundaries' LAGUNA rows (the DRC lists them: SLICE_X8Y660+,
# SLICE_X182Y480:X182Y539, URAM288_X4Y32:Y127, ...). So after its clock regions, every site of
# a pblock that its parent does not hold is taken out again (resize_pblock -remove), the
# children in parent-first order, and SNAPPING_MODE stays off on them (a child is not a
# reconfigurable partition; snapping would put the sites back).

proc otpu_cells {pattern} {
  set cells [get_cells -quiet $pattern]
  if {![llength $cells]} { error "openTPU floorplan: no cells match $pattern" }
  return $cells
}
set otpu_nport [llength [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_nch}]]
puts "OTPU_FLOORPLAN ports=$otpu_nport"

########################################
# pb_otpu_hbm: the HBM fabric, SLR0
########################################
create_pblock pb_otpu_hbm
set_property PARENT pblock_CL [get_pblocks pb_otpu_hbm]
resize_pblock pb_otpu_hbm -add {CLOCKREGION_X0Y0:CLOCKREGION_X3Y3}
resize_pblock pb_otpu_hbm -add {CLOCKREGION_X4Y0:CLOCKREGION_X7Y0}
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_nch}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_hreg}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_port}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_xr.g_x[0].u_dn}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_xr.g_x[0].u_up}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_ocl}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.u_m}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[0].u_dn}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[0].u_up}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/g_rs[0].u_x}]

########################################
# pb_otpu_mid: the chains' middle stage, SLR1
########################################
create_pblock pb_otpu_mid
set_property PARENT pblock_CL [get_pblocks pb_otpu_mid]
resize_pblock pb_otpu_mid -add {CLOCKREGION_X0Y4:CLOCKREGION_X3Y7}
add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_xr.g_x[1].u_dn}]
add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_xr.g_x[1].u_up}]
add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[1].u_dn}]
add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[1].u_up}]
add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/g_rs[1].u_x}]

########################################
# pb_otpu_pcis: the PCIS skid / output registers beside the shell's DMA interface, SLR1
########################################
# Round 3 placed the shell's PCIS AXI4 round trip (its register at SLICE_X176Y399, SLR1, to
# u_host at SLR0's top corner and back through a combinational ready) at -5.0 ns on clk_main_a0;
# round 4's otpu_f2_hbm_host puts every PCIS handshake behind registers (u_pcis_aw / _w / _ar
# skids, u_pcis_b / _r output registers, aclk only, kept hierarchies), and they sit here, the
# CL's clock regions nearest the shell's column (X5Y5:X5Y6 of the HDK's X5Y1:X5Y7), so the
# shell -> register hop stays in one SLR; the registers -> u_host's queues in SLR0 is one hop.
create_pblock pb_otpu_pcis
set_property PARENT pblock_CL [get_pblocks pb_otpu_pcis]
resize_pblock pb_otpu_pcis -add {CLOCKREGION_X5Y5:CLOCKREGION_X5Y6}
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_aw}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_w}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_ar}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_b}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_r}]
# round 6: the links' SLR1 halves (otpu_link_tx after the skids, otpu_link_rx before the output
# registers) sit with them; their SLR0 halves (u_host/u_xs_*) stay with u_host, the stage the
# placer puts at the boundary (or a pblock of their own)
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_aw_tx}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_w_tx}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_ar_tx}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_b_rx}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_r_rx}]

########################################
# pb_otpu_board: the slice side, SLR2
########################################
create_pblock pb_otpu_board
set_property PARENT pblock_CL [get_pblocks pb_otpu_board]
resize_pblock pb_otpu_board -add {CLOCKREGION_X0Y8:CLOCKREGION_X7Y11}
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_board}]
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_xr.g_x[2].u_dn}]
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_xr.g_x[2].u_up}]
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.u_s}]
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[2].u_dn}]
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[2].u_up}]
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/g_rs[2].u_x}]

########################################
# pb_otpu_tmem: the TMEM within the slice (the VPU around it: see the top)
########################################
create_pblock pb_otpu_tmem
set_property PARENT pb_otpu_board [get_pblocks pb_otpu_tmem]
resize_pblock pb_otpu_tmem -add {CLOCKREGION_X2Y9:CLOCKREGION_X5Y10}
add_cells_to_pblock pb_otpu_tmem [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_tmem}]

########################################
# Parameters and containment
########################################
proc otpu_contain {pb} {
  set parent [get_property PARENT [get_pblocks $pb]]
  set held [dict create]
  foreach s [get_sites -of_objects [get_pblocks $parent]] { dict set held $s 1 }
  set extra {}
  foreach s [get_sites -of_objects [get_pblocks $pb]] {
    if {![dict exists $held $s]} { lappend extra $s }
  }
  if {[llength $extra]} { resize_pblock $pb -remove $extra }
  puts "OTPU_FLOORPLAN $pb parent=$parent held=[dict size $held] removed=[llength $extra]"
}
foreach pb {pb_otpu_hbm pb_otpu_mid pb_otpu_pcis pb_otpu_board pb_otpu_tmem} {
  set_property CONTAIN_ROUTING 0 [get_pblocks $pb]
  set_property SNAPPING_MODE OFF [get_pblocks $pb]
  set_property IS_SOFT 0 [get_pblocks $pb]
  otpu_contain $pb
}
puts "OTPU_FLOORPLAN_APPLIED pblocks=5"
