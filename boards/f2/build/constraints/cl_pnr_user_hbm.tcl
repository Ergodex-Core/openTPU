# The HBM build's floorplan, five pblocks (cl_opentpu_hbm, rtl/boards/f2/otpu_f2_hbm_sys.sv;
# docs/v4/f2-timing.md "Round 3", docs/v4/f2-round5.md): boards/f2/build.py --floorplan five (the
# default) appends this file to cl_pnr_user.tcl for an HBM target, which build_level_1_cl.tcl
# reads unmanaged (read_xdc -unmanaged: Tcl, not a managed XDC, which drops proc / if / foreach)
# right before the HDK's opt_design, after the HDK's own small_shell_level_1_fp_cl.xdc, whose
# pblock_CL (the CL's region of the VU47P: all of SLR2, parts of SLR1 and SLR0) is the parent of
# every pblock here. Ranges by complete clock regions, added as the HDK adds its own, then cut
# down to the sites the parent holds (otpu_contain below); the VU47P's SLRs are rows Y0-Y3
# (SLR0, the HBM stacks and their AXI ports below it), Y4-Y7 (SLR1) and Y8-Y11 (SLR2); a clock
# region holds ~13.6K LUTs, ~27K flip-flops, ~21 RAMB36. The alternative, Alex's one SLR0 pblock
# with explicit ranges, is cl_pnr_user_hbm_slr0.tcl (--floorplan slr0).
#
# The f2-hbm CL is one slice (99K LUTs), its adapter (39K, LUT RAM heavy), 8 channel masters
# (~0.5K each), 8 port arbiters, the host path (2K), 8 SmartConnects (the HDK's cl_axi_sc_1x1,
# a few K each: to be read from the first synthesis) and the HBM wrapper (1K). Here:
#   pb_otpu_slr0  (SLR0, the CL's 20 whole clock regions there, ~272K LUTs): the board (the
#                 slice, its adapter, the control block, the trace: u_sys/u_board), the fabric
#                 (the masters, the arbiters, the host path's queues: u_sys/u_mem) and the
#                 SmartConnects (g_sc[*].u_sc) together, beside the HBM's AXI ports under SLR0:
#                 no slice <-> fabric net crosses an SLR, every master sits by its port.
#   pb_otpu_tmem  (a child of pb_otpu_slr0, X0Y2:X3Y3, 8 clock regions, ~109K LUTs): the TMEM
#                 (12K LUTs, 64 RAMB36), the VPU (36K), the DMA (9K) and the quantizer (9K)
#                 together, 61%: the pumped banks' read requests (clk -> clk_f, two fast periods
#                 by otpu_clk2x.tcl) come from the DMA's t_re_q, the VPU's ta_addr / tb_addr,
#                 the MXU's t_raddr and the quantizer's, and a2af0904 placed the DMA's over 4.4
#                 ns of route to a bank's address pins. The MXU's request registers (t_rq, t_ren
#                 and their replicas: the leaf cells, not the MXU) are in the box too since
#                 round 4: 8b2fd3a0 placed u_mxu/t_rq_reg and its _rep copies outside it, 4.1 ns
#                 of route to a bank. The MXU (the slice's rest, ~45K) wraps around the box in
#                 SLR0's other 12 regions with the adapter.
#   pb_otpu_fab   (a child of pb_otpu_slr0, X0Y0:X7Y0, the bottom row, 8 clock regions, ~109K
#                 LUTs): the channel masters, the port arbiters and the SmartConnects by the
#                 HBM's AXI ports under them, each master compact beside its SmartConnect and
#                 that beside its port (round 4's box was the row's left half, X0Y0:X3Y0, for
#                 the 450 MHz bridges; the SmartConnects need the room). The adapter
#                 (u_board/u_mem, 39K, LUT RAM heavy) is not boxed: its loads are the masters
#                 below and the slice above, the placer puts it between. The host block (u_host:
#                 the links' sending ends, the queues, the per-port valids) stays free in SLR0
#                 by its PCIS link stages (pb_otpu_xs); its buses to the arbiters are a core
#                 period's wire each.
#   pb_otpu_pcis  (SLR1, X5Y5:X5Y6, beside the shell's PCIS / OCL column): the host path's
#                 PCIS skid buffers, output registers and link ends (u_host/u_pcis_*, aclk only)
#                 and the OCL crossing (u_ocl: its OCL side faces the shell, its core side holds
#                 one access at a time for the board, a crossing bounded by the core period),
#                 so the shell -> register hop stays in one SLR.
#   pb_otpu_xs    (SLR0, X4Y3, the top corner under the shell's column): the SLR0 ends of the
#                 PCIS links (u_host/u_xs_*), so the SLR1 <-> SLR0 hop runs flip-flop to
#                 flip-flop between pb_otpu_pcis and this box and the stage -> queue paths stay
#                 inside SLR0.
# CONTAIN_ROUTING is off on every pblock (routes may leave the box). The cell queries fail
# closed: a renamed hierarchy stops the build rather than placing freely.
#
# Containment: opt_design's DRC (HDPR-23) requires a child's ranges within its parent's, and
# pblock_CL's are not whole clock regions: the HDK's file takes the shell's columns out, and
# SNAPPING_MODE on a reconfigurable partition's pblock derives site ranges that leave out the
# static shell's sites and the SLR boundaries' LAGUNA rows. So after its clock regions, every
# site of a pblock that its parent does not hold is taken out again (resize_pblock -remove),
# the children in parent-first order, and SNAPPING_MODE stays off on them (a child is not a
# reconfigurable partition; snapping would put the sites back). (rhino's wide-port rounds 3-6
# ran this mechanism on the same shell.)

proc otpu_cells {pattern} {
  set cells [get_cells -quiet $pattern]
  if {![llength $cells]} { error "openTPU floorplan: no cells match $pattern" }
  return $cells
}
set otpu_nport [llength [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_axi}]]
puts "OTPU_FLOORPLAN ports=$otpu_nport"

########################################
# pb_otpu_slr0: the board, the fabric and the SmartConnects, SLR0
########################################
create_pblock pb_otpu_slr0
set_property PARENT pblock_CL [get_pblocks pb_otpu_slr0]
resize_pblock pb_otpu_slr0 -add {CLOCKREGION_X0Y0:CLOCKREGION_X3Y3}
resize_pblock pb_otpu_slr0 -add {CLOCKREGION_X4Y0:CLOCKREGION_X7Y0}
add_cells_to_pblock pb_otpu_slr0 [otpu_cells {WRAPPER/CL/u_sys/u_board}]
add_cells_to_pblock pb_otpu_slr0 [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_axi}]
add_cells_to_pblock pb_otpu_slr0 [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_port}]
add_cells_to_pblock pb_otpu_slr0 [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host}]
add_cells_to_pblock pb_otpu_slr0 [otpu_cells {WRAPPER/CL/g_sc[*].u_sc}]

########################################
# pb_otpu_tmem: the TMEM with the units whose requests reach its pumped banks, inside SLR0
########################################
create_pblock pb_otpu_tmem
set_property PARENT pb_otpu_slr0 [get_pblocks pb_otpu_tmem]
resize_pblock pb_otpu_tmem -add {CLOCKREGION_X0Y2:CLOCKREGION_X3Y3}
add_cells_to_pblock pb_otpu_tmem [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_tmem}]
add_cells_to_pblock pb_otpu_tmem [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_vpu}]
add_cells_to_pblock pb_otpu_tmem [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_dma}]
add_cells_to_pblock pb_otpu_tmem [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_quant}]
add_cells_to_pblock pb_otpu_tmem [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_mxu/t_rq_reg*}]
add_cells_to_pblock pb_otpu_tmem [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_mxu/t_ren_reg*}]

########################################
# pb_otpu_fab: the masters, the arbiters and the SmartConnects by the HBM's ports, SLR0's bottom row
########################################
create_pblock pb_otpu_fab
set_property PARENT pb_otpu_slr0 [get_pblocks pb_otpu_fab]
resize_pblock pb_otpu_fab -add {CLOCKREGION_X0Y0:CLOCKREGION_X7Y0}
add_cells_to_pblock pb_otpu_fab [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_axi}]
add_cells_to_pblock pb_otpu_fab [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_port}]
add_cells_to_pblock pb_otpu_fab [otpu_cells {WRAPPER/CL/g_sc[*].u_sc}]

########################################
# pb_otpu_pcis: the PCIS registers and the OCL crossing beside the shell's column, SLR1
########################################
create_pblock pb_otpu_pcis
set_property PARENT pblock_CL [get_pblocks pb_otpu_pcis]
resize_pblock pb_otpu_pcis -add {CLOCKREGION_X5Y5:CLOCKREGION_X5Y6}
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_aw}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_w}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_ar}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_b}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_r}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_aw_tx}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_w_tx}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_ar_tx}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_b_rx}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_r_rx}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_wb}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_ocl}]

########################################
# pb_otpu_xs: the SLR0 ends of the PCIS links, the top corner of SLR0 under the shell's column
########################################
create_pblock pb_otpu_xs
set_property PARENT pblock_CL [get_pblocks pb_otpu_xs]
resize_pblock pb_otpu_xs -add {CLOCKREGION_X4Y3:CLOCKREGION_X4Y3}
add_cells_to_pblock pb_otpu_xs [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_xs_aw}]
add_cells_to_pblock pb_otpu_xs [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_xs_w}]
add_cells_to_pblock pb_otpu_xs [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_xs_ar}]
add_cells_to_pblock pb_otpu_xs [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_xs_b}]
add_cells_to_pblock pb_otpu_xs [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_xs_r}]
add_cells_to_pblock pb_otpu_xs [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_host/u_xs_wb}]

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
foreach pb {pb_otpu_slr0 pb_otpu_tmem pb_otpu_fab pb_otpu_pcis pb_otpu_xs} {
  set_property CONTAIN_ROUTING 0 [get_pblocks $pb]
  set_property SNAPPING_MODE OFF [get_pblocks $pb]
  set_property IS_SOFT 0 [get_pblocks $pb]
  otpu_contain $pb
}
puts "OTPU_FLOORPLAN_APPLIED pblocks=5"
