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
#   pb_otpu_hbm   (SLR0, the CL's 20 clock regions there + X4Y3): the channels' otpu_native_axi
#                 (u_sys/u_mem/g_port[*].u_naxi) and the SmartConnect fabric (u_sc) by the HBM's
#                 AXI ports, and the SLR0 side of every crossing register chain (rounds 1-8: the
#                 bridges, arbiters and host path, ~30K LUTs at 450 MHz).
#   pb_otpu_mid   (SLR1, X0Y4:X3Y7, 16 clock regions): the middle stage of every chain (~18K
#                 flip-flops, no logic).
#   pb_otpu_pcis  (SLR1, X5Y5:X5Y6, beside the shell's PCIS column): the PCIS gate's skid buffers
#                 (u_mem/u_pcis) and the OCL's crossing (u_ocl; rounds 4-8: u_host's registers).
#   pb_otpu_board (SLR2, all 32 clock regions, ~435K LUTs): u_sys/u_board, the slice with its
#                 adapter, the control block and the trace (226K LUTs, 52%; 416 of the SLR's 672
#                 RAMB36, 1,275 of 3,008 DSPs), and the SLR2 side of the chains. One SLR, so the
#                 adapter's readies into the units, the TMEM's grants into the MXU and the pumped
#                 banks' addresses never cross an SLR. (SLR1's 23 clock regions, ~313K LUTs,
#                 would hold it at 72%: too full for 200 MHz routing.)
#   pb_otpu_tmem  (a child of pb_otpu_board, X2Y9:X5Y10, 8 clock regions, ~109K LUTs): the TMEM
#                 and the VPU together (48K LUTs, 68 RAMB36): the pumped banks' address paths
#                 (clk -> clk_f, one core period: otpu_clk2x.tcl) come from the VPU's ta_addr
#                 and the MXU's t_raddr; the MXU (97K) wraps around the box.
# Since the alex-shell rework (Alex's burst200 memory path, ergodex-openTPU 06c169cb, at 8
# channels; docs/f2_hbm.md): pb_otpu_hbm holds the eight otpu_native_axi (u_mem/g_port[*].u_naxi)
# and the SmartConnect fabric (WRAPPER/CL/u_sc: the eight channel SmartConnects and the host's
# demux) by the HBM's ports; pb_otpu_pcis the PCIS gate (u_mem/u_pcis: its skid buffers, Alex's
# f2_pcis_pipe register slice's place in his pblock_CL_SLR1) with the OCL crossing; pb_otpu_xs
# (round 6's link stages) is gone with otpu_f2_hbm_host and its clock region X4Y3 joins
# pb_otpu_hbm (Alex's pblock_CL_SLR0 holds it too; his partial regions X5Y1:X5Y3, X6Y1 and the
# column ranges beside the shell are not taken: the HBM side's logic is ~a tenth of the rounds'
# 30K LUTs). No logic of ours runs at 450 MHz.
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
set otpu_nport [llength [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_naxi}]]
puts "OTPU_FLOORPLAN ports=$otpu_nport"
# the native ports' chains pumped on clk_f (otpu_f2_hbm_axi NXF > 0, otpu_xr2 g_xf.g_ch[p].u_dn /
# u_up: P3's 16 channels at half the SLR crossings) instead of NXR's core-clock stages (g_xr)
set otpu_xf [get_cells -quiet {WRAPPER/CL/u_sys/u_mem/g_xf.g_ch[*].u_dn}]

########################################
# pb_otpu_hbm: the HBM fabric, SLR0
########################################
create_pblock pb_otpu_hbm
set_property PARENT pblock_CL [get_pblocks pb_otpu_hbm]
resize_pblock pb_otpu_hbm -add {CLOCKREGION_X0Y0:CLOCKREGION_X3Y3}
resize_pblock pb_otpu_hbm -add {CLOCKREGION_X4Y0:CLOCKREGION_X7Y0}
resize_pblock pb_otpu_hbm -add {CLOCKREGION_X4Y3:CLOCKREGION_X4Y3}
# (AWS's own SLR0 outline for its HBM logic, cl_mem_perf's pblock_CL_SLR0: the clock regions
# X5Y1:X5Y3 and X6Y1 and four column ranges beside the shell, next to the HBM stacks' AXI pins)
resize_pblock pb_otpu_hbm -add {CLOCKREGION_X5Y1:CLOCKREGION_X5Y3}
resize_pblock pb_otpu_hbm -add {CLOCKREGION_X6Y1:CLOCKREGION_X6Y1}
resize_pblock pb_otpu_hbm -add {SLICE_X120Y120:SLICE_X145Y179 SLICE_X118Y60:SLICE_X145Y119 SLICE_X206Y60:SLICE_X219Y119 SLICE_X221Y60:SLICE_X232Y119}
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_naxi}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sc}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/g_ch[*].u_rs512}]
if {![llength $otpu_xf]} {
  add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_xr.g_x[0].u_dn}]
  add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_xr.g_x[0].u_up}]
}
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.u_m}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[0].u_dn}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[0].u_up}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/g_rs[0].u_x}]

########################################
# pb_otpu_mid: the chains' middle stage, SLR1
########################################
# With the lane memory's copies (otpu_lane_mem C > 1, below) the box is all of the CL's SLR1 but
# the PCIS box (X5Y5:X5Y6): P3's 9a863228 was not routable ("global congestion level 7", five
# level-7 windows at the SLR2 / SLR1 boundary owned by the pumped chains' SLR2 ends, the eight
# adapters and the MXU lanes, docs/v4/p3-status.md), its 9.2 K crossings all through the LAGUNA
# columns of X0-X3; the copies' 76 K LUTs (45 K LUT RAM) and the chains' SLR2 ends move into SLR1
# (5.6% of its CLBs used) and the crossings spread over SLR1's whole width.
# (build.py's otpu_lmem_slr, the target's lane_memory_slr: 1 for f2-p2's p3c* variants, else 2)
set otpu_lmem {}
if {[info exists otpu_lmem_slr] && $otpu_lmem_slr == 1} {
  set otpu_lmem [get_cells -quiet {WRAPPER/CL/u_sys/u_board/u_mem/g_cp[*].u_adapt}]
  if {[llength $otpu_lmem] < 2} { error "openTPU floorplan: lane_memory_slr 1 without the lane memory's copies" }
}
create_pblock pb_otpu_mid
set_property PARENT pblock_CL [get_pblocks pb_otpu_mid]
if {[llength $otpu_lmem] > 1} {
  resize_pblock pb_otpu_mid -add {CLOCKREGION_X0Y4:CLOCKREGION_X7Y4 CLOCKREGION_X0Y7:CLOCKREGION_X7Y7}
  resize_pblock pb_otpu_mid -add {CLOCKREGION_X0Y5:CLOCKREGION_X4Y6 CLOCKREGION_X6Y5:CLOCKREGION_X7Y6}
  # (the adapters themselves are added below, after u_board's own assignment)
} else {
  resize_pblock pb_otpu_mid -add {CLOCKREGION_X0Y4:CLOCKREGION_X3Y7}
}
if {![llength $otpu_xf]} {
  add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_xr.g_x[1].u_dn}]
  add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_xr.g_x[1].u_up}]
}
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
# (alex-shell: the PCIS gate, otpu_f2_pcis_gate: AW / W / AR / B / R skid buffers and the
# write-pending count, every output a register; its master side meets the SmartConnect fabric's
# host demux in SLR0, as Alex's f2_pcis_pipe in his SLR1 pblock met his memory_sc in SLR0)
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_pcis}]
# (the host chain's first register slice, AWS's AXI4_REG_SLC_HBM position, in the shell's PCIS SLR)
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_hrs512}]
# round 8: the OCL's crossing (otpu_axil_cdc: the shell-facing skid buffers and the aclk state
# machine, its hold registered inside) beside the shell's AXI-Lite master; its core-clock side
# reaches u_cx in SLR0 under the CDC's datapath-only bounds (round 7 placed u_ocl/u_skid_aw ->
# the shell's master over 4.1 ns of route from SLR0)
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_ocl}]

########################################
# pb_otpu_board: the slice side, SLR2
########################################
create_pblock pb_otpu_board
set_property PARENT pblock_CL [get_pblocks pb_otpu_board]
resize_pblock pb_otpu_board -add {CLOCKREGION_X0Y8:CLOCKREGION_X7Y11}
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_board}]
if {![llength $otpu_xf]} {
  add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_xr.g_x[2].u_dn}]
  add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_xr.g_x[2].u_up}]
}
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.u_s}]
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[2].u_dn}]
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[2].u_up}]
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/g_rs[2].u_x}]
# the lane memory's copies (C > 1) in SLR1 (pb_otpu_mid above), after u_board's assignment so
# theirs is the nearer one: the adapters alone; the lane memory's slice-facing registers and gates
# (its readies, idles, the broadcast's request gates) stay with the slice in SLR2, so every
# SLR2 / SLR1 crossing starts or ends at a register of the slice's or the copies' logic
if {[llength $otpu_lmem] > 1} {
  add_cells_to_pblock pb_otpu_mid $otpu_lmem
  puts "OTPU_FLOORPLAN lane memory copies in SLR1: [llength $otpu_lmem] adapters"
}

########################################
# the pumped chains (NXF > 0): each chain's serializer end (its core input register dr and ser) in
# its source's SLR, its NXF clk_f stages
# a third of them an SLR from the serializer's end (two each at NXF 6), its deserializer end in the
# destination's SLR. u_dn: the adapter (pb_otpu_board) -> u_naxi (pb_otpu_hbm); u_up the reverse.
# (Brackets: Vivado takes them literally, so a pattern's are escaped from Tcl, not from get_cells)
########################################
if {[llength $otpu_xf]} {
  set otpu_nxf [llength [otpu_cells "WRAPPER/CL/u_sys/u_mem/g_xf.g_ch\[0\].u_dn/g_x\[*\].u_s"]]
  # (the copies in SLR1: the chains' board ends there too, so a chain crosses SLR1 / SLR0 alone)
  set otpu_bpb [expr {[llength $otpu_lmem] > 1 ? "pb_otpu_mid" : "pb_otpu_board"}]
  foreach {otpu_dir otpu_spb otpu_dpb} [list u_dn $otpu_bpb pb_otpu_hbm u_up pb_otpu_hbm $otpu_bpb] {
    set otpu_b "WRAPPER/CL/u_sys/u_mem/g_xf.g_ch\[*\].$otpu_dir"
    foreach otpu_c {dr_reg* ser_reg* tg_s_reg tgf_s_reg sel_s_reg*} {
      add_cells_to_pblock $otpu_spb [otpu_cells "$otpu_b/$otpu_c"]
    }
    foreach otpu_c {lo_c_reg* hi_f_reg* o_reg* tg_d_reg tgf_d_reg sel_d_reg*} {
      add_cells_to_pblock $otpu_dpb [otpu_cells "$otpu_b/$otpu_c"]
    }
    for {set otpu_k 0} {$otpu_k < $otpu_nxf} {incr otpu_k} {
      set otpu_pb [lindex [list $otpu_spb pb_otpu_mid $otpu_dpb] [expr {$otpu_k * 3 / $otpu_nxf}]]
      add_cells_to_pblock $otpu_pb [otpu_cells "$otpu_b/g_x\[$otpu_k\].u_s"]
    }
  }
  puts "OTPU_FLOORPLAN pumped chains: NXF $otpu_nxf, [llength $otpu_xf] channels"
}

########################################
# pb_otpu_tmem: the TMEM beside the VPU, within the slice
########################################
create_pblock pb_otpu_tmem
set_property PARENT pb_otpu_board [get_pblocks pb_otpu_tmem]
resize_pblock pb_otpu_tmem -add {CLOCKREGION_X2Y9:CLOCKREGION_X5Y10}
add_cells_to_pblock pb_otpu_tmem [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_tmem}]
add_cells_to_pblock pb_otpu_tmem [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice/u_vpu}]

########################################
# the memory path's 450 MHz ends beside their HBM ports, SLR0 (since the first placed build)
########################################
# cd16e195 placed the channels' SmartConnects' 450 MHz side at -1.350 ns, 3.2 of 3.5 ns route
# (sc_c4/inst/m00_nodes/m00_w_node -> m00_exit_pipeline). The HBM's AXI ports sit along SLR0's
# bottom edge: each port's clock region from its site (the HBM IP's AXI interface site,
# BLI_HBM_AXI_INTF_X<p>Y0; if the device names it otherwise, the VU47P's layout, four ports a
# clock region, X<p/4>Y0, with a warning line), and a box of that column's two clock regions
# (Y0:Y1) holds the SmartConnects and the 450 MHz register slices of the ports in it (a child of
# pb_otpu_hbm; the channels' core-clock register slices and otpu_native_axi stay in its parent),
# the host's SmartConnect and 450 MHz slice (u_hsc, u_hrs256) beside its port NPORT the same way.
proc otpu_hbm_port_x {p} {
  set site [get_sites -quiet BLI_HBM_AXI_INTF_X${p}Y0]
  if {[llength $site] == 1} {
    set cr [get_clock_regions -quiet -of_objects $site]
    if {[regexp {^X(\d+)Y0$} [get_property NAME $cr] -> x]} {
      puts "OTPU_FLOORPLAN hbm port $p site $site clock region [get_property NAME $cr]"
      return $x
    }
  }
  set x [expr {$p / 4}]
  puts "OTPU_FLOORPLAN_WARNING hbm port $p: no site BLI_HBM_AXI_INTF_X${p}Y0 (or not in row Y0); X${x}Y0 from the VU47P's four ports a clock region"
  return $x
}
# (the HBM port of channel p: otpu_pstride x p, the host's otpu_hport; build.py sets both from the
# target, hbm_port_stride / hbm_host_port: 1 and NPORT unless a channel window spans pseudo-channels,
# targets/f2-p2.toml's 2 and 15)
if {![info exists otpu_pstride]} { set otpu_pstride 1 }
if {![info exists otpu_hport]} { set otpu_hport $otpu_nport }
puts "OTPU_FLOORPLAN hbm port stride $otpu_pstride host port $otpu_hport"
set otpu_cols {}
for {set otpu_p 0} {$otpu_p <= $otpu_nport} {incr otpu_p} {
  set otpu_x [otpu_hbm_port_x [expr {$otpu_p < $otpu_nport ? $otpu_pstride * $otpu_p : $otpu_hport}]]
  if {$otpu_p < $otpu_nport} {
    dict lappend otpu_cols $otpu_x [otpu_cells "WRAPPER/CL/u_sc/otpu_hbm_sc_i/sc_c$otpu_p"]
    dict lappend otpu_cols $otpu_x [otpu_cells "WRAPPER/CL/g_ch\[$otpu_p\].u_rs256"]   ;# (Vivado takes brackets literally: g_ch[0].u_rs256)
  } else {
    set otpu_hostx $otpu_x
    dict lappend otpu_cols $otpu_x [otpu_cells "WRAPPER/CL/u_hrs256"]
  }
}
set otpu_portbox {}
dict for {otpu_x otpu_cl} $otpu_cols {
  set otpu_pb pb_otpu_hbmx$otpu_x
  create_pblock $otpu_pb
  set_property PARENT pb_otpu_hbm [get_pblocks $otpu_pb]
  resize_pblock $otpu_pb -add "CLOCKREGION_X${otpu_x}Y0:CLOCKREGION_X${otpu_x}Y1"
  add_cells_to_pblock $otpu_pb [concat {*}$otpu_cl]
  lappend otpu_portbox $otpu_pb
  puts "OTPU_FLOORPLAN $otpu_pb X${otpu_x}Y0:X${otpu_x}Y1 cells=[llength $otpu_cl]"
}
# the host's SmartConnect alone in its port's whole column (Y0:Y3, pb_otpu_hsc): cl_axi_sc_1x1's
# 250 MHz entry pipeline at the top beside the crossing from u_hrs512x, its 450 MHz exit at the
# port; the host's 450 MHz register slice u_hrs256 stays in the port's Y0:Y1 box (da7dc503 placed
# the slice up the column: the HBM port's WREADY -> its W pipe's CE at 1.9 ns, -0.041 ns)
create_pblock pb_otpu_hsc
set_property PARENT pb_otpu_hbm [get_pblocks pb_otpu_hsc]
resize_pblock pb_otpu_hsc -add "CLOCKREGION_X${otpu_hostx}Y0:CLOCKREGION_X${otpu_hostx}Y3"
add_cells_to_pblock pb_otpu_hsc [otpu_cells {WRAPPER/CL/u_hsc}]
puts "OTPU_FLOORPLAN pb_otpu_hsc X${otpu_hostx}Y0:X${otpu_hostx}Y3 cells=1"
# the host chain's second register slice (u_hrs512x) in SLR0's top row under the shell's column:
# the SLR crossing from u_hrs512 (pb_otpu_pcis, X5Y5:X5Y6) is one hop into it, the descent to
# the host's column one more (the first 150 MHz build placed u_hrs512 -> the SmartConnect's
# entry FIFO at the bottom of SLR0 at 4.1 ns of route on the 4.0 ns budget: -1.42 ns)
create_pblock pb_otpu_hxs
set_property PARENT pb_otpu_hbm [get_pblocks pb_otpu_hxs]
resize_pblock pb_otpu_hxs -add {CLOCKREGION_X4Y3:CLOCKREGION_X4Y3}
add_cells_to_pblock pb_otpu_hxs [otpu_cells {WRAPPER/CL/u_hrs512x}]
puts "OTPU_FLOORPLAN pb_otpu_hxs X4Y3 cells=1"
# (the host chain's first register slice u_hrs512 sits with the PCIS gate in the shell's PCIS SLR,
# pb_otpu_pcis, below: the SLR crossing runs from its registers to cl_axi_sc_1x1's entry register
# slice beside port NPORT, flop to flop, as the AWS Shell Interface Specification asks)

########################################
# the shell's PCIS boundary: in the SLR of pb_otpu_pcis, found from the linked shell's cells
########################################
# The specification's table puts PCIS in the top SLR; round 3's placed report showed the shell's
# PCIS register at SLICE_X176Y399 (SLR1). Ask the linked design: the cell driving the CL's
# sh_cl_dma_pcis_awvalid input, its SLR; it must be pb_otpu_pcis's (fail closed: none found, or
# another SLR, stops the build with the SLR named).
set otpu_pcis_net [get_nets -quiet WRAPPER/CL/sh_cl_dma_pcis_awvalid]
set otpu_pcis_drv [get_cells -quiet -of_objects [get_pins -quiet -leaf -of_objects $otpu_pcis_net -filter {DIRECTION == OUT}]]
set otpu_pcis_slr [get_slrs -quiet -of_objects $otpu_pcis_drv]
set otpu_box_slr [get_slrs -quiet -of_objects [get_clock_regions -quiet X5Y5]]
if {[llength $otpu_pcis_slr] != 1 || [llength $otpu_box_slr] != 1} {
  error "openTPU floorplan: the shell's PCIS driver of sh_cl_dma_pcis_awvalid: cell '$otpu_pcis_drv', SLR '$otpu_pcis_slr' (pb_otpu_pcis: '$otpu_box_slr')"
}
if {[get_property NAME $otpu_pcis_slr] ne [get_property NAME $otpu_box_slr]} {
  error "openTPU floorplan: the shell's PCIS is in [get_property NAME $otpu_pcis_slr] ($otpu_pcis_drv), pb_otpu_pcis in [get_property NAME $otpu_box_slr]"
}
puts "OTPU_FLOORPLAN shell PCIS [get_property NAME $otpu_pcis_slr] ($otpu_pcis_drv): pb_otpu_pcis there"

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
foreach pb [concat {pb_otpu_hbm} $otpu_portbox {pb_otpu_hsc pb_otpu_hxs pb_otpu_mid pb_otpu_pcis pb_otpu_board pb_otpu_tmem}] {
  set_property CONTAIN_ROUTING 0 [get_pblocks $pb]
  set_property SNAPPING_MODE OFF [get_pblocks $pb]
  set_property IS_SOFT 0 [get_pblocks $pb]
  otpu_contain $pb
}
puts "OTPU_FLOORPLAN_APPLIED pblocks=[expr {7 + [llength $otpu_portbox]}]"

########################################
# the core clock and the pump clock on one insertion delay (round 8)
########################################
# The TMEM's pumped RAMs take their read requests from clk (200 MHz, the whole design's tree:
# ~940K endpoints over the three SLRs) into clk_f (400 MHz, the TMEM's 3.7K) with two fast
# periods (otpu_clk2x.tcl). Round 7 placed the two trees 1.4 ns apart (source clock delay 5.70
# ns, destination 4.32) against those paths. Both clocks are the CL's own MMCM's outputs (u_mmcm
# CLKOUT0 / 6 and CLKOUT1 / 3, phase aligned) through two BUFGs (u_clk, u_clk_f): a clock delay
# group makes the placer put their roots together and match the trees' insertion delays
# (UltraScale clocking: CLOCK_DELAY_GROUP on the nets of buffers in one clock region). Fail
# closed: both nets must be found.
set otpu_ck_nets [get_nets -quiet -of_objects [get_pins -quiet {WRAPPER/CL/u_clk/O WRAPPER/CL/u_clk_f/O}]]
if {[llength $otpu_ck_nets] != 2} {
  error "openTPU floorplan: the core and pump clock nets (u_clk/O, u_clk_f/O): [llength $otpu_ck_nets] found, expected 2"
}
set_property CLOCK_DELAY_GROUP otpu_core_pump $otpu_ck_nets
puts "OTPU_FLOORPLAN clock delay group otpu_core_pump on $otpu_ck_nets"
