# The v5 (full HBM) build's floorplan (cl_opentpu_hbm5, rtl/boards/f2/otpu_f2_hbm_eng.sv;
# docs/full_hbm.md, the 2026-10-20 spec section 6). boards/f2/build.py appends this file to
# cl_pnr_user.tcl for an HBM target whose platform mem_path is "engine", in place of
# cl_pnr_user_hbm.tcl; build_level_1_cl.tcl reads it unmanaged right before the HDK's
# opt_design (a managed XDC drops proc / if / foreach), after the HDK's
# small_shell_level_1_fp_cl.xdc, whose pblock_CL is the parent of every pblock here. The
# mechanisms are cl_pnr_user_hbm.tcl's (whole clock regions cut down to the parent, otpu_contain;
# the HBM port columns from the ports' sites; the shell's PCIS SLR checked; the clock delay group
# of the core and pump clocks), the contents v5's. The CL's outline (AWS's small shell, the
# HDK's pblock_CL): SLR0 X0Y0:X3Y3, X4Y0:X7Y0, X4Y3, X5Y1:X5Y3, X6Y1; SLR1 X0Y4:X3Y7, X4Y4,
# X4Y6:X4Y7, X5Y4:X5Y7; SLR2 X0Y8:X7Y11 (a clock region ~13.6K LUT, 24 RAMB36, 96 DSP; the
# bottom row less).
#   A pblock per slice holding its otpu_slice and its adapter (g_slice[s].u_mem) with the TMEM
#   box inside it, so no core-clock or pump path spans more than the slice's box (the first
#   placed v5 build, 340c4ed6, had slices 2 and 3 in one SLR2 pblock: the adapters' replicated
#   registers 8 ns of route across the die, the DMA 7 ns from its TMEM): slice 0 (u_board/u_slice,
#   u_board/u_mem) in pb_otpu_hbm, SLR0, with the 32 engines (u_mem/g_port[*].u_eng), the 450 MHz
#   register slices and the host chain by the HBM's ports; slice 1 (u_board/g_slice[1]) in
#   pb_otpu_mid, SLR1's CL area, with the collective's hub (u_board/g_xs.u_chub), the control
#   block and the trace (so every hub link crosses one boundary at most); slice 2 in pb_otpu_s2,
#   SLR2's left half (X0Y8:X3Y11; the whole SLR at S = 3), slice 3 in pb_otpu_s3, its right half
#   (X4Y8:X7Y11). A slice with its adapter is ~185K LUT / ~266 RAMB36 / 727 DSP of a half's
#   ~218K / 336 / 1,536 (85% / 79% / 47%; docs/full_hbm.md "Utilization").
#   The native ports' chains (u_mem/g_xr.g_x[k].u_dn / u_up, one register vector over every
#   port) and the write buses (g_xw.g_x[k].u_wb) are placed bit by bit: port p's bits of stage k
#   go to SLR0 (k = 0), to SLR1 (k = 1, a slice at or above SLR1; slice 3's in SLR1's right
#   columns, pb_otpu_mid3, under its half) or SLR0 (k = 1, slice 0), to the slice's pblock
#   (k = 2); the hub links' tx / rx stages with the hub and the slice they serve.
#   A pumped-TMEM box per slice (pb_otpu_tmem<s>) inside the slice's pblock: the TMEM, the VPU
#   and the DMA (340c4ed6 placed slice 3's DMA 7 ns from its TMEM: clk -> clk_f at -3.6 ns).
#   The host chain (wide-port-alexshell deb7ec12 / ba86b5df, which placed at +0.09): u_hrs512
#   with the PCIS gate in SLR1, u_hrs512x in SLR0's top row beside the crossing (pb_otpu_hxs,
#   X4Y3), the host SmartConnect boxed toward its port's column (pb_otpu_hsc), its 450 MHz
#   register slice u_hrs256 and the mux in the port's Y0:Y1 box.
# Every pblock fails closed: a renamed hierarchy stops the build rather than placing freely.

proc otpu_cells {pattern} {
  set cells [get_cells -quiet $pattern]
  if {[llength $cells] == 0} {
    error "openTPU floorplan: no cells match '$pattern' (the hierarchy moved?)"
  }
  return $cells
}
# an engine's leaf cells without its RAMs and its read path: the right half's port columns
# (X4..X7) hold one clock region of CL area beside the HBM ports (AWS's SLR0 outline: X4Y0:X7Y0,
# X5Y1:X5Y3, X6Y1), and 4-5 engines' 28-35 RAMB36 did not fit in one (the first v5 ImplCL: DRC
# UTLZ-1 in pb_otpu_hbmx4/5/6; the read FIFOs are in URAM since, otpu_hbm_rpath), so the
# engines' memories float in pb_otpu_hbm and their logic stays by the port. Fails closed on an
# engine with no leaf cells.
proc otpu_leaves_no_bram {cell} {
  set out {}
  foreach c [get_cells -quiet "$cell/*"] {
    # the read path (otpu_hbm_rpath u_rp: the URAM FIFO, the pairs, the crossing FIFO's
    # distributed RAM) floats with its URAMs: its input and output are registers
    if {[string match */u_rp $c]} continue
    if {[get_property IS_PRIMITIVE $c]} {
      if {![string match BLOCKRAM* [get_property PRIMITIVE_TYPE $c]]} { lappend out $c }
    } else {
      lappend out {*}[otpu_leaves_no_bram $c]
    }
  }
  return $out
}
proc otpu_engine_cells {pattern x} {
  # (every column since the third v5 ImplCL: the macro placer found the left half's boxes at
  # 102-126% of their block RAM too, four engines' 28 RAMB36 in two bottom-row clock regions)
  set cells {}
  foreach e [otpu_cells $pattern] { lappend cells {*}[otpu_leaves_no_bram $e] }
  if {[llength $cells] == 0} { error "openTPU floorplan: no leaf cells under '$pattern' (the hierarchy moved?)" }
  return $cells
}
# the leaf registers r_reg[i] of an otpu_xr stage with lo <= i < hi (a stage's register vector
# holds every port's bits: port p's are p * W .. (p + 1) * W - 1)
proc otpu_bits {stage lo hi} {
  set cells {}
  foreach c [get_cells -quiet "$stage/r_reg\[*\]"] {
    if {[regexp {r_reg\[(\d+)\]$} [get_property NAME $c] -> i] && $i >= $lo && $i < $hi} {
      lappend cells $c
    }
  }
  if {[llength $cells] == 0} {
    error "openTPU floorplan: no registers of '$stage' in \[$lo, $hi) (the hierarchy or the width moved?)"
  }
  return $cells
}
set otpu_nport [llength [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_eng}]]
# the geometry: S slices of NCH channels, the stage widths (otpu_f2_hbm_eng's DW / UW / BW)
set otpu_S [llength [get_cells -quiet {WRAPPER/CL/u_sys/u_board/g_slice[*].u_slice}]]
incr otpu_S
set otpu_nch [expr {$otpu_nport / $otpu_S}]
set otpu_dw 27
set otpu_uw 517
set otpu_bw 580
puts "OTPU_FLOORPLAN ports=$otpu_nport slices=$otpu_S nch=$otpu_nch"
# a slice's SLR: 0, 1, 2, 2; its pblock; its otpu_slice instance (u_board/u_slice, or
# u_board/g_slice[s].u_slice); the pblock of its chains' middle stage
proc otpu_slice_slr {s} { return [expr {$s >= 2 ? 2 : $s}] }
proc otpu_slice_pb {s} {
  switch $s { 0 { return pb_otpu_hbm } 1 { return pb_otpu_mid } default { return pb_otpu_s$s } }
}
proc otpu_slice_path {s} {
  return [expr {$s == 0 ? "WRAPPER/CL/u_sys/u_board/u_slice" : "WRAPPER/CL/u_sys/u_board/g_slice\[$s\].u_slice"}]
}
proc otpu_mid_pb {s} {
  switch $s { 0 { return pb_otpu_hbm } 3 { return pb_otpu_mid3 } default { return pb_otpu_mid } }
}

########################################
# pb_otpu_hbm: SLR0: slice 0, the engines, the host chain, the chains' SLR0 stages
########################################
create_pblock pb_otpu_hbm
set_property PARENT pblock_CL [get_pblocks pb_otpu_hbm]
resize_pblock pb_otpu_hbm -add {CLOCKREGION_X0Y0:CLOCKREGION_X3Y3}
resize_pblock pb_otpu_hbm -add {CLOCKREGION_X4Y0:CLOCKREGION_X7Y0}
resize_pblock pb_otpu_hbm -add {CLOCKREGION_X4Y3:CLOCKREGION_X4Y3}
resize_pblock pb_otpu_hbm -add {CLOCKREGION_X5Y1:CLOCKREGION_X5Y3}
resize_pblock pb_otpu_hbm -add {CLOCKREGION_X6Y1:CLOCKREGION_X6Y1}
resize_pblock pb_otpu_hbm -add {SLICE_X120Y120:SLICE_X145Y179 SLICE_X118Y60:SLICE_X145Y119 SLICE_X206Y60:SLICE_X219Y119 SLICE_X221Y60:SLICE_X232Y119}
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_port[*].u_eng}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_board/u_slice}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_board/u_mem}]
add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[0].u_dn}]
if {$otpu_S > 1} {
  # slice 0's hub link: its rx / tx and port with the slice, the hub's ends in SLR1 (below)
  add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_board/g_xs.g_x[0].u_dn_rx}]
  add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_board/g_xs.g_x[0].u_up_tx}]
  add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_board/g_xs.g_x[0].u_cport}]
  add_cells_to_pblock pb_otpu_hbm [otpu_cells {WRAPPER/CL/u_sys/u_board/g_xs.g_x[0].g_pf.u_pf_tx}]
}

########################################
# pb_otpu_mid: SLR1's CL area: slice 1, the hub, the control block, the trace, the chains'
# middle stages (slice 3's in pb_otpu_mid3, the right columns under its half of SLR2)
########################################
create_pblock pb_otpu_mid
set_property PARENT pblock_CL [get_pblocks pb_otpu_mid]
resize_pblock pb_otpu_mid -add {CLOCKREGION_X0Y4:CLOCKREGION_X3Y7}
# (the CL's columns beside the shell, X4Y4, X4Y6:X4Y7, X5Y4 and X5Y7; X5Y5:X5Y6 is pb_otpu_pcis)
set otpu_mid_right {CLOCKREGION_X4Y4:CLOCKREGION_X4Y4 CLOCKREGION_X4Y6:CLOCKREGION_X4Y7 CLOCKREGION_X5Y4:CLOCKREGION_X5Y4 CLOCKREGION_X5Y7:CLOCKREGION_X5Y7}
foreach otpu_r $otpu_mid_right { resize_pblock pb_otpu_mid -add $otpu_r }
add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_board/u_ctrl}]
add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_board/g_trace.u_trace}]
add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[1].u_dn}]
if {$otpu_S > 1} {
  add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_board/g_xs.u_chub}]
  add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_board/g_xs.g_x[*].u_dn_tx}]
  add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_board/g_xs.g_x[*].u_up_rx}]
  add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_board/g_xs.g_x[0].g_pf.u_pf_rx}]
  add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_board/g_slice[1].u_slice}]
  add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_board/g_slice[1].u_mem}]
  add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_board/g_xs.g_x[1].u_dn_rx}]
  add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_board/g_xs.g_x[1].u_up_tx}]
  add_cells_to_pblock pb_otpu_mid [otpu_cells {WRAPPER/CL/u_sys/u_board/g_xs.g_x[1].u_cport}]
}
set otpu_mid3 {}
if {$otpu_S > 3} {
  create_pblock pb_otpu_mid3
  set_property PARENT pb_otpu_mid [get_pblocks pb_otpu_mid3]
  foreach otpu_r $otpu_mid_right { resize_pblock pb_otpu_mid3 -add $otpu_r }
  set otpu_mid3 pb_otpu_mid3
}

########################################
# pb_otpu_pcis: the PCIS gate, the host chain's first register slice and the OCL crossing
# beside the shell, SLR1 (cl_pnr_user_hbm.tcl's)
########################################
create_pblock pb_otpu_pcis
set_property PARENT pblock_CL [get_pblocks pb_otpu_pcis]
resize_pblock pb_otpu_pcis -add {CLOCKREGION_X5Y5:CLOCKREGION_X5Y6}
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_mem/u_pcis}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_hrs512}]
add_cells_to_pblock pb_otpu_pcis [otpu_cells {WRAPPER/CL/u_sys/u_ocl}]

########################################
# pb_otpu_board: SLR2: the chains' last stages; a pblock per slice inside it (pb_otpu_s2 the
# left half, or the whole SLR at S = 3; pb_otpu_s3 the right half)
########################################
create_pblock pb_otpu_board
set_property PARENT pblock_CL [get_pblocks pb_otpu_board]
resize_pblock pb_otpu_board -add {CLOCKREGION_X0Y8:CLOCKREGION_X7Y11}
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.u_s}]
add_cells_to_pblock pb_otpu_board [otpu_cells {WRAPPER/CL/u_sys/u_cx/g_xr.g_x[2].u_dn}]
set otpu_sboxes {}
for {set otpu_s 2} {$otpu_s < $otpu_S} {incr otpu_s} {
  set otpu_pb [otpu_slice_pb $otpu_s]
  create_pblock $otpu_pb
  set_property PARENT pb_otpu_board [get_pblocks $otpu_pb]
  set otpu_range [expr {$otpu_S == 3 ? "CLOCKREGION_X0Y8:CLOCKREGION_X7Y11" : ($otpu_s == 2 ? "CLOCKREGION_X0Y8:CLOCKREGION_X3Y11" : "CLOCKREGION_X4Y8:CLOCKREGION_X7Y11")}]
  resize_pblock $otpu_pb -add $otpu_range
  add_cells_to_pblock $otpu_pb [otpu_cells "WRAPPER/CL/u_sys/u_board/g_slice\[$otpu_s\].u_slice"]
  add_cells_to_pblock $otpu_pb [otpu_cells "WRAPPER/CL/u_sys/u_board/g_slice\[$otpu_s\].u_mem"]
  add_cells_to_pblock $otpu_pb [otpu_cells "WRAPPER/CL/u_sys/u_board/g_xs.g_x\[$otpu_s\].u_dn_rx"]
  add_cells_to_pblock $otpu_pb [otpu_cells "WRAPPER/CL/u_sys/u_board/g_xs.g_x\[$otpu_s\].u_up_tx"]
  add_cells_to_pblock $otpu_pb [otpu_cells "WRAPPER/CL/u_sys/u_board/g_xs.g_x\[$otpu_s\].u_cport"]
  lappend otpu_sboxes $otpu_pb
  puts "OTPU_FLOORPLAN $otpu_pb $otpu_range slice $otpu_s"
}

########################################
# the native ports' chains and the write buses, stage by stage, port by port
########################################
# stage k of port p (slice s = p / NCH): k = 0 in SLR0 (the engines'); k = 1 in SLR1 for a slice
# at SLR1 or above (slice 3's in pb_otpu_mid3), else SLR0; k = 2 (the slice's end) in the
# slice's pblock. The write bus of slice s: its NXR stages likewise.
for {set otpu_p 0} {$otpu_p < $otpu_nport} {incr otpu_p} {
  set otpu_s [expr {$otpu_p / $otpu_nch}]
  foreach otpu_k {0 1 2} {
    set otpu_pb [expr {$otpu_k == 0 ? "pb_otpu_hbm" : ($otpu_k == 1 ? [otpu_mid_pb $otpu_s] : [otpu_slice_pb $otpu_s])}]
    add_cells_to_pblock $otpu_pb [otpu_bits "WRAPPER/CL/u_sys/u_mem/g_xr.g_x\[$otpu_k\].u_dn" [expr {$otpu_p * $otpu_dw}] [expr {($otpu_p + 1) * $otpu_dw}]]
    add_cells_to_pblock $otpu_pb [otpu_bits "WRAPPER/CL/u_sys/u_mem/g_xr.g_x\[$otpu_k\].u_up" [expr {$otpu_p * $otpu_uw}] [expr {($otpu_p + 1) * $otpu_uw}]]
  }
}
for {set otpu_s 0} {$otpu_s < $otpu_S} {incr otpu_s} {
  foreach otpu_k {0 1 2} {
    set otpu_pb [expr {$otpu_k == 0 ? "pb_otpu_hbm" : ($otpu_k == 1 ? [otpu_mid_pb $otpu_s] : [otpu_slice_pb $otpu_s])}]
    add_cells_to_pblock $otpu_pb [otpu_bits "WRAPPER/CL/u_sys/u_mem/g_xw.g_x\[$otpu_k\].u_wb" [expr {$otpu_s * $otpu_bw}] [expr {($otpu_s + 1) * $otpu_bw}]]
  }
}

########################################
# the pumped TMEM beside its VPU and its DMA, one box a slice (within the slice's pblock)
########################################
# SLR0: X2Y1:X3Y3 (the HBM columns' upper rows); SLR1: X1Y5:X3Y6; SLR2: X0Y9:X3Y10 (slice 2),
# X4Y9:X7Y10 (slice 3). The slice's MXU wraps around the box as v4's did (pb_otpu_tmem). The
# DMA is in the box: its registers into the pumped ports are clk -> clk_f paths with one core
# period (otpu_clk2x.tcl; 340c4ed6 placed slice 3's 7 ns away).
set otpu_tboxes [list {CLOCKREGION_X2Y1:CLOCKREGION_X3Y3} {CLOCKREGION_X1Y5:CLOCKREGION_X3Y6} {CLOCKREGION_X0Y9:CLOCKREGION_X3Y10} {CLOCKREGION_X4Y9:CLOCKREGION_X7Y10}]   ;# slice 0's box three rows tall: at X2Y1:X3Y2 its 68 RAMB36 were 119% of the box
set otpu_tmems {}
for {set otpu_s 0} {$otpu_s < $otpu_S} {incr otpu_s} {
  set otpu_pb pb_otpu_tmem$otpu_s
  create_pblock $otpu_pb
  set_property PARENT [otpu_slice_pb $otpu_s] [get_pblocks $otpu_pb]
  resize_pblock $otpu_pb -add [lindex $otpu_tboxes $otpu_s]
  add_cells_to_pblock $otpu_pb [otpu_cells "[otpu_slice_path $otpu_s]/u_tmem"]
  add_cells_to_pblock $otpu_pb [otpu_cells "[otpu_slice_path $otpu_s]/u_vpu"]
  add_cells_to_pblock $otpu_pb [otpu_cells "[otpu_slice_path $otpu_s]/u_dma"]
  lappend otpu_tmems $otpu_pb
}

########################################
# the 450 MHz ends beside their HBM ports, SLR0 (cl_pnr_user_hbm.tcl's columns)
########################################
# Each port's clock region from its site (BLI_HBM_AXI_INTF_X<p>Y0; else four ports a
# region); a box of that column's two clock regions holds the port's engine (its 450 MHz
# side with its core side: ~1.4K LUTs) and the HDK's 450 MHz register slice; the host's
# 450 MHz register slice and the mux (S x NCH = 32) beside its port.
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
set otpu_host_mux [expr {[llength [get_cells -quiet {WRAPPER/CL/u_sys/u_mem/g_mux.u_mux}]] > 0}]
set otpu_host_port [expr {$otpu_host_mux ? $otpu_nport - 1 : $otpu_nport}]
set otpu_cols {}
for {set otpu_p 0} {$otpu_p < $otpu_nport} {incr otpu_p} {
  set otpu_x [otpu_hbm_port_x $otpu_p]
  dict lappend otpu_cols $otpu_x [otpu_engine_cells "WRAPPER/CL/u_sys/u_mem/g_port\[$otpu_p\].u_eng" $otpu_x]
  dict lappend otpu_cols $otpu_x [otpu_cells "WRAPPER/CL/g_ch\[$otpu_p\].u_rs256"]
}
set otpu_hostx [otpu_hbm_port_x $otpu_host_port]
dict lappend otpu_cols $otpu_hostx [otpu_cells {WRAPPER/CL/u_hrs256}]
if {$otpu_host_mux} { dict lappend otpu_cols $otpu_hostx [otpu_cells {WRAPPER/CL/u_sys/u_mem/g_mux.u_mux}] }
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
puts "OTPU_FLOORPLAN host port $otpu_host_port mux=$otpu_host_mux"
# the host's SmartConnect boxed toward its port's column (pb_otpu_hsc): the column's full
# height, and, for a port in the right half (where the CL has only the bottom row of its
# column), the right half's CL regions above it (X5Y1:X5Y3, X6Y0:X6Y1): cl_axi_sc_1x1's 250 MHz
# entry pipeline at the top beside the crossing from u_hrs512x, its 450 MHz exit at the port;
# u_hrs256 stays in the port's Y0:Y1 box (ba86b5df: the slice up the column placed the port's
# WREADY -> its W pipe's CE at 1.9 ns)
create_pblock pb_otpu_hsc
set_property PARENT pb_otpu_hbm [get_pblocks pb_otpu_hsc]
resize_pblock pb_otpu_hsc -add "CLOCKREGION_X${otpu_hostx}Y0:CLOCKREGION_X${otpu_hostx}Y3"
if {$otpu_hostx >= 4} {
  resize_pblock pb_otpu_hsc -add {CLOCKREGION_X5Y1:CLOCKREGION_X5Y3}
  resize_pblock pb_otpu_hsc -add {CLOCKREGION_X6Y0:CLOCKREGION_X6Y1}
}
add_cells_to_pblock pb_otpu_hsc [otpu_cells {WRAPPER/CL/u_hsc}]
puts "OTPU_FLOORPLAN pb_otpu_hsc X${otpu_hostx}Y0:X${otpu_hostx}Y3[expr {$otpu_hostx >= 4 ? " + X5Y1:X5Y3 X6Y0:X6Y1" : ""}] cells=1"
# the host chain's second register slice (u_hrs512x) in SLR0's top row under the shell's column:
# the SLR crossing from u_hrs512 (pb_otpu_pcis, X5Y5:X5Y6) is one hop into it, the descent to
# the host's column one more (deb7ec12: the first 150 MHz build placed u_hrs512 -> the
# SmartConnect's entry FIFO at 4.1 ns of route; 340c4ed6 the same at 4.06 ns)
create_pblock pb_otpu_hxs
set_property PARENT pb_otpu_hbm [get_pblocks pb_otpu_hxs]
resize_pblock pb_otpu_hxs -add {CLOCKREGION_X4Y3:CLOCKREGION_X4Y3}
add_cells_to_pblock pb_otpu_hxs [otpu_cells {WRAPPER/CL/u_hrs512x}]
puts "OTPU_FLOORPLAN pb_otpu_hxs X4Y3 cells=1"

########################################
# the shell's PCIS boundary: in the SLR of pb_otpu_pcis (cl_pnr_user_hbm.tcl's check)
########################################
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
  if {![llength [get_sites -of_objects [get_pblocks $pb]]]} { error "openTPU floorplan: $pb has no site inside $parent" }
  puts "OTPU_FLOORPLAN $pb parent=$parent held=[dict size $held] removed=[llength $extra]"
}
# (parents first: a child is cut to its parent as already cut)
set otpu_all [concat {pb_otpu_hbm} $otpu_portbox {pb_otpu_hsc pb_otpu_hxs pb_otpu_mid} $otpu_mid3 {pb_otpu_pcis pb_otpu_board} $otpu_sboxes $otpu_tmems]
foreach pb $otpu_all {
  set_property CONTAIN_ROUTING 0 [get_pblocks $pb]
  set_property SNAPPING_MODE OFF [get_pblocks $pb]
  set_property IS_SOFT 0 [get_pblocks $pb]
  otpu_contain $pb
}
puts "OTPU_FLOORPLAN_APPLIED pblocks=[llength $otpu_all]"

########################################
# the core clock and the pump clock on one insertion delay (cl_pnr_user_hbm.tcl's)
########################################
set otpu_ck_nets [get_nets -quiet -of_objects [get_pins -quiet {WRAPPER/CL/u_clk/O WRAPPER/CL/u_clk_f/O}]]
if {[llength $otpu_ck_nets] != 2} {
  error "openTPU floorplan: the core and pump clock nets (u_clk/O, u_clk_f/O): [llength $otpu_ck_nets] found, expected 2"
}
set_property CLOCK_DELAY_GROUP otpu_core_pump $otpu_ck_nets
puts "OTPU_FLOORPLAN clock delay group otpu_core_pump on $otpu_ck_nets"
