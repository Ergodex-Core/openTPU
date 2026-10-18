# The paths between core_clk and core_clk2x: the block design's MMCM output at twice core_clk,
# from the same VCO by integer dividers and rising with it (bd_native.tcl / the BCU1525's bd.tcl,
# CLK2X 1), the RTL's clk_f. One file for every user of the 2x clock, one section each, shared by
# the boards (create_project.tcl, with CLK2X 1) and the out-of-context runs
# (rtl/exp/tmem_nocopy/ooc.tcl, the pumped one copy). Unmanaged Tcl, read in implementation only
# and late (PROCESSING_ORDER LATE), on the synthesized netlist (the F2 build reads it after
# synthesis and again after the link, inside its opt_design wrapper: an exception read into the
# out-of-context checkpoint did not survive the link, docs/wide_port.md "Round 5 RTL"); not
# scoped: each section finds its cells by their module (kept hierarchies).

# ---------------------------------------------------------------- the TMEM's pumped RAMs
# otpu_pump_bank (rtl/mem/otpu_tmem.sv; otpu_tmem IMPL 3 TDP with PMP 1): a true dual-port RAM on
# clk_f. Its reads' requests (ra / rena, rb / renb) come from core_clk and are taken at the even
# fast edge, the next core edge: one core period (2 fast, -end). The words read (qa / qb) change
# only after an even edge and hold through the next core cycle: one core period to core_clk (2
# fast, -start). The write (wv / wa / wd) lands at the odd fast edge, half a core period after the
# core edge that launched it, and the phase (tog -> tq / evr) at the fast edge after the core
# edge: one fast period each, timed as such by default.
# The cells are found by module (REF_NAME / ORIG_REF_NAME) and cross-checked by name, fail
# closed as the floorplan does: every cell named *g_pmp*u_pb (one per pumped RAM piece, otpu_tmem
# g_pmp.g_pc[*].u_pb since round 4's pieces; g_pmp.u_pb before them) must be among them, and so
# must the parent of every block RAM primitive under such a name (a flattened piece would leave
# its primitives with the name and no cell to take the pins); a mismatch is an error naming the
# three counts. Then the exceptions are checked on the design: the worst path through the
# read-request pins must report two fast periods as its requirement (round 4's placement, with
# the file read at synthesis only, placed u_dma/t_re_q_reg -> a piece's ADDRBWRADDR at the single
# period: docs/wide_port.md "Round 5 RTL"; the F2 link wrapper now reads it again after the
# link, boards/f2/build/scripts/build_level_1_cl.tcl).
set c2x_pb [get_cells -quiet -hierarchical -filter {ORIG_REF_NAME == otpu_pump_bank || REF_NAME == otpu_pump_bank}]
set c2x_pn [get_cells -quiet -hierarchical -filter {NAME =~ *g_pmp*u_pb}]
set c2x_br [get_cells -quiet -hierarchical -filter {PRIMITIVE_TYPE =~ BLOCKRAM.* && NAME =~ *g_pmp*u_pb/*}]
set c2x_bp {}
foreach c2x_c $c2x_br {
  set c2x_par [get_property -quiet PARENT $c2x_c]
  if {[lsearch -exact $c2x_bp $c2x_par] < 0} { lappend c2x_bp $c2x_par }
}
if {[llength $c2x_pb] || [llength $c2x_pn] || [llength $c2x_br]} {
  set c2x_names_m [lsort [get_property NAME $c2x_pb]]
  set c2x_names_n [lsort [get_property NAME $c2x_pn]]
  set c2x_names_b [lsort $c2x_bp]
  if {$c2x_names_m ne $c2x_names_n || $c2x_names_m ne $c2x_names_b} {
    error "\[otpu_clk2x.tcl\] pumped TMEM RAM cells: [llength $c2x_pb] by module, [llength $c2x_pn] by name (*g_pmp*u_pb), [llength $c2x_bp] holding the [llength $c2x_br] block RAM primitives named under one: the three must agree (every piece a kept otpu_pump_bank cell)"
  }
}
if {[llength $c2x_pb]} {
  set c2x_c0 [lindex $c2x_pb 0]
  set c2x_cs [get_clocks -quiet -of_objects [get_pins -quiet $c2x_c0/clk]]
  set c2x_cf [get_clocks -quiet -of_objects [get_pins -quiet $c2x_c0/clk_f]]
  set c2x_ts [get_property -quiet -min PERIOD $c2x_cs]
  set c2x_tf [get_property -quiet -min PERIOD $c2x_cf]
  if {[llength $c2x_cs] != 1 || [llength $c2x_cf] != 1 || abs($c2x_ts - 2.0 * $c2x_tf) > 0.001} {
    puts "CRITICAL WARNING: \[otpu_clk2x.tcl\] the pumped RAMs' clocks are '$c2x_cs' ($c2x_ts ns) and '$c2x_cf' ($c2x_tf ns), expected one each, the second at twice the first: their reads unconstrained"
  } else {
    set c2x_rp [get_pins -quiet -of_objects $c2x_pb -filter {REF_PIN_NAME =~ ra[*] || REF_PIN_NAME =~ rb[*] || REF_PIN_NAME == rena || REF_PIN_NAME == renb}]
    set c2x_qp [get_pins -quiet -of_objects $c2x_pb -filter {REF_PIN_NAME =~ qa[*] || REF_PIN_NAME =~ qb[*]}]
    set_multicycle_path 2 -setup -end -from $c2x_cs -through $c2x_rp -to $c2x_cf
    set_multicycle_path 1 -hold -end -from $c2x_cs -through $c2x_rp -to $c2x_cf
    set_multicycle_path 2 -setup -start -from $c2x_cf -through $c2x_qp -to $c2x_cs
    set_multicycle_path 1 -hold -start -from $c2x_cf -through $c2x_qp -to $c2x_cs
    # the check: the worst read-request path reports two fast periods
    set c2x_chk [get_timing_paths -quiet -setup -max_paths 1 -from $c2x_cs -through $c2x_rp -to $c2x_cf]
    if {![llength $c2x_chk]} {
      error "\[otpu_clk2x.tcl\] no timing path from [get_property NAME $c2x_cs] through the [llength $c2x_rp] read-request pins to [get_property NAME $c2x_cf]: the exceptions reach nothing"
    }
    set c2x_req [get_property -quiet REQUIREMENT $c2x_chk]
    if {$c2x_req eq "" || abs($c2x_req - 2.0 * $c2x_tf) > 0.01} {
      error "\[otpu_clk2x.tcl\] the read-request path [get_property STARTPOINT_PIN $c2x_chk] -> [get_property ENDPOINT_PIN $c2x_chk] reports a requirement of '$c2x_req' ns, expected [expr {2.0 * $c2x_tf}] (two fast periods): the multicycle is not in effect"
    }
    puts "otpu_clk2x.tcl: [llength $c2x_pb] pumped TMEM RAMs ([get_property NAME $c2x_cs] $c2x_ts ns, [get_property NAME $c2x_cf] $c2x_tf ns), [llength $c2x_rp] read-request pins, [llength $c2x_qp] word pins, [llength $c2x_br] block RAM primitives; the worst read-request path requires $c2x_req ns"
  }
} else {
  puts "otpu_clk2x.tcl: no pumped TMEM RAMs (otpu_pump_bank)"
}

# ---------------------------------------------------------------- the MX column's pumped DSPs
# (none yet: a pumped MX column at MCOLS 8 brings its multicycle set here, as
# rtl/exp/mx_mxu/ooc.tcl's for otpu_mx_x7)
