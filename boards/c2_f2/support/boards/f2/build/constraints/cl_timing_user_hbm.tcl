# Post-synthesis CDC timing of the HBM build (boards/f2/design/cl_opentpu_hbm.sv; build.py puts
# this file in place of cl_timing_user.tcl for an HBM target). Three clock groups meet in the
# openTPU logic: the core (u_clk: 200 MHz, and u_clk_f: 400 MHz, phase aligned, timed together),
# the shell's clk_main_a0 (250 MHz: OCL, PCIS) and the HBM's AXI clock (u_hclk: 450 MHz). Every
# path between two of them is an otpu_afifo, a held request/response word (otpu_axil_cdc), a gray
# counter (the bridges' and the host path's write counts) or a synchronized level: bound each
# direction by the destination clock's period, datapath only (a crossing's word is read only
# after its pointer, request or count has gone through the destination's two synchronizer
# flip-flops, so a destination period on the data is the conservative bound; bounding both
# directions by the smaller period of the pair, as before docs/v4/f2-timing.md, held the HBM
# side's data into the core, the bridges' return FIFOs and the write counts' synchronizers,
# at 2.2 ns instead of 5 across the die, and pulled the counts' core-side registers to the HBM
# ports), and the gray buses' skew by the smallest period. The clocks are not grouped
# asynchronously (that would override these bounds). The HBM wrapper's own crossings (its
# resets, its ready bits: xpm_cdc) carry the XPM constraints.
proc otpu_required {objects description} {
  if {![llength $objects]} { error "openTPU CDC: missing $description" }
  return $objects
}
set c_core [otpu_required [get_clocks -of_objects [get_pins u_clk/O]] {core clock}]
set c_main [otpu_required [get_clocks clk_main_a0] {shell clock}]
set c_hbm  [otpu_required [get_clocks -of_objects [get_pins u_hclk/O]] {HBM AXI clock}]
if {[llength $c_core] != 1 || [llength $c_main] != 1 || [llength $c_hbm] != 1} {
  error "openTPU CDC: expected one core, shell and HBM AXI clock"
}
set t_core [get_property PERIOD $c_core]
set t_main [get_property PERIOD $c_main]
set t_hbm  [get_property PERIOD $c_hbm]
foreach {a b} [list $c_main $c_core $c_core $c_hbm $c_main $c_hbm] {
  set_max_delay -datapath_only -from $a -to $b [get_property PERIOD $b]
  set_max_delay -datapath_only -from $b -to $a [get_property PERIOD $a]
}
set t_skew [expr {min($t_core, $t_main, $t_hbm)}]
# Gray pointers of every crossing FIFO: per channel two (commands in, data back), the host path
# three, (binary MSBs may merge with the gray registers: both source names)
set fifos [otpu_required [get_cells -hier -filter {ORIG_REF_NAME == otpu_afifo || REF_NAME == otpu_afifo}] {crossing FIFOs}]
foreach f $fifos {
  foreach {src dst} {wgray wgray_r1 rgray rgray_w1} {
    set bin [string map {gray bin} $src]
    set from [get_cells -quiet -filter {IS_SEQUENTIAL} [list $f/${src}_reg* $f/${bin}_reg*]]
    set to [otpu_required [get_cells -quiet -filter {IS_SEQUENTIAL} $f/${dst}_reg*] "$f/$dst"]
    set_bus_skew -from [otpu_required $from "$f/$src"] -to $to $t_skew
  }
}
# The write counts (gray, one step a cycle) of every channel bridge and of the host path
set counters [concat \
  [get_cells -hier -filter {ORIG_REF_NAME == otpu_f2_hbm_nch || REF_NAME == otpu_f2_hbm_nch}] \
  [get_cells -hier -filter {ORIG_REF_NAME == otpu_f2_hbm_host || REF_NAME == otpu_f2_hbm_host}]]
if {[llength $counters] < 2} { error "openTPU CDC: expected the HBM bridges and the host path" }
foreach ch $counters {
  set from [otpu_required [get_cells -quiet -filter {IS_SEQUENTIAL} [list $ch/bgray_reg* $ch/bcnt_reg*]] "$ch/write counter"]
  set to [otpu_required [get_cells -quiet -filter {IS_SEQUENTIAL} $ch/bg1_reg*] "$ch/write synchronizer"]
  set_bus_skew -from $from -to $to $t_skew
}
puts "OTPU_HBM_CDC fifos=[llength $fifos] counters=[llength $counters] core=$t_core main=$t_main hbm=$t_hbm"
# Lock is asynchronous to the generated clocks: only its reset-control fanout is exempted
set_false_path -through [get_pins u_mmcm/LOCKED] -to [get_pins -hier -filter {REF_PIN_NAME == CLR || REF_PIN_NAME == PRE}]
set_false_path -through [get_pins u_hmmcm/LOCKED] -to [get_pins -hier -filter {REF_PIN_NAME == CLR || REF_PIN_NAME == PRE}]
set_false_path -from [get_ports rst_main_n]
