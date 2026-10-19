# Post-synthesis CDC timing of the HBM build (boards/f2/design/cl_opentpu_hbm.sv; build.py puts
# this file in place of cl_timing_user.tcl for an HBM target). Three clock groups exist in the
# CL: the core (u_clk: 200 MHz, and u_clk_f: 400 MHz, phase aligned, timed together), the shell's
# clk_main_a0 (250 MHz: OCL, PCIS) and the HBM's AXI clock (u_hclk: 450 MHz). Since round 5
# (docs/v4/f2-round5.md) openTPU's logic meets only the first two: the host path's crossings
# (otpu_afifo, aclk <-> core), its write count (a gray counter), the OCL crossing's held words
# (otpu_axil_cdc) and synchronized levels. The HBM clock belongs to the vendor IP: the
# cl_axi_sc_1x1 SmartConnects (their crossings from the core clock, with the IP's own
# constraints) and the HDK's HBM wrapper (its xpm_cdc resets); the only openTPU flip-flops on it
# are the wrapper's reset synchronizer (hbm_reset_n), which this file checks, failing closed.
# Every crossing is bound each direction by the destination clock's period, datapath only (a
# crossing's word is read only after its pointer, request or count has gone through the
# destination's two synchronizer flip-flops; docs/v4/f2-timing.md), the vendor crossings
# included (Alex's burst200 image timed its SmartConnect's core -> hbm paths at the core period
# and hbm -> core at the HBM period, both with margin; the IP's own, more specific constraints
# take precedence where it sets any); the gray buses' skew by the smallest period. The clocks
# are not grouped asynchronously (that would override these bounds).
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
# Gray pointers of every crossing FIFO: the host path's three (the PCIS beats and the read
# requests in, the read data back), (binary MSBs may merge with the gray registers: both
# source names)
set fifos [otpu_required [get_cells -hier -filter {ORIG_REF_NAME == otpu_afifo || REF_NAME == otpu_afifo}] {crossing FIFOs}]
foreach f $fifos {
  foreach {src dst} {wgray wgray_r1 rgray rgray_w1} {
    set bin [string map {gray bin} $src]
    set from [get_cells -quiet -filter {IS_SEQUENTIAL} [list $f/${src}_reg* $f/${bin}_reg*]]
    set to [otpu_required [get_cells -quiet -filter {IS_SEQUENTIAL} $f/${dst}_reg*] "$f/$dst"]
    set_bus_skew -from [otpu_required $from "$f/$src"] -to $to $t_skew
  }
}
# The write count (gray, one step a cycle) of the host path, core -> aclk
set counters [get_cells -hier -filter {ORIG_REF_NAME == otpu_f2_hbm_host || REF_NAME == otpu_f2_hbm_host}]
if {[llength $counters] < 1} { error "openTPU CDC: expected the host path" }
foreach ch $counters {
  set from [otpu_required [get_cells -quiet -filter {IS_SEQUENTIAL} [list $ch/bgray_reg* $ch/bcnt_reg*]] "$ch/write counter"]
  set to [otpu_required [get_cells -quiet -filter {IS_SEQUENTIAL} $ch/bg1_reg*] "$ch/write synchronizer"]
  set_bus_skew -from $from -to $to $t_skew
}
puts "OTPU_HBM_CDC fifos=[llength $fifos] counters=[llength $counters] core=$t_core main=$t_main hbm=$t_hbm"
# Nothing of openTPU's on the HBM clock: every sequential cell it clocks is the vendor IP's (the
# SmartConnects g_sc[*].u_sc, the HBM wrapper u_hbm) or the wrapper's reset synchronizer
# (hbm_reset_n_reg*: three flip-flops, cl_pnr_user.tcl's linked-reset exception covers them)
set on_hclk [get_cells -quiet -of_objects $c_hbm -filter {IS_SEQUENTIAL}]
set otpu_on_hclk {}
foreach c $on_hclk {
  set n [get_property NAME $c]
  if {![string match {g_sc[[]*[]].u_sc/*} $n] && ![string match {u_hbm/*} $n] &&
      ![string match {hbm_reset_n_reg*} $n]} { lappend otpu_on_hclk $n }
}
if {[llength $otpu_on_hclk]} {
  error "openTPU CDC: [llength $otpu_on_hclk] openTPU cells on the HBM clock, e.g. [lrange $otpu_on_hclk 0 4]"
}
puts "OTPU_HBM_NO_HCLK_RTL vendor=[llength $on_hclk]"
# Lock is asynchronous to the generated clocks: only its reset-control fanout is exempted
set_false_path -through [get_pins u_mmcm/LOCKED] -to [get_pins -hier -filter {REF_PIN_NAME == CLR || REF_PIN_NAME == PRE}]
set_false_path -through [get_pins u_hmmcm/LOCKED] -to [get_pins -hier -filter {REF_PIN_NAME == CLR || REF_PIN_NAME == PRE}]
set_false_path -from [get_ports rst_main_n]
