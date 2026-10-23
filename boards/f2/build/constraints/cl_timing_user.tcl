# Post-synthesis CDC timing. Do not group shell/core clocks asynchronously: that
# would override these max-delay constraints. Bus skew is a separate assertion.
proc otpu_required {objects description} {
  if {![llength $objects]} { error "openTPU CDC: missing $description" }
  return $objects
}
set c_core [otpu_required [get_clocks -of_objects [get_pins u_clk/O]] {core clock}]
set c_main [otpu_required [get_clocks clk_main_a0] {shell clock}]
# Confirm the exact pinned HDK DDR clock query resolves. Its asynchronous group
# must apply to the DDR UI clock, while shell/core crossings retain max delays.
set c_ddr [otpu_required [get_clocks -of_objects [get_pins -hierarchical -filter {
  NAME =~ *SH_DDR/genblk1.IS_DDR_PRESENT.DDR4_0/inst/u_ddr4_infrastructure/gen_mmcme4.u_mmcme_adv_inst/CLKOUT0
}]] {DDR UI clock used by the HDK constraints}]
if {[llength $c_core] != 1 || [llength $c_main] != 1 || [llength $c_ddr] != 1} {
  error "openTPU CDC: expected one core, shell and DDR UI clock"
}
set t_core [get_property PERIOD $c_core]
set t_main [get_property PERIOD $c_main]
set t_skew [expr {min($t_core, $t_main)}]
# Every path between the shell and core is a FIFO, held request/response word,
# Gray counter, or synchronized status level. Bound each direction by the smaller
# period (4 ns). Preserve timing on the separate core/2x pair for pumped RAMs.
set_max_delay -datapath_only -from $c_main -to $c_core $t_skew
set_max_delay -datapath_only -from $c_core -to $c_main $t_skew
# Gray bus skew per FIFO and write-response counter. Binary MSBs can be merged
# with Gray registers by synthesis; include both possible source names.
set fifos [get_cells -hier -filter {ORIG_REF_NAME == otpu_afifo || REF_NAME == otpu_afifo}]
if {[llength $fifos] != 6} { error "openTPU CDC: expected six FIFOs, found [llength $fifos]" }
foreach f $fifos {
  foreach {src dst} {wgray wgray_r1 rgray rgray_w1} {
    set bin [string map {gray bin} $src]
    set from [get_cells -quiet -filter {IS_SEQUENTIAL} [list $f/${src}_reg* $f/${bin}_reg*]]
    set to [otpu_required [get_cells -quiet -filter {IS_SEQUENTIAL} $f/${dst}_reg*] "$f/$dst"]
    set_bus_skew -from [otpu_required $from "$f/$src"] -to $to $t_skew
  }
}
set channels [get_cells -hier -filter {ORIG_REF_NAME == otpu_f2_nch || REF_NAME == otpu_f2_nch}]
if {[llength $channels] != 2} { error "openTPU CDC: expected two native channels" }
foreach ch $channels {
  set from [otpu_required [get_cells -quiet -filter {IS_SEQUENTIAL} [list $ch/bgray_reg* $ch/bcnt_reg*]] "$ch/write counter"]
  set to [otpu_required [get_cells -quiet -filter {IS_SEQUENTIAL} $ch/bg1_reg*] "$ch/write synchronizer"]
  set_bus_skew -from $from -to $to $t_skew
}
# Lock is asynchronous to the generated clocks. Only its reset-control fanout
# is exempted; ordinary clocks and data remain timed.
set_false_path -through [get_pins u_mmcm/LOCKED] -to [get_pins -hier -filter {REF_PIN_NAME == CLR || REF_PIN_NAME == PRE}]
