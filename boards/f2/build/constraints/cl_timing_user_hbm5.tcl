# Post-synthesis CDC timing of the v5 full-HBM build (boards/f2/design/cl_opentpu_hbm5.sv, the
# engine memory path; build.py puts this file in place of cl_timing_user.tcl for a target whose
# platform mem_path is "engine"; docs/full_hbm.md "Timing"). Three clocks meet in openTPU's
# own logic:
#   - the core (u_clk, 200 MHz; u_clk_f, 2x, phase aligned, is timed with it and otpu_clk2x.tcl
#     holds the pumped TMEM's multicycles) and the shell's clk_main_a0 (250 MHz: OCL, PCIS)
#     through otpu_axil_cdc's held words and synchronized levels (the OCL crossing, the board's
#     LED and HBM-ready levels), as the SmartConnect build: each direction bounded by the
#     destination clock's period, datapath only;
#   - the core and the HBM's AXI clock (u_hclk, 450 MHz), which the SmartConnect build never
#     named (its crossings were SmartConnect's own): here they meet in every port engine's four
#     asynchronous FIFOs (otpu_hbm_ch / otpu_hbm_afifo: the command and write-beat FIFOs core ->
#     hclk, the read-beat and B-token FIFOs hclk -> core; each a gray pointer per direction
#     through two flip-flops, and a RAM written in one clock and read in the other) and in the
#     engine's error level (herr, hclk, through two core flip-flops). Each direction bounded by
#     the destination's period, datapath only (a pointer bit or a level is one flip-flop to a
#     synchronizer; a distributed RAM's word reaches its read register at least two destination
#     cycles after the write, once the pointer announcing it has crossed), and every gray
#     vector's bits held within the shorter period (set_bus_skew: the pointer steps once a
#     source cycle, one bit a step, so its bits must arrive in order).
# The 450 MHz register slices, the 2 x 1 host mux (otpu_hbm_mux2) and the geometry stay in
# hclk (no crossing); the host's cl_axi_sc_1x1 (250 <-> 450) and the HDK's HBM wrapper carry
# their own constraints. The clocks are not grouped asynchronously (that would override the
# IP's bounds and these; validate_route.tcl checks the bounds hold after routing). Fails
# closed: the engines, every FIFO of every engine with both synchronizer stages of both
# pointers, and every error level must be found, in consistent counts, else the build stops
# here; a SmartConnect fabric (u_sc) means the wrong top was built.
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
set t_hskew [expr {min($t_core, $t_hbm)}]
# ---- the engines (one per HBM port: u_sys/u_mem/g_port[p].u_eng; the floorplan names them the
# same way), the mux on port NPORT - 1 at 32 channels (hclk only: counted, not bounded)
if {[llength [get_cells -quiet u_sc]]} { error "openTPU CDC: a SmartConnect fabric u_sc in the engine build" }
set engines [otpu_required [get_cells -hier -filter {NAME =~ *u_mem/g_port*.u_eng}] {the HBM port engines (otpu_hbm_ch)}]
set n_eng [llength $engines]
set n_mux [llength [get_cells -quiet -hier -filter {NAME =~ *u_mem/g_mux.u_mux}]]
# ---- the FIFOs: {instance, write clock}; the pointer width (log2 depth + 1) is read off the
# synchronizers, which must agree across the engines, the two pointers and the two stages
set fifos {u_cq core u_wq core u_rq hbm u_bq hbm}
set widths {}
set n_sync 0
foreach {f wck} $fifos {
  set insts [get_cells -quiet -hier -filter "NAME =~ *u_mem/g_port*.u_eng/$f"]
  if {[llength $insts] != $n_eng} { error "openTPU CDC: $f: [llength $insts] FIFOs for $n_eng engines" }
  set bits 0
  foreach stage {wgray_r1 wgray_r2 rgray_w1 rgray_w2} {
    set n [llength [get_cells -quiet -hier -filter "NAME =~ *u_mem/g_port*.u_eng/$f/${stage}_reg*"]]
    if {$n == 0 || $n % $n_eng} { error "openTPU CDC: $f/$stage: $n synchronizer flip-flops over $n_eng engines" }
    if {$bits == 0} { set bits [expr {$n / $n_eng}] }
    if {$n != $bits * $n_eng} { error "openTPU CDC: $f/$stage: $n flip-flops, the write pointer's synchronizer has [expr {$bits * $n_eng}]" }
  }
  if {$bits < 2} { error "openTPU CDC: $f: a $bits-bit gray pointer" }
  lappend widths "[string range $f 2 end]=$bits"
  incr n_sync [expr {2 * $bits * $n_eng}]
}
set herr [get_cells -quiet -hier -filter {NAME =~ *u_mem/g_port*.u_eng/herr_s1_reg*}]
if {[llength $herr] != $n_eng} { error "openTPU CDC: [llength $herr] engine error synchronizers for $n_eng engines" }
# ---- the bounds: each direction at the destination's period, datapath only
set_max_delay -datapath_only -from $c_main -to $c_core $t_core
set_max_delay -datapath_only -from $c_core -to $c_main $t_main
set_max_delay -datapath_only -from $c_core -to $c_hbm $t_hbm
set_max_delay -datapath_only -from $c_hbm -to $c_core $t_core
# the gray pointers' bits within the shorter period: the write pointer into the read clock's
# synchronizer and the read pointer into the write clock's, per FIFO of every engine (a binary
# MSB can stand in for the gray MSB: both names, as cl_timing_user.tcl)
set n_skew 0
foreach e $engines {
  foreach {f wck} $fifos {
    foreach {src dst} {wgray wgray_r1 rgray rgray_w1} {
      set bin [string map {gray bin} $src]
      set from [get_cells -quiet -filter {IS_SEQUENTIAL} [list $e/$f/${src}_reg* $e/$f/${bin}_reg*]]
      set to [otpu_required [get_cells -quiet -filter {IS_SEQUENTIAL} $e/$f/${dst}_reg*] "$e/$f/$dst"]
      set_bus_skew -from [otpu_required $from "$e/$f/$src"] -to $to $t_hskew
      incr n_skew
    }
  }
}
puts "OTPU_HBM_CDC core=$t_core main=$t_main hbm=$t_hbm engines=$n_eng fifos=[expr {4 * $n_eng}] bits=[join $widths ,] sync_ffs=$n_sync bus_skews=$n_skew err_levels=[llength $herr] mux=$n_mux (hbm crossings: the engines' FIFOs and error levels bounded here; cl_axi_sc_1x1's and the HBM wrapper's their own)"
# Lock is asynchronous to the generated clocks: only its reset-control fanout is exempted
set_false_path -through [get_pins u_mmcm/LOCKED] -to [get_pins -hier -filter {REF_PIN_NAME == CLR || REF_PIN_NAME == PRE}]
set_false_path -through [get_pins u_hmmcm/LOCKED] -to [get_pins -hier -filter {REF_PIN_NAME == CLR || REF_PIN_NAME == PRE}]
set_false_path -from [get_ports rst_main_n]
