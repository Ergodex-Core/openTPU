# F2 HBM only. Read the frozen HDL parameter, not a caller-provided environment override.
# build.py generates this header and records its hash in inputs.json; cl_opentpu_hbm.sv
# includes it as u_sys's actual parameter list.
proc otpu_expected_tmem_pmp {cl_root} {
  set path [file join $cl_root design opentpu_params.vh]
  if {![file isfile $path]} { error "openTPU TMEM: missing frozen parameter header $path" }
  set f [open $path r]
  set params [read $f]
  close $f
  set hits [regexp -all -inline {\.TMEM_PMP\(32'h([0-9a-fA-F]+)\)} $params]
  if {[llength $hits] != 2 || [scan [lindex $hits 1] %x pmp] != 1 || $pmp ni {0 1}} {
    error "openTPU TMEM: $path must contain exactly one TMEM_PMP 0 or 1 parameter"
  }
  return $pmp
}

# PMP 0 must have no surviving pumped memory or fast-clock endpoint. Check every TMEM
# block RAM clock directly: all_registers may not count a RAM primitive as a flip-flop.
proc otpu_assert_core_tmem {} {
  set pump_modules [get_cells -quiet -hierarchical -filter {
    ORIG_REF_NAME == otpu_pump_bank || REF_NAME == otpu_pump_bank}]
  set pump_names [get_cells -quiet -hierarchical -filter {NAME =~ *g_pmp*u_pb}]
  set pump_rams [get_cells -quiet -hierarchical -filter {
    PRIMITIVE_TYPE =~ BLOCKRAM.* && NAME =~ *g_pmp*u_pb/*}]
  if {[llength $pump_modules] || [llength $pump_names] || [llength $pump_rams]} {
    error "openTPU TMEM PMP0: pumped cells remain (modules [llength $pump_modules], names [llength $pump_names], RAMs [llength $pump_rams])"
  }

  set core_pin [get_pins -quiet -hierarchical -filter {NAME =~ *u_mmcm/CLKOUT0}]
  set fast_pin [get_pins -quiet -hierarchical -filter {NAME =~ *u_mmcm/CLKOUT1}]
  if {[llength $core_pin] != 1 || [llength $fast_pin] > 1} {
    error "openTPU TMEM PMP0: expected one core MMCM pin and at most one unused fast pin"
  }
  set core_clock [get_clocks -quiet -of_objects $core_pin]
  set fast_clock {}
  if {[llength $fast_pin]} { set fast_clock [get_clocks -quiet -of_objects $fast_pin] }
  if {[llength $core_clock] != 1 || [llength $fast_clock] > 1} {
    error "openTPU TMEM PMP0: core or fast generated clock is ambiguous"
  }
  if {[llength $fast_clock]} {
    set fast_regs [all_registers -clock $fast_clock -cells]
    if {[llength $fast_regs]} {
      error "openTPU TMEM PMP0: [llength $fast_regs] sequential cells still use the fast clock"
    }
  }

  set tmem_rams [get_cells -quiet -hierarchical -filter {
    PRIMITIVE_TYPE =~ BLOCKRAM.* && NAME =~ *u_tmem/*}]
  if {![llength $tmem_rams]} { error "openTPU TMEM PMP0: no TMEM block RAMs found" }
  set ram_clocks [get_pins -quiet -of_objects $tmem_rams -filter {IS_CLOCK}]
  if {[llength $ram_clocks] < [llength $tmem_rams]} {
    error "openTPU TMEM PMP0: missing TMEM RAM clock pins"
  }
  set core_name [get_property NAME $core_clock]
  foreach pin $ram_clocks {
    set clocks [get_clocks -quiet -of_objects $pin]
    if {[llength $clocks] != 1 || [get_property NAME $clocks] ne $core_name} {
      error "openTPU TMEM PMP0: RAM clock pin $pin is not exclusively on $core_name"
    }
  }
  puts "OTPU_TMEM_PMP0_CHECK RAMs=[llength $tmem_rams] clock_pins=[llength $ram_clocks] core=$core_name fast_endpoints=0"
}
