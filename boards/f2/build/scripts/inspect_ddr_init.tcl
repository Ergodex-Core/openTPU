# Inspect the firmware already stored in a synthesized or routed checkpoint.
# Do not refresh here: implementation uses the verified checkpoint image.
# Select primitive RAMs first, then use a Tcl regexp with literal brackets for
# the HDK's sentinel bank. This avoids glob interpretation of ramloop[0] while
# allowing the controller's prefix and generated implementation names to vary.
set ddr_cal_ram {}
foreach ram [get_cells -hierarchical -filter {REF_NAME == RAMB36E2 && NAME =~ *mcs0*}] {
  set ram_name [get_property NAME $ram]
  if {[regexp {(^|/)mcs0/inst/lmb_bram_I/.*ramloop\[0\]\.ram\.r/} $ram_name]} {
    lappend ddr_cal_ram $ram
  }
}
if {[llength $ddr_cal_ram] != 1} {
  error "openTPU: expected one DDR calibration BRAM sentinel, found [llength $ddr_cal_ram]"
}
set ddr_cal_init [get_property INIT_2C $ddr_cal_ram]
if {![regexp -nocase {^256'h[0-9a-f]{64}$} $ddr_cal_init] ||
    [regexp -nocase {^256'h0+$} $ddr_cal_init]} {
  error "openTPU: DDR calibration firmware is absent (INIT_2C=$ddr_cal_init)"
}
puts "OTPU_DDR_CALIBRATION_INIT_2C=$ddr_cal_init"
