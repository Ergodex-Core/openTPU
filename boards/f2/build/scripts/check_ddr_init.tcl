# Refresh the DDR controller's MicroBlaze calibration firmware before saving the
# checkpoint. The stock HDK footer merely warns (or skips a missing hierarchy).
refresh_meminit
source [file join [file dirname [info script]] inspect_ddr_init.tcl]
