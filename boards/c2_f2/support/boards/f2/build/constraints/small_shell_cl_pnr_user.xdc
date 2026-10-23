# The HDK's small-shell floorplan owns the CL region and DDR placement; the HDK reads this file
# as a managed XDC at placement (build_level_1_cl.tcl: read_xdc, no -unmanaged), so it holds no
# Tcl (a managed XDC drops proc / if / foreach / puts with a CRITICAL WARNING and keeps going).
# openTPU's own placement constraints (the linked reset exception, the HBM floorplan) are
# cl_pnr_user.tcl, read unmanaged by boards/f2/build/scripts/build_level_1_cl.tcl before
# opt_design (docs/v4/f2-timing.md).
