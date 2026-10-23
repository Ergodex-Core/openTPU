# The public openTPU source snapshot is already under CL_DIR/design. Preserve the
# target's compile order; this hook is required by the AWS HDK build driver.
if {$ENCRYPT} { error "openTPU flow expects ENCRYPT=0 (public source snapshot)" }
