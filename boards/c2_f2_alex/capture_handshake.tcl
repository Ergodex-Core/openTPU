# Read-only, bounded, standalone capture. Never changes constraints or the DCP.
# vivado -mode batch -source capture_handshake.tcl -tclargs DCP SHA256 OUTPUT
namespace eval ::c2_handshake {}
proc ::c2_handshake::names {objects} {
    set result {}
    foreach object $objects {lappend result [get_property NAME $object]}
    if {[llength [lsort -unique $result]] != [llength $objects]} {error "Duplicate/empty typed object names"}
    foreach name $result {if {$name eq ""} {error "Empty typed object name"}}
    return $result
}
proc ::c2_handshake::one {kind name} {
    # -filter equality makes vector brackets literal; do not use glob lookup.
    if {[regexp {["\\\n\r]} $name]} {error "Unsafe object name"}
    set command [dict get {cell get_cells pin get_pins} $kind]
    set object [$command -quiet -hierarchical -filter "NAME == \"$name\""]
    if {[names $object] ne [list $name]} {error "Missing/ambiguous $kind $name"}
    return $object
}
proc ::c2_handshake::properties {object keys} {
    set result {}
    foreach key $keys {dict set result $key [get_property $key $object]}
    return $result
}
proc ::c2_handshake::register {out root role name} {
    set cell [one cell $name]
    set pin [one pin $name/C]
    set owners [get_cells -quiet -of_objects $pin]
    set clocks [get_clocks -quiet -of_objects $pin]
    if {[names $owners] ne [list $name] || [llength $clocks]!=1} {error "Ambiguous register owner/clock $name"}
    puts $out [list register root $root role $role cell [properties $cell {NAME REF_NAME ORIG_REF_NAME ASYNC_REG IS_SEQUENTIAL}] \
        pin [properties $pin {NAME REF_PIN_NAME DIRECTION IS_CLOCK}] owners [names $owners] \
        clocks [names $clocks] clock [properties $clocks {NAME PERIOD WAVEFORM}]]
}
proc ::c2_handshake::capture {out checkpoint_sha256} {
    set prefix WRAPPER/CL/u_ocl_cdc/inst/gen_clock_conv.gen_async_lite_conv.clock_conv_lite_
    set expected {}
    foreach suffix {fwd_ar fwd_aw fwd_w resp_b resp_r} {lappend expected ${prefix}${suffix}/handshake}
    puts $out [list schema 1 kind c2_alex_handshake_witness checkpoint_sha256 $checkpoint_sha256 route_qualified 0]
    set all [get_cells -quiet -hierarchical -filter "NAME =~ ${prefix}*/handshake"]
    if {[lsort [names $all]] ne [lsort $expected]} {error "Unexpected OCL handshake root inventory"}
    puts $out [list roots names [lsort [names $all]]]
    foreach root $expected {
        set object [one cell $root]
        puts $out [list root name $root properties [properties $object {NAME REF_NAME ORIG_REF_NAME IS_BLACKBOX}]]
        set selected {}
        foreach {role suffix} {source src_hsdata_ff_reg* destination dest_hsdata_ff_reg* forward_sync xpm_cdc_single_src2dest_inst/syncstages_ff_reg*} {
            set cells [get_cells -quiet -hierarchical -filter "NAME =~ $root/$suffix"]
            set full_names [lsort [names $cells]]
            if {[llength $full_names]<1 || [llength $full_names]>100} {error "Missing/excess handshake register set $root $role"}
            dict set selected $role $full_names
            puts $out [list inventory root $root role $role names $full_names]
            foreach name $full_names {register $out $root $role $name}
        }
        # The XPM hierarchy and all register/clock evidence remain raw facts;
        # the offline predicate determines width/depth and vendor timing bounds.
    }
    puts $out [list complete roots [llength $expected] route_qualified 0]
}
proc ::c2_handshake::sha256 {path} {
    set value [lindex [exec sha256sum -- $path] 0]
    if {![regexp {^[0-9a-f]{64}$} $value]} {error "Invalid checkpoint digest"}
    return $value
}
proc ::c2_handshake::main {arguments} {
    if {[llength $arguments]!=3} {error "Expected DCP SHA256 OUTPUT"}
    lassign $arguments checkpoint expected output
    if {![regexp {^[0-9a-f]{64}$} $expected] || [sha256 $checkpoint] ne $expected} {error "Checkpoint hash mismatch"}
    set out [open $output {WRONLY CREAT EXCL}]
    try {
        open_checkpoint $checkpoint
        capture $out $expected
        if {[sha256 $checkpoint] ne $expected} {error "Checkpoint changed during capture"}
    } finally {close $out}
}
# Vivado -source does not promise that argv0 equals the sourced filename.
# Tests or an already-open read-only collector may explicitly source as library.
if {![info exists ::c2_handshake_library_only] || !$::c2_handshake_library_only} {::c2_handshake::main $argv}
