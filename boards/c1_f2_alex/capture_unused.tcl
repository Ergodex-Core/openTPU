# Source-only C1 raw unused-net collector. Already-open checkpoint/report_drc
# provenance belongs to capture_final's immutable checkpoint/collector wrapper.
namespace eval ::c1_unused_capture {
    variable max_all 50000;variable max_current 128;variable max_connections 256
    variable max_bytes 33554432
}
proc ::c1_unused_capture::emit {out record} {
    variable max_bytes
    if {[tell $out]+[string length [encoding convertto utf-8 $record]]+1>$max_bytes} {error "Unused-net output cap"}
    puts $out $record
}
proc ::c1_unused_capture::bounded {objects cap label} {
    if {[llength $objects]>$cap} {error "$label cap: [llength $objects] > $cap"}
}
proc ::c1_unused_capture::names {objects} {
    set result {};foreach object $objects {lappend result [get_property NAME $object]}
    if {[llength $result]!=[llength [lsort -unique $result]] || "" in $result || "..." in $result} {error "Duplicate/empty/truncated typed inventory"}
    return $result
}
proc ::c1_unused_capture::properties {object keys} {
    set result {};foreach key $keys {dict set result $key [get_property $key $object]}
    return $result
}
proc ::c1_unused_capture::collect {output checkpoint_sha256} {
    variable max_all;variable max_current;variable max_connections
    if {![regexp {^[0-9a-f]{64}$} $checkpoint_sha256]} {error "Exact checkpoint SHA required"}
    if {[version -short] ne "2025.2" || [llength [get_designs -quiet]]!=1} {error "One open Vivado 2025.2 design required"}
    set out [open $output {WRONLY CREAT EXCL}];fconfigure $out -encoding utf-8 -translation lf
    set code [catch {
        emit $out [list schema 1 kind c1_unused_output_nets checkpoint_sha256 $checkpoint_sha256 route_qualified 0]
        set violations [get_drc_violations -quiet];bounded $violations $max_all drc_findings
        set found {};set related_names {};set selected {};array set nets {}
        foreach violation $violations {
            if {[get_property CHECK $violation] ne "RTSTAT-10"} {continue}
            set finding [get_property NAME $violation]
            if {$finding in $found} {error "Duplicate RTSTAT-10 finding"}
            lappend found $finding
            set related [get_nets -quiet -of_objects $violation];bounded $related $max_all related_nets
            set names [names $related]
            emit $out [list finding_inventory name $finding check RTSTAT-10 related_count [llength $names] related_nets $names]
            foreach net $related {
                set name [get_property NAME $net]
                dict set related_names $name 1
                if {[string match {WRAPPER/CL/*} $name]} {set nets($name) $net;if {$name ni $selected} {lappend selected $name}}
            }
            if {[dict size $related_names]>$max_all} {error "Combined related-net cap"}
            bounded $selected $max_current current_CL_nets
        }
        if {![llength $found]} {error "Missing RTSTAT-10 finding inventory"}
        set selected [lsort $selected]
        emit $out [list selected_nets names $selected]
        foreach name $selected {
            # Preserve native net objects, never operate on a printed path as a handle.
            if {[get_property NAME $nets($name)] ne $name || [get_property CLASS $nets($name)] ne "net"} {error "Typed net identity changed"}
            set pins [get_pins -quiet -of_objects $nets($name)]
            bounded $pins $max_connections connected_pins
            if {![llength $pins]} {error "Missing net connection inventory"}
            set segments [get_nets -quiet -segments -of_objects $pins]
            bounded $segments $max_connections net_segments
            set segment_names [names $segments]
            if {$name ni $segment_names} {error "Selected net absent from segments"}
            set leaves [get_pins -quiet -leaf -of_objects $segments]
            set ports [get_ports -quiet -of_objects $segments]
            bounded $leaves $max_connections leaf_pins;bounded $ports $max_connections ports
            emit $out [list net $name properties [properties $nets($name) {NAME CLASS TYPE IS_CLOCK IS_ROUTE_FIXED ROUTE_STATUS}] segments $segment_names pins [names $pins] leaf_pins [names $leaves] ports [names $ports]]
            foreach pin $leaves {
                set owner [get_cells -quiet -of_objects $pin]
                if {[llength $owner]!=1} {error "Missing/multiple leaf owner"}
                emit $out [list leaf $name name [get_property NAME $pin] properties [properties $pin {NAME CLASS DIRECTION REF_NAME REF_PIN_NAME IS_CLOCK IS_INVERTED}] owner [names $owner] owner_properties [properties $owner {NAME CLASS REF_NAME ORIG_REF_NAME IS_PRIMITIVE IS_SEQUENTIAL IS_BLACKBOX DONT_TOUCH}]]
            }
            foreach port $ports {emit $out [list port $name name [get_property NAME $port] properties [properties $port {NAME CLASS DIRECTION}]]}
        }
        emit $out [list complete nets [llength $selected] route_qualified 0]
    } message options]
    if {$code} {catch {emit $out [list error message $message route_qualified 0]}}
    close $out
    if {$code} {return -options $options $message}
    return [dict create kind c1_unused_output_nets checkpoint_sha256 $checkpoint_sha256 nets [llength $selected] findings [llength $found] route_qualified 0]
}
