# The SmartConnect fabric's instance in cl_opentpu_hbm.sv, written from the generated block
# design wrapper's own port list (create_hbm_sc.tcl sources this file; tests/test_f2_build.py
# runs it under tclsh on a sample wrapper). Every wrapper port is mapped to the CL's signals:
#   S_HOST_<sig>  -> h_<sig>          (otpu_f2_pcis_gate's master, aclk)
#   S_N<p>_<sig>  -> ch_<sig>[p]      (otpu_native_axi p's master, core clock; its constants:
#                                      single-beat writes, INCR, 64-byte size, ID 0)
#   M_H<p>_<sig>  -> x_<sig>[p]       (HBM AXI3 port p, through otpu_hbm_geometry)
#   clocks / resets                    core_clk clk, host_clk clk_main_a0, hbm_clk hclk,
#                                      aresetn sc_aresetn, host_aresetn sc_host_aresetn
# The protocol's optional inputs (lock, cache, prot, qos, region, user) are tied to 0, its
# optional outputs left open; an HBM-side ID (M_H<p>_awid / arid, at most the HBM's 6 bits) is
# connected if the wrapper has one, else x_awid / x_arid are tied to 0 (Alex's burst200
# wrapper had none: SmartConnect's single-threaded master). Fail closed: an input the CL does
# not know, a width other than the CL's, or a port the CL needs missing stops the build.

proc otpu_sc_widths {} {
  return {
    H {awid 16 awaddr 64 awlen 8 awsize 3 awburst 2 awvalid 1 awready 1 wdata 512 wstrb 64
       wlast 1 wvalid 1 wready 1 bid 16 bresp 2 bvalid 1 bready 1 arid 16 araddr 64 arlen 8
       arsize 3 arburst 2 arvalid 1 arready 1 rid 16 rdata 512 rresp 2 rlast 1 rvalid 1 rready 1}
    C {awaddr 34 awvalid 1 awready 1 wdata 512 wstrb 64 wvalid 1 wready 1 bresp 2 bvalid 1
       bready 1 araddr 34 arlen 8 arvalid 1 arready 1 rdata 512 rresp 2 rlast 1 rvalid 1 rready 1}
    X {awaddr 34 awlen 4 awsize 3 awburst 2 awvalid 1 awready 1 wdata 256 wstrb 32 wlast 1
       wvalid 1 wready 1 bresp 2 bvalid 1 bready 1 araddr 34 arlen 4 arsize 3 arburst 2
       arvalid 1 arready 1 rdata 256 rresp 2 rlast 1 rvalid 1 rready 1}
  }
}

# the wrapper's ports: {name dir width} from its Verilog declarations ("input [33:0]X;")
proc otpu_sc_ports {text} {
  set ports {}
  foreach line [split $text "\n"] {
    if {[regexp {^\s*(input|output)\s*(?:wire\s*)?(?:\[(\d+):0\])?\s*([A-Za-z_][A-Za-z0-9_]*)\s*;} $line -> dir msb name]} {
      lappend ports [list $name $dir [expr {$msb eq "" ? 1 : $msb + 1}]]
    }
  }
  return $ports
}

proc otpu_sc_instance {text nport} {
  set w [otpu_sc_widths]
  set tie {lock cache prot qos region user}
  set conns {}
  set seen {}
  set xid {}
  foreach p [otpu_sc_ports $text] {
    lassign $p name dir width
    set c ""
    if {[dict exists {core_clk clk host_clk clk_main_a0 hbm_clk hclk aresetn sc_aresetn host_aresetn sc_host_aresetn} $name]} {
      if {$dir ne "input" || $width != 1} { error "openTPU SmartConnect: $name is not a one-bit input" }
      set c [dict get {core_clk clk host_clk clk_main_a0 hbm_clk hclk aresetn sc_aresetn host_aresetn sc_host_aresetn} $name]
    } elseif {[regexp {^(S_HOST|S_N(\d+)|M_H(\d+))_([a-z]+)$} $name -> intf ns nm sig]} {
      set fam [expr {$intf eq "S_HOST" ? "H" : ($ns ne "" ? "C" : "X")}]
      set k [expr {$ns ne "" ? $ns : $nm}]
      if {$k ne "" && ($k >= $nport)} { error "openTPU SmartConnect: $name past NPORT $nport" }
      set idx [expr {$fam eq "H" ? "" : "\[$k\]"}]
      set pre [dict get {H h C ch X x} $fam]
      set opt 0
      foreach t $tie { if {[string match *$t $sig] && ![dict exists [dict get $w $fam] $sig]} { set opt 1 } }
      if {[dict exists [dict get $w $fam] $sig]} {
        set want [dict get [dict get $w $fam] $sig]
        if {$width != $want} { error "openTPU SmartConnect: $name is $width bits, the CL's $want" }
        set c "${pre}_$sig$idx"
        lappend seen "$fam$k.$sig"
      } elseif {$fam eq "C" && $dir eq "input" && [dict exists {awlen '0 awsize 3'd6 awburst 2'b01 arsize 3'd6 arburst 2'b01 wlast 1'b1 awid '0 arid '0} $sig]} {
        set c [dict get {awlen '0 awsize 3'd6 awburst 2'b01 arsize 3'd6 arburst 2'b01 wlast 1'b1 awid '0 arid '0} $sig]
      } elseif {$fam eq "X" && $sig in {awid arid rid bid}} {
        if {$width > 6} { error "openTPU SmartConnect: $name is $width bits, the HBM's IDs 6" }
        set dirok [expr {$sig in {awid arid} ? "output" : "input"}]
        if {$dir ne $dirok} { error "openTPU SmartConnect: $name is an $dir" }
        set c "x_$sig$idx\[[expr {$width - 1}]:0\]"
        if {$sig in {awid arid}} {
          dict set xid "$k.$sig" 1
          if {$width < 6} { lappend conns [list "" "assign x_$sig$idx\[5:$width\] = '0;"] }
        }
      } elseif {$dir eq "output"} {
        set c ""
      } elseif {$opt} {
        set c "'0"
      } else {
        error "openTPU SmartConnect: the CL has no signal for input $name"
      }
    } elseif {$dir eq "output"} {
      set c ""
    } else {
      error "openTPU SmartConnect: unknown wrapper input $name"
    }
    lappend conns [list $name $c]
  }
  # every signal the CL drives or reads, on every interface
  foreach fam {H C X} {
    for {set k 0} {$k < ($fam eq "H" ? 1 : $nport)} {incr k} {
      set kk [expr {$fam eq "H" ? "" : $k}]
      foreach sig [dict keys [dict get $w $fam]] {
        if {"$fam$kk.$sig" ni $seen} {
          set intf [expr {$fam eq "H" ? "S_HOST" : ($fam eq "C" ? "S_N$k" : "M_H$k")}]
          error "openTPU SmartConnect: the wrapper has no ${intf}_$sig"
        }
      }
    }
  }
  set out "// Generated by boards/f2/build/scripts/create_hbm_sc.tcl from otpu_hbm_sc_wrapper's ports.\n"
  append out "otpu_hbm_sc_wrapper u_sc (\n"
  set items {}
  foreach c $conns {
    lassign $c name conn
    if {$name eq ""} { continue }
    lappend items "  .$name\($conn\)"
  }
  append out [join $items ",\n"] "\n);\n"
  foreach c $conns { if {[lindex $c 0] eq ""} { append out [lindex $c 1] "\n" } }
  for {set k 0} {$k < $nport} {incr k} {
    foreach sig {awid arid} {
      if {![dict exists $xid "$k.$sig"]} { append out "assign x_$sig\[$k\] = '0;\n" }
    }
  }
  return $out
}
