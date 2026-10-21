#!/usr/bin/env python3
"""Continue a verified C1/Alex synthesis DCP, without synthesis, publication or FPGA access.

Each explicit stage writes to a fresh directory. Upstream receipt SHA-256 values
are required so a path alone cannot silently select different evidence. Failed
attempts and negative-timing candidates are retained, never overwritten.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location('c1_alex_frozen_builder', HERE / 'build.py')
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
STAGES = ('link', 'implement', 'refine', 'qualify')
PREVIOUS = {'implement': 'link', 'refine': 'implement', 'qualify': 'refine'}
CHECKPOINTS = {'link': 'post_link.dcp', 'implement': 'post_route.dcp', 'refine': 'post_route.dcp'}
FORBIDDEN_CONTROLS = ('tuning.tcl', 'incremental_reference.txt', 'incremental_reference.sha256')
PHYSICAL_STRATEGIES = ('default', 'skip-pre-route-phys-opt', 'align-promoted-carry-pblocks',
                       'align-complete-promoted-carry-pblocks')


# Two synthesis-promoted carry predecessors escaped their original u_board
# hierarchy and inherited the parent Pblock instead of the hard SLR0 child.
# Tighten only those two memberships; do not rewrite logic or region bounds.
CARRY_PBLOCK_SCRIPT = r'''
namespace eval ::c1_carry_pblock {
  variable output [open $::run/carry_pblock_repair.tcldata {WRONLY CREAT EXCL}]
  set run $::run
proc emit {kind value} {
  set line [list $kind $value]
  if {[string bytelength $line] > 1048576} {error "Observation record bound"}
  puts $::c1_carry_pblock::output $line
}
proc selector {name} {return [format {NAME == "%s"} $name]}
proc one {count label} {
  if {$count != 1} {error "Expected one object for $label, got $count"}
}
set pairs [list \
  [list {WRAPPER/CL/engines[15].engine/sums_reg[0][17][3]_i_2} {WRAPPER/CL/u_board/core/cluster/engines[15].engine/sums_reg[0][17][16]_i_11}] \
  [list {WRAPPER/CL/engines[15].engine/sums_reg[0][15][2]_i_1} {WRAPPER/CL/u_board/core/cluster/engines[15].engine/sums_reg[0][15][11]_i_18}]]
set targets {}
foreach pair $pairs {foreach name $pair {dict set targets $name 1}}
emit header [dict create schema 1 kind c1_promoted_carry_pblock_repair route_qualified 0 hardware_actions 0]

# Prove the exact current primitive identities and closed carry chain boundaries.
# CI/CI_TOP are inspected through complete leaf-driver inventories. Every
# connected CO output is checked for other carry successors before mutation.
foreach pair $pairs {
  lassign $pair first last
  foreach name $pair {
    one [llength [get_cells -hierarchical -filter [selector $name]]] $name
    foreach cell [get_cells -hierarchical -filter [selector $name]] {
      if {[get_property REF_NAME $cell] ne "CARRY8" || [get_property CARRY_TYPE $cell] ne "SINGLE_CY8"} {error "Wrong carry primitive"}
      if {[get_property DONT_TOUCH $cell] ni {"" 0 false FALSE}} {error "Protected target"}
      emit target [dict create name $name ref CARRY8 carry_type SINGLE_CY8]
      foreach port {CI CI_TOP} {
        one [llength [get_pins -of_objects $cell -filter [format {REF_PIN_NAME == "%s"} $port]]] "$name/$port"
        foreach pin [get_pins -of_objects $cell -filter [format {REF_PIN_NAME == "%s"} $port]] {
          one [llength [get_nets -of_objects $pin]] "$name/$port net"
          foreach net [get_nets -of_objects $pin] {
            one [llength [get_pins -leaf -of_objects $net -filter {DIRECTION == OUT}]] "$name/$port driver"
            foreach source [get_pins -leaf -of_objects $net -filter {DIRECTION == OUT}] {
              set source_name [get_property NAME $source]
              one [llength [get_cells -of_objects $source]] $source_name
              foreach owner [get_cells -of_objects $source] {
                if {$name eq $last && $port eq "CI"} {
                  if {[get_property NAME $owner] ne $first || [get_property REF_PIN_NAME $source] ne {CO[7]}} {error "Chain input differs"}
                } elseif {[get_property REF_NAME $owner] ne "GND"} {error "Carry chain has another predecessor"}
                emit input [dict create cell $name pin $port source $source_name source_ref [get_property REF_NAME $owner]]
              }
            }
          }
        }
      }
      set successor_count 0
      foreach pin [get_pins -of_objects $cell -filter {DIRECTION == OUT}] {
        if {![regexp {^CO\[[0-7]\]$} [get_property REF_PIN_NAME $pin]]} {continue}
        foreach net [get_nets -of_objects $pin] {
          set count [llength [get_pins -leaf -of_objects $net -filter {DIRECTION == IN}]]
          if {$count > 128} {error "Carry-output endpoint bound"}
          foreach sink [get_pins -leaf -of_objects $net -filter {DIRECTION == IN}] {
            foreach owner [get_cells -of_objects $sink] {
              if {[get_property REF_NAME $owner] eq "CARRY8" && [get_property REF_PIN_NAME $sink] in {CI CI_TOP}} {
                if {$name ne $first || [get_property NAME $owner] ne $last ||
                    [get_property REF_PIN_NAME $pin] ne {CO[7]} || [get_property REF_PIN_NAME $sink] ne "CI"} {error "Unexpected carry successor"}
                incr successor_count
                emit successor [dict create source [get_property NAME $pin] sink [get_property NAME $sink]]
              }
            }
          }
        }
      }
      if {$successor_count != [expr {$name eq $first ? 1 : 0}]} {error "Carry chain closure differs"}
    }
  }
}

proc net_names {pin} {
  set nets [get_nets -of_objects $pin]
  if {![llength $nets]} {return {}}
  return [lsort [get_property NAME $nets]]
}
set block_properties {PARENT GRID_RANGES IS_SOFT CONTAIN_ROUTING SNAPPING_MODE EXCLUDE_PLACEMENT}
set region_before {}
foreach name {pblock_CL pblock_CL_SLR0} {
  set block [get_pblocks -filter [selector $name]]
  one [llength $block] $name
  foreach property $block_properties {
    dict set region_before $name $property [get_property $property $block]
  }
}
if {[dict get $region_before pblock_CL_SLR0 PARENT] ne "pblock_CL" ||
    [dict get $region_before pblock_CL_SLR0 IS_SOFT] ne "0" ||
    [dict get $region_before pblock_CL PARENT] ne "ROOT" ||
    [dict get $region_before pblock_CL IS_SOFT] ne "0"} {error "Unexpected carry region hierarchy"}
emit region_before $region_before
set pin_nets {}
foreach pair $pairs {
  lassign $pair first last
  foreach name $pair {
    set cell [get_cells -hierarchical -filter [selector $name]]
    set wanted [expr {$name eq $first ? "pblock_CL" : "pblock_CL_SLR0"}]
    if {[get_property PBLOCK $cell] ne $wanted ||
        [get_pblocks -of_objects $cell] ne $wanted} {error "Carry Pblock membership differs"}
    emit membership_before [dict create cell $name pblock $wanted]
    foreach pin [get_pins -of_objects $cell] {
      dict set pin_nets [get_property NAME $pin] [net_names $pin]
    }
  }
}
foreach pair $pairs {
  set name [lindex $pair 0]
  add_cells_to_pblock [get_pblocks pblock_CL_SLR0] [get_cells -hierarchical -filter [selector $name]]
}
set pins_after {}
foreach name [dict keys $targets] {
  set cell [get_cells -hierarchical -filter [selector $name]]
  one [llength $cell] $name
  if {[get_property REF_NAME $cell] ne "CARRY8" ||
      [get_property CARRY_TYPE $cell] ne "SINGLE_CY8" ||
      [get_property PBLOCK $cell] ne "pblock_CL_SLR0" ||
      [get_pblocks -of_objects $cell] ne "pblock_CL_SLR0"} {error "Carry membership repair incomplete"}
  foreach pin [get_pins -of_objects $cell] {
    set pin_name [get_property NAME $pin]
    lappend pins_after $pin_name
    if {[net_names $pin] ne [dict get $pin_nets $pin_name]} {error "Carry pin connectivity changed"}
  }
  emit membership_after [dict create cell $name pblock pblock_CL_SLR0]
}
if {[lsort $pins_after] ne [lsort [dict keys $pin_nets]]} {error "Carry pin inventory changed"}
foreach name {pblock_CL pblock_CL_SLR0} {
  foreach property $block_properties {
    if {[get_property $property [get_pblocks $name]] ne [dict get $region_before $name $property]} {error "Carry Pblock region changed"}
  }
}
emit complete [dict create added_cells 2 carry_primitives_and_pin_nets_preserved 1 region_properties_unchanged 1 route_qualified 0 hardware_actions 0]
close $output
}
namespace delete ::c1_carry_pblock
puts C1_CARRY_PBLOCK_REPAIR_COMPLETE_NOT_ROUTE_QUALIFIED
'''

# Complete observed promoted-to-core components. Last suffix in each tuple is
# the existing child-Pblock terminal; preceding suffixes are promoted cells.
# Source: exact post-opt raw 98717de7...11bf4, 159 disjoint components.
CARRY_COMPLETE_COMPONENTS = (
    ('sums_reg[0][0][11]_i_1', 'sums_reg[0][0][16]_i_1'),
    ('sums_reg[0][0][16]_i_8', 'sums_reg[0][0][16]_i_2'),
    ('sums_reg[0][0][2]_i_1', 'sums_reg[0][0][11]_i_18'),
    ('sums_reg[0][0][3]_i_2', 'sums_reg[0][0][16]_i_11'),
    ('sums_reg[0][10][11]_i_1', 'sums_reg[0][10][16]_i_1'),
    ('sums_reg[0][10][16]_i_8', 'sums_reg[0][10][16]_i_2'),
    ('sums_reg[0][10][2]_i_1', 'sums_reg[0][10][11]_i_18'),
    ('sums_reg[0][10][3]_i_2', 'sums_reg[0][10][16]_i_11'),
    ('sums_reg[0][11][11]_i_1', 'sums_reg[0][11][16]_i_1'),
    ('sums_reg[0][11][16]_i_8', 'sums_reg[0][11][16]_i_2'),
    ('sums_reg[0][11][2]_i_1', 'sums_reg[0][11][11]_i_18'),
    ('sums_reg[0][11][3]_i_2', 'sums_reg[0][11][16]_i_11'),
    ('sums_reg[0][12][11]_i_1', 'sums_reg[0][12][16]_i_1'),
    ('sums_reg[0][12][16]_i_8', 'sums_reg[0][12][16]_i_2'),
    ('sums_reg[0][12][2]_i_1', 'sums_reg[0][12][11]_i_18'),
    ('sums_reg[0][12][3]_i_2', 'sums_reg[0][12][16]_i_11'),
    ('sums_reg[0][13][11]_i_1', 'sums_reg[0][13][16]_i_1'),
    ('sums_reg[0][13][16]_i_8', 'sums_reg[0][13][16]_i_2'),
    ('sums_reg[0][13][2]_i_1', 'sums_reg[0][13][11]_i_18'),
    ('sums_reg[0][13][3]_i_2', 'sums_reg[0][13][16]_i_11'),
    ('sums_reg[0][14][11]_i_1', 'sums_reg[0][14][16]_i_1'),
    ('sums_reg[0][14][16]_i_8', 'sums_reg[0][14][16]_i_2'),
    ('sums_reg[0][14][2]_i_1', 'sums_reg[0][14][11]_i_18'),
    ('sums_reg[0][14][3]_i_2', 'sums_reg[0][14][16]_i_11'),
    ('sums_reg[0][15][11]_i_1', 'sums_reg[0][15][16]_i_1'),
    ('sums_reg[0][15][16]_i_8', 'sums_reg[0][15][16]_i_2'),
    ('sums_reg[0][15][2]_i_1', 'sums_reg[0][15][11]_i_18'),
    ('sums_reg[0][15][3]_i_2', 'sums_reg[0][15][16]_i_11'),
    ('sums_reg[0][16][11]_i_1', 'sums_reg[0][16][16]_i_1'),
    ('sums_reg[0][16][16]_i_8', 'sums_reg[0][16][16]_i_2'),
    ('sums_reg[0][16][2]_i_1', 'sums_reg[0][16][11]_i_18'),
    ('sums_reg[0][16][3]_i_2', 'sums_reg[0][16][16]_i_11'),
    ('sums_reg[0][17][11]_i_1', 'sums_reg[0][17][16]_i_1'),
    ('sums_reg[0][17][16]_i_8', 'sums_reg[0][17][16]_i_2'),
    ('sums_reg[0][17][2]_i_1', 'sums_reg[0][17][11]_i_18'),
    ('sums_reg[0][17][3]_i_2', 'sums_reg[0][17][16]_i_11'),
    ('sums_reg[0][18][11]_i_1', 'sums_reg[0][18][16]_i_1'),
    ('sums_reg[0][18][16]_i_8', 'sums_reg[0][18][16]_i_2'),
    ('sums_reg[0][18][2]_i_1', 'sums_reg[0][18][11]_i_18'),
    ('sums_reg[0][18][3]_i_2', 'sums_reg[0][18][16]_i_11'),
    ('sums_reg[0][19][11]_i_1', 'sums_reg[0][19][16]_i_1'),
    ('sums_reg[0][19][16]_i_8', 'sums_reg[0][19][16]_i_2'),
    ('sums_reg[0][19][2]_i_1', 'sums_reg[0][19][11]_i_18'),
    ('sums_reg[0][19][3]_i_2', 'sums_reg[0][19][16]_i_11'),
    ('sums_reg[0][1][11]_i_1', 'sums_reg[0][1][16]_i_1'),
    ('sums_reg[0][1][16]_i_8', 'sums_reg[0][1][16]_i_2'),
    ('sums_reg[0][1][2]_i_1', 'sums_reg[0][1][11]_i_18'),
    ('sums_reg[0][1][3]_i_2', 'sums_reg[0][1][16]_i_11'),
    ('sums_reg[0][20][11]_i_1', 'sums_reg[0][20][16]_i_1'),
    ('sums_reg[0][20][16]_i_8', 'sums_reg[0][20][16]_i_2'),
    ('sums_reg[0][20][2]_i_1', 'sums_reg[0][20][11]_i_18'),
    ('sums_reg[0][20][3]_i_2', 'sums_reg[0][20][16]_i_11'),
    ('sums_reg[0][21][11]_i_1', 'sums_reg[0][21][16]_i_1'),
    ('sums_reg[0][21][16]_i_8', 'sums_reg[0][21][16]_i_2'),
    ('sums_reg[0][21][2]_i_1', 'sums_reg[0][21][11]_i_18'),
    ('sums_reg[0][21][3]_i_2', 'sums_reg[0][21][16]_i_11'),
    ('sums_reg[0][22][11]_i_1', 'sums_reg[0][22][16]_i_1'),
    ('sums_reg[0][22][16]_i_8', 'sums_reg[0][22][16]_i_2'),
    ('sums_reg[0][22][2]_i_1', 'sums_reg[0][22][11]_i_18'),
    ('sums_reg[0][22][3]_i_2', 'sums_reg[0][22][16]_i_11'),
    ('sums_reg[0][23][11]_i_1', 'sums_reg[0][23][16]_i_1'),
    ('sums_reg[0][23][16]_i_8', 'sums_reg[0][23][16]_i_2'),
    ('sums_reg[0][23][2]_i_1', 'sums_reg[0][23][11]_i_18'),
    ('sums_reg[0][23][3]_i_2', 'sums_reg[0][23][16]_i_11'),
    ('sums_reg[0][24][11]_i_1', 'sums_reg[0][24][16]_i_1'),
    ('sums_reg[0][24][16]_i_8', 'sums_reg[0][24][16]_i_2'),
    ('sums_reg[0][24][2]_i_1', 'sums_reg[0][24][11]_i_18'),
    ('sums_reg[0][24][3]_i_2', 'sums_reg[0][24][16]_i_11'),
    ('sums_reg[0][25][11]_i_1', 'sums_reg[0][25][16]_i_1'),
    ('sums_reg[0][25][16]_i_8', 'sums_reg[0][25][16]_i_2'),
    ('sums_reg[0][25][2]_i_1', 'sums_reg[0][25][11]_i_18'),
    ('sums_reg[0][25][3]_i_2', 'sums_reg[0][25][16]_i_11'),
    ('sums_reg[0][26][11]_i_1', 'sums_reg[0][26][16]_i_1'),
    ('sums_reg[0][26][16]_i_8', 'sums_reg[0][26][16]_i_2'),
    ('sums_reg[0][26][2]_i_1', 'sums_reg[0][26][11]_i_18'),
    ('sums_reg[0][26][3]_i_2', 'sums_reg[0][26][16]_i_11'),
    ('sums_reg[0][27][11]_i_1', 'sums_reg[0][27][16]_i_1'),
    ('sums_reg[0][27][16]_i_8', 'sums_reg[0][27][16]_i_2'),
    ('sums_reg[0][27][2]_i_1', 'sums_reg[0][27][11]_i_18'),
    ('sums_reg[0][27][3]_i_2', 'sums_reg[0][27][16]_i_11'),
    ('sums_reg[0][28][11]_i_1', 'sums_reg[0][28][16]_i_1'),
    ('sums_reg[0][28][16]_i_8', 'sums_reg[0][28][16]_i_2'),
    ('sums_reg[0][28][2]_i_1', 'sums_reg[0][28][11]_i_18'),
    ('sums_reg[0][28][3]_i_2', 'sums_reg[0][28][16]_i_11'),
    ('sums_reg[0][29][11]_i_1', 'sums_reg[0][29][16]_i_1'),
    ('sums_reg[0][29][16]_i_8', 'sums_reg[0][29][16]_i_2'),
    ('sums_reg[0][29][2]_i_1', 'sums_reg[0][29][11]_i_18'),
    ('sums_reg[0][29][3]_i_2', 'sums_reg[0][29][16]_i_11'),
    ('sums_reg[0][2][11]_i_1', 'sums_reg[0][2][16]_i_1'),
    ('sums_reg[0][2][16]_i_8', 'sums_reg[0][2][16]_i_2'),
    ('sums_reg[0][2][2]_i_1', 'sums_reg[0][2][11]_i_18'),
    ('sums_reg[0][2][3]_i_2', 'sums_reg[0][2][16]_i_11'),
    ('sums_reg[0][30][11]_i_1', 'sums_reg[0][30][16]_i_1'),
    ('sums_reg[0][30][16]_i_8', 'sums_reg[0][30][16]_i_2'),
    ('sums_reg[0][30][2]_i_1', 'sums_reg[0][30][11]_i_18'),
    ('sums_reg[0][30][3]_i_2', 'sums_reg[0][30][16]_i_11'),
    ('sums_reg[0][31][11]_i_1', 'sums_reg[0][31][16]_i_1'),
    ('sums_reg[0][31][16]_i_8', 'sums_reg[0][31][16]_i_2'),
    ('sums_reg[0][31][2]_i_1', 'sums_reg[0][31][11]_i_18'),
    ('sums_reg[0][31][3]_i_2', 'sums_reg[0][31][16]_i_11'),
    ('sums_reg[0][3][11]_i_1', 'sums_reg[0][3][16]_i_1'),
    ('sums_reg[0][3][16]_i_8', 'sums_reg[0][3][16]_i_2'),
    ('sums_reg[0][3][2]_i_1', 'sums_reg[0][3][11]_i_18'),
    ('sums_reg[0][3][3]_i_2', 'sums_reg[0][3][16]_i_11'),
    ('sums_reg[0][4][11]_i_1', 'sums_reg[0][4][16]_i_1'),
    ('sums_reg[0][4][16]_i_8', 'sums_reg[0][4][16]_i_2'),
    ('sums_reg[0][4][2]_i_1', 'sums_reg[0][4][11]_i_18'),
    ('sums_reg[0][4][3]_i_2', 'sums_reg[0][4][16]_i_11'),
    ('sums_reg[0][5][11]_i_1', 'sums_reg[0][5][16]_i_1'),
    ('sums_reg[0][5][16]_i_8', 'sums_reg[0][5][16]_i_2'),
    ('sums_reg[0][5][2]_i_1', 'sums_reg[0][5][11]_i_18'),
    ('sums_reg[0][5][3]_i_2', 'sums_reg[0][5][16]_i_11'),
    ('sums_reg[0][6][11]_i_1', 'sums_reg[0][6][16]_i_1'),
    ('sums_reg[0][6][16]_i_8', 'sums_reg[0][6][16]_i_2'),
    ('sums_reg[0][6][2]_i_1', 'sums_reg[0][6][11]_i_18'),
    ('sums_reg[0][6][3]_i_2', 'sums_reg[0][6][16]_i_11'),
    ('sums_reg[0][7][11]_i_1', 'sums_reg[0][7][16]_i_1'),
    ('sums_reg[0][7][16]_i_8', 'sums_reg[0][7][16]_i_2'),
    ('sums_reg[0][7][2]_i_1', 'sums_reg[0][7][11]_i_18'),
    ('sums_reg[0][7][3]_i_2', 'sums_reg[0][7][16]_i_11'),
    ('sums_reg[0][8][11]_i_1', 'sums_reg[0][8][16]_i_1'),
    ('sums_reg[0][8][16]_i_8', 'sums_reg[0][8][16]_i_2'),
    ('sums_reg[0][8][2]_i_1', 'sums_reg[0][8][11]_i_18'),
    ('sums_reg[0][8][3]_i_2', 'sums_reg[0][8][16]_i_11'),
    ('sums_reg[0][9][11]_i_1', 'sums_reg[0][9][16]_i_1'),
    ('sums_reg[0][9][16]_i_8', 'sums_reg[0][9][16]_i_2'),
    ('sums_reg[0][9][2]_i_1', 'sums_reg[0][9][11]_i_18'),
    ('sums_reg[0][9][3]_i_2', 'sums_reg[0][9][16]_i_11'),
    ('sums_reg[1][0][7]_i_1', 'sums_reg[1][0][15]_i_1', 'sums_reg[1][0][20]_i_1'),
    ('sums_reg[1][10][7]_i_1', 'sums_reg[1][10][15]_i_1', 'sums_reg[1][10][20]_i_1'),
    ('sums_reg[1][11][7]_i_1', 'sums_reg[1][11][15]_i_1', 'sums_reg[1][11][20]_i_1'),
    ('sums_reg[1][12][7]_i_1', 'sums_reg[1][12][15]_i_1', 'sums_reg[1][12][20]_i_1'),
    ('sums_reg[1][13][7]_i_1', 'sums_reg[1][13][15]_i_1', 'sums_reg[1][13][20]_i_1'),
    ('sums_reg[1][14][7]_i_1', 'sums_reg[1][14][15]_i_1', 'sums_reg[1][14][20]_i_1'),
    ('sums_reg[1][15][7]_i_1', 'sums_reg[1][15][15]_i_1', 'sums_reg[1][15][20]_i_1'),
    ('sums_reg[1][1][7]_i_1', 'sums_reg[1][1][15]_i_1', 'sums_reg[1][1][20]_i_1'),
    ('sums_reg[1][2][7]_i_1', 'sums_reg[1][2][15]_i_1', 'sums_reg[1][2][20]_i_1'),
    ('sums_reg[1][3][7]_i_1', 'sums_reg[1][3][15]_i_1', 'sums_reg[1][3][20]_i_1'),
    ('sums_reg[1][4][7]_i_1', 'sums_reg[1][4][15]_i_1', 'sums_reg[1][4][20]_i_1'),
    ('sums_reg[1][5][7]_i_1', 'sums_reg[1][5][15]_i_1', 'sums_reg[1][5][20]_i_1'),
    ('sums_reg[1][6][7]_i_1', 'sums_reg[1][6][15]_i_1', 'sums_reg[1][6][20]_i_1'),
    ('sums_reg[1][7][7]_i_1', 'sums_reg[1][7][15]_i_1', 'sums_reg[1][7][20]_i_1'),
    ('sums_reg[1][8][7]_i_1', 'sums_reg[1][8][15]_i_1', 'sums_reg[1][8][20]_i_1'),
    ('sums_reg[1][9][7]_i_1', 'sums_reg[1][9][15]_i_1', 'sums_reg[1][9][20]_i_1'),
    ('sums_reg[2][0][7]_i_1', 'sums_reg[2][0][15]_i_1', 'sums_reg[2][0][20]_i_1'),
    ('sums_reg[2][1][7]_i_1', 'sums_reg[2][1][15]_i_1', 'sums_reg[2][1][20]_i_1'),
    ('sums_reg[2][2][7]_i_1', 'sums_reg[2][2][15]_i_1', 'sums_reg[2][2][20]_i_1'),
    ('sums_reg[2][3][7]_i_1', 'sums_reg[2][3][15]_i_1', 'sums_reg[2][3][20]_i_1'),
    ('sums_reg[2][4][7]_i_1', 'sums_reg[2][4][15]_i_1', 'sums_reg[2][4][20]_i_1'),
    ('sums_reg[2][5][7]_i_1', 'sums_reg[2][5][15]_i_1', 'sums_reg[2][5][20]_i_1'),
    ('sums_reg[2][6][7]_i_1', 'sums_reg[2][6][15]_i_1', 'sums_reg[2][6][20]_i_1'),
    ('sums_reg[2][7][7]_i_1', 'sums_reg[2][7][15]_i_1', 'sums_reg[2][7][20]_i_1'),
    ('sums_reg[3][0][7]_i_1', 'sums_reg[3][0][15]_i_1', 'sums_reg[3][0][20]_i_1'),
    ('sums_reg[3][1][7]_i_1', 'sums_reg[3][1][15]_i_1', 'sums_reg[3][1][20]_i_1'),
    ('sums_reg[3][2][7]_i_1', 'sums_reg[3][2][15]_i_1', 'sums_reg[3][2][20]_i_1'),
    ('sums_reg[3][3][7]_i_1', 'sums_reg[3][3][15]_i_1', 'sums_reg[3][3][20]_i_1'),
    ('sums_reg[4][0][7]_i_1', 'sums_reg[4][0][15]_i_1', 'sums_reg[4][0][20]_i_1'),
    ('sums_reg[4][1][7]_i_1', 'sums_reg[4][1][15]_i_1', 'sums_reg[4][1][20]_i_1'),
    ('sums_reg[5][0][7]_i_1', 'sums_reg[5][0][15]_i_1', 'sums_reg[5][0][20]_i_1'),
)

CARRY_COMPLETE_TEMPLATE = r'''
namespace eval ::c1_complete_carry_pblock {
  variable output [open $::run/complete_carry_pblock_repair.tcldata {WRONLY CREAT EXCL}]
  variable emitted 0
proc emit {kind value} {
  set line [list $kind $value]
  incr ::c1_complete_carry_pblock::emitted [expr {[string bytelength $line]+1}]
  if {[string bytelength $line]>1048576 || $::c1_complete_carry_pblock::emitted>16777216} {error "Carry observation bound"}
  puts $::c1_complete_carry_pblock::output $line
}
proc names {objects} {
  set result {}
  foreach object $objects {lappend result [get_property NAME $object]}
  if {[llength $result]>65536 || [llength $result]!=[llength [lsort -unique $result]]} {error "Carry object inventory bound or duplicate"}
  return [lsort $result]
}
proc cell {name} {
  set objects [get_cells -quiet -hierarchical -filter [format {NAME == "%s"} $name]]
  if {[names $objects] ne [list $name]} {error "Missing exact carry cell"}
  return $objects
}
proc pin {name} {
  set objects [get_pins -quiet [list $name]]
  if {[names $objects] ne [list $name]} {error "Missing exact carry pin"}
  return $objects
}
proc driver {name expected} {
  set nets [get_nets -quiet -of_objects [pin $name]]
  if {[llength $nets]!=1 || [llength [get_ports -quiet -of_objects $nets]]} {error "Carry input net boundary differs"}
  set drivers [get_pins -quiet -leaf -of_objects $nets -filter {DIRECTION == OUT}]
  if {[names $drivers] ne [list $expected]} {error "Carry input driver differs"}
  if {[get_property REF_NAME [get_cells -quiet -of_objects $drivers]] ne [expr {[string match */G $expected] ? "GND" : "CARRY8"}]} {error "Carry input driver primitive differs"}
  emit input [list pin $name driver $expected]
}
proc snapshot {name} {
  set object [cell $name];set props {};set available [list_property $object]
  foreach property {NAME PARENT REF_NAME CARRY_TYPE DONT_TOUCH KEEP KEEP_HIERARCHY MARK_DEBUG LOC BEL RLOC RLOC_ORIGIN RLOC_RANGE IS_LOC_FIXED IS_BEL_FIXED LOCK_PINS} {
    if {$property in $available} {dict set props $property [list observed [get_property $property $object]]} else {dict set props $property unavailable}
  }
  set nets {}
  foreach p [get_pins -of_objects $object] {dict set nets [get_property NAME $p] [names [get_nets -quiet -of_objects $p]]}
  return [list properties $props pin_nets $nets]
}
set chains @CHAINS@
set prior @PRIOR@
set promoted {};set terminals {};set all {}
foreach chain $chains {
  if {[llength $chain] ni {2 3}} {error "Carry component length differs"}
  foreach name $chain {
    if {$name in $all} {error "Duplicate carry component cell"}
    lappend all $name
  }
  lappend terminals [lindex $chain end]
  foreach name [lrange $chain 0 end-1] {lappend promoted $name}
}
set promoted [lsort $promoted]
if {[llength $chains]!=159 || [llength $promoted]!=190 || [llength $all]!=349 || [llength $prior]!=2} {error "Carry component inventory differs"}
if {[names [get_cells -quiet -hierarchical -filter {REF_NAME == CARRY8 && NAME =~ WRAPPER/CL/engines*}]] ne $promoted} {error "Complete promoted carry inventory differs"}
foreach name $prior {if {$name ni $promoted} {error "Prior repair outside inventory"}}
emit header [list schema 1 kind c1_complete_promoted_carry_pblock_repair raw_sha256 98717de7476163a94513566d0997ae1680292cc3be616cc9d4cf7f5227511bf4 source_checkpoint_sha256 5865dbfd47ca1636d66cb8a2d03754739541bd7b67a632575f8d9ad6d1ffad95 prior_repair_cells $prior route_qualified 0 hardware_actions 0]
set region_before {};set region_properties {PARENT GRID_RANGES IS_SOFT CONTAIN_ROUTING SNAPPING_MODE EXCLUDE_PLACEMENT}
foreach name {pblock_CL pblock_CL_SLR0} {
  set block [get_pblocks -quiet [list $name]]
  if {[names $block] ne [list $name]} {error "Missing exact carry region"}
  foreach property $region_properties {dict set region_before $name $property [get_property $property $block]}
}
if {[dict get $region_before pblock_CL PARENT] ne "ROOT" || [dict get $region_before pblock_CL IS_SOFT] ne "0" || [dict get $region_before pblock_CL_SLR0 PARENT] ne "pblock_CL" || [dict get $region_before pblock_CL_SLR0 IS_SOFT] ne "0"} {error "Carry hard region hierarchy differs"}
emit region_before $region_before
set before {}
foreach name $all {
  set c [cell $name]
  set parent [expr {$name in $promoted ? "WRAPPER/CL" : {WRAPPER/CL/u_board/core/cluster/engines[15].engine}}]
  set membership [expr {$name in $promoted && $name ni $prior ? "pblock_CL" : "pblock_CL_SLR0"}]
  if {[get_property REF_NAME $c] ne "CARRY8" || [get_property CARRY_TYPE $c] ne "SINGLE_CY8" || [get_property PARENT $c] ne $parent || [get_property DONT_TOUCH $c] ni {"" 0 false FALSE}} {error "Carry primitive or hierarchy differs"}
  if {[get_property PBLOCK $c] ne $membership || [names [get_pblocks -of_objects $c]] ne [list $membership]} {error "Carry pre-repair membership differs"}
  dict set before $name [snapshot $name]
  emit before [list cell $name pblock $membership snapshot [dict get $before $name]]
}
foreach chain $chains {
  set previous WRAPPER/CL/GND/G
  for {set i 0} {$i<[llength $chain]-1} {incr i} {
    set name [lindex $chain $i];set next [lindex $chain [expr {$i+1}]]
    driver $name/CI $previous
    driver $name/CI_TOP WRAPPER/CL/GND_1/G
    set output_pin $name/CO\[7\]
    set nets [get_nets -quiet -segments -of_objects [pin $output_pin]]
    if {![llength $nets] || [llength $nets]>16 || [llength [get_ports -quiet -of_objects $nets]]} {error "Carry output net boundary differs"}
    set leaves [names [get_pins -quiet -leaf -of_objects $nets]]
    if {$leaves ne [lsort [list $output_pin $next/CI]] || [names [get_pins -quiet -leaf -of_objects $nets -filter {DIRECTION == OUT}]] ne [list $output_pin]} {error "Carry output endpoint inventory differs"}
    emit successor [list source $output_pin sink $next/CI]
    set previous $output_pin
  }
}
set added {}
foreach name $promoted {
  if {$name in $prior} {continue}
  add_cells_to_pblock [get_pblocks pblock_CL_SLR0] [cell $name]
  lappend added $name
}
if {[llength $added]!=188} {error "Carry repair cardinality differs"}
foreach name $all {
  set c [cell $name]
  if {[get_property PBLOCK $c] ne "pblock_CL_SLR0" || [names [get_pblocks -of_objects $c]] ne [list pblock_CL_SLR0]} {error "Carry membership repair incomplete"}
  if {[snapshot $name] ne [dict get $before $name]} {error "Carry primitive, pin inventory or connectivity changed"}
  emit after [list cell $name pblock pblock_CL_SLR0]
}
foreach name {pblock_CL pblock_CL_SLR0} {
  foreach property $region_properties {
    if {[get_property $property [get_pblocks $name]] ne [dict get $region_before $name $property]} {error "Carry region changed"}
  }
}
emit complete [list added_cells 188 prior_repaired_cells 2 promoted_cells 190 terminal_carries 159 carry_primitives_and_pin_nets_preserved 1 region_properties_unchanged 1 route_qualified 0 hardware_actions 0]
close $::c1_complete_carry_pblock::output
}
namespace delete ::c1_complete_carry_pblock
puts C1_COMPLETE_CARRY_PBLOCK_REPAIR_COMPLETE_NOT_ROUTE_QUALIFIED
'''

def complete_carry_pblock_script():
    promoted_prefix = 'WRAPPER/CL/engines[15].engine/'
    terminal_prefix = 'WRAPPER/CL/u_board/core/cluster/engines[15].engine/'
    chains = []
    for component in CARRY_COMPLETE_COMPONENTS:
        require(len(component) in (2, 3), 'Invalid observed carry component')
        require(all(re.fullmatch(r'sums_reg\[[0-5]\]\[[0-9]+\]\[[0-9]+\]_i_[0-9]+', suffix)
                    for suffix in component), 'Invalid observed carry suffix')
        names = [promoted_prefix + suffix for suffix in component[:-1]] + [terminal_prefix + component[-1]]
        chains.append('{' + ' '.join('{' + name + '}' for name in names) + '}')
    prior = [promoted_prefix + suffix for suffix in ('sums_reg[0][17][3]_i_2', 'sums_reg[0][15][2]_i_1')]
    return CARRY_COMPLETE_TEMPLATE.replace('@CHAINS@', '{' + ' '.join(chains) + '}').replace(
        '@PRIOR@', '{' + ' '.join('{' + name + '}' for name in prior) + '}')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Expected lowercase SHA-256')
    return value


def read_bound(path, expected):
    require(path.is_file() and not path.is_symlink(), f'Missing or linked receipt: {path}')
    require(b.sha(path) == digest(expected), f'Receipt hash mismatch: {path}')
    return json.loads(path.read_text())


def relative(name):
    require(isinstance(name, str), 'Manifest path must be a string')
    path = Path(name)
    require(isinstance(name, str) and name and not path.is_absolute() and
            '..' not in path.parts and '.' not in path.parts and str(path) == name,
            f'Unsafe manifest path: {name}')
    return path


def verify_files(root, files):
    require(isinstance(files, dict) and files, f'Empty file inventory: {root}')
    for name, record in files.items():
        path = root / relative(name)
        require(path.is_file() and not path.is_symlink(), f'Missing or linked frozen file: {path}')
        require(path.stat().st_size == record['bytes'] and b.sha(path) == digest(record['sha256']),
                f'Frozen file changed: {path}')


def verify_synthesis(run, receipt_sha256, hdk):
    receipt = read_bound(run / 'synth-complete.json', receipt_sha256)
    require(receipt.get('schema') == 1 and receipt.get('kind') == 'c1-alex-synthesis-only' and
            receipt.get('qualification') == 'not-qualified', 'Not a C1/Alex synthesis receipt')
    inputs = read_bound(run / 'inputs.json', receipt['inputs_sha256'])
    require(inputs.get('schema') == 1 and inputs.get('kind') == 'c1-alex-burst200-build-inputs',
            'Wrong synthesis input schema')
    require(re.fullmatch('[0-9a-f]{40}', inputs.get('source_revision', '')) and
            inputs.get('build_id') == inputs['source_revision'][:8] and inputs.get('source_status') == '',
            'Synthesis source was not a clean, identified revision')
    require((inputs.get('part'), inputs.get('core_khz'), inputs.get('hbm_mhz'),
             inputs.get('hbm_ports'), inputs.get('channel_bytes')) ==
            (b.PART, 200000, 450, 8, 2**29), 'Wrong synthesized device/clock/memory geometry')
    require(inputs['alex_revision'] == b.ALEX_REVISION and inputs['archive_sha256'] == b.ARCHIVE_SHA256,
            'Wrong Alex source/archive')
    b.verify_support()
    require(inputs['support_manifest_sha256'] == b.sha(b.SUPPORT / 'manifest.json'),
            'Frozen Alex manifest changed')
    require(b.safe_path(inputs['hdk']) == hdk, 'Installed HDK path differs from synthesis')
    verify_files(run, inputs['prepared_files'])
    # The executor may be a later tooling-only commit, but hardware/generation
    # sources must still match the exact synthesized input bytes.
    require(set(inputs['source_inputs']) == set(b.source_inputs()), 'Incomplete synthesized source inventory')
    verify_files(ROOT, inputs['source_inputs'])
    require(json.loads((run / 'synth/exit.json').read_text()) == {'returncode': 0},
            'Synthesis process did not finish successfully')
    require((run / 'synth/synth_done.txt').read_text().strip() == 'C1_ALEX_SYNTH_ONLY_COMPLETE',
            'Synthesis completion marker missing')
    checkpoint = run / 'synth/post_synth.dcp'
    require(checkpoint.is_file() and not checkpoint.is_symlink() and
            b.sha(checkpoint) == receipt['checkpoint_sha256'], 'Synthesis checkpoint changed')
    binding = json.loads((run / 'ip-binding.json').read_text())
    require(binding['receipt_sha256'] == receipt['ip_receipt_sha256'], 'Synthesis IP binding changed')
    ip_run = b.safe_path(binding['run'])
    require(b.sha(ip_run / 'ip-complete.json') == receipt['ip_receipt_sha256'], 'IP receipt changed')
    b.verify_ip_run(ip_run, inputs)
    installed = b.check_hdk(hdk)
    require(installed == inputs['installed_inputs'], 'Installed HDK inputs differ from synthesis')
    return {'run': run, 'receipt': receipt, 'receipt_sha256': receipt_sha256,
            'inputs': inputs, 'ip_run': ip_run, 'checkpoint': checkpoint, 'installed': installed}


def verify_previous(run, expected_sha256, expected_stage, origin):
    result = read_bound(run / 'stage-result.json', expected_sha256)
    require(result.get('schema') == 1 and result.get('kind') == 'c1-alex-checkpoint-stage' and
            result.get('stage') == expected_stage, 'Wrong upstream checkpoint stage')
    required_status = 'candidate' if expected_stage == 'implement' else 'complete'
    require(result.get('status') == required_status and result.get('route_qualified') is False and
            result.get('returncode') == 0 and result.get('hardware_actions') is False,
            'Upstream stage did not complete at the required gate')
    inputs = read_bound(run / 'stage-inputs.json', result['inputs_sha256'])
    require(inputs.get('schema') == 1 and inputs.get('kind') == 'c1-alex-checkpoint-stage-inputs' and
            inputs['stage'] == expected_stage and inputs.get('resynthesis') is False and
            inputs.get('hardware_actions') is False and type(inputs.get('max_threads')) is int and
            inputs.get('max_threads') in ((4, 8) if expected_stage in ('implement', 'refine') else (4,)) and
            inputs.get('shell_sha256') == b.SHELL_SHA256 and
            inputs['synth_receipt_sha256'] == origin['receipt_sha256'] and
            inputs['synth_inputs_sha256'] == origin['receipt']['inputs_sha256'] and
            inputs['source_revision'] == origin['inputs']['source_revision'] and
            inputs['build_id'] == origin['inputs']['build_id'] and
            inputs['ip_receipt_sha256'] == origin['receipt']['ip_receipt_sha256'],
            'Upstream stage belongs to different synthesized inputs')
    verify_files(run, inputs['prepared_files'])
    verify_files(run, result['outputs'])
    if expected_stage in ('implement', 'refine'):
        strategy = inputs.get('physical_strategy', 'default')
        require(expected_stage == 'implement' or strategy == 'default', 'Refine strategy differs')
        require((run / 'run.tcl').read_text() ==
                physical_script((origin['run'] / expected_stage / 'build.tcl').read_text(), strategy, inputs['max_threads']),
                'Physical strategy/script binding differs')
    if expected_stage == 'implement':
        for name in ('floorplan.tcl', 'core_timing.tcl', 'platform_constraints.tcl'):
            require((run / name).read_bytes() == (origin['run'] / 'implement' / name).read_bytes(),
                    'Physical strategy changed frozen constraints')
    checkpoint = run / CHECKPOINTS[expected_stage]
    require(result.get('checkpoint') == CHECKPOINTS[expected_stage] and
            b.sha(checkpoint) == result.get('checkpoint_sha256'), 'Upstream checkpoint binding mismatch')
    require(result['source_checkpoint_sha256'] == inputs['source_checkpoint']['sha256'],
            'Upstream source checkpoint binding mismatch')
    if expected_stage == 'link':
        require(inputs['previous_run'] is None and inputs['previous_receipt_sha256'] is None,
                'Link must consume synthesis directly')
        source = origin['checkpoint']
    else:
        ancestor = verify_previous(b.safe_path(inputs['previous_run']), inputs['previous_receipt_sha256'],
                                   PREVIOUS[expected_stage], origin)
        source = ancestor['checkpoint']
    require(b.safe_path(inputs['source_checkpoint']['path']) == source and
            b.sha(source) == inputs['source_checkpoint']['sha256'] and
            source.stat().st_size == inputs['source_checkpoint']['bytes'],
            'Upstream checkpoint ancestry changed')
    if expected_stage == 'refine':
        timing = timing_result(run / 'implementation_result.txt')
        require(timing == result['timing'] and min(timing.values()) >= 0,
                'Refinement has not passed setup/hold timing')
    return {'run': run, 'result': result, 'receipt_sha256': expected_sha256,
            'inputs': inputs, 'checkpoint': checkpoint}


def timing_result(path):
    lines = path.read_text().splitlines()
    data = {}
    for key in ('setup_slack_ns', 'hold_slack_ns'):
        matches = [line[len(key)+1:] for line in lines if line.startswith(key + '=')]
        require(len(matches) == 1, f'Missing/duplicate timing field: {key}')
        try:
            value = float(matches[0])
        except ValueError as error:
            raise RuntimeError(f'Invalid timing field: {key}') from error
        require(math.isfinite(value), f'Nonfinite timing field: {key}')
        data[key] = value
    return data


def machine_result(path, required):
    result = {}
    for line in path.read_text().splitlines():
        require('=' in line, 'Malformed stage audit result')
        key, value = line.split('=', 1)
        require(key in required and key not in result, f'Unexpected/duplicate audit field: {key}')
        result[key] = value
    require(set(result) == set(required), 'Missing stage audit field')
    return result


def simulation_helper():
    spec = importlib.util.spec_from_file_location('c1_alex_verified_simulator', HERE / 'simulate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_simulation(run, receipt_sha256, origin):
    sim = simulation_helper()
    receipt = read_bound(run / 'simulation-complete.json', receipt_sha256)
    require(receipt.get('schema') == 1 and receipt.get('kind') == 'c1-alex-vendor-fabric-simulation' and
            receipt.get('state') == 'passed' and receipt.get('hardware_actions') is False and
            receipt.get('qualification') == 'protocol-simulation-only-not-physical-timing',
            'A passed actual-vendor simulation receipt is required')
    inputs = read_bound(run / 'inputs.json', receipt['inputs_sha256'])
    evidence = [run / 'vivado_version.txt']
    for seed, phase in sim.SCENARIOS:
        scenario = run / f'seed-{seed}-phase-{phase}'
        evidence.extend(scenario / name for name in
                        ('rtl', 'simulate.tcl', 'inputs.json', 'command.json',
                         'console.log', 'exit.json', 'result.json'))
    verify_files(run, receipt.get('evidence', {}))
    require(b.inventory(evidence, run) == receipt['evidence'],
            'Vendor simulation evidence inventory is incomplete or changed')
    b.check_vivado((run / 'vivado_version.txt').read_text())
    expected_ip_sha = origin['receipt']['ip_receipt_sha256']
    ip_receipt = read_bound(origin['ip_run'] / 'ip-complete.json', expected_ip_sha)
    ip_inputs = read_bound(origin['ip_run'] / 'inputs.json', ip_receipt['inputs_sha256'])
    require(inputs.get('schema') == 1 and inputs.get('kind') == 'c1-alex-vendor-fabric-simulation' and
            inputs.get('ip_receipt_sha256') == expected_ip_sha and
            inputs.get('build_inputs_sha256') == ip_receipt['inputs_sha256'] and
            inputs.get('ip_source_revision') == ip_inputs['source_revision'] and
            inputs.get('ip_products') == ip_receipt['products'], 'Simulation belongs to different private IP')
    require(inputs.get('clocks_mhz') == {'core': 200, 'host': 250, 'hbm_nominal': 450} and
            inputs.get('hbm_simulated_period_ps') == 2222 and
            inputs.get('hbm_model') == 'functional-not-silicon-timing' and
            inputs.get('qualification') == 'protocol-simulation-only-not-physical-timing' and
            inputs.get('hardware_actions') is False, 'Simulation clock/model contract changed')
    expected_scenarios = [list(pair) for pair in sim.SCENARIOS]
    require(inputs.get('scenarios') == expected_scenarios and len(receipt.get('results', [])) == 4,
            'Four exact seed/phase simulation scenarios are required')
    require(inputs.get('runner_sha256') == b.sha(HERE / 'simulate.py') and
            inputs.get('builder_sha256') == b.sha(HERE / 'build.py'), 'Simulation tool source changed')
    sources = {name: b.sha(path) for name, path in sim.SOURCES.items()}
    require(inputs.get('sources') == sources, 'Simulation source ingredients changed')
    private = run / 'private-ip'
    sim.verified_ip(private, expected_ip_sha)
    # Includes Tcl/headers used by vendor simulation, in addition to the private
    # synthesis products. The simulator owns the precise immutable-file policy.
    require(isinstance(inputs.get('simulation_ip_files'), dict) and inputs['simulation_ip_files'],
            'Missing vendor simulation dependency inventory')
    sim.verify_simulation_ip_files(private, inputs['simulation_ip_files'])
    for (seed, phase), result in zip(sim.SCENARIOS, receipt['results']):
        scenario = run / f'seed-{seed}-phase-{phase}'
        command = json.loads((scenario / 'command.json').read_text())
        require(isinstance(command, list) and len(command) == 5 and
                isinstance(command[0], str) and command[0] and
                command[1:] == ['-mode', 'batch', '-source', str(scenario / 'simulate.tcl')],
                'Vendor scenario command changed')
        require(json.loads((scenario / 'inputs.json').read_text()) ==
                b.inventory([scenario / 'rtl', scenario / 'simulate.tcl'], scenario),
                'Vendor scenario consumed input inventory changed')
        require(json.loads((scenario / 'exit.json').read_text()) == {'returncode': 0},
                'Vendor scenario process did not succeed')
        require(sim.parse_result((scenario / 'console.log').read_text(), seed, phase) == result and
                json.loads((scenario / 'result.json').read_text()) == result,
                'Vendor scenario scoreboards/results do not match the receipt')
        for name, wanted in sources.items():
            require(b.sha(scenario / 'rtl' / name) == wanted, 'Vendor scenario source changed')
        require((scenario / 'simulate.tcl').read_text() == sim.simulation_tcl(scenario, private, seed, phase),
                'Vendor scenario invocation changed')
    return {'run': run, 'receipt_sha256': receipt_sha256, 'inputs_sha256': receipt['inputs_sha256']}


def write_hook_result(stage):
    if stage == 'link':
        return '''set result [::c1_alex_route::audit_linked $run]
if {![dict exists $result structural_pass] || [dict get $result structural_pass] ne "1" ||
    ![dict exists $result route_qualified] || [dict get $result route_qualified] ne "0"} {
  error "C1 Alex linked structural audit did not pass"
}
set f [open $run/audit-result.txt {WRONLY CREAT EXCL}]
puts $f "structural_pass=1\nroute_qualified=0"
close $f
'''
    return '''set result [::c1_alex_route::qualify $run $checkpoint]
if {![dict exists $result route_qualified] || [dict get $result route_qualified] ni {0 1} ||
    ![dict exists $result checkpoint_sha256] ||
    ![regexp {^[0-9a-f]{64}$} [dict get $result checkpoint_sha256]]} {
  error "C1 Alex qualifier returned malformed result"
}
set f [open $run/audit-result.txt {WRONLY CREAT EXCL}]
puts $f "route_qualified=[dict get $result route_qualified]"
puts $f "checkpoint_sha256=[dict get $result checkpoint_sha256]"
close $f
'''


def render_link(origin):
    original = (origin['run'] / 'link/build.tcl').read_text()
    start = original.index('# Linking the supplied static shell is mandatory; never substitute a shell stub.')
    text = original[start:]
    text = text[text.index('create_project -in_memory'):text.index('write_checkpoint -force $run/post_link.dcp')]
    require(text.count('$run/post_synth.dcp') == 2, 'Alex shell-link checkpoint anchors changed')
    text = text.replace('$run/post_synth.dcp', '$checkpoint')
    header = '''# Link a verified existing synthesis DCP. No RTL synthesis is performed.
set run [file normalize [lindex $argv 0]]
set checkpoint [file normalize [lindex $argv 1]]
set hdk [file normalize [lindex $argv 2]]
set shell $hdk/common/shell_stable
set part xcvu47p-fsvh2892-2-e
set core_khz 200000
cd $run
set_param general.maxThreads 4
set_param sta.enableAutoGenClkNamePersistence 0
'''
    script = header + text + '''write_checkpoint -force $run/post_link.dcp
source $run/qualify_c1_alex.tcl
''' + write_hook_result('link') + '''set f [open $run/link_done.txt {WRONLY CREAT EXCL}]
puts $f C1_ALEX_LINK_COMPLETE_NOT_ROUTE_QUALIFIED
close $f
'''
    require('synth_design' not in script and 'read_verilog' not in script and 'generate_target' not in script,
            'Link continuation unexpectedly requests synthesis')
    return script


def executor_inputs(stage):
    paths = [HERE / 'implement.py', HERE / 'build.py']
    if stage in ('link', 'qualify'):
        paths += [HERE / 'qualify_c1_alex.tcl', HERE / 'audit_c1_alex.tcl']
    if stage == 'qualify':
        paths += [HERE / 'simulate.py', HERE / 'routed_host_ready.tcl']
    return b.inventory(paths, ROOT)


def physical_script(script, strategy, max_threads=4):
    """Select one explicit implementation flow without changing its constraints."""
    require(strategy in PHYSICAL_STRATEGIES, 'Unknown physical implementation strategy')
    require('synth_design' not in script, 'Physical continuation unexpectedly requests synthesis')
    require(type(max_threads) is int and max_threads in (4, 8), 'Thread limit must be 4 or 8')
    anchors = re.findall(r'(?m)^set_param general\.maxThreads 4$', script)
    require(anchors and len(anchors) == script.count('general.maxThreads'), 'Thread limit anchor changed')
    script = re.sub(r'(?m)^set_param general\.maxThreads 4$', f'set_param general.maxThreads {max_threads}', script)
    if strategy in ('skip-pre-route-phys-opt', 'align-promoted-carry-pblocks',
                    'align-complete-promoted-carry-pblocks'):
        # The first packed C1 route failed on two CARRY8 cascade pins after
        # AggressiveExplore. Keep placement/routing and every timing constraint;
        # this explicit alternative omits only that pre-route optimization pass.
        old = '  phys_opt_design -directive AggressiveExplore\n'
        require(script.count(old) == 1, 'Physical optimization anchor changed')
        script = script.replace(old, '  puts C1_PRE_ROUTE_PHYS_OPT_SKIPPED_BY_EXPLICIT_STRATEGY\n')
    if strategy in ('align-promoted-carry-pblocks', 'align-complete-promoted-carry-pblocks'):
        old = 'write_checkpoint -force $run/post_opt.dcp\n'
        require(script.count(old) == 1, 'Post-opt checkpoint anchor changed')
        repair = CARRY_PBLOCK_SCRIPT
        if strategy == 'align-complete-promoted-carry-pblocks':
            # Replay the historical repair first: the complete raw inventory was
            # captured after these two exact membership additions.
            repair += complete_carry_pblock_script()
        script = script.replace(old, repair + old)
    return script


def prepare(out, stage, origin, previous, hdk, simulation=None, physical_strategy='default', max_threads=4):
    require(not out.exists(), f'Refusing existing stage output: {out}')
    require(type(max_threads) is int and max_threads in (4, 8) and
            (stage in ('implement', 'refine') or max_threads == 4),
            'Thread limit must be 4, or 8 for implement/refine')
    require(physical_strategy in PHYSICAL_STRATEGIES and
            (stage == 'implement' or physical_strategy == 'default'),
            'Physical strategy applies only to implementation')
    executor_revision = b.require_clean_source()
    tools = executor_inputs(stage)
    require((stage == 'qualify') == bool(simulation), 'Only qualification requires verified vendor simulation')
    checkpoint = origin['checkpoint'] if stage == 'link' else previous['checkpoint']
    source = {'path': str(checkpoint), 'sha256': b.sha(checkpoint), 'bytes': checkpoint.stat().st_size}
    out.mkdir(parents=True)
    if stage == 'link':
        script = render_link(origin)
    elif stage in ('implement', 'refine'):
        script = (origin['run'] / stage / 'build.tcl').read_text()
        script = physical_script(script, physical_strategy, max_threads)
        if stage == 'implement':
            for name in ('floorplan.tcl', 'core_timing.tcl', 'platform_constraints.tcl'):
                shutil.copy2(origin['run'] / 'implement' / name, out / name)
    else:
        script = '''# Independent qualification: reopen the exact final DCP without modifications.
set run [file normalize [lindex $argv 0]]
set checkpoint [file normalize [lindex $argv 1]]
cd $run
set_param general.maxThreads 4
open_checkpoint $checkpoint
source $run/qualify_c1_alex.tcl
''' + write_hook_result('qualify')
    (out / 'run.tcl').write_text(script)
    if stage in ('link', 'qualify'):
        for name in ('qualify_c1_alex.tcl', 'audit_c1_alex.tcl'):
            shutil.copy2(HERE / name, out / name)
    if stage == 'qualify':
        shutil.copy2(HERE / 'routed_host_ready.tcl', out / 'routed_host_ready.tcl')
    inputs = {'schema': 1, 'kind': 'c1-alex-checkpoint-stage-inputs', 'stage': stage,
              'executor_revision': executor_revision, 'executor_sources': tools,
              'source_revision': origin['inputs']['source_revision'],
              'build_id': origin['inputs']['build_id'], 'synth_run': str(origin['run']),
              'synth_receipt_sha256': origin['receipt_sha256'],
              'synth_inputs_sha256': origin['receipt']['inputs_sha256'],
              'ip_run': str(origin['ip_run']), 'ip_receipt_sha256': origin['receipt']['ip_receipt_sha256'],
              'hdk': str(hdk), 'shell_sha256': b.SHELL_SHA256,
              'source_checkpoint': source, 'previous_run': str(previous['run']) if previous else None,
              'previous_receipt_sha256': previous['receipt_sha256'] if previous else None,
              'max_threads': max_threads, 'resynthesis': False, 'hardware_actions': False}
    if stage == 'implement':
        inputs['physical_strategy'] = physical_strategy
    if simulation:
        inputs['simulation'] = {**simulation, 'run': str(simulation['run'])}
    inputs['prepared_files'] = b.inventory([out], out)
    b.write_json(out / 'stage-inputs.json', inputs)
    verify_files(ROOT, tools)
    verify_files(out, inputs['prepared_files'])
    return inputs


def output_inventory(out):
    # Receipt excludes itself and the input manifest to avoid self-references.
    paths = [p for p in out.iterdir() if p.name not in ('stage-inputs.json', 'stage-result.json')]
    return b.inventory(paths, out) if paths else {}


def execute(out, stage, origin, previous, inputs, hdk, vivado):
    checkpoint = Path(inputs['source_checkpoint']['path'])
    require(json.loads((out / 'stage-inputs.json').read_text()) == inputs, 'Prepared input manifest changed')
    inputs_sha256 = b.sha(out / 'stage-inputs.json')
    def recheck():
        require(b.sha(out / 'stage-inputs.json') == inputs_sha256, 'Prepared input manifest changed')
        verify_files(out, inputs['prepared_files'])
        verify_files(ROOT, inputs['executor_sources'])
        require(not any((out / name).exists() for name in FORBIDDEN_CONTROLS),
                'Unreviewed tuning or incremental checkpoint control appeared')
        verify_synthesis(origin['run'], origin['receipt_sha256'], hdk)
        if previous:
            verify_previous(previous['run'], previous['receipt_sha256'], PREVIOUS[stage], origin)
        if stage == 'qualify':
            proof = inputs['simulation']
            checked = verify_simulation(b.safe_path(proof['run']), proof['receipt_sha256'], origin)
            require(checked['inputs_sha256'] == proof['inputs_sha256'], 'Simulation input binding changed')
        require(b.sha(checkpoint) == inputs['source_checkpoint']['sha256'] and
                checkpoint.stat().st_size == inputs['source_checkpoint']['bytes'], 'Source checkpoint changed')
    recheck()
    cmd = [vivado, '-mode', 'batch', '-source', str(out / 'run.tcl'),
           '-tclargs', str(out), str(checkpoint)]
    if stage in ('link', 'implement'):
        cmd.append(str(hdk / 'hdk'))
    b.write_json(out / 'command.json', cmd)
    result = {'schema': 1, 'kind': 'c1-alex-checkpoint-stage', 'stage': stage,
              'inputs_sha256': inputs_sha256,
              'source_checkpoint_sha256': inputs['source_checkpoint']['sha256'],
              'status': 'failed', 'route_qualified': False, 'hardware_actions': False}
    try:
        with (out / 'console.log').open('x') as log:
            process = subprocess.run(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT)
        result['returncode'] = process.returncode
        recheck()
        require(process.returncode == 0, f'{stage} Vivado process failed ({process.returncode})')
        if stage == 'link':
            require((out / 'link_done.txt').read_text().strip() == 'C1_ALEX_LINK_COMPLETE_NOT_ROUTE_QUALIFIED',
                    'Linked-stage completion marker missing')
            require(machine_result(out / 'audit-result.txt', ('structural_pass', 'route_qualified')) ==
                    {'structural_pass': '1', 'route_qualified': '0'}, 'Linked structural gate did not pass')
            result['status'] = 'complete'
        elif stage in ('implement', 'refine'):
            result['timing'] = timing_result(out / 'implementation_result.txt')
            if stage == 'refine':
                require(min(result['timing'].values()) >= 0, 'Refinement did not close setup/hold timing')
                result['status'] = 'complete'
            else:
                result['status'] = 'candidate'
        else:
            audit = machine_result(out / 'audit-result.txt', ('route_qualified', 'checkpoint_sha256'))
            require(audit['checkpoint_sha256'] == inputs['source_checkpoint']['sha256'],
                    'Qualifier audited a different DCP')
            require(audit['route_qualified'] in ('0', '1'), 'Invalid qualification result')
            result['route_qualified'] = audit['route_qualified'] == '1'
            result['status'] = 'complete' if result['route_qualified'] else 'pending-review'
        if stage in CHECKPOINTS:
            name = CHECKPOINTS[stage]
            require((out / name).is_file() and not (out / name).is_symlink(), 'Stage checkpoint missing')
            result['checkpoint'] = name
            result['checkpoint_sha256'] = b.sha(out / name)
        else:
            result['checkpoint'] = str(checkpoint)
            result['checkpoint_sha256'] = inputs['source_checkpoint']['sha256']
    except Exception as error:
        result['status'] = 'failed'
        result['route_qualified'] = False
        result['error'] = str(error)
        result['outputs'] = output_inventory(out)
        b.write_json(out / 'stage-result.json', result)
        raise
    result['outputs'] = output_inventory(out)
    b.write_json(out / 'stage-result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=STAGES)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--hdk', required=True, type=Path, help='Pinned aws-fpga repository root')
    parser.add_argument('--synth-run', required=True, type=Path)
    parser.add_argument('--synth-receipt-sha256', required=True)
    parser.add_argument('--from-run', type=Path)
    parser.add_argument('--from-receipt-sha256')
    parser.add_argument('--simulation-run', type=Path)
    parser.add_argument('--simulation-receipt-sha256')
    parser.add_argument('--vivado', default='vivado')
    parser.add_argument('--max-threads', type=int, choices=(4, 8), default=4,
                        help='Bounded implement/refine concurrency; historical default is 4')
    parser.add_argument('--physical-strategy', choices=PHYSICAL_STRATEGIES, default='default',
                        help='Explicit implementation-only strategy; timing/constraints remain frozen')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args(argv)
    if args.max_threads != 4 and args.stage not in ('implement', 'refine'):
        parser.error('--max-threads 8 applies only to implement/refine')
    if args.stage != 'implement' and args.physical_strategy != 'default':
        parser.error('--physical-strategy applies only to implement')
    if args.stage == 'link':
        if args.from_run or args.from_receipt_sha256:
            parser.error('link consumes synthesis directly; do not specify --from-run')
    elif not args.from_run or not args.from_receipt_sha256:
        parser.error('later stages require --from-run and --from-receipt-sha256')
    if args.stage == 'qualify':
        if not args.simulation_run or not args.simulation_receipt_sha256:
            parser.error('qualify requires --simulation-run and --simulation-receipt-sha256')
        digest(args.simulation_receipt_sha256)
    elif args.simulation_run or args.simulation_receipt_sha256:
        parser.error('simulation evidence is required only for qualify')
    digest(args.synth_receipt_sha256)
    if args.from_receipt_sha256:
        digest(args.from_receipt_sha256)
    out, hdk, synth = map(b.safe_path, (args.out, args.hdk, args.synth_run))
    if args.plan:
        print(json.dumps({'stage': args.stage, 'out': str(out), 'synth_run': str(synth),
              'source': str(args.from_run) if args.from_run else str(synth),
              'physical_strategy': args.physical_strategy,
              'max_threads': args.max_threads, 'resynthesis': False, 'hardware_actions': False,
              'route_qualified': False}, indent=2))
        return 0
    b.require_clean_source()
    origin = verify_synthesis(synth, args.synth_receipt_sha256, hdk)
    previous = verify_previous(b.safe_path(args.from_run), args.from_receipt_sha256,
                               PREVIOUS[args.stage], origin) if args.from_run else None
    simulation = verify_simulation(b.safe_path(args.simulation_run), args.simulation_receipt_sha256,
                                   origin) if args.stage == 'qualify' else None
    version = b.command(args.vivado, '-version')
    b.check_vivado(version)
    inputs = prepare(out, args.stage, origin, previous, hdk, simulation, args.physical_strategy, args.max_threads)
    (out / 'vivado_version.txt').write_text(version + '\n')
    result = execute(out, args.stage, origin, previous, inputs, hdk, args.vivado)
    print(json.dumps({'stage': args.stage, 'status': result['status'],
                      'route_qualified': result['route_qualified'],
                      'receipt': str(out / 'stage-result.json'),
                      'receipt_sha256': b.sha(out / 'stage-result.json')}, indent=2))
    return 3 if result['status'] == 'pending-review' else 0


if __name__ == '__main__':
    raise SystemExit(main())
