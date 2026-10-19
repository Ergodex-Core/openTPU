"""Synthesis audit rejection tests with mocked Vivado objects; no FPGA evidence."""
from pathlib import Path
import shutil
import subprocess

import pytest


AUDIT = Path(__file__).resolve().parents[1] / 'boards/c2_f2_alex/audit_c2_alex.tcl'
MOCK = r'''
lassign $argv script run fault
source $script
set cells {}; set clocks {}; set pins {}; set pin_clocks {}; set members {}
proc add_cell {name ref {opaque 0}} {
    dict set ::cells $name [dict create NAME $name REF_NAME ${ref}_synth ORIG_REF_NAME $ref IS_BLACKBOX $opaque]
}
foreach {name ref} {
 u_board otpu_c2_hbm_kernel u_board/u_mxu otpu_c2_mxu
 u_mem f2_mem_wrapper u_hbm cl_mem_hbm_wrapper u_ocl_cdc f2_ocl_cdc
 u_pcis_pipe f2_pcis_pipe u_core_clk f2_core_clk u_hbm_clk f2_hbm_clk
} {add_cell $name $ref}
for {set p 0} {$p < 8} {incr p} {add_cell u_native$p otpu_native_axi}
foreach {name period} {clk_main_a0 4.000 clk_hbm_ref 10.000 core 5.000 unused400 2.500 hbm 2.222} {
    dict set clocks $name [dict create NAME $name PERIOD $period IS_VIRTUAL 0 IS_PROPAGATED 1]
}
dict set clocks core SOURCE_PINS u_core_clk/inst/mmcm/CLKOUT0
dict set clocks unused400 SOURCE_PINS u_core_clk/inst/mmcm/CLKOUT1
dict set clocks hbm SOURCE_PINS u_hbm_clk/inst/mmcm/CLKOUT0
foreach {pin clock} {u_core_clk/clk_out1 core u_core_clk/clk_out2 unused400 u_hbm_clk/clk_out1 hbm u_mem/core_clk core u_mem/hbm_clk hbm u_mem/host_clk clk_main_a0} {
    dict set pins $pin [dict create NAME $pin]
    dict set pin_clocks $pin $clock
}
set ifaces {S_HOST}
for {set p 0} {$p < 8} {incr p} {lappend ifaces S_N$p M_H$p}
set mem_pins {}
foreach iface $ifaces {
    foreach signal {awvalid awready wvalid wready bvalid bready arvalid arready rvalid rready} {
        set name u_mem/${iface}_$signal
        lappend mem_pins $name; dict set pins $name [dict create NAME $name]
    }
    set aw [expr {$iface eq "S_HOST" ? 64 : 34}]
    set dw [expr {[string match M_H* $iface] ? 256 : 512}]
    foreach {signal width} [list awaddr $aw araddr $aw wdata $dw rdata $dw wstrb [expr {$dw/8}]] {
        for {set bit 0} {$bit < $width} {incr bit} {
            set name [format {u_mem/%s_%s[%d]} $iface $signal $bit]
            lappend mem_pins $name; dict set pins $name [dict create NAME $name]
        }
    }
}
dict set members hbm {u_mem/f2_mem_i/memory_sc/cdc_reg u_hbm/vendor_reg u_rst_hbm/sync_reg}
dict set members core {u_board/data_reg}
dict set members unused400 {}
switch $fault {
    ref_without_orig {
        foreach name [dict keys $cells] {
            dict set cells $name REF_NAME [dict get $cells $name ORIG_REF_NAME]
            dict unset cells $name ORIG_REF_NAME
        }
    }
    ref_empty_orig {
        foreach name [dict keys $cells] {
            dict set cells $name REF_NAME [dict get $cells $name ORIG_REF_NAME]
            dict set cells $name ORIG_REF_NAME {}
        }
    }
    pruned_unused_pin {dict unset pins u_core_clk/clk_out2;dict unset clocks unused400}
    retained_unused_clock {dict unset pins u_core_clk/clk_out2}
    pruned_pin_used_400 {dict unset pins u_core_clk/clk_out2;dict lappend members unused400 u_board/illegal_reg}
    pruned_pin_unknown_source {dict unset pins u_core_clk/clk_out2;dict set clocks unused400 SOURCE_PINS unrelated/clock}
    pruned_pin_no_source {dict unset pins u_core_clk/clk_out2;dict unset clocks unused400 SOURCE_PINS}
    wrong_kernel {dict set cells u_board ORIG_REF_NAME otpu_board}
    missing_ref_property {dict unset cells u_board REF_NAME}
    missing_blackbox_property {dict unset cells u_board IS_BLACKBOX}
    opaque_kernel {dict set cells u_board IS_BLACKBOX 1}
    opaque_mxu {dict set cells u_board/u_mxu IS_BLACKBOX 1}
    missing_native {dict unset cells u_native7}
    extra_native {add_cell u_native8 otpu_native_axi}
    wrong_native {dict set cells u_native0 ORIG_REF_NAME otpu_f2_hbm_nch}
    wrong_vendor {dict set cells u_mem ORIG_REF_NAME wrong_memory}
    vendor_opaque {add_cell u_mem/f2_mem_i/memory_sc vendor_smartconnect 1}
    unknown_opaque {add_cell custom_bad unknown 1}
    lookalike_opaque {add_cell u_mem_extra/custom bad 1}
    opaque_clock {
        add_cell u_core_clk/inst clock_model 1
        dict unset clocks core; dict unset clocks unused400
    }
    missing_clock {dict unset clocks core}
    wrong_clock {dict set clocks core PERIOD 10.000}
    wrong_hbm_clock {dict set clocks hbm PERIOD 2.500}
    wrong_host_clock {dict set clocks clk_main_a0 PERIOD 5.000}
    wrong_reference_clock {dict set clocks clk_hbm_ref PERIOD 8.000}
    disconnected_core {dict set pin_clocks u_mem/core_clk clk_main_a0}
    disconnected_hbm {dict set pin_clocks u_mem/hbm_clk core}
    disconnected_host {dict set pin_clocks u_mem/host_clk core}
    missing_clock_pin {dict unset pins u_core_clk/clk_out1}
    custom_hclk {dict lappend members hbm u_board/custom_reg}
    lookalike_hclk {dict lappend members hbm u_hbm_extra/custom_reg}
    used_400 {dict lappend members unused400 u_board/tmem_reg}
    missing_interface {
        set remaining {}; foreach n $mem_pins {if {![string match u_mem/S_N7_* $n]} {lappend remaining $n}}
        set mem_pins $remaining
    }
    extra_interface {lappend mem_pins u_mem/S_N8_arvalid;dict set pins u_mem/S_N8_arvalid {NAME u_mem/S_N8_arvalid}}
    missing_handshake {set at [lsearch -exact $mem_pins u_mem/M_H7_rready];set mem_pins [lreplace $mem_pins $at $at]}
    indexed_handshakes {
        set updated {}
        foreach name $mem_pins {
            if {[regexp {_(awvalid|awready|wvalid|wready|bvalid|bready|arvalid|arready|rvalid|rready)$} $name]} {
                set name [format {%s[0]} $name]
                dict set pins $name [dict create NAME $name]
            }
            lappend updated $name
        }
        set mem_pins $updated
    }
    duplicate_handshake {lappend mem_pins u_mem/S_HOST_awvalid}
    mixed_handshake {lappend mem_pins {u_mem/S_HOST_awvalid[0]}}
    duplicate_handshake_bit - handshake_extra_bit - handshake_nonzero_bit {
        set at [lsearch -exact $mem_pins u_mem/S_HOST_awvalid]
        set mem_pins [lreplace $mem_pins $at $at]
        if {$fault eq "duplicate_handshake_bit"} {
            lappend mem_pins {u_mem/S_HOST_awvalid[0]} {u_mem/S_HOST_awvalid[0]}
        } elseif {$fault eq "handshake_extra_bit"} {
            lappend mem_pins {u_mem/S_HOST_awvalid[0]} {u_mem/S_HOST_awvalid[1]}
        } else {lappend mem_pins {u_mem/S_HOST_awvalid[1]}}
    }
    wrong_width {set at [lsearch -exact $mem_pins {u_mem/S_N7_awaddr[33]}];set mem_pins [lreplace $mem_pins $at $at]}
    duplicate_bit {lappend mem_pins {u_mem/M_H0_wdata[0]}}
    vector_scalar_duplicate {lappend mem_pins u_mem/S_HOST_awaddr}
}
proc version {args} {return [expr {$::fault eq "version" ? "2025.1" : "2025.2"}]}
proc current_instance {} {return [expr {$::fault eq "scope" ? "WRAPPER/CL" : ""}]}
proc list_property {object} {
    if {$::fault eq "cell_property_inventory_once" && [dict exists $::cells $object]} {
        if {![info exists ::property_queries]} {set ::property_queries {}}
        if {[dict exists $::property_queries $object]} {error "repeated cell property inventory $object"}
        dict set ::property_queries $object 1
    }
    foreach table {cells clocks pins} {if {[dict exists [set ::$table] $object]} {return [dict keys [dict get [set ::$table] $object]]}}
    if {[string match */* $object]} {return NAME}
    error "unknown fixture object $object"
}
proc get_property {key objects} {
    set values {}
    foreach object $objects {
        if {$key eq "NAME"} {lappend values $object;continue}
        set found 0
        foreach table {cells clocks pins} {
            if {[dict exists [set ::$table] $object $key]} {
                lappend values [dict get [set ::$table] $object $key];set found 1;break
            }
        }
        if {!$found} {error "unknown fixture property $key $object"}
    }
    if {[llength $objects] == 1} {return [lindex $values 0]}
    return $values
}
proc get_cells {args} {
    set selected [lindex $args end]
    if {"-filter" in $args} {
        if {$selected eq "IS_BLACKBOX"} {
            set result {};dict for {name props} $::cells {if {[dict get $props IS_BLACKBOX]} {lappend result $name}}
            return $result
        }
        return [expr {$::fault eq "tmem_cell" ? "u_board/u_tmem" : ""}]
    }
    if {"-regexp" in $args} {
        set result {};foreach name [dict keys $::cells] {if {[regexp $selected $name]} {lappend result $name}}
        return $result
    }
    return [expr {[dict exists $::cells $selected] ? $selected : ""}]
}
proc get_pins {args} {
    if {"-of_objects" in $args} {return $::mem_pins}
    set name [lindex $args end]
    return [expr {[dict exists $::pins $name] ? $name : ""}]
}
proc get_clocks {args} {
    set name [lindex $args end]
    if {$name eq "*"} {return [dict keys $::clocks]}
    if {"-of_objects" in $args} {
        if {![dict exists $::pin_clocks $name]} {return {}}
        set name [dict get $::pin_clocks $name]
    }
    return [expr {[dict exists $::clocks $name] ? $name : ""}]
}
proc all_registers {args} {return [dict get $::members [lindex $args 1]]}
proc write_checkpoint {path} {
    if {$::fault eq "checkpoint_error"} {error "mock checkpoint write failure"}
    set f [open $path {WRONLY CREAT EXCL}]
    puts $f {MOCK_DIAGNOSTIC_NOT_QUALIFIED_NOT_FPGA_CONTENT};close $f
}
proc write_xdc {args} {
    set f [open [lindex $args end] w]
    puts $f {# mock export; never physical constraint evidence}
    switch $::fault {
        tmem_exception {puts $f {set_multicycle_path 2 -to [get_pins u_tmem/ADDRARDADDR]}}
        custom_multicycle {puts $f {set_multicycle_path 2 -from [get_clocks core] -to [get_clocks hbm]}}
        vendor_multicycle {puts $f {current_instance u_mem/f2_mem_i/memory_sc};puts $f {set_multicycle_path 2 -to [get_pins vendor_endpoint/D]}}
        incomplete_export {puts $f {set_false_path -to [get_pins}}
    }
    close $f
}
proc emit_report {args} {
    set path [lindex $args end]
    if {$::fault eq "missing_report" && [file tail $path] eq "cdc.rpt"} {return}
    if {$::fault eq "report_error" && [file tail $path] eq "cdc.rpt"} {error "mock CDC report failure"}
    set f [open $path w];puts $f {MOCK_REPORT_NOT_HARDWARE_EVIDENCE};close $f
}
foreach command {report_timing_summary report_drc report_cdc report_bus_skew report_clock_interaction report_exceptions report_methodology report_waivers report_utilization} {
    proc $command {args} {emit_report {*}$args}
}
proc get_drc_violations {args} {return {}}
proc get_methodology_violations {args} {return {}}
if {$fault eq "existing_directory"} {file mkdir [file join $run synth_audit]}
if {$fault eq "existing_checkpoint"} {
    set f [open [file join $run pre_audit.NOT_QUALIFIED.dcp] w]
    puts $f {PREEXISTING};close $f
}
if {[catch {::c2_alex::audit_synthesis $run} result]} {puts stderr $result;exit 1}
puts [list SUMMARY $result]
'''


def run_audit(tmp_path, fault):
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl unavailable')
    driver = tmp_path / 'mock.tcl'
    driver.write_text(MOCK)
    return subprocess.run([tclsh, str(driver), str(AUDIT), str(tmp_path), fault],
                          text=True, capture_output=True, timeout=20)


@pytest.mark.parametrize('fault', ['none', 'vendor_opaque', 'opaque_clock', 'vendor_multicycle',
                                 'ref_without_orig', 'ref_empty_orig', 'pruned_unused_pin',
                                 'retained_unused_clock', 'indexed_handshakes', 'cell_property_inventory_once'])
def test_synthesis_records_limits_and_required_current_vendor_simulation(tmp_path, fault):
    result = run_audit(tmp_path, fault)
    assert result.returncode == 0, result.stderr
    assert 'C2_ALEX_SYNTH_AUDIT_RECORDED route_qualified=0' in result.stdout
    directory = tmp_path / 'synth_audit'
    summary = (directory / 'summary.tcldata').read_text()
    assert 'route_qualified 0 hardware_measured 0' in summary
    assert 'slave_interfaces 9 master_interfaces 8' in summary
    assert 'actual_9SI_8MI_vendor_simulation' in summary
    assert 'full_shell_link_and_complete_clock_CDC_audit' in summary
    assert 'generated_BD_required' in summary
    assert len(list(directory.iterdir())) == 14
    assert 'memory_interface_pins ' in (directory / 'objects.tcldata').read_text()
    assert (tmp_path / 'pre_audit.NOT_QUALIFIED.dcp').is_file()
    assert not list(tmp_path.rglob('*post_synth*.dcp'))
    assert 'qualification NOT_QUALIFIED' in (directory / 'objects.tcldata').read_text()
    assert not list(tmp_path.rglob('*.passed'))
    checklist = (directory / 'REQUIRED_NEXT_STEPS.txt').read_text()
    for required in ('CURRENT nine-SI/eight-MI', 'narrow byte strobes', 'independent IDs',
                     'two-port simulation is insufficient', 'full byte readback'):
        assert required in checklist
    if fault == 'opaque_clock':
        assert 'deferred_clocks {core tmem_unused}' in summary
        assert 'state deferred_opaque_ip' in (directory / 'objects.tcldata').read_text()
    if fault == 'vendor_multicycle':
        assert 'vendor_multicycles_pending_review 1' in summary
    if fault == 'pruned_unused_pin':
        assert 'pruned_unused_outputs tmem_unused' in summary
        assert 'deferred_clocks {}' in summary
        assert 'state pruned_unused_output' in (directory / 'objects.tcldata').read_text()


@pytest.mark.parametrize('fault', [
    'wrong_kernel', 'opaque_kernel', 'opaque_mxu', 'missing_native', 'extra_native',
    'wrong_native', 'wrong_vendor', 'unknown_opaque', 'lookalike_opaque',
    'missing_clock', 'wrong_clock', 'wrong_hbm_clock', 'wrong_host_clock',
    'wrong_reference_clock', 'disconnected_core', 'disconnected_hbm',
    'disconnected_host', 'missing_clock_pin', 'custom_hclk', 'lookalike_hclk',
    'used_400', 'missing_interface', 'extra_interface', 'missing_handshake',
    'wrong_width', 'duplicate_bit', 'version', 'scope', 'tmem_cell',
    'tmem_exception', 'custom_multicycle', 'incomplete_export', 'missing_report',
    'report_error', 'existing_directory',
    'pruned_pin_used_400', 'pruned_pin_unknown_source', 'pruned_pin_no_source',
    'missing_ref_property', 'missing_blackbox_property', 'duplicate_handshake',
    'mixed_handshake', 'duplicate_handshake_bit', 'handshake_extra_bit',
    'handshake_nonzero_bit', 'vector_scalar_duplicate',
    'existing_checkpoint', 'checkpoint_error',
])
def test_synthesis_audit_rejects_wrong_or_incomplete_evidence(tmp_path, fault):
    result = run_audit(tmp_path, fault)
    assert result.returncode != 0, fault
    assert result.stderr.strip(), fault
    assert 'C2_ALEX_SYNTH_AUDIT_RECORDED' not in result.stdout
    assert not (tmp_path / 'synth_audit/summary.tcldata').exists()
    assert not list(tmp_path.rglob('*.passed'))
    assert not list(tmp_path.rglob('*post_synth*.dcp'))
    diagnostic = tmp_path / 'pre_audit.NOT_QUALIFIED.dcp'
    if fault == 'existing_checkpoint':
        assert diagnostic.read_text().strip() == 'PREEXISTING'
    elif fault not in ('version', 'scope', 'existing_directory', 'checkpoint_error'):
        assert diagnostic.read_text().strip() == 'MOCK_DIAGNOSTIC_NOT_QUALIFIED_NOT_FPGA_CONTENT'
