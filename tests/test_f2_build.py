"""The F2 CL snapshot must implement exactly the selected target and retain provenance."""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

import pytest

from opentpu import target as T

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('f2_build', ROOT / 'boards/f2/build.py')
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)


def test_vivado_version_accepts_linux_banner_and_rejects_other_releases():
    for banner in ('vivado v2025.2 (64-bit)', 'Vivado v2025.2 (64-bit)\nSW Build 123'):
        flow.check_vivado_version(banner, '2025.2')
    for banner in ('vivado v2025.1 (64-bit)', 'Vivado v2025.20 (64-bit)', 'build 2025.2'):
        with pytest.raises(RuntimeError, match='requires Vivado 2025.2'):
            flow.check_vivado_version(banner, '2025.2')


def test_ddr_calibration_sentinel_checks_literal_bank_and_firmware():
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl is unavailable')
    sentinel = ('SH_DDR/controller/mcs0/inst/lmb_bram_I/U0/inst_blk_mem_gen/'
                'valid.cstr/ramloop[0].ram.r/primitive.ram')
    other_bank = sentinel.replace('ramloop[0]', 'ramloop[1]')
    script = ROOT / 'boards/f2/build/scripts/check_ddr_init.tcl'
    cases = [
        ([sentinel, other_bank], "256'h" + '0' * 63 + '1', None),
        ([sentinel], "256'h" + '0' * 64, 'firmware is absent'),
        ([sentinel], "256'h" + 'x' * 64, 'firmware is absent'),
        ([other_bank], "256'h" + '1' * 64, 'found 0'),
        ([sentinel, 'duplicate/' + sentinel], "256'h" + '1' * 64, 'found 2'),
    ]
    for cells, init, error in cases:
        harness = '\n'.join([
            'proc refresh_meminit {} {set ::refreshed 1}',
            'proc get_cells {args} {return {' + ' '.join('{' + c + '}' for c in cells) + '}}',
            'proc get_property {prop cell} {',
            '  if {$prop eq "NAME"} {return $cell}',
            '  if {![info exists ::refreshed]} {error "memory was not refreshed"}',
            '  return {' + init + '}',
            '}',
            'if {[catch {source {' + str(script) + '}} message]} {puts stderr $message; exit 1}',
        ])
        result = subprocess.run([tclsh], input=harness, text=True, capture_output=True)
        if error is None:
            assert result.returncode == 0, result.stderr
            assert 'OTPU_DDR_CALIBRATION_INIT_2C=' + init in result.stdout
        else:
            assert result.returncode != 0
            assert error in result.stderr


def test_f2_matches_card_program_architecture():
    t = T.load('f2')
    assert T.config(t) == T.config('ypcb-00338:m4')
    assert t.platform_id == 3
    assert t.tech['part'] == 'xcvu47p-fsvh2892-2-e'
    assert t.core_khz == 100000
    assert t.platform['mem_mts'] == 2133
    assert t.platform['channel_bytes'] == 2**31
    assert t.platform['channels_per_slice'] == 2
    assert t.arch['DRAM_BYTES'] == 2**32
    assert t.uarch['TMEM_PMP'] == 1
    assert T.rtl_params(t)['ARCH_HASH'] == 0x4FFD1E17
    assert all((ROOT / 'rtl' / f).is_file() for f in T.sources(t))


def test_snapshot_preserves_target_and_uncommitted_sources(tmp_path):
    # A local fake HDK provides only the two launch scripts; no vendor tools or
    # firmware are needed to verify the source/parameter handoff to synthesis.
    shell = tmp_path / 'shell'
    scripts = shell / 'build/scripts'
    scripts.mkdir(parents=True)
    for name in ('build_all.tcl', 'build_level_1_cl.tcl'):
        (scripts / name).write_text('# test HDK script\n')
    out = tmp_path / 'run'
    t = T.load('f2')
    cl = flow.prepare(out, shell, t, '01234567')
    manifest = json.loads((out / 'inputs.json').read_text())
    assert manifest['build_id'] == '01234567'
    assert manifest['rtl_parameters']['BUILD_ID'] == 0x01234567
    assert manifest['rtl_parameters']['ARCH_HASH'] == 0x4FFD1E17
    assert manifest['rtl_parameters']['CH_BYTES'] == 2**31
    assert manifest['rtl_parameters']['RPB'] == 64
    assert manifest['rtl_parameters']['DDR_MTS'] == 2133
    assert (cl / 'design/opentpu_params.vh').read_text().count('.TMEM_PMP(') == 1
    assert '.TMEM_TDP(1\'h1)' in (cl / 'design/opentpu_params.vh').read_text()
    assert manifest['sources']['design/cl_opentpu.sv'] == flow.sha(ROOT / 'boards/f2/design/cl_opentpu.sv')
    for name in T.sources(t):
        assert (cl / 'design/rtl' / name).read_bytes() == (ROOT / 'rtl' / name).read_bytes()
    assert (cl / 'build/scripts/build_all.tcl').resolve() == scripts / 'build_all.tcl'
    with pytest.raises(RuntimeError, match='already exists'):
        flow.prepare(out, shell, t, '00000000')


def test_wrong_hdk_revision_is_rejected_before_preparing(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, 'command', lambda *a, **k: '0' * 40)
    with pytest.raises(RuntimeError, match='revision .* expected'):
        flow.check_hdk(tmp_path, T.load('f2'), shell=False)


def completed_synthesis(tmp_path):
    shell = tmp_path / 'shell'
    scripts = shell / 'build/scripts'
    scripts.mkdir(parents=True)
    (scripts / 'build_all.tcl').write_text('# pinned HDK driver\n')
    run = tmp_path / 'synth'
    cl = flow.prepare(run, shell, T.load('f2'), '01234567')
    (run / 'exit_code').write_text('0\n')
    (run / 'console.log').write_text("OTPU_DDR_CALIBRATION_INIT_2C=256'h" +
                                   '1' * 64 + '\nOTPU_STAGE_SYNTHESIS_PASSED\n')
    dcp = cl / 'build/checkpoints/cl_opentpu.01234567.post_synth.dcp'
    dcp.parent.mkdir()
    dcp.write_bytes(b'test checkpoint, not a real FPGA design')
    return shell, run, dcp


def test_implementation_inherits_verified_checkpoint_and_provenance(tmp_path):
    shell, run, dcp = completed_synthesis(tmp_path)
    parent = flow.load_synthesis(run, flow.sha(dcp))
    out = tmp_path / 'impl'
    cl = flow.prepare(out, shell, T.load('f2'), '01234567')
    flow.inherit_synthesis(out, cl, run, *parent)
    info = json.loads((out / 'inputs.json').read_text())
    assert info['parent_synthesis']['checkpoint_sha256'] == flow.sha(dcp)
    assert info['parent_synthesis']['inputs_sha256'] == flow.sha(run / 'inputs.json')
    assert (cl / 'build/checkpoints' / dcp.name).read_bytes() == dcp.read_bytes()
    assert not (cl / 'build/scripts/build_level_1_cl.tcl').is_symlink()


@pytest.mark.parametrize('corruption', ['digest', 'exit', 'firmware', 'source', 'parameters'])
def test_implementation_rejects_invalid_or_mismatched_parent(tmp_path, corruption):
    shell, run, dcp = completed_synthesis(tmp_path)
    if corruption == 'exit':
        (run / 'exit_code').write_text('1\n')
    elif corruption == 'firmware':
        (run / 'console.log').write_text("OTPU_DDR_CALIBRATION_INIT_2C=256'h" +
                                       '0' * 64 + '\nOTPU_STAGE_SYNTHESIS_PASSED\n')
    digest = '0' * 64 if corruption == 'digest' else flow.sha(dcp)
    if corruption in ('digest', 'exit', 'firmware'):
        with pytest.raises(RuntimeError):
            flow.load_synthesis(run, digest)
        return
    parent = flow.load_synthesis(run, digest)
    out = tmp_path / 'impl'
    cl = flow.prepare(out, shell, T.load('f2'), '01234567')
    if corruption == 'source':
        (run / 'cl_opentpu/design/cl_opentpu.sv').write_text('// modified after synthesis\n')
    else:
        parent[0]['rtl_parameters']['CORE_KHZ'] += 1
    with pytest.raises(RuntimeError):
        flow.inherit_synthesis(out, cl, run, *parent)


def completed_routed(tmp_path):
    shell, run, synth_dcp = completed_synthesis(tmp_path)
    cl = run / 'cl_opentpu'
    dcp = synth_dcp.with_name('cl_opentpu.01234567.post_route.dcp')
    dcp.write_bytes(b'qualified routed test bytes, not a real DCP')
    (dcp.parent / 'cl_opentpu.01234567.route_validation.passed').write_text(
        'OTPU_STAGE_ROUTE_VALIDATION_PASSED\n')
    (dcp.parent / '01234567.debug_probes.ltx').write_text('test probes\n')
    with (run / 'console.log').open('a') as f:
        f.write('OTPU_STAGE_ROUTE_VALIDATION_PASSED\n')
    (run / 'outputs.json').write_text('{"historical": true}\n')
    (tmp_path / 'release_version.txt').write_text('RELEASE_VERSION=2.3.3\n')
    # This fixture stands in for the external HDK generator. It reads the staged
    # timestamp aliases and emits real archives, allowing independent metadata
    # and payload corruption to exercise our archive verifier end to end.
    (shell / 'build/scripts/aws_build_dcp_from_cl.py').write_text(r'''
import hashlib, io, os, pathlib, tarfile
TIMESTAMP_FILE_FORMAT = "%Y_%m_%d-%H%M%S"
def generate_dcp_tarball(cl, tag, a, b, c, h):
    directory = pathlib.Path(os.environ['CL_DIR']) / 'build/checkpoints'
    data = (directory / f'{cl}.{tag}.post_route.dcp').read_bytes()
    assert (directory / f'{tag}.debug_probes.ltx').read_text() == 'test probes\n'
    assert 'F001_1D0F' in (directory.parents[1] / 'design/cl_id_defines.vh').read_text()
    manifest = dict(manifest_format_version='2', date=tag,
        pci_vendor_id='0x1D0F', pci_device_id='0xF001',
        pci_subsystem_vendor_id='0xFEDC', pci_subsystem_id='0x1D51',
        shell_version='0x10212415', hdk_version='2.3.3', tool_version='v2025.2',
        dcp_hash=hashlib.sha256(data).hexdigest(), dcp_file_name=f'{tag}.SH_CL_routed.dcp',
        clock_recipe_a=a, clock_recipe_b=b, clock_recipe_c=c, clock_recipe_hbm=h)
    fault = os.environ.get('F2_TEST_BAD_FIELD')
    if fault == 'payload': data += b'corruption'
    elif fault: manifest[fault] = 'invalid'
    with tarfile.open(directory / f'{tag}.Developer_CL.tar', 'w') as tar:
        for name, contents in [(f'{tag}.manifest.txt', '\n'.join(f'{k}={v}' for k,v in manifest.items()).encode()),
                               (f'{tag}.SH_CL_routed.dcp', data)]:
            member = tarfile.TarInfo('to_aws/' + name)
            member.size = len(contents)
            tar.addfile(member, io.BytesIO(contents))
''')
    return shell, run, cl, dcp


def test_timestamp_package_preserves_build_identity_and_records_verified_archive(tmp_path):
    shell, run, cl, dcp = completed_routed(tmp_path)
    info, selected_cl = flow.load_routed_run(run, flow.sha(dcp))
    assert selected_cl == cl
    archive = flow.package_afi(cl, shell, tmp_path, info['build_id'],
                              dict(os.environ, VIVADO_TOOL_VERSION='2025.2'))
    tag = archive.name.removesuffix('.Developer_CL.tar')
    assert re.fullmatch(r'\d{4}_\d{2}_\d{2}-\d{6}', tag)
    assert tag != info['build_id']
    provenance = json.loads((archive.parent / f'{tag}.packaging.json').read_text())
    assert provenance['build_id'] == '01234567'
    assert provenance['package_tag'] == tag
    assert provenance['routed_checkpoint_sha256'] == flow.sha(dcp)
    assert provenance['archive_sha256'] == flow.sha(archive)
    assert provenance['archive_bytes'] == archive.stat().st_size
    assert provenance['inputs_sha256'] == flow.sha(run / 'inputs.json')
    with tarfile.open(archive) as tar:
        assert tar.extractfile(f'to_aws/{tag}.SH_CL_routed.dcp').read() == dcp.read_bytes()
    assert (run / 'outputs.json').read_text() == '{"historical": true}\n'
    assert not list(archive.parent.glob('.afi-package-*'))


@pytest.mark.parametrize('fault', ['date', 'pci_device_id', 'shell_version', 'hdk_version',
                                 'tool_version', 'dcp_hash', 'payload'])
def test_packaging_rejects_bad_manifest_or_changed_payload(tmp_path, fault):
    shell, run, cl, dcp = completed_routed(tmp_path)
    with pytest.raises(RuntimeError, match='AFI (manifest|archive)'):
        flow.package_afi(cl, shell, tmp_path, '01234567',
                         dict(os.environ, VIVADO_TOOL_VERSION='2025.2', F2_TEST_BAD_FIELD=fault))
    assert not list(dcp.parent.glob('*.Developer_CL.tar'))
    assert (run / 'outputs.json').read_text() == '{"historical": true}\n'


@pytest.mark.parametrize('fault', ['digest', 'source', 'exit', 'gate'])
def test_package_only_rejects_unverified_runs(tmp_path, fault):
    shell, run, cl, dcp = completed_routed(tmp_path)
    digest = '0' * 64 if fault == 'digest' else flow.sha(dcp)
    if fault == 'source':
        (cl / 'design/cl_id_defines.vh').write_text('// modified frozen PCI identity\n')
    elif fault == 'exit':
        (run / 'exit_code').write_text('1\n')
    elif fault == 'gate':
        (run / 'console.log').write_text('qualification did not pass\n')
    with pytest.raises(RuntimeError):
        flow.load_routed_run(run, digest)


@pytest.mark.parametrize('fault', ['none', 'timing', 'skew', 'setup', 'hold', 'route',
                                 'drc', 'methodology', 'drc_fatal', 'methodology_fatal',
                                 'drc_schema', 'methodology_schema', 'classification',
                                 'gray', 'crossing', 'firmware', 'hbm', 'hbm_crossing', 'hbm_gray'])
def test_route_qualification_never_marks_a_failed_design_passed(tmp_path, fault):
    """Exercise gate failures without claiming to simulate Vivado timing itself."""
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl is unavailable')
    script = ROOT / 'boards/f2/build/scripts/validate_route.tcl'
    # Classifiers have their own property/query rejection tests. At this seam,
    # exercise the route gate with a remaining blocker or classification error.
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('check_rule_severity.tcl', 'inspect_ddr_init.tcl'):
        shutil.copy2(script.parent / name, scripts / name)
    (scripts / 'classify_methodology.tcl').write_text('''
proc otpu_qualify_methodology {shell prefix audit} {
  if {$::fault eq "classification"} {error "classification failed"}
  set bad [otpu_blocking_violations methodology [get_methodology_violations] $audit]
  return [dict create raw_blockers $bad inherited {} context {} blockers $bad]
}
''')
    harness = r'''
set criticals 0
proc get_msg_config {args} {return $::criticals}
proc report_route_status {args} {}
proc report_timing_summary {args} {if {$::fault eq "timing"} {incr ::criticals}}
proc report_bus_skew {args} {
  if {$::fault eq "skew"} {incr ::criticals}
  set f [open [lindex $args end] w]
  puts $f "1. Bus Skew Report Summary"
  set count [expr {$::fault eq "gray" ? 13 : [string match hbm* $::fault] ? ($::fault eq "hbm_gray" ? 1 : 0) : 14}]
  for {set i 0} {$i < $count} {incr i} {puts $f "$i 100 u_sys/g_ch"}
  puts $f "2. Bus Skew Report Per Constraint"
  close $f
}
foreach command {report_drc report_methodology report_cdc report_exceptions report_design_analysis
                 report_clock_interaction report_utilization} {proc $command {args} {}}
proc get_drc_violations {} {if {[string match drc* $::fault]} {return blocker}; return {}}
proc get_methodology_violations {} {if {[string match methodology* $::fault]} {return blocker}; return {}}
proc list_property {obj} {
  if {[string match *_schema $::fault]} {return {NAME}}
  return {NAME SEVERITY}
}
proc get_nets {args} {if {$::fault eq "route"} {return unrouted}; return {}}
proc get_timing_paths {args} {
  if {$::fault eq "setup" && "-setup" in $args} {return violation}
  if {$::fault eq "hold" && "-hold" in $args} {return violation}
  return {}
}
proc get_pins {args} {return [lindex $args end]}
proc get_clocks {args} {return [lindex $args end]}
proc report_timing {args} {
  if {$::fault eq "crossing"} {return "No paths found"}
  # the HBM CL: main -> core at the core period (150 MHz here), core -> main 4 ns
  if {[string match hbm* $::fault] && [lindex $args [expr {[lsearch $args -to] + 1}]] eq "WRAPPER/CL/u_clk/O"} {
    if {$::fault eq "hbm_crossing"} {return "Requirement: 4.000ns (MaxDelay Path 4.000ns)"}
    return "Requirement: 6.667ns (MaxDelay Path 6.667ns)"
  }
  return "Requirement: 4.000ns (MaxDelay Path 4.000ns)"
}
proc get_cells {args} {
  if {[lindex $args end] eq "WRAPPER/CL/u_sc"} {
    if {[string match hbm* $::fault]} {return WRAPPER/CL/u_sc}
    return {}
  }
  if {[string match hbm* $::fault]} {return {}}
  return {SH_DDR/mcs0/inst/lmb_bram_I/ramloop[0].ram.r/primitive}
}
proc get_property {prop obj} {
  if {$prop eq "NAME"} {return $obj}
  if {$prop eq "PERIOD"} {return [expr {[string match hbm* $::fault] ? 6.66666 : 10.0}]}
  if {$prop eq "SEVERITY"} {
    if {[string match *_fatal $::fault]} {return Fatal}
    return Error
  }
  if {$::fault eq "firmware"} {return "256'h[string repeat 0 64]"}
  return "256'h[string repeat 1 64]"
}
'''
    harness = (f'set fault {fault}\nset reports_dir {{{tmp_path}}}\n'
               f'set checkpoints_dir {{{tmp_path}}}\n'
               f'set scripts_dir {{{scripts}}}\nset CL test\nset TAG 12345678\n'
               f'set env(HDK_SHELL_DIR) {{{tmp_path}}}\n' + harness +
               f'if {{[catch {{source {{{script}}}}} result]}} {{puts stderr $result; exit 1}}\n')
    result = subprocess.run([tclsh], input=harness, text=True, capture_output=True)
    marker = tmp_path / 'test.12345678.route_validation.passed'
    if fault in ('none', 'hbm'):
        assert result.returncode == 0, result.stderr
        assert marker.is_file()
    else:
        assert result.returncode != 0
        assert not marker.exists()


@pytest.mark.parametrize('kind', ['drc', 'methodology'])
@pytest.mark.parametrize('schema,severity,blocker,error', [
    ('direct', 'Fatal', True, None),
    ('direct', 'Error', True, None),
    ('direct', 'Critical Warning', True, None),
    ('direct', 'Warning', False, None),
    ('direct', 'Advisory', False, None),
    ('direct', 'Unknown', None, 'unknown severity'),
    ('direct', '', None, 'unknown severity'),
    ('missing', 'Warning', None, 'neither SEVERITY nor CHECK'),
    ('check', 'Fatal', True, None),
    ('check', 'Warning', False, None),
    ('check_empty', 'Warning', None, 'empty CHECK'),
    ('check_missing', 'Warning', None, 'resolved to 0 rules'),
    ('check_ambiguous', 'Warning', None, 'resolved to 2 rules'),
    ('check_schema', 'Warning', None, 'lacks NAME or SEVERITY'),
    ('check_mismatch', 'Warning', None, 'did not resolve exactly'),
])
def test_rule_severity_is_explicit_and_unknown_schemas_fail_closed(kind, schema, severity,
                                                                 blocker, error):
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl is unavailable')
    script = ROOT / 'boards/f2/build/scripts/check_rule_severity.tcl'
    harness = r'''
proc list_property {obj} {
  if {$obj eq "violation"} {
    if {$::schema eq "direct"} {return {NAME SEVERITY}}
    if {$::schema eq "missing"} {return {NAME}}
    return {NAME CHECK}
  }
  if {$::schema eq "check_schema"} {return {NAME}}
  return {NAME SEVERITY}
}
proc get_property {prop obj} {
  if {$prop ni [list_property $obj]} {error "read of missing property $prop"}
  switch -- $prop {
    SEVERITY {return $::severity}
    CHECK {if {$::schema eq "check_empty"} {return {}}; return RULE-1}
    NAME {if {$::schema eq "check_mismatch"} {return OTHER-1}; return RULE-1}
  }
}
proc rule_lookup {kind name} {
  if {$kind ne $::kind || $name ne "RULE-1"} {error "incorrect rule lookup"}
  if {$::schema eq "check_missing"} {return {}}
  if {$::schema eq "check_ambiguous"} {return {rule1 rule2}}
  return rule1
}
proc get_drc_checks {name} {return [rule_lookup drc $name]}
proc get_methodology_checks {name} {return [rule_lookup methodology $name]}
'''
    harness = (f'set kind {kind}\nset schema {schema}\nset severity {{{severity}}}\n' + harness +
               f'source {{{script}}}\n' +
               'if {[catch {set bad [otpu_blocking_violations $kind {violation} stdout]} result]} '
               '{puts stderr $result; exit 1}\nputs "blockers=[llength $bad]"\n')
    result = subprocess.run([tclsh], input=harness, text=True, capture_output=True)
    if error:
        assert result.returncode != 0
        assert error in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert f'blockers={int(blocker)}' in result.stdout
        assert 'severity' in result.stdout


@pytest.mark.parametrize('fault', ['none', 'boundary', 'registers', 'warning', 'critical'])
def test_linked_reset_exception_is_scoped_and_guarded(fault):
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl is unavailable')
    script = ROOT / 'boards/f2/build/constraints/cl_pnr_user.tcl'
    harness = r'''
set warnings 0
set criticals 0
proc get_msg_config {args} {
  if {[lindex $args end] eq "WARNING"} {return $::warnings}
  return $::criticals
}
proc get_cells {args} {
  set name [lindex $args end]
  if {$::fault eq "registers"} {return [list ${name}0 ${name}1]}
  return [list ${name}0 ${name}1 ${name}2]
}
proc get_pins {args} {
  if {"-of_objects" ni $args} {
    if {$::fault eq "boundary"} {return {}}
    return WRAPPER/CL/rst_main_n
  }
  set cells [lindex $args [expr {[lsearch -exact $args -of_objects] + 1}]]
  set result {}
  foreach cell $cells {lappend result $cell/CLR}
  return $result
}
proc set_false_path {args} {
  if {[lindex $args 0] ne "-through" || [lindex $args 1] ne "WRAPPER/CL/rst_main_n" ||
      [lindex $args 2] ne "-to" || [llength [lindex $args 3]] != 9} {
    error "incorrect reset exception scope"
  }
  if {$::fault eq "warning"} {incr ::warnings}
  if {$::fault eq "critical"} {incr ::criticals}
}
'''
    harness = (f'set fault {fault}\n' + harness +
               f'if {{[catch {{source {{{script}}}}} result]}} {{puts stderr $result; exit 1}}\n')
    result = subprocess.run([tclsh], input=harness, text=True, capture_output=True)
    if fault == 'none':
        assert result.returncode == 0, result.stderr
        assert 'OTPU_LINKED_RESET_EXCEPTION_PASSED endpoints=9' in result.stdout
    else:
        assert result.returncode != 0
        assert 'OTPU_LINKED_RESET_EXCEPTION_PASSED' not in result.stdout


def test_unmaterialized_ddr_ip_is_rejected(tmp_path, monkeypatch):
    t = T.load('f2')
    monkeypatch.setattr(flow, 'command', lambda *a, **k:
        t.platform['ip_commit'] if str(a[2]).endswith('/hdk/common/ip') else t.platform['hdk_commit'])
    shell = tmp_path / 'hdk/common/shell_stable'
    for rel in ('design/interfaces/cl_ports.vh', 'build/scripts/build_all.tcl'):
        f = shell / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text('// available\n')
    (shell / 'shell_version.txt').write_text('small_shell=0x10212415\n')
    ip = tmp_path / 'hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_ddr4/cl_ddr4.xci'
    ip.parent.mkdir(parents=True)
    ip.write_text(json.dumps({'ip_inst': {'parameters': {'component_parameters': {
        'C0.DDR4_TimePeriod': [{'value': '938'}]}}}}))
    generated = tmp_path / 'hdk/common/ip/cl_ip/cl_ip.gen/sources_1/ip/cl_ddr4'
    generated.mkdir(parents=True)
    (generated / 'cl_ddr4.xml').write_text('version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(RuntimeError, match='Git LFS pointer'):
        flow.check_hdk(tmp_path, t, shell=False)


@pytest.mark.parametrize('name,S,nch,dram_log2,portb,khz',
                         [('f2-hbm', 1, 8, 32, 1, 200000), ('f2-hbm:s2', 2, 8, 32, 1, 200000),
                          ('f2-hbm16', 1, 16, 33, 1, 200000),
                          ('f2-hbm:p2', 1, 8, 32, 2, 200000), ('f2-hbm16:p4', 1, 16, 33, 4, 200000),
                          ('f2-hbm:p4rl8a64c150', 1, 8, 32, 4, 150000)])
def test_hbm_snapshot_takes_the_hbm_top_scripts_and_geometry(tmp_path, name, S, nch, dram_log2,
                                                              portb, khz):
    """An HBM target (docs/f2_hbm.md): the HBM top in cl_opentpu's place, its synthesis script
    and CDC constraints in the DDR ones' places, the HDK's wrapper copied in, the geometry the
    target's (S x NCH ports of 512 MiB), the programs' architecture the f2 target's with T2 (the
    p2 / p4 variants: with a wide port B, PORTB in the parameters and the hash)."""
    shell = tmp_path / 'shell'
    scripts = shell / 'build/scripts'
    scripts.mkdir(parents=True)
    (scripts / 'build_all.tcl').write_text('# test HDK script\n')
    wrapper = tmp_path / 'cl_mem_hbm_wrapper.sv'
    wrapper.write_text('module cl_mem_hbm_wrapper; endmodule\n')
    t = T.load(name)
    assert flow.is_hbm(t) and t.core_khz == khz and t.uarch['TMEM_PMP'] == 1
    if S == 1 and khz == 200000:        # f2's programs and T2 (docs/ternary.md)
        assert T.config(t) == dataclasses.replace(T.config('f2'), T2=True, PORTB=portb)
        assert T.rtl_params(t)['ARCH_HASH'] == T.arch_hash(T.config(t)) != 0x4FFD1E17
        if portb > 1:                   # in the hash only when wide (docs/wide_port.md)
            assert T.arch_hash(T.config(t)) != T.arch_hash(T.config(name.split(':')[0]))
    out = tmp_path / 'run'
    cl = flow.prepare(out, shell, t, '89abcdef', wrapper)
    d = cl / 'design'
    assert not (d / 'cl_opentpu.sv').exists() and (d / 'cl_opentpu_hbm.sv').is_file()
    assert (d / 'cl_mem_hbm_wrapper.sv').read_text() == wrapper.read_text()
    # the core MMCM from clk_main_a0 (250 MHz): the core clock and the pumped TMEM's at 2x
    div, mult, d0, d1 = {200000: (1, 4.8, 6.0, 3), 150000: (1, 4.8, 8.0, 4)}[khz]
    assert 250000 * mult / div / d0 == khz and d0 == 2 * d1 and 800 <= 250 * mult / div <= 1600
    assert (d / 'opentpu_geometry.vh').read_text() == (
        f'localparam int S = {S};\nlocalparam int NCH = {nch};\n'
        f'localparam int CORE_KHZ_CL = {khz};\nlocalparam int MMCM_DIVCLK = {div};\n'
        f'localparam real MMCM_MULT = {mult};\nlocalparam real MMCM_DIV0 = {d0};\n'
        f'localparam int MMCM_DIV1 = {d1};\n')
    assert (cl / 'build/scripts/synth_cl_opentpu.tcl').read_bytes() == \
        (ROOT / 'boards/f2/build/scripts/synth_cl_opentpu_hbm.tcl').read_bytes()
    assert (cl / 'build/constraints/cl_timing_user.tcl').read_bytes() == \
        (ROOT / 'boards/f2/build/constraints/cl_timing_user_hbm.tcl').read_bytes()
    pnr = (cl / 'build/constraints/cl_pnr_user.tcl').read_text()
    assert 'hbm_reset_n}' in pnr
    # the floorplan after the reset exception, in the one file build_level_1_cl.tcl reads
    # unmanaged at placement; the managed XDC the HDK reads stays Tcl-free
    floorplan = (ROOT / 'boards/f2/build/constraints/cl_pnr_user_hbm.tcl').read_text()
    assert pnr.endswith('\n' + floorplan) and pnr.index('hbm_reset_n}') < pnr.index('pb_otpu_hbm')
    assert not (cl / 'build/constraints/cl_pnr_user_hbm.tcl').exists()
    managed = (cl / 'build/constraints/small_shell_cl_pnr_user.xdc').read_text()
    assert all(line.startswith('#') for line in managed.splitlines() if line.strip())
    params = json.loads((out / 'inputs.json').read_text())['rtl_parameters']
    assert (params['S'], params['NCH'], params['CH_LOG2'], params['DRAM_LOG2']) == \
        (S, nch, 29, dram_log2)
    assert params['CORE_KHZ'] == khz and 'CH_BYTES' not in params and 'DDR_MTS' not in params
    vh = (d / 'opentpu_params.vh').read_text()
    assert f".NCH(32'h{nch:x})" in vh and ".TMEM_PMP(32'h1)" in vh
    assert params['PORTB'] == portb and f".PORTB(32'h{portb:x})" in vh
    srcs = (d / 'sources.tcl').read_text()
    assert 'rtl/boards/f2/otpu_f2_hbm_sys.sv' in srcs and 'otpu_f2_sys.sv' not in srcs
    # Alex's shell: otpu_native_axi into the SmartConnect fabric, the 450 MHz bridges out of it,
    # the block design's generator in the snapshot (build.py runs it before synthesis)
    for f in ('otpu_native_axi.sv', 'otpu_f2_pcis_gate.sv', 'otpu_f2_hbm_axi.sv', 'otpu_hbm_geometry.sv'):
        assert f'rtl/boards/f2/{f}' in srcs
    for f in ('otpu_f2_hbm_nch', 'otpu_f2_hbm_port', 'otpu_f2_hbm_host', 'otpu_f2_hbm_mem', 'otpu_afifo.sv x'):
        assert f'{f}.sv' not in srcs or f.endswith(' x')
    for f in ('create_hbm_sc.tcl', 'hbm_sc_inst.tcl'):
        assert (cl / 'build/scripts' / f).read_bytes() == (ROOT / 'boards/f2/build/scripts' / f).read_bytes()
    assert flow.hbm_sc_command(cl, t, out / 'hbm_sc.log', tmp_path / 'hdk')[-5:] == [
        str(cl / 'ip'), str(S * nch), 'xcvu47p-fsvh2892-2-e', str(khz * 1000),
        str(tmp_path / 'hdk' / flow.HBM_BD_SRC)]
    # the CL: AWS's cl_dram_hbm_dma chain per channel and for the host's own port, both HBM
    # monitor APBs connected as the AWS Shell Interface Specification asks (else AFI creation fails)
    top = (d / 'cl_opentpu_hbm.sv').read_text()
    for inst in ('axi_register_slice u_rs512', 'cl_axi3_256b_reg_slice u_rs256', 'axi_register_slice u_hrs512', 'axi_register_slice u_hrs512x',
                 'cl_axi_sc_1x1_wrapper u_hsc', 'cl_axi3_256b_reg_slice u_hrs256',
                 'cl_mem_hbm_wrapper #(.NUM_OF_AXI_PORTS(NHBM)'):
        assert inst in top
    for k in (0, 1):
        assert f'.i_hbm_apb_preset_n_{k}(hbm_apb_preset_n_{k})' in top
        for sig in ('paddr', 'pprot', 'psel', 'penable', 'pwrite', 'pwdata', 'pstrb', 'pready',
                    'prdata', 'pslverr'):
            assert f'.o_hbm_apb_{sig}_{k}(hbm_apb_{sig}_{k})' in top
    for f in T.sources(t):
        assert (d / 'rtl' / f).read_bytes() == (ROOT / 'rtl' / f).read_bytes()


def test_hbm_snapshot_refuses_32_ports(tmp_path):
    """f2-hbm:s4 (4 x 8 channels: every HBM port) leaves the host no port of its own."""
    shell = tmp_path / 'shell'
    (shell / 'build/scripts').mkdir(parents=True)
    (shell / 'build/scripts/build_all.tcl').write_text('#\n')
    with pytest.raises(RuntimeError, match='S x NCH <= 31'):
        flow.prepare(tmp_path / 'run', shell, T.load('f2-hbm:s4'), '01234567', tmp_path / 'w.sv')


def test_ddr_snapshot_leaves_the_hbm_files_out(tmp_path):
    shell = tmp_path / 'shell'
    (shell / 'build/scripts').mkdir(parents=True)
    (shell / 'build/scripts/build_all.tcl').write_text('#\n')
    cl = flow.prepare(tmp_path / 'run', shell, T.load('f2'), '01234567')
    assert (cl / 'design/cl_opentpu.sv').is_file()
    assert not list(cl.rglob('*hbm*'))
    assert 'pb_otpu' not in (cl / 'build/constraints/cl_pnr_user.tcl').read_text()


# The HBM floorplan (cl_pnr_user_hbm.tcl): four pblocks under the HDK's pblock_CL by whole clock
# regions, one SLR each (the fabric in SLR0, the crossing chains' middle stage in SLR1, the board
# in SLR2, the TMEM and VPU inside the board's), cut down to the sites their parent holds (the
# mock's pblock_CL lacks "shell" of SLR2's clock regions, the static shell's site), routing not
# contained, and every cell query failing closed (docs/wide_port.md "Placement")
FLOORPLAN_HARNESS = r'''
set pblocks {pblock_CL}
set parents {}
set ranges {}
set removed {}
set cells {}
set props {}
# the sites of a range; pblock_CL's own ranges are the HDK's site ranges (P0 P1 P2 here)
set sites {CLOCKREGION_X0Y0:CLOCKREGION_X3Y3 {s0a} CLOCKREGION_X4Y0:CLOCKREGION_X7Y0 {s0b}
           CLOCKREGION_X0Y4:CLOCKREGION_X3Y7 {s1a} CLOCKREGION_X5Y5:CLOCKREGION_X5Y6 {s1b}
           CLOCKREGION_X4Y3:CLOCKREGION_X4Y3 {s0c}
           CLOCKREGION_X0Y0:CLOCKREGION_X0Y1 {s0a} CLOCKREGION_X1Y0:CLOCKREGION_X1Y1 {s0b}
           CLOCKREGION_X1Y0:CLOCKREGION_X1Y3 {s0b}
           CLOCKREGION_X5Y1:CLOCKREGION_X5Y3 {s0a} CLOCKREGION_X6Y1:CLOCKREGION_X6Y1 {s0b}
           {SLICE_X120Y120:SLICE_X145Y179 SLICE_X118Y60:SLICE_X145Y119 SLICE_X206Y60:SLICE_X219Y119 SLICE_X221Y60:SLICE_X232Y119} {s0c}
           CLOCKREGION_X0Y8:CLOCKREGION_X7Y11 {s2a s2b shell}
           CLOCKREGION_X2Y9:CLOCKREGION_X5Y10 {s2b} P0 {s0a s0b s0c} P1 {s1a s1b} P2 {s2a s2b}}
set ranges {{pblock_CL P0} {pblock_CL P1} {pblock_CL P2}}
proc create_pblock {name} { lappend ::pblocks $name }
proc get_pblocks {name} { if {$name ni $::pblocks} { error "no pblock $name" }; return $name }
proc get_clock_regions {args} {
  set o [lindex $args end]
  if {[regexp {BLI_HBM_AXI_INTF_X(\d+)Y0} $o -> p]} { return "X[expr {$p / 4}]Y0" }
  return $o
}
proc get_slrs {args} { return SLR1 }
proc get_property {prop name} {
  if {$prop eq "NAME"} { return $name }
  foreach p $::props { if {[lindex $p 0] eq $name && [lindex $p 1] eq $prop} { return [lindex $p 2] } }
  error "no $prop on $name"
}
proc get_sites {flag name} {
  if {$flag eq "-quiet"} { return $name }   ;# the HBM AXI port sites: found
  if {$flag ne "-of_objects"} { error "get_sites $flag" }
  set out {}
  foreach r $::ranges {
    if {[lindex $r 0] eq $name} { lappend out {*}[dict get $::sites [lindex $r 1]] }
  }
  foreach r $::removed {
    if {[lindex $r 0] eq $name} { foreach s [lindex $r 1] { set out [lsearch -all -inline -not -exact $out $s] } }
  }
  return $out
}
proc get_cells {args} {
  set pattern [lindex $args end]
  if {$::fault eq "missing" && [string match *u_board* $pattern]} { return {} }
  if {[string match {*g_port\[\*\]*} $pattern]} { return [list ${pattern}0 ${pattern}1 ${pattern}2 ${pattern}3] }
  return [list $pattern]
}
proc resize_pblock {name flag range} {
  if {$name ni $::pblocks} { error "resize $name" }
  if {$flag eq "-add"} { lappend ::ranges [list $name $range]; return }
  if {$flag ne "-remove" || ![llength $range]} { error "resize $name $flag" }
  lappend ::removed [list $name $range]
}
proc add_cells_to_pblock {name cl} {
  if {$name ni $::pblocks} { error "add to $name" }
  foreach c $cl { lappend ::cells [list $name $c] }
}
proc set_property {prop value name} { lappend ::props [list $name $prop $value] }
proc get_pins {args} { return [lindex $args end] }
proc get_nets {args} { set out {}; foreach p [lindex $args end] { lappend out net_of_$p }; return $out }
'''


@pytest.mark.parametrize('fault', ['none', 'missing'])
def test_hbm_floorplan_pblocks_are_one_slr_each_and_fail_closed(fault):
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl is unavailable')
    script = ROOT / 'boards/f2/build/constraints/cl_pnr_user_hbm.tcl'
    report = r'''
puts "PBLOCKS $pblocks"
foreach r $ranges { puts "RANGE [lindex $r 0] [lindex $r 1]" }
foreach r $removed { puts "REMOVED [lindex $r 0] [lindex $r 1]" }
foreach c $cells { puts "CELL [lindex $c 0] [lindex $c 1]" }
foreach p $props { puts "PROP [lindex $p 0] [lindex $p 1] [lindex $p 2]" }
'''
    harness = (f'set fault {fault}\n' + FLOORPLAN_HARNESS +
               f'if {{[catch {{source {{{script}}}}} result]}} {{puts stderr $result; exit 1}}\n' + report)
    result = subprocess.run([tclsh], input=harness, text=True, capture_output=True)
    if fault == 'missing':
        assert result.returncode != 0 and 'no cells match' in result.stderr
        assert 'OTPU_FLOORPLAN_APPLIED' not in result.stdout
        return
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # 4 channel ports (the mock) in X0Y0 and the host's port 4 in X1Y0: two boxes by the ports
    assert 'OTPU_FLOORPLAN ports=4' in out and 'OTPU_FLOORPLAN_APPLIED pblocks=8' in out
    assert 'PBLOCKS pblock_CL pb_otpu_hbm pb_otpu_mid pb_otpu_pcis pb_otpu_board pb_otpu_tmem pb_otpu_hbmx0 pb_otpu_hbmx1 pb_otpu_hxs' in out
    assert 'OTPU_FLOORPLAN shell PCIS SLR1' in out and 'OTPU_FLOORPLAN_WARNING' not in out
    ranges = {}
    for line in out.splitlines():
        if line.startswith('RANGE ') and not line.startswith('RANGE pblock_CL'):
            _, pb, rng = line.split(' ', 2)
            ranges.setdefault(pb, []).append(rng)
    # whole clock regions, one SLR per pblock (Y0-3 SLR0, Y4-7 SLR1, Y8-11 SLR2); X4Y3 (round 6's
    # pb_otpu_xs, gone with otpu_f2_hbm_host) joins the HBM side, as in Alex's pblock_CL_SLR0
    assert ranges == {
        'pb_otpu_hbm': ['CLOCKREGION_X0Y0:CLOCKREGION_X3Y3', 'CLOCKREGION_X4Y0:CLOCKREGION_X7Y0',
                        'CLOCKREGION_X4Y3:CLOCKREGION_X4Y3', 'CLOCKREGION_X5Y1:CLOCKREGION_X5Y3',
                        'CLOCKREGION_X6Y1:CLOCKREGION_X6Y1',
                        'SLICE_X120Y120:SLICE_X145Y179 SLICE_X118Y60:SLICE_X145Y119 '
                        'SLICE_X206Y60:SLICE_X219Y119 SLICE_X221Y60:SLICE_X232Y119'],
        'pb_otpu_mid': ['CLOCKREGION_X0Y4:CLOCKREGION_X3Y7'],
        'pb_otpu_pcis': ['CLOCKREGION_X5Y5:CLOCKREGION_X5Y6'],
        'pb_otpu_board': ['CLOCKREGION_X0Y8:CLOCKREGION_X7Y11'],
        'pb_otpu_tmem': ['CLOCKREGION_X2Y9:CLOCKREGION_X5Y10'],
        'pb_otpu_hbmx0': ['CLOCKREGION_X0Y0:CLOCKREGION_X0Y1'],
        'pb_otpu_hbmx1': ['CLOCKREGION_X1Y0:CLOCKREGION_X1Y3'],     # the host's column: full height
        'pb_otpu_hxs': ['CLOCKREGION_X4Y3:CLOCKREGION_X4Y3']}
    cells = {}
    for line in out.splitlines():
        if line.startswith('CELL '):
            _, pb, cell = line.split(' ', 2)
            cells.setdefault(pb, []).append(cell)
    hbm, mid, pcis, board, tmem = (cells[k] for k in ('pb_otpu_hbm', 'pb_otpu_mid', 'pb_otpu_pcis', 'pb_otpu_board', 'pb_otpu_tmem'))
    # the PCIS gate (Alex's shell: otpu_f2_pcis_gate, every output a register) and the OCL's
    # crossing beside the shell, SLR1
    # (and the host chain's first register slice: the shell's PCIS SLR, the spec's crossing rule)
    assert sorted(pcis) == ['WRAPPER/CL/u_hrs512', 'WRAPPER/CL/u_sys/u_mem/u_pcis', 'WRAPPER/CL/u_sys/u_ocl']
    # the core and pump clock nets in one delay group (round 8)
    assert 'PROP net_of_WRAPPER/CL/u_clk/O net_of_WRAPPER/CL/u_clk_f/O CLOCK_DELAY_GROUP otpu_core_pump' in out
    assert all(c.startswith(('WRAPPER/CL/u_sys/', 'WRAPPER/CL/u_sc', 'WRAPPER/CL/g_ch', 'WRAPPER/CL/u_h'))
               for pb in cells.values() for c in pb)
    # each channel's SmartConnect and 450 MHz register slice beside its port, the host's chain
    # beside its own (port NPORT)
    assert sorted(cells['pb_otpu_hbmx0']) == sorted(
        [f'WRAPPER/CL/u_sc/otpu_hbm_sc_i/sc_c{p}' for p in range(4)] +
        [f'WRAPPER/CL/g_ch[{p}].u_rs256' for p in range(4)])
    assert sorted(cells['pb_otpu_hbmx1']) == ['WRAPPER/CL/u_hrs256', 'WRAPPER/CL/u_hsc']
    assert cells['pb_otpu_hxs'] == ['WRAPPER/CL/u_hrs512x']       # the second host slice at the crossing
    assert 'WRAPPER/CL/g_ch[*].u_rs512' in hbm
    # the channels' otpu_native_axi and the SmartConnect fabric by the HBM's ports
    assert len([c for c in hbm if '.u_naxi' in c]) == 4 and 'WRAPPER/CL/u_sc' in hbm
    assert not any('u_nch' in c or 'u_host' in c or 'u_hreg' in c or 'u_port' in c for c in hbm)
    assert 'WRAPPER/CL/u_sys/u_board' in board
    # the crossing chains: stage k in SLR k, the control path's two sides with their SLRs
    for k, pb in enumerate((hbm, mid, board)):
        for chain in ('u_mem/g_xr', 'u_cx/g_xr'):
            for d in ('u_dn', 'u_up'):
                assert f'WRAPPER/CL/u_sys/{chain}.g_x[{k}].{d}' in pb
        assert f'WRAPPER/CL/u_sys/g_rs[{k}].u_x' in pb
    assert 'WRAPPER/CL/u_sys/u_cx/g_xr.u_m' in hbm and 'WRAPPER/CL/u_sys/u_cx/g_xr.u_s' in board
    assert sorted(tmem) == ['WRAPPER/CL/u_sys/u_board/u_slice/u_tmem', 'WRAPPER/CL/u_sys/u_board/u_slice/u_vpu']
    assert not (set(hbm) & set(mid)) and not (set(hbm) & set(board)) and not (set(mid) & set(board))
    assert not (set(pcis) & (set(hbm) | set(mid) | set(board)))
    props = {(l.split()[1], l.split()[2]): l.split()[3] for l in out.splitlines() if l.startswith('PROP ')}
    for pb in ('pb_otpu_hbm', 'pb_otpu_mid', 'pb_otpu_pcis', 'pb_otpu_board', 'pb_otpu_tmem',
               'pb_otpu_hbmx0', 'pb_otpu_hbmx1', 'pb_otpu_hxs'):
        assert props[(pb, 'CONTAIN_ROUTING')] == '0' and props[(pb, 'IS_SOFT')] == '0'
        assert props[(pb, 'SNAPPING_MODE')] == 'OFF'
        assert props[(pb, 'PARENT')] == {'pb_otpu_tmem': 'pb_otpu_board', 'pb_otpu_hbmx0': 'pb_otpu_hbm',
                                         'pb_otpu_hbmx1': 'pb_otpu_hbm', 'pb_otpu_hxs': 'pb_otpu_hbm'}.get(pb, 'pblock_CL')
    # containment: the one site of SLR2's clock regions that pblock_CL does not hold leaves
    # pb_otpu_board (and pb_otpu_tmem, inside it, is checked against the trimmed board)
    removed = [l.split(' ', 2)[1:] for l in out.splitlines() if l.startswith('REMOVED ')]
    assert removed == [['pb_otpu_board', 'shell']]
    assert 'OTPU_FLOORPLAN pb_otpu_board parent=pblock_CL held=7 removed=1' in out
    assert 'OTPU_FLOORPLAN pb_otpu_tmem parent=pb_otpu_board held=2 removed=0' in out


# build_level_1_cl.tcl: the HDK reads the user placement XDC managed (its Tcl dropped), so the
# wrapper reads cl_pnr_user.tcl, then otpu_clk2x.tcl (the pumped TMEM's multicycles, on the
# linked design), unmanaged inside the HDK's opt_design call, once, and fails if the HDK's
# sequence never reached it
@pytest.mark.parametrize('fault', ['none', 'no_opt', 'bad_tcl'])
def test_link_wrapper_reads_the_placement_tcl_unmanaged_before_opt_design(tmp_path, fault):
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl is unavailable')
    hdk = tmp_path / 'hdk/build/scripts'
    hdk.mkdir(parents=True)
    hdk_steps = 'read_xdc $::constraints_dir/small_shell_cl_pnr_user.xdc\n'
    if fault != 'no_opt':
        hdk_steps += 'opt_design -directive Explore\nplace_design\n'
    (hdk / 'build_level_1_cl.tcl').write_text(hdk_steps)
    cons = tmp_path / 'constraints'
    cons.mkdir()
    (cons / 'small_shell_cl_pnr_user.xdc').write_text('# managed\n')
    (cons / 'cl_pnr_user.tcl').write_text('error "bad tcl"\n' if fault == 'bad_tcl' else
                                          'puts "PNR_TCL_RAN"\n')
    (cons / 'otpu_clk2x.tcl').write_text('puts "CLK2X_TCL_RAN"\n')
    harness = f'''
set HDK_SHELL_DIR {{{tmp_path / 'hdk'}}}
set constraints_dir {{{cons}}}
set checkpoints_dir {{{tmp_path}}}
set scripts_dir {{{tmp_path}}}
set CL cl
set TAG t
set DEVICE_TYPE xcvu47p
set log {{}}
proc set_param {{args}} {{}}
proc create_project {{args}} {{}}
proc read_xdc {{args}} {{
  lappend ::log "read_xdc $args"
  if {{[lindex $args 0] eq "-unmanaged"}} {{ source [lindex $args 1] }}
}}
proc opt_design {{args}} {{ lappend ::log "opt_design $args" }}
proc place_design {{args}} {{ lappend ::log "place_design" }}
proc close_project {{}} {{ error "stop" }}
if {{[catch {{source {{{ROOT / 'boards/f2/build/scripts/build_level_1_cl.tcl'}}}}} result]}} {{
  puts "STOPPED $result"
}}
foreach l $log {{ puts "LOG $l" }}
'''
    result = subprocess.run([tclsh], input=harness, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    log = [l[4:] for l in out.splitlines() if l.startswith('LOG ')]
    if fault == 'none':
        assert 'PNR_TCL_RAN' in out and 'CLK2X_TCL_RAN' in out
        assert log == [f'read_xdc {cons}/small_shell_cl_pnr_user.xdc',
                       f'read_xdc -unmanaged {cons}/cl_pnr_user.tcl',
                       f'read_xdc -unmanaged {cons}/otpu_clk2x.tcl',
                       'opt_design -directive Explore', 'place_design']
        assert 'STOPPED openTPU: implementation produced no routed checkpoint' in out
    elif fault == 'no_opt':
        assert 'STOPPED openTPU: the HDK sequence ran no opt_design' in out and 'PNR_TCL_RAN' not in out
    else:
        assert 'STOPPED bad tcl' in out and 'opt_design' not in ' '.join(log)
        assert 'CLK2X_TCL_RAN' not in out


# The SmartConnect fabric's instance (hbm_sc_inst.tcl, sourced by create_hbm_sc.tcl after
# make_wrapper): a wrapper's ports as Vivado declares them, mapped onto the CL's signals
def _sc_wrapper(nport, *, mid=False, drop=None, wide=None, extra=None, sid=16):
    """a Vivado-style wrapper of the channels' block design: per channel S_N<p> (AXI4 512 b,
    64-bit addresses, sid-bit IDs: cl_axi_sc_1x1's settings) and M_H<p> (AXI3 256 b, 64-bit
    addresses, IDs only with `mid`)"""
    s4 = [('araddr', 'input', 64), ('arburst', 'input', 2), ('arcache', 'input', 4), ('arid', 'input', sid),
          ('arlen', 'input', 8), ('arlock', 'input', 1), ('arprot', 'input', 3), ('arqos', 'input', 4),
          ('arready', 'output', 1), ('arsize', 'input', 3), ('arvalid', 'input', 1),
          ('awaddr', 'input', 64), ('awburst', 'input', 2), ('awcache', 'input', 4), ('awid', 'input', sid),
          ('awlen', 'input', 8), ('awlock', 'input', 1), ('awprot', 'input', 3), ('awqos', 'input', 4),
          ('awready', 'output', 1), ('awsize', 'input', 3), ('awvalid', 'input', 1),
          ('bid', 'output', sid), ('bready', 'input', 1), ('bresp', 'output', 2), ('bvalid', 'output', 1),
          ('rdata', 'output', 512), ('rid', 'output', sid), ('rlast', 'output', 1), ('rready', 'input', 1),
          ('rresp', 'output', 2), ('rvalid', 'output', 1), ('wdata', 'input', 512), ('wlast', 'input', 1),
          ('wready', 'output', 1), ('wstrb', 'input', 64), ('wvalid', 'input', 1)]
    sn = [p for p in s4 if sid or p[0] not in ('arid', 'awid', 'bid', 'rid')]
    m3 = [(n, {'input': 'output', 'output': 'input'}[d], w) for n, d, w in s4
          if n not in ('arid', 'awid', 'bid', 'rid')]
    m3 = [(n, d, {'arlen': 4, 'awlen': 4, 'rdata': 256, 'wdata': 256,
                  'wstrb': 32, 'arlock': 2, 'awlock': 2}.get(n, w)) for n, d, w in m3]
    if mid:
        m3 += [('arid', 'output', 2), ('awid', 'output', 2), ('rid', 'input', 2), ('bid', 'input', 2)]
    lines = ['module otpu_hbm_sc_wrapper', '   (core_clk);']
    lines += [f'  input {n};' for n in ('aresetn', 'core_clk', 'hbm_clk')]
    ports = []
    for p in range(nport):
        ports += [(f'M_H{p}_{n}', d, w) for n, d, w in m3] + [(f'S_N{p}_{n}', d, w) for n, d, w in sn]
    ports += extra or []
    for n, d, w in ports:
        if n == drop:
            continue
        w = wide if (wide and n == 'S_N1_wdata') else w
        lines.append(f'  {d} {f"[{w - 1}:0]" if w > 1 else ""}{n};')
    return '\n'.join(lines) + '\nendmodule\n'


@pytest.mark.parametrize('case', ['aws', 'ids', 'noid', 'unknown', 'width', 'missing'])
def test_hbm_sc_instance_maps_every_wrapper_port_and_fails_closed(tmp_path, case):
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl is unavailable')
    kw = {'ids': {'mid': True}, 'noid': {'sid': 0}, 'unknown': {'extra': [('S_N0_foo', 'input', 3)]},
          'width': {'wide': 256}, 'missing': {'drop': 'M_H3_rdata'}}.get(case, {})
    w = tmp_path / 'w.v'
    w.write_text(_sc_wrapper(8, **kw))
    script = ROOT / 'boards/f2/build/scripts/hbm_sc_inst.tcl'
    r = subprocess.run([tclsh], input=f'source {{{script}}}\nset f [open {{{w}}}]\n'
                       'if {[catch {puts [otpu_sc_instance [read $f] 8]} e]} {puts stderr $e; exit 1}\n',
                       text=True, capture_output=True)
    if case in ('unknown', 'width', 'missing'):
        assert r.returncode != 0 and 'openTPU SmartConnect' in r.stderr, r.stdout[-500:]
        return
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert out.count('otpu_hbm_sc_wrapper u_sc (') == 1
    for k in range(8):
        # the 512-bit side: the channel's register slice (64-bit addresses, 16-bit IDs)
        assert f'.S_N{k}_araddr(cs_araddr[{k}])' in out and f'.S_N{k}_arlen(cs_arlen[{k}])' in out
        assert f'.S_N{k}_wlast(cs_wlast[{k}])' in out and f".S_N{k}_arlock('0)" in out
        if case == 'noid':
            assert f"assign cs_bid[{k}] = '0;" in out and f'.S_N{k}_awid' not in out
        else:
            assert f'.S_N{k}_awid(cs_awid[{k}])' in out and f'.S_N{k}_rid(cs_rid[{k}])' in out
        # the 256-bit side: 64-bit SmartConnect addresses onto the slice's 34 through a wire
        assert f'.M_H{k}_awaddr(sc_M_H{k}_awaddr)' in out and f'logic [63:0] sc_M_H{k}_awaddr;' in out
        assert f"assign cm_awaddr[{k}] = 34'(sc_M_H{k}_awaddr);" in out
        assert f'.M_H{k}_rdata(cm_rdata[{k}])' in out and f'.M_H{k}_awcache()' in out
        if case == 'ids':
            assert f".M_H{k}_arid(sc_M_H{k}_arid)" in out and f"assign cm_arid[{k}] = 6'(sc_M_H{k}_arid);" in out
            assert f".M_H{k}_rid(2'(cm_rid[{k}]))" in out and f"assign cm_arid[{k}] = '0;" not in out
        else:
            assert f"assign cm_arid[{k}] = '0;" in out and f"assign cm_awid[{k}] = '0;" in out
    assert '.core_clk(clk)' in out and '.hbm_clk(hclk)' in out and '.aresetn(sc_aresetn)' in out
    assert 'S_HOST' not in out and 'M_HOST' not in out


# create_hbm_sc.tcl under a Vivado mock: the block design's cells, clocks, connections and address
# map as built (per channel a 2 x 1 SmartConnect on three clocks, the host's 1 x NPORT demux, two
# levels past 16; every master at p x 512 MiB from its channel and from the host), then the
# instance from the (mock) wrapper and ip_done.txt
SC_MOCK = r'''
set calls {}
set opened {}
proc rec {args} { lappend ::calls $args }
foreach c {set_param create_project create_bd_design validate_bd_design save_bd_design generate_target
           write_bd_tcl launch_runs wait_on_run create_ip_run add_files remove_files close_bd_design} { proc $c {args} [format {rec %s {*}$args} $c] }
proc open_bd_design {args} { rec open_bd_design {*}$args; set ::opened ref }
proc set_property {args} { rec set_property {*}$args }
proc create_bd_port {args} { rec create_bd_port {*}$args; return [lindex $args end] }
proc create_bd_intf_port {args} { rec create_bd_intf_port {*}$args; return [lindex $args end] }
proc create_bd_cell {args} { rec create_bd_cell {*}$args; return [lindex $args end] }
# the reference block design (AWS's cl_axi_sc_1x1): one SmartConnect, an AXI4 slave, an AXI3 master
proc get_bd_cells {args} { return ref_sc }
proc get_bd_intf_ports {args} {
  if {[lindex $args 0] eq "-filter"} { return [expr {[string match *Slave* [lindex $args 1]] ? "AXI4" : "AXI3"}] }
  return [lindex $args end]
}
proc get_bd_designs {args} { return [lindex $args end] }
proc list_property {obj pat} {
  return [dict get {ref_sc {CONFIG.NUM_SI CONFIG.NUM_MI CONFIG.NUM_CLKS CONFIG.ADVANCED_PROPERTIES CONFIG.Component_Name}
                    AXI4 {CONFIG.PROTOCOL CONFIG.DATA_WIDTH CONFIG.ADDR_WIDTH CONFIG.FREQ_HZ CONFIG.NUM_READ_OUTSTANDING CONFIG.MAX_BURST_LENGTH CONFIG.ID_WIDTH}
                    AXI3 {CONFIG.PROTOCOL CONFIG.DATA_WIDTH CONFIG.FREQ_HZ CONFIG.NUM_READ_OUTSTANDING CONFIG.MAX_BURST_LENGTH CONFIG.ID_WIDTH}} $obj]
}
proc get_property {args} {
  set k [lindex $args 0]; set o [lindex $args end]
  set v {ref_sc {CONFIG.NUM_SI 1 CONFIG.NUM_MI 1 CONFIG.NUM_CLKS 2 CONFIG.ADVANCED_PROPERTIES 0 CONFIG.Component_Name cl_axi_sc_1x1_smartconnect_0 NAME smartconnect_0 VLNV xilinx.com:ip:smartconnect:1.0}
                     AXI4 {CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 64 CONFIG.FREQ_HZ 250000000 CONFIG.NUM_READ_OUTSTANDING 32 CONFIG.MAX_BURST_LENGTH 64 CONFIG.ID_WIDTH 16 NAME AXI4}
                     AXI3 {CONFIG.PROTOCOL AXI3 CONFIG.DATA_WIDTH 256 CONFIG.FREQ_HZ 450000000 CONFIG.NUM_READ_OUTSTANDING 64 CONFIG.MAX_BURST_LENGTH 16 CONFIG.ID_WIDTH 0 NAME AXI3}}
  if {[string match *S00_AXI $o]} {
    # the channel SmartConnect's slave pin after validation: AWS's 32 / 32, bursts to 64 (or the
    # fault's)
    if {$k eq "CONFIG.MAX_BURST_LENGTH"} { return 64 }
    return [expr {$::sc_fault ? 16 : 32}]
  }
  if {[dict exists $v $o $k]} { return [dict get $v $o $k] }
  return 100%
}
foreach c {get_bd_pins get_bd_intf_pins get_bd_ports get_bd_addr_spaces get_bd_addr_segs} {
  proc $c {args} { return [lindex $args end] }
}
proc connect_bd_net {a b} { rec net $a $b }
proc connect_bd_intf_net {a b} { rec intf $a $b }
proc assign_bd_address {args} { rec addr {*}$args }
proc get_files {f} { return $f }
proc current_project {} { return p }
proc get_runs {args} { return {} }
proc make_wrapper {args} {
  set d $::ipdir_mock/project/otpu_hbm_sc.gen/sources_1/bd/otpu_hbm_sc/hdl
  file mkdir $d
  file copy -force $::wrapper_mock $d/otpu_hbm_sc_wrapper.v
}
'''


@pytest.mark.parametrize('nport,khz,sc_fault', [(8, 150000, 0), (16, 200000, 0), (8, 150000, 1)])
def test_hbm_sc_block_design_is_per_channel(tmp_path, nport, khz, sc_fault):
    """create_hbm_sc.tcl under a Vivado mock: per channel a 1 x 1 SmartConnect with every setting
    copied from the HDK's cl_axi_sc_1x1 (its cell's CONFIG, its AXI4 / AXI3 ports' CONFIG) but the
    slave side's clock, the core's; no host branch, no crossbar"""
    tclsh = shutil.which('tclsh')
    if not tclsh:
        pytest.skip('Tcl is unavailable')
    w = tmp_path / 'w.v'
    w.write_text(_sc_wrapper(nport))
    ipdir = tmp_path / 'ip'
    bd = tmp_path / 'bd/cl_axi_sc_1x1'
    bd.mkdir(parents=True)
    (bd / 'cl_axi_sc_1x1.bd').write_text('{}\n')
    script = ROOT / 'boards/f2/build/scripts/create_hbm_sc.tcl'
    report = 'foreach c $calls { puts "CALL $c" }\n'
    harness = (f'set argv [list {{{ipdir}}} {nport} xcvu47p-fsvh2892-2-e {khz * 1000} {{{tmp_path / "bd"}}}]\n'
               f'set ipdir_mock {{{ipdir}}}\nset wrapper_mock {{{w}}}\nset sc_fault {sc_fault}\n' + SC_MOCK +
               f'if {{[catch {{source {{{script}}}}} e]}} {{puts stderr $e; exit 1}}\n' + report)
    r = subprocess.run([tclsh], input=harness, text=True, capture_output=True)
    if sc_fault:                        # a SmartConnect taking 16 transactions: the build stops
        assert r.returncode != 0 and 'NUM_READ_OUTSTANDING is 16' in r.stderr
        return
    assert r.returncode == 0, r.stderr[-2000:]
    calls = [l[5:] for l in r.stdout.splitlines() if l.startswith('CALL ')]
    assert (ipdir / 'ref/cl_axi_sc_1x1.bd').is_file()          # a copy read, the HDK's untouched
    cells = [c.split()[-1] for c in calls if c.startswith('create_bd_cell')]
    assert cells == [f'sc_c{p}' for p in range(nport)]
    assert all('xilinx.com:ip:smartconnect:1.0' in c for c in calls if c.startswith('create_bd_cell'))
    for p in range(nport):
        for k, v in (('NUM_SI', 1), ('NUM_MI', 1), ('NUM_CLKS', 2), ('ADVANCED_PROPERTIES', 0)):
            assert f'set_property CONFIG.{k} {v} sc_c{p}' in calls
        assert f'set_property CONFIG.Component_Name' not in ' '.join(c for c in calls if c.endswith(f'sc_c{p}'))
        assert f'set_property CONFIG.PROTOCOL AXI4 S_N{p}' in calls and f'set_property CONFIG.ADDR_WIDTH 64 S_N{p}' in calls
        assert f'set_property CONFIG.PROTOCOL AXI3 M_H{p}' in calls and f'set_property CONFIG.DATA_WIDTH 256 M_H{p}' in calls
        assert f'set_property CONFIG.FREQ_HZ {khz * 1000} S_N{p}' in calls
        assert f'set_property CONFIG.FREQ_HZ 250000000 S_N{p}' not in calls
        assert f'set_property CONFIG.FREQ_HZ 450000000 M_H{p}' in calls
        assert f'net core_clk sc_c{p}/aclk' in calls and f'net hbm_clk sc_c{p}/aclk1' in calls
        assert f'intf S_N{p} sc_c{p}/S00_AXI' in calls and f'intf sc_c{p}/M00_AXI M_H{p}' in calls
        base = f'0x{p << 29:09X}'
        assert f'addr -offset {base} -range 0x20000000 -target_address_space S_N{p} M_H{p}/Reg -force' in calls
    # the reference's port settings as they are (reads in flight 32 / 64, bursts 64 / 16), and
    # the SmartConnect's slave checked after validation: >= 32 a direction, bursts >= 8 beats
    for p in range(nport):
        assert f'set_property CONFIG.NUM_READ_OUTSTANDING 32 S_N{p}' in calls
        assert f'set_property CONFIG.MAX_BURST_LENGTH 64 S_N{p}' in calls
        assert f'set_property CONFIG.MAX_BURST_LENGTH 16 M_H{p}' in calls
    assert 'OTPU_HBM_SC sc_c0/S00_AXI CONFIG.NUM_READ_OUTSTANDING 32' in r.stdout
    assert f'set_property CONFIG.FREQ_HZ {khz * 1000} core_clk' in calls
    assert 'set_property CONFIG.FREQ_HZ 450000000 hbm_clk' in calls
    assert not any('S_HOST' in c or 'sc_host' in c for c in calls)
    assert (ipdir / 'ip_done.txt').read_text().startswith('IP_DONE')
    inst = (ipdir / 'otpu_hbm_sc_inst.vh').read_text()
    assert f'.S_N{nport - 1}_araddr(cs_araddr[{nport - 1}])' in inst


def test_hbm_wrapper_gets_a_reset_pipeline_per_port():
    """The copied HDK wrapper drives every HBM port's ARESET_N from its own four-stage pipe
    (62b34824 placed AWS's single pipe at -0.261 ns on the recovery check to port 0); a wrapper
    whose markers moved stops the build."""
    src = ("module cl_mem_hbm_wrapper;\n  logic axi_rst_n;\n" + flow.HBM_RST_PIPE +
           "    `define CONNECT_AXI_HBM_CORE(P)\\\n      .AXI_``P``_ACLK               (axi_clk                        ),\\\n"
           "      " + flow.HBM_RST_PORT + ",\\\nendmodule\n")
    out = flow.hbm_wrapper_per_port_reset(src)
    assert out.count('lib_pipe #(.WIDTH(1), .STAGES(4))') == 2 and 'g_otpu_rst' in out
    assert '(* DONT_TOUCH = "yes" *)' in out and 'axi_rst_n_p[otpu_rp]' in out
    assert '(axi_rst_n_p[P]                 )' in out and flow.HBM_RST_PORT not in out
    assert 'logic [31:0] axi_rst_n_p;' in out and flow.HBM_RST_PIPE in out
    with pytest.raises(RuntimeError, match='per-port reset patch'):
        flow.hbm_wrapper_per_port_reset(src.replace('axi_rst_n                      )', 'axi_rst_n)'))
    with pytest.raises(RuntimeError, match='per-port reset patch'):
        flow.hbm_wrapper_per_port_reset(src + src)
