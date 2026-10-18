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
                                 'gray', 'crossing', 'firmware'])
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
  set count [expr {$::fault eq "gray" ? 13 : 14}]
  for {set i 0} {$i < $count} {incr i} {puts $f "$i 100 u_sys/g_ch"}
  puts $f "2. Bus Skew Report Per Constraint"
  close $f
}
foreach command {report_drc report_methodology report_cdc report_exceptions
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
  return "Requirement: 4.000ns (MaxDelay Path 4.000ns)"
}
proc get_cells {args} {return {SH_DDR/mcs0/inst/lmb_bram_I/ramloop[0].ram.r/primitive}}
proc get_property {prop obj} {
  if {$prop eq "NAME"} {return $obj}
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
    if fault == 'none':
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


@pytest.mark.parametrize('name,S,nch,dram_log2,portb',
                         [('f2-hbm', 1, 8, 32, 1), ('f2-hbm:s2', 2, 8, 32, 1),
                          ('f2-hbm:s4', 4, 8, 32, 1), ('f2-hbm16', 1, 16, 33, 1),
                          ('f2-hbm:p2', 1, 8, 32, 2), ('f2-hbm16:p4', 1, 16, 33, 4)])
def test_hbm_snapshot_takes_the_hbm_top_scripts_and_geometry(tmp_path, name, S, nch, dram_log2,
                                                              portb):
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
    assert flow.is_hbm(t) and t.core_khz == 200000 and t.uarch['TMEM_PMP'] == 1
    if S == 1:                          # f2's programs and T2 (docs/ternary.md)
        assert T.config(t) == dataclasses.replace(T.config('f2'), T2=True, PORTB=portb)
        assert T.rtl_params(t)['ARCH_HASH'] == T.arch_hash(T.config(t)) != 0x4FFD1E17
        if portb > 1:                   # in the hash only when wide (docs/wide_port.md)
            assert T.arch_hash(T.config(t)) != T.arch_hash(T.config(name.split(':')[0]))
    out = tmp_path / 'run'
    cl = flow.prepare(out, shell, t, '89abcdef', wrapper)
    d = cl / 'design'
    assert not (d / 'cl_opentpu.sv').exists() and (d / 'cl_opentpu_hbm.sv').is_file()
    assert (d / 'cl_mem_hbm_wrapper.sv').read_text() == wrapper.read_text()
    assert (d / 'opentpu_geometry.vh').read_text() == (
        f'localparam int S = {S};\nlocalparam int NCH = {nch};\n')
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
    assert params['CORE_KHZ'] == 200000 and 'CH_BYTES' not in params and 'DDR_MTS' not in params
    vh = (d / 'opentpu_params.vh').read_text()
    assert f".NCH(32'h{nch:x})" in vh and ".TMEM_PMP(32'h1)" in vh
    assert params['PORTB'] == portb and f".PORTB(32'h{portb:x})" in vh
    srcs = (d / 'sources.tcl').read_text()
    assert 'rtl/boards/f2/otpu_f2_hbm_sys.sv' in srcs and 'otpu_f2_sys.sv' not in srcs
    for f in T.sources(t):
        assert (d / 'rtl' / f).read_bytes() == (ROOT / 'rtl' / f).read_bytes()


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
           CLOCKREGION_X0Y8:CLOCKREGION_X7Y11 {s2a s2b shell}
           CLOCKREGION_X2Y9:CLOCKREGION_X5Y10 {s2b} P0 {s0a s0b s0c} P1 {s1a s1b} P2 {s2a s2b}}
set ranges {{pblock_CL P0} {pblock_CL P1} {pblock_CL P2}}
proc create_pblock {name} { lappend ::pblocks $name }
proc get_pblocks {name} { if {$name ni $::pblocks} { error "no pblock $name" }; return $name }
proc get_property {prop name} {
  foreach p $::props { if {[lindex $p 0] eq $name && [lindex $p 1] eq $prop} { return [lindex $p 2] } }
  error "no $prop on $name"
}
proc get_sites {flag name} {
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
    assert 'OTPU_FLOORPLAN ports=4' in out and 'OTPU_FLOORPLAN_APPLIED pblocks=6' in out
    assert 'PBLOCKS pblock_CL pb_otpu_hbm pb_otpu_mid pb_otpu_pcis pb_otpu_xs pb_otpu_board pb_otpu_tmem' in out
    ranges = {}
    for line in out.splitlines():
        if line.startswith('RANGE ') and not line.startswith('RANGE pblock_CL'):
            _, pb, rng = line.split(' ', 2)
            ranges.setdefault(pb, []).append(rng)
    # whole clock regions, one SLR per pblock (Y0-3 SLR0, Y4-7 SLR1, Y8-11 SLR2)
    assert ranges == {
        'pb_otpu_hbm': ['CLOCKREGION_X0Y0:CLOCKREGION_X3Y3', 'CLOCKREGION_X4Y0:CLOCKREGION_X7Y0'],
        'pb_otpu_mid': ['CLOCKREGION_X0Y4:CLOCKREGION_X3Y7'],
        'pb_otpu_pcis': ['CLOCKREGION_X5Y5:CLOCKREGION_X5Y6'],
        'pb_otpu_xs': ['CLOCKREGION_X4Y3:CLOCKREGION_X4Y3'],
        'pb_otpu_board': ['CLOCKREGION_X0Y8:CLOCKREGION_X7Y11'],
        'pb_otpu_tmem': ['CLOCKREGION_X2Y9:CLOCKREGION_X5Y10']}
    cells = {}
    for line in out.splitlines():
        if line.startswith('CELL '):
            _, pb, cell = line.split(' ', 2)
            cells.setdefault(pb, []).append(cell)
    hbm, mid, pcis, xs, board, tmem = (cells[k] for k in ('pb_otpu_hbm', 'pb_otpu_mid', 'pb_otpu_pcis', 'pb_otpu_xs', 'pb_otpu_board', 'pb_otpu_tmem'))
    # the SLR0 ends of the PCIS link stages (round 6) in SLR0's top corner, apart from the SLR1 ends
    assert sorted(xs) == [f'WRAPPER/CL/u_sys/u_mem/u_host/u_xs_{k}' for k in ('ar', 'aw', 'b', 'r', 'w')]
    assert not (set(xs) & set(pcis))
    # the PCIS registers (round 4) beside the shell's DMA column, SLR1, apart from u_host (SLR0)
    assert sorted(pcis) == sorted(f'WRAPPER/CL/u_sys/u_mem/u_host/u_pcis_{k}' for k in
                                  ('ar', 'aw', 'b', 'r', 'w', 'aw_tx', 'w_tx', 'ar_tx', 'b_rx', 'r_rx'))
    assert all(c.startswith('WRAPPER/CL/u_sys/') for pb in cells.values() for c in pb)
    assert len([c for c in hbm if '.u_nch' in c]) == 4 and 'WRAPPER/CL/u_sys/u_mem/u_host' in hbm
    assert 'WRAPPER/CL/u_sys/u_board' in board and 'WRAPPER/CL/u_sys/u_ocl' in hbm
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
    for pb in ('pb_otpu_hbm', 'pb_otpu_mid', 'pb_otpu_pcis', 'pb_otpu_xs', 'pb_otpu_board', 'pb_otpu_tmem'):
        assert props[(pb, 'CONTAIN_ROUTING')] == '0' and props[(pb, 'IS_SOFT')] == '0'
        assert props[(pb, 'SNAPPING_MODE')] == 'OFF'
        assert props[(pb, 'PARENT')] == ('pb_otpu_board' if pb == 'pb_otpu_tmem' else 'pblock_CL')
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
