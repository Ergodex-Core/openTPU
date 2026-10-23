"""Recovery preserves the failed parent and cannot bypass route qualification."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('f2_route_opt', ROOT / 'boards/f2/route_opt.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


HBM_TARGET = 'f2-hbm:p4rl8a64c150'


def routed_parent(tmp_path, cl_kind='ddr'):
    """A finished (timing-failed) parent run: the DDR CL with its calibration sentinel in the
    console log, or the HBM CL (the target's provenance as build.py writes it, no sentinel)."""
    run = tmp_path / 'parent'
    cl = run / 'cl_opentpu'
    sources = {
        f"design/cl_opentpu{'_hbm' if cl_kind == 'hbm' else ''}.sv": '// frozen HDL\n',
        'build/constraints/cl_synth_user.xdc': '# original synth constraint\n',
        'build/constraints/cl_timing_user.tcl': '# original CDC constraint\n',
        'build/constraints/small_shell_cl_pnr_user.xdc': '# original placement constraint\n',
        'build/scripts/validate_route.tcl': '# historical validator\n',
    }
    for name, text in sources.items():
        p = cl / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    info = {'build_id': '01234567', 'rtl_parameters': {'BUILD_ID': 0x01234567},
            'sources': {n: recovery.flow.sha(cl / n) for n in sources}}
    if cl_kind == 'hbm':
        info = recovery.flow.target.load(HBM_TARGET).to_dict() | info
    (run / 'inputs.json').write_text(json.dumps(info))
    (run / 'exit_code').write_text('1\n')
    (run / 'console.log').write_text(
        'OTPU_STAGE_ROUTE_VALIDATION_FAILED\n' if cl_kind == 'hbm' else
        "OTPU_DDR_CALIBRATION_INIT_2C=256'h" + '1' * 64 + '\n')
    report = cl / 'build/reports/cl_opentpu.01234567.post_route_timing.rpt'
    report.parent.mkdir()
    report.write_text('timing failed; test fixture\n')
    dcp = cl / 'build/checkpoints/cl_opentpu.01234567.post_route.VIOLATED.dcp'
    dcp.parent.mkdir()
    dcp.write_bytes(b'failed routed test checkpoint')
    return run, dcp


@pytest.mark.parametrize('cl_kind', ['ddr', 'hbm'])
def test_recovery_copies_frozen_sources_and_keeps_parent_checkpoint_out_of_outputs(tmp_path, cl_kind):
    run, dcp = routed_parent(tmp_path, cl_kind)
    before = {str(p.relative_to(run)): recovery.flow.sha(p) for p in run.rglob('*') if p.is_file()}
    out = tmp_path / 'recovery'
    cl, info = recovery.prepare(out, run, dcp, recovery.flow.sha(dcp))
    assert info['parent_implementation']['exit_code'] == 1
    # the HBM CL has no DIMM: the drivers get none_hbm for the sentinel to expect
    assert info['parent_implementation']['cl'] == cl_kind
    assert info['parent_implementation']['ddr_init_2c'] == (
        'none_hbm' if cl_kind == 'hbm' else "256'h" + '1' * 64)
    assert info['route_recovery']['constraints_resourced'] is False
    for name, digest in info['parent_implementation']['source_inventory'].items():
        if name.startswith(('design/', 'build/constraints/')):
            assert recovery.flow.sha(cl / name) == digest
    inherited = Path(info['route_recovery']['parent_checkpoint_copy'])
    assert inherited.read_bytes() == dcp.read_bytes()
    assert not list((cl / 'build/checkpoints').iterdir())
    assert (cl / 'build/scripts/validate_route.tcl').read_bytes() != (
        run / 'cl_opentpu/build/scripts/validate_route.tcl').read_bytes()
    assert before == {str(p.relative_to(run)): recovery.flow.sha(p)
                      for p in run.rglob('*') if p.is_file()}
    # Successful recovery uses the existing package-only verification interface.
    result = cl / 'build/checkpoints/cl_opentpu.01234567.post_route.dcp'
    result.write_bytes(b'qualified recovery test bytes')
    (out / 'exit_code').write_text('0\n')
    (out / 'console.log').write_text('OTPU_STAGE_ROUTE_VALIDATION_PASSED\n')
    assert recovery.flow.load_routed_run(out, recovery.flow.sha(result))[1] == cl
    with pytest.raises(RuntimeError, match='already exists'):
        recovery.prepare(out, run, dcp, recovery.flow.sha(dcp))


@pytest.mark.parametrize('fault', ['digest', 'unfinished', 'report', 'source', 'firmware', 'checkpoint',
                                   'hbm_sentinel', 'hbm_source'])
def test_recovery_rejects_unverified_parent(tmp_path, fault):
    run, dcp = routed_parent(tmp_path, 'hbm' if fault.startswith('hbm') else 'ddr')
    digest = recovery.flow.sha(dcp)
    if fault == 'hbm_sentinel':         # an HBM run's log with a DDR sentinel: another CL's
        with (run / 'console.log').open('a') as f:
            f.write("OTPU_DDR_CALIBRATION_INIT_2C=256'h" + '1' * 64 + '\n')
    elif fault == 'hbm_source':
        (run / 'cl_opentpu/design/cl_opentpu_hbm.sv').write_text('changed\n')
    if fault == 'digest':
        digest = '0' * 64
    elif fault == 'unfinished':
        (run / 'exit_code').unlink()
    elif fault == 'report':
        next((run / 'cl_opentpu/build/reports').iterdir()).unlink()
    elif fault == 'source':
        (run / 'cl_opentpu/design/cl_opentpu.sv').write_text('changed\n')
    elif fault == 'firmware':
        (run / 'console.log').write_text("OTPU_DDR_CALIBRATION_INIT_2C=256'h" + '0' * 64 + '\n')
    elif fault == 'checkpoint':
        dcp = dcp.with_name('unvalidated.dcp')
    with pytest.raises((RuntimeError, OSError)):
        recovery.prepare(tmp_path / 'recovery', run, dcp, digest)


@pytest.mark.parametrize('cl_kind,fault', [
    ('ddr', 'none'), ('ddr', 'reroute'), ('ddr', 'qualification'), ('ddr', 'firmware'), ('ddr', 'hash'),
    ('ddr', 'kind'),
    # the HBM parent (none_hbm): no sentinel read; the checkpoint must hold the SmartConnect fabric
    ('hbm', 'none'), ('hbm', 'reroute'), ('hbm', 'qualification'), ('hbm', 'hash'), ('hbm', 'kind')])
def test_recovery_tcl_persists_only_reopened_qualified_candidate(tmp_path, cl_kind, fault):
    tclsh = shutil.which('tclsh')
    if not tclsh or not shutil.which('sha256sum'):
        pytest.skip('Tcl or SHA256 utility unavailable')
    scripts = tmp_path / 'build/scripts'
    scripts.mkdir(parents=True)
    (tmp_path / 'build/reports').mkdir()
    checkpoints = tmp_path / 'build/checkpoints'
    checkpoints.mkdir()
    for name in ('opt_route.tcl', 'inspect_ddr_init.tcl'):
        shutil.copy2(ROOT / 'boards/f2/build/scripts' / name, scripts / name)
    (scripts / 'validate_route.tcl').write_text(
        'if {$::fault eq "qualification"} {error "qualification failed"}\n'
        'puts "QUALIFICATION_RUN"\n')
    parent = tmp_path / 'parent.dcp'
    parent.write_bytes(b'immutable parent fixture')
    digest = recovery.flow.sha(parent)
    harness = r'''
set opens 0
set routes 0
proc set_param {args} {}
proc open_checkpoint {path} {incr ::opens}
proc close_design {} {}
proc get_cells {args} {
  # the SmartConnect fabric: the HBM CL's marker (fault kind: the other CL's checkpoint)
  if {[lindex $args 0] eq "-quiet"} {
    set hbm [expr {($::cl eq "hbm") != ($::fault eq "kind")}]
    if {$hbm} {return WRAPPER/CL/u_sc} else {return {}}
  }
  if {$::cl eq "hbm"} {error "inspect_ddr_init on the HBM CL"}
  return {SH_DDR/mcs0/inst/lmb_bram_I/ramloop[0].ram.r/primitive}
}
proc get_property {property object} {
  if {$property eq "NAME"} {return $object}
  set hex [expr {$::fault eq "firmware" && $::opens == 2 ? 2 : 1}]
  return "256'h[string repeat $hex 64]"
}
proc report_timing_summary {args} {}
proc report_timing {args} {}
proc phys_opt_design {args} {
  if {$args ne "-directive Explore"} {error "unexpected optimization"}
}
proc get_nets {args} {
  if {$::fault eq "reroute" && !$::routes} {return incomplete}
  return {}
}
proc route_design {args} {
  if {$args ne "-directive Explore -tns_cleanup"} {error "unexpected reroute"}
  incr ::routes
}
proc write_checkpoint {path} {set f [open $path w]; puts $f candidate; close $f}
proc write_debug_probes {args} {set f [open [lindex $args end] w]; close $f}
'''
    expected = '0' * 64 if fault == 'hash' else digest
    init = 'none_hbm' if cl_kind == 'hbm' else "256'h" + '1' * 64
    harness = (f'set fault {fault}\nset cl {cl_kind}\nset env(BUILD_TAG) 01234567\n'
               f'set env(OTPU_PARENT_DCP) {{{parent}}}\n'
               f'set env(OTPU_PARENT_SHA256) {expected}\n'
               f'set env(OTPU_EXPECTED_DDR_INIT) {{{init}}}\n' + harness +
               f'if {{[catch {{source {{{scripts / "opt_route.tcl"}}}}} err]}} '
               '{puts stderr $err; exit 1}\nputs "routes=$routes opens=$opens"\n')
    result = subprocess.run([tclsh], input=harness, text=True, capture_output=True)
    final = checkpoints / 'cl_opentpu.01234567.post_route.dcp'
    assert recovery.flow.sha(parent) == digest
    if fault in ('none', 'reroute'):
        assert result.returncode == 0, result.stderr
        assert final.read_text() == 'candidate\n'
        assert 'QUALIFICATION_RUN' in result.stdout
        assert f'routes={int(fault == "reroute")} opens=2' in result.stdout
    else:
        assert result.returncode != 0
        assert not final.exists()
        if fault == 'kind':
            assert "differs from the parent run's target" in result.stderr



# opt_route.tcl's post-route passes (route_opt.py --phys-opt-directive): the directives in order,
# the setup slack read between passes, stopping at the first pass that meets timing
@pytest.mark.parametrize('slacks,ran', [
    (['-0.073', '0.004'], ['Explore', 'AggressiveExplore']),
    (['-0.073', '-0.020', '-0.010'], ['Explore', 'AggressiveExplore', 'AggressiveFanoutOpt']),
    (['0.001'], ['Explore'])])
def test_recovery_runs_the_post_route_passes_in_order_until_timing_is_met(tmp_path, slacks, ran):
    tclsh = shutil.which('tclsh')
    if not tclsh or not shutil.which('sha256sum'):
        pytest.skip('Tcl or SHA256 utility unavailable')
    scripts = tmp_path / 'build/scripts'
    scripts.mkdir(parents=True)
    (tmp_path / 'build/reports').mkdir()
    (tmp_path / 'build/checkpoints').mkdir()
    shutil.copy2(ROOT / 'boards/f2/build/scripts/opt_route.tcl', scripts / 'opt_route.tcl')
    (scripts / 'validate_route.tcl').write_text('puts "QUALIFICATION_RUN"\n')
    parent = tmp_path / 'parent.dcp'
    parent.write_bytes(b'immutable parent fixture')
    harness = f"""
set env(BUILD_TAG) 01234567
set env(OTPU_PARENT_DCP) {{{parent}}}
set env(OTPU_PARENT_SHA256) {recovery.flow.sha(parent)}
set env(OTPU_EXPECTED_DDR_INIT) none_hbm
set env(OTPU_PHYS_OPT_DIRECTIVES) {{Explore AggressiveExplore AggressiveFanoutOpt}}
set slacks {{{' '.join(slacks)}}}
set passes {{}}
proc set_param {{args}} {{}}
proc open_checkpoint {{path}} {{}}
proc close_design {{}} {{}}
proc get_cells {{args}} {{ return WRAPPER/CL/u_sc }}
proc report_timing_summary {{args}} {{}}
proc report_timing {{args}} {{}}
proc phys_opt_design {{args}} {{ lappend ::passes [lindex $args 1] }}
proc get_timing_paths {{args}} {{ return worst }}
proc get_property {{property object}} {{
  if {{$property ne "SLACK"}} {{ error "unexpected property $property" }}
  return [lindex $::slacks [expr {{[llength $::passes] - 1}}]]
}}
proc get_nets {{args}} {{ return {{}} }}
proc route_design {{args}} {{ error "unexpected reroute" }}
proc write_checkpoint {{path}} {{ set f [open $path w]; puts $f candidate; close $f }}
proc write_debug_probes {{args}} {{ set f [open [lindex $args end] w]; close $f }}
if {{[catch {{source {{{scripts / 'opt_route.tcl'}}}}} err]}} {{ puts stderr $err; exit 1 }}
puts "PASSES $passes"
"""
    result = subprocess.run([tclsh], input=harness, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert f'PASSES {" ".join(ran)}' in result.stdout
    assert 'QUALIFICATION_RUN' in result.stdout


# A recovery run whose candidate met timing but was refused by qualification (DRC AVAL-168: the
# core MMCM's M 4.8) seeds another recovery: its route_opt.unvalidated.dcp is a parent (with its
# route_validation.txt as the evidence), never a package; any other run's candidate is refused
@pytest.mark.parametrize('case', ['recovery', 'not_recovery', 'no_validation'])
def test_recovery_accepts_a_previous_recovery_candidate_as_parent(tmp_path, case):
    run, dcp = routed_parent(tmp_path, 'hbm')
    first = tmp_path / 'recovery1'
    recovery.prepare(first, run, dcp, recovery.flow.sha(dcp))
    seed = first if case != 'not_recovery' else run
    (seed / 'exit_code').write_text('1\n')
    (seed / 'console.log').write_text('openTPU: routed design failed qualification\n')
    cand = seed / 'cl_opentpu/build/checkpoints/cl_opentpu.01234567.route_opt.unvalidated.dcp'
    cand.write_bytes(b'timing-met candidate refused by DRC AVAL-168')
    if case != 'no_validation':
        (seed / 'cl_opentpu/build/reports/cl_opentpu.01234567.route_validation.txt').write_text(
            'setup_failed=0\nhold_failed=0\ndrc_blockers=1\n')
    second = tmp_path / 'recovery2'
    if case != 'recovery':
        with pytest.raises(RuntimeError):
            recovery.prepare(second, seed, cand, recovery.flow.sha(cand), mmcm=(5, 24.0))
        return
    cl, info = recovery.prepare(second, seed, cand, recovery.flow.sha(cand), mmcm=(5, 24.0))
    assert info['parent_implementation']['run'] == str(first)
    assert info['route_recovery']['mmcm'] == {'divclk': 5, 'clkfbout_mult_f': 24.0}
    assert Path(info['route_recovery']['parent_checkpoint_copy']).read_bytes() == cand.read_bytes()


# opt_route.tcl with --mmcm: the core MMCM's DIVCLK_DIVIDE / CLKFBOUT_MULT_F set before the
# post-route passes, only when the VCO (M / D) stays the same
@pytest.mark.parametrize('mult,ok', [('24.000', True), ('25.000', False)])
def test_recovery_tcl_sets_a_legal_mmcm_with_the_same_vco(tmp_path, mult, ok):
    tclsh = shutil.which('tclsh')
    if not tclsh or not shutil.which('sha256sum'):
        pytest.skip('Tcl or SHA256 utility unavailable')
    scripts = tmp_path / 'build/scripts'
    scripts.mkdir(parents=True)
    (tmp_path / 'build/reports').mkdir()
    (tmp_path / 'build/checkpoints').mkdir()
    shutil.copy2(ROOT / 'boards/f2/build/scripts/opt_route.tcl', scripts / 'opt_route.tcl')
    (scripts / 'validate_route.tcl').write_text('puts "QUALIFICATION_RUN"\n')
    parent = tmp_path / 'parent.dcp'
    parent.write_bytes(b'immutable parent fixture')
    harness = f"""
set env(BUILD_TAG) 01234567
set env(OTPU_PARENT_DCP) {{{parent}}}
set env(OTPU_PARENT_SHA256) {recovery.flow.sha(parent)}
set env(OTPU_EXPECTED_DDR_INIT) none_hbm
set env(OTPU_MMCM_DIVCLK) 5
set env(OTPU_MMCM_MULT) {mult}
set log {{}}
proc set_param {{args}} {{}}
proc open_checkpoint {{path}} {{}}
proc close_design {{}} {{}}
proc get_cells {{args}} {{ return [lindex $args end] }}
proc get_property {{property object}} {{
  if {{$property eq "DIVCLK_DIVIDE"}} {{ return 1 }}
  if {{$property eq "CLKFBOUT_MULT_F"}} {{ return 4.800 }}
  error "unexpected property $property"
}}
proc set_property {{args}} {{ lappend ::log "set_property $args" }}
proc report_timing_summary {{args}} {{}}
proc report_timing {{args}} {{}}
proc phys_opt_design {{args}} {{ lappend ::log "phys_opt_design $args" }}
proc get_nets {{args}} {{ return {{}} }}
proc route_design {{args}} {{ error "unexpected reroute" }}
proc write_checkpoint {{path}} {{ set f [open $path w]; puts $f candidate; close $f }}
proc write_debug_probes {{args}} {{ set f [open [lindex $args end] w]; close $f }}
if {{[catch {{source {{{scripts / 'opt_route.tcl'}}}}} err]}} {{ puts "STOPPED $err" }}
foreach l $log {{ puts "LOG $l" }}
"""
    result = subprocess.run([tclsh], input=harness, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    log = [l[4:] for l in result.stdout.splitlines() if l.startswith('LOG ')]
    if ok:
        assert log == ['set_property -dict {DIVCLK_DIVIDE 5 CLKFBOUT_MULT_F 24.000} WRAPPER/CL/u_mmcm',
                       'phys_opt_design -directive Explore']
        assert 'OTPU_MMCM_SET DIVCLK_DIVIDE 1 -> 5 CLKFBOUT_MULT_F 4.800 -> 24.000' in result.stdout
        assert 'QUALIFICATION_RUN' in result.stdout
    else:
        assert 'STOPPED openTPU: MMCM M/D 25.000/5 changes the VCO' in result.stdout
        assert log == []

def test_validation_only_preparation_preserves_parent_design_and_selects_read_only_driver(tmp_path):
    run, dcp = routed_parent(tmp_path)
    cl, info = recovery.prepare(tmp_path / 'validation', run, dcp, recovery.flow.sha(dcp),
                                validate_only=True)
    mode = info['route_recovery']
    assert mode['mode'] == 'validate_only'
    assert mode['directive'] is None
    assert mode['constraints_resourced'] is False
    assert mode['checkpoint_written'] is False
    assert mode['output_must_match_parent'] is True
    assert Path(mode['parent_checkpoint_copy']).read_bytes() == dcp.read_bytes()
    assert (cl / 'build/scripts/validate_only.tcl').is_file()
    assert not (cl / 'build/scripts/opt_route.tcl').exists()
    for name, digest in info['parent_implementation']['source_inventory'].items():
        if name.startswith(('design/', 'build/constraints/')):
            assert recovery.flow.sha(cl / name) == digest


@pytest.mark.parametrize('cl_kind,fault', [('ddr', f) for f in (
    'none', 'qualification', 'firmware', 'hash', 'missing_marker', 'bad_marker', 'report_failure',
    'parent_changed', 'copy_corruption', 'kind')] + [('hbm', f) for f in (
    'none', 'qualification', 'hash', 'missing_marker', 'bad_marker', 'report_failure',
    'parent_changed', 'copy_corruption', 'kind')])
def test_validate_only_tcl_never_mutates_design_and_copies_only_after_all_checks(tmp_path, cl_kind, fault):
    tclsh = shutil.which('tclsh')
    if not tclsh or not shutil.which('sha256sum'):
        pytest.skip('Tcl or SHA256 utility unavailable')
    scripts = tmp_path / 'build/scripts'
    scripts.mkdir(parents=True)
    (tmp_path / 'build/reports').mkdir()
    checkpoints = tmp_path / 'build/checkpoints'
    checkpoints.mkdir()
    for name in ('validate_only.tcl', 'inspect_ddr_init.tcl'):
        shutil.copy2(ROOT / 'boards/f2/build/scripts' / name, scripts / name)
    (scripts / 'validate_route.tcl').write_text(r'''
if {$::fault eq "qualification"} {error "qualification gate failed"}
if {$::fault ne "missing_marker"} {
    set f [open ${checkpoints_dir}/${CL}.${TAG}.route_validation.passed w]
    if {$::fault eq "bad_marker"} {puts $f wrong} else {puts $f OTPU_STAGE_ROUTE_VALIDATION_PASSED}
    close $f
}
puts OTPU_STAGE_ROUTE_VALIDATION_PASSED
''')
    parent = tmp_path / 'parent.dcp'
    original = b'\x00unchanged routed checkpoint\xff\n'
    parent.write_bytes(original)
    digest = recovery.flow.sha(parent)
    harness = r'''
set opens 0
set closes 0
proc set_param {args} {}
proc open_checkpoint {path} {incr ::opens}
proc close_design {} {incr ::closes}
proc get_cells {args} {
    if {[lindex $args 0] eq "-quiet"} {
        set hbm [expr {($::cl eq "hbm") != ($::fault eq "kind")}]
        if {$hbm} {return WRAPPER/CL/u_sc} else {return {}}
    }
    if {$::cl eq "hbm"} {error "inspect_ddr_init on the HBM CL"}
    return {SH_DDR/mcs0/inst/lmb_bram_I/ramloop[0].ram.r/primitive}
}
proc get_property {property object} {
    if {$property eq "NAME"} {return $object}
    set digit [expr {$::fault eq "firmware" ? 2 : 1}]
    return "256'h[string repeat $digit 64]"
}
proc report_timing {args} {
    if {$::fault eq "report_failure"} {error "timing report failed"}
    if {$::fault eq "parent_changed"} {
        set f [open $::env(OTPU_PARENT_DCP) a]; puts $f changed; close $f
    }
    set f [open [lindex $args end] w]; puts $f "qualified timing report"; close $f
}
foreach command {phys_opt_design route_design opt_design link_design read_xdc read_verilog
                 synth_design write_checkpoint create_clock create_generated_clock
                 set_max_delay set_false_path set_clock_groups set_bus_skew refresh_meminit} {
    proc $command {args} {error "FORBIDDEN design mutation command invoked"}
}
rename file original_file
proc file {args} {
    set result [original_file {*}$args]
    if {$::fault eq "copy_corruption" && [lindex $args 0] eq "copy"} {
        set f [open [lindex $args end] a]; puts $f corrupted; close $f
    }
    return $result
}
'''
    expected = '0' * 64 if fault == 'hash' else digest
    init = 'none_hbm' if cl_kind == 'hbm' else "256'h" + '1' * 64
    harness = (f'set fault {fault}\nset cl {cl_kind}\nset env(BUILD_TAG) 01234567\n'
               f'set env(OTPU_PARENT_DCP) {{{parent}}}\n'
               f'set env(OTPU_PARENT_SHA256) {expected}\n'
               f'set env(OTPU_EXPECTED_DDR_INIT) {{{init}}}\n' + harness +
               f'set rc [catch {{source {{{scripts / "validate_only.tcl"}}}}} err]\n'
               'puts "opens=$opens closes=$closes"\n'
               'if {$rc} {puts stderr $err; exit 1}\n')
    result = subprocess.run([tclsh], input=harness, text=True, capture_output=True, timeout=10)
    final = checkpoints / 'cl_opentpu.01234567.post_route.dcp'
    marker = checkpoints / 'cl_opentpu.01234567.route_validation.passed'
    if fault != 'parent_changed':
        assert parent.read_bytes() == original
    if fault == 'none':
        assert result.returncode == 0, result.stderr
        assert final.read_bytes() == original
        assert recovery.flow.sha(final) == digest
        assert marker.read_text().strip() == 'OTPU_STAGE_ROUTE_VALIDATION_PASSED'
        assert 'opens=1 closes=1' in result.stdout
        assert 'OTPU_ROUTE_VALIDATE_ONLY_COMPLETED' in result.stdout
    else:
        assert result.returncode != 0
        assert not final.exists()
        assert not marker.exists()
        assert 'FORBIDDEN' not in result.stderr
        if fault != 'hash':
            assert 'opens=1 closes=1' in result.stdout


@pytest.mark.parametrize('cl_kind,fault', [('ddr', f) for f in (
    'none', 'changed_copy', 'changed_source', 'changed_parent', 'missing_marker', 'vivado_failure')]
    + [('hbm', 'none'), ('hbm', 'changed_copy'), ('hbm', 'vivado_failure'), ('hbm', 'other_target')])
def test_validation_only_driver_requires_identical_bytes_and_invalidates_failed_outputs(tmp_path, monkeypatch, cl_kind, fault):
    run, dcp = routed_parent(tmp_path, cl_kind)
    info_path = run / 'inputs.json'
    info = json.loads(info_path.read_text())
    if cl_kind == 'ddr':                # (the HBM fixture carries its target's provenance)
        info.update({key: recovery.flow.target.load('f2').to_dict()[key] for key in ('platform', 'tech')})
    if fault == 'other_target':         # a parent whose platform is not its named target's
        info['platform'] = dict(info['platform'], core_mhz=200.0)
    info_path.write_text(json.dumps(info))
    digest = recovery.flow.sha(dcp)
    out = tmp_path / 'validation'
    monkeypatch.setattr(sys, 'argv', ['route_opt.py', '--hdk', str(tmp_path / 'hdk'),
        '--out', str(out), '--from-run', str(run), '--checkpoint', str(dcp),
        '--checkpoint-sha256', digest, '--validate-only'])
    checked = []
    monkeypatch.setattr(recovery.flow, 'check_hdk', lambda hdk, t, **kw: (checked.append(t), tmp_path / 'shell')[1])
    monkeypatch.setattr(recovery.flow, 'command', lambda *a: 'Vivado v2025.2 (64-bit)')

    def execute(command, *, cwd, env, stdout, stderr):
        assert command[command.index('-source') + 1] == 'validate_only.tcl'
        assert 'opt_route.tcl' not in command
        # the HBM parent: the HDK checked for its own target, the drivers told no sentinel
        assert env['OTPU_EXPECTED_DDR_INIT'] == ('none_hbm' if cl_kind == 'hbm' else "256'h" + '1' * 64)
        assert [f'{t.name}:{t.variant}' if t.variant else t.name for t in checked] == \
            [HBM_TARGET if cl_kind == 'hbm' else 'f2']
        checkpoints = cwd.parent / 'checkpoints'
        routed = checkpoints / 'cl_opentpu.01234567.post_route.dcp'
        routed.write_bytes(b'changed' if fault == 'changed_copy' else Path(env['OTPU_PARENT_DCP']).read_bytes())
        if fault != 'missing_marker':
            (checkpoints / 'cl_opentpu.01234567.route_validation.passed').write_text('OTPU_STAGE_ROUTE_VALIDATION_PASSED\n')
        (cwd.parent / 'reports/cl_opentpu.01234567.post_route_timing.rpt').write_text('qualified timing\n')
        stdout.write(('' if cl_kind == 'hbm' else "OTPU_DDR_CALIBRATION_INIT_2C=256'h" + '1' * 64 + '\n')
                     + 'OTPU_STAGE_ROUTE_VALIDATION_PASSED\n')
        if fault == 'changed_source':
            (cwd.parent.parent / 'design/cl_opentpu.sv').write_text('unexpected edit\n')
        if fault == 'changed_parent':
            with (run / 'console.log').open('a') as stream:
                stream.write('unexpected parent change\n')
        return subprocess.CompletedProcess(command, 1 if fault == 'vivado_failure' else 0)

    monkeypatch.setattr(recovery.subprocess, 'run', execute)
    if fault == 'none':
        recovery.main()
        outputs = json.loads((out / 'outputs.json').read_text())
        assert outputs['flow'] == 'RouteValidate'
        artifacts = outputs['artifacts']
        routed_name = 'cl_opentpu/build/checkpoints/cl_opentpu.01234567.post_route.dcp'
        assert artifacts[routed_name]['sha256'] == digest
        assert (out / routed_name).read_bytes() == dcp.read_bytes()
        assert (out / 'exit_code').read_text() == '0\n'
        assert recovery.flow.load_routed_run(out, digest)[0]['route_recovery']['mode'] == 'validate_only'
    elif fault == 'other_target':       # refused before any output exists
        with pytest.raises(RuntimeError, match='differs from the pinned F2 target'):
            recovery.main()
        assert not out.exists() and not checked
    else:
        with pytest.raises((RuntimeError, OSError)):
            recovery.main()
        assert (out / 'exit_code').read_text() == '1\n'
        assert not (out / 'outputs.json').exists()
        assert not list((out / 'cl_opentpu/build/checkpoints').iterdir())
    assert recovery.flow.sha(dcp) == digest
