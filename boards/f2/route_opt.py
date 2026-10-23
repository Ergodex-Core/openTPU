#!/usr/bin/env python3
"""Optimize or only revalidate a pinned routed F2 checkpoint in a fresh run."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('f2_build_flow', HERE / 'build.py')
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)

RECOVERY_SCRIPTS = ('opt_route.tcl', 'validate_route.tcl',
                    'otpu_tmem_mode.tcl',
                    'check_rule_severity.tcl', 'inspect_ddr_init.tcl',
                    'classify_methodology.tcl', 'check_shell_baseline.tcl',
                    'shell_baseline_2025_2.tcl', 'check_scoped_clocks.tcl')
VALIDATION_SCRIPTS = ('validate_only.tcl', *RECOVERY_SCRIPTS[1:])
# post-route phys_opt_design directives opt_route.tcl may run, in the order given
PHYS_OPT_DIRECTIVES = ('Explore', 'AggressiveExplore', 'AggressiveFanoutOpt',
                       'AlternateReplication', 'Default')


def cl_kind(info: dict) -> str:
    """The parent's CL from its frozen target: 'hbm' (platform mem "hbm": no DIMM, the
    SmartConnect fabric; boards/f2/design/cl_opentpu_hbm.sv) or 'ddr'."""
    return 'hbm' if info.get('platform', {}).get('mem') == 'hbm' else 'ddr'


def parent_target(info: dict):
    """The frozen target of a parent run (its inputs.json: build.py's provenance holds the
    target's name and variant), an F2 one."""
    name = info.get('name', 'f2')
    variant = info.get('variant', '')
    t = flow.target.load(f'{name}:{variant}' if variant else name)
    if t.platform.get('board') != 'f2':
        raise RuntimeError(f'Parent target {name} is not an F2 target')
    return t


def load_parent(run: Path, checkpoint: Path, expected_sha: str) -> tuple[dict, dict]:
    """Require a finished run and immutable source/DCP evidence, even if timing failed."""
    if not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha):
        raise RuntimeError('--checkpoint-sha256 must be a SHA256 digest')
    info = json.loads((run / 'inputs.json').read_text())
    build_id = info['build_id']
    if not re.fullmatch(r'[0-9a-fA-F]{8}', build_id):
        raise RuntimeError('Invalid parent build ID')
    if info.get('rtl_parameters', {}).get('BUILD_ID') != int(build_id, 16):
        raise RuntimeError('Parent build ID differs from frozen RTL parameters')
    allowed = {run / f'cl_opentpu/build/checkpoints/cl_opentpu.{build_id}.post_route{suffix}.dcp'
               for suffix in ('', '.VIOLATED')}
    # a previous recovery run's candidate (optimized and written, then refused by qualification:
    # its route_validation.txt says why) may seed another recovery; it is never packaged itself
    recovery_parent = bool(info.get('route_recovery'))
    if recovery_parent:
        allowed.add(run / f'cl_opentpu/build/checkpoints/cl_opentpu.{build_id}.route_opt.unvalidated.dcp')
    if checkpoint not in allowed:
        raise RuntimeError('Parent checkpoint must be this run\'s final routed or VIOLATED DCP'
                           ' (or a recovery run\'s unvalidated candidate)')
    # The normal launcher writes exit_code only after Vivado returns. A failed
    # qualification is a permissible parent, never permission to package it.
    exit_code = (run / 'exit_code').read_text().strip()
    if not re.fullmatch(r'[0-9]+', exit_code):
        raise RuntimeError('Parent run has no completed exit status')
    timing = run / f'cl_opentpu/build/reports/cl_opentpu.{build_id}.post_route_timing.rpt'
    if recovery_parent and checkpoint.name.endswith('.route_opt.unvalidated.dcp'):
        timing = timing.with_name(f'cl_opentpu.{build_id}.route_validation.txt')
    if not timing.is_file() or not timing.stat().st_size:
        raise RuntimeError('Parent final routed timing report is absent')
    log = (run / 'console.log').read_text()
    init = re.findall(r"^OTPU_DDR_CALIBRATION_INIT_2C=(256'h[0-9a-fA-F]{64})$", log, re.M)
    if cl_kind(info) == 'hbm':
        # the HBM CL has no DIMM: its validator prints no calibration sentinel (one would be
        # another CL's log); the Tcl drivers take none_hbm as the sentinel to expect
        if init:
            raise RuntimeError('HBM parent run carries a DDR calibration sentinel')
        ddr_init = 'none_hbm'
    elif not init or not int(init[-1].split("'h")[1], 16):
        raise RuntimeError('Parent has no verified persisted DDR calibration sentinel')
    else:
        ddr_init = init[-1]
    if flow.sha(checkpoint) != expected_sha.lower():
        raise RuntimeError('Parent routed checkpoint SHA256 mismatch')
    if not info.get('sources') or not any(n.startswith('design/') for n in info['sources']):
        raise RuntimeError('Parent frozen source inventory is absent')
    for name, digest in info['sources'].items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError(f'Unsafe parent source path: {name}')
        if flow.sha(run / 'cl_opentpu' / name) != digest:
            raise RuntimeError(f'Parent frozen source was modified: {name}')
    evidence = {'run': str(run), 'checkpoint': str(checkpoint),
                'checkpoint_sha256': expected_sha.lower(), 'exit_code': int(exit_code),
                'inputs_sha256': flow.sha(run / 'inputs.json'),
                'console_sha256': flow.sha(run / 'console.log'),
                'exit_code_sha256': flow.sha(run / 'exit_code'),
                'timing_report_sha256': flow.sha(timing), 'ddr_init_2c': ddr_init,
                'cl': cl_kind(info),
                'source_inventory': info['sources']}
    return info, evidence


def prepare(out: Path, run: Path, checkpoint: Path, expected_sha: str,
            *, validate_only: bool = False,
            directives: tuple[str, ...] = ('Explore',),
            mmcm: tuple[int, float] | None = None) -> tuple[Path, dict]:
    if out.exists():
        raise RuntimeError(f'{out} already exists; choose a new run directory')
    info, parent = load_parent(run, checkpoint, expected_sha)
    cl = out / 'cl_opentpu'
    for name, digest in info['sources'].items():
        dest = cl / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / 'cl_opentpu' / name, dest)
        if flow.sha(dest) != digest:
            raise RuntimeError(f'Copied parent source SHA256 mismatch: {name}')
    # Preserve every HDL and constraint byte; only the new driver/qualification
    # scripts replace their historical versions in this new source inventory.
    for name in VALIDATION_SCRIPTS if validate_only else RECOVERY_SCRIPTS:
        shutil.copy2(HERE / 'build/scripts' / name, cl / 'build/scripts' / name)
    inherited = out / 'parent_checkpoint' / checkpoint.name
    inherited.parent.mkdir()
    shutil.copy2(checkpoint, inherited)
    if flow.sha(inherited) != expected_sha.lower():
        raise RuntimeError('Copied routed checkpoint SHA256 mismatch')
    for name in ('reports', 'checkpoints'):
        (cl / 'build' / name).mkdir(exist_ok=True)
    driver_dir = out / 'recovery_driver'
    driver_dir.mkdir()
    for source in (Path(__file__), HERE / 'build.py'):
        shutil.copy2(source, driver_dir / source.name)
    info['sources'] = {str(p.relative_to(cl)): flow.sha(p)
                       for p in sorted(cl.rglob('*')) if p.is_file()}
    info['parent_implementation'] = parent
    if not validate_only and (not directives or set(directives) - set(PHYS_OPT_DIRECTIVES)):
        raise RuntimeError(f'Unknown post-route phys_opt directives: {directives}')
    info['route_recovery'] = {'directive': None if validate_only else ' '.join(directives),
                              'constraints_resourced': False,
                              'parent_checkpoint_copy': str(inherited),
                              'driver_sources': {str(p.relative_to(out)): flow.sha(p)
                                                 for p in sorted(driver_dir.iterdir())}}
    if mmcm:
        info['route_recovery']['mmcm'] = {'divclk': mmcm[0], 'clkfbout_mult_f': mmcm[1]}
    if validate_only:
        info['route_recovery'].update(mode='validate_only', checkpoint_written=False,
                                      output_must_match_parent=True)
    (out / 'inputs.json').write_text(json.dumps(info, indent=2) + '\n')
    # Detect changes during copying, including to the parent manifest/log/DCP.
    if load_parent(run, checkpoint, expected_sha)[1] != parent:
        raise RuntimeError('Parent changed while preparing recovery')
    return cl, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdk', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--from-run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--validate-only', action='store_true',
                        help='qualify and copy the unchanged routed parent; do not optimize or route')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--mmcm', nargs=2, metavar=('DIVCLK', 'MULT'),
                        help="set the core MMCM's DIVCLK_DIVIDE and CLKFBOUT_MULT_F before optimizing "
                             '(the same VCO only: clears DRC AVAL-168 from M 4.8; e.g. 5 24)')
    parser.add_argument('--phys-opt-directive', action='append', choices=PHYS_OPT_DIRECTIVES,
                        help='post-route phys_opt_design directive; repeat for passes in order '
                             '(default: Explore once); the passes stop once setup is met')
    args = parser.parse_args()
    if args.validate_only and args.phys_opt_directive:
        parser.error('--phys-opt-directive does not apply to --validate-only')
    mmcm = None
    if args.mmcm:
        divclk, mult = int(args.mmcm[0]), float(args.mmcm[1])
        if args.validate_only:
            parser.error('--mmcm changes the design: not with --validate-only')
        if not 1 <= divclk <= 106 or not 2.0 <= mult <= 128.0 or (mult * 8) % 1:
            parser.error('--mmcm: DIVCLK 1..106 and CLKFBOUT_MULT_F 2..128 in 0.125 steps')
        mmcm = (divclk, mult)
    directives = tuple(args.phys_opt_directive or ('Explore',))
    out, run, checkpoint = args.out.resolve(), args.from_run.resolve(), args.checkpoint.resolve()
    parent_info, _ = load_parent(run, checkpoint, args.checkpoint_sha256)
    # the parent's own target (f2, or an HBM target: f2-hbm:p4rl8a64c150), as this tree defines
    # it: its frozen platform and tech must be that target's before the HDK is looked at
    target = parent_target(parent_info)
    for key in ('platform', 'tech'):
        if parent_info.get(key) != target.to_dict().get(key):
            raise RuntimeError(f'Parent {key} differs from the pinned F2 target {target.name}')
    shell = flow.check_hdk(args.hdk.resolve(), target, shell=True)
    flow.check_vivado_version(flow.command('vivado', '-version'), target.platform['vivado'])
    cl, info = prepare(out, run, checkpoint, args.checkpoint_sha256,
                       validate_only=args.validate_only, directives=directives, mmcm=mmcm)
    env = dict(os.environ, BUILD_TAG=info['build_id'], HDK_SHELL_DIR=str(shell),
               OTPU_PARENT_DCP=info['route_recovery']['parent_checkpoint_copy'],
               OTPU_PARENT_SHA256=args.checkpoint_sha256.lower(),
               OTPU_EXPECTED_DDR_INIT=info['parent_implementation']['ddr_init_2c'],
               OTPU_PHYS_OPT_DIRECTIVES=' '.join(directives),
               OTPU_MMCM_DIVCLK=str(mmcm[0]) if mmcm else '',
               OTPU_MMCM_MULT=f'{mmcm[1]:.3f}' if mmcm else '')
    driver = 'validate_only.tcl' if args.validate_only else 'opt_route.tcl'
    operation = 'Route validation' if args.validate_only else 'Route recovery'
    cmd = ['vivado', '-mode', 'batch', '-source', driver,
           '-log', str(out / 'vivado.log'), '-journal', str(out / 'vivado.jou')]
    (out / 'invocation.json').write_text(json.dumps({'command': cmd,
        'cwd': str(cl / 'build/scripts'), 'environment': {k: env[k] for k in
        ('BUILD_TAG', 'HDK_SHELL_DIR', 'OTPU_PARENT_DCP', 'OTPU_PARENT_SHA256',
         'OTPU_EXPECTED_DDR_INIT', 'OTPU_PHYS_OPT_DIRECTIVES', 'OTPU_MMCM_DIVCLK',
         'OTPU_MMCM_MULT')}}, indent=2) + '\n')
    if args.prepare_only:
        print(f'Prepared {operation.lower()}: {out}')
        return
    with (out / 'console.log').open('w') as log:
        result = subprocess.run(cmd, cwd=cl / 'build/scripts', env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    # A zero Vivado exit alone does not establish immutable parent provenance.
    # Keep the run unqualified until the Python-side evidence checks also pass.
    (out / 'exit_code').write_text(str(result.returncode or 1) + '\n')
    routed = cl / f'build/checkpoints/cl_opentpu.{info["build_id"]}.post_route.dcp'
    marker = routed.parent / f'cl_opentpu.{info["build_id"]}.route_validation.passed'
    try:
        if load_parent(run, checkpoint, args.checkpoint_sha256)[1] != info['parent_implementation']:
            raise RuntimeError('Parent evidence changed during recovery')
        if result.returncode or not routed.is_file():
            raise RuntimeError(f'{operation} failed; see {out / "console.log"}')
        if marker.read_text().strip() != 'OTPU_STAGE_ROUTE_VALIDATION_PASSED':
            raise RuntimeError(f'{operation} has no exact qualification marker')
        if args.validate_only:
            for path in (routed, Path(info['route_recovery']['parent_checkpoint_copy'])):
                if flow.sha(path) != args.checkpoint_sha256.lower():
                    raise RuntimeError('Validation-only checkpoint differs from the pinned parent')
        artifacts = [routed, marker]
        probes = routed.parent / f'{info["build_id"]}.debug_probes.ltx'
        if probes.is_file():
            artifacts.append(probes)
        outputs = {'flow': 'RouteValidate' if args.validate_only else 'RouteOpt',
                   'inputs_sha256': flow.sha(out / 'inputs.json'),
                   'artifacts': {str(p.relative_to(out)): {'sha256': flow.sha(p), 'bytes': p.stat().st_size}
                                 for p in artifacts}}
        (out / 'exit_code').write_text('0\n')
        flow.load_routed_run(out, flow.sha(routed))
        (out / 'outputs.json').write_text(json.dumps(outputs, indent=2) + '\n')
    except BaseException:
        if args.validate_only:
            # A Tcl pass is provisional until all Python provenance checks pass.
            # Never leave a packageable copy/marker after a later gate fails.
            for path in (routed, marker, out / 'outputs.json'):
                path.unlink(missing_ok=True)
        (out / 'exit_code').write_text(str(result.returncode or 1) + '\n')
        raise
    print(f'{operation} qualified: {routed}')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
