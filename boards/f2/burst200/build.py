#!/usr/bin/env python3
"""Build the burst200 AWS F2 CL against a full, externally installed HDK."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BOARD = ROOT / 'boards/f2/burst200'
SCRIPTS = BOARD / 'scripts'
STAGES = ['ip', 'pcis', 'clocks', 'synth_link', 'implement', 'refine', 'audit']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hdk', required=True, type=Path, help='Full aws-fpga/hdk directory')
    ap.add_argument('--vivado', default='vivado')
    ap.add_argument('--output', type=Path, default=ROOT / 'build/f2/burst200')
    ap.add_argument('--plan', action='store_true', help='Print stages without creating files')
    a = ap.parse_args()
    hdk, output = a.hdk.resolve(), a.output.resolve()
    if a.plan:
        print('HDK:', hdk, '\nOutput:', output)
        print('\n'.join(f'{i + 1}. {stage}' for i, stage in enumerate(STAGES)))
        return
    if output.exists():
        ap.error('Output already exists; select a fresh --output directory')
    required = [
        'common/shell_stable/build/checkpoints/from_aws/cl_bb_routed.small_shell.dcp',
        'common/shell_stable/build/constraints/cl_pins.xdc',
        'common/shell_stable/build/constraints/small_shell_level_1_fp_cl.xdc',
        'common/shell_stable/design/interfaces/cl_ports.vh',
        'common/shell_stable/design/sh_ddr/synth/sh_ddr.sv',
        'common/lib/interfaces.sv',
        'cl/examples/cl_mem_perf/design/cl_mem_hbm_wrapper.sv',
        'common/ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_hbm/cl_hbm.xci',
    ]
    missing = [f for f in required if not (hdk / f).is_file()]
    if missing:
        ap.error('HDK is incomplete: ' + ', '.join(missing))
    if (hdk / 'common/shell_stable/shell_version.txt').read_text().strip() != 'small_shell=0x10212415':
        ap.error('Expected AWS small_shell=0x10212415')
    version = subprocess.check_output([a.vivado, '-version'], text=True)
    if 'vivado v2025.2' not in version.lower():
        ap.error('Vivado 2025.2 is required for this flow')
    output.mkdir(parents=True)
    (output / 'vivado_version.txt').write_text(version)
    dirs = {stage: output / stage for stage in STAGES}
    for directory in dirs.values():
        directory.mkdir()
    script_names = ['create_f2_ip.tcl', 'create_pcis_ip.tcl', 'create_clocks.tcl',
                    'build_f2.tcl', 'implement.tcl', 'refine_route.tcl', 'audit_route.tcl']
    for stage, name in zip(STAGES, script_names):
        shutil.copy2(SCRIPTS / name, dirs[stage] / 'build.tcl')
    for stage in ['synth_link', 'implement']:
        for p in (BOARD / 'constraints').glob('*.tcl'):
            shutil.copy2(p, dirs[stage] / p.name)
    shutil.copytree(ROOT / 'rtl/boards/f2/burst200', dirs['synth_link'] / 'rtl')
    shutil.copy2(ROOT / 'targets/f2-burst200.toml', dirs['synth_link'] / 'f2-burst200.toml')
    with (dirs['synth_link'] / 'config_generation.log').open('w') as log:
        subprocess.run([sys.executable, str(SCRIPTS / 'emit_config.py'),
                        str(dirs['synth_link'])], stdout=log, stderr=subprocess.STDOUT, check=True)
    inputs = [*(ROOT / 'rtl').rglob('*.sv'), *BOARD.rglob('*.tcl'),
              *BOARD.rglob('*.py'), ROOT / 'targets/f2-burst200.toml']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(inputs)}
    (output / 'inputs.sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    commands = [
        ('ip', [dirs['ip'], hdk], 'ip_done.txt'),
        ('pcis', [dirs['pcis']], 'ip_done.txt'),
        ('clocks', [dirs['clocks']], 'ip_done.txt'),
        ('synth_link', [dirs['synth_link'], dirs['ip'], dirs['pcis'], ROOT, hdk,
                        dirs['clocks'], 'link_only'], 'link_done.txt'),
        ('implement', [dirs['implement'], dirs['synth_link'] / 'post_link.dcp', hdk,
                       dirs['synth_link'] / 'link_done.txt'], 'implementation_result.txt'),
        ('refine', [dirs['refine'], dirs['implement'] / 'post_route.dcp'], 'implementation_result.txt'),
        ('audit', [dirs['audit'], dirs['refine'] / 'post_route.dcp'], 'audit.txt'),
    ]
    for stage, argv, marker in commands:
        directory = dirs[stage]
        cmd = [a.vivado, '-mode', 'batch', '-source', str(directory / 'build.tcl'),
               '-tclargs', *map(str, argv)]
        (directory / 'command.json').write_text(json.dumps(cmd, indent=2) + '\n')
        print('Running', stage, '— log:', directory / 'console.log', flush=True)
        with (directory / 'console.log').open('w') as log:
            result = subprocess.run(cmd, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not (directory / marker).is_file():
            raise SystemExit(f'{stage} failed; inspect {directory / "console.log"}')
        if stage == 'refine' and 'ROUTED_TIMING_PASS_PENDING_REPORT_REVIEW' not in (directory / 'console.log').read_text():
            raise SystemExit('Final refinement did not pass timing; inspect its reports')
    print('Checkpoint:', dirs['refine'] / 'post_route.dcp')
    print('Review the new independent audit reports before packaging or deployment.')


if __name__ == '__main__':
    main()
