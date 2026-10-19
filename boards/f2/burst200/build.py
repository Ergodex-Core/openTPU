#!/usr/bin/env python3
"""Build the burst200 AWS F2 CL against a full, externally installed HDK.

--target selects the openTPU target (default targets/f2-burst200.toml, Alex's one slice;
targets/f2-burst200-s3.toml: three slices on six HBM ports, docs/v4/f2-burst200-s3.md). The
target's [platform] rtl_board names the directory of its CL top under rtl/boards/f2
(f2-burst200: burst200/, the checked-in hardware-tested wrapper; f2-burst200-s3: burst200-s3/,
with the adapters taken from burst200/), its floorplan the constraints file placed as
floorplan.tcl (default floorplan.tcl), and its slices and hbm_ports the geometry the IP, PCIS
and link stages read from geometry.tcl."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
BOARD = ROOT / 'boards/f2/burst200'
SCRIPTS = BOARD / 'scripts'
STAGES = ['ip', 'pcis', 'clocks', 'synth_link', 'implement', 'refine', 'audit']
ADAPTERS = ['otpu_native_axi.sv', 'otpu_hbm_geometry.sv']     # Alex's, shared by every top


def plan(target_path: Path) -> dict:
    """What the target asks of the flow: its toml, the RTL files of its CL top, the floorplan
    constraints file and the geometry Tcl (slices, HBM ports) the stages read."""
    from opentpu import target as T
    t = T.load(str(target_path.resolve()))
    board = t.platform.get('rtl_board', 'f2-burst200')
    if not board.startswith('f2-burst200'):
        raise SystemExit(f'{target_path}: rtl_board {board!r} is not a burst200 flow target')
    top_dir = ROOT / 'rtl/boards/f2' / board[len('f2-'):]
    if board == 'f2-burst200':
        rtl = sorted(top_dir.glob('*.sv'))
    else:
        rtl = sorted(top_dir.glob('*.sv')) + [ROOT / 'rtl/boards/f2/burst200' / f for f in ADAPTERS]
    if not any(p.name == 'cl_opentpu.sv' for p in rtl):
        raise SystemExit(f'{top_dir}: no cl_opentpu.sv')
    floorplan = BOARD / 'constraints' / t.platform.get('floorplan', 'floorplan.tcl')
    if not floorplan.is_file():
        raise SystemExit(f'{target_path}: floorplan {floorplan} does not exist')
    S, nch = t.arch['S'], t.platform['channels_per_slice']
    ports = list(t.platform.get('hbm_ports', [0, 4]))
    if nch != 2 or len(ports) != 2 * S or len(set(ports)) != len(ports):
        raise SystemExit(f'{target_path}: two channels a slice on {2 * S} distinct HBM ports')
    geometry = (f'# {target_path.name}: {S} slice(s), window j = 2 s + c on HBM AXI port otpu_ports[j]\n'
                f'set otpu_S {S}\nset otpu_ports {{{" ".join(map(str, ports))}}}\n')
    return dict(target=t, toml=target_path.resolve(), rtl=rtl, floorplan=floorplan,
                geometry=geometry, S=S, ports=ports)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hdk', required=True, type=Path, help='Full aws-fpga/hdk directory')
    ap.add_argument('--vivado', default='vivado')
    ap.add_argument('--output', type=Path, default=ROOT / 'build/f2/burst200')
    ap.add_argument('--target', type=Path, default=ROOT / 'targets/f2-burst200.toml',
                    help='the openTPU target (its rtl_board, floorplan, slices and hbm_ports)')
    ap.add_argument('--plan', action='store_true', help='Print stages without creating files')
    a = ap.parse_args()
    hdk, output = a.hdk.resolve(), a.output.resolve()
    pl = plan(a.target)
    if a.plan:
        print('HDK:', hdk, '\nOutput:', output)
        print('Target:', pl['toml'], f"({pl['S']} slice(s), HBM ports {pl['ports']})")
        print('CL top:', ', '.join(str(p.relative_to(ROOT)) for p in pl['rtl']))
        print('Floorplan:', pl['floorplan'].relative_to(ROOT))
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
            if p.name.startswith('floorplan'):
                continue                    # the target's floorplan alone, as floorplan.tcl
            shutil.copy2(p, dirs[stage] / p.name)
        shutil.copy2(pl['floorplan'], dirs[stage] / 'floorplan.tcl')
    for stage in ['ip', 'pcis', 'synth_link']:
        (dirs[stage] / 'geometry.tcl').write_text(pl['geometry'])
    (dirs['synth_link'] / 'rtl').mkdir()
    for p in pl['rtl']:
        shutil.copy2(p, dirs['synth_link'] / 'rtl' / p.name)
    shutil.copy2(pl['toml'], dirs['synth_link'] / pl['toml'].name)
    with (dirs['synth_link'] / 'config_generation.log').open('w') as log:
        subprocess.run([sys.executable, str(SCRIPTS / 'emit_config.py'),
                        str(dirs['synth_link']), '--target', str(pl['toml'])],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    inputs = [*(ROOT / 'rtl').rglob('*.sv'), *BOARD.rglob('*.tcl'),
              *BOARD.rglob('*.py'), pl['toml']]
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
