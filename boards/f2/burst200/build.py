#!/usr/bin/env python3
"""Build the burst200 AWS F2 CL against a full, externally installed HDK.

--target selects the openTPU target (default targets/f2-burst200.toml, Alex's one slice;
targets/f2-burst200-s3.toml: three slices on six HBM ports, docs/v4/f2-burst200-s3.md). The
target's [platform] rtl_board names the directory of its CL top under rtl/boards/f2
(f2-burst200: burst200/, the checked-in hardware-tested wrapper; f2-burst200-s3: burst200-s3/,
with the adapters taken from burst200/), its floorplan the constraints file placed as
floorplan.tcl (default floorplan.tcl), and its slices and hbm_ports the geometry the IP, PCIS
and link stages read from geometry.tcl.

Variants of one design. --floorplan replaces the target's floorplan file; --place-directive,
--phys-directive and --route-directive replace the implementation's directives (Explore,
AggressiveExplore, Explore), written to implement/strategy.tcl. --from-netlist RUN starts from
an earlier build's synthesized, shell-linked netlist (RUN/synth_link/post_link.dcp) and runs
implement, refine and audit alone, skipping the IP, clock, synthesis and link stages (about 40
minutes); it refuses unless everything that netlist was built from matches this build: the
netlist stages' scripts, the generated geometry, core clock and link config, the CL top and the
RTL the config lists, the HDK and the Vivado build. So the floorplan and the strategy may change
and the netlist may not; several variants of one netlist can run side by side. --incremental-from
DCP implements incrementally from a routed checkpoint of the same design (read_checkpoint
-incremental, with --incremental-directive RuntimeOptimized, Vivado's default, or TimingClosure
or Quick): for a small RTL change to a closed design, the unchanged cells keep their placement
and routing."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
BOARD = ROOT / 'boards/f2/burst200'
SCRIPTS = BOARD / 'scripts'
STAGES = ['ip', 'pcis', 'clocks', 'synth_link', 'implement', 'refine', 'audit']
ADAPTERS = ['otpu_native_axi.sv', 'otpu_hbm_geometry.sv']     # Alex's, shared by every top
SCRIPT = dict(ip='create_f2_ip.tcl', pcis='create_pcis_ip.tcl', clocks='create_clocks.tcl',
              synth_link='build_f2.tcl', implement='implement.tcl', refine='refine_route.tcl',
              audit='audit_route.tcl')
NETLIST_STAGES = STAGES[:4]         # what --from-netlist takes from the earlier build
# Vivado 2025.2's directives (help place_design / phys_opt_design / route_design / read_checkpoint;
# phys_opt's RQS needs a report_qor_suggestions file and is left out)
PLACE_DIRECTIVES = ['Explore', 'AggressiveExplore', 'RuntimeOptimized', 'Quick', 'Default']
PHYS_DIRECTIVES = ['Explore', 'ExploreWithHoldFix', 'ExploreWithAggressiveHoldFix',
                   'AggressiveExplore', 'AlternateReplication', 'AggressiveFanoutOpt',
                   'AlternateFlowWithRetiming', 'AddRetime', 'RuntimeOptimized', 'Default']
ROUTE_DIRECTIVES = ['Explore', 'AggressiveExplore', 'NoTimingRelaxation', 'MoreGlobalIterations',
                    'HigherDelayCost', 'AdvancedSkewModeling', 'AlternateCLBRouting',
                    'RuntimeOptimized', 'Quick', 'Default']
INCREMENTAL_DIRECTIVES = ['RuntimeOptimized', 'TimingClosure', 'Quick']
STRATEGY = dict(place='Explore', phys='AggressiveExplore', route='Explore')    # the flow's own


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
    if (nch != 2 * t.arch.get('CPC', 1) or (nch > 2 and S != 1) or len(ports) != nch * S
            or len(set(ports)) != len(ports)):
        raise SystemExit(f'{target_path}: 2 CPC channels a slice (four on one slice at CPC 2) '
                         f'on {nch * S} distinct HBM ports')
    groups = nch * S // 2               # f2_mem block designs, two windows each
    geometry = (f'# {target_path.name}: {S} slice(s) of {nch} channels, window j = {nch} s + c on HBM '
                f'AXI port otpu_ports[j], windows 2g and 2g + 1 in f2_mem group g\n'
                f'set otpu_S {S}\nset otpu_G {groups}\nset otpu_ports {{{" ".join(map(str, ports))}}}\n')
    core_mhz = float(t.platform['core_mhz'])
    core_clock = (f'# {target_path.name}: the core clock (MHz); the TMEM clock is twice it\n'
                  f'set otpu_core_mhz {core_mhz}\n')
    return dict(target=t, toml=target_path.resolve(), rtl=rtl, floorplan=floorplan,
                geometry=geometry, S=S, ports=ports, core_mhz=core_mhz, core_clock=core_clock)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    """The path from the repository root when it is inside it, else as given."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def emit_config(directory: Path, toml: Path) -> None:
    """The link stage's config.tcl and target.json for the target (emit_config.py)."""
    with (directory / 'config_generation.log').open('w') as log:
        subprocess.run([sys.executable, str(SCRIPTS / 'emit_config.py'), str(directory),
                        '--target', str(toml)], stdout=log, stderr=subprocess.STDOUT, check=True)


def netlist_parent(run: Path, pl: dict, hdk: Path, version: str) -> dict:
    """An earlier build's netlist (run/synth_link/post_link.dcp) for this build, when everything
    it was built from matches: the netlist stages' scripts, the generated geometry, core clock and
    link config, the CL top files, the shared RTL the config lists (against the run's
    inputs.sha256.json), the HDK it was linked against and the Vivado build. Else SystemExit
    naming the first difference. Returns the provenance recorded as netlist_parent.json."""
    def refuse(why):
        raise SystemExit(f'--from-netlist {run}: {why}; build the netlist with the full flow')
    link = run / 'synth_link'
    if not link.is_dir() and (run / 'netlist_parent.json').is_file():
        first = json.loads((run / 'netlist_parent.json').read_text())['run']
        refuse(f'it reused the netlist of {first}; name that run')
    dcp = link / 'post_link.dcp'
    if not (link / 'link_done.txt').is_file() or not dcp.is_file():
        refuse('its link stage did not complete')
    checked = []

    def same(what: str, path: Path, expected: bytes, code_only=False):
        """code_only: the generated geometry and core clock, whose first line comments on the
        target file's name, compare without their comment lines."""
        def norm(b: bytes) -> bytes:
            return b'\n'.join(l for l in b.splitlines() if not l.lstrip().startswith(b'#')) if code_only else b
        if not path.is_file() or norm(path.read_bytes()) != norm(expected):
            refuse(f'{what} differs ({path})')
        checked.append(str(path.relative_to(run)))
    for stage in NETLIST_STAGES:
        same(f'the {stage} stage script', run / stage / 'build.tcl',
             (SCRIPTS / SCRIPT[stage]).read_bytes())
    for stage in ['ip', 'pcis', 'synth_link']:
        same('the geometry', run / stage / 'geometry.tcl', pl['geometry'].encode(), code_only=True)
    same('the core clock', run / 'clocks/core_clock.tcl', pl['core_clock'].encode(), code_only=True)
    with tempfile.TemporaryDirectory() as tmp:
        emit_config(Path(tmp), pl['toml'])
        config = (Path(tmp) / 'config.tcl').read_text()
    same('the link config', link / 'config.tcl', config.encode())
    if sorted(p.name for p in (link / 'rtl').glob('*.sv')) != sorted(p.name for p in pl['rtl']):
        refuse('the CL top files differ')
    for p in pl['rtl']:
        same(f'the CL top file {p.name}', link / 'rtl' / p.name, p.read_bytes())
    try:
        recorded = json.loads((run / 'inputs.sha256.json').read_text())
    except (OSError, ValueError):
        refuse('it has no readable inputs.sha256.json')
    for f in re.search(r'^set rtl_files \{([^}]*)\}', config, re.M).group(1).split():
        key = f'rtl/{f}'
        if recorded.get(key) != sha256(ROOT / key):
            refuse(f'{key} differs from the one it was synthesized from')
        checked.append(key)
    argv = json.loads((link / 'command.json').read_text())
    linked_hdk = Path(argv[argv.index('-tclargs') + 5])
    if linked_hdk != hdk:
        refuse(f'it was linked against the HDK at {linked_hdk}, not {hdk}')
    if not (run / 'vivado_version.txt').is_file() or (run / 'vivado_version.txt').read_text() != version:
        refuse('another Vivado build made it')
    return dict(run=str(run), post_link_dcp=str(dcp), post_link_sha256=sha256(dcp),
                parent_inputs_sha256=sha256(run / 'inputs.sha256.json'), checked=checked)


def strategy_tcl(strategy: dict, incremental_directive: str) -> str:
    """implement/strategy.tcl: the directives implement.tcl runs (the full flow's place,
    phys_opt and route; the incremental flow's read_checkpoint -incremental)."""
    return ('# build.py: the implementation directives implement.tcl runs\n'
            f"set place_directive {strategy['place']}\nset phys_directive {strategy['phys']}\n"
            f"set route_directive {strategy['route']}\nset incremental_directive {incremental_directive}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hdk', required=True, type=Path, help='Full aws-fpga/hdk directory')
    ap.add_argument('--vivado', default='vivado')
    ap.add_argument('--output', type=Path, default=ROOT / 'build/f2/burst200')
    ap.add_argument('--target', type=Path, default=ROOT / 'targets/f2-burst200.toml',
                    help='the openTPU target (its rtl_board, floorplan, slices and hbm_ports)')
    ap.add_argument('--plan', action='store_true', help='Print stages without creating files')
    ap.add_argument('--floorplan', type=Path,
                    help="a floorplan file in place of the target's: a path, or a file name "
                         "under boards/f2/burst200/constraints")
    ap.add_argument('--place-directive', choices=PLACE_DIRECTIVES,
                    help=f"place_design's directive (default {STRATEGY['place']})")
    ap.add_argument('--phys-directive', choices=PHYS_DIRECTIVES,
                    help=f"the post-placement phys_opt_design's directive (default {STRATEGY['phys']})")
    ap.add_argument('--route-directive', choices=ROUTE_DIRECTIVES,
                    help=f"route_design's directive (default {STRATEGY['route']})")
    ap.add_argument('--from-netlist', type=Path, metavar='RUN',
                    help="an earlier build's output directory: implement its synthesized, linked "
                         "netlist (refused unless built from the same sources)")
    ap.add_argument('--incremental-from', type=Path, metavar='DCP',
                    help='a routed checkpoint of the same design to implement incrementally from')
    ap.add_argument('--incremental-directive', choices=INCREMENTAL_DIRECTIVES,
                    help='read_checkpoint -incremental -directive (default RuntimeOptimized)')
    a = ap.parse_args()
    hdk, output = a.hdk.resolve(), a.output.resolve()
    pl = plan(a.target)
    if a.floorplan:
        fp = a.floorplan if a.floorplan.is_file() else BOARD / 'constraints' / a.floorplan
        if not fp.is_file():
            ap.error(f'--floorplan {a.floorplan}: no such file')
        pl['floorplan'] = fp.resolve()
    chosen = {k: getattr(a, f'{k}_directive') for k in STRATEGY}
    if a.incremental_from:
        if any(chosen.values()):
            ap.error('--place-, --phys- and --route-directive set the full flow; '
                     'an incremental run takes --incremental-directive')
        if a.incremental_from.suffix != '.dcp' or not a.incremental_from.is_file():
            ap.error(f'--incremental-from {a.incremental_from}: not a checkpoint file')
    elif a.incremental_directive:
        ap.error('--incremental-directive needs --incremental-from')
    strategy = {k: v or STRATEGY[k] for k, v in chosen.items()}
    incremental_directive = a.incremental_directive or 'RuntimeOptimized'
    reuse = a.from_netlist.resolve() if a.from_netlist else None
    stages = STAGES[len(NETLIST_STAGES):] if reuse else STAGES
    if a.plan:
        print('HDK:', hdk, '\nOutput:', output)
        print('Target:', pl['toml'], f"({pl['S']} slice(s), HBM ports {pl['ports']})")
        print('CL top:', ', '.join(str(p.relative_to(ROOT)) for p in pl['rtl']))
        print('Floorplan:', rel(pl['floorplan']))
        print('Core clock:', f"{pl['core_mhz']} MHz (TMEM {2 * pl['core_mhz']} MHz)")
        if reuse:
            print('Netlist:', reuse / 'synth_link/post_link.dcp', '(checked when the build starts)')
        if a.incremental_from:
            print('Implementation: incremental from', a.incremental_from.resolve(),
                  f'({incremental_directive})')
        else:
            print('Implementation:', f"place_design {strategy['place']}, phys_opt_design "
                  f"{strategy['phys']}, route_design {strategy['route']}")
        print('\n'.join(f'{i + 1}. {stage}' for i, stage in enumerate(stages)))
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
    parent = netlist_parent(reuse, pl, hdk, version) if reuse else None
    output.mkdir(parents=True)
    (output / 'vivado_version.txt').write_text(version)
    if parent:
        (output / 'netlist_parent.json').write_text(json.dumps(parent, indent=2) + '\n')
    dirs = {stage: output / stage for stage in stages}
    for directory in dirs.values():
        directory.mkdir()
    for stage in stages:
        shutil.copy2(SCRIPTS / SCRIPT[stage], dirs[stage] / 'build.tcl')
    for stage in [s for s in ['synth_link', 'implement'] if s in dirs]:
        for p in (BOARD / 'constraints').glob('*.tcl'):
            if p.name.startswith('floorplan'):
                continue                    # the target's floorplan alone, as floorplan.tcl
            shutil.copy2(p, dirs[stage] / p.name)
        shutil.copy2(pl['floorplan'], dirs[stage] / 'floorplan.tcl')
    (dirs['implement'] / 'strategy.tcl').write_text(strategy_tcl(strategy, incremental_directive))
    if a.incremental_from:
        ref = a.incremental_from.resolve()
        (dirs['implement'] / 'incremental_reference.txt').write_text(f'{ref}\n')
        (dirs['implement'] / 'incremental_reference.sha256').write_text(sha256(ref) + '\n')
    if not parent:
        for stage in ['ip', 'pcis', 'synth_link']:
            (dirs[stage] / 'geometry.tcl').write_text(pl['geometry'])
        (dirs['clocks'] / 'core_clock.tcl').write_text(pl['core_clock'])
        (dirs['synth_link'] / 'rtl').mkdir()
        for p in pl['rtl']:
            shutil.copy2(p, dirs['synth_link'] / 'rtl' / p.name)
        shutil.copy2(pl['toml'], dirs['synth_link'] / pl['toml'].name)
        emit_config(dirs['synth_link'], pl['toml'])
    inputs = [*(ROOT / 'rtl').rglob('*.sv'), *BOARD.rglob('*.tcl'),
              *BOARD.rglob('*.py'), pl['toml'], pl['floorplan']]
    hashes = {rel(p): sha256(p) for p in sorted(set(inputs))}
    (output / 'inputs.sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    post_link = Path(parent['post_link_dcp']) if parent else dirs['synth_link'] / 'post_link.dcp'
    commands = [] if parent else [
        ('ip', [dirs['ip'], hdk], 'ip_done.txt'),
        ('pcis', [dirs['pcis']], 'ip_done.txt'),
        ('clocks', [dirs['clocks']], 'ip_done.txt'),
        ('synth_link', [dirs['synth_link'], dirs['ip'], dirs['pcis'], ROOT, hdk,
                        dirs['clocks'], 'link_only'], 'link_done.txt'),
    ]
    commands += [
        ('implement', [dirs['implement'], post_link, hdk, post_link.parent / 'link_done.txt'],
         'implementation_result.txt'),
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
