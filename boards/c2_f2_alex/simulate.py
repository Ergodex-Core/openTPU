#!/usr/bin/env python3
"""Run real C2 Alex 9SI/8MI vendor fabric simulation, never FPGA hardware.

Requires the independently recorded SHA256 of a completed private-IP receipt.
The functional HBM model validates protocol/data, not physical HBM performance.
All vendor simulation runs and their copied IP are isolated from the build.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import signal
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('c2_alex_sim_build', HERE / 'build.py')
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)
SCENARIOS = ((17, 0), (17, 1), (71, 0), (71, 1))
SOURCES = {
    'tb_f2_mem_c2.sv': ROOT / 'rtl/c2_alex/tb_f2_mem_c2.sv',
    'otpu_native_axi.sv': build.SUPPORT / 'rtl/boards/f2/otpu_native_axi.sv',
    'otpu_hbm_geometry.sv': build.SUPPORT / 'rtl/boards/f2/otpu_hbm_geometry.sv',
    'axi_hbm_model.sv': build.SUPPORT / 'tests/f2/axi_hbm_model.sv',
}
BD = 'ip/ip_project/f2_ip.srcs/sources_1/bd/f2_mem/f2_mem.bd'
WRAPPER = 'ip/ip_project/f2_ip.gen/sources_1/bd/f2_mem/hdl/f2_mem_wrapper.v'
PCIS = 'pcis/ip_project/pcis.srcs/sources_1/ip/f2_pcis_pipe/f2_pcis_pipe.xci'


def verified_ip(run, receipt_sha256):
    if not re.fullmatch('[0-9a-f]{64}', receipt_sha256):
        raise RuntimeError('Expected an explicit lower-case SHA256 of ip-complete.json')
    if build.sha(run / 'ip-complete.json') != receipt_sha256:
        raise RuntimeError('Private IP receipt does not match the supplied SHA256')
    build.verify_support()
    origin = json.loads((run / 'inputs.json').read_text())
    required = {'schema': 1, 'kind': 'c2-alex-burst200-build-inputs',
                'alex_revision': build.ALEX_REVISION, 'archive_sha256': build.ARCHIVE_SHA256,
                'support_manifest_sha256': build.sha(build.SUPPORT / 'manifest.json'),
                'hdk_revision': build.HDK_REVISION, 'ip_revision': build.IP_REVISION,
                'part': build.PART, 'core_khz': 200000, 'hbm_mhz': 450,
                'hbm_ports': 8, 'channel_bytes': 2**29, 'source_status': ''}
    for key, value in required.items():
        if origin.get(key) != value:
            raise RuntimeError(f'Private IP input mismatch: {key}')
    # Bind the current reviewed recipe, generator and validator to the receipt's
    # source snapshot. Adding this simulator does not require rebuilding the IP.
    for name in ('rtl/c2_alex/generate.py', 'rtl/c2_alex/create_f2_ip.tcl',
                 'boards/c2_f2_alex/build.py'):
        if origin.get('source_inputs', {}).get(name, {}).get('sha256') != build.sha(ROOT / name):
            raise RuntimeError(f'Private IP source mismatch: {name}')
    return origin, build.verify_ip_run(run, origin)


def simulation_tcl(run, private, seed, phase):
    """Vivado is the compile/elaborate/XSim driver, as in Alex's baseline."""
    return f'''set run {{{run}}}
set iprun {{{private}}}
cd $run
set_param general.maxThreads 4
create_project sim $run/project -part {build.PART} -force
set_property XPM_LIBRARIES {{XPM_CDC XPM_MEMORY XPM_FIFO}} [current_project]
add_files $iprun/{BD}
add_files $iprun/{WRAPPER}
read_ip $iprun/{PCIS}
add_files -fileset sim_1 [glob $run/rtl/*.sv]
set_property top tb_f2_mem_c2 [get_filesets sim_1]
set_property xsim.simulate.runtime {{all}} [get_filesets sim_1]
set_property -name xsim.simulate.xsim.more_options -value {{-testplusarg SEED={seed} -testplusarg PHASE={phase}}} -objects [get_filesets sim_1]
launch_simulation -simset sim_1 -mode behavioral
close_sim
close_project
'''


def parse_result(text, seed, phase):
    """No substring PASS acceptance: require one complete scenario + eight PCs."""
    passes = re.findall(r'^C2_ALEX_FABRIC_PASS seed=(\d+) phase=(\d+) pcs=(\d+) reset_pending=(\d+) host_b=(\d+) host_r=(\d+)\s*$', text, re.M)
    if passes != [(str(seed), str(phase), '8', '8', '994', '1229')]:
        raise RuntimeError('Missing, duplicated, or incorrect scenario PASS record')
    rows = re.findall(r'^C2_PC pc=(\d+) accepted=(\d+) returned=(\d+) consumed=(\d+) axi_beats=(\d+) peak=(\d+) hbm_reads=(\d+) hbm_writes=(\d+)\s*$', text, re.M)
    if len(rows) != 8 or {int(row[0]) for row in rows} != set(range(8)):
        raise RuntimeError('Missing or duplicated per-PC conservation evidence')
    result = []
    for row in rows:
        pc, accepted, returned, consumed, axi, peak, reads, writes = map(int, row)
        want = 610 if pc in (2, 6) else 609
        if (accepted, returned, consumed, axi, peak) != (want, want, want, want, 32) or reads == 0 or writes == 0:
            raise RuntimeError(f'Per-PC conservation or RD32 coverage failed: {pc}')
        result.append(dict(zip(('pc', 'accepted', 'returned', 'consumed', 'axi_beats', 'peak', 'hbm_reads', 'hbm_writes'), map(int, row))))
    # Vivado can return zero after a simulator fatal. Success records alone do
    # not excuse other fatal/error diagnostics from elaboration or simulation.
    if re.search(r'^\s*(?:Fatal:|FATAL:|ERROR:|ERROR\s+\[)', text, re.M):
        raise RuntimeError('Vendor simulation contains an error/fatal diagnostic')
    return {'seed': seed, 'phase': phase, 'pcs': sorted(result, key=lambda x: x['pc']),
            'host_b': 994, 'host_r': 1229, 'reset_pending': 8}


def capture_simulation_ip_files(run):
    """All copied dependencies, including .vh/.tcl files outside build suffixes."""
    trees = [run / stage / 'ip_project' for stage in ('ip', 'pcis', 'clocks')]
    for tree in trees:
        if tree.is_symlink() or any(p.is_symlink() for p in tree.rglob('*')):
            raise RuntimeError('Private IP simulation copy must contain no symlinks')
    return build.inventory(trees, run)


def verify_simulation_ip_files(run, expected):
    if not expected or capture_simulation_ip_files(run) != expected:
        raise RuntimeError('Private simulation IP dependency inventory changed')


def copy_ip(run, out):
    """Copy before Vivado: never let simulation regenerate the synthesis input."""
    all_files = capture_simulation_ip_files(run)
    required_bytes = sum(r['bytes'] for r in all_files.values()) + 2 * 1024**3
    if shutil.disk_usage(out).free < required_bytes:
        raise RuntimeError(f'Insufficient disk for private simulation IP: need {required_bytes} bytes')
    private = out / 'private-ip'
    private.mkdir()
    for stage in ('ip', 'pcis', 'clocks'):
        (private / stage).mkdir()
        shutil.copytree(run / stage / 'ip_project', private / stage / 'ip_project')
        for name in ('build.tcl', 'ip_done.txt'):
            shutil.copy2(run / stage / name, private / stage / name)
    for name in ('inputs.json', 'ip-complete.json'):
        shutil.copy2(run / name, private / name)
    verify_simulation_ip_files(private, all_files)
    return private, all_files


def run_vendor(cmd, *, cwd, stdout, stderr, timeout):
    """Bound the complete local Vivado/XSim process group, including children."""
    process = subprocess.Popen(cmd, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return subprocess.CompletedProcess(cmd, code)
    except BaseException:
        # A child simulator may outlive its parent; always signal the group.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def run_scenario(out, private, seed, phase, vivado, *, runner=run_vendor):
    run = out / f'seed-{seed}-phase-{phase}'
    run.mkdir()
    (run / 'rtl').mkdir()
    for name, source in SOURCES.items():
        shutil.copy2(source, run / 'rtl' / name)
    (run / 'simulate.tcl').write_text(simulation_tcl(run, private, seed, phase))
    consumed = build.inventory([run / 'rtl', run / 'simulate.tcl'], run)
    build.write_json(run / 'inputs.json', consumed)
    cmd = [vivado, '-mode', 'batch', '-source', str(run / 'simulate.tcl')]
    build.write_json(run / 'command.json', cmd)
    with (run / 'console.log').open('x') as log:
        result = runner(cmd, cwd=run, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    build.write_json(run / 'exit.json', {'returncode': result.returncode})
    if result.returncode:
        raise RuntimeError(f'Vendor simulation failed: {run}')
    if build.inventory([run / 'rtl', run / 'simulate.tcl'], run) != consumed:
        raise RuntimeError('Simulation inputs changed while running')
    result = parse_result((run / 'console.log').read_text(), seed, phase)
    build.write_json(run / 'result.json', result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ip-run', type=Path, required=True)
    ap.add_argument('--ip-receipt-sha256', required=True)
    ap.add_argument('--out', type=Path, required=True, help='Fresh isolated output directory')
    ap.add_argument('--vivado', default='vivado')
    ap.add_argument('--plan', action='store_true', help='Validate provenance and print plan, no tool execution or output')
    a = ap.parse_args(argv)
    iprun, out = build.safe_path(a.ip_run), build.safe_path(a.out)
    origin, receipt = verified_ip(iprun, a.ip_receipt_sha256)
    if out.exists():
        raise RuntimeError(f'Refusing existing simulation output: {out}')
    if a.plan:
        print(json.dumps({'scenarios': SCENARIOS, 'private_ip_copy': True,
                         'ip_receipt_sha256': a.ip_receipt_sha256, 'max_threads': 4,
                         'hardware_actions': False, 'qualification': 'protocol-simulation-only'}, indent=2))
        return
    version = build.command(a.vivado, '-version')
    build.check_vivado(version)
    sources_before = {name: build.sha(source) for name, source in SOURCES.items()}
    out.mkdir(parents=True)
    (out / 'vivado_version.txt').write_text(version + '\n')
    private, copied = copy_ip(iprun, out)
    verified_ip(private, a.ip_receipt_sha256)
    inputs = {'schema': 1, 'kind': 'c2-alex-vendor-fabric-simulation',
              'ip_receipt_sha256': a.ip_receipt_sha256, 'build_inputs_sha256': receipt['inputs_sha256'],
              'ip_source_revision': origin['source_revision'], 'ip_products': receipt['products'],
              'simulation_ip_files': copied, 'sources': sources_before,
              'runner_sha256': build.sha(Path(__file__)), 'builder_sha256': build.sha(HERE / 'build.py'),
              'scenarios': SCENARIOS, 'clocks_mhz': {'core': 200, 'host': 250, 'hbm_nominal': 450},
              'hbm_simulated_period_ps': 2222, 'hbm_model': 'functional-not-silicon-timing',
              'qualification': 'protocol-simulation-only-not-physical-timing', 'hardware_actions': False}
    build.write_json(out / 'inputs.json', inputs)
    results = []
    try:
        for seed, phase in SCENARIOS:
            verified_ip(private, a.ip_receipt_sha256)
            verify_simulation_ip_files(private, copied)
            results.append(run_scenario(out, private, seed, phase, a.vivado))
            verified_ip(private, a.ip_receipt_sha256)
            verify_simulation_ip_files(private, copied)
        verified_ip(iprun, a.ip_receipt_sha256)
        if {name: build.sha(source) for name, source in SOURCES.items()} != sources_before:
            raise RuntimeError('Source changed during simulation')
        evidence_paths = [out / 'vivado_version.txt']
        for seed, phase in SCENARIOS:
            run = out / f'seed-{seed}-phase-{phase}'
            evidence_paths.extend(run / name for name in
                ('rtl', 'simulate.tcl', 'inputs.json', 'command.json', 'console.log', 'exit.json', 'result.json'))
        build.write_json(out / 'simulation-complete.json', {
            'schema': 1, 'kind': 'c2-alex-vendor-fabric-simulation', 'state': 'passed',
            'inputs_sha256': build.sha(out / 'inputs.json'), 'results': results,
            'evidence': build.inventory(evidence_paths, out),
            'qualification': 'protocol-simulation-only-not-physical-timing', 'hardware_actions': False})
    except BaseException as exc:
        build.write_json(out / 'failure.json', {'error': str(exc), 'completed_scenarios': results})
        raise


if __name__ == '__main__':
    main()
