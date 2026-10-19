#!/usr/bin/env python3
"""Freeze and build C2 with Alex burst200 private vendor IP. No AFI or hardware actions."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SUPPORT = HERE / 'support'
ALEX_REVISION = '06c169cbcd1721ae8b167de992a93461697a397d'
ARCHIVE_SHA256 = 'fdd38e2be3e87fe80d73b9af111911c2d26f8a699b767dfc37b62b8c5a1505dc'
HDK_REVISION = '20fe90180574464fd509c994680ec67191e9759f'
IP_REVISION = '6d32be972e6da854e61a8d3d6ec0466ab491c1b3'
SHELL_SHA256 = '04a5331321e86cffc7ef740fb6132b29f179e68e2d58ccae5031c989403ecca6'
PART = 'xcvu47p-fsvh2892-2-e'
CORE_SOURCES = ['vpu/otpu_fp.sv', 'vpu/otpu_fpipe.sv', 'top/otpu_pkg.sv',
                'tech/xilinxus/otpu_tech_dsp.sv', 'mxu/otpu_mxu.sv',
                'mxu/otpu_c2_mxu.sv', 'c2/otpu_c2_hbm_kernel.sv']
PRODUCT_SUFFIXES = {'.bd', '.v', '.sv', '.vhd', '.vhdl', '.xdc', '.xci', '.dcp',
                    '.mem', '.dat', '.coe', '.xml', '.bds'}
STAGES = ('ip', 'pcis', 'clocks', 'synth', 'link', 'implement', 'refine', 'audit')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def command(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def verify_support():
    manifest = json.loads((SUPPORT / 'manifest.json').read_text())
    if manifest['commit'] != ALEX_REVISION or manifest['archive_sha256'] != ARCHIVE_SHA256:
        raise RuntimeError('Wrong Alex source provenance')
    names = set(manifest['sources'])
    actual = {str(p.relative_to(SUPPORT)) for p in SUPPORT.rglob('*') if p.is_file()}
    if actual != names | {'manifest.json'}:
        raise RuntimeError('Frozen Alex support inventory changed')
    for name, digest in manifest['sources'].items():
        path = SUPPORT / name
        if path.is_symlink() or sha(path) != digest:
            raise RuntimeError(f'Frozen Alex source changed: {name}')
    return manifest


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Alex source anchor changed: {old[:100]!r}')
    return text.replace(old, new, 1)


def bounded_script(text, ip_jobs=1):
    if ip_jobs not in (1, 2, 3):
        raise RuntimeError('IP jobs must be 1..3')
    text = re.sub(r'set_param general\.maxThreads \d+', 'set_param general.maxThreads 4', text)
    text = re.sub(r'(launch_runs [^\n]* -jobs) \d+', rf'\g<1> {ip_jobs}', text)
    if 'launch_runs ' in text:
        # launch_runs starts child Vivado processes. Bound those processes too;
        # the parent's general.maxThreads setting alone does not bind them.
        hook = ('set hook [open $run/c2_ip_threads.tcl w]\n'
                'puts $hook {set_param general.maxThreads 4}\nclose $hook\n'
                'foreach child [get_runs -filter {IS_SYNTHESIS && NAME != synth_1}] {\n'
                '  set_property STEPS.SYNTH_DESIGN.TCL.PRE $run/c2_ip_threads.tcl $child\n}\n')
        pos = text.index('launch_runs ')
        text = text[:pos] + hook + text[pos:]
    return text


def safe_path(path):
    path = Path(path).resolve()
    if re.search(r'[\s{}\[\]$;\\]', str(path)):
        raise RuntimeError(f'Build paths must not contain whitespace or Tcl metacharacters: {path}')
    return path


def inventory(paths, base, *, products=False):
    """Hash consumed files, reject directory symlinks rather than omit their children."""
    base = Path(base)
    selected = set()
    for path in paths:
        if not path.exists():
            raise RuntimeError(f'Missing build input: {path}')
        if path.is_dir():
            if path.is_symlink():
                raise RuntimeError(f'Directory symlink is not an inventoried tree: {path}')
            for root, dirs, files in os.walk(path, followlinks=False):
                for name in dirs:
                    if (Path(root) / name).is_symlink():
                        raise RuntimeError(f'Directory symlink is not an inventoried tree: {Path(root) / name}')
                for name in files:
                    selected.add(Path(root) / name)
        else:
            selected.add(path)
    result = {}
    for path in sorted(selected):
        if products and path.suffix.lower() not in PRODUCT_SUFFIXES:
            continue
        if not path.is_file():
            raise RuntimeError(f'Nonregular input: {path}')
        result[str(path.relative_to(base))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return result


def hdk_inventory(hdk):
    tree = hdk / 'hdk'
    shell = tree / 'common/shell_stable'
    # shell_stable may itself be an installed alias. Inventory its named contents,
    # not unrelated HDK simulation caches or pre-generated vendor SmartConnect IP.
    return inventory([
        shell / 'shell_version.txt', shell / 'design/interfaces',
        shell / 'design/sh_ddr/synth', shell / 'build/constraints',
        shell / 'build/checkpoints/from_aws/cl_bb_routed.small_shell.dcp',
        tree / 'common/lib/interfaces.sv', tree / 'common/lib/lib_pipe.sv',
        tree / 'cl/examples/cl_mem_perf/design/cl_mem_hbm_wrapper.sv',
        tree / 'common/ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_hbm',
    ], hdk)


def check_hdk(hdk):
    for path, revision in [(hdk, HDK_REVISION), (hdk / 'hdk/common/ip', IP_REVISION)]:
        if command('git', '-C', str(path), 'rev-parse', 'HEAD') != revision:
            raise RuntimeError(f'Wrong installed HDK/IP revision: {path}')
    shell = hdk / 'hdk/common/shell_stable'
    if (shell / 'shell_version.txt').read_text().strip() != 'small_shell=0x10212415':
        raise RuntimeError('Wrong small-shell version')
    if sha(shell / 'build/checkpoints/from_aws/cl_bb_routed.small_shell.dcp') != SHELL_SHA256:
        raise RuntimeError('Wrong small-shell checkpoint')
    xci = hdk / 'hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_hbm/cl_hbm.xci'
    params = json.loads(xci.read_text())['ip_inst']['parameters']['component_parameters']
    for key, value in {'USER_HBM_STACK': '2', 'USER_AXI_CLK_FREQ': '450',
                       'USER_HBM_DENSITY': '16GB', 'USER_SWITCH_ENABLE_00': 'TRUE',
                       'USER_SWITCH_ENABLE_01': 'TRUE'}.items():
        if params.get(key, [{}])[0].get('value') != value:
            raise RuntimeError(f'Unexpected HBM input property {key}')
    return hdk_inventory(hdk)


def check_vivado(output):
    if not re.search(r'\bvivado\s+v?2025\.2\b', output, re.I):
        raise RuntimeError('Vivado 2025.2 is required')


def adapt_scripts(out, ip_jobs):
    original = SUPPORT / 'boards/f2'
    mapping = {'ip': 'create_f2_ip.tcl', 'pcis': 'create_pcis_ip.tcl',
               'clocks': 'create_clocks.tcl', 'synth': 'build_f2.tcl',
               'link': 'build_f2.tcl', 'implement': 'implement.tcl',
               'refine': 'refine_route.tcl', 'audit': 'audit_route.tcl'}
    for stage, name in mapping.items():
        path = ROOT / 'rtl/c2_alex/create_f2_ip.tcl' if stage == 'ip' else original / 'scripts' / name
        text = bounded_script(path.read_text(), ip_jobs)
        if stage == 'synth':
            text = replace_once(text,
                'write_checkpoint -force $run/post_synth.dcp',
                'source $run/audit_c2_alex.tcl\n::c2_alex::audit_synthesis $run\n'
                'write_checkpoint -force $run/post_synth.dcp')
            anchor = '# Linking the supplied static shell is mandatory; never substitute a shell stub.'
            text = replace_once(text, anchor,
                'set done [open $run/synth_done.txt w]; puts $done C2_ALEX_SYNTH_ONLY_COMPLETE; close $done\n'
                'puts C2_ALEX_SYNTH_ONLY_COMPLETE_NOT_QUALIFIED\nclose_project\nexit\n' + anchor)
        if stage in ('synth', 'link'):
            # C2 has no consumer for the second wizard output. No pumped-TMEM
            # exception or required 400-MHz clock survives this adaptation.
            text = replace_once(text, 'foreach {output factor} {clk_out1 1.0 clk_out2 2.0}',
                                      'foreach {output factor} {clk_out1 1.0}')
        if stage == 'audit':
            text = text.replace('{core clk_out1 tmem clk_out2}', '{core clk_out1}')
        (out / stage / 'build.tcl').write_text(text)
    shutil.copy2(HERE / 'audit_c2_alex.tcl', out / 'synth')
    floorplan = (original / 'constraints/floorplan.tcl').read_text()
    floorplan = replace_once(floorplan, 'WRAPPER/CL/u_native0 WRAPPER/CL/u_native1',
        ' '.join(f'WRAPPER/CL/u_native{p}' for p in range(8)))
    core_timing = ('# C2 has no pumped TMEM and no multicycle exceptions.\n'
                   'set core_clock [get_clocks -of_objects [get_pins WRAPPER/CL/u_core_clk/clk_out1]]\n'
                   'if {[llength $core_clock] != 1 || abs([get_property PERIOD $core_clock] - 5.0) > 0.001} '
                   '{error "C2 expected 200 MHz core clock"}\n')
    for stage in ('synth', 'link', 'implement'):
        (out / stage / 'floorplan.tcl').write_text(floorplan)
        (out / stage / 'core_timing.tcl').write_text(core_timing)
        shutil.copy2(original / 'constraints/platform_constraints.tcl', out / stage)


def source_inputs():
    paths = [ROOT / 'rtl' / name for name in CORE_SOURCES]
    paths += [ROOT / 'rtl/c2_alex/generate.py', ROOT / 'rtl/c2_alex/create_f2_ip.tcl',
              HERE / 'build.py', HERE / 'audit_c2_alex.tcl']
    return inventory(paths, ROOT)


def require_clean_source():
    status = command('git', 'status', '--porcelain', '--untracked-files=all', cwd=ROOT)
    if status:
        raise RuntimeError('Build source must be committed and clean before preparing outputs')
    return command('git', 'rev-parse', 'HEAD', cwd=ROOT)


def prepare(out, hdk, build_id, installed, ip_jobs=1):
    if out.exists():
        raise RuntimeError(f'Refusing existing output: {out}')
    support = verify_support()
    revision = require_clean_source()
    inputs = source_inputs()
    out.mkdir(parents=True)
    for stage in STAGES:
        (out / stage).mkdir()
    source = out / 'source'
    for name in CORE_SOURCES:
        dest = source / 'rtl' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'rtl' / name, dest)
    for name in ('otpu_native_axi.sv', 'otpu_hbm_geometry.sv'):
        dest = source / 'rtl/boards/f2' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SUPPORT / 'rtl/boards/f2' / name, dest)
    top = source / 'rtl/boards/f2/cl_opentpu.sv'
    subprocess.run([sys.executable, str(ROOT / 'rtl/c2_alex/generate.py'),
                    '--hdk', str(hdk / 'hdk'), '--output', str(top),
                    '--build-id', build_id], check=True)
    adapt_scripts(out, ip_jobs)
    for stage in ('synth', 'link'):
        shutil.copytree(source / 'rtl/boards/f2', out / stage / 'rtl')
        (out / stage / 'config.tcl').write_text(
            f'set part {{{PART}}}\nset core_khz 200000\nset rtl_files {{' + ' '.join(CORE_SOURCES) + '}\n')
    # Preserve small external source products; large checkpoint hashes bind the
    # installed inputs, which are rechecked before and after each tool stage.
    for name, record in installed.items():
        if record['bytes'] <= 2 * 1024 * 1024:
            dest = out / 'installed-inputs' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(hdk / name, dest)
            if sha(dest) != record['sha256']:
                raise RuntimeError(f'Installed input changed during capture: {name}')
    provenance = {'schema': 1, 'kind': 'c2-alex-burst200-build-inputs',
                  'source_revision': revision, 'source_status': '',
                  'build_id': build_id, 'part': PART, 'core_khz': 200000,
                  'hbm_mhz': 450, 'hbm_ports': 8, 'channel_bytes': 2**29,
                  'alex_revision': support['commit'], 'archive_sha256': support['archive_sha256'],
                  'support_manifest_sha256': sha(SUPPORT / 'manifest.json'),
                  'source_inputs': inputs, 'installed_inputs': installed,
                  'hdk_revision': HDK_REVISION, 'ip_revision': IP_REVISION,
                  'hdk': str(hdk), 'ip_jobs': ip_jobs, 'max_threads': 4,
                  'qualification': 'not-qualified'}
    provenance['prepared_files'] = inventory([out / 'source', *(out / s for s in STAGES)], out)
    write_json(out / 'inputs.json', provenance)
    if source_inputs() != inputs or hdk_inventory(hdk) != installed:
        raise RuntimeError('Input changed during preparation')
    return provenance


def ip_inventory(run):
    paths = [run / stage / 'ip_project' for stage in ('ip', 'pcis', 'clocks')]
    result = inventory(paths, run, products=True)
    required = ['ip/ip_project/f2_ip.srcs/sources_1/bd/f2_mem/f2_mem.bd',
                'ip/ip_project/f2_ip.gen/sources_1/bd/f2_mem/hdl/f2_mem_wrapper.v',
                'ip/ip_project/f2_ip.srcs/sources_1/ip/cl_hbm/cl_hbm.xci',
                'ip/ip_project/f2_ip.srcs/sources_1/ip/f2_ocl_cdc/f2_ocl_cdc.xci',
                'pcis/ip_project/pcis.srcs/sources_1/ip/f2_pcis_pipe/f2_pcis_pipe.xci',
                'clocks/ip_project/clocks.srcs/sources_1/ip/f2_core_clk/f2_core_clk.xci',
                'clocks/ip_project/clocks.srcs/sources_1/ip/f2_hbm_clk/f2_hbm_clk.xci']
    if not set(required) <= result.keys() or not any(name.endswith('.dcp') for name in result):
        raise RuntimeError('Private IP products are incomplete')
    return result


def quantity(value):
    text = str(value)
    match = re.fullmatch(r'(\d+)([KMG])', text)
    if match:
        return int(match[1]) * 1024 ** ('KMG'.index(match[2]) + 1)
    return int(text, 0)


def validate_memory_bd(path):
    """Inspect the generated BD, including effective decode rather than Tcl intent."""
    d = json.loads(Path(path).read_text())['design']
    def require(ok, detail):
        if not ok:
            raise RuntimeError(f'Generated C2 SmartConnect mismatch: {detail}')
    def param(obj, key):
        return obj.get('parameters', {}).get(key, {}).get('value')
    require(d['design_info']['name'] == 'f2_mem' and
            d['design_info']['device'] == PART and
            d['design_info']['tool_version'] == '2025.2' and
            d['design_info']['validated'] == 'true', 'design identity/validation')
    require(set(d['components']) == {'memory_sc'}, 'component inventory')
    sc = d['components']['memory_sc']
    require(sc['vlnv'].startswith('xilinx.com:ip:smartconnect:'), 'vendor IP')
    for key, value in {'NUM_SI': '9', 'NUM_MI': '8', 'NUM_CLKS': '3'}.items():
        require(param(sc, key) == value, key)
    expected = {'S_HOST': ('Slave', 'AXI4', '512', '64', '16', '250000000', 'S00_AXI')}
    for p in range(8):
        expected[f'S_N{p}'] = ('Slave', 'AXI4', '512', '34', '1', '200000000', f'S{p+1:02d}_AXI')
        expected[f'M_H{p}'] = ('Master', 'AXI3', '256', '34', '0', '450000000', f'M{p:02d}_AXI')
    require(set(d['interface_ports']) == set(expected), 'interface inventory')
    nets = [set(net['interface_ports']) for net in d['interface_nets'].values()]
    require(len(nets) == len(expected), 'interface connection count')
    for name, (mode, protocol, width, addr, ident, freq, internal) in expected.items():
        iface = d['interface_ports'][name]
        require(iface['mode'] == mode, f'{name} direction')
        for key, val in {'PROTOCOL': protocol, 'DATA_WIDTH': width, 'ADDR_WIDTH': addr,
                         'ID_WIDTH': ident, 'FREQ_HZ': freq, 'MAX_BURST_LENGTH': '16'}.items():
            require(param(iface, key) == val, f'{name}/{key}')
        require({name, 'memory_sc/' + internal} in nets, f'{name} connectivity')
    ports = d['ports']
    clocks = [('core_clk', '200000000', 'aclk', ':'.join(f'S_N{p}' for p in range(8))),
              ('host_clk', '250000000', 'aclk1', 'S_HOST'),
              ('hbm_clk', '450000000', 'aclk2', ':'.join(f'M_H{p}' for p in range(8)))]
    net_ports = [set(net['ports']) for net in d['nets'].values()]
    for name, freq, pin, associated in clocks:
        require(ports[name]['type'] == 'clk' and param(ports[name], 'FREQ_HZ') == freq,
                f'{name} frequency')
        require(set(str(param(ports[name], 'ASSOCIATED_BUSIF')).split(':')) == set(associated.split(':')),
                f'{name} associated interfaces')
        require({name, 'memory_sc/' + pin} in net_ports, f'{name} net')
    require(param(ports['aresetn'], 'POLARITY') == 'ACTIVE_LOW' and
            {'aresetn', 'memory_sc/aresetn'} in net_ports, 'reset')
    spaces = d['addressing']['/']['address_spaces']
    require(set(spaces) == {'S_HOST', *(f'S_N{p}' for p in range(8))}, 'address-space inventory')
    for source, space in spaces.items():
        targets = range(8) if source == 'S_HOST' else [int(source[3:])]
        want = {(f'/M_H{p}/Reg', p * 2**29, 2**29) for p in targets}
        segments = space['segments'].values()
        actual = {(seg['address_block'], quantity(seg['offset']), quantity(seg['range'])) for seg in segments}
        require(len(space['segments']) == len(want) and actual == want, f'{source} decode windows')
    return {'interfaces': len(expected), 'slave_interfaces': 9, 'master_interfaces': 8,
            'clocks_mhz': [200, 250, 450], 'channel_bytes': 2**29,
            'bd_sha256': sha(path), 'qualification': 'configuration-only-not-timing'}


def verify_ip_run(run, expected):
    receipt = json.loads((run / 'ip-complete.json').read_text())
    origin = json.loads((run / 'inputs.json').read_text())
    if receipt.get('kind') != 'c2-alex-private-ip' or receipt.get('inputs_sha256') != sha(run / 'inputs.json'):
        raise RuntimeError('Private IP receipt does not bind inputs')
    for key in ('alex_revision', 'archive_sha256', 'support_manifest_sha256', 'installed_inputs',
                'hdk_revision', 'ip_revision'):
        if origin[key] != expected[key]:
            raise RuntimeError(f'Private IP input mismatch: {key}')
    for stage in ('ip', 'pcis', 'clocks'):
        name = f'{stage}/build.tcl'
        if origin['prepared_files'][name] != expected['prepared_files'][name] or sha(run / name) != origin['prepared_files'][name]['sha256']:
            raise RuntimeError(f'Private IP recipe mismatch: {name}')
        if (run / stage / 'ip_done.txt').read_text().strip() != 'IP_DONE':
            raise RuntimeError('Private IP completion marker is invalid')
    actual = ip_inventory(run)
    if actual != receipt.get('products'):
        raise RuntimeError('Private IP product inventory changed')
    validate_memory_bd(run / 'ip/ip_project/f2_ip.srcs/sources_1/bd/f2_mem/f2_mem.bd')
    return receipt


def run_stage(out, stage, argv, vivado, hdk, installed):
    inputs = json.loads((out / 'inputs.json').read_text())
    def verify_prepared():
        for name, record in inputs['prepared_files'].items():
            if sha(out / name) != record['sha256'] or (out / name).stat().st_size != record['bytes']:
                raise RuntimeError(f'Prepared build input changed: {name}')
    verify_prepared()
    if hdk_inventory(hdk) != installed:
        raise RuntimeError('Installed inputs changed before Vivado')
    cmd = [vivado, '-mode', 'batch', '-source', str(out / stage / 'build.tcl'), '-tclargs', *map(str, argv)]
    write_json(out / stage / 'command.json', cmd)
    with (out / stage / 'console.log').open('x') as log:
        result = subprocess.run(cmd, cwd=out / stage, stdout=log, stderr=subprocess.STDOUT)
    write_json(out / stage / 'exit.json', {'returncode': result.returncode})
    verify_prepared()
    if hdk_inventory(hdk) != installed:
        raise RuntimeError('Installed inputs changed during Vivado')
    if result.returncode:
        raise RuntimeError(f'{stage} failed; inspect {out / stage / "console.log"}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hdk', type=Path, required=True, help='Pinned aws-fpga repository root (parent of hdk/)')
    ap.add_argument('--out', type=Path, required=True, help='Fresh isolated output directory')
    ap.add_argument('--stage', choices=('ip', 'synth'), required=True)
    ap.add_argument('--ip-run', type=Path, help='Completed --stage ip output, required for synth')
    ap.add_argument('--ip-jobs', type=int, choices=(1, 2, 3), default=1,
                    help='Concurrent IP synth jobs; use 2/3 only with host resources available')
    ap.add_argument('--vivado', default='vivado')
    ap.add_argument('--plan', action='store_true', help='Read-only plan; no Vivado, git/HDK commands, or output creation')
    a = ap.parse_args(argv)
    if (a.stage == 'synth') != bool(a.ip_run):
        ap.error('--ip-run is required only for --stage synth')
    verify_support()
    hdk, out = safe_path(a.hdk), safe_path(a.out)
    if a.plan:
        print(json.dumps({'stage': a.stage, 'out': str(out), 'hdk': str(hdk),
              'ip_run': str(a.ip_run) if a.ip_run else None, 'max_threads': 4,
              'ip_jobs': a.ip_jobs, 'alex_revision': ALEX_REVISION,
              'actions': ['private memory/OCL IP', 'private PCIS IP', 'private clock IP']
              if a.stage == 'ip' else ['verify private IP hashes', 'C2 out-of-context synthesis only'],
              'qualification': 'not-qualified', 'hardware_actions': False}, indent=2))
        return
    installed = check_hdk(hdk)
    version = command(a.vivado, '-version')
    check_vivado(version)
    build_id = command('git', 'rev-parse', '--short=8', 'HEAD', cwd=ROOT)
    if not re.fullmatch('[0-9a-f]{8}', build_id):
        raise RuntimeError('Expected an eight-digit git build ID')
    inputs = prepare(out, hdk, build_id, installed, a.ip_jobs)
    (out / 'vivado_version.txt').write_text(version + '\n')
    if a.stage == 'ip':
        for stage, args in [('ip', [out / 'ip', hdk / 'hdk', a.ip_jobs]), ('pcis', [out / 'pcis']), ('clocks', [out / 'clocks'])]:
            run_stage(out, stage, args, a.vivado, hdk, installed)
            if (out / stage / 'ip_done.txt').read_text().strip() != 'IP_DONE':
                raise RuntimeError(f'{stage} completion marker missing')
        bd_report = validate_memory_bd(out / 'ip/ip_project/f2_ip.srcs/sources_1/bd/f2_mem/f2_mem.bd')
        write_json(out / 'memory-bd-validation.json', bd_report)
        write_json(out / 'ip-complete.json', {'schema': 1, 'kind': 'c2-alex-private-ip',
                   'inputs_sha256': sha(out / 'inputs.json'), 'products': ip_inventory(out),
                   'qualification': 'not-qualified'})
        print(f'Private IP complete: {out}; no C2 timing qualification')
    else:
        ip_run = safe_path(a.ip_run)
        receipt = verify_ip_run(ip_run, inputs)
        write_json(out / 'ip-binding.json', {'run': str(ip_run), 'receipt_sha256': sha(ip_run / 'ip-complete.json')})
        run_stage(out, 'synth', [out / 'synth', ip_run / 'ip', ip_run / 'pcis', out / 'source',
                                 hdk / 'hdk', ip_run / 'clocks', 'synth_only'], a.vivado, hdk, installed)
        if (out / 'synth/synth_done.txt').read_text().strip() != 'C2_ALEX_SYNTH_ONLY_COMPLETE':
            raise RuntimeError('C2 synthesis marker missing')
        verify_ip_run(ip_run, inputs)
        checkpoint = out / 'synth/post_synth.dcp'
        write_json(out / 'synth-complete.json', {'schema': 1, 'kind': 'c2-alex-synthesis-only',
                   'inputs_sha256': sha(out / 'inputs.json'), 'checkpoint_sha256': sha(checkpoint),
                   'ip_receipt_sha256': sha(ip_run / 'ip-complete.json'), 'qualification': 'not-qualified'})
        print(f'C2 synthesis complete: {checkpoint}; shell linking, route and qualification remain')


if __name__ == '__main__':
    main()
