#!/usr/bin/env python3
"""Freeze the F2 CL and run the pinned AWS HDK synthesis/implementation flow."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from opentpu import target  # noqa: E402


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check_vivado_version(output: str, expected: str) -> None:
    # Linux's launcher prints lowercase "vivado"; other releases/platforms
    # capitalize it. Match the tool banner, not an unrelated build timestamp.
    if not re.search(rf'\bvivado\s+v?{re.escape(expected)}\b', output, re.IGNORECASE):
        raise RuntimeError(f'This target requires Vivado {expected}')


def check_hdk(hdk: Path, t: target.Target, *, shell: bool) -> Path:
    for path, wanted in [(hdk, t.platform['hdk_commit']),
                         (hdk / 'hdk/common/ip', t.platform['ip_commit'])]:
        actual = command('git', '-C', str(path), 'rev-parse', 'HEAD')
        if actual != wanted:
            raise RuntimeError(f'{path}: revision {actual}, expected {wanted}')
    sh = hdk / 'hdk/common/shell_stable'
    for rel in ['design/interfaces/cl_ports.vh', 'build/scripts/build_all.tcl']:
        if not (sh / rel).is_file():
            raise RuntimeError(f'Missing HDK input: {sh / rel}; source hdk_setup.sh')
    versions = dict(line.strip().split('=', 1) for line in
                    (sh / 'shell_version.txt').read_text().splitlines() if '=' in line)
    if versions.get('small_shell') != t.platform['shell_version']:
        raise RuntimeError(f'{sh}: small-shell version does not match the target')
    ip = hdk / 'hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_ddr4/cl_ddr4.xci'
    if not ip.is_file():
        raise RuntimeError(f'Missing DDR IP: {ip}; source hdk_setup.sh')
    ip_params = json.loads(ip.read_text())['ip_inst']['parameters']['component_parameters']
    if ip_params['C0.DDR4_TimePeriod'][0]['value'] != '938':
        raise RuntimeError(f'{ip}: unexpected DDR clock period (expected 938 ps)')
    generated = hdk / 'hdk/common/ip/cl_ip/cl_ip.gen/sources_1/ip/cl_ddr4'
    for rel in ('cl_ddr4.xml', 'rtl/ip_top/cl_ddr4.sv',
                'sw/calibration_0/Debug/calibration_ddr.elf'):
        path = generated / rel
        if not path.is_file():
            raise RuntimeError(f'Missing DDR generated product: {path}')
        with path.open('rb') as f:
            header = f.read(128)
        if header.startswith(b'version https://git-lfs.github.com/spec/'):
            raise RuntimeError(f'{path}: Git LFS pointer; run git lfs pull in hdk/common/ip')
        if path.suffix == '.elf' and not header.startswith(b'\x7fELF'):
            raise RuntimeError(f'{path}: not an ELF calibration image')
    if shell:
        dcp = sh / 'build/checkpoints/from_aws/cl_bb_routed.small_shell.dcp'
        expected = '04a5331321e86cffc7ef740fb6132b29f179e68e2d58ccae5031c989403ecca6'
        if not dcp.is_file() or sha(dcp) != expected:
            raise RuntimeError(f'{dcp}: missing or wrong small-shell checkpoint')
    return sh


def prepare(out: Path, sh: Path, t: target.Target, build_id: str) -> Path:
    if out.exists():
        raise RuntimeError(f'{out} already exists; choose a new run directory')
    if t.core_khz != 100000 or t.arch['S'] != 1 or t.platform['channel_bytes'] != 2**31:
        raise RuntimeError('F2 wrapper requires 100 MHz, one slice, 2 GiB per channel')
    cl = out / 'cl_opentpu'
    shutil.copytree(ROOT / 'boards/f2/design', cl / 'design')
    shutil.copytree(ROOT / 'boards/f2/build', cl / 'build')
    files = target.sources(t)
    for name in files:
        dest = cl / 'design/rtl' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'rtl' / name, dest)
    params = target.rtl_params(t)
    for name in ('S', 'NCH', 'CH_LOG2', 'DRAM_LOG2'):
        params.pop(name)
    params.update(CORE_KHZ=t.core_khz, BUILD_ID=int(build_id, 16),
                  CH_BYTES=t.platform['channel_bytes'])
    actual_params = set(re.findall(r'parameter\s+(?:int|bit|logic(?:\s*\[[^\]]+\])?)\s+(\w+)',
                                  (ROOT / 'rtl/boards/f2/otpu_f2_sys.sv').read_text()))
    if set(params) - actual_params:
        raise RuntimeError(f'Unknown RTL parameters: {set(params) - actual_params}')
    bit_params = set(re.findall(r'parameter\s+bit\s+(\w+)',
                        (ROOT / 'rtl/boards/f2/otpu_f2_sys.sv').read_text()))
    (cl / 'design/opentpu_params.vh').write_text(',\n'.join(
        f".{k}({1 if k in bit_params else 64 if k == 'CH_BYTES' else 32}'h{v:x})"
        for k, v in params.items()) + '\n')
    (cl / 'design/sources.tcl').write_text('set otpu_sources {\n' +
        ''.join(f'  rtl/{f}\n' for f in files) + '}\n')
    shutil.copy2(ROOT / 'boards/ypcb-00338/constraints/otpu_clk2x.tcl',
                 cl / 'build/constraints/otpu_clk2x.tcl')
    for name in ('build_all.tcl',):
        (cl / 'build/scripts' / name).symlink_to(sh / 'build/scripts' / name)
    provenance = t.to_dict() | {'build_id': build_id,
        'git_commit': command('git', '-C', str(ROOT), 'rev-parse', 'HEAD'),
        'git_status': command('git', '-C', str(ROOT), 'status', '--porcelain'),
        'rtl_parameters': params,
        'sources': {str(p.relative_to(cl)): sha(p) for p in sorted(cl.rglob('*')) if p.is_file()}}
    (out / 'inputs.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return cl


def load_synthesis(run: Path, expected_sha: str) -> tuple[dict, Path, str]:
    """Accept a completed normal synthesis, never an unvalidated recovery DCP."""
    if not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha):
        raise RuntimeError('--synth-sha256 must be a SHA256 digest')
    info = json.loads((run / 'inputs.json').read_text())
    build_id = info['build_id']
    if not re.fullmatch(r'[0-9a-fA-F]{8}', build_id):
        raise RuntimeError('Invalid synthesis build ID')
    log = (run / 'console.log').read_text()
    if (run / 'exit_code').read_text().strip() != '0' or not re.search(
            r'^OTPU_STAGE_SYNTHESIS_PASSED$', log, re.M):
        raise RuntimeError('Parent synthesis did not complete successfully')
    sentinel = re.search(r"^OTPU_DDR_CALIBRATION_INIT_2C=256'h([0-9a-fA-F]{64})$", log, re.M)
    if not sentinel or not int(sentinel[1], 16):
        raise RuntimeError('Parent synthesis has no verified DDR calibration sentinel')
    dcp = run / f'cl_opentpu/build/checkpoints/cl_opentpu.{build_id}.post_synth.dcp'
    if sha(dcp) != expected_sha.lower():
        raise RuntimeError('Parent synthesis checkpoint SHA256 mismatch')
    return info, dcp, expected_sha.lower()


def inherit_synthesis(out: Path, cl: Path, run: Path, info: dict, dcp: Path, expected_sha: str) -> None:
    """Require identical HDL/parameters/CDC constraints; permit new implementation scripts."""
    current = json.loads((out / 'inputs.json').read_text())
    if current['rtl_parameters'] != info['rtl_parameters']:
        raise RuntimeError('RTL parameters differ from parent synthesis')
    for name in ('platform', 'tech'):
        if current.get(name) != info.get(name):
            raise RuntimeError(f'{name} differs from parent synthesis')
    required = {n for n in current['sources'] if n.startswith('design/')} | {
        'build/constraints/cl_synth_user.xdc',
        'build/constraints/cl_timing_user.tcl',
        'build/constraints/otpu_clk2x.tcl',
    }
    parent_design = {n for n in info['sources'] if n.startswith('design/')}
    if parent_design != {n for n in required if n.startswith('design/')}:
        raise RuntimeError('RTL source inventory differs from parent synthesis')
    for name in sorted(required):
        if current['sources'].get(name) != info['sources'].get(name):
            raise RuntimeError(f'{name} differs from parent synthesis')
        if sha(run / 'cl_opentpu' / name) != info['sources'][name]:
            raise RuntimeError(f'Parent source was modified after synthesis: {name}')
    dest = cl / 'build/checkpoints' / dcp.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dcp, dest)
    digest = expected_sha
    if sha(dcp) != digest or sha(dest) != digest:
        raise RuntimeError('Copied synthesis checkpoint SHA256 mismatch')
    current['parent_synthesis'] = {'run': str(run), 'inputs_sha256': sha(run / 'inputs.json'),
        'checkpoint': str(dcp), 'checkpoint_sha256': digest}
    (out / 'inputs.json').write_text(json.dumps(current, indent=2) + '\n')


def load_routed_run(run: Path, expected_sha: str) -> tuple[dict, Path]:
    """Repackage only a completed, qualified run with an explicitly pinned DCP."""
    if not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha):
        raise RuntimeError('--route-sha256 must be a SHA256 digest')
    info = json.loads((run / 'inputs.json').read_text())
    build_id = info['build_id']
    if not re.fullmatch(r'[0-9a-fA-F]{8}', build_id):
        raise RuntimeError('Invalid routed build ID')
    if ((run / 'exit_code').read_text().strip() != '0' or not re.search(
            r'^OTPU_STAGE_ROUTE_VALIDATION_PASSED$', (run / 'console.log').read_text(), re.M)):
        raise RuntimeError('Routed run did not complete qualification successfully')
    cl = run / 'cl_opentpu'
    if sha(cl / f'build/checkpoints/cl_opentpu.{build_id}.post_route.dcp') != expected_sha.lower():
        raise RuntimeError('Routed checkpoint SHA256 mismatch')
    for name, digest in info['sources'].items():
        if sha(cl / name) != digest:
            raise RuntimeError(f'Routed source snapshot was modified: {name}')
    return info, cl


def package_afi(cl: Path, sh: Path, hdk: Path, build_id: str, env: dict) -> Path:
    """Use the pinned HDK packager, then verify its actual archive and manifest."""
    checkpoints = cl / 'build/checkpoints'
    routed = checkpoints / f'cl_opentpu.{build_id}.post_route.dcp'
    if (checkpoints / f'cl_opentpu.{build_id}.post_route.VIOLATED.dcp').exists():
        raise RuntimeError('Refusing to package a timing-violated checkpoint')
    marker = checkpoints / f'cl_opentpu.{build_id}.route_validation.passed'
    if not marker.is_file() or marker.read_text().strip() != 'OTPU_STAGE_ROUTE_VALIDATION_PASSED':
        raise RuntimeError('Post-route validation has not passed')
    # The official function uses os.system; paths containing shell metacharacters
    # are unsafe for that implementation, even though the main flow uses argv.
    for value in (str(cl), str(sh), str(hdk)):
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
            raise RuntimeError('HDK packaging requires paths without spaces/metacharacters')
    # The official packager uses its tag as manifest.date. Keep this timestamp
    # separate from the immutable eight-digit identity compiled into the RTL.
    tag = datetime.now(timezone.utc).strftime('%Y_%m_%d-%H%M%S')
    archive = checkpoints / f'{tag}.Developer_CL.tar'
    if archive.exists():
        raise RuntimeError(f'{archive} already exists; retry packaging in a later second')
    module = sh / 'build/scripts/aws_build_dcp_from_cl.py'
    dcp_sha = sha(routed)
    ids_header = cl / 'design/cl_id_defines.vh'
    ids = dict(re.findall(r"`define\s+(CL_SH_ID[01])\s+32'h([0-9a-fA-F_]+)", ids_header.read_text()))
    if set(ids) != {'CL_SH_ID0', 'CL_SH_ID1'}:
        raise RuntimeError('Missing frozen PCI identity definitions')
    id0, id1 = (int(ids[name].replace('_', ''), 16) for name in ('CL_SH_ID0', 'CL_SH_ID1'))
    versions = dict(line.split('=', 1) for line in (hdk / 'release_version.txt').read_text().splitlines()
                    if '=' in line)
    expected = {'manifest_format_version': '2', 'date': tag,
        'pci_vendor_id': f'0x{id0 & 0xffff:04x}', 'pci_device_id': f'0x{id0 >> 16:04x}',
        'pci_subsystem_vendor_id': f'0x{id1 & 0xffff:04x}', 'pci_subsystem_id': f'0x{id1 >> 16:04x}',
        'shell_version': '0x10212415', 'hdk_version': versions['RELEASE_VERSION'].strip(),
        'dcp_file_name': f'{tag}.SH_CL_routed.dcp', 'dcp_hash': dcp_sha,
        'clock_recipe_a': 'A1', 'clock_recipe_b': 'B2', 'clock_recipe_c': 'C0', 'clock_recipe_hbm': 'H2'}
    with tempfile.TemporaryDirectory(prefix='.afi-package-', dir=checkpoints) as temp:
        stage = Path(temp)
        (stage / 'design').mkdir()
        shutil.copy2(ids_header, stage / 'design/cl_id_defines.vh')
        staged_checkpoints = stage / 'build/checkpoints'
        staged_checkpoints.mkdir(parents=True)
        os.link(routed, staged_checkpoints / f'cl_opentpu.{tag}.post_route.dcp')
        probes = checkpoints / f'{build_id}.debug_probes.ltx'
        if probes.is_file():
            os.link(probes, staged_checkpoints / f'{tag}.debug_probes.ltx')
        code = ('import importlib.util,datetime; '
                's=importlib.util.spec_from_file_location("aws_package",'+repr(str(module))+'); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                'tag='+repr(tag)+'; '
                'assert datetime.datetime.strptime(tag,m.TIMESTAMP_FILE_FORMAT).strftime(m.TIMESTAMP_FILE_FORMAT)==tag; '
                'm.generate_dcp_tarball("cl_opentpu",tag,"A1","B2","C0","H2")')
        subprocess.run([sys.executable, '-c', code], cwd=staged_checkpoints,
                       env=env | {'AWS_FPGA_REPO_DIR': str(hdk), 'CL_DIR': str(stage)}, check=True)
        staged_archive = staged_checkpoints / archive.name
        with tarfile.open(staged_archive) as tar:
            members = {m.name.removeprefix('./'): m for m in tar.getmembers() if m.isfile()}
            manifest_member = members[f'to_aws/{tag}.manifest.txt']
            manifest = dict(line.split('=', 1) for line in
                            tar.extractfile(manifest_member).read().decode().splitlines() if '=' in line)
            for key, value in expected.items():
                if manifest.get(key, '').lower() != value.lower():
                    raise RuntimeError(f'AFI manifest {key} mismatch: {manifest.get(key)!r}, expected {value!r}')
            if manifest.get('tool_version', '').lower().removeprefix('v') != env['VIVADO_TOOL_VERSION']:
                raise RuntimeError('AFI manifest tool_version mismatch')
            with tar.extractfile(members['to_aws/' + manifest['dcp_file_name']]) as stream:
                embedded_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            if embedded_sha != dcp_sha or sha(routed) != dcp_sha:
                raise RuntimeError('AFI archive checkpoint/hash mismatch')
        with staged_archive.open('rb') as source, archive.open('xb') as dest:
            shutil.copyfileobj(source, dest)
    provenance = {'build_id': build_id, 'package_tag': tag, 'verified_manifest': manifest,
        'source_run': str(cl.parent), 'inputs_sha256': sha(cl.parent / 'inputs.json'),
        'route_validation_marker_sha256': sha(marker),
        'routed_checkpoint': str(routed), 'routed_checkpoint_sha256': dcp_sha,
        'archive': str(archive), 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
        'official_packager_sha256': sha(module), 'build_script_sha256': sha(Path(__file__))}
    (checkpoints / f'{tag}.packaging.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return archive


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hdk', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='fresh run directory; existing run for --package-only')
    p.add_argument('--flow', choices=['SynthCL', 'ImplCL', 'BuildAll'], default='SynthCL')
    p.add_argument('--from-synth', type=Path, help='successful SynthCL run directory for ImplCL')
    p.add_argument('--synth-sha256', help='expected parent post_synth.dcp digest')
    p.add_argument('--package', action='store_true', help='package validated routed DCP for AFI submission')
    p.add_argument('--package-only', action='store_true', help='package a completed run without running Vivado')
    p.add_argument('--route-sha256', help='expected routed checkpoint digest; required with --package-only')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--build-id', default=None, help='eight hexadecimal digits; defaults to HEAD')
    a = p.parse_args()
    t = target.load('f2')
    hdk, out = a.hdk.resolve(), a.out.resolve()
    if a.package_only:
        if (not a.route_sha256 or a.package or a.prepare_only or a.from_synth or
                a.synth_sha256 or a.build_id or a.flow != 'SynthCL'):
            p.error('--package-only requires --route-sha256 and does not accept build options')
        sh = check_hdk(hdk, t, shell=True)
        check_vivado_version(command('vivado', '-version'), t.platform['vivado'])
        info, cl = load_routed_run(out, a.route_sha256)
        env = dict(os.environ, CL_DIR=str(cl), HDK_SHELL_DIR=str(sh), SHELL_MODE='small_shell',
                   VIVADO_TOOL_VERSION=t.platform['vivado'])
        archive = package_afi(cl, sh, hdk, info['build_id'], env)
        print(f'Packaged completed build {info["build_id"]}: {archive}')
        return
    if a.route_sha256:
        p.error('--route-sha256 is only valid with --package-only')
    if (a.flow == 'ImplCL') != bool(a.from_synth and a.synth_sha256):
        p.error('ImplCL requires --from-synth and --synth-sha256; other flows do not accept them')
    if a.flow != 'ImplCL' and (a.from_synth or a.synth_sha256):
        p.error('--from-synth and --synth-sha256 are only valid with ImplCL')
    if a.package and a.flow == 'SynthCL':
        p.error('--package requires ImplCL or BuildAll')
    sh = check_hdk(hdk, t, shell=a.flow != 'SynthCL')
    parent = load_synthesis(a.from_synth.resolve(), a.synth_sha256) if a.from_synth else None
    build_id = (parent[0]['build_id'] if parent else a.build_id or
                command('git', '-C', str(ROOT), 'rev-parse', '--short=8', 'HEAD'))
    if parent and a.build_id and a.build_id != build_id:
        p.error('--build-id must match the inherited synthesis build ID')
    if not re.fullmatch(r'[0-9a-fA-F]{8}', build_id):
        p.error('--build-id must contain exactly eight hexadecimal digits')
    cl = prepare(out, sh, t, build_id)
    if parent:
        inherit_synthesis(out, cl, a.from_synth.resolve(), *parent)
    env = dict(os.environ, CL_DIR=str(cl), CL='cl_opentpu', SHELL_MODE='small_shell',
        BUILD_FLOW=a.flow, BUILD_TAG=build_id, ENCRYPT='0',
        HDK_SHELL_DIR=str(sh), HDK_SHELL_DESIGN_DIR=str(sh / 'design'),
        HDK_COMMON_DIR=str(hdk / 'hdk/common'),
        HDK_IP_SRC_DIR=str(hdk / 'hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/ip'),
        HDK_BD_SRC_DIR=str(hdk / 'hdk/common/ip/cl_ip/cl_ip.srcs/sources_1/bd'),
        HDK_BD_GEN_DIR=str(hdk / 'hdk/common/ip/cl_ip/cl_ip.gen/sources_1/bd'),
        VIVADO_TOOL_VERSION=t.platform['vivado'])
    cmd = ['vivado', '-mode', 'batch', '-source', 'build_all.tcl', '-log', str(out / 'vivado.log'),
        '-journal', str(out / 'vivado.jou'), '-tclargs', 'SSI_SpreadLogic_high',
        'AggressiveExplore', 'AggressiveExplore', 'A1', 'B2', 'C0', 'H2']
    (out / 'invocation.json').write_text(json.dumps({'command': cmd,
        'cwd': str(cl / 'build/scripts'),
        'environment': {k: env[k] for k in env if k.startswith('HDK_') or
            k in ('CL_DIR', 'CL', 'SHELL_MODE', 'BUILD_FLOW', 'BUILD_TAG', 'ENCRYPT', 'VIVADO_TOOL_VERSION')}}, indent=2) + '\n')
    if a.prepare_only:
        print(f'Prepared {cl}; invocation: {out / "invocation.json"}')
        return
    version = command('vivado', '-version')
    check_vivado_version(version, t.platform['vivado'])
    with (out / 'console.log').open('w') as log:
        result = subprocess.run(cmd, cwd=cl / 'build/scripts', env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    (out / 'exit_code').write_text(str(result.returncode) + '\n')
    stage = 'post_synth' if a.flow == 'SynthCL' else 'post_route'
    checkpoint = cl / f'build/checkpoints/cl_opentpu.{build_id}.{stage}.dcp'
    if result.returncode or not checkpoint.is_file():
        raise RuntimeError(f'{a.flow} failed; see {out / "console.log"}')
    if a.flow != 'SynthCL' and not (checkpoint.parent /
            f'cl_opentpu.{build_id}.route_validation.passed').is_file():
        raise RuntimeError(f'Post-route validation failed; see {out / "console.log"}')
    artifacts = [checkpoint]
    if a.package:
        archive = package_afi(cl, sh, hdk, build_id, env)
        artifacts.extend([archive, archive.with_name(
            archive.name.removesuffix('.Developer_CL.tar') + '.packaging.json')])
    (out / 'outputs.json').write_text(json.dumps({'flow': a.flow,
        'inputs_sha256': sha(out / 'inputs.json'),
        'artifacts': {str(f.relative_to(out)): {'sha256': sha(f), 'bytes': f.stat().st_size}
                      for f in artifacts}}, indent=2) + '\n')
    print(f'{a.flow} complete: {checkpoint}')
    print('Review timing/CDC/methodology and DDR calibration reports before hardware deployment.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as e:
        sys.exit(str(e))
