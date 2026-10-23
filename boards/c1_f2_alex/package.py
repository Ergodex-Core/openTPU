#!/usr/bin/env python3
"""Package a qualified portable evidence bundle with the pinned installed HDK.

No synthesis, publication, AWS calls or FPGA access. Bundled Python/Tcl is never
executed. The official packager runs in a separate interpreter and fresh output.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('c1_alex_package_evidence', HERE / 'candidate_evidence.py')
e = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e)
OFFICIAL_SHA256 = 'bc86136e98de039029ecf202d6a676c8006ab0de3dd08adfa5825ec2e0327bdf'
OFFICIAL_BYTES = 13822
HDK_REVISION = '20fe90180574464fd509c994680ec67191e9759f'
IP_REVISION = '6d32be972e6da854e61a8d3d6ec0466ab491c1b3'
MODULE = 'hdk/common/shell_stable/build/scripts/aws_build_dcp_from_cl.py'
SHELL_VERSION = 'hdk/common/shell_stable/shell_version.txt'
VIVADO = Path('/opt/Xilinx/2025.2/Vivado/bin/vivado')
SOURCES = {'official/aws_build_dcp_from_cl.py': MODULE,
           'official/shell_version.txt': SHELL_VERSION,
           'official/release_version.txt': 'release_version.txt'}
RECIPES = dict(clock_recipe_a='A1', clock_recipe_b='B2', clock_recipe_c='C0', clock_recipe_hbm='H2')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def safe_path(value):
    # The official implementation interpolates these into os.system commands.
    path = Path(os.path.abspath(value))
    need(re.fullmatch(r'/[A-Za-z0-9_./-]+', str(path)), 'Unsafe official-packager path')
    need('..' not in path.parts, 'Parent traversal in package path')
    e.no_symlink(path)
    return path


def record(path):
    return {'sha256': e.sha(path), 'bytes': path.stat().st_size}


def write_json(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True); stream.write('\n')


def git_revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def git_file(path, name):
    return subprocess.check_output(['git', '-C', str(path), 'show', HDK_REVISION + ':' + name])


def version_value(raw, key):
    rows = [line.split('=', 1)[1].strip() for line in raw.splitlines() if line.startswith(key + '=')]
    need(len(rows) == 1 and rows[0], 'Missing/duplicate version field: ' + key)
    return rows[0]


def official_inputs(hdk):
    need(git_revision(hdk) == HDK_REVISION and git_revision(hdk / 'hdk/common/ip') == IP_REVISION,
         'Installed HDK/IP revision changed')
    files = {}
    for name, relative in SOURCES.items():
        path = hdk / relative
        # shell_stable is an official HDK alias; require its target inside HDK.
        need(path.is_file() and path.resolve().is_relative_to(hdk.resolve()), 'Official input escapes HDK')
        files[name] = record(path)
    need(files['official/aws_build_dcp_from_cl.py'] == {'sha256': OFFICIAL_SHA256, 'bytes': OFFICIAL_BYTES},
         'Installed official packager is not the reviewed version')
    release = (hdk / 'release_version.txt').read_bytes()
    need(release == git_file(hdk, 'release_version.txt'), 'Installed release file differs from pinned HDK commit')
    need(version_value(release.decode(), 'RELEASE_VERSION') == '2.3.0', 'Wrong HDK release')
    need((hdk / SHELL_VERSION).read_text().strip() == 'small_shell=0x10212415', 'Wrong small-shell version')
    version = subprocess.check_output([str(VIVADO), '-version'], text=True)
    need(re.search(r'^[Vv]ivado v2025\.2(?:\s|$)', version, re.M), 'Vivado 2025.2 required')
    return {'hdk_revision': HDK_REVISION, 'ip_revision': IP_REVISION, 'sources': files,
            'vivado_version': version, 'vivado_path': str(VIVADO)}


def qualified_verifier():
    verifier = HERE / 'evidence.py'
    need(verifier.is_file() and not verifier.is_symlink(), 'C1 final semantic verifier is unavailable')
    spec = importlib.util.spec_from_file_location('c1_qualified_package_evidence', verifier)
    qualified = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualified)
    return qualified


def replay_qualified_evidence(root, manifest_sha256):
    return qualified_verifier().verify_bundle(root, manifest_sha256)


def qualified_manifest(raw, manifest_sha256):
    return qualified_verifier().manifest_data(raw, manifest_sha256)


def verified_evidence(root, manifest_sha256):
    result = replay_qualified_evidence(root, manifest_sha256)
    need(type(result.get('schema')) is int and result['schema'] == 1 and
         result.get('kind') == 'verified-c1-alex-portable-evidence' and
         result.get('route_qualified') is True and result.get('hardware_measured') is False and
         result.get('review_required') is False,
         'Qualified current-route and vendor-simulation evidence required')
    manifest = qualified_manifest((root / e.MANIFEST).read_bytes(), manifest_sha256)
    dcp = root / e.relative(result['final_dcp_relative_path'])
    need(record(dcp) == manifest['files'][result['final_dcp_relative_path']] and
         e.sha(dcp) == result['final_dcp_sha256'], 'Final qualified DCP mismatch')
    top_name = manifest['roots']['synth']['bundle_relative_path'] + '/source/rtl/boards/f2/cl_opentpu.sv'
    top = root / e.relative(top_name)
    need(record(top) == manifest['files'][top_name], 'Compiled top changed')
    text = top.read_text(); constants = {}
    for key in ('cl_sh_id0', 'cl_sh_id1', 'cl_sh_status0', 'cl_sh_status1'):
        rows = re.findall(r'\bassign\s+' + key + r"\s*=\s*32'h([0-9a-fA-F]{8})\s*;", text)
        need(len(rows) == 1, 'Missing/duplicate compiled identity: ' + key)
        constants[key] = rows[0].lower()
    need(constants == dict(cl_sh_id0='f0011d0f', cl_sh_id1='1d51fedd', cl_sh_status0='43314842',
                           cl_sh_status1=result['build_id']), 'Compiled C1 PCI/build identity differs')
    need(re.findall(r"\.BUILD_ID\(32'h([0-9a-fA-F]{8})\)", text) == [result['build_id']],
         'Kernel compiled build identity differs')
    kernel = re.findall(r'otpu_c1_hbm_kernel\s*#\s*\((.*?)\)\s*u_board\s*\(', text, re.S)
    need(len(kernel) == 1, 'Expected exactly one compiled C1 kernel instance')
    for key, value in {'CORE_KHZ': 200000, 'DEPTH': 128, 'MAX_ROWS': 512, 'RD': 32}.items():
        need(re.findall(r'\.' + key + r'\s*\(\s*([0-9]+)\s*\)', kernel[0]) == [str(value)],
             'Compiled C1 kernel geometry differs: ' + key)
    return result, dcp, constants, top_name


def expected_manifest(tag, dcp_sha256, ids):
    id0, id1 = int(ids['cl_sh_id0'], 16), int(ids['cl_sh_id1'], 16)
    return dict(manifest_format_version='2', date=tag, shell_version='0x10212415', hdk_version='2.3.0',
        tool_version='v2025.2', dcp_file_name=f'{tag}.SH_CL_routed.dcp', dcp_hash=dcp_sha256,
        pci_vendor_id=f'0x{id0 & 0xffff:04x}', pci_device_id=f'0x{id0 >> 16:04x}',
        pci_subsystem_vendor_id=f'0x{id1 & 0xffff:04x}', pci_subsystem_id=f'0x{id1 >> 16:04x}', **RECIPES)


def verify_archive(path, tag, dcp, ids):
    expected = expected_manifest(tag, e.sha(dcp), ids)
    names = {'to_aws', f'to_aws/{tag}.manifest.txt', f'to_aws/{tag}.SH_CL_routed.dcp'}
    seen = set(); raw_manifest = None
    before = record(path)
    # Scan past tar end padding so a concatenated second archive cannot hide
    # extra members from the exact-inventory check.
    with tarfile.open(path, 'r:', ignore_zeros=True) as archive:
        for member in archive:
            name = member.name.removeprefix('./')
            if member.isdir():
                name = name.removesuffix('/')
            need(name in names and name not in seen, 'Unexpected/duplicate official archive member')
            seen.add(name)
            need(not member.sparse and not any(k.startswith(('GNU.sparse', 'SCHILY.realsize')) for k in member.pax_headers),
                 'Sparse official archive member')
            if name == 'to_aws':
                need(member.isdir() and member.size == 0, 'Invalid archive directory')
                continue
            need(member.isfile(), 'Link or special member in official archive')
            stream = archive.extractfile(member)
            if name.endswith('.manifest.txt'):
                need(0 < member.size < 65536, 'Invalid manifest length')
                raw_manifest = stream.read(65536).decode()
            else:
                need(member.size == dcp.stat().st_size and hashlib.file_digest(stream, 'sha256').hexdigest() == expected['dcp_hash'],
                     'Embedded DCP differs from final qualified checkpoint')
    need(seen == names and raw_manifest is not None, 'Incomplete official archive')
    manifest = {}
    for line in raw_manifest.splitlines():
        if not line.strip():
            continue
        need('=' in line, 'Malformed official manifest')
        key, value = line.split('=', 1)
        need(key not in manifest, 'Duplicate official manifest field')
        manifest[key] = value
    need(manifest == expected, 'Official manifest identity/clock/tool/hash mismatch')
    need(record(path) == before, 'Archive changed while verifying')
    return manifest


def package(evidence_root, manifest_sha256, hdk, out, *, tag=None):
    root, hdk, out = map(safe_path, (evidence_root, hdk, out))
    need(not out.exists(), 'Refusing existing package output')
    need(not out.is_relative_to(root) and not out.is_relative_to(hdk), 'Output must not mutate evidence or installed HDK')
    evidence, dcp, ids, top_name = verified_evidence(root, manifest_sha256)
    official = official_inputs(hdk)
    tag = tag or datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d-%H%M%S')
    need(re.fullmatch(r'\d{4}_\d{2}_\d{2}-\d{6}', tag) and
         datetime.datetime.strptime(tag, '%Y_%m_%d-%H%M%S').strftime('%Y_%m_%d-%H%M%S') == tag,
         'Invalid official package timestamp')
    out.mkdir(parents=True)
    cl = out / 'cl_opentpu'; checks = cl / 'build/checkpoints'; checks.mkdir(parents=True)
    (cl / 'design').mkdir()
    header = cl / 'design/cl_id_defines.vh'
    header.write_text(f"`define CL_SH_ID0 32'h{ids['cl_sh_id0'][:4]}_{ids['cl_sh_id0'][4:]}\n"
                      f"`define CL_SH_ID1 32'h{ids['cl_sh_id1'][:4]}_{ids['cl_sh_id1'][4:]}\n")
    copied = checks / f'cl_opentpu.{tag}.post_route.dcp'
    with dcp.open('rb') as src, copied.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    need(record(copied) == record(dcp) and not os.path.samestat(copied.stat(), dcp.stat()), 'DCP copy mismatch/hardlink')
    (out / 'official').mkdir()
    for name, original in SOURCES.items():
        shutil.copyfile(hdk / original, out / name)
    (out / 'official/vivado_version.txt').write_text(official['vivado_version'])
    inputs = {'schema': 1, 'kind': 'c1-alex-package-inputs', 'evidence_manifest_sha256': manifest_sha256,
        'evidence': evidence, 'compiled_ids': ids, 'compiled_top_relative_path': top_name,
        'package_tag': tag, 'official': official, 'clocks_mhz': {'core': 200, 'host': 250, 'hbm': 450},
        'clock_recipes': RECIPES, 'packager_sha256': e.sha(Path(__file__)),
        'evidence_verifier_sha256': e.sha(HERE / 'evidence.py'), 'hardware_actions': False,
        'prepared_files': {str(p.relative_to(out)): record(p) for p in [copied, header, *[out / name for name in SOURCES], out / 'official/vivado_version.txt']}}
    write_json(out / 'package-inputs.json', inputs)
    inputs_sha = e.sha(out / 'package-inputs.json')
    code = ("import importlib.util; s=importlib.util.spec_from_file_location('official_aws_packager',"
            + repr(str(hdk / MODULE)) + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            + f"m.generate_dcp_tarball('cl_opentpu',{tag!r},'A1','B2','C0','H2')")
    command = [sys.executable, '-I', '-c', code]
    write_json(out / 'command.json', command)
    env = os.environ | {'AWS_FPGA_REPO_DIR': str(hdk), 'HDK_SHELL_DIR': str(hdk / 'hdk/common/shell_stable'),
                        'SHELL_MODE': 'small_shell', 'CL_DIR': str(cl), 'PATH': str(VIVADO.parent) + ':/usr/bin:/bin'}
    try:
        need(official_inputs(hdk) == official, 'Official inputs changed before packaging')
        with (out / 'console.log').open('x') as log:
            result = subprocess.run(command, cwd=checks, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        write_json(out / 'exit.json', {'returncode': result.returncode})
        need(result.returncode == 0, 'Official packaging process failed')
        need(official_inputs(hdk) == official, 'Official inputs changed while packaging')
        need(e.sha(out / 'package-inputs.json') == inputs_sha, 'Packaging inputs changed')
        need(verified_evidence(root, manifest_sha256)[0] == evidence, 'Evidence changed while packaging')
        for name, wanted in inputs['prepared_files'].items():
            need(record(out / name) == wanted, 'Prepared packaging file changed')
        archive = checks / f'{tag}.Developer_CL.tar'
        manifest = verify_archive(archive, tag, copied, ids)
        receipt = {'schema': 1, 'kind': 'c1-alex-official-package', 'state': 'verified',
            'inputs_sha256': inputs_sha, 'evidence_manifest_sha256': manifest_sha256,
            'source_revision': evidence['source_revision'], 'build_id': evidence['build_id'],
            'archive': str(archive.relative_to(out)), 'archive_sha256': e.sha(archive), 'archive_bytes': archive.stat().st_size,
            'checkpoint': str(copied.relative_to(out)), 'dcp_sha256': evidence['final_dcp_sha256'],
            'verified_manifest': manifest, 'official_packager_sha256': OFFICIAL_SHA256,
            'hardware_actions': False, 'outputs': {name: record(out / name) for name in ('command.json', 'console.log', 'exit.json')}}
        write_json(out / 'package.json', receipt)
        return verify_package(out, root, manifest_sha256, e.sha(out / 'package.json'))
    except Exception as error:
        write_json(out / 'failure.json', {'error': str(error), 'state': 'failed', 'hardware_actions': False})
        raise


def verify_package(package_run, evidence_root, evidence_manifest_sha256, package_receipt_sha256):
    out, root = map(safe_path, (package_run, evidence_root))
    e.digest(package_receipt_sha256)
    need(e.sha(out / 'package.json') == package_receipt_sha256, 'Package receipt SHA mismatch')
    receipt = e.decode((out / 'package.json').read_bytes())
    need(receipt.get('schema') == 1 and receipt.get('kind') == 'c1-alex-official-package' and
         receipt.get('state') == 'verified' and receipt.get('hardware_actions') is False and
         not (out / 'failure.json').exists(), 'Package is not verified')
    evidence, dcp, ids, top_name = verified_evidence(root, evidence_manifest_sha256)
    need(e.sha(out / 'package-inputs.json') == receipt['inputs_sha256'], 'Package inputs changed')
    inputs = e.decode((out / 'package-inputs.json').read_bytes())
    need(inputs.get('schema') == 1 and inputs.get('kind') == 'c1-alex-package-inputs' and
         inputs.get('evidence') == evidence and inputs.get('evidence_manifest_sha256') == evidence_manifest_sha256 and
         inputs.get('compiled_ids') == ids and inputs.get('compiled_top_relative_path') == top_name and
         inputs.get('clocks_mhz') == {'core': 200, 'host': 250, 'hbm': 450} and inputs.get('clock_recipes') == RECIPES and
         inputs.get('hardware_actions') is False, 'Package input evidence/identity/clock binding differs')
    need(inputs.get('packager_sha256') == e.sha(Path(__file__)) and
         inputs.get('evidence_verifier_sha256') == e.sha(HERE / 'evidence.py'), 'Reviewed packaging/verifier code changed')
    official = inputs['official']
    need(official.get('hdk_revision') == HDK_REVISION and official.get('ip_revision') == IP_REVISION and
         set(official.get('sources', {})) == set(SOURCES) and
         official['sources']['official/aws_build_dcp_from_cl.py'] == {'sha256': OFFICIAL_SHA256, 'bytes': OFFICIAL_BYTES},
         'Official packager/HDK identity changed')
    tag = inputs['package_tag']
    need(isinstance(tag, str) and re.fullmatch(r'\d{4}_\d{2}_\d{2}-\d{6}', tag), 'Invalid package tag')
    wanted_prepared = {*SOURCES, 'official/vivado_version.txt', 'cl_opentpu/design/cl_id_defines.vh',
                       f'cl_opentpu/build/checkpoints/cl_opentpu.{tag}.post_route.dcp'}
    need(set(inputs['prepared_files']) == wanted_prepared and
         set(receipt['outputs']) == {'command.json', 'console.log', 'exit.json'}, 'Incomplete package provenance inventory')
    for inventory in (inputs['prepared_files'], official['sources'], receipt['outputs']):
        for name, wanted in inventory.items():
            path = out / e.relative(name); e.no_symlink(path)
            need(record(path) == wanted, 'Package file changed: ' + name)
    need(version_value((out / 'official/release_version.txt').read_text(), 'RELEASE_VERSION') == '2.3.0' and
         (out / 'official/shell_version.txt').read_text().strip() == 'small_shell=0x10212415' and
         (out / 'official/vivado_version.txt').read_text() == official['vivado_version'] and
         re.search(r'^[Vv]ivado v2025\.2(?:\s|$)', official['vivado_version'], re.M), 'Preserved official version inputs differ')
    need(e.decode((out / 'exit.json').read_bytes()) == {'returncode': 0}, 'Official process did not succeed')
    archive = out / e.relative(receipt['archive']); copied = out / e.relative(receipt['checkpoint'])
    e.no_symlink(archive); e.no_symlink(copied)
    need(receipt['archive'] == f"cl_opentpu/build/checkpoints/{inputs['package_tag']}.Developer_CL.tar" and
         receipt['checkpoint'] == f"cl_opentpu/build/checkpoints/cl_opentpu.{inputs['package_tag']}.post_route.dcp",
         'Unexpected package artifact paths')
    need(record(archive) == {'sha256': receipt['archive_sha256'], 'bytes': receipt['archive_bytes']} and
         record(copied) == record(dcp) and receipt['dcp_sha256'] == evidence['final_dcp_sha256'], 'Package artifact/DCP mismatch')
    need(receipt.get('official_packager_sha256') == OFFICIAL_SHA256 and
         receipt.get('evidence_manifest_sha256') == evidence_manifest_sha256 and
         receipt.get('source_revision') == evidence['source_revision'] and receipt.get('build_id') == evidence['build_id'],
         'Package receipt provenance mismatch')
    need(verify_archive(archive, inputs['package_tag'], copied, ids) == receipt['verified_manifest'], 'Package manifest changed')
    return {'schema': 1, 'archive': str(archive), 'package_sha256': receipt['archive_sha256'],
            'package_bytes': receipt['archive_bytes'], 'dcp_sha256': evidence['final_dcp_sha256'],
            'inputs_sha256': evidence['synth_inputs_sha256'], 'evidence_manifest_sha256': evidence_manifest_sha256,
            'source_revision': evidence['source_revision'], 'build_id': evidence['build_id'],
            'ip_receipt_sha256': evidence['ip_receipt_sha256'], 'simulation_receipt_sha256': evidence['simulation_receipt_sha256'],
            'qualification_receipt_sha256': evidence['qualification_receipt_sha256'],
            'package_receipt_sha256': package_receipt_sha256}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', required=True, type=Path); p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--hdk', required=True, type=Path); p.add_argument('--out', required=True, type=Path)
    a = p.parse_args(argv)
    print(json.dumps(package(a.evidence, a.manifest_sha256, a.hdk, a.out), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
