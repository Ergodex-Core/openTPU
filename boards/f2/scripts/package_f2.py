#!/usr/bin/env python3
"""Create an AWS DCP submission tar locally; does not upload or create an AFI.

Run on F2 after reviewing the independent route audit. The review JSON binds
the reviewed checkpoint to its SHA-256 and records timing, routing, DRC and CDC
conclusions. The official installed HDK supplies manifest/package generation.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, re, shutil, sys, tarfile

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--hdk-root', type=Path, required=True)
    p.add_argument('--ids', type=Path, required=True)
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.tag):
        p.error('Tag may contain letters, numbers, underscore and hyphen only')
    checkpoint = a.checkpoint.resolve(strict=True)
    review_path = a.review.resolve(strict=True)
    ids = a.ids.resolve(strict=True)
    review = json.loads(review_path.read_text())
    digest = sha256(checkpoint)
    if review.get('checkpoint_sha256') != digest:
        p.error('Review does not match this checkpoint')
    for gate in ('setup_hold_pulse_width', 'fully_routed', 'drc', 'cdc', 'bus_skew', 'constraints'):
        if review.get(gate, {}).get('pass') is not True:
            p.error('Review gate not passed: ' + gate)
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cl = out / 'cl'
    checkpoints = cl / 'build/checkpoints'
    checkpoints.mkdir(parents=True)
    (cl / 'design').mkdir()
    shutil.copy2(ids, cl / 'design/cl_id_defines.vh')
    shutil.copy2(review_path, out / 'review.json')
    target = checkpoints / f'cl_opentpu.{a.tag}.post_route.dcp'
    # Hard-link immutable input on this filesystem; copy only if needed.
    try:
        os.link(checkpoint, target)
    except OSError:
        shutil.copy2(checkpoint, target)
    probes = review_path.parent / 'debug_probes.ltx'
    if probes.exists():
        shutil.copy2(probes, checkpoints / f'{a.tag}.debug_probes.ltx')
    hdk = a.hdk_root.resolve(strict=True)
    shell = hdk / 'hdk/common/shell_stable'
    official = shell / 'build/scripts/aws_build_dcp_from_cl.py'
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('aws_dcp_package', official)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.environ.update(CL_DIR=str(cl), HDK_SHELL_DIR=str(shell),
                      AWS_FPGA_REPO_DIR=str(hdk), SHELL_MODE='small_shell')
    os.environ['PATH'] = '/opt/Xilinx/2025.2/Vivado/bin:' + os.environ['PATH']
    os.chdir(out)
    # Clock recipes describe the HDK defaults. The CL uses its own clock IP;
    # actual core/TMEM/HBM clocks are embedded in the reviewed checkpoint.
    module.generate_dcp_tarball('cl_opentpu', a.tag, 'A1', 'B2', 'C0', 'H2')
    archive = checkpoints / f'{a.tag}.Developer_CL.tar'
    with tarfile.open(archive) as tf:
        manifest_member = next(m for m in tf.getmembers() if m.name.endswith('.manifest.txt'))
        manifest = tf.extractfile(manifest_member).read().decode()
        (out / 'manifest.txt').write_text(manifest)
        if f'dcp_hash={digest}' not in manifest:
            raise RuntimeError('Official package hash mismatch')
        dcp_member = next(m for m in tf.getmembers() if m.name.endswith('.dcp'))
        h = hashlib.sha256()
        with tf.extractfile(dcp_member) as f:
            for block in iter(lambda: f.read(8 << 20), b''):
                h.update(block)
        if h.hexdigest() != digest:
            raise RuntimeError('Archived checkpoint differs from reviewed checkpoint')
    (out / 'package.json').write_text(json.dumps(dict(
        archive=str(archive), archive_sha256=sha256(archive),
        checkpoint_sha256=digest, official_packager=str(official),
        official_packager_sha256=sha256(official), uploaded=False), indent=2) + '\n')
    print(archive)

if __name__ == '__main__':
    main()
