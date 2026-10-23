"""Prepare/verify the pinned common runtime using local Git data only.

No network, installation, synthesis, FPGA access or model execution. A prepared
directory is a minimal source closure for offline qualification, not a new image.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def dependency():
    return json.loads((Path(__file__).resolve().parents[1] / 'docs/phonon2/tdt_runtime_dependency.json').read_text())


def verify(runtime, manifest):
    runtime = Path(runtime)
    actual = json.loads((runtime/'source_manifest.json').read_text())
    if actual != manifest:
        raise RuntimeError('common runtime manifest mismatch')
    for name, expected in manifest['files'].items():
        if hashlib.sha256((runtime/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('common runtime source mismatch: '+name)


def prepare(checkout, destination, manifest):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('destination must be a new directory')
    # Validate all source bytes before creating any output. Git objects are read
    # as data; no checkout hooks or setup code execute.
    files = {}
    for name, expected in manifest['files'].items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('unsafe manifest path')
        raw = subprocess.run(['git','-C',str(checkout),'show',manifest['revision']+':'+name],capture_output=True,check=True).stdout
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError('Git source hash mismatch: '+name)
        files[name] = raw
    destination.mkdir(parents=True)
    for name, raw in files.items():
        path = destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    (destination/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    verify(destination,manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-root');p.add_argument('--prepare-from-git');p.add_argument('--destination')
    a=p.parse_args();manifest=dependency()['runtime_manifest']
    if a.verify_root and not a.prepare_from_git and not a.destination:
        verify(a.verify_root,manifest)
    elif a.prepare_from_git and a.destination and not a.verify_root:
        prepare(a.prepare_from_git,a.destination,manifest)
    else:p.error('choose --verify-root, or --prepare-from-git plus --destination')
    print('common runtime source hashes verified')
