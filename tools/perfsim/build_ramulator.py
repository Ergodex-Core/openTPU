#!/usr/bin/env python3
"""Build pinned dependencies in a private directory. --fetch explicitly permits downloads.

No sudo, system install, credentials, cloud resources or implicit simple-memory fallback.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def call(args,cwd=None):
    subprocess.run(list(map(str,args)),cwd=cwd,check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--build',type=Path,default=ROOT/'build/perfsim/bridge')
    ap.add_argument('--fetch',action='store_true')
    ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args()
    source=args.source.resolve(); build=args.build.resolve()
    lock=json.loads((HERE/'dependencies.lock.json').read_text())
    def checkout(path,dep):
        if not path.exists():
            if not args.fetch: raise SystemExit(f'{path} missing; use --fetch to download pinned public sources')
            call(['git','clone',dep['url'],path])
        head=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if head!=dep['commit']:
            # This must be a disposable checkout. Never discard local edits.
            if subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip():
                raise SystemExit(f'{path} has edits and wrong revision; refusing checkout')
            call(['git','-C',path,'checkout','--detach',dep['commit']])
    checkout(source,lock['ramulator'])
    for name,dep in lock['ramulator_dependencies'].items():
        checkout(source/'ext'/name,dep)
    call([sys.executable,HERE/'ramulator/prepare.py',source])
    opts=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_POLICY_VERSION_MINIMUM=3.5','-DFETCHCONTENT_FULLY_DISCONNECTED=ON']
    for name in lock['ramulator_dependencies']:
        opts.append(f'-DFETCHCONTENT_SOURCE_DIR_{name.upper()}={source / "ext" / name}')
    arch=[f'-DCMAKE_OSX_ARCHITECTURES={platform.machine()}'] if sys.platform=='darwin' else []
    call(['cmake','-S',source,'-B',source/'build',*opts,*arch])
    call(['cmake','--build',source/'build','-j',args.jobs])
    # A build-local snapshot avoids synced-folder transient reads by CMake.
    snapshot=build/'bridge-source';snapshot.mkdir(parents=True,exist_ok=True)
    snapshot_hashes={}
    for name in ('CMakeLists.txt','bridge.cpp'):
        raw=(HERE/'ramulator'/name).read_bytes()
        if not raw:raise RuntimeError(f'empty bridge source: {name}')
        (snapshot/name).write_bytes(raw);snapshot_hashes[name]=hashlib.sha256(raw).hexdigest()
    call(['cmake','-S',snapshot,'-B',build,f'-DRAMULATOR_ROOT={source}',*arch])
    call(['cmake','--build',build,'-j',args.jobs])
    libraries={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in build.glob('*opentpu_ramulator*') if p.is_file()}
    if not libraries:raise RuntimeError('CMake returned success without a bridge library')
    diff=subprocess.check_output(['git','-C',str(source),'diff','--binary'])
    receipt={'lock':lock,'python_architecture':platform.machine(),'patch_sha256':hashlib.sha256(diff).hexdigest(),
             'bridge_sources':snapshot_hashes,'libraries':libraries}
    (build/'dependency-build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(build)

if __name__=='__main__': main()
