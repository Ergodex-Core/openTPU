#!/usr/bin/env python3
"""Plan or explicitly execute a portable-evidence-qualified C2 Alex session.

The canonical C2 session owns locks, device discovery, image loading, cleanup and
restoration. This adapter supplies only the new evidence/image identity contract
and the thin Alex benchmark entrypoint. Default planning never touches hardware.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

S=module('c2_alex_canonical_session',ROOT/'tools/c2_f2_session.py')
E=module('c2_alex_portable_evidence',ROOT/'boards/c2_f2_alex/evidence.py')
D=module('c2_alex_deployment',ROOT/'boards/c2_f2_alex/deployment.py')
C=S.C
# Reuse the original session, including load-partial-failure and disk-full
# restoration behavior. No second copy of that critical control flow exists.
session=S.session
SOURCE_FILES=('tools/c2_f2_session.py','tools/c2_f2_alex_session.py','tools/c2_f2_alex_bench.py',
              'boards/c2_f2_alex/evidence.py','boards/c2_f2_alex/deployment.py',
              'opentpu/c2_f2.py','opentpu/host/c2_f2.py','opentpu/host/runstate.py',
              'opentpu/lens.py','opentpu/lens_app.html')


def qualify(evidence_dir,manifest_sha256,image_record,image_record_sha256):
    root=Path(os.path.abspath(evidence_dir))
    E.no_symlink(root)
    verified=E.verify_bundle(root,manifest_sha256)
    path=Path(image_record)
    E.digest(image_record_sha256)
    if path.is_symlink() or S.sha(path)!=image_record_sha256:
        raise ValueError('Available-image record differs from independently pinned SHA256')
    record=E.decode(path.read_bytes())
    q=D.verify_deployment(record,verified)
    source=root/E.relative(verified['synth_inputs_relative_path'])
    if not source.is_relative_to(root) or S.sha(source)!=q['inputs_sha256']:
        raise ValueError('Verified synthesis input locator/hash changed')
    if S.sha(path)!=image_record_sha256:
        raise ValueError('Available-image record changed during qualification')
    return dict(q,schema=1,kind='c2-alex-session-qualification',hardware_measured=False,
        build_manifest=str(source),deployment_sha256=image_record_sha256,
        evidence_directory=str(root),verified_evidence=verified,
        software_sources={name:S.sha(ROOT/name) for name in SOURCE_FILES})


def verify_aws_image(response,q):
    # Preserve the existing owner, available state, AFI/AGFI, shell and four-tag
    # checks; add the portable evidence hash as a fifth required live tag.
    result=S.verify_aws_image(response,q)
    tags=result['tags']
    if tags.get('c2:evidence_manifest_sha256')!=q['evidence_manifest_sha256']:
        raise RuntimeError('AWS image does not bind the qualified portable evidence')
    return result


class AlexLocalOps(S.LocalOps):
    def verify_image_binding(self,q):
        raw=self.command('describe-afi',['aws','ec2','describe-fpga-images','--region','us-east-1',
            '--fpga-image-ids',q['afi_id'],'--output','json','--no-cli-pager'],timeout=60)
        return verify_aws_image(json.loads(raw),q)

    def benchmark(self,bdf,lock,q,fixture,repeats,idx):
        self.require_owner(bdf)
        if S.sha(q['build_manifest'])!=q['inputs_sha256']:
            raise RuntimeError('Frozen Alex synthesis manifest changed')
        if S.sha(q['qualification_file'])!=q['qualification_sha256']:
            raise RuntimeError('Frozen Alex session qualification changed')
        for name,digest in q['software_sources'].items():
            if S.sha(ROOT/name)!=digest:raise RuntimeError('Session/benchmark source changed: '+name)
        env=dict(os.environ,OTPU_LOCK_HELD='f2-'+bdf,
                 OTPU_LOCK_PID=str(lock.holder or os.getpid()),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        self.command('benchmark',[self.python,str(ROOT/'tools/c2_f2_alex_bench.py'),
            '--fixture',str(fixture),'--bdf',bdf,'--expected-build-id','0x'+q['build_id'],
            '--qualification',q['qualification_file'],'--qualification-sha256',q['qualification_sha256'],
            '--out',str(self.out/f'benchmark-{idx:02d}'),'--repeats',str(repeats)],
            env=env,timeout=900,cancellable=True)


def prepare(args):
    if not 1<=args.repeats<=100:raise ValueError('repeats must be1..100')
    q=qualify(args.evidence_dir,args.evidence_manifest_sha256,args.image_record,args.image_record_sha256)
    loaded=[];fixtures=[]
    for path in args.fixture:
        f=C.Fixture.load(path);f.reference();loaded.append(f)
        fixtures.append(dict(fixture=str(path.resolve()),fixture_sha256=f.digest(),
            fixture_file_sha256=S.sha(path),rows=f.rows,kb=f.kb))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'fixtures').mkdir()
    for idx,(entry,f) in enumerate(zip(fixtures,loaded)):
        path=out/'fixtures'/f'{idx:02d}.npz';f.save(path)
        entry.update(frozen_path=str(path),frozen_file_sha256=S.sha(path))
    source=out/'synthesis-inputs.json';source.write_bytes(Path(q['build_manifest']).read_bytes())
    if S.sha(source)!=q['inputs_sha256']:raise RuntimeError('Synthesis input changed while freezing session')
    q['build_manifest']=str(source)
    frozen=out/'qualification.json';S.write_json(frozen,q)
    q=dict(q,qualification_file=str(frozen),qualification_sha256=S.sha(frozen))
    S.write_json(out/'plan.json',dict(qualification=q,fixtures=fixtures,repeats=args.repeats,execute=args.execute,
        stages=['fresh AWS evidence/image binding','dedicated host and canonical lock',
                'baseline image/idle check','load candidate','rescan and rediscover BDF',
                'canonical C2 benchmark with inherited lock','restore original image and identity'],
        limitation='Restores the original image, not prior HBM/ACT/model contents. SIGKILL/host failure can prevent cleanup.'))
    return out,q,fixtures


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True);p.add_argument('--evidence-manifest-sha256',required=True)
    p.add_argument('--image-record',type=Path,required=True);p.add_argument('--image-record-sha256',required=True)
    p.add_argument('--fixture',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--python',default=sys.executable)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    out,q,fixtures=prepare(args)
    if not args.execute:
        print(f'Plan only: {out/"plan.json"}; no AWS, FPGA utility, lock or BAR access');return
    ops=AlexLocalOps(out,args.python)
    try:
        with S.deferred_signals(ops):
            session(ops,q,[Path(f['frozen_path']) for f in fixtures],repeats=args.repeats)
    except BaseException as error:
        ops.restoring=True
        ops.evidence('failure evidence',lambda:S.write_json(out/'error.json',dict(
            error=f'{type(error).__name__}: {error}',
            instruction='Inspect session.json restoration status and command logs before any further FPGA action.')))
        raise
    print(f'Benchmark complete; original image restored. Evidence: {out}')

if __name__=='__main__':main()
