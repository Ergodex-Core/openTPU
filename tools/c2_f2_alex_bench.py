#!/usr/bin/env python3
"""Alex session benchmark adapter; reuses canonical C2 ABI/arithmetic/Lens code.

Never loads images. Execution requires the parent session's frozen qualification
record SHA and inherited canonical lock. Offline transports cannot emit profiles.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opentpu import c2_f2 as C, lens


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_qualification(path,digest):
    if sha(path)!=digest:raise ValueError('Frozen Alex session qualification changed')
    q=json.loads(Path(path).read_text())
    if (q.get('schema')!=1 or q.get('kind')!='c2-alex-session-qualification' or
        q.get('route_qualified') is not True or q.get('hardware_measured') is not False):
        raise ValueError('Not a qualified Alex session record')
    for name,expected in q.get('software_sources',{}).items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen benchmark source changed: '+name)
    source=Path(q['build_manifest'])
    if sha(source)!=q['inputs_sha256']:raise ValueError('Frozen synthesis inputs changed')
    inputs=json.loads(source.read_text())
    if (inputs.get('kind')!='c2-alex-burst200-build-inputs' or
        inputs.get('source_revision')!=q['source_revision'] or inputs.get('build_id')!=q['build_id'] or
        (inputs.get('core_khz'),inputs.get('hbm_mhz'),inputs.get('hbm_ports'),inputs.get('channel_bytes'))!=(200000,450,8,2**29)):
        raise ValueError('Alex synthesis identity/geometry mismatch')
    if sha(path)!=digest:raise ValueError('Qualification changed during verification')
    return q


def checked_runs(transport,fixture,build_id,repeats,timeout, failure_callback=None):
    """The canonical client owns preload fences/readback, numerics and counters."""
    bench=C.Benchmark(transport,expected_build_id=build_id)
    repeat=mode=None
    try:
        bench.preload(fixture)
        for repeat in range(repeats):
            for mode in ((0,1) if repeat%2==0 else (1,0)):
                result=bench.run(mode,timeout=timeout)
                yield dict(result,repeat=repeat)
    except BaseException as exc:
        failure=getattr(exc,'c2_failure',None)
        if failure is None:
            failure=dict(kind='c2-f2-run-failure',bit_exact=False,profile_eligible=False,
                         error_type=type(exc).__name__,error=str(exc))
            try:
                if transport.reg_read(C.R_STATUS)&(C.BUSY|C.DRAINING):
                    bench.abort()
                    failure['abort']=dict(attempted=True,drained=True)
            except BaseException as cleanup_error:
                failure['cleanup_error']=dict(error_type=type(cleanup_error).__name__,error=str(cleanup_error))
        failure.update(repeat=repeat,mode=mode)
        if failure_callback is not None:
            try:
                failure_callback(failure)
            except BaseException as write_error:
                message=f'Could not persist failure evidence: {type(write_error).__name__}: {write_error}'
                if hasattr(exc,'add_note'):
                    exc.add_note(message)
                else:  # Python 3.10 is supported; reporting must preserve the original failure.
                    exc.c2_failure_write_error=message
        raise


def write_failure(path, failure, provenance):
    """Failure evidence is separate from successful results and Lens profiles."""
    record=dict(failure,source=provenance,bit_exact=False,profile_eligible=False)
    with Path(path).open('x') as stream:
        stream.write(json.dumps(record,indent=2)+'\n')


def run(args):
    if not 1<=args.repeats<=100 or args.timeout<=0:raise ValueError('Invalid repeat/timeout')
    q=read_qualification(args.qualification,args.qualification_sha256)
    if int(q['build_id'],16)!=args.expected_build_id:raise ValueError('Expected BUILD_ID differs from qualification')
    # Require the controller's retained canonical owner; do not silently start a
    # second independent session around an already loaded, unowned image.
    from opentpu.host.c2_f2 import C2F2Transport,canonical_bdf
    bdf=canonical_bdf(args.bdf)
    if os.environ.get('OTPU_RUN_DIR','/tmp/otpu')!='/tmp/otpu' or os.environ.get('OTPU_LOCK_HELD')!='f2-'+bdf or not os.environ.get('OTPU_LOCK_PID','').isdigit():
        raise RuntimeError('Alex benchmark requires the inherited canonical session lock')
    fixture=C.Fixture.load(args.fixture)
    fixture.reference()
    args.out.mkdir(parents=True,exist_ok=False)
    provenance=dict(kind='c2-alex-physical-session',qualification=q,
        qualification_sha256=args.qualification_sha256,fixture_file_sha256=sha(args.fixture),
        software_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        software_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),
        sources={str(p.relative_to(ROOT)):sha(p) for p in
                 (ROOT/'opentpu/c2_f2.py',ROOT/'opentpu/host/c2_f2.py',Path(__file__).resolve())})
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    profiles=[];results=[]
    with C2F2Transport(bdf,expected_build_id=args.expected_build_id) as transport:
        for result in checked_runs(transport,fixture,args.expected_build_id,args.repeats,args.timeout,
                                   failure_callback=lambda failure:write_failure(args.out/'failure.json',failure,provenance)):
            result['source']=provenance
            name=f"C2 Alex {'decode' if result['mode'] else 'legacy M1'} N{fixture.rows} KB{fixture.kb} repeat{result['repeat']}"
            # Canonical exporter refuses injected/model data and non-bit-exact output.
            profile=C.lens_data(result,name)
            results.append(result);profiles.append(profile)
            (args.out/f"result-{result['repeat']}-{result['mode']}.json").write_text(json.dumps(result,indent=2)+'\n')
            lens.save(profiles,args.out/'c2-f2.otpuprof')
            print(f"{name}: {result['cycles']} device cycles, {result['weight_bytes']} native bytes, exact bits",flush=True)
    (args.out/'c2-f2.html').write_text(lens.render(profiles,title='C2 Alex F2 matrix benchmark — hardware counters'))
    (args.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--bdf',required=True)
    p.add_argument('--expected-build-id',type=lambda s:int(s,0),required=True)
    p.add_argument('--qualification',type=Path,required=True);p.add_argument('--qualification-sha256',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--timeout',type=float,default=30.)
    run(p.parse_args(argv))

if __name__=='__main__':main()
