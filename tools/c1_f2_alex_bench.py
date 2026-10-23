#!/usr/bin/env python3
"""C1 child runner: requires a live inherited canonical lock; no image load."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from opentpu import c1_f2 as C
from opentpu.host.c1_f2 import C1F2Transport,canonical_bdf
from boards.c1_f2_alex import session_results as F

def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def failure_snapshot(transport):
    """Bounded CSR reads only; failed snapshots never become performance profiles."""
    fields={'id':C.R_ID,'abi':C.R_ABI,'build_id':C.R_BUILD,'status':C.R_STATUS,'error':C.R_ERROR,'count':C.R_COUNT,'mode':C.R_MODE,'kb':C.R_KB,'rows':C.R_ROWS,'base_beat':C.R_BASE,'hbm_ready':C.R_HBM_READY,'core_khz':C.R_CORE_KHZ,'max_rows':C.R_MAX_ROWS,'depth':C.R_DEPTH,'credits':C.R_CREDITS}
    errors={};reads=0
    def read(label,address):
        nonlocal reads
        reads+=1
        try:return transport.reg_read(address)
        except BaseException as exc:errors[label]=f'{type(exc).__name__}: {exc}';return None
    values={name:read(name,reg) for name,reg in fields.items()}
    counters={name:read('counter.'+name,reg) for name,reg in C.COUNTERS.items()}
    pcs=[{name:read(f'pc{pc}.'+name,0x100+pc*0x20+offset) for name,offset in C.PC_COUNTERS.items()} for pc in range(8)]
    cycles=None;cycle_reads=[]
    for attempt in range(3):
        hi=read('cycles_hi1',C.R_CYCLES_HI);lo=read('cycles_lo',C.R_CYCLES_LO);hi2=read('cycles_hi2',C.R_CYCLES_HI)
        cycle_reads.append([hi,lo,hi2])
        if hi is not None and lo is not None and hi==hi2:cycles=(hi<<32)|lo;break
    return dict(kind='c1-f2-failure-snapshot',profile=False,hardware_measured=False,raw_registers=values,counters=counters,pcs=pcs,cycles=cycles,cycle_reads=cycle_reads,coherent_cycles=cycles is not None,counters_may_have_wrapped=cycles is None or cycles>=2**32,read_errors=errors,csr_reads=reads,limitation='Failed-run raw evidence only; no bandwidth or completed-performance claim.')

class DiagnosticBenchmark(C.Benchmark):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.failure=None
    def capture_failure(self,original):
        if self.failure is not None:return
        self.failure=dict(kind='c1-f2-benchmark-failure',profile=False,error=f'{type(original).__name__}: {original}',abort=None,abort_attempts=[])
        try:self.failure['snapshot']=failure_snapshot(self.t)
        except BaseException as error:self.failure['snapshot_error']=f'{type(error).__name__}: {error}'
    def abort(self,timeout=5.0):
        # Canonical run invokes self.abort inside its active exception handler.
        original=sys.exception()
        if original is not None:self.capture_failure(original)
        try:
            super().abort(timeout=timeout)
            if self.failure is not None:
                self.failure['abort']=dict(drained=True);self.failure['abort_attempts'].append(self.failure['abort'].copy())
        except BaseException as error:
            if self.failure is not None:
                self.failure['abort']=dict(drained=False,error=f'{type(error).__name__}: {error}');self.failure['abort_attempts'].append(self.failure['abort'].copy())
            if original is None:raise
            # Preserve run's active exception: its following bare raise remains original.
    def run(self,*args,**kwargs):
        self.failure=None
        try:return super().run(*args,**kwargs)
        except BaseException as error:self.capture_failure(error);raise

def checked_child(bdf,expected_build_id,fixture,out,repeats):
    out=Path(out);canonical=canonical_bdf(bdf)
    if os.environ.get('OTPU_LOCK_HELD')!='f2-'+canonical or not os.environ.get('OTPU_LOCK_PID','').isdigit() or os.environ.get('OTPU_RUN_DIR','/tmp/otpu')!='/tmp/otpu':raise RuntimeError('Inherited canonical session lock required')
    out.mkdir(parents=False,exist_ok=False);bench=None;original=None
    try:
        with C1F2Transport(canonical,expected_build_id=expected_build_id) as transport:
            # Transport otherwise falls back to taking a new lock if inheritance fails.
            # Reject that fallback before preload/START writes.
            if transport._lock.holder is None or transport._lock.lost():raise RuntimeError('Parent session lock was not inherited')
            bench=DiagnosticBenchmark(transport,expected_build_id=expected_build_id)
            try:
                bench.preload(fixture)
                for repeat in range(repeats):
                    result=dict(bench.run(timeout=30.0),repeat=repeat)
                    facts=F.checked_result_facts(result,fixture,expected_identity=bench.identity,expected_device=transport.dev,repeat=repeat)
                    save(out/f'raw-{repeat:03d}.json',result)
                save(out/'complete.json',dict(kind='c1-checked-fixture-results',repeats=repeats,fixture_sha256=fixture.digest(),route_qualified=False,session_restoration_verified=False))
            except BaseException as error:
                original=error;bench.capture_failure(error);raise
            finally:
                try:
                    if transport.reg_read(C.R_STATUS)&(C.BUSY|C.DRAINING):bench.abort()
                except BaseException as error:
                    if bench.failure is not None:bench.failure['cleanup_error']=f'{type(error).__name__}: {error}'
                    if original is None:raise
    except BaseException as error:
        record=bench.failure if bench is not None and bench.failure is not None else dict(kind='c1-f2-benchmark-failure',profile=False,error=f'{type(error).__name__}: {error}')
        try:save(out/'failure.json',record)
        except BaseException as log_error:print('Could not save failure evidence: '+str(log_error),file=sys.stderr)
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--bdf',required=True);p.add_argument('--expected-build-id',required=True,type=lambda s:int(s,0));p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    if not 1<=a.repeats<=100:raise ValueError('Repeats outside bound')
    checked_child(a.bdf,a.expected_build_id,C.Fixture.load(a.fixture),a.out,a.repeats)
if __name__=='__main__':main()
