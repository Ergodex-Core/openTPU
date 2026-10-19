#!/usr/bin/env python3
"""Prepare C1 fixtures offline; checked_runs is for a future qualified session.

This CLI cannot open PCI devices or configure an FPGA. A session controller must
supply its already-owned, identity-checked transport to checked_runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opentpu import c1_f2 as C


def checked_runs(transport,fixture,expected_build_id,*,repeats=3,timeout=30.):
    if not 1<=repeats<=100 or timeout<=0:
        raise ValueError('Invalid repeat count or timeout')
    bench=C.Benchmark(transport,expected_build_id=expected_build_id)
    try:
        bench.preload(fixture)
        for repeat in range(repeats):
            result=bench.run(timeout=timeout)
            yield dict(result,repeat=repeat)
    finally:
        if transport.reg_read(C.R_STATUS)&(C.BUSY|C.DRAINING):bench.abort()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',required=True,type=Path)
    p.add_argument('--rows',type=int,default=17);p.add_argument('--kb',type=int,default=9)
    p.add_argument('--seed',type=int,default=20261019);p.add_argument('--base-beat',type=int,default=16)
    a=p.parse_args(argv)
    manifest=a.fixture.with_suffix('.json')
    if a.fixture.suffix!='.npz' or a.fixture.exists() or manifest.exists():
        raise ValueError('Use a fresh .npz fixture and sidecar path')
    f=C.make_fixture(a.rows,a.kb,a.seed,a.base_beat);bits,integers=f.reference()
    a.fixture.parent.mkdir(parents=True,exist_ok=True);f.save(a.fixture)
    manifest.write_text(json.dumps(dict(kind='offline-c1-canonical-reference',hardware_measured=False,
        rows=f.rows,kb=f.kb,seed=a.seed,base_beat=f.base_beat,fixture_sha256=f.digest(),
        fixture_file_sha256=hashlib.sha256(a.fixture.read_bytes()).hexdigest(),
        expected_bits=bits.tolist(),expected_integers=integers,expected_pc_beats=f.expected_pc_beats(),
        useful_weight_bytes=f.rows*f.kb*128,native_read_bytes=sum(f.expected_pc_beats())*64),indent=2)+'\n')
    print(f'Prepared {a.fixture}; canonical arithmetic checked, no hardware accessed')


if __name__=='__main__':main()
