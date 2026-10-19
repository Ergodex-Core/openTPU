#!/usr/bin/env python3
"""Measure native HBM/DMA throughput independently of host data transfers.

Overwrites a 32 MiB test area before model upload. Requires hardware smoke PASS.
"""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OTPU_RUN_DIR',str(ROOT/'build/f2/runtime'))
os.environ.setdefault('OTPU_TARGET',str(ROOT/'targets/f2.toml'))
import numpy as np
from opentpu.host.f2_transport import F2Transport
from opentpu.host.board import Board
from opentpu.host import regs as R
from opentpu import isa as I


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    if json.loads(a.smoke.read_text()).get('pass') is not True:
        raise SystemExit('Successful hardware smoke report required')
    tr=F2Transport();board=None
    try:
        board=Board(transport=tr);info=board.info()
        chunk=65536;count=128;size=chunk*count
        src=0x1000000;dst=0x2000000;prog=0x3000000
        data=np.random.default_rng(47).integers(0,256,size,dtype=np.uint8)
        board.write(src,data)
        results=[]
        for mode in ('read_linear','read_reuse','write_linear'):
            if mode=='write_linear':
                ops=[I.ld(src,0,chunk//4)]+[I.st(dst+i*chunk,0,chunk//4) for i in range(count)]
                expected=np.tile(data[:chunk],count);check_addr=dst
            else:
                offsets=[i*chunk if mode=='read_linear' else 0 for i in range(count)]
                ops=[I.ld(src+off,0,chunk//4) for off in offsets]+[I.st(dst,0,chunk//4)]
                expected=data[offsets[-1]:offsets[-1]+chunk];check_addr=dst
            board.load(prog,ops+[I.halt()])
            for repetition in range(3):
                before=board.snapshot();stats=board.run(timeout=30);after=board.snapshot()
                delta={k:int(after[k]-before[k]) for k in before if k!='snaps'}
                np.testing.assert_array_equal(board.read(check_addr,len(expected)),expected)
                status=tr.reg_read(R.R_STATUS)
                if status&(R.ST_AXI_ERR|R.ST_ERROR):raise RuntimeError(f'Hardware error: {status:#x}')
                seconds=stats['cycles']/(info['core_khz']*1000)
                row=dict(mode=mode,repetition=repetition,payload_bytes=size,
                         payload_GBps=size/seconds/1e9,cycles=stats['cycles'],
                         counters=delta,run=stats)
                results.append(row)
                print(mode,repetition,f"{row['payload_GBps']:.3f} GB/s",flush=True)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps({'pass':True,'device':info,'results':results},indent=2)+'\n')
    finally:
        if board is not None:board.close()
        tr.close()


if __name__=='__main__':main()
