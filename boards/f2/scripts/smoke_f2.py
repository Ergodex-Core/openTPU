#!/usr/bin/env python3
"""HBM address/data + instruction/TMEM/DMA smoke test for a loaded openTPU F2 image."""
from pathlib import Path
import argparse, json, os, sys, time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OTPU_RUN_DIR',str(ROOT/'build/f2/runtime'))
os.environ.setdefault('OTPU_TARGET',str(ROOT/'targets/f2.toml'))
os.environ.setdefault('HF_HOME',str(ROOT/'build/f2/hf-cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'build/f2/runtime/cache'))
(ROOT/'build/f2/runtime/tmp').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('TMPDIR',str(ROOT/'build/f2/runtime/tmp'))
import numpy as np
from opentpu.host.f2_transport import F2Transport
from opentpu.host.board import Board, device_config
from opentpu.host import regs as R
from opentpu import isa as I
from opentpu import target

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    tr=F2Transport();board=None
    try:
        board=Board(transport=tr)
        status=tr.poll(R.R_STATUS,R.ST_CALIB0|R.ST_CALIB1,R.ST_CALIB0|R.ST_CALIB1,timeout=30)
        if status&R.ST_AXI_ERR:raise RuntimeError(f'AXI error before testing: {status:#x}')
        info=board.info();cfg=device_config(info)
        expected=target.current()
        expected_hash=target.arch_hash(target.config(expected))
        if (info['cfg']['platform']!=expected.platform_id or
            info['cfg']['hash']!=expected_hash or info['core_khz']!=expected.core_khz):
            raise RuntimeError('Device platform, architecture hash or clock readback differs from OTPU_TARGET')
        if cfg.DRAM_BYTES!=1<<32:raise RuntimeError('Expected 4 GiB logical capacity')
        # The configuration register states the intended clock. UPTIME provides
        # an independent host-timed measurement of the actual core clock.
        a0=time.perf_counter();s0=board.snapshot();a1=time.perf_counter()
        time.sleep(0.5)
        b0=time.perf_counter();s1=board.snapshot();b1=time.perf_counter()
        elapsed=(b0+b1-a0-a1)/2
        measured_mhz=(s1['UPTIME']-s0['UPTIME'])/elapsed/1e6
        clock_probe={'frequency_mhz':measured_mhz,'elapsed_seconds':elapsed,
                     'first_snapshot_seconds':a1-a0,'last_snapshot_seconds':b1-b0}
        if abs(measured_mhz/(info['core_khz']/1000)-1)>0.02:
            raise RuntimeError(f'Core clock measurement differs by more than 2%: {clock_probe}')
        rng=np.random.default_rng(0xF247)
        patterns={}
        # Keep all writes until readback to detect aliases among pseudochannels/windows.
        for c in (0,1):
            for off in (0,1<<21,1<<29,1<<30,3<<29,(1<<31)-4096):
                data=rng.integers(0,256,4096,dtype=np.uint8)
                patterns[c,off]=data
                tr.mem_write(c,off,data)
        for (c,off),data in patterns.items():
            np.testing.assert_array_equal(tr.mem_read(c,off,len(data)),data)
        data=rng.integers(0,256,8195,dtype=np.uint8)
        board.write(0x400001,data)
        np.testing.assert_array_equal(board.read(0x400001,len(data)),data)
        # The unchanged instruction loader, native reads, 2x TMEM and DMA store.
        data=rng.integers(0,256,16384,dtype=np.uint8)
        board.write(0x100000,data)
        board.write(0x200000,np.zeros_like(data))
        board.load(0x300000,[I.ld(0x100000,0,4096),I.st(0x200000,0,4096),I.halt()])
        stats=board.run(timeout=30)
        np.testing.assert_array_equal(board.read(0x200000,len(data)),data)
        status=tr.reg_read(R.R_STATUS)
        if status&(R.ST_AXI_ERR|R.ST_ERROR):raise RuntimeError(f'Error after DMA copy: {status:#x}')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        result={'pass':True,'device':info,'raw_probes':len(patterns),'raw_probe_bytes':4096,
                'logical_unaligned_bytes':8195,'dma_copy_bytes':len(data),'run':stats,
                'clock_probe':clock_probe,'status':status}
        args.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
        print('PASS HBM windows/aliasing, hashed unaligned access, instruction load, TMEM and DMA copy')
    finally:
        if board is not None:board.close()
        tr.close()
if __name__=='__main__':main()
