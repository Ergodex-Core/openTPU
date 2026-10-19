"""C1 native-HBM-boundary RTL checks; synthetic memory timing, not device results."""
import fcntl
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
from . import c1_rtl
ROOT=c1_rtl.ROOT
SOURCES=[*c1_rtl.SOURCES[:-1],ROOT/'rtl/c1/otpu_c1_native.sv',ROOT/'rtl/c1/otpu_c1_hbm_kernel.sv',ROOT/'sim/c1/tb_hbm.sv']


def build():
    version=subprocess.check_output(['verilator','--version'],text=True)
    digest=hashlib.sha256(version.encode())
    for p in SOURCES:digest.update(p.read_bytes())
    out=ROOT/'build/c1-hbm'/digest.hexdigest()[:16]
    out.mkdir(parents=True,exist_ok=True)
    exe=out/'Vtb_c1_hbm'
    with (out/'build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not exe.exists():
            r=subprocess.run(['verilator','--binary','--timing','--assert','-Wno-fatal','-j','2','--top-module','tb_c1_hbm','--Mdir',str(out),*map(str,SOURCES)],capture_output=True,text=True)
            (out/'build.log').write_text(r.stdout+r.stderr)
            if r.returncode:raise RuntimeError(f'C1 HBM build failed: {out}/build.log')
    return exe


def write_fixture(case,path):
    if case.n0 or case.k0 or case.ab or case.n>512 or case.kb>128:raise ValueError('C1 hardware benchmark supports origin-zero, N<=512, KB<=128')
    ops=[]
    def window(addr,values):
        ops.append((0x24,addr));ops.extend((0x28,int(v)) for v in values)
    window(0,np.frombuffer(case.state.act[0,:case.kb*128].tobytes(),dtype='<u4'))
    window(0x10000,np.frombuffer(case.state.adoff[0,:case.kb,:].tobytes(),dtype='<u4'))
    window(0x11000,case.state.ascale[0,:1].view(np.uint32))
    window(0x40000,np.frombuffer(case.scales[:,:4].copy().tobytes(),dtype='<u4'))
    for row in range(case.n):window(0x20000+row*128,case.scales[row,4:4+case.kb])
    beats=((case.n+15)//16)*4*case.kb
    banks=[bytearray(beats*64) for _ in range(8)]
    for row in range(case.n):
        offset=((row//16)*2+(row%2))*case.kb*128
        banks[(row%16)//2][offset:offset+case.kb*128]=case.weights[row,:case.kb*128].tobytes()
    with Path(path).open('wb') as f:
        f.write(struct.pack('>6I',case.n,case.kb,0,64,beats,len(ops)))
        for bank in banks:
            for i in range(0,len(bank),64):f.write(bank[i:i+64][::-1])
        for a,v in ops:f.write(struct.pack('>II',a,v))


def run(case,out,*,latency=37,stall=25,delay_pc0=91,fault=0,seed=17):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_fixture(case,out/'input.bin')
    exe=build()
    r=subprocess.run([str(exe),f'+input={out/"input.bin"}',f'+events={out/"events.txt"}',f'+results={out/"results.txt"}',f'+latency={latency}',f'+stall={stall}',f'+delay_pc0={delay_pc0}',f'+fault={fault}',f'+seed={seed}'],capture_output=True,text=True,timeout=180)
    (out/'console.log').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    counters={};bits=[];ints=[]
    for line in (out/'results.txt').read_text().splitlines():
        fields=line.split()
        if fields[0]=='C':counters[int(fields[1])]=int(fields[2])
        elif fields[0]=='R':
            assert int(fields[1])==len(bits)
            bits.append(int(fields[2],16));ints.append(int(fields[3],16))
    assert bits==case.expected.tolist(),(bits,case.expected.tolist())
    assert ints==case.accumulators,(ints,case.accumulators)
    assert counters[0x30]==0 and counters[0x34]==case.n
    assert counters[0x40]==case.n*case.kb*4 and counters[0x44]==case.n
    assert counters[0x48]==counters[0x4c]==case.n*case.kb*2
    result=dict(cycles=counters[0x38]+(counters[0x3c]<<32),counters=counters,rows=case.n,kb=case.kb,bit_exact=True,evidence='native kernel RTL with synthetic memory, not FPGA')
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def lens_data(result,events,name):
    from . import lens
    from .isasim import Config
    from .profile import UNITS
    c=result['counters'];cycles=result['cycles'];n=result['rows'];kb=result['kb']
    floor=max(c[0x100+32*p] for p in range(8))
    detail=f'C1 E16/S4 N{n} KB{kb}; resident ACT/scales; eight native ports'
    return dict(kind='rtl',name=name+' [KERNEL RTL / SYNTHETIC NATIVE SERVICE]',
        cycles=cycles,clock_mhz=200.0,
        config=lens._config(Config(S=1,D=128,MCOLS=4,ACT_BLOCKS=128,TMEM_WORDS=512),
                            {'C':1,'PCS':8,'C1_ENGINES':16,'C1_SLOTS':4}),
        roofline=dict(bound=floor,efficiency=floor/cycles,per_slice=[dict(bound=floor,portb=c[0x4c],porta=0)]),
        macs=n*kb*128,bytes=c[0x4c]*64,peak_macs=512,peak_bytes=512,units=UNITS,
        slices=[dict(ports={},busy={},lose={},buckets={})],
        instrs=[[0,0,0,1,'C1_ROWS',detail,0,0,-1,0,cycles,-1,{},n*kb,n*kb,0]],
        programs=[[[0,'C1_ROWS',detail,'C1 native benchmark RTL',-1]]],sources=[],mxu_gaps={},
        c1_native_rtl=dict(counters={str(k):v for k,v in c.items()},events=events,
             fields=['run','cycle','request_mask','response_mask','accepted_32B_groups','commit','output_row','pending'],
             source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}),
        notes=[dict(level='warn',text='Actual kernel RTL cycles with synthetic native memory service; no measured HBM bandwidth, FPGA timing or full-token throughput.'),
               dict(level='info',text='COMMAND through final result, excluding OCL preload, ACT snapshot copy and result readback. Scales are resident on chip; HBM holds weight bytes only. Stall counters overlap and are not exclusive causal attribution.')])
