"""DPI entry point. The RTL uses the same Python PortController and ctypes Ramulator bridge.

Data is sampled/updated on logical-port acceptance, as in the existing behavioral RAM.
Timing and completion are closed-loop; no request trace is replayed. Native masks must
represent contiguous enabled bytes, otherwise the harness fails explicitly.
"""
import json
from pathlib import Path
import numpy as np
from opentpu.perfsim.config import MemoryConfig
from opentpu.perfsim.ports import PortController


class Driver:
    def __init__(self,profile,directory,width,words):
        obj=json.loads(Path(profile).read_text());self.period=obj['architecture']['core_period_ps']
        cfg=obj['memory'].copy();base=Path(profile).resolve().parent
        for key in ('bridge','ramulator_config'):
            if cfg.get(key):cfg[key]=str((base/cfg[key]).resolve())
        if not cfg.get('native_ports'):raise ValueError('RTL comparison requires native_ports=true')
        self.controller=PortController(MemoryConfig(**cfg));self.width=width;self.dir=Path(directory)
        data=np.fromfile(self.dir/'dram_0.bin',dtype='>u4').astype('<u4').tobytes()
        self.data=bytearray(words*4);self.data[:len(data)]=data
        self.next_ident=0;self.pending={};self.offers={};self.read_data={}

    def step(self,cycle,a,b,sw):
        # Each descriptor: valid, write, byte address, mask, tag, 128-byte payload.
        for port,desc in (('A',a),('B',b),('SW',sw)):
            valid,write,addr,mask,tag,payload=desc
            if not valid:
                if port in self.offers:raise RuntimeError('RTL dropped an unaccepted port request')
                continue
            if port in self.offers:continue
            if write:
                enabled=[k for k in range(self.width if port=='B' else 4) if (mask>>(k//4 if port=='B' else k))&1]
                if not enabled:raise ValueError('empty native write')
                lo,hi=min(enabled),max(enabled)+1
                if enabled!=list(range(lo,hi)):raise ValueError('non-contiguous native write mask unsupported')
                addr+=lo;size=hi-lo;payload=payload[lo:hi]
            else:
                size=32 if port=='A' else self.width
                if port=='A':addr=addr//32*32  # native scale-line addresses are word addresses within the aligned line
            if addr<0 or addr+size>len(self.data):raise ValueError('native address outside DRAM image')
            ident=self.next_ident;self.next_ident+=1
            self.offers[port]=(ident,addr,size,bool(write),0)
            self.pending[ident]=(port,addr,size,bool(write),tag,payload)
        accepted,done=self.controller.step(cycle,self.period,self.offers)
        ready=0
        for port in accepted:
            ident,addr,size,write,_=self.offers.pop(port);_,_,_,_,tag,payload=self.pending[ident]
            if write:self.data[addr:addr+size]=payload
            else:self.read_data[ident]=bytes(self.data[addr:addr+size])
            ready|={'A':1,'B':2,'SW':4}[port]
        av=0;ad=bytes(32);bv=0;bt=0;bd=bytes(self.width)
        for port,ident in done:
            _,addr,size,write,tag,payload=self.pending.pop(ident)
            if not write:
                data=self.read_data.pop(ident)
                if port=='A':av=1;ad=data
                else:bv=1;bt=tag;bd=data
        return ready,av,ad,bv,bt,bd,int(self.controller.read_idle),int(self.controller.write_idle)

    def dump(self):
        (self.dir/'dram_out_0.bin').write_bytes(self.data)
        m=self.controller.memory
        (self.dir/'memory-rtl.json').write_text(json.dumps({'backend':m.cfg.backend,'statistics':dict(m.stats),'transactions':m.trace,'ports':self.controller.events},indent=2)+'\n')

    def close(self):
        self.controller.memory.close()
