"""AWS Small Shell OCL/PCIS transport for upstream Board/BoardBackend.

Bring-up transport using PCIe BARs, not a claim of DMA performance. Requires
root access to AppPF resource0/resource4 and the matching openTPU image.
"""
from pathlib import Path
import ctypes as C
import mmap, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from opentpu.host.board import XdmaTransport, BASE
from opentpu.host.runstate import DeviceLock
from opentpu.host.regs import ID_OTPU, R_ID

class F2Transport:
    ecc=False
    threaded=True
    streams=True
    keeps_state=True
    run_h2c=0
    host_idle=0.0
    bases=BASE
    # Reuse only the device-independent, measured polling algorithm.
    poll=XdmaTransport.poll
    reg_read_many=XdmaTransport.reg_read_many

    def __init__(self, dev=None, dma=True, *, sysfs_root='/sys/bus/pci/devices', library=None):
        bdf=os.environ.get('OTPU_F2_BDF','0000:34:00.0')
        if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',bdf):
            raise ValueError('OTPU_F2_BDF must be a PCI BDF')
        self.dev=bdf;self.devname='f2_'+bdf.replace(':','_').replace('.','_')
        self._otpu_lock=None;self.regs=None;self.mem=None;self.mem_wc=None
        self._reg_view=None;self._mem_view=None;self._wc_view=None
        self._mutex=threading.RLock()
        self._read_pool=None
        self.read_threads=int(os.environ.get('OTPU_F2_READ_THREADS','1'))
        if not 1<=self.read_threads<=16:
            raise ValueError('OTPU_F2_READ_THREADS must be between 1 and 16')
        self.transfer_bytes={'h2c':0,'c2h':0}
        self.transfer_seconds={'h2c':0.0,'c2h':0.0}
        self.read_mode=os.environ.get('OTPU_F2_READ_MODE','scalar')
        if self.read_mode not in ('scalar','stream'):
            raise ValueError('OTPU_F2_READ_MODE must be scalar or stream')
        libpath=Path(library or os.environ.get('OTPU_F2_MMIO_LIBRARY') or
                     Path(__file__).with_name('libf2_mmio.so'))
        self.lib=C.CDLL(str(libpath.resolve()))
        self.lib.otpu_reg_read.argtypes=[C.c_void_p,C.c_size_t]
        self.lib.otpu_reg_read.restype=C.c_uint32
        self.lib.otpu_reg_write.argtypes=[C.c_void_p,C.c_size_t,C.c_uint32]
        self.lib.otpu_reg_write.restype=None
        for k in ('otpu_mem_read','otpu_mem_write'):
            f=getattr(self.lib,k);f.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t];f.restype=None
        if self.read_mode=='stream':
            self.lib.otpu_stream_read_supported.restype=C.c_int
            if not self.lib.otpu_stream_read_supported():
                raise RuntimeError('Streaming reads require SSE4.1')
            self.lib.otpu_mem_read_stream.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
            self.lib.otpu_mem_read_stream.restype=None
        path=Path(sysfs_root)/bdf
        try:
            if int((path/'vendor').read_text(),16)!=0x1d0f:
                raise RuntimeError('Selected PCI function is not an AWS FPGA')
            if dma:self._otpu_lock=DeviceLock(self.devname)
            self.regs=self._map(path/'resource0',4096)
            self._reg_view=C.c_char.from_buffer(self.regs)
            self._reg_ptr=C.addressof(self._reg_view)
            if self.reg_read(R_ID)!=ID_OTPU:
                raise RuntimeError('Selected FPGA does not report openTPU; refusing memory access')
            if dma:
                self.mem=self._map(path/'resource4',1<<32)
                self._mem_view=C.c_char.from_buffer(self.mem)
                self._mem_ptr=C.addressof(self._mem_view)
                # WC improves upload rate; normal mapping remains the read path.
                if (path/'resource4_wc').exists():
                    self.mem_wc=self._map(path/'resource4_wc',1<<32)
                    self._wc_view=C.c_char.from_buffer(self.mem_wc)
                    self._write_ptr=C.addressof(self._wc_view)
                else:self._write_ptr=self._mem_ptr
                if self.read_mode=='stream' and self.mem_wc is None:
                    raise RuntimeError('Streaming HBM reads require resource4_wc')
        except BaseException:
            self.close();raise

    @staticmethod
    def _map(path,size):
        fd=os.open(path,os.O_RDWR|os.O_SYNC)
        try:return mmap.mmap(fd,size,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE)
        finally:os.close(fd)

    def reg_read(self,off):
        if off<0 or off>=4096 or off%4:raise ValueError('Invalid OCL register offset')
        return int(self.lib.otpu_reg_read(self._reg_ptr,off))

    def reg_write(self,off,val):
        if off<0 or off>=4096 or off%4:raise ValueError('Invalid OCL register offset')
        self.lib.otpu_reg_write(self._reg_ptr,off,val&0xffffffff)

    def _address(self,ch,off,n):
        if ch not in (0,1) or off<0 or n<0 or off+n>(1<<31):
            raise ValueError('Access outside the two 2-GiB HBM channel windows')
        if self.mem is None:raise RuntimeError('Transport opened without memory access')
        return self.bases[ch]+off

    def mem_write(self,ch,off,data):
        a=np.ascontiguousarray(data,dtype=np.uint8).reshape(-1)
        addr=self._address(ch,off,a.size)
        if not a.size:return
        with self._mutex:
            start=time.perf_counter()
            self.lib.otpu_mem_write(self._write_ptr,addr,a.ctypes.data,a.size)
            # Flush posted PCIe writes with an ordered read on the same BAR.
            # Device tests must verify host/core visibility before benchmarking.
            self.lib.otpu_reg_read(self._mem_ptr,(addr+a.size-1)&~3)
            self.transfer_seconds['h2c']+=time.perf_counter()-start
            self.transfer_bytes['h2c']+=a.size

    def mem_read(self,ch,off,n,out=None):
        addr=self._address(ch,off,n)
        a=np.empty(n,dtype=np.uint8) if out is None else out
        if a.dtype!=np.uint8 or not a.flags.c_contiguous or a.size<n:
            raise ValueError('Read buffer must be contiguous uint8 with sufficient capacity')
        with self._mutex:
            start=time.perf_counter()
            if self.read_mode=='stream':
                fn=self.lib.otpu_mem_read_stream
                if self.read_threads>1 and n>=65536:
                    if self._read_pool is None:
                        self._read_pool=ThreadPoolExecutor(max_workers=self.read_threads,
                                                         thread_name_prefix='otpu_f2_read')
                    block=((n+self.read_threads-1)//self.read_threads+63)&~63
                    futures=[self._read_pool.submit(fn,self._write_ptr,addr+i,a.ctypes.data+i,
                                                    min(block,n-i)) for i in range(0,n,block)]
                    for future in futures:future.result()
                else:fn(self._write_ptr,addr,a.ctypes.data,n)
            else:
                self.lib.otpu_mem_read(self._mem_ptr,addr,a.ctypes.data,n)
            self.transfer_seconds['c2h']+=time.perf_counter()-start
            self.transfer_bytes['c2h']+=n
        return a

    def close(self):
        if self._read_pool is not None:
            self._read_pool.shutdown(wait=True);self._read_pool=None
        self._reg_view=self._mem_view=self._wc_view=None
        for name in ('mem_wc','mem','regs'):
            mapping=getattr(self,name,None)
            if mapping is not None:mapping.close();setattr(self,name,None)
        if self._otpu_lock is not None:self._otpu_lock.release();self._otpu_lock=None
