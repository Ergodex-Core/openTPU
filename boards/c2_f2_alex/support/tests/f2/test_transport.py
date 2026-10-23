"""Host address mapping on file-backed fake BARs; never touches a PCI device."""
import os, sys, struct
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
from opentpu.host.f2_transport import F2Transport
from opentpu.host.board import Board, beat_at
from opentpu.host.regs import ID_OTPU,R_ID

@pytest.fixture(scope='session')
def mmio_library(tmp_path_factory):
    import subprocess
    lib = tmp_path_factory.mktemp('f2_mmio') / 'libf2_mmio.so'
    subprocess.run(['gcc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
                    '-shared', '-fPIC', str(ROOT / 'opentpu/host/f2_mmio.c'),
                    '-o', str(lib)], check=True)
    return lib

@pytest.fixture(autouse=True)
def configure_library(monkeypatch, mmio_library):
    monkeypatch.setenv('OTPU_F2_MMIO_LIBRARY', str(mmio_library))

@pytest.fixture(params=[('scalar',1),('stream',1),('stream',8)])
def tr(tmp_path,monkeypatch,request):
    monkeypatch.setenv('OTPU_RUN_DIR',str(tmp_path/'locks'))
    monkeypatch.setenv('OTPU_F2_BDF','0000:34:00.0')
    mode,threads=request.param
    monkeypatch.setenv('OTPU_F2_READ_MODE',mode)
    monkeypatch.setenv('OTPU_F2_READ_THREADS',str(threads))
    dev=tmp_path/'0000:34:00.0';dev.mkdir()
    (dev/'vendor').write_text('0x1d0f')
    regs=bytearray(4096);struct.pack_into('<I',regs,R_ID,ID_OTPU)
    (dev/'resource0').write_bytes(regs)
    with (dev/'resource4').open('wb') as f:f.truncate(1<<32)
    os.link(dev/'resource4',dev/'resource4_wc')
    t=F2Transport(sysfs_root=tmp_path)
    yield t
    t.close()

def test_windows_boundaries_and_unaligned_bytes(tr):
    for ch in (0,1):
        for off,n in ((0,129),(511,79),((1<<31)-129,129)):
            data=np.arange(n,dtype=np.uint8)^np.uint8(ch*91)
            tr.mem_write(ch,off,data)
            np.testing.assert_array_equal(tr.mem_read(ch,off,n),data)
    with pytest.raises(ValueError):tr.mem_read(0,(1<<31)-1,2)
    with pytest.raises(ValueError):tr.mem_write(2,0,np.zeros(8,dtype=np.uint8))
    with pytest.raises(ValueError):tr.reg_write(3,0)

def test_logical_hash_mapping(tr):
    # Test actual upstream Board interleave against independently checked beat locations.
    b=Board(transport=tr,check=False,lock=False,calibrate=False)
    b._info={'caps':{'chash':True}}
    # chash is a property derived from the CAPS register; set only that capability.
    from opentpu.host import regs as R
    tr.reg_write(R.R_CAPS,R.CAP_CHASH)
    data=np.random.default_rng(17).integers(0,256,4096,dtype=np.uint8)
    b.write(0,data)
    np.testing.assert_array_equal(b.read(0,len(data)),data)
    for a in range(0,len(data),64):
        m=a//128;c=((a//64)&1)^(m.bit_count()&1)
        np.testing.assert_array_equal(tr.mem_read(c,m*64,64),data[a:a+64])
    b.close()

def test_stream_read_alignment_edges_and_refresh(tr):
    # The file-backed test checks bounds/alignment and fresh contents. Real WC
    # coherency is separately checked on hardware, including model logits.
    import ctypes as C
    lib=tr.lib
    lib.otpu_stream_read_supported.restype=C.c_int
    if not lib.otpu_stream_read_supported():pytest.skip('SSE4.1 unavailable')
    fn=lib.otpu_mem_read_stream
    fn.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t];fn.restype=None
    rng=np.random.default_rng(128)
    for offset in range(64):
        for length in (0,1,15,16,17,63,64,65,128,513):
            for flip in (0,255):
                data=rng.integers(0,256,length,dtype=np.uint8)^np.uint8(flip)
                tr.mem_write(0,offset,data)
                out=np.full(length+2,0xA5,dtype=np.uint8)
                fn(tr._mem_ptr,offset,out.ctypes.data+1,length)
                np.testing.assert_array_equal(out[1:-1],data)
                assert out[0]==0xA5 and out[-1]==0xA5

def test_large_read_parallel_chunks_and_output_guards(tr):
    rng=np.random.default_rng(64)
    for off,n in ((0,607744),(29,1048579),((1<<31)-1048579,1048579)):
        for _ in range(2):
            data=rng.integers(0,256,n,dtype=np.uint8)
            tr.mem_write(1,off,data)
            out=np.full(n+2,0xA5,dtype=np.uint8)
            tr.mem_read(1,off,n,out=out[1:-1])
            np.testing.assert_array_equal(out[1:-1],data)
            assert out[0]==0xA5 and out[-1]==0xA5
