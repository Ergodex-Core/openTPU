"""Exclusive BAR0/BAR4 access to an already loaded C2 benchmark image.

Based on frozen F2 transport semantics at tpubench 67dcf0c8: WC write, MFENCE,
uncached PCIS read completion before START. This transport has its own image
identity and never loads AFIs, changes PCI setup, or bypasses device ownership.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from opentpu.c2_f2 import ABI, ID, PC_BYTES, R_ABI, R_BUILD, R_ID
from .runstate import DeviceLock, LockLost

_BDF = re.compile(r"(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-1][0-9a-fA-F]\.[0-7]")
_SOURCE = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if !defined(__x86_64__)
#error C2 F2 MMIO requires x86-64
#endif
uint32_t rd32(volatile uint32_t *p) { return *p; }
void wr32(volatile uint32_t *p, uint32_t x, volatile uint32_t *barrier) {
    *p=x; __asm__ volatile("mfence" ::: "memory"); (void)*barrier;
}
void write_bytes(void *wc, volatile uint32_t *uc, const void *src, size_t n) {
    memcpy(wc, src, n); __asm__ volatile("mfence" ::: "memory"); (void)*uc;
}
void read_bytes(void *dst, const void *uc, size_t n) { memcpy(dst, uc, n); }
"""


def canonical_bdf(value):
    if not _BDF.fullmatch(value):
        raise ValueError("explicit application-PF PCI BDF required, e.g. 0000:34:00.0")
    return ("0000:"+value if value.count(":") == 1 else value).lower()


def _library():
    with tempfile.TemporaryDirectory(prefix="c2-f2-mmio-") as temp:
        path = Path(temp)/"mmio.so"
        subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-x", "c", "-", "-o", str(path)],
                       input=_SOURCE, text=True, capture_output=True, timeout=30, check=True)
        lib = ctypes.CDLL(str(path))
    ptr, size = ctypes.c_void_p, ctypes.c_size_t
    lib.rd32.argtypes, lib.rd32.restype = [ptr], ctypes.c_uint32
    lib.wr32.argtypes, lib.wr32.restype = [ptr, ctypes.c_uint32, ptr], None
    lib.write_bytes.argtypes, lib.write_bytes.restype = [ptr, ptr, ptr, size], None
    lib.read_bytes.argtypes, lib.read_bytes.restype = [ptr, ptr, size], None
    return lib


def _mapping(path, size):
    fd = os.open(path, os.O_RDWR | os.O_SYNC)
    try:
        return mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)


def _address(mapping):
    return ctypes.addressof(ctypes.c_char.from_buffer(mapping))


class C2F2Transport:
    evidence_kind = "f2-hardware"

    def __init__(self, bdf, *, expected_build_id):
        if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64") or sys.byteorder != "little":
            raise RuntimeError("C2 F2 BAR access requires Linux x86-64 little endian")
        self.bdf = canonical_bdf(bdf)
        self.dev, self.devname = "f2:"+self.bdf, "f2-"+self.bdf
        self._lock = self._regs = self._wc = self._uc = None
        self._closed = False
        try:
            self._lock = DeviceLock(self.devname)
            root = Path("/sys/bus/pci/devices")/self.bdf
            ids = tuple(int((root/name).read_text(), 0) for name in ("vendor", "device"))
            if ids != (0x1D0F, 0xF001):
                raise RuntimeError(f"unexpected F2 PCI identity {ids}")
            self._lib = _library()
            self._regs = _mapping(root/"resource0", 4096)
            self._reg_addr = _address(self._regs)
            identity = tuple(self.reg_read(r) for r in (R_ID, R_ABI, R_BUILD))
            if identity != (ID, ABI, expected_build_id):
                raise RuntimeError(f"wrong C2 benchmark image: {identity!r}; no writes performed")
            self._uc = _mapping(root/"resource4", 8*PC_BYTES)
            self._wc = _mapping(root/"resource4_wc", 8*PC_BYTES)
            self._uc_addr, self._wc_addr = _address(self._uc), _address(self._wc)
        except BaseException:
            self.close()
            raise

    def _own(self):
        if self._closed or self._lock is None:
            raise RuntimeError("C2 transport is closed")
        if self._lock.lost():
            raise LockLost(self.devname, self._lock.holder)

    def _reg(self, offset):
        self._own()
        if not isinstance(offset, int) or offset < 0 or offset >= 4096 or offset % 4:
            raise ValueError("C2 CSR address must be aligned and inside BAR0")
        return self._reg_addr+offset

    def reg_read(self, offset):
        return int(self._lib.rd32(self._reg(offset)))

    def reg_write(self, offset, value):
        if not 0 <= int(value) <= 0xFFFFFFFF:
            raise ValueError("C2 CSR data must be uint32")
        self._lib.wr32(self._reg(offset), int(value), self._reg_addr+R_ID)

    def _range(self, pc, offset, size):
        self._own()
        if not 0 <= pc < 8 or offset < 0 or size < 0 or offset+size > PC_BYTES:
            raise ValueError("C2 PCIS access exceeds its owned PC")
        if offset % 64 or size % 64:
            raise ValueError("C2 benchmark PCIS transfers require whole aligned native beats")
        return pc*PC_BYTES+offset

    def mem_write(self, pc, offset, data):
        a = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
        pos = self._range(pc, offset, len(a))
        if len(a):
            self._lib.write_bytes(self._wc_addr+pos, self._uc_addr+pos, a.ctypes.data, len(a))

    def mem_read(self, pc, offset, n):
        pos = self._range(pc, offset, n)
        a = np.empty(n, np.uint8)
        if n:
            self._lib.read_bytes(a.ctypes.data, self._uc_addr+pos, n)
        return a

    def close(self):
        if self._closed:
            return
        for attr in ("_wc", "_uc", "_regs"):
            mapping = getattr(self, attr)
            if mapping is not None:
                mapping.close()
                setattr(self, attr, None)
        self._closed = True
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
