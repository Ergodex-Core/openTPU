"""AWS F2 small-shell transport: OCL BAR0 registers and PCIS BAR4 DIMM memory.

Select ``OTPU_DEVICE=f2:0`` (AWS slot number), or ``f2:0000:34:00.0`` (application PF BDF).
Slots are resolved by the AWS SDK's fpga-describe-local-image-slots tool; explicit BDFs need
only Linux sysfs. The transport never loads an image, unbinds a driver or changes PCI setup.

BAR4 writes use resource4_wc, followed in the same native call by MFENCE and a read from the
uncached resource4 mapping. The CPU fence drains write combining; the PCIe read completion and
otpu_f2_mem's write barrier make those bytes visible before a later OCL RUN/WAITW write.
This requires a C compiler on the F2 host for a small process-local MMIO helper. BAR0 uses
32-bit memoryview accesses, never byte copies of a control register. Call Board.scrub() once
after configuration before payload transfers (BoardBackend does this automatically): sh_ddr
calibration alone is not evidence that the DDR ECC bits have been initialized.
"""
from __future__ import annotations

import ctypes
import functools
import mmap
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import regs as R
from .board import XdmaTransport, _dma_lock
from .runstate import DeviceLock, LockLost

SYSFS = Path("/sys/bus/pci/devices")
BAR0_BYTES = 4096
PCI_VENDOR, PCI_DEVICE = 0x1D0F, 0xF001
PLATFORM = 3
_BDF = re.compile(r"(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-1][0-9a-fA-F]\.[0-7]")


def resolve_device(device: str = "f2:0") -> str:
    """Return the canonical application-PF BDF; slot and BDF aliases share one lock."""
    if not device.startswith("f2:"):
        raise ValueError("F2 device must be f2:<slot> or f2:<PCI BDF>")
    value = device[3:]
    if value.isascii() and value.isdecimal():
        slot = int(value)
        try:
            r = subprocess.run(["fpga-describe-local-image-slots"], check=True,
                               capture_output=True, text=True, timeout=15)
        except FileNotFoundError as e:
            raise RuntimeError("AWS FPGA SDK slot discovery is unavailable; use f2:<PCI BDF>") from e
        matches = [p[-1] for line in r.stdout.splitlines()
                   if len(p := line.split()) == 5 and p[0] == "AFIDEVICE"
                   and p[1].isdigit() and int(p[1]) == slot]
        if len(matches) != 1 or not _BDF.fullmatch(matches[0]):
            raise RuntimeError(f"AWS FPGA slot {slot} has no unambiguous application PF")
        value = matches[0]
    if not _BDF.fullmatch(value):
        raise ValueError(f"invalid F2 PCI address: {value!r}")
    return ("0000:" + value if value.count(":") == 1 else value).lower()


def _require_host() -> None:
    if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("F2 BAR access requires the Linux x86-64 FPGA instance")
    if sys.byteorder != "little" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("F2 BAR access requires a 64-bit little-endian Python")


_MMIO_SOURCE = r"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if !defined(__x86_64__)
#error F2 MMIO requires x86-64
#endif
void otpu_write(void *wc, volatile uint32_t *uc, const void *src, size_t bytes) {
    memcpy(wc, src, bytes);
    __asm__ volatile("mfence" ::: "memory");
    (void)*uc;
}
void otpu_read(void *dst, const void *uc, size_t bytes) {
    memcpy(dst, uc, bytes);
}
"""


@functools.lru_cache(maxsize=1)
def _mmio():
    # A private directory, never a shared executable cache. dlopen keeps the library mapped
    # after its file is removed. No executable anonymous memory or AWS SDK ABI dependency.
    with tempfile.TemporaryDirectory(prefix="otpu-f2-mmio-") as d:
        libpath = Path(d) / "mmio.so"
        try:
            r = subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-x", "c", "-", "-o", str(libpath)],
                               input=_MMIO_SOURCE, capture_output=True, text=True, timeout=30)
        except FileNotFoundError as e:
            raise RuntimeError("F2 MMIO requires a C compiler (cc) on the FPGA host") from e
        if r.returncode:
            raise RuntimeError(f"F2 MMIO helper compilation failed: {r.stderr.strip()}")
        lib = ctypes.CDLL(str(libpath))
    lib.otpu_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.otpu_write.restype = None
    lib.otpu_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.otpu_read.restype = None
    return lib


def _map(path: Path, size: int) -> mmap.mmap:
    fd = os.open(path, os.O_RDWR | os.O_SYNC)
    try:
        return mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)


def _address(mapping: mmap.mmap) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(mapping))


class F2Transport:
    """One loaded openTPU F2 image; dma=False opens registers for a monitor only."""
    ecc = True                          # conservatively scrub the exposed 4 GiB after a load
    threaded = True
    streams = True
    host_idle = 50e-6
    poll = XdmaTransport.poll

    def __init__(self, dev: str = "f2:0", dma: bool = True):
        _require_host()
        self.bdf = resolve_device(dev)
        self.dev = "f2:" + self.bdf
        self.devname = "f2-" + self.bdf
        self._otpu_lock = None
        self.regs = self.words = self._uc = self._wc = None
        self._lib = None
        self._data = dma
        self._closed = False
        self._dma = _dma_lock(self.devname)
        try:
            # Lock before opening a PCI resource, just as for XDMA. Numeric slot aliases
            # resolve to the same BDF lock as explicit PCI addresses.
            if dma:
                self._otpu_lock = DeviceLock(self.devname)
            root = SYSFS / self.bdf
            ids = tuple(int((root / f).read_text().strip(), 0) for f in ("vendor", "device"))
            if ids != (PCI_VENDOR, PCI_DEVICE):
                raise RuntimeError(f"{self.dev}: unexpected PCI IDs {ids[0]:04x}:{ids[1]:04x}")
            self.regs = _map(root / "resource0", BAR0_BYTES)
            self.words = memoryview(self.regs).cast("I")
            if self.reg_read(R.R_ID) != R.ID_OTPU:
                raise RuntimeError(f"{self.dev}: no openTPU image in BAR0")
            if R.regmap(self.reg_read(R.R_REGMAP)) < R.CFG_SINCE or self.reg_read(R.R_CFG_PLATFORM) != PLATFORM:
                raise RuntimeError(f"{self.dev}: image is not the F2 openTPU platform")
            dram = self.reg_read(R.R_CFG_DRAM)
            channel_shift, channels, total_shift = (dram >> 16) & 255, (dram >> 8) & 255, dram & 255
            if (self.reg_read(R.R_CFG_SLICES) != 1 or channels != 2 or not 6 <= channel_shift <= 31
                    or total_shift != channel_shift + 1):
                raise RuntimeError(f"{self.dev}: unsupported F2 memory geometry {dram:#x}")
            self.channel_bytes = 1 << channel_shift
            self.bases = (0, self.channel_bytes)
            if dma:
                self._lib = _mmio()
                size = channels * self.channel_bytes
                self._uc = _map(root / "resource4", size)
                self._wc = _map(root / "resource4_wc", size)
                self._uc_addr, self._wc_addr = _address(self._uc), _address(self._wc)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        # Stop a Board's run while registers and ownership still exist. Keep the device lock
        # until every mapping is gone so the next owner can safely reconfigure/rescan it.
        lock = self._otpu_lock
        if lock is not None:
            cleanup, lock.cleanup = lock.cleanup, []
            for ref in cleanup:
                callback = ref()
                if callback is not None:
                    callback()
        if self.words is not None:
            self.words.release()
            self.words = None
        for name in ("_wc", "_uc", "regs"):
            resource = getattr(self, name)
            if resource is not None:
                resource.close()
                setattr(self, name, None)
        self._closed = True
        if lock is not None:
            lock.release()
            self._otpu_lock = None

    def _own(self) -> None:
        if self._otpu_lock is None:
            raise RuntimeError("F2 writes require the device lock")
        if self._otpu_lock.lost():
            raise LockLost(self.devname, self._otpu_lock.holder)

    def _reg(self, off: int) -> bool:
        if self._closed:
            raise RuntimeError("F2 transport is closed")
        if off < 0 or off % 4 or off >= 1 << 32:
            raise ValueError("F2 register offsets must be aligned 32-bit addresses")
        return off < BAR0_BYTES

    def reg_read(self, off: int) -> int:
        return int(self.words[off // 4]) if self._reg(off) else 0

    def reg_read_many(self, offs: list[int]) -> list[int]:
        return [self.reg_read(off) for off in offs]

    def reg_write(self, off: int, val: int) -> None:
        inside = self._reg(off)
        if self._data:
            self._own()
        elif off != R.R_SNAP:
            raise RuntimeError("F2 monitor can only read registers and latch a snapshot")
        if inside:
            with self._dma:
                self.words[off // 4] = int(val) & 0xFFFFFFFF

    def _range(self, ch: int, off: int, size: int) -> int:
        if self._closed or self._uc is None:
            raise RuntimeError("F2 memory transport is not open")
        if not 0 <= ch < len(self.bases) or off < 0 or size < 0 or off + size > self.channel_bytes:
            raise ValueError("F2 memory access exceeds its channel")
        return self.bases[ch] + off

    def mem_write(self, ch: int, off: int, data: np.ndarray) -> None:
        a = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
        pos = self._range(ch, off, len(a))
        self._own()
        if not len(a):
            return
        with self._dma:
            self._lib.otpu_write(self._wc_addr + pos, self._uc_addr + (pos & ~3),
                                 a.ctypes.data, len(a))

    def mem_read(self, ch: int, off: int, n: int, out: np.ndarray | None = None) -> np.ndarray:
        pos = self._range(ch, off, n)
        a = np.empty(n, dtype=np.uint8) if out is None else out
        if (not isinstance(a, np.ndarray) or a.dtype != np.uint8 or not a.flags.c_contiguous
                or not a.flags.writeable or a.ndim != 1 or len(a) != n):
            raise ValueError("F2 read output must be a writable contiguous uint8 vector of n bytes")
        if n:
            with self._dma:
                self._lib.otpu_read(a.ctypes.data, self._uc_addr + pos, n)
        return a
