"""F2 transport against file-backed BARs: identity, ownership, addressing and write fences."""
import ctypes
from dataclasses import replace
import os
import platform
import shutil
import struct
import subprocess

import numpy as np
import pytest

from opentpu.host import board, f2, regs as R, runstate
from opentpu.target import config

BDF = "0000:34:00.0"


class MMIO:
    def __init__(self):
        self.writes = []

    def otpu_write(self, wc, uc, src, n):
        ctypes.memmove(wc, src, n)
        # An uncached read is part of every completed write, before the transport returns.
        self.writes.append((wc, uc, n, ctypes.c_uint32.from_address(uc).value))

    def otpu_read(self, dst, uc, n):
        ctypes.memmove(dst, uc, n)


@pytest.fixture
def pci(tmp_path, monkeypatch):
    monkeypatch.setattr(f2, "_require_host", lambda: None)
    monkeypatch.setattr(f2, "SYSFS", tmp_path / "pci")
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("OTPU_LOCK_HELD", raising=False)
    mmio = MMIO()
    monkeypatch.setattr(f2, "_mmio", lambda: mmio)
    root = f2.SYSFS / BDF
    root.mkdir(parents=True)
    (root / "vendor").write_text("0x1d0f\n")
    (root / "device").write_text("0xf001\n")
    regs = bytearray(4096)
    for off, value in {R.R_ID: R.ID_OTPU, R.R_REGMAP: 4, R.R_CFG_PLATFORM: 3,
                       R.R_CFG_SLICES: 1, R.R_CFG_DRAM: 13 | 2 << 8 | 12 << 16,
                       R.R_VERSION: 128 << 16 | 4 << 8 | 8}.items():
        struct.pack_into("<I", regs, off, value)
    (root / "resource0").write_bytes(regs)
    (root / "resource4").write_bytes(bytes(8192))
    os.link(root / "resource4", root / "resource4_wc")
    return root, mmio


def test_slot_and_bdf_select_the_same_device(monkeypatch):
    def slots(*args, **kw):
        assert args[0] == ["fpga-describe-local-image-slots"]
        return subprocess.CompletedProcess(args[0], 0, "Type FpgaImageSlot VendorId DeviceId DBDF\n"
                                           "AFIDEVICE 0 0x1d0f 0xf001 0000:34:00.0\n"
                                           "AFIDEVICE 1 0x1d0f 0xf001 0000:36:00.0\n")
    monkeypatch.setattr(f2.subprocess, "run", slots)
    assert f2.resolve_device("f2:0") == f2.resolve_device("f2:34:00.0") == BDF
    assert f2.resolve_device("f2:1") == "0000:36:00.0"
    with pytest.raises(RuntimeError, match="no unambiguous"):
        f2.resolve_device("f2:2")
    for bad in ("f2:", "f2:../resource0", "f2:-1", "f2:34:20.0", "f2:34:00.8", "xdma0"):
        with pytest.raises(ValueError):
            f2.resolve_device(bad)


def test_partial_memory_writes_fence_before_control(pci):
    root, native = pci
    t = f2.F2Transport("f2:" + BDF)
    try:
        a = np.arange(137, dtype=np.uint8)
        t.mem_write(1, 3, a)
        expected = bytes(4099) + a.tobytes() + bytes(8192 - 4099 - len(a))
        assert (root / "resource4").read_bytes() == expected
        assert native.writes[-1][1] == t._uc_addr + 4096
        assert native.writes[-1][2] == 137
        assert len(native.writes) == 1
        t.reg_write(R.R_CTRL, 0x12345678)       # reached only after the memory fence returned
        assert t.reg_read(R.R_CTRL) == 0x12345678
        assert (root / "resource0").read_bytes()[R.R_CTRL:R.R_CTRL + 4] == b"\x78\x56\x34\x12"
        out = np.empty(len(a), dtype=np.uint8)
        assert t.mem_read(1, 3, len(a), out) is out
        assert np.array_equal(out, a)
        assert np.array_equal(t.mem_read(0, 0, 64), np.zeros(64, np.uint8))
        # Other boards' DDR CSR windows are acknowledged without hitting F2 control words.
        t.reg_write(R.R_DDR4, 0xffffffff)
        assert t.reg_read(R.R_DDR4) == 0
        for ch, off, n in [(-1, 0, 4), (2, 0, 4), (0, -1, 4), (0, 4094, 4), (0, 0, -1)]:
            with pytest.raises(ValueError):
                t.mem_read(ch, off, n)
        with pytest.raises(ValueError):
            t.mem_write(0, 4095, np.zeros(2, np.uint8))
        with pytest.raises(ValueError, match="aligned"):
            t.reg_write(3, 0)
        with pytest.raises(ValueError, match="output"):
            t.mem_read(0, 0, 4, np.zeros(8, np.uint8)[::2])
        with pytest.raises(ValueError, match="output"):
            t.mem_read(0, 0, 4, np.zeros(4, np.uint32))
    finally:
        t.close()
    with pytest.raises(RuntimeError, match="closed"):
        t.reg_read(R.R_ID)


def test_lock_precedes_bar_open_and_failed_identity_releases_it(pci, monkeypatch):
    root, _ = pci
    held = runstate.DeviceLock("f2-" + BDF)
    seen = []
    real_map = f2._map
    monkeypatch.setattr(f2, "_map", lambda *args: seen.append(args))
    try:
        with pytest.raises(runstate.DeviceBusy):
            f2.F2Transport("f2:34:00.0")
        assert seen == []
    finally:
        held.release()
    monkeypatch.setattr(f2, "_map", real_map)
    for off, wrong, match in [(R.R_ID, 0, "no openTPU"), (R.R_CFG_PLATFORM, 1, "not the F2")]:
        raw = bytearray((root / "resource0").read_bytes())
        saved = raw[:]
        struct.pack_into("<I", raw, off, wrong)
        (root / "resource0").write_bytes(raw)
        with pytest.raises(RuntimeError, match=match):
            f2.F2Transport("f2:" + BDF)
        runstate.DeviceLock("f2-" + BDF, wait=0).release()
        (root / "resource0").write_bytes(saved)
    (root / "vendor").write_text("0x1234\n")
    with pytest.raises(RuntimeError, match="unexpected PCI"):
        f2.F2Transport("f2:" + BDF)


def test_monitor_does_not_lock_or_map_memory(pci):
    root, _ = pci
    (root / "resource4").unlink()
    (root / "resource4_wc").unlink()
    held = runstate.DeviceLock("f2-" + BDF)
    try:
        t = f2.F2Transport("f2:" + BDF, dma=False)
        assert t._otpu_lock is None and t.reg_read(R.R_ID) == R.ID_OTPU
        t.reg_write(R.R_SNAP, 1)
        with pytest.raises(RuntimeError, match="monitor"):
            t.reg_write(R.R_CTRL, R.CTRL_RUN)
        with pytest.raises(RuntimeError, match="not open"):
            t.mem_read(0, 0, 4)
        t.close()
    finally:
        held.release()


def test_close_stops_before_unmapping_and_releases_lock_last(pci):
    t = f2.F2Transport("f2:" + BDF)
    lock = t._otpu_lock
    released = lock.release
    events = []

    def stop():
        assert t.regs is not None and t._uc is not None
        t.reg_write(R.R_CTRL, 0)
        events.append("stop")

    def release():
        assert t.regs is None and t.words is None and t._wc is None and t._uc is None
        events.append("release")
        released()

    lock.cleanup.append(lambda: stop)
    lock.release = release
    t.close()
    assert events == ["stop", "release"]


def test_ecc_scrub_covers_the_exposed_channels_once(pci, monkeypatch):
    root, _ = pci
    (root / "resource4").write_bytes(b"\xff" * 8192)
    t = f2.F2Transport("f2:" + BDF)
    b = board.Board(t, calibrate=False)
    # Small chunks for the miniature file-backed geometry (the real driver uses 64 MiB).
    monkeypatch.setattr(board, "placed", lambda n, card: np.zeros(64, np.uint8))
    try:
        assert b.scrub()
        assert (root / "resource4").read_bytes() == bytes(8192)
        assert t.reg_read(R.R_SCRATCH) == R.DRAM_INIT
        assert not b.scrub()
    finally:
        b.close()
        t.close()


def test_board_default_factory_and_close(pci, monkeypatch):
    monkeypatch.setenv("OTPU_DEVICE", "f2:" + BDF)
    b = board.Board(calibrate=False)
    assert isinstance(b.t, f2.F2Transport)
    b.write(7, np.arange(250, dtype=np.uint8))
    assert np.array_equal(b.read(7, 250), np.arange(250, dtype=np.uint8))
    b.close()
    assert b.t._closed
    runstate.DeviceLock("f2-" + BDF, wait=0).release()
    calls = []
    monkeypatch.setattr(board, "XdmaTransport", lambda *a, **kw: calls.append((a, kw)))
    board.open_transport("/dev/xdma3", dma=False)
    assert calls == [(("/dev/xdma3",), {"dma": False})]


def test_backend_owned_transport_unmaps_before_releasing_lock(pci, monkeypatch):
    t = f2.F2Transport("f2:" + BDF)
    t.ecc = False
    monkeypatch.setattr(board, "check_fits", lambda *args: None)
    cfg = replace(config("f2"), DRAM_BYTES=8192, IMEM_WORDS=64, SEQ_WC=False)
    backend = board.BoardBackend(cfg, [np.zeros(64, np.uint8)], transport=t,
                                 status=False, own_transport=True)
    lock = t._otpu_lock
    release = lock.release

    def checked_release():
        assert t._closed and t.regs is None and t._uc is None and t._wc is None
        release()

    monkeypatch.setattr(lock, "release", checked_release)
    backend.close()
    assert backend.board.lock is None and t._otpu_lock is None
    runstate.DeviceLock("f2-" + BDF, wait=0).release()


@pytest.mark.parametrize("failed_stage", ["scrub", "upload"])
def test_backend_constructor_failure_closes_owned_transport(pci, monkeypatch, failed_stage):
    t = f2.F2Transport("f2:" + BDF)
    t.ecc = False
    monkeypatch.setattr(board, "check_fits", lambda *args: None)
    cfg = replace(config("f2"), DRAM_BYTES=8192, IMEM_WORDS=64, SEQ_WC=False)

    def fail(*args):
        raise OSError("injected " + failed_stage + " failure")

    if failed_stage == "scrub":
        monkeypatch.setattr(board.Board, "scrub", fail)
    else:
        monkeypatch.setattr(t, "mem_write", fail)
    with pytest.raises(OSError, match="injected"):
        board.BoardBackend(cfg, [np.zeros(64, np.uint8)], transport=t,
                           status=False, own_transport=True)
    assert t._closed and t.regs is None and t._uc is None and t._wc is None
    runstate.DeviceLock("f2-" + BDF, wait=0).release()


@pytest.mark.skipif(platform.machine().lower() not in ("x86_64", "amd64") or not shutil.which("cc"),
                    reason="native F2 MMIO helper needs x86-64 and a C compiler")
def test_native_helper_copies_bytes_and_reads_back():
    lib = f2._mmio()
    src = ctypes.create_string_buffer(bytes(range(128)))
    dst = ctypes.create_string_buffer(256)
    result = ctypes.create_string_buffer(128)
    p = ctypes.addressof(dst)
    lib.otpu_write(p + 3, p, ctypes.addressof(src), 128)
    lib.otpu_read(ctypes.addressof(result), p + 3, 128)
    assert result.raw == bytes(range(128))
    assert dst.raw[:3] == bytes(3) and dst.raw[131:] == bytes(125)


@pytest.mark.parametrize("slices,nch", [(1, 8), (2, 8), (1, 16), (4, 8)])
def test_hbm_geometry_maps_every_slice_channel_and_beat(pci, slices, nch):
    """The HBM build's geometry (docs/f2_hbm.md): S slices x NCH pseudo-channels at BAR4 offset
    (NCH s + c) x channel bytes; Board.write / read interleave a slice's logical bytes over its
    NCH channels by the adapter's map (beat_at: the pair rotation and its CHASH fold), edges and
    whole groups alike, and read them back. A miniature 4 KiB per channel stands in for 512 MiB."""
    root, _ = pci
    regs = bytearray((root / "resource0").read_bytes())
    struct.pack_into("<I", regs, R.R_CFG_SLICES, slices)
    struct.pack_into("<I", regs, R.R_CFG_DRAM, (12 + (nch.bit_length() - 1)) | nch << 8 | 12 << 16)
    struct.pack_into("<I", regs, R.R_CAPS, R.CAP_CHASH)
    (root / "resource0").write_bytes(regs)
    size = slices * nch * 4096
    os.unlink(root / "resource4_wc")
    (root / "resource4").write_bytes(bytes(size))
    os.link(root / "resource4", root / "resource4_wc")
    t = f2.F2Transport("f2:" + BDF)
    b = board.Board(t, calibrate=False)
    try:
        assert t.bases == tuple(j * 4096 for j in range(slices * nch))
        assert b.nch == nch and b.slices == slices and b.chash
        rng = np.random.default_rng(nch * 10 + slices)
        for s in range(slices):
            n = nch * 4096 - 3 * 64 - 5
            data = rng.integers(0, 256, n, dtype=np.uint8)
            b.write(70, data, s)                       # unaligned edges, whole groups between
            assert np.array_equal(b.read(70, n, s), data)
            raw = np.frombuffer((root / "resource4").read_bytes(), np.uint8)
            for x in range(70, 70 + n, 61):           # each byte where the adapter puts it
                c, off = board.beat_at(x, True, nch)
                assert raw[(nch * s + c) * 4096 + off] == data[x - 70]
    finally:
        b.close()
        t.close()


def test_hbm_host_map_is_the_adapters():
    """beat_at is the inverse of rtl/mem/otpu_native_dram.sv's chan_of / cb_of as its testbench
    states it (sim/verilator/tb_native_nch.sv lbeat): chunk m = u x NCH / 2 + (pair ^ fold(u))."""
    for nch in (2, 4, 8, 16):
        pb = (nch // 2).bit_length() - 1
        for c in range(nch):
            for u in (0, 1, 2, 3, 7, 255, 4096 + 5, (1 << 23) + 77):
                f = 0
                for k in range(0, 27, pb or 27):
                    f ^= (u >> k) & ((1 << pb) - 1)
                m = u * (nch // 2) + ((c // 2 ^ f) & (nch // 2 - 1))
                b = 2 * m + (c % 2 ^ (m.bit_count() & 1))
                assert board.beat_at(b * 64 + 9, True, nch) == (c, u * 64 + 9)
