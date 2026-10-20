"""The lane memory's host side (P2: targets/f2-p2.toml, docs/v4/p2-f2.md "Host"): on a bitstream
whose CFG_DRAM reports copies of the slice's image ([31:24] + 1), the F2 transport writes every
byte to every copy (copy k's window of channel c at (k x S x NCH + c) x the channel's bytes) and
reads copy 0; a Board on it sees one copy's two channels. File-backed BARs, as e04/tpubench's
tests/test_f2_host.py; a miniature 4 KiB a channel stands in for 1 GiB."""
import ctypes
import os
import struct

import numpy as np
import pytest

from opentpu.host import board, f2, regs as R

BDF = "0000:34:00.0"
WIN = 4096


class MMIO:
    def otpu_write(self, wc, uc, src, n):
        ctypes.memmove(wc, src, n)

    def otpu_read(self, dst, uc, n):
        ctypes.memmove(dst, uc, n)


def _card(tmp_path, monkeypatch, dram: int, size: int):
    monkeypatch.setattr(f2, "_require_host", lambda: None)
    monkeypatch.setattr(f2, "SYSFS", tmp_path / "pci")
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("OTPU_LOCK_HELD", raising=False)
    monkeypatch.setattr(f2, "_mmio", lambda: MMIO())
    root = f2.SYSFS / BDF
    root.mkdir(parents=True)
    (root / "vendor").write_text("0x1d0f\n")
    (root / "device").write_text("0xf001\n")
    regs = bytearray(4096)
    for off, value in {R.R_ID: R.ID_OTPU, R.R_REGMAP: 4, R.R_CFG_PLATFORM: 3,
                       R.R_CFG_SLICES: 1, R.R_CFG_DRAM: dram, R.R_CAPS: R.CAP_CHASH,
                       R.R_VERSION: 128 << 16 | 4 << 8 | 8}.items():
        struct.pack_into("<I", regs, off, value)
    (root / "resource0").write_bytes(regs)
    (root / "resource4").write_bytes(bytes(size))
    os.link(root / "resource4", root / "resource4_wc")
    return root


@pytest.mark.parametrize("copies", [1, 2, 4, 8])
def test_every_copy_takes_every_write_and_reads_come_from_copy_0(tmp_path, monkeypatch, copies):
    root = _card(tmp_path, monkeypatch, 13 | 2 << 8 | 12 << 16 | (copies - 1) << 24,
                 copies * 2 * WIN)
    t = f2.F2Transport("f2:" + BDF)
    b = board.Board(t, calibrate=False)
    try:
        assert (t.copies, t.nwin) == (copies, 2)
        assert t.bases == tuple(j * WIN for j in range(2 * copies))
        assert b.slices == 1 and b.info()["cfg"]["copies"] == copies
        rng = np.random.default_rng(copies)
        n = 2 * WIN - 3 * 64 - 5
        data = rng.integers(0, 256, n, dtype=np.uint8)
        b.write(70, data)                              # unaligned edges, whole chunks between
        raw = np.frombuffer((root / "resource4").read_bytes(), np.uint8)
        for k in range(1, copies):                     # each copy byte for byte copy 0's
            assert np.array_equal(raw[k * 2 * WIN:(k + 1) * 2 * WIN], raw[:2 * WIN]), k
        for x in range(70, 70 + n, 61):                # copy 0 as the adapter maps it
            c, off = board.beat_at(x, True)
            assert raw[c * WIN + off] == data[x - 70]
        assert np.array_equal(b.read(70, n), data)
        if copies > 1:                                 # reads are copy 0's alone
            junk = bytearray((root / "resource4").read_bytes())
            junk[2 * WIN:] = b"\xa5" * (len(junk) - 2 * WIN)
            (root / "resource4").write_bytes(bytes(junk))
            assert np.array_equal(b.read(70, n), data)
    finally:
        b.close()
        t.close()


@pytest.mark.parametrize("dram", [
    14 | 4 << 8 | 12 << 16 | 1 << 24,                  # copies of a 4-channel slice
    31 | 2 << 8 | 30 << 16 | 15 << 24,                 # 16 copies of 2 GiB: past the 16 GiB HBM
])
def test_copies_only_of_one_two_channel_slice_within_the_hbm(tmp_path, monkeypatch, dram):
    _card(tmp_path, monkeypatch, dram, 8 * WIN)
    with pytest.raises(RuntimeError, match="geometry"):
        f2.F2Transport("f2:" + BDF)


def test_cfg_reports_copies_and_chunk_lanes():
    """CFG_DRAM [31:24] + 1 copies, CFG_UARCH [28:25] + 1 chunk lanes (rtl/shell/otpu_ctrl.sv);
    every bitstream before reads 1 and 1, and cfg_words inverts cfg."""
    old = R.cfg([1, 65536, 32768, 128, 32 | 2 << 8 | 31 << 16, 2 | 3 << 4 | 2 << 8 | 8 << 16 | 1 << 24,
                 1, 0x4ffd1e17])
    assert (old["copies"], old["MXU_C"]) == (1, 1)
    p2 = R.cfg([1, 65536, 32768, 128, 31 | 2 << 8 | 30 << 16 | 3 << 24,
                2 | 3 << 4 | 2 << 8 | 8 << 16 | 1 << 24 | 3 << 25, 3, 0])
    assert (p2["copies"], p2["MXU_C"], p2["channels"], p2["channel_bytes"], p2["DRAM_BYTES"]) == \
        (4, 4, 2, 1 << 30, 1 << 31)
    assert R.cfg(R.cfg_words(p2)) == p2
