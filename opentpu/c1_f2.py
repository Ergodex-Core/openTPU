"""C1 isolated F2 benchmark ABI, byte-preserving fixtures and checked counter runs.

This is the E16/S4 weight-only microbenchmark on eight 64-byte native ports.
Every native beat supplies two 32-byte groups. ACT/scales are resident. Results
retain their transport evidence kind; this module does not render board profiles
or expose a command that can load an FPGA.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import fp32 as F, isa as I
from .isasim import Config, Slice

ID, ABI = 0x43314D42, 1
PC_BYTES = 512 << 20
R_ID, R_ABI, R_BUILD, R_STATUS, R_CTRL = 0x000, 0x004, 0x008, 0x00C, 0x010
R_MODE, R_KB, R_ROWS, R_BASE = 0x014, 0x018, 0x01C, 0x020
R_MEM_ADDR, R_MEM_WDATA, R_MEM_RDATA = 0x024, 0x028, 0x02C
R_ERROR, R_COUNT, R_CYCLES_LO, R_CYCLES_HI = 0x030, 0x034, 0x038, 0x03C
R_CORE_KHZ, R_MAX_ROWS, R_DEPTH, R_CREDITS, R_HBM_READY = 0x06C, 0x070, 0x074, 0x078, 0x07C
BUSY, DONE, ERROR, ABORTED, DRAINING = 1, 2, 4, 8, 16
START, ABORT, CLEAR = 1, 2, 4
ACT, DA, SA, DW, SW = 0x00000, 0x10000, 0x11000, 0x20000, 0x40000
RESULT, INTLO, INTHI = 0x50000, 0x60000, 0x70000
COUNTERS = dict(accepted_32B_groups=0x040, committed_rows=0x044, read_issued=0x048,
                read_returned=0x04C, advance=0x050, native_blocked_cycles=0x054,
                weight_wait=0x058, scale_wait=0x05C, act_wait=0x060,
                output_wait=0x064, max_outstanding=0x068)
PC_COUNTERS = dict(issued=0x00, returned=0x04, max_outstanding=0x08,
                   blocked_cycles=0x0C, reserved_latency=0x10,
                   credit_wait_cycles=0x14)


@dataclass
class Fixture:
    weights: np.ndarray
    acts: np.ndarray
    dw: np.ndarray
    da: np.ndarray
    sw: np.ndarray
    sa: np.ndarray
    base_beat: int = 16

    @property
    def rows(self):
        return self.weights.shape[0]

    @property
    def kb(self):
        return self.weights.shape[1]

    def validate(self):
        if self.weights.ndim != 3 or self.weights.shape[2] != 128:
            raise ValueError("weights must be ROWS x KB x 128")
        if not 1 <= self.rows <= 512 or not 1 <= self.kb <= 128:
            raise ValueError("benchmark supports ROWS=1..512 and KB=1..128")
        for name, shape, dtype in (
            ("weights", (self.rows, self.kb, 128), np.int8),
            ("acts", (self.kb, 128), np.int8), ("dw", (self.rows, self.kb), np.uint8),
            ("da", (self.kb, 4), np.uint8), ("sw", (self.rows,), np.uint32),
            ("sa", (1,), np.uint32)):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
                raise ValueError(f"invalid {name} shape/dtype")
        if not isinstance(self.base_beat, int) or self.base_beat < 0 or (
                self.base_beat + ((self.rows + 15) // 16) * 4 * self.kb) * 64 > PC_BYTES:
            raise ValueError("native allocation exceeds a 512 MiB PC")

    def banks(self):
        self.validate()
        banks = [np.zeros(((self.rows + 15) // 16) * 2 * self.kb * 128, np.uint8) for _ in range(8)]
        for row in range(self.rows):
            pc = (row % 16) // 2
            offset = ((row // 16) * 2 + row % 2) * self.kb * 128
            banks[pc][offset:offset+self.kb*128] = self.weights[row].reshape(-1).view(np.uint8)
        return tuple(banks)

    def expected_pc_beats(self):
        return [(len(range(2*pc, self.rows, 16)) + len(range(2*pc+1, self.rows, 16))) * self.kb * 2
                for pc in range(8)]

    @classmethod
    def from_workload(cls, case, base_beat=16):
        """Use the exact canonical C1 RTL fixture operands and plain-output origin."""
        if case.n0 or case.k0 or case.ab:
            raise ValueError("C1 benchmark requires origin-zero plain output")
        value = cls(case.weights.reshape(case.n, case.kb, 128).view(np.int8).copy(),
                    case.state.act[0, :case.kb*128].reshape(case.kb, 128).copy(),
                    case.scales[:, 4:4+case.kb].copy(), case.state.adoff[0, :case.kb].copy(),
                    np.frombuffer(case.scales[:, :4].copy().tobytes(), dtype='<u4').copy(),
                    F.bits(case.state.ascale[0, :1]).copy(), base_beat)
        value.validate()
        return value

    def digest(self):
        self.validate()
        h = hashlib.sha256(json.dumps(dict(layout="c1-f2-e16-two-engines-per-pc-v1", rows=self.rows,
                     kb=self.kb, base_beat=self.base_beat), sort_keys=True).encode())
        for name in ("weights", "acts", "dw", "da", "sw", "sa"):
            h.update(np.ascontiguousarray(getattr(self, name)).tobytes())
        return h.hexdigest()

    def save(self, path):
        self.validate()
        np.savez(path, **{name: getattr(self, name) for name in
                         ("weights", "acts", "dw", "da", "sw", "sa")},
                 base_beat=np.array(self.base_beat, np.uint32))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            result = cls(**{name: data[name].copy() for name in
                           ("weights", "acts", "dw", "da", "sw", "sa")},
                         base_beat=int(data["base_beat"]))
        result.validate()
        return result

    def reference(self):
        """Canonical ISA MM bits plus independently computed signed48 diagnostics."""
        self.validate()
        rs, srs = self.kb * 128, ((self.kb + 7) // 4) * 4
        ssa = self.rows * rs
        cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, TMEM_WORDS=512,
                     DRAM_BYTES=ssa + self.rows * srs + 128)
        dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
        dram[:ssa] = self.weights.reshape(-1).view(np.uint8)
        for r in range(self.rows):
            dram[ssa+r*srs:ssa+r*srs+4] = self.sw[r:r+1].view(np.uint8)
            dram[ssa+r*srs+4:ssa+r*srs+4+self.kb] = self.dw[r]
        sl = Slice(cfg, 0, [], dram)
        sl.act[0, :self.kb*128] = self.acts.reshape(-1)
        sl.adoff[0, :self.kb] = self.da
        sl.ascale[0, :self.kb] = F.from_bits(self.sa)[0]
        sl._mm(I.mm(0, ssa, 0, self.rows, self.kb, rs, 1, 1, 0, srs))
        dots = (self.weights.astype(np.int64)*self.acts.astype(np.int64)).reshape(
            self.rows, self.kb, 4, 32).sum(3)
        shifts = 16-self.dw.astype(np.int64)[:, :, None]-self.da.astype(np.int64)[None]
        ints = np.where(shifts >= 0, dots << np.maximum(0, shifts), 0).sum((1, 2))
        return sl.tmem[:self.rows].copy(), ints.tolist()


def make_fixture(rows=31, kb=9, seed=20261019, base_beat=16):
    if not 1 <= rows <= 512 or not 1 <= kb <= 128:
        raise ValueError("benchmark supports ROWS=1..512 and KB=1..128")
    from . import c1_rtl
    return Fixture.from_workload(c1_rtl.workload(n=rows, kb=kb, seed=seed, ab=0, k0=0), base_beat)


class Benchmark:
    """Checked ABI client over injected reg_read/reg_write/mem_read/mem_write transport."""
    def __init__(self, transport, *, expected_build_id):
        self.t = transport
        self.identity = {name: self.t.reg_read(reg) for name, reg in (
            ("id", R_ID), ("abi", R_ABI), ("build_id", R_BUILD), ("core_khz", R_CORE_KHZ),
            ("max_rows", R_MAX_ROWS), ("depth", R_DEPTH), ("credits", R_CREDITS))}
        if (self.identity["id"], self.identity["abi"], self.identity["build_id"]) != (ID, ABI, expected_build_id):
            raise RuntimeError(f"wrong benchmark image identity: {self.identity}")
        if self.identity["max_rows"] != 512 or self.identity["depth"] != 128 or not 1 <= self.identity["credits"] <= 32:
            raise RuntimeError("unsupported benchmark memory/credit geometry")
        if self.identity["core_khz"] != 200000:
            raise RuntimeError("benchmark build must declare the qualified 200 MHz clock")
        self.fixture = self.expected = None

    def idle(self):
        status = self.t.reg_read(R_STATUS)
        if status & (BUSY | DRAINING):
            raise RuntimeError("benchmark still owns native transactions")
        return status

    def _write_words(self, addr, words):
        self.t.reg_write(R_MEM_ADDR, addr)
        for value in words:
            self.t.reg_write(R_MEM_WDATA, int(value))

    def _read_words(self, addr, n):
        self.t.reg_write(R_MEM_ADDR, addr)
        return [self.t.reg_read(R_MEM_RDATA) for _ in range(n)]

    def preload(self, fixture):
        self.idle()
        fixture.validate()
        if not self.t.reg_read(R_HBM_READY) & 1:
            raise RuntimeError("HBM is not ready")
        self.t.reg_write(R_CTRL, CLEAR)
        # Each write is completed by the transport's UC read barrier before START.
        for pc, data in enumerate(fixture.banks()):
            self.t.mem_write(pc, fixture.base_beat*64, data)
            if not np.array_equal(self.t.mem_read(pc, fixture.base_beat*64, len(data)), data):
                raise RuntimeError(f"HBM preload readback mismatch on PC {pc}")
        self._write_words(ACT, np.frombuffer(fixture.acts.tobytes(), dtype="<u4"))
        self._write_words(DA, np.frombuffer(fixture.da.tobytes(), dtype="<u4"))
        self._write_words(SA, fixture.sa)
        self._write_words(SW, fixture.sw)
        for r in range(fixture.rows):
            self._write_words(DW+r*128, fixture.dw[r])
        self.fixture = fixture
        self.fixture_sha = fixture.digest()
        self.expected = fixture.reference()

    def abort(self, timeout=5.0):
        """Drain native responses before releasing ownership; timeout remains an error."""
        self.t.reg_write(R_CTRL, ABORT)
        deadline = time.monotonic()+timeout
        while self.t.reg_read(R_STATUS) & (BUSY | DRAINING):
            if time.monotonic() >= deadline:
                raise TimeoutError("C1 abort did not drain; benchmark/image recovery required")
            time.sleep(0.001)

    def run(self, mode=0, *, timeout=30.0):
        if mode != 0 or self.fixture is None or timeout <= 0:
            raise ValueError("preload a fixture, select C1 plain mode 0 and a positive timeout")
        self.idle()
        f = self.fixture
        if f.digest() != self.fixture_sha:
            raise RuntimeError("fixture changed after device preload")
        self.t.reg_write(R_CTRL, CLEAR)
        for reg, value in ((R_MODE, mode), (R_KB, f.kb), (R_ROWS, f.rows), (R_BASE, f.base_beat)):
            self.t.reg_write(reg, value)
        self.t.reg_write(R_CTRL, START)
        deadline = time.monotonic()+timeout
        try:
            while True:
                status = self.t.reg_read(R_STATUS)
                if status & DONE and not status & (BUSY | DRAINING):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("C1 benchmark completion timeout")
                time.sleep(0.001)
        except BaseException:
            self.abort()
            raise
        error = self.t.reg_read(R_ERROR)
        if error or status & (ERROR | ABORTED):
            raise RuntimeError(f"C1 failed: status={status:#x} error={error:#x}")
        count = self.t.reg_read(R_COUNT)
        if count != f.rows:
            raise RuntimeError(f"C1 result count {count} != {f.rows}")
        cycles = self.t.reg_read(R_CYCLES_LO) | (self.t.reg_read(R_CYCLES_HI) << 32)
        if cycles >= 1 << 32:
            raise RuntimeError("run may have wrapped 32-bit stall counters; no counter profile exported")
        counters = {name: self.t.reg_read(reg) for name, reg in COUNTERS.items()}
        pcs = [{name: self.t.reg_read(0x100+pc*0x20+offset) for name, offset in PC_COUNTERS.items()}
               for pc in range(8)]
        bits = self._read_words(RESULT, count)
        lo, hi = self._read_words(INTLO, count), self._read_words(INTHI, count)
        integers = [a | ((b & 0xFFFF) << 32) for a, b in zip(lo, hi)]
        integers = [v-(1 << 48) if v & (1 << 47) else v for v in integers]
        if bits != self.expected[0].tolist() or integers != self.expected[1]:
            bad = next(r for r in range(count) if bits[r] != int(self.expected[0][r]) or integers[r] != self.expected[1][r])
            raise RuntimeError(f"C1 arithmetic mismatch at row {bad}: bits={bits[bad]:08x} integer={integers[bad]}")
        beats = f.expected_pc_beats()
        groups = f.rows * f.kb * 4
        if (cycles <= 0 or counters["accepted_32B_groups"] != groups or
                counters["committed_rows"] != f.rows or counters["advance"] != groups):
            raise RuntimeError("C1 cycle/group counters inconsistent with the completed run")
        if counters["read_issued"] != sum(beats) or counters["read_returned"] != sum(beats):
            raise RuntimeError("C1 aggregate native traffic mismatch")
        for pc, (values, expected) in enumerate(zip(pcs, beats)):
            if values["issued"] != expected or values["returned"] != expected or values["max_outstanding"] > 2*self.identity["credits"] or values["reserved_latency"] != 0:
                raise RuntimeError(f"C1 PC {pc} traffic/credit mismatch")
        if any(counters[name] for name in ("scale_wait", "act_wait", "output_wait")):
            raise RuntimeError("C1 reserved stall counters must remain zero in ABI 1")
        if counters["max_outstanding"] > 16*self.identity["credits"]:
            raise RuntimeError("C1 aggregate receive reservation overflow")
        if self.t.reg_read(R_STATUS) != status:
            raise RuntimeError("C1 completion snapshot changed while being read")
        return dict(evidence_kind=getattr(self.t, "evidence_kind", "test"),
                    measured_at_utc=datetime.now(timezone.utc).isoformat(),
                    device=getattr(self.t, "dev", "injected transport"), identity=self.identity.copy(),
                    mode=mode, rows=f.rows, kb=f.kb, base_beat=f.base_beat,
                    fixture_sha256=self.fixture_sha, cycles=cycles, counters=counters,
                    pcs=pcs, bits=bits, integers=integers, bit_exact=True,
                    weight_bytes=sum(beats)*64, status=status, error=error,
                    measured_window="COMMAND acceptance through final fp32 result; excludes OCL preload, ACT snapshot copy and host readback")

