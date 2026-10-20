"""C2 isolated F2 benchmark ABI, byte-preserving fixtures and checked counter runs.

This is a separate eight-PC microbenchmark image, not the production TPU ABI or
the sixteen-PC C2 proposal image. Only hardware transport results can be exported
as board measurements. No event times are inferred from aggregate counters.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import fp32 as F, isa as I, lens
from .isasim import Config, Slice
from .profile import UNITS

ID, ABI = 0x43324D42, 1
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
COUNTERS = dict(accepted_groups=0x040, committed_groups=0x044, read_issued=0x048,
                read_returned=0x04C, advance=0x050, native_blocked_cycles=0x054,
                weight_wait=0x058, scale_wait=0x05C, act_wait=0x060,
                output_wait=0x064, max_outstanding=0x068)
PC_COUNTERS = dict(issued=0x00, returned=0x04, max_outstanding=0x08,
                   blocked_cycles=0x0C, pending_no_response_cycles=0x10,
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
        if not 1 <= self.rows <= 512 or not 1 <= self.kb <= 255:
            raise ValueError("benchmark supports ROWS=1..512 and KB=1..255")
        for name, shape, dtype in (
            ("weights", (self.rows, self.kb, 128), np.int8),
            ("acts", (self.kb, 128), np.int8), ("dw", (self.rows, self.kb), np.uint8),
            ("da", (self.kb, 4), np.uint8), ("sw", (self.rows,), np.uint32),
            ("sa", (1,), np.uint32)):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
                raise ValueError(f"invalid {name} shape/dtype")
        if not isinstance(self.base_beat, int) or self.base_beat < 0 or (
                self.base_beat + ((self.rows + 3) // 4) * self.kb) * 64 > PC_BYTES:
            raise ValueError("native allocation exceeds a 512 MiB PC")

    def banks(self):
        self.validate()
        banks = [np.zeros(((self.rows + 3) // 4, self.kb, 64), np.uint8) for _ in range(8)]
        for r in range(self.rows):
            banks[2*(r % 4)][r//4] = self.weights[r, :, :64].view(np.uint8)
            banks[2*(r % 4)+1][r//4] = self.weights[r, :, 64:].view(np.uint8)
        return tuple(bank.reshape(-1) for bank in banks)

    def expected_pc_beats(self):
        return [len(range(pc//2, self.rows, 4)) * self.kb for pc in range(8)]

    def digest(self):
        self.validate()
        h = hashlib.sha256(json.dumps(dict(layout="c2-f2-pairpc-v1", rows=self.rows,
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
        cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=256, TMEM_WORDS=1024,
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
    if not 1 <= rows <= 512 or not 1 <= kb <= 255:
        raise ValueError("benchmark supports ROWS=1..512 and KB=1..255")
    rng = np.random.default_rng(seed)
    result = Fixture(rng.integers(-128, 128, (rows, kb, 128), dtype=np.int16).astype(np.int8),
                     rng.integers(-128, 128, (kb, 128), dtype=np.int16).astype(np.int8),
                     rng.integers(0, 12, (rows, kb), dtype=np.uint8),
                     rng.integers(0, 12, (kb, 4), dtype=np.uint8),
                     F.bits(np.ldexp(rng.standard_normal(rows).astype(np.float32), -9)),
                     F.bits(np.array([0.03125], np.float32)), base_beat)
    result.da[0] = [0, 16, 17, 255]
    result.dw[:, 0] = np.resize(np.array([0, 1, 16, 255], np.uint8), rows)
    return result


class Benchmark:
    """Checked ABI client over injected reg_read/reg_write/mem_read/mem_write transport."""
    def __init__(self, transport, *, expected_build_id):
        self.t = transport
        self.identity = {name: self.t.reg_read(reg) for name, reg in (
            ("id", R_ID), ("abi", R_ABI), ("build_id", R_BUILD), ("core_khz", R_CORE_KHZ),
            ("max_rows", R_MAX_ROWS), ("depth", R_DEPTH), ("credits", R_CREDITS))}
        if (self.identity["id"], self.identity["abi"], self.identity["build_id"]) != (ID, ABI, expected_build_id):
            raise RuntimeError(f"wrong benchmark image identity: {self.identity}")
        if self.identity["max_rows"] != 512 or self.identity["depth"] != 256 or not 1 <= self.identity["credits"] <= 32:
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
            self._write_words(DW+r*256, fixture.dw[r])
        self.fixture = fixture
        self.fixture_sha = fixture.digest()
        self.expected = fixture.reference()

    def abort(self, timeout=5.0):
        """Drain native responses before releasing ownership; timeout remains an error."""
        self.t.reg_write(R_CTRL, ABORT)
        deadline = time.monotonic()+timeout
        while self.t.reg_read(R_STATUS) & (BUSY | DRAINING):
            if time.monotonic() >= deadline:
                raise TimeoutError("C2 abort did not drain; benchmark/image recovery required")
            time.sleep(0.001)

    def failure_snapshot(self):
        """Bounded, read-only observations; live counters are not an atomic profile."""
        snapshot = dict(kind="c2-f2-failure-snapshot", bit_exact=False,
                        profile_eligible=False, atomic=False,
                        observed_at_utc=datetime.now(timezone.utc).isoformat(),
                        identity={}, registers={}, counters={}, pcs=[], read_errors=[])
        def read(target, name, address):
            try:
                target[name] = self.t.reg_read(address)
            except Exception as exc:
                snapshot["read_errors"].append(dict(address=address, name=name,
                    error_type=type(exc).__name__, error=str(exc)))
        for name, address in (("id", R_ID), ("abi", R_ABI), ("build_id", R_BUILD),
                              ("core_khz", R_CORE_KHZ), ("max_rows", R_MAX_ROWS),
                              ("depth", R_DEPTH), ("credits", R_CREDITS)):
            read(snapshot["identity"], name, address)
        for name, address in (("status_before", R_STATUS), ("error", R_ERROR),
                              ("count", R_COUNT), ("mode", R_MODE), ("kb", R_KB),
                              ("rows", R_ROWS), ("base_beat", R_BASE),
                              ("hbm_ready", R_HBM_READY)):
            read(snapshot["registers"], name, address)
        snapshot["cycle_samples"] = []
        for _ in range(3):
            sample = {}
            for name, address in (("high_before", R_CYCLES_HI),
                                  ("low", R_CYCLES_LO), ("high_after", R_CYCLES_HI)):
                read(sample, name, address)
            snapshot["cycle_samples"].append(sample)
            if len(sample) == 3 and sample["high_before"] == sample["high_after"]:
                snapshot["cycles"] = sample["low"] | (sample["high_after"] << 32)
                break
        for name, address in COUNTERS.items():
            read(snapshot["counters"], name, address)
        for pc in range(8):
            values = {}
            for name, offset in PC_COUNTERS.items():
                read(values, name, 0x100+pc*0x20+offset)
            snapshot["pcs"].append(values)
        read(snapshot["registers"], "status_after", R_STATUS)
        snapshot["counters_may_wrap"] = snapshot.get("cycles", 1 << 32) >= 1 << 32
        snapshot["limitations"] = [
            "Sequential live register observations, not a coherent counter profile.",
            "32-bit counters may wrap during a timeout; no derived rates or success claim.",
            "Read count is bounded; an individual MMIO access remains subject to host/device behavior."]
        return snapshot

    def run(self, mode, *, timeout=30.0):
        if mode not in (0, 1) or self.fixture is None or timeout <= 0:
            raise ValueError("preload a fixture, select mode 0/1 and a positive timeout")
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
                    raise TimeoutError("C2 benchmark completion timeout")
                time.sleep(0.001)
        except BaseException as exc:
            failure = dict(kind="c2-f2-run-failure", bit_exact=False, profile_eligible=False,
                           mode=mode, rows=f.rows, kb=f.kb, base_beat=f.base_beat,
                           fixture_sha256=self.fixture_sha, timeout_s=timeout,
                           error_type=type(exc).__name__, error=str(exc))
            exc.c2_failure = failure
            try:
                failure["before_abort"] = self.failure_snapshot()
            except BaseException as snapshot_error:
                failure["snapshot_error"] = dict(error_type=type(snapshot_error).__name__,
                                                error=str(snapshot_error))
            try:
                self.abort()
                failure["abort"] = dict(attempted=True, drained=True)
            except BaseException as abort_error:
                failure["abort"] = dict(attempted=True, drained=False,
                                       error_type=type(abort_error).__name__, error=str(abort_error))
            raise
        error = self.t.reg_read(R_ERROR)
        if error or status & (ERROR | ABORTED):
            raise RuntimeError(f"C2 failed: status={status:#x} error={error:#x}")
        count = self.t.reg_read(R_COUNT)
        if count != f.rows:
            raise RuntimeError(f"C2 result count {count} != {f.rows}")
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
            raise RuntimeError(f"C2 arithmetic mismatch at row {bad}: bits={bits[bad]:08x} integer={integers[bad]}")
        beats = f.expected_pc_beats()
        groups = f.rows if mode == 0 else (f.rows+3)//4
        if cycles <= 0 or counters["accepted_groups"] != groups or counters["committed_groups"] != groups:
            raise RuntimeError("C2 cycle/group counters inconsistent with the completed run")
        if counters["read_issued"] != sum(beats) or counters["read_returned"] != sum(beats):
            raise RuntimeError("C2 aggregate native traffic mismatch")
        for pc, (values, expected) in enumerate(zip(pcs, beats)):
            if values["issued"] != expected or values["returned"] != expected or values["max_outstanding"] > self.identity["credits"]:
                raise RuntimeError(f"C2 PC {pc} traffic/credit mismatch")
        if self.t.reg_read(R_STATUS) != status:
            raise RuntimeError("C2 completion snapshot changed while being read")
        return dict(evidence_kind=getattr(self.t, "evidence_kind", "test"),
                    measured_at_utc=datetime.now(timezone.utc).isoformat(),
                    device=getattr(self.t, "dev", "injected transport"), identity=self.identity.copy(),
                    mode=mode, rows=f.rows, kb=f.kb, base_beat=f.base_beat,
                    fixture_sha256=self.fixture_sha, cycles=cycles, counters=counters,
                    pcs=pcs, bits=bits, integers=integers, bit_exact=True,
                    weight_bytes=sum(beats)*64, status=status, error=error,
                    measured_window="START acceptance through final core commit; excludes preload and host readback")


def lens_data(result, name):
    if result.get("evidence_kind") != "f2-hardware" or not result.get("bit_exact"):
        raise ValueError("board profiles require bit-verified physical F2 counter evidence")
    cycles, c = result["cycles"], result["counters"]
    # Maximum per-PC beats is only a native-interface bandwidth lower bound, not
    # an observed critical path, HBM peak or serialized sum of waits.
    bound = max(pc["returned"] for pc in result["pcs"])
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=256, TMEM_WORDS=512)
    return dict(kind="board", name=name+" [F2 / aggregate counters]", cycles=cycles,
        clock_mhz=result["identity"]["core_khz"]/1000,
        config=lens._config(cfg, {"MEM": "hbm", "PCS": 8, "C2_COLUMNS": 4, "C": 1}),
        measurements=dict(unit_utilization=False, instruction_timeline=False),
        roofline=dict(kind="native-interface-lower-bound", bound=bound, efficiency=bound/cycles,
                      per_slice=[dict(bound=bound, portb=sum(pc["returned"] for pc in result["pcs"]), porta=0)],
                      note="native 8 x 64-byte interface floor only, not HBM peak or a measured critical path"),
        macs=result["rows"]*result["kb"]*128, bytes=result["weight_bytes"],
        peak_macs=512, peak_bytes=512, units=UNITS,
        slices=[dict(ports={}, busy={}, lose={}, buckets={})],
        instrs=[], programs=[[]], sources=[], mxu_gaps={}, board=result,
        notes=[dict(level="info", text="Physical F2 counters; no per-instruction or per-cycle timeline. "
                    "Elapsed cycles are read from the device, not inferred from host wall time. "
                    "Clock comes from the image configuration; it is not a frequency measurement."),
               dict(level="info", text=result["measured_window"]+". Same weight addresses and bytes in both modes; "
                    "ACT/scales resident, local result sink, no full-token speedup claim."),
               dict(level="info", text=f"Operand-fill occupancy: weight {c['weight_wait']}, "
                    f"scale {c['scale_wait']}, ACT {c['act_wait']} cycles with at least one required "
                    "operand bank incomplete. These include successful transfers and can overlap "
                    "compute in the other bank; they are not missing-response or bubble cycles."),
               dict(level="info", text=f"Native command backpressure {c['native_blocked_cycles']} "
                    f"cycles; output backpressure {c['output_wait']} cycles. Per-PC pending_no_response "
                    "counts cycles with outstanding requests and no response, not per-request latency. "
                    "Per-PC credit_wait counts full receive reservations, including returned data "
                    "awaiting its paired half or core consumption. These counters overlap and cannot "
                    "be added into lost cycles or a critical path.")])
