"""Experimental C2 MM_ROWS reference, not an encoded ISA or RTL implementation.

The packed image is a byte-preserving snapshot of static int8 weights. Four columns
consume separate weight rows and one ACT row. Existing MM remains the executable
fallback. The reference intentionally computes from packed banks, independently of
Slice._mm, using the existing fp32 operations only for its defined epilogue.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json

import numpy as np

from . import fp32 as F, isa as I
from .isasim import Config, SimError, Slice

VERSION = 1
LAYOUT = "opentpu.c2.rows4.pc4.int8.v1"
D, COLS, PCS = 128, 4, 16


@dataclass(frozen=True)
class Matrix:
    """Canonical allocation. scale_k0 need not equal the packed block index.

    For a column part, sa may already point into its original matrix, while ssa
    either names full scale rows (scale_k0 > 0) or copied part scales (scale_k0=0).
    n0 and k_begin in a request are relative to *this* allocation.
    """
    rows: int
    kb: int
    sa: int
    ssa: int
    rs: int
    srs: int
    scale_k0: int = 0

    def validate(self, dram_bytes: int) -> None:
        if self.rows < 1 or self.kb < 1 or self.scale_k0 < 0:
            raise ValueError("C2: invalid matrix dimensions/scale origin")
        if self.sa < 0 or self.sa % D or self.rs < self.kb * D or self.rs % D:
            raise ValueError("C2: invalid canonical weight allocation")
        if self.ssa < 0 or self.ssa % 4 or self.srs % 4 or self.srs < 4 + self.scale_k0 + self.kb:
            raise ValueError("C2: invalid canonical scale allocation")
        if self.sa + (self.rows - 1) * self.rs + self.kb * D > dram_bytes:
            raise ValueError("C2: canonical weights exceed DRAM")
        if self.ssa + self.rows * self.srs > dram_bytes:
            raise ValueError("C2: canonical scales exceed DRAM")
        wend = self.sa + (self.rows - 1) * self.rs + self.kb * D
        if self.sa < self.ssa + self.rows * self.srs and self.ssa < wend:
            raise ValueError("C2: canonical weight/scale allocations overlap")


@dataclass(frozen=True)
class PackedRows:
    matrix: Matrix
    bases: tuple[int, ...]
    banks: tuple[np.ndarray, ...]  # each bank stores its allocation, not a base-sized prefix
    scales: np.ndarray           # complete original scale rows, including padding
    capacity: int
    version: int = VERSION
    layout: str = LAYOUT

    @property
    def pitch(self) -> int:
        return (self.matrix.kb + 3) // 4

    @property
    def bank_bytes(self) -> int:
        return ((self.matrix.rows + 3) // 4) * self.pitch * D

    def validate(self) -> None:
        if self.version != VERSION or self.layout != LAYOUT:
            raise ValueError("C2: unsupported packed descriptor version/layout")
        if len(self.bases) != PCS or len(self.banks) != PCS:
            raise ValueError("C2: descriptor requires sixteen PC allocations")
        self.matrix.validate(max(self.matrix.sa + self.matrix.rows * self.matrix.rs,
                                 self.matrix.ssa + self.matrix.rows * self.matrix.srs))
        for base, bank in zip(self.bases, self.banks):
            if base < 0 or base % D or base + self.bank_bytes > self.capacity:
                raise ValueError("C2: PC allocation alignment/capacity")
            if bank.dtype != np.uint8 or bank.shape != (self.bank_bytes,):
                raise ValueError("C2: truncated or invalid PC allocation")
        if self.scales.dtype != np.uint8 or self.scales.shape != (self.matrix.rows, self.matrix.srs):
            raise ValueError("C2: invalid scale snapshot")

    def address(self, row: int, block: int) -> tuple[int, int]:
        """PC and absolute byte address; padding is never a legal logical address."""
        if not 0 <= row < self.matrix.rows or not 0 <= block < self.matrix.kb:
            raise ValueError("C2: logical address outside matrix (including padding)")
        pc = 4 * (row % 4) + block % 4
        return pc, self.bases[pc] + D * ((row // 4) * self.pitch + block // 4)

    def read(self, row: int, block: int) -> np.ndarray:
        pc, addr = self.address(row, block)
        start = addr - self.bases[pc]
        return self.banks[pc][start:start + D].view(np.int8)

    def descriptor(self) -> dict:
        """Versioned review artifact; no binary opcode or device ABI is assigned."""
        header = {"version": self.version, "layout": self.layout,
                  "matrix": asdict(self.matrix), "bases": list(self.bases),
                  "limits": [b + self.bank_bytes for b in self.bases],
                  "capacity": self.capacity}
        h = hashlib.sha256(json.dumps(header, sort_keys=True).encode())
        h.update(self.scales.tobytes())
        for bank in self.banks:
            h.update(bank.tobytes())
        return {**header, "content_sha256": h.hexdigest()}


def pack(dram: np.ndarray, matrix: Matrix, *, bases=(0,) * PCS,
         capacity: int = 512 << 20, occupied=()) -> PackedRows:
    """Copy static int8 bytes and scale records into bounded, nonoverlapping PCs.

    occupied contains (pc, start, end) intervals reserved by other allocations,
    including any canonical image in the same physical PCs. Allocation is offline.
    """
    if dram.dtype != np.uint8 or dram.ndim != 1:
        raise ValueError("C2: DRAM must be a one-dimensional uint8 image")
    matrix.validate(len(dram))
    bases = tuple(bases)
    size = ((matrix.rows + 3) // 4) * ((matrix.kb + 3) // 4) * D
    if len(bases) != PCS or capacity <= 0 or any(b < 0 or b % D or b + size > capacity for b in bases):
        raise ValueError("C2: PC allocation alignment/capacity")
    for pc, start, end in occupied:
        if not 0 <= pc < PCS or not 0 <= start <= end <= capacity:
            raise ValueError("C2: invalid occupied interval")
        if start < end and bases[pc] < end and start < bases[pc] + size:
            raise ValueError("C2: PC allocations overlap")
    banks = tuple(np.zeros(size, np.uint8) for _ in range(PCS))
    scales = np.stack([dram[matrix.ssa + n * matrix.srs:matrix.ssa + (n + 1) * matrix.srs]
                       for n in range(matrix.rows)])
    result = PackedRows(matrix, bases, banks, scales, capacity)
    for n in range(matrix.rows):
        for k in range(matrix.kb):
            pc, addr = result.address(n, k)
            src = matrix.sa + n * matrix.rs + k * D
            off = addr - bases[pc]
            banks[pc][off:off + D] = dram[src:src + D]
    for bank in banks:
        bank.flags.writeable = False
    scales.flags.writeable = False
    result.validate()
    return result


@dataclass(frozen=True)
class Dispatch:
    operation: str
    reason: str


def select(sl: Slice, ins: I.Instr, packed: PackedRows | None, *, n0=0, k_begin=0,
           capability=True, purpose="linear", static=True) -> Dispatch:
    """Explicit lowering decision. Invalid versioned descriptors are errors, not MM.

    static is the caller's ownership promise: the canonical matrix has not changed
    since packing. This offline reference does not intercept arbitrary DRAM writes.
    """
    if ins.op != I.MM:
        raise ValueError("C2: selection accepts only canonical MM instructions")
    if packed is not None:
        packed.validate()
    w = ins.w
    N, KB, k0 = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, w[3] >> 24
    M, ab = (w[5] >> 16) & 0xFF, w[5] >> 24
    why = None
    if not capability:
        why = "target has no MM_ROWS capability"
    elif packed is None:
        why = "no packed descriptor"
    elif not static or purpose != "linear":
        why = "only static linear weights are eligible (no KV/gather)"
    elif sl.cfg.D != D or sl.cfg.MCOLS != COLS:
        why = "MM_ROWS requires D=128 and MCOLS=4"
    elif M != 1:
        why = "M must be one"
    elif (ins.flags >> I.WF_SHIFT) & 3 != I.W8:
        why = "only int8 is eligible"
    elif ins.flags & ~I.F_ACC:
        why = "only plain MM and ACC flags are eligible"
    elif N < 1 or KB < 1 or ab + KB > sl.cfg.ACT_BLOCKS:
        why = "invalid operation or ACT range"
    elif n0 < 0 or n0 % COLS or n0 + N > packed.matrix.rows:
        why = "unaligned or out-of-range matrix rows"
    elif k_begin < 0 or k_begin + KB > packed.matrix.kb:
        why = "out-of-range packed blocks"
    else:
        m = packed.matrix
        sa = (sl.reg(ins.ra) + w[0]) & 0xFFFFFFFF
        ssa = (sl.reg(ins.rb) + w[1]) & 0xFFFFFFFF
        if (sa, ssa, w[4], w[6], k0) != (m.sa + n0 * m.rs + k_begin * D,
                                           m.ssa + n0 * m.srs, m.rs, m.srs,
                                           m.scale_k0 + k_begin):
            why = "canonical addresses/strides/scale origin do not match descriptor"
        elif (sa + (N - 1) * m.rs + KB * D > sl.cfg.DRAM_BYTES or
              ssa + (N - 1) * m.srs + 4 + k0 + KB > sl.cfg.DRAM_BYTES):
            why = "canonical operation exceeds DRAM"
    return Dispatch("MM" if why else "MM_ROWS", why or "eligible static int8 row group")


@dataclass
class Trace:
    """Actual reference events. Slots are serialized arithmetic steps, not RTL cycles."""
    events: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)

    def add(self, op, **data):
        self.events.append({"slot": len(self.events), "op": op, **data})

    def counts(self) -> dict:
        reads = [read for e in self.events for read in e.get("reads", [])]
        pc_bytes = [0] * PCS
        for pc, _ in reads:
            pc_bytes[pc] += D
        advances = [e for e in self.events if e["op"] == "MM_ROWS.advance"]
        return {"advance_slots": len(advances), "commit_slots": sum(e["op"] == "MM_ROWS.commit" for e in self.events),
                "weight_chunks": len(reads), "weight_bytes": len(reads) * D,
                "scale_payload_bytes": sum(e.get("scale_bytes", 0) for e in self.events),
                "products": len(reads) * D, "pc_bytes": pc_bytes,
                "column_utilization": len(reads) / (COLS * len(advances)) if advances else 0,
                "fallbacks": sum(d["operation"] == "MM" for d in self.decisions)}


def execute(sl: Slice, ins: I.Instr, packed: PackedRows | None, *, n0=0, k_begin=0,
            capability=True, purpose="linear", static=True, trace: Trace | None = None) -> Dispatch:
    """Execute selected packed arithmetic or the unchanged canonical MM fallback."""
    dispatch = select(sl, ins, packed, n0=n0, k_begin=k_begin, capability=capability,
                      purpose=purpose, static=static)
    if trace is not None:
        trace.decisions.append(asdict(dispatch))
    if dispatch.operation == "MM":
        sl._mm(ins)
        if trace is not None:
            trace.add("MM.fallback", reason=dispatch.reason)
        return dispatch
    w, m = ins.w, packed.matrix
    N, KB, ab = w[3] & 0xFFFF, (w[3] >> 16) & 0xFF, w[5] >> 24
    out = (sl.reg(ins.rc) + w[2]) & 0xFFFFFFFF
    # Validate the complete destination and ACT scale range before writing anything.
    sl.tget(out + np.arange(N))
    scales = sl.ascale[0, ab:ab + KB]
    if np.any(scales.view(np.uint32) != scales[:1].view(np.uint32)):
        raise SimError("MM_ROWS: ACT blocks hold different Sa (more than one QACT)")
    sa = F.mul(scales[0], np.float32(2.0 ** -I.MX_W[I.W8]))
    for r in range(0, N, COLS):
        rows = list(range(n0 + r, n0 + min(r + COLS, N)))
        acc = np.zeros(len(rows), np.int64)
        for q in range(KB):
            block = k_begin + q
            weights = np.stack([packed.read(n, block) for n in rows]).astype(np.int64)
            act = sl.act[0, (ab + q) * D:(ab + q + 1) * D].astype(np.int64)
            c = (weights * act).reshape(len(rows), 4, D // 4).sum(axis=2)
            dw = packed.scales[rows, 4 + m.scale_k0 + block].astype(np.int64)
            da = sl.adoff[0, ab + q].astype(np.int64)
            sh = I.MX_W[I.W8] - dw[:, None] - da
            acc += np.where(sh >= 0, c << np.maximum(sh, 0), 0).sum(axis=1)
            if trace is not None:
                trace.add("MM_ROWS.advance", row=r, matrix_rows=rows, block=block,
                          valid_mask=(1 << len(rows)) - 1,
                          reads=[packed.address(n, block) for n in rows], scale_bytes=len(rows))
        if np.any(np.abs(acc) >= 1 << 47):
            raise SimError("MM_ROWS: exact accumulator exceeds ISA bound")
        sw = np.frombuffer(b"".join(packed.scales[n, :4].tobytes() for n in rows), dtype="<f4")
        y = F.mul(F.mul(F.i2f(acc), sw), sa)
        idx = out + r + np.arange(len(rows))
        if ins.flags & I.F_ACC:
            y = F.add(sl.tget(idx), y)
        sl.tput(idx, y)
        if trace is not None:
            trace.add("MM_ROWS.commit", row=r, outputs=idx.tolist(), scale_bytes=4 * len(rows),
                      accumulator=acc.tolist(), output_bits=y.view(np.uint32).tolist(),
                      acc=bool(ins.flags & I.F_ACC))
    return dispatch


def lens_data(trace: Trace, cfg: Config, name="C2 executed reference events") -> dict:
    """Lens v1 adapter; the x-axis is a serialized reference-slot model, not timing.

    Only packed events are accepted: a fallback's one functional event has no
    meaningful timing. No memory latency/refresh/queues or pipelined overlap is
    invented. The full event evidence and allocation traffic remain in the file.
    """
    from . import lens
    from .profile import UNITS
    if any(e["op"] == "MM.fallback" for e in trace.events):
        raise ValueError("C2 Lens: fallback timing is unmodeled; profile packed events separately")
    counts = trace.counts()
    instrs, programs = [], []
    for i, e in enumerate(trace.events):
        advance = e["op"] == "MM_ROWS.advance"
        detail = (f"r={e['row']} k={e['block']} mask={e['valid_mask']:04b}; {e['reads']}"
                  if advance else f"outputs={e['outputs']} ACC={e['acc']}")
        programs.append([i, e["op"], detail, "executed functional reference event", -1])
        instrs.append([0, i, i, 1, e["op"], detail, i, i, -1, i, i + 1, -1, {}, 1,
                       len(e.get("reads", [])), 0])
    slots = len(trace.events)
    bound = (counts["weight_chunks"] + COLS - 1) // COLS
    return {"kind": "isa", "name": name + " [REFERENCE SLOTS, NOT RTL]", "cycles": slots,
            "clock_mhz": 200.0, "config": lens._config(cfg, {"C": 1, "MEM": "hbm", "PCS": PCS,
                                                             "C2_COLUMNS": COLS}),
            "roofline": {"bound": bound, "efficiency": bound / slots if slots else 0,
                         "per_slice": [{"portb": counts["weight_chunks"], "porta": 0, "bound": bound}]},
            "macs": counts["products"], "bytes": counts["weight_bytes"],
            "peak_macs": D * COLS, "peak_bytes": D * COLS, "units": UNITS,
            "slices": [{"ports": {}, "busy": {u: slots if u == "MXU" else 0 for u in UNITS},
                        "lose": {}, "buckets": {}}],
            "instrs": instrs, "programs": [programs], "sources": [], "mxu_gaps": {},
            "notes": [{"level": "warn", "text": "Executed C2 functional reference. The axis counts "
                       "serialized arithmetic advances and commits, NOT RTL cycles or measured F2 "
                       "time. 200 MHz only converts these hypothetical slots. No HBM scheduling, "
                       "cache-line scale transfers, stalls, pipelining, QACT or vector work modeled."},
                      {"level": "info", "text": "Weight-only roofline; scale payload bytes are counted "
                       "separately in c2.counts. The floorplan retains the legacy schematic; use "
                       "Timeline/Instructions for per-column bank reads and valid-tail masks."}],
            "c2": {"model": "serialized-functional-events-v1", "counts": counts,
                   "decisions": trace.decisions, "events": trace.events}}
