"""C3 vocabulary engine, milestone 1: immutable layout and executable reference.

This module does not dispatch hardware. It preserves canonical int8x bytes, executes
the proposed row ownership independently of the ISA MM, and records actual work for
Lens. Timing is an explicitly serial service model, never measured RTL/HBM timing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

import numpy as np

from .. import fp32 as F, isa as I, quant as Q

VERSION = 1
D = 128
ENGINES = 4
PCS_PER_ENGINE = 4
HEADER = 32


def _integer(x, lo, hi, name):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or not lo <= x <= hi:
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    return int(x)


@dataclass(frozen=True)
class Layout:
    """Per-PC addresses; bases/limits are local byte addresses, not a global HBM map.

    Reduced vocabulary/hidden dimensions allow bounded verification workloads. The
    production eligibility check below accepts only Gemma 3 1B's actual dimensions.
    """
    vocab: int = 262144
    hidden: int = 1152
    bases: tuple[int, ...] = (0,) * 16
    limits: tuple[int, ...] = (1 << 32,) * 16
    version: int = VERSION

    def __post_init__(self):
        object.__setattr__(self, "version", _integer(self.version, VERSION, VERSION, "version"))
        object.__setattr__(self, "vocab", _integer(self.vocab, ENGINES, 1 << 24, "vocab"))
        object.__setattr__(self, "hidden", _integer(self.hidden, D, (HEADER - 4) * D, "hidden"))
        if self.hidden % D:
            raise ValueError("hidden must be a multiple of 128")
        object.__setattr__(self, "bases", tuple(self.bases))
        object.__setattr__(self, "limits", tuple(self.limits))
        if len(self.bases) != 16 or len(self.limits) != 16:
            raise ValueError("exactly 16 PC bases and limits required")
        for pc, (base, limit, size) in enumerate(zip(self.bases, self.limits, self.pc_sizes)):
            _integer(base, 0, (1 << 64) - 1, f"PC {pc} base")
            _integer(limit, 0, 1 << 64, f"PC {pc} limit")
            if base % 32 or base + size > limit:
                raise ValueError(f"PC {pc}: unaligned base or image exceeds limit")

    @property
    def kb(self):
        return self.hidden // D

    @property
    def scale_bytes(self):
        return Q.scale_bytes(self.hidden, "int8", D)

    @property
    def pitch(self):
        return HEADER + self.hidden

    @property
    def rows_per_engine(self):
        return (self.vocab + ENGINES - 1) // ENGINES

    def row_ids(self, engine):
        _integer(engine, 0, ENGINES - 1, "engine")
        lo = engine * self.rows_per_engine
        return range(lo, min(lo + self.rows_per_engine, self.vocab))

    @property
    def pc_sizes(self):
        return tuple(len(self.row_ids(e)[p::PCS_PER_ENGINE]) * self.pitch
                     for e in range(ENGINES) for p in range(PCS_PER_ENGINE))

    def locate(self, token_id):
        """(engine, PC, PC-local address), with no wrap or modulo acceptance of bad IDs."""
        token_id = _integer(token_id, 0, self.vocab - 1, "token_id")
        e, r = divmod(token_id, self.rows_per_engine)
        slot, p = divmod(r, PCS_PER_ENGINE)
        pc = PCS_PER_ENGINE * e + p
        return e, pc, self.bases[pc] + slot * self.pitch

    def bursts(self, token_id, max_beats=16):
        """Read transactions (PC, address, byte count), 32-byte beats, no 4-KiB crossing."""
        _integer(max_beats, 1, 256, "max_beats")
        _, pc, address = self.locate(token_id)
        left = self.pitch
        out = []
        while left:
            n = min(left, 32 * max_beats, 4096 - address % 4096)
            out.append((pc, address, n))
            address, left = address + n, left - n
        return tuple(out)


def _canonical(layout, weights, scales):
    w, s = np.asarray(weights), np.asarray(scales)
    if w.dtype != np.uint8 or w.shape != (layout.vocab, layout.hidden):
        raise ValueError("weights must be canonical uint8 [vocab, hidden] bytes")
    if s.dtype != np.uint8 or s.shape != (layout.vocab, layout.scale_bytes):
        raise ValueError("scales must be canonical uint8 scale-row bytes")
    if np.any(s[:, 4:4 + layout.kb] > 31):
        raise ValueError("canonical int8x offsets must be in 0..31")
    return np.ascontiguousarray(w), np.ascontiguousarray(s)


def _digest(layout, weights, scales):
    h = hashlib.sha256(f"C3/int8x/{layout.vocab}/{layout.hidden}/".encode())
    h.update(memoryview(weights))
    h.update(memoryview(scales))
    return h.hexdigest()


@dataclass(frozen=True)
class Sidecar:
    layout: Layout
    pcs: tuple[bytes, ...]
    source_sha256: str

    @classmethod
    def build(cls, layout, weights, scales, *, embedding_weights, embedding_scales):
        """Copy existing quantization. Mandatory tied inputs are checked before copying."""
        w, s = _canonical(layout, weights, scales)
        ew, es = _canonical(layout, embedding_weights, embedding_scales)
        digest = _digest(layout, w, s)
        if _digest(layout, ew, es) != digest:
            raise ValueError("tied embedding and head bytes differ")
        pcs = []
        for e in range(ENGINES):
            for p in range(PCS_PER_ENGINE):
                ids = list(layout.row_ids(e)[p::PCS_PER_ENGINE])
                data = np.zeros((len(ids), layout.pitch), np.uint8)
                data[:, :layout.scale_bytes] = s[ids]
                data[:, HEADER:] = w[ids]
                pcs.append(data.tobytes())
        return cls(layout, tuple(pcs), digest)

    def __post_init__(self):
        if len(self.pcs) != 16 or any(type(p) is not bytes for p in self.pcs):
            raise ValueError("PC images must be 16 immutable byte strings")
        object.__setattr__(self, "pcs", tuple(self.pcs))
        if tuple(map(len, self.pcs)) != self.layout.pc_sizes:
            raise ValueError("PC image lengths do not match descriptor")
        if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256):
            raise ValueError("invalid source checksum")

    def row(self, token_id):
        _, pc, address = self.layout.locate(token_id)
        start = address - self.layout.bases[pc]
        return np.frombuffer(self.pcs[pc], np.uint8, self.layout.pitch, start)

    def descriptor(self):
        return {"format": "C3-vocabulary-sidecar", "layout": asdict(self.layout),
                "source_sha256": self.source_sha256,
                "pc_sha256": [hashlib.sha256(p).hexdigest() for p in self.pcs]}

    @classmethod
    def from_descriptor(cls, descriptor, pcs):
        if descriptor.get("format") != "C3-vocabulary-sidecar":
            raise ValueError("not a C3 sidecar")
        obj = cls(Layout(**descriptor["layout"]), tuple(pcs), descriptor["source_sha256"])
        if obj.descriptor()["pc_sha256"] != descriptor["pc_sha256"]:
            raise ValueError("sidecar payload checksum mismatch")
        obj.verify_payload()
        return obj

    def verify_payload(self):
        """Recover canonical order, including scale padding, and verify its checksum."""
        w = np.empty((self.layout.vocab, self.layout.hidden), np.uint8)
        s = np.empty((self.layout.vocab, self.layout.scale_bytes), np.uint8)
        for e in range(ENGINES):
            for p in range(PCS_PER_ENGINE):
                ids = list(self.layout.row_ids(e)[p::PCS_PER_ENGINE])
                rows = np.frombuffer(self.pcs[e * 4 + p], np.uint8).reshape(-1, self.layout.pitch)
                if rows[:, self.layout.scale_bytes:HEADER].any():
                    raise ValueError("nonzero sidecar header padding")
                s[ids], w[ids] = rows[:, :self.layout.scale_bytes], rows[:, HEADER:]
        _canonical(self.layout, w, s)
        if _digest(self.layout, w, s) != self.source_sha256:
            raise ValueError("sidecar does not match canonical source checksum")

    def verify_tied(self, weights, scales, embedding_weights, embedding_scales):
        self.verify_payload()
        for w, s in ((weights, scales), (embedding_weights, embedding_scales)):
            w, s = _canonical(self.layout, w, s)
            if _digest(self.layout, w, s) != self.source_sha256:
                raise ValueError("stale sidecar or inconsistent tied embedding/head")


@dataclass(frozen=True)
class Activation:
    values: bytes                    # signed int8 elements
    scale_bits: int                   # one Sa for the whole QACT row
    offsets: bytes                    # one offset per 32-element group

    def __post_init__(self):
        if type(self.values) is not bytes or type(self.offsets) is not bytes:
            raise ValueError("activation values and offsets must be immutable bytes")
        _integer(self.scale_bits, 0, 0xFFFFFFFF, "scale_bits")
        if not self.values or len(self.values) % D or len(self.offsets) != len(self.values) // 32:
            raise ValueError("activation must contain full 128-element blocks and 32-element offsets")
        if max(self.offsets) > 31:
            raise ValueError("activation offsets must be in 0..31")

    @classmethod
    def quantize(cls, normalized):
        x = np.asarray(normalized, np.float32)
        if x.ndim != 1 or not x.size or x.size % D:
            raise ValueError("one normalized activation row, a multiple of 128 elements, required")
        q, s, d = F.quantize_mx(x[None, :], 32)
        return cls(q.tobytes(), int(F.bits(s[0])), d.tobytes())


@dataclass(frozen=True)
class Winner:
    bits: int
    token_id: int

    def __post_init__(self):
        _integer(self.bits, 0, 0xFFFFFFFF, "logit bits")
        object.__setattr__(self, "token_id", _integer(self.token_id, 0, (1 << 24) - 1, "token ID"))
        object.__setattr__(self, "bits", int(F.bits(F.ftz(F.from_bits(self.bits)))))


def better(a: Winner | None, b: Winner) -> Winner:
    if a is None:
        return b
    ka, kb = (int(F._key(F.from_bits(x.bits))) for x in (a, b))
    return b if kb > ka or (kb == ka and b.token_id < a.token_id) else a


def reduce_shards(layout, winners):
    """Completion tuples (engine, winner), in any order; reject duplicates/missing shards."""
    seen, best = set(), None
    for engine, winner in winners:
        _integer(engine, 0, ENGINES - 1, "engine")
        if engine in seen or winner.token_id not in layout.row_ids(engine):
            raise ValueError("duplicate engine or winner outside its vocabulary ownership")
        seen.add(engine)
        best = better(best, winner)
    expected = {e for e in range(ENGINES) if len(layout.row_ids(e))}
    if seen != expected:
        raise ValueError("missing shard completion")
    return best


def row_logit(row, activation, hidden):
    """Scalar row engine: exact Python integer sum, independent of ISA's vectorized MM."""
    if len(activation.values) != hidden or len(row) != HEADER + hidden:
        raise ValueError("row/activation shape mismatch")
    act = np.frombuffer(activation.values, np.int8).astype(np.int64)
    weights = row[HEADER:].view(np.int8).astype(np.int64)
    acc, dropped = 0, 0
    for group, da in enumerate(activation.offsets):
        shift = 16 - int(row[4 + group // 4]) - da
        if shift < 0:
            dropped += 1
            continue
        lo = group * 32
        acc += int(np.dot(act[lo:lo + 32], weights[lo:lo + 32])) << shift
    if not -(1 << 47) <= acc < (1 << 47):
        raise ValueError("48-bit accumulator overflow")
    sw = F.from_bits(int.from_bytes(row[:4].tobytes(), "little"))
    sa = F.from_bits(activation.scale_bits)
    y = F.mul(F.mul(F.i2f(acc), sw), F.mul(sa, np.float32(2.0 ** -16)))
    return int(F.bits(y)), dropped


@dataclass(frozen=True)
class Work:
    rows: int
    bytes: int
    macs: int
    dropped_groups: int
    transactions: int


@dataclass(frozen=True)
class Execution:
    layout: Layout
    winner: Winner
    shards: tuple[Winner | None, ...]
    logits_bits: np.ndarray
    work: tuple[Work, ...]
    reference_seconds: float


def execute(sidecar, activation, completion_order=None):
    """Diagnostic execution; validates payload and visits every row exactly once."""
    sidecar.verify_payload()
    layout = sidecar.layout
    if len(activation.values) != layout.hidden:
        raise ValueError("activation hidden dimension differs from descriptor")
    order = range(layout.vocab) if completion_order is None else tuple(completion_order)
    if completion_order is not None:
        for row in order:
            _integer(row, 0, layout.vocab - 1, "completion row")
        if len(order) != layout.vocab or len(set(order)) != layout.vocab:
            raise ValueError("completion order must visit every row exactly once")
    start = time.perf_counter()
    logits = np.empty(layout.vocab, np.uint32)
    winners = [None] * ENGINES
    counters = [[0] * 5 for _ in range(ENGINES)]
    for token_id in order:
        engine, _, _ = layout.locate(token_id)
        bits, dropped = row_logit(sidecar.row(token_id), activation, layout.hidden)
        logits[token_id] = bits
        winners[engine] = better(winners[engine], Winner(bits, token_id))
        c = counters[engine]
        for i, n in enumerate((1, layout.pitch, layout.hidden, dropped, len(layout.bursts(token_id)))):
            c[i] += n
    pairs = [(e, w) for e, w in enumerate(winners) if w is not None]
    winner = reduce_shards(layout, pairs)
    logits.flags.writeable = False
    return Execution(layout, winner, tuple(winners), logits, tuple(Work(*c) for c in counters),
                     time.perf_counter() - start)


def fallback_reasons(layout, *, model_type, tied=True, head_format="int8", greedy=True,
                     repetition_penalty=1.0, keep_logits=False, capability_version=None):
    """Guard for future dispatch. No production caller or hardware capability exists yet."""
    reasons = []
    for ok, why in ((model_type in ("gemma3", "gemma3_text"), "model type"),
                    ((layout.vocab, layout.hidden) == (262144, 1152), "model dimensions"),
                    (tied, "untied embeddings"), (head_format == "int8", "head format"),
                    (greedy, "sampling"), (repetition_penalty == 1.0, "repetition penalty"),
                    (not keep_logits, "full-logit output"),
                    (type(capability_version) is int and capability_version == VERSION,
                     "C3 capability version")):
        if not ok:
            reasons.append(why)
    return tuple(reasons)


def canonical_program(layout, weights, scales, normalized):
    """Existing ISA LD/QACT/MM/ST head workload, for an independent oracle and Lens."""
    from ..isasim import Config
    w, s = _canonical(layout, weights, scales)
    x = np.asarray(normalized, np.float32)
    if x.shape != (layout.hidden,):
        raise ValueError("normalized activation shape mismatch")
    align = lambda n: (n + D - 1) // D * D
    wa = align(x.nbytes)
    sa = align(wa + w.nbytes)
    out = align(sa + s.nbytes)
    dram = np.zeros(align(out + layout.vocab * 4), np.uint8)
    dram[:x.nbytes] = x.view(np.uint8)
    dram[wa:wa + w.nbytes] = w.reshape(-1)
    dram[sa:sa + s.nbytes] = s.reshape(-1)
    cfg = Config(S=1, D=D, MCOLS=4, ACT_BLOCKS=max(32, layout.kb), DRAM_BYTES=len(dram))
    prog = [I.ld(0, 0, layout.hidden), I.qact(0, 1, 0, layout.kb, layout.hidden)]
    for lo in range(0, layout.vocab, 8192):
        n = min(8192, layout.vocab - lo)
        prog += [I.mm(wa + lo * layout.hidden, sa + lo * layout.scale_bytes, 0, n, layout.kb,
                      layout.hidden, n, 1, 0, layout.scale_bytes), I.st(out + lo * 4, 0, n)]
    return cfg, prog + [I.halt()], dram, out


def lens_profile(run, *, mhz=200.0, read_bytes_per_cycle=128, epilogue_cycles=8):
    """Work-accounting schedule of an executed reference, serialized phases per engine.

    Each engine reads its complete sidecar, computes all rows, applies epilogues and
    reduces them; engines run concurrently. No latency, overlap, queue or HBM claim.
    """
    from ..profile import UNITS
    if not math.isfinite(mhz) or mhz <= 0:
        raise ValueError("positive finite MHz required")
    _integer(read_bytes_per_cycle, 1, 4096, "read_bytes_per_cycle")
    _integer(epilogue_cycles, 1, 1024, "epilogue_cycles")
    layout, instrs, programs, slices = run.layout, [], [], []
    broadcast = (layout.hidden + 4 + layout.hidden // 32 + read_bytes_per_cycle - 1) // read_bytes_per_cycle
    ends = []
    for e, work in enumerate(run.work):
        t, prog = 0, []
        busy = {u: 0 for u in UNITS}
        phases = [("activation broadcast", 0, broadcast, 0),
                  ("sidecar read", 0, math.ceil(work.bytes / read_bytes_per_cycle), work.bytes / D),
                  ("integer GEMV", 1, work.rows * layout.kb, 0),
                  ("fp32 epilogue", 3, work.rows * epilogue_cycles, 0),
                  ("local ARGMAX", 4, work.rows, 0)]
        for name, unit, cycles, pb in phases:
            pc = len(prog)
            detail = f"engine {e}: {work.rows} executed rows; serial service model"
            prog.append([pc, name, detail, "C3 reference event", -1, [0] * 8, [0] * 4, 0, 0])
            instrs.append([e, pc, pc, unit, name, detail, t, t, -1, t, t + cycles,
                           -1, {}, cycles, pb, 0])
            busy[UNITS[unit]] += cycles
            t += cycles
        programs.append(prog)
        slices.append({"ports": {}, "busy": busy, "lose": {}, "buckets": {}})
        ends.append(t)
    end = max(ends)
    pc = len(programs[0])
    programs[0].append([pc, "global ARGMAX", "four winner pairs", "C3 reference event", -1,
                        [0] * 8, [0] * 4, 0, 0])
    instrs.append([0, pc, pc, 4, "global ARGMAX", "four winner pairs", end, end, -1,
                   end, end + ENGINES, -1, {}, ENGINES, 0, 0])
    slices[0]["busy"][UNITS[4]] += ENGINES
    cycles = end + ENGINES
    bounds = [math.ceil(w.bytes / read_bytes_per_cycle) for w in run.work]
    assumptions = {"read_bytes_per_cycle_per_engine": read_bytes_per_cycle,
                   "integer_products_per_cycle_per_engine": D, "epilogue_cycles_per_row": epilogue_cycles,
                   "stages_overlap": False, "HBM_latency_or_contention_modeled": False}
    return {"kind": "model", "name": "C3 executed reference — serial service model, NOT RTL timing",
            "cycles": cycles, "clock_mhz": mhz,
            "config": {"S": ENGINES, "D": D, "MCOLS": 1, "LANES": 1, "CL": 1,
                       "C": 1, "ACT_BLOCKS": layout.kb, "TMEM_WORDS": 0,
                       "MEM": "hbm", "PCS": 16, "model": "C3 reference service accounting"},
            "roofline": {"bound": max(bounds), "efficiency": max(bounds) / cycles,
                         "per_slice": [{"portb": w.bytes / D, "porta": 0, "bound": b}
                                       for w, b in zip(run.work, bounds)],
                         "note": "Sidecar byte-service bound under explicit assumptions, not HBM measurement"},
            "macs": sum(w.macs for w in run.work), "bytes": sum(w.bytes for w in run.work),
            "peak_macs": D * ENGINES, "peak_bytes": read_bytes_per_cycle * ENGINES,
            "units": UNITS, "slices": slices, "instrs": instrs, "programs": programs,
            "sources": [], "mxu_gaps": {},
            "notes": [{"level": "warn", "text": "Functional reference executed every row. Displayed cycles are a "
                        "serialized service model, not RTL/HBM measurements or predicted token throughput."},
                      {"level": "info", "text": json.dumps(assumptions, sort_keys=True)},
                      {"level": "info", "text": "S denotes independent head engines here, not full TPU slices. "
                       "The generic Lens floorplan is not a physical C3 floorplan. No transformer layers included."}],
            "c3": {"work": [asdict(w) for w in run.work], "assumptions": assumptions,
                   "reference_seconds": run.reference_seconds, "winner": asdict(run.winner)}}
