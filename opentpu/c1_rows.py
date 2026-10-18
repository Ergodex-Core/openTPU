"""Experimental C1 row-local image and bit-exact GEMV reference, not an ISA opcode or RTL.

Only immutable, ordinary-scale int8 M=1 matrices with plain output/ACC are eligible.
The canonical image remains authoritative for every unsupported operation. PC payloads
contain only their allocated records; physical bases are metadata, never array offsets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from . import fp32 as F, isa as I, quant as Q

FORMAT = "opentpu-c1-rows"
VERSION = 1
D = 128


@dataclass(frozen=True)
class Descriptor:
    rows: int
    blocks: int
    pc_ids: tuple[int, ...]
    bases: tuple[int, ...]
    limits: tuple[int, ...]                 # exclusive physical PC-local allocation bounds
    version: int = VERSION

    def __post_init__(self):
        for name in ("pc_ids", "bases", "limits"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        values = (self.rows, self.blocks, self.version, *self.pc_ids, *self.bases, *self.limits)
        if any(type(v) is not int for v in values):
            raise ValueError("descriptor fields must be integers")
        if self.version != VERSION or self.rows < 1 or not 1 <= self.blocks <= 255:
            raise ValueError("unsupported version or matrix shape")
        if not 1 <= self.engines <= 32 or len(set(self.pc_ids)) != self.engines:
            raise ValueError("one distinct PC per engine is required")
        if any(not 0 <= pc < 32 for pc in self.pc_ids):
            raise ValueError("PC identity outside 0..31")
        if len(self.bases) != self.engines or len(self.limits) != self.engines:
            raise ValueError("one base and limit per PC are required")
        for e, (base, limit) in enumerate(zip(self.bases, self.limits)):
            if base < 0 or base % 32 or limit < base or base + self.payload_bytes(e) > limit:
                raise ValueError("PC allocation is unaligned or too small")

    @property
    def engines(self):
        return len(self.pc_ids)

    @property
    def header(self):
        return ((4 + self.blocks + 31) // 32) * 32

    @property
    def pitch(self):
        return self.header + D * self.blocks

    def payload_bytes(self, engine):
        return max(0, (self.rows + self.engines - 1 - engine) // self.engines) * self.pitch

    def location(self, row):
        if not 0 <= row < self.rows:
            raise ValueError("matrix row out of bounds")
        e = row % self.engines
        return self.pc_ids[e], self.bases[e] + (row // self.engines) * self.pitch

    def to_dict(self):
        return {"format": FORMAT, "weight_format": "int8", "block_bytes": D, **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        obj = dict(value)
        if (obj.pop("format", None), obj.pop("weight_format", None),
                obj.pop("block_bytes", None)) != (FORMAT, "int8", D):
            raise ValueError("unsupported C1 descriptor format")
        if set(obj) != {"rows", "blocks", "pc_ids", "bases", "limits", "version"}:
            raise ValueError("unexpected or missing descriptor fields")
        return cls(**obj)


@dataclass(frozen=True)
class PackedRows:
    descriptor: Descriptor
    payloads: tuple[bytes, ...]

    def __post_init__(self):
        object.__setattr__(self, "payloads", tuple(bytes(p) for p in self.payloads))
        if len(self.payloads) != self.descriptor.engines or any(
                len(p) != self.descriptor.payload_bytes(e) for e, p in enumerate(self.payloads)):
            raise ValueError("PC payload length does not match descriptor")

    def row(self, n):
        d = self.descriptor
        d.location(n)                   # bounds check before indexing
        off = (n // d.engines) * d.pitch
        record = np.frombuffer(self.payloads[n % d.engines], np.uint8, d.pitch, off)
        return record[d.header:], record[:Q.scale_bytes(d.blocks * D, "int8", D)]

    def save(self, path):
        """Portable sidecar: versioned JSON plus immutable per-PC bytes and SHA-256 checks."""
        path = Path(path)
        manifest = {"descriptor": self.descriptor.to_dict(),
                    "sha256": [hashlib.sha256(p).hexdigest() for p in self.payloads]}
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
            for pc, payload in zip(self.descriptor.pc_ids, self.payloads):
                z.writestr(f"pc{pc}.bin", payload)
        return path

    @classmethod
    def load(cls, path):
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read("manifest.json"))
            d = Descriptor.from_dict(manifest["descriptor"])
            if len(manifest["sha256"]) != d.engines:
                raise ValueError("missing PC digest")
            payloads = []
            for e, pc in enumerate(d.pc_ids):
                name = f"pc{pc}.bin"
                if z.getinfo(name).file_size != d.payload_bytes(e):
                    raise ValueError("PC payload length does not match descriptor")
                p = z.read(name)
                if hashlib.sha256(p).hexdigest() != manifest["sha256"][e]:
                    raise ValueError("PC payload checksum mismatch")
                payloads.append(p)
        return cls(d, tuple(payloads))


def pack(weights, scales, *, engines=16, pc_ids=None, bases=None, limits=None):
    """Copy canonical quantize_mxu int8 rows without re-quantizing or altering scale bits.

    Scale padding in the canonical record is preserved too. Physical allocation bounds
    must be checked against the selected shell before these sidecars reach any device.
    """
    w, s = np.asarray(weights), np.asarray(scales)
    if w.dtype != np.uint8 or s.dtype != np.uint8 or w.ndim != 2 or s.ndim != 2:
        raise ValueError("canonical rows must be two-dimensional uint8 arrays")
    if w.shape[1] % D or not w.shape[1]:
        raise ValueError("C1 requires complete 128-byte int8 blocks")
    rows, blocks = w.shape[0], w.shape[1] // D
    if s.shape != (rows, Q.scale_bytes(w.shape[1], "int8", D)):
        raise ValueError("canonical int8 scale shape mismatch")
    ids = tuple(range(engines)) if pc_ids is None else tuple(pc_ids)
    if len(ids) != engines:
        raise ValueError("PC count must equal engines")
    d = Descriptor(rows, blocks, ids, tuple(bases) if bases is not None else (0,) * engines,
                   tuple(limits) if limits is not None else (512 << 20,) * engines)
    payloads = [bytearray(d.payload_bytes(e)) for e in range(engines)]
    for n in range(rows):
        e, off = n % engines, n // engines * d.pitch
        payloads[e][off:off + s.shape[1]] = s[n].tobytes()
        payloads[e][off + d.header:off + d.pitch] = w[n].tobytes()
    return PackedRows(d, tuple(payloads))


@dataclass(frozen=True)
class Command:
    n0: int
    n: int
    k0: int
    kb: int
    ab: int
    out: int
    flags: int = 0
    m: int = 1
    immutable: bool = True


class FallbackRequired(ValueError):
    """The original MM must execute this operation; it is not a C1 fast-path command."""


def fallback_reason(cfg, command):
    if cfg.D != D:
        return "C1 requires D=128"
    if not command.immutable:
        return "mutable matrices require the canonical MM path"
    if command.m != 1:
        return "M!=1 (including embedding gather and prefill) requires the canonical MM path"
    if command.flags & ~I.F_ACC:
        return "only ordinary-scale int8 plain output/ACC is supported"
    return None


def validate_command(d, command, cfg):
    c = command
    if any(type(v) is not int for v in (c.n0, c.n, c.k0, c.kb, c.ab, c.out, c.flags, c.m)):
        raise ValueError("command fields must be integers")
    if type(c.immutable) is not bool:
        raise ValueError("matrix immutability must be explicit")
    reason = fallback_reason(cfg, c)
    if reason:
        raise FallbackRequired(reason)
    if not (0 <= c.n0 and 1 <= c.n <= 65535 and c.n0 + c.n <= d.rows):
        raise ValueError("command row range out of bounds")
    if not (0 <= c.k0 <= 255 and 1 <= c.kb <= 255 and c.k0 + c.kb <= d.blocks):
        raise ValueError("command weight/scale block range out of bounds")
    if not (0 <= c.ab <= 255 and c.ab + c.kb <= cfg.ACT_BLOCKS):
        raise ValueError("command ACT range out of bounds")
    if not (0 <= c.out and c.out + c.n <= cfg.TMEM_WORDS):
        raise ValueError("command output range out of bounds")


def execute(packed, command, state):
    """Execute the reference against an existing isasim.Slice; no cycle claims are made.

    This independent group-at-a-time implementation reads the sidecar, not the canonical
    matrix. It shares only the established fp32 primitives with the ISA golden model.
    Validate and calculate before writing, so rejected commands do not partially commit.
    """
    d, c = packed.descriptor, command
    validate_command(d, c, state.cfg)
    sa = state.ascale[0, c.ab:c.ab + c.kb]
    if np.any(sa.view(np.uint32) != sa[:1].view(np.uint32)):
        raise ValueError("ACT blocks hold different Sa (more than one QACT)")
    act = state.act[0, c.ab * D:(c.ab + c.kb) * D].reshape(c.kb, 4, 32).astype(np.int64)
    da = state.adoff[0, c.ab:c.ab + c.kb]
    values = np.empty(c.n, np.float32)
    for i, n in enumerate(range(c.n0, c.n0 + c.n)):
        w, scales = packed.row(n)
        w = w.view(np.int8).reshape(d.blocks, 4, 32)
        total = 0
        for k in range(c.kb):
            for g in range(4):
                shift = I.MX_W[I.W8] - int(scales[4 + c.k0 + k]) - int(da[k, g])
                if shift >= 0:
                    group = int(np.dot(act[k, g], w[c.k0 + k, g].astype(np.int64)))
                    total += group << shift
        if not -(1 << 47) <= total < (1 << 47):
            raise ValueError("row accumulator exceeds signed 48-bit range")
        sw = scales[:4].copy().view("<f4")[0]
        value = F.mul(F.mul(F.i2f(np.array(total, np.int64)), sw),
                      F.mul(sa[0], np.float32(2.0 ** -I.MX_W[I.W8])))
        if c.flags & I.F_ACC:
            value = F.add(state.tget(c.out + i), value)
        values[i] = value
    state.tput(c.out + np.arange(c.n), values)
    return values
