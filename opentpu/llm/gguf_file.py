"""A dependency-free GGUF reader (v2/v3), with Prism ML's ternary types.

The `gguf` package rejects the PrismML llama.cpp fork's private types (PQ2_0 = 142,
PTQ1_0 = 143), which Ternary-Bonsai 2's GGUFs use. This reads the header, the metadata and
the tensor table, and memory-maps the data; tensors decode to fp32 (`dequant`) or, for the
ternary types, to their exact parts (`ternary`: int8 codes in {-1, 0, +1} [N, K] and one fp32
scale per group [N, K / g]).

Ternary packings (ggml-common.h, PrismML-Eng/llama.cpp branch prism):
  Q2_0   (42)  per 64 weights: fp16 d, 16 bytes of 2-bit codes (element j: byte j / 4, bits
               2 (j % 4)), value (q - 1) d (q = 3, +2, is unused by ternary checkpoints)
  PQ2_0  (142) the same at 128 weights a group
  TQ2_0  (35)  per 256: 64 bytes of codes then fp16 d
  PTQ1_0 (143) per 128: 24 bytes of 5 trits each (120 values), 2 bytes of 4 trits (8), fp16 d
"""
from __future__ import annotations

import mmap
import struct
from pathlib import Path

import numpy as np

F32, F16, Q4_0, Q8_0, BF16, TQ2_0, Q2_0, PQ2_0, PTQ1_0 = 0, 1, 2, 8, 30, 35, 42, 142, 143

# type: (elements a block, bytes a block)
BLOCK = {F32: (1, 4), F16: (1, 2), BF16: (1, 2), Q4_0: (32, 18), Q8_0: (32, 34),
         TQ2_0: (256, 66), Q2_0: (64, 18), PQ2_0: (128, 34), PTQ1_0: (128, 28)}
TERNARY = (Q2_0, PQ2_0, PTQ1_0, TQ2_0)
NAMES = {F32: "F32", F16: "F16", BF16: "BF16", Q4_0: "Q4_0", Q8_0: "Q8_0", TQ2_0: "TQ2_0",
         Q2_0: "Q2_0", PQ2_0: "PQ2_0", PTQ1_0: "PTQ1_0"}

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
_STR, _ARR = 8, 9


class Tensor:
    def __init__(self, f: "GGUF", name: str, shape: tuple, typ: int, off: int):
        self.f, self.name, self.type, self.off = f, name, typ, off
        self.shape = shape                      # numpy order: (rows, ..., K), K contiguous

    @property
    def nbytes(self) -> int:
        n = int(np.prod(self.shape))
        be, bb = BLOCK[self.type]
        return n // be * bb

    def raw(self) -> np.ndarray:
        return np.frombuffer(self.f.mm, np.uint8, self.nbytes, self.f.data + self.off)

    def _blocks(self, rows=None) -> tuple[np.ndarray, int]:
        """The tensor's blocks as [rows, blocks a row, bytes a block] (rows: an index list)."""
        K = self.shape[-1]
        be, bb = BLOCK[self.type]
        b = self.raw().reshape(-1, K // be, bb)
        return (b if rows is None else b[np.asarray(rows)]), K

    def ternary(self, rows=None) -> tuple[np.ndarray, np.ndarray, int]:
        """codes int8 [N, K] in {-1, 0, +1}, scales fp32 [N, K / g], g."""
        if self.type not in TERNARY:
            raise ValueError(f"{self.name}: {NAMES.get(self.type, self.type)} is not ternary")
        b, K = self._blocks(rows)
        N, nb = b.shape[:2]
        if self.type in (Q2_0, PQ2_0):
            d, qs = b[..., :2], b[..., 2:]
        elif self.type == TQ2_0:
            qs, d = b[..., :64], b[..., 64:]
        if self.type in (Q2_0, PQ2_0):
            q = (qs[..., :, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3
            q = q.reshape(N, nb, -1).astype(np.int8) - 1
        elif self.type == TQ2_0:          # ggml's TQ2_0: 32-byte chunks, 4 planes of 32
            q = (qs.reshape(N, nb, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8)
                 .reshape(1, 1, 1, 4, 1)) & 3
            q = q.reshape(N, nb, 256).astype(np.int8) - 1
        else:
            d = b[..., 26:28]
            q = _trits(b[..., :24], b[..., 24:26]).reshape(N, nb, 128)
        if np.any(q > 1):
            raise ValueError(f"{self.name}: code +2 in a ternary tensor")
        s = d.copy().view(np.float16).astype(np.float32).reshape(N, nb)
        return q.reshape(N, K), s, K // nb

    def dequant(self, rows=None) -> np.ndarray:
        if self.type in TERNARY:
            q, s, g = self.ternary(rows)
            N, K = q.shape
            return (q.reshape(N, K // g, g) * s[..., None]).reshape(N, K).astype(np.float32)
        b, K = self._blocks(rows)
        if self.type == F32:
            return b.reshape(b.shape[0], -1).view(np.float32).reshape(-1, K).copy() \
                if len(self.shape) > 1 else b.reshape(-1).view(np.float32).copy()
        if self.type in (F16, BF16):
            v = b.reshape(b.shape[0], -1).copy()
            if self.type == F16:
                v = v.view(np.float16).astype(np.float32)
            else:
                v = (v.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            return v.reshape(-1, K) if len(self.shape) > 1 else v.reshape(-1)
        if self.type == Q8_0:
            d = b[..., :2].copy().view(np.float16).astype(np.float32)
            q = b[..., 2:].view(np.int8).astype(np.float32)
            return (q * d).reshape(-1, K)
        if self.type == Q4_0:
            d = b[..., :2].copy().view(np.float16).astype(np.float32)
            qs = b[..., 2:]
            q = np.concatenate([qs & 15, qs >> 4], -1).astype(np.float32) - 8
            return (q * d).reshape(-1, K)
        raise ValueError(f"{self.name}: unsupported GGUF type {self.type}")

    def array(self) -> np.ndarray:
        """fp32 in the tensor's own shape."""
        return self.dequant().reshape(self.shape)


def _trits(qs: np.ndarray, qh: np.ndarray) -> np.ndarray:
    """PTQ1_0's trits as TQ1_0 packs them (ggml-quants.c dequantize_row_tq1_0, group 128):
    qs [.., 24] holds 5 trits a byte in 5 planes of 24 (pow3 multiply, then (x * 3) >> 8),
    qh [.., 2] 4 a byte in 4 planes of 2. Returns int8 [.., 128] in {-1, 0, +1}."""
    pow3 = np.array([1, 3, 9, 27, 81], np.uint16)
    a = ((qs[..., None, :].astype(np.uint16) * pow3[:, None]) & 0xFF)        # [.., 5, 24]
    a = ((a * 3) >> 8).reshape(*qs.shape[:-1], 120)
    h = ((qh[..., None, :].astype(np.uint16) * pow3[:4, None]) & 0xFF)       # [.., 4, 2]
    h = ((h * 3) >> 8).reshape(*qh.shape[:-1], 8)
    return np.concatenate([a, h], -1).astype(np.int8) - 1


class GGUF:
    def __init__(self, path):
        self.path = Path(path)
        fh = open(self.path, "rb")
        self.mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        fh.close()
        if self.mm[:4] != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        self.pos = 4
        ver, nt, nkv = self._u("<I"), self._u("<Q"), self._u("<Q")
        if ver not in (2, 3):
            raise ValueError(f"{path}: GGUF version {ver}")
        self.meta = {}
        for _ in range(nkv):
            k = self._str()
            self.meta[k] = self._val(self._u("<I"))
        self.align = int(self.meta.get("general.alignment", 32))
        table = []
        for _ in range(nt):
            name = self._str()
            nd = self._u("<I")
            dims = [self._u("<Q") for _ in range(nd)]
            typ, off = self._u("<I"), self._u("<Q")
            table.append((name, tuple(reversed(dims)), typ, off))
        self.data = -(-self.pos // self.align) * self.align
        self.tensors = {n: Tensor(self, n, s, t, o) for n, s, t, o in table}

    def _u(self, fmt):
        v = struct.unpack_from(fmt, self.mm, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return v

    def _str(self) -> str:
        n = self._u("<Q")
        s = bytes(self.mm[self.pos:self.pos + n]).decode("utf-8", "replace")
        self.pos += n
        return s

    def _val(self, t):
        if t in _SCALAR:
            return self._u(_SCALAR[t])
        if t == _STR:
            return self._str()
        if t == _ARR:
            et, n = self._u("<I"), self._u("<Q")
            if et in _SCALAR and et != 7:
                fmt = _SCALAR[et]
                a = np.frombuffer(self.mm, np.dtype(fmt), n, self.pos).copy()
                self.pos += n * struct.calcsize(fmt)
                return a
            return [self._val(et) for _ in range(n)]
        raise ValueError(f"GGUF value type {t}")

    def __getitem__(self, name) -> Tensor:
        return self.tensors[name]

    def __contains__(self, name) -> bool:
        return name in self.tensors
