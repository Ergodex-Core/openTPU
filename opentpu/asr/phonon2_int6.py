"""Audited INT6 row-scale record decoding and immutable weight representation.

The container is data, never executable. Packing keeps the original signed
six-bit integers and FP16 scales; no additional weight quantization is applied.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Int6Weight:
    q: np.ndarray
    scale: np.ndarray

    def __post_init__(self):
        q, scale = np.asarray(self.q), np.asarray(self.scale)
        if q.ndim != 2 or any(n <= 0 for n in q.shape) or q.size > 64 * 1024 * 1024:
            raise ValueError("INT6 weight must be a bounded positive matrix")
        if not np.issubdtype(q.dtype, np.integer) or np.any(q < -32) or np.any(q > 31):
            raise ValueError("INT6 codes must be integers in [-32,31]")
        if scale.shape != (q.shape[0],) or not np.isfinite(scale).all() or np.any(scale < 0):
            raise ValueError("INT6 row scales must be finite and nonnegative")
        with np.errstate(over="ignore"):
            if not np.array_equal(scale, scale.astype(np.float16).astype(scale.dtype)):
                raise ValueError("INT6 scales must be exactly FP16 representable")
        for name, value, dtype in (("q", q, np.int8), ("scale", scale, np.float32)):
            value = np.array(value, dtype=dtype, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def shape(self):
        return self.q.shape

    def dense(self):
        """Offline reference expansion only."""
        return self.q.astype(np.float32) * self.scale[:, None]


def decode_int6(blob, shape):
    if (not isinstance(shape, (tuple, list)) or len(shape) < 2 or
            any(type(n) is not int or n <= 0 for n in shape) or math.prod(shape) > 64 * 1024 * 1024):
        raise ValueError("invalid bounded INT6 record shape")
    total, rows = math.prod(shape), shape[0]
    body = 3 * ((total + 3) // 4)
    if len(blob) != body + 2 * rows:
        raise ValueError("INT6 record length does not match shape")
    packed = np.frombuffer(blob, np.uint8, count=body).reshape(-1, 3).astype(np.uint32)
    word = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
    unsigned = np.stack([(word >> shift) & 63 for shift in (0, 6, 12, 18)], axis=1).ravel()[:total]
    scale = np.frombuffer(blob, "<f2", count=rows, offset=body)
    return Int6Weight((unsigned.astype(np.int16) - 32).reshape(rows, -1), scale)
