"""Phonon-2 five-value projections using the existing OpenTPU ISA.

Offline unpack: a base-3 sign stream (five trits/byte), one high/low bit per
nonzero, then FP16 low and high magnitudes per output row. See the pinned
fermion-five-value-parakeet-v1 format in docs/phonon2_asr.md.

The device receives two disjoint signed INT8 planes, each with an FP32 row scale.
QACT quantizes the activation once; two MM operations and FP32 accumulation
produce the projection. Weight levels are preserved exactly. Activation
quantization and accumulation differ from the dense upstream FP32 reference.
This is a bounded projection slice, not an encoder or transcription engine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import language as ol
from ..compiler import QTensor, Tensor
from ..isasim import Config


def _shape(shape) -> tuple[int, int]:
    if (not isinstance(shape, (tuple, list)) or len(shape) != 2
            or any(type(n) is not int or n <= 0 for n in shape)
            or shape[0] * shape[1] > 64 * 1024 * 1024):
        raise ValueError("five-value shape must be positive [outputs, inputs], at most 64M elements")
    return tuple(shape)


@dataclass(frozen=True)
class FiveValue:
    sign: np.ndarray
    high: np.ndarray
    low_scale: np.ndarray
    high_scale: np.ndarray

    def __post_init__(self):
        shape = _shape(list(np.shape(self.sign)))
        sign = np.asarray(self.sign)
        high = np.asarray(self.high)
        low, upper = np.asarray(self.low_scale), np.asarray(self.high_scale)
        if not np.isin(sign, [-1, 0, 1]).all():
            raise ValueError("sign entries must be -1, 0 or 1")
        if high.shape != shape or high.dtype != np.bool_:
            raise ValueError("high must be a boolean matrix matching sign")
        if low.shape != (shape[0],) or upper.shape != low.shape:
            raise ValueError("one low/high scale is required per output row")
        if not np.isfinite(low).all() or not np.isfinite(upper).all() or np.any(low < 0) or np.any(upper < 0):
            raise ValueError("low/high scales must be finite and nonnegative")
        # No second rounding: inputs must be representable in the container's FP16.
        with np.errstate(over="ignore"):
            if not np.array_equal(low, low.astype(np.float16).astype(low.dtype)) or not np.array_equal(upper, upper.astype(np.float16).astype(upper.dtype)):
                raise ValueError("low/high scales must be exactly representable in FP16")
        for name, value, dtype in (("sign", sign, np.int8), ("high", high, np.bool_),
                                    ("low_scale", low, np.float32), ("high_scale", upper, np.float32)):
            copy = np.array(value, dtype=dtype, copy=True)
            copy.flags.writeable = False
            object.__setattr__(self, name, copy)

    @property
    def shape(self):
        return self.sign.shape

    def dense(self) -> np.ndarray:
        """Offline/reference expansion only, never part of an inference adapter."""
        return self.sign.astype(np.float32) * np.where(self.high, self.high_scale[:, None], self.low_scale[:, None])


def decode_five_value(blob: bytes, shape: tuple[int, int]) -> FiveValue:
    """Decode one bounded record as data, without importing the upstream loader."""
    rows, cols = _shape(shape)
    row_bytes = (cols + 4) // 5
    sign_bytes = rows * row_bytes
    if len(blob) < sign_bytes + 4 * rows:
        raise ValueError("truncated five-value record")
    packed = np.frombuffer(blob, np.uint8, count=sign_bytes).reshape(rows, row_bytes)
    if np.any(packed >= 243):
        raise ValueError("invalid base-3 byte")
    trits = np.stack([(packed // (3 ** i)) % 3 for i in range(5)], axis=-1)
    codes = trits.reshape(rows, row_bytes * 5)[:, :cols]
    nonzero = codes != 1
    nnz = int(nonzero.sum())
    bits_bytes = (nnz + 7) // 8
    scale_at = sign_bytes + bits_bytes
    if len(blob) != scale_at + 4 * rows:
        raise ValueError("five-value record size does not match signs/high bits/scales")
    high_bits = np.unpackbits(np.frombuffer(blob, np.uint8, count=bits_bytes, offset=sign_bytes), bitorder="little")[:nnz]
    high = np.zeros((rows, cols), bool)
    high[nonzero] = high_bits
    low = np.frombuffer(blob, dtype="<f2", count=rows, offset=scale_at)
    upper = np.frombuffer(blob, dtype="<f2", count=rows, offset=scale_at + 2 * rows)
    return FiveValue(codes.astype(np.int8) - 1, high, low, upper)


@ol.jit
def five_value_linear(x, low, high, out):
    q = ol.quantize(ol.load(x))
    y = ol.dot(q, low)
    ol.dot(q, high, acc=y)
    ol.store(out, y)


@dataclass(frozen=True)
class LinearSlice:
    cfg: Config
    program: list
    image: np.ndarray
    input_at: int
    output_at: int
    rows: int
    inputs: int
    padded_inputs: int
    outputs: int
    weight_bytes: int

    def budget(self) -> dict:
        return {"dram_image_bytes": int(self.image.size), "weight_bytes": self.weight_bytes,
                "instructions": len(self.program), "instruction_words": 8 * len(self.program),
                "activation_rows": self.rows, "activation_blocks": self.padded_inputs // self.cfg.D,
                "weight_format": "two_disjoint_signed_int8_planes_with_fp32_row_scales",
                "activation_format": "QACT_INT8", "hardware_validated": False}


def compile_linear_slice(weight: FiveValue, rows: int, cfg: Config) -> LinearSlice:
    """Compile one bounded projection and immutable weights; performs no model inference.

    The initial input/output byte ranges are zero. The caller transfers input PCM-
    independent activations and reads output only in a kernel qualification harness.
    Deployment must keep intermediate activations on device in the eventual graph.
    """
    if cfg.S != 1:
        raise ValueError("Phonon-2 projection slice currently supports one device slice")
    if type(rows) is not int or not 0 < rows <= cfg.act_rows:
        raise ValueError("projection rows must fit the configured ACT rows")
    outputs, inputs = weight.shape
    padded = (inputs + cfg.D - 1) // cfg.D * cfg.D
    if padded // cfg.D > min(cfg.ACT_BLOCKS, 255):
        raise ValueError("projection input exceeds QACT/ACT block capacity")
    # Predict the exact aligned allocation before making any potentially large arrays.
    sizes = [4 * rows * padded, outputs * padded, outputs * padded,
             4 * outputs, 4 * outputs, 4 * rows * outputs]
    footprint = sum((n + 127) // 128 * 128 for n in sizes)
    if footprint > cfg.DRAM_BYTES:
        raise ValueError("projection image exceeds device DRAM budget")
    image = np.zeros(footprint, np.uint8)
    cursor = 0

    def alloc(size, data=None):
        nonlocal cursor
        at = cursor
        cursor += (size + 127) // 128 * 128
        if data is not None:
            raw = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
            image[at:at + size] = raw
        return at

    x_at = alloc(sizes[0])
    low_codes = np.zeros((outputs, padded), np.int8)
    high_codes = np.zeros_like(low_codes)
    low_codes[:, :inputs] = np.where(weight.high, 0, weight.sign)
    high_codes[:, :inputs] = np.where(weight.high, weight.sign, 0)
    low_at = alloc(sizes[1], low_codes)
    high_at = alloc(sizes[2], high_codes)
    low_scale_at = alloc(sizes[3], weight.low_scale.astype("<f4"))
    high_scale_at = alloc(sizes[4], weight.high_scale.astype("<f4"))
    out_at = alloc(sizes[5])
    bound = dict(x=Tensor(x_at, (rows, padded), (padded, 1)),
                 low=QTensor(low_at, low_scale_at, (outputs, padded), padded, 4, cfg.D, rsc=True),
                 high=QTensor(high_at, high_scale_at, (outputs, padded), padded, 4, cfg.D, rsc=True),
                 out=Tensor(out_at, (rows, outputs), (outputs, 1)))
    program = five_value_linear.trace(cfg, 0, bound).finish()
    if len(program) * 8 > cfg.IMEM_WORDS:
        raise ValueError("projection program exceeds device instruction memory")
    image.flags.writeable = False
    return LinearSlice(cfg, program, image, x_at, out_at, rows, inputs, padded, outputs,
                       sizes[1] + sizes[2] + sizes[3] + sizes[4])
