"""Residual-QACT variants of the validated subsampler's three affine phases.

The original 9ad37535 source, masks, convolution phases and receipt are unchanged.
All correction arithmetic is in device instructions; constants use explicit
disjoint upload ranges, including a D-by-D identity at the image's end.
"""
from dataclasses import dataclass
import math

import numpy as np

from .. import language as ol
from ..compiler import QTensor, Tensor
from ..kernels.vit import row_groups
from .phonon2_subsampler import compile_subsampler, weight_shapes, SubsampleShape
from .phonon2_precision import linear


@ol.jit
def _pointwise(x, out, weight, bias, identity, valid: int, freq: int, channels: int, padded: int, R: int):
    b = ol.load(bias)
    for t in ol.range(valid):
        def body(f, n):
            a = ol.zeros((n, padded))
            a[:, :channels].set(ol.load(x[t, f:f + n, :]))
            y = linear(a, weight, identity, 2)
            ol.store(out[t, f:f + n, :], ol.maximum(y + b[None, :], 0.0))
        row_groups(freq, R, body)


@ol.jit
def _project(x, out, weight, bias, identity, frames: int, R: int):
    b = ol.load(bias)
    def body(t, n):
        y = linear(ol.load(x[t:t + n, :]), weight, identity, 2)
        ol.store(out[t:t + n, :], y + b[None, :])
    row_groups(frames, R, body)


@dataclass(frozen=True)
class PreciseSubsampler:
    base: object
    programs: tuple
    phase_budgets: tuple
    image: np.ndarray
    constant_ranges: tuple

    def __getattr__(self, name):
        return getattr(self.base, name)

    def budget(self):
        b = self.base.budget()
        b.update(dram_image_bytes=int(self.image.size), phases=list(self.phase_budgets),
                 max_instruction_words=max(x["instruction_words"] for x in self.phase_budgets),
                 max_tmem_live_words=max(x["tmem_peak_live_words"] for x in self.phase_budgets),
                 activation_passes=2, constant_ranges=[list(x) for x in self.constant_ranges])
        return b


def compile_precise_subsampler(weights, frames, valid, cfg, shape=SubsampleShape()):
    base = compile_subsampler(weights, frames, valid, cfg, shape)
    cursor, bound = 0, {}
    def alloc(size):
        nonlocal cursor
        at = cursor
        cursor += (size + 127) // 128 * 128
        return at
    for name, (kind, dims) in weight_shapes(shape).items():
        if kind == "int6":
            o, k = dims[0], math.prod(dims[1:])
            kp = math.ceil(k / cfg.D) * cfg.D
            bound[name] = QTensor(alloc(o * kp), alloc(4 * o), (o, kp), kp, 4, cfg.D, rsc=True)
        else:
            at = alloc(4 * math.prod(dims))
            if name.endswith("bias"):
                bound[name] = Tensor(at, dims, (1,))
    if cursor != base.input_at:
        raise ValueError("validated subsampler's immutable layout changed")
    identity_data = np.eye(cfg.D, dtype=np.int8)
    identity_at = base.image.size
    if identity_at + identity_data.nbytes > cfg.DRAM_BYTES:
        raise ValueError("identity exceeds DRAM budget")
    image = np.zeros(identity_at + identity_data.nbytes, np.uint8)
    image[:base.image.size] = base.image
    image[identity_at:] = identity_data.view(np.uint8).ravel()
    image.flags.writeable = False
    identity = QTensor(identity_at, None, (cfg.D, cfg.D), cfg.D, 0, cfg.D)
    def desc(name):
        s = base.checkpoints[name]
        return Tensor(s["offset"], s["shape"], s["strides"])
    programs, budgets = list(base.programs), list(base.phase_budgets)
    def replace(at, kernel, **kw):
        builder = kernel.trace(cfg, 0, kw)
        program = builder.finish()
        if len(program) * 8 > cfg.IMEM_WORDS:
            raise ValueError("precise subsampler phase exceeds IMEM")
        programs[at] = program
        budgets[at] = {"name": budgets[at]["name"] + "_residual", "instructions": len(program),
                       "instruction_words": len(program) * 8, "tmem_peak_live_words": builder.tmem_peak}
    common = dict(identity=identity, R=min(2, cfg.act_rows))
    for at, depth, point, stage, freq in ((4, 2, 3, 2, 32), (6, 5, 6, 3, 16)):
        replace(at, _pointwise, x=desc(f"depthwise{depth}"), out=desc(f"pointwise{point}"),
                weight=bound[f"layers.{point}.weight"], bias=bound[f"layers.{point}.bias"],
                valid=base.valid_lengths[stage], freq=freq, channels=shape.channels,
                padded=math.ceil(shape.channels / cfg.D) * cfg.D, **common)
    replace(8, _project, x=desc("flat"), out=desc("output"), weight=bound["linear.weight"],
            bias=bound["linear.bias"], frames=base.stored_lengths[3], **common)
    return PreciseSubsampler(base, tuple(programs), tuple(budgets), image,
                             ((0, base.input_at), (identity_at, image.size)))
