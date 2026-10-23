"""Phonon-2's complete learned 8x subsampler compiled to the existing ISA.

Compilation performs immutable weight packing only. Convolutions, masking,
ReLU, layout conversion and the final projection run as device instructions.
No hardware driver or serving fallback is provided. Lengths are static metadata.
"""
from dataclasses import dataclass
import math

import numpy as np

from .. import language as ol
from ..compiler import QTensor, Tensor
from ..isasim import Config
from ..kernels.vit import row_groups
from .phonon2_int6 import Int6Weight


@dataclass(frozen=True)
class SubsampleShape:
    channels: int = 256
    hidden: int = 1024

    def validate(self):
        if (type(self.channels) is not int or type(self.hidden) is not int or
                not 1 <= self.channels <= 256 or not 1 <= self.hidden <= 1024):
            raise ValueError("require channels in [1,256] and hidden in [1,1024]")


def weight_shapes(shape=SubsampleShape()):
    c = shape.channels
    result = {"layers.0.weight": ("fp16", (c, 1, 3, 3)), "layers.0.bias": ("fp16", (c,))}
    for depth, point in ((2, 3), (5, 6)):
        result[f"layers.{depth}.weight"] = ("fp16", (c, 1, 3, 3))
        result[f"layers.{depth}.bias"] = ("fp16", (c,))
        result[f"layers.{point}.weight"] = ("int6", (c, c, 1, 1))
        result[f"layers.{point}.bias"] = ("fp16", (c,))
    result["linear.weight"] = ("int6", (shape.hidden, c * 16))
    result["linear.bias"] = ("fp16", (shape.hidden,))
    return result


def lengths(frames, valid):
    if type(frames) is not int or type(valid) is not int or not 1 <= valid <= frames <= 3001:
        raise ValueError("require 1 <= valid frames <= stored frames <= 3001")
    stored, masked = [frames], [valid]
    for _ in range(3):
        stored.append((stored[-1] + 1) // 2)
        masked.append((masked[-1] + 1) // 2)
    return tuple(stored), tuple(masked)


@ol.jit
def _clear(scratch, words: int):
    """Reset every halo, masked cell, tail and output on each invocation."""
    full, tail = divmod(words, 512)
    zero = ol.zeros((512,))
    for block in ol.range(full):
        ol.store(scratch[block * 512:block * 512 + 512], zero)
    if tail:
        ol.store(scratch[full * 512:full * 512 + tail], zero[:tail])


@ol.jit
def _pad_input(x, padded, frames: int):
    for t in ol.range(frames):
        ol.store(padded[t + 1, 1:129, 0], ol.load(x[t, :]))


@ol.jit
def _first_conv(x, out, weight, bias, valid: int):
    w, b = ol.load(weight), ol.load(bias)
    for t in ol.range(valid):
        for f in ol.range(64):
            patch = ol.empty((9,))
            for dt in range(3):
                for df in range(3):
                    at = 3 * dt + df
                    patch[at:at + 1].set(ol.load(x[2 * t + dt, 2 * f + df, :]))
            ol.store(out[t + 1, f + 1, :], ol.maximum((w @ patch) + b, 0.0))


@ol.jit
def _depthwise(x, out, taps, bias, valid: int, freq: int, channels: int):
    w, b = ol.load(taps), ol.load(bias)
    for t in ol.range(valid):
        for f in ol.range(freq):
            acc = ol.zeros((channels,))
            for dt in range(3):
                for df in range(3):
                    acc.set(acc + ol.load(x[2 * t + dt, 2 * f + df, :]) * w[3 * dt + df, :])
            ol.store(out[t, f, :], acc + b)


@ol.jit
def _pointwise(x, out, weight, bias, valid: int, freq: int, channels: int, padded: int, R: int):
    b = ol.load(bias)
    for t in ol.range(valid):
        def body(f, n):
            a = ol.zeros((n, padded))
            a[:, :channels].set(ol.load(x[t, f:f + n, :]))
            y = ol.dot(ol.quantize(a), weight)
            ol.store(out[t + 1, f + 1:f + n + 1, :], ol.maximum(y + b[None, :], 0.0))
        row_groups(freq, R, body)


@ol.jit
def _flatten(x, out, frames: int, channels: int):
    for t in ol.range(frames):
        source = ol.load(x[t + 1, 1:17, :])  # [frequency, channel]
        target = ol.empty((channels, 16), dense=True)
        for f in range(16):
            target[:, f:f + 1].set(source[f, :].column())
        # Exact upstream transpose(N,C,T,F)->(N,T,C,F)->flatten(C,F).
        ol.store(out[t, :channels * 16], target.reshape(1, channels * 16)[0, :])


@ol.jit
def _project(x, out, weight, bias, frames: int, R: int):
    b = ol.load(bias)
    def body(t, n):
        y = ol.dot(ol.quantize(ol.load(x[t:t + n, :])), weight)
        # Upstream applies no time mask after its final Linear: retain its bias
        # even on a padded encoder frame whose convolutional input is all zero.
        ol.store(out[t:t + n, :], y + b[None, :])
    row_groups(frames, R, body)


@dataclass(frozen=True)
class SubsamplerSlice:
    cfg: Config
    programs: tuple
    phase_budgets: tuple
    image: np.ndarray
    input_at: int
    output_at: int
    checkpoints: dict
    stored_lengths: tuple
    valid_lengths: tuple
    shape: SubsampleShape
    weight_bytes: int

    def budget(self):
        return {"dram_image_bytes": int(self.image.size), "weight_bytes_aligned": self.weight_bytes,
                "phases": list(self.phase_budgets),
                "max_instruction_words": max(x["instruction_words"] for x in self.phase_budgets),
                "max_tmem_live_words": max(x["tmem_peak_live_words"] for x in self.phase_budgets),
                "stored_lengths": list(self.stored_lengths), "valid_lengths": list(self.valid_lengths),
                "max_act_blocks": math.ceil(self.shape.channels * 16 / self.cfg.D),
                "hardware_validated": False, "audio_to_transcript_validated": False}


def compile_subsampler(weights, frames, valid_frames, cfg: Config, shape=SubsampleShape()):
    """Compile a batch-one subsampler; preserves the official masks/layout/bias.

    The first input is FP32 log-mel from a future device frontend. Offline test
    harnesses may supply golden features, but serving never calls a CPU frontend.
    Mask-after-convolution is fused by computing valid time rows and resetting
    invalid outputs. No pointwise or final projection runs on the host.
    """
    shape.validate()
    stored, valid = lengths(frames, valid_frames)
    if cfg.S != 1 or cfg.D <= 0 or cfg.act_rows < 1:
        raise ValueError("require one device slice, positive D and ACT rows")
    padded_c = math.ceil(shape.channels / cfg.D) * cfg.D
    padded_flat = math.ceil(shape.channels * 16 / cfg.D) * cfg.D
    if padded_flat // cfg.D > min(cfg.ACT_BLOCKS, 255):
        raise ValueError("projection exceeds ACT block capacity")
    expected = weight_shapes(shape)
    if set(weights) != set(expected):
        raise ValueError("subsampler weight keys differ from pinned graph")
    for name, (kind, dims) in expected.items():
        value = weights[name]
        if kind == "int6":
            if not isinstance(value, Int6Weight) or value.shape != (dims[0], math.prod(dims[1:])):
                raise ValueError(f"invalid INT6 weight {name}")
        elif np.shape(value) != dims or not np.isfinite(value).all():
            raise ValueError(f"invalid finite FP16 array {name}")
        else:
            a = np.asarray(value)
            with np.errstate(over="ignore"):
                if not np.array_equal(a, a.astype(np.float16).astype(a.dtype)):
                    raise ValueError(f"FP16 array {name} would be requantized")

    chunks, cursor = [], 0
    def alloc(size, data=None):
        nonlocal cursor
        at = cursor
        cursor += (size + 127) // 128 * 128
        if cursor > cfg.DRAM_BYTES:
            raise ValueError("subsampler image exceeds DRAM budget")
        if data is not None:
            chunks.append((at, np.ascontiguousarray(data).view(np.uint8).reshape(-1)))
        return at
    def tensor(dims, data=None):
        stride, strides = 1, []
        for n in dims[::-1]:
            strides.insert(0, stride)
            stride *= n
        return Tensor(alloc(4 * stride, None if data is None else np.asarray(data, dtype="<f4")), tuple(dims), tuple(strides))
    def array(a):
        a = np.asarray(a, dtype="<f4")
        return tensor(a.shape, a)
    def int6(weight):
        o, k = weight.shape
        kp = math.ceil(k / cfg.D) * cfg.D
        if kp > k:
            codes = np.zeros((o, kp), np.int8)
            codes[:, :k] = weight.q
        else:
            codes = weight.q
        return QTensor(alloc(codes.nbytes, codes), alloc(4 * o, weight.scale.astype("<f4")),
                       (o, kp), kp, 4, cfg.D, rsc=True)
    w = {}
    for name, (kind, _) in expected.items():
        value = weights[name]
        if kind == "int6":
            w[name] = int6(value)
        elif name in ("layers.2.weight", "layers.5.weight"):
            w[name] = array(np.asarray(value)[:, 0].reshape(shape.channels, 9).T)
        elif name == "layers.0.weight":
            w[name] = array(np.asarray(value).reshape(shape.channels, 9))
        else:
            w[name] = array(value)
    weight_bytes = cursor
    x = tensor((frames, 128))
    scratch_start = cursor
    padded = [tensor((stored[0] + 2, 130, 1))]
    for stage, freq in enumerate((64, 32, 16), 1):
        padded.append(tensor((stored[stage] + 2, freq + 2, shape.channels)))
    dw1 = tensor((stored[2], 32, shape.channels))
    dw2 = tensor((stored[3], 16, shape.channels))
    flat = tensor((stored[3], padded_flat))
    out = tensor((stored[3], shape.hidden))
    scratch = Tensor(scratch_start, ((cursor - scratch_start) // 4,), (1,))
    programs, budgets = [], []
    def phase(label, kernel, **kw):
        builder = kernel.trace(cfg, 0, kw)
        program = builder.finish()
        if len(program) * 8 > cfg.IMEM_WORDS:
            raise ValueError(f"phase {label} exceeds instruction memory")
        programs.append(program)
        budgets.append({"name": label, "instructions": len(program), "instruction_words": len(program) * 8,
                        "tmem_peak_live_words": builder.tmem_peak})
    phase("reset_scratch", _clear, scratch=scratch, words=scratch.shape[0])
    phase("pad_input", _pad_input, x=x, padded=padded[0], frames=frames)
    phase("conv0_mask_relu", _first_conv, x=padded[0], out=padded[1], weight=w["layers.0.weight"], bias=w["layers.0.bias"], valid=valid[1])
    for stage, depth, point, freq, dw in ((2, 2, 3, 32, dw1), (3, 5, 6, 16, dw2)):
        phase(f"depthwise{depth}_mask", _depthwise, x=padded[stage - 1], out=dw,
              taps=w[f"layers.{depth}.weight"], bias=w[f"layers.{depth}.bias"], valid=valid[stage], freq=freq, channels=shape.channels)
        phase(f"pointwise{point}_mask_relu", _pointwise, x=dw, out=padded[stage],
              weight=w[f"layers.{point}.weight"], bias=w[f"layers.{point}.bias"], valid=valid[stage], freq=freq,
              channels=shape.channels, padded=padded_c, R=min(4, cfg.act_rows))
    phase("flatten_channel_then_frequency", _flatten, x=padded[3], out=flat, frames=stored[3], channels=shape.channels)
    phase("project_with_unmasked_bias", _project, x=flat, out=out, weight=w["linear.weight"], bias=w["linear.bias"], frames=stored[3], R=min(4, cfg.act_rows))
    image = np.zeros(cursor, np.uint8)
    for at, raw in chunks:
        image[at:at + raw.size] = raw
    image.flags.writeable = False
    def spec(desc):
        return {"offset": desc.base, "shape": desc.shape, "strides": desc.strides}
    checkpoints = {"input": spec(x), "depthwise2": spec(dw1), "depthwise5": spec(dw2), "flat": spec(flat), "output": spec(out)}
    for name, stage, freq in (("conv0", 1, 64), ("pointwise3", 2, 32), ("pointwise6", 3, 16)):
        checkpoints[name] = spec(padded[stage][1:stored[stage] + 1, 1:freq + 1, :])
    return SubsamplerSlice(cfg, tuple(programs), tuple(budgets), image, x.base, out.base,
                           checkpoints, stored, valid, shape, weight_bytes)
