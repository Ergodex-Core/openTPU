"""Bounded, complete Phonon-2 Conformer block on the existing ISA.

Compilation only: no host activation computation, no device access, no serving
fallback. The phases share device DRAM. A qualification harness may execute them
with Machine.load(); a future hardware driver may sequence the same phases.
This is one block, batch 1, 1..8 stored frames, not an ASR implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .. import language as ol
from ..compiler import QTensor, Tensor
from ..isasim import Config
from ..kernels.lib import sigmoid, silu
from ..kernels.vit import layernorm_, row_groups
from .phonon2_linear import FiveValue


@dataclass(frozen=True)
class BlockShape:
    hidden: int = 1024
    intermediate: int = 4096
    heads: int = 8
    kernel: int = 9

    def validate(self, cfg):
        values = (self.hidden, self.intermediate, self.heads, self.kernel)
        if any(type(n) is not int or n <= 0 for n in values):
            raise ValueError("block dimensions must be positive integers")
        if self.hidden > 1024 or self.intermediate > 4096 or self.kernel != 9:
            raise ValueError("slice supports at most H=1024/F=4096 and kernel=9")
        if self.hidden % self.heads or self.hidden % cfg.D or self.intermediate % cfg.D:
            raise ValueError("hidden/head and projection/D dimensions must divide exactly")
        if self.hidden % 2 or max(self.hidden, self.intermediate) // cfg.D > min(cfg.ACT_BLOCKS, 255):
            raise ValueError("projection exceeds ACT block capacity")


def weight_shapes(shape: BlockShape) -> dict:
    h, f = shape.hidden, shape.intermediate
    result = {}
    for ff in ("feed_forward1", "feed_forward2"):
        result[ff + ".linear1"] = ("five_value", (f, h))
        result[ff + ".linear2"] = ("five_value", (h, f))
    for p in ("q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"):
        result["self_attn." + p] = ("five_value", (h, h))
    result["conv.pointwise_conv1"] = ("five_value", (2 * h, h))
    result["conv.pointwise_conv2"] = ("five_value", (h, h))
    result["conv.depthwise_conv.weight"] = ("array", (h, 1, 9))
    for norm in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
        for suffix in ("weight", "bias"):
            result[norm + "." + suffix] = ("array", (h,))
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        result["conv.norm." + suffix] = ("array", (h,))
    for suffix in ("bias_u", "bias_v"):
        result["self_attn." + suffix] = ("array", (shape.heads, h // shape.heads))
    return result


def relative_positions(frames, hidden):
    """Immutable sinusoidal constants generated at compile time, not from audio."""
    inv = 1.0 / (np.float32(10000) ** (np.arange(0, hidden, 2, dtype=np.float32) / hidden))
    freq = np.arange(frames - 1, -frames, -1, dtype=np.float32)[:, None] * inv[None, :]
    return np.stack((np.sin(freq), np.cos(freq)), axis=-1).reshape(2 * frames - 1, hidden)


def _linear(x, w):
    q = ol.quantize(x)
    y = ol.dot(q, w[0])
    ol.dot(q, w[1], acc=y)
    return y


def _norm(x, weights, name):
    layernorm_(x, ol.load(weights[name + ".weight"]), ol.load(weights[name + ".bias"]), 1e-5)
    return x


@ol.jit
def _ff(w, x, out, frames: int, R: int, name: str):
    def body(r, n):
        residual = ol.load(x[r:r + n, :])
        a = _norm(ol.load(x[r:r + n, :]), w, "norm_" + name)
        up = _linear(a, w[name + ".linear1"])
        down = _linear(silu(up), w[name + ".linear2"])
        ol.store(out[r:r + n, :], residual + down * .5)
    row_groups(frames, R, body)


@ol.jit
def _qkv(w, x, q, k, v, frames: int, R: int):
    def body(r, n):
        a = _norm(ol.load(x[r:r + n, :]), w, "norm_self_att")
        for name, out in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
            ol.store(out[r:r + n, :], _linear(a, w["self_attn." + name]))
    row_groups(frames, R, body)


@ol.jit
def _position(w, positions, p, frames: int, R: int):
    def body(r, n):
        ol.store(p[r:r + n, :], _linear(ol.load(positions[r:r + n, :]), w["self_attn.relative_k_proj"]))
    row_groups(2 * frames - 1, R, body)


@ol.jit
def _attend_head(w, q, k, v, p, out, frames: int, valid: int, h: int, dim: int):
    """Full FP32 RDOT attention; preserves finite-min masking for padded queries.

    No QACT on scores, probabilities, K or V. A bounded sequence permits a VPU
    implementation without dynamic transposes or a new relative-attention opcode.
    """
    c0, c1 = h * dim, (h + 1) * dim
    kt = ol.load(k[:, c0:c1])
    vt = ol.load(v[:, c0:c1])
    # Transpose the short V tile using on-device column copies, not host math.
    transposed = ol.empty((dim, frames))
    for j in range(frames):
        transposed[:, j:j + 1].set(vt[j, :].column())
    bu = ol.load(w["self_attn.bias_u"][h, :])
    bv = ol.load(w["self_attn.bias_v"][h, :])
    for i in range(frames):
        qi = ol.load(q[i, c0:c1])
        ac = (kt @ (qi + bu)) * (dim ** -.5)
        # The upstream pad/view/drop/view relative shift becomes this row slice.
        pr = ol.load(p[frames - 1 - i:2 * frames - 1 - i, c0:c1])
        bd = (pr @ (qi + bv)) * (dim ** -.5)
        if i >= valid:
            bd.set(float(np.finfo(np.float32).min))
        elif valid < frames:
            bd[valid:frames].set(float(np.finfo(np.float32).min))
        scores = ac + bd
        prob = ol.exp2((scores - ol.max(scores, axis=-1)) * ol.LOG2E)
        prob = prob / ol.sum(prob, axis=-1)
        ol.store(out[i, c0:c1], transposed @ prob)


@ol.jit
def _attention_residual(w, x, attention, out, frames: int, R: int):
    def body(r, n):
        a = _linear(ol.load(attention[r:r + n, :]), w["self_attn.o_proj"])
        ol.store(out[r:r + n, :], ol.load(x[r:r + n, :]) + a)
    row_groups(frames, R, body)


@ol.jit
def _glu(w, x, glu, frames: int, valid: int, R: int, hidden: int):
    def body(r, n):
        a = _norm(ol.load(x[r:r + n, :]), w, "norm_conv")
        pw = _linear(a, w["conv.pointwise_conv1"])
        ol.store(glu[r:r + n, :], pw[:, :hidden] * sigmoid(pw[:, hidden:2 * hidden]))
    row_groups(frames, R, body)
    # Exact stage of upstream masking: after GLU, before the depthwise convolution.
    for i in range(valid, frames):
        ol.store(glu[i, :], ol.zeros((hidden,)))


@ol.jit
def _depthwise(w, glu, conv, frames: int, hidden: int):
    for i in range(frames):
        acc = ol.zeros((hidden,))
        for tap in range(9):
            j = i + tap - 4
            if 0 <= j < frames:
                acc.set(acc + ol.load(glu[j, :]) * ol.load(w["depthwise_taps"][tap, :]))
        a = (acc - ol.load(w["conv.norm.running_mean"])) * ol.rsqrt(ol.load(w["conv.norm.running_var"]) + 1e-5)
        a = a * ol.load(w["conv.norm.weight"]) + ol.load(w["conv.norm.bias"])
        ol.store(conv[i, :], silu(a))


@ol.jit
def _convolution_residual(w, x, conv, out, frames: int, R: int):
    def body(r, n):
        a = _linear(ol.load(conv[r:r + n, :]), w["conv.pointwise_conv2"])
        ol.store(out[r:r + n, :], ol.load(x[r:r + n, :]) + a)
    row_groups(frames, R, body)


@ol.jit
def _final_norm(w, x, out, frames: int, R: int):
    def body(r, n):
        ol.store(out[r:r + n, :], _norm(ol.load(x[r:r + n, :]), w, "norm_out"))
    row_groups(frames, R, body)


@dataclass(frozen=True)
class ConformerSlice:
    cfg: Config
    programs: tuple
    phase_budgets: tuple
    image: np.ndarray
    input_at: int
    output_at: int
    checkpoints: dict
    shape: BlockShape
    frames: int
    valid_frames: int
    weight_bytes: int

    def budget(self):
        return {"dram_image_bytes": int(self.image.size), "weight_bytes_aligned": self.weight_bytes,
                "phases": list(self.phase_budgets),
                "max_instruction_words": max(p["instruction_words"] for p in self.phase_budgets),
                "max_tmem_live_words": max(p["tmem_peak_live_words"] for p in self.phase_budgets),
                "max_projection_act_blocks": max(self.shape.hidden, self.shape.intermediate) // self.cfg.D,
                "activation_rows": min(4, self.cfg.act_rows),
                "stored_frames": self.frames, "valid_frames": self.valid_frames,
                "hardware_validated": False, "audio_to_transcript_validated": False}


def compile_conformer_slice(weights, frames: int, valid_frames: int, cfg: Config,
                            shape: BlockShape = BlockShape()) -> ConformerSlice:
    """Compile all block stages. Weights/constants are the only host computations.

    The static valid-frame prefix comes from file length (or a prior device stage).
    Reject holes, empty inputs, unbounded sequences, incompatible shapes and budgets.
    No inference code calls this automatically; it remains a qualification slice.
    """
    if cfg.S != 1:
        raise ValueError("one slice is required")
    if type(frames) is not int or type(valid_frames) is not int or not 1 <= valid_frames <= frames <= 8:
        raise ValueError("require 1 <= valid_frames <= stored frames <= 8")
    shape.validate(cfg)
    if cfg.act_rows < 1:
        raise ValueError("at least one ACT row is required")
    expected = weight_shapes(shape)
    if set(weights) != set(expected):
        raise ValueError("block weight keys do not match the pinned eval graph")
    for name, (kind, dims) in expected.items():
        value = weights[name]
        if kind == "five_value":
            if not isinstance(value, FiveValue) or value.shape != dims:
                raise ValueError(f"invalid five-value weight {name}; expected {dims}")
        elif np.shape(value) != dims or not np.isfinite(value).all():
            raise ValueError(f"invalid finite array {name}; expected {dims}")
    if np.any(np.asarray(weights["conv.norm.running_var"]) < 0):
        raise ValueError("BatchNorm running variance must be nonnegative")

    h, R = shape.hidden, min(4, cfg.act_rows)
    chunks, cursor = [], 0
    def alloc(data):
        nonlocal cursor
        data = np.ascontiguousarray(data)
        at = cursor
        cursor += (data.nbytes + 127) // 128 * 128
        if cursor > cfg.DRAM_BYTES:
            raise ValueError("block image exceeds device DRAM budget")
        chunks.append((at, data.view(np.uint8).reshape(-1)))
        return at
    def tensor(data):
        data = np.asarray(data, dtype="<f4")
        return Tensor(alloc(data), data.shape, tuple(s // 4 for s in np.ascontiguousarray(data).strides))
    w = {}
    for name, (kind, _) in expected.items():
        value = weights[name]
        if kind == "five_value":
            rows, cols = value.shape
            pair = []
            for high, scales in ((False, value.low_scale), (True, value.high_scale)):
                codes = np.where(value.high == high, value.sign, 0).astype(np.int8)
                pair.append(QTensor(alloc(codes), alloc(scales.astype("<f4")), value.shape, cols, 4, cfg.D, rsc=True))
            w[name] = tuple(pair)
        elif name == "conv.depthwise_conv.weight":
            w["depthwise_taps"] = tensor(np.asarray(value)[:, 0, :].T)
        else:
            w[name] = tensor(value)
    weight_bytes = cursor
    positions = tensor(relative_positions(frames, h))
    scratch = {name: tensor(np.zeros((frames, h), np.float32)) for name in
               ("input", "ff1", "q", "k", "v", "attention", "att_residual", "glu", "conv", "conv_residual", "ff2", "output")}
    p = tensor(np.zeros((2 * frames - 1, h), np.float32))
    programs, budgets = [], []
    def phase(label, kernel, **kw):
        trace = kernel.trace(cfg, 0, dict(w=w, **kw))
        program = trace.finish()
        if len(program) * 8 > cfg.IMEM_WORDS:
            raise ValueError(f"phase {label} exceeds instruction memory")
        programs.append(program)
        budgets.append({"name": label, "instructions": len(program), "instruction_words": 8 * len(program),
                        "tmem_peak_live_words": trace.tmem_peak})
    common = dict(frames=frames, R=R)
    s = scratch
    phase("ff1", _ff, x=s["input"], out=s["ff1"], name="feed_forward1", **common)
    phase("qkv", _qkv, x=s["ff1"], q=s["q"], k=s["k"], v=s["v"], **common)
    phase("relative_keys", _position, positions=positions, p=p, **common)
    for head in range(shape.heads):
        phase(f"attention_head_{head}", _attend_head, q=s["q"], k=s["k"], v=s["v"], p=p,
              out=s["attention"], frames=frames, valid=valid_frames, h=head, dim=h // shape.heads)
    phase("attention_residual", _attention_residual, x=s["ff1"], attention=s["attention"], out=s["att_residual"], **common)
    phase("glu", _glu, x=s["att_residual"], glu=s["glu"], valid=valid_frames, hidden=h, **common)
    phase("depthwise_bn_silu", _depthwise, glu=s["glu"], conv=s["conv"], frames=frames, hidden=h)
    phase("convolution_residual", _convolution_residual, x=s["att_residual"], conv=s["conv"], out=s["conv_residual"], **common)
    phase("ff2", _ff, x=s["conv_residual"], out=s["ff2"], name="feed_forward2", **common)
    phase("final_norm", _final_norm, x=s["ff2"], out=s["output"], **common)
    image = np.zeros(cursor, np.uint8)
    for at, raw in chunks:
        image[at:at + raw.size] = raw
    image.flags.writeable = False
    locations = {name: desc.base for name, desc in scratch.items()}
    return ConformerSlice(cfg, tuple(programs), tuple(budgets), image, locations["input"], locations["output"],
                          locations, shape, frames, valid_frames, weight_bytes)
