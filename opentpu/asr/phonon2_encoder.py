"""Whole-context Conformer layer plans for a complete 24-layer device encoder.

One fixed weight slot is restaged with immutable constants between layers;
dynamic activations stay in device DRAM. Phase programs fit IMEM individually.
No device access, remote runtime, model execution on host or serving backend.
"""
from dataclasses import dataclass, fields
import math

import numpy as np

from .. import language as ol
from ..compiler import Affine, QTensor, Tensor
from ..isasim import Config
from ..kernels.lib import silu, sigmoid
from ..kernels.vit import layernorm_, row_groups
from .phonon2_conformer import BlockShape, weight_shapes, relative_positions
from .phonon2_linear import FiveValue
from .phonon2_precision import linear

SHARED_PROFILE = "f2-hbm:p4rl8a64c150"
SHARED_AGFI = "agfi-0001de1dd3df8e21e"
SHARED_ARCH_HASH = "dd6b92a6"


def shared_profile_config():
    expected = dict(S=1, D=128, MCOLS=8, ACT_BLOCKS=128, ACT_ROWS=16,
                    TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=4 << 30,
                    LANES=8, PAIR=True, DSTEP=True, STREAM=True, SEQ_WC=True,
                    PORTB=4, T2=True, ROWLANES=True, ALINE=64)
    missing = set(expected) - {f.name for f in fields(Config)}
    if missing:
        raise RuntimeError("ASR_SHARED_PROFILE_RUNTIME_REQUIRED: missing Config fields " + ",".join(sorted(missing)))
    return Config(**expected)


def _norm(x, w, name):
    layernorm_(x, ol.load(w[name + ".weight"]), ol.load(w[name + ".bias"]), 1e-5)
    return x


def _linear(x, w, name, passes):
    return linear(x, w[name], w["identity"], passes)


@ol.jit
def clear(scratch, words: int):
    zero = ol.zeros((512,))
    full, tail = divmod(words, 512)
    for b in ol.range(full):
        ol.store(scratch[b * 512:b * 512 + 512], zero)
    if tail:
        ol.store(scratch[full * 512:full * 512 + tail], zero[:tail])


@ol.jit
def copy_rows(x, out, rows: int):
    for t in ol.range(rows):
        ol.store(out[t, :], ol.load(x[t, :]))


@ol.jit
def _ff(w, x, out, frames: int, R: int, name: str, passes: int):
    def body(r, n):
        residual = ol.load(x[r:r + n, :])
        a = _norm(ol.load(x[r:r + n, :]), w, "norm_" + name)
        up = _linear(a, w, name + ".linear1", passes)
        down = _linear(silu(up), w, name + ".linear2", passes)
        ol.store(out[r:r + n, :], residual + down * .5)
    row_groups(frames, R, body)


@ol.jit
def _qkv(w, x, q, k, v, frames: int, R: int, passes: int):
    def body(r, n):
        a = _norm(ol.load(x[r:r + n, :]), w, "norm_self_att")
        for name, out in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
            ol.store(out[r:r + n, :], _linear(a, w, "self_attn." + name, passes))
    row_groups(frames, R, body)


@ol.jit
def _positions(w, x, out, frames: int, R: int, passes: int):
    def body(r, n):
        ol.store(out[r:r + n, :], _linear(ol.load(x[r:r + n, :]), w, "self_attn.relative_k_proj", passes))
    row_groups(2 * frames - 1, R, body)


def _chunks(start, count, tile, body):
    full, tail = divmod(count, tile)
    if full:
        for b in ol.range(full):
            body(Affine.of(start) + b * tile, tile)
    if tail:
        body(Affine.of(start) + full * tile, tail)


@ol.jit
def _scores(w, q, k, p, scores, frames: int, valid: int, head: int, dim: int, tile: int):
    c0, c1 = head * dim, (head + 1) * dim
    bu, bv = ol.load(w["self_attn.bias_u"][head, :]), ol.load(w["self_attn.bias_v"][head, :])
    def query(i, valid_query):
        qi = ol.load(q[i, c0:c1])
        qu, qv = qi + bu, qi + bv
        def keys(j, n, masked):
            content = (ol.load(k[j:j + n, c0:c1]) @ qu) * (dim ** -.5)
            if masked:
                # Exact masked_fill result before adding the content term.
                score = content + float(np.finfo(np.float32).min)
            else:
                start = Affine.of(frames - 1) - Affine.of(i) + j
                relative = (ol.load(p[start:start + n, c0:c1]) @ qv) * (dim ** -.5)
                score = content + relative
            ol.store(scores[i, j:j + n], score)
        if valid_query:
            _chunks(0, valid, tile, lambda j, n: keys(j, n, False))
            _chunks(valid, frames - valid, tile, lambda j, n: keys(j, n, True))
        else:
            _chunks(0, frames, tile, lambda j, n: keys(j, n, True))
    for i in ol.range(valid):
        query(i, True)
    if valid < frames:
        for i in ol.range(frames - valid):
            query(valid + i, False)


@ol.jit
def _values(scores, v, out, frames: int, head: int, dim: int, tile: int):
    c0, c1 = head * dim, (head + 1) * dim
    for i in ol.range(frames):
        score = ol.load(scores[i, :])
        probability = ol.exp2((score - ol.max(score, axis=-1)) * ol.LOG2E)
        probability = probability / ol.sum(probability, axis=-1)
        ol.store(scores[i, :], probability)
        accumulator = ol.zeros((dim,))
        def part(j, n):
            value = ol.load(v[j:j + n, c0:c1])
            transpose = ol.empty((dim, n), dense=True)
            for c in range(n):
                transpose[:, c:c + 1].set(value[c, :].column())
            probability_part = ol.load(scores[i, j:j + n])
            accumulator.set(accumulator + transpose @ probability_part)
        _chunks(0, frames, tile, part)
        ol.store(out[i, c0:c1], accumulator)


@ol.jit
def _att_residual(w, x, attention, out, frames: int, R: int, passes: int):
    def body(r, n):
        a = _linear(ol.load(attention[r:r + n, :]), w, "self_attn.o_proj", passes)
        ol.store(out[r:r + n, :], ol.load(x[r:r + n, :]) + a)
    row_groups(frames, R, body)


@ol.jit
def _glu(w, x, out, frames: int, valid: int, R: int, hidden: int, passes: int):
    def body(r, n):
        a = _norm(ol.load(x[r:r + n, :]), w, "norm_conv")
        pw = _linear(a, w, "conv.pointwise_conv1", passes)
        ol.store(out[r:r + n, :], pw[:, :hidden] * sigmoid(pw[:, hidden:2 * hidden]))
    row_groups(frames, R, body)
    if valid < frames:
        zero = ol.zeros((hidden,))
        for i in ol.range(frames - valid):
            ol.store(out[valid + i, :], zero)


@ol.jit
def _pad_conv(x, halo, frames: int):
    for i in ol.range(frames):
        ol.store(halo[i + 4, :], ol.load(x[i, :]))


@ol.jit
def _depthwise(w, halo, out, frames: int, hidden: int):
    for i in ol.range(frames):
        acc = ol.zeros((hidden,))
        for tap in range(9):
            acc.set(acc + ol.load(halo[i + tap, :]) * ol.load(w["depthwise_taps"][tap, :]))
        a = (acc - ol.load(w["conv.norm.running_mean"])) * ol.rsqrt(ol.load(w["conv.norm.running_var"]) + 1e-5)
        a = a * ol.load(w["conv.norm.weight"]) + ol.load(w["conv.norm.bias"])
        ol.store(out[i, :], silu(a))


@ol.jit
def _conv_residual(w, x, conv, out, frames: int, R: int, passes: int):
    def body(r, n):
        a = _linear(ol.load(conv[r:r + n, :]), w, "conv.pointwise_conv2", passes)
        ol.store(out[r:r + n, :], ol.load(x[r:r + n, :]) + a)
    row_groups(frames, R, body)


@ol.jit
def _norm_out(w, x, out, frames: int, R: int):
    def body(r, n):
        ol.store(out[r:r + n, :], _norm(ol.load(x[r:r + n, :]), w, "norm_out"))
    row_groups(frames, R, body)


@dataclass(frozen=True)
class LayerPlan:
    cfg: Config
    programs: tuple
    phase_budgets: tuple
    image: np.ndarray
    input_at: int
    output_at: int
    frames: int
    valid_frames: int
    shape: BlockShape
    passes: int
    checkpoints: dict
    weight_bytes: int
    attention_scores_at: int

    def budget(self):
        return {"dram_image_bytes": int(self.image.size), "immutable_prefix_bytes": self.input_at,
                "weight_bytes": self.weight_bytes, "phase_count": len(self.programs),
                "max_instruction_words": max(x["instruction_words"] for x in self.phase_budgets),
                "max_tmem_live_words": max(x["tmem_peak_live_words"] for x in self.phase_budgets),
                "phases": list(self.phase_budgets), "frames": self.frames, "valid_frames": self.valid_frames,
                "activation_passes": self.passes, "attention": "whole context, FP32 tiled RDOT",
                "hardware_validated": False}


def scheduled_programs(plan):
    """Serialize actual program+WAITC images for the shared sequencer.

    The conservative schedule waits for all preceding units/VPU producers. It
    trades concurrency for a straightforward correctness baseline; the checker
    must validate it before physical acceptance. No optimistic timing claim.
    """
    from .. import waits as W
    result = []
    for program in plan.programs:
        words = np.asarray([i.encode() for i in program], dtype="<u4").reshape(-1)
        entries = np.full(len(program), W.DEFAULT, np.uint64)
        result.append((words, entries, W.loadable(words, entries, plan.cfg.D)))
    return tuple(result)


def compile_encoder_layer(weights, frames, valid_frames, cfg, shape=BlockShape(), *, passes=1, tile=32):
    if type(frames) is not int or type(valid_frames) is not int or not 1 <= valid_frames <= frames <= 376:
        raise ValueError("require 1 <= valid frames <= frames <= 376")
    if cfg.S != 1 or passes not in (1, 2) or tile not in (8, 16, 32):
        raise ValueError("require one slice, one/two activation passes, and tile 8/16/32")
    shape.validate(cfg)
    expected = weight_shapes(shape)
    if set(weights) != set(expected):
        raise ValueError("layer weight keys differ from pinned graph")
    for name, (kind, dims) in expected.items():
        a = weights[name]
        if kind == "five_value":
            if not isinstance(a, FiveValue) or a.shape != dims:
                raise ValueError("bad projection " + name)
        elif np.shape(a) != dims or not np.isfinite(a).all():
            raise ValueError("bad finite parameter " + name)
    if np.any(weights["conv.norm.running_var"] < 0):
        raise ValueError("negative BatchNorm variance")
    chunks, cursor = [], 0
    def alloc(size, data=None):
        nonlocal cursor
        at = cursor
        cursor += (size + 127) // 128 * 128
        if cursor > cfg.DRAM_BYTES:
            raise ValueError("layer image exceeds DRAM")
        if data is not None:
            chunks.append((at, np.ascontiguousarray(data).view(np.uint8).ravel()))
        return at
    def tensor(dims, data=None):
        return Tensor(alloc(4 * math.prod(dims), None if data is None else np.asarray(data, dtype="<f4")),
                      dims, (dims[1], 1) if len(dims) == 2 else (1,))
    w = {}
    for name, (kind, _) in expected.items():
        value = weights[name]
        if kind == "five_value":
            o, k = value.shape
            pair = []
            for high, scale in ((False, value.low_scale), (True, value.high_scale)):
                code = np.where(value.high == high, value.sign, 0).astype(np.int8)
                pair.append(QTensor(alloc(code.nbytes, code), alloc(scale.nbytes, scale), (o, k), k, 4, cfg.D, rsc=True))
            w[name] = tuple(pair)
        else:
            value = np.asarray(value, np.float32)
            if name == "conv.depthwise_conv.weight":
                value = np.ascontiguousarray(value[:, 0, :].T)
                name = "depthwise_taps"
            w[name] = tensor(value.shape, value)
    weight_bytes = cursor
    identity = np.eye(cfg.D, dtype=np.int8)
    w["identity"] = QTensor(alloc(identity.nbytes, identity), None, (cfg.D, cfg.D), cfg.D, 0, cfg.D)
    h = shape.hidden
    positions = tensor((2 * frames - 1, h), relative_positions(frames, h))
    s = {n: tensor((frames, h)) for n in ("input", "ff1", "q", "k", "v", "attention", "att_residual", "glu", "conv", "conv_residual", "ff2", "output")}
    p, scores, halo = tensor((2 * frames - 1, h)), tensor((frames, frames)), tensor((frames + 8, h))
    reset_at = s["input"].base + 4 * frames * h
    reset = Tensor(reset_at, ((cursor - reset_at) // 4,), (1,))
    programs, budgets = [], []
    def phase(label, kernel, **kw):
        builder = kernel.trace(cfg, 0, kw)
        program = builder.finish()
        if len(program) * 8 > cfg.IMEM_WORDS:
            raise ValueError(label + " exceeds IMEM")
        programs.append(program)
        budgets.append({"name": label, "instructions": len(program), "instruction_words": len(program) * 8,
                        "tmem_peak_live_words": builder.tmem_peak})
    R = min(2 if passes == 2 else 4, cfg.act_rows)
    common = dict(w=w, frames=frames, R=R, passes=passes)
    phase("reset_layer_scratch", clear, scratch=reset, words=reset.shape[0])
    phase("ff1", _ff, x=s["input"], out=s["ff1"], name="feed_forward1", **common)
    phase("qkv", _qkv, x=s["ff1"], q=s["q"], k=s["k"], v=s["v"], **common)
    phase("relative_keys", _positions, x=positions, out=p, **common)
    for head in range(shape.heads):
        phase(f"scores_{head}", _scores, w=w, q=s["q"], k=s["k"], p=p, scores=scores,
              frames=frames, valid=valid_frames, head=head, dim=h // shape.heads, tile=tile)
        phase(f"values_{head}", _values, scores=scores, v=s["v"], out=s["attention"],
              frames=frames, head=head, dim=h // shape.heads, tile=tile)
    phase("attention_residual", _att_residual, x=s["ff1"], attention=s["attention"], out=s["att_residual"], **common)
    phase("glu", _glu, x=s["att_residual"], out=s["glu"], hidden=h, valid=valid_frames, **common)
    phase("pad_convolution", _pad_conv, x=s["glu"], halo=halo, frames=frames)
    phase("depthwise_bn_silu", _depthwise, w=w, halo=halo, out=s["conv"], frames=frames, hidden=h)
    phase("convolution_residual", _conv_residual, x=s["att_residual"], conv=s["conv"], out=s["conv_residual"], **common)
    phase("ff2", _ff, x=s["conv_residual"], out=s["ff2"], name="feed_forward2", **common)
    phase("final_norm", _norm_out, w=w, x=s["ff2"], out=s["output"], frames=frames, R=R)
    image = np.zeros(cursor, np.uint8)
    for at, raw in chunks:
        image[at:at + raw.size] = raw
    image.flags.writeable = False
    return LayerPlan(cfg, tuple(programs), tuple(budgets), image, s["input"].base, s["output"].base,
                     frames, valid_frames, shape, passes, {name: desc.base for name, desc in s.items()}, weight_bytes, scores.base)
