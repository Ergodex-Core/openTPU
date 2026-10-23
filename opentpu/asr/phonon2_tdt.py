"""Bounded device LSTM/TDT decoder for the pinned Transformers greedy policy.

All input-dependent lookup, state/cache control, logits selection and UTF-8
piece selection execute in the ISA. Host output is scatter/gather byte I/O,
never token lookup or whitespace processing. No physical device is opened.
"""
from dataclasses import dataclass
import math

import numpy as np

from .. import language as ol
from ..compiler import DevVar, QTensor, Tensor, current
from ..kernels.lib import sigmoid
from ..kernels.vit import row_groups
from .phonon2_int6 import Int6Weight
from .phonon2_precision import linear
from .phonon2_encoder import clear

POLICY = "transformers-ea52711b-tdt-cache-eos3-metaspace-watchdog-v1"
FRAME, PENDING, INITIALIZED, STEPS, EMITTED, STATUS, TEXT_SEEN, TEXT_BYTES, ZERO_RUN = range(9)
STATE_WORDS = 16
TEXT_STRIDE = 32


@dataclass(frozen=True)
class DecoderShape:
    encoder: int = 1024
    hidden: int = 640
    vocab: int = 8193
    blank: int = 8192
    eos: int = 3

    def validate(self, cfg):
        if (cfg.S != 1 or self.encoder % cfg.D or self.hidden % cfg.D or
                not 1 <= self.hidden <= 1024 or not 4 <= self.vocab <= 8193 or
                not 0 <= self.blank < self.vocab or not 0 <= self.eos < self.vocab or self.blank == self.eos):
            raise ValueError("invalid one-slice aligned decoder geometry")


def weight_shapes(shape=DecoderShape()):
    h, result = shape.hidden, {}
    result["encoder_projector.weight"] = ("int6", (h, shape.encoder))
    result["encoder_projector.bias"] = ("array", (h,))
    result["decoder.embedding.weight"] = ("int6", (shape.vocab, h))
    for layer in range(2):
        for kind in ("ih", "hh"):
            result[f"decoder.lstm.weight_{kind}_l{layer}"] = ("int6", (4 * h, h))
            result[f"decoder.lstm.bias_{kind}_l{layer}"] = ("array", (4 * h,))
    result["decoder.decoder_projector.weight"] = ("int6", (h, h))
    result["decoder.decoder_projector.bias"] = ("array", (h,))
    result["joint.head.weight"] = ("int6", (shape.vocab + 5, h))
    result["joint.head.bias"] = ("array", (shape.vocab + 5,))
    return result


def integer_equal(x, value):
    return 1.0 - ol.minimum(ol.abs(x - value), 1.0)


def integer_less(x, value):
    return ol.minimum(ol.maximum(value - x, 0.0), 1.0)


def conditional(flag, body):
    """A device 0/1 trip-count loop, with a bounded RLD; no host decision."""
    b, r = current(), current().scratch()
    b.rld(r, flag, bound=(0, 1))
    loop = b.begin_loop(0, rcount=r)
    body()
    b.end_loop(loop)
    b.unscratch(r)


def indexed(base, byte_offset, shape, strides, bound, body):
    b, r = current(), current().scratch()
    b.rld(r, byte_offset, bound=bound)
    desc = Tensor(base + DevVar("decoder_index", r, align=4), shape, strides)
    body(desc)
    b.unscratch(r)


def affine(x, w, name, passes):
    return linear(x, w[name + ".weight"], w["identity"], passes) + ol.load(w[name + ".bias"])[None, :]


def tanh(x):
    return 2.0 * sigmoid(2.0 * x) - 1.0


@ol.jit
def reset_decoder(scratch, words: int, state, blank: int):
    # Same bounded clearing instructions as the encoder, preserving its input.
    zero = ol.zeros((512,))
    full, tail = divmod(words, 512)
    for i in ol.range(full):
        ol.store(scratch[i * 512:i * 512 + 512], zero)
    if tail:
        ol.store(scratch[full * 512:full * 512 + tail], zero[:tail])
    ol.store(state[PENDING:PENDING + 1], ol.full((1,), float(blank)))


@ol.jit
def project_encoder(w, x, out, frames: int, passes: int):
    def body(i, n):
        ol.store(out[i:i + n, :], affine(ol.load(x[i:i + n, :]), w, "encoder_projector", passes))
    row_groups(frames, 2 if passes == 2 else 4, body)


def predictor(w, state, hidden, cell, cache, shape, passes):
    h = shape.hidden
    token = ol.load(state[PENDING:PENDING + 1])
    x = ol.empty((1, h))
    indexed(w["embedding"].base, token * (4 * h), (1, h), (h, 1),
            (0, (shape.vocab - 1) * 4 * h), lambda desc: ol.load(desc, out=x))
    for layer in range(2):
        old_h = ol.load(hidden[layer:layer + 1, :])
        gates = (affine(x, w, f"decoder.lstm.ih{layer}", passes) +
                 affine(old_h, w, f"decoder.lstm.hh{layer}", passes))
        c = (sigmoid(gates[:, h:2 * h]) * ol.load(cell[layer:layer + 1, :]) +
             sigmoid(gates[:, :h]) * tanh(gates[:, 2 * h:3 * h]))
        x = sigmoid(gates[:, 3 * h:]) * tanh(c)
        ol.store(hidden[layer:layer + 1, :], x)
        ol.store(cell[layer:layer + 1, :], c)
    ol.store(cache, affine(x, w, "decoder.decoder_projector", passes))
    ol.store(state[INITIALIZED:INITIALIZED + 1], ol.full((1,), 1.0))


def token_control(state, logits, trace, text, text_lengths, table, lengths, visible,
                  shape, valid, step, max_steps):
    b, vocab = current(), shape.vocab
    token = b.argmax(logits[:, :vocab])[1:2]
    duration = b.argmax(logits[:, vocab:vocab + 5])[1:2]
    blank = integer_equal(token, float(shape.blank))
    duration = duration + blank * integer_equal(duration, 0.0)
    frame = ol.load(state[FRAME:FRAME + 1])
    ol.store(trace[step, :1], token)
    ol.store(trace[step, 1:2], duration)
    ol.store(trace[step, 2:3], frame)
    ol.store(trace[step, 3:4], 1.0 - blank)
    zero_run = (ol.load(state[ZERO_RUN:ZERO_RUN + 1]) + 1.0) * integer_equal(duration, 0.0)
    ol.store(state[ZERO_RUN:ZERO_RUN + 1], zero_run)
    ol.store(state[EMITTED:EMITTED + 1], ol.load(state[EMITTED:EMITTED + 1]) + 1.0 - blank)
    is_visible = ol.empty((1,))
    indexed(visible.base, token * 4.0, (1,), (1,), (0, (vocab - 1) * 4),
            lambda desc: ol.load(desc, out=is_visible))
    def emit_text():
        # Tokenizers Metaspace: remove replacement chars in the first retained
        # token; replace them by spaces in every later token. No CTC merging.
        following = ol.minimum(ol.load(state[TEXT_SEEN:TEXT_SEEN + 1]), 1.0)
        raw = ol.empty((TEXT_STRIDE // 4,))
        indexed(table.base, token * (2 * TEXT_STRIDE) + following * TEXT_STRIDE,
                (TEXT_STRIDE // 4,), (1,), (0, vocab * 2 * TEXT_STRIDE - TEXT_STRIDE),
                lambda desc: ol.load(desc, out=raw))
        ol.store(text[step, :], raw)  # raw UTF-8 bytes, copied without VPU arithmetic
        count = ol.empty((1,))
        indexed(lengths.base, token * 8.0 + following * 4.0, (1,), (1,), (0, vocab * 8 - 4),
                lambda desc: ol.load(desc, out=count))
        ol.store(text_lengths[step:step + 1], count)
        ol.store(state[TEXT_BYTES:TEXT_BYTES + 1], ol.load(state[TEXT_BYTES:TEXT_BYTES + 1]) + count)
        ol.store(state[TEXT_SEEN:TEXT_SEEN + 1], ol.load(state[TEXT_SEEN:TEXT_SEEN + 1]) + 1.0)
    conditional(is_visible * (1.0 - blank), emit_text)
    next_frame = frame + duration
    eos = integer_equal(token, float(shape.eos))
    finished = 1.0 - integer_less(next_frame, float(valid))
    ol.store(state[STATUS:STATUS + 1], eos * 2.0 + (1.0 - eos) * finished)
    ol.store(state[FRAME:FRAME + 1], next_frame)
    ol.store(state[PENDING:PENDING + 1], token)
    ol.store(state[STEPS:STEPS + 1], ol.load(state[STEPS:STEPS + 1]) + 1.0)


@ol.jit
def decode(w, enc, state, hidden, cell, cache, trace, text, text_lengths, table, lengths, visible,
           audit_hidden, audit_cache, frames: int, valid: int, max_steps: int, passes: int,
           shape: DecoderShape, audit: bool):
    h = shape.hidden
    for step in ol.range(max_steps):
        active = integer_equal(ol.load(state[STATUS:STATUS + 1]), 0.0)
        def body():
            needs = ol.minimum(1.0 - ol.load(state[INITIALIZED:INITIALIZED + 1]) +
                               1.0 - integer_equal(ol.load(state[PENDING:PENDING + 1]), float(shape.blank)), 1.0)
            conditional(needs, lambda: predictor(w, state, hidden, cell, cache, shape, passes))
            frame = ol.load(state[FRAME:FRAME + 1])
            encoder = ol.empty((1, h))
            indexed(enc.base, frame * (4 * h), (1, h), (h, 1), (0, (valid - 1) * h * 4),
                    lambda desc: ol.load(desc, out=encoder))
            logits = affine(ol.maximum(encoder + ol.load(cache), 0.0), w, "joint.head", passes)
            token_control(state, logits, trace, text, text_lengths, table, lengths, visible,
                          shape, valid, step, max_steps)
            if audit:
                for layer in range(2):
                    ol.store(audit_hidden[step, layer * h:(layer + 1) * h], ol.load(hidden[layer, :]))
                    ol.store(audit_hidden[step, (layer + 2) * h:(layer + 3) * h], ol.load(cell[layer, :]))
                ol.store(audit_cache[step, :], ol.load(cache)[0, :])
        conditional(active, body)
    status = ol.load(state[STATUS:STATUS + 1])
    ol.store(state[STATUS:STATUS + 1], status + integer_equal(status, 0.0) * 3.0)


@dataclass(frozen=True)
class DecoderPlan:
    cfg: object
    programs: tuple
    phase_budgets: tuple
    image: np.ndarray
    input_at: int
    checkpoints: dict
    frames: int
    valid: int
    max_steps: int
    shape: DecoderShape
    audit: bool

    def budget(self):
        return {"dram_image_bytes": int(self.image.size), "immutable_prefix_bytes": self.input_at,
                "phases": list(self.phase_budgets), "max_instruction_words": max(p["instruction_words"] for p in self.phase_budgets),
                "max_tmem_live_words": max(p["tmem_peak_live_words"] for p in self.phase_budgets),
                "frames": self.frames, "valid": self.valid, "max_steps": self.max_steps,
                "policy": POLICY, "text_format": "UTF-8 scatter/gather segments; device-selected bytes and lengths",
                "hardware_validated": False, "model_ready": False}


def compile_decoder(weights, text_table, frames, valid, cfg, shape=DecoderShape(), *, passes=2, max_steps=None, audit=False):
    shape.validate(cfg)
    if type(frames) is not int or type(valid) is not int or not 1 <= valid <= frames <= 376:
        raise ValueError("require 1 <= valid <= stored encoder frames <= 376")
    max_steps = 10 * frames - 1 if max_steps is None else max_steps
    if type(max_steps) is not int or not 1 <= max_steps <= 3760 or passes not in (1, 2):
        raise ValueError("bounded step budget and one/two activation passes required")
    expected = weight_shapes(shape)
    if set(weights) != set(expected):
        raise ValueError("decoder weight keys differ")
    for name, (kind, dims) in expected.items():
        a = weights[name]
        if kind == "int6":
            if not isinstance(a, Int6Weight) or a.q.shape != dims:
                raise ValueError("bad INT6 shape " + name)
        elif np.shape(a) != dims or not np.isfinite(a).all():
            raise ValueError("bad finite parameter " + name)
    table, byte_lengths, visible_flags = text_table
    if (table.shape != (shape.vocab, 2, TEXT_STRIDE) or table.dtype != np.uint8 or
            byte_lengths.shape != (shape.vocab, 2) or visible_flags.shape != (shape.vocab,) or
            not np.isin(visible_flags, [0, 1]).all() or not np.isfinite(byte_lengths).all() or
            np.any(byte_lengths < 0) or np.any(byte_lengths > TEXT_STRIDE) or
            np.any(byte_lengths != np.floor(byte_lengths))):
        raise ValueError("invalid bounded immutable UTF-8 table")
    chunks, cursor = [], 0
    def alloc(size, data=None):
        nonlocal cursor
        at = cursor; cursor += (size + 127) // 128 * 128
        if cursor > cfg.DRAM_BYTES:
            raise ValueError("decoder image exceeds DRAM")
        if data is not None:
            chunks.append((at, np.ascontiguousarray(data).view(np.uint8).ravel()))
        return at
    def tensor(dims, data=None):
        return Tensor(alloc(4 * math.prod(dims), None if data is None else np.asarray(data, dtype="<f4")),
                      dims, (dims[1], 1) if len(dims) == 2 else (1,))
    w = {}
    for name, (kind, dims) in expected.items():
        value = weights[name]
        if name == "decoder.embedding.weight":
            w["embedding"] = tensor(dims, value.dense())  # immutable exact weight decode, not input compute
        elif kind == "int6":
            w[name] = QTensor(alloc(value.q.nbytes, value.q), alloc(value.scale.nbytes, value.scale), dims, dims[1], 4, cfg.D, rsc=True)
        else:
            w[name] = tensor(dims, value)
    for layer in range(2):
        for kind in ("ih", "hh"):
            for suffix in ("weight", "bias"):
                w[f"decoder.lstm.{kind}{layer}.{suffix}"] = w[f"decoder.lstm.{suffix}_{kind}_l{layer}"]
    identity = np.eye(cfg.D, dtype=np.int8)
    w["identity"] = QTensor(alloc(identity.nbytes, identity), None, (cfg.D, cfg.D), cfg.D, 0, cfg.D)
    td = Tensor(alloc(table.nbytes, table), (shape.vocab * 2 * TEXT_STRIDE // 4,), (1,))
    ld, vd = tensor(byte_lengths.shape, byte_lengths), tensor(visible_flags.shape, visible_flags)
    x = tensor((frames, shape.encoder))
    reset_at = cursor
    enc = tensor((frames, shape.hidden))
    state, hidden, cell, cache = tensor((STATE_WORDS,)), tensor((2, shape.hidden)), tensor((2, shape.hidden)), tensor((1, shape.hidden))
    trace = tensor((max_steps, 4)); text = tensor((max_steps, TEXT_STRIDE // 4)); text_lengths = tensor((max_steps,))
    ah = tensor((max_steps if audit else 1, 4 * shape.hidden)); ac = tensor((max_steps if audit else 1, shape.hidden))
    scratch = Tensor(reset_at, ((cursor - reset_at) // 4,), (1,))
    programs, budgets = [], []
    def phase(name, kernel, **kwargs):
        b = kernel.trace(cfg, 0, kwargs); p = b.finish()
        if len(p) * 8 > cfg.IMEM_WORDS:
            raise ValueError(name + " exceeds IMEM")
        programs.append(p); budgets.append({"name": name, "instruction_words": len(p) * 8, "tmem_peak_live_words": b.tmem_peak})
    phase("reset_decoder", reset_decoder, scratch=scratch, words=scratch.shape[0], state=state, blank=shape.blank)
    phase("encoder_projector", project_encoder, w=w, x=x, out=enc, frames=valid, passes=passes)
    phase("bounded_device_tdt", decode, w=w, enc=enc, state=state, hidden=hidden, cell=cell, cache=cache,
          trace=trace, text=text, text_lengths=text_lengths, table=td, lengths=ld, visible=vd,
          audit_hidden=ah, audit_cache=ac, frames=frames, valid=valid, max_steps=max_steps, passes=passes, shape=shape, audit=audit)
    image = np.zeros(cursor, np.uint8)
    for at, raw in chunks:
        image[at:at + raw.size] = raw
    image.flags.writeable = False
    descs = dict(encoder_projected=enc, state=state, hidden=hidden, cell=cell, cache=cache, trace=trace,
                 text=text, text_lengths=text_lengths, audit_hidden=ah, audit_cache=ac)
    checkpoints = {n: {"offset": d.base, "shape": d.shape} for n, d in descs.items()}
    return DecoderPlan(cfg, tuple(programs), tuple(budgets), image, x.base, checkpoints, frames, valid, max_steps, shape, audit)
