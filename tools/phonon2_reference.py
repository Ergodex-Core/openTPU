"""Offline CPU qualification references only; never imported by the ASR adapter.

These independent NumPy equations reproduce the pinned eval graph, not the
upstream executable. No remote imports, inference backend or transcription path.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from opentpu.asr.phonon2 import CONTAINER_SHA256
from opentpu.asr.phonon2_conformer import BlockShape, weight_shapes
from opentpu.asr.phonon2_linear import decode_five_value


def read_block(path, layer=0):
    """Hash-check the existing local container; decode only one known block as data."""
    if type(layer) is not int or not 0 <= layer < 24:
        raise ValueError("layer must be in [0,24)")
    path = Path(path)
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != CONTAINER_SHA256:
            raise ValueError("container SHA256 differs from the pinned model")
        f.seek(0)
        n = int.from_bytes(f.read(8), "little")
        if not 0 < n <= 1 << 20:
            raise ValueError("invalid bounded header length")
        header = json.loads(f.read(n))
        if header["format"] != "fermion-five-value-parakeet-v1":
            raise ValueError("unsupported container format")
        prefix = f"encoder.layers.{layer}."
        expected, weights, hashes = weight_shapes(BlockShape()), {}, {}
        for entry in header["index"]:
            name, size, kind = entry["n"], entry["b"], entry["k"]
            if type(size) is not int or not 0 <= size <= 64 << 20:
                raise ValueError("invalid bounded record size")
            suffix = name.removeprefix(prefix) if name.startswith(prefix) else None
            if suffix not in expected:
                f.seek(size, 1)
                continue
            if suffix in weights or tuple(entry["shape"]) != expected[suffix][1]:
                raise ValueError("duplicate or mismatched block tensor")
            blob = f.read(size)
            if len(blob) != size:
                raise ValueError("truncated record")
            hashes[name] = hashlib.sha256(blob).hexdigest()
            dims = tuple(entry["shape"])
            if expected[suffix][0] == "five_value" and kind == "five_value":
                weights[suffix] = decode_five_value(blob, dims)
            elif suffix == "conv.depthwise_conv.weight" and kind == "int6":
                total, rows = math.prod(dims), dims[0]
                body = 3 * ((total + 3) // 4)
                if size != body + 2 * rows:
                    raise ValueError("invalid INT6 record size")
                b = np.frombuffer(blob[:body], np.uint8).reshape(-1, 3).astype(np.uint32)
                packed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
                q = np.stack([(packed >> shift) & 63 for shift in (0, 6, 12, 18)], axis=1).ravel()[:total]
                q = q.astype(np.int32) - 32
                scale = np.frombuffer(blob[body:], "<f2").astype(np.float32)
                weights[suffix] = (q.reshape(rows, -1) * scale[:, None]).astype(np.float32).reshape(dims)
            elif expected[suffix][0] == "array" and kind == "fp16" and size == 2 * math.prod(dims):
                weights[suffix] = np.frombuffer(blob, "<f2").astype(np.float32).reshape(dims)
            else:
                raise ValueError("unexpected record kind")
        if f.tell() != path.stat().st_size or set(weights) != set(expected):
            raise ValueError("invalid container extent or missing block tensor")
    return weights, hashes


def conformer_reference(weights, x, valid, shape=BlockShape(), *, quantize_D=None):
    """Dense CPU eval, optionally with the ISA activation quantization at linears.

    Attention uses the literal upstream pad/reshape/drop/reshape shift, independent
    of the device kernel's selected row slices. Returns per-stage audit snapshots.
    """
    w, x = weights, np.asarray(x, np.float32).copy()
    t, h = x.shape
    snapshots = {"input": x.copy()}
    def linear(a, name):
        if quantize_D is not None:
            from opentpu import fp32 as F
            q, scale, offsets = F.quantize_mx(a, quantize_D // 4)
            a = (q.astype(np.float32).reshape(len(a), -1, quantize_D // 4) *
                 np.ldexp(scale[:, None], -offsets.astype(int))[:, :, None]).reshape(a.shape)
        return (a @ w[name].dense().T).astype(np.float32)
    def norm(a, name):
        centered = a - a.mean(-1, keepdims=True)
        return centered / np.sqrt((centered ** 2).mean(-1, keepdims=True) + np.float32(1e-5)) * w[name + ".weight"] + w[name + ".bias"]
    def sig(a):
        with np.errstate(over="ignore"):
            return 1 / (1 + np.exp(-a))
    def ff(a, name):
        z = linear(norm(a, "norm_" + name), name + ".linear1")
        return a + .5 * linear(z * sig(z), name + ".linear2")
    x = ff(x, "feed_forward1")
    snapshots["ff1"] = x.copy()
    z = norm(x, "norm_self_att")
    q, k, v = [linear(z, "self_attn." + name + "_proj") for name in ("q", "k", "v")]
    snapshots.update(q=q.copy(), k=k.copy(), v=v.copy())
    # Explicit sinusoidal construction, separate from compiler's helper.
    inv = np.power(np.float32(10000), -np.arange(0, h, 2, dtype=np.float32) / h)
    theta = np.arange(t - 1, -t, -1, dtype=np.float32)[:, None] * inv
    pos = np.empty((2 * t - 1, h), np.float32)
    pos[:, ::2], pos[:, 1::2] = np.sin(theta), np.cos(theta)
    p = linear(pos, "self_attn.relative_k_proj")
    attention = np.empty_like(x)
    dim = h // shape.heads
    mask = (np.arange(t)[:, None] < valid) & (np.arange(t)[None, :] < valid)
    for head in range(shape.heads):
        part = slice(head * dim, (head + 1) * dim)
        ac = ((q[:, part] + w["self_attn.bias_u"][head]) @ k[:, part].T) * np.float32(dim ** -.5)
        bd = (q[:, part] + w["self_attn.bias_v"][head]) @ p[:, part].T
        bd = np.pad(bd, ((0, 0), (1, 0))).reshape(-1, t)[1:].reshape(t, 2 * t - 1)[:, :t]
        bd = bd * np.float32(dim ** -.5)
        score = ac + np.where(mask, bd, np.finfo(np.float32).min)
        probability = np.exp(score - score.max(-1, keepdims=True))
        probability /= probability.sum(-1, keepdims=True)
        attention[:, part] = probability @ v[:, part]
    snapshots["attention"] = attention.copy()
    x = x + linear(attention, "self_attn.o_proj")
    snapshots["att_residual"] = x.copy()
    pw = linear(norm(x, "norm_conv"), "conv.pointwise_conv1")
    glu = pw[:, :h] * sig(pw[:, h:])
    glu[valid:] = 0
    snapshots["glu"] = glu.copy()
    padded = np.pad(glu, ((4, 4), (0, 0)))
    conv = sum(padded[tap:tap + t] * w["conv.depthwise_conv.weight"][:, 0, tap] for tap in range(9))
    conv = (conv - w["conv.norm.running_mean"]) / np.sqrt(w["conv.norm.running_var"] + np.float32(1e-5))
    conv = conv * w["conv.norm.weight"] + w["conv.norm.bias"]
    conv = conv * sig(conv)
    snapshots["conv"] = conv.copy()
    x = x + linear(conv, "conv.pointwise_conv2")
    snapshots["conv_residual"] = x.copy()
    x = ff(x, "feed_forward2")
    snapshots["ff2"] = x.copy()
    snapshots["output"] = norm(x, "norm_out")
    return snapshots


def slaney_mel():
    """128 Slaney-normalized triangular filters; same float32 storage as librosa."""
    def hz_to_mel(hz):
        return 15 + np.log(hz / 1000) / (np.log(6.4) / 27) if hz >= 1000 else hz / (200 / 3)
    m = np.linspace(0, hz_to_mel(8000), 130)
    hz = (200 / 3) * m
    log = m >= 15
    hz[log] = 1000 * np.exp((np.log(6.4) / 27) * (m[log] - 15))
    freq = np.fft.rfftfreq(512, 1 / 16000)
    bank = np.zeros((128, 257), np.float32)
    for i in range(128):
        bank[i] = np.maximum(0, np.minimum((freq - hz[i]) / (hz[i + 1] - hz[i]),
                                          (hz[i + 2] - freq) / (hz[i + 2] - hz[i + 1])))
    bank *= (2 / (hz[2:130] - hz[:128]))[:, None]
    return bank


def frontend_reference(pcm_s16le: bytes, *, dither=0.0, sample_rate=16000, channels=1):
    """Audited, nondithered CPU golden-vector generator, not production preprocessing.

    Strict mono 16kHz PCM contract. Reject fewer than two valid feature frames to
    avoid the upstream unbiased-variance division by zero; cap at 30 seconds.
    Uses NumPy FFT, so values are tolerance references, not Torch bitwise claims.
    """
    if sample_rate != 16000 or channels != 1 or dither != 0.0:
        raise ValueError("qualification policy is 16kHz mono, dither=0 only")
    if not isinstance(pcm_s16le, bytes) or len(pcm_s16le) % 2 or not 320 <= len(pcm_s16le) // 2 <= 480000:
        raise ValueError("require aligned PCM16 with 320..480000 samples (20ms..30s)")
    x = np.frombuffer(pcm_s16le, "<i2").astype(np.float32) / np.float32(32768)
    valid = len(x) // 160
    pre = x.copy()
    pre[1:] = x[1:] - np.float32(.97) * x[:-1]
    window = np.pad(np.hanning(400).astype(np.float32), (56, 56))
    padded = np.pad(pre, (256, 256))
    frames = np.lib.stride_tricks.sliding_window_view(padded, 512)[::160]
    spectrum = np.fft.rfft(frames * window, axis=-1).astype(np.complex64)
    magnitude = np.sqrt(spectrum.real ** 2 + spectrum.imag ** 2) ** 2
    log_mel = np.log(magnitude @ slaney_mel().T + np.float32(2 ** -24))
    mask = np.arange(len(log_mel)) < valid
    selected = log_mel * mask[:, None]
    mean = selected.sum(0) / valid
    variance = (((selected - mean) ** 2) * mask[:, None]).sum(0) / (valid - 1)
    features = ((log_mel - mean) / (np.sqrt(variance) + np.float32(1e-5))) * mask[:, None]
    return {"preemphasis": pre, "log_mel": log_mel, "features": features.astype(np.float32),
            "attention_mask": mask, "valid_frames": valid, "stored_frames": len(log_mel)}
