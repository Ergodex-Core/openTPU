"""Offline CPU audit of the pinned learned subsampler; never a serving backend."""
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from opentpu.asr.phonon2 import CONTAINER_SHA256
from opentpu.asr.phonon2_int6 import decode_int6
from opentpu.asr.phonon2_subsampler import SubsampleShape, weight_shapes, lengths


def read_subsampler(path):
    path = Path(path)
    with path.open("rb") as f:
        digest = hashlib.sha256()
        for part in iter(lambda: f.read(1 << 20), b""):
            digest.update(part)
        if digest.hexdigest() != CONTAINER_SHA256:
            raise ValueError("container SHA256 differs from pinned checkpoint")
        f.seek(0)
        size = int.from_bytes(f.read(8), "little")
        if not 0 < size <= 1 << 20:
            raise ValueError("invalid bounded header length")
        header = json.loads(f.read(size))
        if header["format"] != "fermion-five-value-parakeet-v1":
            raise ValueError("unexpected container format")
        expected, weights, hashes = weight_shapes(), {}, {}
        for entry in header["index"]:
            full_name, count = entry["n"], entry["b"]
            if type(count) is not int or not 0 <= count <= 64 << 20:
                raise ValueError("invalid record byte count")
            name = full_name.removeprefix("encoder.subsampling.") if full_name.startswith("encoder.subsampling.") else None
            if name not in expected:
                f.seek(count, 1)
                continue
            kind, dims = expected[name]
            if name in weights or entry["k"] != kind or tuple(entry["shape"]) != dims:
                raise ValueError("duplicate or mismatched subsampler tensor")
            data = f.read(count)
            if len(data) != count:
                raise ValueError("truncated record")
            hashes[full_name] = hashlib.sha256(data).hexdigest()
            if kind == "int6":
                weights[name] = decode_int6(data, dims)
            elif count == 2 * math.prod(dims):
                weights[name] = np.frombuffer(data, "<f2").astype(np.float32).reshape(dims)
            else:
                raise ValueError("FP16 record size differs from shape")
        if f.tell() != path.stat().st_size or set(weights) != set(expected):
            raise ValueError("invalid container extent or missing tensor")
    return weights, hashes


def _activation(a, D):
    if D is None:
        return a
    from opentpu import fp32 as F
    rows, cols = a.shape
    kp = (cols + D - 1) // D * D
    padded = np.pad(a, ((0, 0), (0, kp - cols)))
    q, scale, offsets = F.quantize_mx(padded, D // 4)
    return (q.astype(np.float32).reshape(rows, -1, D // 4) *
            np.ldexp(scale[:, None], -offsets.astype(int))[:, :, None]).reshape(rows, kp)[:, :cols]


def subsampler_reference(weights, features, valid_frames, shape=SubsampleShape(), *, quantize_D=None):
    """Independent NumPy conv equations; optional activation quantization at linears."""
    x = np.asarray(features, np.float32)
    stored, valid = lengths(len(x), valid_frames)
    w, snapshots = weights, {"input": x.copy()}
    source = np.pad(x, ((1, 1), (1, 1)))
    a = np.zeros((stored[1], 64, shape.channels), np.float32)
    for dt in range(3):
        for df in range(3):
            a += source[dt:dt + 2 * stored[1]:2, df:df + 128:2, None] * w["layers.0.weight"][None, None, :, 0, dt, df]
    a += w["layers.0.bias"]
    a[valid[1]:] = 0
    a = np.maximum(a, 0)
    snapshots["conv0"] = a.copy()
    for stage, depth, point, freq in ((2, 2, 3, 32), (3, 5, 6, 16)):
        source = np.pad(a, ((1, 1), (1, 1), (0, 0)))
        a = np.zeros((stored[stage], freq, shape.channels), np.float32)
        for dt in range(3):
            for df in range(3):
                a += source[dt:dt + 2 * stored[stage]:2, df:df + 2 * freq:2] * w[f"layers.{depth}.weight"][None, None, :, 0, dt, df]
        a += w[f"layers.{depth}.bias"]
        a[valid[stage]:] = 0
        snapshots[f"depthwise{depth}"] = a.copy()
        projected = _activation(a.reshape(-1, shape.channels), quantize_D) @ w[f"layers.{point}.weight"].dense().T
        a = projected.reshape(a.shape) + w[f"layers.{point}.bias"]
        a[valid[stage]:] = 0
        a = np.maximum(a, 0)
        snapshots[f"pointwise{point}"] = a.copy()
    flat = a.transpose(0, 2, 1).reshape(stored[3], shape.channels * 16)
    padded = flat if quantize_D is None else np.pad(flat, ((0, 0), (0, (-flat.shape[1]) % quantize_D)))
    snapshots["flat"] = padded.copy()
    snapshots["output"] = _activation(flat, quantize_D) @ w["linear.weight"].dense().T + w["linear.bias"]
    return snapshots


def subsampler_torch(weights, features, valid_frames, shape=SubsampleShape()):
    """Own audit using native CPU conv2d and the literal upstream tensor layout."""
    import torch
    import torch.nn.functional as f
    w = {name: torch.from_numpy(value.dense() if hasattr(value, "dense") else np.array(value, dtype=np.float32))
         for name, value in weights.items()}
    x = torch.from_numpy(np.array(features, dtype=np.float32))[None, None]
    valid = valid_frames
    snapshots = {"input": np.array(features, dtype=np.float32)}
    def save(name, a):
        snapshots[name] = a.permute(0, 2, 3, 1)[0].numpy().copy()
    x = f.conv2d(x, w["layers.0.weight"], w["layers.0.bias"], stride=2, padding=1)
    valid = (valid + 1) // 2
    x[:, :, valid:] = 0
    x = f.relu(x)
    save("conv0", x)
    for depth, point in ((2, 3), (5, 6)):
        x = f.conv2d(x, w[f"layers.{depth}.weight"], w[f"layers.{depth}.bias"], stride=2, padding=1, groups=shape.channels)
        valid = (valid + 1) // 2
        x[:, :, valid:] = 0
        save(f"depthwise{depth}", x)
        x = f.conv2d(x, w[f"layers.{point}.weight"].reshape(shape.channels, shape.channels, 1, 1), w[f"layers.{point}.bias"])
        x[:, :, valid:] = 0
        x = f.relu(x)
        save(f"pointwise{point}", x)
    flat = x.transpose(1, 2).reshape(1, x.shape[2], -1)
    snapshots["flat"] = flat[0].numpy().copy()
    snapshots["output"] = f.linear(flat, w["linear.weight"], w["linear.bias"])[0].numpy()
    return snapshots


def read_tutorial_audio(path):
    """Bounded RIFF/WAVE data decoder for the tutorial's mono float32 recording.

    File decoding and conversion here are explicit offline golden preparation.
    Serving does not call this. No audio resampling or model inference occurs.
    """
    data = Path(path).read_bytes()
    if len(data) > 2_000_000 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("expected bounded RIFF WAVE data")
    cursor, fmt, pcm = 12, None, None
    while cursor + 8 <= len(data):
        key, count = data[cursor:cursor + 4], int.from_bytes(data[cursor + 4:cursor + 8], "little")
        cursor += 8
        if cursor + count > len(data):
            raise ValueError("truncated RIFF chunk")
        if key == b"fmt ":
            fmt = data[cursor:cursor + count]
        elif key == b"data":
            pcm = data[cursor:cursor + count]
        cursor += count + (count & 1)
    if fmt is None or len(fmt) < 16 or pcm is None:
        raise ValueError("missing WAVE format/data")
    kind, channels, rate, _, align, bits = struct.unpack_from("<HHIIHH", fmt)
    if (kind, channels, rate, align, bits) != (3, 1, 16000, 4, 32) or len(pcm) % 4:
        raise ValueError("require tutorial mono 16kHz IEEE float32 WAVE")
    audio = np.frombuffer(pcm, "<f4")
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
        raise ValueError("invalid float32 audio amplitude")
    return np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2")
