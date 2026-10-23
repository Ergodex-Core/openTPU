"""Offline bounded checkpoint reader and native-Torch encoder audit equations.

Never a serving backend. The checkpoint is hashed once and only a layer's
records are decoded at a time. No upstream model code is installed/imported.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from opentpu.asr.phonon2 import CONTAINER_SHA256
from opentpu.asr.phonon2_conformer import BlockShape, weight_shapes
from opentpu.asr.phonon2_linear import decode_five_value
from opentpu.asr.phonon2_int6 import decode_int6
from opentpu.asr.phonon2_subsampler import weight_shapes as subsample_shapes


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as f:
            sha = hashlib.sha256()
            for b in iter(lambda: f.read(1 << 20), b""):
                sha.update(b)
            if sha.hexdigest() != CONTAINER_SHA256:
                raise ValueError("checkpoint differs from pinned container")
            f.seek(0)
            n = int.from_bytes(f.read(8), "little")
            if not 0 < n <= 1 << 20:
                raise ValueError("invalid bounded header")
            header = json.loads(f.read(n))
        if header["format"] != "fermion-five-value-parakeet-v1":
            raise ValueError("unsupported container")
        self.index, offset = {}, 8 + n
        for entry in header["index"]:
            name, size = entry["n"], entry["b"]
            if name in self.index or type(size) is not int or not 0 <= size <= 64 << 20:
                raise ValueError("duplicate or invalid record extent")
            self.index[name] = (entry, offset)
            offset += size
        if offset != self.path.stat().st_size:
            raise ValueError("container extent mismatch")

    def _read(self, prefix, expected):
        weights, hashes = {}, {}
        with self.path.open("rb") as f:
            for name, (expected_kind, dims) in expected.items():
                full = prefix + name
                entry, at = self.index[full]
                if tuple(entry["shape"]) != dims:
                    raise ValueError("shape mismatch " + full)
                f.seek(at)
                blob = f.read(entry["b"])
                if len(blob) != entry["b"]:
                    raise ValueError("truncated record")
                hashes[full] = hashlib.sha256(blob).hexdigest()
                kind = entry["k"]
                if expected_kind == "five_value" and kind == "five_value":
                    weights[name] = decode_five_value(blob, dims)
                elif kind == "int6" and (expected_kind == "int6" or name == "conv.depthwise_conv.weight"):
                    weight = decode_int6(blob, dims)
                    weights[name] = weight.dense().reshape(dims) if expected_kind == "array" else weight
                elif kind == "fp16" and expected_kind in ("array", "fp16") and len(blob) == 2 * math.prod(dims):
                    weights[name] = np.frombuffer(blob, "<f2").astype(np.float32).reshape(dims)
                else:
                    raise ValueError("kind mismatch " + full)
        return weights, hashes

    def layer(self, index):
        if type(index) is not int or not 0 <= index < 24:
            raise ValueError("layer index must be in [0,24)")
        return self._read(f"encoder.layers.{index}.", weight_shapes(BlockShape()))

    def subsampler(self):
        return self._read("encoder.subsampling.", subsample_shapes())


def dequant_activation(x, D):
    from opentpu import fp32 as F
    shape = x.shape
    a = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, shape[-1])
    q, scale, offset = F.quantize_mx(a, D // 4)
    return (q.astype(np.float32).reshape(len(a), -1, D // 4) *
            np.ldexp(scale[:, None], -offset.astype(int))[:, :, None]).reshape(shape)


def layer_torch(weights, x, valid, *, activation_passes=0, D=128, shape=BlockShape()):
    """Pinned eager eval graph; optional QACT decomposition is an audit variant."""
    import torch
    import torch.nn.functional as f
    w = {name: torch.from_numpy(value.dense() if hasattr(value, "dense") else np.array(value, dtype=np.float32))
         for name, value in weights.items()}
    x = torch.from_numpy(np.array(x, dtype=np.float32))[None]
    t, h, dim = x.shape[1], shape.hidden, shape.hidden // shape.heads
    def linear(a, name):
        if not activation_passes:
            return f.linear(a, w[name])
        source = a.numpy()
        first = dequant_activation(source, D)
        y = f.linear(torch.from_numpy(first), w[name])
        if activation_passes == 2:
            residual = dequant_activation(source - first, D)
            y = y + f.linear(torch.from_numpy(residual), w[name])
        return y
    def norm(a, name):
        return f.layer_norm(a, (h,), w[name + ".weight"], w[name + ".bias"], eps=1e-5)
    def ff(a, name):
        return a + .5 * linear(f.silu(linear(norm(a, "norm_" + name), name + ".linear1")), name + ".linear2")
    x = ff(x, "feed_forward1")
    z = norm(x, "norm_self_att")
    q, k, v = [linear(z, "self_attn." + n + "_proj").view(1, t, shape.heads, dim).transpose(1, 2) for n in ("q", "k", "v")]
    inv = 1 / (10000 ** (torch.arange(0, h, 2, dtype=torch.float32) / h))
    theta = torch.arange(t - 1, -t, -1, dtype=torch.float32)[:, None] * inv
    positions = torch.stack((theta.sin(), theta.cos()), dim=-1).reshape(1, 2 * t - 1, h)
    p = linear(positions, "self_attn.relative_k_proj").view(1, 2 * t - 1, shape.heads, dim)
    bd = (q + w["self_attn.bias_v"][None, :, None, :]) @ p.permute(0, 2, 3, 1)
    bd = f.pad(bd, (1, 0)).view(1, shape.heads, -1, t)[:, :, 1:].reshape(1, shape.heads, t, 2 * t - 1)[..., :t]
    bd *= dim ** -.5
    mask = (torch.arange(t)[:, None] < valid) & (torch.arange(t)[None, :] < valid)
    bd.masked_fill_(~mask[None, None], torch.finfo(torch.float32).min)
    score = ((q + w["self_attn.bias_u"][None, :, None, :]) @ k.transpose(-1, -2)) * (dim ** -.5) + bd
    a = (torch.softmax(score, dim=-1) @ v).transpose(1, 2).reshape(1, t, h)
    x = x + linear(a, "self_attn.o_proj")
    glu = f.glu(linear(norm(x, "norm_conv"), "conv.pointwise_conv1"), dim=-1).transpose(1, 2)
    glu[:, :, valid:] = 0
    conv = f.conv1d(glu, w["conv.depthwise_conv.weight"], padding=4, groups=h)
    conv = f.batch_norm(conv, w["conv.norm.running_mean"], w["conv.norm.running_var"], w["conv.norm.weight"], w["conv.norm.bias"], training=False, eps=1e-5)
    x = x + linear(f.silu(conv).transpose(1, 2), "conv.pointwise_conv2")
    return norm(ff(x, "feed_forward2"), "norm_out")[0].numpy().copy()
