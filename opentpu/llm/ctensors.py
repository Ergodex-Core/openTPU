"""compressed-tensors "pack-quantized" checkpoints (llm-compressor's W4A16 / W8A16 weight-only
quantization, e.g. RedHatAI/Qwen3.8-27B-INT4): a quantized Linear's weight is stored as

  X.weight_packed   int32 [N, ceil(K b / 32)]: its intB codes, 32/b a word, code i of a word
                    in bits b*i .. b*i+b-1 (the least significant first), along the input
                    dimension K; offset-binary (the code is value + 2^(b-1): a 4-bit -8 .. 7
                    is stored 0 .. 15, NOT two's complement)
  X.weight_scale    bf16 (or fp16 / fp32) [N, K / group]: one scale per group of `group`
                    input columns of each output row (strategy "group"; "channel": one per row)
  X.weight_shape    int64 [2]: (N, K)
  X.weight_zero_point  only for asymmetric schemes (packed like the weight along N); refused

and weight = code * scale[row, column // group]. The convention is compressed-tensors'
compressors/pack_quantized/helpers.py (pack_to_int32 / unpack_from_int32, v0.19.0): a value
offset by 1 << (num_bits - 1), element i at global bit position i * num_bits, int32 words.
An "actorder" of "static" / "weight" permutes the quantization order only (the stored
weight and its groups are in the original column order: no g_idx); "group" / "dynamic"
stores X.weight_g_idx and is refused.

`scheme(model_dir)` reads config.json's quantization_config; `names(keys)` the virtual
"X.weight" names of a checkpoint's packed tensors; `dequantized(get, triple, scheme)` the
fp32 matrix of one of them, on demand and one at a time (qwen3.Weights / LazyWeights: the
dequantized fp32 model of a 27B checkpoint is ~100 GB, the image build quantizes one tensor
at a time). docs/qwen38.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PACKED = ".weight_packed"
SCALE = ".weight_scale"
SHAPE = ".weight_shape"
ZERO = ".weight_zero_point"
G_IDX = ".weight_g_idx"


def scheme(model_dir) -> dict | None:
    """The weight scheme of a compressed-tensors pack-quantized checkpoint: {"num_bits",
    "group_size" (0: per channel), "symmetric", "type"}; None for a checkpoint without a
    quantization_config of quant_method compressed-tensors. ValueError for a compressed-tensors
    format or scheme this reader does not hold (not pack-quantized, float weights, asymmetric,
    a non-group strategy, bits that do not divide 32, activation ordering by groups)."""
    cfg = Path(model_dir) / "config.json"
    if not cfg.exists():
        return None
    q = json.loads(cfg.read_text()).get("quantization_config")
    if not q or q.get("quant_method") != "compressed-tensors":
        return None
    fmt = q.get("format")
    if fmt != "pack-quantized":
        raise ValueError(f"compressed-tensors format {fmt!r}: only pack-quantized is read")
    groups = [g for g in (q.get("config_groups") or {}).values() if g.get("weights")]
    if not groups:
        raise ValueError("compressed-tensors: no weight quantization in config_groups")
    out = None
    for g in groups:
        w = g["weights"]
        bits, strategy = int(w.get("num_bits", 4)), w.get("strategy", "group")
        s = {"num_bits": bits, "symmetric": bool(w.get("symmetric", True)),
             "type": w.get("type", "int"),
             "group_size": int(w.get("group_size") or 0) if strategy == "group" else 0}
        if s["type"] != "int":
            raise ValueError(f"compressed-tensors: {s['type']} weights (int only)")
        if not s["symmetric"]:
            raise ValueError("compressed-tensors: an asymmetric scheme (zero points)")
        if strategy not in ("group", "channel"):
            raise ValueError(f"compressed-tensors: weight strategy {strategy!r} (group, channel)")
        if 32 % bits:
            raise ValueError(f"compressed-tensors: {bits}-bit codes (1, 2, 4 or 8 pack a word)")
        if (w.get("actorder") or "").lower() in ("group", "dynamic"):
            raise ValueError("compressed-tensors: activation ordering by groups (g_idx)")
        if out is not None and out != s:
            raise ValueError(f"compressed-tensors: config groups differ ({out} / {s})")
        out = s
    return out


def names(keys) -> dict:
    """{"X.weight": ("X.weight_packed", "X.weight_scale", "X.weight_shape" or None)} for the
    packed tensors among `keys` (checkpoint names, or the loaders' text-only ones)."""
    ks = set(keys)
    out = {}
    for k in ks:
        if not k.endswith(PACKED):
            continue
        base = k[:-len(PACKED)]
        if base + ZERO in ks or base + G_IDX in ks:
            raise ValueError(f"{base}: a zero point or g_idx tensor (asymmetric or "
                             f"group-ordered compressed-tensors are not read)")
        out[base + ".weight"] = (k, base + SCALE, base + SHAPE if base + SHAPE in ks else None)
    return out


def unpack(packed: np.ndarray, num_bits: int, K: int) -> np.ndarray:
    """The intB codes [N, K] (int8) of int32 words [N, words] packed as compressed-tensors
    packs them (unpack_from_int32): code i of a word in bits num_bits*i .. (the least
    significant first), offset by 2^(num_bits-1)."""
    if 32 % num_bits:
        raise ValueError(f"{num_bits}-bit codes do not pack a word")
    per = 32 // num_bits
    w = np.ascontiguousarray(packed).view(np.uint32)
    N, words = w.shape
    if words * per < K:
        raise ValueError(f"{words} words of {per} codes hold {words * per} < K = {K}")
    shifts = (np.arange(per, dtype=np.uint32) * np.uint32(num_bits))
    mask = np.uint32((1 << num_bits) - 1)
    u = ((w[:, :, None] >> shifts) & mask).astype(np.int16).reshape(N, words * per)[:, :K]
    return (u - np.int16(1 << (num_bits - 1))).astype(np.int8)


def dequantize(packed: np.ndarray, scale: np.ndarray, shape, num_bits: int,
               group: int) -> np.ndarray:
    """The fp32 matrix [N, K] of packed codes and their scales [N, K / group] (group 0: one
    per row): code * scale, computed in fp32 (a bf16 scale times a small integer is exact)."""
    N, K = (int(x) for x in shape)
    q = unpack(packed, num_bits, K)
    s = np.asarray(scale, np.float32)
    if group == 0:
        s = s.reshape(N, 1)
        if s.shape[0] != N:
            raise ValueError(f"scales {s.shape} for {N} rows")
        return q.astype(np.float32) * s
    if K % group:
        raise ValueError(f"K = {K} is not a multiple of the group size {group}")
    s = s.reshape(N, -1)
    if s.shape[1] != K // group:
        raise ValueError(f"scales {s.shape} for [{N}, {K}] in groups of {group}")
    out = np.empty((N, K), np.float32)
    v = out.reshape(N, K // group, group)
    np.multiply(q.reshape(N, K // group, group), s[:, :, None], out=v)
    return out


def dequantized(get, triple: tuple, sch: dict) -> np.ndarray:
    """One packed tensor as fp32: get(name) reads a checkpoint tensor (a torch tensor);
    `triple` is names()' (packed, scale, shape or None)."""
    import torch
    kp, ks, kshape = triple
    packed = get(kp)
    if packed.dtype != torch.int32:
        raise ValueError(f"{kp}: {packed.dtype}, not int32")
    packed = packed.numpy()
    scale = get(ks).to(torch.float32).numpy()
    g = sch["group_size"]
    if kshape is not None:
        shape = tuple(int(x) for x in get(kshape).tolist())
    else:
        shape = (packed.shape[0], scale.shape[-1] * g if g else packed.shape[1] * 32 // sch["num_bits"])
    return dequantize(packed, scale, shape, sch["num_bits"], g)


def pack(values: np.ndarray, num_bits: int) -> np.ndarray:
    """compressed-tensors' pack_to_int32 for codes [N, K] in [-2^(b-1), 2^(b-1) - 1] whose K
    fills whole words (the tests' reference is the library's own function; this is its
    numpy twin for a quick synthetic checkpoint)."""
    per = 32 // num_bits
    v = (np.asarray(values, np.int64) + (1 << (num_bits - 1)))
    N, K = v.shape
    if K % per:
        raise ValueError(f"K = {K} does not fill words of {per}")
    v = v.reshape(N, K // per, per).astype(np.uint32)
    shifts = (np.arange(per, dtype=np.uint32) * np.uint32(num_bits))
    return (v << shifts).sum(-1, dtype=np.uint32).view(np.int32)
