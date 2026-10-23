"""Pinned original-Mimi encoder transformer reference for offline checks.

Loads only encoder_transformer checkpoint tensors. Native streaming positions,
including the original full-ring delta<=0 behavior, are captured unchanged.
This module never serves inference or accesses an FPGA.
"""

from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import struct

import numpy as np


_SOURCE_SHA256 = {
    "models/loaders.py": "24d484efc2198c118f6d2b22877c765cd82e89887c52774c42aac5410b299148",
    "models/compression.py": "7af4b2791d16fd0add4ef4f51d4322fc9fec159cc96604e569bbb53fe756b4ed",
    "modules/conv.py": "d4f9512cdc1343e959b0222dc2f18632f986ce8862d1cae5293542e0e282c51e",
    "modules/resample.py": "136931781a9de1e43afd19c4ffc9817e462f20b695853d415695d34f912405d9",
    "modules/streaming.py": "9fbdd77237d53ef6d95d1a08bce3b2438d661223ad8b795edfccc1b87eb4b81b",
    "modules/transformer.py": "0eccf9b539ba738910d201a66809c68f5dad14f6b020fee862a592fd00e1de55",
    "modules/rope.py": "10930a26f7241f53f1d6eb03ed04e8d2b94b49f23d225e206fdbddf8cf95b04f",
}
_CODEC_BYTES = 384644900
_HEADER_BYTES = 39320
_HEADER_SHA256 = "ec30eb84e7a32c11b2913db3d89595da90a422320dea3b66a3b39e16f821cc93"
_TENSOR_COUNT = 318
_SELECTED_COUNT = 80
_SELECTED_BYTES = 100761600
_CONSTRUCTOR = {
    "input_dimension": 512, "output_dimensions": [512], "d_model": 512,
    "num_heads": 8, "num_layers": 8, "causal": True, "context": 250,
    "layer_scale": 0.01, "conv_layout": True, "max_period": 10000.0,
    "gating": "none", "norm": "layer_norm", "positional_embedding": "rope",
    "dim_feedforward": 2048, "device": "cpu",
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root(contract, *objects):
    roots = {Path(inspect.getfile(obj)).resolve().parents[1] for obj in objects}
    if len(roots) != 1 or contract.get("installed_source_sha256") != _SOURCE_SHA256:
        raise ValueError("Native input transformer source contract differs")
    root = roots.pop()
    for name, digest in _SOURCE_SHA256.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or _sha256(path) != digest:
            raise ValueError("Native input transformer source changed: " + name)
    return root


def _selected_header(spec):
    h, f = spec.hidden, spec.ffn
    expected = {}
    for index in range(spec.layers):
        prefix = f"encoder_transformer.transformer.layers.{index}."
        entries = (
            ("layer_scale_1.scale", (h,)), ("layer_scale_2.scale", (h,)),
            ("linear1.weight", (f, h)), ("linear2.weight", (h, f)),
            ("norm1.bias", (h,)), ("norm1.weight", (h,)),
            ("norm2.bias", (h,)), ("norm2.weight", (h,)),
            ("self_attn.in_proj_weight", (3 * h, h)),
            ("self_attn.out_proj.weight", (h, h)))
        expected.update({prefix + name: shape for name, shape in entries})
    return expected


def _checked_header(codec_path, spec, contract):
    recorded = contract.get("cached_codec_header")
    if not isinstance(recorded, dict):
        raise ValueError("Native input transformer contract lacks cached codec header")
    for name, value in (("file_bytes", _CODEC_BYTES), ("header_bytes", _HEADER_BYTES),
                        ("header_sha256", _HEADER_SHA256), ("tensor_count", _TENSOR_COUNT),
                        ("selected_count", _SELECTED_COUNT), ("selected_bytes", _SELECTED_BYTES)):
        if recorded.get(name) != value:
            raise ValueError("Cached input transformer contract differs: " + name)
    path = Path(codec_path).resolve()
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    if before[0] != _CODEC_BYTES:
        raise ValueError("Cached codec file size differs")
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8 or struct.unpack("<Q", raw_size)[0] != _HEADER_BYTES:
            raise ValueError("Cached codec header length differs")
        raw = handle.read(_HEADER_BYTES)
    if len(raw) != _HEADER_BYTES or hashlib.sha256(raw).hexdigest() != _HEADER_SHA256:
        raise ValueError("Cached codec header hash differs")
    header = json.loads(raw)
    if len([name for name in header if name != "__metadata__"]) != _TENSOR_COUNT:
        raise ValueError("Cached codec tensor count differs")
    expected = _selected_header(spec)
    actual = {name: meta for name, meta in header.items()
              if name.startswith("encoder_transformer.")}
    if (set(actual) != set(expected) or len(actual) != _SELECTED_COUNT
            or actual != recorded.get("selected_tensors")):
        raise ValueError("Cached input transformer tensor metadata differs")
    total = 0
    for name, shape in expected.items():
        meta = actual[name]
        offsets = meta.get("data_offsets")
        size = 4 * int(np.prod(shape))
        if (meta.get("dtype") != "F32" or tuple(meta.get("shape", ())) != shape
                or not isinstance(offsets, list) or len(offsets) != 2
                or type(offsets[0]) is not int or type(offsets[1]) is not int
                or offsets[1] - offsets[0] != size):
            raise ValueError("Cached input transformer tensor shape/offset differs: " + name)
        total += size
    if total != _SELECTED_BYTES or (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise ValueError("Cached input transformer payload or file metadata changed")
    return path, before, expected


def _array(tensor, shape, name):
    value = tensor.detach().cpu().numpy().copy()
    if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Native input transformer tensor differs: " + name)
    return np.ascontiguousarray(value)


def native_encoder_transformer(codec_path, spec, contract):
    """Strict-load original CPU encoder transformer and ordered FP32 weights."""
    import torch
    from safetensors import safe_open
    from moshi.modules.transformer import ProjectedTransformer
    from moshi.modules.rope import apply_rope

    if (spec.hidden != 512 or spec.n_heads != 8 or spec.ffn != 2048 or spec.layers != 8
            or spec.context != 250 or spec.eps != 1e-5 or spec.rope_base != 10000.0
            or not 0 < spec.max_frames <= 128
            or contract.get("device_spec") != dict(hidden=512, n_heads=8, ffn=2048,
                layers=8, context=250, max_frames=128, eps=1e-5, rope_base=10000.0)
            or contract.get("native_constructor") != _CONSTRUCTOR):
        raise ValueError("Native input transformer requires pinned production dimensions")
    _source_root(contract, ProjectedTransformer, apply_rope)
    path, before, selected = _checked_header(codec_path, spec, contract)
    transformer = ProjectedTransformer(**_CONSTRUCTOR)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        state = {name.removeprefix("encoder_transformer."): handle.get_tensor(name)
                 for name in selected}
    if (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise ValueError("Cached codec changed during input transformer load")
    transformer.load_state_dict(state, strict=True)
    transformer.eval()
    del state

    hidden, ffn = spec.hidden, spec.ffn
    layers = []
    for index, layer in enumerate(transformer.transformer.layers):
        qkv = _array(layer.self_attn.in_proj_weight, (3 * hidden, hidden), f"layer{index}.qkv")
        layers.append({
            "norm1_weight": _array(layer.norm1.weight, (hidden,), f"layer{index}.norm1.weight"),
            "norm1_bias": _array(layer.norm1.bias, (hidden,), f"layer{index}.norm1.bias"),
            "norm2_weight": _array(layer.norm2.weight, (hidden,), f"layer{index}.norm2.weight"),
            "norm2_bias": _array(layer.norm2.bias, (hidden,), f"layer{index}.norm2.bias"),
            "scale1": _array(layer.layer_scale_1.scale, (hidden,), f"layer{index}.scale1"),
            "scale2": _array(layer.layer_scale_2.scale, (hidden,), f"layer{index}.scale2"),
            "wq": np.ascontiguousarray(qkv[:hidden]),
            "wk": np.ascontiguousarray(qkv[hidden:2 * hidden]),
            "wv": np.ascontiguousarray(qkv[2 * hidden:]),
            "wo": _array(layer.self_attn.out_proj.weight, (hidden, hidden), f"layer{index}.wo"),
            "w1": _array(layer.linear1.weight, (ffn, hidden), f"layer{index}.w1"),
            "w2": _array(layer.linear2.weight, (hidden, ffn), f"layer{index}.w2"),
        })
    if len(layers) != spec.layers:
        raise ValueError("Native input transformer layer count differs")

    # Build RoPE from the installed original FP32 operator, preserving its
    # adjacent real/imaginary basis and two-token streaming positions.
    basis = torch.zeros((1, 1, 2 * spec.max_frames, spec.head_dim), dtype=torch.float32)
    basis[..., 0::2] = 1.0
    with torch.no_grad():
        rotated, _ = apply_rope(basis, basis, torch.zeros(1, dtype=torch.long),
                                max_period=spec.rope_base, time_before_heads=False)
    coeffs = rotated.detach().numpy().reshape(spec.max_frames, 2, spec.head_dim // 2, 2)
    cos = np.ascontiguousarray(coeffs[..., 0].copy())
    sin = np.ascontiguousarray(coeffs[..., 1].copy())
    if (cos.shape != (spec.max_frames, 2, spec.head_dim // 2)
            or not np.isfinite(cos).all() or not np.isfinite(sin).all()):
        raise ValueError("Native input transformer RoPE extraction failed")
    transformer._mimi_native_input_spec = spec
    return transformer, layers, (cos, sin)


def collect_native_encoder_transformer(transformer, inputs):
    """Capture original two-token streaming output/layers and actual KV slots."""
    import torch

    spec = getattr(transformer, "_mimi_native_input_spec", None)
    if spec is None:
        raise ValueError("Use transformer returned by native_encoder_transformer")
    values = np.asarray(inputs)
    if (values.dtype != np.float32 or values.ndim != 3
            or values.shape[1:] != (2, spec.hidden)
            or not 0 < values.shape[0] <= spec.max_frames or not np.isfinite(values).all()):
        raise ValueError("Native input transformer needs finite FP32 [frames,2,512]")
    values = np.ascontiguousarray(values)
    count = len(values)
    output = []
    layers = [[] for _ in range(spec.layers)]
    positions = [[] for _ in range(spec.layers)]
    handles, originals = [], []
    with ExitStack() as stack:
        stack.enter_context(transformer.streaming(1))
        try:
            for index, layer in enumerate(transformer.transformer.layers):
                def layer_hook(module, args, result, index=index):
                    layers[index].append(_array(result[0], (2, spec.hidden),
                                                f"native encoder transformer layer{index}"))
                handles.append(layer.register_forward_hook(layer_hook))
                cache = layer.self_attn._streaming_state.kv_cache
                original = cache.complete
                def complete(k, v, original=original, index=index):
                    result = original(k, v)
                    positions[index].append(result.positions.detach().cpu().numpy().copy())
                    return result
                cache.complete = complete
                originals.append((cache, original))
            with torch.no_grad():
                for frame in values:
                    pair = torch.from_numpy(frame.T.copy()[None, :, :])
                    (encoded,) = transformer(pair)
                    if tuple(encoded.shape) != (1, spec.hidden, 2):
                        raise ValueError("Native encoder transformer did not emit two tokens")
                    output.append(_array(encoded[0].transpose(0, 1), (2, spec.hidden),
                                         "native encoder transformer output"))
        finally:
            for handle in handles:
                handle.remove()
            for cache, original in originals:
                cache.complete = original
    if any(len(frames) != count for frames in layers + positions):
        raise ValueError("Native encoder transformer hook count differs")
    return {"output": np.stack(output),
            "layers": np.stack([np.stack(frames) for frames in layers], axis=1),
            "positions": positions}
