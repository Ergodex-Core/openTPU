"""Original Mimi decoder diagnostics for offline checkpoint qualification only.

This module never serves inference or accesses an FPGA.  The production path
uses resident ISA programs; these helpers load the pinned CPU modules solely to
compare their outputs and to extract immutable weights/RoPE tables.
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


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root(contract, *source_objects):
    """Reject a changed installed package before touching checkpoint payloads."""
    roots = {Path(inspect.getfile(obj)).resolve().parents[1] for obj in source_objects}
    if len(roots) != 1:
        raise ValueError("Native Mimi classes came from different source trees")
    root = roots.pop()
    recorded = contract["installed_source_sha256"]
    for name, digest in _SOURCE_SHA256.items():
        if recorded.get(name) != digest:
            raise ValueError(f"Decoder source contract differs: {name}")
    for name, digest in recorded.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or len(digest) != 64:
            raise ValueError(f"Invalid native source contract entry: {name}")
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or _sha256(path) != digest:
            raise ValueError(f"Native codec source changed: {name}")
    return root


def _selected_header(spec):
    h, f = spec.hidden, spec.ffn
    expected = {"upsample.convtr.convtr.convtr.weight": (h, 1, 4)}
    for index in range(spec.layers):
        prefix = f"decoder_transformer.transformer.layers.{index}."
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
    header_contract = contract.get("cached_codec_header", contract.get("codec_header"))
    if header_contract is None:
        raise ValueError("Native decoder contract lacks cached codec header")
    for name, value in (("file_bytes", _CODEC_BYTES), ("header_bytes", _HEADER_BYTES),
                        ("header_sha256", _HEADER_SHA256), ("tensor_count", _TENSOR_COUNT)):
        if header_contract.get(name) != value:
            raise ValueError(f"Cached codec contract differs: {name}")
    path = Path(codec_path).resolve()
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    if before[0] != _CODEC_BYTES:
        raise ValueError("Cached codec file size differs")
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Cached codec header is incomplete")
        size = struct.unpack("<Q", size_bytes)[0]
        if size != _HEADER_BYTES:
            raise ValueError("Cached codec header length differs")
        raw = handle.read(size)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != _HEADER_SHA256:
        raise ValueError("Cached codec header hash differs")
    header = json.loads(raw)
    if len([key for key in header if key != "__metadata__"]) != _TENSOR_COUNT:
        raise ValueError("Cached codec tensor count differs")
    expected = _selected_header(spec)
    actual = {name: header[name] for name in header
              if name.startswith("upsample.") or name.startswith("decoder_transformer.")}
    if set(actual) != set(expected):
        raise ValueError("Cached codec decoder tensor names differ")
    for name, shape in expected.items():
        if actual[name]["dtype"] != "F32" or tuple(actual[name]["shape"]) != shape:
            raise ValueError(f"Cached codec decoder tensor shape differs: {name}")
    recorded = header_contract.get("selected_tensors")
    if recorded is not None and recorded != actual:
        raise ValueError("Cached codec decoder tensor metadata differs")
    return path, before, expected


def _array(tensor, shape, name):
    value = tensor.detach().cpu().numpy().copy()
    if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Native decoder tensor differs: {name}")
    return np.ascontiguousarray(value)


def native_modules(codec_path, spec, contract):
    """Load only the original CPU upsampler and decoder transformer, strictly.

    Returns (upsample, transformer, kernel_np, layers, (cos, sin)).  All arrays
    use source layout; the production image applies its own row permutation and
    quantization.  The checkpoint header and installed source are pinned first.
    """
    import torch
    from safetensors import safe_open
    from moshi.modules.resample import ConvTrUpsample1d
    from moshi.modules.transformer import ProjectedTransformer
    from moshi.modules.rope import apply_rope

    if (spec.hidden != 512 or spec.n_heads != 8 or spec.ffn != 2048 or spec.layers != 8
            or spec.context != 250 or spec.eps != 1e-5 or spec.rope_base != 10000.0
            or not 0 < spec.max_frames <= 128):
        raise ValueError("Native decoder requires the pinned production dimensions")
    _source_root(contract, ConvTrUpsample1d, ProjectedTransformer, apply_rope)
    path, before, selected = _checked_header(codec_path, spec, contract)

    upsample = ConvTrUpsample1d(stride=2, dimension=spec.hidden, causal=True,
                                learnt=True, channel_wise=True)
    transformer = ProjectedTransformer(
        input_dimension=spec.hidden, output_dimensions=[spec.hidden], d_model=spec.hidden,
        num_heads=spec.n_heads, num_layers=spec.layers, causal=True, context=spec.context,
        layer_scale=0.01, conv_layout=True, max_period=spec.rope_base, gating="none",
        norm="layer_norm", positional_embedding="rope", dim_feedforward=spec.ffn,
        device="cpu")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        up_state = {name.removeprefix("upsample."): handle.get_tensor(name)
                    for name in selected if name.startswith("upsample.")}
        tf_state = {name.removeprefix("decoder_transformer."): handle.get_tensor(name)
                    for name in selected if name.startswith("decoder_transformer.")}
    if (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise ValueError("Cached codec changed during native decoder load")
    upsample.load_state_dict(up_state, strict=True)
    transformer.load_state_dict(tf_state, strict=True)
    upsample.eval()
    transformer.eval()
    del up_state, tf_state

    hidden, ffn = spec.hidden, spec.ffn
    kernel = _array(upsample.convtr.convtr.convtr.weight, (hidden, 1, 4),
                    "upsample.weight").reshape(hidden, 4)
    layers = []
    for index, layer in enumerate(transformer.transformer.layers):
        qkv = _array(layer.self_attn.in_proj_weight, (3 * hidden, hidden),
                     f"layer{index}.qkv")
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

    # Construct coefficients through the installed FP32 RoPE implementation.
    # Adjacent [real, imaginary] basis pairs [1, 0] rotate to [cos, sin].
    basis = torch.zeros((1, 1, 2 * spec.max_frames, spec.head_dim), dtype=torch.float32)
    basis[..., 0::2] = 1.0
    with torch.no_grad():
        rotated, _ = apply_rope(basis, basis, torch.zeros(1, dtype=torch.long),
                                max_period=spec.rope_base, time_before_heads=False)
    coeffs = rotated.detach().numpy().reshape(spec.max_frames, 2, spec.head_dim // 2, 2)
    cos = np.ascontiguousarray(coeffs[..., 0].copy())
    sin = np.ascontiguousarray(coeffs[..., 1].copy())
    if cos.shape != (spec.max_frames, 2, spec.head_dim // 2) or not np.isfinite(cos).all():
        raise ValueError("Native RoPE coefficient extraction failed")
    upsample._mimi_native_spec = spec
    transformer._mimi_native_spec = spec
    return upsample, transformer, kernel, layers, (cos, sin)


def collect_native(upsample, transformer, latents):
    """Capture an independent original-source two-token streaming decode.

    `latents` is contiguous FP32 [frames, hidden].  Each frame is passed once
    through the original upsampler and once through the original transformer;
    no diagnostic read controls device execution or feeds a later native frame.
    """
    import torch

    spec = getattr(transformer, "_mimi_native_spec", None)
    if spec is None or getattr(upsample, "_mimi_native_spec", None) != spec:
        raise ValueError("Use original modules returned by native_modules")
    values = np.asarray(latents)
    if (values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != spec.hidden
            or not 0 < values.shape[0] <= spec.max_frames or not np.isfinite(values).all()):
        raise ValueError("Native decoder latents must be finite FP32 [frames, hidden]")
    values = np.ascontiguousarray(values)
    count = values.shape[0]
    upsampled, output = [], []
    layers = [[] for _ in range(spec.layers)]
    positions = [[] for _ in range(spec.layers)]
    handles, originals = [], []

    with ExitStack() as stack:
        stack.enter_context(upsample.convtr.streaming(1))
        stack.enter_context(transformer.streaming(1))
        try:
            for index, layer in enumerate(transformer.transformer.layers):
                def layer_hook(module, inputs, result, index=index):
                    layers[index].append(_array(result[0], (2, spec.hidden),
                                                f"native layer{index} output"))
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
                for frame in range(count):
                    latent = torch.from_numpy(values[frame].reshape(1, spec.hidden, 1))
                    pair = upsample(latent)
                    if tuple(pair.shape) != (1, spec.hidden, 2):
                        raise ValueError("Native upsample did not emit two tokens")
                    upsampled.append(_array(pair[0].transpose(0, 1), (2, spec.hidden),
                                            "native upsampled frame"))
                    (decoded,) = transformer(pair)
                    if tuple(decoded.shape) != (1, spec.hidden, 2):
                        raise ValueError("Native transformer did not return two tokens")
                    output.append(_array(decoded[0].transpose(0, 1), (2, spec.hidden),
                                         "native decoder output"))
        finally:
            for handle in handles:
                handle.remove()
            for cache, original in originals:
                cache.complete = original
    if any(len(frames) != count for frames in layers + positions):
        raise ValueError("Native decoder hook count differs from input frames")
    return {
        "upsampled": np.stack(upsampled),
        "output": np.stack(output),
        "layers": np.stack([np.stack(frames) for frames in layers], axis=1),
        "positions": positions,
    }
