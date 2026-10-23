"""Pinned original-Mimi SEANet encoder reference for offline qualification.

Only the cached checkpoint's encoder tensors are loaded. Inputs and outputs
remain raw FP32; no resampling, clipping, scaling, or board access occurs.
"""

from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import struct

import numpy as np


_SOURCE_SHA256 = {
    "models/compression.py": "7af4b2791d16fd0add4ef4f51d4322fc9fec159cc96604e569bbb53fe756b4ed",
    "models/loaders.py": "24d484efc2198c118f6d2b22877c765cd82e89887c52774c42aac5410b299148",
    "modules/conv.py": "d4f9512cdc1343e959b0222dc2f18632f986ce8862d1cae5293542e0e282c51e",
    "modules/seanet.py": "1611e47987b635bb90b19944fc3deca91c4100270f2117bf9fd9cc70bf0da9cc",
    "modules/streaming.py": "9fbdd77237d53ef6d95d1a08bce3b2438d661223ad8b795edfccc1b87eb4b81b",
    "server.py": "cdb9f68a326328308fa4acd43e7e75d79030d2214c43f828d186a0e03a13da99",
}
_CODEC_BYTES = 384644900
_HEADER_BYTES = 39320
_HEADER_SHA256 = "ec30eb84e7a32c11b2913db3d89595da90a422320dea3b66a3b39e16f821cc93"
_TENSOR_COUNT = 318
_ENCODER_COUNT = 28
_ENCODER_BYTES = 50513024
_KWARGS = {
    "channels": 1, "dimension": 512, "causal": True, "n_filters": 64,
    "n_residual_layers": 1, "activation": "ELU", "compress": 2,
    "dilation_base": 2, "disable_norm_outer_blocks": 0, "kernel_size": 7,
    "residual_kernel_size": 3, "last_kernel_size": 3, "norm": "none",
    "pad_mode": "constant", "ratios": [8, 6, 5, 4], "true_skip": True,
}
# Top-level model index, underlying RawStreamingConv1d path and native shape.
# Every convolution has a bias and retains PyTorch [out,in,kernel] layout.
_CONVOLUTIONS = (
    (0, "conv.conv", (64, 1, 7)),
    (1, "block.1.conv.conv", (32, 64, 3)),
    (1, "block.3.conv.conv", (64, 32, 1)),
    (3, "conv.conv", (128, 64, 8)),
    (4, "block.1.conv.conv", (64, 128, 3)),
    (4, "block.3.conv.conv", (128, 64, 1)),
    (6, "conv.conv", (256, 128, 10)),
    (7, "block.1.conv.conv", (128, 256, 3)),
    (7, "block.3.conv.conv", (256, 128, 1)),
    (9, "conv.conv", (512, 256, 12)),
    (10, "block.1.conv.conv", (256, 512, 3)),
    (10, "block.3.conv.conv", (512, 256, 1)),
    (12, "conv.conv", (1024, 512, 16)),
    (14, "conv.conv", (512, 1024, 3)),
)
_STAGE_SHAPES = (
    (1920, 64), (1920, 64), (1920, 64),
    (480, 128), (480, 128), (480, 128),
    (96, 256), (96, 256), (96, 256),
    (16, 512), (16, 512), (16, 512),
    (2, 1024), (2, 1024), (2, 512),
)
_STAGE_CLASSES = (
    "StreamingConv1d", "SEANetResnetBlock", "ELU",
    "StreamingConv1d", "SEANetResnetBlock", "ELU",
    "StreamingConv1d", "SEANetResnetBlock", "ELU",
    "StreamingConv1d", "SEANetResnetBlock", "ELU",
    "StreamingConv1d", "ELU", "StreamingConv1d",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root(contract, encoder_class):
    root = Path(inspect.getfile(encoder_class)).resolve().parents[1]
    if contract.get("installed_source_sha256") != _SOURCE_SHA256:
        raise ValueError("Native encoder installed-source contract differs")
    for name, digest in _SOURCE_SHA256.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or _sha256(path) != digest:
            raise ValueError("Installed native encoder source changed: " + name)
    return root


def _expected_tensors():
    expected = {}
    for index, subpath, shape in _CONVOLUTIONS:
        prefix = f"encoder.model.{index}.{subpath}."
        expected[prefix + "weight"] = shape
        expected[prefix + "bias"] = (shape[0],)
    return expected


def _read_header(path):
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8 or struct.unpack("<Q", raw_size)[0] != _HEADER_BYTES:
            raise ValueError("Cached codec header length differs")
        raw = handle.read(_HEADER_BYTES)
    if len(raw) != _HEADER_BYTES or hashlib.sha256(raw).hexdigest() != _HEADER_SHA256:
        raise ValueError("Cached codec header hash differs")
    return json.loads(raw)


def _checked_header(codec_path, contract):
    recorded = contract.get("cached_codec_header")
    if not isinstance(recorded, dict):
        raise ValueError("Native encoder contract lacks cached codec header")
    for name, value in (("file_bytes", _CODEC_BYTES), ("header_bytes", _HEADER_BYTES),
                        ("header_sha256", _HEADER_SHA256), ("tensor_count", _TENSOR_COUNT),
                        ("encoder_count", _ENCODER_COUNT), ("encoder_bytes", _ENCODER_BYTES)):
        if recorded.get(name) != value:
            raise ValueError("Cached codec encoder contract differs: " + name)
    path = Path(codec_path).resolve()
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    if before[0] != _CODEC_BYTES:
        raise ValueError("Cached codec file size differs")
    header = _read_header(path)
    if len([name for name in header if name != "__metadata__"]) != _TENSOR_COUNT:
        raise ValueError("Cached codec tensor count differs")
    expected = _expected_tensors()
    actual = {name: meta for name, meta in header.items() if name.startswith("encoder.")}
    if (set(actual) != set(expected) or len(actual) != _ENCODER_COUNT
            or actual != recorded.get("encoder_tensors")):
        raise ValueError("Cached codec encoder tensor metadata differs")
    total = 0
    for name, shape in expected.items():
        meta = actual[name]
        offsets = meta.get("data_offsets")
        size = 4 * int(np.prod(shape))
        if (meta.get("dtype") != "F32" or tuple(meta.get("shape", ())) != shape
                or not isinstance(offsets, list) or len(offsets) != 2
                or type(offsets[0]) is not int or type(offsets[1]) is not int
                or offsets[1] - offsets[0] != size):
            raise ValueError("Cached codec encoder tensor shape/offset differs: " + name)
        total += size
    if total != _ENCODER_BYTES or (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise ValueError("Cached codec encoder payload or file metadata changed")
    return path, before, expected


def _array(tensor, shape, name):
    value = tensor.detach().cpu().numpy().copy()
    if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Native encoder tensor differs: " + name)
    return np.ascontiguousarray(value)


def _submodule(root, path):
    value = root
    for part in path.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return value


def native_encoder_module(codec_path, contract):
    """Strict-load original CPU SEANetEncoder and extract ordered FP32 weights."""
    from safetensors import safe_open
    from moshi.modules.seanet import SEANetEncoder

    _source_root(contract, SEANetEncoder)
    if (contract.get("constructor_kwargs") != _KWARGS
            or contract.get("device_spec") != dict(hidden=512, n_filters=64,
                ratios=[8, 6, 5, 4], output_rows=2)
            or contract.get("stage_shapes") != [list(shape) for shape in _STAGE_SHAPES]
            or contract.get("stage_classes") != list(_STAGE_CLASSES)):
        raise ValueError("Native encoder topology contract differs")
    path, before, selected = _checked_header(codec_path, contract)
    encoder = SEANetEncoder(**_KWARGS)
    if ([type(module).__name__ for module in encoder.model] != list(_STAGE_CLASSES)
            or len(encoder.model) != len(_STAGE_SHAPES)):
        raise ValueError("Native encoder model topology differs")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        state = {name.removeprefix("encoder."): handle.get_tensor(name) for name in selected}
    if (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise ValueError("Cached codec changed during native encoder load")
    _read_header(path)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    del state
    weights = []
    for index, subpath, shape in _CONVOLUTIONS:
        module = _submodule(encoder.model[index], subpath)
        weights.append({"kind": "conv", "model_index": index, "subpath": subpath,
                        "weight": _array(module.weight, shape, f"model.{index}.{subpath}.weight"),
                        "bias": _array(module.bias, (shape[0],), f"model.{index}.{subpath}.bias")})
    encoder._mimi_native_encoder_contract = _HEADER_SHA256
    return encoder, weights


def collect_native_encoder(encoder, frames, forced_prefix=None):
    """Capture original streaming stage outputs from finite raw FP32 frames.

    A fresh native stream is opened per call. If supplied, ``forced_prefix``
    contains saved outputs of the first N stages, with N <= 14; each following
    native stage receives the corresponding saved prior-stage output while its
    own streaming state still advances. This isolates individual stage error.
    """
    import torch

    if getattr(encoder, "_mimi_native_encoder_contract", None) != _HEADER_SHA256:
        raise ValueError("Use the encoder returned by native_encoder_module")
    values = np.asarray(frames)
    if (values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 1920
            or not 0 < values.shape[0] <= 128 or not np.isfinite(values).all()):
        raise ValueError("Native encoder input must be finite raw FP32 [frames,1920]")
    values = np.ascontiguousarray(values)
    prefix = [] if forced_prefix is None else list(forced_prefix)
    if len(prefix) > len(_STAGE_SHAPES) - 1:
        raise ValueError("Forced encoder prefix may contain at most 14 stage outputs")
    for index, value in enumerate(prefix):
        array = np.asarray(value)
        if (array.dtype != np.float32 or array.shape != (len(values), *_STAGE_SHAPES[index])
                or not np.isfinite(array).all()):
            raise ValueError(f"Forced encoder stage {index} must be finite FP32")
        prefix[index] = np.ascontiguousarray(array)
    captured = [[] for _ in _STAGE_SHAPES]
    with ExitStack() as stack:
        stack.enter_context(encoder.streaming(1))
        with torch.no_grad():
            for frame_index, frame in enumerate(values):
                value = torch.from_numpy(frame.reshape(1, 1, 1920))
                for index, module in enumerate(encoder.model):
                    if index and index - 1 < len(prefix):
                        prior = prefix[index - 1][frame_index]
                        value = torch.from_numpy(prior.T.copy()[None, :, :])
                    value = module(value)
                    rows, channels = _STAGE_SHAPES[index]
                    if tuple(value.shape) != (1, channels, rows):
                        raise ValueError(f"Native encoder stage {index} output shape differs")
                    captured[index].append(_array(value[0].transpose(0, 1), (rows, channels),
                                                  f"encoder stage {index}"))
    layers = [np.stack(stage) for stage in captured]
    return {"output": layers[-1], "layers": layers}
