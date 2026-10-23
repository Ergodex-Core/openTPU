"""Pinned original Mimi learned input downsample, for offline CPU checks only.

The helper loads exactly one checkpoint tensor into the original Moshi class.
It neither constructs an input RVQ nor performs hardware or serving work.
"""
import hashlib
import inspect
import json
from pathlib import Path
import struct

import numpy as np


_SOURCE_SHA256 = {
    "models/loaders.py": "24d484efc2198c118f6d2b22877c765cd82e89887c52774c42aac5410b299148",
    "modules/conv.py": "d4f9512cdc1343e959b0222dc2f18632f986ce8862d1cae5293542e0e282c51e",
    "modules/resample.py": "136931781a9de1e43afd19c4ffc9817e462f20b695853d415695d34f912405d9",
    "modules/streaming.py": "9fbdd77237d53ef6d95d1a08bce3b2438d661223ad8b795edfccc1b87eb4b81b",
}
_CODEC_BYTES = 384644900
_HEADER_BYTES = 39320
_HEADER_SHA256 = "ec30eb84e7a32c11b2913db3d89595da90a422320dea3b66a3b39e16f821cc93"
_TENSOR_COUNT = 318
_NAME = "downsample.conv.conv.conv.weight"
_SHAPE = (512, 512, 4)
_OFFSETS = [159660164, 163854468]
_CONSTRUCTOR = dict(stride=2, dimension=512, causal=True,
                    learnt=True, channel_wise=False)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root(contract, native_class):
    if contract.get("installed_source_sha256") != _SOURCE_SHA256:
        raise ValueError("Native input downsample source contract differs")
    root = Path(inspect.getfile(native_class)).resolve().parents[1]
    for name, digest in _SOURCE_SHA256.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or _sha256(path) != digest:
            raise ValueError("Native input downsample source changed: " + name)
    return root


def _checked_header(codec_path, contract):
    expected_meta = {"dtype": "F32", "shape": list(_SHAPE),
                     "data_offsets": _OFFSETS}
    recorded = contract.get("cached_codec_header")
    if not isinstance(recorded, dict):
        raise ValueError("Native input downsample contract lacks cached codec header")
    for name, value in (("file_bytes", _CODEC_BYTES), ("header_bytes", _HEADER_BYTES),
                        ("header_sha256", _HEADER_SHA256), ("tensor_count", _TENSOR_COUNT),
                        ("selected_count", 1), ("selected_bytes", 4 * np.prod(_SHAPE))):
        if recorded.get(name) != value:
            raise ValueError("Cached input downsample contract differs: " + name)
    if recorded.get("selected_tensors") != {_NAME: expected_meta}:
        raise ValueError("Cached input downsample selected tensor contract differs")
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
    selected = {name: meta for name, meta in header.items()
                if name.startswith("downsample.")}
    if selected != {_NAME: expected_meta}:
        raise ValueError("Cached input downsample tensor metadata differs")
    if (_OFFSETS[1] - _OFFSETS[0] != 4 * np.prod(_SHAPE)
            or (path.stat().st_size, path.stat().st_mtime_ns) != before):
        raise ValueError("Cached input downsample payload or file metadata changed")
    return path, before


def native_input_downsample(codec_path, spec, contract):
    """Return strict-loaded source module and its sole FP32 weight tensor."""
    import torch
    from safetensors import safe_open
    from moshi.modules.resample import ConvDownsample1d

    if (spec.hidden != 512 or spec.input_rows != 2 or spec.output_rows != 1
            or contract.get("device_spec") != dict(hidden=512, input_rows=2,
                                                     output_rows=1)
            or contract.get("native_constructor") != _CONSTRUCTOR):
        raise ValueError("Native input downsample requires pinned production dimensions")
    _source_root(contract, ConvDownsample1d)
    path, before = _checked_header(codec_path, contract)
    module = ConvDownsample1d(**_CONSTRUCTOR)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(_NAME)
    if (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise ValueError("Cached codec changed during input downsample load")
    module.load_state_dict({"conv.conv.conv.weight": tensor}, strict=True)
    module.eval()
    weight = tensor.detach().cpu().numpy().copy()
    if weight.dtype != np.float32 or weight.shape != _SHAPE or not np.isfinite(weight).all():
        raise ValueError("Native learned input downsample weight differs")
    module._mimi_native_downsample_contract = _HEADER_SHA256
    return module, np.ascontiguousarray(weight)


def collect_native_input_downsample(module, inputs):
    """Return raw FP32 [frames,1,512] from a fresh source streaming context."""
    import torch

    if getattr(module, "_mimi_native_downsample_contract", None) != _HEADER_SHA256:
        raise ValueError("Use the module returned by native_input_downsample")
    values = np.asarray(inputs)
    if (values.dtype != np.float32 or values.ndim != 3
            or values.shape[1:] != (2, 512) or not 0 < len(values) <= 128
            or not np.isfinite(values).all()):
        raise ValueError("Native input downsample needs finite FP32 [frames,2,512]")
    outputs = []
    with torch.no_grad(), module.conv.streaming(1):
        for frame in values:
            source = torch.from_numpy(frame.T.copy()[None, :, :])
            output = module(source).detach().cpu().numpy()[0].T.copy()
            if output.shape != (1, 512) or output.dtype != np.float32 or not np.isfinite(output).all():
                raise ValueError("Native input downsample output differs")
            outputs.append(output)
    return np.stack(outputs)
