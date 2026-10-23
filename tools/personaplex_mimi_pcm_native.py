"""Pinned original-Mimi SEANet diagnostics for offline PCM qualification.

This module is a CPU reference and checkpoint extractor, never a serving path.
It does not access an FPGA.  The returned arrays retain PyTorch's native
Conv1d [out, in, kernel] and ConvTranspose1d [in, out, kernel] layouts.
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
_DECODER_COUNT = 28
_KWARGS = {
    "channels": 1, "dimension": 512, "causal": True, "n_filters": 64,
    "n_residual_layers": 1, "activation": "ELU", "compress": 2,
    "dilation_base": 2, "disable_norm_outer_blocks": 0, "kernel_size": 7,
    "residual_kernel_size": 3, "last_kernel_size": 3, "norm": "none",
    "pad_mode": "constant", "ratios": [8, 6, 5, 4], "true_skip": True,
}
# (top-level model stage, path to the underlying torch convolution, kind,
#  native PyTorch weight shape).  All fourteen convolutions have a bias.
_CONVOLUTIONS = (
    (0, "conv.conv", "conv", (1024, 512, 7)),
    (2, "convtr.convtr", "transpose", (1024, 512, 16)),
    (3, "block.1.conv.conv", "conv", (256, 512, 3)),
    (3, "block.3.conv.conv", "conv", (512, 256, 1)),
    (5, "convtr.convtr", "transpose", (512, 256, 12)),
    (6, "block.1.conv.conv", "conv", (128, 256, 3)),
    (6, "block.3.conv.conv", "conv", (256, 128, 1)),
    (8, "convtr.convtr", "transpose", (256, 128, 10)),
    (9, "block.1.conv.conv", "conv", (64, 128, 3)),
    (9, "block.3.conv.conv", "conv", (128, 64, 1)),
    (11, "convtr.convtr", "transpose", (128, 64, 8)),
    (12, "block.1.conv.conv", "conv", (32, 64, 3)),
    (12, "block.3.conv.conv", "conv", (64, 32, 1)),
    (14, "conv.conv", "conv", (1, 64, 3)),
)
_STAGE_SHAPES = (
    (2, 1024), (2, 1024), (16, 512), (16, 512), (16, 512),
    (96, 256), (96, 256), (96, 256), (480, 128), (480, 128),
    (480, 128), (1920, 64), (1920, 64), (1920, 64), (1920, 1),
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root(contract, decoder_class):
    root = Path(inspect.getfile(decoder_class)).resolve().parents[1]
    recorded = contract.get("installed_source_sha256")
    if recorded != _SOURCE_SHA256:
        raise ValueError("Native SEANet installed-source contract differs")
    for name, digest in _SOURCE_SHA256.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or _sha256(path) != digest:
            raise ValueError(f"Installed native SEANet source changed: {name}")
    return root


def _expected_tensors():
    expected = {}
    for index, subpath, kind, shape in _CONVOLUTIONS:
        prefix = f"decoder.model.{index}.{subpath}."
        out_channels = shape[1] if kind == "transpose" else shape[0]
        expected[prefix + "weight"] = shape
        expected[prefix + "bias"] = (out_channels,)
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
        raise ValueError("Native SEANet contract lacks cached codec header")
    for name, value in (("file_bytes", _CODEC_BYTES), ("header_bytes", _HEADER_BYTES),
                        ("header_sha256", _HEADER_SHA256), ("tensor_count", _TENSOR_COUNT),
                        ("decoder_count", _DECODER_COUNT)):
        if recorded.get(name) != value:
            raise ValueError(f"Cached codec contract differs: {name}")
    path = Path(codec_path).resolve()
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    if before[0] != _CODEC_BYTES:
        raise ValueError("Cached codec file size differs")
    header = _read_header(path)
    if len([name for name in header if name != "__metadata__"]) != _TENSOR_COUNT:
        raise ValueError("Cached codec tensor count differs")
    expected = _expected_tensors()
    actual = {name: meta for name, meta in header.items() if name.startswith("decoder.")}
    if set(actual) != set(expected) or actual != recorded.get("decoder_tensors"):
        raise ValueError("Cached codec SEANet tensor metadata differs")
    for name, shape in expected.items():
        if actual[name]["dtype"] != "F32" or tuple(actual[name]["shape"]) != shape:
            raise ValueError(f"Cached codec SEANet tensor shape differs: {name}")
    if (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise ValueError("Cached codec changed during header read")
    return path, before, expected


def _array(tensor, shape, name):
    array = tensor.detach().cpu().numpy().copy()
    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Native SEANet value differs: {name}")
    return np.ascontiguousarray(array)


def _submodule(root, path):
    value = root
    for part in path.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return value


def native_pcm_module(codec_path, contract):
    """Strict-load the original CPU SEANet decoder and extract ordered weights.

    Returns ``(decoder, ordered_weights)``.  Each weight record contains
    ``kind``, ``model_index``, ``subpath``, and finite FP32 ``weight``/``bias``.
    Native source, header, tensor names/shapes, and file metadata are pinned.
    """
    from safetensors import safe_open
    from moshi.modules.seanet import SEANetDecoder

    _source_root(contract, SEANetDecoder)
    path, before, selected = _checked_header(codec_path, contract)
    decoder = SEANetDecoder(**_KWARGS)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        state = {name.removeprefix("decoder."): handle.get_tensor(name)
                 for name in selected}
    if (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise ValueError("Cached codec changed during native SEANet load")
    # Reopen after payload extraction: the pinned digest covers the complete
    # safetensors metadata while the stat pair guards changes to its payload.
    _read_header(path)
    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    del state

    ordered_weights = []
    for index, subpath, kind, shape in _CONVOLUTIONS:
        module = _submodule(decoder.model[index], subpath)
        channels = shape[1] if kind == "transpose" else shape[0]
        ordered_weights.append({
            "model_index": index,
            "subpath": subpath,
            "kind": kind,
            "weight": _array(module.weight, shape, f"model.{index}.{subpath}.weight"),
            "bias": _array(module.bias, (channels,), f"model.{index}.{subpath}.bias"),
        })
    decoder._mimi_native_pcm_contract = _HEADER_SHA256
    return decoder, ordered_weights


def collect_native_pcm(decoder, transformer_outputs):
    """Replay consecutive [2,512] frames through fresh native streaming state.

    Returns ``pcm`` as raw, finite FP32 [frames,1920] samples without clipping
    or scaling. ``layers`` is a list indexed by the 15 top-level model stages;
    each item is finite FP32 [frames,time,channels] in source stage order.
    """
    import torch

    if getattr(decoder, "_mimi_native_pcm_contract", None) != _HEADER_SHA256:
        raise ValueError("Use the decoder returned by native_pcm_module")
    values = np.asarray(transformer_outputs)
    if (values.dtype != np.float32 or values.ndim != 3
            or values.shape[1:] != (2, 512) or not 0 < values.shape[0] <= 128
            or not np.isfinite(values).all()):
        raise ValueError("Native PCM input must be finite FP32 [frames,2,512]")
    values = np.ascontiguousarray(values)
    captured = [[] for _ in _STAGE_SHAPES]
    handles = []
    with ExitStack() as stack:
        stack.enter_context(decoder.streaming(1))
        try:
            for index, module in enumerate(decoder.model):
                def hook(_module, _inputs, output, index=index):
                    time, channels = _STAGE_SHAPES[index]
                    if tuple(output.shape) != (1, channels, time):
                        raise ValueError(f"Native SEANet stage {index} shape differs")
                    captured[index].append(_array(output[0].transpose(0, 1),
                                                  (time, channels), f"stage {index}"))
                handles.append(module.register_forward_hook(hook))
            pcm = []
            with torch.no_grad():
                for frame in values:
                    source = torch.from_numpy(frame.T.copy()[None, :, :])
                    output = decoder(source)
                    if tuple(output.shape) != (1, 1, 1920):
                        raise ValueError("Native SEANet did not emit 1920 PCM samples")
                    pcm.append(_array(output[0, 0], (1920,), "native PCM frame"))
        finally:
            for handle in handles:
                handle.remove()
    count = values.shape[0]
    if any(len(stage) != count for stage in captured):
        raise ValueError("Native SEANet stage capture count differs")
    return {"pcm": np.stack(pcm),
            "layers": [np.stack(stage) for stage in captured]}
