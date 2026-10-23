"""Lazy import of the Nanbeige 4.2 compressed-tensors INT8 checkpoint.

This is a weight importer, not an implementation of published W8A16 inference.
The source uses symmetric per-output-channel INT8 weights, offset-packed in
INT32, with BF16 embeddings, norms and LM head. We reproduce compressed-tensors'
decompression (including rounding to the scale dtype), and expose those values
as float32 arrays. The openTPU image builder subsequently requantizes them into
its MXU format and quantizes activations to INT8. That second conversion needs
an independent quality check; it is not a lossless GPTQ repack.

Only metadata and small scales are inspected on construction. Large tensors
are memory-mapped and converted on demand, one at a time; ``rows`` permits
bounded-size export without materializing a full embedding or LM head.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np


_DTYPES = {"I32": "<i4", "I64": "<i8", "F32": "<f4", "F16": "<f2", "BF16": "<u2"}
_FLOATS = {"F32", "F16", "BF16"}
_PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _integer(value, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: expected an integer >= {minimum}, got {value!r}")
    return value


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def validate_nanbeige_config(config: Mapping) -> None:
    """Reject formats and architecture extensions this target importer cannot represent."""
    if config.get("model_type") != "nanbeige":
        raise ValueError("expected model_type='nanbeige' (DSpark is a separate draft model)")
    if config.get("architectures", ["NanbeigeForCausalLM"]) != ["NanbeigeForCausalLM"]:
        raise ValueError("unsupported Nanbeige architecture")
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size"):
        _integer(config.get(key), key, 1)
    _integer(config.get("num_loops", 1), "num_loops", 1)
    if config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if config["head_dim"] % 2:
        raise ValueError("head_dim must be even for RoPE")
    if config.get("tie_word_embeddings", False):
        raise ValueError("this Nanbeige importer requires the untied LM head")
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError("only Nanbeige SiLU is supported")
    for key in ("attention_bias", "mlp_bias", "qk_norm", "enable_mhc", "enable_hc",
                "enable_depth_attention", "enable_double_loop_split", "loop_share_kv",
                "enable_ngram", "use_ngram", "use_sliding_window"):
        if config.get(key):
            raise ValueError(f"unsupported Nanbeige option: {key}")
    q = config.get("quantization_config")
    if not isinstance(q, dict):
        raise ValueError("expected compressed-tensors INT8 quantization_config")
    if q.get("quant_method") != "compressed-tensors" or q.get("format") != "pack-quantized":
        raise ValueError("only compressed-tensors pack-quantized INT8 is supported")
    if q.get("quantization_status") != "compressed":
        raise ValueError("expected quantization_status='compressed'")
    if q.get("ignore") != ["lm_head"]:
        raise ValueError("expected only lm_head in quantization ignore list")
    for key in ("kv_cache_scheme", "sparsity_config", "transform_config"):
        if q.get(key):
            raise ValueError(f"unsupported quantization option: {key}")
    groups = q.get("config_groups")
    if not isinstance(groups, dict) or len(groups) != 1:
        raise ValueError("expected one Linear weight quantization group")
    group = next(iter(groups.values()))
    if group.get("format", "pack-quantized") != "pack-quantized" or group.get("targets") != ["Linear"]:
        raise ValueError("expected pack-quantized Linear group")
    if group.get("input_activations") is not None or group.get("output_activations") is not None:
        raise ValueError("source must be weight-only W8A16, with no activation quantization")
    weight = group.get("weights", {})
    expected = {"num_bits": 8, "strategy": "channel", "symmetric": True, "type": "int"}
    if any(weight.get(k) != v for k, v in expected.items()):
        raise ValueError("expected symmetric channelwise integer INT8 weights")
    for key in ("actorder", "block_structure", "group_size", "dynamic"):
        if weight.get(key) not in (None, False):
            raise ValueError(f"unsupported weight quantization option: {key}")


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: checkpoint contains NaN or infinity")
    return value


def _round_bf16(value: np.ndarray) -> np.ndarray:
    """Round finite float32 values to BF16, ties to even; return float32 values."""
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def unpack_int8(packed: np.ndarray, columns: int) -> np.ndarray:
    """Decode compressed-tensors offset-binary bytes, not two's-complement bytes."""
    packed = np.asarray(packed)
    _integer(columns, "columns", 1)
    if packed.ndim != 2 or packed.dtype.kind != "i" or packed.dtype.itemsize != 4:
        raise ValueError("weight_packed must be a rank-2 INT32 tensor")
    if packed.shape[1] != (columns + 3) // 4:
        raise ValueError("weight_packed columns disagree with weight_shape")
    # Signed shifts are harmless after the byte mask, including INT32's sign bit.
    out = np.empty((packed.shape[0], packed.shape[1] * 4), dtype=np.int8)
    for lane in range(4):
        out[:, lane::4] = (((packed >> (8 * lane)) & 255) - 128).astype(np.int8)
    return out[:, :columns]


def dequantize_int8(packed: np.ndarray, scale: np.ndarray, columns: int,
                   scale_dtype: str = "F32") -> np.ndarray:
    """Match compressed-tensors dequantize's default output dtype=scale.dtype."""
    scale = np.asarray(scale, dtype=np.float32)
    if scale.shape != (packed.shape[0], 1):
        raise ValueError("weight_scale must have shape [output_channels, 1]")
    _finite(scale, "weight_scale")
    if np.any(scale <= 0):
        raise ValueError("weight_scale must be strictly positive")
    if scale_dtype not in _FLOATS:
        raise ValueError(f"unsupported scale dtype: {scale_dtype}")
    out = unpack_int8(packed, columns).astype(np.float32) * scale
    if scale_dtype == "BF16":
        out = _round_bf16(out)
    elif scale_dtype == "F16":
        with np.errstate(over="ignore"):
            out = out.astype(np.float16).astype(np.float32)
    return _finite(out, "dequantized weight")


class _Shard:
    """Minimal strictly validated safetensors reader for this checkpoint's five dtypes.

    Direct numeric mmap avoids importing torch merely to convert BF16 to numpy.
    No Python/pickle payload or model remote code is executed.
    """
    def __init__(self, path: Path):
        self.path = path
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw_len = stream.read(8)
            if len(raw_len) != 8:
                raise ValueError(f"{path}: truncated safetensors header")
            length = int.from_bytes(raw_len, "little")
            if not 2 <= length <= min(100_000_000, size - 8):
                raise ValueError(f"{path}: invalid safetensors header length")
            header = json.loads(stream.read(length), object_pairs_hook=_unique_object)
        if not isinstance(header, dict):
            raise ValueError(f"{path}: safetensors header must be an object")
        self.offset = 8 + length
        self.tensors = {}
        ranges = []
        for name, info in header.items():
            if name == "__metadata__":
                if not isinstance(info, dict) or not all(isinstance(v, str) for v in info.values()):
                    raise ValueError(f"{path}: invalid safetensors metadata")
                continue
            if not isinstance(info, dict) or info.get("dtype") not in _DTYPES:
                raise ValueError(f"{name}: unsupported safetensors dtype")
            shape = info.get("shape")
            if not isinstance(shape, list):
                raise ValueError(f"{name}: invalid shape")
            for v in shape:
                _integer(v, f"{name} shape")
            offsets = info.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"{name}: invalid data_offsets")
            a, b = (_integer(v, f"{name} offset") for v in offsets)
            nbytes = math.prod(shape) * np.dtype(_DTYPES[info["dtype"]]).itemsize
            if b - a != nbytes or b > size - self.offset:
                raise ValueError(f"{name}: data offsets disagree with tensor size")
            self.tensors[name] = (info["dtype"], tuple(shape), a)
            ranges.append((a, b))
        end = 0
        for a, b in sorted(ranges):
            if a != end:
                raise ValueError(f"{path}: tensor data has gaps or overlaps")
            end = b
        if end != size - self.offset:
            raise ValueError(f"{path}: unreferenced or truncated tensor data")

    def read(self, name: str, start: int | None = None, stop: int | None = None) -> np.ndarray:
        dtype, shape, offset = self.tensors[name]
        raw = np.memmap(self.path, dtype=_DTYPES[dtype], mode="r", offset=self.offset + offset,
                        shape=shape)
        if start is not None:
            raw = raw[start:stop]
        if dtype == "BF16":
            return (raw.astype(np.uint32) << 16).view(np.float32)
        if dtype in _FLOATS:
            return np.asarray(raw, dtype=np.float32)
        return raw


class NanbeigeWeights(Mapping):
    """Validated, lazy float32 view of a Nanbeige compressed-tensors checkpoint.

    ``precision`` records both the source and conversion for benchmark manifests.
    No large converted tensor is cached. Callers choose their own tensor lifetime.
    """
    precision = {
        "source": "compressed-tensors/symmetric-channel-W8A16",
        "decompression": "offset-int8-times-scale-rounded-to-scale-dtype",
        "view_dtype": "float32",
        "device_conversion": "openTPU MXU requantization; INT8 activations; not W8A16-equivalent",
    }

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text(),
                                 object_pairs_hook=_unique_object)
        validate_nanbeige_config(self.config)
        self._where = {}
        files = sorted(self.model_dir.glob("*.safetensors"))
        if not files:
            raise ValueError(f"{model_dir}: no safetensors checkpoint found")
        for path in files:
            shard = _Shard(path)
            for name in shard.tensors:
                if name in self._where:
                    raise ValueError(f"duplicate checkpoint tensor: {name}")
                self._where[name] = shard
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(), object_pairs_hook=_unique_object)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or set(weight_map) != set(self._where):
                raise ValueError("safetensors index tensor names disagree with checkpoint files")
            for name, filename in weight_map.items():
                if filename != self._where[name].path.name:
                    raise ValueError(f"{name}: safetensors index points to the wrong shard")
        c = self.config
        h, f, v = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
        q, k = c["num_attention_heads"] * c["head_dim"], c["num_key_value_heads"] * c["head_dim"]
        self._shapes = {"model.embed_tokens.weight": (v, h), "lm_head.weight": (v, h),
                        "model.norm.weight": (h,)}
        self._packed = set()
        for layer in range(c["num_hidden_layers"]):
            prefix = f"model.layers.{layer}."
            for name, shape in zip(_PROJECTIONS, ((q, h), (k, h), (k, h), (h, q),
                                                   (f, h), (f, h), (h, f))):
                key = prefix + name + ".weight"
                self._shapes[key] = shape
                self._packed.add(key)
            for norm in ("input_layernorm", "post_attention_layernorm"):
                self._shapes[prefix + norm + ".weight"] = (h,)
        required = set()
        for name, shape in self._shapes.items():
            if name not in self._packed:
                required.add(name)
                self._check(name, shape, _FLOATS)
                continue
            rows, cols = shape
            for suffix, expected_shape, allowed in (("_packed", (rows, (cols + 3) // 4), {"I32"}),
                                                    ("_scale", (rows, 1), _FLOATS),
                                                    ("_shape", (2,), {"I32", "I64"})):
                key = name + suffix
                required.add(key)
                self._check(key, expected_shape, allowed)
            if tuple(self._where[name + "_shape"].read(name + "_shape")) != shape:
                raise ValueError(f"{name}: weight_shape disagrees with model config")
            scale = self._where[name + "_scale"].read(name + "_scale")
            if not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError(f"{name}: weight_scale must be finite and strictly positive")
        unexpected = set(self._where) - required
        if unexpected:
            raise ValueError(f"unexpected checkpoint tensor(s): {sorted(unexpected)[:5]}")

    def _check(self, name: str, shape: tuple, dtypes: set) -> None:
        if name not in self._where:
            raise ValueError(f"missing checkpoint tensor: {name}")
        dtype, actual, _ = self._where[name].tensors[name]
        if actual != shape or dtype not in dtypes:
            raise ValueError(f"{name}: expected shape {shape}, dtype {sorted(dtypes)}; "
                             f"got {actual}, {dtype}")

    def rows(self, name: str, start: int, stop: int) -> np.ndarray:
        shape = self._shapes[name]
        if len(shape) != 2:
            raise ValueError("rows() requires a matrix tensor")
        _integer(start, "start")
        _integer(stop, "stop")
        if not start < stop <= shape[0]:
            raise ValueError("row range must satisfy 0 <= start < stop <= output rows")
        return self._read(name, start, stop)

    def _read(self, name: str, start=None, stop=None) -> np.ndarray:
        shape = self._shapes[name]
        if name in self._packed:
            pk, sk = name + "_packed", name + "_scale"
            packed = self._where[pk].read(pk, start, stop)
            scale = self._where[sk].read(sk, start, stop)
            dtype = self._where[sk].tensors[sk][0]
            return dequantize_int8(packed, scale, shape[1], dtype)
        return _finite(self._where[name].read(name, start, stop), name)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._read(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._shapes)

    def __len__(self) -> int:
        return len(self._shapes)

    def __contains__(self, name) -> bool:
        return name in self._shapes


class FloatSafetensorWeights(Mapping):
    """Lazy unquantized tensors under their exact checkpoint names (e.g. DSpark).

    This validates storage, dtypes and finite values; the draft runtime must
    validate its own model-specific names and shapes. Integer token mappings
    are exposed without float conversion so vocabulary IDs remain exact.
    """
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self._where = {}
        for path in sorted(self.model_dir.glob("*.safetensors")):
            shard = _Shard(path)
            for name, (dtype, _, _) in shard.tensors.items():
                if name in self._where:
                    raise ValueError(f"duplicate checkpoint tensor: {name}")
                if name.endswith(("weight_packed", "weight_scale", "weight_zero_point")):
                    raise ValueError("FloatSafetensorWeights does not decode quantized checkpoints")
                self._where[name] = shard
        if not self._where:
            raise ValueError(f"{model_dir}: no safetensors tensors found")

    def shape(self, name: str) -> tuple:
        return self._where[name].tensors[name][1]

    def rows(self, name: str, start: int, stop: int) -> np.ndarray:
        shape = self.shape(name)
        _integer(start, "start")
        _integer(stop, "stop")
        if not shape or not start < stop <= shape[0]:
            raise ValueError("row range must satisfy 0 <= start < stop <= output rows")
        return _finite(self._where[name].read(name, start, stop), name)

    def __getitem__(self, name: str) -> np.ndarray:
        return _finite(self._where[name].read(name), name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._where)

    def __len__(self) -> int:
        return len(self._where)

    def __contains__(self, name) -> bool:
        return name in self._where


def load_nanbeige_weights(model_dir) -> NanbeigeWeights:
    return NanbeigeWeights(model_dir)


def iter_nanbeige_weights(model_dir) -> Iterator[tuple[str, np.ndarray]]:
    weights = load_nanbeige_weights(model_dir)
    for name in weights:
        yield name, weights[name]
