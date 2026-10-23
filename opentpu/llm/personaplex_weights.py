"""Read the split PersonaPlex MLX checkpoint without requiring MLX/Apple Silicon.

The checkpoint stores PyTorch-style [output, input] projections. MLX affine
quantization packs low-bit-first unsigned codes into uint32 along the last axis;
each group dequantizes as code * scale + bias. These are *quantizer* biases, not
linear-layer biases. Values must be requantized by the openTPU image builder.

Names and shapes follow soniqo/speech-swift's PersonaPlex/WeightLoading.swift
and NVIDIA/personaplex's moshi/models/{lm,loaders}.py and modules/gating.py.
Neither the obsolete model-card size table nor its n_q=8 field describes the
17 input streams: the upstream LM has 16 audio embeddings and one text stream.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np

FLOAT_DTYPES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
DELAYS = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]


def _host_name(filename: str, key: str) -> str:
    """One canonical key map, shared by header validation and actual loading."""
    mapped = "depformer." + key if filename == "depformer.safetensors" and key.startswith("layers.") else key
    return mapped.replace(".self_attn.out_proj_weight", ".self_attn.out_proj.weight")


def dequantize_affine(weight, scales, biases, *, bits: int = 4,
                      group_size: int = 64) -> np.ndarray:
    """Decode MLX affine packed rows exactly in float32, with no transposition."""
    q, s, b = np.asarray(weight), np.asarray(scales), np.asarray(biases)
    if bits not in (4, 8):
        raise ValueError(f"unsupported MLX quantization bits {bits}; expected 4 or 8")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if q.ndim != 2 or q.dtype.kind != "u" or q.dtype.itemsize != 4:
        raise ValueError(f"packed weight must be a uint32 matrix, got {q.dtype} {q.shape}")
    width = q.shape[1] * (32 // bits)
    if width % group_size or s.shape != (q.shape[0], width // group_size) or b.shape != s.shape:
        raise ValueError(f"affine shapes disagree: packed={q.shape}, scales={s.shape}, "
                         f"biases={b.shape}, group_size={group_size}")
    if s.dtype.kind != "f" or b.dtype.kind != "f" or not np.isfinite(s).all() or not np.isfinite(b).all():
        raise ValueError("quantization scales and biases must be finite floating-point arrays")
    result = np.empty((q.shape[0], width), np.float32)
    shifts = np.arange(0, 32, bits, dtype=np.uint32)
    # Bound unpacking temporaries; a single FFN projection is hundreds of MB.
    for start in range(0, q.shape[0], 64):
        end = min(start + 64, q.shape[0])
        codes = ((q[start:end, :, None] >> shifts) & ((1 << bits) - 1)).reshape(end - start, width)
        grouped = codes.astype(np.float32).reshape(end - start, -1, group_size)
        grouped *= s[start:end, :, None].astype(np.float32)
        grouped += b[start:end, :, None].astype(np.float32)
        result[start:end] = grouped.reshape(end - start, width)
    if not np.isfinite(result).all():
        raise ValueError("dequantized weight contains non-finite values")
    return result


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def read_config(model_dir) -> dict:
    """Read and validate the actual split MLX config, not a generic HF LLM config."""
    config = json.loads((Path(model_dir) / "config.json").read_text())
    if config.get("model_type") != "personaplex":
        raise ValueError("expected model_type='personaplex'")
    temporal = config.get("temporal")
    if not isinstance(temporal, dict):
        raise ValueError("missing temporal configuration")
    for field in ("dim", "num_layers", "num_heads", "context", "text_card", "card"):
        _positive_int(temporal.get(field), f"temporal.{field}")
    for field, value in (("n_q", 8), ("card", 2048), ("text_card", 32000)):
        if type(temporal.get(field)) is not int or temporal[field] != value:
            raise ValueError(f"unsupported temporal.{field}; expected {value}")
    dim, heads = temporal["dim"], temporal["num_heads"]
    if dim % heads or (dim // heads) % 2:
        raise ValueError("temporal dimension must divide evenly into even-width attention heads")
    scale = temporal.get("hidden_scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("temporal.hidden_scale must be positive and finite")
    # Exactly NVIDIA ActivationGating's sizing rule, including its special case.
    feedforward = int(scale * dim)
    hidden = 21 * dim // 8 if feedforward == 4 * dim else 2 * feedforward // 3
    if hidden <= 0:
        raise ValueError("temporal feedforward width must be positive")
    period = temporal.get("max_period")
    if isinstance(period, bool) or not isinstance(period, (int, float)) or not np.isfinite(period) or period <= 0:
        raise ValueError("temporal.max_period must be positive and finite")
    for name in ("rms_norm_eps", "rmsNormEps", "eps"):
        if name in temporal and temporal[name] != 1e-8:
            raise ValueError(f"unsupported PersonaPlex RMSNorm epsilon {temporal[name]}")
    quant = config.get("quantization")
    if not isinstance(quant, dict) or type(quant.get("bits")) is not int or quant["bits"] not in (4, 8):
        raise ValueError("expected affine MLX quantization with 4 or 8 bits")
    if type(quant.get("group_size")) is not int or quant["group_size"] != 64:
        raise ValueError("supported PersonaPlex checkpoints use quantization.group_size=64")
    if quant.get("mode", "affine") != "affine":
        raise ValueError("only MLX affine quantization is supported")
    if set(quant) - {"bits", "group_size", "mode", "quantized_components"}:
        raise ValueError("unsupported component-specific quantization configuration")
    components = quant.get("quantized_components")
    if (not isinstance(components, list) or not all(isinstance(c, str) for c in components)
            or len(set(components)) != len(components) or "temporal" not in components
            or set(components) - {"temporal", "depformer"}):
        raise ValueError("quantized_components must contain temporal and optionally depformer")
    depformer = config.get("depformer")
    expected_depformer = {"dim": 1024, "num_layers": 6, "num_heads": 16,
                          "dim_feedforward": 2816, "num_steps": 16, "card": 2048,
                          "text_card": 32000, "context": 8, "weights_per_step": True,
                          "multi_linear": True}
    if not isinstance(depformer, dict):
        raise ValueError("missing depformer configuration")
    for field, expected in expected_depformer.items():
        if type(depformer.get(field)) is not type(expected) or depformer[field] != expected:
            raise ValueError(f"unsupported depformer.{field}; expected {expected!r}")
    if set(depformer) - set(expected_depformer):
        raise ValueError("unsupported depformer configuration fields")
    if config.get("delays") != DELAYS:
        raise ValueError("unsupported PersonaPlex stream delays")
    return config


def spec_from_mlx(model_dir):
    """Construct the temporal FPGA Spec; embeddings and audio streams remain on host."""
    from .personaplex_temporal import Spec
    t = read_config(model_dir)["temporal"]
    ff = int(t["hidden_scale"] * t["dim"])
    hidden = 21 * t["dim"] // 8 if ff == 4 * t["dim"] else 2 * ff // 3
    return Spec(hidden=t["dim"], layers=t["num_layers"], n_heads=t["num_heads"],
                ffn=hidden, eps=1e-8, rope_base=float(t["max_period"]), context=t["context"])


class _TensorFile:
    """Read-only access: NumPy for packed U32, CPU torch for BF16 and floats."""

    def __init__(self, path: Path, bits: int, group_size: int):
        from safetensors import safe_open
        self.path, self.bits, self.group_size = path, bits, group_size
        self.handle = safe_open(str(path), framework="pt", device="cpu")
        self._numpy_handle = None
        self.keys = set(self.handle.keys())

    def shape(self, key: str) -> tuple[int, ...]:
        if key not in self.keys:
            raise ValueError(f"{self.path.name}: missing required tensor {key}")
        return tuple(self.handle.get_slice(key).get_shape())

    def raw(self, key: str, rows: tuple[int, int] | None = None) -> np.ndarray:
        self.shape(key)
        if self.handle.get_slice(key).get_dtype() == "U32":
            # MLX packs unsigned words, including values above INT32_MAX. Use
            # safetensors' NumPy path directly, including row slices, without
            # depending on torch's version-specific unsigned dtype support.
            # Opening this handle is lazy: metadata preflight reads no payload.
            if self._numpy_handle is None:
                from safetensors import safe_open
                self._numpy_handle = safe_open(str(self.path), framework="np")
            return (self._numpy_handle.get_tensor(key) if rows is None else
                    self._numpy_handle.get_slice(key)[rows[0]:rows[1]])
        tensor = (self.handle.get_tensor(key) if rows is None else
                  self.handle.get_slice(key)[rows[0]:rows[1]])
        # NumPy has no built-in BF16, so floating tensors keep the torch path.
        if tensor.is_floating_point():
            tensor = tensor.float()
        return tensor.numpy()

    @staticmethod
    def metadata_keys(key: str) -> tuple[str, str]:
        if key.endswith("_weight"):
            return key[:-7] + "_scales", key[:-7] + "_biases"
        if key.endswith(".weight"):
            return key[:-7] + ".scales", key[:-7] + ".biases"
        return key + ".scales", key + ".biases"

    def is_quantized(self, key: str) -> bool:
        skey, bkey = self.metadata_keys(key)
        if (skey in self.keys) != (bkey in self.keys):
            raise ValueError(f"{self.path.name}: incomplete quantization metadata for {key}")
        return skey in self.keys

    def logical_shape(self, key: str) -> tuple[int, ...]:
        shape = self.shape(key)
        dtype = self.handle.get_slice(key).get_dtype()
        if self.is_quantized(key):
            if len(shape) != 2 or dtype != "U32":
                raise ValueError(f"{key}: packed weight must be a uint32 matrix, got {dtype} {shape}")
            logical = shape[0], shape[1] * (32 // self.bits)
            skey, bkey = self.metadata_keys(key)
            expected = (shape[0], logical[1] // self.group_size)
            if logical[1] % self.group_size or any(self.shape(k) != expected for k in (skey, bkey)):
                raise ValueError(f"{key}: quantization metadata shapes do not match {logical}")
            if any(self.handle.get_slice(k).get_dtype() not in ("F16", "BF16", "F32", "F64")
                   for k in (skey, bkey)):
                raise ValueError(f"{key}: quantization metadata must be floating-point")
            return logical
        if dtype not in ("F16", "BF16", "F32", "F64"):
            raise ValueError(f"{key}: integer matrix has no affine quantization metadata")
        return shape

    def matrix(self, key: str, expected: tuple[int, int] | None = None,
               rows: tuple[int, int] | None = None) -> np.ndarray:
        shape = self.logical_shape(key)
        if len(shape) != 2 or (expected is not None and shape != expected):
            raise ValueError(f"{key}: expected matrix {expected}, got {shape}")
        if rows is not None and not (0 <= rows[0] <= rows[1] <= shape[0]):
            raise ValueError(f"{key}: invalid row range {rows} for {shape}")
        weight = self.raw(key, rows)
        if self.is_quantized(key):
            skey, bkey = self.metadata_keys(key)
            result = dequantize_affine(weight, self.raw(skey, rows), self.raw(bkey, rows),
                                       bits=self.bits, group_size=self.group_size)
        else:
            if weight.dtype.kind != "f":
                raise ValueError(f"{key}: integer matrix has no affine quantization metadata")
            result = weight.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError(f"{key}: non-finite checkpoint values")
        return np.ascontiguousarray(result)

    def norm(self, key: str, dim: int) -> np.ndarray:
        if self.shape(key) not in ((dim,), (1, 1, dim)):
            raise ValueError(f"{key}: expected norm shape ({dim},) or (1,1,{dim})")
        if self.handle.get_slice(key).get_dtype() not in ("F16", "BF16", "F32", "F64"):
            raise ValueError(f"{key}: norm must be floating-point")
        result = np.asarray(self.raw(key), np.float32).reshape(dim)
        if not np.isfinite(result).all():
            raise ValueError(f"{key}: non-finite norm")
        return result

    def metadata(self, key: str) -> dict:
        """Logical shape/type and physical bytes from safetensors headers only."""
        shape = self.logical_shape(key)
        dtype = self.handle.get_slice(key).get_dtype()
        quantized = self.is_quantized(key)
        if key.endswith((".weight", "_weight")) and len(shape) != 2:
            raise ValueError(f"{key}: expected a matrix, got {shape}")
        physical = (key, *self.metadata_keys(key)) if quantized else (key,)
        sizes = {**FLOAT_DTYPES, "U32": 4}
        storage_bytes = sum(math.prod(self.shape(k)) * sizes[self.handle.get_slice(k).get_dtype()]
                            for k in physical)
        return {"shape": shape, "dtype": "F32", "source_dtype": dtype,
                "quantized": quantized, "storage_bytes": storage_bytes,
                "source_file": self.path.name, "source_key": key}


class _Layer(Mapping):
    KEYS = ("norm1", "norm2", "wq", "wk", "wv", "wo", "wg", "wu", "wd")

    def __init__(self, weights: "MLXWeights", index: int):
        self.owner, self.index = weights, index

    def __iter__(self):
        return iter(self.KEYS)

    def __len__(self):
        return len(self.KEYS)

    def __getitem__(self, key):
        if key not in self.KEYS:
            raise KeyError(key)
        f, s = self.owner.temporal, self.owner.spec
        p, h, ff = f"layers.{self.index}.", s.hidden, s.ffn
        if key.startswith("norm"):
            return f.norm(p + key + ".alpha", h)
        if key in ("wq", "wk", "wv"):
            offset = {"wq": 0, "wk": h, "wv": 2 * h}[key]
            return f.matrix(p + "self_attn.in_proj_weight", (3 * h, h), (offset, offset + h))
        if key == "wo":
            return f.matrix(p + "self_attn.out_proj.weight", (h, h))
        if key in ("wg", "wu"):
            offset = 0 if key == "wg" else ff
            return f.matrix(p + "gating.linear_in.weight", (2 * ff, h), (offset, offset + ff))
        return f.matrix(p + "gating.linear_out.weight", (h, ff))


class MLXWeights:
    """Lazy temporal layer dictionaries and a streaming iterator of CPU LM tensors.

    Only matrices requested by an image builder are unpacked. No all-model FP32
    copy is built or cached. q/k retain adjacent-pair ordering: the FPGA runtime
    is responsible for any permutation required by its own RoPE primitive.
    """

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.config = read_config(model_dir)
        self.spec = spec_from_mlx(model_dir)
        q = self.config["quantization"]
        self.temporal = _TensorFile(self.model_dir / "temporal.safetensors", q["bits"], q["group_size"])
        self.layers: Sequence[Mapping] = tuple(_Layer(self, i) for i in range(self.spec.layers))
        # Check all required names and logical dimensions without reading big arrays.
        h, ff = self.spec.hidden, self.spec.ffn
        for i in range(self.spec.layers):
            p = f"layers.{i}."
            for name in ("norm1", "norm2"):
                if self.temporal.logical_shape(p + name + ".alpha") not in ((h,), (1, 1, h)):
                    raise ValueError(f"{p}{name}.alpha: invalid RMSNorm shape")
            for key, shape in (("self_attn.in_proj_weight", (3 * h, h)),
                               ("self_attn.out_proj.weight", (h, h)),
                               ("gating.linear_in.weight", (2 * ff, h)),
                               ("gating.linear_out.weight", (h, ff))):
                if self.temporal.logical_shape(p + key) != shape:
                    raise ValueError(f"{p}{key}: expected shape {shape}, got "
                                     f"{self.temporal.logical_shape(p + key)}")
        if self.temporal.logical_shape("out_norm.alpha") not in ((h,), (1, 1, h)):
            raise ValueError("out_norm.alpha: invalid RMSNorm shape")

    @property
    def final_norm(self):
        return self.temporal.norm("out_norm.alpha", self.spec.hidden)

    def _text_head_reader(self):
        # Open only the embeddings file; a token diagnostic must not materialize
        # the depth transformer or the other CPU model tensors.
        reader = getattr(self, "_text_head_file", None)
        if reader is None:
            q = self.config["quantization"]
            reader = _TensorFile(self.model_dir / "embeddings.safetensors",
                                 q["bits"], q["group_size"])
            self._text_head_file = reader
        return reader

    def text_head_metadata(self):
        """Header-only shape/type check of the untied vocabulary projection."""
        reader = self._text_head_reader()
        metadata = reader.metadata("text_linear.weight")
        expected = (self.config["temporal"]["text_card"], self.spec.hidden)
        if tuple(metadata["shape"]) != expected:
            raise ValueError(f"text_linear.weight: expected matrix {expected}, "
                             f"got {metadata['shape']}")
        return metadata

    @property
    def text_head(self):
        """Read only the finite FP32 text head for offline device-image conversion."""
        self.text_head_metadata()
        return self._text_head_reader().matrix(
            "text_linear.weight", (self.config["temporal"]["text_card"], self.spec.hidden))

    def input_embedding_metadata(self):
        """Header-only input tables in API order: text, then the audio streams."""
        reader = self._text_head_reader()
        temporal = self.config["temporal"]
        names = [("text_emb.weight", temporal["text_card"] + 1)] + [
            (f"emb.{index}.weight", temporal["card"] + 1)
            for index in range(len(self.config["delays"]) - 1)]
        result = []
        for name, rows in names:
            item = reader.metadata(name)
            if tuple(item["shape"]) != (rows, self.spec.hidden):
                raise ValueError(f"{name}: invalid input embedding shape {item['shape']}")
            result.append(item)
        return result

    @property
    def input_embeddings(self):
        """Materialize only input tables for offline FP32 device-image preparation."""
        entries = self.input_embedding_metadata()
        reader = self._text_head_reader()
        return tuple(reader.matrix(item["source_key"], tuple(item["shape"])) for item in entries)

    def audio_head_metadata(self):
        """Header-only codebook head bank, excluding depth-transformer tensors."""
        reader = self._text_head_reader()
        spec = self.config["depformer"]
        result = []
        for index in range(spec["num_steps"]):
            name = f"linears.{index}.weight"
            item = reader.metadata(name)
            if tuple(item["shape"]) != (spec["card"], spec["dim"]):
                raise ValueError(f"{name}: invalid audio head shape {item['shape']}")
            result.append(item)
        return result

    @property
    def audio_heads(self):
        """Materialize only the audio output heads for offline device conversion."""
        reader = self._text_head_reader()
        entries = self.audio_head_metadata()
        return np.stack([reader.matrix(item["source_key"], tuple(item["shape"])) for item in entries])

    def _depth_input_reader(self):
        reader = getattr(self, "_depth_input_file", None)
        if reader is None:
            q = self.config["quantization"]
            reader = _TensorFile(self.model_dir / "depformer.safetensors", q["bits"], q["group_size"])
            self._depth_input_file = reader
        return reader

    def depth_conditioning_metadata(self):
        """Header-only native forward_depformer input projection/embedding banks."""
        projections, embeddings = self._depth_input_reader(), self._text_head_reader()
        dep = self.config["depformer"]
        count = dep["num_steps"] if dep["multi_linear"] else 1
        def entry(reader, name, shape):
            item = reader.metadata(name)
            if tuple(item["shape"]) != shape:
                raise ValueError(f"{name}: invalid depth conditioning shape {item['shape']}")
            if item["quantized"] and "depformer" not in self.config["quantization"]["quantized_components"]:
                raise ValueError(f"{name}: depth quantization disagrees with config")
            return item
        return {"projections": [entry(projections, f"depformer_in.{i}.weight", (dep["dim"], self.spec.hidden))
                                 for i in range(count)],
                "embeddings": [entry(embeddings, "depformer_text_emb.weight", (dep["text_card"] + 1, dep["dim"]))] +
                    [entry(embeddings, f"depformer_emb.{i}.weight", (dep["card"] + 1, dep["dim"]))
                     for i in range(dep["num_steps"] - 1)]}

    @property
    def depth_input_projections(self):
        entries = self.depth_conditioning_metadata()["projections"]
        reader = self._depth_input_reader()
        return np.stack([reader.matrix(item["source_key"], tuple(item["shape"])) for item in entries])

    @property
    def depth_input_embeddings(self):
        entries = self.depth_conditioning_metadata()["embeddings"]
        reader = self._text_head_reader()
        return tuple(reader.matrix(item["source_key"], tuple(item["shape"])) for item in entries)

    def host_checkpoint_fingerprint(self):
        """Invalidate a cached preflight if checkpoint files change on disk."""
        if read_config(self.model_dir) != self.config:
            raise ValueError("Checkpoint config changed; construct a new MLXWeights reader")
        files = ("config.json", "embeddings.safetensors", "depformer.safetensors")
        return tuple((name, (st := (self.model_dir / name).stat()).st_dev,
                      st.st_ino, st.st_size, st.st_mtime_ns) for name in files)

    def _host_entries(self):
        q = self.config["quantization"]
        for filename in ("embeddings.safetensors", "depformer.safetensors"):
            f = _TensorFile(self.model_dir / filename, q["bits"], q["group_size"])
            consumed = set()
            for key in sorted(f.keys):
                if key.endswith((".scales", ".biases", "_scales", "_biases")):
                    continue
                if f.is_quantized(key):
                    if filename != "depformer.safetensors" or "depformer" not in q["quantized_components"]:
                        raise ValueError(f"{filename}:{key}: quantization disagrees with config")
                    consumed.update(f.metadata_keys(key))
                consumed.add(key)
                mapped = _host_name(filename, key)
                metadata = f.metadata(key)
                if mapped.endswith(".alpha") and len(metadata["shape"]) == 1:
                    metadata["shape"] = (1, 1, metadata["shape"][0])
                yield f, key, mapped, metadata
            if consumed != f.keys:
                raise ValueError(f"{filename}: orphan quantization metadata: {sorted(f.keys - consumed)}")

    def iter_host_metadata(self) -> Iterator[tuple[str, dict]]:
        """Yield canonical names and validated metadata without reading tensor data."""
        for _, _, mapped, metadata in self._host_entries():
            yield mapped, metadata

    def iter_host_tensors(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield dequantized non-temporal LM tensors under NVIDIA PyTorch keys.

        This excludes Mimi (use the upstream codec checkpoint) and final_norm,
        which belongs to the temporal adapter. The consumer must check these
        names/shapes against its LMModel and refuse missing/unexpected weights.
        This reader does not silently duplicate depth-step weights.
        """
        for f, key, mapped, metadata in self._host_entries():
            if key.endswith((".weight", "_weight")):
                yield mapped, f.matrix(key)
            else:
                value = np.asarray(f.raw(key), np.float32).reshape(metadata["shape"])
                if not np.isfinite(value).all():
                    raise ValueError(f"{f.path.name}:{key}: non-finite checkpoint values")
                yield mapped, value


def load_weights(model_dir) -> MLXWeights:
    return MLXWeights(model_dir)
