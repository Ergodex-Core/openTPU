"""CPU Moshi orchestration with PersonaPlex's temporal transformer on openTPU.

The native NVIDIA PersonaPlex LMGen retains token delays, voice conditioning,
sampling, and the depth transformer. Its temporal transformer is replaced by a
StreamingModule that invokes ``personaplex_temporal.Engine``. The engine includes
the final RMSNorm, so the host's out_norm becomes Identity. The original temporal
transformer is removed; there is no CPU fallback.

Install before LMGen starts streaming. Moshi's recursive streaming lifecycle then
resets the device KV state on both warmup and new conversations. A bounded device
context raises rather than silently resetting partway through a conversation.
Mimi remains the upstream CPU codec and must use its original codec checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time

import numpy as np


@dataclass
class BridgeStats:
    frames: int = 0
    temporal_seconds: float = 0.0
    resets: int = 0

    @property
    def temporal_ms_per_frame(self) -> float:
        return self.temporal_seconds * 1000 / self.frames if self.frames else 0.0


class _EngineState:
    def __init__(self, engine, stats: BridgeStats):
        self.engine = engine
        self.stats = stats
        self.reset()

    def reset(self):
        self.engine.reset()
        self.stats.frames = 0
        self.stats.temporal_seconds = 0.0
        self.stats.resets += 1


def _make_transformer(engine, streaming_type, output_dtype):
    import torch

    class FPGATemporalTransformer(streaming_type):
        def __init__(self):
            super().__init__()
            self.engine = engine
            self.stats = BridgeStats()

        def _init_streaming_state(self, batch_size: int):
            if batch_size != 1:
                raise ValueError("PersonaPlex FPGA bridge supports one conversation at a time")
            return _EngineState(self.engine, self.stats)

        @torch.no_grad()
        def forward(self, embeddings):
            if self._streaming_state is None:
                raise RuntimeError("Start LMGen streaming before FPGA temporal inference")
            if embeddings.device.type != "cpu":
                raise ValueError("PersonaPlex FPGA host components must run on CPU")
            hidden = self.engine.spec.hidden
            if embeddings.ndim != 3 or embeddings.shape[0] != 1 or embeddings.shape[2] != hidden:
                raise ValueError(f"Expected embeddings [1, T, {hidden}], got {tuple(embeddings.shape)}")
            if not embeddings.is_floating_point():
                raise ValueError("PersonaPlex embeddings must be floating point")
            if not bool(torch.isfinite(embeddings).all()):
                raise ValueError("PersonaPlex embeddings contain non-finite values")
            frames = embeddings.shape[1]
            if frames == 0:
                return embeddings.to(dtype=output_dtype).clone()
            # Do not consume a partial voice/text prompt and only then discover
            # its tail exceeds the bounded device context.
            if hasattr(self.engine, "cap") and self.engine.pos + frames > self.engine.cap:
                raise RuntimeError(f"Prompt/frame chunk exceeds temporal context "
                                   f"({self.engine.pos}+{frames}>{self.engine.cap}); reset required")
            values = embeddings.detach().float().numpy()
            # Finite FP64 values can overflow when converted to the engine's
            # FP32 inputs. Validate the whole converted chunk before consuming
            # any prompt frames or updating persistent KV state.
            if not np.isfinite(values).all():
                raise ValueError("PersonaPlex embeddings must remain finite in float32")
            output = np.empty((1, frames, hidden), dtype=np.float32)
            for index in range(frames):
                start = time.perf_counter()
                result = np.asarray(self.engine.step(values[0, index]), dtype=np.float32)
                if result.shape != (hidden,) or not np.isfinite(result).all():
                    raise RuntimeError("FPGA temporal engine returned invalid hidden activations")
                output[0, index] = result
                self.stats.temporal_seconds += time.perf_counter() - start
                self.stats.frames += 1
            # Saved voice-prompt embeddings can be BF16 even when the CPU model
            # is FP32; text_linear/depformer must receive their weight dtype.
            return torch.from_numpy(output).to(dtype=output_dtype)

    return FPGATemporalTransformer()


def install_bridge(lm, engine):
    """Replace temporal computation and return the installed streaming adapter.

    Must be called before constructing/starting LMGen, because LMGen caches the
    forward callables. ``lm.forward_embeddings`` remains upstream's implementation.
    """
    import torch
    from moshi.modules.streaming import StreamingModule

    if getattr(lm, "_streaming_state", None) is not None:
        raise ValueError("Install the FPGA bridge before starting Moshi streaming")
    for attr in ("transformer", "out_norm", "text_linear", "forward_embeddings"):
        if not hasattr(lm, attr):
            raise TypeError(f"Unsupported PersonaPlex LMModel: missing {attr}")
    if lm.text_linear.in_features != engine.spec.hidden:
        raise ValueError("Host text head and FPGA temporal hidden sizes differ")
    if any(p.device.type not in ("cpu", "meta") for p in lm.parameters()):
        raise ValueError("PersonaPlex FPGA host components must run on CPU")
    bridge = _make_transformer(engine, StreamingModule, lm.text_linear.weight.dtype)
    lm.transformer = bridge
    lm.out_norm = torch.nn.Identity()
    return bridge


def _validate_upstream_config(kwargs, spec):
    """Reject incompatible Moshi installs rather than loading them permissively."""
    expected = {
        "dim": 4096, "text_card": 32000, "existing_text_padding_id": 3,
        "n_q": 16, "card": 2048, "num_heads": 32, "num_layers": 32,
        "hidden_scale": 4.125, "context": 3000, "max_period": 10000,
        "gating": "silu", "norm": "rms_norm_f32", "positional_embedding": "rope",
        "depformer_dim": 1024, "depformer_num_heads": 16,
        "depformer_dim_feedforward": 4224,
        "depformer_num_layers": 6, "depformer_multi_linear": True,
        "depformer_weights_per_step": True, "depformer_pos_emb": "none",
        "depformer_causal": True, "depformer_gating": "silu",
        "depformer_layer_scale": None, "layer_scale": None, "causal": True,
        "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    }
    mismatches = [key for key, value in expected.items() if kwargs.get(key) != value]
    if mismatches:
        raise ValueError("Unsupported PersonaPlex upstream config: " + ", ".join(mismatches))
    for name, value in (("hidden", 4096), ("layers", 32), ("n_heads", 32), ("ffn", 11264),
                        ("eps", 1e-8), ("rope_base", 10000), ("context", 3000)):
        if getattr(spec, name) != value:
            raise ValueError(f"Host checkpoint requires temporal {name}={value}")


def _meta_host_model(spec, dtype):
    """Use the installed NVIDIA model's exact parameter schema without storage."""
    import torch
    try:
        from moshi.models import loaders
    except ImportError as exc:
        raise RuntimeError("Install NVIDIA/personaplex's moshi package for host checkpoint preflight") from exc
    kwargs = dict(loaders._lm_kwargs)
    _validate_upstream_config(kwargs, spec)
    kwargs["dep_q"] = 16
    with torch.device("meta"):
        lm = loaders.LMModel(device="meta", dtype=dtype, **kwargs)
        lm.transformer = torch.nn.Identity()
        lm.out_norm = torch.nn.Identity()
    if any(t.device.type != "meta" for t in lm.state_dict().values()):
        raise RuntimeError("Installed PersonaPlex created non-meta tensors during schema validation")
    return lm


def _attention_expansion(name, shape, target):
    match = re.fullmatch(r"depformer\.layers\.\d+\.self_attn\.(in_proj_weight|out_proj\.weight)", name)
    if match is None or len(shape) != 2 or len(target) != 2:
        return False
    factor = 3 if match[1] == "in_proj_weight" else 1
    # Exactly 8 sets -> 16 sets; do not allow arbitrary half-height tensors.
    return (target[0] == 16 * factor * target[1] and shape[1] == target[1]
            and shape[0] == 8 * factor * target[1])


def _copied_step_source(name):
    patterns = (r"(linears|depformer_in|depformer_emb)\.(\d+)\.(weight)",
                r"(depformer\.layers\.\d+\.gating)\.(\d+)\.(linear_(?:in|out)\.weight)")
    for pattern in patterns:
        match = re.fullmatch(pattern, name)
        if match and 8 <= int(match[2]) < 16:
            return f"{match[1]}.{int(match[2]) - 8}.{match[3]}"
    return None


def _validate_host_metadata(expected, entries):
    """Validate header records against a meta state_dict; never access array data."""
    seen, expanded, copies = {}, set(), {}
    storage_bytes = 0
    for name, metadata in entries:
        if name not in expected:
            raise ValueError(f"Unexpected host checkpoint tensor: {name}")
        if name in seen:
            raise ValueError(f"Duplicate host checkpoint tensor: {name}")
        shape = tuple(metadata["shape"])
        target = tuple(expected[name].shape)
        if metadata["dtype"] != "F32" or metadata["source_dtype"] not in ("F16", "BF16", "F32", "F64", "U32"):
            raise ValueError(f"Unsupported host checkpoint dtype for {name}")
        if (metadata["source_dtype"] == "U32") != bool(metadata["quantized"]):
            raise ValueError(f"Invalid host checkpoint quantization metadata for {name}")
        if shape != target:
            if _attention_expansion(name, shape, target):
                expanded.add(name)
            else:
                raise ValueError(f"Host checkpoint shape mismatch for {name}: {shape} != {target}")
        seen[name] = target
        storage_bytes += metadata["storage_bytes"]
    for name, value in expected.items():
        if name in seen:
            continue
        source = _copied_step_source(name)
        if source in seen and seen[source] == tuple(value.shape):
            copies[name] = source
    missing = sorted(set(expected) - set(seen) - set(copies))
    if missing:
        raise ValueError("Missing host checkpoint tensors: " + ", ".join(missing[:12]))
    parameters = sum(math.prod(t.shape) for t in expected.values())
    return {"header_only": True, "source_tensor_count": len(seen),
            "loaded_tensor_count": len(expected), "copied_tensor_count": len(copies),
            "expanded_attention_count": len(expanded), "source_storage_bytes": storage_bytes,
            "host_parameter_count": parameters, "host_fp32_bytes": parameters * 4}


def validate_host_checkpoint(weights, spec):
    """Header-only validation against genuine installed NVIDIA PersonaPlex.

    Run before opening a board or constructing its DRAM image. A successful
    result validates names/shapes/types, not data values or model quality. The
    same weights object's cached result is reused while source file metadata,
    model spec, and the installed upstream model/config remain unchanged.
    """
    import torch
    try:
        from moshi.models import loaders
    except ImportError as exc:
        raise RuntimeError("Install NVIDIA/personaplex's moshi package for host checkpoint preflight") from exc
    if weights.spec != spec:
        raise ValueError("Checkpoint temporal specification does not match the engine")
    fingerprint = (weights.host_checkpoint_fingerprint(), repr(spec), id(loaders.LMModel),
                   json.dumps(loaders._lm_kwargs, sort_keys=True))
    cached = getattr(weights, "_host_preflight", None)
    if cached is not None and cached[0] == fingerprint:
        return dict(cached[1])
    lm = _meta_host_model(spec, torch.float32)
    report = _validate_host_metadata(lm.state_dict(), weights.iter_host_metadata())
    weights._host_preflight = (fingerprint, dict(report))
    return report


def _assign_host_tensors(lm, tensors, dtype):
    """Strictly assign dequantized host tensors without allocating temporal weights."""
    import torch

    expected = lm.state_dict()
    state = {}
    for name, value in tensors:
        if name.startswith("transformer.") or name.startswith("out_norm."):
            raise ValueError(f"Host tensor iterator unexpectedly includes temporal tensor {name}")
        if name not in expected:
            raise ValueError(f"Unexpected host checkpoint tensor: {name}")
        if name in state:
            raise ValueError(f"Duplicate host checkpoint tensor: {name}")
        arr = np.asarray(value)
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite host checkpoint tensor: {name}")
        target = expected[name]
        if tuple(arr.shape) != tuple(target.shape):
            # NVIDIA's loader extends an 8-step pretrained depformer to 16 steps
            # by repeating its packed attention weights; no other shape coercion.
            if _attention_expansion(name, arr.shape, tuple(target.shape)):
                arr = np.concatenate((arr, arr), axis=0)
            else:
                raise ValueError(f"Host checkpoint shape mismatch for {name}: "
                                 f"{arr.shape} != {tuple(target.shape)}")
        state[name] = torch.from_numpy(np.array(arr, copy=True)).to(dtype=dtype)

    # The matching upstream loader copies depth projections/gates/embeddings from
    # steps 0..7 to 8..15 when a checkpoint contains only the first eight steps.
    for name in expected:
        if name in state:
            continue
        source = _copied_step_source(name)
        if source in state and state[source].shape == expected[name].shape:
            state[name] = state[source].clone()
    missing = sorted(set(expected) - set(state))
    if missing:
        raise ValueError("Missing host checkpoint tensors: " + ", ".join(missing[:12]))
    lm.load_state_dict(state, strict=True, assign=True)
    remaining = [name for name, p in lm.named_parameters() if p.device.type == "meta"]
    if remaining:
        raise RuntimeError("Uninitialized host parameters remain: " + ", ".join(remaining[:12]))


def load_host_model(model_dir_or_weights, engine, *, dtype=None):
    """Load CPU host parts of the requested MLX checkpoint, returning an LMModel.

    Requires NVIDIA's PersonaPlex version of ``moshi``. No original full LM
    checkpoint is required. ``model_dir_or_weights`` may be a model directory or
    ``personaplex_weights.MLXWeights``; its iterator excludes temporal parameters.
    Use float32 on the CPU by default. Dequantized CPU depth weights consume more
    host RAM than their stored MLX INT4 representation.
    """
    import torch
    from .personaplex_weights import MLXWeights

    if dtype is None:
        dtype = torch.float32
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("CPU PersonaPlex host dtype must be float32 or bfloat16")
    weights = (MLXWeights(model_dir_or_weights)
               if isinstance(model_dir_or_weights, (str, Path)) else model_dir_or_weights)
    validate_host_checkpoint(weights, engine.spec)
    lm = _meta_host_model(engine.spec, dtype)
    _assign_host_tensors(lm, weights.iter_host_tensors(), dtype)
    # Upstream creates text_linear in the default dtype, even when dtype is
    # provided for the transformer. Capture its actual loaded dtype only now.
    install_bridge(lm, engine)
    lm.eval()
    return lm
