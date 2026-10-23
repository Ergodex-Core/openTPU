"""Bounded offline original MimiModel.encode qualification against native components.

For a bounded offline EC2 diagnostic only. Call with the modules returned by the
existing strict-loading native helpers and their native-chain reference arrays.
There is no CLI, download, hardware path, output-codec reference or serving path.

Source-only reviewed against the full pinned compression.py, streaming.py and
utils/compile.py exports. Original CPU Mimi streaming constructs CUDAGraphed
with disable=True, so its original __call__ delegates to the transformer with
no warmup/capture. This helper observes that path without substituting direct
transformer calls or repairing native ring positions. Its qualification claims
apply only after the guarded EC2 CPU call completes successfully.
"""

import hashlib
import inspect
from pathlib import Path


_COMPRESSION_SHA256 = "7af4b2791d16fd0add4ef4f51d4322fc9fec159cc96604e569bbb53fe756b4ed"
_STREAMING_SHA256 = "9fbdd77237d53ef6d95d1a08bce3b2438d661223ad8b795edfccc1b87eb4b81b"
_COMPILE_SHA256 = "cdf0bb28eff09caefcf91d0180faac350bc2b6b55e09f1419ad41cdb533db618"
_ORCHESTRATION_SOURCES = {
    "models/compression.py": _COMPRESSION_SHA256,
    "modules/streaming.py": _STREAMING_SHA256,
    "utils/compile.py": _COMPILE_SHA256,
}
_HEADER_SHA256 = "ec30eb84e7a32c11b2913db3d89595da90a422320dea3b66a3b39e16f821cc93"
_RVQ_CONTRACT_SHA256 = "5342c087a41a95fd899bf62ac92c2c2ad31e8fa09d60788eda74214df5dd4aa1"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _exact(actual, expected, label):
    if (actual.dtype != expected.dtype or actual.shape != expected.shape
            or actual.tobytes() != expected.tobytes()):
        raise AssertionError(label + ": raw bytes differ")


def _checked_transformer_spec(spec):
    if (spec.hidden, spec.n_heads, spec.layers, spec.context) != (512, 8, 8, 250):
        raise ValueError("Require the pinned original encoder transformer geometry")


def _pinned_modules(encoder, transformer, downsample, quantizer, mimi_class):
    if (getattr(encoder, "_mimi_native_encoder_contract", None) != _HEADER_SHA256
            or getattr(downsample, "_mimi_native_downsample_contract", None) != _HEADER_SHA256
            or getattr(quantizer, "_mimi_input_rvq_contract", None) != _RVQ_CONTRACT_SHA256
            or getattr(transformer, "_mimi_native_input_spec", None) is None):
        raise ValueError("Supply modules returned by the existing pinned native helpers")
    roots = {Path(inspect.getfile(type(module))).resolve().parents[1]
             for module in (encoder, transformer, downsample, quantizer)}
    roots.add(Path(inspect.getfile(mimi_class)).resolve().parents[1])
    if len(roots) != 1:
        raise ValueError("Native input modules come from different Moshi trees")
    root = roots.pop()
    for name, digest in _ORCHESTRATION_SOURCES.items():
        if _sha256(root / name) != digest:
            raise ValueError("Original full-input orchestration source changed: " + name)
    for owner in (encoder, transformer, downsample, quantizer):
        if any(getattr(module, "_streaming_state", None) is not None
               for module in owner.modules()):
            raise ValueError("Native input modules must be outside a streaming context")
        if any(value.device.type != "cpu"
               for value in [*owner.parameters(), *owner.buffers()]):
            raise ValueError("This diagnostic requires existing CPU native modules")
    if (encoder.dimension != 512 or encoder.hop_length != 960
            or quantizer.num_codebooks != 8):
        raise ValueError("Require original mono 24kHz/25Hz encoder and eight input codebooks")
    _checked_transformer_spec(transformer._mimi_native_input_spec)
    return root


def _collect_in_stream(model, frames):
    """Observe original streamed encode and cache.complete; return all results unchanged."""
    import numpy as np
    import torch
    from moshi.utils.compile import CUDAGraphed

    recorded = {name: [] for name in ("encoder", "transformer", "latents", "ids")}
    positions = [[] for _ in range(8)]
    handles, originals = [], []
    state = model._streaming_state
    graph = getattr(state, "graphed_tr_enc", None)
    if (type(graph) is not CUDAGraphed or graph.disable is not True
            or graph.func is not model.encoder_transformer
            or graph._graph is not None or graph._args is not None
            or graph._output is not None
            or getattr(state, "graphed_tr_dec", None) is not None):
        raise ValueError("Original Mimi streaming did not construct its disabled CPU encoder graph")
    graph_file = Path(inspect.getfile(type(graph))).resolve()
    if _sha256(graph_file) != _COMPILE_SHA256:
        raise ValueError("Original Mimi CPU graph-wrapper source changed")
    graph_metadata = dict(class_name=type(graph).__qualname__,
                          module=type(graph).__module__,
                          source_path=str(graph_file), source_sha256=_COMPILE_SHA256,
                          disabled_on_cpu=True, warmup_or_capture_used=False)

    def hook(name, rows, tuple_output=False):
        def capture(_module, _args, output):
            value = output[0] if tuple_output else output
            if value.dtype != torch.float32 or tuple(value.shape) != (1, 512, rows):
                raise AssertionError("Full Mimi native boundary differs: " + name)
            array = value.detach().cpu().numpy()[0].T.copy()
            if not np.isfinite(array).all():
                raise AssertionError("Full Mimi native boundary is nonfinite: " + name)
            recorded[name].append(array)
        return capture

    try:
        handles.append(model.encoder.register_forward_hook(hook("encoder", 2)))
        handles.append(model.encoder_transformer.register_forward_hook(
            hook("transformer", 2, tuple_output=True)))
        handles.append(model.downsample.register_forward_hook(hook("latents", 1)))
        layers = model.encoder_transformer.transformer.layers
        if len(layers) != 8:
            raise ValueError("Original full-input encoder transformer layer count differs")
        for index, layer in enumerate(layers):
            cache = layer.self_attn._streaming_state.kv_cache
            original = cache.complete
            def complete(k, v, original=original, index=index):
                result = original(k, v)
                positions[index].append(result.positions.detach().cpu().numpy().copy())
                return result
            cache.complete = complete
            originals.append((cache, original))
        with torch.no_grad():
            for frame in frames:
                source = torch.from_numpy(frame.reshape(1, 1, 1920).copy())
                codes = model.encode(source)
                if codes.dtype != torch.int64 or tuple(codes.shape) != (1, 8, 1):
                    raise AssertionError("Original full Mimi encode did not emit eight IDs")
                recorded["ids"].append(codes.detach().cpu().numpy()[0, :, 0].copy())
    finally:
        for handle in handles:
            handle.remove()
        for cache, original in originals:
            cache.complete = original
    if (any(len(value) != len(frames) for value in recorded.values())
            or any(len(value) != len(frames) for value in positions)):
        raise AssertionError("Original full Mimi hooks/cache captures did not run once per frame")
    result = {name: np.stack(value) for name, value in recorded.items()}
    result["positions"] = np.stack([np.stack(value) for value in positions])
    result["graph_wrapper"] = graph_metadata
    return result


def check_native_full_input(encoder, transformer, downsample, quantizer,
                            frames, component_reference):
    """Require bit-exact full-entrypoint/component-chain boundaries and IDs.

    ``frames`` is the preserved raw FP32 [F,1920], 1 <= F <= 128 fixture.
    ``component_reference`` contains native arrays named encoder/transformer
    [F,2,512], latents [F,1,512], ids int64 [F,8], and actual native positions
    int64 [8,F,250]. It must come from the native component chain, not ISA
    outputs. The ISA accumulated/same-input ID qualification remains separate.

    Original fresh-stream and reset_streaming replays are checked independently.
    No encode_to_latent call is interleaved with encode (which would advance the
    encoder/transformer/downsample stream twice for one PCM frame).
    """
    import numpy as np
    import torch
    from moshi.models.compression import MimiModel

    values = np.asarray(frames)
    if (values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 1920
            or not 0 < len(values) <= 128 or not np.isfinite(values).all()):
        raise ValueError("Require finite raw preserved FP32 [1..128,1920] PCM")
    shapes = dict(encoder=(len(values), 2, 512), transformer=(len(values), 2, 512),
                  latents=(len(values), 1, 512), ids=(len(values), 8),
                  positions=(8, len(values), 250))
    expected = {}
    for name, shape in shapes.items():
        array = np.asarray(component_reference[name])
        dtype = np.int64 if name in ("ids", "positions") else np.float32
        if array.dtype != dtype or array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Native component reference shape/dtype differs: " + name)
        expected[name] = np.ascontiguousarray(array)
    native_root = _pinned_modules(encoder, transformer, downsample, quantizer, MimiModel)
    # MimiModel constructs fresh downsample/upsample modules. Strictly load its
    # input resampler AFTER construction; leaving constructor weights is invalid.
    model = MimiModel(encoder, torch.nn.Identity(), quantizer,
        channels=1, sample_rate=24000, frame_rate=12.5,
        encoder_frame_rate=24000 / encoder.hop_length, causal=True,
        resample_method="conv", encoder_transformer=transformer,
        decoder_transformer=None, torch_compile_encoder_decoder=False).cpu().eval()
    source_state = downsample.state_dict()
    model.downsample.load_state_dict(source_state, strict=True)
    for name, value in model.downsample.state_dict().items():
        if not torch.equal(value, source_state[name]):
            raise AssertionError("MimiModel constructed downsample did not strict-load: " + name)
    # Identity decoder and random constructor upsampler are never called and
    # provide no output-codec qualification. Record this scope explicitly.
    with model.streaming(1):
        first = _collect_in_stream(model, values)
        model.reset_streaming()
        reset = _collect_in_stream(model, values)
    with model.streaming(1):
        fresh = _collect_in_stream(model, values)
    for name in shapes:
        _exact(first[name], expected[name], "full Mimi versus native component " + name)
        _exact(reset[name], first[name], "full Mimi logical reset " + name)
        _exact(fresh[name], first[name], "full Mimi fresh stream " + name)
    for name, digest in _ORCHESTRATION_SOURCES.items():
        if _sha256(native_root / name) != digest:
            raise RuntimeError("Original full-input source changed during comparison: " + name)
    first["qualification"] = dict(
        scope="Original MimiModel.encode input orchestration only; offline CPU reference",
        frames=len(values), component_boundaries_bit_exact=True,
        input_ids_bit_exact=True, native_positions_bit_exact=True,
        logical_reset_bit_exact=True, fresh_stream_bit_exact=True,
        decoder_is_identity=True, output_codec_qualified=False,
        constructor_upsample_unused=True, native_ring_behavior_changed=False,
        hardware_used=False, serving_fallback=False,
        full_speech_qualified=False, checkpoint_payload_hashes_verified=False,
        native_orchestration_sha256=dict(_ORCHESTRATION_SOURCES))
    return first
