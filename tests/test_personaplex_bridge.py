"""Bridge contracts independent of downloading a speech checkpoint or owning an F2."""
from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from opentpu.llm.personaplex_bridge import (_assign_host_tensors, install_bridge,
                                           _validate_host_metadata, validate_host_checkpoint)


class StreamingModule(torch.nn.Module):
    """Minimal upstream lifecycle: recurse through StreamingModule children."""
    def __init__(self):
        super().__init__()
        self._streaming_state = None

    def _init_streaming_state(self, batch_size):
        return SimpleNamespace(reset=lambda: None)

    def streaming_forever(self, batch_size):
        for m in self.modules():
            if isinstance(m, StreamingModule):
                m._streaming_state = m._init_streaming_state(batch_size)

    def reset_streaming(self):
        for m in self.modules():
            if isinstance(m, StreamingModule):
                m._streaming_state.reset()

    @contextmanager
    def streaming(self, batch_size):
        self.streaming_forever(batch_size)
        try:
            yield
        finally:
            for m in self.modules():
                if isinstance(m, StreamingModule):
                    m._streaming_state = None


class NeverRun(torch.nn.Module):
    def forward(self, x):
        raise AssertionError("Original host temporal computation must never execute")


class HostModel(StreamingModule):
    def __init__(self, device="cpu"):
        super().__init__()
        self.transformer = NeverRun()
        self.out_norm = NeverRun()
        self.text_linear = torch.nn.Linear(4, 7, bias=False, device=device)

    def forward_embeddings(self, x):
        # Actual upstream protocol, including where final RMSNorm is applied.
        result = self.out_norm(self.transformer(x))
        return result, self.text_linear(result)[:, None]


class TemporalEngine:
    def __init__(self):
        self.spec = SimpleNamespace(hidden=4)
        self.pos = 0
        self.calls = []
        self.reset_count = 0

    def reset(self):
        self.pos = 0
        self.reset_count += 1

    def step(self, value):
        self.calls.append(value.copy())
        out = value + np.float32(self.pos)
        self.pos += 1
        return out


@pytest.fixture
def moshi_streaming(monkeypatch):
    module = ModuleType("moshi.modules.streaming")
    module.StreamingModule = StreamingModule
    monkeypatch.setitem(sys.modules, "moshi.modules.streaming", module)


def test_prompt_chunk_and_single_frames_have_same_protocol_and_order(moshi_streaming):
    engine = TemporalEngine()
    host = HostModel()
    bridge = install_bridge(host, engine)
    embeddings = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    with host.streaming(1):
        hidden, logits = host.forward_embeddings(embeddings)
        expected = embeddings + torch.arange(3)[:, None]
        torch.testing.assert_close(hidden, expected)
        torch.testing.assert_close(logits, host.text_linear(expected)[:, None])
        assert logits.shape == (1, 1, 3, 7)
        assert bridge.stats.frames == 3
        assert bridge.stats.temporal_seconds >= 0
        host.reset_streaming()
        parts = [host.forward_embeddings(embeddings[:, i:i+1])[0] for i in range(3)]
        torch.testing.assert_close(torch.cat(parts, dim=1), expected)
        assert engine.reset_count == 2


def test_parent_lmgen_reset_after_warmup_resets_fpga_kv(moshi_streaming):
    engine = TemporalEngine()
    host = HostModel()
    bridge = install_bridge(host, engine)
    lmgen = StreamingModule()
    lmgen.lm_model = host
    lmgen.streaming_forever(1)
    x = torch.ones(1, 1, 4)
    first = host.forward_embeddings(x)[0]
    host.forward_embeddings(x)
    assert engine.pos == 2
    lmgen.reset_streaming()
    assert engine.pos == 0
    assert bridge.stats.frames == 0
    torch.testing.assert_close(host.forward_embeddings(x)[0], first)


def test_saved_bfloat16_voice_prompt_matches_float32_host_head(moshi_streaming):
    host = HostModel()
    engine = TemporalEngine()
    install_bridge(host, engine)
    with host.streaming(1):
        hidden, logits = host.forward_embeddings(torch.ones(1, 1, 4, dtype=torch.bfloat16))
    assert hidden.dtype == logits.dtype == torch.float32
    assert engine.calls[0].dtype == np.float32


def test_bfloat16_host_loader_uses_assigned_head_dtype(moshi_streaming, monkeypatch):
    """Test loader ordering with real torch tensors, not a genuine Moshi runtime."""
    import opentpu.llm.personaplex_bridge as bridge

    # Upstream's text_linear constructor omits dtype: its meta weight starts
    # FP32 even when the surrounding LMModel was requested in BF16.
    host = HostModel(device="meta")
    assert host.text_linear.weight.dtype == torch.float32
    monkeypatch.setattr(bridge, "_meta_host_model", lambda spec, dtype: host)
    monkeypatch.setattr(bridge, "validate_host_checkpoint", lambda weights, spec: {})
    values = np.arange(28, dtype=np.float32).reshape(7, 4) / 16
    weights = SimpleNamespace(iter_host_tensors=lambda: iter([("text_linear.weight", values)]))
    engine = TemporalEngine()

    loaded = bridge.load_host_model(weights, engine, dtype=torch.bfloat16)
    assert loaded.text_linear.weight.dtype == torch.bfloat16
    with loaded.streaming(1):
        hidden, logits = loaded.forward_embeddings(torch.ones(1, 1, 4))
    assert hidden.dtype == logits.dtype == torch.bfloat16
    torch.testing.assert_close(logits, loaded.text_linear(hidden)[:, None])


@pytest.mark.parametrize("shape", [(2, 1, 4), (1, 4), (1, 1, 3)])
def test_bad_shapes_do_not_advance_state(moshi_streaming, shape):
    engine = TemporalEngine()
    host = HostModel()
    install_bridge(host, engine)
    with host.streaming(1):
        with pytest.raises(ValueError, match="Expected embeddings"):
            host.forward_embeddings(torch.zeros(shape))
    assert engine.pos == 0


def test_streaming_required_and_batch_one(moshi_streaming):
    host = HostModel()
    engine = TemporalEngine()
    install_bridge(host, engine)
    with pytest.raises(RuntimeError, match="Start LMGen"):
        host.forward_embeddings(torch.zeros(1, 1, 4))
    with pytest.raises(ValueError, match="one conversation"):
        host.streaming_forever(2)


def test_nonfinite_rejected_and_empty_frame_preserves_state(moshi_streaming):
    host = HostModel()
    engine = TemporalEngine()
    install_bridge(host, engine)
    with host.streaming(1):
        empty, logits = host.forward_embeddings(torch.empty(1, 0, 4))
        assert empty.shape == (1, 0, 4) and logits.shape == (1, 1, 0, 7)
        with pytest.raises(ValueError, match="non-finite"):
            host.forward_embeddings(torch.full((1, 1, 4), float("nan")))
    assert engine.pos == 0


def test_install_refuses_live_stream(moshi_streaming):
    host = HostModel()
    host.streaming_forever(1)
    with pytest.raises(ValueError, match="before starting"):
        install_bridge(host, TemporalEngine())


@pytest.mark.parametrize("overflow", [1e100, -1e100])
def test_float64_prompt_overflow_is_rejected_before_consuming_any_frame(moshi_streaming, overflow):
    host = HostModel()
    engine = TemporalEngine()
    bridge = install_bridge(host, engine)
    with host.streaming(1):
        host.forward_embeddings(torch.ones(1, 1, 4))
        seconds = bridge.stats.temporal_seconds
        embeddings = torch.ones(1, 2, 4, dtype=torch.float64)
        embeddings[0, 1, 0] = overflow
        assert bool(torch.isfinite(embeddings).all())
        with pytest.raises(ValueError, match="finite in float32"):
            host.forward_embeddings(embeddings)
        assert engine.pos == len(engine.calls) == bridge.stats.frames == 1
        assert bridge.stats.temporal_seconds == seconds
        # The rejected chunk left the next position and conversation usable.
        hidden, _ = host.forward_embeddings(torch.ones(1, 1, 4, dtype=torch.float64))
        torch.testing.assert_close(hidden, torch.full((1, 1, 4), 2.0))
        assert engine.pos == bridge.stats.frames == 2


def test_prompt_that_exceeds_context_is_rejected_before_any_frame(moshi_streaming):
    host = HostModel()
    engine = TemporalEngine()
    engine.cap = 2
    install_bridge(host, engine)
    with host.streaming(1):
        with pytest.raises(RuntimeError, match="exceeds temporal context"):
            host.forward_embeddings(torch.ones(1, 3, 4))
    assert engine.calls == []


def test_meta_host_assignment_requires_every_parameter(moshi_streaming):
    host = HostModel(device="meta")
    install_bridge(host, TemporalEngine())
    with pytest.raises(ValueError, match="Missing host checkpoint"):
        _assign_host_tensors(host, [], torch.float32)
    weights = np.arange(28, dtype=np.float32).reshape(7, 4)
    _assign_host_tensors(host, [("text_linear.weight", weights)], torch.float32)
    assert host.text_linear.weight.device.type == "cpu"
    torch.testing.assert_close(host.text_linear.weight, torch.from_numpy(weights))


@pytest.mark.parametrize("name,value,match", [
    ("transformer.layers.0.weight", np.zeros((1,)), "unexpectedly includes temporal"),
    ("unexpected.weight", np.zeros((1,)), "Unexpected host"),
    ("text_linear.weight", np.zeros((4, 7)), "shape mismatch"),
    ("text_linear.weight", np.full((7, 4), np.nan), "Non-finite"),
])
def test_host_assignment_rejects_invalid_tensors(moshi_streaming, name, value, match):
    host = HostModel(device="meta")
    install_bridge(host, TemporalEngine())
    with pytest.raises(ValueError, match=match):
        _assign_host_tensors(host, [(name, value)], torch.float32)


def test_host_assignment_matches_upstream_eight_to_sixteen_depth_expansion():
    host = torch.nn.Module()
    host.depformer = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.in_proj_weight = torch.nn.Parameter(torch.empty(96, 2, device="meta"))
    host.depformer.layers = torch.nn.ModuleList([layer])
    host.linears = torch.nn.ModuleList([
        torch.nn.Linear(2, 2, bias=False, device="meta") for _ in range(16)])
    qkv = np.arange(96, dtype=np.float32).reshape(48, 2)
    weights = [("depformer.layers.0.self_attn.in_proj_weight", qkv)]
    weights += [(f"linears.{i}.weight", np.full((2, 2), i, np.float32)) for i in range(8)]
    _assign_host_tensors(host, weights, torch.float32)
    torch.testing.assert_close(layer.self_attn.in_proj_weight,
                               torch.from_numpy(np.concatenate((qkv, qkv))))
    for i in range(16):
        torch.testing.assert_close(host.linears[i].weight, torch.full((2, 2), float(i % 8)))


def _header(shape, *, source_dtype="F32", quantized=False):
    return {"shape": shape, "dtype": "F32", "source_dtype": source_dtype,
            "quantized": quantized, "storage_bytes": int(np.prod(shape)) * 4}


def test_header_only_schema_checks_exact_expansion_and_missing_parameters():
    expected = {"depformer.layers.0.self_attn.in_proj_weight": torch.empty(96, 2, device="meta"),
                "depformer.layers.0.self_attn.out_proj.weight": torch.empty(32, 2, device="meta"),
                "text_emb.weight": torch.empty(5, 2, device="meta")}
    expected.update({f"linears.{i}.weight": torch.empty(2, 2, device="meta") for i in range(16)})
    entries = [("depformer.layers.0.self_attn.in_proj_weight", _header((48, 2))),
               ("depformer.layers.0.self_attn.out_proj.weight", _header((16, 2))),
               ("text_emb.weight", _header((5, 2)))]
    entries += [(f"linears.{i}.weight", _header((2, 2))) for i in range(8)]
    report = _validate_host_metadata(expected, iter(entries))
    assert report["header_only"]
    assert report["source_tensor_count"] == 11 and report["loaded_tensor_count"] == 19
    assert report["expanded_attention_count"] == 2 and report["copied_tensor_count"] == 8
    assert report["host_fp32_bytes"] == (96 * 2 + 32 * 2 + 5 * 2 + 16 * 2 * 2) * 4
    assert all(t.device.type == "meta" for t in expected.values())
    with pytest.raises(ValueError, match="Missing host"):
        _validate_host_metadata(expected, entries[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        _validate_host_metadata(expected, entries + entries[:1])
    with pytest.raises(ValueError, match="shape mismatch"):
        _validate_host_metadata(expected, [(entries[0][0], _header((47, 2)))] + entries[1:])


@pytest.mark.parametrize("name,metadata,message", [
    ("other.weight", _header((5, 2)), "Unexpected"),
    ("text_emb.weight", _header((5, 2), source_dtype="I32"), "dtype"),
    ("text_emb.weight", _header((5, 2), source_dtype="U32"), "quantization"),
    ("text_emb.weight", _header((2, 5)), "shape mismatch"),
])
def test_invalid_headers_fail_before_tensor_materialization(name, metadata, message):
    expected = {"text_emb.weight": torch.empty(5, 2, device="meta")}
    with pytest.raises(ValueError, match=message):
        _validate_host_metadata(expected, [(name, metadata)])


def test_preflight_cache_invalidates_when_files_change(monkeypatch):
    import opentpu.llm.personaplex_bridge as bridge
    models = ModuleType("moshi.models")
    models.loaders = SimpleNamespace(LMModel=object, _lm_kwargs={})
    monkeypatch.setitem(sys.modules, "moshi.models", models)
    built = []
    def build(spec, dtype):
        built.append(spec)
        return torch.nn.Linear(2, 3, bias=False, device="meta")
    monkeypatch.setattr(bridge, "_meta_host_model", build)
    class HeaderReader:
        spec, version = "spec", 1
        def host_checkpoint_fingerprint(self):
            return self.version
        def iter_host_metadata(self):
            return iter([("weight", _header((3, 2)))])
        def iter_host_tensors(self):
            raise AssertionError("preflight must not request tensor data")
    reader = HeaderReader()
    first = validate_host_checkpoint(reader, "spec")
    assert validate_host_checkpoint(reader, "spec") == first and len(built) == 1
    reader.version += 1
    assert validate_host_checkpoint(reader, "spec") == first and len(built) == 2
    with pytest.raises(ValueError, match="specification"):
        validate_host_checkpoint(reader, "different spec")


def test_genuine_installed_moshi_schema_is_meta_only():
    # Optional dependency test: no downloads, checkpoint weights, or GPU required.
    pytest.importorskip("moshi.models.loaders", exc_type=ImportError,
                       reason="NVIDIA PersonaPlex runtime or its native dependencies are unavailable")
    from opentpu.llm.personaplex_bridge import _meta_host_model
    from opentpu.llm.personaplex_temporal import Spec
    model = _meta_host_model(Spec(), torch.float32)
    expected = model.state_dict()
    assert expected and all(t.device.type == "meta" for t in expected.values())
    assert not any(n.startswith(("transformer.", "out_norm.")) for n in expected)
    assert expected["depformer.layers.0.self_attn.in_proj_weight"].shape == (16 * 3 * 1024, 1024)
    entries = [(name, _header(tuple(value.shape))) for name, value in expected.items()]
    report = _validate_host_metadata(expected, entries)
    assert report["copied_tensor_count"] == report["expanded_attention_count"] == 0
