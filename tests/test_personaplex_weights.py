"""MLX affine bytes and PersonaPlex projection slicing against known tensors."""
import json

import numpy as np
import pytest

from opentpu.llm.personaplex_weights import MLXWeights, dequantize_affine, read_config, DELAYS, _TensorFile

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")


def _save_file(values, path):
    """Use the public serializer for mixed U32/BF16 fixtures on old and new releases.

    safetensors 0.4.5's torch writer lacks U32 in its Python dtype table even
    though the format and NumPy reader support it. Keep packed words as NumPy,
    and encode BF16 bytes directly instead of changing dependency internals.
    """
    flattened, keep_alive = {}, []
    tensor_spec = getattr(safetensors, "TensorSpec", None)
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            shape, dtype = tuple(value.shape), str(value.dtype).removeprefix("torch.")
            data = value.view(torch.uint8).numpy()
        else:
            value = np.ascontiguousarray(value)
            shape, dtype, data = value.shape, value.dtype.name, value
        keep_alive.append(data)
        if tensor_spec is None:  # Public serializer API in pinned 0.4.5.
            flattened[key] = {"dtype": dtype, "shape": shape, "data": data.tobytes()}
        else:  # 0.8 replaces the descriptor dict with TensorSpec.
            flattened[key] = tensor_spec(dtype=dtype, shape=shape,
                                         data_ptr=data.ctypes.data, data_len=data.nbytes)
    safetensors.serialize_file(flattened, str(path))


def test_known_packed_words_have_low_nibble_first_and_affine_offsets():
    packed = np.array([[0x76543210, 0xFEDCBA98], [0xFFFFFFFF, 0x00000000]], np.uint32)
    scales = np.array([[2, -0.5], [0, 3]], np.float32)
    biases = np.array([[-3, 10], [7, -2]], np.float32)
    result = dequantize_affine(packed, scales, biases, group_size=8)
    expected = np.array([[-3, -1, 1, 3, 5, 7, 9, 11, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5],
                         [7] * 8 + [-2] * 8], np.float32)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(
        dequantize_affine(np.array([[0xFF800100]], np.uint32),
                          np.array([[0.25]], np.float32), np.array([[-1]], np.float32),
                          bits=8, group_size=4), [[-1, -0.75, 31, 62.75]])


@pytest.mark.parametrize("bad", ["signed", "group", "metadata", "nan"])
def test_bad_quantization_is_refused(bad):
    q, s, b = np.zeros((2, 8), np.uint32), np.ones((2, 1), np.float32), np.zeros((2, 1), np.float32)
    if bad == "signed":
        q = q.astype(np.int32)
    elif bad == "group":
        q = q[:, :-1]
    elif bad == "metadata":
        s = np.ones((2, 2), np.float32)
    else:
        b[0, 0] = np.nan
    with pytest.raises(ValueError):
        dequantize_affine(q, s, b)


def test_mixed_uint32_bfloat16_file_preserves_unsigned_words_and_row_slices(tmp_path):
    packed = np.array([[0, 0x7FFFFFFF], [0x80000000, 0xFFFFFFFF],
                       [0x76543210, 0xFEDCBA98]], dtype=np.uint32)
    floating = torch.tensor([[1.25, -2.5], [0.03125, 128], [-16, 0]], dtype=torch.bfloat16)
    path = tmp_path / "mixed.safetensors"
    _save_file({"packed": packed, "floating": floating}, path)
    reader = _TensorFile(path, bits=4, group_size=64)
    assert reader.handle.get_slice("packed").get_dtype() == "U32"
    assert reader.handle.get_slice("floating").get_dtype() == "BF16"
    whole, rows = reader.raw("packed"), reader.raw("packed", (1, 3))
    assert whole.dtype == rows.dtype == np.uint32
    np.testing.assert_array_equal(whole, packed)
    np.testing.assert_array_equal(rows, packed[1:3])
    np.testing.assert_array_equal(reader.raw("floating"), floating.float().numpy())
    np.testing.assert_array_equal(reader.raw("floating", (1, 3)), floating[1:3].float().numpy())


def _quantized(rng, rows, cols):
    codes = rng.integers(0, 16, (rows, cols), dtype=np.uint32)
    packed = np.zeros((rows, cols // 8), np.uint32)
    for nibble in range(8):
        packed |= codes[:, nibble::8] << (4 * nibble)
    scales = rng.normal(0, 0.03, (rows, cols // 64)).astype(np.float32)
    biases = rng.normal(0, 0.02, scales.shape).astype(np.float32)
    reference = codes.astype(np.float32) * scales.repeat(64, axis=1) + biases.repeat(64, axis=1)
    return packed, scales, biases, reference


@pytest.fixture
def checkpoint(tmp_path):
    rng = np.random.default_rng(21)
    # Small but with both input dimensions divisible by the real group size64.
    h, ff = 128, 256
    c = {"model_type": "personaplex", "temporal": {
        "dim": h, "num_heads": 2, "num_layers": 1, "hidden_scale": 3,
        "context": 23, "card": 2048, "text_card": 32000, "n_q": 8, "max_period": 10000},
        "depformer": {"dim": 1024, "num_layers": 6, "num_heads": 16,
                      "dim_feedforward": 2816, "num_steps": 16, "card": 2048,
                      "text_card": 32000, "context": 8, "weights_per_step": True,
                      "multi_linear": True}, "delays": DELAYS,
        "quantization": {"bits": 4, "group_size": 64, "quantized_components": ["temporal", "depformer"]}}
    (tmp_path / "config.json").write_text(json.dumps(c))
    values, reference = {}, {}
    for key, shape in (("self_attn.in_proj_weight", (3 * h, h)),
                       ("self_attn.out_proj.weight", (h, h)),
                       ("gating.linear_in.weight", (2 * ff, h)),
                       ("gating.linear_out.weight", (h, ff))):
        packed, scale, bias, ref = _quantized(rng, *shape)
        name = "layers.0." + key
        values[name] = packed
        ending = "_" if key.endswith("_weight") else "."
        values[name[:-7] + ending + "scales"] = torch.from_numpy(scale)
        values[name[:-7] + ending + "biases"] = torch.from_numpy(bias)
        reference[key] = ref
    for name in ("layers.0.norm1.alpha", "layers.0.norm2.alpha", "out_norm.alpha"):
        values[name] = torch.arange(h, dtype=torch.float32).reshape(1, 1, h) / h + 0.5
    _save_file(values, tmp_path / "temporal.safetensors")
    return tmp_path, reference, values


def test_layer_conversion_preserves_packed_qkv_gate_order_and_no_transpose(checkpoint):
    path, ref, _ = checkpoint
    weights = MLXWeights(path)
    s, layer = weights.spec, weights.layers[0]
    assert (s.hidden, s.layers, s.n_heads, s.ffn, s.eps, s.rope_base, s.context) == (
        128, 1, 2, 256, 1e-8, 10000, 23)
    for index, key in enumerate(("wq", "wk", "wv")):
        np.testing.assert_array_equal(layer[key], ref["self_attn.in_proj_weight"][128 * index:128 * (index + 1)])
    np.testing.assert_array_equal(layer["wo"], ref["self_attn.out_proj.weight"])
    np.testing.assert_array_equal(layer["wg"], ref["gating.linear_in.weight"][:256])
    np.testing.assert_array_equal(layer["wu"], ref["gating.linear_in.weight"][256:])
    np.testing.assert_array_equal(layer["wd"], ref["gating.linear_out.weight"])
    np.testing.assert_array_equal(weights.final_norm, np.arange(128, dtype=np.float32) / 128 + 0.5)
    # Non-square projections and an independent matvec detect accidental transposes.
    x = np.arange(128, dtype=np.float32) / 128
    np.testing.assert_allclose(layer["wu"] @ x, ref["gating.linear_in.weight"][256:] @ x)
    assert set(layer) == {"norm1", "norm2", "wq", "wk", "wv", "wo", "wg", "wu", "wd"}
    with pytest.raises(KeyError):
        layer["q_norm"]


def test_text_head_reader_is_header_only_until_requested_and_skips_other_tensors(
        checkpoint, monkeypatch):
    path, _, _ = checkpoint
    expected = np.linspace(-1, 1, 32000 * 128, dtype=np.float32).reshape(32000, 128).astype(np.float16)
    _save_file({"text_linear.weight": expected,
                "text_emb.weight": np.zeros((2, 128), np.float16)}, path / "embeddings.safetensors")
    weights = MLXWeights(path)
    raw = _TensorFile.raw
    reads = []

    def checked_read(reader, key, rows=None):
        reads.append((reader.path.name, key))
        assert reader.path.name == "embeddings.safetensors" and key == "text_linear.weight"
        return raw(reader, key, rows)

    monkeypatch.setattr(_TensorFile, "raw", checked_read)
    metadata = weights.text_head_metadata()
    assert tuple(metadata["shape"]) == (32000, 128) and metadata["source_dtype"] == "F16"
    assert reads == []
    actual = weights.text_head
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    assert reads == [("embeddings.safetensors", "text_linear.weight")]


@pytest.mark.parametrize("fault", ["missing", "shape", "nonfinite"])
def test_text_head_rejects_invalid_checkpoint_projection(checkpoint, fault):
    path, _, _ = checkpoint
    values = {}
    if fault != "missing":
        head = np.ones((31999 if fault == "shape" else 32000, 128), np.float16)
        if fault == "nonfinite":
            head[3, 4] = np.nan
        values["text_linear.weight"] = head
    _save_file(values, path / "embeddings.safetensors")
    weights = MLXWeights(path)
    with pytest.raises(ValueError):
        _ = weights.text_head


def test_component_readers_preserve_order_and_do_not_materialize_other_components(checkpoint, monkeypatch):
    path, _, _ = checkpoint
    weights = MLXWeights(path)
    # Exercise the reader with small independent shapes, without relaxing the
    # production read_config contract or generating a large test checkpoint.
    weights.config["temporal"].update(text_card=8, card=4)
    weights.config["delays"] = [0, 0, 1, 1]
    weights.config["depformer"].update(num_steps=3, dim=32, card=7)
    values = {"text_emb.weight": np.full((9, 128), 0.25, np.float16)}
    values.update({f"emb.{i}.weight": np.full((5, 128), i + 1, np.float16) for i in range(3)})
    values.update({f"linears.{i}.weight": np.full((7, 32), i + 5, np.float16) for i in range(3)})
    values["text_linear.weight"] = np.zeros((2, 2), np.float16)  # deliberately unrelated
    _save_file(values, path / "embeddings.safetensors")
    original, reads = _TensorFile.raw, []
    def read(reader, key, rows=None):
        reads.append(key)
        return original(reader, key, rows)
    monkeypatch.setattr(_TensorFile, "raw", read)
    assert len(weights.input_embedding_metadata()) == 4
    assert len(weights.audio_head_metadata()) == 3
    assert reads == []
    embeddings = weights.input_embeddings
    assert reads == ["text_emb.weight", "emb.0.weight", "emb.1.weight", "emb.2.weight"]
    heads = weights.audio_heads
    assert reads[4:] == [f"linears.{i}.weight" for i in range(3)]
    for index, table in enumerate(embeddings):
        np.testing.assert_array_equal(table, values[reads[index]].astype(np.float32))
    np.testing.assert_array_equal(heads[:, 0, 0], [5, 6, 7])


@pytest.mark.parametrize("kind", ["input", "audio"])
@pytest.mark.parametrize("fault", ["missing", "shape", "nonfinite"])
def test_component_readers_reject_invalid_selected_payload(checkpoint, kind, fault):
    path, _, _ = checkpoint
    weights = MLXWeights(path)
    weights.config["temporal"].update(text_card=8, card=4)
    weights.config["delays"] = [0, 0]
    weights.config["depformer"].update(num_steps=1, dim=32, card=7)
    values = {"text_emb.weight": np.ones((9, 128), np.float16),
              "emb.0.weight": np.ones((5, 128), np.float16),
              "linears.0.weight": np.ones((7, 32), np.float16)}
    key = "text_emb.weight" if kind == "input" else "linears.0.weight"
    if fault == "missing":
        del values[key]
    elif fault == "shape":
        values[key] = values[key][:-1]
    else:
        values[key][0, 0] = np.nan
    _save_file(values, path / "embeddings.safetensors")
    with pytest.raises(ValueError):
        _ = weights.input_embeddings if kind == "input" else weights.audio_heads


def _depth_reader_fixture(checkpoint):
    path, _, _ = checkpoint
    weights = MLXWeights(path)
    weights.config["depformer"].update(num_steps=3, dim=32, text_card=8, card=4)
    values = {f"depformer_in.{i}.weight": np.full((32, 128), i + 1, np.float16) for i in range(3)}
    values["depformer_text_emb.weight"] = np.full((9, 32), 5, np.float16)
    values.update({f"depformer_emb.{i}.weight": np.full((5, 32), i + 6, np.float16) for i in range(2)})
    return path, weights, values


def _save_depth_fixture(values, path):
    projections = {k: v for k, v in values.items() if k.startswith("depformer_in.")}
    embeddings = {k: v for k, v in values.items() if not k.startswith("depformer_in.")}
    # A same-named decoy makes selecting the wrong checkpoint file observable.
    embeddings["depformer_in.0.weight"] = np.full((32, 128), -99, np.float16)
    _save_file(projections, path / "depformer.safetensors")
    _save_file(embeddings, path / "embeddings.safetensors")


def test_depth_input_readers_validate_headers_and_materialize_only_requested_bank(checkpoint, monkeypatch):
    path, weights, values = _depth_reader_fixture(checkpoint)
    values["linears.0.weight"] = np.ones((1, 1), np.float16)  # unrelated malformed audio head
    _save_depth_fixture(values, path)
    original, reads = _TensorFile.raw, []
    def read(reader, key, rows=None):
        assert reader.path.name == ("depformer.safetensors" if key.startswith("depformer_in.") else
                                    "embeddings.safetensors")
        reads.append(key)
        return original(reader, key, rows)
    monkeypatch.setattr(_TensorFile, "raw", read)
    metadata = weights.depth_conditioning_metadata()
    assert len(metadata["projections"]) == len(metadata["embeddings"]) == 3
    assert reads == []
    projection = weights.depth_input_projections
    assert reads == [f"depformer_in.{i}.weight" for i in range(3)]
    np.testing.assert_array_equal(projection[:, 0, 0], [1, 2, 3])
    embeddings = weights.depth_input_embeddings
    assert reads[3:] == ["depformer_text_emb.weight", "depformer_emb.0.weight", "depformer_emb.1.weight"]
    assert all(e.dtype == np.float32 for e in embeddings)
    np.testing.assert_array_equal([e[0, 0] for e in embeddings], [5, 6, 7])


@pytest.mark.parametrize("kind", ["projection", "embedding"])
@pytest.mark.parametrize("fault", ["missing", "shape", "nonfinite"])
def test_depth_input_reader_rejects_bad_selected_checkpoint(checkpoint, kind, fault):
    path, weights, values = _depth_reader_fixture(checkpoint)
    key = "depformer_in.1.weight" if kind == "projection" else "depformer_emb.1.weight"
    if fault == "missing":
        del values[key]
    elif fault == "shape":
        values[key] = values[key][:-1]
    else:
        values[key][0, 0] = np.nan
    _save_depth_fixture(values, path)
    with pytest.raises(ValueError):
        _ = weights.depth_input_projections if kind == "projection" else weights.depth_input_embeddings


@pytest.mark.parametrize("wrong_config", [False, True])
def test_depth_input_projection_affine_int4_is_read_from_depformer_file(checkpoint, wrong_config):
    path, weights, values = _depth_reader_fixture(checkpoint)
    expected = []
    rng = np.random.default_rng(438)
    for index in range(3):
        q, scale, bias, reference = _quantized(rng, 32, 128)
        values[f"depformer_in.{index}.weight"] = q
        values[f"depformer_in.{index}.scales"] = scale
        values[f"depformer_in.{index}.biases"] = bias
        expected.append(reference)
    _save_depth_fixture(values, path)
    if wrong_config:
        weights.config["quantization"]["quantized_components"] = ["temporal"]
        with pytest.raises(ValueError, match="quantization disagrees"):
            weights.depth_conditioning_metadata()
    else:
        metadata = weights.depth_conditioning_metadata()
        assert all(m["quantized"] and m["source_file"] == "depformer.safetensors"
                   for m in metadata["projections"])
        np.testing.assert_array_equal(weights.depth_input_projections, np.stack(expected))


def _depth_layer_fixture(checkpoint):
    path, _, _ = checkpoint
    owner = MLXWeights(path)
    owner.config["depformer"].update(dim=128, num_steps=3, num_layers=2, dim_feedforward=256)
    rng, values, reference = np.random.default_rng(531), {}, {}
    shapes = {"self_attn.in_proj_weight": (3 * 3 * 128, 128),
              "self_attn.out_proj.weight": (3 * 128, 128)}
    shapes.update({f"gating.{i}." + name: shape for i in range(3) for name, shape in
                   (("linear_in.weight", (512, 128)), ("linear_out.weight", (128, 256)))})
    for key, shape in shapes.items():
        q, scales, biases, expected = _quantized(rng, *shape)
        name = "layers.0." + key
        values[name] = q
        stem = name[:-7] + ("_" if name.endswith("_weight") else ".")
        values[stem + "scales"], values[stem + "biases"] = scales, biases
        reference[key] = expected
    for name in ("norm1", "norm2"):
        values[f"layers.0.{name}.alpha"] = np.linspace(0.8, 1.2, 128, dtype=np.float32).reshape(1, 1, 128)
    return path, owner, values, reference


def test_depth_layer_reader_preserves_contiguous_step_qkv_gate_order_and_shared_norms(checkpoint, monkeypatch):
    path, owner, values, reference = _depth_layer_fixture(checkpoint)
    _save_file(values, path / "depformer.safetensors")
    raw, reads = _TensorFile.raw, []
    def read(reader, key, rows=None):
        reads.append((key, rows))
        assert reader.path.name == "depformer.safetensors" and key.startswith("layers.0.")
        return raw(reader, key, rows)
    monkeypatch.setattr(_TensorFile, "raw", read)
    assert len(owner.depth_layer_metadata(0)) == 10
    layer = owner.depth_layer_weights(0)
    assert reads == [] and set(layer) == {"norm1", "norm2", "steps"}
    np.testing.assert_array_equal(layer["norm1"], values["layers.0.norm1.alpha"].reshape(-1))
    for step in range(3):
        bank = layer["steps"][step]
        assert set(bank) == {"wq", "wk", "wv", "wo", "wg", "wu", "wd"}
        for part, name in enumerate(("wq", "wk", "wv")):
            start = step * 384 + part * 128
            np.testing.assert_array_equal(bank[name], reference["self_attn.in_proj_weight"][start:start + 128])
        np.testing.assert_array_equal(bank["wo"], reference["self_attn.out_proj.weight"][step * 128:(step + 1) * 128])
        np.testing.assert_array_equal(bank["wg"], reference[f"gating.{step}.linear_in.weight"][:256])
        np.testing.assert_array_equal(bank["wu"], reference[f"gating.{step}.linear_in.weight"][256:])
        np.testing.assert_array_equal(bank["wd"], reference[f"gating.{step}.linear_out.weight"])
    with pytest.raises(KeyError):
        _ = layer["steps"][0]["q_norm"]
    with pytest.raises(KeyError):
        _ = layer["final_norm"]
    assert reads


@pytest.mark.parametrize("fault", ["missing", "shape", "metadata", "nonfinite_norm", "nonfinite_matrix"])
def test_depth_layer_reader_refuses_invalid_selected_checkpoint(checkpoint, fault):
    path, owner, values, _ = _depth_layer_fixture(checkpoint)
    key = "layers.0.gating.2.linear_in.weight"
    if fault == "missing":
        del values[key]
    elif fault == "shape":
        values[key] = values[key][:-1]
    elif fault == "metadata":
        del values["layers.0.gating.2.linear_in.scales"]
    elif fault == "nonfinite_norm":
        values["layers.0.norm1.alpha"][0, 0, 4] = np.nan
    else:
        values["layers.0.gating.2.linear_in.biases"][3, 0] = np.inf
    _save_file(values, path / "depformer.safetensors")
    with pytest.raises(ValueError):
        layer = owner.depth_layer_weights(0)
        if fault == "nonfinite_norm":
            _ = layer["norm1"]
        elif fault == "nonfinite_matrix":
            _ = layer["steps"][2]["wg"]


@pytest.mark.parametrize("index", [-1, 2, True, 0.0])
def test_depth_layer_reader_rejects_bad_index(checkpoint, index):
    path, owner, values, _ = _depth_layer_fixture(checkpoint)
    _save_file(values, path / "depformer.safetensors")
    with pytest.raises(ValueError, match="index"):
        owner.depth_layer_weights(index)


@pytest.mark.parametrize("fault", ["missing", "shape", "metadata"])
def test_corrupt_checkpoint_is_refused(checkpoint, fault):
    path, _, values = checkpoint
    key = "layers.0.self_attn.in_proj_weight"
    if fault == "missing":
        del values[key]
    elif fault == "shape":
        values[key] = values[key][:-1]
    else:
        del values["layers.0.self_attn.in_proj_biases"]
    _save_file(values, path / "temporal.safetensors")
    with pytest.raises(ValueError):
        MLXWeights(path)


def test_bfloat_norm_and_unquantized_old_checkpoint_projection(checkpoint):
    path, ref, values = checkpoint
    key = "layers.0.self_attn.in_proj_weight"
    values[key] = torch.from_numpy(ref["self_attn.in_proj_weight"]).to(torch.bfloat16)
    del values["layers.0.self_attn.in_proj_scales"], values["layers.0.self_attn.in_proj_biases"]
    values["out_norm.alpha"] = values["out_norm.alpha"].to(torch.bfloat16)
    _save_file(values, path / "temporal.safetensors")
    w = MLXWeights(path)
    np.testing.assert_array_equal(w.layers[0]["wq"], values[key][:128].float().numpy())
    np.testing.assert_array_equal(w.final_norm, values["out_norm.alpha"].float().numpy().reshape(-1))


def test_host_key_remapping_dequantizes_and_does_not_include_temporal(checkpoint):
    path, _, _ = checkpoint
    rng = np.random.default_rng(45)
    p, s, b, ref = _quantized(rng, 128, 128)
    values = {"layers.0.self_attn.out_proj_weight": p,
              "layers.0.self_attn.out_proj_scales": torch.from_numpy(s),
              "layers.0.self_attn.out_proj_biases": torch.from_numpy(b),
              "layers.0.norm1.alpha": torch.ones(1, 1, 128),
              "depformer_in.0.weight": torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)}
    _save_file(values, path / "depformer.safetensors")
    emb = torch.arange(256, dtype=torch.float32).reshape(2, 128).to(torch.bfloat16)
    _save_file({"text_emb.weight": emb}, path / "embeddings.safetensors")
    host = dict(MLXWeights(path).iter_host_tensors())
    assert set(host) == {"text_emb.weight", "depformer.layers.0.self_attn.out_proj.weight",
                         "depformer.layers.0.norm1.alpha", "depformer_in.0.weight"}
    np.testing.assert_array_equal(host["depformer.layers.0.self_attn.out_proj.weight"], ref)
    np.testing.assert_array_equal(host["text_emb.weight"], emb.float().numpy())


def test_configuration_refuses_wrong_model_or_math(checkpoint):
    path, _, _ = checkpoint
    original = json.loads((path / "config.json").read_text())
    for section, name, value in ((None, "model_type", "llama"),
                                 ("temporal", "num_heads", 3),
                                 ("temporal", "n_q", 16),
                                 ("temporal", "card", 4096),
                                 ("temporal", "text_card", 32001),
                                 ("temporal", "eps", 1e-5),
                                 ("depformer", "dim_feedforward", 4224),
                                 ("depformer", "quantization", {"bits": 8}),
                                 ("quantization", "depformer_bits", 8),
                                 ("quantization", "group_size", 32),
                                 ("quantization", "mode", "mxfp4")):
        c = json.loads(json.dumps(original))
        (c if section is None else c[section])[name] = value
        (path / "config.json").write_text(json.dumps(c))
        with pytest.raises(ValueError):
            read_config(path)


def test_host_metadata_reads_only_headers_and_shares_load_mapping(checkpoint, monkeypatch):
    path, _, _ = checkpoint
    packed, scale, bias, reference = _quantized(np.random.default_rng(66), 128, 128)
    _save_file({"layers.0.self_attn.out_proj_weight": packed,
                          "layers.0.self_attn.out_proj_scales": torch.from_numpy(scale),
                          "layers.0.self_attn.out_proj_biases": torch.from_numpy(bias),
                          "layers.0.norm1.alpha": torch.ones(128)},
                         str(path / "depformer.safetensors"))
    _save_file({"text_emb.weight": torch.ones(2, 128, dtype=torch.bfloat16)},
                         str(path / "embeddings.safetensors"))
    reader = MLXWeights(path)
    real = dict(reader.iter_host_tensors())
    def forbid_data(*args, **kwargs):
        raise AssertionError("header preflight must not read tensor data")
    monkeypatch.setattr(_TensorFile, "raw", forbid_data)
    # Even construction is header-only now; final_norm is lazy.
    reader = MLXWeights(path)
    metadata = dict(reader.iter_host_metadata())
    assert set(metadata) == set(real)
    for name, value in real.items():
        assert metadata[name]["shape"] == value.shape
    assert metadata["depformer.layers.0.norm1.alpha"]["shape"] == (1, 1, 128)
    assert metadata["text_emb.weight"]["source_dtype"] == "BF16"
    q = metadata["depformer.layers.0.self_attn.out_proj.weight"]
    assert q["source_dtype"] == "U32" and q["quantized"]
    assert q["storage_bytes"] == packed.nbytes + scale.nbytes + bias.nbytes
    assert q["shape"] == reference.shape


def test_orphan_metadata_and_invalid_host_dtype_are_rejected_from_headers(checkpoint):
    path, _, _ = checkpoint
    _save_file({"text_emb.weight": torch.ones(2, 128)}, path / "embeddings.safetensors")
    for values in ({"layers.0.norm1.alpha": torch.ones(128, dtype=torch.int64)},
                   {"orphan.scales": torch.ones(2, 2)}):
        _save_file(values, path / "depformer.safetensors")
        with pytest.raises(ValueError):
            list(MLXWeights(path).iter_host_metadata())
