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
