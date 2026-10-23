"""Checkpoint-codec and validation tests independent of the multi-GB model download."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from opentpu.llm.nanbeige_weights import (FloatSafetensorWeights, NanbeigeWeights, _Shard, dequantize_int8,
                                          load_nanbeige_weights, unpack_int8,
                                          validate_nanbeige_config)


def _config():
    c = json.loads((Path(__file__).parents[1] / "tools/data/nanbeige/target-config.json").read_text())
    c.update(hidden_size=4, intermediate_size=6, num_hidden_layers=2,
             num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=10)
    return c


def _pack(q):
    """Independent scalar packing oracle, including non-multiple-of-four rows."""
    result = np.zeros((q.shape[0], (q.shape[1] + 3) // 4), dtype=np.uint32)
    for row in range(q.shape[0]):
        for col in range(q.shape[1]):
            result[row, col // 4] |= np.uint32((int(q[row, col]) + 128) << (8 * (col % 4)))
    return result.view(np.int32)


def _save(path, values):
    header, payload = {}, bytearray()
    names = {"float32": "F32", "int32": "I32", "int64": "I64", "uint16": "BF16"}
    for name, arr in values.items():
        arr = np.asarray(arr)
        data = arr.tobytes()
        header[name] = {"dtype": names[arr.dtype.name], "shape": list(arr.shape),
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def _tiny_values():
    values = {"model.embed_tokens.weight": np.arange(40, dtype=np.float32).reshape(10, 4),
              "lm_head.weight": np.arange(40, dtype=np.float32).reshape(10, 4) / 8,
              "model.norm.weight": np.ones(4, dtype=np.float32)}
    matrices = {"self_attn.q_proj": (8, 4), "self_attn.k_proj": (4, 4),
                "self_attn.v_proj": (4, 4), "self_attn.o_proj": (4, 8),
                "mlp.gate_proj": (6, 4), "mlp.up_proj": (6, 4), "mlp.down_proj": (4, 6)}
    expected = dict(values)
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            values[prefix + norm + ".weight"] = np.ones(4, dtype=np.float32)
        for name, shape in matrices.items():
            name = prefix + name + ".weight"
            q = (np.arange(np.prod(shape), dtype=np.int32).reshape(shape) - 16).astype(np.int8)
            scale = (np.arange(shape[0], dtype=np.float32) + 1)[:, None] / 8
            values[name + "_packed"] = _pack(q)
            values[name + "_scale"] = scale
            values[name + "_shape"] = np.array(shape, dtype=np.int64)
            expected[name] = q.astype(np.float32) * scale
    return values, expected


def _checkpoint(tmp_path, values=None):
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    if values is None:
        values, _ = _tiny_values()
    _save(tmp_path / "model.safetensors", values)
    return tmp_path


def test_offset_binary_packing_all_int8_values_and_tail():
    q = np.arange(-128, 128, dtype=np.int16).astype(np.int8).reshape(2, 128)
    assert np.array_equal(unpack_int8(_pack(q), 128), q)
    q = np.array([[-128, -1, 0, 127, 37]], dtype=np.int8)
    assert int(_pack(q)[0, 0].view(np.uint32)) == 0xff807f00
    assert np.array_equal(unpack_int8(_pack(q), 5), q)


def test_bf16_decompression_matches_stored_dtype_rounding():
    q = np.array([[-128, -3, 0, 3, 127]], dtype=np.int8)
    scale = np.array([[0.10009765625]], dtype=np.float32)  # BF16 0x3dcd
    got = dequantize_int8(_pack(q), scale, 5, "BF16")
    assert np.array_equal(got, np.array([[-12.8125, -0.30078125, 0, 0.30078125, 12.6875]], np.float32))
    assert not np.array_equal(got, dequantize_int8(_pack(q), scale, 5, "F32"))


def test_real_metadata_contract_is_supported():
    path = Path(__file__).parents[1] / "tools/data/nanbeige/target-config.json"
    validate_nanbeige_config(json.loads(path.read_text()))


def test_lazy_checkpoint_shapes_values_and_row_reads(tmp_path, monkeypatch):
    values, expected = _tiny_values()
    path = _checkpoint(tmp_path, values)
    reads = []
    original = _Shard.read
    def spy(self, name, *args):
        reads.append(name)
        return original(self, name, *args)
    monkeypatch.setattr(_Shard, "read", spy)
    weights = load_nanbeige_weights(path)
    assert all(k.endswith(("_scale", "_shape")) for k in reads)
    assert len(weights) == 21  # 3 global + 9 per physical layer; no loop duplication
    assert "model.layers.0.self_attn.q_proj.weight_packed" not in weights
    for name, array in expected.items():
        assert weights[name].dtype == np.float32
        assert np.array_equal(weights[name], array)
        if array.ndim == 2:
            assert np.array_equal(weights.rows(name, 1, 3), array[1:3])
    with pytest.raises(KeyError):
        weights["missing.weight"]
    with pytest.raises(ValueError, match="row range"):
        weights.rows("lm_head.weight", 0, 11)
    assert "not W8A16-equivalent" in weights.precision["device_conversion"]


def test_plain_bf16_tensor_is_read_without_torch(tmp_path):
    values, _ = _tiny_values()
    values["model.norm.weight"] = np.array([0x3f80, 0x3f00, 0xbf80, 0x0000], dtype=np.uint16)
    weights = NanbeigeWeights(_checkpoint(tmp_path, values))
    assert np.array_equal(weights["model.norm.weight"], np.array([1, .5, -1, 0], np.float32))


@pytest.mark.parametrize("mutation,match", [
    (lambda v: v.pop("lm_head.weight"), "missing checkpoint"),
    (lambda v: v.__setitem__("model.layers.0.self_attn.q_proj.weight_shape", np.array([4, 4], np.int64)), "weight_shape"),
    (lambda v: v.__setitem__("model.layers.0.self_attn.q_proj.weight_packed", np.zeros((4, 1), np.int32)), "expected shape"),
    (lambda v: v.__setitem__("model.layers.0.self_attn.q_proj.weight_scale", np.zeros((8, 1), np.float32)), "strictly positive"),
    (lambda v: v.__setitem__("unrecognized.weight", np.ones(4, np.float32)), "unexpected checkpoint"),
])
def test_reject_bad_checkpoint_contract(tmp_path, mutation, match):
    values, _ = _tiny_values()
    mutation(values)
    with pytest.raises(ValueError, match=match):
        NanbeigeWeights(_checkpoint(tmp_path, values))


@pytest.mark.parametrize("change,match", [
    ({"model_type": "qwen3"}, "model_type"),
    ({"enable_mhc": True}, "enable_mhc"),
    ({"tie_word_embeddings": True}, "untied"),
    ({"hidden_size": 0}, "hidden_size"),
])
def test_reject_unsupported_architecture(change, match):
    with pytest.raises(ValueError, match=match):
        validate_nanbeige_config(dict(_config(), **change))


@pytest.mark.parametrize("change", [{"num_bits": 4}, {"symmetric": False},
                                      {"strategy": "group"}, {"group_size": 128}])
def test_reject_other_quantization_contracts(change):
    c = copy.deepcopy(_config())
    c["quantization_config"]["config_groups"]["group_0"]["weights"].update(change)
    with pytest.raises(ValueError):
        validate_nanbeige_config(c)


def test_nonfinite_plain_values_fail_when_read(tmp_path):
    values, _ = _tiny_values()
    values["lm_head.weight"][0, 0] = np.nan
    weights = NanbeigeWeights(_checkpoint(tmp_path, values))
    with pytest.raises(ValueError, match="NaN"):
        weights["lm_head.weight"]


def test_duplicate_shard_and_incorrect_index_are_rejected(tmp_path):
    _checkpoint(tmp_path)
    (tmp_path / "extra.safetensors").write_bytes((tmp_path / "model.safetensors").read_bytes())
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        NanbeigeWeights(tmp_path)
    (tmp_path / "extra.safetensors").unlink()
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    with pytest.raises(ValueError, match="index tensor names"):
        NanbeigeWeights(tmp_path)


def test_truncated_or_overlapping_safetensors_fail(tmp_path):
    _checkpoint(tmp_path)
    path = tmp_path / "model.safetensors"
    blob = path.read_bytes()
    path.write_bytes(blob[:-1])
    with pytest.raises(ValueError, match="offsets"):
        NanbeigeWeights(tmp_path)
    n = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8:8+n])
    header["lm_head.weight"]["data_offsets"] = header["model.embed_tokens.weight"]["data_offsets"]
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + blob[8+n:])
    with pytest.raises(ValueError, match="overlaps"):
        NanbeigeWeights(tmp_path)


def test_float_draft_reader_preserves_names_bf16_and_integer_mapping(tmp_path):
    _save(tmp_path / "model.safetensors", {
        "markov_head.markov_w1.weight": np.array([[0x3f80, 0xbf80], [0x3f00, 0]], np.uint16),
        "d2t": np.array([0, 123456789], np.int64),
    })
    weights = FloatSafetensorWeights(tmp_path)
    name = "markov_head.markov_w1.weight"
    assert weights.shape(name) == (2, 2)
    assert np.array_equal(weights.rows(name, 0, 1), [[1, -1]])
    assert np.array_equal(weights[name], [[1, -1], [.5, 0]])
    assert weights["d2t"].dtype == np.int64
    assert weights["d2t"][1] == 123456789


def test_float_draft_reader_rejects_packed_target(tmp_path):
    _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="does not decode quantized"):
        FloatSafetensorWeights(tmp_path)
