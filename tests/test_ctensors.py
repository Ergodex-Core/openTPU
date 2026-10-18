"""compressed-tensors pack-quantized checkpoints (opentpu/llm/ctensors.py): the packing
convention is the library's, and a checkpoint's packed tensors read as "X.weight" through
qwen3's Weights and LazyWeights, dequantized."""
import json
import math

import numpy as np
import pytest
import torch

from opentpu.llm import ctensors as CT
from opentpu.llm.qwen3 import LazyWeights, Weights, load_weights


def pack_to_int32(value: torch.Tensor, num_bits: int, packed_dim: int = 1) -> torch.Tensor:
    """compressed_tensors/compressors/pack_quantized/helpers.py (v0.19.0), verbatim but for
    the checks: the library's packing, the reference for ctensors.unpack."""
    offset = 1 << (num_bits - 1)
    value = value.to(torch.int32) + offset
    device = value.device
    if packed_dim == 0:
        value = value.transpose(0, 1)
    rows, cols = value.shape
    packed_cols = math.ceil(cols * num_bits / 32)
    padded_cols = math.ceil(cols / 32) * 32
    if padded_cols > cols:
        value = torch.nn.functional.pad(value, (0, padded_cols - cols))
    num_groups = padded_cols // 32
    rows_g = rows * num_groups
    value_g = value.reshape(rows_g, 32)
    output_g = torch.zeros(rows_g, num_bits, dtype=torch.int32, device=device)
    elem_i = torch.arange(32, device=device, dtype=torch.int32)
    bit_starts = elem_i * num_bits
    word_idx = (bit_starts // 32).long()
    bit_offset = bit_starts % 32
    output_g.scatter_add_(1, word_idx.unsqueeze(0).expand(rows_g, -1),
                          value_g << bit_offset.unsqueeze(0))
    ov = bit_offset + num_bits - 32
    ov_mask = ov > 0
    if ov_mask.any():
        ov_vals = value_g[:, ov_mask] >> (num_bits - ov[ov_mask]).unsqueeze(0)
        output_g.scatter_add_(1, (word_idx[ov_mask] + 1).unsqueeze(0).expand(rows_g, -1),
                              ov_vals)
    output = output_g.view(rows, num_groups * num_bits)[:, :packed_cols]
    if packed_dim == 0:
        output = output.transpose(0, 1)
    return output


@pytest.mark.parametrize("bits", [4, 8])
def test_unpack_is_the_librarys_inverse(bits):
    """Codes packed by the library's pack_to_int32 come back through ctensors.unpack, and
    ctensors.pack packs them the same (offset-binary, the least significant code first)."""
    rng = np.random.default_rng(1)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    codes = rng.integers(lo, hi + 1, (5, 256)).astype(np.int8)
    codes[0, :8] = [lo, hi, 0, -1, 1, lo + 1, hi - 1, 0]
    packed = pack_to_int32(torch.from_numpy(codes), bits)
    assert packed.dtype == torch.int32 and packed.shape == (5, 256 * bits // 32)
    assert np.array_equal(CT.unpack(packed.numpy(), bits, 256), codes)
    assert np.array_equal(CT.pack(codes, bits), packed.numpy())
    # a 4-bit -8 is the nibble 0, +7 the nibble 15: not two's complement
    if bits == 4:
        one = pack_to_int32(torch.tensor([[-8, 7, 0, 1, 2, 3, 4, 5]], dtype=torch.int8), 4)
        assert int(one[0, 0]) & 0xFFFFFFFF == 0xDCBA98F0
        assert [(int(one[0, 0]) >> (4 * i)) & 15 for i in range(8)] == \
            [0, 15, 8, 9, 10, 11, 12, 13]
        assert np.array_equal(CT.unpack(one.numpy(), 4, 8)[0], [-8, 7, 0, 1, 2, 3, 4, 5])


def test_dequantize_group_scales():
    rng = np.random.default_rng(2)
    N, K, g = 4, 512, 128
    codes = rng.integers(-8, 8, (N, K)).astype(np.int8)
    scale = (rng.uniform(0.001, 0.1, (N, K // g)).astype(np.float32)
             .astype(np.float32)).view(np.uint32) & 0xFFFF0000        # bf16 values
    scale = scale.view(np.float32)
    packed = CT.pack(codes, 4)
    want = codes.astype(np.float32).reshape(N, K // g, g) * scale[:, :, None]
    got = CT.dequantize(packed, scale, (N, K), 4, g)
    assert got.dtype == np.float32 and np.array_equal(got, want.reshape(N, K))
    assert np.array_equal(CT.dequantize(packed, scale[:, :1], (N, K), 4, 0),
                          codes.astype(np.float32) * scale[:, :1])
    with pytest.raises(ValueError, match="scales"):
        CT.dequantize(packed, scale[:, :2], (N, K), 4, g)


def _checkpoint(tmp_path, asym: bool = False):
    """A two-tensor pack-quantized checkpoint: model.language_model.x (a packed Linear) and
    model.language_model.n (bf16 as usual), with its config.json."""
    from safetensors.torch import save_file
    rng = np.random.default_rng(3)
    N, K = 6, 256
    codes = rng.integers(-8, 8, (N, K)).astype(np.int8)
    scale = torch.from_numpy(rng.uniform(0.001, 0.1, (N, K // 128)).astype(np.float32)).bfloat16()
    norm = torch.randn(K).bfloat16()
    t = {"model.language_model.x.weight_packed": pack_to_int32(torch.from_numpy(codes), 4),
         "model.language_model.x.weight_scale": scale,
         "model.language_model.x.weight_shape": torch.tensor([N, K]),
         "model.language_model.n.weight": norm,
         "model.visual.v.weight": torch.randn(2, 2)}
    if asym:
        t["model.language_model.x.weight_zero_point"] = torch.zeros(1, N, dtype=torch.int32)
    save_file(t, str(tmp_path / "model.safetensors"))
    w = {"num_bits": 4, "type": "int", "symmetric": not asym, "strategy": "group",
         "group_size": 128, "actorder": "static"}
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "quantization_config": {
            "quant_method": "compressed-tensors", "format": "pack-quantized",
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": w}}}}))
    want = codes.astype(np.float32).reshape(N, K // 128, 128) * \
        scale.float().numpy()[:, :, None]
    return want.reshape(N, K), norm.float().numpy()


def test_weights_present_the_packed_tensor_as_weight(tmp_path):
    want, norm = _checkpoint(tmp_path)
    assert CT.scheme(tmp_path) == {"num_bits": 4, "symmetric": True, "type": "int",
                                   "group_size": 128}
    for W in (load_weights(tmp_path), Weights(tmp_path), LazyWeights(tmp_path)):
        assert "model.x.weight" in W and "model.x.weight_packed" in W
        assert "model.visual.v.weight" not in W
        v = W["model.x.weight"]
        assert v.dtype == np.float32 and np.array_equal(v, want)
        assert np.array_equal(W["model.n.weight"], norm)
        assert "model.x.weight" in set(W.keys())
    # a plain checkpoint has no scheme and no virtual names
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert CT.scheme(tmp_path) is None
    assert "model.x.weight" not in Weights(tmp_path)


def test_refuses_what_it_cannot_read(tmp_path):
    _checkpoint(tmp_path, asym=True)
    with pytest.raises(ValueError, match="asymmetric"):
        Weights(tmp_path)
    c = json.loads((tmp_path / "config.json").read_text())
    c["quantization_config"]["config_groups"]["group_0"]["weights"]["symmetric"] = True
    (tmp_path / "config.json").write_text(json.dumps(c))
    with pytest.raises(ValueError, match="zero point"):    # (the tensor is still there)
        Weights(tmp_path)
    c["quantization_config"]["format"] = "naive-quantized"
    (tmp_path / "config.json").write_text(json.dumps(c))
    with pytest.raises(ValueError, match="pack-quantized"):
        CT.scheme(tmp_path)
