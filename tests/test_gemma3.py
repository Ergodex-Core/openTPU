"""Gemma 3: HF math, int8 / int4 ISA decode, and checkpoint configuration / names."""
import json

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm import gemma3 as G
from opentpu.llm import load_spec, model_dir
from opentpu.llm.qwen3 import Engine, load_weights


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.gemma3")
    torch.manual_seed(31)
    hc = transformers.Gemma3TextConfig(
        hidden_size=128, num_hidden_layers=3, num_attention_heads=2,
        num_key_value_heads=1, head_dim=128, intermediate_size=256, vocab_size=256,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=256, query_pre_attn_scalar=128, max_position_embeddings=1024)
    hc._attn_implementation = "eager"
    m = transformers.Gemma3ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name:
                p.copy_(torch.randn_like(p) * 0.1)
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    d = tmp_path_factory.mktemp("gemma3")
    (d / "config.json").write_text(json.dumps(hc.to_dict()))
    return m, W, G.Spec.from_hf(d), d


def test_spec_and_stop_ids(tiny):
    _, _, spec, path = tiny
    assert isinstance(load_spec(path), G.Spec)
    assert spec.kinds == (G.SLIDE, G.SLIDE, G.FULL)
    assert spec.kv_src == (0, 1, 2) and spec.ple_dim == 0
    assert spec.norm_offset == 1 and not spec.v_norm and not spec.layer_scale
    assert spec.attn_scale == 128 ** -0.5 and spec.global_rot == 64
    assert spec.truncated([0, 2]).kv_src == (0, 1)
    assert type(spec.truncated([0, 2])) is G.Spec
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 50, 106]}))
    assert G.Spec.from_hf(path).eos == (1, 50, 106)


def test_reference_matches_hf(tiny):
    import torch
    m, W, spec, _ = tiny
    tokens = [2, 18, 72, 30, 5, 10, 63, 125]
    with torch.no_grad():
        hf = m(torch.tensor([tokens])).logits[0].numpy()
    np.testing.assert_allclose(G.reference_logits(spec, W, tokens), hf, atol=2e-5, rtol=2e-5)


def test_reference_sliding_window(tiny):
    import torch
    m, W, spec, _ = tiny
    tokens = np.random.default_rng(2).integers(3, 256, 261).tolist()
    with torch.no_grad():
        hf = m(torch.tensor([tokens])).logits[0, -3:].numpy()
    ref = G.reference_logits(spec, W, tokens)[-3:]
    np.testing.assert_allclose(ref, hf, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("fmt", ["int8", "int4"])
def test_isa_matches_reference_and_emulation(tiny, fmt):
    import torch
    m, W, spec, _ = tiny
    tokens = [2, 18, 72, 30]
    cfg = board_config(DRAM_BYTES=1 << 24)
    eng = Engine(spec, W, cap=512, cfg=cfg, wformat=fmt, resident=True, rows=1)
    dev = np.array([eng.step(t) for t in tokens])
    with torch.no_grad():
        hf = m(torch.tensor([tokens])).logits[0].numpy()
    emu = G.emulated_logits(spec, W, tokens, wformat=fmt)
    assert _cos(dev, hf).min() > (0.99 if fmt == "int8" else 0.89)
    assert _cos(dev, emu).min() > 0.999
    np.testing.assert_array_equal(dev.argmax(-1), emu.argmax(-1))


def test_hf_and_mlx_weights(tiny, tmp_path):
    from safetensors.numpy import save_file
    _, W, _, path = tiny
    (tmp_path / "config.json").write_text((path / "config.json").read_text())
    # MLX's unprefixed names; leave out the tied duplicate head.
    raw = {k.removeprefix("model."): v.copy() for k, v in W.items() if k != "lm_head.weight"}
    save_file(raw, str(tmp_path / "model.safetensors"))
    got = load_weights(tmp_path)
    assert isinstance(got, G.Weights)
    for name in ("model.layers.0.self_attn.q_norm.weight", "model.embed_tokens.weight"):
        np.testing.assert_array_equal(got[name], W[name])
    np.testing.assert_array_equal(got.rows("model.embed_tokens.weight", [3, 8]),
                                  W["model.embed_tokens.weight"][[3, 8]])


def test_reject_scaled_rope(tiny, tmp_path):
    _, _, _, path = tiny
    c = json.loads((path / "config.json").read_text())
    c["rope_parameters"] = {"rope_type": "linear", "factor": 2.0}
    (tmp_path / "config.json").write_text(json.dumps(c))
    with pytest.raises(ValueError, match="default RoPE"):
        G.Spec.from_hf(tmp_path)


@pytest.mark.parametrize("name,layers,hidden", [("gemma3", 26, 1152), ("functiongemma", 18, 640)])
def test_real_compile(name, layers, hidden):
    path = model_dir(name)
    if not (path / "config.json").exists():
        pytest.skip(f"{path}: checkpoint not present")
    spec = G.Spec.from_hf(path)
    assert (spec.layers, spec.hidden) == (layers, hidden)
    assert spec.kinds[5] == G.FULL and spec.global_rot == 128
    img = spec.image(board_config(DRAM_BYTES=2 << 30), 1024, rows=1, wformat="int4", lookup=True)
    programs, _ = img.compile_decode(1, 0)
    assert len(programs) == 1 and len(programs[0]) > 0


def test_tiled_prompt_matches_decode(tiny):
    _, W, spec, _ = tiny
    tokens = [2, 18, 72, 30, 5, 10, 63, 125]
    cfg = board_config(DRAM_BYTES=1 << 24)
    decode = Engine(spec, W, cap=512, cfg=cfg, wformat="int4", resident=True, rows=1)
    expected = decode.prefill(tokens)
    tiled = Engine(spec, W, cap=512, cfg=cfg, wformat="int4", resident=True, rows=4,
                   prompt_runs=True, tile_runs=True)
    got = list(tiled.prefill_chunks(tokens))[-1][1]
    assert _cos(expected, got) > 0.9999
    assert expected.argmax() == got.argmax()


def test_gguf_lazy_weights(tmp_path):
    gguf = pytest.importorskip("gguf")
    from opentpu import quant as Q
    writer = gguf.GGUFWriter(str(tmp_path / "tiny.gguf"), "gemma3")
    norm = np.linspace(0.25, 2.0, 128, dtype=np.float32)
    emb = np.arange(256, dtype=np.float32).reshape(2, 128) / 128 - 1
    packed = gguf.quants.quantize(emb, gguf.GGMLQuantizationType.Q4_0)
    writer.add_tensor("token_embd.weight", packed, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    writer.add_tensor("output_norm.weight", norm)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    W = G.load_weights(tmp_path)
    np.testing.assert_array_equal(W["model.norm.weight"], norm - 1)
    np.testing.assert_array_equal(W["model.embed_tokens.weight"], Q.dequantize_q4_0(packed))
    np.testing.assert_array_equal(W.rows("model.embed_tokens.weight", [1]),
                                  Q.dequantize_q4_0(packed)[[1]])
