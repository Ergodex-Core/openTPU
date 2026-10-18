"""Gemma 3 on openTPU (opentpu/llm/gemma3.py, on Gemma 4's code): the 1 + w norms, the
attention scale folded into q_norm, no V norm, every layer with its own K / V, the global
layers' full-head RoPE; against Hugging Face transformers on a tiny random model, through a
saved checkpoint (load_spec / load_weights) and otpu-chat."""
import json
from pathlib import Path

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm import gemma3 as G3
from opentpu.llm import gemma4 as G
from opentpu.llm import load_spec, model_dir
from opentpu.llm.qwen3 import Engine, load_weights

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.gemma3")

REAL = model_dir("gemma3")
LAYERS = 8          # 5 sliding, a global, 2 sliding: the loops (s x 5, f), then s x 2


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def _model(d, layers=LAYERS, window=512):
    """A tiny random Gemma 3 (HF's layer pattern: 5 sliding then a global): (HF model, the
    weights folded for Gemma 4's code, Spec), its config.json in d."""
    torch.manual_seed(0)
    hc = transformers.Gemma3TextConfig(
        hidden_size=256, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=1, head_dim=128, intermediate_size=512, vocab_size=1000,
        sliding_window=window, sliding_window_pattern=6, query_pre_attn_scalar=128,
        max_position_embeddings=4096, final_logit_softcapping=None,
        attn_logit_softcapping=None)
    hc._attn_implementation = "eager"
    m = transformers.Gemma3ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:                 # the gains near 1 + 0 (Gemma 3's 1 + w)
                p.copy_(0.1 * torch.randn_like(p))
    cfg = hc.to_dict()
    (d / "config.json").write_text(json.dumps(cfg))
    W = G3.fold_weights({k: v.float().numpy() for k, v in m.state_dict().items()}, cfg)
    return m, W, G3.spec_from_hf(d)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return _model(tmp_path_factory.mktemp("gemma3"))


def _cfg(S: int = 1):
    """The card's configuration (S slices: the tensor-parallel split, docs/gemma4.md
    "Slices") with a DRAM the tiny model's image fits."""
    return board_config(DRAM_BYTES=1 << 26, S=S)


def test_spec(tiny):
    _, _, spec = tiny
    assert spec.kinds == ("sliding",) * 5 + ("full",) + ("sliding",) * 2
    assert spec.kv_src == tuple(range(LAYERS)) and not spec.ple_dim and not spec.v_norm
    assert spec.global_rot == spec.head_dim // 2 == 64 and spec.global_rope_div == 1.0
    assert spec.theta == 1e4 and spec.global_theta == 1e6 and spec.softcap is None
    assert spec.eos == (1,) and spec.bos == 2 and spec.tied


def test_reference_matches_hf_past_the_window(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 600)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(G.reference_logits(spec, W, toks) - hf).max() < 1e-4


def test_tiny_matches_hf_and_emulation(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 24)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    eng = Engine(spec, W, cap=1024, cfg=_cfg())
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, hf).min() > 0.985
    emu = G.emulated_logits(spec, W, toks[:12])
    assert _cos(dev[:12], emu).min() > 0.99


def test_resident_and_prefill_are_bit_exact(tiny):
    """The resident decode (the rows gathered on the device) and a chunked prefill give the
    per-position programs' logits, across the sliding ring's wrap (768 slots)."""
    _, W, spec = tiny
    rng = np.random.default_rng(1)
    toks = [int(t) for t in rng.integers(0, 1000, 800)]
    cfg = _cfg()
    a = Engine(spec, W, cap=1024, cfg=cfg)
    ref = [a.step(t) for t in toks]
    b = Engine(spec, W, cap=1024, cfg=cfg, resident=True)
    got = [b.step(t) for t in toks]
    for i in (0, 1, 5, 255, 256, 511, 512, 767, 768, 799):
        np.testing.assert_array_equal(got[i], ref[i])
    c = Engine(spec, W, cap=1024, cfg=cfg, resident=True)
    np.testing.assert_array_equal(c.prefill(toks[:790]), ref[789])
    for t, r in zip(toks[790:], ref[790:]):
        np.testing.assert_array_equal(c.step(t), r)


@pytest.mark.parametrize("S", [2, 4])
def test_slices_match_hf_and_one_slice(tiny, S):
    """Proposal P4's split (docs/gemma4.md "Slices"): on S slices every matrix is split by
    output rows with the inputs all-gathered, so every fp32 sum is the one-slice program's:
    the logits of the per-position programs and of the resident decode are bit-exact to
    S = 1's (and so match Hugging Face as they do), on every slice's own copy of the K / V."""
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 20)]
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    ref = Engine(spec, W, cap=1024, cfg=_cfg())
    one = np.array([ref.step(t) for t in toks])
    eng = Engine(spec, W, cap=1024, cfg=_cfg(S))
    assert eng.cfg.S == S and eng.image.v_loc == spec.vocab // S
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, hf).min() > 0.985
    np.testing.assert_array_equal(dev, one)
    res = Engine(spec, W, cap=1024, cfg=_cfg(S), resident=True)
    assert res.resident
    np.testing.assert_array_equal(np.array([res.step(t) for t in toks]), one)


@pytest.fixture(scope="module")
def tiny_w256(tmp_path_factory):
    """The tiny model with a 256-token window: its sliding ring (window + an attention
    block) wraps at 512 slots."""
    return _model(tmp_path_factory.mktemp("gemma3_w256"), window=256)


def test_four_slices_resident_and_prefill_are_bit_exact(tiny_w256):
    """At S = 4 the resident decode and a chunked prefill give the per-position programs'
    logits across the sliding ring's wrap (512 slots at a 256-token window), as the S = 1
    test above does at 768."""
    _, W, spec = tiny_w256
    assert spec.window == 256
    rng = np.random.default_rng(4)
    toks = [int(t) for t in rng.integers(0, 1000, 530)]
    cfg = _cfg(4)
    a = Engine(spec, W, cap=1024, cfg=cfg)
    assert a.image.ring == 512
    ref = [a.step(t) for t in toks]
    b = Engine(spec, W, cap=1024, cfg=cfg, resident=True)
    got = [b.step(t) for t in toks]
    for i in (0, 1, 5, 255, 256, 511, 512, 529):
        np.testing.assert_array_equal(got[i], ref[i])
    c = Engine(spec, W, cap=1024, cfg=cfg, resident=True)
    np.testing.assert_array_equal(c.prefill(toks[:520]), ref[519])
    assert max(s.get("rows", 1) for s in c.stats) > 1            # runs of several rows
    for t, r in zip(toks[520:], ref[520:]):
        np.testing.assert_array_equal(c.step(t), r)


def test_slices_the_model_does_not_split_over():
    """Spec.check: S in (1, 2, 4) and the dimensions divisible by S."""
    spec = G.Spec(hidden=256, kinds=(G.SLIDE,) * 2, kv_src=(0, 1), ffn=(512, 512), n_q=4,
                  n_kv=1, head_dim=128, global_head_dim=128, vocab=1000, ple_dim=0)
    spec.check(_cfg(4))
    with pytest.raises(ValueError, match="slices S in"):
        spec.check(_cfg(3))
    with pytest.raises(ValueError, match="n_q 2 % S"):
        G.Spec(hidden=256, kinds=(G.SLIDE,) * 2, kv_src=(0, 1), ffn=(512, 512), n_q=2,
               n_kv=1, head_dim=128, global_head_dim=128, vocab=1000, ple_dim=0,
               ).check(_cfg(4))
    with pytest.raises(ValueError, match="vocab 1002 % S"):
        G.Spec(hidden=256, kinds=(G.SLIDE,) * 2, kv_src=(0, 1), ffn=(512, 512), n_q=4,
               n_kv=1, head_dim=128, global_head_dim=128, vocab=1002, ple_dim=0,
               ).check(_cfg(4))


def checkpoint(d: Path) -> Path:
    """The tiny model saved by Hugging Face in d, with a word-level tokenizer ("t17" is token
    17) and a chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    d.mkdir(parents=True, exist_ok=True)
    m, _, _ = _model(d)
    m.save_pretrained(d)
    tk = Tokenizer(models.WordLevel({f"t{i}": i for i in range(1000)}, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(tokenizer_object=tk).save_pretrained(d)
    (d / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }} {% endfor %}")
    return d


def test_checkpoint_weights_are_folded(tiny, tmp_path):
    """load_spec / load_weights on the saved checkpoint: the norms read as 1 + w, q_norm with
    the attention scale, a layer scalar of 1, and the same reference logits."""
    m, W, spec = tiny
    checkpoint(tmp_path)
    s2, W2 = load_spec(tmp_path), load_weights(tmp_path)
    assert s2 == spec
    raw = m.state_dict()
    g = "model.layers.0.input_layernorm.weight"
    np.testing.assert_array_equal(W2[g], 1 + raw[g].float().numpy())
    q = "model.layers.0.self_attn.q_norm.weight"
    np.testing.assert_allclose(W2[q], (1 + raw[q].float().numpy()) * 128 ** -0.5, rtol=1e-6)
    np.testing.assert_array_equal(W2["model.layers.3.layer_scalar"], [1.0])
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 16)]
    np.testing.assert_allclose(G.reference_logits(s2, W2, toks),
                               G.reference_logits(spec, W, toks), atol=1e-5)


def test_otpu_chat_on_the_isa_simulator(tmp_path, monkeypatch, capfd):
    import sys

    from opentpu.host import chat as C
    checkpoint(tmp_path)
    monkeypatch.setenv("OTPU_IMG_CACHE", "0")
    plain = C.Chat.ask_plain
    monkeypatch.setattr(C.Chat, "ask_plain", lambda self, text: plain(self, text, sys.stdout))
    capfd.readouterr()
    C.main(["--model", str(tmp_path), "--prompt", "t5 t6 t7", "--greedy", "--max-new", "3",
            "--cap", "1024", "--no-prog-cache"])
    out = capfd.readouterr().out
    assert "weights: int8, head int8" in out and "prefill 3 tokens" in out and "decode" in out


@pytest.mark.skipif(not (REAL / "config.json").exists(), reason="models/gemma-3-1b-it")
def test_real_spec():
    spec = load_spec(REAL)
    assert spec.layers == 26 and spec.hidden == 1152 and spec.kinds.count("full") == 4
    assert spec.n_q == 4 and spec.n_kv == 1 and spec.head_dim == 256 and spec.window == 512
    assert spec.eos == (1, 106) and spec.softcap is None and spec.vocab == 262144
    spec.check(board_config())
