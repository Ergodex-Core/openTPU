"""Speculative decoding with prompt-lookup drafts on the card (opentpu/llm/specdec.py,
docs/specdec.md): the loop's tokens are plain greedy decode's (Engine.generate_card) bit for bit,
whatever the drafts, across attention buckets, at stop ids and at the tokens' limit."""
import numpy as np
import pytest

from opentpu.llm.specdec import ngram_drafts

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_ngram_drafts_take_the_latest_longest_match():
    h = [5, 1, 2, 3, 9, 1, 2, 3, 7, 4, 2, 3]
    assert ngram_drafts(h, 2) == [7, 4]             # g = 2 (2, 3), latest: after 1 2 3 at 5
    assert ngram_drafts(h + [1, 2, 3], 3) == [7, 4, 2]   # 1 2 3 at 5 .. 7, the latest
    assert ngram_drafts([1, 2, 3, 4], 2) == []      # no earlier occurrence
    assert ngram_drafts([8, 8], 3) == [8, 8, 8]     # a period of one
    assert ngram_drafts([4, 6, 4], 4) == [6, 4, 6, 4]   # g = 1, the period 4 6


def _model(tied: bool, zero_layers: bool):
    """A tiny Qwen3 whose greedy tokens vary (initializer_range 0.2). zero_layers: the layers'
    outputs (W_o, W_down) zero, so the next token is a function of the token alone and the
    greedy tokens fall into a cycle (the drafts are then accepted), untied (a tied head would
    give each token itself)."""
    from opentpu.llm.qwen3 import Spec
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=128, intermediate_size=512, vocab_size=1000, rms_norm_eps=1e-6,
        rope_theta=1e6, tie_word_embeddings=tied, max_position_embeddings=4096,
        initializer_range=0.2)
    m = transformers.Qwen3ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    if zero_layers:
        W = {k: np.zeros_like(v) if k.endswith(("o_proj.weight", "down_proj.weight")) else v
             for k, v in W.items()}
    return W, Spec(256, 2, 4, 2, 128, 512, 1000, tied=tied)


@pytest.fixture(scope="module", params=["random", "cycle"])
def tiny(request):
    return (request.param,) + _model(request.param == "random", request.param == "cycle")


def _engines(spec, W, K: int, cap: int = 512, wformat: str = "int8"):
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    cfg = device_config(spec, cap, rows=PREFILL_ROWS, lookup=True, spec_rows=K + 1, S=1,
                        wformat=wformat)
    return (Engine(spec, W, cap=cap, cfg=cfg, resident=True, spec_rows=K + 1, wformat=wformat),
            Engine(spec, W, cap=cap, cfg=cfg, resident=True, wformat=wformat))


@pytest.mark.parametrize("K", [1, 3])
def test_the_loop_gives_plain_greedy_tokens(tiny, K):
    """From position 200 for 100 tokens (across the bucket boundary at 256): the plain loop's
    tokens, and Engine.pos after them; the cycling model's drafts are accepted, a K + 1 row
    verify emitting up to K + 1 tokens; a second call continues the context."""
    from opentpu.llm.specdec import SpecDecoder
    name, W, spec = tiny
    a, b = _engines(spec, W, K)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 200)]
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref = b.generate_card(t0, 100, stop_ids=[])
    d = SpecDecoder(a)
    got = d.generate_card(t0, 70, stop_ids=[], context=toks + [t0])
    assert got == ref[:70] and a.pos == 200 + 70
    st = d.last
    assert st.iterations + st.accepted == 70 and st.verifies <= st.iterations
    if name == "cycle":
        assert st.accepted >= 15 and st.accepted > K * 0.5 * st.verifies
    seen = []
    got2 = d.generate_card(got[-1], 30, stop_ids=[], context=toks + [t0] + got,
                           on_token=seen.append)
    assert got2 == ref[70:100] == seen and a.pos == 300
    assert sorted(d._done) == [1, 2]


def test_stop_ids_and_the_tokens_limit(tiny):
    """A stop id ends the loop where plain decode's does (returned, not fed), also inside an
    accepted run of drafts; at most n tokens, also when the drafts would give more."""
    from opentpu.llm.specdec import SpecDecoder
    name, W, spec = tiny
    a, b = _engines(spec, W, 3)
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 120)]
    t0 = int(np.argmax(a.prefill(toks)))
    b.prefill(toks)
    ref = b.generate_card(t0, 60, stop_ids=[])
    d = SpecDecoder(a)
    for n in (1, 2, 5, 7):
        a.rollback(120)
        assert d.generate_card(t0, n, stop_ids=[], context=toks + [t0]) == ref[:n]
        assert a.pos == 120 + n
    j = next(j for j in range(40, 60) if ref[j] not in ref[:j]) if name == "random" else 45
    a.rollback(120)
    got = d.generate_card(t0, 60, stop_ids=[ref[j], 1001], context=toks + [t0])
    k = ref.index(ref[j])
    assert got == ref[:k + 1] and a.pos == 120 + k + 1      # (as generate_card leaves it)


def test_int4_weights(tiny):
    """4-bit weights: the K + 1 rows' MMs take no column reuse (PAIR, 2 rows at MCOLS 4), the
    decode step's does; the outputs are the same integers, so the same tokens."""
    from opentpu.llm.specdec import SpecDecoder
    name, W, spec = tiny
    a, b = _engines(spec, W, 3, wformat="int4")
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 240)]
    t0 = int(np.argmax(a.prefill(toks)))
    b.prefill(toks)
    ref = b.generate_card(t0, 40, stop_ids=[])
    assert SpecDecoder(a).generate_card(t0, 40, stop_ids=[], context=toks + [t0]) == ref


# ---- a draft model on the same device (specdec.draft_engine)
def _with_draft(spec, W, K: int, dW, dspec, cap: int = 512):
    from dataclasses import replace

    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    from opentpu.llm.specdec import draft_engine
    cfg = device_config(spec, cap, rows=PREFILL_ROWS, lookup=True, spec_rows=K + 1, S=1)
    cfg = replace(cfg, DRAM_BYTES=4 * cfg.DRAM_BYTES)       # (room for the draft's image)
    a = Engine(spec, W, cap=cap, cfg=cfg, resident=True, spec_rows=K + 1, prompt_runs=True)
    b = Engine(spec, W, cap=cap, cfg=cfg, resident=True)
    return a, b, draft_engine(a, None, wformat="int8", W=dW, spec=dspec)


@pytest.mark.parametrize("draft", ["same", "other"])
def test_a_draft_model_gives_plain_greedy_tokens(draft):
    """A draft model's drafts (its decode steps on the card, D): the plain loop's tokens across
    the bucket boundary at 256, the context continued by a second call (the draft's K / V
    follows: SpecDecoder.dctx), after a rollback and with stop ids. The target itself as the
    draft has every draft accepted; another model (a larger vocabulary, cut to the target's)
    some. Its image sits above the target's in the one DRAM."""
    from opentpu.llm.qwen3 import Spec
    from opentpu.llm.specdec import SpecDecoder
    W, spec = _model(True, False)
    if draft == "same":
        dW, dspec = W, spec
    else:
        rng = np.random.default_rng(5)
        dW = {k: v + 0.3 * v.std() * rng.standard_normal(v.shape).astype(np.float32)
              if v.ndim == 2 else v for k, v in W.items()}
        e = dW["model.embed_tokens.weight"]
        dW["model.embed_tokens.weight"] = np.concatenate([e, e[:24]])
        dspec = Spec(256, 2, 4, 2, 128, 512, 1024)
    K = 3
    a, b, dr = _with_draft(spec, W, K, dW, dspec)
    assert dr.image.base >= a.image.nbytes and dr.image.nbytes <= a.cfg.DRAM_BYTES
    assert dr.spec.vocab == 1000
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 200)]
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref = b.generate_card(t0, 100, stop_ids=[])
    d = SpecDecoder(a, draft=dr)
    got = d.generate_card(t0, 70, stop_ids=[], context=toks + [t0])
    assert got == ref[:70] and a.pos == 270 and dr.pos in (269, 270)
    st = d.last
    assert st.iterations + st.accepted == 70 and st.verifies <= st.iterations
    if draft == "same":
        assert st.accepted >= 2.5 * st.iterations - 10
    got2 = d.generate_card(got[-1], 30, stop_ids=[], context=toks + [t0] + got)
    assert got2 == ref[70:100]
    a.rollback(200)                     # the draft goes back with the context (dctx)
    j = 50
    got3 = d.generate_card(t0, 60, stop_ids=[ref[j]], context=toks + [t0])
    k = ref.index(ref[j])
    assert got3 == ref[:k + 1] and a.pos == 200 + k + 1
