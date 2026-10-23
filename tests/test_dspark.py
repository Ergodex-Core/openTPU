"""DSpark algorithm contracts independent of any installed GPU runtime or model download."""
from dataclasses import replace
import json

import numpy as np
import pytest

from opentpu.llm.dspark import Context, Reference, Spec, _attention, _rope, verify_greedy


def tiny_spec(**kw):
    return replace(Spec(hidden=8, layers=2, n_q=4, n_kv=2, head_dim=4, ffn=12,
                        vocab=7, draft_vocab=7, target_hidden=8, target_layers=(0, 3),
                        mask_token=6, markov_rank=3), **kw)


def weights(s, seed=12):
    rng = np.random.default_rng(seed)
    out = {}
    def put(name, shape, norm=False):
        out[name] = (np.ones(shape) if norm else rng.normal(0, .2, shape)).astype(np.float32)
    put("embed_tokens.weight", (s.vocab, s.hidden))
    put("lm_head.weight", (s.draft_vocab, s.hidden))
    put("fc.weight", (s.hidden, len(s.target_layers) * s.target_hidden))
    for norm in ("norm", "hidden_norm"):
        put(norm + ".weight", (s.hidden,), True)
    put("markov_head.markov_w1.weight", (s.vocab, s.markov_rank))
    put("markov_head.markov_w2.weight", (s.draft_vocab, s.markov_rank))
    for l in range(s.layers):
        for name in ("input_layernorm", "post_attention_layernorm"):
            put(f"layers.{l}.{name}.weight", (s.hidden,), True)
        for name, rows in (("q", s.n_q), ("k", s.n_kv), ("v", s.n_kv)):
            put(f"layers.{l}.self_attn.{name}_proj.weight", (rows * s.head_dim, s.hidden))
        put(f"layers.{l}.self_attn.o_proj.weight", (s.hidden, s.n_q * s.head_dim))
        for name in ("q", "k"):
            put(f"layers.{l}.self_attn.{name}_norm.weight", (s.head_dim,), True)
        for name, shape in (("gate", (s.ffn, s.hidden)), ("up", (s.ffn, s.hidden)),
                            ("down", (s.hidden, s.ffn))):
            put(f"layers.{l}.mlp.{name}_proj.weight", shape)
    if s.confidence_head:
        size = s.hidden + (s.markov_rank if s.confidence_with_markov else 0)
        put("confidence_head.proj.weight", (1, size))
        put("confidence_head.proj.bias", (1,))
    return out


def empty_context(s):
    return Context(np.array([], dtype=np.int64),
                   tuple(np.zeros((0, s.n_kv, s.head_dim), dtype=np.float32) for _ in range(s.layers)),
                   tuple(np.zeros((0, s.n_kv, s.head_dim), dtype=np.float32) for _ in range(s.layers)))


def test_noncausal_gqa_against_scalar_attention():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(3, 12, 4)).astype(np.float32)  # six Q heads per KV head
    k, v = (rng.normal(size=(5, 2, 4)).astype(np.float32) for _ in range(2))
    expected = np.zeros_like(q)
    for i in range(3):
        for h in range(12):
            logits = np.array([q[i, h].dot(k[j, h // 6]) / 2 for j in range(5)])
            p = np.exp(logits - max(logits))
            p /= sum(p)
            expected[i, h] = sum(p[j] * v[j, h // 6] for j in range(5))
    np.testing.assert_allclose(_attention(q, k, v), expected, rtol=1e-6, atol=1e-7)


def test_rope_layouts_preserve_pairs():
    x = np.arange(8, dtype=np.float32).reshape(1, 1, 8)
    for neox in (True, False):
        result = _rope(x, [7], 10, neox)
        a, b = np.split(x, 2, -1) if neox else (x[..., ::2], x[..., 1::2])
        c, d = np.split(result, 2, -1) if neox else (result[..., ::2], result[..., 1::2])
        np.testing.assert_allclose(a*a + b*b, c*c + d*d, rtol=1e-6)
    np.testing.assert_array_equal(_rope(x, [0], 10, True), x)
    assert not np.allclose(_rope(x, [7], 10, True), _rope(x, [7], 10, False))


def test_context_incremental_projection_and_truncation():
    s = tiny_spec()
    ref = Reference(s, weights(s))
    aux = np.random.default_rng(9).normal(size=(5, 2, 8)).astype(np.float32)
    full = ref.project_context(aux, np.arange(5))
    prefix = ref.project_context(aux[:2], [0, 1])
    joined = prefix.append(ref.project_context(aux[2:], [2, 3, 4]))
    for a, b in zip(full.keys + full.values, joined.keys + joined.values):
        # BLAS may select a different FP32 accumulation order for 2 vs 5 rows.
        # Allow a few ulps through projection, RMS normalization, and RoPE.
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
    truncated = joined.truncate(2)
    np.testing.assert_array_equal(truncated.positions, prefix.positions)
    for a, b in zip(truncated.keys + truncated.values, prefix.keys + prefix.values):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="strictly advance"):
        prefix.append(prefix)
    with pytest.raises(ValueError, match="auxiliary state shape"):
        ref.project_context(aux[:, :1], np.arange(5))


def test_backbone_is_noncausal_and_draft_does_not_mutate_context():
    s = tiny_spec()
    ref = Reference(s, weights(s))
    context = ref.project_context(np.ones((2, 2, 8), np.float32), [0, 1])
    before = [x.copy() for x in context.keys + context.values]
    first = ref.backbone([1, 6, 6], [2, 3, 4], context)
    changed_future = ref.backbone([1, 6, 4], [2, 3, 4], context)
    assert not np.allclose(first[0], changed_future[0])
    proposal = ref.draft(1, 2, context, 3)
    np.testing.assert_array_equal(proposal.hidden, first)
    for a, b in zip(before, context.keys + context.values):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="precede the anchor"):
        ref.draft(1, 1, context, 3)


def test_checkpoint_block_size_is_default_and_changes_prefix_states():
    s = tiny_spec(block_size=7)
    ref = Reference(s, weights(s))
    full = ref.draft(1, 0, empty_context(s))
    short = ref.draft(1, 0, empty_context(s), 3)
    prefix = ref.draft(1, 0, empty_context(s), 3, query_tokens=7)
    assert len(full.tokens) == 7 and len(short.tokens) == 3
    # Query slots attend across the full block, including slots not verified.
    assert not np.allclose(full.hidden[:3], short.hidden)
    np.testing.assert_array_equal(prefix.hidden, full.hidden[:3])
    np.testing.assert_allclose(prefix.logits, full.logits[:3], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(prefix.tokens, full.tokens[:3])
    with pytest.raises(ValueError, match="block size"):
        ref.draft(1, 0, empty_context(s), 3, query_tokens=2)


def test_markov_head_is_sequential_and_applies_d2t_offsets():
    s = tiny_spec(vocab=7, draft_vocab=3, markov_rank=3)
    w = weights(s)
    w["lm_head.weight"].fill(0)
    w["d2t"] = np.array([0, 2, 3], np.int64)  # draft 0,1,2 -> target 0,3,5
    w["markov_head.markov_w1.weight"].fill(0)
    w["markov_head.markov_w2.weight"] = np.eye(3, dtype=np.float32)
    w["markov_head.markov_w1.weight"][1] = [0, 5, 0]  # anchor1 -> target3
    w["markov_head.markov_w1.weight"][3] = [0, 0, 5]  # target3 -> target5
    w["markov_head.markov_w1.weight"][5] = [5, 0, 0]  # target5 -> target0
    ref = Reference(s, w, target_vocab_size=6)
    p = ref.draft(1, 0, empty_context(s), 3)
    np.testing.assert_array_equal(p.tokens, [3, 5, 0])
    np.testing.assert_array_equal(np.argmax(p.logits, -1), [1, 2, 0])
    w["d2t"] = np.array([0, -1, 0], np.int64)
    with pytest.raises(ValueError, match="duplicate"):
        Reference(s, w, target_vocab_size=6)


def test_confidence_and_topk_are_explicit():
    s = tiny_spec(confidence_head=True, confidence_with_markov=True, topk=2)
    w = weights(s)
    w["confidence_head.proj.weight"].fill(0)
    w["confidence_head.proj.bias"].fill(0)
    p = Reference(s, w).draft(1, 0, empty_context(s), 3)
    np.testing.assert_array_equal(p.confidence, [.5, .5, .5])
    assert np.all(np.sum(np.isfinite(p.logits), axis=1) == 2)
    del w["confidence_head.proj.weight"]
    with pytest.raises(ValueError, match="confidence_head"):
        Reference(s, w).draft(1, 0, empty_context(s), 3)


def test_optional_shared_embedding_and_lm_head_require_exact_shapes():
    s = tiny_spec()
    w = weights(s)
    shared = {"model.embed_tokens.weight": w.pop("embed_tokens.weight"),
              "lm_head.weight": w.pop("lm_head.weight")}
    Reference(s, w, target_weights=shared)
    with pytest.raises(ValueError, match="explicit compatible target"):
        Reference(s, w)
    shared["lm_head.weight"] = shared["lm_head.weight"][:-1]
    with pytest.raises(ValueError, match="shape"):
        Reference(s, w, target_weights=shared)


def test_corrupt_weights_cannot_silently_emit_argmax_token():
    s = tiny_spec()
    w = weights(s)
    w["markov_head.markov_w2.weight"][0, 0] = np.nan
    ref = Reference(s, w)
    with pytest.raises(ValueError, match="nonfinite"):
        ref.draft(1, 0, empty_context(s), 3)
    with pytest.raises(ValueError, match="nonfinite"):
        ref.project_context(np.full((1, 2, 8), np.inf, np.float32), [0])


def test_verify_prefix_bonus_eos_and_budget():
    logits = np.eye(8, dtype=np.float32)[[2, 3, 4, 5]]
    assert verify_greedy([2, 3, 7], logits).tokens == (2, 3, 4)
    assert verify_greedy([2, 3, 7], logits).accepted == 2
    assert verify_greedy([2, 3, 4], logits).tokens == (2, 3, 4, 5)
    assert verify_greedy([2, 3, 4], logits, eos_tokens=[3]).tokens == (2, 3)
    assert verify_greedy([2, 3, 4], logits, max_tokens=1).tokens == (2,)
    assert verify_greedy([2, 3, 4], logits, max_tokens=0).tokens == ()
    with pytest.raises(ValueError, match="one logits row"):
        verify_greedy([2, 3, 4], logits[:-1])


def test_config_never_invents_target_layers_or_unsupported_semantics(tmp_path):
    cfg = dict(hidden_size=8, num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=2, head_dim=4, intermediate_size=12, vocab_size=7,
               markov_rank=3, rms_norm_eps=1e-6,
               dflash_config=dict(mask_token_id=6, target_layer_ids=[0, 3]))
    def parse(**kw):
        (tmp_path / "config.json").write_text(json.dumps(dict(cfg, **kw)))
        return Spec.from_hf(tmp_path)
    assert parse().target_layers == (0, 3)
    assert parse().capture_layers == (1, 4)
    assert parse(block_size=9).block_size == 9
    with pytest.raises(ValueError, match="block_size"):
        parse(block_size=0)
    nested = dict(cfg["dflash_config"], enable_confidence_head=True,
                  confidence_head_with_markov=True)
    assert parse(dflash_config=nested).confidence_with_markov
    assert parse(dflash_config=nested).confidence_head
    with pytest.raises(ValueError, match="target_layer_ids"):
        parse(dflash_config=dict(mask_token_id=6))
    for kw, match in ((dict(is_causal=True), "noncausal"),
                      (dict(attention_bias=True), "biases"),
                      (dict(sample_from_anchor=False), "sample_from_anchor"),
                      (dict(quantization_config={"quant_method": "gptq"}), "floating-point"),
                      (dict(rope_scaling={"rope_type": "yarn"}), "RoPE")):
        with pytest.raises(ValueError, match=match):
            parse(**kw)
