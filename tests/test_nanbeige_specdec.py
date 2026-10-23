"""End-to-end greedy equality and cache rollback with the real ISA target engine."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu.compiler import CompileError
from opentpu.isasim import Config
from opentpu.llm.dspark import Reference, Spec as DraftSpec
from opentpu.llm.nanbeige import Engine, Spec
from opentpu.llm.nanbeige_specdec import Decoder


def _weights(hidden, layers, nq, nkv, hd, ffn, vocab, prefix, rng):
    out = {}
    def put(name, shape, norm=False):
        out[prefix + name + ".weight"] = (np.ones(shape) if norm else
            rng.normal(0, .12 / np.sqrt(shape[-1]), shape)).astype(np.float32)
    put("embed_tokens", (vocab, hidden))
    put("norm", (hidden,), True)
    for l in range(layers):
        for name in ("input_layernorm", "post_attention_layernorm"):
            put(f"layers.{l}.{name}", (hidden,), True)
        for name, rows in (("q", nq), ("k", nkv), ("v", nkv)):
            put(f"layers.{l}.self_attn.{name}_proj", (rows * hd, hidden))
        put(f"layers.{l}.self_attn.o_proj", (hidden, nq * hd))
        for name, shape in (("gate", (ffn, hidden)), ("up", (ffn, hidden)),
                            ("down", (hidden, ffn))):
            put(f"layers.{l}.mlp.{name}_proj", shape)
    out["lm_head.weight"] = rng.normal(0, .1, (vocab, hidden)).astype(np.float32)
    return out


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(92)
    target = Spec(128, 2, 2, 1, 128, 256, 128, capture_layers=(1, 3), eos=())
    cfg = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, ACT_ROWS=16,
                 TMEM_WORDS=65536, IMEM_WORDS=32768, DRAM_BYTES=8 << 20)
    tw = _weights(128, 2, 2, 1, 128, 256, 128, "model.", rng)
    draft = DraftSpec(128, 1, 2, 1, 64, 256, 128, 128, 128, (0, 2), 127, 8)
    dw = _weights(128, 1, 2, 1, 64, 256, 128, "", rng)
    # Exercise the actual released checkpoint's omitted embedding/LM head.
    del dw["embed_tokens.weight"], dw["lm_head.weight"]
    dw["fc.weight"] = rng.normal(0, .03, (128, 256)).astype(np.float32)
    dw["hidden_norm.weight"] = np.ones(128, np.float32)
    dw["layers.0.self_attn.q_norm.weight"] = np.ones(64, np.float32)
    dw["layers.0.self_attn.k_norm.weight"] = np.ones(64, np.float32)
    dw["markov_head.markov_w1.weight"] = rng.normal(0, .1, (128, 8)).astype(np.float32)
    dw["markov_head.markov_w2.weight"] = rng.normal(0, .1, (128, 8)).astype(np.float32)
    plain = Engine(target, tw, cfg=cfg, cap=128, rows=4, prog_cache=False)
    prompt = (2, 7)
    captures = []
    for token in prompt:
        logits = plain.step(token)
        captures.append(plain.read_hidden((1, 3))[0])
    expected = [int(np.argmax(logits))]
    for _ in range(6):
        logits = plain.step(expected[-1])
        captures.append(plain.read_hidden((1, 3))[0])
        expected.append(int(np.argmax(logits)))
    return target, cfg, tw, draft, dw, prompt, tuple(expected), np.stack(captures)


def make_decoder(fixture, *, scripted=False, reject=False, fallback=False,
                 draft_query_tokens=None):
    target, cfg, tw, draft, dw, prompt, expected, _ = fixture
    engine = Engine(target, tw, cfg=cfg, cap=128, rows=4, prog_cache=False)
    ref = Reference(draft, dw, target_weights=tw)
    ref.query_calls = []
    if scripted:
        stream = prompt + expected
        def propose(anchor, position, context, n, *, query_tokens=None):
            assert anchor == stream[position]
            ref.query_calls.append(query_tokens)
            tokens = list(stream[position + 1:position + n + 1])
            tokens += [0] * (n - len(tokens))  # Guesses beyond this test's requested output.
            if reject and len(tokens) >= 2:
                tokens[1] = (tokens[1] + 1) % target.vocab
            return SimpleNamespace(tokens=np.array(tokens, dtype=np.int64))
        ref.draft = propose
    if fallback:
        original = engine.run_rows
        def compile_limited(rows, *args, **kwargs):
            if len(rows) > 2:
                raise CompileError("deliberate two-row test limit")
            return original(rows, *args, **kwargs)
        engine.run_rows = compile_limited
    return Decoder(engine, ref, draft_tokens=3, prefill_chunk=2,
                   draft_query_tokens=draft_query_tokens)


def test_real_reference_draft_matches_plain_target_tokens(fixture):
    decoder = make_decoder(fixture)
    result = decoder.generate(fixture[5], max_new_tokens=5)
    assert result.tokens == fixture[6][:5]
    assert result.stats.generated_tokens == 5
    assert result.stats.decode_tokens == 4
    assert result.stats.draft_backend == "numpy-fp32-reference"
    assert result.stats.target_seconds > 0 and result.stats.draft_seconds > 0
    assert result.stats.draft_query_tokens == 7
    assert result.stats.computed_query_tokens == result.stats.draft_calls * 7
    assert result.stats.prefill_seconds + result.stats.decode_seconds <= result.stats.total_seconds
    raw = result.stats.target_run_stats
    split = result.stats.target_prefill_runs
    assert split > 0 and result.stats.target_decode_runs > 0
    assert len(raw) == split + result.stats.target_decode_runs
    # The functional ISA simulator has no hardware cycle measurements.
    assert result.stats.target_prefill_cycles is None
    assert result.stats.target_decode_cycles is None


@pytest.mark.parametrize("reject,fallback", [(False, False), (True, False), (False, True)])
def test_accepted_prefix_rollback_and_capture_match_baseline(fixture, reject, fallback):
    decoder = make_decoder(fixture, scripted=True, reject=reject, fallback=fallback)
    result = decoder.generate(fixture[5], max_new_tokens=7)
    assert result.tokens == fixture[6]
    assert decoder.target.pos == len(fixture[5]) + 6
    assert decoder.context.positions.tolist() == list(range(decoder.target.pos))
    expected_ctx = decoder.draft.project_context(fixture[7], np.arange(decoder.target.pos))
    for actual, expected in zip(decoder.context.keys + decoder.context.values,
                                expected_ctx.keys + expected_ctx.values):
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
    assert result.stats.accepted_tokens > 0
    if reject:
        assert result.stats.accepted_tokens < result.stats.proposed_tokens
    else:
        assert result.stats.accepted_tokens == result.stats.proposed_tokens
    assert bool(result.stats.compile_fallbacks) == fallback
    assert decoder.draft.query_calls and set(decoder.draft.query_calls) == {7}


def test_full_block_is_drafted_even_when_only_one_proposal_fits_budget(fixture):
    decoder = make_decoder(fixture, scripted=True)
    result = decoder.generate(fixture[5], max_new_tokens=3)
    assert result.tokens == fixture[6][:3]
    assert decoder.draft.query_calls == [7]
    assert result.stats.proposed_tokens == 1
    assert result.stats.computed_draft_tokens == 1
    assert result.stats.computed_query_tokens == 7
    assert result.stats.iterations[0]["verify_rows"] == 2


def test_short_draft_is_an_explicit_distinct_configuration(fixture):
    decoder = make_decoder(fixture, scripted=True, draft_query_tokens=3)
    result = decoder.generate(fixture[5], max_new_tokens=5)
    assert result.tokens == fixture[6][:5]
    assert set(decoder.draft.query_calls) == {3}
    assert result.stats.draft_query_tokens == 3
    with pytest.raises(ValueError, match="verification prefix"):
        make_decoder(fixture, draft_query_tokens=2)


def test_initial_eos_and_zero_generation(fixture):
    decoder = make_decoder(fixture)
    zero = decoder.generate(fixture[5], max_new_tokens=0)
    assert zero.tokens == () and zero.stats.target_calls == 0
    eos = decoder.generate(fixture[5], max_new_tokens=7, eos_tokens=(fixture[6][0],))
    assert eos.tokens == fixture[6][:1]
    assert eos.stop_reason == "eos" and eos.stats.draft_calls == 0


def test_missing_capture_is_rejected_before_running(fixture):
    target, cfg, tw, draft, dw, *_ = fixture
    engine = Engine(replace(target, capture_layers=()), tw, cfg=cfg, cap=128,
                    rows=4, prog_cache=False)
    with pytest.raises(ValueError, match="capture_layers"):
        Decoder(engine, Reference(draft, dw, target_weights=tw))


@pytest.fixture
def validation_decoder():
    """Validation needs no weights, compiler or ISA execution."""
    target = SimpleNamespace(
        batch=1, stateful=False, cap=8, pos=5, resets=0,
        spec=SimpleNamespace(hidden=4, vocab=8, eos=()),
        image=SimpleNamespace(capture_slots=(0,), rows=2),
        backend=object(), stats=[{"preserved": True}],
    )

    def reset():
        target.resets += 1
        target.pos = 0

    target.reset = reset
    draft = SimpleNamespace(
        spec=SimpleNamespace(target_hidden=4, capture_layers=(0,), block_size=2),
        target_vocab=8, backend="validation-stub",
    )
    decoder = Decoder(target, draft, draft_tokens=1, prefill_chunk=1)
    decoder.stats, decoder.context = object(), object()
    decoder._phase = "preserved"
    return decoder


@pytest.mark.parametrize("prompt", [
    [-0.5, 7], [2.9, 7], [2.0, 7], ["2", 7], [True, 7],
    [np.bool_(False), 7], [None, 7], [np.nan, 7], [np.inf, 7],
    [], [-1, 7], [8, 7], [1] * 9,
])
def test_invalid_prompt_preserves_decoder_state(validation_decoder, prompt):
    decoder = validation_decoder
    stats, context = decoder.stats, decoder.context
    with pytest.raises(ValueError):
        decoder.generate(prompt, max_new_tokens=0)
    assert decoder.target.resets == 0 and decoder.target.pos == 5
    assert decoder.target.stats == [{"preserved": True}]
    assert decoder.stats is stats and decoder.context is context
    assert decoder._phase == "preserved"


@pytest.mark.parametrize("budget", [
    1.5, 0.0, np.float64(2), np.nan, np.inf, "0", True, np.bool_(False), None, -1,
])
def test_invalid_budget_preserves_decoder_state(validation_decoder, budget):
    decoder = validation_decoder
    stats, context = decoder.stats, decoder.context
    with pytest.raises(ValueError):
        decoder.generate([2, 7], max_new_tokens=budget)
    assert decoder.target.resets == 0 and decoder.target.pos == 5
    assert decoder.target.stats == [{"preserved": True}]
    assert decoder.stats is stats and decoder.context is context
    assert decoder._phase == "preserved"


def test_numpy_integer_prompt_and_zero_budget_are_accepted(validation_decoder):
    decoder = validation_decoder
    result = decoder.generate(np.array([2, 7], dtype=np.int64), max_new_tokens=np.int64(0))
    assert result.tokens == () and result.stop_reason == "length"
    assert result.stats.prompt_tokens == 2 and result.stats.target_calls == 0
    assert decoder.target.resets == 1 and decoder.target.pos == 0
