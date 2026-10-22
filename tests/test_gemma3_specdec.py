"""Bounded ISA qualification of Gemma 3's greedy device prompt-lookup loop.

No checkpoint download, card, host draft, or host logits enter the candidate.
The reference engine is used only by the tests to establish expected tokens.
"""

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm import generate as G
from opentpu.llm.gemma_specdec import (E, V, GemmaPromptLookupDecoder, S_ACC, S_ITER,
                                      S_VER, compile_gen)
from opentpu.llm.qwen3 import Engine
from test_gemma3 import tiny


def _engine(spec, weights, candidate=False, cap=1024):
    return Engine(spec, weights, cap=cap,
                  cfg=board_config(DRAM_BYTES=1 << 24, MCOLS=8, PORTB=4,
                                   ROWLANES=True, PAIR=True, T2=True,
                                   ALINE=64, ACT_ROWS=16, SEQ_WC=True),
                  rows=2, wformat="int8", head_format="int8", resident=True,
                  prog_cache=False, prompt_runs=False,
                  **({"gemma_prompt_lookup": True} if candidate else {}))


def _bits(x):
    return np.asarray(x, np.float32).view(np.uint32)


def test_full_device_loop_matches_plain_greedy_and_resumes(tiny):
    _, weights, spec, _ = tiny
    reference, candidate = _engine(spec, weights), _engine(spec, weights, True)
    dec = GemmaPromptLookupDecoder(candidate)
    prompt = [2, 18, 72, 30, 18, 72, 30]
    reference.prefill(prompt[:-1])
    expected = reference.generate_card(prompt[-1], 6, stop_ids=[])
    got = dec.generate_from_unfed_prompt(prompt, 6, stop_ids=[])
    assert got == expected and candidate.pos == reference.pos
    assert dec.last.tokens == got
    assert dec.last.iterations + dec.last.accepted == len(got)
    assert dec.last.verifies <= dec.last.iterations

    # A subsequent request seeds history from the complete context.  The last
    # output has not yet been fed to either model.
    continuation = prompt + got
    expected2 = reference.generate_card(expected[-1], 3, stop_ids=[])
    got2 = dec.generate_card(got[-1], 3, stop_ids=[], context=continuation)
    assert got2 == expected2 and candidate.pos == reference.pos
    assert dec.last.iterations + dec.last.accepted == len(got2)


def test_budget_one_and_eos_end_without_a_draft(tiny):
    _, weights, spec, _ = tiny
    prompt = [2, 18, 72, 30]
    oracle = _engine(spec, weights)
    oracle.prefill(prompt[:-1])
    first = oracle.generate_card(prompt[-1], 1, stop_ids=[])[0]

    for budget, stops in ((1, []), (5, [first])):
        candidate = _engine(spec, weights, True)
        dec = GemmaPromptLookupDecoder(candidate)
        got = dec.generate_from_unfed_prompt(prompt, budget, stop_ids=stops)
        assert got == [first] and candidate.pos == len(prompt)
        assert dec.last.iterations == 1 and dec.last.accepted == 0
        assert dec.last.verifies == 0


@pytest.mark.parametrize("accept", [False, True], ids=["reject", "accept"])
def test_single_device_verify_updates_only_accepted_prefix(tiny, accept):
    _, weights, spec, _ = tiny
    reference, candidate = _engine(spec, weights), _engine(spec, weights, True)
    GemmaPromptLookupDecoder(candidate)  # write only the immutable position ramp
    prefix = [2, 18, 72, 30, 5, 10, 63, 125]
    reference.prefill(prefix)
    candidate.prefill(prefix)
    p, seed = len(prefix), 31
    a0 = int(np.argmax(reference.step(seed)))
    draft = a0 if accept else (a0 + 1) % spec.vocab
    a = candidate.image.lookup["gemma_sd"]
    g = candidate.image.lookup["gen"]
    hist = np.zeros(a["n"], np.float32)
    hist[:p] = prefix
    hist[p:p + 2] = [seed, draft]
    state = G.state_words(spec, seed, p, 2, [], candidate.block)
    state[G.S_HALT] = 1.0  # finish after this V; it still verifies both input rows
    candidate.backend.write(0, a["hist"], hist)
    candidate.backend.write(0, g["state"], state)
    candidate.backend.write(0, g["out"] + 4 * (p + 1), np.full(2, G.OUT_MARK, np.uint32))
    programs = compile_gen(candidate.image, p // candidate.block + 1, V)
    assert programs and G.fits(candidate.image, programs)
    candidate.backend.run(programs)
    raw = candidate.backend.read(0, g["out"] + 4 * (p + 1), 8).view(np.uint32)
    assert int(raw[0].view(np.float32)) == a0
    if accept:
        assert raw[1] != G.OUT_MARK
        a1 = int(np.argmax(reference.step(draft)))
        assert int(raw[1].view(np.float32)) == a1
        candidate.pos = p + 2
        np.testing.assert_array_equal(_bits(candidate.step(a1)), _bits(reference.step(a1)))
    else:
        assert raw[1] == G.OUT_MARK
        candidate.pos = p + 1
        # This step overwrites the rejected row's positional KV before reading it.
        np.testing.assert_array_equal(_bits(candidate.step(a0)), _bits(reference.step(a0)))
    sw = candidate.backend.read(0, g["state"], 4 * G.STATE_WORDS).view(np.float32)
    assert (int(sw[S_ITER]), int(sw[S_ACC]), int(sw[S_VER])) == (1, int(accept), 1)


@pytest.mark.parametrize("cap", [256, 1024], ids=["last-slot", "next-bucket"])
def test_bucket_boundary_starts_with_e_and_continues_on_device(tiny, cap):
    _, weights, spec, _ = tiny
    reference, candidate = _engine(spec, weights, cap=cap), _engine(spec, weights, True, cap=cap)
    prefix = np.random.default_rng(17).integers(3, spec.vocab, 255).tolist()
    seed = 31
    reference.prefill(prefix)
    candidate.prefill(prefix)
    assert compile_gen(candidate.image, 1, E) is not None
    expected = reference.generate_card(seed, 3, stop_ids=[])
    got = GemmaPromptLookupDecoder(candidate).generate_card(
        seed, 3, stop_ids=[], context=prefix + [seed])
    assert got == expected and candidate.pos == reference.pos == min(cap, 258)


def test_reject_after_sliding_ring_wrap_keeps_next_logits(tiny):
    _, weights, spec, _ = tiny
    reference, candidate = _engine(spec, weights), _engine(spec, weights, True)
    p = 527
    assert candidate.image.ring == 512 and p > candidate.image.ring
    prefix = np.random.default_rng(23).integers(3, spec.vocab, p).tolist()
    reference.prefill(prefix)
    candidate.prefill(prefix)
    seed = 31
    a0 = int(np.argmax(reference.step(seed)))
    draft = (a0 + 1) % spec.vocab
    a, g = candidate.image.lookup["gemma_sd"], candidate.image.lookup["gen"]
    hist = np.zeros(a["n"], np.float32)
    hist[:p] = prefix
    hist[p:p + 2] = [seed, draft]
    state = G.state_words(spec, seed, p, 2, [], candidate.block)
    state[G.S_HALT] = 1.0
    GemmaPromptLookupDecoder(candidate)
    candidate.backend.write(0, a["hist"], hist)
    candidate.backend.write(0, g["state"], state)
    candidate.backend.write(0, g["out"] + 4 * (p + 1), np.full(2, G.OUT_MARK, np.uint32))
    candidate.backend.run(compile_gen(candidate.image, p // candidate.block + 1, V))
    candidate.pos = p + 1
    np.testing.assert_array_equal(_bits(candidate.step(a0)), _bits(reference.step(a0)))


@pytest.mark.parametrize("budget,stops,expected,accepted", [
    (4, [11], [11], 0),
    (4, [12], [11, 12], 1),
    (1, [], [11], 0),
], ids=["first-row-eos", "second-row-eos", "one-output-left"])
@pytest.mark.parametrize("p", [4, 254], ids=["ordinary-slot", "last-two-slots"])
def test_device_verify_eos_and_budget_control(tiny, monkeypatch, budget, stops,
                                            expected, accepted, p):
    """Test only device commit control with fixed head picks, not model numerics.

    Separate tests compare actual head ARGMAX and future logits to plain decode.
    Constants here make both EOS branches reachable independent of random weights.
    """
    from opentpu import language as ol
    _, weights, spec, _ = tiny
    candidate = _engine(spec, weights, True, cap=256)
    GemmaPromptLookupDecoder(candidate)
    seed, draft = 31, 11
    budget = min(budget, candidate.cap - p)
    a, g = candidate.image.lookup["gemma_sd"], candidate.image.lookup["gen"]
    hist = np.zeros(a["n"], np.float32)
    hist[p:p + 2] = [seed, draft]
    state = G.state_words(spec, seed, p, budget, stops, candidate.block)
    candidate.backend.write(0, a["hist"], hist)
    candidate.backend.write(0, g["state"], state)
    slots = min(4, candidate.cap - p)
    candidate.backend.write(0, g["out"] + 4 * (p + 1), np.full(slots, G.OUT_MARK, np.uint32))
    picks = iter((11, 12))
    monkeypatch.setattr(G.Greedy, "token", lambda self: ol.full([1], float(next(picks))))
    candidate.backend.run(compile_gen(candidate.image, 1, V))
    raw = candidate.backend.read(0, g["out"] + 4 * (p + 1), 4 * slots).view(np.uint32)
    assert raw[:len(expected)].view(np.float32).astype(int).tolist() == expected
    assert (raw[len(expected):] == G.OUT_MARK).all()
    sw = candidate.backend.read(0, g["state"], 4 * G.STATE_WORDS).view(np.float32)
    assert int(sw[G.S_TOK]) == expected[-1]
    assert int(sw[G.S_TPOS]) == (p + len(expected)) % candidate.block
    assert int(sw[G.S_LEFT]) == budget - len(expected)
    assert (int(sw[S_ITER]), int(sw[S_ACC]), int(sw[S_VER])) == (1, accepted, 1)


def test_zero_budget_and_invalid_requests_do_not_mutate_prefix(tiny):
    _, weights, spec, _ = tiny
    candidate = _engine(spec, weights, True)
    dec = GemmaPromptLookupDecoder(candidate)
    before = len(candidate.stats)
    assert dec.generate_from_unfed_prompt([2, 18], 0) == []
    for prompt, budget in [([True], 1), ([2.5], 1), ([2], -1), ([2], True),
                           ([2] * 1025, 1)]:
        with pytest.raises(ValueError):
            dec.generate_from_unfed_prompt(prompt, budget)
    for context in ([True], [2.5], [], [2, 3]):
        with pytest.raises(ValueError):
            dec.generate_card(2, 1, context=context)
    assert candidate.pos == 0 and len(candidate.stats) == before
