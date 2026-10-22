"""Bounded ISA qualification of a two-row Gemma3 verifier candidate.

This proves numerical and live KV equivalence for a proposed seed/draft row
pair. It does not implement a speculative decoder, a card run, or a speed claim.
"""

import numpy as np
import pytest

from opentpu.compiler import vt_tile
from opentpu.isasim import board_config
from opentpu.llm import gemma3 as G
from opentpu.llm.qwen3 import Engine
from test_gemma3 import tiny


def _engine(spec, weights):
    return Engine(spec, weights, cap=1024,
                  cfg=board_config(DRAM_BYTES=1 << 24, MCOLS=8, PORTB=4, ROWLANES=True, PAIR=True,
                                   T2=True, ALINE=64, ACT_ROWS=16, SEQ_WC=True), rows=2,
                  wformat="int8", head_format="int8", resident=True,
                  prog_cache=False, prompt_runs=False)


def _bits(values):
    return np.asarray(values, np.float32).view(np.uint32)


def _live_kv(eng, positions):
    """Only committed positions' K rows, V columns and scales, never future slots."""
    image, spec = eng.image, eng.spec
    dram = eng.backend.machine.slices[0].dram
    live = []
    for layer in range(spec.layers):
        if spec.kv_src[layer] != layer:
            continue
        kind = spec.kinds[layer]
        ck = image.ring if kind == G.SLIDE else image.cap
        tb = vt_tile(ck)
        d = spec.hd(layer)
        ks = image.ks[kind]
        base = image._off(layer).const
        for head in image.bofs[image._key(layer)]["kv"]:
            for pos in positions:
                slot = pos % ck
                k = base + head["k"] + slot * ks
                vs = base + head["vs"] + 4 * slot
                q, r = divmod(slot, tb)
                vt = base + head["vt"] + q * d * tb + r
                live.append((layer, pos,
                             dram[k:k + ks].tobytes(),
                             dram[vt + np.arange(d) * tb].tobytes(),
                             dram[vs:vs + 4].tobytes()))
    assert live
    return live


@pytest.mark.parametrize("position", [8, 255, 527],
                         ids=["short", "bucket-boundary", "ring-wrap"])
@pytest.mark.parametrize("accept", [True, False], ids=["draft-accepted", "draft-rejected"])
def test_two_row_logits_and_committed_kv_match_sequential(tiny, accept, position,
                                                       record_property):
    _, weights, spec, _ = tiny
    ref, pair, clean = (_engine(spec, weights) for _ in range(3))
    assert ref.image.rows == pair.image.rows == 2
    assert ref.image.head_format == pair.image.head_format == "int8"
    assert ref.device_inputs and pair.device_inputs
    if position == 527:
        assert pair.image.ring == 512 and position > pair.image.ring
    prefix = np.random.default_rng(19).integers(3, spec.vocab, position).tolist()
    for token in prefix:
        np.testing.assert_array_equal(_bits(ref.step(token)), _bits(pair.step(token)))
        np.testing.assert_array_equal(_bits(ref.backend.read(
            0, ref.image.io["logits"], 4 * spec.vocab).view(np.float32)),
            _bits(clean.step(token)))
    p = len(prefix)
    assert ref.pos == pair.pos == p
    assert _live_kv(ref, range(p)) == _live_kv(pair, range(p))

    seed = 31
    first = ref.step(seed)
    np.testing.assert_array_equal(_bits(first), _bits(clean.step(seed)))
    correct = int(np.argmax(first))
    draft = correct if accept else (correct + 1) % spec.vocab
    assert (draft == correct) == accept
    second = ref.step(draft)
    seq_instructions = sum(int(s["instructions"][0]) for s in ref.stats[-2:])

    # Direct ISA run deliberately bypasses Engine.prefill_chunks: its single
    # two-row program must produce both heads, and Engine.pos is advanced only
    # after that explicit run. No observed logits enter the program.
    programs = pair.image.compile_rows([(0, p), (0, p + 1)], [0, 1],
                                        tokens=[seed, draft])
    assert len(programs) == 1 and programs[0]
    st = pair.backend.run(programs)
    got = pair.backend.read(0, pair.image.io["logits"], 8 * spec.vocab)
    got = got.view(np.uint32).reshape(2, spec.vocab)
    np.testing.assert_array_equal(got[0], _bits(first))
    np.testing.assert_array_equal(got[1], _bits(second))
    pair.pos = p + 2
    assert ref.pos == pair.pos
    assert _live_kv(ref, range(p + 2)) == _live_kv(pair, range(p + 2))

    assert st["instructions"][0] > 0 and seq_instructions > 0
    record_property("two_row_isa_instructions", int(st["instructions"][0]))
    record_property("two_sequential_isa_instructions", seq_instructions)
    # IsaBackend reports instructions but no DRAM read-byte counter. Do not
    # infer shared weight reads or speed from the lower instruction count.

    if not accept:
        # A rejected draft is not committed. Both engines discard its logical
        # position, then feed the verifier's correct row-0 choice there.
        assert pair.rollback(p + 1) == p + 1
        # This oracle never saw the rejected draft. Equality against two
        # rollback paths could otherwise hide the same stale-cache error.
        assert clean.pos == p + 1
        repaired_pair = pair.step(correct)
        repaired_ref = clean.step(correct)
        np.testing.assert_array_equal(_bits(repaired_pair), _bits(repaired_ref))
        assert pair.pos == clean.pos == p + 2
        assert _live_kv(clean, range(p + 2)) == _live_kv(pair, range(p + 2))
    else:
        np.testing.assert_array_equal(_bits(second), _bits(clean.step(correct)))
        assert _live_kv(clean, range(p + 2)) == _live_kv(pair, range(p + 2))
    # A reset must make every future row independent of rejected/dead entries.
    pair.reset()
    clean.reset()
    np.testing.assert_array_equal(_bits(pair.step(42)), _bits(clean.step(42)))
    assert _live_kv(pair, range(1)) == _live_kv(clean, range(1))
