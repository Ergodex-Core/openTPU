"""ISA check for the separate, two-row Gemma 3 device-greedy verifier."""

import numpy as np
import pytest

from opentpu.compiler import vt_tile
from opentpu import isa as I
from opentpu.isasim import board_config
from opentpu.llm import gemma3
from opentpu.llm.gemma_verifier import compile_two_row_verify
from opentpu.llm.qwen3 import Engine
from test_gemma3 import tiny


def _engine(spec, weights):
    return Engine(spec, weights, cap=512,
                  cfg=board_config(DRAM_BYTES=1 << 24), rows=2,
                  wformat="int8", head_format="int8", resident=True,
                  prog_cache=False, prompt_runs=False)


def _live_kv(eng, positions):
    """Snapshot only committed K/V slots and their scales."""
    image, spec = eng.image, eng.spec
    dram = eng.backend.machine.slices[0].dram
    live = []
    for layer in range(spec.layers):
        if spec.kv_src[layer] != layer:
            continue
        kind = spec.kinds[layer]
        ck = image.ring if kind == gemma3.SLIDE else image.cap
        tb = vt_tile(ck)
        d, ks = spec.hd(layer), image.ks[kind]
        base = image._off(layer).const
        for head in image.bofs[image._key(layer)]["kv"]:
            for pos in positions:
                slot = pos % ck
                k = base + head["k"] + slot * ks
                vs = base + head["vs"] + 4 * slot
                q, r = divmod(slot, tb)
                vt = base + head["vt"] + q * d * tb + r
                live.append((dram[k:k + ks].tobytes(),
                             dram[vt + np.arange(d) * tb].tobytes(),
                             dram[vs:vs + 4].tobytes()))
    return live


def test_verifier_ids_match_full_heads_and_kv(tiny, monkeypatch, record_property):
    _, weights, spec, _ = tiny
    full, verify = _engine(spec, weights), _engine(spec, weights)
    assert full.device_inputs and verify.device_inputs
    prefix = [2, 18, 72, 30]
    for token in prefix:
        full.step(token)
        verify.step(token)
    assert _live_kv(full, range(len(prefix))) == _live_kv(verify, range(len(prefix)))

    pos, seed, draft = len(prefix), 31, 47
    # The reference writes two full vocabulary heads.  The verifier never reads
    # those heads or the observed picks; both IDs are obtained by ISA ARGMAX.
    reference = full.image.compile_rows([(0, pos), (0, pos + 1)], [0, 1],
                                        tokens=[seed, draft])
    full.backend.run(reference)
    heads = full.backend.read(0, full.image.io["logits"], 8 * spec.vocab)
    expected = heads.view(np.float32).reshape(2, spec.vocab).argmax(axis=1)

    before_layout = dict(verify.image.io)
    program = compile_two_row_verify(verify.image, pos, [seed, draft])
    assert len(program) == 1 and program[0]
    assert verify.image.io == before_layout
    # Inspect instructions actually executed by the ISA slice, including any
    # hardware loops. M is the logical stationary-row count, not an inferred
    # number of native reads or a throughput measurement.
    slice0 = verify.backend.machine.slices[0]
    old_mm = slice0._mm
    executed_m2 = []

    def trace_mm(ins):
        wf, pair, quad = I.mm_format(ins)
        m = (ins.w[5] >> 16) & 0xFF
        n, kb = ins.w[3] & 0xFFFF, (ins.w[3] >> 16) & 0xFF
        if m == 2 and not pair and not quad and n > 1 and kb > 0:
            executed_m2.append((wf, n, kb))
        return old_mm(ins)

    monkeypatch.setattr(slice0, "_mm", trace_mm)
    verify.backend.run(program)
    assert executed_m2, "the verifier executed no non-PAIR/QUAD two-row MM"
    record_property("verifier_isa_m2_mm_count", len(executed_m2))
    record_property("verifier_isa_m2_mm_example", executed_m2[0])
    ids = verify.backend.read(0, verify.image.io["logits"], 8).view(np.float32)
    np.testing.assert_array_equal(ids, expected.astype(np.float32))
    assert _live_kv(full, range(pos + 2)) == _live_kv(verify, range(pos + 2))


def test_verifier_rejects_invalid_inputs(tiny):
    _, weights, spec, _ = tiny
    eng = _engine(spec, weights)
    for pos, tokens in [(-1, [1, 2]), (511, [1, 2]), (True, [1, 2]),
                        (0, [1]), (0, [1, spec.vocab]), (0, [1, -1]),
                        (0, [1, True])]:
        with pytest.raises(ValueError):
            compile_two_row_verify(eng.image, pos, tokens)
    no_lookup = spec.image(eng.image.cfg, cap=512, rows=2, lookup=False)
    with pytest.raises(ValueError, match="device inputs"):
        compile_two_row_verify(no_lookup, 0, [1, 2])


def test_verifier_rejects_non_gemma_image():
    with pytest.raises(ValueError, match="Gemma 3"):
        compile_two_row_verify(object(), 0, [1, 2])
