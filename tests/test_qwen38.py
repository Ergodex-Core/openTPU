"""Qwen3.8-27B-INT4's paths on a tiny Qwen3.5 (docs/qwen38.md): the truncated Spec, and the
embedding table on the host (embed_host) served to every slice of a tensor-parallel decode."""
import dataclasses

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm.qwen3 import Engine
from test_qwen35 import KINDS, _tiny_model


@pytest.fixture(scope="module")
def tiny():
    return _tiny_model(4)


def test_truncated_spec(tiny):
    _, _, spec = tiny
    t = spec.truncated(3)
    assert t.kinds == KINDS[:3] and t.layers == 3 and t.hidden == spec.hidden
    with pytest.raises(ValueError):
        spec.truncated(len(KINDS) + 1)


def test_embed_host_rows_serve_every_slice(tiny):
    """A dense model whose int8 embedding table stays on the host (embed_host: Qwen3.8-27B's
    4-bit image does not fit 4 GiB a slice with it on the card) on two slices: the decode
    loop on the device asks the host for each token's row; every slice posts the request in
    lockstep and waits on its own served word, so the row server writes every slice's slot
    and mailbox (offload.BackendDram slices). generate_card gives the resident step's
    tokens."""
    _, W, spec = tiny
    W = dict(W, **{"lm_head.weight": W["model.embed_tokens.weight"] * np.float32(1.25)})
    spec = dataclasses.replace(spec, tied=False, embed="int8")
    cfg = board_config(S=2, DRAM_BYTES=1 << 25, PAIR=True)
    a = Engine(spec, W, cap=256, cfg=cfg, resident=True, embed_host=True)
    b = Engine(spec, W, cap=256, cfg=cfg, resident=True)
    assert a.image.embed_host and not b.image.embed_host and a.row_server is not None
    assert a.row_server.mem.slices == [0, 1]
    toks = [int(t) for t in np.random.default_rng(5).integers(0, 1000, 20)]
    t0 = int(np.argmax(a.prefill(toks)))
    assert int(np.argmax(b.prefill(toks))) == t0
    ref, t = [], t0
    for _ in range(6):
        t = int(np.argmax(b.step(t)))
        ref.append(t)
    assert a.generate_card(t0, 6, stop_ids=[]) == ref
    assert a.row_server.seq >= 5
