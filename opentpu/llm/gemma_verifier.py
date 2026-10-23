"""A two-row, greedy Gemma 3 verifier compiled entirely to the existing ISA.

The caller supplies the seed and proposed draft token IDs at compile time.  This
is a single verification run, not a speculative decode loop: deciding whether to
accept the draft and advancing the logical KV position remain the caller's job.
"""

from __future__ import annotations

from numbers import Integral

from .. import language as ol
from ..compiler import current
from . import gemma3
from . import generate as G
from .gemma4 import Image, gemma4_step
from .qwen3 import head_rows_chunk


@ol.jit
def _verify_two_rows(m, pos, tokens):
    b = current()
    m.lm_sinks = [G.Greedy(b, m.v_loc, head_rows_chunk(2)) for _ in range(2)]
    gemma4_step.fn(m, [pos, pos + 1], [0, 1], m.block, tokens)
    ids = [sink.token() for sink in m.lm_sinks]
    m.lm_sinks = None
    # The old logits area is large enough for two rows.  Only its first two fp32
    # words are the result of this program; no full vocabulary row is stored.
    for r, token in enumerate(ids):
        ol.store(m.logitsr[0, r:r + 1], token)


def compile_two_row_verify(image: Image, pos: int, tokens) -> list:
    """Compile one Gemma 3 verifier program; return two IDs as fp32 at logits[0:2].

    `tokens` is (seed, draft), with those IDs gathered from the resident device
    embedding table.  Both heads perform greedy ARGMAX in the program.  The caller
    must have prepared the KV prefix through `pos - 1` before running it.
    """
    if not isinstance(image, Image) or type(image.spec) is not gemma3.Spec:
        raise ValueError("the verifier requires a Gemma 3 image")
    if image.cfg.S != 1 or image.batch != 1 or image.rows < 2 or not image.lookup:
        raise ValueError("the verifier requires one slice, two I/O rows, and device inputs")
    if image.spec.experts or image.spec.ple_dim:
        raise ValueError("the verifier requires a dense Gemma 3 text model")
    if not isinstance(pos, Integral) or isinstance(pos, bool) or not 0 <= pos < image.cap - 1:
        raise ValueError("the verifier needs two consecutive in-capacity positions")
    if not isinstance(tokens, (tuple, list)) or len(tokens) != 2 or any(
            not isinstance(t, Integral) or isinstance(t, bool) or
            not 0 <= t < image.spec.vocab for t in tokens):
        raise ValueError("the verifier needs two in-vocabulary seed and draft IDs")
    m = image.descriptors(0)
    m.fill = False
    return [_verify_two_rows.trace(image.cfg, 0, {"m": m, "pos": int(pos),
                                                   "tokens": [int(t) for t in tokens]}).finish()]
