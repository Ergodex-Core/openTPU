"""Resident text/codec embeddings, the existing projection, and one TTS talker block.

The block uses text-only (three equal position axes) rotary embeddings. It does
not implement the complete CustomVoice prompt or the frame-level generation loop.
"""
import math

from .. import isa as I
from .. import language as ol
from ..compiler import Affine, DevVar, Tensor, current
from .attention import _attend
from .lib import rmsnorm, rope
from .mlp import swiglu_down
from .tts_projection import text_projection


def _embedding(table, token):
    """Read a resident fp32 row; address multiplication is an ISA RLD operation."""
    b = current()
    register = b.scratch()
    b.emit(I.rld(register, token.base, mul=4 * table.strides[0],
                 comment="Qwen TTS resident embedding row"))
    row = ol.load(Tensor(Affine.of(table.base) + DevVar("tts embedding", register),
                         (1, table.shape[1]), table.strides))
    b.unscratch(register)
    return row


@ol.jit
def talker_block(m, position: int, attention_block: int):
    """One aligned text+codec input row -> layer-0 output and device-resident KV."""
    spec = m.spec
    ids = ol.load(m.ids)
    # Reuse the previously validated projection verbatim. These scratch transfers
    # are device DRAM -> TMEM -> device DRAM, never host round trips.
    ol.store(m.text_scratch, _embedding(m.text_embedding, ids[0:1]))
    text_projection.fn(m.text_scratch, m.fc1, m.bias1, m.fc2, m.bias2, m.projected)
    x = ol.load(m.projected) + _embedding(m.codec_embedding, ids[1:2])
    ol.store(m.assembled, x)

    xs = ol.quantize(rmsnorm(x, ol.load(m.input_norm), spec.eps))
    q = rmsnorm(ol.dot(xs, m.wq).reshape(spec.n_q, spec.head_dim),
                ol.load(m.q_norm), spec.eps)
    k = rmsnorm(ol.dot(xs, m.wk).reshape(spec.n_kv, spec.head_dim),
                ol.load(m.k_norm), spec.eps)
    v = ol.dot(xs, m.wv).reshape(spec.n_kv, spec.head_dim)
    cosine, sine = ol.load(m.cosine[position, :]), ol.load(m.sine[position, :])
    q, k = rope(q, cosine, sine), rope(k, cosine, sine)
    for head in ol.static_range(spec.n_kv):
        ol.kv_append(m.kv, head, position, k[head:head + 1, :], v[head:head + 1, :])

    groups = spec.n_q // spec.n_kv
    context = ol.empty((spec.n_q, spec.head_dim), dense=True)
    for head in ol.static_range(spec.n_kv):
        first = head * groups
        attended = _attend(q[first:first + groups, :], m.kv, head,
                           position + 1, attention_block,
                           scale=ol.LOG2E / math.sqrt(spec.head_dim))
        context[first:first + groups, :].set(attended)
    residual = x + ol.dot(context.reshape(1, spec.n_q * spec.head_dim), m.wo)
    ol.store(m.attention_residual, residual)
    normalized = ol.quantize(rmsnorm(residual, ol.load(m.post_norm), spec.eps))
    # The bounded layer fits IMEM with static chunks. A loop-dependent column
    # slice of the monolithic scaled down matrix is not an ISA MM immediate.
    out = residual + swiglu_down(normalized, m.gate, m.up, m.down)
    ol.store(m.output, out)
