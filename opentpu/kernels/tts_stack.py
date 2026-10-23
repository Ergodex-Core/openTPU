"""Sequential talker layers and final codec logits/greedy; no generation service."""
import math
from .. import language as ol
from ..compiler import current
from .attention import _attend
from .lib import rmsnorm, rope
from .mlp import swiglu_down
from .tts_talker import _embedding
from .tts_projection import text_projection


@ol.jit
def talker_role_row(m):
    """Source role prefix: projected text only, with no codec embedding added."""
    ids = ol.load(m.ids)
    ol.store(m.text_scratch, _embedding(m.text_embedding, ids[0:1]))
    text_projection.fn(m.text_scratch, m.fc1, m.bias1, m.fc2, m.bias2, m.projected)
    projected = ol.load(m.projected)
    ol.store(m.assembled, projected)
    ol.store(m.output, projected)


@ol.jit
def talker_layer(m, position: int, attention_block: int):
    spec = m.spec
    x = ol.load(m.output)
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


@ol.jit
def codec_logits(m, audio_vocab: int, eos: int):
    normalized = rmsnorm(ol.load(m.output), ol.load(m.final_norm), m.spec.eps)
    ol.store(m.normalized, normalized)
    logits = ol.dot(normalized, m.codec_head)
    ol.store(m.logits, logits)
    ol.store(m.greedy_raw, current().argmax(logits))
    # Fixed source suppression only; repetition penalty/sampling are not implied.
    masked = ol.empty(logits.shape, dense=True)
    masked.set(logits)
    if audio_vocab < eos:
        masked[:, audio_vocab:eos].set(float('-inf'))
    if eos + 1 < m.spec.codec_vocab:
        masked[:, eos + 1:].set(float('-inf'))
    ol.store(m.greedy_masked, current().argmax(masked))
