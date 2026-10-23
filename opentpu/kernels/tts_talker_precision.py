"""Split corrected talker programs using the existing shared ISA."""
import math
from .. import language as ol
from ..compiler import current
from .lib import rmsnorm, rope, silu
from .mlp import down_chunk
from .tts_talker import _embedding
from .tts_predictor_precision import activation_pair, corrected_dot


def project(m, text):
    first = corrected_dot(activation_pair(text, m.identities[m.spec.text_hidden]), m.fc1, m.fc1_residual) + ol.load(m.bias1)[None, :]
    return corrected_dot(activation_pair(silu(first), m.identities[m.spec.text_hidden]), m.fc2, m.fc2_residual) + ol.load(m.bias2)[None, :]


@ol.jit
def corrected_input(m, role: bool):
    ids = ol.load(m.ids)
    projected = project(m, _embedding(m.text_embedding, ids[0:1]))
    ol.store(m.projected, projected)
    x = projected if role else projected + _embedding(m.codec_embedding, ids[1:2])
    ol.store(m.assembled, x)
    ol.store(m.output, x)


@ol.jit
def corrected_feedback(m):
    projected = project(m, _embedding(m.text_embedding, ol.load(m.pad_id)))
    ol.store(m.projected, projected)
    x = ol.load(m.codec_sum) + projected
    ol.store(m.assembled, x)
    ol.store(m.output, x)


@ol.jit
def corrected_qkv(m, position: int):
    s = m.spec
    x = ol.load(m.output)
    pair = activation_pair(rmsnorm(x, ol.load(m.input_norm), s.eps), m.identities[s.hidden])
    q = rmsnorm(corrected_dot(pair, m.wq, m.wq_residual).reshape(s.n_q, s.head_dim), ol.load(m.q_norm), s.eps)
    k = rmsnorm(corrected_dot(pair, m.wk, m.wk_residual).reshape(s.n_kv, s.head_dim), ol.load(m.k_norm), s.eps)
    v = corrected_dot(pair, m.wv, m.wv_residual).reshape(s.n_kv, s.head_dim)
    cos, sin = ol.load(m.cosine[position, :]), ol.load(m.sine[position, :])
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    ol.store(m.q_current, q)
    for head in ol.static_range(s.n_kv):
        ol.store(m.kfp[head][position:position + 1, :], k[head:head + 1, :])
        ol.store(m.vfp[head][position:position + 1, :], v[head:head + 1, :])


@ol.jit
def corrected_attention_head(m, position: int, head: int):
    s, n = m.spec, position + 1
    groups = s.n_q // s.n_kv
    q = ol.load(m.q_current)
    context = ol.empty((groups, s.head_dim), dense=True)
    for j in ol.static_range(groups):
        first = head * groups + j
        scores = ol.empty((n,), dense=True)
        # Keep K tiles bounded even when the resident cache contains 512 rows.
        for start in ol.static_range(0, n, 64):
            stop = min(start + 64, n)
            keys = ol.load(m.kfp[head][start:stop, :])
            values = ol.sum(keys * q[first, :][None, :], axis=1) * (ol.LOG2E / math.sqrt(s.head_dim))
            scores[start:stop].set(values)
        p = ol.exp2(scores - ol.max(scores))
        p = p * ol.recip(ol.sum(p))
        acc = ol.zeros((1, s.head_dim))
        for t in ol.static_range(n):
            value = ol.load(m.vfp[head][t:t + 1, :])
            acc.set(acc + value * p[t:t + 1][:, None])
        context[j:j + 1, :].set(acc)
    first = head * groups
    ol.store(m.context[first:first + groups, :], context)


@ol.jit
def corrected_finish(m):
    s = m.spec
    x = ol.load(m.output)
    context = ol.load(m.context).reshape(1, s.n_q * s.head_dim)
    residual = x + corrected_dot(activation_pair(context, m.identities[s.n_q * s.head_dim]), m.wo, m.wo_residual)
    ol.store(m.attention_residual, residual)
    pair = activation_pair(rmsnorm(residual, ol.load(m.post_norm), s.eps), m.identities[s.hidden])
    chunk = down_chunk(s.ffn, ol.block_size(), 'int8')
    output = None
    for start in ol.static_range(0, s.ffn, chunk):
        stop = start + chunk
        gate = silu(corrected_dot(pair, m.gate[start:stop, :], m.gate_residual[start:stop, :]))
        up = corrected_dot(pair, m.up[start:stop, :], m.up_residual[start:stop, :])
        term = corrected_dot(activation_pair(gate * up, m.identities[chunk]), m.down[:, start:stop], m.down_residual[:, start:stop])
        output = term if output is None else output + term
    ol.store(m.output, residual + output)


@ol.jit
def corrected_codec_logits(m, audio_vocab: int, eos: int):
    normalized = rmsnorm(ol.load(m.output), ol.load(m.final_norm), m.spec.eps)
    ol.store(m.normalized, normalized)
    logits = corrected_dot(activation_pair(normalized, m.identities[m.spec.hidden]), m.codec_head, m.codec_head_residual)
    ol.store(m.logits, logits)
    ol.store(m.greedy_raw, current().argmax(logits))
    masked = ol.empty(logits.shape, dense=True)
    masked.set(logits)
    if audio_vocab < eos:
        masked[:, audio_vocab:eos].set(float('-inf'))
    if eos + 1 < m.spec.codec_vocab:
        masked[:, eos + 1:].set(float('-inf'))
    ol.store(m.greedy_masked, current().argmax(masked))
