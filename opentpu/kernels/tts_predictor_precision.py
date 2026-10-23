"""Device-only predictor precision experiment using existing MM/VOP/LD/ST.

Two INT8 weight terms and two activation terms, plus unquantized short-cache
attention. No host residual activations or neural computation are involved.
"""
import math
from .. import language as ol
from ..compiler import current
from .lib import rmsnorm, rope, silu
from .mlp import down_chunk


def activation_pair(x, identity):
    high = ol.quantize(x)
    # ACT RAM has no load/dequantize instruction. An exact integer identity MM
    # reconstructs the represented operand on device, including MX drop rules.
    reconstructed = ol.dot(high, identity)
    low = ol.quantize(x - reconstructed)
    return high, low


def corrected_dot(pair, high, low):
    ah, al = pair
    result = ol.dot(ah, high)
    ol.dot(ah, low, acc=result)
    ol.dot(al, high, acc=result)
    ol.dot(al, low, acc=result)
    return result


@ol.jit
def predictor_precision_layer(m, position: int):
    s = m.spec
    x = ol.load(m.output)
    pair = activation_pair(rmsnorm(x, ol.load(m.input_norm), s.eps), m.identities[s.hidden])
    q = rmsnorm(corrected_dot(pair, m.wq, m.wq_residual).reshape(s.n_q, s.head_dim), ol.load(m.q_norm), s.eps)
    k = rmsnorm(corrected_dot(pair, m.wk, m.wk_residual).reshape(s.n_kv, s.head_dim), ol.load(m.k_norm), s.eps)
    v = corrected_dot(pair, m.wv, m.wv_residual).reshape(s.n_kv, s.head_dim)
    cos, sin = ol.load(m.cosine[position, :]), ol.load(m.sine[position, :])
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    context = ol.empty((s.n_q, s.head_dim), dense=True)
    groups = s.n_q // s.n_kv
    for head in ol.static_range(s.n_kv):
        ol.store(m.kfp[head][position:position + 1, :], k[head:head + 1, :])
        ol.store(m.vfp[head][position:position + 1, :], v[head:head + 1, :])
        keys = ol.load(m.kfp[head][0:position + 1, :])
        for j in ol.static_range(groups):
            first = head * groups + j
            scores = ol.sum(keys * q[first, :][None, :], axis=1) * (ol.LOG2E / math.sqrt(s.head_dim))
            p = ol.exp2(scores - ol.max(scores))
            p = p * ol.recip(ol.sum(p))
            acc = ol.zeros((1, s.head_dim))
            for t in ol.static_range(position + 1):
                value = ol.load(m.vfp[head][t:t + 1, :])
                acc.set(acc + value * p[t:t + 1][:, None])
            context[first:first + 1, :].set(acc)
    pair = activation_pair(context.reshape(1, s.n_q * s.head_dim), m.identities[s.n_q * s.head_dim])
    residual = x + corrected_dot(pair, m.wo, m.wo_residual)
    ol.store(m.attention_residual, residual)
    pair = activation_pair(rmsnorm(residual, ol.load(m.post_norm), s.eps), m.identities[s.hidden])
    chunk = down_chunk(s.ffn, ol.block_size(), 'int8')
    output = None
    for start in ol.static_range(0, s.ffn, chunk):
        stop = start + chunk
        gate = silu(corrected_dot(pair, m.gate[start:stop, :], m.gate_residual[start:stop, :]))
        up = corrected_dot(pair, m.up[start:stop, :], m.up_residual[start:stop, :])
        down_pair = activation_pair(gate * up, m.identities[chunk])
        term = corrected_dot(down_pair, m.down[:, start:stop], m.down_residual[:, start:stop])
        output = term if output is None else output + term
    ol.store(m.output, residual + output)


@ol.jit
def predictor_precision_head(m, p, head: int):
    normalized = rmsnorm(ol.load(p.output), ol.load(p.final_norm), p.spec.eps)
    ol.store(p.normalized, normalized)
    logits = corrected_dot(activation_pair(normalized, p.identities[p.spec.hidden]), p.heads[head], p.heads_residual[head])
    ol.store(p.logits, logits)
    best = current().argmax(logits)
    ol.store(p.greedy, best)
    ol.store(m.code_ids[head + 1:head + 2], best[1:2])
