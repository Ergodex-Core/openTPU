"""Tokenizer decoder transformer, FP32 ring KV and corrected MX affines."""
import math
from .. import language as ol
from .lib import rmsnorm, rope, silu
from .mlp import down_chunk
from .tts_predictor_precision import activation_pair, corrected_dot


@ol.jit
def decoder_input(m):
    x = corrected_dot(activation_pair(ol.load(m.input), m.identities[m.spec.latent]), m.input_proj, m.input_proj_residual)
    ol.store(m.output, x + ol.load(m.input_bias)[None, :])


@ol.jit
def decoder_qkv(m, position: int):
    s = m.spec
    pair = activation_pair(rmsnorm(ol.load(m.output), ol.load(m.input_norm), s.eps), m.identities[s.hidden])
    # The tokenizer decoder has Identity Q/K norms, unlike the talker.
    q = corrected_dot(pair, m.wq, m.wq_residual).reshape(s.heads, s.head_dim)
    k = corrected_dot(pair, m.wk, m.wk_residual).reshape(s.heads, s.head_dim)
    v = corrected_dot(pair, m.wv, m.wv_residual).reshape(s.heads, s.head_dim)
    cos, sin = ol.load(m.cosine[position, :]), ol.load(m.sine[position, :])
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    ol.store(m.query, q)
    slot = position % s.window
    for head in ol.static_range(s.heads):
        ol.store(m.keys[head][slot:slot + 1, :], k[head:head + 1, :])
        ol.store(m.values[head][slot:slot + 1, :], v[head:head + 1, :])


@ol.jit
def decoder_attention_head(m, position: int, head: int):
    s = m.spec
    n = min(position + 1, s.window)
    first = position + 1 - n
    q = ol.load(m.query[head:head + 1, :])
    scores = ol.empty((n,), dense=True)
    # Traverse ring slots in chronological source order. The source mask admits
    # current and preceding window-1 keys, never an uninitialized/future slot.
    offset = 0
    while offset < n:
        slot = (first + offset) % s.window
        count = min(32, n - offset, s.window - slot)
        keys = ol.load(m.keys[head][slot:slot + count, :])
        score = ol.sum(keys * q[0, :][None, :], axis=1) * (ol.LOG2E / math.sqrt(s.head_dim))
        scores[offset:offset + count].set(score)
        offset += count
    p = ol.exp2(scores - ol.max(scores))
    p = p * ol.recip(ol.sum(p))
    value = ol.zeros((1, s.head_dim))
    for index in ol.static_range(n):
        slot = (first + index) % s.window
        v = ol.load(m.values[head][slot:slot + 1, :])
        value.set(value + v * p[index:index + 1][:, None])
    ol.store(m.context[head:head + 1, :], value)


@ol.jit
def decoder_finish(m):
    s = m.spec
    branch = corrected_dot(activation_pair(ol.load(m.context).reshape(1, s.heads * s.head_dim), m.identities[s.heads * s.head_dim]), m.wo, m.wo_residual)
    residual = ol.load(m.output) + branch * ol.load(m.attention_scale)[None, :]
    ol.store(m.attention_residual, residual)
    pair = activation_pair(rmsnorm(residual, ol.load(m.post_norm), s.eps), m.identities[s.hidden])
    chunk = down_chunk(s.ffn, ol.block_size(), 'int8')
    result = None
    for start in ol.static_range(0, s.ffn, chunk):
        stop = start + chunk
        gate = silu(corrected_dot(pair, m.gate[start:stop, :], m.gate_residual[start:stop, :]))
        up = corrected_dot(pair, m.up[start:stop, :], m.up_residual[start:stop, :])
        term = corrected_dot(activation_pair(gate * up, m.identities[chunk]), m.down[:, start:stop], m.down_residual[:, start:stop])
        result = term if result is None else result + term
    ol.store(m.output, residual + result * ol.load(m.mlp_scale)[None, :])


@ol.jit
def decoder_output(m):
    x = rmsnorm(ol.load(m.output), ol.load(m.final_norm), m.spec.eps)
    ol.store(m.normalized, x)
    result = corrected_dot(activation_pair(x, m.identities[m.spec.hidden]), m.output_proj, m.output_proj_residual)
    ol.store(m.transformed, result + ol.load(m.output_bias)[None, :])
