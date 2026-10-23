"""Resident decoder RVQ and causal pre-convolution; no host learned arithmetic."""
from .. import language as ol
from .tts_talker import _embedding
from .tts_predictor_precision import activation_pair, corrected_dot


def codebook_row(m, book, token):
    row = _embedding(m.embeddings[book], token)
    usage = _embedding(m.usages[book], token).reshape(1, 1)[0, :]
    scale = ol.recip(ol.maximum(usage, m.spec.epsilon))
    return row * scale[:, None]


@ol.jit
def rvq_lookup(m):
    ids = ol.load(m.ids)
    semantic = codebook_row(m, 0, ids[0:1])
    acoustic = ol.zeros((1, m.spec.code_dim))
    for book in ol.static_range(1, m.spec.codebooks):
        acoustic.set(acoustic + codebook_row(m, book, ids[book:book + 1]))
    ol.store(m.semantic, semantic)
    ol.store(m.acoustic, acoustic)


@ol.jit
def rvq_project(m):
    semantic = corrected_dot(activation_pair(ol.load(m.semantic), m.identities[m.spec.code_dim]), m.first, m.first_residual)
    acoustic = corrected_dot(activation_pair(ol.load(m.acoustic), m.identities[m.spec.code_dim]), m.rest, m.rest_residual)
    ol.store(m.latent, semantic + acoustic)


@ol.jit
def decoder_preconv(m):
    oldest, previous, current = ol.load(m.history0), ol.load(m.history1), ol.load(m.latent)
    frames = [oldest, previous, current]
    output = None
    for tap in ol.static_range(3):
        term = corrected_dot(activation_pair(frames[tap], m.identities[m.spec.rvq_hidden]), m.conv[tap], m.conv_residual[tap])
        output = term if output is None else output + term
    ol.store(m.preconv, output + ol.load(m.conv_bias)[None, :])
    ol.store(m.history0, previous)
    ol.store(m.history1, current)
