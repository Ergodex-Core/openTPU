"""Resident codec-frame predictor, greedy IDs and feedback; no waveform decoder."""
from .. import language as ol
from ..compiler import current
from .lib import rmsnorm
from .tts_talker import _embedding
from .tts_projection import text_projection


@ol.jit
def primary_select(m, audio_vocab: int, eos: int, allow_eos: bool):
    logits = ol.load(m.logits)
    selected = ol.empty(logits.shape, dense=True)
    selected.set(logits)
    selected[:, audio_vocab:].set(float('-inf'))
    if allow_eos:
        selected[:, eos:eos + 1].set(logits[:, eos:eos + 1])
    best = current().argmax(selected)
    ol.store(m.primary, best)
    ol.store(m.code_ids[0:1], best[1:2])


@ol.jit
def predictor_hidden(m, p):
    ol.store(p.output, ol.load(m.normalized))


@ol.jit
def predictor_primary(m, p):
    primary = _embedding(m.codec_embedding, ol.load(m.code_ids[0:1]))
    ol.store(m.codec_sum, primary)
    ol.store(p.output, primary)


@ol.jit
def predictor_logits(m, p, head: int):
    normalized = rmsnorm(ol.load(p.output), ol.load(p.final_norm), p.spec.eps)
    ol.store(p.normalized, normalized)
    logits = ol.dot(normalized, p.heads[head])
    ol.store(p.logits, logits)
    best = current().argmax(logits)
    ol.store(p.greedy, best)
    ol.store(m.code_ids[head + 1:head + 2], best[1:2])


@ol.jit
def predictor_accumulate(m, p, head: int):
    embed = _embedding(p.embeddings[head], ol.load(m.code_ids[head + 1:head + 2]))
    ol.store(m.codec_sum, ol.load(m.codec_sum) + embed)
    # The last embedding contributes to feedback, but is never a predictor input.
    if head < len(p.heads) - 1:
        ol.store(p.output, embed)


@ol.jit
def codec_feedback(m):
    ol.store(m.text_scratch, _embedding(m.text_embedding, ol.load(m.pad_id)))
    text_projection.fn(m.text_scratch, m.fc1, m.bias1, m.fc2, m.bias2, m.projected)
    assembled = ol.load(m.codec_sum) + ol.load(m.projected)
    ol.store(m.assembled, assembled)
    ol.store(m.output, assembled)
