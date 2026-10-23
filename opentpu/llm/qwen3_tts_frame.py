"""Complete 16-codebook frame and feedback in ISA memory, diagnostic session only.

Greedy policy: do_sample=False, subtalker_dosample=False, repetition_penalty=1,
min_new_tokens=2, fixed source suppression. This is an explicit supported source
generation configuration, not the upstream sampled default or a waveform engine.
"""
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from .. import quant as Q
from ..compiler import KVDesc, QTensor, k_scale_bytes
from ..kernels.tts_stack import talker_layer, codec_logits
from ..kernels.tts_frame import (primary_select, predictor_hidden, predictor_primary,
                                 predictor_logits, predictor_accumulate, codec_feedback)
from .qwen3_tts_talker import _tensor
from .qwen3_tts_stack import TalkerStackImage, StackISASession


def check_predictor_config(config):
    c = config['talker_config']['code_predictor_config']
    expected = dict(hidden_size=1024, intermediate_size=3072, num_hidden_layers=5,
                    num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                    vocab_size=2048, num_code_groups=16, rope_theta=1e6,
                    rms_norm_eps=1e-6, attention_bias=False, hidden_act='silu',
                    rope_scaling=None, use_sliding_window=False, sliding_window=None)
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError('unsupported code predictor configuration')


class CodecFrameImage(TalkerStackImage):
    def __init__(self, spec, cfg, capacity=512, attention_block=None, *, layers=28,
                 predictor_layers=5, codebooks=16, audio_vocab=2048, eos=2150,
                 pad_id=151671):
        if not (type(codebooks) is int and 2 <= codebooks <= 16 and
                type(predictor_layers) is int and 1 <= predictor_layers <= 5):
            raise ValueError('invalid predictor layer/codebook count')
        if type(pad_id) is not int or not 0 <= pad_id < spec.text_vocab:
            raise ValueError('text pad ID out of range')
        super().__init__(spec, cfg, capacity, attention_block, layers=layers,
                         audio_vocab=audio_vocab, eos=eos)
        self.predictor_layer_count, self.codebooks, self.pad_id = predictor_layers, codebooks, pad_id

        def alloc(name, size):
            address = self.nbytes
            self.regions[name] = (address, size)
            self.nbytes = ((address + size + 127) // 128) * 128
            return address

        def tensor(name, shape, checkpoint=None):
            t = _tensor(alloc(name, 4 * math.prod(shape)), shape)
            if checkpoint is not None:
                self.parameters[checkpoint] = ('fp32', t, shape)
            return t

        def weight(name, shape, checkpoint):
            n, k = shape
            data = alloc(name + '.data', n * Q.row_bytes(k, 'int8', cfg.D))
            scale = alloc(name + '.scales', n * Q.scale_bytes(k, 'int8', cfg.D))
            t = QTensor.weight(data, scale, n, k, 'int8', cfg.D)
            self.parameters[checkpoint] = ('int8', t, shape)
            return t

        m, h = self.desc, spec.hidden
        m.primary = tensor('primary', (2,))
        m.code_ids = tensor('code_ids', (codebooks,))
        m.codec_sum = tensor('codec_sum', (1, h))
        m.pad_id = tensor('pad_id', (1,))
        # Equal widths make small_to_mtp_projection the source Identity module.
        p = SimpleNamespace(spec=replace(spec, codec_vocab=audio_vocab),
                            cosine=m.cosine, sine=m.sine)
        p.output = tensor('predictor.output', (1, h))
        p.attention_residual = tensor('predictor.attention_residual', (1, h))
        p.normalized = tensor('predictor.normalized', (1, h))
        p.logits = tensor('predictor.logits', (1, audio_vocab))
        p.greedy = tensor('predictor.greedy', (2,))
        p.final_norm = tensor('predictor.final_norm', (h,), 'talker.code_predictor.model.norm.weight')
        p.heads, p.embeddings = [], []
        for i in range(codebooks - 1):
            p.heads.append(weight(f'predictor.head.{i}', (audio_vocab, h),
                                  f'talker.code_predictor.lm_head.{i}.weight'))
            p.embeddings.append(tensor(f'predictor.embedding.{i}', (audio_vocab, h),
                                      f'talker.code_predictor.model.codec_embedding.{i}.weight'))
        originals = list(self.parameters.items())
        attrs = {id(getattr(m, n)): n for n in
                 ('input_norm', 'post_norm', 'q_norm', 'k_norm', 'wq', 'wk', 'wv', 'wo', 'gate', 'up', 'down')}
        self.predictor_layers = []
        self.predictor_capacity = cfg.D  # only positions 0..15; causal length guards stale tail
        for index in range(predictor_layers):
            layer = SimpleNamespace(**vars(p))
            for key, (fmt, desc, shape) in originals:
                if not key.startswith('talker.model.layers.0.'):
                    continue
                attr = attrs[id(desc)]
                checkpoint = key.replace('talker.model.layers.0.', f'talker.code_predictor.model.layers.{index}.')
                name = f'predictor.layer.{index}.{attr}'
                setattr(layer, attr, tensor(name, shape, checkpoint) if fmt == 'fp32'
                        else weight(name, shape, checkpoint))
            heads = {}
            for head in range(spec.n_kv):
                heads[head] = {k: alloc(f'predictor.layer.{index}.kv.{head}.{k}', size)
                               for k, size in [('k', cfg.D * spec.head_dim),
                                   ('ks', cfg.D * k_scale_bytes(spec.head_dim, cfg.D)),
                                   ('vt', cfg.D * spec.head_dim), ('vs', cfg.D * 4)]}
            layer.kv = KVDesc(heads, cfg.D, spec.head_dim, cfg.D, 1, 0)
            self.predictor_layers.append(layer)
        self.predictor = p
        if codebooks > cfg.D or self.nbytes > min(cfg.DRAM_BYTES, 2**32):
            raise MemoryError(f'codec frame needs {self.nbytes} resident device bytes; no offload')

    def build(self, weights, **kwargs):
        packed = super().build(weights, **kwargs)
        a = self.desc.pad_id.base.const
        packed[a:a + 4] = np.asarray([self.pad_id], np.float32).view(np.uint8)
        return packed

    def part(self, name, kernel, **args):
        b = kernel.trace(self.cfg, 0, args)
        program = b.finish()
        budget = dict(instructions=len(program), imem_words=8 * len(program),
                      tmem_peak_words=b.tmem_peak, dram_bytes=self.nbytes)
        if budget['imem_words'] > self.cfg.IMEM_WORDS or b.tmem_peak > self.cfg.TMEM_WORDS:
            raise MemoryError('codec-frame part exceeds on-chip program/scratch memory')
        return name, [program], budget

    def predictor_parts(self, position, head=None):
        if type(position) is not int or not 0 <= position < self.codebooks:
            raise ValueError('predictor position outside frame')
        if head is not None and (type(head) is not int or head != position - 1 or not 0 <= head < self.codebooks - 1):
            raise ValueError('predictor head/position mismatch')
        parts = [self.part(f'predictor.layer.{i}', talker_layer, m=layer,
                           position=position, attention_block=self.predictor_capacity)
                 for i, layer in enumerate(self.predictor_layers)]
        if head is not None:
            parts += [self.part(f'predictor.head.{head}', predictor_logits, m=self.desc, p=self.predictor, head=head),
                      self.part(f'predictor.embedding.{head}', predictor_accumulate, m=self.desc, p=self.predictor, head=head)]
        return parts

    def feedback_parts(self, position):
        if type(position) is not int or not 0 <= position < self.capacity:
            raise ValueError('talker position outside resident KV capacity')
        return [self.part('codec.feedback', codec_feedback, m=self.desc)] + [
            self.part(f'layer.{i}', talker_layer, m=layer, position=position,
                      attention_block=self.attention_block) for i, layer in enumerate(self.layers)] + [
            self.part('codec_head', codec_logits, m=self.desc, audio_vocab=self.audio_vocab, eos=self.eos)]


class CodecFrameISASession(StackISASession):
    """Explicit CPU ISA diagnostic; no production transport or fallback.

    After prefill, no host token/embedding/hidden writes occur. Host dispatches
    known frame positions and checks the already device-selected EOS as control.
    Predictor positions restart at zero: every visible KV entry is overwritten
    before use, and future stale entries are masked by exact causal length.
    """
    def __init__(self, image, packed):
        super().__init__(image, packed)
        self.frames, self.stopped, self.failed = 0, False, False

    def step(self, *args, **kwargs):
        if self.failed or self.frames or self.stopped:
            raise ValueError('reset required before further prefill')
        try:
            return super().step(*args, **kwargs)
        except Exception:
            self.failed = True
            raise

    def execute(self, parts, observe=None):
        if self.failed:
            raise ValueError('reset required after failed frame')
        for name, programs, _ in parts:
            try:
                self.machine.load(programs).run()
            except Exception:
                self.failed = True
                raise
            if observe is not None:
                observe(name, self)

    def frame(self, observe=None):
        im, m, p = self.image, self.image.desc, self.image.predictor
        if self.position == 0 or self.stopped or self.failed:
            raise ValueError('prefill/reset required or EOS already reached')
        if self.position >= im.capacity:
            raise ValueError('no resident KV slot for codec feedback')
        self.execute([im.part('primary.select', primary_select, m=m, audio_vocab=im.audio_vocab,
                             eos=im.eos, allow_eos=self.frames >= 2)], observe)
        if int(self.read('primary')[1]) == im.eos:
            self.stopped = True
            return None
        self.execute([im.part('predictor.hidden', predictor_hidden, m=m, p=p)], observe)
        self.execute(im.predictor_parts(0), observe)
        self.execute([im.part('predictor.primary', predictor_primary, m=m, p=p)], observe)
        for head in range(im.codebooks - 1):
            self.execute(im.predictor_parts(head + 1, head), observe)
        ids = self.read('code_ids').astype(np.int64)
        self.execute(self.feedback_parts(), observe)
        self.position += 1
        self.frames += 1
        return ids

    def feedback_parts(self):
        return self.image.feedback_parts(self.position)

    def reset(self):
        super().reset()
        self.frames, self.stopped, self.failed = 0, False, False
