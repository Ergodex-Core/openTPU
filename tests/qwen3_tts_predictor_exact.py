"""Predictor-specific exact arithmetic replay; no compiler or ISA interpreter math."""
import numpy as np
from qwen3_tts_exact_mx import ExactMX


def read_predictor(session, name):
    t = getattr(session.image.predictor, name)
    return session.machine.slices[0].dram[t.base.const:t.base.const + 4 * int(np.prod(t.shape))].view(np.float32).copy().reshape(-1)


class PredictorExactReplay:
    def __init__(self, image, weights):
        prefix = 'talker.code_predictor.'
        renamed = {k.replace(prefix, 'talker.'): v for k, v in weights.items()
                   if k.startswith(prefix + 'model.layers.') or k == prefix + 'model.norm.weight'}
        for i in range(image.codebooks - 1):
            renamed[f'talker.codec_head.{i}.weight'] = weights[f'{prefix}lm_head.{i}.weight']
        renamed['talker.codec_head.weight'] = renamed['talker.codec_head.0.weight']
        self.exact = ExactMX(image.predictor.spec, renamed, image.predictor_layer_count)
        self.pending, self.position, self.frame = None, 0, -1
        self.rows = []

    def observe(self, name, session):
        if name == 'predictor.hidden':
            self.exact.reset()
            self.position, self.frame = 0, self.frame + 1
            self.pending = self.exact.hidden_step(read_predictor(session, 'output'))
        elif name == 'predictor.primary':
            self.position = 1
            self.exact.packed['talker.codec_head.weight'] = self.exact.packed['talker.codec_head.0.weight']
            self.pending = self.exact.hidden_step(read_predictor(session, 'output'))
        elif name.startswith('predictor.embedding.'):
            head = int(name.rsplit('.', 1)[1])
            if head + 1 < session.image.codebooks - 1:
                self.position = head + 2
                self.exact.packed['talker.codec_head.weight'] = self.exact.packed[f'talker.codec_head.{head + 1}.weight']
                self.pending = self.exact.hidden_step(read_predictor(session, 'output'))
        elif name.startswith('predictor.layer.') or name.startswith('predictor.head.'):
            layer = name.startswith('predictor.layer.')
            index = int(name.rsplit('.', 1)[1])
            want = self.pending['layers'][index] if layer else self.pending['logits']
            got = read_predictor(session, 'output' if layer else 'logits')
            self.rows.append(dict(frame=self.frame, position=self.position, part=name,
                                  bitwise_equal=bool(np.array_equal(got.view(np.uint32), want.view(np.uint32))),
                                  max_absolute_error=float(np.max(np.abs(got - want))),
                                  reference_greedy=None if layer else int(np.argmax(want)),
                                  device_greedy=None if layer else int(np.argmax(got))))
