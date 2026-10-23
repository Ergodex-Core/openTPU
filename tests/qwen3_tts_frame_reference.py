"""Independent source-order FP32/BF16 codec-frame algebra; no upstream code import."""
import numpy as np
import torch
import torch.nn.functional as F
from qwen3_tts_stack_reference import StackReference


class FrameReference:
    def __init__(self, image, weights, *, bf16=False):
        self.image, self.dtype = image, torch.bfloat16 if bf16 else torch.float32
        talker = {k: v for k, v in weights.items() if not k.startswith('talker.code_predictor.')}
        self.talker = StackReference(image.spec, talker, image.layer_count, dtype=self.dtype)
        # The predictor has its own decoder stack/cache and 15 heads. Rename only
        # dictionary keys to reuse independently authored transformer algebra.
        pw = {k.replace('talker.code_predictor.', 'talker.'): v for k, v in weights.items()
              if k.startswith('talker.code_predictor.model.layers.') or k == 'talker.code_predictor.model.norm.weight'}
        pw['talker.codec_head.weight'] = weights['talker.code_predictor.lm_head.0.weight']
        self.predictor = StackReference(image.predictor.spec, pw, image.predictor_layer_count, dtype=self.dtype)
        self.embeddings = [torch.from_numpy(weights[f'talker.code_predictor.model.codec_embedding.{i}.weight']).to(self.dtype)
                           for i in range(image.codebooks - 1)]
        self.heads = [torch.from_numpy(weights[f'talker.code_predictor.lm_head.{i}.weight']).to(self.dtype)
                      for i in range(image.codebooks - 1)]
        self.reset()

    def reset(self):
        self.talker.reset()
        self.predictor.reset()
        self.last, self.frames, self.stopped = None, 0, False

    def prefill(self, row):
        self.last = self.talker.step(row.text_id, row.codec_id, input_kind=row.kind)
        return self.last

    def choose_primary(self):
        logits = self.last['logits'].copy()
        logits[self.image.audio_vocab:] = -np.inf
        if self.frames >= 2:
            logits[self.image.eos] = self.last['logits'][self.image.eos]
        return int(np.argmax(logits))

    def predictor_frame(self, hidden, primary, *, force_ids=None):
        """Optional teacher forcing diagnoses identical-prefix conditional logits."""
        im = self.image
        self.predictor.reset()
        self.predictor.hidden_step(torch.as_tensor(hidden, dtype=self.dtype).reshape(1, -1))
        embeddings = [self.talker.weights['talker.model.codec_embedding.weight'][primary].reshape(1, -1)]
        ids, rows = [primary], []
        for i in range(im.codebooks - 1):
            row = self.predictor.hidden_step(embeddings[-1])
            norm = torch.as_tensor(row['normalized'], dtype=self.dtype).reshape(1, -1)
            logits = self.predictor.array(F.linear(norm, self.heads[i]))
            selected = int(np.argmax(logits))
            ids.append(selected if force_ids is None else int(force_ids[i + 1]))
            rows.append(dict(logits=logits, greedy=selected, layers=row['layers']))
            embeddings.append(self.embeddings[i][ids[-1]].reshape(1, -1))
        # Source concatenates then sum(1), rather than adding on the host in the
        # deployed path. This reference preserves source dtype/reduction behavior.
        summed = torch.cat(embeddings, dim=0).sum(0, keepdim=True)
        return np.asarray(ids), summed, rows

    def feedback(self, summed):
        w, p = self.talker.weights, 'talker.text_projection.'
        text = w['talker.model.text_embedding.weight'][self.image.pad_id].reshape(1, -1)
        projected = F.linear(F.silu(F.linear(text, w[p + 'linear_fc1.weight'], w[p + 'linear_fc1.bias'])),
                             w[p + 'linear_fc2.weight'], w[p + 'linear_fc2.bias'])
        self.last = self.talker.hidden_step(summed + projected)
        self.frames += 1
        return self.last

    def frame(self):
        primary = self.choose_primary()
        if primary == self.image.eos:
            self.stopped = True
            return None
        ids, summed, rows = self.predictor_frame(self.last['normalized'], primary)
        feedback = self.feedback(summed)
        return dict(ids=ids, codec_sum=self.talker.array(summed), predictor=rows, feedback=feedback)
