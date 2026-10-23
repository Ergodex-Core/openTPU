"""Optional corrected talker+predictor codec-frame layout, context bounded to 512."""
import math
from .. import quant as Q
from ..compiler import Affine, QTensor
from ..kernels.tts_talker_precision import (corrected_input, corrected_feedback, corrected_qkv,
                                           corrected_attention_head, corrected_finish, corrected_codec_logits)
from .qwen3_tts_talker import _tensor
from .qwen3_tts_predictor_precision import PredictorPrecisionImage


class TalkerPrecisionFrameImage(PredictorPrecisionImage):
    def __init__(self, spec, cfg, capacity=512, attention_block=None, **kwargs):
        if type(capacity) is not int or capacity > 512:
            raise ValueError('corrected talker supports at most 512 resident positions; no offload or truncation')
        super().__init__(spec, cfg, capacity, attention_block, **kwargs)

        def alloc(name, size):
            address = self.nbytes
            self.regions[name] = (address, size)
            self.nbytes = ((address + size + 127) // 128) * 128
            return address

        def tensor(name, shape):
            return _tensor(alloc(name, 4 * math.prod(shape)), shape)

        def residual(name):
            _, high, shape = self.parameters[name]
            n, k = shape
            data = alloc(name + '.residual.data', n * Q.row_bytes(k, 'int8', cfg.D))
            scale = alloc(name + '.residual.scale', n * Q.scale_bytes(k, 'int8', cfg.D))
            low = QTensor.weight(data, scale, n, k, 'int8', cfg.D)
            self.residual_parameters[name] = (high, low, shape)
            return low

        if spec.text_hidden not in self.identities:
            width = spec.text_hidden
            self.identities[width] = QTensor(Affine(alloc(f'talker.precision.identity.{width}', width * width)),
                                             None, (width, width), width, 0, cfg.D)
        m = self.desc
        m.identities = self.identities
        m.fc1_residual = residual('talker.text_projection.linear_fc1.weight')
        m.fc2_residual = residual('talker.text_projection.linear_fc2.weight')
        m.codec_head_residual = residual('talker.codec_head.weight')
        q_current = tensor('talker.precision.q_current', (spec.n_q, spec.head_dim))
        context = tensor('talker.precision.context', (spec.n_q, spec.head_dim))
        suffixes = dict(wq='self_attn.q_proj.weight', wk='self_attn.k_proj.weight', wv='self_attn.v_proj.weight',
                        wo='self_attn.o_proj.weight', gate='mlp.gate_proj.weight', up='mlp.up_proj.weight', down='mlp.down_proj.weight')
        for i, layer in enumerate(self.layers):
            layer.identities, layer.q_current, layer.context = self.identities, q_current, context
            for attr, suffix in suffixes.items():
                setattr(layer, attr + '_residual', residual(f'talker.model.layers.{i}.{suffix}'))
            layer.kfp, layer.vfp = [], []
            for head in range(spec.n_kv):
                layer.kfp.append(tensor(f'talker.precision.layer.{i}.kv.{head}.kfp', (capacity, spec.head_dim)))
                layer.vfp.append(tensor(f'talker.precision.layer.{i}.kv.{head}.vfp', (capacity, spec.head_dim)))
        if self.nbytes > min(cfg.DRAM_BYTES, 2**32):
            raise MemoryError(f'corrected talker needs {self.nbytes} resident bytes; no offload')

    def layer_parts(self, position):
        if type(position) is not int or not 0 <= position < self.capacity:
            raise ValueError('talker position outside corrected resident capacity')
        parts = []
        for i, layer in enumerate(self.layers):
            parts.append(self.part(f'talker.{i}.qkv', corrected_qkv, m=layer, position=position))
            for head in range(self.spec.n_kv):
                parts.append(self.part(f'talker.{i}.head.{head}', corrected_attention_head,
                                       m=layer, position=position, head=head))
            parts.append(self.part(f'layer.{i}', corrected_finish, m=layer))
        return parts

    def compile_parts(self, position, input_kind='aligned'):
        if input_kind not in ('aligned', 'role'):
            raise ValueError('unsupported talker input row kind')
        parts = self.layer_parts(position)
        return [self.part('prompt.corrected', corrected_input, m=self.desc, role=input_kind=='role')] + parts + [
            self.part('codec_head', corrected_codec_logits, m=self.desc, audio_vocab=self.audio_vocab, eos=self.eos)]

    def feedback_parts(self, position):
        return [self.part('codec.feedback', corrected_feedback, m=self.desc)] + self.layer_parts(position) + [
            self.part('codec_head', corrected_codec_logits, m=self.desc, audio_vocab=self.audio_vocab, eos=self.eos)]
