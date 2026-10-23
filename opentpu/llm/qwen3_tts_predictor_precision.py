"""Optional resident predictor precision experiment; no production backend."""
import math
import numpy as np
from .. import fp32 as F, quant as Q
from ..compiler import Affine, QTensor
from ..kernels.mlp import down_chunk
from ..kernels.tts_frame import predictor_accumulate
from ..kernels.tts_predictor_precision import predictor_precision_layer, predictor_precision_head
from .qwen3_tts_talker import _tensor
from .qwen3_tts_frame import CodecFrameImage


class PredictorPrecisionImage(CodecFrameImage):
    """Keeps the talker path unchanged; corrects predictor weights/activations.

    Residual weight packing is offline. Activation correction, short FP32 KV,
    attention, heads and argmax are device programs. No native FP16 MM is implied.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg, s, p = self.cfg, self.spec, self.predictor
        self.residual_parameters, self.identities = {}, {}

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

        widths = {s.hidden, s.n_q * s.head_dim, down_chunk(s.ffn, cfg.D, 'int8')}
        for width in sorted(widths):
            address = alloc(f'predictor.precision.identity.{width}', width * width)
            self.identities[width] = QTensor(Affine(address), None, (width, width), width, 0, cfg.D)
        p.identities = self.identities
        p.heads_residual = [residual(f'talker.code_predictor.lm_head.{i}.weight') for i in range(self.codebooks - 1)]
        suffixes = dict(wq='self_attn.q_proj.weight', wk='self_attn.k_proj.weight', wv='self_attn.v_proj.weight',
                        wo='self_attn.o_proj.weight', gate='mlp.gate_proj.weight', up='mlp.up_proj.weight', down='mlp.down_proj.weight')
        for i, layer in enumerate(self.predictor_layers):
            layer.identities = self.identities
            for attr, suffix in suffixes.items():
                setattr(layer, attr + '_residual', residual(f'talker.code_predictor.model.layers.{i}.{suffix}'))
            layer.kfp, layer.vfp = [], []
            for head in range(s.n_kv):
                layer.kfp.append(tensor(f'predictor.precision.layer.{i}.kv.{head}.kfp', (self.predictor_capacity, s.head_dim)))
                layer.vfp.append(tensor(f'predictor.precision.layer.{i}.kv.{head}.vfp', (self.predictor_capacity, s.head_dim)))
        if self.nbytes > min(cfg.DRAM_BYTES, 2**32):
            raise MemoryError(f'precision predictor needs {self.nbytes} resident bytes; no offload')

    def build(self, weights, **kwargs):
        packed = super().build(weights, **kwargs)
        for width, desc in self.identities.items():
            identity = np.eye(width, dtype=np.int8).view(np.uint8).reshape(-1)
            packed[desc.data.const:desc.data.const + len(identity)] = identity
        for name, (high, low, (n, k)) in self.residual_parameters.items():
            raw = packed[high.data.const:high.data.const + n * high.rs].reshape(n, high.rs)
            scales = packed[high.scale.const:high.scale.const + n * high.srs].reshape(n, high.srs)
            represented = Q.dequantize_mxu(raw, scales, 'int8', k, self.cfg.D).astype(np.float32)
            correction = F.sub(np.asarray(weights[name], np.float32), represented)
            data, scale = Q.quantize_mxu(correction, 'int8', self.cfg.D)
            packed[low.data.const:low.data.const + data.size] = data.reshape(-1)
            packed[low.scale.const:low.scale.const + scale.size] = scale.reshape(-1)
        return packed

    def predictor_parts(self, position, head=None):
        if type(position) is not int or not 0 <= position < self.codebooks:
            raise ValueError('predictor position outside frame')
        if head is not None and (type(head) is not int or head != position - 1 or not 0 <= head < self.codebooks - 1):
            raise ValueError('predictor head/position mismatch')
        parts = [self.part(f'predictor.layer.{i}', predictor_precision_layer, m=layer, position=position)
                 for i, layer in enumerate(self.predictor_layers)]
        if head is not None:
            parts += [self.part(f'predictor.head.{head}', predictor_precision_head, m=self.desc, p=self.predictor, head=head),
                      self.part(f'predictor.embedding.{head}', predictor_accumulate, m=self.desc, p=self.predictor, head=head)]
        return parts
