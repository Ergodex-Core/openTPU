"""Functional FP32 quantization ablation only; never a production backend."""
import numpy as np
import torch
import torch.nn.functional as TF
from opentpu import fp32 as F
from qwen3_tts_stack_reference import StackReference


def qact(x, depth):
    a = x.detach().float().numpy()
    shape = a.shape
    q, scale, offsets = F.quantize_mx(a.reshape(-1, shape[-1]), depth // 4)
    factors = np.ldexp(scale[:, None], -offsets.astype(np.int32)).astype(np.float32)
    deq = F.mul(q.reshape(q.shape[0], -1, depth // 4), factors[..., None]).reshape(shape)
    return torch.from_numpy(deq)


def row_quant(x):
    a = x.detach().float().numpy()
    q, scale = F.quantize(a)
    return torch.from_numpy(F.mul(q, scale[:, None])), torch.from_numpy(q.astype(np.float32)), torch.from_numpy(scale)


class QuantReference(StackReference):
    def __init__(self, spec, weights, *, activation=False, kv=False, attention=False):
        super().__init__(spec, weights, 28, dtype=torch.float32)
        self.activation_quant, self.kv_quant, self.attention_quant = activation, kv, attention

    def reset(self):
        super().reset()
        self._vscales, self._vcodes = [], []
        self._cache_layer = 0

    def linear(self, x, weight, bias=None):
        return TF.linear(qact(x, self.spec.head_dim) if self.activation_quant else x, weight, bias)

    def mlp(self, z, prefix):
        if not self.activation_quant:
            return super().mlp(z, prefix)
        w, chunk = self.weights, 384
        gate = TF.silu(self.linear(z, w[prefix + "mlp.gate_proj.weight"]))
        up = self.linear(z, w[prefix + "mlp.up_proj.weight"])
        a = gate * up
        out = None
        for start in range(0, self.spec.ffn, chunk):
            term = self.linear(a[:, start:start + chunk], w[prefix + "mlp.down_proj.weight"][:, start:start + chunk])
            out = term if out is None else out + term
        return out

    def cache(self, k, v):
        if self.kv_quant:
            k, _, _ = row_quant(k)
            v, codes, scales = row_quant(v)
        else:
            codes, scales = v, torch.ones(v.shape[0])
        index = self._cache_layer
        if len(self._vcodes) <= index:
            self._vcodes.append([])
            self._vscales.append([])
        self._vcodes[index].append(codes)
        self._vscales[index].append(scales)
        self._cache_layer = (index + 1) % self.layers
        self._attend_layer = index
        return k, v

    def attend(self, q, keys, values, frequency_dtype):
        if not self.attention_quant:
            return super().attend(q, keys, values, frequency_dtype)
        # Short-prefill ablation: one flash-attention block, including the actual
        # QACT scale-before-Q and V-scale-folded probability quantization.
        n = keys.shape[1]
        if n > 256:
            raise ValueError("functional attention ablation is limited to one 256-token block")
        q = qact(q * np.float32(np.log2(np.e) / np.sqrt(self.spec.head_dim)), self.spec.head_dim)
        scores = torch.matmul(q[:, None], keys.transpose(1, 2)).squeeze(1)
        p = torch.exp2(scores - scores.max(-1, keepdim=True).values)
        denominator = p.sum(-1, keepdim=True)
        index = self._attend_layer
        groups = self.spec.n_q // self.spec.n_kv
        codes = torch.stack(self._vcodes[index], 1).repeat_interleave(groups, 0)
        scales = torch.stack(self._vscales[index], 1).repeat_interleave(groups, 0)
        padded = TF.pad(p * scales, (0, (-n) % self.spec.head_dim))
        pq = qact(padded, self.spec.head_dim)[:, :n]
        return (torch.matmul(pq[:, None], codes).squeeze(1) / denominator).reshape(1, -1)
