"""Operator-level MX/FP32 diagnostic, independent of compiler/kernels/Machine.

Shares the documented numeric primitives and offline weight packer to isolate
lowering errors from quantization; not an independent speech-quality reference.
"""
import numpy as np
from opentpu import fp32 as F, quant as Q


class ExactMX:
    def __init__(self, spec, weights, layers=28):
        self.spec, self.layers = spec, layers
        self.weights = weights
        self.packed = {}
        for name, value in weights.items():
            if value.ndim == 2 and "embedding" not in name:
                raw, scales = Q.quantize_mxu(value, "int8", spec.head_dim)
                sw, offsets, _ = Q.unpack_scales(scales, "int8", value.shape[1] // spec.head_dim)
                self.packed[name] = (raw.view(np.int8), sw, offsets[:, :, 0])
        self.reset()

    def reset(self):
        self.keys, self.values, self.kscales, self.vscales = ([[] for _ in range(self.layers)] for _ in range(4))

    def mm(self, x, weight, *, unit=False):
        d = self.spec.head_dim
        x = np.asarray(x, np.float32).reshape(-1, x.shape[-1])
        raw, sw, dw = weight
        m, k = x.shape
        n, kb = raw.shape[0], k // d
        qa, sa, da = F.quantize_mx(x, d // 4)
        da = da.reshape(m, kb, 4)
        c = np.einsum("mkgd,nkgd->mnkg", qa.reshape(m, kb, 4, d // 4).astype(np.float64),
                      raw.reshape(n, kb, 4, d // 4).astype(np.float64), optimize=True).astype(np.int64)
        shift = 16 - dw[None, :, :, None].astype(np.int64) - da[:, None].astype(np.int64)
        factor = np.where(shift >= 0, np.left_shift(np.int64(1), np.maximum(shift, 0)), 0)
        acc = (c * factor).sum((2, 3))
        return F.mul(F.mul(F.i2f(acc), sw[None]), F.mul(sa[:, None], np.float32(2.0 ** -16)))

    def linear(self, x, name, rows=None, cols=None):
        raw, sw, offsets = self.packed[name]
        if rows is not None:
            raw, sw, offsets = raw[rows], sw[rows], offsets[rows]
        if cols is not None:
            start, stop = cols
            raw, offsets = raw[:, start:stop], offsets[:, start // self.spec.head_dim:stop // self.spec.head_dim]
        return self.mm(x, (raw, sw, offsets))

    def norm(self, x, gamma):
        inverse = F.rsqrt(F.add(F.mul(F.rdot(x, x), np.float32(1.0 / x.shape[-1])), np.float32(self.spec.eps)))
        return F.mul(F.mul(x, inverse[:, None]), gamma[None])

    def silu(self, x):
        return F.mul(x, F.recip(F.add(F.exp2(F.mul(x, np.float32(-np.log2(np.e)))), np.float32(1))))

    def step(self, text_id, codec_id, *, input_kind="aligned"):
        s, w, d = self.spec, self.weights, self.spec.head_dim
        p = "talker.text_projection."
        text = w["talker.model.text_embedding.weight"][text_id:text_id + 1]
        a = F.add(self.linear(text, p + "linear_fc1.weight"), w[p + "linear_fc1.bias"])
        projected = F.add(self.linear(self.silu(a), p + "linear_fc2.weight"), w[p + "linear_fc2.bias"])
        x = projected if input_kind == "role" else F.add(projected, w["talker.model.codec_embedding.weight"][codec_id])
        result = {"projected": projected.reshape(-1), "assembled": x.reshape(-1), "layers": []}
        return self.hidden_step(x, result)

    def hidden_step(self, x, result=None):
        s, w, d = self.spec, self.weights, self.spec.head_dim
        x = np.asarray(x, np.float32).reshape(1, -1)
        result = result or {"assembled": x.reshape(-1).copy(), "layers": []}
        position = len(self.keys[0])
        inv = np.float32(1) / np.power(np.float32(s.theta), np.arange(0, d, 2, dtype=np.float32) / d)
        angle = np.float32(position) * inv
        cos, sin = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)

        def rope(z):
            a, b = z[:, :d // 2], z[:, d // 2:]
            return np.concatenate((F.sub(F.mul(a, cos), F.mul(b, sin)),
                                   F.add(F.mul(b, cos), F.mul(a, sin))), -1)

        for index in range(self.layers):
            p = f"talker.model.layers.{index}."
            z = self.norm(x, w[p + "input_layernorm.weight"])
            q = self.norm(self.linear(z, p + "self_attn.q_proj.weight").reshape(s.n_q, d), w[p + "self_attn.q_norm.weight"])
            k = self.norm(self.linear(z, p + "self_attn.k_proj.weight").reshape(s.n_kv, d), w[p + "self_attn.k_norm.weight"])
            v = self.linear(z, p + "self_attn.v_proj.weight").reshape(s.n_kv, d)
            q, k = rope(q), rope(k)
            kq, ks = F.quantize(k)
            vq, vs = F.quantize(v)
            self.keys[index].append(kq)
            self.kscales[index].append(ks)
            self.values[index].append(vq)
            self.vscales[index].append(vs)
            n = position + 1
            if n > 256:
                raise ValueError("exact replay currently covers one 256-token attention block")
            context = []
            groups = s.n_q // s.n_kv
            for head in range(s.n_kv):
                keys = np.stack([a[head] for a in self.keys[index]])
                key_scales = np.asarray([a[head] for a in self.kscales[index]], np.float32)
                query = F.mul(q[head * groups:(head + 1) * groups], np.float32(np.log2(np.e) / np.sqrt(d)))
                score = self.mm(query, (keys, key_scales, np.zeros((n, 1), np.uint8)))
                probability = F.exp2(F.sub(score, F.chain_max(score)[:, None]))
                scales = np.asarray([a[head] for a in self.vscales[index]], np.float32)
                folded = F.mul(probability, scales[None])
                padded = np.pad(folded, ((0, 0), (0, (-n) % d)))
                values = np.stack([a[head] for a in self.values[index]]).T
                values = np.pad(values, ((0, 0), (0, (-n) % d)))
                pv = self.mm(padded, (values, np.ones(d, np.float32), np.zeros((d, padded.shape[-1] // d), np.uint8)))
                context.append(F.mul(pv, F.recip(F.interleaved_sum(probability, F.RED_PARTIALS))[:, None]))
            context = np.concatenate(context, 0).reshape(1, -1)
            residual = F.add(x, self.linear(context, p + "self_attn.o_proj.weight"))
            z = self.norm(residual, w[p + "post_attention_layernorm.weight"])
            output = None
            # Same independent arithmetic order as the ISA's static down chunks.
            chunk = max(2 * d, (s.ffn // 8) // d * d)
            while s.ffn % chunk:
                chunk -= d
            for start in range(0, s.ffn, chunk):
                stop = start + chunk
                gate = self.silu(self.linear(z, p + "mlp.gate_proj.weight", rows=slice(start, stop)))
                up = self.linear(z, p + "mlp.up_proj.weight", rows=slice(start, stop))
                term = self.linear(F.mul(gate, up), p + "mlp.down_proj.weight", cols=(start, stop))
                output = term if output is None else F.add(output, term)
            x = F.add(residual, output)
            result["layers"].append(x.reshape(-1).copy())
        normalized = self.norm(x, w["talker.model.norm.weight"])
        result["normalized"] = normalized.reshape(-1)
        result["logits"] = self.linear(normalized, "talker.codec_head.weight").reshape(-1)
        return result
