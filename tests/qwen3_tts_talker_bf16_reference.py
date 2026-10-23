"""Locally reviewed CPU eager-BF16 algebra, not imported Qwen model code.

Matches the pinned source's cast locations and equal-axis rotary path. CUDA
FlashAttention execution may round differently; this is a diagnostic reference.
"""
import torch
import torch.nn.functional as F


class BF16Reference:
    def __init__(self, spec, weights):
        torch.set_num_threads(1)
        self.spec = spec
        self.weights = {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in weights.items()}
        self.keys, self.values = [], []

    def step(self, text_id, codec_id):
        s, w = self.spec, self.weights
        p = "talker.text_projection."
        text = w["talker.model.text_embedding.weight"][text_id].reshape(1, -1)
        a = F.linear(text, w[p + "linear_fc1.weight"], w[p + "linear_fc1.bias"])
        projected = F.linear(F.silu(a), w[p + "linear_fc2.weight"], w[p + "linear_fc2.bias"])
        assembled = projected + w["talker.model.codec_embedding.weight"][codec_id]
        p = "talker.model.layers.0."

        def norm(x, name):
            z = x.float()
            z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + s.eps)
            return w[p + name] * z.to(x.dtype)

        z = norm(assembled, "input_layernorm.weight")
        q = norm(F.linear(z, w[p + "self_attn.q_proj.weight"]).reshape(s.n_q, s.head_dim),
                 "self_attn.q_norm.weight")
        k = norm(F.linear(z, w[p + "self_attn.k_proj.weight"]).reshape(s.n_kv, s.head_dim),
                 "self_attn.k_norm.weight")
        v = F.linear(z, w[p + "self_attn.v_proj.weight"]).reshape(s.n_kv, s.head_dim)
        half = s.head_dim // 2
        inv = 1.0 / (s.theta ** (torch.arange(0, s.head_dim, 2).float() / s.head_dim))
        angle = len(self.keys) * inv
        cos, sin = angle.cos().to(q.dtype), angle.sin().to(q.dtype)

        def rotate(x):
            cos2, sin2 = torch.cat((cos, cos)), torch.cat((sin, sin))
            rotated = torch.cat((-x[:, half:], x[:, :half]), dim=-1)
            return x * cos2 + rotated * sin2

        q, k = rotate(q), rotate(k)
        self.keys.append(k)
        self.values.append(v)
        groups = s.n_q // s.n_kv
        keys = torch.stack(self.keys, dim=1).repeat_interleave(groups, dim=0)
        values = torch.stack(self.values, dim=1).repeat_interleave(groups, dim=0)
        scores = torch.matmul(q[:, None, :], keys.transpose(1, 2)) * (s.head_dim ** -0.5)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        context = torch.matmul(probabilities, values).reshape(1, -1)
        residual = assembled + F.linear(context, w[p + "self_attn.o_proj.weight"])
        z = norm(residual, "post_attention_layernorm.weight")
        gate = F.silu(F.linear(z, w[p + "mlp.gate_proj.weight"]))
        up = F.linear(z, w[p + "mlp.up_proj.weight"])
        out = residual + F.linear(gate * up, w[p + "mlp.down_proj.weight"])
        return {k: v.float().numpy().reshape(-1).copy() for k, v in
                {"projected": projected, "assembled": assembled,
                 "attention_residual": residual, "output": out}.items()}
