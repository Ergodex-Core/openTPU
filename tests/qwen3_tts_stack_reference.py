"""Independent locally reviewed Torch FP64/BF16 talker algebra; no Qwen import."""
import torch
import torch.nn.functional as F


class StackReference:
    def __init__(self, spec, weights, layers, *, bf16=False):
        torch.set_num_threads(1)
        self.spec, self.layers, self.bf16 = spec, layers, bf16
        self.dtype = torch.bfloat16 if bf16 else torch.float64
        self.weights = {k: torch.from_numpy(v).to(self.dtype) for k, v in weights.items()}
        self.reset()

    def reset(self):
        self.keys = [[] for _ in range(self.layers)]
        self.values = [[] for _ in range(self.layers)]

    def norm(self, x, weight):
        z = x.float() if self.bf16 else x
        z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + self.spec.eps)
        return weight * z.to(x.dtype)

    def array(self, x):
        return (x.float() if self.bf16 else x).numpy().reshape(-1).copy()

    def step(self, text_id, codec_id):
        s, w = self.spec, self.weights
        p = "talker.text_projection."
        text = w["talker.model.text_embedding.weight"][text_id].reshape(1, -1)
        projected = F.linear(F.silu(F.linear(text, w[p + "linear_fc1.weight"], w[p + "linear_fc1.bias"])),
                             w[p + "linear_fc2.weight"], w[p + "linear_fc2.bias"])
        x = projected + w["talker.model.codec_embedding.weight"][codec_id]
        result = {"projected": self.array(projected), "assembled": self.array(x), "layers": []}
        frequency_dtype = torch.float32 if self.bf16 else torch.float64
        inv = 1.0 / s.theta ** (torch.arange(0, s.head_dim, 2).to(frequency_dtype) / s.head_dim)
        angle = len(self.keys[0]) * inv
        cos, sin = angle.cos().to(self.dtype), angle.sin().to(self.dtype)

        def rotate(z):
            half = s.head_dim // 2
            return z * torch.cat((cos, cos)) + torch.cat((-z[:, half:], z[:, :half]), -1) * torch.cat((sin, sin))

        for index in range(self.layers):
            p = f"talker.model.layers.{index}."
            z = self.norm(x, w[p + "input_layernorm.weight"])
            q = self.norm(F.linear(z, w[p + "self_attn.q_proj.weight"]).reshape(s.n_q, s.head_dim), w[p + "self_attn.q_norm.weight"])
            k = self.norm(F.linear(z, w[p + "self_attn.k_proj.weight"]).reshape(s.n_kv, s.head_dim), w[p + "self_attn.k_norm.weight"])
            v = F.linear(z, w[p + "self_attn.v_proj.weight"]).reshape(s.n_kv, s.head_dim)
            q, k = rotate(q), rotate(k)
            self.keys[index].append(k)
            self.values[index].append(v)
            groups = s.n_q // s.n_kv
            keys = torch.stack(self.keys[index], 1).repeat_interleave(groups, 0)
            values = torch.stack(self.values[index], 1).repeat_interleave(groups, 0)
            scores = torch.matmul(q[:, None], keys.transpose(1, 2)) * (s.head_dim ** -0.5)
            probabilities = F.softmax(scores, dim=-1, dtype=frequency_dtype).to(self.dtype)
            context = torch.matmul(probabilities, values).reshape(1, -1)
            residual = x + F.linear(context, w[p + "self_attn.o_proj.weight"])
            z = self.norm(residual, w[p + "post_attention_layernorm.weight"])
            gate, up = F.silu(F.linear(z, w[p + "mlp.gate_proj.weight"])), F.linear(z, w[p + "mlp.up_proj.weight"])
            x = residual + F.linear(gate * up, w[p + "mlp.down_proj.weight"])
            result["layers"].append(self.array(x))
        normalized = self.norm(x, w["talker.model.norm.weight"])
        result["normalized"] = self.array(normalized)
        result["logits"] = self.array(F.linear(normalized, w["talker.codec_head.weight"]))
        return result
