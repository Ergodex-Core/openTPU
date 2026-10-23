"""Independent locally reviewed Torch FP64/BF16 talker algebra; no Qwen import."""
import torch
import torch.nn.functional as F


class StackReference:
    def __init__(self, spec, weights, layers, *, bf16=False, dtype=None):
        torch.set_num_threads(1)
        self.spec, self.layers = spec, layers
        self.dtype = dtype or (torch.bfloat16 if bf16 else torch.float64)
        self.bf16 = self.dtype == torch.bfloat16
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

    def linear(self, x, weight, bias=None):
        return F.linear(x, weight, bias)

    def cache(self, k, v):
        return k, v

    def attend(self, q, keys, values, frequency_dtype):
        scores = torch.matmul(q[:, None], keys.transpose(1, 2)) * (self.spec.head_dim ** -0.5)
        probabilities = F.softmax(scores, dim=-1, dtype=frequency_dtype).to(self.dtype)
        return torch.matmul(probabilities, values).reshape(1, -1)

    def mlp(self, z, prefix):
        w = self.weights
        gate, up = F.silu(self.linear(z, w[prefix + "mlp.gate_proj.weight"])), self.linear(z, w[prefix + "mlp.up_proj.weight"])
        return self.linear(gate * up, w[prefix + "mlp.down_proj.weight"])

    def step(self, text_id, codec_id, *, input_kind="aligned"):
        s, w = self.spec, self.weights
        p = "talker.text_projection."
        text = w["talker.model.text_embedding.weight"][text_id].reshape(1, -1)
        projected = self.linear(F.silu(self.linear(text, w[p + "linear_fc1.weight"], w[p + "linear_fc1.bias"])),
                             w[p + "linear_fc2.weight"], w[p + "linear_fc2.bias"])
        x = projected if input_kind == "role" else projected + w["talker.model.codec_embedding.weight"][codec_id]
        result = {"projected": self.array(projected), "assembled": self.array(x), "layers": []}
        return self.hidden_step(x, result)

    def hidden_step(self, x, result=None):
        s, w = self.spec, self.weights
        result = result or {"assembled": self.array(x), "layers": []}
        frequency_dtype = torch.float64 if self.dtype == torch.float64 else torch.float32
        inv = 1.0 / s.theta ** (torch.arange(0, s.head_dim, 2).to(frequency_dtype) / s.head_dim)
        angle = len(self.keys[0]) * inv
        cos, sin = angle.cos().to(self.dtype), angle.sin().to(self.dtype)

        def rotate(z):
            half = s.head_dim // 2
            return z * torch.cat((cos, cos)) + torch.cat((-z[:, half:], z[:, :half]), -1) * torch.cat((sin, sin))

        for index in range(self.layers):
            p = f"talker.model.layers.{index}."
            z = self.norm(x, w[p + "input_layernorm.weight"])
            q = self.norm(self.linear(z, w[p + "self_attn.q_proj.weight"]).reshape(s.n_q, s.head_dim), w[p + "self_attn.q_norm.weight"])
            k = self.norm(self.linear(z, w[p + "self_attn.k_proj.weight"]).reshape(s.n_kv, s.head_dim), w[p + "self_attn.k_norm.weight"])
            v = self.linear(z, w[p + "self_attn.v_proj.weight"]).reshape(s.n_kv, s.head_dim)
            q, k = rotate(q), rotate(k)
            k, v = self.cache(k, v)
            self.keys[index].append(k)
            self.values[index].append(v)
            groups = s.n_q // s.n_kv
            keys = torch.stack(self.keys[index], 1).repeat_interleave(groups, 0)
            values = torch.stack(self.values[index], 1).repeat_interleave(groups, 0)
            context = self.attend(q, keys, values, frequency_dtype)
            residual = x + self.linear(context, w[p + "self_attn.o_proj.weight"])
            z = self.norm(residual, w[p + "post_attention_layernorm.weight"])
            x = residual + self.mlp(z, p)
            result["layers"].append(self.array(x))
        normalized = self.norm(x, w["talker.model.norm.weight"])
        result["normalized"] = self.array(normalized)
        result["logits"] = self.array(self.linear(normalized, w["talker.codec_head.weight"]))
        return result
