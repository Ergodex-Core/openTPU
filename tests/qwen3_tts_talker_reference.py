"""Independent source-derived FP64 math for one Qwen3-TTS talker layer.

Source: QwenLM/Qwen3-TTS 022e286b98fbec7e1e916cb940cdf532cd9f488e,
modeling_qwen3_tts.py: Qwen3TTSTalkerResizeMLP, Qwen3TTSTalkerAttention,
Qwen3TTSTalkerDecoderLayer, Qwen3TTSTalkerTextMLP, Qwen3TTSRMSNorm,
apply_multimodal_rotary_pos_emb (three equal position axes).

This intentionally imports no compiler/ISA/quantizer helpers or upstream Python.
It is full-precision mathematics, not a BF16/PyTorch execution claim.
"""
import numpy as np


class Reference:
    def __init__(self, spec, weights):
        self.spec = spec
        self.weights = {name: np.asarray(value, np.float64) for name, value in weights.items()}
        self.keys, self.values = [], []

    def step(self, text_id, codec_id):
        s, w = self.spec, self.weights
        p = "talker.text_projection."
        text = w["talker.model.text_embedding.weight"][text_id]
        a = w[p + "linear_fc1.weight"] @ text + w[p + "linear_fc1.bias"]
        projected = w[p + "linear_fc2.weight"] @ (a / (1 + np.exp(-a))) + w[p + "linear_fc2.bias"]
        assembled = projected + w["talker.model.codec_embedding.weight"][codec_id]
        p = "talker.model.layers.0."

        def norm(x, name):
            return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + s.eps) * w[p + name]

        z = norm(assembled, "input_layernorm.weight")
        q = norm((w[p + "self_attn.q_proj.weight"] @ z).reshape(s.n_q, s.head_dim),
                 "self_attn.q_norm.weight")
        k = norm((w[p + "self_attn.k_proj.weight"] @ z).reshape(s.n_kv, s.head_dim),
                 "self_attn.k_norm.weight")
        v = (w[p + "self_attn.v_proj.weight"] @ z).reshape(s.n_kv, s.head_dim)
        # The three position axes are equal in the audited text-only path. An
        # interleaved selection from equal cosine/sine axes is an identity.
        half = s.head_dim // 2
        angle = len(self.keys) * s.theta ** (-np.arange(half) * 2.0 / s.head_dim)
        cos, sin = np.cos(angle), np.sin(angle)

        def rotate(x):
            return np.concatenate((x[:, :half] * cos - x[:, half:] * sin,
                                   x[:, :half] * sin + x[:, half:] * cos), axis=-1)

        q, k = rotate(q), rotate(k)
        self.keys.append(k)
        self.values.append(v)
        keys, values = np.stack(self.keys, axis=1), np.stack(self.values, axis=1)
        context = []
        for head in range(s.n_q):
            kv = head // (s.n_q // s.n_kv)
            logits = keys[kv] @ q[head] / np.sqrt(s.head_dim)
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)
            context.append(probs @ values[kv])
        residual = assembled + w[p + "self_attn.o_proj.weight"] @ np.concatenate(context)
        z = norm(residual, "post_attention_layernorm.weight")
        gate = w[p + "mlp.gate_proj.weight"] @ z
        up = w[p + "mlp.up_proj.weight"] @ z
        out = residual + w[p + "mlp.down_proj.weight"] @ ((gate / (1 + np.exp(-gate))) * up)
        return {"projected": projected, "assembled": assembled,
                "attention_residual": residual, "output": out}
