"""Gemma 3 1B and FunctionGemma 270M text decoders on Gemma 4's kernels.

Gemma 3 uses the same four-norm residual blocks and GeGLU, with effective norm gains
1 + weight, no V norm, attention scaled by query_pre_attn_scalar**-0.5, full RoPE on
both layer kinds, and no PLE or layer scalar. Every sixth layer is global; the others
use a 512-token window. The tied embedding and LM head use the image's head format.

HF and MLX safetensors are read lazily. A directory can instead contain one GGUF and
its HF config/tokenizer; GGUF's norm gains already include one (removed when read).
Q4_0 tensors are dequantized exactly before openTPU's weight quantizer runs: GGUF and
openTPU int4 have different scale encodings, so this is not a lossless GGUF repack.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import quant as Q
from . import gemma4 as G
from .gemma4 import (FULL, SLIDE, Image, emulated_logits, reference_logits, rope_freqs,
                     rope_tables, softcap, text_ids)
from .qwen3 import finite


@dataclass(frozen=True)
class Spec(G.Spec):
    norm_offset: float = 1.0
    v_norm: bool = False
    attn_scale: float = 1 / 16
    layer_scale: bool = False
    softcap: float | None = None

    @staticmethod
    def from_hf(model_dir) -> "Spec":
        top = json.loads((Path(model_dir) / "config.json").read_text())
        c = top.get("text_config", top)
        if c.get("model_type", "gemma3_text") != "gemma3_text":
            raise ValueError("not a Gemma 3 text configuration")
        if c.get("attention_bias") or c.get("mlp_bias") or c.get("use_bidirectional_attention"):
            raise ValueError("Gemma 3 biases and bidirectional attention are not supported")
        if not top.get("tie_word_embeddings", c.get("tie_word_embeddings", True)):
            raise ValueError("Gemma 3 requires a tied embedding / LM head")
        if c.get("attn_logit_softcapping"):
            raise ValueError("Gemma 3 attention softcapping is not supported")
        if c.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise ValueError("Gemma 3 requires gelu_pytorch_tanh")
        L, d = c["num_hidden_layers"], c["head_dim"]
        pattern = c.get("sliding_window_pattern", c.get("_sliding_window_pattern", 6))
        if pattern < 1:
            raise ValueError("sliding_window_pattern must be positive")
        kinds = c.get("layer_types") or ["full_attention" if (i + 1) % pattern == 0 else
                                        "sliding_attention" for i in range(L)]
        if len(kinds) != L or set(kinds) - {"sliding_attention", "full_attention"}:
            raise ValueError("invalid Gemma 3 layer_types")
        rp = c.get("rope_parameters") or c.get("rope_scaling") or {}
        # Recent transformers serializes per-kind rope_parameters; older configs have
        # rope_theta / rope_local_base_freq and rope_scaling (null for 1B and 270M).
        g = rp.get("full_attention", rp)
        s = rp.get("sliding_attention", {})
        for r in (g, s):
            if r.get("rope_type", r.get("type", "default")) != "default" or \
                    r.get("partial_rotary_factor", 1.0) != 1.0:
                raise ValueError("only full default RoPE is supported for Gemma 3")
        eos = c.get("eos_token_id", [1, 106])
        gc = Path(model_dir) / "generation_config.json"
        if gc.exists():
            eos = json.loads(gc.read_text()).get("eos_token_id", eos)
        eos = tuple(eos) if isinstance(eos, list) else (eos,)
        return Spec(hidden=c["hidden_size"], kinds=tuple(FULL if k == "full_attention" else
                    SLIDE for k in kinds), kv_src=tuple(range(L)),
                    ffn=(c["intermediate_size"],) * L, n_q=c["num_attention_heads"],
                    n_kv=c["num_key_value_heads"], head_dim=d, global_head_dim=d,
                    vocab=c["vocab_size"], ple_dim=0, window=c.get("sliding_window", 512),
                    theta=s.get("rope_theta", c.get("rope_local_base_freq", 1e4)),
                    global_theta=g.get("rope_theta", c.get("rope_theta", 1e6)),
                    global_rot=d // 2, eps=c.get("rms_norm_eps", 1e-6),
                    attn_scale=c.get("query_pre_attn_scalar", d) ** -0.5,
                    softcap=c.get("final_logit_softcapping"),
                    tied=top.get("tie_word_embeddings", c.get("tie_word_embeddings", True)),
                    bos=c.get("bos_token_id", 2), eos=eos)


class Weights(G.Weights):
    """Gemma 3's HF / MLX names, or GGUF tensors, exposed under HF text-model names."""
    def __init__(self, model_dir):
        from safetensors import safe_open
        dict.__init__(self)
        self.model_dir, self._files, self._gguf = Path(model_dir), {}, {}
        for p in sorted(self.model_dir.glob("*.safetensors")):
            f = safe_open(str(p), "pt")
            for k in f.keys():
                name = k
                for prefix in ("model.language_model.", "language_model.model.", "language_model."):
                    if name.startswith(prefix):
                        name = "model." + name[len(prefix):]
                        break
                if name.startswith(("layers.", "embed_tokens.", "norm.")):
                    name = "model." + name
                if name.startswith(("model.", "lm_head.")):
                    self._files[name] = (f, k)
        if self._files:
            return
        paths = list(self.model_dir.glob("*.gguf"))
        if len(paths) != 1:
            raise ValueError(f"{model_dir}: expected safetensors or exactly one GGUF")
        import gguf
        self._reader = gguf.GGUFReader(str(paths[0]))
        for t in self._reader.tensors:
            name = _gguf_name(t.name)
            if name is not None:
                self._gguf[name] = t
        self._files = dict.fromkeys(self._gguf)
        if "model.embed_tokens.weight" not in self:
            raise ValueError(f"{paths[0]}: no Gemma embedding")

    def __getitem__(self, k):
        if not self._gguf:
            return super().__getitem__(k)
        t = self._gguf[k]
        v = self._decode(t, t.data)
        # llama.cpp stores the effective norm gain. The shared reference / image adds one.
        return finite(v - np.float32(1) if "norm.weight" in k else v, k)

    @staticmethod
    def _decode(t, data):
        import gguf
        if t.tensor_type == gguf.GGMLQuantizationType.Q4_0:
            return Q.dequantize_q4_0(data)
        if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            return np.asarray(data, np.float32)
        raise ValueError(f"{t.name}: unsupported GGUF type {t.tensor_type}")

    def rows(self, k, idx):
        if not self._gguf:
            return super().rows(k, idx)
        t = self._gguf[k]
        return finite(self._decode(t, t.data[list(idx)]), k)


def _gguf_name(name):
    common = {"token_embd.weight": "model.embed_tokens.weight",
              "output_norm.weight": "model.norm.weight", "output.weight": "lm_head.weight"}
    if name in common:
        return common[name]
    m = re.fullmatch(r"blk\.(\d+)\.(.+)\.weight", name)
    names = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
             "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
             "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
             "attn_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm",
             "ffn_norm": "pre_feedforward_layernorm", "post_ffw_norm": "post_feedforward_layernorm",
             "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
    if m and m[2] in names:
        return f"model.layers.{m[1]}.{names[m[2]]}.weight"
    return None


def load_weights(model_dir) -> Weights:
    return Weights(model_dir)
