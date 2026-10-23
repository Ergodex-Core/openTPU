"""Gemma 3 text decoders on openTPU: google/gemma-3-1b-it (and the base gemma-3-1b-pt).

Gemma 3 is Gemma 4 (gemma4.py) without the per-layer embeddings, the KV sharing, the layer
scalars and the V norm, so it runs on Gemma 4's Spec, DRAM image, kernels and Engine; this
module reads its config (spec_from_hf) and folds its parameters into Gemma 4's (folder). What
differs, and where it goes:
  norms        Gemma 3's RMSNorm scales by 1 + w (Gemma 4's by w): every norm weight is read as
               1 + w (gemma4.Weights folds it when the checkpoint is a Gemma 3)
  attention    scaled by query_pre_attn_scalar^-1/2 (1 / 16 at 256-wide heads; Gemma 4: 1):
  scale        folded into q_norm's gain (before RoPE, which commutes with a scalar; the
               query's int8 rounding is per block and scale-free)
  V            not normed (gemma4.Spec v_norm False); K and q are, as Gemma 4's
  layer_scalar none: read as 1
  global RoPE  the whole 256-wide head (global_rot = head_dim / 2), theta 1e6; sliding
               layers' theta 1e4 (rope_local_base_freq). Linear scaling of the global
               frequencies (the 4B+ checkpoints' factor 8) divides them (global_rope_div)
  layers       5 sliding (a 512-token window) then 1 global (sliding_window_pattern 6, or
               layer_types), every layer with its own K / V; GeGLU MLP, gelu_tanh
  head         tied to the embedding (x sqrt(H) on input), no soft cap on gemma-3-1b
The embedding rows are gathered on the device from the int8 LM head, as Gemma 4's.

    hf download google/gemma-3-1b-it --local-dir models/gemma-3-1b-it      # MODELS' gemma3
    otpu-chat --model gemma3 --backend isa
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import formats as FM
# the references and kernels are Gemma 4's: the F2 tools (tools/fpga_inference_check.py,
# tools/gemma3_check.py) take them from here with a folded gemma4.Weights (load_weights)
from .gemma4 import emulated_logits, reference_logits  # noqa: F401

MODEL_TYPES = ("gemma3_text", "gemma3")


def _text_config(model_dir) -> dict:
    top = json.loads((Path(model_dir) / "config.json").read_text())
    return top, top.get("text_config", top)


def spec_from_hf(model_dir):
    """The gemma4.Spec of a Gemma 3 checkpoint, as Hugging Face's Gemma3 modeling code reads
    its config.json."""
    from .gemma4 import FULL, SLIDE, Spec
    top, c = _text_config(model_dir)
    mt = c.get("model_type")
    if mt not in MODEL_TYPES:
        raise ValueError(f"{model_dir}: model type {mt!r} is not a Gemma 3 one {MODEL_TYPES}")
    if c.get("attention_bias"):
        raise ValueError(f"{model_dir}: attention biases are not supported")
    if c.get("attn_logit_softcapping"):
        raise ValueError(f"{model_dir}: attention logit soft capping is not supported")
    if c.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        raise ValueError(f"activation {c['hidden_activation']} is not gelu_tanh")
    L, H, d = c["num_hidden_layers"], c["hidden_size"], c["head_dim"]
    types = c.get("layer_types")
    if types is None:                       # Gemma3TextConfig's default
        n = c.get("sliding_window_pattern", 6)
        types = ["sliding_attention" if (i + 1) % n else "full_attention" for i in range(L)]
    kinds = tuple(FULL if t == "full_attention" else SLIDE for t in types)
    # RoPE: rope_parameters (transformers >= 5) over rope_theta / rope_local_base_freq /
    # rope_scaling (the checkpoints' configs): the global layers' may be linearly scaled
    rp = c.get("rope_parameters") or {}
    g, s = dict(rp.get("full_attention") or {}), dict(rp.get("sliding_attention") or {})
    rs = c.get("rope_scaling") or {}
    g.setdefault("rope_theta", c.get("rope_theta", 1e6))
    s.setdefault("rope_theta", c.get("rope_local_base_freq", 1e4))
    gkind = g.get("rope_type", rs.get("rope_type", rs.get("type", "default")))
    if gkind not in ("default", "linear") or s.get("rope_type", "default") != "default":
        raise ValueError(f"{model_dir}: RoPE types {gkind} (global) / "
                         f"{s.get('rope_type', 'default')} (sliding) are not supported")
    div = float(g.get("factor", rs.get("factor", 1.0))) if gkind == "linear" else 1.0
    # the stop ids: the config's and the generation config's (<eos> 1, <end_of_turn> 106)
    eos, gc = [], Path(model_dir) / "generation_config.json"
    for e in (c.get("eos_token_id", 1),
              json.loads(gc.read_text()).get("eos_token_id") if gc.exists() else None):
        eos += e if isinstance(e, list) else [] if e is None else [e]
    return Spec(hidden=H, kinds=kinds, kv_src=tuple(range(L)),
                ffn=(c["intermediate_size"],) * L, n_q=c["num_attention_heads"],
                n_kv=c["num_key_value_heads"], head_dim=d, global_head_dim=d,
                vocab=c["vocab_size"], ple_dim=0, window=c["sliding_window"],
                theta=float(s["rope_theta"]), global_theta=float(g["rope_theta"]),
                global_rot=d // 2, global_rope_div=div, eps=c.get("rms_norm_eps", 1e-6),
                softcap=c.get("final_logit_softcapping"),
                tied=top.get("tie_word_embeddings", c.get("tie_word_embeddings", True)),
                bos=c.get("bos_token_id", 2), eos=tuple(dict.fromkeys(eos)),
                mix=FM.mix_for(c), v_norm=False)


def folder(config: dict):
    """fold(name, array) -> array for a Gemma 3 checkpoint's config (None for any other): the
    tensors as Gemma 4's code reads them. Every norm weight 1 + w (Gemma3RMSNorm), q_norm's
    also times the attention scale query_pre_attn_scalar^-1/2; a layer_scalar (which Gemma 3
    has not: `array` None) is 1."""
    c = config.get("text_config", config)
    if c.get("model_type") not in MODEL_TYPES:
        return None
    qs = np.float32(float(c.get("query_pre_attn_scalar", c["head_dim"])) ** -0.5)

    def fold(k: str, a):
        if a is None:
            return np.ones(1, np.float32) if k.endswith("layer_scalar") else None
        if k.endswith("norm.weight"):
            a = np.float32(1) + np.asarray(a, np.float32)
            if k.endswith("q_norm.weight"):
                a = a * qs
        return a
    return fold


def fold_weights(W: dict, config: dict) -> dict:
    """A Gemma 3 state dict (Hugging Face's names, fp32 arrays) folded for Gemma 4's code
    (folder), with a layer_scalar of 1 per layer: tests and tools that hold the weights."""
    fold = folder(config)
    if fold is None:
        raise ValueError("not a Gemma 3 config")
    out = {k: fold(k, np.asarray(v, np.float32)) for k, v in W.items()}
    c = config.get("text_config", config)
    for i in range(c["num_hidden_layers"]):
        out[f"model.layers.{i}.layer_scalar"] = fold(f"model.layers.{i}.layer_scalar", None)
    return out
