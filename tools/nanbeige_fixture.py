"""Deterministic small model pair for DSpark ISA and FPGA arithmetic gates."""
import numpy as np
from opentpu.llm.dspark import Spec as DraftSpec
from opentpu.llm.nanbeige import Spec as TargetSpec


def tiny_fixture(seed=92):
    """Public deterministic fixture for the standalone FPGA/ISA hardware gate."""
    rng = np.random.default_rng(seed)
    target = TargetSpec(128, 2, 2, 1, 128, 256, 128,
                        eps=1e-5, theta=7e7, capture_layers=(1, 3), eos=())
    draft = DraftSpec(128, 1, 2, 1, 128, 256, 128, 128, 128, (0, 2), 127, 8,
                      eps=1e-5, theta=7e7, block_size=7)

    def matrices(spec, prefix):
        out = {}
        def put(name, shape, gain=False):
            out[prefix + name + ".weight"] = (rng.uniform(.8, 1.2, shape) if gain else
                rng.normal(0, .24 / np.sqrt(shape[-1]), shape)).astype(np.float32)
        put("norm", (spec.hidden,), True)
        for layer in range(spec.layers):
            p = f"layers.{layer}."
            for name in ("input_layernorm", "post_attention_layernorm"):
                put(p + name, (spec.hidden,), True)
            for name, heads in (("q", spec.n_q), ("k", spec.n_kv), ("v", spec.n_kv)):
                put(p + f"self_attn.{name}_proj", (heads * spec.head_dim, spec.hidden))
            put(p + "self_attn.o_proj", (spec.hidden, spec.n_q * spec.head_dim))
            for name, shape in (("gate", (spec.ffn, spec.hidden)),
                                ("up", (spec.ffn, spec.hidden)),
                                ("down", (spec.hidden, spec.ffn))):
                put(p + f"mlp.{name}_proj", shape)
        return out

    tw = matrices(target, "model.")
    tw["model.embed_tokens.weight"] = rng.normal(0, .3, (128, 128)).astype(np.float32)
    tw["lm_head.weight"] = rng.normal(0, .04, (128, 128)).astype(np.float32)
    dw = matrices(draft, "")
    dw["fc.weight"] = rng.normal(0, .04, (128, 256)).astype(np.float32)
    dw["hidden_norm.weight"] = rng.uniform(.8, 1.2, 128).astype(np.float32)
    dw["layers.0.self_attn.q_norm.weight"] = np.ones(128, np.float32)
    dw["layers.0.self_attn.k_norm.weight"] = np.ones(128, np.float32)
    dw["markov_head.markov_w1.weight"] = rng.normal(0, .1, (128, 8)).astype(np.float32)
    dw["markov_head.markov_w2.weight"] = rng.normal(0, .1, (128, 8)).astype(np.float32)
    return target, tw, draft, dw
