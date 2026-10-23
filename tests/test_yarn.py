"""YaRN RoPE (opentpu.llm.yarn_rope) as Qwen3's Spec takes it: Ternary-Bonsai-1.7B's config
(factor 4 of 8192 positions, theta 1e6, head 128). Against transformers' own YaRN init when it is
installed; the closed-form properties always."""
import json
import math

import numpy as np
import pytest

from opentpu.llm import yarn_rope
from opentpu.llm.qwen3 import Spec

BONSAI = {"vocab_size": 151669, "max_position_embeddings": 32768, "hidden_size": 2048,
          "intermediate_size": 6144, "num_hidden_layers": 28, "num_attention_heads": 16,
          "num_key_value_heads": 8, "head_dim": 128, "rms_norm_eps": 1e-6,
          "rope_theta": 1000000.0, "tie_word_embeddings": True, "model_type": "qwen3",
          "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                           "original_max_position_embeddings": 8192},
          "eos_token_id": 151645, "architectures": ["Qwen3ForCausalLM"]}


def _spec(tmp_path, cfg=BONSAI):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return Spec.from_hf(tmp_path)


def test_yarn_closed_form(tmp_path):
    s = _spec(tmp_path)
    assert s.rope_scale == pytest.approx(0.1 * math.log(4.0) + 1.0)
    d = np.asarray(s.rope_div)
    assert len(d) == 64 and d[0] == 1.0 and d[-1] == pytest.approx(4.0)
    assert np.all(np.diff(d) >= 0)                  # high frequencies kept, low ones interpolated


def test_default_rope_has_no_divisors(tmp_path):
    cfg = dict(BONSAI)
    cfg.pop("rope_scaling")
    s = _spec(tmp_path, cfg)
    assert s.rope_div == () and s.rope_scale == 1.0


def test_yarn_matches_transformers(tmp_path):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    s = _spec(tmp_path)
    cfg = tf.AutoConfig.from_pretrained(tmp_path)
    inv_hf, att = ROPE_INIT_FUNCTIONS["yarn"](cfg, "cpu")
    half = s.rope_dim // 2
    inv = 1.0 / (s.theta ** (np.arange(half) * 2 / s.rope_dim)) / np.asarray(s.rope_div)
    np.testing.assert_allclose(inv, inv_hf.double().numpy(), rtol=1e-6)
    assert s.rope_scale == pytest.approx(att)


def test_yarn_rope_mscale_pair():
    div, af = yarn_rope({"factor": 4.0, "mscale": 1.0, "mscale_all_dim": 1.0,
                         "original_max_position_embeddings": 8192}, 1e6, 128, 32768)
    assert af == pytest.approx(1.0) and len(div) == 64
