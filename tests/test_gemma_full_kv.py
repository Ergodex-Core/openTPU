"""Storage-only full-context Gemma KV option; no physical transports."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu.host.chat import Chat, sampler
from opentpu.llm import gemma3 as G
from opentpu.llm.qwen3 import Engine
from opentpu.target import config


@pytest.fixture(autouse=True)
def forbid_physical(monkeypatch):
    from opentpu.host import board, f2
    attempts = []
    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("PHYSICAL_ACCESS_FORBIDDEN")
    monkeypatch.setattr(board, "open_transport", forbidden)
    monkeypatch.setattr(f2.F2Transport, "__init__", forbidden)
    yield
    assert attempts == []


@pytest.fixture(scope="module")
def model():
    spec = G.Spec(hidden=128, kinds=(G.SLIDE, G.FULL), kv_src=(0, 1),
                  ffn=(256, 256), n_q=1, n_kv=1, head_dim=128,
                  global_head_dim=128, global_rot=64, vocab=128, ple_dim=0)
    rng = np.random.default_rng(923)
    def mat(shape):
        return rng.normal(0, 0.025, shape).astype(np.float32)
    weights = {"model.embed_tokens.weight": mat((128, 128)),
               "model.norm.weight": mat((128,))}
    for i in range(2):
        p = f"model.layers.{i}."
        for name in ["input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
            weights[p + name + ".weight"] = mat((128,))
        for name in ["q_norm", "k_norm"]:
            weights[p + "self_attn." + name + ".weight"] = mat((128,))
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weights[p + "self_attn." + name + ".weight"] = mat((128, 128))
        for name, shape in [("gate_proj", (256, 128)), ("up_proj", (256, 128)),
                            ("down_proj", (128, 256))]:
            weights[p + "mlp." + name + ".weight"] = mat(shape)
    return spec, weights


def engine(model, cap=1024, full=False):
    return Engine(*model, cap=cap, cfg=config("f2-hbm:p4rl8a64c150"),
                  resident=True, pipeline=False, rows=1, snapshots=0,
                  kv_full=full, prog_cache=False)


def test_default_layout_unchanged_at_768(model):
    cfg = config("f2-hbm:p4rl8a64c150")
    spec, weights = model
    old = spec.image(cfg, 768, lookup=True)
    new = spec.image(cfg, 768, lookup=True, kv_full=True)
    assert old.ring == new.ring == 768
    assert "kv_full" not in old.choices and new.choices["kv_full"] is True
    assert old.nbytes == new.nbytes
    for a, b in zip(old.build(weights), new.build(weights)):
        np.testing.assert_array_equal(a, b)


def test_strict_guard_and_worker_reconstruction(model):
    a, b = engine(model), engine(model, full=True)
    assert a.stateful and a.image.ring == 768
    assert not b.stateful and b.image.ring == 1024
    assert b._image_kw["kv_full"] is True
    reconstructed = b.spec.image(b.cfg, b.cap, b.batch, b.rows, **b._image_kw)
    assert reconstructed.ring == b.image.ring
    assert reconstructed.nbytes == b.image.nbytes
    pick = sampler(0, 0, 1, 0, softcap=b.spec.softcap)
    with pytest.raises(ValueError, match="stateless-KV"):
        Chat(a, None, False, pick, 64, fpga_only=True)
    Chat(b, None, False, pick, 64, fpga_only=True)


def test_exact_decode_across_original_ring_and_reset(model):
    # The old ring physically wraps at 768; the candidate does not. The exact
    # 512-token sliding mask and chronological accumulation must still agree.
    a, b = engine(model), engine(model, full=True)
    tokens = np.random.default_rng(41).integers(3, 128, 780).tolist()
    for t in tokens:
        np.testing.assert_array_equal(a.step(t), b.step(t))
    assert b.rollback(16) == 16
    expected = []
    for t in tokens[16:20]:
        expected.append(b.step(t))
    b.reset()
    got = [b.step(t) for t in tokens[:20]][16:20]
    np.testing.assert_array_equal(expected, got)


def test_chain_crosses_bucket_and_replays(model):
    a, b = engine(model), engine(model, full=True)
    tokens = np.random.default_rng(73).integers(3, 128, 255).tolist()
    a.prefill(tokens)
    b.prefill(tokens)
    ga = a.generate_card(31, 4, stop_ids=[])
    gb = b.generate_card(31, 4, stop_ids=[])
    assert len(ga) == 4 and ga == gb and a.pos == b.pos == 259
    b.reset()
    b.prefill(tokens)
    assert b.generate_card(31, 4, stop_ids=[]) == gb


@pytest.mark.parametrize("cap", [1280, 2304, 4096, 4352])
def test_layout_bounds_and_attention_window(model, cap):
    spec, _ = model
    cfg = config("f2-hbm:p4rl8a64c150")
    image = spec.image(cfg, cap, lookup=True, kv_full=True)
    assert image.ring == cap and image.state_regions() == []
    assert image.spec.window == spec.window == 512
    assert image.nbytes < cfg.DRAM_BYTES
    with pytest.raises(ValueError, match="multiple"):
        spec.image(cfg, cap + 1, kv_full=True)


def test_full_kv_rejects_other_model_family(model):
    from opentpu.llm.qwen3 import Spec
    spec = Spec(hidden=128, ffn=256, layers=1, n_q=1, n_kv=1,
                head_dim=128, vocab=128)
    with pytest.raises(ValueError, match="Gemma"):
        engine((spec, model[1]), full=True)


def test_all_17_buckets_preserve_sliding_math(model):
    # A complete small model executes every input position through 4096+64.
    # Existing global-attention storage does not change; isolate the changed
    # sliding storage to keep this long numerical regression inexpensive.
    spec = replace(model[0], kinds=(G.SLIDE,), kv_src=(0,), ffn=(256,))
    small = (spec, model[1])
    a, b = engine(small, cap=4352), engine(small, cap=4352, full=True)
    for token in np.random.default_rng(712).integers(3, 128, 4160):
        np.testing.assert_array_equal(a.step(int(token)), b.step(int(token)))
    assert a.pos == b.pos == 4160
